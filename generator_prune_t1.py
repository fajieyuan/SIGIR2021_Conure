import tensorflow as tf
# import ops_prune as ops
import ops
import numpy as np
import config

class NextItNet_Decoder:

    def __init__(self, model_para):
        self.model_para = model_para
        self.embedding_width =  model_para['dilated_channels']
        self.taskID=model_para['taskID']
        self.allitem_embeddings = tf.get_variable('allitem_embeddings',
                                                    [model_para['bigemb'], self.embedding_width],
                                                    initializer=tf.truncated_normal_initializer(stddev=0.02))

    def train_graph(self, is_negsample=False,ispre=True):
        self.itemseq_input = tf.placeholder('int32',
                                         [None, None], name='itemseq_input')


        label_seq, self.dilate_input=self.model_graph(self.itemseq_input, train=True, ispre=ispre)

        model_para = self.model_para
        if is_negsample:
            logits_2D = tf.reshape(self.dilate_input, [-1,model_para['dilated_channels']])
            self.softmax_w = tf.get_variable("softmax_w_{}".format(self.taskID), [model_para['item_size'],  model_para['dilated_channels']],tf.float32,tf.random_normal_initializer(0.0, 0.01))
            self.softmax_b = tf.get_variable("softmax_b_{}".format(self.taskID), [model_para['item_size']], tf.float32, tf.constant_initializer(0.1))
            label_flat = tf.reshape(label_seq, [-1, 1])  # 1 is the number of positive example
            num_sampled = int(0.2* model_para['item_size'])#sample 20% as negatives
            loss =tf.nn.sampled_softmax_loss(self.softmax_w, self.softmax_b, label_flat, logits_2D, num_sampled,model_para['item_size'])

        else:
            logits = ops.conv1d(tf.nn.relu(self.dilate_input), model_para['item_size'], name='logits')
            logits_2D = tf.reshape(logits, [-1, model_para['item_size']])
            label_flat = tf.reshape(label_seq, [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_flat, logits=logits_2D)

        self.loss = tf.reduce_mean(loss)
        self.arg_max_prediction = tf.argmax(logits_2D, 1)
    def model_graph(self, itemseq_input, train=True,ispre=True):
        model_para = self.model_para
        context_seq = itemseq_input[:, 0:-1]
        label_seq = itemseq_input[:, 1:]


        self.context_embedding = tf.nn.embedding_lookup(self.allitem_embeddings,
                                                   context_seq, name="context_embedding")
        # positional embedding
        if self.model_para['has_positionalembedding']:
            pos_emb = self.embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(context_seq)[1]), 0),
                        [tf.shape(itemseq_input)[0], 1]),
                max_position=model_para['max_position'],
                num_units=self.embedding_width,
                zero_pad=False,
                scale=False,
                l2_reg=0.0,
                scope="dec_pos",
                with_t=False
            )
            dilate_input=self.context_embedding+pos_emb
        else:
            dilate_input = self.context_embedding

        residual_channels = dilate_input.get_shape().as_list()[-1]
        for layer_id, dilation in enumerate(model_para['dilations']):
            if ispre == True:
                dilate_input = ops.nextitnet_residual_block_withmask_pre_beforeln(dilate_input, dilation,
                                                                                  layer_id, residual_channels,
                                                                                  model_para['kernel_size'],
                                                                                  self.taskID,
                                                                                  causal=True, train=train)
            else:
                dilate_input = ops.nextitnet_residual_block_withmask_fine_beforeln(dilate_input, dilation,
                                                                                   layer_id, residual_channels,
                                                                                   model_para['kernel_size'],
                                                                                   self.taskID,
                                                                                   causal=True, train=train)

        return label_seq, dilate_input

    def predict_graph(self, is_negsample=False, reuse=False,ispre=True):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        self.input_predict = tf.placeholder('int32', [None, None], name='input_predict')

        label_seq, dilate_input = self.model_graph(self.input_predict, train=False,ispre=ispre)
        model_para = self.model_para

        if is_negsample:
            logits_2D = tf.reshape(dilate_input[:, -1:, :], [-1, model_para['dilated_channels']])
            logits_2D = tf.matmul(logits_2D, tf.transpose(self.softmax_w))
            logits_2D = tf.nn.bias_add(logits_2D, self.softmax_b)
        else:
            logits = ops.conv1d(tf.nn.relu(dilate_input[:, -1:, :]), model_para['item_size'], name='logits')
            logits_2D = tf.reshape(logits, [-1, model_para['item_size']])

        label_flat = tf.reshape(label_seq[:, -1], [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_flat, logits=logits_2D)
        self.loss_test = tf.reduce_mean(loss)
        probs_flat = tf.nn.softmax(logits_2D)
        self.g_probs = tf.reshape(probs_flat, [-1, 1, model_para['item_size']])
        self.top_k = tf.nn.top_k(self.g_probs[:, -1], k=5, name='top-k')

    def save_impwei(self, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        init_zeros = tf.zeros_initializer()
        trainable_vars = tf.trainable_variables()
        variables_to_restore = [v for v in trainable_vars if v.name.find("weight") != -1]
        self.mask_val_list = []

        kernel_num = 1 * self.model_para['kernel_size'] * self.embedding_width * self.embedding_width
        cutoff_rank = tf.cast((1.0 - config.maskp_task1) * kernel_num, tf.int32)
        graph = tf.get_default_graph()

        with tf.variable_scope("mask_filter", reuse=tf.AUTO_REUSE):
            for layer_id, dilation in enumerate(self.model_para['dilations']):
                mask_name = "mask_val_layer_{}_{}".format(layer_id, dilation)

                dilated_conv1 = variables_to_restore[2 * layer_id]
                dilated_conv1_norm = tf.abs(dilated_conv1)
                dilated_conv1_onedim = tf.reshape(dilated_conv1_norm, [kernel_num])
                top_k_dilated_conv1 = tf.nn.top_k(dilated_conv1_onedim, cutoff_rank + 1).values[
                    cutoff_rank]  # e.g., 2.3

                one = tf.ones_like(dilated_conv1_norm)
                zero = tf.zeros_like(dilated_conv1_norm)
                mask_dilated_conv1 = tf.where(dilated_conv1_norm < top_k_dilated_conv1, x=zero, y=one)
                # mask_dilated_conv1 = tf.where(dilated_conv1_norm < top_k_dilated_conv1, x=one, y=zero)

                dilated_conv2 = variables_to_restore[2 * layer_id + 1]
                dilated_conv2_norm = tf.abs(dilated_conv2)

                dilated_conv2_onedim = tf.reshape(dilated_conv2_norm, [kernel_num])
                top_k_dilated_conv2 = tf.nn.top_k(dilated_conv2_onedim, cutoff_rank + 1).values[
                    cutoff_rank]  # e.g., 2.3

                mask_dilated_conv2 = tf.where(dilated_conv2_norm < top_k_dilated_conv2, x=zero, y=one)

                self.mask_val_list.append(mask_dilated_conv1)
                self.mask_val_list.append(mask_dilated_conv2)

    def embedding(self, inputs, max_position, num_units, zero_pad=True, scale=True, l2_reg=0.0, scope="embedding",
                  with_t=False):
        with tf.variable_scope(scope):
            lookup_table = tf.get_variable('lookup_table_position',
                                           dtype=tf.float32,
                                           shape=[max_position, num_units],
                                           # initializer=tf.contrib.layers.xavier_initializer(),
                                           regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
            if zero_pad:
                lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                          lookup_table[1:, :]), 0)
            outputs = tf.nn.embedding_lookup(lookup_table, inputs)

            if scale:
                outputs = outputs * (num_units ** 0.5)
        if with_t:
            return outputs, lookup_table
        else:
            return outputs

























