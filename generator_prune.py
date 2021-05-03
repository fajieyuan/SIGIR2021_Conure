import tensorflow as tf
import ops
import numpy as np
import config

# import pandas as pd
# this class is used for true system, only predict but no evaluation
class NextItNet_Decoder:

    def __init__(self, model_para):
        self.model_para = model_para
        self.embedding_width =  model_para['dilated_channels']
        self.taskID= model_para['taskID']
        self.allitem_embeddings = tf.get_variable('allitem_embeddings',
                                                    [model_para['bigemb'], self.embedding_width ],
                                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.allitem_embeddings_out = tf.get_variable("softmax_w_{}".format(self.taskID),
                                                  [model_para['target_item_size'], self.embedding_width ],
                                                  initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                  regularizer=tf.contrib.layers.l2_regularizer(0.02))

    def train_graph(self,  ispre=True):
        model_para = self.model_para
        self.itemseq_input = tf.placeholder('int32',
                                         [None, None], name='itemseq_input')
        if ispre==True:
            self.dilate_input = self.model_graph(self.itemseq_input, train=True)
        else:
            self.dilate_input = self.model_graph(self.itemseq_input, train=True,ispre=False)


    def model_graph(self, itemseq_input, train=True, ispre=True):
        model_para = self.model_para
        context_seq = itemseq_input[:, 0:-1]
        # label_seq = itemseq_input[:, 1:]

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
            # dilate_input = tf.concat([self.context_embedding, pos_emb], -1)
            dilate_input =self.context_embedding+pos_emb
        else:
            dilate_input = self.context_embedding

        # dilate_input =  self.context_embedding
        residual_channels = dilate_input.get_shape().as_list()[-1]

        for layer_id, dilation in enumerate(model_para['dilations']):
            if ispre == True:
                dilate_input = ops.nextitnet_residual_block_withmask_pre_beforeln(dilate_input, dilation,
                                                                              layer_id, residual_channels,
                                                                              model_para['kernel_size'], self.taskID,
                                                                              causal=True, train=train)
            else:
                dilate_input = ops.nextitnet_residual_block_withmask_fine_beforeln(dilate_input, dilation,
                                                                     layer_id, residual_channels,
                                                                     model_para['kernel_size'], self.taskID,
                                                                     causal=True, train=train)


        return dilate_input

    #saving important weights
    def save_impwei(self,mask_var,weight, curtaskID, reuse=False):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            init_zeros = tf.zeros_initializer()
            trainable_vars = tf.trainable_variables()
            self.mask_val_list_task = []
            kernel_num = 1 * self.model_para['kernel_size'] * self.embedding_width * self.embedding_width

            # cutoff_rank_task_remain =  config.maskp_task1 * config.maskp_task2 * config.maskp_task3* (1-config.maskp_task4)
            if curtaskID == config.taskID_1st:
                cutoff = 1 - config.maskp_task1
            elif curtaskID == config.taskID_2nd:
                cutoff = config.maskp_task1 * (1 - config.maskp_task2)
            elif curtaskID == config.taskID_3rd:
                cutoff = config.maskp_task1 * config.maskp_task2 * (1 - config.maskp_task3)
            elif curtaskID == config.taskID_4th:
                cutoff = config.maskp_task1 * config.maskp_task2 * config.maskp_task3 * (
                        1 - config.maskp_task4)
            elif curtaskID == config.taskID_5th:
                cutoff = config.maskp_task1 * config.maskp_task2 * config.maskp_task3 * config.maskp_task4 * (
                        1 - config.maskp_task5)
            elif curtaskID == config.taskID_6th:
                cutoff = config.maskp_task1 * config.maskp_task2 * config.maskp_task3 * config.maskp_task4 * config.maskp_task5 * (
                        1 - config.maskp_task6)

            cutoff_rank_task_remain = cutoff

            cutoff_rank = tf.cast(cutoff_rank_task_remain* kernel_num, tf.int32)
            graph = tf.get_default_graph()


            with tf.variable_scope("mask_filter", reuse=tf.AUTO_REUSE):
                for layer_id, dilation in enumerate(self.model_para['dilations']):
                    mask_name = "mask_val_layer_{}_{}".format(layer_id, dilation)
                    mask_name_layid_dilation="_{}_{}".format(layer_id, dilation)
                    mask_lay_dilation=[v for v in mask_var if v.name.find(mask_name_layid_dilation) != -1]# mask according to layer_id and dilation

                    frozen_matrix_conv1 = tf.zeros_like(weight[0])
                    frozen_matrix_conv2 = tf.zeros_like(weight[0])

                    for index in xrange(curtaskID-config.taskID_1st):
                        taskid=config.taskID_1st+index
                        mask_task=[v for v in mask_lay_dilation if v.name.find(str(taskid)) != -1]
                        mask_task_conv1=[v for v in mask_task if v.name.find("conv1") != -1]
                        mask_task_conv2 = [v for v in mask_task if v.name.find("conv2") != -1]
                        frozen_matrix_conv1+=mask_task_conv1[0]
                        frozen_matrix_conv2 += mask_task_conv2[0]

                    dilated_conv1 = [v for v in weight if
                                     v.name.find("conv1") != -1 and v.name.find(mask_name_layid_dilation) != -1][0]
                    dilated_conv2 = [v for v in weight if
                                     v.name.find("conv2") != -1 and v.name.find(mask_name_layid_dilation) != -1][0]

                    mask_conv1_ = tf.abs(frozen_matrix_conv1 - 1)
                    dilated_conv1_norm = tf.abs(dilated_conv1*mask_conv1_)
                    dilated_conv1_onedim = tf.reshape(dilated_conv1_norm, [kernel_num])
                    top_k_dilated_conv1 = tf.nn.top_k(dilated_conv1_onedim, cutoff_rank + 1).values[cutoff_rank]

                    one = tf.ones_like(dilated_conv1_norm)
                    zero = tf.zeros_like(dilated_conv1_norm)
                    mask_dilated_conv1 = tf.where(dilated_conv1_norm < top_k_dilated_conv1, x=zero, y=one)
                    mask_conv2_ = tf.abs(frozen_matrix_conv2 - 1)

                    dilated_conv2_norm = tf.abs(dilated_conv2*mask_conv2_)
                    dilated_conv2_onedim = tf.reshape(dilated_conv2_norm, [kernel_num])
                    top_k_dilated_conv2 = tf.nn.top_k(dilated_conv2_onedim, cutoff_rank + 1).values[
                        cutoff_rank]  # e.g., 2.3
                    mask_dilated_conv2 = tf.where(dilated_conv2_norm < top_k_dilated_conv2, x=zero, y=one)
                    # mask_dilated_conv2 = tf.where(dilated_conv2_norm < top_k_dilated_conv2, x=one, y=zero)

                    self.mask_val_list_task.append(mask_dilated_conv1)
                    self.mask_val_list_task.append(mask_dilated_conv2)


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








