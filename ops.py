import tensorflow as tf
import math
import numpy as np
import config


#config e.g. dilations: [1,4,16,] In most cases[1,4,] is enough
def nextitnet_residual_block(input_, dilation, layer_id,
                            residual_channels, kernel_size,
                            causal=True, train=True):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):

        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)


        dilated_conv = conv1d(relu1,  residual_channels,
                              2 *dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        return input_ + relu1

#layer norm is  before cnn
def nextitnet_residual_block_beforeln(input_, dilation, layer_id,
                            residual_channels, kernel_size,taskID,
                            causal=True, train=True):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):

        input_ln = layer_norm(input_, name=str(taskID) + "_layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)

        dilated_conv = conv1d(relu1, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )

        input_ln = layer_norm(dilated_conv, name=str(taskID) + "_layer_norm2", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d(relu1,  residual_channels,
                              2 *dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )
        return input_ + dilated_conv

def nextitnet_residual_block_beforeln_rezero(input_, dilation, layer_id,
                            residual_channels, kernel_size,taskID,
                            causal=True, train=True):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):
        rez = tf.get_variable('rez', [1],
                              initializer=tf.constant_initializer(0.0))

        input_ln = layer_norm(input_, name=str(taskID) + "_layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)

        dilated_conv = conv1d(relu1, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )

        input_ln = layer_norm(dilated_conv, name=str(taskID) + "_layer_norm2", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d(relu1,  residual_channels,
                              2 *dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )
        return input_ + dilated_conv*rez



def nextitnet_residual_block_withmask(input_, dilation, layer_id,
                            residual_channels, kernel_size,taskID,
                            causal=True, train=True):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):
        dilated_conv = conv1d_mask(input_, residual_channels,
                              dilation, kernel_size,taskID,
                              causal=causal,
                              name="dilated_conv1"
                              )
        #not useful for training, but to make sure these variables are available
        for index in range(taskID-config.taskID_1st):
            t_name=config.taskID_1st+index
            layernorm_task = layer_norm(dilated_conv, name=str(t_name)+"_layer_norm1", trainable=train)

        input_ln = layer_norm(dilated_conv, name=str(taskID)+"_layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)


        dilated_conv = conv1d_mask(relu1,  residual_channels,
                              2 *dilation, kernel_size,taskID,
                              causal=causal,
                              name="dilated_conv2"
                              )
        #not useful for training, but to make sure these variables are available
        for index in range(taskID - config.taskID_1st):
            t_name = config.taskID_1st + index
            layernorm_task = layer_norm(dilated_conv, name=str(t_name) + "_layer_norm2", trainable=train)
        input_ln = layer_norm(dilated_conv, name=str(taskID)+"_layer_norm2", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        return input_ + relu1


def nextitnet_residual_block_withmask_beforeln(input_, dilation, layer_id,
                            residual_channels, kernel_size,taskID,
                            causal=True, train=True):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    # not useful for training, but to make sure these variables are available
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):
        # for index in range(taskID - config.taskID_1st):
        #     t_name = config.taskID_1st + index
        #     layernorm_task = layer_norm(input_, name=str(t_name) + "_layer_norm1", trainable=train)
        input_ln = layer_norm(input_, name=str(config.taskID_1st) + "_layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d_mask(relu1, residual_channels,
                              dilation, kernel_size,taskID,
                              causal=causal,
                              name="dilated_conv1"
                              )
        # not useful for training, but to make sure these variables are available
        # for index in range(taskID - config.taskID_1st):
        #     t_name = config.taskID_1st + index
        #     layernorm_task = layer_norm(dilated_conv, name=str(t_name) + "_layer_norm2", trainable=train)
        input_ln = layer_norm(dilated_conv, name=str(config.taskID_1st) + "_layer_norm2", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d_mask(relu1,  residual_channels,
                              2 *dilation, kernel_size,taskID,
                              causal=causal,
                              name="dilated_conv2"
                              )
        return input_ + dilated_conv

def nextitnet_residual_block_withmask_beforeln_rezero(input_, dilation, layer_id,
                            residual_channels, kernel_size,taskID,
                            causal=True, train=True):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    # not useful for training, but to make sure these variables are available
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):
        # for index in range(taskID - config.taskID_1st):
        #     t_name = config.taskID_1st + index
        #     layernorm_task = layer_norm(input_, name=str(t_name) + "_layer_norm1", trainable=train)
        rez = tf.get_variable('rez', [1],
                              initializer=tf.constant_initializer(0.0))

        input_ln = layer_norm(input_, name=str(config.taskID_1st) + "_layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d_mask(relu1, residual_channels,
                              dilation, kernel_size,taskID,
                              causal=causal,
                              name="dilated_conv1"
                              )
        # not useful for training, but to make sure these variables are available
        # for index in range(taskID - config.taskID_1st):
        #     t_name = config.taskID_1st + index
        #     layernorm_task = layer_norm(dilated_conv, name=str(t_name) + "_layer_norm2", trainable=train)
        input_ln = layer_norm(dilated_conv, name=str(config.taskID_1st) + "_layer_norm2", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d_mask(relu1,  residual_channels,
                              2 *dilation, kernel_size,taskID,
                              causal=causal,
                              name="dilated_conv2"
                              )
        return input_ + dilated_conv*rez
#almost the same with nextitnet_residual_block_withmask with only difference conv1d_fine instead of conv1d_mask
#finetune for the  task after the first
def nextitnet_residual_block_withmask_fine(input_, dilation, layer_id,
                            residual_channels, kernel_size,taskID,
                            causal=True, train=True):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):
        dilated_conv = conv1d_fine(input_, residual_channels,
                              dilation, kernel_size,taskID,
                              causal=causal,
                              name="dilated_conv1"
                              )
        for index in range(taskID-config.taskID_1st):
            t_name=config.taskID_1st+index
            layernorm_task = layer_norm(dilated_conv, name=str(t_name)+"_layer_norm1", trainable=train)

        input_ln = layer_norm(dilated_conv, name=str(taskID)+"_layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)



        dilated_conv = conv1d_fine(relu1,  residual_channels,
                              2 *dilation, kernel_size,taskID,
                              causal=causal,
                              name="dilated_conv2"
                              )
        for index in range(taskID - config.taskID_1st):
            t_name = config.taskID_1st + index
            layernorm_task = layer_norm(dilated_conv, name=str(t_name) + "_layer_norm2", trainable=train)
        input_ln = layer_norm(dilated_conv, name=str(taskID)+"_layer_norm2", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        return input_ + relu1

def nextitnet_residual_block_withmask_pre_beforeln(input_, dilation, layer_id,
                            residual_channels, kernel_size,taskID,
                            causal=True, train=True):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):
        # for index in range(taskID - config.taskID_1st):
        #     t_name = config.taskID_1st + index
        #     layernorm_task = layer_norm(input_, name=str(t_name) + "_layer_norm1", trainable=train)
        input_ln = layer_norm(input_, name=str(config.taskID_1st) + "_layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d_pre(relu1, residual_channels,
                              dilation, kernel_size,taskID,
                              causal=causal,
                              name="dilated_conv1"
                              )
        # for index in range(taskID - config.taskID_1st):
        #     t_name = config.taskID_1st + index
        #     layernorm_task = layer_norm(dilated_conv, name=str(t_name) + "_layer_norm2", trainable=train)
        input_ln = layer_norm(dilated_conv, name=str(config.taskID_1st)+"_layer_norm2", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d_pre(relu1,  residual_channels,
                              2 *dilation, kernel_size,taskID,
                              causal=causal,
                              name="dilated_conv2"
                              )
        return input_ + dilated_conv

def nextitnet_residual_block_withmask_pre_beforeln_rezero(input_, dilation, layer_id,
                            residual_channels, kernel_size,taskID,
                            causal=True, train=True):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):
        # for index in range(taskID - config.taskID_1st):
        #     t_name = config.taskID_1st + index
        #     layernorm_task = layer_norm(input_, name=str(t_name) + "_layer_norm1", trainable=train)
        rez = tf.get_variable('rez', [1],
                              initializer=tf.constant_initializer(0.0))
        input_ln = layer_norm(input_, name=str(config.taskID_1st) + "_layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d_pre(relu1, residual_channels,
                              dilation, kernel_size,taskID,
                              causal=causal,
                              name="dilated_conv1"
                              )
        # for index in range(taskID - config.taskID_1st):
        #     t_name = config.taskID_1st + index
        #     layernorm_task = layer_norm(dilated_conv, name=str(t_name) + "_layer_norm2", trainable=train)
        input_ln = layer_norm(dilated_conv, name=str(config.taskID_1st)+"_layer_norm2", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d_pre(relu1,  residual_channels,
                              2 *dilation, kernel_size,taskID,
                              causal=causal,
                              name="dilated_conv2"
                              )
        return input_ + dilated_conv*rez

#for testing, i.e., peterrec
def nextitnet_residual_block_withmask_pre_beforeln_fortest(input_, dilation, layer_id,
                            residual_channels, kernel_size,taskID,
                            causal=True, train=True):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):
        # for index in range(taskID - config.taskID_1st):
        #     t_name = config.taskID_1st + index
        #     layernorm_task = layer_norm(input_, name=str(t_name) + "_layer_norm1", trainable=train)
        input_ln = layer_norm(input_, name=str(config.taskID_1st) + "_layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d_pre_peterrec(relu1, residual_channels,
                              dilation, kernel_size,taskID,
                              causal=causal,
                              name="dilated_conv1"
                              )
        # for index in range(taskID - config.taskID_1st):
        #     t_name = config.taskID_1st + index
        #     layernorm_task = layer_norm(dilated_conv, name=str(t_name) + "_layer_norm2", trainable=train)
        input_ln = layer_norm(dilated_conv, name=str(config.taskID_1st)+"_layer_norm2", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d_pre_peterrec(relu1,  residual_channels,
                              2 *dilation, kernel_size,taskID,
                              causal=causal,
                              name="dilated_conv2"
                              )
        return input_ + dilated_conv

#no other tasks
def nextitnet_residual_block_noothertask(input_, dilation, layer_id,
                            residual_channels, kernel_size,taskID,
                            causal=True, train=True):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):
        # for index in range(taskID - config.taskID_1st):
        #     t_name = config.taskID_1st + index
        #     layernorm_task = layer_norm(input_, name=str(t_name) + "_layer_norm1", trainable=train)
        input_ln = layer_norm(input_, name=str(config.taskID_1st) + "_layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d(relu1, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        # for index in range(taskID - config.taskID_1st):
        #     t_name = config.taskID_1st + index
        #     layernorm_task = layer_norm(dilated_conv, name=str(t_name) + "_layer_norm2", trainable=train)
        input_ln = layer_norm(dilated_conv, name=str(config.taskID_1st)+"_layer_norm2", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d(relu1,  residual_channels,
                              2 *dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )
        return input_ + dilated_conv



def nextitnet_residual_block_withmask_fine_beforeln(input_, dilation, layer_id,
                            residual_channels, kernel_size,taskID,
                            causal=True, train=True):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):
        # for index in range(taskID - config.taskID_1st):
        #     t_name = config.taskID_1st + index
        #     layernorm_task = layer_norm(input_, name=str(t_name) + "_layer_norm1", trainable=train)
        input_ln = layer_norm(input_, name=str(config.taskID_1st) + "_layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d_fine(relu1, residual_channels,
                              dilation, kernel_size,taskID,
                              causal=causal,
                              name="dilated_conv1"
                              )
        # for index in range(taskID - config.taskID_1st):
        #     t_name = config.taskID_1st + index
        #     layernorm_task = layer_norm(dilated_conv, name=str(t_name) + "_layer_norm2", trainable=train)
        input_ln = layer_norm(dilated_conv, name=str(config.taskID_1st)+"_layer_norm2", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d_fine(relu1,  residual_channels,
                              2 *dilation, kernel_size,taskID,
                              causal=causal,
                              name="dilated_conv2"
                              )
        return input_ + dilated_conv

def nextitnet_residual_block_withmask_fine_beforeln_rezero(input_, dilation, layer_id,
                            residual_channels, kernel_size,taskID,
                            causal=True, train=True):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):
        # for index in range(taskID - config.taskID_1st):
        #     t_name = config.taskID_1st + index
        #     layernorm_task = layer_norm(input_, name=str(t_name) + "_layer_norm1", trainable=train)
        rez = tf.get_variable('rez', [1],
                              initializer=tf.constant_initializer(0.0))
        input_ln = layer_norm(input_, name=str(config.taskID_1st) + "_layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d_fine(relu1, residual_channels,
                              dilation, kernel_size,taskID,
                              causal=causal,
                              name="dilated_conv1"
                              )
        # for index in range(taskID - config.taskID_1st):
        #     t_name = config.taskID_1st + index
        #     layernorm_task = layer_norm(dilated_conv, name=str(t_name) + "_layer_norm2", trainable=train)
        input_ln = layer_norm(dilated_conv, name=str(config.taskID_1st)+"_layer_norm2", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d_fine(relu1,  residual_channels,
                              2 *dilation, kernel_size,taskID,
                              causal=causal,
                              name="dilated_conv2"
                              )
        return input_ + dilated_conv*rez
#Aggregated Residual Transformations for Deep Neural Networks block1  =resnet if cardinality==1
def get_mp(input_,cardinality=32, name="mp"):
    with tf.variable_scope(name):
        residual_channels = input_.get_shape()[-1]
        hidden_size = residual_channels / (cardinality * 8)
        blocksets = list()
        for i in range(cardinality):
            conv_down_i = conv1d(input_, hidden_size,
                               name="mp_conv1_down_{}".format(i)
                               )
            conv_down_i = gelu(conv_down_i)
            conv_up_i = conv1d(conv_down_i, residual_channels,
                             name="mp_conv1_up_{}".format(i)
                             )
            blocksets.append(conv_up_i)

        output = tf.add_n(blocksets)
        return input_+output


# peter_2mp_parallel
def peter_2mp_parallel(input_, dilation, layer_id,
                            residual_channels, kernel_size,
                            causal=True, train=True,mp=True,cardinality=32):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):

        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        if mp:
            after_adapter = get_mp(input_, cardinality,name="mp_1")
            dilated_conv = tf.add(dilated_conv, after_adapter)


        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        #input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?
        relu1 = tf.nn.relu(input_ln)


        dilated_conv = conv1d(relu1,  residual_channels,
                              2 *dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        if mp:
            after_adapter=get_mp(relu1,cardinality,name="mp_2")
            dilated_conv = tf.add(dilated_conv, after_adapter)


        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        return relu1+input_


# peter_2mp_parallel  peter_2mp_serial
def peter_2mp_serial(input_, dilation, layer_id,
                            residual_channels, kernel_size,
                            causal=True, train=True,mp=True,cardinality=32):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):
        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        if mp:
            after_adapter = get_mp(dilated_conv, cardinality,name="mp_1")
            dilated_conv = after_adapter
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d(relu1,  residual_channels,
                              2 *dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )
        if mp:
            after_adapter=get_mp(dilated_conv,cardinality,name="mp_2")
            dilated_conv = after_adapter
        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        return input_ + relu1


def peter_mp_serial(input_, dilation, layer_id,
                            residual_channels, kernel_size,
                            causal=True, train=True,mp=True,cardinality=32):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):

        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)


        dilated_conv = conv1d(relu1,  residual_channels,
                              2 *dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        if mp:
            after_adapter=get_mp(dilated_conv,cardinality)
            dilated_conv = after_adapter


        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        relu1 = tf.nn.relu(input_ln)

        return input_ + relu1


def peter_mp_serial_oneblock_beforeln(input_, dilation, layer_id,
                                      residual_channels, kernel_size,
                                      causal=True, train=True, mp=True, cardinality=32):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name, reuse=tf.AUTO_REUSE):
        input_ln = layer_norm(input_, name=str(config.taskID_1st) + "_layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d(relu1, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )

        input_ln = layer_norm(dilated_conv, name=str(config.taskID_1st) + "_layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d(relu1, residual_channels,
                              2 * dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        if mp:
            after_adapter = get_mp(dilated_conv, cardinality)
            dilated_conv = after_adapter

        return input_ + dilated_conv

def conv1d(input_, output_channels,
           dilation=1, kernel_size=1, causal=False,
           name="dilated_conv"):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [1, kernel_size, input_.get_shape()[-1], output_channels],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
        bias = tf.get_variable('bias', [output_channels],
                               initializer=tf.constant_initializer(0.0))
        if causal:
            padding = [[0, 0], [(kernel_size - 1) * dilation, 0], [0, 0]]
            padded = tf.pad(input_, padding)
            input_expanded = tf.expand_dims(padded, dim=1)
            out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='VALID') + bias
        else:
            input_expanded = tf.expand_dims(input_, dim=1)
            # out = tf.nn.conv2d(input_expanded, weight, strides=[1, 1, 1, 1], padding="SAME") + bias
            out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='SAME') + bias
        return tf.squeeze(out, [1])

#load mask for transformer ffn
def conv1d_loadmask(input_, output_channels,
           dilation=1, kernel_size=1, causal=False, taskID = 1,
           name="dilated_conv",pretrain=True):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [1, kernel_size, input_.get_shape()[-1], output_channels],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
        if pretrain:
            weight = maskload_pretrain(name='weight', weight=weight, taskID=taskID)
        else:
            weight = maskload_retrain(name='weight', weight=weight, taskID=taskID)


        bias = tf.get_variable('bias', [output_channels],
                               initializer=tf.constant_initializer(0.0))
        if causal:
            padding = [[0, 0], [(kernel_size - 1) * dilation, 0], [0, 0]]
            padded = tf.pad(input_, padding)
            input_expanded = tf.expand_dims(padded, dim=1)
            out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='VALID') + bias
        else:
            input_expanded = tf.expand_dims(input_, dim=1)
            # out = tf.nn.conv2d(input_expanded, weight, strides=[1, 1, 1, 1], padding="SAME") + bias
            out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='SAME') + bias
        return tf.squeeze(out, [1])



def conv1d_mask(input_, output_channels,
           dilation=1, kernel_size=1,taskID=config.taskID_1st, causal=False,
           name="dilated_conv"):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [1, kernel_size, input_.get_shape()[-1], output_channels],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
        bias = tf.get_variable('bias', [output_channels],
                               initializer=tf.constant_initializer(0.0),trainable=False)
        with tf.variable_scope("mask_filter"):
            init_zeros = tf.zeros_initializer()
            one = tf.ones_like(weight)
            zero = tf.zeros_like(weight)
            #restore previous masks but will not be used
            for index in range(taskID - config.taskID_1st):
                t_name = config.taskID_1st + index
                t_name= "{}_mask_val".format(t_name)
                #be careful we only use the mask matrix of the last task,but we need restore all matrices from previous tasks
                mask_val = tf.get_variable(t_name, [1, kernel_size, input_.get_shape()[-1], output_channels],
                                           initializer=init_zeros, trainable=False)
            mask_val_ = tf.abs(mask_val - 1)#reverse
            # weight = weight * mask_val
            weight=tf.stop_gradient(weight * mask_val)+weight* mask_val_
            # weight = tf.stop_gradient(weight * mask_val+weight* mask_val_)
            # weight=weight * mask_val+weight* mask_val_
        if causal:
            padding = [[0, 0], [(kernel_size - 1) * dilation, 0], [0, 0]]
            padded = tf.pad(input_, padding)
            input_expanded = tf.expand_dims(padded, dim=1)
            out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='VALID') + bias
        else:
            input_expanded = tf.expand_dims(input_, dim=1)
            # out = tf.nn.conv2d(input_expanded, weight, strides=[1, 1, 1, 1], padding="SAME") + bias
            out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='SAME') + bias
        return tf.squeeze(out, [1])

def conv1d_pre(input_, output_channels,
           dilation=1, kernel_size=1,taskID=config.taskID_1st, causal=False,
           name="dilated_conv"):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [1, kernel_size, input_.get_shape()[-1], output_channels],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
        bias = tf.get_variable('bias', [output_channels],
                               initializer=tf.constant_initializer(0.0),trainable=False)
        with tf.variable_scope("mask_filter"):
            init_zeros = tf.zeros_initializer()
            one = tf.ones_like(weight)
            zero = tf.zeros_like(weight)
            weight_ = zero
            task_count = taskID - config.taskID_1st
            mask_val_list=[]
            for index in range(task_count):
                t_name = config.taskID_1st + index
                t_name = "{}_mask_val".format(t_name)
                # be careful we only use the mask matrix of the last task,but we need restore all matrices from previous tasks
                mask_val = tf.get_variable(t_name, [1, kernel_size, input_.get_shape()[-1], output_channels],
                                           initializer=init_zeros, trainable=False)
                mask_val_list.append(mask_val)
            frozen_mask =zero
            for mask in mask_val_list:
                frozen_mask+=mask
            frozen_mask_ = tf.abs(frozen_mask - 1)  # reverse
            weight_=tf.stop_gradient(weight*frozen_mask)+weight*frozen_mask_
            weight = weight_

        if causal:
            padding = [[0, 0], [(kernel_size - 1) * dilation, 0], [0, 0]]
            padded = tf.pad(input_, padding)
            input_expanded = tf.expand_dims(padded, dim=1)
            out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='VALID') + bias
        else:
            input_expanded = tf.expand_dims(input_, dim=1)
            # out = tf.nn.conv2d(input_expanded, weight, strides=[1, 1, 1, 1], padding="SAME") + bias
            out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='SAME') + bias
        return tf.squeeze(out, [1])

def conv1d_pre_peterrec(input_, output_channels,
           dilation=1, kernel_size=1,taskID=config.taskID_1st, causal=False,
           name="dilated_conv"):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [1, kernel_size, input_.get_shape()[-1], output_channels],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
        bias = tf.get_variable('bias', [output_channels],
                               initializer=tf.constant_initializer(0.0),trainable=False)
        with tf.variable_scope("mask_filter"):
            init_zeros = tf.zeros_initializer()
            one = tf.ones_like(weight)
            zero = tf.zeros_like(weight)
            weight_ = zero
            task_count = taskID - config.taskID_1st
            mask_val_list=[]
            for index in range(task_count):
                t_name = config.taskID_1st + index
                t_name = "{}_mask_val".format(t_name)
                # be careful we only use the mask matrix of the last task,but we need restore all matrices from previous tasks
                mask_val = tf.get_variable(t_name, [1, kernel_size, input_.get_shape()[-1], output_channels],
                                           initializer=init_zeros, trainable=False)
                mask_val_list.append(mask_val)
            frozen_mask =zero
            for mask in mask_val_list:
                frozen_mask+=mask
            frozen_mask_ = tf.abs(frozen_mask - 1)  # reverse
            # weight_=tf.stop_gradient(weight*frozen_mask)+weight*frozen_mask_
            weight_ = weight * frozen_mask + weight * frozen_mask_
            weight = weight_

        if causal:
            padding = [[0, 0], [(kernel_size - 1) * dilation, 0], [0, 0]]
            padded = tf.pad(input_, padding)
            input_expanded = tf.expand_dims(padded, dim=1)
            out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='VALID') + bias
        else:
            input_expanded = tf.expand_dims(input_, dim=1)
            # out = tf.nn.conv2d(input_expanded, weight, strides=[1, 1, 1, 1], padding="SAME") + bias
            out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='SAME') + bias
        return tf.squeeze(out, [1])

def conv1d_fine(input_, output_channels,
           dilation=1, kernel_size=1, taskID=config.taskID_1st,causal=False,
           name="dilated_conv"):

    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [1, kernel_size, input_.get_shape()[-1], output_channels],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
        bias = tf.get_variable('bias', [output_channels],
                               initializer=tf.constant_initializer(0.0),trainable=False)
        with tf.variable_scope("mask_filter"):
            init_zeros = tf.zeros_initializer()
            one = tf.ones_like(weight)
            zero = tf.zeros_like(weight)
            weight_=zero
            task_count=taskID - config.taskID_1st+1
            for index in range(task_count):
                t_name = config.taskID_1st + index
                t_name = "{}_mask_val".format(t_name)
                # be careful we only use the mask matrix of the last task,but we need restore all matrices from previous tasks
                mask_val = tf.get_variable(t_name, [1, kernel_size, input_.get_shape()[-1], output_channels],
                                           initializer=init_zeros, trainable=False)
                if index<taskID - config.taskID_1st:
                    weight_mask=tf.stop_gradient(weight * mask_val)
                    # weight_mask = weight * mask_val
                    weight_+=weight_mask
                else:
                    weight_mask =weight* mask_val
                    weight_ += weight_mask
            weight=weight_
        # 'nextitnet_residual_blockdecoder_layer_0_1/dilated_conv1/mask_filter/mask_val:0'
        if causal:
            padding = [[0, 0], [(kernel_size - 1) * dilation, 0], [0, 0]]
            padded = tf.pad(input_, padding)
            input_expanded = tf.expand_dims(padded, dim=1)
            out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='VALID') + bias
        else:
            input_expanded = tf.expand_dims(input_, dim=1)
            # out = tf.nn.conv2d(input_expanded, weight, strides=[1, 1, 1, 1], padding="SAME") + bias
            out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='SAME') + bias
        return tf.squeeze(out, [1])


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                dropout_rate=0.2,
                is_training=True,
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        # params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
        #           "activation": tf.nn.relu, "use_bias": True}
        # outputs = tf.layers.conv1d(**params)
        outputs = conv1d(tf.nn.relu(inputs), num_units[0],name="conv1d_1")
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        # params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
        #           "activation": None, "use_bias": True}
        # outputs = tf.layers.conv1d(**params)
        outputs = conv1d(tf.nn.relu(outputs), num_units[1], name="conv1d_2")
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Residual connection
        outputs += inputs

        # Normalize
        # outputs = normalize(outputs)

    return outputs

# def feedforward_loadmask(inputs,
#                 num_units=[2048, 512],
#                 scope="multihead_attention",
#                 dropout_rate=0.2,
#                 is_training=True,
#                 reuse=None):
#     '''Point-wise feed forward net.
#
#     Args:
#       inputs: A 3d tensor with shape of [N, T, C].
#       num_units: A list of two integers.
#       scope: Optional scope for `variable_scope`.
#       reuse: Boolean, whether to reuse the weights of a previous layer
#         by the same name.
#
#     Returns:
#       A 3d tensor with the same shape and dtype as inputs
#     '''
#     with tf.variable_scope(scope, reuse=reuse):
#         # Inner layer
#         # params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
#         #           "activation": tf.nn.relu, "use_bias": True}
#         # outputs = tf.layers.conv1d(**params)
#         outputs = conv1d_loadmask(tf.nn.relu(inputs), num_units[0],name="conv1d_1")
#         outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
#         # Readout layer
#         # params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
#         #           "activation": None, "use_bias": True}
#         # outputs = tf.layers.conv1d(**params)
#         outputs = conv1d_loadmask(tf.nn.relu(outputs), num_units[1], name="conv1d_2")
#         outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
#
#         # Residual connection
#         outputs += inputs
#
#         # Normalize
#         # outputs = normalize(outputs)
#
#     return outputs


def feedforward_withmask(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                dropout_rate=0.2,
                is_training=True,
                taskID=None,
                reuse=None,
                pretrain=True):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        # params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
        #           "activation": tf.nn.relu, "use_bias": True}
        # outputs = tf.layers.conv1d(**params)
        outputs = conv1d_loadmask(tf.nn.relu(inputs), num_units[0], taskID = taskID,name="conv1d_1", pretrain=pretrain)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        # params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
        #           "activation": None, "use_bias": True}
        # outputs = tf.layers.conv1d(**params)
        outputs = conv1d_loadmask(tf.nn.relu(outputs), num_units[1],taskID = taskID, name="conv1d_2",pretrain=pretrain)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Residual connection
        outputs += inputs

        # Normalize
        # outputs = normalize(outputs)

    return outputs


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None,
                        with_qk=False):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # num_seq = queries.get_shape().as_list[1]
        num_seq = tf.shape(queries)[1]

        # Linear projections
        # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        weight_Q = tf.get_variable('weight_Q', [num_units, num_units],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
        bias_Q = tf.get_variable('bias_Q', [num_units],
                               initializer=tf.constant_initializer(0.0))
        weight_K = tf.get_variable('weight_K', [num_units, num_units],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
        bias_K = tf.get_variable('bias_K', [num_units],
                                 initializer=tf.constant_initializer(0.0))
        weight_V = tf.get_variable('weight_V', [num_units, num_units],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
        bias_V = tf.get_variable('bias_V', [num_units],
                                 initializer=tf.constant_initializer(0.0))

        queries = tf.reshape(queries, [-1, num_units])
        keys = tf.reshape(keys, [-1, num_units])


        Q = tf.matmul(queries, weight_Q)
        Q = tf.nn.bias_add(Q, bias_Q)

        K = tf.matmul(keys, weight_K )
        K = tf.nn.bias_add(K, bias_K)

        V = tf.matmul(keys, weight_V)
        V = tf.nn.bias_add(V, bias_V)

        Q = tf.reshape(Q, [-1, num_seq, num_units])
        K = tf.reshape(K, [-1, num_seq, num_units])
        V = tf.reshape(V, [-1, num_seq, num_units])

        queries= tf.reshape(queries, [-1,num_seq, num_units])
        keys = tf.reshape(keys, [-1, num_seq, num_units])

        #original implementation
        # Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            # tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        # outputs = normalize(outputs) # (N, T_q, C)

    if with_qk:
        return Q, K
    else:
        return outputs

# during retraining phase, how to load mask
def maskload_retrain(name, weight, taskID):
    with tf.variable_scope(name):
        init_zeros = tf.zeros_initializer()
        one = tf.ones_like(weight)
        zero = tf.zeros_like(weight)
        weight_ = zero
        task_count = taskID - config.taskID_1st + 1
        for index in range(task_count):
            t_name = config.taskID_1st + index
            t_name = "{}_mask".format(t_name)
            # be careful we only use the mask
            # matrix of the last task,but we
            # need restore all matrices from previous tasks
            mask_val = tf.get_variable(t_name, weight.get_shape(),
                                       initializer=init_zeros, trainable=False)
            if index < taskID - config.taskID_1st:
                weight_mask = tf.stop_gradient(weight * mask_val)
                # weight_mask = weight * mask_val
                weight_ += weight_mask
            else:
                weight_mask = weight * mask_val
                weight_ += weight_mask
        weight = weight_
    return weight

# load mask for pretraining phrase
def maskload_pretrain(name, weight, taskID):
    with tf.variable_scope(name):
        init_zeros = tf.zeros_initializer()
        one = tf.ones_like(weight)
        zero = tf.zeros_like(weight)
        weight_ = zero
        task_count = taskID - config.taskID_1st
        mask_val_list = []
        for index in range(task_count):
            t_name = config.taskID_1st + index
            t_name = "{}_mask".format(t_name)
            # be careful we only use the mask matrix of the last task,but we need restore all matrices from previous tasks
            mask_val = tf.get_variable(t_name, weight.get_shape(),
                                       initializer=init_zeros, trainable=False)
            mask_val_list.append(mask_val)
        frozen_mask = zero
        for mask in mask_val_list:
            frozen_mask += mask
        frozen_mask_ = tf.abs(frozen_mask - 1)  # reverse
        weight_ = tf.stop_gradient(weight * frozen_mask) + weight * frozen_mask_
        weight = weight_
    return weight




def multihead_attention_withmask(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None,
                        taskID=None,
                        with_qk=False,
                        pretrain=True):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # num_seq = queries.get_shape().as_list[1]
        num_seq = tf.shape(queries)[1]

        # Linear projections
        # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        weight_Q = tf.get_variable('weight_Q', [num_units, num_units],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
        bias_Q = tf.get_variable('bias_Q', [num_units],
                               initializer=tf.constant_initializer(0.0))
        weight_K = tf.get_variable('weight_K', [num_units, num_units],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
        bias_K = tf.get_variable('bias_K', [num_units],
                                 initializer=tf.constant_initializer(0.0))
        weight_V = tf.get_variable('weight_V', [num_units, num_units],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
        bias_V = tf.get_variable('bias_V', [num_units],
                                 initializer=tf.constant_initializer(0.0))
        if pretrain:
            weight_Q = maskload_pretrain(name='weight_Q', weight=weight_Q, taskID=taskID)
            weight_K = maskload_pretrain(name='weight_K', weight=weight_K, taskID=taskID)
            weight_V = maskload_pretrain(name='weight_V', weight=weight_V, taskID=taskID)
        else:
            weight_Q = maskload_retrain(name='weight_Q', weight=weight_Q, taskID=taskID)
            weight_K = maskload_retrain(name='weight_K', weight=weight_K, taskID=taskID)
            weight_V = maskload_retrain(name='weight_V', weight=weight_V, taskID=taskID)

        queries = tf.reshape(queries, [-1, num_units])
        keys = tf.reshape(keys, [-1, num_units])


        Q = tf.matmul(queries, weight_Q)
        Q = tf.nn.bias_add(Q, bias_Q)

        K = tf.matmul(keys, weight_K )
        K = tf.nn.bias_add(K, bias_K)

        V = tf.matmul(keys, weight_V)
        V = tf.nn.bias_add(V, bias_V)

        Q = tf.reshape(Q, [-1, num_seq, num_units])
        K = tf.reshape(K, [-1, num_seq, num_units])
        V = tf.reshape(V, [-1, num_seq, num_units])

        queries= tf.reshape(queries, [-1,num_seq, num_units])
        keys = tf.reshape(keys, [-1, num_seq, num_units])

        #original implementation
        # Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            # tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        # outputs = normalize(outputs) # (N, T_q, C)

    if with_qk:
        return Q, K
    else:
        return outputs


def count_varsize(var):
    ''' varaible size'''
    variable_parameters = 1
    for dim in var.get_shape():
        variable_parameters *= dim.value
    return variable_parameters



# tf.contrib.layers.layer_norm
def layer_norm(x, name, epsilon=1e-8, trainable=True):
    with tf.variable_scope(name):
        shape = x.get_shape()
        beta = tf.get_variable('beta', [int(shape[-1])],
                               initializer=tf.constant_initializer(0), trainable=trainable)
        gamma = tf.get_variable('gamma', [int(shape[-1])],
                                initializer=tf.constant_initializer(1), trainable=trainable)

        mean, variance = tf.nn.moments(x, axes=[len(shape) - 1], keep_dims=True)

        x = (x - mean) / tf.sqrt(variance + epsilon)

        return gamma * x + beta

def gelu(x):
        """Gaussian Error Linear Unit.

        This is a smoother version of the RELU.
        Original paper: https://arxiv.org/abs/1606.08415
        Args:
          x: float Tensor to perform activation.

        Returns:
          `x` with the GELU activation applied.
        """
        cdf = 0.5 * (1.0 + tf.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        return x * cdf

# def entry_stop_gradients(target, mask):
#     mask_h = tf.abs(mask-1)
#     return tf.stop_gradient(mask_h * target) + mask * target