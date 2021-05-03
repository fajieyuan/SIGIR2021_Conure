import tensorflow as tf
import data_loader_neg as data_loader
import generator_prune_regbig as generator_recsys
import time
import numpy as np
import argparse
import config


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t == s:
        t = np.random.randint(l, r)
    return t

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t == s:
        t = np.random.randint(l, r)
    return t


def random_negs(l,r,no,s):
    # set_s=set(s)
    negs = []
    for i in range(no):
        t = np.random.randint(l, r)
        # while (t in set_s):
        while (t== s):
            t = np.random.randint(l, r)
        negs.append(t)
    return negs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    #history_sequences_20181014_fajie_smalltest.csv
    parser.add_argument('--datapath', type=str, default='Data/Session/original_desen_finetune_like_nouser15per.csv',
                        help='data path')
    parser.add_argument('--datapath_index', type=str, default='Data/Session/pretrain_smal_index.csv',
                        help='data path')
    parser.add_argument('--eval_iter', type=int, default=500,
                        help='Sample generator output evry x steps')
    parser.add_argument('--save_para_every', type=int, default=500,
                        help='save model parameters every')
    parser.add_argument('--tt_percentage', type=float, default=0.2,
                        help='0.2 means 80% training 20% testing')
    parser.add_argument('--rho', type=float, default=0.3,
                        help='static sampling in LambdaFM paper')
    parser.add_argument('--is_generatesubsession', type=bool, default=False,
                        help='whether generating a subsessions, e.g., 12345-->01234,00123,00012  It may be useful for very some very long sequences')
    parser.add_argument('--has_positionalembedding', type=bool, default=False,
                        help='whether contains positional embedding before performing cnnn')
    parser.add_argument('--max_position', type=int, default=1000,
                         help='maximum number of for positional embedding, it has to be larger than the sequence lens')



    args = parser.parse_args()

    dl = data_loader.Data_Loader(
        {'model_type': 'generator', 'dir_name': args.datapath, 'dir_name_index': args.datapath_index,
         'lambdafm_rho': args.rho})

    items = dl.item_dict
    items_len = len(items)
    print "len(source)", len(items)
    # targets_len=len(targets)+items_len

    targets = dl.target_dict
    targets_len=len(targets)
    print "len(targets)", targets_len
    targets_len_nozero = targets_len - 1
    print "len(allitems)", dl.embed_len
    bigemb = dl.embed_len

    top_k=args.top_k
    all_samples = dl.example



    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    all_samples = all_samples[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(args.tt_percentage * float(len(all_samples)))
    train_set, valid_set = all_samples[:dev_sample_index], all_samples[dev_sample_index:]

    model_para = {
        'item_size': len(items),
        'bigemb':bigemb,
        'dilated_channels': 256,
        'target_item_size': targets_len,
        'dilations': [1,4,1,4,1,4,1,4,],
        'kernel_size': 3,
        'learning_rate':0.0001,
        'batch_size':512,
        'iterations':200,
        'has_positionalembedding': args.has_positionalembedding,
        'max_position': args.max_position,
        'is_negsample':True,
        'taskID':config.taskID_3rd #the second task indexing from 10001
    }

    sess = tf.Session()
    taskID=model_para['taskID']
    itemrec = generator_recsys.NextItNet_Decoder(model_para)
    itemrec.train_graph(ispre=False)

    for index in range(taskID - config.taskID_1st):
        t_name = config.taskID_1st + index
        if index == 0:
            softmax_w = tf.get_variable("softmax_w_{}".format(t_name),
                                        [model_para['item_size'], model_para['dilated_channels']],
                                        tf.float32,
                                        tf.random_normal_initializer(0.0, 0.01))
            softmax_b = tf.get_variable("softmax_b_{}".format(t_name), [model_para['item_size']], tf.float32,
                                        tf.constant_initializer(0.1))
        else:
            softmax_size = config.task_conf['task_itemsize'][index]
            softmax_w = tf.get_variable("softmax_w_{}".format(t_name),
                                        [softmax_size, model_para['dilated_channels']],
                                        tf.float32,
                                        tf.random_normal_initializer(0.0, 0.01))


    init = tf.global_variables_initializer()
    trainable_vars = tf.trainable_variables()
    allable_vars=tf.all_variables()
    variables_to_restore =trainable_vars
    bias=[v for v in allable_vars if v.name.find("bias") != -1]
    mask_var_all=[v for v in allable_vars if v.name.find("mask_val") != -1]

    # ln_var_all = [v for v in trainable_vars if v.name.find("layer_norm") != -1]
    # ln_var = [v for v in ln_var_all if v.name.find(str(taskID) + "_layer_norm") != -1]  # current
    weight = [v for v in trainable_vars if v.name.find("weight") != -1]
    softmax_name_curtask = "softmax_w_{}".format(taskID)
    softmax_var = [v for v in trainable_vars if v.name.find(softmax_name_curtask) != -1]

    variables_to_restore.extend(bias)
    variables_to_restore.extend(mask_var_all)


    sess.run(init)
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, "Data/Models/generation_model_t3/model_nextitnet_transfer_pretrain.ckpt")

    saver_ft = tf.train.Saver()
    source_item_embedding = itemrec.dilate_input
    # source_item_embedding = tf.reduce_mean(source_item_embedding, 1)
    source_item_embedding = tf.reduce_mean(source_item_embedding[:, -1:, :], 1)  # use the last token
    embedding_size = tf.shape(source_item_embedding)[-1]

    with tf.variable_scope("target-item"):

        allitem_embeddings_target = itemrec.allitem_embeddings_out  # only difference
        is_training = tf.placeholder(tf.bool, shape=())

        # training
        itemseq_input_target_pos = tf.placeholder('int32',
                                                  [None, None], name='itemseq_input_pos')
        itemseq_input_target_neg = tf.placeholder('int32',
                                                  [None, None], name='itemseq_input_neg')
        target_item_embedding_pos = tf.nn.embedding_lookup(allitem_embeddings_target,
                                                           itemseq_input_target_pos,
                                                           name="target_item_embedding_pos")
        target_item_embedding_neg = tf.nn.embedding_lookup(allitem_embeddings_target,
                                                           itemseq_input_target_neg,
                                                           name="target_item_embedding_neg")

        pos_score = source_item_embedding * tf.reshape(target_item_embedding_pos, [-1, embedding_size])
        neg_score = source_item_embedding * tf.reshape(target_item_embedding_neg, [-1, embedding_size])
        pos_logits = tf.reduce_sum(pos_score, -1)
        neg_logits = tf.reduce_sum(neg_score, -1)

        logits_2D = tf.matmul(source_item_embedding, tf.transpose(allitem_embeddings_target))
        top_k_test = tf.nn.top_k(logits_2D, k=args.top_k, name='top-k')
        tf.add_to_collection("top_k", top_k_test[1])

        # target_loss = tf.reduce_mean(
        #     - tf.log(tf.sigmoid(pos_logits) + 1e-24) -
        #     tf.log(1 - tf.sigmoid(neg_logits) + 1e-24)
        # )
        target_loss = -tf.reduce_mean(tf.log(tf.sigmoid(pos_logits - neg_logits))) + 1e-24
        reg_losses = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # reg_losses = 0.001 * tf.reduce_mean([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        target_loss += reg_losses


        loss =  target_loss
    # optimizer = tf.train.AdamOptimizer(model_para['learning_rate'], beta1=args.beta1, name='Adam2').minimize(loss)
    # optimizer = tf.train.AdamOptimizer(model_para['learning_rate'], beta1=args.beta1, name='Adam2').minimize(loss,var_list=[softmax_var])
    optimizer = tf.train.AdamOptimizer(model_para['learning_rate'], beta1=args.beta1, name='Adam2').minimize(loss,
                                                                                                             var_list=[
                                                                                                                 softmax_var,weight])

    unitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            unitialized_vars.append(var)

    initialize_op = tf.variables_initializer(unitialized_vars)
    sess.run(initialize_op)


    numIters = 1
    for iter in range(model_para['iterations']):
        batch_no = 0
        batch_size = model_para['batch_size']
        while (batch_no + 1) * batch_size < train_set.shape[0]:
            start = time.time()
            # the first n-1 is source, the last one is target
            # item_batch=[[1,2,3],[4,5,6]]
            item_batch = train_set[batch_no * batch_size: (batch_no + 1) * batch_size, :]

            pos_batch = item_batch[:, -1]  # [3 6] used for negative sampling
            source_batch = item_batch[:, :-2]  #
            pos_target = item_batch[:, -1:]  # [[3][6]]
            neg_target = np.random.choice(targets_len_nozero, len(pos_batch), p=dl.prob)
            neg_target = np.array(neg_target + 1)
            neg_target = neg_target[:, np.newaxis]
            _, loss_out = sess.run(
                [optimizer, loss],
                feed_dict={
                    itemrec.itemseq_input: item_batch,
                    itemseq_input_target_pos: pos_target,
                    itemseq_input_target_neg: neg_target

                })
            end = time.time()

            if numIters % args.eval_iter == 0:
                print "-------------------------------------------------------train1"
                print "LOSS: {}\tITER: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                    loss_out, iter, batch_no, numIters, train_set.shape[0] / batch_size)
                # print "TIME FOR BATCH", end - start
                print "TIME FOR ITER (mins)", (end - start) * (train_set.shape[0] / batch_size) / 60.0
            batch_no += 1

            if numIters % args.eval_iter == 0:
                batch_no_test = 0
                batch_size_test = batch_size * 1
                # batch_size_test =  1
                hits = []  # 1
                mrrs = []  # ---add 1

                while (batch_no_test + 1) * batch_size_test < valid_set.shape[0]:
                    if (numIters / (args.eval_iter) < 10):
                        if (batch_no_test > 95):
                            break
                    else:
                        if (batch_no_test > 95):
                            break
                    item_batch = valid_set[batch_no_test * batch_size_test: (batch_no_test + 1) * batch_size_test, :]
                    pos_batch = item_batch[:, -1]  # [3 6] used for negative sampling

                    [top_k_batch] = sess.run(
                        [top_k_test],
                        feed_dict={
                            itemrec.itemseq_input: item_batch
                            # itemseq_input_target_label: target
                        })
                    top_k = np.squeeze(
                        top_k_batch[1])  # remove one dimension since e.g., [[[1,2,4]],[[34,2,4]]]-->[[1,2,4],[34,2,4]]
                    for i in range(top_k.shape[0]):
                        top_k_per_batch = top_k[i]
                        predictmap = {ch: i for i, ch in enumerate(top_k_per_batch)}  # add 2
                        true_item = pos_batch[i]
                        rank = predictmap.get(true_item)  # add 3
                        if rank == None:
                            hits.append(0.0)
                            mrrs.append(0.0)  # add 5
                        else:
                            hits.append(1.0)
                            mrrs.append(1.0 / (rank + 1))  # add 4
                    batch_no_test += 1
                print "-------------------------------------------------------Accuracy"
                if len(hits) != 0:
                    print "Accuracy hit_n:", sum(hits) / float(len(hits)), "MRR_n:", sum(mrrs) / float(len(mrrs))  # 5

            if numIters % args.save_para_every == 0:
                # print "weght_0", sess.run(weight[11])
                save_path = saver_ft.save(sess,
                                       "Data/Models/generation_model_finetune_t3/model_nextitnet_transfer_pretrain.ckpt".format(iter, numIters))
                print "Save models done!"
            numIters += 1




if __name__ == '__main__':
    main()
