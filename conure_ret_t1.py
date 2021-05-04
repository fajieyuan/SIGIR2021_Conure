import tensorflow as tf
import data_loader_t1 as data_loader
import generator_prune_t1 as generator_recsys
import math
import numpy as np
import argparse
import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    parser.add_argument('--datapath', type=str, default='Data/Session/original_desen_pretrain.csv',
                        help='data path')
    parser.add_argument('--datapath_index', type=str, default='Data/Session/index.csv',
                        help='data path')
    parser.add_argument('--eval_iter', type=int, default=4000,
                        help='Sample generator output evry x steps')
    parser.add_argument('--save_para_every', type=int, default=4000,
                        help='save model parameters every')
    parser.add_argument('--tt_percentage', type=float, default=0.2,
                        help='0.2 means 80% training 20% testing')
    parser.add_argument('--has_positionalembedding', type=bool, default=False,
                        help='whether contains positional embedding before performing cnnn')
    parser.add_argument('--max_position', type=int, default=1000,
                         help='maximum number of for positional embedding, it has to be larger than the sequence lens')

    args = parser.parse_args()

    dl = data_loader.Data_Loader({'model_type': 'generator', 'dir_name': args.datapath,'dir_name_index': args.datapath_index})

    all_samples = dl.item
    items = dl.item_dict
    bigemb= dl.embed_len
    print "len(source)",len(items)
    print "len(allitems)", bigemb


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
        'dilations': [1,4,1,4,1,4,1,4,],
        'kernel_size': 3,
        'learning_rate':0.001,
        'batch_size':32,
        'iterations':10,
        'has_positionalembedding': args.has_positionalembedding,
        'max_position': args.max_position,
        'is_negsample':True, #False denotes using full softmax
        'taskID': config.taskID_1st #this is the start taskID index from 10001  i.e., ID=1
    }

    sess = tf.Session()
    itemrec = generator_recsys.NextItNet_Decoder(model_para)
    itemrec.train_graph(model_para['is_negsample'],ispre=False)

    trainable_vars = tf.trainable_variables()
    weight = [v for v in trainable_vars if v.name.find("weight") != -1]
    softmax_var = [v for v in trainable_vars if v.name.find("softmax") != -1]
    ln_var_all = [v for v in trainable_vars if v.name.find("layer_norm") != -1]

    optimizer = tf.train.AdamOptimizer(model_para['learning_rate'], beta1=args.beta1).minimize(itemrec.loss,
                                                                                               var_list=[weight,softmax_var])
    itemrec.predict_graph(model_para['is_negsample'], reuse=True,ispre=False)

    init = tf.global_variables_initializer()
    trainable_vars = tf.trainable_variables()
    allable_vars=tf.all_variables()



    variables_to_restore=trainable_vars

    mask_var = [v for v in allable_vars if v.name.find("mask_filter") != -1]
    variables_to_restore.extend(mask_var)
    # layer_norm2 = [v for v in trainable_vars if v.name.find("layer_norm2") != -1]  # not very necessary for re-training
    sess.run(init)

    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, "Data/Models/generation_model_t1/model_nextitnet_transfer_pretrain.ckpt")
    # print "weight", sess.run(variables_to_restore[41])
    saver_ft = tf.train.Saver()
    numIters = 1

    for iter in range(model_para['iterations']):
        batch_no = 0
        batch_size = model_para['batch_size']
        while (batch_no + 1) * batch_size < train_set.shape[0]:
            item_batch = train_set[batch_no * batch_size: (batch_no + 1) * batch_size, :]
            _, loss = sess.run(
                [optimizer, itemrec.loss],
                feed_dict={
                    itemrec.itemseq_input: item_batch
                })
            if numIters % args.eval_iter == 0:
                print "-------------------------------------------------------train1"
                print "LOSS: {}\tITER: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                    loss, iter, batch_no, numIters, train_set.shape[0] / batch_size)

            if numIters % args.eval_iter == 0:
                print "-------------------------------------------------------test1"
                if (batch_no + 1) * batch_size < valid_set.shape[0]:
                    item_batch = valid_set[(batch_no) * batch_size: (batch_no + 1) * batch_size, :]
                loss = sess.run(
                    [itemrec.loss_test],
                    feed_dict={
                        itemrec.input_predict: item_batch
                    })
                print "LOSS: {}\tITER: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                    loss, iter, batch_no, numIters, valid_set.shape[0] / batch_size)
            batch_no += 1
            if numIters % args.eval_iter == 0:
                batch_no_test = 0
                batch_size_test = batch_size*1
                curr_preds_5=[]
                rec_preds_5=[]
                ndcg_preds_5=[]
                while (batch_no_test + 1) * batch_size_test < valid_set.shape[0]:
                    if (numIters / (args.eval_iter) < 5):
                        if (batch_no_test > 9000):
                            break
                    else:
                        if (batch_no_test > 9000):
                            break
                    item_batch = valid_set[batch_no_test * batch_size_test: (batch_no_test + 1) * batch_size_test, :]
                    [top_k_batch] = sess.run(
                        [itemrec.top_k],
                        feed_dict={
                            itemrec.input_predict: item_batch,
                        })
                    top_k = np.squeeze(top_k_batch[1])
                    for bi in range(top_k.shape[0]):
                        pred_items_5 = top_k[bi][:5]
                        true_item = item_batch[bi][-1]
                        predictmap_5 = {ch: i for i, ch in enumerate(pred_items_5)}

                        rank_5 = predictmap_5.get(true_item)
                        if rank_5 == None:
                            curr_preds_5.append(0.0)
                            rec_preds_5.append(0.0)
                            ndcg_preds_5.append(0.0)
                        else:
                            MRR_5 = 1.0 / (rank_5 + 1)
                            Rec_5 = 1.0  # 3
                            ndcg_5 = 1.0 / math.log(rank_5 + 2, 2)  # 3
                            curr_preds_5.append(MRR_5)
                            rec_preds_5.append(Rec_5)
                            ndcg_preds_5.append(ndcg_5)
                    batch_no_test += 1
                    if (numIters / (args.eval_iter) < 5):
                        if (batch_no_test == 9000):
                            print "mrr_5:", sum(curr_preds_5) / float(len(curr_preds_5)), "hit_5:", sum(rec_preds_5) / float(
                                len(rec_preds_5)),   "ndcg_5:", sum(ndcg_preds_5) / float(
                                len(ndcg_preds_5))
                    else:
                        if (batch_no_test == 9000):
                            print "mrr_5:", sum(curr_preds_5) / float(len(curr_preds_5)), "hit_5:", sum(rec_preds_5) / float(
                                len(rec_preds_5)),  "ndcg_5:", sum(ndcg_preds_5) / float(
                                len(ndcg_preds_5))
            if numIters % args.save_para_every == 0:
                save_path = saver_ft.save(sess,
                                       "Data/Models/generation_model_finetune_t1/model_nextitnet_transfer_pretrain.ckpt".format(iter, numIters))
                print "Save models done!"
            numIters += 1


if __name__ == '__main__':
    main()
