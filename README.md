# SIGIR2021_Conurre
# One Person, One Model, One World: Learning Continual User Representation without Forgetting

  
```
Please cite our paper if you use our code or datasets in your publication.
@article{yuan2020one,
  title={One Person, One Model, One World: Learning Continual User Representation without Forgetting},
  author={Yuan, Fajie and Zhang, Guoxiao and Karatzoglou, Alexandros and Jose, Joemon and Kong, Beibei and Li, Yudong},
  journal={arXiv preprint arXiv:2009.13724},
  year={2020}
}
```
## If you want to use Conure in real production system. I strongly suggest: (1) understand our code released here ; (2)using TFRecord (tf.data.Dataset) and tf.estimator to replace feed_dict (slow), which is several times faster; (3) contact yuanfajie@westlake.edu.cn if you could not achieve expected results. (E.g., No personalization for new user recommendation, 99% there are bugs in your project!!). Please also note that the code attached here was rewritten by Fajie after leaving Tencent. Though it was not the original code used for this paper, you should reproduce all results reported in the paper. 

---------------------------------------------------

conure_tp_t1.py: Conure is trained on Task1 and then is pruned after convergence.

conure_ret_t1.py: Conure retrains the pruned architecture of Task1

conure_tp_t2.py: Conure is trained on Task2 and then is pruned after convergence.

conure_ret_t2.py: Conure retrains the pruned architecture of Task2





## Running our code:

First download the dataset from:  https://drive.google.com/file/d/1imhHUsivh6oMEtEW-RwVc4OsDqn-xOaP/view?usp=sharin
Put these dataset on Data/Session

FOllowing these steps:

python conure_tp_t1.py          After convergence (it takes more than 24 hours for training). We suggest 4 iterations. Parameters will be automatioally saved.

python conure_tp_t1.py          You can manually stop this job if the results are satisfied (better than results reported in conure_tp_t1.py). Parameters will be automatioally saved.

python conure_tp_t2.py

python conure_tp_t2.py 

python conure_tp_t3.py

python conure_tp_t3.py  

python conure_tp_t4.py

python conure_tp_t4.py  


### DataSet （desensitized）Links
```
https://drive.google.com/file/d/1imhHUsivh6oMEtEW-RwVc4OsDqn-xOaP/view?usp=sharin

```




## Environments
* Tensorflow (version: 1.10.0)
* python 2.7

## Related work:
```
[1]
@inproceedings{yuan2019simple,
  title={A simple convolutional generative network for next item recommendation},
  author={Yuan, Fajie and Karatzoglou, Alexandros and Arapakis, Ioannis and Jose, Joemon M and He, Xiangnan},
  booktitle={Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining},
  pages={582--590},
  year={2019}
}
```
```
[2]
@inproceedings{yuan2020parameter,
  title={Parameter-efficient transfer from sequential behaviors for user modeling and recommendation},
  author={Yuan, Fajie and He, Xiangnan and Karatzoglou, Alexandros and Zhang, Liguang},
  booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1469--1478},
  year={2020}
}
```
```
[3]
@inproceedings{yuan2020future,
  title={Future Data Helps Training: Modeling Future Contexts for Session-based Recommendation},
  author={Yuan, Fajie and He, Xiangnan and Jiang, Haochuan and Guo, Guibing and Xiong, Jian and Xu, Zhezhao and Xiong, Yilin},
  booktitle={Proceedings of The Web Conference 2020},
  pages={303--313},
  year={2020}
}
```
```
[4]
@article{sun2020generic,
  title={A Generic Network Compression Framework for Sequential Recommender Systems},
  author={Sun, Yang and Yuan, Fajie and Yang, Ming and Wei, Guoao and Zhao, Zhou and Liu, Duo},
  journal={Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining},
  year={2020}
}
```
```
[5]
@inproceedings{yuan2016lambdafm,
  title={Lambdafm: learning optimal ranking with factorization machines using lambda surrogates},
  author={Yuan, Fajie and Guo, Guibing and Jose, Joemon M and Chen, Long and Yu, Haitao and Zhang, Weinan},
  booktitle={Proceedings of the 25th ACM International on Conference on Information and Knowledge Management},
  pages={227--236},
  year={2016}
}
```
```
[6]
@article{wang2020stackrec,
  title={StackRec: Efficient Training of Very Deep Sequential Recommender Models by Layer Stacking},
  author={Wang, Jiachun and Yuan, Fajie and Chen, Jian and Wu, Qingyao and Li, Chengmin and Yang, Min and Sun, Yang and Zhang, Guoxiao},
  journal={arXiv preprint arXiv:2012.07598},
  year={2020}
}
```


