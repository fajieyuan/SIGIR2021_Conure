# SIGIR2021_Conure
# One Person, One Model, One World: Learning Continual User Representation without Forgetting
Posts:
https://zhuanlan.zhihu.com/p/437671278

## Pytorch Code is here: https://github.com/yuangh-x/2022-NIPS-Tenrec


🤗 **New Resources**: four Large-scale datasets for evaluating foundation / transferable / cross-domain recommendaiton models.

-  **PixelRec**: https://github.com/westlake-repl/PixelRec

-  **MicroLens(DeepMind Talk)**: https://github.com/westlake-repl/MicroLens
  
-  **NineRec(TPAMI)**: https://github.com/westlake-repl/NineRec 

-  **Tenrec(NeurIPS)**: https://github.com/yuangh-x/2022-NIPS-Tenrec


```
@inproceedings{yuan2021one,
  title={One person, one model, one world: Learning continual user representation without forgetting},
  author={Yuan, Fajie and Zhang, Guoxiao and Karatzoglou, Alexandros and Jose, Joemon and Kong, Beibei and Li, Yudong},
  booktitle={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={696--705},
  year={2021}
}
```

### NextItNet pytorch version: https://github.com/syiswell/NextItNet-Pytorch
### GRec pytorch version: https://github.com/hangjunguo/GRec

---------------------------------------------------

conure_tp_t1.py: Conure is trained on Task1, and once converged it will be pruned.

conure_ret_t1.py: Conure retrains the pruned architecture of Task1

conure_tp_t2.py: Conure is trained on Task2 and once converged it will be pruned.

conure_ret_t2.py: Conure retrains the pruned architecture of Task2






## Large-scale Recommendation Dataset for pretraining，transfer learning (crosss-domain recommendation) and user representation learning：

Download the TTL dataset from:  https://drive.google.com/file/d/1imhHUsivh6oMEtEW-RwVc4OsDqn-xOaP/view?usp=sharin
            or the ML dataset from: https://drive.google.com/file/d/1-_KmnZFaOdH11keLYVcgkf-kW_BaM266/view?usp=sharing

TTL Dataset: 可用于推荐系统预训练，迁移学习，跨域推荐，冷启动推荐，用户表征学习，自监督学习等任务。
## Running our code: 
Put these dataset on Data/Session

FOllowing these steps:

python conure_tp_t1.py          After convergence (it takes more than 24 hours for training). We suggest 4 iterations. Parameters will be automatioally saved.

python conure_ret_t1.py          You can manually stop this job if the results are satisfied (better than results reported in conure_tp_t1.py). Parameters will be automatioally saved.

python conure_tp_t2.py

python conure_ret_t2.py 

python conure_tp_t3.py

python conure_ret_t3.py  

python conure_tp_t4.py

python conure_ret_t4.py  


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

#  Hiring
If you want to work with Fajie https://fajieyuan.github.io/, Please contact him by email yuanfajie@westlake.edu.cn. His lab is now recruiting visiting students, interns, research assistants, posdocs, and research scientists. You can also contact him if you want to  pursue a Phd degree at Westlake University. 
Please feel free to talk to him (by weichat: wuxiangwangyuan) if you have ideas or papers for collaboration. He is open to various collaborations. 
西湖大学原发杰团队长期招聘：推荐系统和生物信息（尤其蛋白质相关）方向 ，科研助理，博士生，博后，访问学者，研究员系列。
