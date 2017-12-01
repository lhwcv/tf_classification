cifar 10 上loss function 实验
(网络结构见 net.PNG)
base_lr: 0.1  采用explonential_decay   (NUM_EPOCHS_PER_DECAY = 350.0  FACTOR=0.1)

10K step == 25.6  epoch

log     20K:  0.842       40K : 0.857     100K:  0.873
log2    20K:   0.816   后续训练收敛到了NaN值(可能是偶然)
mse     20K:   0.756    40K:  0.800  
hinge2  20K:  0.362          40K: 0.4   (不知道怎么这么差,loss波动厉害，降低lr如下)
hinge2（base_lr: 0.05）  20K: 0.427          40K: 0.445  (训练数据上也很差,不知道是不是收敛得不够)