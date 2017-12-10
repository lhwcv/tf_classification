##  MobileNet on [Dogs Vs Cats](https://www.kaggle.com/c/dogs-vs-cats)
Please refer Google's paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
![](https://github.com/lhwcv/tf_classification/blob/master/MobileNet/imgs/base_module.png)


## Accuracy on Dogs Vs Cats Dataset
We using 2w images to train and 5k images to val in train.zip</br>
(Here we only trained 8K steps with batch size 64)</br>
| Model | Width Multiplier | Accuracy-Top1
|--------|:---------:|:------:|
| MobileNet |1.0| 95.8% |

You  can get the trained model in [BaiduYun](https://pan.baidu.com/s/1dEGFXtf) 
and  train.zip in [Dogs Vs Cats](https://www.kaggle.com/c/dogs-vs-cats)

First you need generate tfrecords by build_tfrecords.py</br>
and then just simply modify train.py to eval or train</br>  
