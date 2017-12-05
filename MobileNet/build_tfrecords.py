import numpy as np
import cv2
import tensorflow as tf
import  os
import random


classes_lut_map= {'cat':0,'dog':1}

DOGS_VS_CATS_TRAIN_IMG_DIR = 'D:\\into_DL\\dataset\\Dogs_vs_Cats\\train\\'
RESIZE_W = 256
RESIZE_H = 256

def split_train_into_trainval(train_img_dir, val_num=5000):
    """
    (Dogs vs Cats) We using 2w of 2.5w train images to train and the rest 5k to val
    :param train_img_dir:
    :return: train_img_list, val_img_list
    """
    all_imgs_list = os.listdir(train_img_dir)
    val_imgs_list = random.sample(all_imgs_list,val_num)
    train_imgs_list = [item for item in all_imgs_list if item not in val_imgs_list]
    return train_imgs_list,val_imgs_list

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def build_tfrecords_from_img_label_list(img_list,label_list,base_dir,tfrecords_dst_filename):
    assert  len(img_list)==len(label_list)
    writer = tf.python_io.TFRecordWriter(tfrecords_dst_filename)
    for i,img_path in enumerate(img_list):
        img = cv2.imread(base_dir+img_path)
        img = cv2.resize(img,(RESIZE_W,RESIZE_H))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        height= img.shape[0]
        width = img.shape[1]
        img_raw = img.tobytes()
        example= tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'label': _int64_feature(label_list[i])
        }))
        writer.write(example.SerializeToString())
    writer.close()


def get_labels_by_img_filename_list(img_list,classes_lut_map=classes_lut_map):
    img_labels = np.zeros(len(img_list), np.uint8)
    for i, c in enumerate(img_list):
        label = [classes_lut_map[k] for k in classes_lut_map if k in c][0]
        img_labels[i] = label
    return img_labels


def build_data():
    train_img_list,val_img_list = split_train_into_trainval(DOGS_VS_CATS_TRAIN_IMG_DIR,5000)
    print('Train images: %d, Val images: %d'%(len(train_img_list),len(val_img_list)))
    print('Build train and val tfrecords...')
    train_img_labels = get_labels_by_img_filename_list(train_img_list,classes_lut_map=classes_lut_map)
    val_img_labels = get_labels_by_img_filename_list(val_img_list, classes_lut_map=classes_lut_map)
    build_tfrecords_from_img_label_list(train_img_list, train_img_labels,
                                        DOGS_VS_CATS_TRAIN_IMG_DIR, 'train.tfrecords')
    build_tfrecords_from_img_label_list(val_img_list, val_img_labels,
                                        DOGS_VS_CATS_TRAIN_IMG_DIR, 'val.tfrecords')
    print('Write tfrecords to train.tfrecords and val.tfrecords Done!!')


##  http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz
if __name__ =='__main__':
    build_data()
