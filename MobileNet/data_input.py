import numpy as np
import tensorflow as tf

def data_augmentation(img,crop_h,crop_w):
    #angle = np.random.uniform(low=-10.0, high=10.0)

    img = tf.random_crop(img,(crop_h,crop_w,3))
    #img = tf.contrib.image.rotate(img, angle)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.image.random_brightness(img, max_delta=16. / 255.)
    return img

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 20000
def train_input(filename_queue,batch_size,crop_h,crop_w):
    reader = tf.TFRecordReader()
    _,serialized_example=reader.read(filename_queue)
    fea = tf.parse_single_example(serialized_example,
                                  features={
                                      'height':tf.FixedLenFeature([],tf.int64),
                                      'width': tf.FixedLenFeature([], tf.int64),
                                      'image_raw':tf.FixedLenFeature([],tf.string),
                                      'label': tf.FixedLenFeature([], tf.int64),
                                  })
    height = tf.cast(fea['height'], tf.int32)
    width = tf.cast(fea['width'], tf.int32)
    label = tf.cast(fea['label'], tf.int32)

    img = tf.decode_raw(fea['image_raw'], tf.uint8)
    img = tf.reshape(img, [height, width, 3])
    img = data_augmentation(img,crop_h,crop_w)
    min_fraction_of_examples_in_queue = 0.8
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                     batch_size=batch_size,
                                                     capacity=min_queue_examples+3*batch_size,
                                                     num_threads=4,
                                                     min_after_dequeue=min_queue_examples)

    return img_batch,label_batch



def val_input(filename_queue,batch_size,crop_h,crop_w):
    reader = tf.TFRecordReader()
    _,serialized_example=reader.read(filename_queue)
    fea = tf.parse_single_example(serialized_example,
                                  features={
                                      'height':tf.FixedLenFeature([],tf.int64),
                                      'width': tf.FixedLenFeature([], tf.int64),
                                      'image_raw':tf.FixedLenFeature([],tf.string),
                                      'label': tf.FixedLenFeature([], tf.int64),
                                  })
    height = tf.cast(fea['height'], tf.int32)
    width = tf.cast(fea['width'], tf.int32)
    label = tf.cast(fea['label'], tf.int32)
    img = tf.decode_raw(fea['image_raw'], tf.uint8)
    img = tf.reshape(img, [height, width, 3])
    img = tf.image.resize_image_with_crop_or_pad(img,crop_h,crop_w)
    img_batch, label_batch = tf.train.batch([img, label],
                                                batch_size=batch_size,
                                                capacity=3000,
                                                num_threads=4)
    return img_batch,label_batch



