from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf

CROP_SIZE_W=24
CROP_SIZE_H=24
IMAGE_SIZE = 24
NUM_CLASSES=10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_TEST  = 10000

def decode_bin_data(filename_queue):
    '''
    :param filename_queue:  input filename eg: data_batch_1.bin
    :return: An object representing a single example
              (height,width,depth,label, uint8image
    '''
    class Cifar10Record:
        pass
    result = Cifar10Record()
    result.height=32
    result.width=32
    result.depth=3
    label_bytes=1 #  1 for cifar 10 ; 2 for cifar 100
    img_bytes= result.height*result.width*result.depth
    record_bytes= label_bytes+img_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value,tf.uint8)
    result.label = tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)
    img_data = tf.reshape(tf.strided_slice(record_bytes,[label_bytes],[label_bytes+img_bytes]),
                          [result.depth,result.height,result.width])

    result.uint8image = tf.transpose(img_data, [1, 2, 0])
    return result

def _generate_img_label_batch(img,label,min_queue_examples,batch_size,shuffle):
    num_threads=6
    if shuffle:
        imgs,labels= tf.train.shuffle_batch([img,label],batch_size=batch_size,num_threads=num_threads,
                                      capacity=min_queue_examples+3*batch_size,
                                      min_after_dequeue=min_queue_examples,
                                      )
    else:
        imgs, labels = tf.train.batch([img, label], batch_size=batch_size, num_threads=num_threads,
                                              capacity=min_queue_examples + 3 * batch_size
                                              )
    tf.summary.image('imgs',imgs)
    return imgs,tf.reshape(labels, [batch_size])

def train_input(data_dir,batch_size):
    filenames = [os.path.join(data_dir,'data_batch_%d.bin'%i) for i in range(1,6) ]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise  ValueError('File not found: '+f)
    filename_queue = tf.train.string_input_producer(filenames)
    example = decode_bin_data(filename_queue)
    image = tf.cast(example.uint8image,tf.float32 )
    distorted_image = tf.random_crop(image, [CROP_SIZE_H, CROP_SIZE_W, 3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)
    float_image = tf.image.per_image_standardization(distorted_image)
    # Set the shapes of tensors.

    float_image.set_shape([CROP_SIZE_H, CROP_SIZE_W, 3])
    example.label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    return _generate_img_label_batch(float_image,example.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)

def test_inputs(eval_data, data_dir, batch_size,visualise=False):
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in range(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TEST
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)
  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)
  # Read examples from files in the filename queue.
  read_input = decode_bin_data(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  height = CROP_SIZE_H
  width = CROP_SIZE_W
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,height, width)
  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(resized_image)
  # Set the shapes of tensors.
  float_image.set_shape([height, width, 3])
  read_input.label.set_shape([1])
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)
  if visualise:
      float_image = read_input.uint8image
  return _generate_img_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)

