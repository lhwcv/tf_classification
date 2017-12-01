from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import argparse
import  cifar10_input
parser= argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int, default=128,
                    help='Number of images in a batch')
parser.add_argument('--data_dir',type=str, default='../dataset/cifar10/',
                    help='cifar10 data dir ')
FLAGS = parser.parse_args()

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TEST

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1      # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _variable_with_wd(name,shape,stddev,wd ):
    var = tf.get_variable(name,shape,dtype=tf.float32,
                          initializer=tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay= tf.multiply(tf.nn.l2_loss(var),wd,name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def inference_without_fc(images,ksize,weight_decay):
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_wd('weight',[ksize,ksize,3,64],5e-2,weight_decay)
        conv = tf.nn.conv2d(images,kernel,[1,1,1,1],padding='SAME')
        bias = tf.get_variable('bias', [64],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        pred = tf.nn.bias_add(conv,bias)
        conv1 = tf.nn.relu(pred,name=scope.name)
    # overlaping pool
    pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool1')
    # LRN
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_wd('weight',[ksize,ksize,64,64],5e-2,weight_decay)
        conv = tf.nn.conv2d(norm1,kernel,[1,1,1,1],padding='SAME')
        bias = tf.get_variable('bias', [64],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        pred = tf.nn.bias_add(conv,bias)
        conv2 = tf.nn.relu(pred,name=scope.name)
    # overlaping pool
    pool2 = tf.nn.max_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool2')
    # LRN
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    return  norm2

def fcn_part(images,weight_decay):
    w = images.get_shape()[1].value

    with tf.variable_scope('fcn1') as scope:
        kernel = _variable_with_wd('weight',[w,w,128,384],5e-2,weight_decay)
        conv = tf.nn.conv2d(images,kernel,[1,1,1,1],padding='VALID')
        bias = tf.get_variable('bias', [384],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
        pred = tf.nn.bias_add(conv,bias)
        fcn1 = tf.nn.relu(pred,name=scope.name)

    with tf.variable_scope('fcn2') as scope:
        kernel = _variable_with_wd('weight',[1,1,384,192],5e-2,weight_decay)
        conv = tf.nn.conv2d(fcn1,kernel,[1,1,1,1],padding='SAME')
        bias = tf.get_variable('bias', [192],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        pred = tf.nn.bias_add(conv,bias)
        fcn2 = tf.nn.relu(pred,name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        kernel = _variable_with_wd('weight',[1,1,192,NUM_CLASSES],5e-2,weight_decay)
        conv = tf.nn.conv2d(fcn2,kernel,[1,1,1,1],padding='SAME')
        bias = tf.get_variable('bias', [NUM_CLASSES],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        pred = tf.nn.bias_add(conv,bias)
    pred = tf.squeeze(pred)
    print(pred)
    return pred

##  using Fully Convolutinal Network instead of  Fully Connected
def inference_fcn(images,weight_decay):
    with tf.variable_scope('net1'):
        x1 = inference_without_fc(images,3,weight_decay)
    with tf.variable_scope('net2'):
        x2 = inference_without_fc(images,5,weight_decay)
    x = tf.concat([x1,x2],axis=3)
    return fcn_part(x,weight_decay)

def inference(images,weight_decay):
    with tf.variable_scope('net1'):
        x1 = inference_without_fc(images,3,weight_decay)
    with tf.variable_scope('net2'):
        x2 = inference_without_fc(images,5,weight_decay)
    x = tf.concat([x1,x2],axis=3)
    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(x, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_wd('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = tf.get_variable('biases', [384],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_wd('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = tf.get_variable('biases', [192], dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_wd('weights', [192, NUM_CLASSES],
                                              stddev=1 / 192.0, wd=0.0)
        biases = tf.get_variable('biases', [NUM_CLASSES],
                                 dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

    return softmax_linear

def loss(logits, labels):
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  #one_hot = tf.one_hot(labels,10)
  #hinge = tf.losses.hinge_loss(labels=one_hot, logits=logits)
  #hinge = tf.reduce_mean(hinge, name='hinge')
  #mse = tf.losses.mean_squared_error(labels=one_hot, predictions=logits)
  #mse = tf.reduce_mean(mse, name='mse')
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

  tf.add_to_collection('losses',  cross_entropy_mean)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])
  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))
  return loss_averages_op

def train(total_loss, global_step):
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)
  # Generate moving averages of all losses and associated summaries
  loss_averages_op = _add_loss_summaries(total_loss)
  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)
  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)
  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)
  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())
  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')
  return train_op

