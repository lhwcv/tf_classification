import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import  control_flow_ops
import  slim.nets.mobilenet_v1 as mobilenet_v1
import argparse
import data_input
import  mobile_net
import os
import numpy as np
import random
import cv2
parse = argparse.ArgumentParser()
parse.add_argument('--finetune_model',type=str,default='')
parse.add_argument('--train_records',type=str,default='train.tfrecords')
parse.add_argument('--val_records',type=str,default='val.tfrecords')
parse.add_argument('--train_dir',type=str,default='./log/train/')
parse.add_argument('--batch_size',type=int,default=64)
parse.add_argument('--num_classes',type=int,default=3)
parse.add_argument('--crop_w',type=int,default=224)
parse.add_argument('--crop_h',type=int,default=224)
parse.add_argument('--base_lr',type=float,default=0.02)
parse.add_argument('--decay_steps',type=int,default=10000)
parse.add_argument('--decay_factor',type=float,default=0.1)
parse.add_argument('--weight_decay',type=float,default=0.0)
parse.add_argument('--max_steps',type=int,default=20000)
parse.add_argument('--save_model_per_steps',type=int,default=2000)
FLAGS = parse.parse_args()

def get_variables_mapping(inputs,model_variables):
    logits_slim, end_points_slim = mobilenet_v1.mobilenet_v1(inputs, scope='MobileNetV1')
    model_variables_slim = slim.get_model_variables(scope='MobileNetV1')
    model_variables_maping = {}
    for i, j in zip(model_variables, model_variables_slim):
        if 'Logits' not in j.name:
            model_variables_maping[j.name] = i
    return model_variables_maping

def train():
    assert os.path.exists(FLAGS.train_records)
    filename_queue_train = tf.train.string_input_producer([FLAGS.train_records])
    img_batch_train,label_batch_train=data_input.train_input(filename_queue_train,
                                                             FLAGS.batch_size,FLAGS.crop_w,FLAGS.crop_h)
    logits,end_points=mobile_net.mobile_net_inference(img_batch_train,FLAGS.num_classes,is_training=True)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch_train,
                                                          logits=logits)
    cross_entropy_sum =  tf.reduce_mean(cross_entropy, name='cross_entropy_sum')
    tf.add_to_collection('losses', cross_entropy_sum)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    tf.summary.scalar('cross_entropy_loss',  cross_entropy_sum)
    tf.summary.scalar('total_loss', total_loss)

    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
    lr = tf.train.exponential_decay(FLAGS.base_lr,
                                    global_step=global_step,
                                   decay_steps= FLAGS.decay_steps,
                                   decay_rate= FLAGS.decay_factor
                                   )
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)

    train_op = slim.learning.create_train_op(cross_entropy_sum, optimizer, global_step=global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        cross_entropy_sum = control_flow_ops.with_dependencies([updates], cross_entropy_sum)

    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies([tf.group(*update_ops)]):
    #     train_op = optimizer.minimize(total_loss, global_step=global_step)

    init_op = tf.global_variables_initializer()
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
    saver = tf.train.Saver()
    with tf.Session()  as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step_np = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            tf.assign(global_step,global_step_np)
        else:
            print('No past checkpoint file found')

        for step in range(FLAGS.max_steps):
            loss,global_step_np,lr_np, summary_string, _ = sess.run([total_loss,global_step,lr,
                                                      merged_summary_op,
                                                      train_op])
            if step % 10 == 0 and step > 0:
                print("loss: " + str(loss) + " \t lr: " + str(lr_np)+"\tstep:"+str(global_step_np))
                summary_writer.add_summary(summary_string, step)
            if global_step_np % FLAGS.save_model_per_steps == 0 and step > 0:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                save_path = saver.save(sess,checkpoint_path, global_step=global_step)
                print("Save model in: %s" % save_path)
        coord.request_stop()
        coord.join(threads)
    summary_writer.close()

def eval(val_num=5000):
    assert os.path.exists(FLAGS.val_records)
    with tf.Session()  as sess:
        filename_queue_val = tf.train.string_input_producer([FLAGS.val_records])
        img_batch_val, label_batch_val = data_input.val_input(filename_queue_val,
                                                             50, FLAGS.crop_w, FLAGS.crop_h)
        logits, end_points = mobile_net.mobile_net_inference(img_batch_val, FLAGS.num_classes, is_training=False)
        top_k_op = tf.nn.in_top_k(logits, label_batch_val, 1)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        true_count = 0
        for step in range(int(val_num/50)):
            predictions,imgs,labels,pred = sess.run([top_k_op,img_batch_val,label_batch_val,logits])
            true_count += np.sum(predictions)
        print("Ap@1:" , true_count/val_num)
        coord.request_stop()
        coord.join(threads)



if __name__=='__main__':
    eval()
    #train()

