import tensorflow as tf
import tensorflow.contrib.slim as slim
import  slim.nets.mobilenet_v1 as mobilenet_v1
import argparse
import data_input
import  mobile_net
import os
import numpy as np
parse = argparse.ArgumentParser()
parse.add_argument('--finetune_model',type=str,default='../pretrained_model/mobilenet_v1_1.0_224_2017_06_14/mobilenet_v1_1.0_224.ckpt')
parse.add_argument('--train_records',type=str,default='train.tfrecords')
parse.add_argument('--val_records',type=str,default='val.tfrecords')
parse.add_argument('--train_dir',type=str,default='./log/train/')
parse.add_argument('--batch_size',type=int,default=64)
parse.add_argument('--num_classes',type=int,default=2)
parse.add_argument('--crop_w',type=int,default=224)
parse.add_argument('--crop_h',type=int,default=224)
parse.add_argument('--base_lr',type=float,default=0.01)
parse.add_argument('--decay_steps',type=int,default=10000)
parse.add_argument('--decay_factor',type=float,default=0.1)
parse.add_argument('--weight_decay',type=float,default=0.0)
parse.add_argument('--max_steps',type=int,default=20000)
parse.add_argument('--save_model_per_steps',type=int,default=1000)
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
    logits,end_points=mobile_net.mobile_net_inference(img_batch_train,FLAGS.num_classes)
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
    train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(total_loss,global_step=global_step)
    init_op = tf.global_variables_initializer()
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
    saver = tf.train.Saver()
    with tf.Session()  as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        for step in range(FLAGS.max_steps):
            loss,global_step_np,lr_np, summary_string, _ = sess.run([total_loss,global_step,lr,
                                                      merged_summary_op,
                                                      train_op])
            if step % 10 == 0 and step > 0:
                print("loss: " + str(loss) + " \t lr: " + str(lr_np)+"\tstep:"+str(global_step_np))
                summary_writer.add_summary(summary_string, step)
            if step % FLAGS.save_model_per_steps == 0 and step > 0:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                save_path = saver.save(sess,checkpoint_path, global_step=global_step)
                print("Save model in: %s" % save_path)
        coord.request_stop()
        coord.join(threads)
    summary_writer.close()

def eval():
    assert os.path.exists(FLAGS.val_records)
    filename_queue_val = tf.train.string_input_producer([FLAGS.val_records])
    img_batch_val,label_batch_val = data_input.val_input(filename_queue_val,
                                                        FLAGS.batch_size,FLAGS.crop_w,FLAGS.crop_h)
    logits,end_points=mobile_net.mobile_net_inference(img_batch_val,FLAGS.num_classes)
    top_k_op = tf.nn.in_top_k(logits, label_batch_val, 1)
    saver = tf.train.Saver()
    with tf.Session()  as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        true_count = 0
        steps_n = int(5000 / FLAGS.batch_size)
        for step in range(steps_n):
            predictions = sess.run([top_k_op])
            true_count += np.sum(predictions)
        print("Ap@1:" , true_count/(steps_n*FLAGS.batch_size))
        coord.request_stop()
        coord.join(threads)

if __name__=='__main__':
    #eval()
    train()

