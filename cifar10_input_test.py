import tensorflow as tf
import cv2
import cifar10_input
import numpy as np
img_batch, label_batch= cifar10_input.test_inputs(True,'../dataset/cifar10/',1,True)
with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
        # for qr in queue_runners:
        #     print(type(qr.queue))
        #     print(qr.queue.name)
        #     for opt in qr.enqueue_ops:
        #        print(type(opt))
        #        print(opt.name)
        for i in range(100):
            img,label = sess.run([img_batch,label_batch])
            img = np.squeeze(img)
            cv2.namedWindow('img',0)
            cv2.imshow('img',img)
            cv2.waitKey(0)


