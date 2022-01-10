import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
try:
    with tf.device('/device:GPU:0'):
        v = tf.Variable(tf.zeros([10, 10]))
        print(v)
except:
    print('no gpu')

