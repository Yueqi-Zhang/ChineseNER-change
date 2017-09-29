# encoding=utf8
import os
import codecs
import numpy as np
import tensorflow as tf

#this is a test

a = [[1,2],[2,3],[3,4],[4,5],[5,6]]
b = tf.expand_dims(a,0)
bs = tf.expand_dims(tf.shape(a)[0],0)
mul = tf.concat([bs,tf.constant([1,1])],0)
c = tf.tile(b,mul)
num_steps = tf.shape(a)[0]
pad_logits = tf.cast(3 * tf.ones([tf.shape(a)[0], num_steps, 2]), tf.float32)
f = tf.expand_dims()tf.ones([tf.shape(a)[0]],tf.int32)*2

sess = tf.Session()
x = sess.run(f)
print(x)
print(np.shape(x))

