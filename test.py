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

pad_logits = tf.cast(3 * tf.ones([bs, self.num_steps, 2]), tf.float32)

sess = tf.Session()
x,v = sess.run([mul,c])
print(x)
print('******')
print(np.shape(v))
