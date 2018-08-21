# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
运行时，将cifar10文件夹、data_convert.py文件和此文件放在同一目录。
"""

import tensorflow as tf
import time

#解决编译时提示不兼容 AVX2, FMA的问题。
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import data_convert
"""
思路：
输入：（10000， 3072） --> reshap(10000, 32, 32, 3)
输入： x (batch, 32, 32, 3) 
卷积1： h_conv1 
(batch, 32, 32, 3) w = (5, 5, 3, 32) strides = (1, 1, 1, 1) padding = 'SAME'
conv1 = (batch, 32, 32, 32)
pool1 = (batch, 16, 16, 32)  ksize = (1, 2, 2, 1) strides = [1, 2, 2, 1] padding = 'SAME'

layer2:  input (batch, 16, 16, 32 )
conv2 (batch, 16, 16, 64)
pool2 (batch, 8, 8, 64)

layer3: full connection input(batch, 8, 8, 64)
w = (8*8*64, 1024) b = ([1024])
output (1, 1024)

layer4: full connection input (1, 1024)
w = (1024, 10)   b = (10)
output (1, 10)
softmax()
 """
 
learn_rate = 1e-3
epoch = 2000
#keep_prob = 1
batch_size = 5000

def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    init = tf.constant(0.1, shape = shape)
    return tf.Variable(init)

#tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
#def conv2d(value, weight, strd):
#    return tf.nn.conv2d(value, weight, strides = [1, strd, strd, 1], padding = 'SAME')
def conv2d(value, weight):
    return tf.nn.conv2d(value, weight, strides = [1, 1, 1, 1], padding = 'SAME')


#tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
def max_pool_2x2(value, strd):
    return tf.nn.max_pool(value, ksize = [1, 2, 2, 1], strides = [1, strd, strd, 1], padding = 'SAME' )


x = tf.placeholder(tf.float32, shape = [None, 3072])
x_image = tf.reshape(x, [-1, 32, 32, 3])

keep_prob = tf.placeholder(tf.float32)
#Layer1: convlution
#x (batch, 32, 32, 3)  w = (5, 5, 3, 32) (conv)s = 1 (pool)s = 2
conv1_w = weight_variable([5, 5, 3, 8])
conv1_b = bias_variable([8]) 
#conv1 = tf.nn.relu(conv2d(x_image, conv1_w, 1) + conv1_b)  # output (batch, 32, 32, 32)
conv1 = tf.nn.relu(conv2d(x_image, conv1_w) + conv1_b) 
pool1 = max_pool_2x2(conv1, 2)   #out (batch, 16, 16, 32)
drop1 = tf.nn.dropout(pool1, keep_prob)
#drop1 = tf.nn.lrn(drop1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

# Layer2: convlution and max pooling
# x (batch, 16, 16, 32) w = (5, 5, 32, 64), (conv)s = 1, (pool)s = 2 
conv2_w = weight_variable([5, 5, 8, 16])
conv2_b = bias_variable([16])
conv2 = tf.nn.relu(conv2d(drop1, conv2_w) + conv2_b)  #output (batch, 16, 16, 64)
pool2 = max_pool_2x2(conv2, 2)  #output (batch, 8, 8, 64)
drop2 = tf.nn.dropout(pool2, keep_prob)
#drop2 = tf.nn.lrn(drop2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

#Layer3 : full connection
fc1_w = weight_variable([8*8*16, 1024])
fc1_b = bias_variable([1024])
fc1_input = tf.reshape(drop2, [-1, 8*8*16])
fc1_output = tf.nn.relu(tf.matmul(fc1_input, fc1_w) + fc1_b) #output (1, 1024)
fc1_drop =  tf.nn.dropout(fc1_output, keep_prob) 

#Layer4 full connection
fc2_w = weight_variable([1024, 10])
fc2_b = bias_variable([10])
fc2_linear = tf.matmul(fc1_drop, fc2_w)+ fc2_b
#fc2_linear = data_nor(fc2_linear)
fc2_output = tf.nn.softmax(fc2_linear)
y_conv = fc2_output

# loss, train_step
y = tf.placeholder(tf.float32, [None, 10])
loss = -tf.reduce_sum(y * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)
#train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)  效果很差
prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_conv, 1))
accuracy_rate = tf.reduce_mean(tf.cast(prediction, 'float'))


sess = tf.Session()
sess.run(tf.global_variables_initializer())
#xx_data, yy_data = data_convert()
test_images, test_labels = data_convert.get_test_batch()
tarin_images, tarin_labels = data_convert.get_train_batch()
start_time =  time.time()
max_accur = 0.0
for i in range(epoch):
    random_batch_x, random_batch_y = data_convert.next_random_batch(tarin_images, tarin_labels, batch_size)
    sess.run(train_step, feed_dict = {x:random_batch_x, y:random_batch_y, keep_prob:1 })
  
    if (i+1) % 100 == 0:
        print("epoch:" , i+1, "loss", sess.run(loss, feed_dict={x:random_batch_x, y:random_batch_y, keep_prob:1}))
        trian_accur = sess.run(accuracy_rate, feed_dict={x:random_batch_x, y:random_batch_y,keep_prob:1})
        if trian_accur > max_accur:
            max_accur = trian_accur
        print("Train accuracy_rate", trian_accur)

end_time = time.time()
test_accur = sess.run(accuracy_rate, feed_dict={x:test_images, y:test_labels, keep_prob:1})
print("========================Test accuracy_rate", test_accur)
print("````````````````````````Max Train accuracy rate is:", max_accur)
print("Total tarining time is:",str((end_time - start_time) * 1000) + 'ms')
sess.close()

 
 
 
 