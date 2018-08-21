# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 16:46:00 2018

@author: tommy
运行时，将cifar10文件夹和此文件放在同一目录。ciar10数据集下载地址：http://www.cs.toronto.edu/~kriz/cifar.html
"""

"""
data convert for cifar10
"""
import pickle
import numpy as np
import os


"""
  numpy one hot转换
>>> a
array([2, 4, 5, 6, 7, 8, 9, 0])
>>> na = np.max(a)+1
>>> np.eye(na)[a]
array([[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
"""    
"""
函数名： get_test_batch()
功能说明：从当前文件路径下，打开文件夹cifar10中的test_batch文件，并将其中的test data和test labels进行numpy array转换后输出。
         注意，这里取出的label直接是表示类别的数字，所以将其进行numpy one-hot转换，对应为10维的稀疏矩阵，数字对应矩阵中索引下标为1，其他为0。
         例如，2对应为[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]
输入：无
输出：numpy array 格式的test_images，test_lables：test_images (10000, 3072) ， test_lables (10000, 10) 

"""
def get_test_batch():
    pwd =  os.getcwd()    #获取当前文件路径
    file = pwd + '/' + "cifar10/test_batch"  #指定当前路径下的cifar10文件夹
    #file = "cifar10/test_batch"
    fs = open(file, 'rb')
    batch = pickle.load(fs, encoding='bytes')
    
    test_images = batch[b'data']
    test_lables = batch[b'labels']
    test_lables = np.array(test_lables)
    #print(test_lables)
    
    num_test_lables = np.max(test_lables) + 1
    test_lables = np.eye(num_test_lables)[test_lables]

    fs.close()
    
    print("test_lables:", test_lables.shape)
    print("test_images:", test_images.shape)
    return test_images, test_lables  #test_images (10000, 3072) ， test_lables (10000, 10)
  

"""
函数名： get_train_batch()
功能说明：从当前文件路径下，打开文件夹cifar10中的包含有“batch_”字段的文件，并进行拼接后输出为train_images和train_lables的numpy array。
         注意1：这里取出的label直接是表示类别的数字，所以将其进行numpy one-hot转换，对应为10维的稀疏矩阵，数字对应矩阵中索引下标为1，其他为0。
              例如，2对应为[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]
         注意2：这里在拼接numpy array时，用到了np.concatenate， 先初始化一个和每一个batch文件大小相等的np.zeros，作为拼接的基础，在后边输出时， 在将其去掉。
输入：无
输出：numpy array 格式的train_images, train_lables   ，train_images (50000, 3072), train_lables (50000, 10)

"""
def get_train_batch():
    pwd =  os.getcwd()    #获取当前文件路径
    root_dir = pwd + '/' + "cifar10"  #指定当前路径下的cifar10文件夹

    train_images = np.zeros([10000, 3072]) #初始化10000 * 3072的0 numpy array, 用于拼接读出来的train image
    train_lables = np.zeros([10000, 10])   #初始化10000 * 10的0 numpy array, 用于拼接读出来的train label
    #print(train_lables.shape)
    
    #查询目录下文件，并判断文件名中是否包含"batch_", 并将包含的文件名和路径拼接成文件名，用pickle读取，处理其中数据。
    dir_list = os.listdir(root_dir)
    for files in dir_list:
        if "batch_" in files:
            #print(root_dir + '/' + files)
            #print(root_dir)
            fs = open(root_dir + '/' + files, 'rb')
            batch = pickle.load(fs, encoding='bytes')
        
            temp_images = batch[b'data']
            temp_lables = batch[b'labels']
            temp_lables = np.array(temp_lables)
            #print(test_lables)
            num_temp_lables = np.max(temp_lables) + 1
            temp_lables = np.eye(num_temp_lables)[temp_lables]
            
            #numpy的拼接模块
            train_images = np.concatenate([train_images, temp_images])
            train_lables = np.concatenate([train_lables, temp_lables])
            
    train_images = train_images[10000:]  #去掉为了拼接初始化的前10000个元素,返回后边拼接的50000个元素的numpy array
    train_lables = train_lables[10000:]  #去掉为了拼接初始化的前10000个元素,返回后边拼接的50000个元素的numpy array
    #print(type(train_images), type(train_lables))
    print("train_images:", train_images.shape)
    print("train_lables:", train_lables.shape)
    fs.close()
    
    return train_images, train_lables   #train_images (50000, 3072), train_lables (50000, 10)


"""
函数名：  next_random_batch(x_images, y_labels, batch_size)
功能说明：对输入的x_images和y_labels， 随即获取batch_size大小的子集，并输出numpy array格式的子集
输入：x_images：需要获取batch的images数据集
     y_labels：需要获取batch的labels数据集
     batch_size:需要获取的batch大小
输出：numpy array 格式的子集batch_x, batch_y

"""    
def next_random_batch(x_images, y_labels, batch_size):
    index = [i for i in range(0, len(x_images))]
    np.random.shuffle(index)
    batch_x = []
    batch_y = []

    for i in range(0, batch_size):
        batch_x.append(x_images[index[i]])
        batch_y.append(y_labels[index[i]])
        
    return np.array(batch_x) / 255, np.array(batch_y)


