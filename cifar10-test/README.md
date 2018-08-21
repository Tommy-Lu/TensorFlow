>>> >>> 参数设置：
conv1_w = weight_variable([3, 3, 3, 8])
conv1_b = bias_variable([8]) 
conv2_w = weight_variable([3, 3, 8, 16])
conv2_b = bias_variable([16])

learn_rate = 1e-3
epoch = 200
keep_prob = 1
batch_size = 3000
使用[3， 3]的卷积核， 每次选择3000个数据，进行200轮训练  
Python 3.5.2 (default, Sep 14 2017, 22:51:06) 
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> runfile('/home/tommy/Desktop/tensorflow_test/cifar10-test/cifar10-cnn.py', wdir='/home/tommy/Desktop/tensorflow_test/cifar10-test')
test_lables: (10000, 10)
test_images: (10000, 3072)
train_images: (50000, 3072)
train_lables: (50000, 10)
epoch: 10 loss 6853.9526
......
Train accuracy_rate 0.462
epoch: 180 loss 4383.0547
Train accuracy_rate 0.49033332
epoch: 190 loss 4363.3413
Train accuracy_rate 0.48733333
epoch: 200 loss 4339.1006
Train accuracy_rate 0.49633333
========================Test accuracy_rate 0.3241
````````````````````````Max Train accuracy rate is: 0.49633333
Total tarining time is: 208351.97591781616ms


>>> >>> 参数设置：
conv1_w = weight_variable([5, 5, 3, 8])
conv1_b = bias_variable([8]) 
conv2_w = weight_variable([5, 5, 8, 16])
conv2_b = bias_variable([16])

learn_rate = 1e-3
epoch = 200
keep_prob = 1
batch_size = 5000
使用[3， 3]的卷积核， 每次选择5000个数据，进行200轮训练  

Python 3.5.2 (default, Sep 14 2017, 22:51:06) 
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> runfile('/home/tommy/Desktop/tensorflow_test/cifar10-test/cifar10-cnn.py', wdir='/home/tommy/Desktop/tensorflow_test/cifar10-test')
test_lables: (10000, 10)
test_images: (10000, 3072)
train_images: (50000, 3072)
train_lables: (50000, 10)
epoch: 10 loss 12045.4
Train accuracy_rate 0.115
epoch: 20 loss 10467.584
Train accuracy_rate 0.244
epoch: 30 loss 9640.866
Train accuracy_rate 0.3146
.....
Train accuracy_rate 0.532
epoch: 170 loss 6581.963
Train accuracy_rate 0.536
epoch: 180 loss 6365.5
Train accuracy_rate 0.555
epoch: 190 loss 6443.7266
Train accuracy_rate 0.558
epoch: 200 loss 6420.2056
Train accuracy_rate 0.5496
========================Test accuracy_rate 0.3872
````````````````````````Max Train accuracy rate is: 0.558
Total tarining time is: 708790.1556491852ms


>>> >>> 参数设置：
learn_rate = 1e-3
epoch = 500
keep_prob = 1
batch_size = 5000
每个batch大小取随机5000个数据，进行500个loop的训练，train accuracy可以到0.68，但是Test accuracy一直上不来，大约0.40。

Python 3.5.2 (default, Sep 14 2017, 22:51:06) 
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> runfile('/home/tommy/Desktop/tensorflow_test/cifar10-test/cifar10-cnn.py', wdir='/home/tommy/Desktop/tensorflow_test/cifar10-test')
test_lables: (10000, 10)
test_images: (10000, 3072)
train_images: (50000, 3072)
train_lables: (50000, 10)
epoch: 10 loss 11381.252
Train accuracy_rate 0.1218
epoch: 20 loss 10489.008
Train accuracy_rate 0.2462
epoch: 30 loss 9770.899
Train accuracy_rate 0.296
epoch: 40 loss 9027.956
Train accuracy_rate 0.3542
epoch: 50 loss 8487.083
Train accuracy_rate 0.4026
epoch: 60 loss 7994.788
Train accuracy_rate 0.4396
epoch: 70 loss 7721.5596
Train accuracy_rate 0.4656
......
Train accuracy_rate 0.653
epoch: 420 loss 4947.2744
Train accuracy_rate 0.667
epoch: 430 loss 4900.2373
Train accuracy_rate 0.6712
epoch: 440 loss 4761.0713
Train accuracy_rate 0.673
epoch: 450 loss 4994.479
Train accuracy_rate 0.6584
epoch: 460 loss 4812.958
Train accuracy_rate 0.6676
epoch: 470 loss 4699.216
Train accuracy_rate 0.6858
epoch: 480 loss 4720.2754
Train accuracy_rate 0.6768
epoch: 490 loss 4600.6055
Train accuracy_rate 0.6832
epoch: 500 loss 4791.76
Train accuracy_rate 0.6702
========================Test accuracy_rate 0.405
````````````````````````Max Train accuracy rate is: 0.6858
Total tarining time is: 1668279.4711589813ms


>>> >>> 参数设置：
learn_rate = 1e-3
epoch = 2000
keep_prob = 1
batch_size = 5000
每个batch大小取随机5000个数据，进行2000个loop的训练，发现train accuracy可以到0.97，但是Test accuracy一直上不来，大约0.389。 考虑，是否数据过拟合。
test_lables: (10000, 10)
test_images: (10000, 3072)
train_images: (50000, 3072)
train_lables: (50000, 10)
epoch: 10 loss 11235.318
Train accuracy_rate 0.1536
epoch: 20 loss 10480.924
Train accuracy_rate 0.2472
epoch: 30 loss 9718.287
Train accuracy_rate 0.3144
epoch: 40 loss 9076.797
Train accuracy_rate 0.3648
epoch: 50 loss 8480.027
Train accuracy_rate 0.4128
......
epoch: 1970 loss 752.5939
Train accuracy_rate 0.9752
epoch: 1980 loss 810.66833
Train accuracy_rate 0.9718
epoch: 1990 loss 750.829
Train accuracy_rate 0.978
epoch: 2000 loss 776.15515
Train accuracy_rate 0.9748
========================Test accuracy_rate 0.3897
````````````````````````Max Train accuracy rate is: 0.978
Total tarining time is: 6549065.231800079ms
