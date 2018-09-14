#!/home/hongbin/miniconda3/envs/tensor/bin python3.5
# -*- coding: utf-8 -*-
# @Time    : 2018/9/10 18:47
# @Author  : Li Hongbin
# @File    : motor_detector10_Gray.py
# # TODO  完成 卷积层和全连接层的BN和 数据尺寸缩减 和 数量增加， 但是6.0版本网络过拟合
#   TODO  7.0 版本 1.加入l1或者l2正则项  weight decay
#   TODO          2. relu 变成 leaky_re
#   TODO          3. 加入dropout
#   TODO  9.0 版本 1. 加入大量tensorboard代码
#   TODO          2. 训练和验证顺序进行
#   TODO 10.0 版本 1.数据变成FFT后未经二值化处理的 1s时频图   1s_gray_dataset

import tensorflow as tf
import numpy as np
import os                 # 路径需要
from datetime import datetime
import time
from math import sqrt

# ****************** 用于控制所用显存*********************
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.1
config.gpu_options.allow_growth = True
# 用于选择某个显卡进行训练
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'   # 指定M4000GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = '0'   # 指定gtx 1080ti GPU可用
# ****************** 用于控制所用显存*********************


# ****************** 超参数设置 **************************
# datadir = '/home/hongbin/data/motor_sound_data/bin_dataset/'
datadir = '/home/hongbin/data/motor_sound_data/1s_gray_dataset/'
mode = 'train'
filename = mode + '.tfrecords'
writer_path = datadir + 'save/'
train_file_path = writer_path + filename

dev_file_path = datadir + 'save/' + 'develop.tfrecords'

image_height = 129   # 先height再width
image_width = 92
image_channel = 1  # 变成binary， 为单通道


batch_size = 40
dev_batch_size = 50

num_threads = 5   # cpu线程？？？
capacity = 4000
min_after_dequeue = 3000  # need to less than capacity

# lr = 1e-4
lr = 1e-3
prob = 0.5

epoch_num = 1000
grad_clip = 1   # 'set the threshold of gradient clipping, -1 denotes no clipping'

train_num = 12100
test_num = 4000
dev_num = 4100

keep = False

# tensorboard+save
now = '10.0-gray-test-1'
root_logdir = "motor_dectector_record"
tensorboard_dir = "/home/hongbin/data/{}/run-{}/tensorboard/".format(root_logdir, now)
savedir = "/home/hongbin/data/{}/run-{}/save/".format(root_logdir, now)
logdir = "/home/hongbin/data/{}/run-{}/logging/".format(root_logdir, now)

logfile = os.path.join(logdir, str(datetime.strftime(datetime.now(), '%Y-%m-%d-%H:%M:%S')
                                   + '.txt').replace(' ', '').replace('/', ''))

# for logging
config = {'No.': now,
          'trainable params': 0,
          'all params': 0,
          'num_layer': 'BN + l2 R ',
          'num_class': '2',
          'activation': 'relu',
          'optimizer': 'adam',
          'learning rate': lr,
          'batch size': batch_size}
# ****************** 超参数设置 **************************


# ******************** 函数定义 *********************************

def read_decode(file_path1, epoch1):
    # def read_decode(file_path1):
    """读取tfrecord的二进制图像和label，然后解码成数组"""
    #  生成文件名队列  string_input_producer()中必须是1-D tensor, 把TFRecorder的路径放进文件名队列
    file_queue = tf.train.string_input_producer([file_path1], num_epochs=epoch1)  # 这里num_epochs 表示
    # file_queue = tf.train.string_input_producer([file_path1])
    # 读取器tf.TFRecordReader
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)  # 返回文件名和文件
    # 解析器tf.parse_single_example
    # Configuration for parsing a fixed-length input feature.
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    # 解码
    img1 = tf.decode_raw(features['img_raw'], tf.uint8)
    img2 = tf.reshape(img1, [image_height, image_width, image_channel])
    # 归一化,对图片进行归一化操作将【0，255】之间的像素归一化到【-0.5，0.5】 , 需要两个吗？
    img3 = tf.cast(img2, tf.float32)
    # img4 = tf.image.per_image_standardization(img3)  # 标准化

    label1 = tf.cast(features['label'], tf.int32)

    return img3, label1


def logging(config1, logfile1, errorRate, epoch=0, delta_time=0, mode1='train'):
    """log the cost and error rate and time while training or testing"""
    if mode1 != 'train' and mode1 != 'test'and mode1 != 'develop':
        raise TypeError('mode should be train or test or config1.')

    elif mode1 == 'train':
        with open(logfile1, "a") as myfile:
            myfile.write(str(config1) + '\n')
            myfile.write(str(time.strftime('%X %x %Z'))+'\n')
            myfile.write("Epoch:"+str(epoch+1)+' '+"train error rate:"+str(errorRate)+'\n')
            myfile.write("Epoch:"+str(epoch+1)+' '+"train time:"+str(delta_time)+' s\n')
    elif mode1 == 'test':
        logfile1 = logfile1+'_TEST'
        with open(logfile1, "a") as myfile:
            myfile.write(str(config1)+'\n')
            myfile.write(str(time.strftime('%X %x %Z'))+'\n')
            myfile.write("test error rate:"+str(errorRate)+'\n')
    elif mode1 == 'develop':
        logfile1 = logfile1+'_DEV'
        with open(logfile1, "a") as myfile:
            myfile.write(str(config1)+'\n')
            myfile.write(str(time.strftime('%X %x %Z'))+'\n')
            myfile.write("development error rate:"+str(errorRate)+'\n')


def check_path_exists(path):
    """ check a path exists or not
    """
    if isinstance(path, list):
        for p in path:
            if not os.path.exists(p):
                os.makedirs(p)
    else:
        if not os.path.exists(path):
            os.makedirs(path)


def count_params(var, mode2='trainable'):
    """count all parameters of a tensorflow graph"""
    if mode2 == 'all':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in var])
    elif mode2 == 'trainable':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in var])
    else:
        raise TypeError('mode should be all or trainable.')
    print('number of '+mode2+' parameters: '+str(num))
    return num


def variable_with_weight_2loss(shape, stddev=0.05, wl=0.):
    """
    variable_with_weight_2loss 计算网络weight的 l2 loss

    :param shape:
        网络 weight 的 shape
    :param stddev:  float
        weight 正太分布的方差， 默认参数 0.05
    :param wl:      float
        正则化参数， 默认值为 0
    :return:    weights
        返回 可训练
    """
    weights = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(weights), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
        # 将weight loss 加入到 图中，放在 losses 的 类似于 list中 ，重复命令则 加1
    return weights


def variable_he_initialization_with_weight_2loss(shape, wl=0.):
    """
    使用He_intialization 初始化参数 ，用于全连接层 ,variable_with_weight_2loss 计算网络weight的 l2 loss

    :param shape:
        网络 weight 的 shape
    :param wl:      float
        正则化参数， 默认值为 0
    :return:    weights
        返回 可训练
    """
    n_input = shape[0]
    n_output = shape[1]
    stddev = sqrt(2) * sqrt(2./(n_input+n_output))

    weights = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(weights), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
        # 将weight loss 加入到 图中，放在 losses 的 类似于 list中 ，重复命令则 加1
    return weights


def variable_summaries(var_name, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(var_name):
        # 计算参数的均值，并使用tf.summary.scalar记录
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        # 计算参数的标准差
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # 用直方图记录参数的分布
        tf.summary.histogram('histogram', var)


def fully_connected_bn(layer_name, prev_layer, num_units, is_bn_training=True, epsilon=1e-3, decay=0.90, stddev=0.05, wl=0.):
    """
    num_units参数传递该层神经元的数量，根据prev_layer参数传入值作为该层输入创建全连接神经网络。

    :param layer_name : String
            网络层名字
    :param prev_layer: Tensor
        该层神经元输入
    :param num_units: int
        该层神经元结点个数
    :param is_bn_training: bool or Tensor
        表示该网络当前是否正在训练，告知Batch Normalization层是否应该更新或者使用均值或方差的分布信息， 默认参数 True
    :param epsilon: float
        表示BN中的参数epsilon, 默认参数 1e-3
    :param decay: float
        表示滑动平均算法中更新均值和方差的速率， 默认参数 0.99
    :param stddev:  float
        表示weight初始化时正态分布的方差， 默认0.05
    :param wl:      float
        表示网络该层l2正则化系数， 默认为0，不正则化
    :returns Tensor
        一个新的全连接神经网络层
    """

    # layer = tf.layers.dense(prev_layer, num_units, use_bias=False, activation=None)
    with tf.name_scope(layer_name):

        in_num_units = prev_layer.get_shape().as_list()[1]

        with tf.name_scope('weights'):
            fc_w = variable_he_initialization_with_weight_2loss([in_num_units, num_units], wl)
            variable_summaries('fully_weights', fc_w)

        with tf.name_scope('linear_output'):
            layer = tf.matmul(prev_layer, fc_w)
            tf.summary.histogram('linear', layer)

        with tf.name_scope('batch_normalization'):
            gamma = tf.Variable(tf.ones([num_units]))
            beta = tf.Variable(tf.zeros([num_units]))
            variable_summaries('gamma', gamma)
            variable_summaries('beta', beta)

            pop_mean = tf.Variable(tf.zeros([num_units]), trainable=False)
            pop_variance = tf.Variable(tf.ones([num_units]), trainable=False)
            variable_summaries('pop_mean', pop_mean)
            variable_summaries('pop_variance', pop_variance)

        def batch_norm_training():
            batch_mean, batch_variance = tf.nn.moments(layer, [0])

            train_mean = tf.assign(pop_mean, pop_mean*decay + batch_mean*(1 - decay))
            train_variance = tf.assign(pop_variance, pop_variance*decay + batch_variance*(1 - decay))

            with tf.control_dependencies([train_mean, train_variance]):
                return tf.nn.batch_normalization(layer, batch_mean, batch_variance, beta, gamma, epsilon)

        def batch_norm_inference():
            return tf.nn.batch_normalization(layer, pop_mean, pop_variance, beta, gamma, epsilon)

        with tf.name_scope('BN_output'):
            batch_normalized_output = tf.cond(is_bn_training, batch_norm_training, batch_norm_inference)
            tf.summary.histogram('bn_output', batch_normalized_output)
        with tf.name_scope('RELU_output'):
            relu_output = tf.nn.leaky_relu(batch_normalized_output)
            tf.summary.histogram('relu_output', relu_output)
    return relu_output


def conv_layer_bn(layer_name, prev_layer, out_channel, shape=3, strides=1, padding='SAME', is_bn_training=True,
                  epsilon=1e-3, decay=0.90, stddev=0.05, wl=0.):
    """
       使用给定的参数作为输入创建卷积层
        :param layer_name : String
            网络层名字
        :param prev_layer: Tensor
            传入该层神经元作为输入
        :param out_channel: int
            我们将根据网络中图层的深度设置特征图的步长和数量。
            这不是实践CNN的好方法，但它可以帮助我们用很少的代码创建这个示例。
        :param shape: int
            卷积核的大小, 默认参数3
        :param strides: int
            卷积核的同步移动， 默认参数 1
        :param padding: 'VALID' or 'SAME'
            卷积网络padding方式， 默认参数为 'SAME'
        :param is_bn_training: bool or Tensor
            表示该网络当前是否正在训练，告知Batch Normalization层是否应该更新或者使用均值或方差的分布信息, 默认参数 True
        :param epsilon: float
            表示BN中的参数epsilon, 默认参数 1e-3
        :param decay: float
            表示滑动平均算法中更新均值和方差的速率， 默认参数 0.99
        :param stddev:  float
        表示weight初始化时正态分布的方差， 默认0.05
        :param wl:      float
        表示网络该层l2正则化系数， 默认为0，不正则化
        :returns Tensor
            一个新的卷积层
        """
    in_channels = prev_layer.get_shape().as_list()[3]
    out_channels = out_channel

    # 设置命名空间
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            c_weights = variable_with_weight_2loss([shape, shape, in_channels, out_channels], stddev, wl)
            variable_summaries('conv_weights', c_weights)

        with tf.name_scope('linear_output'):
            layer = tf.nn.conv2d(prev_layer, c_weights, strides=[1, strides, strides, 1], padding=padding)
            tf.summary.histogram('linear', layer)

        with tf.name_scope('batch_normalization'):
            gamma = tf.Variable(tf.ones([out_channels]))
            beta = tf.Variable(tf.zeros([out_channels]))
            variable_summaries('gamma', gamma)
            variable_summaries('beta', beta)

            pop_mean = tf.Variable(tf.zeros([out_channels]), trainable=False)
            pop_variance = tf.Variable(tf.ones([out_channels]), trainable=False)
            variable_summaries('pop_mean', pop_mean)
            variable_summaries('pop_variance', pop_variance)

        def batch_norm_training():
            # 一定要使用正确的维度确保计算的是每个特征图上的平均值和方差而不是整个网络节点上的统计分布值
            batch_mean, batch_variance = tf.nn.moments(layer, [0, 1, 2], keep_dims=False)

            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_variance = tf.assign(pop_variance, pop_variance * decay + batch_variance * (1 - decay))

            with tf.control_dependencies([train_mean, train_variance]):
                return tf.nn.batch_normalization(layer, batch_mean, batch_variance, beta, gamma, epsilon)

        def batch_norm_inference():
            return tf.nn.batch_normalization(layer, pop_mean, pop_variance, beta, gamma, epsilon)

        with tf.name_scope('BN_output'):
            batch_normalized_output = tf.cond(is_bn_training, batch_norm_training, batch_norm_inference)
            tf.summary.histogram('bn_output', batch_normalized_output)
        # return tf.nn.relu(batch_normalized_output)
        with tf.name_scope('RELU_output'):
            relu_output = tf.nn.leaky_relu(batch_normalized_output)
            tf.summary.histogram('relu_output', relu_output)

    return relu_output

# ******************** 函数定义 *********************************


# ******************** 开发集数据读取 ****************************
dev_img, dev_label = read_decode(dev_file_path, epoch_num)
dev_img_batch, dev_label_batch = tf.train.batch([dev_img, dev_label], batch_size=dev_batch_size,
                                                num_threads=num_threads, capacity=capacity)
# one_hot编码
dev_labels_batch = tf.expand_dims(dev_label_batch, 1)
dev_indices = tf.expand_dims(tf.range(0, dev_batch_size, 1), 1)
dev_concated = tf.concat([dev_indices, dev_labels_batch], 1)
# stack()保证[batch_size, 2]是tensor
dev_labels_onehot_batch = tf.sparse_to_dense(dev_concated, tf.stack([dev_batch_size, 2]), 1.0, 0.0)
# ******************** 数据读取 *********************************


# ******************** 测试集数据读取 ****************************
train_img, train_label = read_decode(train_file_path, epoch_num)
train_img_batch, train_label_batch = tf.train.shuffle_batch([train_img, train_label], batch_size=batch_size,
                                                            num_threads=num_threads, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
# one_hot编码
train_labels_batch = tf.expand_dims(train_label_batch, 1)
train_indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
train_concated = tf.concat([train_indices, train_labels_batch], 1)
# stack()保证[batch_size, 2]是tensor
train_labels_onehot_batch = tf.sparse_to_dense(train_concated, tf.stack([batch_size, 2]), 1.0, 0.0)
# ******************** 数据读取 *********************************


# ******************** 数据训练 *********************************
# 输入层
with tf.name_scope('input'):
    is_training = tf.placeholder(tf.bool)
    tf_X = tf.cond(is_training, lambda: train_img_batch, lambda: dev_img_batch)
    tf_Y = tf.cond(is_training, lambda: train_labels_onehot_batch, lambda: dev_labels_onehot_batch)
    tf.summary.image('input_image', tf_X, 10)

print(tf_X.shape, tf_Y.shape, is_training)

# 输入层数据为 1*129*92 的 tensor结构（具体会带上batch_size）

# TODO 试着 1.  C1+C2+S1+C3+FU1+FU2+SOFTMAX,               参数1400w 一样问题, 卷积到全连接参数过大，需要池化
# TODO     2。 C1+C2+S1+C3+S2+FU1+FU2+SOFTMAX             参数550W  epoch2 就过拟合了
# TODO     3. 1s数据 C1+C2+S1+C3+S2+FU1+FU2+SOFTMAX       参数
# ***********************1-卷积层1：  4个3*3的卷积核，步长为1, valid  输出 4*127*465******************************

conv_layer_bn1 = conv_layer_bn('conv_layer_bn1', tf_X, 4, 7, strides=1, padding='VALID', is_bn_training=is_training,
                               epsilon=1e-3, decay=0.99)
print('conv_layer_bn1: ', conv_layer_bn1)
# # *******************2-卷积层2：  输入4*127*465   4个3*3的卷积核，步长为1, valid  输出 4*125*463******************

# conv_layer_bn2 = conv_layer_bn('conv_layer_bn2', conv_layer_bn1, 16, 7, strides=1, padding='VALID',
#                                is_bn_training=is_training, epsilon=1e-3, decay=0.99)
# print('conv_layer_bn2: ', conv_layer_bn2)
# *****************************3-池化层1:   输入4*125*463    输出 4*62*231****************************************
# with tf.name_scope('max_pool1'):
#     max_pool1 = tf.nn.max_pool(conv_layer_bn2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
# print('max_pool1: ', max_pool1)
# **********************4-卷积层3:   输入4*62*231  4个3*3的卷积核，步长为1, valid  输出 4*60*229********************
# conv_layer_bn3 = conv_layer_bn('conv_layer_bn3', max_pool1, 8, 7, strides=1, padding='VALID', is_bn_training=is_training,
#                                epsilon=1e-3, decay=0.99)
# print('conv_layer_bn3: ', conv_layer_bn3)
# *****************************5-池化层2:   输入4*60*229    输出 4*30*114****************************************
# with tf.name_scope('max_pool2'):
#     max_pool2 = tf.nn.max_pool(conv_layer_bn3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
# print('max_pool2: ', max_pool2)
# ********************************  6-将卷积层输出扁平化处理
# with tf.name_scope('flat'):
#     orig_shape = max_pool2.get_shape().as_list()
#     flat = tf.reshape(max_pool2, shape=[-1, orig_shape[1]*orig_shape[2]*orig_shape[3]])
# print('flat: ', flat)
with tf.name_scope('flat'):
    orig_shape = conv_layer_bn1.get_shape().as_list()
    flat = tf.reshape(conv_layer_bn1, shape=[-1, orig_shape[1]*orig_shape[2]*orig_shape[3]])
print('flat: ', flat)
# ********************************* 7-全连接层1:  输入 4*30*114  输出 100
fully_connected_bn1 = fully_connected_bn('fully_1', flat, 4, is_bn_training=is_training, epsilon=1e-3,
                                         decay=0.99, wl=0.004)
with tf.name_scope('dropout1'):
    fully_dropout1 = tf.contrib.layers.dropout(fully_connected_bn1, keep_prob=prob, is_training=is_training)
# ********************************* 8-全连接层2:  输入 100   输出 64
# fully_connected_bn2 = fully_connected_bn('fully_2', fully_dropout1, 10, is_bn_training=is_training, epsilon=1e-3,
#                                          decay=0.99, wl=0.004)
#
# with tf.name_scope('dropout2'):
#     fully_dropout2 = tf.contrib.layers.dropout(fully_connected_bn2, keep_prob=prob, is_training=is_training)
# ***********************************9-输出层=softmax
# out_w1 = tf.Variable(tf.truncated_normal([10, 2]))
# out_b1 = tf.Variable(tf.truncated_normal([2]))
# logits = tf.matmul(fully_dropout2, out_w1) + out_b1
# pred = tf.nn.softmax(logits)
out_w1 = tf.Variable(tf.truncated_normal([4, 2]))
out_b1 = tf.Variable(tf.truncated_normal([2]))
logits = tf.matmul(fully_dropout1, out_w1) + out_b1
pred = tf.nn.softmax(logits)

# loss
with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf_Y, logits=logits, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
    tf.add_to_collection('losses', cross_entropy_mean)
    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

loss_summary = tf.summary.scalar('loss', loss)

# 计算参数数量
var_list = tf.global_variables()
var_trainable_list = tf.trainable_variables()

all_num_params = count_params(var_list, mode2='all')
num_params = count_params(var_trainable_list, mode2='trainable')
config['trainable params'] = num_params
config['all params'] = all_num_params


# train_step
with tf.name_scope('train'):
    if grad_clip == -1:
        # not apply gradient clipping
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    else:
        # apply gradient clipping
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, var_trainable_list), grad_clip)
        opti = tf.train.AdamOptimizer(lr)
        train_step = opti.apply_gradients(zip(grads, var_trainable_list))


# accuracy
with tf.name_scope('ACCURACY'):
    y_pred = tf.arg_max(pred, 1)  # 因为是列向量，所以dimension = 1. 返回最大值的index
    bool_pred = tf.equal(tf.arg_max(tf_Y, 1), y_pred)
    accuracy = tf.reduce_mean(tf.cast(bool_pred, tf.float32))   # 求出每个batch—size的准确率

accuracy_summary = tf.summary.scalar('accuracy', accuracy)

# file_writer = tf.summary.FileWriter(tensorboard_dir, tf.get_default_graph())

# summaries合并
merged = tf.summary.merge_all()
# 写到指定的磁盘路径中
train_writer = tf.summary.FileWriter(tensorboard_dir + '/train', tf.get_default_graph())
test_writer = tf.summary.FileWriter(tensorboard_dir + '/test')


# 保存
saver = tf.train.Saver(var_list, max_to_keep=2)
# ******************** 数据训练 *********************************


# ******************** Session 部分 ****************************
with tf.Session() as sess:

    if keep:  # 用于重新训练 keep == True
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()])
        ckpt = tf.train.get_checkpoint_state(savedir)
        # Returns CheckpointState proto from the "checkpoint" file.
        if ckpt and ckpt.model_checkpoint_path:  # The checkpoint file
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model restored from:' + savedir)  # 由于使用TFRecord，直接读取效果不对
    else:
        # initializer for num_epochs
        print('Initializing')
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()])
        check_path_exists([root_logdir, tensorboard_dir, savedir, logdir])

    # 用Coordinator协同线程，并启动线程
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        # while not (coord.should_stop()):
        # Run training steps or whatever
        print("hey**********************************************************")
        #  train*******************************************

        for epoch in range(epoch_num):

            train_step_num = int(train_num / batch_size)
            train_batchErrors = np.zeros(train_step_num)

            dev_step_num = int(dev_num / dev_batch_size)
            dev_batchErrors = np.zeros(dev_step_num)

            # **********training ！！！ 每个epoch 都进行tensorboard记录， 每隔10 epoch 保存模型，只记录最近两个 *********
            start = time.time()
            for step in range(train_step_num):
                _, summary, los, acc = sess.run([train_step, merged, loss, accuracy],
                                                feed_dict={is_training: True})
                train_batchErrors[step] = acc  # 用于统计最后的Epoch er
                print("epoch", epoch+1, "step", step+1, "loss", los, "accuracy", acc)

                if step == train_step_num-1:
                    train_writer.add_summary(summary, epoch+1)

            end = time.time()
            delta_time = end - start
            print('epoch ' + str(epoch + 1) + ' needs time:' + str(delta_time) + ' s')

            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(savedir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=epoch)  # 第三个参数将训练的次数作为后缀加入到模型名字中
                print('Model has been saved in {}'.format(savedir))

            train_epochER = train_batchErrors.sum() / train_step_num

            logging(config, logfile, train_epochER, epoch, delta_time, mode1=mode)
            # training*****************************************************************************

            # ************************  develop ***************************************************
            start = time.time()
            for step in range(dev_step_num):
                summary, acc = sess.run([merged, accuracy],
                                        feed_dict={is_training: False})
                dev_batchErrors[step] = acc  # 用于统计最后的Epoch er
                print("step: ", step + 1, 'accuracy: ', acc)

                if step == dev_step_num-1:
                    test_writer.add_summary(summary, epoch+1)

            end = time.time()
            delta_time = end - start
            print('ALL develop dataset need : ' + str(delta_time) + ' s')
            dev_epochER = dev_batchErrors.sum() / dev_step_num
            print('Accuracy :', dev_epochER)
            logging(config, logfile, dev_epochER, epoch=1, delta_time=delta_time, mode1='develop')
            # ************************  develop ***************************************************

        train_writer.close()
        test_writer.close()  # tensorboard

    except tf.errors.OutOfRangeError:
        # 和try的while一起使用，保证数据全部进入网络进行训练
        print('Done training -- epoch limit reached')

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
        coord.join(thread)
# ******************** Session 部分 ****************************

