#!/home/hongbin/miniconda3/envs/tensor/bin python3.5
# -*- coding: utf-8 -*-
# @Time    : 2018/9/14 14:26
# @Author  : Li Hongbin
# @File    : taskUtils.py

import tensorflow as tf
import numpy as np
import time


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
    img2 = tf.cast(img1, tf.float32)

    label1 = tf.cast(features['label'], tf.int32)

    return img2, label1


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
