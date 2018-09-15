#!/home/hongbin/miniconda3/envs/tensor/bin python3.5
# -*- coding: utf-8 -*-
# @Time    : 2018/9/15 19:48
# @Author  : Li Hongbin
# @File    : matrices_data_processed_TFRecord.py
# TODO  功能与 rgb_data_processed_TFRecord.py 相似 ，数据集为 1s_matrices_dataset
# TODO  尝试  标准化， 归一化等 数据影响


import os
import tensorflow as tf
import numpy as np

# ****************** 超参数设置 **************************
datadir = 'F:\\motor_sound_data\\1s_matrices_dataset'
mode = 'train'
filename = mode + '.tfrecords'
writer_path = datadir + '\\save\\'
file_path = writer_path + filename
matrices_height = 129
matrices_width = 92
# ****************** 超参数设置 **************************

# ****************** TFrecord **************************

writer = tf.python_io.TFRecordWriter(file_path)
for i in os.listdir(os.path.join(datadir, mode)):
    feature_path = os.path.join(datadir, mode, i, 'feature')
    label_path = os.path.join(datadir, mode, i, 'label')
    for name in os.listdir(feature_path):
        matrices_path = os.path.join(feature_path, name)
        matrices = np.loadtxt(matrices_path)

        # # 归一化
        # max_mat = np.max(matrices)
        # min_mat = np.min(matrices)
        #
        # fin_matrices = (matrices - min_mat) / (max_mat - min_mat)

        # 标准化
        mean_mat = np.average(matrices)
        deviation_mat = np.std(matrices)

        fin_matrices = (matrices - mean_mat) / deviation_mat

        i_int = int(i)

        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[i_int])),
            "matrices_raw": tf.train.Feature(bytes_list=tf.train.FloatList(value=[fin_matrices]))
        }))
        writer.write(example.SerializeToString())  # 序列化为字符串

writer.close()


