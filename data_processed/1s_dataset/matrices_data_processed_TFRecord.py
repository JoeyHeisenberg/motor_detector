#!/home/hongbin/miniconda3/envs/tensor/bin python3.5
# -*- coding: utf-8 -*-
# @Time    : 2018/9/15 19:48
# @Author  : Li Hongbin
# @File    : matrices_data_processed_TFRecord.py
# TODO  功能与 rgb_data_processed_TFRecord.py 相似 ，数据集为 1s_matrices_dataset


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

        i_int = int(i)

        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[i_int])),
            "img_raw": tf.train.Feature(bytes_list=tf.train.FloatList(value=[matrices]))
        }))
        writer.write(example.SerializeToString())  # 序列化为字符串

writer.close()

# TODO 雅正resize顺序
# img_path = r'F:\motor_sound_data\1s_processed\0\0-001-001.png'
#
# img = Image.open(img_path)
# image_height = 129
# image_width = 92
# img = img.resize((image_width, image_height))
# # img = img.resize((image_height, image_width))
# img.show()
