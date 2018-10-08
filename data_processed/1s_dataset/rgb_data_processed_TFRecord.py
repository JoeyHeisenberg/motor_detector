#!/home/hongbin/miniconda3/envs/tensor/bin python3.5
# -*- coding : UTF-8 -*-
# Author：LiHongbin   Time: 2018/6/27
# # TODO  将dataset和1s_dataset的数据保存为TFRecord，放在save\文件中

import os
import tensorflow as tf
from PIL import Image

# ****************** 超参数设置 **************************
datadir = 'F:\\motor_sound_data_4.0\\1s_dataset'
mode = 'test'
# filename = mode + '.tfrecords'
filename = 'test_for_order.tfrecords'
writer_path = datadir + '\\save\\'
file_path = writer_path + filename
image_height = 129
image_width = 92

k = 0
# ****************** 超参数设置 **************************

# ****************** TFrecord **************************

writer = tf.python_io.TFRecordWriter(file_path)
for i in os.listdir(os.path.join(datadir, mode)):
    feature_path = os.path.join(datadir, mode, i, 'feature')
    label_path = os.path.join(datadir, mode, i, 'label')
    for name in os.listdir(feature_path):
        img_path = os.path.join(feature_path, name)
        img = Image.open(img_path)
        # img = tf.reshape(img, [image_width, image_height, image_channel])     # 作用可有可无？？？反正变成二进制
        img = img.resize((image_width, image_height))

        # 去掉alpha通道
        r, g, b, a = img.split()  # 分离三通道
        img = Image.merge('RGB', (r, g, b))  # 合并三通道

        i_int = int(i)
        k = k + 1
        img_raw = img.tobytes()         # 将图片转化成byte格式（二进制数据）
        print(img_path)
        print(k)
        example = tf.train.Example(features=tf.train.Features(feature={
            "order": tf.train.Feature(int64_list=tf.train.Int64List(value=[k])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[i_int])),
            "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
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
