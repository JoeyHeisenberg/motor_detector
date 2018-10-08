#!/home/hongbin/miniconda3/envs/tensor/bin python3.5
# -*- coding: utf-8 -*-
# @Time    : 2018/10/8 10:48
# @Author  : Li Hongbin
# @File    : motor_detector_test_rgb_window.py
# TODO      1. window平台在测试集上测试错误率，精确率 和 召回率

import tensorflow as tf
import numpy as np
import os                 # 路径需要
from datetime import datetime
import time

from Utils import read_decode, logging, check_path_exists, count_params, logging_total_acc
from Models import conv_layer_bn, conv_layer_bn_1_dimension, fully_connected_bn


# ********
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# **********

# # ****************** 用于控制所用显存*********************
# config = tf.ConfigProto()
# # config.gpu_options.per_process_gpu_memory_fraction = 0.1
# config.gpu_options.allow_growth = True
# # 用于选择某个显卡进行训练
# # os.environ["CUDA_VISIBLE_DEVICES"] = '1'   # 指定M4000GPU可用
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'   # 指定gtx 1080ti GPU可用
# # ****************** 用于控制所用显存*********************

# tensorboard+save
now = 'rgb-test12-4.0-2'
root_logdir = "motor_detector_record"
tensorboard_dir = "F:\\motor_sound_data_4.0\\{}\\run-{}\\tensorboard\\".format(root_logdir, now)
savedir = "F:\\motor_sound_data_4.0\\{}\\run-{}\\save".format(root_logdir, now)
logdir = "F:\\motor_sound_data_4.0\\{}\\run-{}\\logging\\".format(root_logdir, now)

logfile = os.path.join(logdir, str(datetime.strftime(datetime.now(), '%Y-%m-%d-%H:%M:%S')
                                   + '.txt').replace(' ', '').replace('/', ''))

# ****************** 超参数设置 **************************
datadir = 'F:\\motor_sound_data_4.0\\1s_dataset\\'

# dev_file_path = datadir + 'save/' + 'train.tfrecords'  # !!!!!!!
dev_file_path = datadir + 'save\\' + 'test.tfrecords'  # !!!!!!!
# dev_file_path = datadir + 'save/' + 'develop.tfrecords'  # !!!!!!!

image_height = 129   # 先height再width
image_width = 92
image_channel = 3  # rgb channels

dev_batch_size = 5

num_threads = 1   # cpu线程？？？
capacity = 12100


# lr = 1e-4
lr = 1e-4
prob = 0.5

epoch_num = 1
grad_clip = 1   # 'set the threshold of gradient clipping, -1 denotes no clipping'

train_num = 15600
test_num = 3800
# dev_num = 15600
dev_num = 3800  # 改成test的数量
# dev_num = 3700

keep = True

# for logging
config = {
          'trainable params': 0,
          'all params': 0,
          }
# ****************** 超参数设置 **************************


# ******************** 开发集数据读取 ****************************
dev_img, dev_label = read_decode(dev_file_path, 1)
dev_img1 = tf.reshape(dev_img, [image_height, image_width, image_channel])
# 添加 标准化 模块
dev_img2 = tf.image.per_image_standardization(dev_img1)  # 标准化

dev_img_batch, dev_label_batch = tf.train.batch([dev_img2, dev_label], batch_size=dev_batch_size,
                                                num_threads=num_threads, capacity=capacity)
# one_hot编码
dev_labels_batch = tf.expand_dims(dev_label_batch, 1)
dev_indices = tf.expand_dims(tf.range(0, dev_batch_size, 1), 1)
dev_concated = tf.concat([dev_indices, dev_labels_batch], 1)
# stack()保证[batch_size, 2]是tensor
dev_labels_onehot_batch = tf.sparse_to_dense(dev_concated, tf.stack([dev_batch_size, 2]), 1.0, 0.0)
# ******************** 数据读取 *********************************


# ********************                     网络结构                           *********************************
# 输入层
with tf.name_scope('input'):
    is_training = tf.placeholder(tf.bool)
    lr = tf.placeholder(tf.float32)
    tf_X = dev_img_batch
    tf_Y = dev_labels_onehot_batch
    tf.summary.image('input_image', tf_X, 10)

print(tf_X.shape, tf_Y.shape, is_training)

# 输入层数据为 3*129*92 的 tensor结构（具体会带上batch_size）
# ***********************  1-卷积层1：                    ******************************

conv_layer_bn1 = conv_layer_bn_1_dimension('conv_layer_bn1', tf_X, 32, 7, strides=2, padding='VALID',
                                           is_bn_training=is_training, epsilon=1e-3, decay=0.99)
print('conv_layer_bn1: ', conv_layer_bn1)
# ***********************  2-池化层1:                    ******************************
with tf.name_scope('max_pool1'):
    max_pool1 = tf.nn.max_pool(conv_layer_bn1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
print('max_pool1: ', max_pool1)
# ***********************  3-卷积层2：                    ******************************
conv_layer_bn2 = conv_layer_bn('conv_layer_bn2', max_pool1, 16, 5, strides=1, padding='VALID',
                               is_bn_training=is_training, epsilon=1e-3, decay=0.99)
print('conv_layer_bn2: ', conv_layer_bn2)
# ***********************  4-卷积层2：                    ******************************
conv_layer_bn3 = conv_layer_bn('conv_layer_bn3', conv_layer_bn2, 16, 5, strides=1, padding='VALID',
                               is_bn_training=is_training, epsilon=1e-3, decay=0.99)
print('conv_layer_bn3: ', conv_layer_bn3)
# ***********************   5-将卷积层输出扁平化处理         ******************************
with tf.name_scope('flat'):
    orig_shape = conv_layer_bn3.get_shape().as_list()
    flat = tf.reshape(conv_layer_bn3, shape=[-1, orig_shape[1]*orig_shape[2]*orig_shape[3]])
print('flat: ', flat)
# ***********************   6-全连接层1:                   ******************************
fully_connected_bn1 = fully_connected_bn('fully_1', flat, 50, is_bn_training=is_training, epsilon=1e-3,
                                         decay=0.99, wl=0.004)
with tf.name_scope('dropout1'):
    fully_dropout1 = tf.contrib.layers.dropout(fully_connected_bn1, keep_prob=prob, is_training=is_training)
# ***********************   7-全连接层2:                   ******************************
fully_connected_bn2 = fully_connected_bn('fully_2', fully_dropout1, 10, is_bn_training=is_training, epsilon=1e-3,
                                         decay=0.99, wl=0.004)
with tf.name_scope('dropout2'):
    fully_dropout2 = tf.contrib.layers.dropout(fully_connected_bn2, keep_prob=prob, is_training=is_training)
# ***********************   8-输出层                      ******************************
out_w1 = tf.Variable(tf.truncated_normal([10, 2], stddev=0.05))
out_b1 = tf.Variable(tf.truncated_normal([2]))
logits = tf.matmul(fully_dropout2, out_w1) + out_b1
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

# 保存
saver = tf.train.Saver(var_list, max_to_keep=2)
# ******************** 数据训练 *********************************

# ******************** Session 部分 ****************************
with tf.Session() as sess:

    sess.run([tf.global_variables_initializer(),
              tf.local_variables_initializer()])
    ckpt = tf.train.get_checkpoint_state(savedir)
    # Returns CheckpointState proto from the "checkpoint" file.

    if ckpt and ckpt.model_checkpoint_path:  # The checkpoint file
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model restored from:' + savedir)  # 由于使用TFRecord，直接读取效果不对
    else:
        print('NO CKPT file')

    # 用Coordinator协同线程，并启动线程
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        # while not (coord.should_stop()):
        # Run training steps or whatever
        print("hey**********************************************************")

        dev_step_num = int(dev_num / dev_batch_size)
        dev_batchErrors = np.zeros(dev_step_num)

        dev_motor_num = int(dev_num / 5)
        dev_motorErrors = np.zeros(dev_motor_num)

        # ************************  develop ***************************************************
        start = time.time()
        TP = 0
        FP = 0
        FN = 0
        for step in range(dev_step_num):

            acc = sess.run(accuracy,
                           feed_dict={is_training: False})
            dev_batchErrors[step] = acc  # 用于统计最后的Epoch er

            # 计算5s内判断
            if acc > 0.5:
                dev_motorErrors[step] = 1.
            else:
                dev_motorErrors[step] = 0.

            # 计算精确率（因为网络问题导致出厂含坏电机比例）
            if step < 361 and acc > 0.5:
                TP = TP + 1
            if step > 360 and acc < 0.5:
                FP = FP + 1
            # 计算召回率（因为网络问题导致损失正常电机比例）
            if step < 361 and acc < 0.5:
                FN = FN + 1
            print("step: ", step + 1, 'accuracy: ', acc)

        end = time.time()
        delta_time = end - start
        print('ALL test dataset need : ' + str(delta_time) + ' s')
        dev_epochER = dev_batchErrors.sum() / dev_step_num
        dev_motor_epochER = dev_motorErrors.sum() / dev_motor_num
        print('Accuracy :', dev_epochER)
        print('5s Accuracy :', dev_motor_epochER)
        dev_precision = TP / (TP + FP)
        dev_recall = TP / (TP + FN)
        print('TP, FP, FN', TP, FP, FN)
        print('5s Precision :', dev_precision)
        print('5s Recall :', dev_recall)
        # logging(config, logfile, dev_epochER, 1, delta_time=delta_time, mode1='test')

        # ************************  develop ***************************************************

    except tf.errors.OutOfRangeError:
        # 和try的while一起使用，保证数据全部进入网络进行训练
        print('Done training -- epoch limit reached')

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
        coord.join(thread)
# ******************** Session 部分 ****************************
