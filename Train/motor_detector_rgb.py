#!/home/hongbin/miniconda3/envs/tensor/bin python3.5
# -*- coding: utf-8 -*-
# @Time    : 2018/9/14 13:50
# @Author  : Li Hongbin
# @File    : motor_detector_rgb.py
# TODO 1-dimentional CNN   C1+S1+C2+C3+FLATTEN+FU1+FU2


import tensorflow as tf
import numpy as np
import os                 # 路径需要
from datetime import datetime
import time

from Utils import read_decode, logging, check_path_exists, count_params, logging_total_acc
from Models import conv_layer_bn_1_dimension, conv_layer_bn, fully_connected_bn


# ****************** 用于控制所用显存*********************
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
# 用于选择某个显卡进行训练
os.environ["CUDA_VISIBLE_DEVICES"] = '1'   # 指定M4000GPU可用
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'   # 指定gtx 1080ti GPU可用
# ****************** 用于控制所用显存*********************

# tensorboard+save
now = 'rgb_test12-4.0-2'
root_logdir = "motor_dectector_record"
tensorboard_dir = "/home/hongbin/data/{}/run-{}/tensorboard/".format(root_logdir, now)
savedir = "/home/hongbin/data/{}/run-{}/save/".format(root_logdir, now)
logdir = "/home/hongbin/data/{}/run-{}/logging/".format(root_logdir, now)

logfile = os.path.join(logdir, str(datetime.strftime(datetime.now(), '%Y-%m-%d-%H:%M:%S')
                                   + '.txt').replace(' ', '').replace('/', ''))

# ****************** 超参数设置 **************************
datadir = '/home/hongbin/data/motor_sound_data_4.0/1s_dataset/'
mode = 'train'
filename = mode + '.tfrecords'
writer_path = datadir + 'save/'
train_file_path = writer_path + filename

test_file_path = datadir + 'save/' + 'test.tfrecords'

image_height = 129   # 先height再width
image_width = 92
image_channel = 3  # rgb channels


batch_size = 40
dev_batch_size = 50
test_batch_size = 50

num_threads = 5   # cpu线程？？？
capacity = 11900
min_after_dequeue = 6000  # need to less than capacity

# lr = 1e-4
# lr = 1e-4
prob = 0.5

epoch_num = 1000
grad_clip = 1   # 'set the threshold of gradient clipping, -1 denotes no clipping'

train_num = 15600
test_num = 3800


# keep = False
keep = True
# for logging
config = {'No.': now,
          'trainable params': 0,
          'all params': 0,
          'num_layer': 'BN + l2 R ',
          'num_class': '2',
          'activation': 'leaky_relu',
          'optimizer': 'adam',
          'batch size': batch_size}
# ****************** 超参数设置 **************************

# ******************** 测试集数据读取 ****************************
test_img, test_label = read_decode(test_file_path, epoch_num)
test_img1 = tf.reshape(test_img, [image_height, image_width, image_channel])
# 添加 标准化 模块
test_img2 = tf.image.per_image_standardization(test_img1)  # 标准化

test_img_batch, test_label_batch = tf.train.batch([test_img2, test_label], batch_size=test_batch_size,
                                                  num_threads=num_threads, capacity=capacity)
# one_hot编码
test_labels_batch = tf.expand_dims(test_label_batch, 1)
test_indices = tf.expand_dims(tf.range(0, test_batch_size, 1), 1)
test_concated = tf.concat([test_indices, test_labels_batch], 1)
# stack()保证[batch_size, 2]是tensor
test_labels_onehot_batch = tf.sparse_to_dense(test_concated, tf.stack([test_batch_size, 2]), 1.0, 0.0)
# ******************** 数据读取 *********************************


# ******************** 测试集数据读取 ****************************
train_img, train_label = read_decode(train_file_path, epoch_num)
train_img1 = tf.reshape(train_img, [image_height, image_width, image_channel])
# 添加 标准化 模块
train_img2 = tf.image.per_image_standardization(train_img1)  # 标准化

train_img_batch, train_label_batch = tf.train.shuffle_batch([train_img2, train_label], batch_size=batch_size,
                                                            num_threads=num_threads, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
# one_hot编码
train_labels_batch = tf.expand_dims(train_label_batch, 1)
train_indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
train_concated = tf.concat([train_indices, train_labels_batch], 1)
# stack()保证[batch_size, 2]是tensor
train_labels_onehot_batch = tf.sparse_to_dense(train_concated, tf.stack([batch_size, 2]), 1.0, 0.0)
# ******************** 数据读取 *********************************


# ********************                     网络结构                           *********************************
# 输入层
with tf.name_scope('input'):
    is_training = tf.placeholder(tf.bool)
    lr = tf.placeholder(tf.float32)
    tf_X = tf.cond(is_training, lambda: train_img_batch, lambda: test_img_batch)
    tf_Y = tf.cond(is_training, lambda: train_labels_onehot_batch, lambda: test_labels_onehot_batch)
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
            train_batchLoss = np.zeros(train_step_num)

            test_step_num = int(test_num / test_batch_size)
            test_batchErrors = np.zeros(test_step_num)
            test_batchLoss = np.zeros(test_step_num)

            # **********training ！！！ 每个epoch 都进行tensorboard记录， 每隔10 epoch 保存模型，只记录最近两个 *********
            start = time.time()
            # if epoch > 20:
            #     learning_r = 1e-6
            # else:
            #     learning_r = 1e-4
            learning_r = 1e-6

            for step in range(train_step_num):
                _, summary, los, acc = sess.run([train_step, merged, loss, accuracy],
                                                feed_dict={is_training: True,
                                                           lr: learning_r})
                train_batchErrors[step] = acc  # 用于统计最后的Epoch er
                train_batchLoss[step] = los
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
            logging_total_acc(logfile, train_epochER, 'train')

            # training*****************************************************************************

            # # ************************  test ***************************************************
            start = time.time()
            for step in range(test_step_num):
                summary, acc, los = sess.run([merged, accuracy, loss],
                                             feed_dict={is_training: False})
                test_batchErrors[step] = acc  # 用于统计最后的Epoch er
                test_batchLoss[step] = los
                print("step: ", step + 1, 'accuracy: ', acc)

                if step == test_step_num - 1:
                    test_writer.add_summary(summary, epoch + 1)

            end = time.time()
            delta_time = end - start
            print('ALL test dataset need : ' + str(delta_time) + ' s')
            test_epochER = test_batchErrors.sum() / test_step_num
            print('Accuracy :', test_epochER)
            logging(config, logfile, test_epochER, epoch, delta_time=delta_time, mode1='test')
            logging_total_acc(logfile, test_epochER, 'dev')

            # # ************************  test ***************************************************

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

