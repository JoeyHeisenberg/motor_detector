#!/home/hongbin/miniconda3/envs/tensor/bin python3.5
# -*- coding: utf-8 -*-
# @Time    : 2018/9/14 14:34
# @Author  : Li Hongbin
# @File    : CNNCell.py

import tensorflow as tf


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
