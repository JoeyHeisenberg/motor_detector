#!/home/hongbin/miniconda3/envs/tensor/bin python3.5
# -*- coding: utf-8 -*-
# @Time    : 2018/7/31 14:26
# @Author  : Li Hongbin
# @File    : convert_csv_to_one_fifths_png.py
# @Software: PyCharm
# TODO 将电机信号csv文件根据FFT原理转换成时频特征图, 由于csv为 5s数据过长导致网络复杂，所以分成5份

import numpy as np
import matplotlib.pylab as plt
import os

# ****************** 超参数设置 **************************
source_path = r'F:\motor_sound_data\source'               # r 表示 取消\转义符
processed_path = r'F:\motor_sound_data\1s_processed'

framerate = 12000
nfft = 256
noverlap = 128
# ****************** 超参数设置 **************************


# ******************  数据转化 **************************
for motorMode in os.listdir(source_path):
    source_mode_path = os.path.join(source_path, motorMode)
    save_path = os.path.join(processed_path, motorMode)

    for motorList in os.listdir(source_mode_path):
        source_mode_list_path = os.path.join(source_mode_path, motorList)

        for csvList in os.listdir(source_mode_list_path):
            csvPath = os.path.join(source_mode_list_path, csvList)

            for i in range(5):

                str_data = np.loadtxt(open(csvPath, 'rb'), delimiter="\n", skiprows=0)
                part_data = str_data[i*framerate: i*framerate+framerate]

                # 根据 freqs和bins长度设定图片像素
                fig = plt.figure(figsize=(0.92, 1.29))

                # 设置axes填充全图和坐标轴消失
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)

                plt.specgram(part_data, NFFT=nfft, Fs=framerate, noverlap=noverlap, mode='psd')

                filepath, tempfilename = os.path.split(csvPath)
                filename, extension = os.path.splitext(tempfilename)

                front = tempfilename[0:6]
                k = tempfilename[6:9]

                # int_k = int(k) + i * 20
                # 需要改成原来第二张图 ，序号为6,7,8,9,10
                int_k = (int(k)-1) * 5 + (i + 1)

                fin_k = str(int_k)
                fin_k = fin_k.zfill(3)
                fig.savefig(save_path + '\\' + front + fin_k)
                print("path: ", save_path + '\\' + front + fin_k)
                plt.close()
# ******************  数据转化 **************************
