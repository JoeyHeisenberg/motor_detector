#!/home/hongbin/miniconda3/envs/tensor/bin python3.5
# -*- coding: utf-8 -*-
# @Time    : 2018/9/5 16:11
# @Author  : Li Hongbin
# @File    : 1s_convert_processed_train_test_dev.py
# TODO     功能1. 将 1s_processed的数据 按着 Description.txt的要求 转移到 1s_dataset
# TODO     功能2. 将 1s_processed_matrices的数据 转移到 1s_matrices_dataset

import os
import shutil
processed_path = r'F:\motor_sound_data_2.0\1s_processed'
s_dataset_path = r'F:\motor_sound_data_2.0\1s_dataset'

# processed_path = r'F:\motor_sound_data_2.0\1s_processed_matrices'
# s_dataset_path = r'F:\motor_sound_data_2.0\1s_matrices_dataset'

for motorMode in os.listdir(processed_path):
    processed_motor_path = os.path.join(processed_path, motorMode)

    for pngList in os.listdir(processed_motor_path):
        pngPath = os.path.join(processed_motor_path, pngList)

        filepath, tempfilename = os.path.split(pngPath)
        filename, extension = os.path.splitext(tempfilename)

        mode_motor_serial_num = tempfilename[2:5]
        if int(motorMode) == 0:

            if 0 < int(mode_motor_serial_num) < 61:
                s_dataset_mode_motor_path = s_dataset_path + '\\train\\' + motorMode + "\\feature\\"
                save_path = s_dataset_mode_motor_path + tempfilename
                shutil.copy(pngPath, save_path)

            if 60 < int(mode_motor_serial_num) < 81:
                s_dataset_mode_motor_path = s_dataset_path + '\\test\\' + motorMode + "\\feature\\"
                save_path = s_dataset_mode_motor_path + tempfilename
                shutil.copy(pngPath, save_path)

            if 80 < int(mode_motor_serial_num) < 101:
                s_dataset_mode_motor_path = s_dataset_path + '\\develop\\' + motorMode + "\\feature\\"
                save_path = s_dataset_mode_motor_path + tempfilename
                shutil.copy(pngPath, save_path)

        if int(motorMode) == 1:

            if 0 < int(mode_motor_serial_num) < 62:
                s_dataset_mode_motor_path = s_dataset_path + '\\train\\' + motorMode + "\\feature\\"
                save_path = s_dataset_mode_motor_path + tempfilename
                shutil.copy(pngPath, save_path)

            if 61 < int(mode_motor_serial_num) < 83:
                s_dataset_mode_motor_path = s_dataset_path + '\\test\\' + motorMode + "\\feature\\"
                save_path = s_dataset_mode_motor_path + tempfilename
                shutil.copy(pngPath, save_path)

            if 82 < int(mode_motor_serial_num) < 104:
                s_dataset_mode_motor_path = s_dataset_path + '\\develop\\' + motorMode + "\\feature\\"
                save_path = s_dataset_mode_motor_path + tempfilename
                shutil.copy(pngPath, save_path)
