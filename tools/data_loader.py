import os
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import torchvision.transforms
import cv2
from tools import data_processing, tool_functions, config

'''
通过继承Data.Dataset，实现将一组Tensor数据对封装成Tensor数据集
至少要重载__init__，__len__和__getitem__方法
'''


class Dataset(Data.Dataset):
    def __init__(self, img_dir, label_dir, input_mode, preImg_num, model_type, prediction_mode, data_shape):
        # 初始化
        self.img_dir = img_dir
        self.img_files = os.listdir(self.img_dir)
        self.preImg_path = os.path.join(img_dir.split("Product_9dvf")[0], "Origin/projection")
        self.label_dir = label_dir
        self.label_files = os.listdir(self.label_dir)
        self.input_mode = input_mode
        self.model_type = model_type
        self.preImg_num = preImg_num
        self.prediction_mode = prediction_mode
        self.data_shape = data_shape

    def __len__(self):
        # 返回数据集的大小
        return len(self.img_files)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        img_name = self.img_files[index]
        name_number = img_name.split("_")[1]
        # 获取转换图作为输入图像
        input_img = torch.tensor(data_processing.get_input_array(
            model_type=self.model_type,
            image_path=self.img_dir,
            image_name=img_name,
            preImg_num=self.preImg_num,
            pre_image_path=self.preImg_path
        ), dtype=torch.float)


        label = data_processing.get_label(
            prediction_mode=self.prediction_mode,
            label_path=self.label_dir,
            image_name=img_name,
            data_shape=self.data_shape
        )

        return input_img, label


if __name__ == '__main__':
    pass
