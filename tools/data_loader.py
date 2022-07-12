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

args = config.get_args()

'''
通过继承Data.Dataset，实现将一组Tensor数据对封装成Tensor数据集
至少要重载__init__，__len__和__getitem__方法
'''


class Dataset_CBCT(Data.Dataset):
    def __init__(self, img_folder, target_folder, input_mode_names, model_type):
        # 初始化
        self.img_folder = img_folder
        self.img_files = os.listdir(self.img_folder)
        self.target_folder = target_folder
        self.target_files = os.listdir(self.target_folder)
        self.input_mode_names = input_mode_names
        self.model_type = model_type

    def __len__(self):
        # 返回数据集的大小
        return len(self.img_files)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        root_path = tool_functions.get_poject_path("Pulmonary-2D-3D-Image-Registration")
        name_number = self.img_files[index].split("projection")[1]
        img_name = os.path.join(self.img_folder, self.img_files[index])  # 返回第index的文件名
        img = np.fromfile(img_name, dtype='float32').reshape((100, 240, 300))[25]  # 读取img

        # 获取转换图作为输入图像
        input_img = data_processing.input_mode_concat_variable(img, standardization_method="max_min",
                                                               input_mode_names=self.input_mode_names,
                                                               resize=(150, 150))
        # 获取对应的标签
        target_name = os.path.join(self.target_folder, ('CT_dcm' + name_number))  # 返回第index的文件名
        target = np.fromfile(target_name, dtype='float32').reshape((150, 256, 256))[:, 53:203, 53:203]
        if self.model_type == "spaceAndTime":
            pre_imgs = data_processing.load_projection_sequence(self.img_folder,
                                                                data_processing.get_preImgName_sequence(
                                                                    img_name, args.preImg_num))
            input_preImgs = [data_processing.input_mode_concat_variable(pre_img, standardization_method="max_min",
                                                                        input_mode_names=self.input_mode_names,
                                                                        resize=(150, 150)) for pre_img in pre_imgs]

            return torch.tensor(input_preImgs + [input_img]), target

        return input_img, target


class Dataset_PCA(Data.Dataset):
    def __init__(self, img_folder, target_folder, input_mode_names, model_type):
        # 初始化
        self.img_folder = img_folder
        self.img_files = os.listdir(self.img_folder)
        self.target_folder = target_folder
        self.target_files = os.listdir(self.target_folder)
        self.input_mode_names = input_mode_names
        self.model_type = model_type

    def __len__(self):
        # 返回数据集的大小
        return len(self.img_files)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        img_name = self.img_files[index]
        name_number = img_name.split("_")[1]
        img_path = os.path.join(self.img_folder, img_name)  # 返回第index文件的路径
        # img = np.fromfile(img_path, dtype='float32').reshape((100, 240, 300))[25]  # 读取img.bin的所有投影
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = np.array(img, dtype="float32")
        # 获取转换图作为输入图像
        input_img = data_processing.input_mode_concat_variable(img, standardization_method="max_min",
                                                               input_mode_names=self.input_mode_names,
                                                               resize=(120, 120))
        # 获取对应的标签
        target_path = os.path.join(self.target_folder, ('PCA_' + name_number))  # 返回第index的文件名
        target = np.fromfile(target_path, dtype='float32')  # 读取img
        if self.model_type == "spaceAndTime":
            pre_imgs = data_processing.load_projection_sequence(self.img_folder,
                                                                data_processing.get_preImgName_sequence(
                                                                    img_name, args.preImg_num))
            input_preImgs = [data_processing.input_mode_concat_variable(pre_img, standardization_method="max_min",
                                                                        input_mode_names=self.input_mode_names,
                                                                        resize=(120, 120)) for pre_img in pre_imgs]

            return torch.tensor(input_preImgs + [input_img]), target
        # 返回值自动转换为torch的tensor类型
        return input_img, target


if __name__ == '__main__':
    args = config.get_args()
    root_path = tool_functions.get_poject_path("PCA")
    img_folder = os.path.join(root_path, args.img_folder)
    target_folder = os.path.join(root_path, args.target_folder)
    methods = "origin_sub_multiAngle_edge"
    dataset = Dataset_PCA(img_folder, target_folder, methods, 2)
    input_preImg, input_img, target = dataset.__getitem__(10)

    print(input_img.shape)
    print(input_img[0].max(), input_img[0].min())
    print(input_img[1].max(), input_img[1].min())
    plt.imshow(input_img[0], cmap='gray')
    plt.show()
    plt.imshow(input_img[1], cmap='gray')
    plt.show()
    plt.imshow(input_img[2], cmap='gray')
    plt.show()
    plt.imshow(input_img[3], cmap='gray')
    plt.show()
    plt.imshow(input_img[0], cmap='gray')
    plt.show()
    plt.imshow(input_preImg[1], cmap='gray')
    plt.show()
    plt.imshow(input_preImg[2], cmap='gray')
    plt.show()
    plt.imshow(input_preImg[3], cmap='gray')
    plt.show()
    plt.imshow(input_preImg[0] - input_img[0], cmap='gray')
    plt.show()
    print(target.max(), target.min())
