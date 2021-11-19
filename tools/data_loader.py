import os
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
from tools.other.Laplacian import laplacian_img
from tools.config import get_args
from tools.tool_functions import get_poject_path
import torchvision.transforms
import cv2

'''
通过继承Data.Dataset，实现将一组Tensor数据对封装成Tensor数据集
至少要重载__init__，__len__和__getitem__方法
'''


def resize_img(img, resize):
    """
    :param img: img.shape(C,H,W) or (H,W)
    :param resize: (resize_h,reszie_w)
    :return: resizeImg.shape(C,H,W) or (H,W)
    """
    if len(img.shape) == 3:
        # 默认认为shape为CHW
        img_trans = img.transpose((1, 2, 0))
        resizeImg_trans = cv2.resize(img_trans, resize, interpolation=cv2.INTER_AREA)
        resizeImg = resizeImg_trans.transpose((2, 0, 1))
    else:
        resizeImg = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
    return resizeImg


"""对数据进行标准差归一化"""
def data_std_all(data):
    PCA_all = np.fromfile(r'/Dataset/train/PCA_all\pca_960', dtype='float32').reshape((960, 3))
    PCA_mean = np.mean(PCA_all, axis=0)
    PCA_std = np.std(PCA_all, axis=0)
    data_std = (data - PCA_mean) / PCA_std
    return data_std


"""对标准差归一化的数据进行还原"""


def stdData_reverse(data_std, PCA_std, PCA_mean):
    data_reverse = data_std * PCA_std + PCA_mean
    return data_reverse


"""最大-最小归一化数据进行还原"""


def normalData_reverse(normaldata, min, max):
    data_reverse = normaldata * (max - min) + min
    return data_reverse


"""对数据进行最大-最小标准化"""


def data_normal_max_min(data, min, max):
    data_normal_x = (data - min) / (max - min)
    return data_normal_x


"""对数据进行标准差标准化"""


def data_normal_mean_std(data, mean, std):
    data_normal_x = (data - mean) / std
    return data_normal_x


"""  对数据进行归一化 """


def img_normal_f(img, method):
    if (method == "max_min"):
        max_x = max(map(max, img))
        min_x = min(map(min, img))
        img_normal_x = data_normal_max_min(img, min_x, max_x)
    else:
        mean_x = np.mean(img)
        std_x = np.std(img)
        img_normal_x = data_normal_mean_std(img, mean_x, std_x)
    return img_normal_x


""" 加载标准差和最大最小值函数"""


def return_normal_para(target_file):
    target_all = np.fromfile(target_file, dtype='float32').reshape((-1, 3))
    std = np.std(target_all, axis=0)
    mean = np.mean(target_all, axis=0)
    min_x = np.min(target_all, axis=0)
    max_x = np.max(target_all, axis=0)
    return std, mean, min_x, max_x


def trans_PCAFile(target_folder):
    target_all = np.fromfile(os.path.join(target_folder, 'PCA_all'), dtype='float32')
    target_tran = np.reshape(target_all, (-1, 3))
    target_trans_folder = os.path.join(target_folder, '..\\PCA_trans\\')
    if not os.path.exists(target_trans_folder):
        os.makedirs(target_trans_folder)
    for i in range(target_tran.shape[0]):
        print(target_tran[i])
        target_tran[i].tofile(os.path.join(target_trans_folder, 'PCA_{}'.format(i)))


def data_normal_0_255(img):
    ymax = 255
    ymin = 0
    xmax = max(map(max, img))  # 进行两次求max值
    xmin = min(map(min, img))
    img_normal_0_255 = np.round((ymax - ymin) * (img - xmin) / (xmax - xmin) + ymin)
    return img_normal_0_255

def data_normal_0_255_imgs(imgs):
    # ask img.shape is (C,H,W)
    imgs_normal = []
    for i in range(imgs.shape[0]):
        imgs_normal_0_255 = data_normal_0_255(imgs[i])
        imgs_normal.append(imgs_normal_0_255)
    return np.array(imgs_normal)


def make_subimg(img):
    root_ = get_poject_path("PCA")
    img_fix = np.fromfile((root_ + "/Dataset/Test_9dvf/projection_0_0"), dtype="float32").reshape((100, 240, 300))[25,
              :, :]
    sub_img = img - img_fix
    return sub_img

def get_preImg_name(imgName,phase_num):
    pca_num = (int(imgName.split("_")[1]) - 1)
    pca_num = pca_num+phase_num if pca_num == 0 else pca_num
    random_num = imgName.split("_")[2]
    init_name = imgName.split("_")[0]
    preImgName = init_name+"_"+str(pca_num)+"_"+random_num
    return preImgName


def img_deal_cat(img, method, resize=None):
    # 灰度图像进行归一化
    img_normal = img_normal_f(img, method)

    # 获取差值图
    sub_img = make_subimg(img)
    sub_img_normal = img_normal_f(sub_img, method)

    # 获取边缘图
    edge_img = laplacian_img(img)
    edge_img_normal = img_normal_f(edge_img, method)

    if resize == None:
        # 图片在深度上进行叠加作为输入
        input_img = np.stack([img_normal, edge_img_normal, sub_img_normal], axis=0)
    else:
        input_img = np.stack(
            [resize_img(img_normal, resize), resize_img(edge_img_normal, resize), resize_img(sub_img_normal, resize)],
            axis=0)
    return input_img


# def img_deal_cat_variable(imgs, normal_method, data_method, resize=None):
#     img_orth = imgs[25]  # 25是正投影
#     img_shape = list(img_orth[np.newaxis,...].shape)
#     img_shape[0] = 0
#     img_cat = np.ones(shape=img_shape,dtype="float32")
#     if data_method["origin"]:
#         img_orth_normal = img_normal_f(img_orth, normal_method)[np.newaxis, ...]
#         img_cat = np.vstack([img_cat, img_orth_normal])
#     if data_method["multi_angle"]:
#         img_side = imgs[0]
#         img_side_normal = img_normal_f(img_side, normal_method)[np.newaxis, ...]
#         img_cat = np.vstack([img_cat, img_side_normal])
#     if data_method["edge"]:
#         edge_img = laplacian_img(img_orth)
#         edge_img_normal = img_normal_f(edge_img, normal_method)[np.newaxis, ...]
#         img_cat = np.vstack([img_cat, edge_img_normal])
#     if data_method["sub"]:
#         sub_img = make_subimg(img_orth)
#         sub_img_normal = img_normal_f(sub_img, normal_method)[np.newaxis, ...]
#         img_cat = np.vstack([img_cat, sub_img_normal])
#     if resize:
#         input_img = resize_img(img_cat.squeeze(), resize)
#     if len(input_img.shape) == 2:
#         input_img = input_img[np.newaxis, ...]
#     return input_img

def img_deal_cat_variable(imgs:np.ndarray, normal_method, data_method:list, resize=None)->np.ndarray:
    """
    :param imgs: shape(number,H,W)
    :param normal_method: "max-min" or "mean-std"
    :param data_method: ["origin","sub","multiAngle","edge"]
    :param resize: 将要调整的图像大小（resize_H,resize_W）
    :return:input_img.shape(C,H,W)
    """
    img_orth = imgs[25]  # 25是正投影
    # img_shape = 1 * W * H
    img_shape = list(img_orth[np.newaxis,...].shape)
    img_shape[0] = 0
    num = 0
    # img_cat.shape = 0 * W * H
    img_cat = np.ones(shape=img_shape,dtype="float32")
    if data_method.find("origin") != -1:
        num += 1
        img_orth_normal = img_normal_f(img_orth, normal_method)[np.newaxis, ...]
        img_cat = np.vstack([img_cat, img_orth_normal])
    if data_method.find("multiAngle") != -1:
        num += 1
        img_side = imgs[0]
        img_side_normal = img_normal_f(img_side, normal_method)[np.newaxis, ...]
        img_cat = np.vstack([img_cat, img_side_normal])
    if data_method.find("edge") != -1:
        num += 1
        edge_img = laplacian_img(img_orth)
        edge_img_normal = img_normal_f(edge_img, normal_method)[np.newaxis, ...]
        img_cat = np.vstack([img_cat, edge_img_normal])
    if data_method.find("sub") != -1:
        num += 1
        sub_img = make_subimg(img_orth)
        sub_img_normal = img_normal_f(sub_img, normal_method)[np.newaxis, ...]
        img_cat = np.vstack([img_cat, sub_img_normal])

    if num != len(data_method.split("_")):
        raise ValueError("dataMethodName Error!")
    # 调整尺寸
    if resize:
        input_img = resize_img(img_cat.squeeze(), resize)
    # 如果图像为 W*H，将其转换为 1*W*H
    if len(input_img.shape) == 2:
        input_img = input_img[np.newaxis, ...]

    return input_img


class Dataset(Data.Dataset):
    def __init__(self, img_folder, target_folder, PCA_all_folder):
        # 初始化
        self.img_folder = img_folder
        self.img_files = os.listdir(self.img_folder)
        self.target_folder = target_folder
        self.target_files = os.listdir(self.target_folder)
        self.PCA_all_folder = PCA_all_folder

    def __len__(self):
        # 返回数据集的大小
        return len(self.img_files)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        root_path = get_poject_path("PCA")
        name_number = self.img_files[index].split("_")[1]
        img_name = os.path.join(self.img_folder, self.img_files[index])  # 返回第index的文件名
        img = np.fromfile(img_name, dtype='float32').reshape((100, 240, 300))[25, :, :]  # 读取img

        # 获取转换图作为输入图像
        input_img = img_deal_cat(img, method="max_min", resize=(120, 120))
        # 获取对应的标签
        target_name = os.path.join(self.target_folder, ('PCA_' + name_number))  # 返回第index的文件名
        target = np.fromfile(target_name, dtype='float32')  # 读取img
        # std, mean, min_x, max_x = return_normal_para(os.path.join(root_path,"Dataset/Test1/DVF_trans_PCAs/PCA_10_phase"))

        # 返回值自动转换为torch的tensor类型
        return input_img, target


class Dataset_variable(Data.Dataset):
    def __init__(self, img_folder, target_folder, PCA_all_folder, methods):
        # 初始化
        self.img_folder = img_folder
        self.img_files = os.listdir(self.img_folder)
        self.target_folder = target_folder
        self.target_files = os.listdir(self.target_folder)
        self.PCA_all_folder = PCA_all_folder
        self.data_method = methods

    def __len__(self):
        # 返回数据集的大小
        return len(self.img_files)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        root_path = get_poject_path("PCA")
        name_number = self.img_files[index].split("_")[1]
        img_name = os.path.join(self.img_folder, self.img_files[index])  # 返回第index的文件名
        img = np.fromfile(img_name, dtype='float32').reshape((100, 240, 300))  # 读取img的所有投影

        # 获取转换图作为输入图像
        input_img = img_deal_cat_variable(img, normal_method="max_min", data_method=self.data_method, resize=(120, 120))
        # 获取对应的标签
        target_name = os.path.join(self.target_folder, ('PCA_' + name_number))  # 返回第index的文件名
        target = np.fromfile(target_name, dtype='float32')  # 读取img
        # 返回值自动转换为torch的tensor类型
        return input_img, target


class Dataset_variable_TestNum(Data.Dataset):
    def __init__(self, img_folder, target_folder, PCA_all_folder, methods,TestNum):
        # 初始化
        self.img_folder = img_folder
        self.img_files = os.listdir(self.img_folder)
        self.target_folder = target_folder
        self.target_files = os.listdir(self.target_folder)
        self.PCA_all_folder = PCA_all_folder
        self.data_method = methods
        self.TestNum = TestNum

    def __len__(self):
        # 返回数据集的大小
        return len(self.img_files)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        root_path = get_poject_path("PCA")
        name_number = self.img_files[index].split("_")[1]
        img_path = os.path.join(self.img_folder, self.img_files[index])  # 返回第index文件的路径
        img = np.fromfile(img_path, dtype='float32').reshape((100, 240, 300))  # 读取img的所有投影
        # 获取转换图作为输入图像
        input_img = img_deal_cat_variable(img, normal_method="max_min", data_method=self.data_method, resize=(120, 120))
        # 获取对应的标签
        target_path = os.path.join(self.target_folder, ('PCA_' + name_number))  # 返回第index的文件名
        target = np.fromfile(target_path, dtype='float32')  # 读取img
        if self.TestNum == 2:
            pre_img_path = os.path.join(self.img_folder, get_preImg_name(self.img_files[index], 9))
            pre_img = np.fromfile(pre_img_path, dtype='float32').reshape((100, 240, 300))  # 读取pre_img的所有投影
            input_preImg = img_deal_cat_variable(pre_img, normal_method="max_min", data_method=self.data_method, resize=(120, 120))
            return input_preImg,input_img,target
        # 返回值自动转换为torch的tensor类型
        return input_img, target


if __name__ == '__main__':
    args = get_args()
    root_path = get_poject_path("PCA")
    img_folder = os.path.join(root_path, args.img_folder)
    target_folder = os.path.join(root_path, args.target_folder)
    PCA_all_folder = os.path.join(root_path, args.PCA_all_folder)
    methods = "origin_sub_multiAngle_edge"
    dataset = Dataset_variable_TestNum(img_folder, target_folder, PCA_all_folder, methods,2)
    input_preImg, input_img, target= dataset.__getitem__(10)

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
    plt.imshow(input_preImg[0]-input_img[0],cmap = 'gray')
    plt.show()
    print(target.max(), target.min())
