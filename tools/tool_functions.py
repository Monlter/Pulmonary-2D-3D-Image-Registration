from asyncore import compact_traceback
import numpy as np
import os
import logging
import torch
import random
import re
import time
import cv2
from tools import data_processing
import SimpleITK as sitk
import csv
import yaml
import math
from .config import get_args


def get_poject_path(PROJECT_NAME):
    project_path = os.path.abspath(os.path.dirname(__file__))
    root_path = project_path[:project_path.find("{}".format(PROJECT_NAME)) + len("{}".format(PROJECT_NAME))]
    return root_path


def choose_by_prediction_mode(prediction_mode, choose_list):
    for choose_item in choose_list:
        if prediction_mode in choose_item:
            return choose_item


def normalization_2d_img(img, method='max_min'):
    img = np.array(img)
    if method == 'max_min':
        img = (img - img.min()) / (img.max() - img.min())
    elif method == 'mean_std':
        img = (img - img.mean()) / img.std()
    return img


"""删除list空项"""


def trim(list):
    list_out = []
    for i in list:
        if i != '':
            list_out.append(i)
    return list_out


def load_all_file(file_folder, shape=None):
    file_name_list = sorted(os.listdir(file_folder))
    file_list = []
    for i, file_name in enumerate(file_name_list):
        file_path = os.path.join(file_folder, file_name)
        if os.path.isdir(file_path):
            # 为dcm文件
            file = data_processing.readDicomSeries(file_path)
        else:
            file = np.fromfile(os.path.join(file_folder, file_name), dtype='float32')
        file_list.append(file)
    files_array = np.array(file_list)
    if shape:
        shape = (i,) + shape
        files_array.reshape(shape)
    return files_array


def load_odd_file(filename, shape=None):
    file_array = np.fromfile(filename, dtype='float32')
    if shape:
        file_array = file_array.reshape(shape)
    return file_array


def file_save(data, name, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    data.tofile(os.path.join(save_folder, name))


def make_dir(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def get_dataMethodName(data_methods):
    returnstr = "origin"
    if data_methods["edge"]:
        returnstr += "_edge"
    if data_methods["sub"]:
        returnstr += "_sub"
    if data_methods["multi_angle"]:
        returnstr += "_multiAngle"
    return returnstr


def get_csv(filename, header):
    f = open(filename, 'w', newline='')
    csv_writer = csv.DictWriter(f, fieldnames=header)
    csv_writer.writeheader()
    return csv_writer


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # formatter = logging.Formatter(
    #     "[%(asctime)s] %(message)s"
    # )
    # %(asctime)s：当前时间
    # %(message)s ：用户输出的消息
    formatter = logging.Formatter(
        "%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def adjust_multichannels(img):
    if len(img.shape) == 2:
        return img[..., np.newaxis]
    else:
        return img


def recode_progressNum(num):
    print("-" * 100)
    print("progress_" + str(num), "start:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def methodsName_combine(args):
    returnstr = ''
    if args.testName[5:] == "spaceAndTime":
        if args.cpName == "model_cp":
            returnstr = args.modelMethod + "(" + args.inputMode + "_" + args.lossFunctionMethod + "_pre" + str(
                args.preImg_num) + ")"
        elif args.cpName == "data_cp":
            returnstr = args.inputMode + "(" + args.modelMethod + "_" + args.lossFunctionMethod + "_pre" + str(
                args.preImg_num) + ")"
        elif args.cpName == "loss_cp":
            returnstr = args.lossFunctionMethod + "(" + args.modelMethod + "_" + args.inputMode + "_pre" + str(
                args.preImg_num) + ")"
        print("modelMethod:", args.modelMethod, "\tdataMethod:", args.inputMode, "\tlossfunction:",
              args.lossFunctionMethod, "\tpreImg_num:", args.preImg_num)
    else:
        if args.cpName == "model_cp":
            returnstr = args.modelMethod + "(" + args.inputMode + "_" + args.lossFunctionMethod + ")"
        elif args.cpName == "data_cp":
            returnstr = args.inputMode + "(" + args.modelMethod + "_" + args.lossFunctionMethod + ")"
        elif args.cpName == "loss_cp":
            returnstr = args.lossFunctionMethod + "(" + args.modelMethod + "_" + args.inputMode + ")"
        print("modelMethod:", args.modelMethod, "\tdataMethod:", args.inputMode, "\tlossfunction:",
              args.lossFunctionMethod)
    return returnstr


def get_out_result_dir(args):
    if args.testName.split("_")[1] == "spaceAndTime":
        returnstr = os.path.join(args.root_path, "Out_result", args.testName, args.cpName)
    else:
        returnstr = os.path.join(args.root_path, "Out_result", args.testName, args.cpName)
    _ = make_dir(returnstr)
    print("running model class is :", args.testName.split("_")[1])
    return returnstr


def get_checkpoint_dir(args):
    if args.testName.split("_")[1] == "spaceAndTime":
        returnstr = os.path.join(args.root_path, "checkpoint", args.testName, args.cpName, args.workFileName)
    else:
        returnstr = os.path.join(args.root_path, "checkpoint", args.testName, args.cpName, args.workFileName)
    _ = make_dir(returnstr)
    return returnstr


def get_filename(file):
    filename = os.path.basename(file)  # 返回path最后的文件名
    return filename


# 获取实验名：pca_space ,pca_spaceAndTime
def get_testName(file):
    project_path = os.path.abspath(os.path.dirname(file))
    testName = project_path.split("\\")[-1]
    return testName


def get_cpName(file):
    cpName = re.findall(re.compile("[{](.*?)[}]"), file)[0]  # 返回{}内的内容
    return cpName


def get_fileType(filename):
    returnstr = filename[:filename.find("_")]
    return returnstr


def get_channelNum(dataMethodName):
    num = 0
    if dataMethodName.find("origin") != -1:
        num += 1
    if dataMethodName.find("multiAngle") != -1:
        num += 1
    if dataMethodName.find("edge") != -1:
        num += 1
    if dataMethodName.find("sub") != -1:
        num += 1
    return num


def get_logfilename(num_cp, modelMethod, dataMethod, lossFunctionMethod):
    returnstr = ''
    if num_cp == 1:
        returnstr = modelMethod
    elif num_cp == 2:
        returnstr = dataMethod
    elif num_cp == 3:
        returnstr = modelMethod
    elif num_cp == 4:
        returnstr = lossFunctionMethod
    return returnstr


def resize_img(img, resize):
    """
    :param img: img.shape(C,H,W) or (H,W)
    :param resize: (resize_h,reszie_w)
    :return: resizeImg.shape(C,H,W) or (H,W)
    """
    if len(img.shape) == 3:
        # 默认认为shape为CHW
        if img.shape[0] > 4:
            img_result = []
            for i in range(img.shape[0]):
                img_result.append(cv2.resize(img[i], resize, interpolation=cv2.INTER_AREA))
            return np.array(img_result)
        img_trans = img.transpose((1, 2, 0))
        resizeImg_trans = cv2.resize(img_trans, resize, interpolation=cv2.INTER_AREA)
        resizeImg = resizeImg_trans.transpose((2, 0, 1))
    else:
        resizeImg = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
    return resizeImg


def save_png(imgs_numpy, save_path, save_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_numpy = data_processing.data_standardization_0_255(imgs_numpy)
    cv2.imwrite(os.path.join(save_path, save_name + ".png"), img_numpy)


def load_pca_para(PCA_para_folder):
    file_list = os.listdir(PCA_para_folder)
    pca_components = 0
    pca_mean = 0
    for file_name in file_list:
        if file_name.startswith("PCA_components_"):
            pca_components = np.fromfile(os.path.join(PCA_para_folder, file_name), dtype='float32').reshape(
                (3, -1))
            print("pca_components 已经加载")
        elif file_name.startswith("PCA_mean_"):
            pca_mean = np.fromfile(os.path.join(PCA_para_folder, file_name), dtype='float32')
            print("pca_mean 已经加载")
    return pca_components, pca_mean


def pca_trans_origin(pca, pca_component, pca_mean):
    # inverse_transform()
    return np.dot(pca, pca_component) + pca_mean


def ImageResampleBySpacing(sitk_image,old_spacing,new_spacing=[1.0, 1.0, 1.0], is_label=False):
    '''
    sitk_image:
    new_spacing: x,y,z
    is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
    '''
    size = np.array(sitk_image.GetSize())
    spacing = np.array(old_spacing)
    sitk_image.SetSpacing(spacing)
    new_spacing = np.array(new_spacing)
    new_size = size * spacing / new_spacing
    new_spacing_refine = size * spacing / new_size
    new_spacing_refine = [float(s) for s in new_spacing_refine]
    new_size = [int(math.ceil(s)) for s in new_size]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing_refine)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    newimage = resample.Execute(sitk_image)
    return newimage


def ImageResizeByShape(sitk_image,old_shape,new_shape,resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    oldSize = sitk_image.GetSize()
    oldSpacing = sitk_image.GetSpacing()
    newSize = np.array(new_shape, float)
    newSize = newSize.astype(np.int)  # spacing肯定不能是整数
    resampler.SetReferenceImage(sitk_image)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    factor = oldSize / newSize
    newSpacing = oldSpacing * factor
    newSize = newSize.astype(np.int)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(sitk_image)  # 得到重新采样后的图像
    # itkimgResampled.SetOutputSpacing(oldSpacing.tolist())
    return itkimgResampled




def load_cfg(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        args = get_args(dataset=cfg["DATASET"])
    return args, cfg


def calc_param(model, mode="time"):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0
    total_params = sum(p.numel() for p in model.parameters()) / 1000000.0
    return trainable_params, total_params


def save_all_split(image,shape,save_path,prefix_name,suffix_name):
    assert len(shape) == len(suffix_name),"list shape different!"
    for i, cur_shape in enumerate(shape):
        for j in range(cur_shape):
            if i == 0:
                img = image[j,:,:]
            elif i == 1:
                img = image[:,j,:]
            elif i == 2:
                img = image[:,:,j]
            img_split = normalization_2d_img(img) * 255
            cv2.imwrite(os.path.join(save_path, str(prefix_name) + "_"+ suffix_name[i] + "(" + str(j) + ").png"), img_split)


if __name__ == '__main__':
    pass
