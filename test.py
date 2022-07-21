import csv

import numpy as np
import os
from tools import config, tool_functions, data_loader, data_processing
import torch
import logging
from functools import partial
import math
import cv2
import yaml
# 加载自己库
from Model import *
from tools.loss_tool import PCA_loss, Log_cosh, PCA_smoothL1Loss
from tools import instanceExam, data_processing


def load_cfg(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def exam_test(args, cfg):
    # 初始化
    exam_process_dict = cfg['EXAM_PROCESS']
    total_exam_num = len(exam_process_dict)
    cur_exam_num = 0

    # 模型方式
    model_methods = {
        "CNN": CNN_model.CNN_net,
        "Unet": partial(Unet_model.UNet_net, n_classes=3),
        "Resnet": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2]),
        "Resnet_outCBAM": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_outAttention="CBAM"),
        "Resnet_outSPA": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_outAttention="SPA"),
        "Resnet_inCBAM": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="CBAM"),
        "Resnet_outSE": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_outAttention="SE"),
        "Resnet_inSPA": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="SPA"),
        "Resnet_inSE": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="SE"),
        "Resnet_allSPA": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="SPA",
                                 is_outAttention="SPA"),
        "Resnet_allCBAM": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="CBAM",
                                  is_outAttention="CBAM"),
        "Resnet_allSE": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="SE",
                                is_outAttention="SE"),
        "Resnet_inSPA_outCBAM": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="SPA",
                                        is_outAttention="CBAM"),
        "Resnet_inCBAM_outSPA": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="CBAM",
                                        is_outAttention="SPA"),
        "Resnet_inCBAM_outSE": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="CBAM",
                                       is_outAttention="SE"),
        "Resnet_inSPA_outSE": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="SPA",
                                      is_outAttention="SE"),
        "Resnet_inSE_outSPA": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="SE",
                                      is_outAttention="SPA"),
        "Resnet_inSE_outCBAM": partial(Resnet_attention.resnet, layers=[2, 2, 2, 2], is_inlineAttention="SE",
                                       is_outAttention="CBAM"),
    }
    # 评估函数方式
    estimate_method = {
        'PCA': ['MAE', 'MAE_percentage', 'R2'],
        'CBCT': ['MAD', 'MAD_precentage', 'NCC', 'SSIM'],
    }

    tool_functions.setup_seed(12)
    # 超参数设定
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for exam_cfg in exam_process_dict:
            # 实例化
            exam_instance = instanceExam.InstanceExam(args, cfg, exam_cfg)

            # 加载模型
            load_model_file = os.path.join(exam_instance.cur_ckpt_dir, str(args.EPOCH) + ".pth")
            if not os.path.exists(load_model_file):
                continue
            model = model_methods[exam_instance.model_method](exam_instance.inChannel_num)
            model.load_state_dict(torch.load(load_model_file))
            model.to(device)

            # 生成log文件
            logger = tool_functions.get_logger(
                filename=os.path.join(exam_instance.log_dir, exam_instance.work_fileName + "_test.log"),
                verbosity=1,
                name=exam_instance.work_fileName)
            # 生成csv文件
            csv_writer = tool_functions.get_csv(
                filename=os.path.join(exam_instance.csv_dir, exam_instance.work_fileName + "_test.csv"),
                header=["name"] + estimate_method[exam_instance.prediction_mode])

            #  保存切片的路径
            split_img_path = exam_instance.split_img_dir

            # 加载数据
            val_img_folder = os.path.join(args.root_path, args.val_img_folder)
            val_target_folder = os.path.join(args.root_path, args.val_target_folder)
            val_files = os.listdir(val_img_folder)
            for val_name in val_files:
                # 加载输入数据
                input_imgs = torch.tensor(data_processing.return_input_array(
                    model_type=exam_instance.model_type,
                    image_path=val_img_folder,
                    image_name=val_name,
                    preImg_num=args.preImg_num,
                ), dtype=torch.float)
                input_imgs = input_imgs.unsqueeze(0)

                # 加载GT
                val_number = val_name.split('.')[0].split('_')[1]
                GT_numpy = data_processing.load_odd_GT(exam_instance.prediction_mode, val_target_folder, val_number,
                                                       exam_instance.data_shape)
                # 预测
                prediction = model(input_imgs.to(device))
                prediction_numpy = prediction[0].cpu().numpy()  # 取出第一个预测值
                # 打印相关参数
                print(val_name, '----------------------------->')
                estimate_value_dict = {"name": val_name}
                cur_estimate_method_list = estimate_method[exam_instance.prediction_mode]
                estimate_value_dict.update(
                    data_processing.estimate_calc(GT_numpy, prediction_numpy, cur_estimate_method_list))
                logger.info(estimate_value_dict)
                csv_writer.writerow(estimate_value_dict)

                if exam_instance.prediction_mode == "CBCT":
                    split_num = args.split_num
                    cor_img = prediction_numpy[split_num[0], :, :]
                    sag_img = prediction_numpy[:, split_num[1], :]
                    tra_img = prediction_numpy[:, :, split_num[2]]
                    cv2.imread(os.path.join(split_img_path, val_number + "_cor" + "(" + split_num[0] + ").png"),
                               cor_img)
                    cv2.imread(os.path.join(split_img_path, val_number + "_sag" + "(" + split_num[0] + ").png"),
                               sag_img)
                    cv2.imread(os.path.join(split_img_path, val_number + "_tra" + "(" + split_num[0] + ").png"),
                               tra_img)
            logging.shutdown()


if __name__ == '__main__':
    args, cfg = load_cfg(yaml_path="./tools/cfg/pca_space.yaml")
    exam_test(args, cfg)
