import numpy as np
import os

from tools import config, tool_functions, data_loader, pca_data_processing
import torch

import logging
# 加载自己模型
from Model import *

args = config.get_args()


def init_args(modelMethodName, inputModeName, lossFunctionMethodName):
    args.modelMethod = modelMethodName
    args.inputMode = inputModeName
    args.lossFunctionMethod = lossFunctionMethodName

    current_file = tool_functions.get_filename(__file__)
    cpName = tool_functions.get_cpName(current_file)
    args.cpName = cpName

    testName = tool_functions.get_testName(__file__)
    args.testName = testName

    root_path = tool_functions.get_poject_path("Pulmonary-2D-3D-Image-Registration")
    args.root_path = root_path

    workFileName = tool_functions.methodsName_combine(args)
    args.workFileName = workFileName


def exam_test(modelMethodName=args.modelMethod, inputModeName=args.inputMode,
              lossFunctionMethodName=args.lossFunctionMethod):
    in_channels = tool_functions.get_channelNum(inputModeName)
    model_methods = {
        "ConvLSTM": convLSTM_model.ConvLSTM_Liner(in_channels),
    }
    init_args(modelMethodName, inputModeName, lossFunctionMethodName)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model_methods[modelMethodName].to(device)
    load_model_file = os.path.join(tool_functions.get_checkpoint_dir(args),
                                   str(args.EPOCH) + ".pth")
    model.load_state_dict(torch.load(load_model_file))
    out_result_dir = tool_functions.get_out_result_dir(args)
    log_dir = tool_functions.make_dir(os.path.join(out_result_dir, 'log/'))
    logger = tool_functions.get_logger(log_dir + args.workFileName + "_val.log", 1, args.workFileName)

    # 加载数据
    val_img_folder = os.path.join(args.root_path, args.val_img_folder)
    val_target_folder = os.path.join(args.root_path, args.val_target_folder)
    val_files = os.listdir(val_img_folder)
    loss_mse = []
    loss_mse_ratio = []
    for val_name in val_files:
        # 加载输入数据
        cur_img = tool_functions.load_odd_file(os.path.join(val_img_folder, val_name)).reshape((100, 240, 300))[25]
        input_cur_img = pca_data_processing.input_mode_concat_variable(cur_img, standardization_method="max_min",
                                                                       input_mode_names=inputModeName,
                                                                       resize=(120, 120))
        pre_imgs = pca_data_processing.load_projection_sequence(val_img_folder,
                                                                pca_data_processing.get_preImgName_sequence(val_name,
                                                                                                            args.preImg_num))
        input_pre_imgs = [
            pca_data_processing.input_mode_concat_variable(pre_img, standardization_method="max_min",
                                                           input_mode_names=inputModeName,
                                                           resize=(120, 120)) for pre_img in pre_imgs]
        print(np.array(input_pre_imgs).shape)
        print(np.array([input_cur_img]).shape)
        imgs = input_pre_imgs + [input_cur_img]
        input_imgs = torch.tensor(np.array(imgs)[np.newaxis, ...])

        print(input_imgs.shape)

        # 加载输入标签
        img_number = val_name.split("_")[1]
        pca_name = "PCA_" + img_number
        GT_pca = tool_functions.load_odd_file(os.path.join(val_target_folder, pca_name))

        # 预测
        prediction = model(input_imgs.to(device))
        prediction = prediction[0].cpu().detach().numpy()
        print(val_name)
        print("prediction:", prediction, "GT:", GT_pca)
        sub_value = prediction - GT_pca
        sub_ratio = (abs(sub_value) / abs(GT_pca)) * 100
        logger.info(
            val_name + "sub_value:" + str(sub_value) + "\tsub_ration(%):" + str(sub_ratio) + "\tprediction:" + str(
                prediction) + "\tGT:" + str(GT_pca))
        print("-" * 100)
        loss_mse.append(sub_value)
        loss_mse_ratio.append(sub_ratio)
    logger.info("PCA_3_mean:" + str(np.mean(np.abs(loss_mse), axis=0)) + "\tPCA_3_mean_ratio(%)" + str(
        np.mean(loss_mse_ratio, axis=0)))
    logger.info("PCA_mse:" + str(np.mean(np.abs(loss_mse))))
    logging.shutdown()


if __name__ == '__main__':
    tool_functions.recode_progressNum(1)
    exam_test(modelMethodName="ConvLSTM")
