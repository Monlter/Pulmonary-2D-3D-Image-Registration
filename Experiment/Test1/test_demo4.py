import numpy as np
import os

from tools.config import get_args
from tools.tool_functions import get_poject_path, load_odd_file
import torch
from tools.tool_functions import *
from tools.data_loader import Dataset, img_deal_cat_variable, return_normal_para
import logging
# 加载自己模型
from Model import *

args = get_args()


def exam_test(modelMethodName=args.common_model_name, dataMethodName=args.common_data_name,
              lossFuntionMethodName=args.common_lossfunction_name):
    in_channels = get_dataMethod_num(dataMethodName)
    model_methods = {
        "CNN": CNN_model.CNN_net(in_channels),
        "Unet": Unet_model.UNet_net(in_channels, 3),
        "Resnet": Resnet_attention.resnet(in_channels),
        "Resnet_Triplet":Resnet_Triplet_atttention.resnet(in_channels,is_Triplet=True),
        "Resnet_CBAM":Resnet_attention.resnet(in_channels,is_CBAM=True),
        "Resnet_dilation":Resnet_attention.resnet(in_channels,dilation=3),
        "Resnet_CBAM_dilation": Resnet_attention.resnet(in_channels,is_CBAM=True, dilation=3)
    }
    root_path = get_poject_path("Pulmonary-2D-3D-Image-Registration")
    file_name = get_filename(__file__)
    num_cp = get_fileNum(file_name)
    testName = get_testName(__file__)
    workFileName = methodsName_combine(num_cp, modelMethodName, dataMethodName, lossFuntionMethodName)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model_methods[modelMethodName].to(device)
    load_model_file = os.path.join(get_savedir(num_cp, root_path, testName, args.gen_pca_method, workFileName), str(args.EPOCH) + ".pth")
    model.load_state_dict(torch.load(load_model_file))
    experiment_dir = get_experimentDir(num_cp, root_path, testName, args.gen_pca_method) # "PCA/Experiment/Test1/PCA_origin/"
    log_dir = make_dir(os.path.join(experiment_dir, "log/"))
    logger = get_logger(log_dir + workFileName+ "_val.log",1,workFileName)

    # 加载数据
    val_img_folder = os.path.join(root_path, args.val_img_folder)
    val_target_folder = os.path.join(root_path, args.val_target_folder)
    val_files = os.listdir(val_img_folder)
    loss_mse = []
    loss_mse_ratio = []
    for val_name in val_files:
        img = load_odd_file(os.path.join(val_img_folder, val_name)).reshape((100, 240, 300))
        input_img = torch.tensor(img_deal_cat_variable(img, normal_method="max_min", data_method=dataMethodName, resize=(120, 120))[np.newaxis, ...])
        img_number = val_name.split("_")[1]
        pca_name = "PCA_" + img_number
        pca = load_odd_file(os.path.join(val_target_folder, pca_name))
        prediction = model(input_img.to(device))
        prediction = prediction[0].cpu().detach().numpy()
        print(val_name)
        print("prediction:", prediction, "orgin:", pca)
        sub_value = prediction - pca
        sub_ratio = (abs(sub_value) / abs(pca)) * 100
        logger.info(
        val_name + "sub_value:" + str(sub_value) + "\tsub_ration(%):" + str(sub_ratio) + "\tprediction:" + str(
            prediction) + "\torgin:" + str(pca))
        print("-" * 100)
        loss_mse.append(sub_value)
        loss_mse_ratio.append(sub_ratio)
    logger.info("PCA_3_mean:" + str(np.mean(np.abs(loss_mse), axis=0)) + "\tPCA_3_mean_ratio(%)" + str(
        np.mean(loss_mse_ratio, axis=0)))
    logger.info("PCA_mse:" + str(np.mean(np.abs(loss_mse))))
    logging.shutdown()

if __name__ == '__main__':
    recode_progressNum(1)
    exam_test("Unet")
