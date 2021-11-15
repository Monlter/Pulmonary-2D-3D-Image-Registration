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
np.set_printoptions(suppress=True)

args = get_args()




def exam_test(modelMethodName=args.common_model_name, dataMethodName=args.common_data_name,
              lossFuntionMethodName=args.common_lossfunction_name):
    in_channels = get_dataMethod_num(dataMethodName)
    model_methods = {
        "CNN": CNN_model.CNN_net(in_channels),
        "Unet": Unet_model.UNet_net(in_channels, 3),
        "Resnet": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2]),
        "Resnet_outTriplet": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_outAttention="Triplet"),
        "Resnet_outCBAM": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_outAttention="CBAM"),
        "Resnet_dilation": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], dilation=3),
        "Resnet_outSPA_dilation": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_outAttention="SPA",
                                                          dilation=3),
        "Resnet_outCBAM_dilation": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_outAttention="CBAM",
                                                           dilation=3),
        "Resnet_outSPA": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_outAttention="SPA"),
        "Resnet_inCBAM": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="CBAM"),
        "Resnet_outSE": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_outAttention="SE"),
        "Resnet_inSPA": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="SPA"),
        "Resnet_inSE": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="SE"),
        "Resnet_allSPA": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="SPA",
                                                 is_outAttention="SPA"),
        "Resnet_allCBAM": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="CBAM",
                                                  is_outAttention="CBAM"),
        "Resnet_allSE": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="SE",
                                                is_outAttention="SE"),
        "Resnet_inSPA_outCBAM": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="SPA",
                                                        is_outAttention="CBAM"),
        "Resnet_inCBAM_outSPA": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="CBAM",
                                                        is_outAttention="SPA"),
        "Resnet_inCBAM_outSE": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="CBAM",
                                                       is_outAttention="SE"),
        "Resnet_inSPA_outSE": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="SPA",
                                                      is_outAttention="SE"),
        "Resnet_inSE_outSPA": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="SE",
                                                      is_outAttention="SPA"),
        "Resnet_inSE_outCBAM": Resnet_attention.resnet(in_channels, layers=[2, 2, 2, 2], is_inlineAttention="SE",
                                                       is_outAttention="CBAM"),
    }
    root_path = get_poject_path("Pulmonary-2D-3D-Image-Registration")
    file_name = get_filename(__file__)
    num_cp = get_fileNum(file_name)
    testName = get_testName(__file__)
    pca_components_path = os.path.join(root_path,args.dvf_trans_pca,"PCA_components_(3,29491200)")
    pca_mean_path = os.path.join(root_path,args.dvf_trans_pca,"PCA_mean_(29491200,)")
    workFileName = methodsName_combine(num_cp, modelMethodName, dataMethodName, lossFuntionMethodName)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model_methods[modelMethodName].to(device)
    load_model_file = os.path.join(get_savedir(num_cp, root_path, testName, args.gen_pca_method, workFileName),
                                   str(args.EPOCH) + ".pth")
    model.load_state_dict(torch.load(load_model_file),strict=False)
    experiment_dir = get_experimentDir(num_cp, root_path, testName,
                                       args.gen_pca_method)  # "Pulmonary-2D-3D-Image-Registration/Experiment/Test1/PCA_origin/"
    log_dir = make_dir(os.path.join(experiment_dir, "log/"))
    logger = get_logger(log_dir + workFileName + "_val.log",1,workFileName)

    # 加载数据
    val_img_folder = os.path.join(root_path, args.val_img_folder)
    val_target_folder = os.path.join(root_path, args.val_target_folder)
    val_files = os.listdir(val_img_folder)
    val_files.sort()
    loss_mse = []
    loss_mse_ratio = []

    logger.info(
        "Experiment:" + str(testName) + "\tdata_method:" + str(args.gen_pca_method) + "\tmodel:" + str(
            modelMethodName) + '\tdataMethod:' + str(dataMethodName))

    # 开始运行
    for val_name in val_files:
        with torch.no_grad():
            img = load_odd_file(os.path.join(val_img_folder, val_name)).reshape((100, 240, 300))
            input_img = torch.tensor(
                img_deal_cat_variable(img, normal_method="max_min", data_method=dataMethodName, resize=(120, 120))[
                    np.newaxis, ...])
            img_number = val_name.split("_")[1]
            pca_name = "PCA_" + img_number
            pca = load_odd_file(os.path.join(val_target_folder, pca_name))
            prediction = model(input_img.to(device))
            prediction = prediction[0].cpu().detach().numpy()
            # 保存pca_trans_dvf
            pca_components = load_odd_file(pca_components_path).reshape(3,29491200)
            pca_mean = load_odd_file(pca_mean_path)
            predict_dvf = pca_trans_origin(prediction,pca_components,pca_mean)
            file_save(predict_dvf,"predict_dvf_"+img_number,os.path.join(root_path,args.predict_dvf,workFileName))
            # 保存log文件
            print(val_name)
            print("prediction:", prediction, "orgin:", pca)
            sub_value = prediction - pca
            sub_ratio = (abs(sub_value) / abs(pca)) * 100
            logger.info(
                val_name + "\tprediction:" + str(np.around(prediction,2)) + "\torgin:" + str(np.around(pca,2))+ "\tsub_value:" + str(np.around(sub_value,2)) + "\tsub_ration(%):" + str(np.around(sub_ratio,2)))
            print("-" * 100)
            loss_mse.append(sub_value)
            loss_mse_ratio.append(sub_ratio)
    logger.info("PCA_3_sub_value_mean:" + str(np.around(np.mean(np.abs(loss_mse), axis=0),2)) + "\tPCA_3_sub_value_mean_ratio(%)" + str(np.around(np.mean(loss_mse_ratio, axis=0),2)))
    logger.info("PCA_sub_value_mse:" + str(np.around(np.mean(np.abs(loss_mse)),2)))
    logging.shutdown()


if __name__ == '__main__':
    # recode_progressNum(1)
    # exam_test(modelMethodName="CNN")
    # recode_progressNum(2)
    # exam_test(modelMethodName="Unet")
    # recode_progressNum(3)
    # exam_test(modelMethodName="Resnet")
    # recode_progressNum(4)
    # exam_test(modelMethodName="Resnet_dilation")
    recode_progressNum(5)
    exam_test(modelMethodName="Resnet_outCBAM")
