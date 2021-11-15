import numpy as np
import os
import argparse
import cv2
from tools.config import get_args
from tools.tool_functions import get_poject_path, load_odd_file
import torch
from tools.tool_functions import *
from tools.data_loader import Dataset, img_deal_cat_variable, return_normal_para
import logging
# 加载自己模型
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image

from Model import *
from tools.config import get_args
from tools.tool_functions import *
from tools.data_loader import *

args = get_args()


def exam_test(modelMethodName=args.common_model_name, dataMethodName=args.common_data_name,
              lossFuntionMethodName=args.common_lossfunction_name):
    in_channels = get_dataMethod_num(dataMethodName)
    model_methods = {
        "CNN": CNN_model.CNN_net(in_channels),
        "Unet": Unet_model.UNet_net(in_channels, 3),
        "Resnet": Resnet_attention.resnet(in_channels),
        "Resnet_Triplet": Resnet_Triplet_atttention.resnet(in_channels, is_Triplet=True),
        "Resnet_CBAM": Resnet_attention.resnet(in_channels, is_CBAM=True),
        "Resnet_dilation": Resnet_attention.resnet(in_channels, dilation=3),
        "Resnet_CBAM_dilation": Resnet_attention.resnet(in_channels, is_CBAM=True, dilation=3)
    }
    root_path = get_poject_path("PCA")
    file_name = get_filename(__file__)
    num_cp = get_fileNum(file_name)
    testName = get_testName(__file__)
    workFileName = methodsName_combine(num_cp, modelMethodName, dataMethodName, lossFuntionMethodName)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model_methods[modelMethodName].to(device)
    load_model_file = os.path.join(get_savedir(num_cp, root_path, testName, args.gen_pca_method, workFileName),
                                   str(args.EPOCH) + ".pth")
    model.load_state_dict(torch.load(load_model_file))
    experiment_dir = get_experimentDir(num_cp, root_path, testName,
                                       args.gen_pca_method)  # "PCA/Experiment/Test1/PCA_origin/"
    log_dir = make_dir(os.path.join(experiment_dir, "log/"))
    logger = get_logger(log_dir + workFileName + "_val.log", 1, workFileName)

    # 加载数据
    val_img_folder = os.path.join(root_path, args.val_img_folder)
    val_target_folder = os.path.join(root_path, args.val_target_folder)
    val_files = os.listdir(val_img_folder)
    loss_mse = []
    loss_mse_ratio = []
    for val_name in val_files:
        img = load_odd_file(os.path.join(val_img_folder, val_name)).reshape((100, 240, 300))
        input_img = torch.tensor(
            img_deal_cat_variable(img, normal_method="max_min", data_method=dataMethodName, resize=(120, 120))[
                np.newaxis, ...])
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


def cam_model(modelMethodName=args.common_model_name, dataMethodName=args.common_data_name,
              lossFunctionMethodName=args.common_lossfunction_name):
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}

    root_path = get_poject_path("PCA")
    num_cp = get_fileNum(get_filename(__file__))
    testName = get_testName(__file__)
    workFileName = methodsName_combine(num_cp, modelMethodName, dataMethodName, lossFunctionMethodName)
    load_model_file = os.path.join(get_savedir(num_cp, root_path, testName, args.gen_pca_method, workFileName),
                                   str(args.EPOCH) + ".pth")
    model = Resnet_attention.resnet(1)
    model.load_state_dict(torch.load(load_model_file), strict=False)
    target_layer = model.layer4[2]

    cam = methods[args.cam_method](model=model, target_layer=target_layer, use_cuda=args.use_cuda)

    img_moving = np.fromfile(os.path.join(root_path, "Dataset/Test_9dvf/VAL/projection/projection_1_phase"),
                             dtype='float32').reshape((100, 240, 300))
    img_test_load = img_deal_cat_variable(img_moving, normal_method="max_min", data_method=dataMethodName,
                                          resize=(120, 120)).squeeze()
    img_test = img_test_load / img_test_load.max()
    img_test = img_test[..., np.newaxis]

    input_tensor = preprocess_image(img_test, mean=[0.485], std=[0.229])
    print(input_tensor.shape)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None
    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category,
                        aug_smooth=args.aug_smooth,
                        eigen_smooth=args.eigen_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]  # 获取cam的输出图
    cam_image = show_cam_on_image(img_test, grayscale_cam)
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    img_test_load = data_normal_0_255(img_test_load)

    cv2.imwrite(os.path.join(root_path, args.output_folder, f"{dataMethodName}_(img).jpg"), img_test_load)
    cv2.imwrite(os.path.join(root_path, args.output_folder, f'{dataMethodName}_(cam)_{args.cam_method}.jpg'), cam_image)
    cv2.imwrite(os.path.join(root_path, args.output_folder, f'{dataMethodName}_(gb)_{args.cam_method}.jpg'), gb)
    cv2.imwrite(os.path.join(root_path, args.output_folder, f'{dataMethodName}_(cam_gb)_{args.cam_method}.jpg'), cam_gb)


if __name__ == '__main__':
    # exam_test("Unet")
    cam_model(dataMethodName="origin",modelMethodName="Unet")
    cam_model(dataMethodName="sub",modelMethodName="Unet")
    cam_model(dataMethodName="edge",modelMethodName="Unet")
    cam_model(dataMethodName="multiAngle",modelMethodName="Unet")
