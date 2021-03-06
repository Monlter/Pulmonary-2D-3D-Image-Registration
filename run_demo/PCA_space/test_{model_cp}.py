import numpy as np
import os
import cv2
from tools.config import get_args
from tools.tool_functions import get_poject_path, load_odd_file
import torch
from tools.tool_functions import *
from tools.data_loader import Dataset, input_mode_concat_variable
import logging
# 加载自己模型
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
# 加载自己模型
from Model import *
np.set_printoptions(suppress=True)

args = get_args()


def init_args(modelMethodName, inputModeName,lossFunctionMethodName):
    args.modelMethod = modelMethodName
    args.inputMode = inputModeName
    args.lossFunctionMethod = lossFunctionMethodName

    current_file = get_filename(__file__)
    cpName = get_cpName(current_file)
    args.cpName = cpName

    testName = get_testName(__file__)
    args.testName = testName

    root_path = get_poject_path("Pulmonary-2D-3D-Image-Registration")
    args.root_path = root_path

    workFileName = methodsName_combine(args)
    args.workFileName = workFileName


def cam_model(model_obj,model_name,img,save_dir,img_name):
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}
    if model_name == "CNN":
        target_layer = model_obj.conv3[-1]
    elif "Unet":
        target_layer = model_obj.up4.conv.double_conv[-1]
    else :
        target_layer = model_obj.layer4[-1]

    print("求导层：",target_layer)
    cam = methods[args.cam_method](model=model_obj, target_layer=target_layer,use_cuda=args.use_cuda)
    img_normal = img/img.max()
    out_img = np.array(img_normal).squeeze(0).transpose((1,2,0))
    print(out_img.shape)
    if args.use_cuda:
        input_tensor = preprocess_image(out_img,mean=[0.485], std=[0.229]).to("cuda:0")
    else:
        input_tensor = preprocess_image(out_img,mean=[0.485], std=[0.229])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None
    cam.batch_size = 1

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category,
                        aug_smooth=args.aug_smooth,
                        eigen_smooth=args.eigen_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0]  # 获取cam的输出图
    cam_image = show_cam_on_image(out_img, grayscale_cam)

    gb_model = GuidedBackpropReLUModel(model=model_obj, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(os.path.join(save_dir, f"{img_name}_(img).jpg"), data_normal_0_255(out_img))
    cv2.imwrite(os.path.join(save_dir,f'{img_name}_(cam)_{args.cam_method}.jpg'), cam_image)
    cv2.imwrite(os.path.join(save_dir,f'{img_name}_(gb)_{args.cam_method}.jpg'), gb)
    cv2.imwrite(os.path.join(save_dir,f'{img_name}_(cam_gb)_{args.cam_method}.jpg'),cam_gb )


def exam_test(modelMethodName=args.modelMethod, inputModeName=args.inputMode,lossFunctionMethodName=args.lossFunctionMethod):
    in_channels = get_channelNum(inputModeName)
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
    init_args(modelMethodName, inputModeName, lossFunctionMethodName)
    root_path = get_poject_path("Pulmonary-2D-3D-Image-Registration")
    pca_components_path = os.path.join(root_path,args.dvf_trans_pca,"PCA_components_(3,29491200)")
    pca_mean_path = os.path.join(root_path,args.dvf_trans_pca,"PCA_mean_(29491200,)")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model_methods[modelMethodName].to(device)
    load_model_file = os.path.join(get_checkpoint_dir(args),
                                   str(args.EPOCH) + ".pth")
    model.load_state_dict(torch.load(load_model_file),strict=False)
    out_result_dir = get_out_result_dir(args)  # "Pulmonary-2D-3D-Image-Registration/run_demo/Test_space/PCA_origin/"
    log_dir = make_dir(os.path.join(out_result_dir, "log/"))
    logger = get_logger(log_dir + args.workFileName + "_val.log",1,args.workFileName)

    # 加载数据
    val_img_folder = os.path.join(root_path, args.val_img_folder)
    val_target_folder = os.path.join(root_path, args.val_target_folder)
    val_files = os.listdir(val_img_folder)
    val_files.sort()
    loss_mse = []
    loss_mse_ratio = []

    logger.info(
        "TestName:" + str(args.testName) + "\tdata_method:" + str(args.gen_pca_method) + "\tmodel:" + str(
            modelMethodName) + '\tdataMethod:' + str(inputModeName) + '\tloss_function:' + str(lossFunctionMethodName))

    # 开始运行
    for val_name in val_files:
        # with torch.no_grad():
            img = load_odd_file(os.path.join(val_img_folder, val_name)).reshape((100, 240, 300))[25]
            input_img = torch.tensor(input_mode_concat_variable(img, standardization_method="max_min", input_mode_names=inputModeName, resize=(120, 120))[np.newaxis, ...])
            # cam_model(model,modelMethodName,input_img,os.path.join(root_path,"anayle/cam",str(modelMethodName)),val_name)
            img_number = val_name.split("_")[1]
            pca_name = "PCA_" + img_number
            GT_pca = load_odd_file(os.path.join(val_target_folder, pca_name))
            prediction = model(input_img.to(device))
            prediction = prediction[0].cpu().detach().numpy()
            # 保存pca_trans_dvf
            pca_components = load_odd_file(pca_components_path).reshape(3,29491200)
            pca_mean = load_odd_file(pca_mean_path)
            predict_dvf = pca_trans_origin(prediction,pca_components,pca_mean)
            file_save(predict_dvf,"predict_dvf_"+img_number,os.path.join(root_path,args.predict_dvf,args.workFileName))
            # 保存log文件
            print(val_name)
            print("prediction:", prediction, "orgin:", GT_pca)
            sub_value = prediction - GT_pca
            sub_ratio = (abs(sub_value) / abs(GT_pca)) * 100
            logger.info(
                val_name + "sub_value:" + str(sub_value) + "\tsub_ration(%):" + str(sub_ratio) + "\tprediction:" + str(
                    prediction) + "\tGT:" + str(GT_pca))
            print("-"*100)
            loss_mse.append(sub_value)
            loss_mse_ratio.append(sub_ratio)
    logger.info("PCA_3_sub_value_mean:" + str(np.around(np.mean(np.abs(loss_mse), axis=0),2)) + "\tPCA_3_sub_value_mean_ratio(%)" + str(np.around(np.mean(loss_mse_ratio, axis=0),2)))
    logger.info("PCA_sub_value_mse:" + str(np.around(np.mean(np.abs(loss_mse)),2)))
    logging.shutdown()


if __name__ == '__main__':
    recode_progressNum(1)
    exam_test(modelMethodName="CNN")
    # recode_progressNum(2)
    # exam_test(modelMethodName="Unet")
    # recode_progressNum(3)
    # exam_test(modelMethodName="Resnet")
    recode_progressNum(4)
    exam_test(modelMethodName="Resnet_outCBAM")
    recode_progressNum(5)
    exam_test(modelMethodName="Resnet_dilation")
    # recode_progressNum(6)
    # exam_test(modelMethodName="Resnet_outSPA_dilation")
    # recode_progressNum(7)
    # exam_test(modelMethodName="Resnet_outCBAM_dilation")
    recode_progressNum(8)
    exam_test(modelMethodName="Resnet_outSPA")
    recode_progressNum(9)
    exam_test(modelMethodName="Resnet_inCBAM")
    recode_progressNum(10)
    exam_test(modelMethodName="Resnet_inSPA")
    recode_progressNum(11)
    exam_test(modelMethodName="Resnet_allSPA")
    recode_progressNum(12)
    exam_test(modelMethodName="Resnet_allCBAM")
    recode_progressNum(13)
    exam_test(modelMethodName="Resnet_inSPA_outCBAM")
    # recode_progressNum(14)
    # exam_test(modelMethodName="Resnet_inCBAM_outSPA")
    # recode_progressNum(15)
    # exam_test(modelMethodName="Resnet_outSE")
    # recode_progressNum(16)
    # exam_test(modelMethodName="Resnet_inSE")
    # recode_progressNum(17)
    # exam_test(modelMethodName="Resnet_allSE")
    # recode_progressNum(18)
    # exam_test(modelMethodName="Resnet_inSPA_outSE")
    # recode_progressNum(19)
    # exam_test(modelMethodName="Resnet_inSE_outSPA")
    # recode_progressNum(20)
    # exam_test(modelMethodName="Resnet_inCBAM_outSE")
    # recode_progressNum(21)
    # exam_test(modelMethodName="Resnet_inSE_outCBAM")
