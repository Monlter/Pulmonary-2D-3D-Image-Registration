# copy from https://github.com/jacobgil/pytorch-grad-cam/blob/master/cam.py

import argparse
import cv2
import numpy as np
import torch
from torchvision import models
import os
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
# 加载自己的库
from Model.Resnet_attention import resnet
from tools.data_loader import *
from tools.tool_functions import *
from tools.config import get_args as get_myargs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default=r'E:\code\jupyter notebook\test_code\image\both.png',
                        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++', 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam', 'eigengradcam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


args_my = get_myargs()
args = get_args()
def cam_model(modelMethodName=args_my.common_model_name,dataMethodName=args_my.common_data_name,lossFunctionMethodName=args_my.common_lossfunction_name):
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}

    root_path = get_poject_path("Pulmonary-2D-3D-Image-Registration")
    num_cp = get_fileNum(get_filename(__file__))
    workFileName = methodsName_combine(num_cp, modelMethodName, dataMethodName, lossFunctionMethodName)
    load_model_file = os.path.join(root_path,get_savedir(num_cp, root_path, "Test1", args_my.gen_pca_method, workFileName),
                                   str(args_my.EPOCH) + ".pth")
    model = resnet(1,[2,2,2,2]).to("cuda:0")
    model.load_state_dict(torch.load(load_model_file), strict=False)
    target_layer = model.layer4[-1]


    cam = methods[args.method](model=model, target_layer=target_layer, use_cuda=args.use_cuda)

    img_moving = np.fromfile(os.path.join(root_path, "Dataset/Origin/VAL/projection/projection_1_phase"), dtype='float32').reshape((100, 240, 300))
    img_test_load = img_deal_cat_variable(img_moving,normal_method="max_min",data_method=dataMethodName, resize=(120, 120)).squeeze()
    img_test = img_test_load / img_test_load.max()
    img_test = img_test[..., np.newaxis]

    input_tensor = preprocess_image(img_test, mean=[0.485], std=[0.229])
    input_tensor = torch.Tensor(img_test[np.newaxis,...]).permute((0,3,1,2)).to("cuda:0")
    print(input_tensor.size())
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None
    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 1
    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category,
                        aug_smooth=args.aug_smooth,
                        eigen_smooth=args.eigen_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]   # 获取cam的输出图
    cam_image = show_cam_on_image(img_test, grayscale_cam)
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    img_test_load = data_normal_0_255(img_test_load)

    cv2.imwrite("origin_img.jpg", img_test_load)
    cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
    cv2.imwrite(f'{args.method}_gb.jpg', gb)
    cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        train. Guided Back Propagation
        3. Combining both
    """
    cam_model()



