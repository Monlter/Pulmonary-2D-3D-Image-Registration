# copy from https://github.com/jacobgil/pytorch-grad-cam/blob/master/cam.py

import argparse
import cv2
import numpy as np
import torch
from torchvision import models

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
# 加载自己的库
from Model.Resnet_dilated import resnet34




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


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        train. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}

    # # 加载模型
    model = models.resnet50(pretrained=False)
    model.load_state_dict(torch.load("../checkpoint/other/resnet50-pre.pth"), strict=False)
    # Choose the target layer you want to compute the visualization for. Usually this will be the last convolutional layer in the model.
    target_layer = model.layer4[-1]


    cam = methods[args.method](model=model, target_layer=target_layer, use_cuda=args.use_cuda)


    # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    # rgb_img = np.float32(rgb_img) / 255
    # input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
    #                                          std=[0.229, 0.224, 0.225])


    img_test_load = np.fromfile(r"E:\code\pycharm\CNN_PCA\Dataset\VAL\origin\1\projection\projection_0_30",dtype='float32').reshape((100, 240, 300))
    img_test = img_test_load[25, :, :] / img_test_load[25, :, :].max()
    img_test = img_test[...,np.newaxis]
    input_tensor = preprocess_image(img_test, mean=[0.485],std=[0.229])
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
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(img_test, grayscale_cam)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
    cv2.imwrite(f'{args.method}_gb.jpg', gb)
    cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)
