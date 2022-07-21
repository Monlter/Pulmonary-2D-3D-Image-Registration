import argparse
import torch
from tools.tool_functions import get_poject_path
import os


def get_args(dataset="Dataset/Digital_phantom/"):
    root_path = get_poject_path('Pulmonary-2D-3D-Image-Registration')

    parser = argparse.ArgumentParser()
    # 路径
    parser.add_argument('--root_path', type=str, default=root_path)
    parser.add_argument('--train_DRR_dir', type=str,default=os.path.join(root_path, dataset, 'Product_9dvf/projections'))
    parser.add_argument('--train_DVF_dir', type=str, default=os.path.join(root_path, dataset, 'Product_9dvf/DVFs'))
    parser.add_argument('--train_CBCT_dir', type=str, default=os.path.join(root_path, dataset, 'Product_9dvf/CTs'))
    parser.add_argument('--real_DRR_folder', type=str, default=os.path.join(root_path, dataset, "Origin/projection"))
    parser.add_argument('--real_DVF_folder', type=str, default=os.path.join(root_path, dataset, 'origin/DVF'))
    parser.add_argument('--real_CBCT_folder', type=str, default=os.path.join(root_path, dataset, 'origin/CT'))
    parser.add_argument('--PCA_dir', type=str, default=os.path.join(root_path, dataset, 'origin/PCA'))
    parser.add_argument('--val_DRR_folder', type=str,default=os.path.join(root_path, dataset, 'Product_9dvf/VAL/projections'))
    parser.add_argument("--dvf_trans_pca", type=str, default=os.path.join(root_path, dataset, "origin/DVF_trans_PCA"))

    # 超参数
    parser.add_argument('--batch_size', '-b', type=int, default=8, dest='batch_size')
    parser.add_argument('--lr', '-l', type=float, default=0.005, dest='lr')
    parser.add_argument('--val_ratio', '-r', type=float, default='0.3', dest='val_ratio')
    parser.add_argument('--epoch', '-e', type=int, default='150', dest='EPOCH')

    # 实验变量
    parser.add_argument("--preImg_num", type=int, default=1, dest="preImg_num")
    parser.add_argument('--extend_num', type=int, default=120, dest='extend_num')
    parser.add_argument("--split_num", type=int, default=(153, 300, 300), dest='split_num')

    # 函数方法
    parser.add_argument('--cam_method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++', 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam', 'eigengradcam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                             'of cam_weights*activations')

    # 设备
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    return args


if __name__ == '__main__':
    args = get_args()
