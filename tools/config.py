import argparse
import torch
from tools.tool_functions import get_poject_path


def get_args():
    parser = argparse.ArgumentParser()
    # 路径
    parser.add_argument('--img_folder', type=str, default='Dataset/Patient/5/Product_9dvf/projections',
                        dest='img_folder')
    parser.add_argument('--PCA_folder', type=str, default='Dataset/Patient/5/Product_9dvf/PCAs', dest='PCA_folder')
    # parser.add_argument('--DVF_folder', type=str, default='Dataset/Digital_phantom/Product_9dvf/CTs', dest='DVF_folder')
    # parser.add_argument('--CBCT_folder', type=str, default='Dataset/Digital_phantom/Product_9dvf/PCAs',
    #                     dest='CBCT_folder')
    parser.add_argument('--PCA_all_folder', type=str, default='Dataset/Patient/5/Product_9dvf/DVF_trans_PCAs',
                        dest='PCA_all_folder')
    parser.add_argument('--val_img_folder', type=str, default='Dataset/Patient/5/Origin/VAL/projection',
                        dest='val_img_folder')
    parser.add_argument('--val_target_folder', type=str, default='Dataset/Patient/5/Product_9dvf/PCAs/',
                        dest='val_target_folder')
    parser.add_argument("--output_folder", type=str, default='Dataset/Patient/5/Product_9dvf/Output/',
                        dest="output_folder")
    parser.add_argument("--dvf_trans_pca", type=str, default="Dataset/Patient/5/Product_9dvf/DVF_trans_PCAs")
    # parser.add_argument("--predict_dvf", type=str, default="Dataset/Digital_phantom/Product_9dvf/Output/dvf",
    #                     dest="predict_dvf")
    # parser.add_argument("--predict_ct", type=str, default="Dataset/Digital_phantom/Product_9dvf/Output/CT_dcm",
    #                     dest="predict_ct")
    # parser.add_argument("--real_ct", type=str, default="Dataset/origin/CT_dcm", dest="real_ct")
    # parser.add_argument("--reference_CBCT", type=str, default="Dataset/Digital_phantom/Origin/CT_dcm/ct_0.bin",
    #                     dest="reference_CBCT")
    # 超参数
    parser.add_argument('--batch_size', '-b', type=int, default=8, dest='batch_size')
    parser.add_argument('--lr', '-l', type=float, default=0.005, dest='lr')
    parser.add_argument('--val_ratio', '-r', type=float, default='0.3', dest='val_ratio')
    parser.add_argument('--epoch', '-e', type=int, default='150', dest='EPOCH')

    # 参数变量
    parser.add_argument('--modelMethod', '-m', type=str, default='Resnet', dest='modelMethod')
    parser.add_argument('--inputMode', type=str, default='origin', dest='inputMode')
    parser.add_argument('--lossFunctionMethod', type=str, default='MSE', dest='lossFunctionMethod')

    # 实验变量
    parser.add_argument('--testName', type=str, default='Test_space', dest='testName')
    parser.add_argument("--preImg_num", type=int, default=1, dest="preImg_num")
    parser.add_argument('--cpName', type=str, default="Test_space", dest='cpName')
    parser.add_argument('--workFileName', type=str, default="Resnet(origin_MSE)", dest='workFileName')
    parser.add_argument('--root_path', type=str, default=None, dest='root_path')
    parser.add_argument('--extend_num', type=int, default=120, dest='extend_num')
    parser.add_argument('--gen_pca_method', type=str, default='PCA_origin', dest='gen_pca_method')
    parser.add_argument("--pca_frame", type=str,
                        default="Out_result/Test_space/PCA_origin/model_cp/anayle/loss_out_csv/out_val.csv",
                        dest="pca_frame")

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
    args.root_path = get_poject_path('Pulmonary-2D-3D-Image-Registration')
    return args
