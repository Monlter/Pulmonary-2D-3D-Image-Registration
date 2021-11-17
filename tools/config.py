import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', '-i', type=str, default='Dataset/Test_9dvf/projections', dest='img_folder')
    parser.add_argument('--target_folder', '-t', type=str, default='Dataset/Test_9dvf/PCAs', dest='target_folder')
    parser.add_argument('--PCA_all_folder', '-p', type=str, default=r'Dataset/Test_9dvf/DVF_trans_PCAs',
                        dest='PCA_all_folder')
    parser.add_argument('--val_img_folder', type=str, default='Dataset/Test_9dvf/VAL/projection/',
                        dest='val_img_folder')
    parser.add_argument('--val_target_folder', type=str, default='Dataset/Test_9dvf/PCAs/', dest='val_target_folder')
    parser.add_argument("--output_folder", type=str, default='Dataset/Test_9dvf/Output/', dest="output_folder")
    parser.add_argument('--batch_size', '-b', type=int, default=8, dest='batch_size')
    parser.add_argument('--lr', '-l', type=float, default=0.001, dest='lr')
    parser.add_argument('--save_dir', '-s', type=str, default=r'checkpoint/', dest='save_dir')
    parser.add_argument('--val_ratio', '-r', type=float, default='0.3', dest='val_ratio')
    parser.add_argument('--epoch', '-e', type=int, default='150', dest='EPOCH')
    parser.add_argument('--common_model_name', '-m', type=str, default='Resnet', dest='common_model_name')
    parser.add_argument('--common_data_name', type=str, default='origin', dest='common_data_name')
    parser.add_argument('--common_lossfuntion_name', type=str, default='MSE', dest='common_lossfunction_name')
    parser.add_argument('--Experiment', '-n', type=str, default='Test1', dest='experiment')
    parser.add_argument('--extend_num', type=int, default=300, dest='extend_num')
    parser.add_argument('--gen_pca_method', type=str, default='PCA_origin', dest='gen_pca_method')
    parser.add_argument("--dvf_trans_pca", type=str,default="Dataset/Test_9dvf/DVF_trans_PCAs")
    parser.add_argument("--predict_dvf", type=str, default="Dataset/Test_9dvf/Output/dvf",dest="predict_dvf")
    parser.add_argument("--predict_ct", type=str, default="Dataset/Test_9dvf/Output/CT", dest="predict_ct")
    parser.add_argument("--real_ct", type=str, default="Dataset/Origin/CT", dest="real_ct")
    parser.add_argument('--cam_method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++', 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam', 'eigengradcam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam')
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                             'of cam_weights*activations')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    return args
