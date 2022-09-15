import numpy as np
import os
import pandas as pd
import csv
import logging
import cv2
# 加载自己的库
from tools import data_processing,instanceExam,tool_functions


def exam_extimate(args,cfg):
    exam_process_dict = cfg['EXAM_PROCESS']
    # 评估函数方式
    estimate_method = {
        'PCA': ['MAE', 'MAE_percentage', 'R2'],
        'CT': ['MAD', 'MAD_percentage', 'NCC', 'SSIM'],
    }
    for exam_cfg in exam_process_dict:
        # 实例化
        exam_instance = instanceExam.InstanceExam(args, cfg, exam_cfg)
        print(exam_instance.compare_mode,"--",exam_instance.work_fileName, "开始评估---------------------------->")
        pred_PCA_dir = os.path.join(exam_instance.pred_dir, "PCA", exam_instance.work_fileName)
        pred_CT_dir = os.path.join(exam_instance.pred_dir, "CT", exam_instance.work_fileName)
        real_PCA_dir = os.path.join(args.root_path, args.real_PCA_dir)
        real_CT_dir = os.path.join(args.root_path, args.real_CT_dir)
        save_GT_split_img_flag = False

        # 生成log文件
        logger = tool_functions.get_logger(
            filename=os.path.join(exam_instance.log_dir, exam_instance.work_fileName + "_estimate.log"),
            verbosity=1,
            name=exam_instance.work_fileName)
        # 生成csv文件
        csv_writer = tool_functions.get_csv(
            filename=os.path.join(exam_instance.csv_dir, exam_instance.work_fileName + "_estimate.csv"),
            header=["name"] + estimate_method["PCA"] + estimate_method["CT"])

        #  保存切片的路径
        split_img_path = exam_instance.split_img_dir
        if not save_GT_split_img_flag:
            GT_split_img_path = os.path.join(exam_instance.split_img_dir,"..","GT")
            os.makedirs(GT_split_img_path,exist_ok=True)

        pred_file_suffix = [file_name.split("_")[1] for file_name in os.listdir(pred_PCA_dir)]
        # 开始评估：
        for cur_file_suffix in pred_file_suffix:
            cur_number = cur_file_suffix.split(".")[0]
            print("projection"+cur_number, "-->")
            estimate_value_dict = {"name": "projection"+cur_number}
            for pred_mode in estimate_method:
                pred_dir = tool_functions.choose_by_prediction_mode(pred_mode,[pred_PCA_dir, pred_CT_dir])
                GT_dir = tool_functions.choose_by_prediction_mode(pred_mode,[real_PCA_dir, real_CT_dir])
                pred_value = tool_functions.load_odd_file(os.path.join(pred_dir, pred_mode+"_"+cur_file_suffix), shape=exam_instance.data_shape[pred_mode])
                GT_value = tool_functions.load_odd_file(os.path.join(GT_dir, pred_mode+"_"+cur_file_suffix), shape=exam_instance.data_shape[pred_mode])
                estimate_value_dict.update(
                    data_processing.estimate_calc(GT_value, pred_value, estimate_method[pred_mode]))

            logger.info(estimate_value_dict)
            csv_writer.writerow(estimate_value_dict)

            if pred_mode == "CT":
                # 保存切片
                # split_num = args.split_num
                #
                # cor_img = tool_functions.normalization_2d_img(pred_value[split_num[0], :, :]) * 255
                # sag_img = tool_functions.normalization_2d_img(pred_value[:, split_num[1], :]) * 255
                # tra_img = tool_functions.normalization_2d_img(pred_value[:, :, split_num[2]]) * 255
                # cv2.imwrite(os.path.join(split_img_path, str(cur_number) + "_cor" + "(" + str(split_num[0]) + ").png"),
                #            cor_img)
                # cv2.imwrite(os.path.join(split_img_path, str(cur_number) + "_sag" + "(" + str(split_num[1]) + ").png"),
                #            sag_img)
                # cv2.imwrite(os.path.join(split_img_path, str(cur_number) + "_tra" + "(" + str(split_num[2]) + ").png"),
                #            tra_img)
                tool_functions.save_all_split(pred_value,pred_value.shape,split_img_path,str(cur_number),["cor","sag","tra"])



                # 保存GT切片
                if not save_GT_split_img_flag:
                    # GT_cor_img = tool_functions.normalization_2d_img(GT_value[split_num[0], :, :]) * 255
                    # GT_sag_img = tool_functions.normalization_2d_img(GT_value[:, split_num[1], :]) * 255
                    # GT_tra_img = tool_functions.normalization_2d_img(GT_value[:, :, split_num[2]]) * 255
                    # cv2.imwrite(os.path.join(GT_split_img_path, str(cur_number) + "_cor" + "(" + str(split_num[0]) + ").png"),
                    #            GT_cor_img)
                    # cv2.imwrite(os.path.join(GT_split_img_path, str(cur_number) + "_sag" + "(" + str(split_num[1]) + ").png"),
                    #            GT_sag_img)
                    # cv2.imwrite(os.path.join(GT_split_img_path, str(cur_number) + "_tra" + "(" + str(split_num[2]) + ").png"),
                    #            GT_tra_img)
                    tool_functions.save_all_split(GT_value, pred_value.shape, GT_split_img_path, str(cur_number),
                                                  ["cor", "sag", "tra"])
        save_GT_split_img_flag = True



        logging.shutdown()


if __name__ == '__main__':
    args, cfg = tool_functions.load_cfg(yaml_path="./tools/cfg/pca_spaceAndTime.yaml")
    exam_extimate(args, cfg)



