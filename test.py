import csv
import numpy as np
import os

import torch
import logging
import time
import cv2
# 加载自己库

from tools import instanceExam, data_processing, tool_functions, models_init


def exam_test(args, cfg):
    # 初始化
    exam_process_dict = cfg['EXAM_PROCESS']
    model_methods, lossfunction_methods = models_init.optional_init()
    total_exam_num = len(exam_process_dict)
    cur_exam_num = 0

    # 评估函数方式
    estimate_method = {
        'PCA': ['MAE', 'MAE_percentage', 'R2'],
        'CT': ['MAD', 'MAD_precentage', 'NCC', 'SSIM'],
    }

    tool_functions.setup_seed(12)
    # 超参数设定
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(args.root_path,"Out_result",str(cfg["MODEL_TYPE"]+"_"+cfg["PREDICTION_MODE"]) ,"DVF_path.txt"),"w") as DVF_path,\
         open(os.path.join(args.root_path,"Out_result",str(cfg["MODEL_TYPE"]+"_"+cfg["PREDICTION_MODE"]) ,"CT_path.txt"),"w") as CT_path:

        with torch.no_grad():
            for exam_cfg in exam_process_dict:
                # 实例化
                exam_instance = instanceExam.InstanceExam(args, cfg, exam_cfg)

                # 加载模型
                load_model_file = os.path.join(exam_instance.cur_ckpt_dir, str(args.EPOCH) + ".pth")
                if not os.path.exists(load_model_file):
                    print(load_model_file, "文件不存在！")
                    continue
                model = model_methods[exam_instance.model_method](exam_instance.inChannel_num)
                model.load_state_dict(torch.load(load_model_file))
                model.to(device)


                trainable_params ,total_params = tool_functions.calc_param(model)
                # 生成log文件
                logger = tool_functions.get_logger(
                    filename=os.path.join(exam_instance.log_dir, exam_instance.work_fileName + "_test.log"),
                    verbosity=1,
                    name=exam_instance.work_fileName)

                # 生成csv文件
                csv_writer = tool_functions.get_csv(
                    filename=os.path.join(exam_instance.csv_dir, exam_instance.work_fileName + "_test.csv"),
                    header=["name","trainable_params","total_params","time"] + estimate_method[exam_instance.prediction_mode])
                #  保存预测值的路径
                pred_dir = os.path.join(exam_instance.pred_dir, exam_instance.prediction_mode, exam_instance.work_fileName)
                os.makedirs(pred_dir, exist_ok=True)
                #  保存切片的路径
                split_img_path = exam_instance.split_img_dir

                # 加载数据
                val_DRR_dir = os.path.join(args.root_path, args.val_DRR_dir)
                real_PCA_dir = os.path.join(args.root_path, args.real_PCA_dir)
                real_CT_dir = os.path.join(args.root_path, args.real_CT_dir)
                real_DVF_dir = os.path.join(args.root_path, args.real_DVF_dir)
                DVF_trans_PCA_dir = os.path.join(args.root_path, args.dvf_trans_pca)

                # Ground_truth 路径
                GT_dir = tool_functions.choose_by_prediction_mode(exam_instance.prediction_mode,
                                                                  [real_PCA_dir, real_CT_dir, real_DVF_dir])

                val_DRR_file = os.listdir(val_DRR_dir)
                for val_DRR_name in val_DRR_file:
                    # 加载输入数据
                    input_imgs = torch.tensor(data_processing.get_input_array(
                        model_type=exam_instance.model_type,
                        image_path=val_DRR_dir,
                        image_name=val_DRR_name,
                        preImg_num=args.preImg_num,
                        pre_image_path=val_DRR_dir
                    ), dtype=torch.float)
                    input_imgs = input_imgs.unsqueeze(0)

                    # 加载GT
                    cur_number = val_DRR_name.split('.')[0].split('_')[1]
                    GT_numpy = data_processing.load_odd_GT(exam_instance.prediction_mode, GT_dir, cur_number,
                                                           exam_instance.data_shape)
                    # 预测
                    start_time = time.time()
                    prediction = model(input_imgs.to(device))
                    end_time = time.time()
                    cost_time = end_time - start_time
                    prediction_numpy = prediction[0].cpu().numpy().astype(np.float32)  # 取出第一个预测值
                    # 保存预测值
                    prediction_numpy.tofile(
                        os.path.join(pred_dir, exam_instance.prediction_mode + "_" + cur_number + ".bin"))

                    if exam_instance.prediction_mode == "PCA":
                        # 保存DVF
                        pred_DVF_dir = os.path.join(exam_instance.pred_dir, "DVF", exam_instance.work_fileName)
                        os.makedirs(pred_DVF_dir, exist_ok=True)
                        pred_CT_dir = os.path.join(exam_instance.pred_dir, "CT", exam_instance.work_fileName)
                        os.makedirs(pred_CT_dir, exist_ok=True)

                        PCA_components = None
                        PCA_mean = None
                        for file_name in os.listdir(DVF_trans_PCA_dir):
                            if file_name.find("components") >= 0:
                                shape = list(map(lambda x: int(x), file_name.split("(")[1].split(")")[0].split(",")))
                                PCA_components = np.fromfile(os.path.join(DVF_trans_PCA_dir, file_name),
                                                             dtype=np.float32).reshape(shape)
                            if file_name.find("mean") >= 0:
                                PCA_mean = np.fromfile(os.path.join(DVF_trans_PCA_dir, file_name), dtype=np.float32)
                        assert (PCA_components.all() != None and PCA_mean.all() != None), "PCA_components or PCA_mean not exist!"
                        pred_DVF = np.matmul(prediction_numpy, PCA_components) + PCA_mean
                        pred_DVF.astype(np.float32).tofile(os.path.join(pred_DVF_dir, "DVF_" + cur_number + ".bin"))
                        DVF_path.write(os.path.join(pred_DVF_dir, "DVF_" + cur_number + ".bin\n"))
                        CT_path.write(os.path.join(pred_CT_dir, "CT_" + cur_number + ".bin\n"))

                    elif exam_instance.prediction_mode == "CT":
                        # 保存切片
                        split_num = args.split_num
                        cor_img = prediction_numpy[split_num[0], :, :]
                        sag_img = prediction_numpy[:, split_num[1], :]
                        tra_img = prediction_numpy[:, :, split_num[2]]
                        cv2.imread(os.path.join(split_img_path, cur_number + "_cor" + "(" + split_num[0] + ").png"),
                                   cor_img)
                        cv2.imread(os.path.join(split_img_path, cur_number + "_sag" + "(" + split_num[0] + ").png"),
                                   sag_img)
                        cv2.imread(os.path.join(split_img_path, cur_number + "_tra" + "(" + split_num[0] + ").png"),
                                   tra_img)

                    # 打印相关参数
                    print(val_DRR_name, '----------------------------->')
                    estimate_value_dict = {"name": val_DRR_name}
                    cur_estimate_method_list = estimate_method[exam_instance.prediction_mode]
                    estimate_value_dict.update(
                        data_processing.estimate_calc(GT_numpy, prediction_numpy, cur_estimate_method_list))
                    estimate_value_dict.update({
                        "trainable_params": int(trainable_params),
                        "total_params": int(total_params),
                        "time":cost_time
                    })
                    logger.info(estimate_value_dict)
                    csv_writer.writerow(estimate_value_dict)

                logging.shutdown()
        DVF_path.write("\n")
        CT_path.write("\n")


if __name__ == '__main__':
    args, cfg = tool_functions.load_cfg(yaml_path="./tools/cfg/pca_spaceAndTime.yaml")
    exam_test(args, cfg)
