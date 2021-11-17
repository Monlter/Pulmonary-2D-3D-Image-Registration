import numpy as np
import os
import pandas as pd
from tools.tool_functions import *
import pandas
from tools.config import get_args
from tools.estimate_methods import *
import csv

def estimate_calc(real_CTs_numpy,predict_CTs_numpy,estimate_methods_list):
    estimate_method_function = {
        "NMI":NMI,
        "SSD":SSD,
        "SAD":SAD,
        "MSE":MSE,
        "SSIM":SSIM,
        "NCC":NCC
    }
    estimate_data = []
    for estimate_method in estimate_methods_list:
        estimate_data.append(estimate_method_function[estimate_method](real_CTs_numpy, predict_CTs_numpy))
    return estimate_data

def composite_all_excel(all_excel_path):
    all_excel_list = os.listdir(all_excel_path)
    dataframe_list = []
    for excel_name in all_excel_list:
        excel_path = os.path.join(all_excel_path,excel_name)
        dataframe = pd.read_csv(excel_path,sep=',',index_col=0,header=0)
        dataframe_list.append(dataframe)
    composite_dataframe_mean = pd.DataFrame(np.mean(dataframe_list,axis=0),index=dataframe.index,columns=dataframe.columns)
    composite_dataframe_std = pd.DataFrame(np.std(dataframe_list,axis=0),index=dataframe.index,columns=dataframe.columns)
    composite_dataframe_mean.to_csv(os.path.join(all_excel_path,"composite_out_CTs_mean.csv"),header=True,index=True,sep=',')
    composite_dataframe_std.to_csv(os.path.join(all_excel_path,"composite_out_CTs_std.csv"),header=True,index=True,sep=',')


if __name__ == '__main__':
    args = get_args()
    estimate_data = {}
    root_path = get_poject_path("Pulmonary-2D-3D-Image-Registration")

    # # out_path
    # num_cp = get_fileNum(get_filename(__file__))
    # testName = get_testName(__file__)  # TEST1'
    # experiment_dir = get_experimentDir(num_cp, root_path, testName,args.gen_pca_method)  # E:\code\pycharm\PCA\Experiment\Test1/PCA_origin/model_cp
    # out_dir = os.path.join(experiment_dir, "anayle")
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    # # 评估方法
    # estimate_methods_list = ["NMI","SSD","SAD","MSE","NCC","SSIM"]
    # # 获取CT_list
    # real_CT_path = os.path.join(root_path, args.real_ct)
    # real_CT_list = os.listdir(real_CT_path)
    # predcict_CT_list = os.listdir(os.path.join(root_path,"Dataset/Test_9dvf/Output/CT/CNN(origin_MSE)"))
    # for i in range(9):
    #     # real_CT
    #     real_CT_numpy = load_odd_file(os.path.join(real_CT_path,real_CT_list[i+1])).reshape(150,256,256).transpose(1,2,0)
    #     print(real_CT_list[i+1])
    #     # predict_ct
    #     all_predict_CT_path = os.path.join(root_path,args.predict_ct)
    #     predict_CT_class_list = os.listdir(all_predict_CT_path)
    #     for predict_CT_class in predict_CT_class_list:
    #         predict_CT_path = os.path.join(all_predict_CT_path,predict_CT_class,predcict_CT_list[i])
    #         print(predcict_CT_list[i])
    #         predict_CT_numpy = load_odd_file(predict_CT_path).reshape(150,256,256).transpose(1,2,0)
    #         estimate_data_odd_list = estimate_calc(real_CT_numpy,predict_CT_numpy,estimate_methods_list)
    #         estimate_data[predict_CT_class] = estimate_data_odd_list
    #     estimate_frame = pd.DataFrame(estimate_data,index=estimate_methods_list)
    #     estimate_frame.to_csv(os.path.join(out_dir,"estimate_out_CT"+str(i+1)+".csv"), sep=",",index=True,header=True)

    all_excel_path = os.path.join(root_path,"Experiment/Test1/PCA_origin/model_cp/anayle")
    composite_all_excel(all_excel_path)






