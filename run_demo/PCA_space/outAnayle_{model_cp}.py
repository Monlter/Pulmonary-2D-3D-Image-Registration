import numpy as np
import os

import pandas as pd

from tools import config,tool_functions,data_loader,pca_data_processing


def pd_save_csv(log_list):
    log_train_data = {}
    log_val_data = {}
    for log_file in log_list:
        trainOrVal = log_file.split(")_")[1].split(".")[0]
        print(trainOrVal)
        print(trainOrVal)
        # train log
        if trainOrVal == "train":
            fp = open(os.path.join(log_path, log_file))
            for i, line in enumerate(fp.readlines()):
                # 读取第一行的model名称
                if i == 0:
                    model_name = line.split("\t")[2].split(":")[1]
                    train_loss_list = []
                    test_loss_list = []

                elif i >= 4 and i <= 103:
                    epoch_num = line.split(":[")[1].split("/")[0]
                    train_loss = line.split("train_loss=")[1].split("\t")[0]
                    test_loss = line.split("test_loss=")[1].split('\n')[0]
                    train_loss_list.append(train_loss)
                    test_loss_list.append(test_loss)

            log_train_data[model_name + "_train_loss"] = train_loss_list
            log_train_data[model_name + "_test_loss"] = test_loss_list

        elif trainOrVal == "val":
            fp = open(os.path.join(log_path, log_file))
            val_list = []
            val_origin = []
            for i, line in enumerate(fp.readlines()):
                if i == 0:
                    model_name = line.split("\t")[2].split(":")[1]
                elif i<=9:
                    prediction = line.split("\t")[1].split(":")[1]
                    origin = line.split("\t")[2].split(":")[1]
                    val_list.append(prediction)
                    val_origin.append(origin)
            if not "origin" in log_val_data:
               log_val_data["origin"] = val_origin
            log_val_data[model_name] = val_list

    train_frame = pd.DataFrame(log_train_data)
    train_frame.to_csv(os.path.join(out_dir, "out_train.csv"), sep=",", index=True, header=True)

    val_frame = pd.DataFrame(log_val_data)
    val_frame_T = pd.DataFrame(val_frame.T, index=val_frame.columns,columns=val_frame.index)
    val_frame_T.to_csv(os.path.join(out_dir, "out_val.csv"), sep=",", index=True, header=True)

    return train_frame, val_frame


args = config.get_args()

def init_args(modelMethodName, inputModeName,lossFunctionMethodName):
    args.modelMethod = modelMethodName
    args.inputMode = inputModeName
    args.lossFunctionMethod = lossFunctionMethodName

    current_file = tool_functions.get_filename(__file__)
    cpName = tool_functions.get_cpName(current_file)
    args.cpName = cpName

    testName = tool_functions.get_testName(__file__)
    args.testName = testName

    root_path = tool_functions.get_poject_path("Pulmonary-2D-3D-Image-Registration")
    args.root_path = root_path

    workFileName = tool_functions.methodsName_combine(args)
    args.workFileName = workFileName

if __name__ == '__main__':

    root_path = tool_functions.get_poject_path("Pulmonary-2D-3D-Image-Registration")

    log_path = os.path.join(tool_functions.get_out_result_dir(args), "log")
    out_dir = os.path.join( tool_functions.get_out_result_dir(args), "anayle/loss_out_csv")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    log_list = os.listdir(log_path)
    print(log_list)
    train_frame,val_frame = pd_save_csv(log_list)

