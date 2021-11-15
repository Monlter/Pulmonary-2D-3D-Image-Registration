import numpy as np
import os
import logging
import torch
import random
import re
import time
comparsion_mode = ["model_cp", "data_cp", "heatmap_cp", "lossFunction_cp","noise_cp"]


def get_poject_path(PROJECT_NAME):
    project_path = os.path.abspath(os.path.dirname(__file__))
    root_path = project_path[:project_path.find("{}".format(PROJECT_NAME)) + len("{}".format(PROJECT_NAME))]
    return root_path


"""¼ÓÔØfile"""


def load_all_file(file_folder):
    file_name_list = os.listdir(file_folder)
    file_list = []
    for file_name in file_name_list:
        file = np.fromfile(os.path.join(file_folder, file_name), dtype='float32')
        file_list.append(file)
    return np.array(file_list)


def load_odd_file(filename):
    file = np.fromfile(filename, dtype='float32')
    return file


def file_save(data, name, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    data.tofile(os.path.join(save_folder, name),)


"""¶Ôpca½øÐÐorigin_fileµÄ»¹Ô­"""


def pca_trans_origin(pca, pca_component, pca_mean):
    # inverse_transform()
    return np.dot(pca, pca_component) + pca_mean


"""¼ÓÔØpca»¹Ô­ÐèÒªµÄÏà¹Ø²ÎÊý"""


def load_pca_para(PCA_para_folder):
    pca_components = np.fromfile(os.path.join(PCA_para_folder, "pca_components_(3,9830400)"), dtype='float32').reshape(
        (3, -1))
    pca_mean = np.fromfile(os.path.join(PCA_para_folder, "pca_mean_(9830400,)"), dtype='float32')
    return pca_components, pca_mean


"""´´½¨ÎÄ¼þ¼Ð"""


def make_dir(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def get_dataMethodName(data_methods):
    returnstr = "origin"
    if data_methods["edge"]:
        returnstr += "_edge"
    if data_methods["sub"]:
        returnstr += "_sub"
    if data_methods["multi_angle"]:
        returnstr += "_multiAngle"
    return returnstr


def  get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s"
    )
    # %(asctime)s：当前时间
    # %(message)s ：用户输出的消息
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def recode_progressNum(num):
    print("-"*100)
    print("progress_"+str(num), "start:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def methodsName_combine(num_cp, modelMethod, dataMethod, lossFunctionMethod):
    returnstr = ''
    if num_cp == 1:
        returnstr = modelMethod + "(" + dataMethod + "_" + lossFunctionMethod + ")"
    elif num_cp == 2:
        returnstr = dataMethod + "(" + modelMethod + "_" + lossFunctionMethod + ")"
    elif num_cp == 3:
        returnstr = modelMethod + "(" + dataMethod + "_" + lossFunctionMethod + ")"
    elif num_cp == 4:
        returnstr = lossFunctionMethod + "(" + modelMethod + "_" + dataMethod + ")"
    elif num_cp == 5:
        returnstr = "("+modelMethod+"_"+dataMethod+"_"+lossFunctionMethod+")"
    print("modelMethod:", modelMethod, "\tdataMethod:", dataMethod, "\tlossfunction:", lossFunctionMethod)
    return returnstr


def get_experimentDir(num_cp, root_path, experiment, pca_method):
    returnstr = os.path.join(root_path, "Experiment",
                             str(experiment + "/" + pca_method + "/" + comparsion_mode[num_cp - 1]))
    print("Experiment:", experiment, "\tpca_method:", pca_method)
    return returnstr


def get_savedir(num_cp, root_path, experiment, pca_method, workFileName):
    returnstr = os.path.join(root_path, "checkpoint",
                             (experiment + "/" + pca_method + "/" + comparsion_mode[
                                 num_cp - 1] + "/" + workFileName + "/"))
    _ = make_dir(returnstr)
    return returnstr


def get_filename(file):
    filename = os.path.basename(file)
    return filename

def get_testName(file):
    project_path = os.path.abspath(os.path.dirname(file))
    testName = project_path[project_path.find("Test"):project_path.find("Test") + 5]
    return testName

def get_fileNum(filename):
    num = re.findall(re.compile(r'\d+'), filename)
    return int(num[0])

def get_fileType(filename):
    returnstr = filename[:filename.find("_")]
    return returnstr

def get_dataMethod_num(dataMethodName):
    num = 0
    if dataMethodName.find("origin") != -1:
        num += 1
    if dataMethodName.find("multiAngle") != -1:
        num += 1
    if dataMethodName.find("edge") != -1:
        num += 1
    if dataMethodName.find("sub") != -1:
        num += 1
    return num

def get_logfilename(num_cp, modelMethod, dataMethod, lossFunctionMethod):
    returnstr = ''
    if num_cp == 1:
        returnstr = modelMethod
    elif num_cp == 2:
        returnstr = dataMethod
    elif num_cp == 3:
        returnstr = modelMethod
    elif num_cp == 4:
        returnstr = lossFunctionMethod
    return returnstr





if __name__ == '__main__':
    pass
