import numpy as np
import torch
import os
from sklearn.decomposition import PCA
import gc
import sys
from tools.config import get_args
from tools import tool_functions, data_processing

"""加载file"""


def load_file(file_folder):
    file_name_list = os.listdir(file_folder)
    file_list = []
    for file_name in file_name_list:
        file_path = os.path.join(file_folder, file_name)
        if os.path.isdir(file_path):
            # 为dcm文件
            file = data_processing.readDicomSeries(file_path)
        else:
            file = np.fromfile(os.path.join(file_folder, file_name), dtype='float32')
        file_list.append(file)
    return np.array(file_list)




"""对pca进行origin_file的还原"""


def pca_trans_origin(pca, pca_component, pca_mean):
    # inverse_transform()
    return np.dot(pca, pca_component) + pca_mean


"""加载pca还原需要的相关参数"""





if __name__ == '__main__':
    args = get_args()
    root_path = tool_functions.get_poject_path('Pulmonary-2D-3D-Image-Registration')
    # 加载CT
    CT_folder = os.path.join(root_path, "Dataset/Patient/Origin/CT_dcm")
    CTs_arr = load_file(CT_folder)
    pca_class = PCA(n_components=3)
    pca_class.fit(CTs_arr)
    ct_trans_pcas = pca_class.fit_transform(CTs_arr)
    print(pca_class.mean_.shape)
    print(pca_class.components_.shape)
    print(pca_class.explained_variance_ratio_)

    file_save(pca_class.components_, 'pca_components_(3,9830400)',
              os.path.join(root_path, "Dataset/Patient/Product_9dvf/DVF_trans_PCAs"))
    file_save(pca_class.mean_, 'pca_mean_(9830400,)',
              os.path.join(root_path, "Dataset/Patient/Product_9dvf/DVF_trans_PCAs"))
    for i in range(ct_trans_pcas.shape[0]):
        file_save(ct_trans_pcas[i], "PCA_" + str(i),
                  os.path.join(root_path, "Dataset/Patient/Product_9dvf/PCAs"))

    # pca_all = np.ones((0, 3), dtype="float32")
    # print("总：***************************")
    # print(ct_trans_pcas)
    # # 均分PCA
    # linspace_num = args.extend_num
    # for i in range(len(ct_trans_pcas) - 1):
    #     # 每2个PCA系数之间重新生成num个PCA系数
    #     pca_linspace = np.linspace(ct_trans_pcas[i], ct_trans_pcas[i+1], num=linspace_num)
    #     print("第{}个***************************".format(i))
    #     print(pca_linspace)
    #     for j in range(linspace_num):
    #         pca_trans_cts = pca_class.inverse_transform(pca_linspace[j])   # 将每个PCA进行DVF的还原
    #         file_save(pca_trans_cts, ("DVF"+str(i)+'_'+str(j)), os.path.join(root_path,"Dataset/trainData(9dvf)/DVFs"))   # 保存对应的DVF
    #         del pca_trans_cts     # 减少内存的使用
    #         print(str(i)+'_'+str(j), '已经保存')
    #     gc.collect()   # 垃圾回收机制
    #
    # # 保存总的PCA
    # file_save(pca_all, ("pca_"+str((i+1) * linspace_num)), os.path.join(root_path,"Dataset/trainData(9dvf)/DVF_trans_PCAs"))   # 保存对应的pca
