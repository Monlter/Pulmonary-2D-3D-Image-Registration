from tools import tool_functions, data_processing, config, models_init
import os
import numpy as np
import SimpleITK as sitk
import torch
from sklearn.decomposition import PCA
import gc
import torch.nn.functional as F
from PIL import Image
from thop import profile
from torchinfo import summary
import time
from skimage import transform


def dcm_trans_bin():
    root_path = tool_functions.get_poject_path("Pulmonary-2D-3D-Image-Registration")
    dcm_folder = os.path.join(root_path, "Dataset/Patient/Origin/CT_dcm")
    dcm_list = os.listdir(dcm_folder)
    for i, dcm_phase in enumerate(dcm_list):
        ct_array = data_processing.readDicomSeries(os.path.join(dcm_folder, dcm_phase)).astype(np.float32)
        tool_functions.file_save(ct_array,
                                 "ct_" + str(i) + "(" + str(ct_array.shape[0]) + "_"
                                 + str(ct_array.shape[1]) + "_"
                                 + str(ct_array.shape[2]) + ").bin",
                                 os.path.join(root_path, "Dataset/Patient/Origin/CT"))
    print("Dcm_trans_Binary 已经成功！")


def read_file(path):
    root_path = tool_functions.get_poject_path("Pulmonary-2D-3D-Image-Registration")
    folder = os.path.join(root_path, path)
    name_file = "../cuda_c++_cu113/data/file_name.txt"
    file_list = os.listdir(folder)
    # 输入文件
    with open(name_file, 'w') as f1:
        for file_name in file_list:
            f1.write(file_name)
            f1.write('\n')


def dvf_save_nii():
    root_path = tool_functions.get_poject_path("Pulmonary-2D-3D-Image-Registration")
    dvf_path = os.path.join(root_path, "\Dataset\Test_9dvf\Output\dvf\Resnet_allCBAM(origin_MSE)\predict_dvf_1")
    dvf = tool_functions.load_odd_file(dvf_path).reshape(3, 150, 256, 256).transpose(2, 3, 1, 0)
    sitk_dvf = sitk.GetImageFromArray(dvf)
    sitk.WriteImage(sitk_dvf, "dvf.nii")


def save_ct_split():
    def ct_split(img_path, split_phase, split_mode):
        ct_numpy = tool_functions.load_odd_file(img_path).reshape(150, 256, 256)
        if split_mode == "overlook":
            ct_split_numpy = ct_numpy[split_phase, :, :][53:203, 53:203]
        elif split_mode == "positiveview":
            ct_split_numpy = ct_numpy[:, split_phase, :][:, 53:203]
        else:
            ct_split_numpy = ct_numpy[:, :, split_phase][:, 53:203]
        return ct_split_numpy

    args = config.get_args()
    split_phase = 150
    root_path = tool_functions.get_poject_path("Pulmonary-2D-3D-Image-Registration")
    predict_ct_path = os.path.join(root_path, args.predict_ct)
    model_name_list = os.listdir(predict_ct_path)
    print("predict_CT:切片开始--------------------------")
    for model_name in model_name_list:
        ct_list = os.listdir(os.path.join(predict_ct_path, model_name))
        for ct in ct_list:
            for split_phase in range(60, 100):
                ct_split_numpy = ct_split(os.path.join(predict_ct_path, model_name, ct), split_phase, "overlook")
                tool_functions.save_png(ct_split_numpy, os.path.join(predict_ct_path, "../CT_split_img", model_name),
                                        str(ct) + "_split_phase(" + str(split_phase) + ")_overlook")
            for split_phase in range(120, 150):
                ct_split_numpy = ct_split(os.path.join(predict_ct_path, model_name, ct), split_phase, "positiveview")
                tool_functions.save_png(ct_split_numpy, os.path.join(predict_ct_path, "../CT_split_img", model_name),
                                        str(ct) + "_split_phase(" + str(split_phase) + ")_positiveview")
            for split_phase in range(76, 106):
                ct_split_numpy = ct_split(os.path.join(predict_ct_path, model_name, ct), split_phase, "sideview")
                tool_functions.save_png(ct_split_numpy, os.path.join(predict_ct_path, "../CT_split_img", model_name),
                                        str(ct) + "_split_phase(" + str(split_phase) + ")_sideview")
            print(model_name, str(ct) + "已经保存！")
    print("GT_CT:切片开始--------------------------")
    GT_ct_path = os.path.join(root_path, args.real_ct)
    GT_ct_list = os.listdir(GT_ct_path)
    for GT_ct in GT_ct_list:
        for split_phase in range(60, 100):
            ct_split_numpy = ct_split(os.path.join(GT_ct_path, GT_ct), split_phase, "overlook")
            tool_functions.save_png(ct_split_numpy, os.path.join(predict_ct_path, "../CT_split_img", "GT"),
                                    str(GT_ct) + "_split_phase(" + str(split_phase) + ")_overlook")
        for split_phase in range(120, 150):
            ct_split_numpy = ct_split(os.path.join(GT_ct_path, GT_ct), split_phase, "positiveview")
            tool_functions.save_png(ct_split_numpy, os.path.join(predict_ct_path, "../CT_split_img", "GT"),
                                    str(GT_ct) + "_split_phase(" + str(split_phase) + ")_positiveview")
        for split_phase in range(76, 106):
            ct_split_numpy = ct_split(os.path.join(GT_ct_path, GT_ct), split_phase, "sideview")
            tool_functions.save_png(ct_split_numpy, os.path.join(predict_ct_path, "../CT_split_img", "GT"),
                                    str(GT_ct) + "_split_phase(" + str(split_phase) + ")_sideview")
        print("GT", str(GT_ct) + "已经保存！")


def PCA_train_by_DVFs(path):
    root_path = tool_functions.get_poject_path('Pulmonary-2D-3D-Image-Registration')
    DVF_folder = os.path.join(root_path, path, "Origin/DVF")
    PCA_train_parameter_folder = os.path.join(root_path, path, "Origin/DVF_trans_PCAs")
    os.makedirs(PCA_train_parameter_folder,exist_ok=True)
    PCA_coefficient_folder = os.path.join(root_path, path, "Origin/PCA")
    os.makedirs(PCA_coefficient_folder, exist_ok=True)
    gen_DVF_folder = os.path.join(root_path, path, "Product_9dvf/DVFs")
    os.makedirs(gen_DVF_folder, exist_ok=True)
    gen_CT_folder = os.path.join(root_path, path, "Product_9dvf/CTs")
    os.makedirs(gen_CT_folder, exist_ok=True)
    gen_projection_folder = os.path.join(root_path, path, "Product_9dvf/projections")
    os.makedirs(gen_projection_folder, exist_ok=True)
    file_name_path = os.path.join(root_path, path, "Product_9dvf")
    os.makedirs(file_name_path, exist_ok=True)


    args = config.get_args()
    # 加载DVF
    DVFs_arr = tool_functions.load_all_file(DVF_folder)
    # 训练PCA
    pca_class = PCA(n_components=3)
    pca_class.fit(DVFs_arr)
    ct_trans_pcas = pca_class.fit_transform(DVFs_arr)
    # 保存PCA相关参数
    tool_functions.file_save(pca_class.components_,
                             'PCA_components_{}'.format(str(pca_class.components_.shape)) + '.bin',
                             PCA_train_parameter_folder)
    tool_functions.file_save(pca_class.mean_, 'PCA_mean_{}'.format(str(pca_class.mean_.shape)) + '.bin',
                             PCA_train_parameter_folder)
    for i in range(ct_trans_pcas.shape[0]):
        tool_functions.file_save(ct_trans_pcas[i], "PCA_" + str(i + 1) + ".bin", PCA_coefficient_folder)

    # 生成DVF数据
    linspace_num = args.extend_num
    diff = np.vstack([ct_trans_pcas[0], ct_trans_pcas, ct_trans_pcas[-1]])  # 首尾添加
    diff = diff[1:] - diff[0:-1]
    print("diff:", diff.shape)

    with open(os.path.join(file_name_path, "DVF_path.txt"), "w") as DVF_path, \
            open(os.path.join(file_name_path, "CT_path.txt"), "w") as CT_path, \
            open(os.path.join(file_name_path, "projection_path.txt"), "w") as projection_path:

        for i in range(len(ct_trans_pcas)):
            # 每2个PCA系数之间重新生成num个PCA系数
            pca_linspace_left = np.linspace(ct_trans_pcas[i] - diff[i] * 0.15, ct_trans_pcas[i], num=linspace_num // 2)
            pca_linspace_right = np.linspace(ct_trans_pcas[i], ct_trans_pcas[i] + diff[i + 1] * 0.15,
                                             num=linspace_num // 2)
            pca_linspace = np.vstack([pca_linspace_left, pca_linspace_right])
            print("PCA_linspace_left:", pca_linspace_left.shape)
            print("PCA_linspace:", pca_linspace.shape)
            print("第{}组DVF生成：extend_nums({})***************************".format(i + 1, len(pca_linspace)))
            print(pca_linspace)
            for j in range(linspace_num):
                pca_trans_cts = pca_class.inverse_transform(pca_linspace[j])  # 将每个PCA进行DVF的还原
                save_DVF_name = "DVF" + str(i + 1) + '_' + str(j + 1) + ".bin"
                save_CT_name = save_DVF_name.replace("DVF", "CT")
                save_projection_name = save_DVF_name.replace("DVF", "projection")
                tool_functions.file_save(pca_trans_cts, save_DVF_name, gen_DVF_folder)  # 保存对应的DVF
                DVF_path.write(os.path.join(gen_DVF_folder, save_DVF_name + "\n"))
                CT_path.write(os.path.join(gen_CT_folder, save_CT_name + "\n"))
                projection_path.write(os.path.join(gen_projection_folder, save_projection_name + "\n"))
                del pca_trans_cts  # 减少内存的使用
                print(str(i + 1) + '_' + str(j + 1), '已经保存')
            gc.collect()  # 垃圾回收机制
        DVF_path.write("\n")
        CT_path.write("\n")
        projection_path.write("\n")


def adjust_size_by_spacing(input_path, save_path, shape, spacing):
    root_path = tool_functions.get_poject_path('Pulmonary-2D-3D-Image-Registration')
    input_list = os.listdir(os.path.join(root_path, input_path))
    for input_name in input_list:
        input_array = np.fromfile(os.path.join(root_path, input_path, input_name), dtype='float32').reshape(shape)
        input_img = sitk.GetImageFromArray(input_array)
        resize_img = tool_functions.ImageResampleBySpacing(input_img, old_spacing=spacing)
        resize_array = sitk.GetArrayFromImage(resize_img)
        resize_array.tofile(os.path.join(root_path, save_path, input_name))
        print(input_name, "已经转成功!", shape, "--->", resize_array.shape)


def bin_trans_png(projection_path, shape):
    root_path = tool_functions.get_poject_path('Pulmonary-2D-3D-Image-Registration')
    projection_list = os.listdir(os.path.join(root_path, projection_path))
    for projection_name in projection_list:
        projection_array = np.fromfile(os.path.join(root_path, projection_path, projection_name), dtype="float32")
        projection_array = projection_array.reshape(shape)
        projection_array = tool_functions.normalization_2d_img(projection_array, 'max_min') * 255
        img = Image.fromarray(projection_array)
        img = img.convert("L")  # 转成灰度图
        save_path = os.path.join(root_path, projection_path, "../trans_projections")
        os.makedirs(save_path,exist_ok=True)
        img.save(
            os.path.join(os.path.join(save_path, projection_name.split(".bin")[0] + ".png")))
        print("转{}文件成功".format(projection_name))


def rename(path, old_name, new_name):
    root_path = tool_functions.get_poject_path('Pulmonary-2D-3D-Image-Registration')
    path = os.path.join(root_path, path)
    file_name_list = os.listdir(path)
    for file_name in file_name_list:
        old_file = os.path.join(path, file_name)
        new_file = os.path.join(path, file_name.replace(old_name, new_name))
        os.rename(old_file, new_file)
        print(new_file, "保存成功！")


def calc_each_model_params():
    root_path = tool_functions.get_poject_path('Pulmonary-2D-3D-Image-Registration')
    csv_writer = tool_functions.get_csv(
        filename=os.path.join(root_path, "Out_result", "models_param.csv"),
        header=["model", "train_params", "total_params", "time"]
    )

    dummy_input = torch.randn(1, 1, 120, 120).to('cuda:0')
    dummy_input_seq = torch.randn(1, 1, 1, 120, 120).to("cuda:0")
    model_methods, _ = models_init.optional_init()
    for model_name in model_methods:
        model = model_methods[model_name](1).to('cuda:0')
        train_params, total_params = tool_functions.calc_param(model)
        if "ConvLSTM" not in model_name:
            start_time = time.time()
            pred_value = model(dummy_input)
            end_time = time.time()
            # print(summary(model,input_size=(1,1,120,120)))
        else:
            start_time = time.time()
            pred_value = model(dummy_input_seq)
            end_time = time.time()
            # print(summary(model,input_size=(1,1,1,120,120)))

        print("model:", model_name, "train_params:", train_params, "total_params:", total_params, "time:",
              end_time - start_time)
        csv_writer.writerow({
            "model": model_name,
            "train_params": train_params,
            "total_params": total_params,
            "time": end_time - start_time

        })


# def trans_size(input_path,old_shape,new_shape):
#     root_path = tool_functions.get_poject_path('Pulmonary-2D-3D-Image-Registration')
#     input_list = os.listdir(os.path.join(root_path, input_path))
#     for input_name in input_list:
#         input_array = np.fromfile(os.path.join(root_path, input_path, input_name), dtype='float32').reshape(old_shape)
#         input_img = sitk.GetImageFromArray(input_array)
#         resize_img = tool_functions.ImageResampleBySpacing(input_img,[1.0,1.0,1.0],[2.0,2.0,2.0])
#         resize_array = sitk.GetArrayFromImage(resize_img)
#         save_path = os.path.join(root_path, input_path, "..", "trans_")
#         os.makedirs(save_path,exist_ok=True)
#         resize_array.tofile(os.path.join(save_path, input_name))
#         print(input_name, "已经转成功!", old_shape, "--->", resize_array.shape)

def trans_size(input_path, save_path, input_shape, output_shape):
    root_path = tool_functions.get_poject_path('Pulmonary-2D-3D-Image-Registration')
    input_list = os.listdir(os.path.join(root_path, input_path))
    for input_name in input_list:
        input_array = np.fromfile(os.path.join(root_path, input_path, input_name), dtype='float32').reshape(input_shape)
        resized_image = transform.resize(input_array, output_shape).astype('float32')
        resized_image.tofile(os.path.join(root_path, save_path, input_name))
        print(input_name, "已经转成功!", input_shape, "--->", output_shape)


def concat_dvfSplit(path, shape, type):
    dvf_list = os.listdir(path)
    save_path = os.path.join(path, "trans_dvf")
    os.makedirs(save_path, exist_ok=True)
    for dvf_name in dvf_list:
        if ("Fx" not in dvf_name):
            continue
        dvf_Fx = np.fromfile(os.path.join(path, dvf_name), dtype=type).reshape(shape)
        dvf_Fy = np.fromfile(os.path.join(path, dvf_name.replace("Fx", "Fy")), dtype=type).reshape(shape)
        dvf_Fz = np.fromfile(os.path.join(path, dvf_name.replace("Fx", "Fz")), dtype=type).reshape(shape)
        dvf = np.stack((dvf_Fx, dvf_Fy, dvf_Fz), axis=0).astype("float32")
        save_name = dvf_name.split(".")[0].split("_Fx")[0]
        dvf.tofile(os.path.join(save_path, save_name + '.bin'))
        print(save_name, "已经保存")


if __name__ == '__main__':
    # PCA_train_by_DVFs(path="Dataset/Patient/9")
    # read_file(path="Dataset/Patient/5/Origin/CT")
    # trans_size(input_path='Dataset/Patient/5/Origin/CT', save_path='Dataset/Patient/5/Origin/resize_CT',
    #            shape=(102, 512, 512))
    # bin_trans_png("Dataset/Patient/8/Product_9dvf/projections", shape=(384,512))
    # rename("Dataset/Patient/9/Origin/DVF",old_name="dvf_ct0",new_name="dvf_")
    # calc_each_model_params()
    trans_size(input_path="Dataset/Patient/9/Origin/CT1", save_path="Dataset/Patient/9/Origin/trans", input_shape=[315, 553, 553], output_shape=[158,277,277])
    # concat_dvfSplit(
    #     path=r"C:\Users\ck\Downloads\009 dvf  277x277x158\009 dvf  277x277x158",
    #     shape=[158,277,277],
    #     type='float64')
    pass
