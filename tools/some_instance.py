from tools import tool_functions, data_processing, config
import os
import numpy as np
import SimpleITK as sitk
import torch
from sklearn.decomposition import PCA
import gc
import torch.nn.functional as F
from PIL import Image


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
    PCA_train_parameter_folder = os.path.join(root_path, path, "Product_9dvf/DVF_trans_PCAs")
    PCA_coefficient_folder = os.path.join(root_path, path, "Product_9dvf/PCAs")
    gen_DVF_folder = os.path.join(root_path, path, "Product_9dvf/DVFs")

    args = config.get_args()
    # 加载DVF
    DVFs_arr = tool_functions.load_all_file(DVF_folder)
    # 训练PCA
    pca_class = PCA(n_components=3)
    pca_class.fit(DVFs_arr)
    ct_trans_pcas = pca_class.fit_transform(DVFs_arr)
    # 保存PCA相关参数
    tool_functions.file_save(pca_class.components_,
                             'pca_components_({})'.format(str(pca_class.components_.shape)),
                             PCA_train_parameter_folder)
    tool_functions.file_save(pca_class.mean_, 'pca_mean_({})'.format(str(pca_class.mean_.shape)),
                             PCA_train_parameter_folder)
    for i in range(ct_trans_pcas.shape[0]):
        tool_functions.file_save(ct_trans_pcas[i], "PCA_" + str(i + 1), PCA_coefficient_folder)

    # 生成DVF数据
    linspace_num = args.extend_num
    diff = np.vstack([ct_trans_pcas[0], ct_trans_pcas, ct_trans_pcas[-1]])  # 首尾添加
    diff = diff[1:] - diff[0:-1]
    print("diff:", diff.shape)

    for i in range(len(ct_trans_pcas)):
        # 每2个PCA系数之间重新生成num个PCA系数
        pca_linspace_left = np.linspace(ct_trans_pcas[i] - diff[i] * 0.15, ct_trans_pcas[i], num=linspace_num // 2)
        pca_linspace_right = np.linspace(ct_trans_pcas[i], ct_trans_pcas[i] + diff[i + 1] * 0.15, num=linspace_num // 2)
        pca_linspace = np.vstack([pca_linspace_left, pca_linspace_right])
        print("PCA_linspace_left:", pca_linspace_left.shape)
        print("PCA_linspace:", pca_linspace.shape)
        print("第{}组DVF生成：extend_nums({})***************************".format(i + 1, len(pca_linspace)))
        print(pca_linspace)
        for j in range(linspace_num):
            pca_trans_cts = pca_class.inverse_transform(pca_linspace[j])  # 将每个PCA进行DVF的还原
            tool_functions.file_save(pca_trans_cts, ("DVF" + str(i + 1) + '_' + str(j + 1)), gen_DVF_folder)  # 保存对应的DVF
            del pca_trans_cts  # 减少内存的使用
            print(str(i + 1) + '_' + str(j + 1), '已经保存')
        gc.collect()  # 垃圾回收机制


def trans_size(input_path, save_path, shape):
    root_path = tool_functions.get_poject_path('Pulmonary-2D-3D-Image-Registration')
    input_list = os.listdir(os.path.join(root_path, input_path))
    for input_name in input_list:
        input_array = np.fromfile(os.path.join(root_path, input_path, input_name), dtype='float32').reshape(shape)
        input_img = sitk.GetImageFromArray(input_array)
        resize_img = tool_functions.ImageResample(input_img)
        resize_array = sitk.GetArrayFromImage(resize_img)
        resize_array.tofile(os.path.join(root_path, save_path, input_name))
        print(input_name, "已经转成功!", shape, "--->", resize_array.shape)


def demons_registration():
    import sys
    def command_iteration(filter):
        print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")

    if len(sys.argv) < 4:
        print(
            f"Usage: {sys.argv[0]} <fixedImageFilter> <movingImageFile> <outputTransformFile>")
        sys.exit(1)

    fixed = sitk.ReadImage(sys.argv[1], sitk.sitkFloat32)

    moving = sitk.ReadImage(sys.argv[2], sitk.sitkFloat32)

    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    moving = matcher.Execute(moving, fixed)

    # The basic Demons Registration Filter
    # Note there is a whole family of Demons Registration algorithms included in
    # SimpleITK
    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations(50)
    # Standard deviation for Gaussian smoothing of displacement field
    demons.SetStandardDeviations(1.0)

    demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))

    displacementField = demons.Execute(fixed, moving)

    print("-------")
    print(f"Number Of Iterations: {demons.GetElapsedIterations()}")
    print(f" RMS: {demons.GetRMSChange()}")

    outTx = sitk.DisplacementFieldTransform(displacementField)

    sitk.WriteTransform(outTx, sys.argv[3])


def bin_trans_png(projection_path, shape):
    root_path = tool_functions.get_poject_path('Pulmonary-2D-3D-Image-Registration')
    projection_list = os.listdir(os.path.join(root_path, projection_path))
    for projection_name in projection_list:
        projection_array = np.fromfile(os.path.join(root_path, projection_path, projection_name), dtype="float32")
        projection_array = projection_array.reshape(shape)[25]
        projection_array = tool_functions.normalization_2d_img(projection_array, 'max_min') * 255
        img = Image.fromarray(projection_array)
        img = img.convert("L")  # 转成灰度图
        img.save(
            os.path.join(os.path.join(root_path, projection_path, "../trans_projections/", projection_name + ".png")))
        print("转{}文件成功".format(projection_name))


def rename(path):
    root_path = tool_functions.get_poject_path('Pulmonary-2D-3D-Image-Registration')
    path = os.path.join(root_path, path)
    file_name_list = os.listdir(path)
    for file_name in file_name_list:
        old_file = os.path.join(path, file_name)
        new_file = os.path.join(path, file_name.replace("PCA", "pca") + ".bin")
        os.rename(old_file, new_file)
        print(new_file, "保存成功！")


if __name__ == '__main__':
    # PCA_train_by_DVFs(path="Dataset/Patient/5")
    # read_file(path="Dataset/Patient/5/Origin/CT")
    # trans_size(input_path='Dataset/Patient/5/Origin/CT', save_path='Dataset/Patient/5/Origin/resize_CT',
    #            shape=(102, 512, 512))
    # bin_trans_png("Dataset/Digital_phantom/Product_9dvf/VAL/projection", shape=(100, 240, 300))
    rename("Dataset/Patient/5/Origin/PCA")
    pass
