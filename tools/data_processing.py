import matplotlib.pyplot as plt
import numpy as np
import os
from tools import tool_functions
import cv2
import SimpleITK as sitk
"""对数据进行标准差归一化"""


def data_std_all(data):
    PCA_all = np.fromfile(r'/Dataset/train/PCA_all\pca_960', dtype='float32').reshape((960, 3))
    PCA_mean = np.mean(PCA_all, axis=0)
    PCA_std = np.std(PCA_all, axis=0)
    data_std = (data - PCA_mean) / PCA_std
    return data_std


"""对标准差归一化的数据进行还原"""


def stdData_reverse(data_std, PCA_std, PCA_mean):
    data_reverse = data_std * PCA_std + PCA_mean
    return data_reverse


"""最大-最小归一化数据进行还原"""


def standardizationData_reverse(standardizationdata, min, max):
    data_reverse = standardizationdata * (max - min) + min
    return data_reverse


"""对数据进行最大-最小标准化"""


def data_standardization_max_min(data, min, max):
    data_standardization_x = (data - min) / (max - min)
    return data_standardization_x


"""对数据进行标准差标准化"""


def data_standardization_mean_std(data, mean, std):
    data_standardization_x = (data - mean) / std
    return data_standardization_x


"""  对数据进行归一化 """


def img_standardization_f(img, method):
    if (method == "max_min"):
        max_x = max(map(max, img))
        min_x = min(map(min, img))
        img_standardization_x = data_standardization_max_min(img, min_x, max_x)
    else:
        mean_x = np.mean(img)
        std_x = np.std(img)
        img_standardization_x = data_standardization_mean_std(img, mean_x, std_x)
    return img_standardization_x


""" 加载标准差和最大最小值函数"""


def return_standardization_para(target_file):
    target_all = np.fromfile(target_file, dtype='float32').reshape((-1, 3))
    std = np.std(target_all, axis=0)
    mean = np.mean(target_all, axis=0)
    min_x = np.min(target_all, axis=0)
    max_x = np.max(target_all, axis=0)
    return std, mean, min_x, max_x


def trans_PCAFile(target_folder):
    target_all = np.fromfile(os.path.join(target_folder, 'PCA_all'), dtype='float32')
    target_tran = np.reshape(target_all, (-1, 3))
    target_trans_folder = os.path.join(target_folder, '..\\PCA_trans\\')
    if not os.path.exists(target_trans_folder):
        os.makedirs(target_trans_folder)
    for i in range(target_tran.shape[0]):
        print(target_tran[i])
        target_tran[i].tofile(os.path.join(target_trans_folder, 'PCA_{}'.format(i)))


def data_standardization_0_255(img):
    ymax = 255
    ymin = 0
    xmax = max(map(max, img))  # 进行两次求max值
    xmin = min(map(min, img))
    img_standardization_0_255 = np.round((ymax - ymin) * (img - xmin) / (xmax - xmin) + ymin)
    return img_standardization_0_255


def data_standardization_0_255_imgs(imgs):
    # ask img.shape is (C,H,W)
    imgs_standardization = []
    for i in range(imgs.shape[0]):
        imgs_standardization_0_255 = data_standardization_0_255(imgs[i])
        imgs_standardization.append(imgs_standardization_0_255)
    return np.array(imgs_standardization)


def make_subimg(img):
    root_ = tool_functions.get_poject_path("Pulmonary-2D-3D-Image-Registration")
    img_fix = np.fromfile((root_ + "/Dataset/trainData(9dvf)/projection_0_0"), dtype="float32").reshape(
        (100, 240, 300))[25,
              :, :]
    sub_img = img - img_fix
    return sub_img


def get_preImgName_sequence(imgName, preImg_num):
    phase_num = 9
    cur_num = int(imgName.split("_")[1])
    random_num = imgName.split("_")[2]
    init_name = imgName.split("_")[0]
    preImgName_sequence = []
    for i in range(preImg_num):
        pre_num = cur_num - 1
        pre_num = pre_num + phase_num if pre_num == 0 else pre_num
        preImgName = init_name + "_" + str(pre_num) + "_" + random_num
        preImgName_sequence.append(preImgName)
        cur_num = pre_num
    return preImgName_sequence[::-1]


def load_projection_sequence(projection_dir, projection_name_squence, projection_view=25):
    projection_sequence = []
    for projection_name in projection_name_squence:
        projection = \
            np.fromfile(os.path.join(projection_dir, projection_name), dtype='float32').reshape((100, 240, 300))[
                projection_view]
        projection_sequence.append(projection)
    return projection_sequence


def input_mode_concat_variable(img: np.ndarray, standardization_method, input_mode_names: list,
                               resize=None) -> np.ndarray:
    """
    :param img: shape(H,W)
    :param standardization_method: "max-min" or "mean-std"
    :param data_process_methods: ["origin","sub","edge"]
    :param resize: 将要调整的图像大小（resize_H,resize_W）
    :return:input_img.shape(C,H,W)
    """
    # img_shape = 1 * W * H
    img_shape = list(img[np.newaxis, ...].shape)
    img_shape[0] = 0
    num = 0
    # img_cat.shape = 0 * W * H
    img_cat = np.ones(shape=img_shape, dtype="float32")
    if input_mode_names.find("origin") != -1:
        num += 1
        img_standardization = img_standardization_f(img, standardization_method)[np.newaxis, ...]
        img_cat = np.vstack([img_cat, img_standardization])

    if input_mode_names.find("edge") != -1:
        num += 1
        edge_img = laplacian_img(img)
        edge_img_standardization = img_standardization_f(edge_img, standardization_method)[
            np.newaxis, ...]
        img_cat = np.vstack([img_cat, edge_img_standardization])
    if input_mode_names.find("sub") != -1:
        num += 1
        sub_img = make_subimg(img)
        sub_img_standardization = img_standardization_f(sub_img, standardization_method)[
            np.newaxis, ...]
        img_cat = np.vstack([img_cat, sub_img_standardization])

    if num != len(input_mode_names.split("_")):
        raise ValueError("The data processing name in the data_process_methods is incorrectly written!")
    # 调整尺寸
    if resize:
        input_img = tool_functions.resize_img(img_cat.squeeze(), resize)
    # 如果图像为 W*H，将其转换为 1*W*H
    if len(input_img.shape) == 2:
        input_img = input_img[np.newaxis, ...]

    return input_img


def laplacian_img(img):
    blur = cv2.GaussianBlur(img, (7, 7), 1, 1)
    result = cv2.Laplacian(blur, cv2.CV_32F, ksize=1)
    return result


def intensity_correction(img, reference_img):
    hist_img = cv2.calcHist([img], [0], None, [256], [0.0, 256.0])
    hist_reference_img = cv2.calcHist([reference_img], [0], None, [256], [0.0, 256.0])
    similarity = cv2.compareHist(hist_img, hist_reference_img, 0)
    return similarity, hist_img, hist_reference_img


def readDicomSeries(folder_name):
    series_reader = sitk.ImageSeriesReader()
    fileNames = series_reader.GetGDCMSeriesFileNames(folder_name)
    series_reader.SetFileNames(fileNames)
    image = series_reader.Execute()
    image_array = sitk.GetArrayFromImage(image)
    return image_array

if __name__ == '__main__':
    root_path = tool_functions.get_poject_path("Pulmonary-2D-3D-Image-Registration")
    reference_img = tool_functions.load_odd_file(
        os.path.join(root_path, "Dataset/origin/CT_dcm/4d_lung_phantom_w_lesion_atn_2.bin")).reshape(150, 256, 256)[:,100,:]
    plt.imshow(reference_img)
    plt.show()
    # plt.subplot(2, 1, 1)
    # plt.plot(hist_img)
    # plt.subplot(2, 1, 2)
    # plt.plot(hist_reference_img)
