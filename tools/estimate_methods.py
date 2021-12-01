import numpy as np
from sklearn.metrics.cluster import mutual_info_score, normalized_mutual_info_score
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score


def return_shape_multi(img_shape):
    multi = 1
    for i in img_shape:
        multi *= i
    return multi


"""TRE"""

"""MAE"""


def MAE(real, predict):
    real_copy = np.copy(real)
    predict_copy = np.copy(predict)
    if (len(real_copy.shape) == 3):
        MAE_error_sum = 0
        for i in range(real_copy.shape[2]):
            MAE_error_sum += mean_absolute_error(real_copy[:, :, i], predict_copy[:, :, i])
        MAE_error = MAE_error_sum / real_copy.shape[2]
    else:
        MAE_error = mean_absolute_error(real_copy, predict_copy)
    return MAE_error


def MAE_percentage(real, predict):
    real_copy = np.copy(real)
    predict_copy = np.copy(predict)
    if (len(real_copy.shape) == 3):
        MAE_precentage_sum = 0
        for i in range(real_copy.shape[2]):
            MAE_precentage_sum += mean_absolute_percentage_error(real_copy[:, :, i], predict_copy[:, :, i])
        MAE_precentage = MAE_precentage_sum / real_copy.shape[2]
    else:
        MAE_precentage = mean_absolute_percentage_error(real_copy, predict_copy)
    return MAE_precentage


"""R2"""


def R2(real, predict):
    real_copy = np.copy(real)
    predict_copy = np.copy(predict)
    r2 = r2_score(real_copy, predict_copy)
    return r2


"""SSIM"""


def SSIM(real, predict):
    real_copy = np.copy(real)
    predict_copy = np.copy(predict)
    if (len(real_copy.shape) == 3):
        return structural_similarity(real_copy, predict_copy, multichannel=True)
    else:
        return structural_similarity(real_copy, predict_copy)


"""NCC"""


def NCC(real, predict):
    real_copy = np.copy(real)
    predict_copy = np.copy(predict)
    return np.mean(np.multiply((real_copy - np.mean(real_copy)), (predict_copy - np.mean(predict_copy)))) / (
            np.std(real_copy) * np.std(predict_copy))


"""NMI"""


def NMI(real_copy, predict_copy):
    predict_copy = predict_copy.flatten()
    real_copy = real_copy.flatten()
    return normalized_mutual_info_score(real_copy, predict_copy)


def MI(real_copy, predict_copy):
    predict_copy = predict_copy.flatten()
    real_copy = real_copy.flatten()
    return mutual_info_score(real_copy, predict_copy)


"""MSE"""


def SSD(real_copy, predict_copy):
    return np.sum(np.square(predict_copy - real_copy))


def SAD(real_copy, predict_copy):
    return np.sum(np.abs(predict_copy - real_copy))


def MSE(real_copy, predict_copy):
    # return mean_squared_error(real_copy, predict_copy)
    return np.mean(np.square(predict_copy - real_copy))


"""SC"""


def SC(real_copy, predict_copy):
    sc_score = sum(sum(real_copy, real_copy)) / sum(sum(predict_copy, predict_copy))
    return sc_score


"""NAE"""


def NAE(real_copy, predict_copy):
    nae_score = sum(sum(abs(real_copy - predict_copy))) / sum(sum(real_copy))
    return nae_score


if __name__ == '__main__':
    pass
