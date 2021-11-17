import numpy as np
from sklearn.metrics.cluster import mutual_info_score, normalized_mutual_info_score
from skimage.metrics import structural_similarity,mean_squared_error

"""TRE"""


"""DSC"""


"""ASD"""


"""NMI"""
def NMI(real,predict):
    predict = predict.flatten()
    real = real.flatten()
    return normalized_mutual_info_score(real,predict)

def MI(real,predict):
    predict = predict.flatten()
    real = real.flatten()
    return mutual_info_score(real, predict)


"""MSE"""
def SSD(real,predict):
    return np.sum(np.square(predict-real))

def SAD(real,predict):
    return np.sum(np.abs(predict-real))

def MSE(real,predict):
    # return mean_squared_error(real, predict)
    return np.mean(np.square(predict-real))


"""SSIM"""
def SSIM(real,predict):
    return structural_similarity(real, predict, multichannel=True)


"""NCC"""
def NCC(real,predict):
    return np.mean(np.multiply((real-np.mean(real)),(predict-np.mean(predict))))/(np.std(real)*np.std(predict))


if __name__ == '__main__':
    a = np.ones(shape=(225,225,150))
    a[1,1,0] = 2
    b = np.ones(shape=(225,225,150))
    b[2,2,24] = 8

    print(MSE(a,b))
    print(SSIM(a,b))
    print(NCC(a,b))

