import cv2
import numpy as np
import matplotlib.pyplot as plt



def laplacian_img(img):
    blur = cv2.GaussianBlur(img, (7, 7), 1, 1)
    result = cv2.Laplacian(blur, cv2.CV_32F, ksize=1)
    return result


if __name__ == '__main__':
    img_name = r"../../Dataset/Test1/projections/projection_1_1"
    gray_img = np.fromfile(img_name, dtype='float32').reshape((100, 240, 300))[25, ...]
    result = laplacian_img(gray_img)
    plt.imshow(result)
    plt.show()
