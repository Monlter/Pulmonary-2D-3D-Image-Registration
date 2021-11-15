import cv2
import numpy as np
import matplotlib.pyplot as plt
from tools.data_loader import data_normal_0_255

def Canny_demo(image):
    plt.imshow(image,cmap="gray")
    plt.show()
    gray_blur = cv2.GaussianBlur(image, (3, 3), 0)    # 进行高斯滤波
    gradx = cv2.Sobel(gray_blur, cv2.CV_16SC1, 1, 0)  # 使用sobel算子进行图像梯度的计算
    grady = cv2.Sobel(gray_blur, cv2.CV_16SC1, 0, 1)
    edge_output = cv2.Canny(gradx, grady, 50, 150)
    # edge_output = cv.Canny(gray, 50, 150) 可以替代前三行
    plt.imshow(edge_output,cmap="gray")
    plt.show()
    dst = cv2.bitwise_and(image, image, mask=edge_output)   # 对二进制数据进行“与”操作
    plt.imshow(dst,cmap="gray")
    plt.show()

if __name__ == '__main__':
    img_name = r"../Dataset/Test1/projections/projection_1_1"
    gray_img = np.fromfile(img_name, dtype='float32').reshape((100, 240, 300))[25, ...]
    gray_img = data_normal_0_255(gray_img)
    print(gray_img.max())
    Canny_demo(gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
