import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.util import random_noise
from tools.config import get_args
from tools.tool_functions import *
from tools.data_loader import data_normal_0_255_imgs,data_normal_0_255
def add_noise(img,noise_type,ask_255=False):
    img_noise = random_noise(img,mode=noise_type,seed=12,clip=False)
    if ask_255:
        img_noise = img_noise * 255
        return img_noise
    return np.array(img_noise,dtype=np.float32)


if __name__ == '__main__':
    args = get_args()
    root_path = get_poject_path("PCA")
    img_floder = os.path.join(root_path,args.img_folder)
    img_list = os.listdir(img_floder)
    new_floder = make_dir(os.path.join(os.path.dirname(img_floder), "projections_noise"))
    for img_name in img_list:
        imgs = load_odd_file(os.path.join(img_floder,img_name)).reshape((100, 240, 300))
        img_noise = add_noise(imgs, "poisson", False)
        img_noise = add_noise(img_noise,"gaussian",False)
        file_save(img_noise,img_name+"_poissonAndGaussian",new_floder)
        print(img_name)


    # for img_name in img_list:
    #     imgs = load_odd_file(os.path.join(img_floder,img_name)).reshape((100,240,300))
    #     img_noise1 = add_noise(imgs[25], "poisson", False)
    #     img_noise2 = add_noise(img_noise1, "gaussian", False)
    #     plt.imshow(img_noise2)
    #     plt.show()
    #     sns.distplot(img_noise2)
    #     plt.show()
    #     img_noise3 = add_noise(imgs[25], "gaussian", False)
    #     plt.imshow(img_noise3)
    #     plt.show()
    #     sns.distplot(img_noise3)
    #     plt.show()
    #     break
    #     img_noise2 = add_noise(imgs[25],"gaussian",False)
    #     plt.imshow(img_noise2)
    #     plt.show()
    #     sns.distplot(img_noise2)
    #     plt.show()
    #     img_noise3 = add_noise(imgs[25], "speckle", False)
    #     plt.imshow(img_noise3)
    #     plt.show()
    #     sns.distplot(img_noise3)
    #     plt.show()






