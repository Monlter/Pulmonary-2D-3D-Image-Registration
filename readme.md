## 数据转换形式
### CT：
+ 格式：二进制，float32(.bin文件)
+ 形状：(9830400,)
+ 需要转换的格式：reshape(150,256,256)--256*256为每张图像的大小

### DVF：
+ 格式：二进制，float32
+ 形状：(29491200,)
+ 需要转换的格式：reshape(3,150,256,256)--256*256为每张图像的大小

### DRR(projection):
+ 格式：二进制，float32
+ 形状：(1800000,)
+ 需要转换的格式：reshape(100,240,300)--240*300为每张图像的大小,其中第25张为标准正投


## 数据集
+ 位置：
  + Dataset/Digital_phantom  :数字模体数据
  + Dataset/Patient: 真实病人数据
+ 结构：
  + Origin
    + CT
    + DVF
    + PCA： 存放每个DVF对应的PCA系数
    + projection：每个CT正向投影的DRR图像
    + DVF_trans_PCA：存放所有DVF的特征向量
  + Product_9dvf
    + projections: 进行数据扩充的DRR图像


## 程序设置：
+ tools/Config.py : 设置一些基础设置（学习率，batch_size,以及一些文件路径）
+ tools/cfg/xxx.yaml : 特殊设置 （当前输入图像数量，模型，损失函数）
  + pca_space.ymal : 使用空间网络（接受单个图像输入）
  + pca_spaceAndTime.yaml : 使用时空网络（接受序列输入）

## 程序运行：
+ train.py : 程序训练
+ test.py : 使用训练好的权重进行测试
+ estimate.py : 对测试后的图像进行各个指标的验证
  
