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
