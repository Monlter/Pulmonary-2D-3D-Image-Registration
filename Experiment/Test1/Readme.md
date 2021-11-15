# 数据结构
prediction.shape:(Batch,3)
target.shape:(Batch,3)

# train_demo1
进行的模型的对比

# train_demo2
输入数据不同类型的对比

# train_demo3
使用热力图对不同模型的显示

# train_demo4
使用不同的损失函数进行对比

# 组织结构
**checkpoint:**
```
├─checkpoint
│  └─Test1
│      └─PCA_origin
│          ├─data_cp
│          │  ├─origin
│          │  └─origin_sub
│          ├─lossFunction_cp
│          │   └─MSE
│          └─heatmap_cp
│              ├─Unet
│              └─Resnet
```
**Experiment:**
```
├─Experiment
│  └─Test1
│      └─PCA_origin
│          ├─data_cp
│          │  ├─log
│          │  └─run
│          │      ├─Resnet
│          │      └─Unet
│          ├─heatmap_cp
│          │  ├─log
│          │  └─run
│          │      └─Resnet
│          └─model_cp
│              ├─log
│              └─run
│                  ├─Resnet
│                  └─Unet
```