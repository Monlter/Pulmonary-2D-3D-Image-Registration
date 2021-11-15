import numpy as np
import torch
import torch.nn as nn

if __name__ == '__main__':
    arr = np.array(range(1, 26))
    arr = arr.reshape((5, 5))
    arr = np.expand_dims(arr, 2)
    arr = arr.transpose((2, 0, 1))
    arr = torch.Tensor(arr)
    arr = arr.unsqueeze(0)
    print(arr.size())  # torch.Size([1, 1, 5, 5])
    conv1 = nn.Conv2d(1, 1, 3, stride=1, bias=False, dilation=1)  # 普通卷积
    conv2 = nn.Conv2d(1, 1, 3, stride=1, bias=False, dilation=2)  # dilation就是空洞率，间隔为2.多出来的用0填充

    # 给参数初始化为1，方便运算
    nn.init.constant_(conv1.weight, 1)
    nn.init.constant_(conv2.weight, 1)
    out1 = conv1(arr)
    out2 = conv2(arr)
    print('standare conv:\n', out1.data.numpy())
    print('dilated conv:\n', out2.data.numpy())