import torch
import torch.nn as nn


# ---------------module 中的hook函数-------------------
# define a simple Net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # in_channel,out_channel,kernel_size
        self.conv1 = nn.Conv2d(1, 2, 3)
        self.conv2 = nn.Conv2d(2, 2, 3)
        # 卷积核尺寸
        self.pool1 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        return x


def forward_hook(module, data_input, data_output):
    fmap_block.append(data_output)
    input_block.append(data_input)


def forward_pre_hook(module, data_input):
    print("forward_pre_hook input:{}".format(data_input))


def backward_hook(module, grad_input, grad_output):
    print("backward hook input:{}".format(grad_input))
    print("backward hook output:{}".format(grad_output))


# 初始化网络
net = Net()
# detach()就是为例截断反向传播的梯度流
# 卷积核1：值为1
net.conv1.weight[0].detach().fill_(1)
# 卷积核2：值为2
net.conv1.weight[1].detach().fill_(2)
# 偏置
net.conv1.bias.data.detach().zero_()

# 注册hook函数
fmap_block = list()
input_block = list()
net.conv1.register_forward_hook(forward_hook)
net.conv1.register_forward_pre_hook(forward_pre_hook)
net.conv1.register_backward_hook(backward_hook)

# batch size * channel * H * W
fake_img = torch.ones((1, 1, 4, 4))
output = net(fake_img)

loss_fnc = nn.L1Loss()
target = torch.randn_like(output)
loss = loss_fnc(target, output)
loss.backward()

print("output shape: {}\noutput value: {}\n".format(output.shape, output))
print("feature maps shape: {}\noutput value: {}\n".format(fmap_block[0].shape, fmap_block[0]))
print("input shape: {}\ninput value: {}".format(input_block[0][0].shape, input_block[0]))
