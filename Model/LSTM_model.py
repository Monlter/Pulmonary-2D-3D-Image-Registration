import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CNN(nn.Module):
    def __init__(self, in_channels):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=8,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, 3, 1, 1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 1, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )
        self.conv_reduce = torch.nn.Conv2d(32,8,1)
        # self.mlp1 = nn.Linear(15 * 15 * 32, 1024)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv_reduce(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_classes, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.conv_reduce = nn.Conv2d(512*4,512,1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_reduce(x)
        return x


def resnet34_net(in_classes):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=Bottleneck,
                   layers=[3, 4, 6, 3],
                   in_classes=in_classes
                   )
    return model



class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer,
                            batch_first=True)
        # self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        out, _ = self.lstm(x)   # x‘s shape (batch_size, 序列长度, 序列中每个数据的长度)  # out‘s shape (batch_size, 序列长度, hidden_dim)
        out = out[:, -1, :]     # 中间的序列长度取-1，表示取序列中的最后一个数据，这个数据长度为hidden_dim，
        # out = self.classifier(out)
        return out             # 得到的out的shape为(batch_size, hidden_dim)


class CRN(nn.Module):
    def __init__(self, in_channels):
        super(CRN, self).__init__()
        self.in_channels = in_channels
        self.cnn_block = CNN(self.in_channels)
        self.rnn_block = Rnn(15 * 15 * 8, 1024, 2)
        self.fc = nn.Linear(1024, 3)

    def forward(self,x1,x2):
        x1 = self.cnn_block(x1)
        x1 = x1.view(x1.size(0),-1)
        x2 = self.cnn_block(x2)
        x2 = x2.view(x2.size(0),-1)
        x = torch.stack([x1,x2],dim=1)
        x = self.rnn_block(x)
        x = self.fc(x)
        return x


class CRN_resnet(nn.Module):
    def __init__(self, in_channels):
        super(CRN_resnet, self).__init__()
        self.in_channels = in_channels
        self.cnn_block = resnet34_net(self.in_channels)
        self.rnn_block = Rnn(512*4*4, 1024, 2)
        self.fc = nn.Linear(1024, 3)

    def forward(self,x1,x2):
        x1 = self.cnn_block(x1)
        x1 = x1.view(x1.size(0),-1)
        x2 = self.cnn_block(x2)
        x2 = x2.view(x2.size(0),-1)
        x = torch.stack([x1,x2],dim=1)
        x = self.rnn_block(x)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = CRN_resnet(1).to("cuda:0")
    print(model)
    print(summary(model, [[1, 120, 120],[1, 120, 120]]))