import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import math

RANDOM_SEED = 1
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
NUM_EPOCHS = 20

# Architecture
NUM_FEATURES = 28 * 28
NUM_CLASSES = 10

# Other
DEVICE = "cuda:0"
GRAYSCALE = True

"""
CBAM-attention -----start-----------------------------------------------------------------------------------------------
"""


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM_attention(nn.Module):
    def __init__(self, in_planes):
        super(CBAM_attention, self).__init__()
        self.in_planes = in_planes
        self.ca = ChannelAttention(self.in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x_ca = self.ca(x)
        x_sa = self.sa(x_ca)
        x_out = x_sa
        return x_out


"""
CBAM-attention-----end-------------------------------------------------------------------------------------------------
SPAnet-attention----start ----------------------------------------------------------------------------------------------
"""


class SPPLayer(torch.nn.Module):
    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size()  # num:样本数量 c:通道数 h:高 w:宽
        for i in range(self.num_levels):
            level = i + 1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (
                math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            if self.pool_type == 'avg_pool':
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)

            # 展开、拼接
            if (i == 0):
                SPP = tensor.view(num, -1)
            else:
                SPP = torch.cat((SPP, tensor.view(num, -1)), 1)
        return SPP.unsqueeze(2).unsqueeze(2)


class SPAnet(nn.Module):
    def __init__(self, in_planes):
        super(SPAnet, self).__init__()
        self.spplayer = SPPLayer(3)
        self.fc1 = nn.Conv2d(in_planes * 14, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_spp = self.spplayer(x)
        x1 = self.fc1(x_spp)
        x1_relu = self.relu1(x1)
        x2 = self.fc2(x1_relu)
        out = x2
        return self.sigmoid(out)


"""
SPAnet-attention-----end------------------------------------------------------------------------------------------------

Triplet attention----start----------------------------------------------------------------------------------------------
"""


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


"""
Triplet attention-----end-----------------------------------------------------------------------------------------------

Resnet Model----start----------------------------------------------------------------------------------------------
"""
"""
Resnet_block ----  start  ---------------------------------------------------------------------------------------------
"""


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    _attention_methods = {
        "CBAM":CBAM_attention,
        "SPA":SPAnet,
        "Triplet":TripletAttention
    }
    def __init__(self, inplanes, planes, stride=1, downsample=None, is_inlineAttention=None):
        super(Bottleneck, self).__init__()
        self.__dict__.update(self._attention_methods)
        self.is_inlineAttention = is_inlineAttention
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if self.is_inlineAttention:
            self.attention_method = self._attention_methods[self.is_inlineAttention](planes * 4)


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

        if self.is_inlineAttention:
            self.attention_method(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, in_dim, is_CBAM, dilation, is_SPA, is_inlineAttention):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.CBAM = is_CBAM
        self.is_SPA = is_SPA
        self.is_inlineAttention = is_inlineAttention
        self.cbam1 = CBAM_attention(self.inplanes)
        self.spa1 = SPAnet(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.cbam2 = CBAM_attention(self.inplanes)
        self.spa2 = SPAnet(self.inplanes)
        if dilation == 1:
            self.fc1 = nn.Linear(512 * block.expansion * 4 * 4, 512 * block.expansion)
        elif dilation == 3:
            self.fc1 = nn.Linear(512 * block.expansion * 4 * 4, 512 * block.expansion)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)

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
        layers.append(block(self.inplanes, planes, stride, downsample, is_inlineAttention=self.is_inlineAttention))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.CBAM:
            self.cbam1(x) * x
        if self.is_SPA:
            x = self.spa1(x) * x
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.CBAM:
            x = self.cbam2(x) * x
        if self.is_SPA:
            x = self.spa2(x) * x
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        logits = self.fc2(x)
        return logits


def resnet(in_channel, layers=[3, 4, 6, 3], is_CBAM=False, dilation=1, is_SPA=False, is_inlineAttention=None):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=Bottleneck,
                   layers=layers,
                   num_classes=3,
                   in_dim=in_channel,
                   is_CBAM=is_CBAM,
                   dilation=dilation,
                   is_SPA=is_SPA,
                   is_inlineAttention=is_inlineAttention)
    return model


if __name__ == '__main__':
    model = resnet(3, [3, 4, 6, 3]).to('cuda:0')
    print(model)
    print(summary(model, (3, 120, 120)))
