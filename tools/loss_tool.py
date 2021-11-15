import torch.nn as nn
import torch
import torch.nn.functional as func
import numpy as np
import math
from torch.autograd import Variable, Function


def loss_function(y, prediction):
    wcoeff = torch.FloatTensor([[2 / math.sqrt(6), 1 / math.sqrt(6), 1 / math.sqrt(6)]]).to("cuda:0")
    U = torch.FloatTensor([[0.00018971, 0.00048171, 0.00063309]]).to("cuda:0")
    # loss_mse = torch.mean(torch.sum((1 / 3) * torch.norm(U * wcoeff * (y - prediction), dim=1), dim=0))
    loss_mse = torch.mean((1 / 3) * torch.norm(U * wcoeff * (y - prediction), dim=1), dim=0)
    return loss_mse


class PCA_loss(nn.Module):
    def __init__(self, wcoeff):
        super(PCA_loss, self).__init__()
        self.wcoeff = wcoeff

    def forward(self, target, prediction):
        loss_mse = torch.mean(self.wcoeff * torch.norm((target - prediction), dim=0), dim=0)
        return loss_mse


class PCA_smoothL1Loss(nn.Module):
    def __init__(self, wcoeff, threshold=200):
        super(PCA_smoothL1Loss, self).__init__()
        self.wcoeff = wcoeff
        self.threshold = threshold

    def forward(self, target, prediction):
        diff = torch.mean(torch.abs(target - prediction).float())
        if (diff < self.threshold):
            print("MSE")
            loss_result = torch.mean((self.wcoeff * torch.norm((target - prediction), dim=0)), dim=0)
        else:
            print("MAE")
            loss_result = torch.mean((torch.norm((target - prediction), p=1, dim=0) - 0.5 / torch.pow(self.wcoeff, 2)),
                                     dim=0)
        return loss_result


class Log_cosh(nn.Module):
    def __init__(self, wcoeff):
        super(Log_cosh, self).__init__()
        self.wcoeff = wcoeff

    def forward(self, target, prediction):
        loss = np.log(np.cosh(self.wcoeff * (prediction - target)))
        return np.sum(loss)


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
