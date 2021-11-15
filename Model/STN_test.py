import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


"""
input : 输入tensor， shape为 [N, C, D_in, H_in, W_in]
grid: 一个field flow， shape为[N, D_in, H_out, W_out, 3],数值范围被归一化到[-1,1]。
out: 输出tensor shape[N, C, D_in, H_in, W_in]
"""

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        self.mode = mode

    def standardization(data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma


    def forward(self, src, flow):
        new_locs = flow      # grid的作用是什么
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = self.standardization(new_locs[:, i, ...])

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode)



