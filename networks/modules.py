import torch.nn as nn
import torch


class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, bn, bn_threshold):
        bn0, bn1 = bn[0].weight.abs(), bn[1].weight.abs()
        x0, x1 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x0[:] = x[0][:]
        x0[:, bn0 < bn_threshold] = x[1][:, bn0 < bn_threshold]
        x1[:] = x[1][:]
        x1[:, bn1 < bn_threshold] = x[0][:, bn1 < bn_threshold]
        return [x0, x1]


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]
