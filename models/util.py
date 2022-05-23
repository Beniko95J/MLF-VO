import torch.nn as nn

from third_party.CEN.semantic_segmentation.models.modules import ModuleParallel


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                    stride=stride, padding=1, bias=bias))


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                    stride=stride, padding=0, bias=bias))


class conv7x7Parallel(nn.Module):
    def __init__(self, in_planes, out_planes, num_parallel, stride=1, bias=False):
        super(conv7x7Parallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'conv7x7_' + str(i), nn.Conv2d(in_planes, out_planes, kernel_size=7,
                                                         stride=stride, padding=3, bias=bias))

    def forward(self, x_parallel):
        return [getattr(self, 'conv7x7_' + str(i))(x) for i, x in enumerate(x_parallel)]


class conv3x3Parallel(nn.Module):
    def __init__(self, in_planes, out_planes, num_parallel, stride=1, bias=False):
        super(conv3x3Parallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'conv3x3_' + str(i), nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                                         stride=stride, padding=1, bias=bias))

    def forward(self, x_parallel):
        return [getattr(self, 'conv3x3_' + str(i))(x) for i, x in enumerate(x_parallel)]


class conv1x1Parallel(nn.Module):
    def __init__(self, in_planes, out_planes, num_parallel, stride=1, bias=False):
        super(conv1x1Parallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'conv1x1_' + str(i), nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                                         stride=stride, padding=0, bias=bias))

    def forward(self, x_parallel):
        return [getattr(self, 'conv1x1_' + str(i))(x) for i, x in enumerate(x_parallel)]


class linearParallel(nn.Module):
    def __init__(self, in_features, out_features, num_parallel) -> None:
        super().__init__()
        for i in range(num_parallel):
            setattr(self, 'linear_' + str(i), nn.Linear(in_features, out_features))

    def forward(self, x_parallel):
        return [getattr(self, 'linear_' + str(i))(x) for i, x in enumerate(x_parallel)]
