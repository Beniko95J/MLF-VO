import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import conv3x3, conv1x1
from third_party.CEN.modules import Exchange, BatchNorm2dParallel, ModuleParallel


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, num_parallel, bn_threshold, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2dParallel(planes, num_parallel)

        self.num_parallel = num_parallel
        self.stride = stride
        self.downsample = downsample

        self.exchange = Exchange()
        self.bn_threshold = bn_threshold
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if len(x) > 1:
            out = self.exchange(out, self.bn2_list, self.bn_threshold)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out


class PoseNet(nn.Module):
    def __init__(self, block, layers, num_parallel, bn_threshold=2e-2):
        super(PoseNet, self).__init__()

        self.inplanes = 64
        self.num_parallel = num_parallel

        self.conv1 = ModuleParallel(nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False))

        self.bn1 = BatchNorm2dParallel(64, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.maxpool = ModuleParallel(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(block, 64, layers[0], bn_threshold)
        self.layer2 = self._make_layer(block, 128, layers[1], bn_threshold, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], bn_threshold, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], bn_threshold, stride=2)

        self.fc1 = ModuleParallel(nn.Linear(512 * 6 * 20, 512))
        self.fc2 = ModuleParallel(nn.Linear(512, 512))
        self.fc3_t = ModuleParallel(nn.Linear(512, 3))
        self.fc3_r = ModuleParallel(nn.Linear(512, 3))

        self.alpha = nn.Parameter(torch.ones(num_parallel, requires_grad=True))
        self.register_parameter('alpha', self.alpha)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, bn_threshold, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                BatchNorm2dParallel(planes * block.expansion, self.num_parallel)
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold))

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

        x = [_x.flatten(start_dim=1) for _x in x]

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        x_t = self.fc3_t(x)
        x_r = self.fc3_r(x)
        x = [torch.cat([_x_r, _x_t], dim=1) for _x_r, _x_t in zip(x_r, x_t)]

        alpha_soft = F.softmax(self.alpha, dim=0)
        axisangle = alpha_soft[0] * x[0][:, 0:3] + alpha_soft[1] * x[1][:, 0:3]
        translation = alpha_soft[0] * x[0][:, 3:6] + alpha_soft[1] * x[1][:, 3:6]
        axisangle = 0.001 * axisangle.view(-1, 3)
        translation = 0.001 * translation.view(-1, 3)

        return axisangle, translation


def create_posenet_multi_layer(num_parallel=2, bn_threshold=2e-2):
    model = PoseNet(BasicBlock, [2, 2, 2, 2], num_parallel, bn_threshold)

    return model
