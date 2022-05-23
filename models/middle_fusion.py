import torch
import torch.nn as nn

from .util import conv7x7Parallel, conv3x3Parallel, conv1x1Parallel, linearParallel
from third_party.CEN.semantic_segmentation.models.modules import BatchNorm2dParallel, ModuleParallel


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, num_parallel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3Parallel(inplanes, planes, num_parallel, stride)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.conv2 = conv3x3Parallel(planes, planes, num_parallel)
        self.bn2 = BatchNorm2dParallel(planes, num_parallel)

        self.num_parallel = num_parallel
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out


class PoseNet(nn.Module):
    def __init__(self, block, layers, num_parallel):
        super(PoseNet, self).__init__()

        self.inplanes = 64
        self.num_parallel = num_parallel

        self.conv1 = conv7x7Parallel(6, 64, 2, stride=2)

        self.bn1 = BatchNorm2dParallel(64, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.relu_ = nn.ReLU(inplace=True)
        self.maxpool = ModuleParallel(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fc1 = linearParallel(512 * 6 * 20, 256, 2)
        self.fc2_1 = nn.Linear(512, 512)
        self.fc2_2 = nn.Linear(512, 512)
        self.soft_fusion = nn.Linear(512, 512)
        self.fc3_t = nn.Linear(512, 3)
        self.fc3_r = nn.Linear(512, 3)

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1Parallel(self.inplanes, planes * block.expansion, 2, stride=stride),
                BatchNorm2dParallel(planes * block.expansion, self.num_parallel)
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.num_parallel, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, self.num_parallel))

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

        x = torch.cat(x, dim=1)
        mask = self.sigmoid(self.soft_fusion(x))
        x = mask * x

        x = self.fc2_1(x)
        x = self.relu_(x)
        x = self.fc2_2(x)
        x = self.relu_(x)

        x_t = self.fc3_t(x)
        x_r = self.fc3_r(x)
        x = torch.cat([x_r, x_t], dim=1)

        axisangle = x[:, 0:3]
        translation = x[:, 3:6]
        axisangle = 0.01 * axisangle.view(-1, 3)
        translation = 0.01 * translation.view(-1, 3)

        return axisangle, translation


def create_posenet_middle(num_parallel=2):
    model = PoseNet(BasicBlock, [2, 2, 2, 2], num_parallel)

    return model
