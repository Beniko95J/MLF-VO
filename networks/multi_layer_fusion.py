import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *


model_urls = {
    '18_imagenet': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    '50_imagenet': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}


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


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, num_parallel, bn_threshold, stride=1, downsample=None, parallel=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride) if not parallel else conv3x3Parallel(inplanes, planes, 2, stride)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.conv2 = conv3x3(planes, planes) if not parallel else conv3x3Parallel(planes, planes, 2)
        self.bn2 = BatchNorm2dParallel(planes, num_parallel)
        self.num_parallel = num_parallel
        self.downsample = downsample
        self.stride = stride

        self.exchange = Exchange()
        self.bn_threshold = bn_threshold
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)
        
        self.planes = planes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if len(x) > 1: # FIXME
            # if self.planes == 256 or self.planes == 512:
            out = self.exchange(out, self.bn2_list, self.bn_threshold)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, num_parallel, bn_threshold, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = BatchNorm2dParallel(planes, num_parallel)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = BatchNorm2dParallel(planes * 4, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.num_parallel = num_parallel
        self.downsample = downsample
        self.stride = stride

        self.exchange = Exchange()
        self.bn_threshold = bn_threshold
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x):
        residual = x
        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if len(x) > 1:
            out = self.exchange(out, self.bn2_list, self.bn_threshold)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

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

        self.conv1 = ModuleParallel(nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)) # FIXME
        # self.conv1 = conv7x7Parallel(6, 64, 2, stride=2, bias=False)
        
        self.bn1 = BatchNorm2dParallel(64, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.maxpool = ModuleParallel(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(block, 64, layers[0], bn_threshold)
        self.layer2 = self._make_layer(block, 128, layers[1], bn_threshold, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], bn_threshold, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], bn_threshold, stride=2)
        
        self.fc1 = ModuleParallel(nn.Linear(512 * 6 * 20, 512)) # FIXME
        self.fc2 = ModuleParallel(nn.Linear(512, 512)) # FIXME
        # self.fc1 = linearParallel(512 * 6 * 20, 512, 2)
        # self.fc2 = linearParallel(512, 512, 2)

        self.fc3_t = ModuleParallel(nn.Linear(512, 3)) # FIXME
        self.fc3_r = ModuleParallel(nn.Linear(512, 3)) # FIXME

        self.sigmoid = ModuleParallel(nn.Sigmoid())
        self.tanh = ModuleParallel(nn.Tanh())
        self.softmax = nn.Softmax(dim=0)

        self.alpha = nn.Parameter(torch.ones(num_parallel, requires_grad=True)) # FIXME
        # self.beta = nn.Parameter(torch.ones(num_parallel, requires_grad=True))
        self.register_parameter('alpha', self.alpha)
        # self.register_parameter('beta', self.beta)
        # self.alpha = ModuleParallel(nn.Linear(512, 1))
        # self.beta = ModuleParallel(nn.Linear(512, 1))

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
                # conv1x1Parallel(self.inplanes, planes * block.expansion, 2, stride=stride) if (planes == 64 or planes == 128) else conv1x1(self.inplanes, planes * block.expansion, stride=stride), # FIXME
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                BatchNorm2dParallel(planes * block.expansion, self.num_parallel)
            )

        layers = []

        # layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold, stride, downsample, parallel=(planes == 64 or planes == 128))) # FIXME
        layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold, stride, downsample))
        
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            # layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold, parallel=(planes == 64 or planes == 128))) # FIXME
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

        x = [torch.flatten(x[0], start_dim=1), torch.flatten(x[1], start_dim=1)]

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        x_t = self.fc3_t(x)

        x_r = self.fc3_r(x)

        x = [
                torch.cat([x_r[0], x_t[0]], dim=1),
                torch.cat([x_r[1], x_t[1]], dim=1),
            ]

        alpha_soft = F.softmax(self.alpha, dim=0) # FIXME
        # beta_soft = F.softmax(self.beta, dim=0)

        axisangle = alpha_soft[0] * x[0][:, 0:3] + alpha_soft[1] * x[1][:, 0:3] # FIXME
        translation = alpha_soft[0] * x[0][:, 3:6] + alpha_soft[1] * x[1][:, 3:6]

        # axisangle = x[1][:, 0:3]
        # translation = x[1][:, 3:6]

        axisangle = 0.001 * axisangle.view(-1, 1, 1, 3)
        translation = 0.001 * translation.view(-1, 1, 1, 3)

        return axisangle, translation


def _posenet(block, layers, num_parallel, bn_threshold):
    model = PoseNet(block, layers, num_parallel, bn_threshold)
    return model


def posenet18(num_parallel, bn_threshold):
    return _posenet(BasicBlock, [2, 2, 2, 2], num_parallel, bn_threshold)


def posenet50(num_parallel, bn_threshold):
    return _posenet(Bottleneck, [3, 4, 6, 3], num_parallel, bn_threshold)


def model_init(model, num_layers, num_parallel):
    key = str(num_layers) + '_imagenet'
    url = model_urls[key]
    state_dict = maybe_download(key, url)
    state_dict['conv1.weight'] = torch.cat(
            [state_dict['conv1.weight']] * 2, 1) / 2
    model_dict = expand_model_dict(model.state_dict(), state_dict, num_parallel)
    model.load_state_dict(model_dict, strict=True)

    return model


def maybe_download(model_name, model_url, model_dir=None, map_location=None):
    import os, sys
    import urllib
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = '{}.pth.tar'.format(model_name)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        url = model_url
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urllib.request.urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def expand_model_dict(model_dict, state_dict, num_parallel):
    model_dict_keys = model_dict.keys()
    state_dict_keys = state_dict.keys()
    for model_dict_key in model_dict_keys:
        # if 'bn' not in model_dict_key:
        #     print(model_dict_key)
        model_dict_key_re = model_dict_key.replace('module.', '')
        if model_dict_key_re in state_dict_keys:
            model_dict[model_dict_key] = state_dict[model_dict_key_re]
        for i in range(num_parallel):
            bn = '.bn_%d' % i
            replace = True if bn in model_dict_key_re else False
            model_dict_key_re = model_dict_key_re.replace(bn, '')
            if replace and model_dict_key_re in state_dict_keys:
                model_dict[model_dict_key] = state_dict[model_dict_key_re]
        
        for i in range(num_parallel):
            conv1x1 = '.conv1x1_%d' % i
            replace = True if conv1x1 in model_dict_key_re else False
            model_dict_key_re = model_dict_key_re.replace(conv1x1, '')
            if replace and model_dict_key_re in state_dict_keys:
                model_dict[model_dict_key] = state_dict[model_dict_key_re]
        
        for i in range(num_parallel):
            conv3x3 = '.conv3x3_%d' % i
            replace = True if conv3x3 in model_dict_key_re else False
            model_dict_key_re = model_dict_key_re.replace(conv3x3, '')
            if replace and model_dict_key_re in state_dict_keys:
                model_dict[model_dict_key] = state_dict[model_dict_key_re]
        
        for i in range(num_parallel):
            conv7x7 = '.conv7x7_%d' % i
            replace = True if conv7x7 in model_dict_key_re else False
            model_dict_key_re = model_dict_key_re.replace(conv7x7, '')
            if replace and model_dict_key_re in state_dict_keys:
                model_dict[model_dict_key] = state_dict[model_dict_key_re]

    return model_dict


if __name__ == '__main__':
    model = posenet18(num_parallel=2, bn_threshold=2e-2, r_type='euler')
    model_init(model, 18, 2)
    inputs = [torch.randn(12, 6, 192, 640), torch.randn(12, 6, 192, 640)]
    outputs = model(inputs)
    print(len(outputs))
