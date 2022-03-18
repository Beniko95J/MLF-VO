import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d

from .modules import *


model_urls = {
    '18_imagenet': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    '50_imagenet': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                     stride=stride, padding=0, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class PoseNet(nn.Module):

    def __init__(self, block, layers):
        super(PoseNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fc1 = nn.Linear(512 * 6 * 20, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3_t = nn.Linear(512, 3)
        self.fc3_r = nn.Linear(512, 3)

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
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                BatchNorm2d(planes * block.expansion)
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.cat(x, dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        x_t = self.fc3_t(x)
        x_r = self.fc3_r(x)
        x = torch.cat([x_r, x_t], dim=1)

        axisangle = x[:, 0:3]
        translation = x[:, 3:6]

        axisangle = 0.001 * axisangle.view(-1, 1, 1, 3)
        translation = 0.001 * translation.view(-1, 1, 1, 3)

        return axisangle, translation


def _posenet(block, layers):
    model = PoseNet(block, layers)
    return model


def posenet18():
    return _posenet(BasicBlock, [2, 2, 2, 2])


def posenet50():
    return _posenet(Bottleneck, [3, 4, 6, 3])


def model_init(model, num_layers, num_parallel):
    key = str(num_layers) + '_imagenet'
    url = model_urls[key]
    state_dict = maybe_download(key, url)
    state_dict['conv1.weight'] = torch.cat(
            [state_dict['conv1.weight']] * 4, 1) / 4
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
    model = posenet18()
    model_init(model, 18, 2)
    inputs = [torch.randn(12, 6, 192, 640), torch.randn(12, 6, 192, 640)]
    outputs = model(inputs)
