from evaluation.eval_constant import eval_const
from networks import *
from layers import transformation_from_parameters

import os
import numpy as np
import PIL.Image as pil
import torch
import torch.nn as nn
import torchvision.transforms as transforms


def read_rgb_image(file: str, do_flip=False) -> pil.Image:
    img = pil.open(file).convert('RGB')
    if do_flip:
        return img.transpose(pil.FLIP_LEFT_RIGHT)
    else:
        return img


def preprocess_img(img: pil.Image) -> torch.Tensor:
    to_tensor = transforms.ToTensor()
    
    img = transforms.Resize((eval_const.HEIGHT, eval_const.WIDTH), interpolation=pil.ANTIALIAS)(img)
    img = to_tensor(img)
    img = img.unsqueeze(0)
    img = img.cuda()
    return img


def postprocess_disp(disp: torch.Tensor) -> np.ndarray:
    disp = disp.squeeze()
    disp = disp.cpu().numpy()
    return disp


def load_state_dict(model: nn.Module, file: str) -> None:
    if 'encoder.pth' in file:
        l_dict = torch.load(file)
        model.load_state_dict({k : v for k, v in l_dict.items() if k in model.state_dict()})
    else:
        model.load_state_dict(torch.load(file))
    
    model.eval()


class BaseModel:
    def __init__(self) -> None:
        self.model = {}

    def load_model(self, path: str) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def infer_disp(self, img: torch.Tensor) -> torch.Tensor:
        feature = self.model['encoder'](img)
        disp = self.model['depth'](feature)['disp', 0]
        return disp

    @torch.no_grad()
    def infer_pose(self, img1: torch.Tensor, img2: torch.Tensor) -> np.ndarray:
        raise NotImplementedError


class Model_multimodal(BaseModel):
    def __init__(self, path) -> None:
        super().__init__()
        self.load_model(path)
    
    def load_model(self, path: str) -> None:        
        self.model['encoder'] = ResnetEncoder(
            num_layers=eval_const.NUM_LAYERS,
            pretrained=False
        ).eval().cuda()
        load_state_dict(self.model['encoder'], os.path.join(path, 'encoder.pth'))

        self.model['depth'] = DepthDecoder(
            num_ch_enc=self.model['encoder'].num_ch_enc,
            scales=range(4)
        ).eval().cuda()
        load_state_dict(self.model['depth'], os.path.join(path, 'depth.pth'))

        self.model['pose'] = posenet18(num_parallel=2, bn_threshold=2e-2).eval().cuda()
        # self.model['pose'] = posenet18().eval().cuda()
        load_state_dict(self.model['pose'], os.path.join(path, 'pose.pth'))

    @torch.no_grad()
    def infer_pose(self, img1: torch.Tensor, img2: torch.Tensor, disp1: torch.Tensor, disp2: torch.Tensor) -> np.ndarray:
        rgb_inputs = torch.cat([img1, img2], 1)

        disp1 = disp1.repeat(1, 3, 1, 1)
        disp2 = disp2.repeat(1, 3, 1, 1)
        depth_inputs = torch.cat([disp1, disp2], 1)

        pose_input = [rgb_inputs, depth_inputs]
        axisangle, translation = self.model['pose'](pose_input)

        T = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)
        T = T.cpu().numpy()[0]

        return T
