import os
import argparse
from pathlib import Path
import yaml
from easydict import EasyDict as edict
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

from models import create_posenet_early, create_posenet_late, create_posenet_middle, create_posenet_multi_layer
from libs.kinetics import from_euler_t
from libs.misc import create_dir_if_not_exist
from third_party.monodepth2.networks import ResnetEncoder
from third_party.monodepth2.networks import DepthDecoder


posenet_type_dict = {
    'early': create_posenet_early,
    'late': create_posenet_late,
    'middle': create_posenet_middle,
    'multi_layer': create_posenet_multi_layer
}


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class KIITIOdomDataset:
    def __init__(self, config_dict) -> None:
        super().__init__()

        self.seq_dir = os.path.join(config_dict.data.path, 'image_2')
        self.height = config_dict.data.height
        self.width = config_dict.data.width

        self.filename_list = sorted(os.listdir(self.seq_dir))
        self.loader = pil_loader
        self.resize = transforms.Resize((self.height, self.width), interpolation=Image.ANTIALIAS)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.filename_list)

    def preprocess(self, rgb_image):
        rgb_image = self.to_tensor(self.resize(rgb_image))
        return rgb_image[None, ...].cuda()

    def getitem(self, index):
        filename = self.filename_list[index]
        file_path = os.path.join(self.seq_dir, filename)
        rgb_image = self.loader(file_path)
        rgb_image = self.preprocess(rgb_image)

        return rgb_image


def initialize_models(config_dict):
    models = {}
    models['encoder'] = ResnetEncoder(num_layers=18, pretrained=False)
    models['depth'] = DepthDecoder(num_ch_enc=models['encoder'].num_ch_enc)

    if config_dict.model.type not in posenet_type_dict:
        raise ValueError("{} is not a valid type of posenet".format(config_dict.model.type))

    models['pose'] = posenet_type_dict[config_dict.model.type]()

    state_dict = torch.load(Path(config_dict.model.path) / 'encoder.pth')
    models['encoder'].load_state_dict({k : v for k, v in state_dict.items() if k in models['encoder'].state_dict()})
    models['depth'].load_state_dict(torch.load(Path(config_dict.model.path) / 'depth.pth'))
    models['pose'].load_state_dict(torch.load(Path(config_dict.model.path) / 'pose.pth'))

    models['encoder'] = models['encoder'].cuda().eval()
    models['depth'] = models['depth'].cuda().eval()
    models['pose'] = models['pose'].cuda().eval()

    return models


def save_traj(file_path: str, pose_list: list) -> None:
    pose_list = [p[0:3, 0:4].reshape((12,)) for p in pose_list]

    with open(file_path, 'w') as f:
        np.savetxt(f, pose_list, delimiter=' ')

    print('Trajectory saved to {}.'.format(file_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, default='configs/run_odometry.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.load(f, yaml.FullLoader)
        config_dict = edict(config_dict)

    odom_dataset = KIITIOdomDataset(config_dict)
    models = initialize_models(config_dict)
    seq_name = config_dict.data.path.split('/')[-1]
    create_dir_if_not_exist(config_dict.output.path)

    traj = [np.eye(4)]
    prev_rgb = None
    for i in tqdm(range(len(odom_dataset))):
        cur_rgb = odom_dataset.getitem(i)
        cur_disp = models['depth'](models['encoder'](cur_rgb))['disp', 0]

        if prev_rgb is not None:
            rgb_inputs = torch.cat([prev_rgb, cur_rgb], dim=1)
            disp_inputs = torch.cat([prev_disp.repeat(1, 3, 1, 1), cur_disp.repeat(1, 3, 1, 1)], dim=1)

            with torch.no_grad():
                euler_angle, translation = models['pose']([rgb_inputs, disp_inputs])
                T = from_euler_t(euler_angle, translation, invert=True)
                traj.append(np.dot(traj[-1], T.cpu().numpy()[0]))

        prev_rgb = cur_rgb
        prev_disp = cur_disp

    save_traj(os.path.join(config_dict.output.path, '{}.txt'.format(seq_name)), traj)

if __name__ == '__main__':
    main()
