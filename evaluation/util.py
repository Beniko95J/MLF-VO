import yaml

import numpy as np
from easydict import EasyDict as edict


def read_yaml(file: str) -> dict:
    if file is not None:
        with open(file, 'r') as f:
            return yaml.load(f, yaml.FullLoader)
    else:
        return {}


def update_dict(dict1: dict, dict2: dict) -> dict:
    for item in dict2:
        if dict1.get(item, -1) != -1:
            if isinstance(dict1[item], dict):
                dict1[item] = update_dict(dict1[item], dict2[item])
            else:
                dict1[item] = dict2[item]
        else:
            dict1[item] = dict2[item]
    return dict1


def merge_cfg(files: list) -> edict:
    cfg = {}
    for f in files:
        if f is not None:
            cfg = update_dict(cfg, read_yaml(f))
    return edict(cfg)


def load_poses_from_txt(file: str) -> dict:
    f = open(file, 'r')
    s = f.readlines()
    f.close()
    pose = {}

    for cnt, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split(' ')]
        withIdx = int(len(line_split) == 13)
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row * 4 + col + withIdx]
            if withIdx:
                frame_idx = line_split[0]
            else:
                frame_idx = cnt
            pose[frame_idx] = P

    return pose


def save_traj(file: str, poses: list) -> None:
    poses = [p[0:3, 0:4].reshape((12,)) for p in poses]

    with open(file, 'w') as f:
        np.savetxt(f, poses, delimiter=' ')
    
    print('Trajectory saved.')
