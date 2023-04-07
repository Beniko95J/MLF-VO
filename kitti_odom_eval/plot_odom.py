from kitti_odometry import umeyama_alignment

import json
import argparse
from collections import OrderedDict
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil


# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return np.array(xyzs)


def convert_global_to_local(poses):
    local_poses = []
    for i in range(1, len(poses)):
        local_poses.append(np.dot(np.linalg.inv(poses[i-1]), poses[i]))
    
    return np.array(local_poses)


def load_poses(file_path):
    #FIXME: Should support KITTI format or TUM format.
    f = open(file_path, 'r')
    s = f.readlines()
    f.close()
    poses = []
    for _, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split(" ") if i!=""]
        withIdx = len(line_split) == 13
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row*4 + col + withIdx]
        poses.append(np.array(P))
    return np.array(poses)


def load_all_poses(base_path, subpath_list, seq):
    """
    Return
        dict, {method(str) : [4x4 array], ...}
    """
    all_poses = {}
    for path in subpath_list:
        method = path.split('/')[0]
        file_name = os.path.join(base_path, path, '{:02d}.txt'.format(seq))
        if not os.path.exists(file_name):
            continue
        else:
            poses = load_poses(file_name)
        
        all_poses[method] = poses
    
    return all_poses


def pose_alignment(all_poses, alignment='scale_7dof'):
    poses_gt = all_poses['Ground-truth']
    local_gt = convert_global_to_local(poses_gt)
    xyz_gt = dump_xyz(local_gt)
    for method, poses_pred in all_poses.items():
        if method == 'Ground-truth':
            continue
        else:
            local_pred = convert_global_to_local(poses_pred)
            xyz_pred = dump_xyz(local_pred)
            length = min(len(xyz_pred), len(xyz_gt))
            r, t, scale = umeyama_alignment(xyz_pred[:length].transpose(1, 0), xyz_gt[:length].transpose(1, 0), alignment!="6dof")

            align_transformation = np.eye(4)
            align_transformation[:3:, :3] = r
            align_transformation[:3, 3] = t

            for cnt in range(len(poses_pred)):
                poses_pred[cnt][:3, 3] *= scale
                poses_pred[cnt] = align_transformation @ poses_pred[cnt]

            all_poses[method] = poses_pred

    return all_poses


def plot_multiple_trajectory(all_poses, colormap, seq, output_path=None):
    """
    Args
        all_poses (dict): {method(str) : {idx : 4x4 array, ...}, ...}
    """
    fig = plt.figure(constrained_layout=True, figsize=(9,9))
    ax = plt.gca()
    ax.set_aspect('equal', 'datalim')
    font_size = 24

    for method, poses in all_poses.items():
        xyz = np.array([p[0:3, 3] for p in poses])
        
        local_poses = convert_global_to_local(poses)
        xyz = dump_xyz(local_poses)
        
        plt.plot(xyz[:, 0], xyz[:, 2], '-o', label=method, markersize=1, color=colormap[method])
    
    ax.legend(prop={'size': font_size})
    
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel('x (m)', fontsize=font_size)
    plt.ylabel('z (m)', fontsize=font_size)

    if output_path is not None:
        png_title = "sequence_{:02}.pdf".format(seq)
        fig_pdf = os.path.join(output_path, png_title)
        fig.savefig(fig_pdf, pad_inches=0)
    
    plt.close(fig)


def plot_multiple_trajectory_animation(all_poses, colormap, seq_path, seq_length):
    image_files = sorted(os.listdir(seq_path))

    xlim = None
    ylim = None
    for method, poses in all_poses.items():
        local_poses = convert_global_to_local(poses)
        xyz = dump_xyz(local_poses)
        if xlim is None:
            xlim = [np.min(xyz[:, 0]), np.max(xyz[:, 0])]
            ylim = [np.min(xyz[:, 2]), np.max(xyz[:, 2])]
        else:
            a = xlim[0]
            b = np.min(xyz[:, 0])
            xlim[0] = min(xlim[0], np.min(xyz[:, 0]))
            xlim[1] = max(xlim[1], np.max(xyz[:, 0]))
            ylim[0] = min(ylim[0], np.min(xyz[:, 2]))
            ylim[1] = max(ylim[1], np.max(xyz[:, 2]))

    for idx in range(seq_length):
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(40, 3)
        ax1 = fig.add_subplot(gs[:12, :3])
        ax2 = fig.add_subplot(gs[12:, :3])
        ax2.set_xlim([xlim[0], xlim[1]])
        ax2.set_ylim([ylim[0], ylim[1]])
        ax2.set_aspect('equal')

        file = os.path.join(seq_path, image_files[idx])
        img = pil.open(file).convert('RGB')
        ax1.imshow(img)
        ax1.set_xticks([])
        ax1.set_yticks([])

        for method, poses in all_poses.items():
            local_poses = convert_global_to_local(poses[0:idx+1])
            xyz = dump_xyz(local_poses)
            ax2.plot(xyz[:, 0], xyz[:, 2], '-o', label=method, markersize=0.1, color=colormap[method])
        
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
        ax2.set_xticks([])
        ax2.set_yticks([])

        png_title = "{:04}.png".format(idx)
        fig_png = os.path.join('assets', png_title)
        fig.savefig(fig_png, bbox_inches='tight', pad_inches=0)

        print('\r', idx+1, seq_length, end=' '*10)

        plt.close(fig=fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', required=True)
    parser.add_argument('--seq', type=int, required=True)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--config', default='plot_config.json')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not os.path.exists(args.config):
        print("Config file does not exist.")
        sys.exit(1)

    json_data = {}
    with open(args.config, 'r') as config_file:
        json_data = json.load(config_file, object_pairs_hook=OrderedDict)
    plot_info = json_data['settings']

    all_poses = load_all_poses(args.base_path, plot_info['methods'], args.seq)
    
    seq_path = os.path.join(plot_info['data_path'], "sequences/{:02d}/image_2".format(args.seq))
    seq_length = len(os.listdir(seq_path))

    all_poses = pose_alignment(all_poses)

    if plot_info['mode'] == 'frame':
        plot_multiple_trajectory(all_poses, plot_info['colormap'], args.seq, output_path=args.output_path)
    elif plot_info['mode'] == 'animation':
        plot_multiple_trajectory_animation(all_poses, plot_info['colormap'], seq_path, seq_length)
    else:
        print('Mode is not supported!')


if __name__ == "__main__":
    main()
