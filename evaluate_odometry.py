from evaluation.util import *
from evaluation.model_factory import *
from evaluation.frame_drawer import FrameDrawer

import os
import argparse


model_type_dict = {
    'multimodal' : Model_multimodal,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate odometry')
    parser.add_argument('-c', '--config', type=str,
                        default=None,
                        help='configuration file')
    parser.add_argument('--selective', action='store_true')
    parser.add_argument('-s', '--seqs', type=int,
                        nargs='+',
                        default=[9, 10],
                        help='sequences to process')
    args = parser.parse_args()
    return args


class DataFeeder:
    def __init__(self, cfg) -> None:
        self.seq_path = os.path.join(
            cfg.data_dir,
            "sequences/{:02d}/image_2".format(cfg.seq)
        )

        self.image_files = sorted(os.listdir(self.seq_path))     

    def feed_data(self):
        for i, file in enumerate(self.image_files):
            file = os.path.join(self.seq_path, file)
            yield (i, file)
    
    def __len__(self):
        return len(self.image_files)


class VisualOdometry:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model_type = cfg.model_type
        self.do_vis = cfg.do_vis

        # print('Loading models...')
        self.model = model_type_dict[cfg.model_type](cfg.model_dir)
        # print('Model loaded.')

        # print('Loading gt poses...')
        gt_poses_file = os.path.join(cfg.gt_dir, '{:02d}.txt'.format(cfg.seq))
        self.gt_poses = load_poses_from_txt(gt_poses_file)
        # print('Gt poses loaded.')

        # print('Initializing data feeder...')
        self.data_feeder = DataFeeder(cfg)
        # print('Data feeder initialized.')

        assert len(self.data_feeder) == len(self.gt_poses), "lengths of data feeder and gt poses are mismatched!"

        if self.do_vis:
            # print('Initializing frame drawer...')
            self.drawer = FrameDrawer(cfg.vis.window_h, cfg.vis.window_w)
            self.init_frame_drawer()
            self.drawer.set_traj_init_xy(self.gt_poses)
            # print('Frame drawer initialized.')

        # infer and plot members
        self.prev_img = None
        self.prev_disp = None
        self.cur_img = None
        self.cur_disp = None
        self.mask = None

        self.prev_img_plot = None
        self.prev_disp_plot = None
        self.cur_img_plot = None
        self.cur_disp_plot = None
        self.mask_plot = None

        self.transformations = []
        self.pred_poses = [np.eye(4)]

    def init_frame_drawer(self):
        h = self.drawer.h
        w = self.drawer.w
        
        self.drawer.assign_data(
            item='traj',
            top_left=[0, 0],
            bottom_right=[int(h), int(w/2)]
        )

        self.drawer.assign_data(
            item='prev_img',
            top_left=[int(h/3*0), int(w/4*2)],
            bottom_right=[int(h/3*1), int(w/4*3)]
        )

        self.drawer.assign_data(
            item='prev_disp',
            top_left=[int(h/3*0), int(w/4*3)],
            bottom_right=[int(h/3*1), int(w/4*4)]
        )

        self.drawer.assign_data(
            item='cur_img',
            top_left=[int(h/3*1), int(w/4*2)],
            bottom_right=[int(h/3*2), int(w/4*3)]
        )

        self.drawer.assign_data(
            item='cur_disp',
            top_left=[int(h/3*1), int(w/4*3)],
            bottom_right=[int(h/3*2), int(w/4*4)]
        )

        self.drawer.assign_data(
            item='mask',
            top_left=[int(h/3*2), int(w/4*2)],
            bottom_right=[int(h/3*3), int(w/4*4)]
        )

    def main(self):
        for idx, file in self.data_feeder.feed_data():
            self.cur_img_plot = read_rgb_image(file)
            self.cur_img = preprocess_img(self.cur_img_plot)
            self.cur_disp = self.model.infer_disp(self.cur_img)
            self.cur_disp_plot = postprocess_disp(self.cur_disp)

            if self.prev_img is not None:
                T = self.model.infer_pose(self.prev_img, self.cur_img, self.prev_disp, self.cur_disp)
                self.transformations += [T]
                self.pred_poses.append(np.dot(self.pred_poses[-1], self.transformations[-1]))

            if self.do_vis:
                self.drawer.plot(self)
            
            self.prev_img = self.cur_img
            self.prev_disp = self.cur_disp
            self.prev_img_plot = self.cur_img_plot
            self.prev_disp_plot = self.cur_disp_plot
            
            print('\r', idx+1, len(self.data_feeder), end=' '*10)
    
        if self.cfg.output_dir == 'None':
            file_path = '{:02d}.txt'
        else:
            file_path = os.path.join(self.cfg.output_dir, '{:02d}.txt')
        
        save_traj(file_path.format(self.cfg.seq), self.pred_poses)


if __name__ == '__main__':
    args = parse_args()
    config_files = [args.config]
    cfg = merge_cfg(config_files)

    if not args.selective:
        vo = VisualOdometry(cfg)
        vo.main()
    else:
        for seq in args.seqs:
            print('Processing seq.{:02d}'.format(seq))
            cfg.seq = seq
            vo = VisualOdometry(cfg)
            vo.main()
