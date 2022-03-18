from evaluation.util import *
from evaluation.model_factory import *

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
    parser.add_argument('--select_mode', action='store_true')
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

        print('Loading models...')
        self.model = model_type_dict[cfg.model_type](cfg.model_dir)

        print('Loading gt poses...')
        gt_poses_file = os.path.join(cfg.gt_dir, '{:02d}.txt'.format(cfg.seq))
        self.gt_poses = load_poses_from_txt(gt_poses_file)

        print('Initializing data feeder...')
        self.data_feeder = DataFeeder(cfg)

        assert len(self.data_feeder) == len(self.gt_poses), "lengths of data feeder and gt poses are mismatched!"

        self.prev_img = None
        self.prev_disp = None
        self.cur_img = None
        self.cur_disp = None

        self.prev_img_plot = None
        self.prev_disp_plot = None
        self.cur_img_plot = None
        self.cur_disp_plot = None

        self.transformations = []
        self.pred_poses = [np.eye(4)]

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

                # save inferred depth, resized RGB
            
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

    if not args.select_mode:
        print('Processing seq.{:02d}'.format(cfg.seq))
        vo = VisualOdometry(cfg)
        vo.main()
    else:
        print('Select mode.')
        for seq in args.seqs:
            print('Processing seq.{:02d}'.format(seq))
            cfg.seq = seq
            vo = VisualOdometry(cfg)
            vo.main()
