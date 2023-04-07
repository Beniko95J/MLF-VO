from evaluation.util import *
from evaluation.model_factory import *

import argparse
import matplotlib.pyplot as plt


model_type_dict = {
    'multimodal' : Model_multimodal,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate eigen depth.')
    parser.add_argument('-c', '--config', type=str,
                        default=None,
                        help='configuration file')
    args = parser.parse_args()
    return args


class DataFeeder:
    def __init__(self, cfg) -> None:
        self.eigen_path = cfg.eigen_path
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

        self.eigen_files = []
        with open(cfg.eigen_split_file) as file:
            for line in file.readlines():
                if line.strip():
                    self.eigen_files.append(line)
    
    def get_image_path(self, folder, frame_index, side) -> str:
        f_str = "{:010d}.png".format(frame_index)
        image_path = os.path.join(
            self.eigen_path, folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path
    
    def feed_data(self):
        for i, line in enumerate(self.eigen_files):
            folder, frame_index, side = line.split()
            frame_index = int(frame_index)
            file = self.get_image_path(folder, frame_index, side)
            yield(i, file)
    
    def __len__(self):
        return len(self.eigen_files)


class DepthEstimator:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

        self.model_name = []
        self.model = []
        self.model_type = []
        for i, model_info in enumerate(cfg.models):
            print('Loading model {}...'.format(i+1))
            self.model_name.append(model_info.model_name)
            self.model.append(model_type_dict[model_info.model_type](model_info.model_path))
            self.model_type.append(model_info.model_type)
            print('Model {} loaded.'.format(i+1))
        
        self.n_models = len(self.model_name)

        print('Initializing data feeder...')
        self.data_feeder = DataFeeder(cfg)
        print('Data feeder initialized.')
    
    def main(self):
        for idx, file in self.data_feeder.feed_data():
            self.cur_img_plot = read_rgb_image(file)
            self.cur_img = preprocess_img(self.cur_img_plot)
            
            self.cur_disp_plot = []
            for model in self.model:
                cur_disp = model.infer_disp(self.cur_img)
                self.cur_disp_plot.append(postprocess_disp(cur_disp))
            
            fig = plt.figure(constrained_layout=True)
            axs = []
            for i in range(self.n_models+1):
                axs.append(fig.add_subplot(self.n_models+1, 1, i+1))
            
            for i, ax in enumerate(axs):
                if i == 0:
                    ax.imshow(self.cur_img_plot)
                    ax.set_title('RGB')
                else:
                    ax.imshow(self.cur_disp_plot[i-1], cmap='magma', vmax=np.percentile(self.cur_disp_plot[i-1], 95))
                    ax.set_title(self.model_name[i-1])
                ax.set_xticks([])
                ax.set_yticks([])
            
            png_title = "{:04}.png".format(idx)
            fig_png = os.path.join('assets/depths', png_title)
            fig.savefig(fig_png, bbox_inches='tight', pad_inches=0)

            print('\r', idx, len(self.data_feeder), end=' '*10)

            plt.close(fig=fig)


if __name__ == '__main__':
    args = parse_args()
    config_files = [args.config]
    cfg = merge_cfg(config_files)
    
    de = DepthEstimator(cfg)
    de.main()
