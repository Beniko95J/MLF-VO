# MLF-VO

This repo implements the network described in the ICRA2022 paper:

[Self-Supervised Ego-Motion Estimation Based on Multi-Layer Fusion of RGB and Inferred Depth](https://arxiv.org/pdf/2203.01557.pdf)

Zijie Jiang, Hajime Taira, Naoyuki Miyashita and Masatoshi Okutomi

Please also visit our [project page](http://www.ok.sc.e.titech.ac.jp/res/MLF-VO/). The pretrained weights of models and results can be found [here](https://drive.google.com/drive/folders/1bogcNuteWNce_551jscX-leo54YYhYZY?usp=sharing).

If you find our work useful for your research, please consider citing the following paper:

```
@inproceedings{jiang2022mlfvo,
  title={Self-Supervised Ego-Motion Estimation Based on Multi-Layer Fusion of RGB and Inferred Depth},
  author={Jiang, Zijie and Taira, Hajime and Miyashita, Naoyuki and Okutomi, Masatoshi},
  booktitle={2022 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2022}
}
```

# Updates

- [x] 2022.05.20. Pretrained weights and demo for testing are open.

# Requirements

Our experiments are conducted on a machine installed with Ubuntu 18.04, Pytorch 1.7.1, CUDA 10.1. You can install other required packages by:

``` bash
git clone --recurse https://github.com/Beniko95J/MLF-VO.git
pip install requirements.txt
```

Our demo partly depends on [Monodepth2](https://github.com/nianticlabs/monodepth2) and [CEN](https://github.com/yikaiw/CEN). Please also refer their original repositories.

# Prepare dataset

Please download the KITTI odometry benchmark from their [site](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) and create a soft link by executing the following command:

``` bash
ln -s /path/to/dataset/data_odometry_color/dataset data
```

# Odometry evaluation

You can download the pretrained weights of models from [here](https://drive.google.com/drive/folders/1bogcNuteWNce_551jscX-leo54YYhYZY?usp=sharing). To obtain the results reported in the paper, create a soft link:

``` bash
ln -s /path/to/models data
```

The demo specifies several paths and configurations in `configs/run_odometry.yaml`. Please modify line 2, 6, or 9 to use your own paths to dataset, model weights, or outputs.

``` bash
python run_odometry -c configs/run_odometry.yaml
```

The estimated trajectory will be exported to `data/outputs/09.txt`. Each line in the file represents the estimated camera, corresponding to each frame in the sequence, in the following format:

```
T11 T12 T13 T14 T21 T22 T23 T24 T31 T32 T33 T34
...
```

where T11, T12, ... , T34 are the elements of 3x4 camera transformation matrix (from camera to world coordinate system):

```
T11 T12 T13 T14
T21 T22 T23 T24
T31 T32 T33 T34
```

We recommend to use this [toolbox](https://github.com/Huangying-Zhan/kitti-odom-eval) to evaluate the inferred trajectory.

# Acknowledgement

We are grateful to the authors of [Monodepth2](https://github.com/nianticlabs/monodepth2) and [CEN](https://github.com/yikaiw/CEN) for publicly sharing their codes.

# License

MLF-VO is released under GPLv3 License.
