# MLF-VO

This repo implements the network described in the ICRA2022 paper:

[Self-Supervised Ego-Motion Estimation Based on Multi-Layer Fusion of RGB and Inferred Depth](https://arxiv.org/pdf/2203.01557.pdf)

Zijie Jiang, Hajime Taira, Naoyuki Miyashita and Masatoshi Okutomi

The pretrained weights of models and results can be found [here](https://drive.google.com/drive/folders/1bogcNuteWNce_551jscX-leo54YYhYZY?usp=sharing).

If you find our work useful for your research, please consider citing the following paper:

```
@inproceedings{jiang2022mlfvo,
  title={Self-Supervised Ego-Motion Estimation Based on Multi-Layer Fusion of RGB and Inferred Depth},
  author={Jiang, Zijie and Taira, Hajime and Miyashita, Naoyuki and Okutomi, Masatoshi},
  booktitle={2022 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2022}
}
```

## Contents

1. Requirements
2. Prepare dataset
3. Odometry evaluation
4. Training

## 1. Requirements

Install the requirements by executing:

``` bash
pip install requirements.txt
```

## 2. Prepare dataset

Please download the KITTI odometry benchmark from their [site](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) and create a soft link by executing the following command:

``` bash
ln -s /home/YourAccount/Downloads/data_odometry_color/dataset data
```

## 3. Odometry evaluation

You can download the pretrained weights of models from [here](https://drive.google.com/drive/folders/1bogcNuteWNce_551jscX-leo54YYhYZY?usp=sharing). To obtain the results reported in the paper, create a soft link:

``` bash
ln -s /home/YourAccount/Downloads/models data
```

and configure paths and model type in `configs/run_odometry.yaml` and run:

``` bash
python run_odometry -c configs/run_odometry.yaml
```

We recommend to use this [toolbox](https://github.com/Huangying-Zhan/kitti-odom-eval) to evaluted the inferred trajectory.

## 4. Training

The codes for training will be released after cleaning.

## Acknowledgement

We are grateful to the authors of [Monodepth2](https://github.com/nianticlabs/monodepth2) and [CEN](https://github.com/yikaiw/CEN) for publicly sharing their codes.

## License

MLF-VO is released under MIT License.
