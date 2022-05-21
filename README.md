# MLF-VO

This repo implements the network described in the ICRA2022 paper:

[Self-Supervised Ego-Motion Estimation Based on Multi-Layer Fusion of RGB and Inferred Depth](https://arxiv.org/pdf/2203.01557.pdf)

Zijie Jiang, Hajime Taira, Naoyuki Miyashita and Masatoshi Okutomi

Our homepage is [here](http://www.ok.sc.e.titech.ac.jp/res/MLF-VO/). The pretrained weights of models and results can be found [here](https://drive.google.com/drive/folders/1bogcNuteWNce_551jscX-leo54YYhYZY?usp=sharing).

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

Our experiments are conducted on a machine installed with Ubuntu 18.04, Pytorch 1.7.1, CUDA 10.1. Install the requirements by executing:

``` bash
pip install requirements.txt
```

Considering the strict LICENSE of [Monodepth2](https://github.com/nianticlabs/monodepth2), you can download the necessary files by executing:

``` bash
cd third_party/Monodepth2
wget https://raw.githubusercontent.com/nianticlabs/monodepth2/master/networks/depth_decoder.py
wget https://raw.githubusercontent.com/nianticlabs/monodepth2/master/networks/resnet_encoder.py
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

The output is a `.txt` file which contains the estimated trajectory in the following format:

```
T11 T12 T13 T14 T21 T22 T23 T24 T31 T32 T33 T34 # A flattened 3*4 transformation matrix.
...
```

We recommend to use this [toolbox](https://github.com/Huangying-Zhan/kitti-odom-eval) to evaluate the inferred trajectory.

## 4. Training

The codes for training will be released after cleaning.

## Acknowledgement

We are grateful to the authors of [Monodepth2](https://github.com/nianticlabs/monodepth2) and [CEN](https://github.com/yikaiw/CEN) for publicly sharing their codes.

## License

MLF-VO is released under MIT License.
