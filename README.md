# Training

```
python train.py --data_path ~/Documents/datasets/kitti_odom/dataset --log_dir ~/Documents/MLF-VO/KITTI --model_name test --num_epochs 20 --dataset kitti_odom --split odom
```

Please change L55 in `trainer.py` into the PoseNet we provide in the master branch.

```python
self.models['pose'] = posenet_type_dict[config_dict.model.type]()
```

I wll find some time to clean the code in this branch for easy use.
