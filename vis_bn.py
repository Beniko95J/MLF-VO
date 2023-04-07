import torch
import numpy as np
import matplotlib.pyplot as plt

from networks import posenet18

# model_path = '/home/jiang/Documents/icra2022/cen_alpha/models/weights_35/pose.pth'
model_path = '/mnt/cronus_hdd/icra2022/debug3/models/weights_36/pose.pth'
model = posenet18(num_parallel=2, bn_threshold=2e-2, r_type='euler')
pretrained_dict = torch.load(model_path)
model.load_state_dict(pretrained_dict)

model = model.eval()

bn_weights = None
for name, param in model.named_parameters():
    if 'layer3.0.bn2.bn_0.weight' in name:
        print(torch.mean(param))
        bn_weights = param.detach().numpy()
        # bn_weights = bn_weights[:128]

plt.bar(range(len(bn_weights)), bn_weights)
plt.show()
