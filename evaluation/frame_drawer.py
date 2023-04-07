import numpy as np
import cv2
from matplotlib import cm


class FrameDrawer:
    def __init__(self, h: int, w: int) -> None:
        self.h = h
        self.w = w
        self.img = np.zeros((h, w, 3), dtype=np.uint8)
        self.data = {}
        
        self.traj_x0 = None
        self.traj_z0 = None
    
    def set_traj_init_xy(self, poses: list) -> None:
        gt_Xs = []
        gt_Zs = []
        for cnt, i in enumerate(poses):
            true_x, _, true_z = poses[i][:3, 3]
            gt_Xs.append(true_x)
            gt_Zs.append(true_z)
            if cnt == 0:
                x0 = true_x
                z0 = true_z
        
        min_x, max_x = np.min(gt_Xs), np.max(gt_Xs)
        min_z, max_z = np.min(gt_Zs), np.max(gt_Zs)
        ratio_x = (x0 - min_x) / (max_x - min_x)
        ratio_z = (z0 - min_z) / (max_z - min_z)

        crop = [0.2, 0.8]
        self.traj_x0 = int(self.w/2 * (crop[1] - crop[0]) * ratio_x + self.w/2 * crop[0])
        self.traj_z0 = int(self.h * crop[1] -self.h * (crop[1] - crop[0]) * ratio_z)
    
    def assign_data(self, item: str, top_left: list, bottom_right: list) -> None:
        self.data[item] = self.img[
                                    top_left[0]:bottom_right[0],
                                    top_left[1]:bottom_right[1]]
    
    def update_data(self, item: str, data: np.ndarray) -> None:
        data = np.array(data)
        
        if 'mask' in item:
            data = np.expand_dims(data, 2) * 255
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        elif 'disp' in item:
            data = data / (np.amax(data) + 1e-7)
            data = np.uint8(cm.get_cmap('magma')(data) * 255)[:, :, :3]
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        else:
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        vis_h, vis_w, _ = self.data[item].shape
        data = cv2.resize(data, (vis_w, vis_h))
        self.data[item][...] = data

    def draw_traj(self, pred_poses: list, gt_poses: list, cfg) -> None:
        traj = self.data['traj']
        latest_id = len(pred_poses) - 1

        draw_scale = cfg.vis.draw_scale
        mono_scale = cfg.vis.mono_scale
        pred_draw_scale = draw_scale * mono_scale

        true_XX, _, true_ZZ = gt_poses[latest_id][:3, 3]
        true_x = int(true_XX * draw_scale) + self.traj_x0
        true_z = -int(true_ZZ * draw_scale) + self.traj_z0
        cv2.circle(traj, (true_x, true_z), 1, (0, 0, 255), 1)

        x, y, z = pred_poses[latest_id][:3, 3]
        draw_x = int(x * pred_draw_scale) + self.traj_x0
        draw_y = -int(z * pred_draw_scale) + self.traj_z0
        cv2.circle(traj, (draw_x, draw_y), 1, (0, 255, 0), 1)
    
    def plot(self, vo) -> None:
        vo.drawer.draw_traj(
            pred_poses=vo.pred_poses,
            gt_poses=vo.gt_poses,
            cfg=vo.cfg
        )

        if vo.prev_img_plot is not None:
            vo.drawer.update_data('prev_img', vo.prev_img_plot)
        
        if vo.prev_disp_plot is not None:
            vo.drawer.update_data('prev_disp', vo.prev_disp_plot)

        if vo.cur_img_plot is not None:
            vo.drawer.update_data('cur_img', vo.cur_img_plot)
        
        if vo.cur_disp_plot is not None:
            vo.drawer.update_data('cur_disp', vo.cur_disp_plot)

        if vo.mask is not None:
            vo.drawer.update_data('mask', vo.cur_mask_plot)
        
        cv2.imshow('VO', vo.drawer.img)

        if len(vo.pred_poses) == len(vo.data_feeder):
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)
