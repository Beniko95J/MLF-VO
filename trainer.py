from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssim = SSIM()
ssim.to(device)


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        # Model definition
        self.models = {}
        self.parameters_to_train = []

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        self.models["pose"] = networks.posenet18(
            num_parallel=2, bn_threshold=2e-2)
        networks.model_init(self.models["pose"], 18, 2)
        self.models["pose"].to(device)
        self.parameters_to_train += list(self.models["pose"].parameters())

        self.slim_params = []
        for name, param in self.models["pose"].named_parameters():
            if param.requires_grad and name.endswith('weight') and 'bn2' in name:
                if 'bn_0' in name:
                    self.slim_params.append(param[:len(param) // 2])
                else:
                    self.slim_params.append(param[len(param) // 2:])

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.5)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n", self.opt.model_name)
        print("The used memory is about {:.2f}Mb".format(
            sum(param.numel() for param in self.parameters_to_train) * 4 / 1024 / 1024))
        print("Models and tensorboard events files are saved to:\n",
              self.opt.log_dir)
        print("Training is using:\n", device)

        # Data
        datasets_dict = {"kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png'

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        print("Using split:\n", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        # Misc definition
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(device)

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self): 
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
            
            slim_params_list = []
            for slim_param in self.slim_params:
                slim_params_list.extend(slim_param.cpu().data.numpy())
            slim_params_list = np.array(sorted(slim_params_list))
            print('Epoch %d, 3%% smallest slim_params: %.4f, largest slim_params: %.4f'
                    % (self.epoch, slim_params_list[len(slim_params_list) // 33], slim_params_list[len(slim_params_list)-1]),
                    flush=True)
            print('Epoch %d, portion of slim_params < %.e: %.4f'
                    % (self.epoch, 2e-2, sum(slim_params_list < 2e-2) / len(slim_params_list)),
                    flush=True)

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time
            
            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        self.model_lr_scheduler.step()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, value in inputs.items():
            inputs[key] = value.to(device)
        
        outputs = {}
        for f_i in self.opt.frame_ids:
            outputs["disp", f_i] = self.models["depth"](
                self.models["encoder"](inputs["color_aug", f_i, 0]))
        outputs.update(self.predict_poses(inputs, outputs))

        self.generate_images_pred(inputs, outputs)
        self.generate_depths_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, outputs):
        """Predict poses between input frames for monocular sequences.
        """
        pose_outputs = {}
        rgbs = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
        for f_i in self.opt.frame_ids[1:]:
            if f_i < 0:
                rgb_inputs = [rgbs[f_i], rgbs[0]]
                depth_inputs = [outputs["disp", f_i][("disp", 0)], outputs["disp", 0][("disp", 0)]]
            else:
                rgb_inputs = [rgbs[0], rgbs[f_i]]
                depth_inputs = [outputs["disp", 0][("disp", 0)], outputs["disp", f_i][("disp", 0)]]

            rgb_inputs = torch.cat(rgb_inputs, 1)
            depth_inputs[0] = depth_inputs[0].repeat(1, 3, 1, 1)
            depth_inputs[1] = depth_inputs[1].repeat(1, 3, 1, 1)
            depth_inputs = torch.cat(depth_inputs, 1)

            pose_inputs = [rgb_inputs, depth_inputs]
            # -1 -> 0 and 0 -> 1
            axisangle, translation = self.models["pose"](pose_inputs)
            # 0 -> -1 and 0 -> 1
            pose_outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        return pose_outputs

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs["disp", 0]["disp", scale]
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs["depth", 0, scale] = depth

            for frame_id in self.opt.frame_ids[1:]:
                T = outputs["cam_T_cam", 0, frame_id]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs["inv_K", source_scale])
                pix_coords, computed_depth = self.project_3d[source_scale](
                    cam_points, inputs["K", source_scale], T)

                outputs["sample", frame_id, scale] = pix_coords

                outputs["color", frame_id, scale] = F.grid_sample(
                    inputs["color", frame_id, source_scale],
                    outputs["sample", frame_id, scale],
                    padding_mode="border", align_corners=True)

                outputs["computed_depth", frame_id, scale] = computed_depth.clamp(min=self.opt.min_depth, max=self.opt.max_depth)

                valid_points = pix_coords.abs().max(dim=-1)[0] <= 1
                valid_mask = valid_points.unsqueeze(1).float()
                outputs["valid_mask", frame_id, scale] = valid_mask

    def generate_depths_pred(self, inputs, outputs):
        for scale in self.opt.scales:
            for frame_id in self.opt.frame_ids[1:]:
                disp = outputs['disp', frame_id][("disp", scale)]
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                
                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

                outputs[("projected_depth", frame_id, scale)] = F.grid_sample(
                    depth,
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

    def compute_photometric_loss(self, inputs, outputs, loss_dict):
        loss = 0

        for scale in self.opt.scales:
            losses = []
            target = inputs[("color", 0, 0)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                losses.append(compute_reprojection_loss(pred, target))
            
            for frame_id in self.opt.frame_ids[1:]:
                source = inputs[("color", frame_id, 0)]
                losses.append(compute_reprojection_loss(source, target))
            
            losses = torch.cat(losses, dim=1)
            losses, min_indices = torch.min(losses, dim=1)
            auto_mask = (min_indices < len(self.opt.frame_ids[1:])).float()
            outputs["auto_mask", scale] = auto_mask
            
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                occ_mask = (min_indices == i).float()
                outputs["occ_mask", frame_id, scale] = occ_mask.unsqueeze(dim=1)
            
            loss_dict["Lp/{}".format(scale)] = losses.mean()
            loss += loss_dict["Lp/{}".format(scale)]
        
        loss /= len(self.opt.scales)
        return loss

    def compute_smoothness_loss(self, inputs, outputs, loss_dict):
        loss = 0

        for scale in self.opt.scales:
            disp = outputs["disp", 0][("disp", scale)]
            color = inputs[("color", 0, scale)]

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            loss_dict["Ls/{}".format(scale)] = get_smooth_loss(norm_disp, color)
            loss += loss_dict["Ls/{}".format(scale)]
        
        loss /= len(self.opt.scales)
        return loss

    def compute_geometric_loss(self, inputs, outputs, loss_dict):
        loss = 0

        for scale in self.opt.scales:
            tmp = 0
            for frame_id in self.opt.frame_ids[1:]:
                computed_depth = outputs["computed_depth", frame_id, scale]
                projected_depth = outputs["projected_depth", frame_id, scale]

                valid_mask = outputs["valid_mask", frame_id, scale]
                occ_mask = outputs["occ_mask", frame_id, scale]
                mask = valid_mask * occ_mask

                diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)

                tmp += mean_on_mask(diff_depth, mask)

            loss_dict["Lg/{}".format(scale)] = tmp
            loss += loss_dict["Lg/{}".format(scale)]
        
        loss /= len(self.opt.scales)
        return loss

    def compute_bn_regularization(self):
        return sum([L1_penalty(m).cuda() - 0.1 * polorize(m).cuda() for m in self.slim_params])

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        loss_dict = {}
        
        Lp = self.compute_photometric_loss(inputs, outputs, loss_dict)
        Lg = self.compute_geometric_loss(inputs, outputs, loss_dict)
        Ls = self.compute_smoothness_loss(inputs, outputs, loss_dict)
        Lb = self.compute_bn_regularization()

        loss_dict["Lp"] = Lp
        loss_dict["Lg"] = Lg
        loss_dict["Ls"] = Ls
        loss_dict["Lb"] = Lb

        loss_dict["loss"] = Lp + 1e-2 * Lg + 1e-3 * Ls + 2e-5 * Lb

        return loss_dict

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    tensor2array(normalize_image(1 / outputs[("depth", 0, s)][j])), self.step)
            
            writer.add_image(
                "auto_mask/{}".format(j),
                outputs["auto_mask", 0][j][None, ...], self.step)

            for frame_id in self.opt.frame_ids:
                writer.add_image(
                    "color_{}/{}".format(frame_id, j),
                    inputs[("color", frame_id, 0)][j].data, self.step)
    
                if frame_id != 0:
                    writer.add_image(
                        "color_pred_{}/{}".format(frame_id, j),
                        outputs[("color", frame_id, 0)][j].data, self.step)

                    writer.add_image(
                        "computed_depth_{}/{}".format(frame_id, j),
                        tensor2array(normalize_image(1 / outputs["computed_depth", frame_id, 0][j])), self.step)

                    writer.add_image(
                        "projected_depth_{}/{}".format(frame_id, j),
                        tensor2array(normalize_image(1 / outputs["projected_depth", frame_id, 0][j])), self.step)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


def L1_penalty(x):
    return torch.abs(x).sum()


def polorize(x):
    return torch.abs(x - torch.mean(x)).sum()


def compute_reprojection_loss(pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss


def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    if mask.sum() > 10000:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0).float().to(device)
    return mean_value
