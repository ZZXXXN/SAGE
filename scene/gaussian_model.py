#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.linear_layer import LinearLayer
from scene.gaussian_conv import GaussianConv
from scene.style_transfer import MulLayer, local_align
import torch.nn.functional as F
from torch.cuda.amp import autocast

class ViewAdaptiveMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_dim = 12
        self.hidden = 64
        self.out_dim = 3
        
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.out_dim)
        )
        
    def forward(self, x):
        return self.mlp(x)

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self, is_feature_model=False, is_style_model=False):
        
        if is_feature_model:
            return (
                self._xyz,
                self._scaling,
                self._rotation,
                self._opacity,
                self._vgg_features,
                self.feature_linear.state_dict(),
            )
        
        if is_style_model:
            return (
                self._xyz,
                self._scaling,
                self._rotation,
                self._opacity,
                self.final_vgg_features,
                self.decoder.state_dict(),
            )

        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args=None, from_feature_model=False, from_style_model=False):

        if from_feature_model:
            (self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self._vgg_features,
            self.feature_linear_state_dict) = model_args
            self.feature_linear = LinearLayer(inChanel=32, out_dim=256).cuda()
            self.feature_linear.load_state_dict(self.feature_linear_state_dict)
            return
        
        if from_style_model:
            (self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self.final_vgg_features,
            self.decoder_state_dict) = model_args
            self.decoder = GaussianConv(self.get_xyz.detach()).cuda()
            self.decoder.load_state_dict(self.decoder_state_dict)
            self.style_transfer = MulLayer().cuda()
            return

        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup_reconstruction(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        if hasattr(self, 'vacm_enabled') and self.vacm_enabled:
            # Truncate to first 9 SH coefficients (4th order SH) when VACM is enabled
            features_rest = features_rest[:, :9, :]  # [N, 9, 3]
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup_reconstruction(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        # Setup VACM if enabled
        self.vacm_enabled = False
        if hasattr(training_args, 'use_vacm') and training_args.use_vacm:
            self.vacm = ViewAdaptiveMLP().cuda()
            self.vacm_enabled = True
            # Initialize spatial extent for VACM displacement clamping
            self.spatial_extent = self.spatial_lr_scale if self.spatial_lr_scale > 0 else 1.0

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        # Verify optimizer structure at initialization
        self._verify_optimizer_structure(phase="reconstruction_init")
    
    def _verify_optimizer_structure(self, phase="unknown"):
        """Verify optimizer parameter group structure (one-time check per phase)"""
        print(f"\n{'='*60}")
        print(f"[OPTIMIZER CHECK] Phase: {phase}")
        print(f"{'='*60}")
        
        core_groups = ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation']
        
        for group in self.optimizer.param_groups:
            group_name = group.get('name', 'unnamed')
            num_params = len(group['params'])
            
            if num_params > 0:
                param = group['params'][0]
                shape = tuple(param.shape) if hasattr(param, 'shape') else 'N/A'
                device = str(param.device) if hasattr(param, 'device') else 'N/A'
                dtype = str(param.dtype) if hasattr(param, 'dtype') else 'N/A'
                requires_grad = param.requires_grad if hasattr(param, 'requires_grad') else 'N/A'
            else:
                shape = device = dtype = requires_grad = 'N/A'
            
            status = "✓" if num_params == 1 else "✗ WARN"
            print(f"  [{status}] {group_name:15s} | len={num_params} | shape={str(shape):20s} | "
                  f"device={device:10s} | dtype={dtype:15s} | grad={requires_grad}")
            
            # Alert if core group has wrong number of params
            if group_name in core_groups and num_params != 1:
                print(f"      ⚠️  WARNING: Core group '{group_name}' should have exactly 1 param, got {num_params}")
            
            # Alert if vacm_scale group exists and has wrong number
            if group_name == 'vacm_scale' and num_params != 1:
                print(f"      ⚠️  WARNING: vacm_scale group should have exactly 1 param, got {num_params}")
        
        print(f"{'='*60}\n")
        
    def training_setup_feature(self, training_args):
        # delete spherical harmonics because we don't need them for feature reconstruction
        del self._features_rest
        del self._features_dc
        
        # Freeze vacm_scale if exists (only train in reconstruction phase)
        if hasattr(self, 'vacm_scale') and self.vacm_scale is not None:
            self.vacm_scale.requires_grad_(False)
            print(f"[FEATURE SETUP] Frozen vacm_scale (requires_grad=False)")

        _vgg_features = torch.randn((self.get_xyz.shape[0], 32), device="cuda").requires_grad_(True)
        self._vgg_features = nn.Parameter(_vgg_features)
        self.feature_linear = LinearLayer(inChanel=32, out_dim=256).cuda()

        l = [
            {'params': [self._vgg_features], 'lr': 0.01, "name": "vgg_features"},
            {'params': self.feature_linear.parameters(), 'lr': 1e-3, "name": "feature_linear"}
        ]

        self.optimizer = torch.optim.Adam(l, eps=1e-15)
        
        # Verify optimizer structure
        self._verify_optimizer_structure(phase="feature_setup")

    def training_setup_decoder(self, training_args):
        # compute the final vgg features for each point
        self.final_vgg_features = self.feature_linear.forward_directly_on_point(self._vgg_features)

        # delete vgg features and linear layer because we have the perpoint features now
        del self._vgg_features
        del self.feature_linear

        # init gaussian conv
        self.decoder = GaussianConv(self.get_xyz.detach()).cuda()
       
        l = [
            {'params': self.decoder.parameters(), 'lr': 2e-3, "name": "decoder"}
        ]

        self.optimizer = torch.optim.Adam(l, eps=1e-15)

    def training_setup_style(self, training_args, decoder_path, photorealistic=False):
        # compute the final vgg features for each point
        self.final_vgg_features = self.feature_linear.forward_directly_on_point(self._vgg_features)
        self.final_vgg_features += torch.randn_like(self.final_vgg_features) # Hack: randomness improves stylization quality

        # delete vgg features and linear layer because we have the perpoint features now
        del self._vgg_features
        del self.feature_linear
        
        # Freeze vacm_scale if exists (only train in reconstruction phase)
        if hasattr(self, 'vacm_scale') and self.vacm_scale is not None:
            self.vacm_scale.requires_grad_(False)
            print(f"[STYLE SETUP] Frozen vacm_scale (requires_grad=False)")

        # init gaussian conv
        self.decoder = GaussianConv(self.get_xyz.detach(), K=(1 if photorealistic else 8)).cuda()
        if decoder_path:
            print('Init decoder from {}'.format(decoder_path))
            (_xyz,
            _scaling,
            _rotation,
            _opacity,
            final_vgg_features,
            decoder_state_dict) = torch.load(decoder_path)
            self.decoder.load_state_dict(decoder_state_dict)

        # init style transfer module
        self.style_transfer = MulLayer().cuda()

        l = [
            {'params': self.decoder.parameters(), 'lr': 1e-3, "name": "decoder"},
            {'params': self.style_transfer.parameters(), 'lr': 1e-3, "name": "style_transfer"}
        ]

        self.optimizer = torch.optim.Adam(l, eps=1e-15)
        
        # Verify optimizer structure
        self._verify_optimizer_structure(phase="style_setup")


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            group_name = group.get("name", None)
            
            # Skip non-Gaussian parameters (e.g., vacm_scale)
            # These don't have point-wise structure and shouldn't be pruned
            if group_name not in ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation']:
                continue
            
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group_name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group_name] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # Skip parameter groups that are not in tensors_dict (e.g., vacm_scale)
            group_name = group.get("name", None)
            if group_name not in tensors_dict:
                # Skip non-densify parameter groups
                continue
            
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group_name]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group_name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group_name] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_point_num=3e5):
        if self.get_xyz.shape[0] > max_point_num: return
        
        # Gate-keeping check: verify core parameter groups have exactly 1 param
        core_groups = ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation']
        for group in self.optimizer.param_groups:
            group_name = group.get('name', '')
            if group_name in core_groups:
                num_params = len(group['params'])
                if num_params != 1:
                    if not hasattr(self, '_densify_guard_warned'):
                        print(f"\n⚠️  [DENSIFY GUARD] Skipping densification: group '{group_name}' has {num_params} params (expected 1)")
                        print(f"    This prevents AssertionError in cat_tensors_to_optimizer.")
                        self._densify_guard_warned = True
                    return  # Skip this densification cycle

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def apply_vacm(self, iter, poses_vec, overlap_mask=None):
        """Apply view-adaptive position correction (non-inplace version)
        
        Args:
            iter: Current iteration number (for scheduling, handled externally now)
            poses_vec: Camera poses vectors [V, 12] float32
            overlap_mask: Optional boolean mask [P] for points to update
        """
        if not hasattr(self, 'vacm_enabled') or not self.vacm_enabled:
            return
        
        # Early return if overlap_mask is invalid
        if overlap_mask is None or not overlap_mask.any():
            return
        
        # Initialize vacm_scale if not exists (trainable parameter for regularization)
        if not hasattr(self, 'vacm_scale'):
            self.vacm_scale = nn.Parameter(torch.ones(1, device=self._xyz.device, dtype=torch.float32))
            
            # Add to optimizer as a separate parameter group (not in xyz group)
            # This avoids breaking cat_tensors_to_optimizer which expects 1 param per group
            vacm_scale_in_optimizer = False
            for group in self.optimizer.param_groups:
                if any(id(p) == id(self.vacm_scale) for p in group['params']):
                    vacm_scale_in_optimizer = True
                    break
            
            if not vacm_scale_in_optimizer:
                # Create new parameter group for vacm_scale with same lr as xyz
                xyz_lr = None
                for group in self.optimizer.param_groups:
                    if group["name"] == "xyz":
                        xyz_lr = group['lr']
                        break
                
                if xyz_lr is not None:
                    self.optimizer.add_param_group({
                        'params': [self.vacm_scale],
                        'lr': xyz_lr,
                        'name': 'vacm_scale'
                    })
                    
                    # Verify optimizer structure after adding vacm_scale
                    print(f"\n[VACM] Added vacm_scale parameter group to optimizer")
                    self._verify_optimizer_structure(phase="vacm_scale_added")
            
        with autocast():
            v_emb = self.vacm(poses_vec)  # [V, 3]
            v_mean = v_emb.mean(dim=0)  # [3]
            
            # Compute displacement with trainable scale and spatial extent normalization
            spatial_extent = getattr(self, "spatial_extent", 1.0)
            max_disp = 0.0015 * spatial_extent
            
            # Apply scale and clamp to prevent excessive displacement
            delta = (self.vacm_scale * 0.01 * v_mean).view(1, 3)  # [1, 3]
            delta = delta.clamp(min=-max_disp, max=max_disp)
            
        # Apply displacement with mask using NON-INPLACE update
        # Create new tensor to avoid version number conflicts
        with torch.no_grad():
            mask = overlap_mask.view(-1, 1).to(self._xyz.dtype)
            delta = delta.to(self._xyz.dtype)
            
            # Create new tensor (non-inplace)
            new_xyz = self._xyz + mask * delta
            
            # Debug logging (only once to avoid spam)
            if not hasattr(self, '_vacm_first_apply_logged'):
                print(f"[DEBUG] VACM applied: delta.mean()={delta.mean().item():.6f}, delta.abs().max()={delta.abs().max().item():.6f}")
                self._vacm_first_apply_logged = True
            
            # Update optimizer state with new tensor
            for group in self.optimizer.param_groups:
                if group["name"] == "xyz":
                    stored_state = self.optimizer.state.get(group['params'][0], None)
                    
                    if stored_state is not None:
                        # Preserve optimizer state (momentum, etc.) - keep the same state
                        # No modification needed as the state tensors are still valid
                        pass
                    
                    # Replace parameter with new tensor
                    del self.optimizer.state[group['params'][0]]
                    group['params'][0] = nn.Parameter(new_xyz.requires_grad_(True))
                    
                    # Restore optimizer state
                    if stored_state is not None:
                        self.optimizer.state[group['params'][0]] = stored_state
                    
                    # Update model parameter
                    self._xyz = group['params'][0]
                    break
                
        # Periodically clear GPU cache (every 1000 iterations)
        if iter % 1000 == 0:
            torch.cuda.empty_cache()

    def local_align(self, content_feats, style_feats, K=5):
        """对每个内容特征，找到风格特征中最近的K个邻居，做加权平均（可扩展为更复杂的局部对齐）。
        
        Args:
            content_feats: [N, C]
            style_feats: [M, C]
            
        Returns:
            [N, C]
        """
        N, C = content_feats.shape
        aligned_feats = []
        for i in range(N):
            c = content_feats[i]  # [C]
            dists = torch.norm(style_feats - c[None, :], dim=1)  # [M]
            idx = torch.topk(dists, K, largest=False).indices  # 最近K个
            local_style = style_feats[idx]  # [K, C]
            aligned = local_style.mean(dim=0)
            aligned_feats.append(aligned)
        return torch.stack(aligned_feats, dim=0)  # [N, C]

    def style_transfer(self, content_features, style_features, content_mask=None, style_mask=None):
        """实现局部对齐的风格迁移
        
        Args:
            content_features: 内容特征 [B, C, H, W]
            style_features: 风格特征 [B, C, H, W]
            content_mask: 内容图像的分割掩码 [B, 1, H, W]
            style_mask: 风格图像的分割掩码 [B, 1, H, W]
            
        Returns:
            stylized_features: 风格化后的特征 [B, C, H, W]
        """
        # 1. 计算局部特征统计量
        content_mean, content_std = self.compute_local_statistics(content_features, content_mask)
        style_mean, style_std = self.compute_local_statistics(style_features, style_mask)
        
        # 2. 计算局部对齐矩阵
        alignment_matrix = self.compute_local_alignment(content_features, style_features, content_mask, style_mask)
        
        # 3. 应用局部风格迁移
        stylized_features = self.apply_local_style_transfer(
            content_features, 
            content_mean, 
            content_std,
            style_mean, 
            style_std,
            alignment_matrix
        )
        
        return stylized_features
        
    def compute_local_statistics(self, features, mask=None):
        """计算局部特征统计量
        
        Args:
            features: 特征图 [B, C, H, W]
            mask: 分割掩码 [B, 1, H, W]
            
        Returns:
            mean: 局部均值 [B, C, H, W]
            std: 局部标准差 [B, C, H, W]
        """
        if mask is not None:
            features = features * mask
            
        # 使用卷积计算局部统计量
        kernel_size = 3
        padding = kernel_size // 2
        
        # 计算局部均值
        mean = F.avg_pool2d(features, kernel_size, stride=1, padding=padding)
        
        # 计算局部方差
        var = F.avg_pool2d(features ** 2, kernel_size, stride=1, padding=padding) - mean ** 2
        std = torch.sqrt(var + 1e-6)
        
        return mean, std
        
    def compute_local_alignment(self, content_features, style_features, content_mask=None, style_mask=None):
        """计算局部对齐矩阵
        
        Args:
            content_features: 内容特征 [B, C, H, W]
            style_features: 风格特征 [B, C, H, W]
            content_mask: 内容图像的分割掩码 [B, 1, H, W]
            style_mask: 风格图像的分割掩码 [B, 1, H, W]
            
        Returns:
            alignment_matrix: 局部对齐矩阵 [B, H*W, H*W]
        """
        B, C, H, W = content_features.shape
        
        # 将特征图展平为向量
        content_flat = content_features.view(B, C, -1)  # [B, C, H*W]
        style_flat = style_features.view(B, C, -1)      # [B, C, H*W]
        
        # 计算余弦相似度
        content_norm = F.normalize(content_flat, dim=1)
        style_norm = F.normalize(style_flat, dim=1)
        similarity = torch.bmm(content_norm.transpose(1, 2), style_norm)  # [B, H*W, H*W]
        
        # 应用掩码
        if content_mask is not None and style_mask is not None:
            content_mask_flat = content_mask.view(B, 1, -1)  # [B, 1, H*W]
            style_mask_flat = style_mask.view(B, 1, -1)      # [B, 1, H*W]
            mask = torch.bmm(content_mask_flat.transpose(1, 2), style_mask_flat)  # [B, H*W, H*W]
            similarity = similarity * mask
            
        # 归一化对齐矩阵
        alignment_matrix = F.softmax(similarity / 0.1, dim=-1)  # 使用温度参数0.1
        
        return alignment_matrix
        
    def apply_local_style_transfer(self, content_features, content_mean, content_std, 
                                 style_mean, style_std, alignment_matrix):
        """应用局部风格迁移
        
        Args:
            content_features: 内容特征 [B, C, H, W]
            content_mean: 内容特征局部均值 [B, C, H, W]
            content_std: 内容特征局部标准差 [B, C, H, W]
            style_mean: 风格特征局部均值 [B, C, H, W]
            style_std: 风格特征局部标准差 [B, C, H, W]
            alignment_matrix: 局部对齐矩阵 [B, H*W, H*W]
            
        Returns:
            stylized_features: 风格化后的特征 [B, C, H, W]
        """
        B, C, H, W = content_features.shape
        
        # 将特征图展平为向量
        content_flat = content_features.view(B, C, -1)  # [B, C, H*W]
        content_mean_flat = content_mean.view(B, C, -1)  # [B, C, H*W]
        content_std_flat = content_std.view(B, C, -1)    # [B, C, H*W]
        style_mean_flat = style_mean.view(B, C, -1)      # [B, C, H*W]
        style_std_flat = style_std.view(B, C, -1)        # [B, C, H*W]
        
        # 计算局部风格迁移
        # 1. 使用对齐矩阵对风格统计量进行加权
        aligned_style_mean = torch.bmm(style_mean_flat, alignment_matrix.transpose(1, 2))  # [B, C, H*W]
        aligned_style_std = torch.bmm(style_std_flat, alignment_matrix.transpose(1, 2))    # [B, C, H*W]
        
        # 2. 应用局部风格迁移
        stylized_flat = aligned_style_std * (content_flat - content_mean_flat) / (content_std_flat + 1e-6) + aligned_style_mean
        
        # 3. 重塑回原始形状
        stylized_features = stylized_flat.view(B, C, H, W)
        
        return stylized_features