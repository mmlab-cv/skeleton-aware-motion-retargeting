from einops import rearrange, repeat
import pytorch_lightning as pl
import wandb
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from transformer import Transformer
import pickle as pkl
import sys
import bpy
from geometry import *
from dataset.bvh_writer import BvhWriter
import sys 
from FKinematics import ForwardKinematics
import os
from options import Options
from load_bvh import load_bvh
from scene import make_scene, add_material_for_character, add_rendering_parameters
from dataset.bvh_parser import BvhData
from dataset.skeleton import build_bone_topology, _concat_together, de_normalize, convert_to_bvh_write_format
from utils.BVH_FILE import read_bvh, save_bvh
import time


class DATRetarget(LightningModule):
    def __init__(
        self,
        FLAGS,
        datamodule,
        **kwargs,
    ):
        super().__init__()
        self.FLAGS = FLAGS

        # # Original
        # self.encoder = nn.Sequential(
        #     nn.Linear(4, 128),
        #     nn.LeakyReLU(),
        #     Transformer(128, 6, 16, 64, 128, 0.),
        #     nn.Linear(128, 4)
        # )

        # Tiny
        # dim, depth, heads, dim_head, mlp_dim, dropout
        
        # if self.FLAGS.mode == 'demo':
        #     self.dataset_mixamo = datamodule.demo_dataset
            
        if self.FLAGS.mode == 'train':
            self.dataset_mixamo = datamodule.train_dataset
            self.dataset_mixamo_val = datamodule.val_dataset
            

        self.mapping = [0,1,2,3,4,5,6,7,8,9,11,12,13,15,16,17,18,20,21,22,23]

        self.mapping_human = list(range(21))
        self.mapping_dog = [0,1,2,4,5,6,8,9,10,11,13,14,15,16,17,18,19,20,21,22]

        # [0,13,14,15,16,17,18,1,2,3,4,5,6,7,8,9,10,11,12,19,20]
        
        edges_m = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(12,13),(13,14),(12,15),(15,16),(16,17),(17,18),(18,19),(12,20),(20,21),(21,22),(22,23),(23,24)]
        edges = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(12,13),(11,14),(14,15),(15,16),(16,17),(11,18),(18,19),(19,20),(20,21)]
        edges_cmu = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 6), (6, 7), (7, 8), (8, 9), (9, 10), (0, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (20, 23), (13, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (27, 30)]
        edges_human = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(12,13),(9,14),(14,15),(15,16),(16,17),(9,18),(18,19),(19,20),(20,21)]
        edges_dog = [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(7,8),(1,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(0,16),(16,17),(17,18),(0,19),(19,20)]

        self.fk_transform_m = ForwardKinematics(edges=edges_m)
        self.fk_transform = ForwardKinematics(edges=edges)
        self.fk_transform_cmu = ForwardKinematics(edges=edges_cmu)
        self.fk_human = ForwardKinematics(edges=edges_human)
        self.fk_dog = ForwardKinematics(edges=edges_dog)
        
        tiny = [192, 12, 3, 64, 768, 0.]
        small = [384, 12, 6, 64, 1536, 0.]
        base = [768, 12, 12, 64, 3072, 0.]
        large = [1024, 24, 16, 64, 4096, 0.]

        
        # dim, depth, heads, dim_head, mlp_dim, dropout
        tiny_4Xheads = [192, 12, 12, 16, 768, 0.]

        C = tiny_4Xheads


        self.encoder = nn.Sequential(
            nn.Linear(4, C[0]),
            nn.LeakyReLU()
        )

        self.transformer = Transformer(C[0], C[1], C[2], C[3], C[4], C[5])  # dim_head = dim/heads

        self.decoder = nn.Linear(C[0], 4)


        self.pos_enc_spatial = nn.Parameter(torch.randn(1,self.FLAGS.n_joints-1,C[0]),requires_grad=True)
        self.pos_enc_temporal = nn.Parameter(torch.randn(1,self.FLAGS.window_size,C[0]),requires_grad=True)

        if self.FLAGS.use_cls:
            self.pos_enc_cls = nn.Parameter(torch.randn(1,1,C[0]),requires_grad=True)

        self.mse_loss = nn.MSELoss()


    def forward(self, z, cls = None, mapping=None):
        if self.FLAGS.mode == 'train':
            if self.FLAGS.dataset == "HumanDog":
                if len(mapping) == 21:
                    rand_index = torch.tensor(np.sort(np.random.choice(mapping, self.FLAGS.masked_joints, replace=False)))
                    total_rand_index = np.append(rand_index, [12,3,7,21,22])
                    # total_rand_index = np.array([12,2,3,6,7,21,22,23,24]) 
                    z[:,:,total_rand_index,:] = torch.zeros(z[:,:,total_rand_index,:].size(), device="cuda")
                else:
                    rand_index = torch.tensor(np.sort(np.random.choice(mapping, self.FLAGS.masked_joints, replace=False)))
                    total_rand_index = np.append(rand_index,  [12,3,7,21,22])
                    # total_rand_index = np.array([12,2,3,6,7,21,22,23,24]) 
                    z[:,:,total_rand_index,:] = torch.zeros(z[:,:,total_rand_index,:].size(), device="cuda")
            else:
                if mapping == None:
                    total_rand_index = torch.tensor(np.sort(np.random.choice(range(self.FLAGS.n_joints-1), self.FLAGS.masked_joints, replace=False)))
                    z[:,:,total_rand_index,:] = torch.zeros(z[:,:,total_rand_index,:].size(), device="cuda")
                else:
                    rand_index = torch.tensor(np.sort(np.random.choice(mapping, self.FLAGS.masked_joints, replace=False)))
                    total_rand_index = np.append(rand_index, [])
                    z[:,:,total_rand_index,:] = torch.zeros(z[:,:,total_rand_index,:].size(), device="cuda")
                    # z[:,:,total_rand_index,:] = torch.rand_like(z[:,:,total_rand_index,:])
                    # z[:,:,total_rand_index,:] = z[:,:,total_rand_index,:] + torch.rand_like(z[:,:,total_rand_index,:]) * 0.1
        else:
            # total_rand_index = np.array([12,3,7,21,22]) #13,3,4,8
            # z[:,:,total_rand_index,:] = torch.zeros(z[:,:,total_rand_index,:].size(), device="cuda")
            total_rand_index = torch.tensor(np.sort(np.random.choice(range(self.FLAGS.n_joints-1), self.FLAGS.masked_joints, replace=False)))
            # total_rand_index = torch.tensor(np.array([3,13,15,21,29,19,10]))
            z[:,:,total_rand_index,:] = torch.zeros(z[:,:,total_rand_index,:].size(), device="cuda")


        # keep_index = []
        # for i in range(self.FLAGS.n_joints-1):
        #     if i not in total_rand_index:
        #         keep_index.append(i)
                
        keep_index = [item for item in np.array(range(self.FLAGS.n_joints-1)) if item not in total_rand_index]

        pos_enc_spatial = repeat(self.pos_enc_spatial, 'b j q -> b (t j) q', t = self.FLAGS.window_size)
        pos_enc_temporal = repeat(self.pos_enc_temporal, 'b t q -> b (t j) q', j = self.FLAGS.n_joints-1)
        pos_enc = pos_enc_spatial + pos_enc_temporal
        
        z = rearrange(z, 'b w j q -> b (w j) q')

        z = self.encoder(z)

        if cls is not None:
            pos_enc = torch.cat((pos_enc,self.pos_enc_cls),dim=1)
            cls = cls.unsqueeze(1).unsqueeze(2)
            cls_token = repeat(cls, 'b 1 1 -> b 1 d', d = 192)
            z = torch.cat((z,cls_token),dim=1)

        z += pos_enc

        z = self.transformer(z)

        z = self.decoder(z)

        if cls is not None:
            return z[:,:-1,:], total_rand_index, keep_index
        else:
            return z, total_rand_index, keep_index

    def training_step(self, batch, batch_idx):
        
        pos = batch['msk_rot'][:, :, 0:1, 0:3]

        if self.FLAGS.use_cls:
            char_id = batch['id'].float()
        else:
            char_id = None

        index = batch['index'][0].item()
        
        N = pos.shape[0]

        joint = batch['msk_rot'][:, :, 1:, :]

        joint_z = rearrange(joint ,'b w j q -> b w j q')     

        z = joint.clone()

        z = rearrange(z,'b w j q -> b w j q')
        
        self.edges = self.dataset_mixamo.edges[int(index)]
        self.offset = self.dataset_mixamo.offsets[int(index)]
        
        if self.FLAGS.dataset == 'MIXAMO':
            if len(self.edges) != 24:
                mapping = self.mapping
                fk = self.fk_transform
            else:
                mapping = None
                fk = self.fk_transform_m
        elif self.FLAGS.dataset == 'HumanDog':
            if len(self.edges) == 21:
                mapping = self.mapping_human
                fk = self.fk_human
            else:
                mapping = self.mapping_dog
                fk = self.fk_dog
        else:
            mapping = None
            fk = self.fk_transform_cmu

        self.offsets = [each.permute(1, 0) for each in [self.offset]]
        # GROUND TRUTH
        joints_rot4d = rearrange(joint, 'b w j q -> (b w) j q')
        pos_root = rearrange(pos, 'b w j q -> (b w) j q')

        concat_tensor_gt = _concat_together(joints_rot4d, pos_root)
        concat_tensor_gt = rearrange(concat_tensor_gt, '(b w) j q -> b q j w', b = N, w = self.FLAGS.window_size, j = joints_rot4d.shape[1]+1,q=4)

        denorm_gt_rot, denorm_gt_pos = de_normalize(concat_tensor_gt)
        
        if mapping != None:
            denorm_gt_rot = denorm_gt_rot[:,:,mapping,:]
        
        gt_pos = fk.forward_from_raw(denorm_rot=denorm_gt_rot,
                                                         de_norm_pos=denorm_gt_pos,
                                                         offset=self.offsets[0].permute(1, 0).unsqueeze(0).repeat(denorm_gt_rot.shape[0], 1, 1))

        gt_pos = fk.from_local_to_world(gt_pos / 236.57)
        


        # PREDICT
        
        generated, mask_index, no_mask_index = self(z,cls = char_id, mapping=mapping)
        
        generated = rearrange(generated, 'b (w j) q -> b w j q', w = self.FLAGS.window_size, j =self.FLAGS.n_joints-1, q=4)

        total_generated = torch.zeros(joint_z.shape).bfloat16().cuda()
        total_generated = total_generated.clone()
        total_generated[:,:,mask_index,:] = generated[:,:,mask_index,:]
        total_generated[:,:,no_mask_index,:] = joint_z[:,:,no_mask_index,:].bfloat16()

        total_rot_pred_4d = rearrange(total_generated, 'b w j q -> (b w) j q ')


        # PREDICTION
        concat_tensor_pred = _concat_together(total_rot_pred_4d, pos_root)
        concat_tensor_pred = rearrange(concat_tensor_pred, '(b w) j q -> b q j w', b = N, w = self.FLAGS.window_size, j = self.FLAGS.n_joints,q=4)
        denorm_pred_rot, denorm_pred_pos = de_normalize(concat_tensor_pred)
        # [B, 4, J-1, frame]
        
        if mapping != None:
            denorm_pred_rot = denorm_pred_rot[:,:,mapping,:]

        pred_pos = fk.forward_from_raw(denorm_rot=denorm_pred_rot,
                                                         de_norm_pos=denorm_pred_pos,
                                                         offset=self.offsets[0].permute(1, 0).unsqueeze(0).repeat(denorm_pred_rot.shape[0], 1, 1))

        pred_pos = fk.from_local_to_world(pred_pos / 236.57)
        
        pred_positions = pred_pos
        pred_positions = rearrange(pred_positions,'b w j q -> b (q w j)')
        gt_positions = gt_pos
        gt_positions = rearrange(gt_positions,'b w j q -> b (q w j)')


        loss = self.mse_loss(pred_positions,gt_positions)
        self.log("Training/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        
        pos_val = batch['msk_rot'][:, :, 0:1, 0:3]

        if self.FLAGS.use_cls:
            char_id = batch['id'].float()
        else:
            char_id = None

        index = batch['index'][0].item()
        
        N = pos_val.shape[0]

        joint_val = batch['msk_rot'][:, :, 1:, :]

        joint_z = rearrange(joint_val ,'b w j q -> b w j q')     

        z = joint_val.clone()

        z = rearrange(z,'b w j q -> b w j q')
        
        self.edges = self.dataset_mixamo_val.val_edges[int(index)]
        self.offset = self.dataset_mixamo_val.val_offsets[int(index)]
        
        if self.FLAGS.dataset == 'MIXAMO':
            if len(self.edges) != 24:
                mapping = self.mapping
                fk = self.fk_transform
            else:
                mapping = None
                fk = self.fk_transform_m
        elif self.FLAGS.dataset == 'HumanDog':
            if len(self.edges) == 21:
                mapping = self.mapping_human
                fk = self.fk_human
            else:
                mapping = self.mapping_dog
                fk = self.fk_dog
        else:
            mapping = None
            fk = self.fk_transform_cmu

        self.offsets = [each.permute(1, 0) for each in [self.offset]]
        # GROUND TRUTH
        joints_rot4d = rearrange(joint_val, 'b w j q -> (b w) j q')
        pos_root = rearrange(pos_val, 'b w j q -> (b w) j q')
        concat_tensor_gt = self.dataset_mixamo._concat_together(joints_rot4d, pos_root)
        concat_tensor_gt = rearrange(concat_tensor_gt, '(b w) j q -> b q j w', b = N, w = self.FLAGS.window_size, j = joints_rot4d.shape[1]+1,q=4)

        denorm_gt_rot, denorm_gt_pos = self.dataset_mixamo.de_normalize(concat_tensor_gt)
        
        if mapping != None:
            denorm_gt_rot = denorm_gt_rot[:,:,mapping,:]

        gt_pos = fk.forward_from_raw(denorm_rot=denorm_gt_rot,
                                                         de_norm_pos=denorm_gt_pos,
                                                         offset=self.offset.unsqueeze(0).repeat(denorm_gt_rot.shape[0], 1, 1))

        gt_pos = fk.from_local_to_world(gt_pos / 236.57)
        

        # PREDICT
        
        generated, mask_index, no_mask_index = self(z,cls = char_id,mapping=mapping)
        
        generated = rearrange(generated, 'b (w j) q -> b w j q', w = self.FLAGS.window_size, j =self.FLAGS.n_joints-1, q=4)

        total_generated = torch.zeros(joint_z.shape).bfloat16().cuda()
        total_generated[:,:,mask_index,:] = generated[:,:,mask_index,:]
        total_generated[:,:,no_mask_index,:] = joint_z[:,:,no_mask_index,:].bfloat16()

        total_rot_pred_4d = rearrange(total_generated, 'b w j q -> (b w) j q ')


        # PREDICTION
        concat_tensor_pred = self.dataset_mixamo._concat_together(total_rot_pred_4d, pos_root)
        concat_tensor_pred = rearrange(concat_tensor_pred, '(b w) j q -> b q j w', b = N, w = self.FLAGS.window_size, j = self.FLAGS.n_joints,q=4)
        denorm_pred_rot, denorm_pred_pos = self.dataset_mixamo.de_normalize(concat_tensor_pred)
        # [B, 4, J-1, frame]
        if mapping != None:
            denorm_pred_rot = denorm_pred_rot[:,:,mapping,:]
        
        pred_pos = fk.forward_from_raw(denorm_rot=denorm_pred_rot,
                                                         de_norm_pos=denorm_pred_pos,
                                                         offset=self.offsets[0].permute(1, 0).unsqueeze(0).repeat(denorm_pred_rot.shape[0], 1, 1))

        pred_pos = fk.from_local_to_world(pred_pos / 236.57)
        
        
        pred_positions = pred_pos
        pred_positions = rearrange(pred_positions,'b w j q -> b (q w j)')
        gt_positions = gt_pos
        gt_positions = rearrange(gt_positions,'b w j q -> b (q w j)')


        loss = self.mse_loss(pred_positions,gt_positions)
        self.log("Validation/loss", loss, prog_bar=True)

        self.total_generated = total_generated
        self.joint_val = joint_val
        self.pos_val = pos_val
        self.mapping_val = mapping

        return loss
    
    

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.FLAGS.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.95)

        return [opt], [scheduler]