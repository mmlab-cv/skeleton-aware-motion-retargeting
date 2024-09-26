import torch
from model import DATRetarget
from dataset.dataloader_mixamo import MixamoDataModule
from dataset.dataloader_mixamo import MixamoDataset
from absl import app
from absl import flags
import data_flags
from dataset.bvh_parser import BvhData
from utils.IK import fix_foot_contact, get_character_height
from dataset import get_bvh_file_names
from absl import app
from absl import flags
from einops import rearrange, repeat
import sys
import os
from dataset.bvh_writer import BvhWriter
import utils.BVH as BVH
from data_flags import FLAGS
import utils.Animation_deep as Animation
import numpy as np
import wandb
import natsort
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def main(argv):
    dm = MixamoDataModule(FLAGS)
    model = DATRetarget.load_from_checkpoint(FLAGS.load_checkpoint, FLAGS = FLAGS, mixamo_datamodule = dm).cuda()
    model.eval()
    # characters_input = ['Aj', 'Malcolm_m','BigVegas', 'Kaya', 'SportyGranny', 'Remy_m', 
    #                                 'Maria_m', 'Knight_m','Liam_m', 'Parasite', 
    #                                 'Michelle_m', 'LolaB_m','Pumpkinhulk_m', 'Ortiz_m', 'Paladin_m', 
    #                                 'James_m', 'Joe_m','Olivia_m', 'Yaku_m', 'Timmy_m', 'Racer_m', 'Abe_m']
    # characters_target = ["Aj","BigVegas","Goblin_m","Kaya","Mousey_m","Mremireh_m","SportyGranny","Vampire_m"] #"Mutant",
    
    characters_input = ["Aj","BigVegas","Goblin","Kaya","Mousey","Warrok","PeasantMan"]
    characters_target = ["Ortiz","SportyGranny","Aj","BigVegas","Goblin","Kaya","Mousey","Warrok","PeasantMan","XBot","Man","CastleGuard"]
    
    mapping_init = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,16,17,18,19,21,22,23,24]
    
    dataset = dm.demo_dataset
    
    all_err = []
    
    unique_combinations = []
    
    for i in range(len(characters_input)):
        for j in range(len(characters_target)):
            unique_combinations.append((characters_input[i], characters_target[j]))
    
    for combination in tqdm(unique_combinations):
        char_input, char_target = combination[0], combination[1]
        
        char_input = 'XBot'
        char_target = 'BigVegas'
        
        if FLAGS.retargeting == 'homeo':
            if '_m' not in char_input and '_m' not in char_target:
                continue
            elif '_m' in char_input and '_m' in char_target:
                continue
        
        # LOAD INPUT CHAR DATA
        if '_m' in char_input:
            char_in = char_input.split('_m')[0]
        else:
            char_in = char_input
  
        std_bvh_data_in = BvhData(char_input, motion_file_name=char_in+'.bvh', FLAGS=FLAGS)
        edges_in = std_bvh_data_in.edges
        offset_in = torch.from_numpy(std_bvh_data_in.offset).float()
        
        bvh_file_list_in = get_bvh_file_names(FLAGS.mode, character_name=char_input)
        
        # LOAD OUT CHAR DATA
        if '_m' in char_target:
            char_out = char_target.split('_m')[0]
        else:
            char_out = char_target

        std_bvh_data_out = BvhData(char_target, motion_file_name=char_out+'.bvh', FLAGS=FLAGS)
        edges_out = std_bvh_data_out.edges
        offset_out = torch.from_numpy(std_bvh_data_out.offset).float()
        
        bvh_file_list_out = get_bvh_file_names(FLAGS.mode, character_name=char_out)
        
        bvh_file_list = intersection(bvh_file_list_in, bvh_file_list_out)
        
        
        for j in tqdm(range(len(bvh_file_list))):
            
            # bvh_name = 'Short Right Side Step.bvh'
            bvh_name = 'GanGnam Style.bvh'
            
            ## Process BVH Data Input
            bvh_name = bvh_file_list[j]+'.bvh'
            bvh_data = BvhData(char_input, motion_file_name=bvh_name,FLAGS = FLAGS)
            names_in = std_bvh_data_in.names
            rotation = bvh_data.get_rotation()
            root_position = bvh_data.get_root_position()
            rotation, root_position = dataset._normalize(rotation, root_position)
            concat_tensor = dataset._concat_together(rotation, root_position)
            rotation_super = torch.zeros(size=(concat_tensor.shape[0],25,4))
            if concat_tensor.shape[1] == 25:
                rotation_super = concat_tensor
            else:
                rotation_super[:,mapping_init,:] = concat_tensor
            final_output_rot_in = dataset._to_format_tensor(rotation_super)
            final_output_rot_in = dataset._slice_to_equal_frame_len(final_output_rot_in)
            final_output_rot_in = rearrange(final_output_rot_in, 'b q j w -> b w j q')
            
            ## Process BVH Data Out
            bvh_name = bvh_file_list[j]+'.bvh'
            bvh_data = BvhData(char_target, motion_file_name=bvh_name,FLAGS = FLAGS)
            rotation = bvh_data.get_rotation()
            root_position = bvh_data.get_root_position()
            rotation, root_position = dataset._normalize(rotation, root_position)
            concat_tensor = dataset._concat_together(rotation, root_position)
            rotation_super = torch.zeros(size=(concat_tensor.shape[0],25,4))
            if concat_tensor.shape[1] == 25:
                rotation_super = concat_tensor
            else:
                rotation_super[:,mapping_init,:] = concat_tensor
            final_output_rot_out = dataset._to_format_tensor(rotation_super)
            final_output_rot_out = dataset._slice_to_equal_frame_len(final_output_rot_out)
            final_output_rot_out = rearrange(final_output_rot_out, 'b q j w -> b w j q')
            
            ## Clone for GT
            joint_rotation = final_output_rot_in[:, :, 1:, :]
            joint_position = final_output_rot_in[:, :, 0:1, 0:3]
            
            joint_rotation_gt = final_output_rot_out[:, :, 1:, :]
            joint_position_gt = final_output_rot_out[:, :, 0:1, 0:3]
            
            if joint_rotation.shape[0] != joint_rotation_gt.shape[0]:
                continue
            
            N = joint_position_gt.shape[0]
            
            ## Predict
            if FLAGS.retargeting == 'homeo':
                if '_m' in char_input and '_m' not in char_target:
                    mapping = [0,1,2,3,4,5,6,7,8,9,11,12,13,15,16,17,18,20,21,22,23]
                elif '_m' not in char_input and '_m' in char_target:
                    mapping = None
                    mapping_in = [0,1,2,3,4,5,6,7,8,9,11,12,13,15,16,17,18,20,21,22,23]
            else:
                mapping = [0,1,2,3,4,5,6,7,8,9,11,12,13,15,16,17,18,20,21,22,23]
                mapping_gt = mapping
            
            
            joint_z = joint_rotation.clone().cuda()
            generated, mask_index, no_mask_index = model(joint_z,mapping=mapping)
            generated = rearrange(generated, 'b (w j) q -> b w j q', w = FLAGS.window_size, j =FLAGS.n_joints-1, q=4)
            total_generated = torch.empty(joint_rotation.shape).cuda()
            total_generated = total_generated.clone()
            total_generated[:,:,mask_index,:] = generated[:,:,mask_index,:]
            total_generated[:,:,no_mask_index,:] = joint_rotation[:,:,no_mask_index,:].cuda()
            total_rot_pred_4d = rearrange(total_generated, 'b w j q -> (b w) j q ')
            
            ## Save Result
            
            if FLAGS.retargeting == 'homeo':
                if '_m' in char_input and '_m' not in char_target:
                    joints_pred_4d = rearrange(total_generated[:,:,mapping,:], 'n w j q -> (n w) j q ')
                    names = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot',
                    'RightToeBase', 'Spine', 'Spine1','Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm',
                    'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
                    joints_gt = rearrange(joint_rotation_gt[:,:,mapping,:], 'n w j q -> (n w) j q')
                    pos_gt = rearrange(joint_position_gt, 'n w j q -> (n w) j q')
                    joints_in = rearrange(joint_rotation, 'n w j q -> (n w) j q')
                    pos_in = rearrange(joint_position, 'n w j q -> (n w) j q')
                elif '_m' not in char_input and '_m' in char_target:
                    joints_pred_4d = rearrange(total_generated, 'n w j q -> (n w) j q ')
                    names = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot',
                'RightToeBase', 'Spine', 'Spine1', 'Spine1_split','Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftShoulder_split', 'LeftArm',
                'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightShoulder_split','RightArm', 'RightForeArm', 'RightHand']
                    joints_gt = rearrange(joint_rotation_gt, 'n w j q -> (n w) j q')
                    pos_gt = rearrange(joint_position_gt, 'n w j q -> (n w) j q')
                    joints_in = rearrange(joint_rotation[:,:,mapping_in,:], 'n w j q -> (n w) j q')
                    pos_in = rearrange(joint_position, 'n w j q -> (n w) j q')     
            else:
                joints_pred_4d = rearrange(total_generated[:,:,mapping,:], 'n w j q -> (n w) j q ')
                joints_gt = rearrange(joint_rotation_gt[:,:,mapping_gt,:], 'n w j q -> (n w) j q')
                pos_gt = rearrange(joint_position_gt, 'n w j q -> (n w) j q')
                joints_in = rearrange(joint_rotation[:,:,mapping_gt,:], 'n w j q -> (n w) j q')
                pos_in = rearrange(joint_position, 'n w j q -> (n w) j q')
                names = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot',
            'RightToeBase', 'Spine', 'Spine1','Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm',
            'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
                
            
            # Define the writer
            bvh_writer_in = BvhWriter(edges=edges_in, names=names_in,offset=offset_in)
            bvh_writer_out = BvhWriter(edges=edges_out, names=names,offset=offset_out)
            
            # INPUT ANIMATION
            concat_tensor_gt = dataset._concat_together(joints_in, pos_in)
            concat_tensor_gt = rearrange(concat_tensor_gt, '(b w) j q -> b q j w', b = total_generated.shape[0], w = FLAGS.window_size, j = joints_in.shape[1]+1,q=4)          
            bvh_write_tensor_gt = dataset.convert_to_bvh_write_format(concat_tensor_gt)
            bvh_writer_in.write_raw(bvh_write_tensor_gt, 'quaternion', os.path.join(FLAGS.save_path_eval,'gt_in.bvh'))
            
            # GROUND TRUTH
            
            concat_tensor_gt = dataset._concat_together(joints_gt, pos_gt)
            concat_tensor_gt = rearrange(concat_tensor_gt, '(b w) j q -> b q j w', b = total_generated.shape[0], w = FLAGS.window_size, j = joints_gt.shape[1]+1,q=4)          
            bvh_write_tensor_gt = dataset.convert_to_bvh_write_format(concat_tensor_gt)
            bvh_writer_out.write_raw(bvh_write_tensor_gt, 'quaternion', os.path.join(FLAGS.save_path_eval,'gt_quat.bvh'))
            
            # PREDICTION
            concat_tensor_pred = dataset._concat_together(joints_pred_4d, pos_gt)
            concat_tensor_pred = rearrange(concat_tensor_pred, '(b w) j q -> b q j w', b = total_generated.shape[0], w = FLAGS.window_size, j = joints_pred_4d.shape[1]+1,q=4)
            bvh_write_tensor_pred = dataset.convert_to_bvh_write_format(concat_tensor_pred)
            bvh_writer_out.write_raw(bvh_write_tensor_pred, 'quaternion', os.path.join(FLAGS.save_path_eval,'pred_quat.bvh'))
            
            ### GET ERROR ###
            anim, names, ftime = BVH.load(os.path.join(FLAGS.save_path_eval,'pred_quat.bvh'))
            index = []
            for k, name in enumerate(names):
                if 'virtual' in name:
                    continue
                index.append(k)
            
            anim_ref, _, _ = BVH.load(os.path.join(FLAGS.save_path_eval,'gt_quat.bvh'))
            ref_height = get_character_height('/media/mmlab/Volume2/MixamoBVH/{}/{}.bvh'.format(char_out, char_out))
            anim_ref.rotations[:,0] = anim.rotations[:,0]
            anim_ref.positions[:,0] = anim.positions[:,0]
            pos = Animation.positions_global(anim)  # [T, J, 3]
            pos_ref = Animation.positions_global(anim_ref)
            final_path = FLAGS.save_final_anim + '/' + char_input + '_to_' + char_target
            if pos.shape[0] == pos_ref.shape[0]:
                if not os.path.exists(final_path):
                    os.mkdir(final_path)
                BVH.save(os.path.join(final_path,bvh_name), anim, names, ftime)
                pos = pos[:, index, :]
                pos_ref = pos_ref[:, index, :]
                err = 1. / ref_height * ((pos_ref - pos) ** 2).sum(axis=-1)
                all_err.append(err)
    total_error = np.concatenate(all_err).mean()
    print('MSE: ',total_error)
    
    
if __name__ == '__main__':
    app.run(main)