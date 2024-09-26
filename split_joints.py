"""
This script splits three joints as we describe in the paper.
It automatically detects all the dirs in "./datasets/Mixamo" and
finds these can be split then create a new dir with an extra _m
as suffix to store the split files in the new dir.
"""

import sys
import os
import numpy as np
from tqdm import tqdm
from utils.bvh_parser import BVH_file
import utils.BVH_mod as BVH


def split_joint(file_name, save_file=None):
    if save_file is None:
        save_file = file_name
    target_joints = ['Spine1', 'LeftShoulder', 'RightShoulder']
    target_idx = [-1] * len(target_joints)
    anim, names, ftime = BVH.load(file_name)

    n_joint = len(anim.parents)

    for i, name in enumerate(names):
        if ':' in name:
            name = name[name.find(':') + 1:]
            names[i] = name

        for j, joint in enumerate(target_joints):
            if joint == names[i]:
                target_idx[j] = i

    new_anim = anim.copy()
    new_anim.offsets = []
    new_anim.parents = []
    new_anim.rotations = []
    new_names = []

    target_idx.sort()

    bias = 0
    new_id = {-1: -1}
    target_idx.append(-1)
    for i in range(n_joint):
        new_id[i] = i + bias
        if i == target_idx[bias]: bias += 1

    identity = np.zeros_like(anim.rotations)
    identity = identity[:, :1, :]

    bias = 0
    for i in range(n_joint):
        new_anim.parents.append(new_id[anim.parents[i]])
        new_names.append(names[i])
        new_anim.rotations.append(anim.rotations[:, [i], :])

        if i == target_idx[bias]:
            new_anim.offsets.append(anim.offsets[i] / 2)

            new_anim.parents.append(i + bias)
            new_names.append(names[i] + '_split')
            new_anim.offsets.append(anim.offsets[i] / 2)

            new_anim.rotations.append(identity)

            new_id[i] += 1
            bias += 1
        else:
            new_anim.offsets.append(anim.offsets[i])

    new_anim.offsets = np.array(new_anim.offsets)

    offset_spine = anim.offsets[target_idx[0]] + anim.offsets[target_idx[0] + 1]
    new_anim.offsets[target_idx[0]:target_idx[0]+3, :] = offset_spine / 3

    new_anim.rotations = np.concatenate(new_anim.rotations, axis=1)
    if not os.path.exists(os.path.split(save_file)[0]):
        os.system('mkdir -p {}'.format(os.path.split(save_file)[0]))
    BVH.save(save_file, new_anim, names=new_names, frametime=ftime, order='xyz')


def batch_split(source, dest):
    files = [f for f in os.listdir(source) if f.endswith('.bvh')]
    try:
        bvh_file = BVH_file(os.path.join(source, files[0]))
        if bvh_file.skeleton_type != 1: return
    except:
        return

    print("Working on {}".format(os.path.split(source)[-1]))
    if not os.path.exists(dest):
        os.system('mkdir -p {}'.format(dest))
    files = [f for f in os.listdir(source) if f.endswith('.bvh')]
    for i, file in tqdm(enumerate(files), total=len(files)):
        in_file = os.path.join(source, file)
        out_file = os.path.join(dest, file)
        split_joint(in_file, out_file)


if __name__ == '__main__':
    prefix = '/media/mmlab/Volume2/MixamoBVH/'
    names = [f for f in os.listdir(prefix) if os.path.isdir(os.path.join(prefix, f))]
    no_characters = ["SportyGranny","Kaya","Claire","BigVegas","Aj","Remy","Knight","Michelle"]
    char = ["Malcolm"]
    for name in names:
        if name in char:
            batch_split(os.path.join(prefix, name), os.path.join(prefix, name + '_m'))