import random
import torch
from torch.utils import data
import SimpleITK as itk
import numpy as np
import scipy
import scipy.io as sio
import os
from utils import central_padding_3D

def get_input(volume_path, para_path, flip=False, size_in=None):
    if size_in is None:
        size_in = [128, 128, 128]
    volume_input_original = itk.GetArrayFromImage(itk.ReadImage(volume_path))
    volume_input_original = volume_input_original.astype(np.float32)
    volume_input_original = volume_input_original.transpose(2, 1, 0) / 1000
    if flip:
        volume_input_original = np.flip(volume_input_original, axis=0)
        volume_input_original = np.flip(volume_input_original, axis=1)

    para = sio.loadmat(para_path)
    lpmtx_input = para['normlpmtxw']
    lpmtx_input = lpmtx_input.astype(np.float32)
    lpmtx_input = np.asarray(scipy.sparse.coo_matrix(lpmtx_input).todense())

    adj_input = para['adjw']
    adj_input = adj_input.astype(np.float32)
    adj_input = np.asarray(scipy.sparse.coo_matrix(adj_input).todense())

    centroid_input = para['newmid'].astype(np.int32)

    # Cutting
    if 'region' in para.keys():
        region = np.squeeze(para['region'])
    else:
        region = [1, 128, 1, 128, 1, 128]

    volume_input_original = volume_input_original[region[0] - 1:region[1] - 1, region[2] - 1:region[3] - 1,
                            region[4] - 1:region[5] - 1]
    centroid_input[:, 0] = centroid_input[:, 0] - region[0] + 1
    centroid_input[:, 1] = centroid_input[:, 1] - region[2] + 1
    centroid_input[:, 2] = centroid_input[:, 2] - region[4] + 1

    volume_input, centroid_input = central_padding_3D(volume_input_original, centroid_input, size_in)
    volume_input = np.expand_dims(volume_input, axis=0)
    volume_input = volume_input.astype(np.float32)

    gt_input = para['is_forward'].astype(np.float32)
    volume_elements = {"volume": volume_input, "centroid": centroid_input, "adj": adj_input, "lpmtx": lpmtx_input,
                       "gt": gt_input, "filename": volume_path.split('/')[-1].split('.')[0]}
    return volume_elements


class CBCT_Syn_Folder(data.Dataset):
    def __init__(self, syn_folder, img_prefix='img', para_prefix='para', flip=False):
        # Initializes LF paths and load into memory
        self.volume_path = []
        self.para_path = []
        self.syn_names = {}
        self.syn_folder = syn_folder
        self.img_prefix = img_prefix
        self.para_prefix = para_prefix
        self.flip = flip
        for root in sorted(os.listdir(syn_folder)):
            self.syn_names[root] = []
            cur_syn_folder = os.path.join(syn_folder, root)
            if not os.path.isdir(cur_syn_folder):
                continue
            cur_img_path = os.path.join(cur_syn_folder, img_prefix)
            cur_para_path = os.path.join(cur_syn_folder, para_prefix)
            for fname in sorted(os.listdir(cur_img_path)):
                self.syn_names[root].append(fname)
                path_volume = os.path.join(cur_img_path, fname)
                self.volume_path.append([root, path_volume])
                name_mat = fname.split('.')[0] + '.mat'
                path_mat = os.path.join(cur_para_path, name_mat)
                self.para_path.append([root, path_mat])

        self.length = len(self.volume_path)
        print("Data count in {} path: {}".format(syn_folder, self.length))

    def __getitem__(self, index):

        first_name, first_img_path = self.volume_path[index]
        _, first_para_path = self.para_path[index]

        cur_syn_folder = os.path.join(self.syn_folder, first_name)
        cur_img_path = os.path.join(cur_syn_folder, self.img_prefix)
        cur_para_path = os.path.join(cur_syn_folder, self.para_prefix)

        length_files = len(self.syn_names[first_name])
        second_index = random.randint(1, length_files - 1)
        second_img_name = self.syn_names[first_name][second_index]
        second_para_name = second_img_name.split('.')[0] + '.mat'
        second_img_path = os.path.join(cur_img_path, second_img_name)
        second_para_path = os.path.join(cur_para_path, second_para_name)

        return get_input(volume_path=first_img_path, para_path=first_para_path, flip=self.flip), \
            get_input(volume_path=second_img_path, para_path=second_para_path, flip=self.flip)

    def __len__(self):
        # Returns the total number of font files
        return len(self.volume_path)


def get_CBCT_syn_loader(syn_folder, flip=False, batch_size=1, num_workers=0, shuffle=True,
                        img_prefix='img', para_prefix='para'):
    # Builds and returns Dataloader
    dataset = CBCT_Syn_Folder(syn_folder, flip=flip, img_prefix='img', para_prefix='para')
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=False)
    return data_loader