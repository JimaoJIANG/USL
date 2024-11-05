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
    # centroid_input = temp['middle'].astype(np.int32)

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

    hand_feainput = para['fea']
    hand_feainput = hand_feainput.astype(np.float32)
    volume_elements = {"volume": volume_input, "centroid": centroid_input, "adj": adj_input, "lpmtx": lpmtx_input,
                       "hand_fea": hand_feainput, "filename": volume_path.split('/')[-1].split('.')[0]}
    return volume_elements


class CBCT_Folder(data.Dataset):
    def __init__(self, volume_root, para_root, flip=False):
        # Initializes LF paths and load into memory

        self.volume_path = []
        self.para_path = []
        self.volume_root = volume_root
        self.flip = flip
        for root, _, fnames in sorted(os.walk(volume_root)):
            for fname in fnames:
                path_volume = os.path.join(root, fname)
                self.volume_path.append(path_volume)
                name_mat = fname.split('.')[0] + '.mat'
                path_mat = os.path.join(para_root, name_mat)
                self.para_path.append(path_mat)

        self.length = len(self.volume_path)
        print("Data count in {} path: {}".format(volume_root, self.length))

    def __getitem__(self, index):
        next_index = random.randint(1, self.length - 1)
        return get_input(volume_path=self.volume_path[index], para_path=self.para_path[index], flip=self.flip), \
            get_input(volume_path=self.volume_path[next_index], para_path=self.para_path[next_index], flip=self.flip)

    def __len__(self):
        # Returns the total number of font files
        return len(self.volume_path)


def get_CBCT_loader(volume_path, para_path, flip=False, batch_size=1, num_workers=0, shuffle=True):
    # Builds and returns Dataloader
    dataset = CBCT_Folder(volume_path, para_path, flip=flip)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=False)
    return data_loader

