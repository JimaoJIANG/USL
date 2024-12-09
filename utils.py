import torch
import numpy as np
import yaml
from typing import Dict

def read_yaml(path) -> Dict:
    """
    Helper function to read yaml file
    """
    file = open(path, "r", encoding="utf-8")
    string = file.read()
    data_dict = yaml.safe_load(string)

    return data_dict


def make_opt_P(P, axis=0, device='cpu'):
    N1, N2 = P.shape
    if axis == 0:
        [_, row] = torch.min(P, dim=1)
        row = torch.vstack([torch.arange(0, N1).to(device), row])
        indices = row
        values = torch.as_tensor([1.] * indices.shape[1]).to(device)
    else:
        [_, col] = torch.min(P, dim=0)
        col = torch.vstack([col, torch.arange(0, N2).to(device)])
        indices = col
        values = torch.as_tensor([1.] * indices.shape[1]).to(device)
    P_opt = torch.sparse_coo_tensor(indices, values, (N1, N2)).to_dense()
    P_opt[P_opt > 0] = 1.0
    return P_opt

def cal_P(C, base1, base2):
    P1 = torch.matmul(torch.matmul(base2, C), base1.T)  # 1->2
    pmax = torch.max(P1)
    pmin = torch.min(P1)
    if pmax == pmin:
        pmax = 1.0
        pmin = 0.0
    P_12 = (P1 - pmin) / (pmax - pmin)
    return P_12

def central_padding_3D(img, centroid, out_size):
    in_size = list(img.shape)
    if in_size[0] > out_size[0]:
        img = img[0:out_size[0], :, :]
        in_size[0] = out_size[0]
    if in_size[1] > out_size[1]:
        img = img[:, 0:out_size[1], :]
        in_size[1] = out_size[1]
    if in_size[2] > out_size[2]:
        img = img[:, :, 0:out_size[2]]
        in_size[2] = out_size[2]
    if (out_size[0] - in_size[0]) % 2 == 0:
        c_left = (out_size[0] - in_size[0]) // 2
        c_right = (out_size[0] - in_size[0]) // 2
    else:
        c_left = (out_size[0] - in_size[0]) // 2
        c_right = (out_size[0] - in_size[0]) // 2 + 1
    if (out_size[1] - in_size[1]) % 2 == 0:
        up = (out_size[1] - in_size[1]) // 2
        down = (out_size[1] - in_size[1]) // 2
    else:
        up = (out_size[1] - in_size[1]) // 2
        down = (out_size[1] - in_size[1]) // 2 + 1
    if (out_size[2] - in_size[2]) % 2 == 0:
        left = (out_size[2] - in_size[2]) // 2
        right = (out_size[2] - in_size[2]) // 2
    else:
        left = (out_size[2] - in_size[2]) // 2
        right = (out_size[2] - in_size[2]) // 2 + 1
    img_padded = np.pad(img, ((c_left, c_right), (up, down), (left, right)), 'constant', constant_values=0)
    centroid[:, 0] = centroid[:, 0] + c_left
    centroid[:, 1] = centroid[:, 1] + up
    centroid[:, 2] = centroid[:, 2] + left
    return img_padded, centroid