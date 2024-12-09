import torch
from model import USLModel
import os
import scipy.io as sio
from dataloader_cbct import get_CBCT_loader, get_input
from utils import read_yaml
import argparse

torch.manual_seed(1)

parser = argparse.ArgumentParser(description="USL Training")
parser.add_argument("--model_config_dir", default="./configs", type=str)
parser.add_argument("--model_config_name", default="default_test.yaml", type=str)
print("Starting...", flush=True)
args = parser.parse_args()
cfg = read_yaml(os.path.join(args.model_config_dir, args.model_config_name))
print("model_config:", cfg, flush=True)

os.makedirs(cfg["ENGINE"]["save_dir"], exist_ok=True)

device = torch.device(f'cuda:{cfg["ENGINE"]["device"]}' if cfg["ENGINE"]["device"] > 0 and torch.cuda.is_available()
                      else 'cpu')

template_volume = get_input(cfg["DATASET"]["template_volume_path"], cfg["DATASET"]["template_para_path"])
template_volume["volume"] = torch.as_tensor(template_volume["volume"]).unsqueeze(0)
template_volume["centroid"] = torch.as_tensor(template_volume["centroid"]).unsqueeze(0)
template_volume["adj"] = torch.as_tensor(template_volume["adj"]).unsqueeze(0)

test_loader = get_CBCT_loader(cfg["DATASET"]["test_volume_path"], cfg["DATASET"]["test_para_path"],
                              batch_size=1, num_workers=0)

net = USLModel(in_channels=cfg["MODEL"]["in_chns"], base_num=cfg["MODEL"]["base_num"], device=device).to(device)
net.load_state_dict(torch.load(cfg["MODEL"]["pretrained_path"]))
net.eval()

for target_volume, _ in test_loader:
    target_name = target_volume["filename"]
    C_21, c_1, c_2, P_21, rebase_1, rebase_2, fea_1, fea_2, latent_fea = net(template_volume, target_volume)
    export_path = os.path.join(cfg["ENGINE"]["save_dir"], 'test_%s.mat' % (target_name[0]))
    sio.savemat(export_path, {'C': C_21.cpu().detach().numpy(), 're_base': rebase_2.cpu().detach().numpy(),
                              're_fea': fea_2.cpu().detach().numpy()})





