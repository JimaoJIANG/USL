import os
import time
import datetime
import argparse
import matplotlib
import torch.nn.functional
from model import USLModel
from loss import USLLoss
from dataloader_cbct import get_CBCT_loader
from dataloader_cbct_syn import get_CBCT_syn_loader
import matplotlib.pyplot as plt
from utils import read_yaml
import gc

matplotlib.use('Agg')
torch.manual_seed(1)

parser = argparse.ArgumentParser(description="USL Training")
parser.add_argument("--model_config_dir", default="./configs", type=str)
parser.add_argument("--model_config_name", default="default.yaml", type=str)
parser.add_argument("--model_save_dir", default="./checkpoints", type=str, help="log directory")
parser.add_argument("--plot_dir", default="./temp_plots", type=str, help="log directory")
parser.add_argument("--log_dir", default="./logs", type=str, help="log directory")
parser.add_argument("--log_name", default="training_logs", type=str, help="log directory")
print("Starting...", flush=True)
args = parser.parse_args()
cfg = read_yaml(os.path.join(args.model_config_dir, args.model_config_name))
print("model_config:", cfg, flush=True)

model_path = args.model_save_dir
log_name = args.log_name
plot_dir = args.plot_dir
log_dir = args.log_dir
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, log_name + '.txt')
device = torch.device(f'cuda:{cfg["ENGINE"]["device"]}' if cfg["ENGINE"]["device"] > 0 and torch.cuda.is_available()
                      else 'cpu')

real_loader_train = get_CBCT_loader(cfg["DATASET"]["real_volume_train_path"], cfg["DATASET"]["real_para_train_path"],
                                    batch_size=1, num_workers=0)
real_loader_val = get_CBCT_loader(cfg["DATASET"]["real_volume_valid_path"], cfg["DATASET"]["real_para_valid_path"],
                                    batch_size=1, num_workers=0)
syn_loader_train = get_CBCT_syn_loader(cfg["DATASET"]["syn_folder_train_path"], batch_size=1, num_workers=0)
syn_loader_val = get_CBCT_syn_loader(cfg["DATASET"]["syn_folder_valid_path"], batch_size=1, num_workers=0)

net = USLModel(in_channels=cfg["MODEL"]["in_chns"], base_num=cfg["MODEL"]["base_num"], device=device).to(device)
if cfg["MODEL"]["pretrained"]:
    state_dict_cur = torch.load(cfg["MODEL"]["pretrained_path"])
    net.load_state_dict(state_dict_cur)


# Setup optimizer
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), cfg["OPTIM"]["lr"], 
                             (cfg["OPTIM"]["beta1"], cfg["OPTIM"]["beta2"]))


# Loss
usl_loss = USLLoss(lambda_emb=cfg["WEIGHTS"]["lambda_emb"], lambda_C_reg=cfg["WEIGHTS"]["lambda_C_reg"],
                   lambda_C_align=cfg["WEIGHTS"]["lambda_C_align"], lambda_cycle=cfg["WEIGHTS"]["lambda_cycle"],
                   lambda_intri=cfg["WEIGHTS"]["lambda_intri"], lambda_syn=cfg["WEIGHTS"]["lambda_syn"],
                   lambda_real=cfg["WEIGHTS"]["lambda_real"], base_num=cfg["MODEL"]["base_num"], device=device)

# Training
for epoch in range(cfg["ENGINE"]["epoch"]):
    current_time = datetime.datetime.now()
    print('[Epoch] %d/%d,' % (epoch + 1, cfg["ENGINE"]["epoch"]), current_time.strftime('%Y.%m.%d-%H:%M:%S'))
    log = open(log_file_path, 'a+')
    log.write('[Epoch] %d/%d,' % (epoch + 1, cfg["ENGINE"]["epoch"]))
    log.write(current_time.strftime('%Y.%m.%d-%H:%M:%S'))
    log.write('\n')
    start_time = time.time()
    net.train(True)

    if epoch > cfg["ENGINE"]["epoch"] - 20:
        optimizer.param_groups[0]['lr'] = cfg["OPTIM"]["lr"] / 10
        if epoch > cfg["ENGINE"]["epoch"] - 10:
            optimizer.param_groups[0]['lr'] = cfg["OPTIM"]["lr"] / 100

    # Train for real data
    if cfg["ENGINE"]["train_real"]:
        net.train(True)
        if epoch >= cfg["ENGINE"]["real_start_epoch"] and epoch < cfg["ENGINE"]["real_end_epoch"]:
            batch_steps = 0
            epoch_loss = 0.0
            for volume_1, volume_2 in real_loader_train:
                optimizer.zero_grad()
                C_21, c_1, c_2, P_21, rebase_1, rebase_2, fea_1, fea_2, latent_fea = net(volume_1, volume_2)
                loss = usl_loss(rebase_1, rebase_2, fea_1, fea_2, c_1, c_2, C_21, P_21, latent_fea, volume_1, volume_2)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
                batch_steps += 1

                if (batch_steps + 1) % cfg["ENGINE"]["real_log_step"] == 0:
                    print('[REAL Train] Loss:%f' % (loss))
                    log.write('[REAL Train] Loss:%f\n' % (loss))
                if (batch_steps + 1) % cfg["ENGINE"]["real_plot_step"] == 0:
                    plt.imshow(c_1.cpu().detach().numpy())
                    plt.savefig(os.path.join(plot_dir, 'REAL_c1_' + log_name + '.png'))
                    plt.imshow(c_2.cpu().detach().numpy())
                    plt.savefig(os.path.join(plot_dir, 'REAL_c2_' + log_name + '.png'))
                    plt.imshow(C_21.cpu().detach().numpy())
                    plt.savefig(os.path.join(plot_dir, 'REAL_C_' + log_name + '.png'))
                    plt.imshow(P_21.cpu().detach().numpy())
                    plt.savefig(os.path.join(plot_dir, 'REAL_P_' + log_name + '.png'))
                    plt.imshow(torch.matmul(rebase_1.T, rebase_1).cpu().detach().numpy())
                    plt.savefig(os.path.join(plot_dir, 'REAL_rebase_1_' + log_name + '.png'))
                    plt.cla()
                    plt.clf()
                    plt.close('all')

            print('---------------------------------------------------')
            print('[REAL Train Total] Loss:%f' % (epoch_loss / batch_steps))
            print('[Current LR]:', optimizer.param_groups[0]['lr'])
            print('(LR: %f) Time lasts: %.8fs' % (optimizer.param_groups[0]['lr'], time.time() - start_time))
            print('---------------------------------------------------')
            log.write('---------------------------------------------------\n')
            log.write('[REAL Train Total] Loss:%f\n' % (epoch_loss / batch_steps))
            log.write('[Current LR]:%f\n' % optimizer.param_groups[0]['lr'])
            log.write('(LR: %f) Time lasts: %.8fs\n' % (optimizer.param_groups[0]['lr'], time.time() - start_time))
            log.write('---------------------------------------------------\n')

            # Validation for real data
            net.train(False)
            net.eval()

            start_time = time.time()
            batch_steps = 0
            epoch_loss = 0.0
            for volume_1, volume_2 in real_loader_val:
                C_21, c_1, c_2, P_21, rebase_1, rebase_2, fea_1, fea_2, latent_fea = net(volume_1, volume_2)
                loss = usl_loss(rebase_1, rebase_2, fea_1, fea_2, c_1, c_2, C_21, P_21, latent_fea, volume_1, volume_2)
                epoch_loss += loss.detach().item()
                batch_steps += 1

            print('---------------------------------------------------')
            print('[REAL VALID] Loss:%f' % (epoch_loss / batch_steps))
            print('(LR: %f) Time lasts: %.8fs' % (optimizer.param_groups[0]['lr'], time.time() - start_time))
            print('---------------------------------------------------')
            log.write('---------------------------------------------------\n')
            log.write('[REAL VALID] Loss:%f\n' % (epoch_loss / batch_steps))
            log.write('(LR: %f) Time lasts: %.8fs\n' % (optimizer.param_groups[0]['lr'], time.time() - start_time))
            log.write('---------------------------------------------------\n')

    if cfg["ENGINE"]["train_syn"]:
        # Train for synthetic data
        net.train(True)
        if epoch >= cfg["ENGINE"]["syn_start_epoch"] or epoch < cfg["ENGINE"]["syn_end_epoch"]:
            batch_steps = 0
            epoch_loss = 0.0
            epoch_acc = 0.0
            for volume_1, volume_2 in syn_loader_train:
                optimizer.zero_grad()
                C_21, c_1, c_2, P_21, rebase_1, rebase_2, fea_1, fea_2, latent_fea = net(volume_1, volume_2)
                loss, acc = usl_loss(rebase_1, rebase_2, fea_1, fea_2, c_1, c_2, C_21, P_21, latent_fea, volume_1, volume_2,
                                     is_syn=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
                epoch_acc += acc
                batch_steps += 1

                if (batch_steps + 1) % cfg["ENGINE"]["syn_log_step"] == 0:
                    print('[SYN Train] Loss:%f' % (loss))
                    print('[SYN Train] ACC:{}'.format(acc))
                    log.write('[SYN Train] Loss:%f\n' % (loss))
                    log.write('[SYN Train] ACC:{}\n'.format(acc))
                if (batch_steps + 1) % cfg["ENGINE"]["syn_plot_step"] == 0:
                    plt.imshow(c_1.cpu().detach().numpy())
                    plt.savefig(os.path.join(plot_dir, 'SYN_c1_' + log_name + '.png'))
                    plt.imshow(c_2.cpu().detach().numpy())
                    plt.savefig(os.path.join(plot_dir, 'SYN_c2_' + log_name + '.png'))
                    plt.imshow(C_21.cpu().detach().numpy())
                    plt.savefig(os.path.join(plot_dir, 'SYN_C_' + log_name + '.png'))
                    plt.imshow(P_21.cpu().detach().numpy())
                    plt.savefig(os.path.join(plot_dir, 'SYN_P_' + log_name + '.png'))
                    plt.imshow(torch.matmul(rebase_1.T, rebase_1).cpu().detach().numpy())
                    plt.savefig(os.path.join(plot_dir, 'SYN_rebase_1_' + log_name + '.png'))
                    plt.cla()
                    plt.clf()
                    plt.close('all')

            print('---------------------------------------------------')
            print('[SYN Train Total] Loss:%f' % (epoch_loss / batch_steps))
            print('[SYN Train Total] ACC:%f' % (epoch_acc / batch_steps))
            print('[Current LR]:', optimizer.param_groups[0]['lr'])
            print('(LR: %f) Time lasts: %.8fs' % (optimizer.param_groups[0]['lr'], time.time() - start_time))
            print('---------------------------------------------------')
            log.write('---------------------------------------------------\n')
            log.write('[SYN Train Total] Loss:%f\n' % (epoch_loss / batch_steps))
            log.write('[SYN Train Total] ACC:%f\n' % (epoch_acc / batch_steps))
            log.write('[Current LR]:%f\n' % optimizer.param_groups[0]['lr'])
            log.write('(LR: %f) Time lasts: %.8fs\n' % (optimizer.param_groups[0]['lr'], time.time() - start_time))
            log.write('---------------------------------------------------\n')

            # Validation for synthetic data
            net.train(False)
            net.eval()

            start_time = time.time()
            batch_steps = 0
            epoch_loss = 0.0
            epoch_acc = 0
            for volume_1, volume_2 in syn_loader_val:
                C_21, c_1, c_2, P_21, rebase_1, rebase_2, fea_1, fea_2, latent_fea = net(volume_1, volume_2)
                loss, acc = usl_loss(rebase_1, rebase_2, fea_1, fea_2, c_1, c_2, C_21, P_21, latent_fea, volume_1, volume_2,
                                     is_syn=True)
                epoch_loss += loss.detach().item()
                epoch_acc += acc
                batch_steps += 1
            print('---------------------------------------------------')
            print('[SYN VALID] Loss:%f' % (epoch_loss / batch_steps))
            print('[SYN VALID] ACC:%f' % (epoch_acc / batch_steps))
            print('(LR: %f) Time lasts: %.8fs' % (optimizer.param_groups[0]['lr'], time.time() - start_time))
            print('---------------------------------------------------')
            log.write('---------------------------------------------------\n')
            log.write('[SYN VALID] Loss:%f\n' % (epoch_loss / batch_steps))
            log.write('[SYN VALID] ACC:%f\n' % (epoch_acc / batch_steps))
            log.write('(LR: %f) Time lasts: %.8fs\n' % (optimizer.param_groups[0]['lr'], time.time() - start_time))
            log.write('---------------------------------------------------\n')
    log.close()
    # Save model
    if (epoch + 1) % cfg["ENGINE"]["save_model_epoch"] == 0:
        state_dict_cur = net.state_dict()
        net_path = os.path.join(model_path, log_name + '_for_cbct_Net_%d-%d_%.6f.pkl'
                                % (epoch + 1, cfg["ENGINE"]["epoch"], cfg["OPTIM"]["lr"]))
        torch.save(state_dict_cur, net_path)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
