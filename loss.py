import torch
import torch.nn as nn
import torch.nn.functional as F

class USLLoss(nn.Module):
    def __init__(self, lambda_emb=10, lambda_C_reg=1, lambda_C_align=1, lambda_cycle=1, lambda_intri=1, lambda_syn=10,
                 lambda_real=1, base_num=48, device='cuda'):
        super().__init__()
        self.lambda_emb = lambda_emb
        self.lambda_C_reg = lambda_C_reg
        self.lambda_C_align = lambda_C_align
        self.lambda_cycle = lambda_cycle
        self.lambda_intri = lambda_intri
        self.lambda_syn = lambda_syn
        self.lambda_real = lambda_real
        self.base_num = base_num
        self.device = device

    def forward(self, rebase_1, rebase_2, fea_1, fea_2, c_1, c_2, C_21, P_21, latent_fea, volume_1, volume_2, is_syn=False):
        lpmtx_1 = volume_1["lpmtx"].squeeze(0).to(self.device)
        lpmtx_2 = volume_2["lpmtx"].squeeze(0).to(self.device)

        loss_ort = F.mse_loss(torch.eye(self.base_num).to(self.device), torch.matmul(rebase_1.T, rebase_1)) + \
                   F.mse_loss(torch.eye(self.base_num).to(self.device), torch.matmul(rebase_2.T, rebase_2))
        loss_diag_tmp_1 = torch.matmul(rebase_1.T, torch.matmul(lpmtx_1, rebase_1)) * (
                1.0 - torch.eye(self.base_num).to(self.device))
        loss_diag_tmp_2 = torch.matmul(rebase_2.T, torch.matmul(lpmtx_2, rebase_2)) * (
                1.0 - torch.eye(self.base_num).to(self.device))
        loss_diag = torch.mean(torch.abs(loss_diag_tmp_1)) + torch.mean(torch.abs(loss_diag_tmp_2))

        loss_emb = loss_ort + loss_diag

        fea_spec_1 = torch.matmul(rebase_1.T, fea_1)
        fea_spec_2 = torch.matmul(rebase_2.T, fea_2)

        loss_C_align = torch.nn.functional.mse_loss(torch.matmul(C_21, fea_spec_2), fea_spec_1)
        loss_C_reg = F.mse_loss(torch.eye(self.base_num).to(self.device), torch.matmul(c_1, c_1.T)) + \
                   F.mse_loss(torch.eye(self.base_num).to(self.device), torch.matmul(c_2, c_2.T))

        loss_cycle = F.mse_loss(torch.matmul(c_1, fea_spec_1), torch.matmul(c_2, fea_spec_2))
        loss_intri = F.mse_loss(torch.matmul(c_1, fea_spec_1), latent_fea) + \
                     F.mse_loss(torch.matmul(c_2, fea_spec_2), latent_fea)
        if is_syn:
            gt_1 = volume_1["gt"].to(self.device).squeeze(0)
            gt_2 = volume_2["gt"].to(self.device).squeeze(0)
            dismap = torch.cdist(gt_2, gt_1, p=1.0)
            dismap[:, :] = (dismap[:, :] == 0)
            loss_syn = F.l1_loss(torch.matmul(dismap, rebase_1), torch.matmul(rebase_2, C_21))

            loss = self.lambda_emb * loss_emb + self.lambda_C_reg * loss_C_reg + self.lambda_C_align * loss_C_align + \
                   self.lambda_cycle * loss_cycle + self.lambda_intri * loss_intri + self.lambda_syn * loss_syn

            correct_prediction = torch.argmax(dismap, dim=0) == torch.argmin(P_21, dim=0)  # dim???
            accuracy = torch.mean(correct_prediction.to(torch.float32))
            acc = accuracy.detach().item()
            return loss, acc

        hand_fea_1 = volume_1["hand_fea"].to(self.device).squeeze(0)
        hand_fea_2 = volume_2["hand_fea"].to(self.device).squeeze(0)
        hand_spec_1 = torch.matmul(rebase_1.T, hand_fea_1)
        hand_spec_2 = torch.matmul(rebase_2.T, hand_fea_2)
        loss_real = torch.nn.functional.mse_loss(torch.matmul(C_21, hand_spec_1), hand_spec_2)
        loss = self.lambda_emb * loss_emb + self.lambda_C_reg * loss_C_reg + self.lambda_C_align * loss_C_align + \
               self.lambda_cycle * loss_cycle + self.lambda_intri * loss_intri + self.lambda_real * loss_real
        return loss

