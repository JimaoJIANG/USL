import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class USLModel(nn.Module):
    def __init__(self, in_channels=1, base_num=48, method="deep", device='cuda'):
        super(USLModel, self).__init__()
        self.method = method
        self.device = device
        ## Encoder
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))

        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))

        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2))

        self.conv5 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.2))

        ## Decoder
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))
        self.conv5_1 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2))

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))
        self.conv4_1 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2))

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(256, 64, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(0.2))
        self.conv3_1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2))

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(0.2))
        self.conv2_1 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2))

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(64, 4, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(0.2))
        self.conv1_1 = nn.Sequential(
            nn.Conv3d(4, 4, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2))

        ## Linear
        self.dense1 = nn.Sequential(
            nn.Linear(484, 600),
            nn.LeakyReLU(0.2))
        self.dense2 = nn.Sequential(
            nn.Linear(600, 600),
            nn.LeakyReLU(0.2))

        ## GCN
        # 3
        self.weight_gcn_1_1 = nn.init.trunc_normal_(nn.Parameter(torch.FloatTensor(484, 256), requires_grad=True))
        self.bias_gcn_1_1 = nn.init.trunc_normal_(nn.Parameter(torch.FloatTensor(256), requires_grad=True))

        self.weight_gcn_1_2 = nn.init.trunc_normal_(nn.Parameter(torch.FloatTensor(256, 128), requires_grad=True))
        self.bias_gcn_1_2 = nn.init.trunc_normal_(nn.Parameter(torch.FloatTensor(128), requires_grad=True))

        self.weight_gcn_1_3 = nn.init.trunc_normal_(nn.Parameter(torch.FloatTensor(128, base_num // 3), requires_grad=True))
        self.bias_gcn_1_3 = nn.init.trunc_normal_(nn.Parameter(torch.FloatTensor(base_num // 3), requires_grad=True))

        # 2
        self.weight_gcn_2_1 = nn.init.trunc_normal_(nn.Parameter(torch.FloatTensor(484, 128), requires_grad=True))
        self.bias_gcn_2_1 = nn.init.trunc_normal_(nn.Parameter(torch.FloatTensor(128), requires_grad=True))

        self.weight_gcn_2_2 = nn.init.trunc_normal_(nn.Parameter(torch.FloatTensor(128, base_num // 3), requires_grad=True))
        self.bias_gcn_2_2 = nn.init.trunc_normal_(nn.Parameter(torch.FloatTensor(base_num // 3), requires_grad=True))

        # 1
        tmp_channels = base_num - 2 * base_num // 3
        self.weight_gcn_3_1 = nn.init.trunc_normal_(nn.Parameter(torch.FloatTensor(484, tmp_channels), requires_grad=True))
        self.bias_gcn_3_1 = nn.init.trunc_normal_(nn.Parameter(torch.FloatTensor(tmp_channels), requires_grad=True))

        self.denseblock = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 48, 48 * 48),
            nn.Linear(48 * 48, 48 * 48),
            nn.Linear(48 * 48, 48 * 48)
        )

        self.latent_fea = nn.init.orthogonal_(nn.Parameter(torch.FloatTensor(48, 600), requires_grad=True))
        self.tao = nn.init.trunc_normal_(nn.Parameter(torch.FloatTensor(1), requires_grad=True))

    @staticmethod
    def gather_middle(params, indices):
        out = params[:, indices[:, 0], indices[:, 1], indices[:, 2]]
        return out.permute(1, 0).contiguous()

    def spectral_elements(self, x, centroid, gcn_adj):
        # Encoder
        h_conv1 = self.conv1(x)
        h_conv2 = self.conv2(h_conv1)
        h_conv3 = self.conv3(h_conv2)
        h_conv4 = self.conv4(h_conv3)
        h_conv5 = self.conv5(h_conv4)

        # Decoder
        h_deconv5 = self.deconv5(h_conv5)
        h_deconv5 = F.pad(h_deconv5, (1, 2, 1, 2, 1, 2), "constant", 0)
        h_conv5_1 = self.conv5_1(h_deconv5)

        h_conv5_cat = torch.cat([h_conv5_1, h_conv4], dim=1)

        h_deconv4 = self.deconv4(h_conv5_cat)
        h_deconv4 = F.pad(h_deconv4, (1, 2, 1, 2, 1, 2), "constant", 0)
        h_conv4_1 = self.conv4_1(h_deconv4)

        h_conv4_cat = torch.cat([h_conv4_1, h_conv3], dim=1)

        h_deconv3 = self.deconv3(h_conv4_cat)
        h_conv3_1 = self.conv3_1(h_deconv3)

        h_conv3_cat = torch.cat([h_conv3_1, h_conv2], dim=1)

        h_deconv2 = self.deconv2(h_conv3_cat)
        h_conv2_1 = self.conv2_1(h_deconv2)

        h_conv2_cat = torch.cat([h_conv2_1, h_conv1], dim=1)
        h_deconv1 = self.deconv1(h_conv2_cat)
        h_conv1_1 = self.conv1_1(h_deconv1)

        # Supervoxel feature gathering
        centroid_1 = torch.floor(centroid.to(torch.float32) / 2).to(torch.long)
        feature_0 = self.gather_middle(h_conv2_1[0, ...], centroid_1)

        centroid_2 = torch.floor(centroid.to(torch.float32) / 4).to(torch.long)
        feature_0 = torch.cat([self.gather_middle(h_conv3_1[0, ...], centroid_2), feature_0], dim=1)

        centroid_3 = torch.floor(centroid.to(torch.float32) / 8).to(torch.long)
        feature_0 = torch.cat([self.gather_middle(h_conv4_1[0, ...], centroid_3), feature_0], dim=1)

        centroid_4 = torch.floor(centroid.to(torch.float32) / 16).to(torch.long)
        feature_0 = torch.cat([self.gather_middle(h_conv5_1[0, ...], centroid_4), feature_0], dim=1)

        feature_0 = torch.cat([self.gather_middle(h_conv1_1[0, ...], centroid), feature_0], dim=1)
        feature_0_l2 = torch.nn.functional.normalize(feature_0, p=2, dim=0)

        # Linear projection
        f_conv1 = self.dense1(feature_0)
        re_fea = self.dense2(f_conv1)

        ## GCN
        # 3
        g_conv1_1 = torch.matmul(torch.matmul(gcn_adj, feature_0_l2), self.weight_gcn_1_1) + self.bias_gcn_1_1
        g_conv1_2 = torch.matmul(torch.matmul(gcn_adj, g_conv1_1), self.weight_gcn_1_2) + self.bias_gcn_1_2
        g_conv1_3 = torch.matmul(torch.matmul(gcn_adj, g_conv1_2), self.weight_gcn_1_3) + self.bias_gcn_1_3

        # 2
        g_conv2_1 = torch.matmul(torch.matmul(gcn_adj, feature_0_l2), self.weight_gcn_2_1) + self.bias_gcn_2_1
        g_conv2_2 = torch.matmul(torch.matmul(gcn_adj, g_conv2_1), self.weight_gcn_2_2) + self.bias_gcn_2_2

        # 1
        g_conv3_1 = torch.matmul(torch.matmul(gcn_adj, feature_0_l2), self.weight_gcn_3_1) + self.bias_gcn_3_1

        # concat
        re_base = torch.nn.functional.normalize(torch.cat([g_conv3_1, g_conv2_2, g_conv1_3], dim=1), p=2, dim=0)
        return re_fea, re_base, h_conv1_1[0, 3, ...]

    def cal_C_deep(self, fea_spec_1, fea_spec_2):
        similarity = torch.matmul(fea_spec_1, fea_spec_2.T) / \
                     (torch.norm(fea_spec_1, p=2) * torch.norm(fea_spec_2, p=2) + 1e-6)
        C_flattened = self.denseblock(similarity.unsqueeze(0).unsqueeze(0))
        C = torch.reshape(C_flattened.squeeze(0).squeeze(0), (48, 48))
        return C

    def cal_P(self, C, base1, base2):
        P_21 = euclidean_dist(base2, torch.matmul(C, base1.T).T)
        P_21 = torch.nn.functional.normalize(torch.nn.functional.normalize(P_21, p=2, dim=0), p=2, dim=1)
        return P_21

    def forward(self, input_1, input_2):
        # Get input for network
        volume_1 = input_1["volume"].to(self.device)
        centroid_1 = input_1["centroid"].to(torch.long).squeeze(0).to(self.device)
        gcn_adj_1 = input_1["adj"].squeeze(0).to(self.device)

        volume_2 = input_2["volume"].to(self.device)
        centroid_2 = input_2["centroid"].to(torch.long).squeeze(0).to(self.device)
        gcn_adj_2 = input_2["adj"].squeeze(0).to(self.device)

        # Supervoxel-level features
        feature_1, rebase_1, revolume_1 = self.spectral_elements(volume_1, centroid_1, gcn_adj_1)
        feature_2, rebase_2, revolume_2 = self.spectral_elements(volume_2, centroid_2, gcn_adj_2)

        # Feature projection
        fea_spec_1 = torch.matmul(rebase_1.T, feature_1)
        fea_spec_2 = torch.matmul(rebase_2.T, feature_2)

        # Decomposed maps
        c_1 = self.cal_C_deep(fea_spec_1, self.latent_fea)
        c_2 = self.cal_C_deep(fea_spec_2, self.latent_fea)

        C_12 = torch.matmul(c_1, c_2.T)
        C_21 = C_12.T

        # Permutation matrix
        P = self.cal_P(C_21, rebase_1, rebase_2)
        return C_21, c_1, c_2, P, rebase_1, rebase_2, feature_1, feature_2, self.latent_fea
