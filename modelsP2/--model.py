import torch
import torch.nn as nn
from models.utils import Conv1d, PointNet_FP_Module, PointNet_SA_Module
from models.transformer import Transformer
from torch_geometric.utils import to_dense_batch
from typing import List
from torch_geometric.data import Data, Batch

from pugcn_lib.upsample import GeneralUpsampler, PointShuffle

from Snowflake_model.utils import PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, Transformer
from Snowflake_model.skip_transformer import SkipTransformer
import torch.nn.functional as F
import A3_FPS



class sn_FeatFeatureExtractor(nn.Module):
    def __init__(self, out_dim=1024):
        """Encoder that encodes information of partial point cloud
        """
        super(sn_FeatFeatureExtractor, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = Transformer(256, dim=64)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

    def forward(self, point_cloud):
        """
        Args:
             point_cloud: b, 3, n

        Returns:
            l3_points: (B, out_dim, 1)
        """
        l0_xyz = point_cloud
        l0_points = point_cloud

        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l1_points = self.transformer_1(l1_points, l1_xyz)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        l2_points = self.transformer_2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)

        return l3_points


class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1 = self.ps(feat)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (b, 128, 256)
        completion = self.mlp_4(x3)  # (b, 3, 256)
        return completion


class SPD(nn.Module):
    def __init__(self, dim_feat=512, up_factor=2, i=0, radius=1):
        """Snowflake Point Deconvolution"""
        super(SPD, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + dim_feat, layer_dims=[256, 128])

        self.skip_transformer = SkipTransformer(in_channel=128, dim=64)

        self.mlp_ps = MLP_CONV(in_channel=128, layer_dims=[64, 32])
        self.ps = nn.ConvTranspose1d(32, 128, up_factor, up_factor, bias=False)   # point-wise splitting

        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)

        self.mlp_delta = MLP_CONV(in_channel=128, layer_dims=[64, 3])

    def forward(self, pcd_prev, feat_global, K_prev=None):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            feat_global: Tensor, (B, dim_feat, 1)
            K_prev: Tensor, (B, 128, N_prev)

        Returns:
            pcd_child: Tensor, up sampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        b, _, n_prev = pcd_prev.shape
        feat_1 = self.mlp_1(pcd_prev)
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                            feat_global.repeat(1, 1, feat_1.size(2))], 1)
        Q = self.mlp_2(feat_1)

        H = self.skip_transformer(pcd_prev, K_prev if K_prev is not None else Q, Q)

        feat_child = self.mlp_ps(H)
        feat_child = self.ps(feat_child)  # (B, 128, N_prev * up_factor)
        H_up = self.up_sampler(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))

        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / self.radius**self.i  # (B, 3, N_prev * up_factor)
        pcd_child = self.up_sampler(pcd_prev)
        pcd_child = pcd_child + delta

        return pcd_child, K_curr


class sn_Decoder(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256, num_p0=512, radius=1, up_factors=None):
        super(sn_Decoder, self).__init__()
        self.num_p0 = num_p0
        self.decoder_coarse = SeedGenerator(dim_feat=dim_feat, num_pc=num_pc)

        up_factors = [2, 2,2]
        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = [1] + up_factors

        uppers = []
        for i, factor in enumerate(up_factors):
            uppers.append(SPD(dim_feat=dim_feat, up_factor=factor, i=i, radius=radius))

        self.uppers = nn.ModuleList(uppers)

    def forward(self, feat, partial, return_P0=False):
        """
        Args:
            feat: Tensor, (b, dim_feat, n)
            partial: Tensor, (b, n, 3)
        """
        arr_pcd = []
        pcd = self.decoder_coarse(feat).permute(0, 2, 1).contiguous()  # (B, num_pc, 3)
        arr_pcd.append(pcd)
        pcd = fps_subsample(torch.cat([pcd, partial], 1), self.num_p0)
        if return_P0:
            arr_pcd.append(pcd)
        K_prev = None
        pcd = pcd.permute(0, 2, 1).contiguous()
        for upper in self.uppers:
            pcd, K_prev = upper(pcd, feat, K_prev)
            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())

        return arr_pcd




class Unit(nn.Module):
    def __init__(self, step=1, in_channel=256):
        super(Unit, self).__init__()
        self.step = step
        if step == 1:
            return

        self.conv_z = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=torch.sigmoid)
        self.conv_r = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=torch.sigmoid)
        self.conv_h = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=torch.relu)

    def forward(self, cur_x, prev_s):
        """
        Args:
            cur_x: Tensor, (B, in_channel, N)
            prev_s: Tensor, (B, in_channel, N)

        Returns:
            h: Tensor, (B, in_channel, N)
            h: Tensor, (B, in_channel, N)
        """
        if self.step == 1:
            return cur_x, cur_x

        z = self.conv_z(torch.cat([cur_x, prev_s], 1))
        r = self.conv_r(torch.cat([cur_x, prev_s], 1))
        h_hat = self.conv_h(torch.cat([cur_x, r * prev_s], 1))
        h = (1 - z) * cur_x + z * h_hat
        return h, h


class StepModel(nn.Module):
    def __init__(self, step=1):
        super(StepModel, self).__init__()
        self.step = step
        self.sa_module_1 = PointNet_SA_Module(512, 32, 0.2, 3, [64, 64, 128], group_all=False)
        self.sa_module_2 = PointNet_SA_Module(128, 32, 0.4, 128, [128, 128, 256], group_all=False)
        self.sa_module_3 = PointNet_SA_Module(None, None, None, 256, [256, 512, 1024], group_all=True)

        self.fp_module_3 = PointNet_FP_Module(1024, [256, 256], use_points1=True, in_channel_points1=256)
        self.fp_module_2 = PointNet_FP_Module(256, [256, 128], use_points1=True, in_channel_points1=128)
        self.fp_module_1 = PointNet_FP_Module(128, [128, 128, 128], use_points1=True, in_channel_points1=6)

        self.unit_3 = Unit(step=step, in_channel=256)
        self.unit_2 = Unit(step=step, in_channel=128)
        self.unit_1 = Unit(step=step, in_channel=128)

        mlp = [128, 64, 3]
        last_channel = 128 + 32
        mlp_conv = []
        for out_channel in mlp[:-1]:
            mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=True))
            last_channel = out_channel
        mlp_conv.append(Conv1d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*mlp_conv)

    def forward(self, point_cloud, prev_s):
        device = point_cloud.device
        l0_xyz = point_cloud
        l0_points = point_cloud

        l1_xyz, l1_points = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        # print('l1_xyz, l1_points', l1_xyz.shape, l1_points.shape)
        l2_xyz, l2_points = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        # print('l2_xyz, l2_points', l2_xyz.shape, l2_points.shape)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, 1024, 1)
        # print('l3_xyz, l3_points', l3_xyz.shape, l3_points.shape)

        l2_points = self.fp_module_3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points, prev_s['l2'] = self.unit_3(l2_points, prev_s['l2'])
        # print('l2_points, prev_s[l2]', l2_points.shape, prev_s['l2'].shape)

        l1_points = self.fp_module_2(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points, prev_s['l1'] = self.unit_2(l1_points, prev_s['l1'])
        # print('l1_points, prev_s[l1]', l1_points.shape, prev_s['l1'].shape)

        l0_points = self.fp_module_1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)
        l0_points, prev_s['l0'] = self.unit_1(l0_points, prev_s['l0'])  # (B, 128, 2048)
        # print('l0_points, prev_s[l0]', l0_points.shape, prev_s['l0'].shape)

        b, _, n = l0_points.shape
        noise = torch.normal(mean=0, std=torch.ones((b, 32, n), device=device))
        delta_xyz = torch.tanh(self.mlp_conv(torch.cat([l0_points, noise], 1))) * 1.0 / 10 ** (self.step - 1)
        point_cloud = point_cloud + delta_xyz
        return point_cloud, delta_xyz


class StepModelNoise(nn.Module):
    def __init__(self, step=1, if_noise=False, noise_dim=3, noise_stdv=1e-2):
        super(StepModelNoise, self).__init__()
        self.step = step
        self.if_noise = if_noise
        self.noise_dim = noise_dim
        self.noise_stdv = noise_stdv
        self.sa_module_1 = PointNet_SA_Module(512, 32, 0.2, 3 + (self.noise_dim if self.if_noise else 0), [64, 64, 128],
                                              group_all=False)
        self.sa_module_2 = PointNet_SA_Module(128, 32, 0.4, 128, [128, 128, 256], group_all=False)
        self.sa_module_3 = PointNet_SA_Module(None, None, None, 256, [256, 512, 1024], group_all=True)

        self.fp_module_3 = PointNet_FP_Module(1024, [256, 256], use_points1=True, in_channel_points1=256)
        self.fp_module_2 = PointNet_FP_Module(256, [256, 128], use_points1=True, in_channel_points1=128)
        self.fp_module_1 = PointNet_FP_Module(128, [128, 128, 128], use_points1=True,
                                              in_channel_points1=6 + (self.noise_dim if self.if_noise else 0))

        self.unit_3 = Unit(step=step, in_channel=256)
        self.unit_2 = Unit(step=step, in_channel=128)
        self.unit_1 = Unit(step=step, in_channel=128)

        mlp = [128, 64, 3]
        last_channel = 128 + 32
        mlp_conv = []
        for out_channel in mlp[:-1]:
            mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=True))
            last_channel = out_channel
        mlp_conv.append(Conv1d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*mlp_conv)

    def forward(self, point_cloud, prev_s):
        device = point_cloud.device
        l0_xyz = point_cloud
        l0_points = point_cloud

        b, _, n = l0_points.shape

        fef = self.noise_dim if self.if_noise else 0
        #print(l0_points.shape)
        if fef == 0:
            l0_points = l0_points #  没有拼接的意思
        else:
            noise_points = torch.normal(mean=0, std=torch.ones((b, (fef), n), device=device) * self.noise_stdv )
            l0_points = torch.cat([l0_points, noise_points], 1)
        #print(l0_points.shape)

        l1_xyz, l1_points = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)

        l2_xyz, l2_points = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)

        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, 1024, 1)

        l2_points = self.fp_module_3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points, prev_s['l2'] = self.unit_3(l2_points, prev_s['l2'])

        l1_points = self.fp_module_2(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points, prev_s['l1'] = self.unit_2(l1_points, prev_s['l1'])

        l0_points = self.fp_module_1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)
        l0_points, prev_s['l0'] = self.unit_1(l0_points, prev_s['l0'])  # (B, 128, 2048)

        noise = torch.normal(mean=0, std=torch.ones((b, 32, n), device=device))
        delta_xyz = torch.tanh(self.mlp_conv(torch.cat([l0_points, noise], 1))) * 1.0 / 10 ** (self.step - 1)
        point_cloud = point_cloud + delta_xyz
        return point_cloud, delta_xyz


class PMPNet(nn.Module):
    def __init__(self, dataset='Completion3D', noise_dim=3, noise_stdv=1e-2):
        super(PMPNet, self).__init__()
        if dataset == 'ShapeNet':
            self.step_1 = StepModelNoise(step=1, if_noise=True, noise_dim=noise_dim, noise_stdv=noise_stdv)
        else:
            self.step_1 = StepModel(step=1)

        self.step_2 = StepModelNoise(step=2)
        self.step_3 = StepModelNoise(step=3)

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: Tensor, (B, 2048, 3)
        """
        b, npoint, _ = point_cloud.shape
        device = point_cloud.device
        point_cloud = point_cloud.permute(0, 2, 1).contiguous()
        prev_s = {
            'l0': torch.normal(mean=0, std=torch.ones((b, 128, npoint), dtype=torch.float, device=device) * 0.01),
            'l1': torch.normal(mean=0, std=torch.ones((b, 128, 512), dtype=torch.float, device=device) * 0.01),
            'l2': torch.normal(mean=0, std=torch.ones((b, 256, 128), dtype=torch.float, device=device) * 0.01)
        }

        pcd_out_1, delta1 = self.step_1(point_cloud, prev_s)
        pcd_out_2, delta2 = self.step_2(pcd_out_1, prev_s)
        pcd_out_3, delta3 = self.step_3(pcd_out_2, prev_s)

        return [pcd_out_1.permute(0, 2, 1).contiguous(), pcd_out_2.permute(0, 2, 1).contiguous(),
                pcd_out_3.permute(0, 2, 1).contiguous()], [delta1, delta2, delta3]

def switchupdata(batch_input):
    temparry = []

    for i in range(batch_input.shape[0]):
            onetorch = batch_input[i].squeeze()
            temparry.append(Data(onetorch))
            # print(onetorch.shape)
    switch = Batch().from_data_list(temparry)
    x = switch.x
    batch= switch.batch
    return x,batch



class StepModelTransformer(nn.Module):
    def __init__(self, step=1, if_noise=False, noise_dim=3, noise_stdv=1e-2, dim_tail=32):
        super(StepModelTransformer, self).__init__()


        self.step = step
        self.if_noise =  False        # if_noise
        self.noise_dim = noise_dim
        self.noise_stdv = noise_stdv
        self.dim_tail = dim_tail
        self.sa_module_1 = PointNet_SA_Module(512, 32, 0.2, 3 + (self.noise_dim if self.if_noise else 0), [64, 64, 128],
                                              group_all=False)
        self.transformer_start_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointNet_SA_Module(128, 32, 0.4, 128, [128, 128, 256], group_all=False)
        self.transformer_start_2 = Transformer(256, dim=64)
        self.sa_module_3 = PointNet_SA_Module(None, None, None, 256, [256, 512, 1024], group_all=True)

        self.fp_module_3 = PointNet_FP_Module(1024, [256, 256], use_points1=True, in_channel_points1=256)
        self.fp_module_2 = PointNet_FP_Module(256, [256, 128], use_points1=True, in_channel_points1=128)
        self.fp_module_1 = PointNet_FP_Module(128, [128, 128, 128], use_points1=True, in_channel_points1=6)

        self.unit_3 = Unit(step=step, in_channel=256)
        self.unit_2 = Unit(step=step, in_channel=128)
        self.unit_1 = Unit(step=step, in_channel=128)


        mlp = [128, 64, 24]
        # last_channel = 128 + self.dim_tail  # (32 if self.step == 1 else 0)
        last_channel = 128
        mlp_conv = []
        for out_channel in mlp[:-1]:
            mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=True))
            last_channel = out_channel
        mlp_conv.append(Conv1d(last_channel, mlp[-1], if_bn=False, activation_fn=None))

        self.mlp_conv1 = nn.Sequential(*mlp_conv)

        self.mlp_conv_24 = Conv1d(24, 24, if_bn=True, activation_fn=torch.relu)
        self.mlp_conv_end = Conv1d(24, 3, if_bn=True, activation_fn=torch.relu)

        #self.conv_h = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=torch.relu)

        #######################################
        # mlp = [24,3]
        # # last_channel = 128 + self.dim_tail  # (32 if self.step == 1 else 0)
        # last_channel = 24
        # mlp_conv = []
        # for out_channel in mlp[:-1]:
        #     mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=True))
        #     last_channel = out_channel
        # mlp_conv.append(Conv1d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        #
        # self.mlp_conv2 = nn.Sequential(*mlp_conv)

        ###################################
        self.upsampler = GeneralUpsampler(
            in_channels=24,
            out_channels=24,
            k=20,
            r=4 ,  #  z ratio   # 理论要四倍
            upsampler="nodeshuffle",
            conv="edge" ,
        )
        self.reconstructor = torch.nn.Sequential(
            torch.nn.Linear(24, 24),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(24, 3),
        )
        ####################################################
        dim_feat = 512
        num_pc = 256
        num_p0 = 512
        radius = 1
        up_factors = None
        self.sn_feat_extractor =  sn_FeatFeatureExtractor(out_dim=dim_feat)
        self.sn_decoder = sn_Decoder(dim_feat=dim_feat, num_pc=num_pc, num_p0=num_p0, radius=radius, up_factors=up_factors)
        ##########################################################
        ### PFnet
        self.crop_point_num = 1024
        self.fc1 = nn.Linear(1024, 1024)  # 1920*1  拉成 1024*1
        self.fc2 = nn.Linear(1024, 512)  # 1024*1  拉成 512*1
        self.fc3 = nn.Linear(512, 256)  # 512*1  拉成 256*1

        self.fc1_1 = nn.Linear(1024, 128 * 512)
        self.fc2_1 = nn.Linear(512, 64 * 128)  # nn.Linear(512,64*256) !
        self.fc3_1 = nn.Linear(256, 64 * 3)  # 256*1  拉成 192*1 （64*3）#

        #
        self.conv1_1 = torch.nn.Conv1d(512, 512, 1)  # torch.nn.Conv1d(256,256,1) !
        self.conv1_2 = torch.nn.Conv1d(512, 256, 1)
        self.conv1_3 = torch.nn.Conv1d(256, int((self.crop_point_num * 3) / 128), 1)
        self.conv2_1 = torch.nn.Conv1d(128, 6, 1)  # torch.nn.Conv1d(256,12,1) !

    def forward(self, point_cloud, gt_ones_4096,  gt_ones_4224):   #[B,3,2048]
        myreadpoint=point_cloud
        point_cloud = point_cloud.permute(0, 2, 1).contiguous()
        device = point_cloud.device
        l0_xyz = point_cloud
        l0_points = point_cloud
        b, _, n = l0_points.shape
        # noise_points = torch.normal(mean=0, std=torch.ones(( b, (self.noise_dim if self.if_noise else 0), n),
        #                                                    device=device) * self.noise_stdv)

        # l0_points = torch.cat([l0_points, noise_points], 1)  # [B,6,2048]=[B,3,2048] [B,32048]

        l1_xyz, l1_points = self.sa_module_1(l0_xyz, l0_xyz)  # (B, 3, 512), (B, 128, 512)
        #l1_points = self.transformer_start_1(l1_points, l1_xyz)
        l2_xyz, l2_points = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        #l2_points = self.transformer_start_2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, 1024, 1)

        l2_points = self.fp_module_3(l2_xyz, l3_xyz, l2_points, l3_points)                                   #[16,256,128]
        l1_points = self.fp_module_2(l1_xyz, l2_xyz, l1_points, l2_points)                                   #[16,128,512]
        l0_points = self.fp_module_1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_xyz], 1), l1_points)  #[16,128,2048]


        #########################################################################

        # # 以上：三个原样，卷积后形成1个特征图
        # # 以下： 从1个特征图中，上采样形成三层的特征图
        l3_points = l3_points.squeeze(2)
        x_1 = F.relu(self.fc1(l3_points))  # 1024   #形成三种特征金字塔
        x_2 = F.relu(self.fc2(x_1))  # 512
        x_3 = F.relu(self.fc3(x_2))  # 256

        pc1_feat = self.fc3_1(x_3)  # Linear： 256*1 拉成 192*1 ，  192是 64个点的意思
        pc1_xyz = pc1_feat.reshape(-1, 64, 3)  # [8,192] -> [8,64,3]   # -1 保留 # 特征向量还原 拉成立体

        pc2_feat = F.relu(self.fc2_1(x_2))  # #Linear：拉成..
        pc2_feat = pc2_feat.reshape(-1, 128, 64)  # [8,8192] -> [8,128,64]
        pc2_xyz = self.conv2_1(pc2_feat)  # [8,128,64] -> [8,6,64]   # nn.Conv1d(128,6,1)

        yyyyyy = self.fc1_1(x_1)
        pc3_feat = F.relu(yyyyyy)  # # #Linear：拉成..
        pc3_feat = pc3_feat.reshape(-1, 512, 128)
        pc3_feat = F.relu(self.conv1_1(pc3_feat))
        pc3_feat = F.relu(self.conv1_2(pc3_feat))
        pc3_xyz = self.conv1_3(pc3_feat)

        # 以上： 从1个特征图中，上采样形成三层的特征图
        # 以下： 三个特征图再卷积，之后再拼接，  还原出立体形
        pc1_xyz_expand = torch.unsqueeze(pc1_xyz, 2)

        pc2_xyz = pc2_xyz.transpose(1, 2)  # [8,6,64]   [8,64，6]
        pc2_xyz = pc2_xyz.reshape(-1, 64, 2, 3)  # [8,64，6]  [8,64，2,3]   # 为了做加法， 做维度操作
        pc2_xyz = pc1_xyz_expand + pc2_xyz  # [8,64，2,3]  = [8,64，2,3] +  [8,64,1，3]
        pc2_xyz = pc2_xyz.reshape(-1, 128, 3)  # [8,64，2,3] ->  #  [8,128，3]
        pc2_xyz_expand = torch.unsqueeze(pc2_xyz, 2)  # [8,128，3]-> [8,128，1,3]

        pc3_xyz = pc3_xyz.transpose(1, 2)  # -> [8,128,12]
        pc3_xyz = pc3_xyz.reshape(-1, 128, int(self.crop_point_num / 128), 3)  # [8,128,12]  -> [8,128,4,3]
        pc3_xyz = pc2_xyz_expand + pc3_xyz  # [8,128,4,3]  =  [8,128,1,3]  +  [8,128,4,3]
        #pred_1024 = pc3_xyz.reshape(-1, self.crop_point_num, 3)  # [8,128,4,3]-> [8,512,3]
        #pred_FPnet = pc3_xyz.reshape(-1, 24,128)  # [8,128,4,3]-> [8,512,3]

        ########################################################################
        ## 上采样
        ##
        middleF0=self.mlp_conv1(l0_points)         # 中间value
        # middleP = self.mlp_conv2(middleF)

        middleF = middleF0.permute(0, 2, 1).contiguous()
        pos,batch = switchupdata(myreadpoint)
        up_middleF,_ = switchupdata(middleF)
        # （32768，24） = （8192，24）（2，163840），（8192，3）（8192，）
        #                 (点云 通道书增加)（可后弄） （原始点云）（点云batch分布）
        x = self.upsampler(up_middleF, edge_index=None, pos=pos, batch=batch)
        # q = self.reconstructor(x)  # （32768，3）
        _, PYG_batch_gt = switchupdata(gt_ones_4096)
        pred, _ = to_dense_batch(x , PYG_batch_gt)  # [B, N * r, 3]
        #####################################
        ##   点云 转 图矩阵       #############
        temp_pos, temp_batch = switchupdata(pred)  # 点云 转 矩阵


        PF_F = pc3_xyz.reshape(-1, 128, 24)  # [8,8192] -> [8,128,64]

        tensorlist=[]
        for i in range(PF_F.shape[0]):

            catend = torch.cat((pred[i], PF_F[i]), 0)
            # print(catend.shape)
            tensorlist.append(catend)

        #####################################
        cat_end = torch.cat(tensorlist, 0)
        q = self.reconstructor(cat_end)  # （32768，3）
        _, PYG_batch_4224 = switchupdata(gt_ones_4224)
        pred, _ = to_dense_batch(q , PYG_batch_4224)  # [B, N * r, 3]

        # print(cat_end.shape)


        # xx= pred.permute(0, 2, 1).contiguous()
        #
        # xx = self.mlp_conv_24(xx)
        # pre_4096 = self.mlp_conv_end(xx)
        # pre_4096 = pre_4096.permute(0, 2, 1).contiguous()


        # _, PYG_batch_gt= switchupdata(gt)
        # pred_3072, _ = to_dense_batch(x, PYG_batch_gt)  # [B, N * r, 3]
        # pred_3072 = pred_3072.permute(0, 2, 1).contiguous()
        #
        # ####################################################################################################
        # #end = pred_3072 + pred_FPnet
        #
        # end = x + end2
        # end = end.reshape(-1, 24)
        # end2 = pred_FPnet.reshape(-1, 24, 1, 128)
        # pre_4096 = self.mlp_conv_end(end)
        # pre_4096 = pre_4096.permute(0, 2, 1).contiguous()
        # # pc2_xyz = pc1_xyz_expand + pc2_xyz  # [8,64，2,3]  = [8,64，2,3] +  [8,64,1，3]
        # # catend=torch.cat([pred,point_1024],1)
        # #####################################################################################################



        return pred,pred

class PMPNetPlus(nn.Module):
    def __init__(self, dataset='Completion3D', dim_tail=32):
        super(PMPNetPlus, self).__init__()
        self.step_1 = StepModelTransformer(step=1, if_noise=True, dim_tail=dim_tail)
        self.step_2 = StepModelTransformer(step=2, if_noise=True, dim_tail=dim_tail)
        self.step_3 = StepModelTransformer(step=3, if_noise=True, dim_tail=dim_tail)

    def forward(self, point_cloud, gt_ones_4096,  gt_ones_4224):
        """
        Args:
            point_cloud: Tensor, (B, 2048, 3)
        """
        b, npoint, _ = point_cloud.shape
        device = point_cloud.device



        pcd_out_1 = self.step_1(point_cloud, gt_ones_4096,  gt_ones_4224)

        # pcd_out_2, delta2 = self.step_2(pcd_out_1)
        # pcd_out_3, delta3 = self.step_3(pcd_out_2)

        return pcd_out_1