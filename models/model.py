#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Xiang

import torch
import torch.nn as nn
from models.utils import PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, Transformer
from models.skip_transformer import SkipTransformer
from modelsP2.transformer import Transformer
from modelsP2.utils import Conv1d, PointNet_FP_Module, PointNet_SA_Module
import torch.nn.functional as F



class PFdropBatchFeature(nn.Module):
    def  __init__( self ):
        super(PFdropBatchFeature,self).__init__()
        self.crop_point_num = 1536

        # self.fc1 = nn.Linear(1920, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4_1 = nn.Linear(256, 128)
        self.fc4_2 = nn.Linear(128, 64)
        # self.fc4_3 = nn.Linear(64, 24)

        self.fc1_1 = nn.Linear(1024, 128 * 512)
        self.fc2_1 = nn.Linear(512, 64 * 128)  # nn.Linear(512,64*256) !
        self.fc3_1 = nn.Linear(256, 64 * 3)

        self.conv1_1 = torch.nn.Conv1d(512, 512, 1)  # torch.nn.Conv1d(256,256,1) !
        self.conv1_2 = torch.nn.Conv1d(512, 256, 1)
        self.conv1_3 = torch.nn.Conv1d(256, int((self.crop_point_num * 3) / 128), 1)
        self.conv2_1 = torch.nn.Conv1d(128, 6, 1)  # torch.nn.Conv1d(256,12,1) !

    def forward(self,point2F_l3):
        x_1        = point2F_l3
        # x_1 = F.relu(self.fc1(x)) #1024
        x_2 = F.relu(self.fc2(x_1)) # 
        x_3 = F.relu(self.fc3(x_2))  #256

        x_4 = F.relu(self.fc4_1(x_3))  #128
        LineF_64 = F.relu(self.fc4_2(x_4))  #64
        LineF_64 = torch.unsqueeze(LineF_64, 2)
        # LineF24 = F.relu(self.fc4_3(x_4))

        pc1_feat = self.fc3_1(x_3)
        pc1_xyz = pc1_feat.reshape(-1,64,3) #64x3 center1

        pc2_feat = F.relu(self.fc2_1(x_2))
        pc2_feat = pc2_feat.reshape(-1,128,64)
        pc2_xyz =self.conv2_1(pc2_feat) #6x64 center2

        pc3_feat = F.relu(self.fc1_1(x_1))
        pc3_feat = pc3_feat.reshape(-1,512,128)
        pc3_feat = F.relu(self.conv1_1(pc3_feat))
        pc3_feat = F.relu(self.conv1_2(pc3_feat))
        pc3_xyz = self.conv1_3(pc3_feat) #12x128 fine

        pc1_xyz_expand = torch.unsqueeze(pc1_xyz,2)
        pc2_xyz = pc2_xyz.transpose(1,2)
        pc2_xyz = pc2_xyz.reshape(-1,64,2,3)
        pc2_xyz = pc1_xyz_expand+pc2_xyz
        pc2_xyz = pc2_xyz.reshape(-1,128,3)

        pc2_xyz_expand = torch.unsqueeze(pc2_xyz,2)
        pc3_xyz = pc3_xyz.transpose(1,2)
        pc3_xyz = pc3_xyz.reshape(-1,128,int(self.crop_point_num/128),3)
        pc3_xyz = pc2_xyz_expand+pc3_xyz
        pc3_xyz = pc3_xyz.reshape(-1,self.crop_point_num,3)

        return  pc1_xyz, pc2_xyz,pc3_xyz



class FeatureExtractor(nn.Module):
    def __init__(self, out_dim=1024):
        """Encoder that encodes information of partial point cloud
        """
        super(FeatureExtractor, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = Transformer(256, dim=64)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)
        self.fc1 = nn.Linear(512, 1024)
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


        line_1024 = torch.squeeze(l3_points,2)  # 1
        line_1024 = F.relu(self.fc1(line_1024)) #1024

        return line_1024,l3_points

class StepModel(nn.Module):
    def __init__(self, step=1):
        super(StepModel, self).__init__()
        self.step = step

        self.sa_module_1 = PointNet_SA_Module(512, 32, 0.2, 3, [64, 64, 128], group_all=False)
        self.transformer_start_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointNet_SA_Module(128, 32, 0.4, 128, [128, 128, 256], group_all=False)
        self.transformer_start_2 = Transformer(256, dim=64)
        self.sa_module_3 = PointNet_SA_Module(None, None, None, 256, [256, 512, 1024], group_all=True)


        self.fp_module_3 = PointNet_FP_Module(1024, [256, 256], use_points1=True, in_channel_points1=256)
        self.fp_module_2 = PointNet_FP_Module(256, [256, 128], use_points1=True, in_channel_points1=128)
        self.fp_module_1 = PointNet_FP_Module(128, [128, 128, 128], use_points1=True, in_channel_points1=6)

        mlp = [128, 64, 3]
        last_channel = 128 + 32
        mlp_conv = []
        for out_channel in mlp[:-1]:
            mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=True))
            last_channel = out_channel
        mlp_conv.append(Conv1d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*mlp_conv)


        self.conv1 = torch.nn.Conv1d(2,1,1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, point_cloud,F_last):
        device = point_cloud.device
        l0_xyz = point_cloud
        l0_points = point_cloud


        # pcd = fps_subsample(point_cloud, 1024)

        l1_xyz, l1_points = self.sa_module_1(l0_xyz, l0_xyz)  # (B, 3, 512), (B, 128, 512)
        l1_points = self.transformer_start_1(l1_points, l1_xyz)
        l2_xyz, l2_points = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        l2_points = self.transformer_start_2(l2_points, l2_xyz)

        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, 1024, 1)

        if F_last ==None :
              pass
        else:
            latentfeature = torch.cat([l3_points,F_last],2)
            latentfeature = latentfeature.transpose(1,2)
            latentfeature = F.relu(self.bn1(self.conv1(latentfeature)))
            l3_points = latentfeature.transpose(1, 2)


        l2_points = self.fp_module_3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp_module_2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp_module_1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)

        b, _, n = l0_points.shape
        noise = torch.normal(mean=0, std=torch.ones((b, 32, n), device=device))
        delta_xyz = torch.tanh(self.mlp_conv(torch.cat([l0_points, noise], 1))) * 1.0 / 10 ** (self.step - 1)
        point_cloud = point_cloud + delta_xyz

        return l3_points,point_cloud

#   Point++
class PPP(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256, num_p0=512, radius=1, up_factors=None):
        super(PPP, self).__init__()

        self.step_1 = StepModel(step=1)
        self.step_2 = StepModel(step=1)
        self.step_3 = StepModel(step=1)

    def forward(self, point_cloud): # (B,2048,3)  fps   (B,XX,3)

        point_512 = fps_subsample( point_cloud  , 512)
        point_1024 = fps_subsample(point_cloud  , 1024)
        point_2048 = point_cloud

        point_512 = point_512.permute(0, 2, 1).contiguous()
        point_1024 = point_1024.permute(0, 2, 1).contiguous()
        point_2048 = point_2048.permute(0, 2, 1).contiguous()

        F512, whole512 = self.step_1(point_512, None)
        F1024, whole1024 = self.step_2(point_1024, F512)
        _, whole2048 = self.step_3(point_2048, F1024)

        whole512 = whole512.permute(0, 2, 1)
        whole1024 = whole1024.permute(0, 2, 1)
        whole2048 = whole2048.permute(0, 2, 1)

        return whole512,whole1024,whole2048




class _netG_1536(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256, num_p0=512, radius=2, up_factors=None):
        super(_netG_1536, self).__init__()
        self.feat_extractor1 = FeatureExtractor(out_dim=dim_feat)
        self.feat_extractor2 = FeatureExtractor(out_dim=dim_feat)
        self.feat_extractor3 = FeatureExtractor(out_dim=dim_feat)

        # self.decoder = Decoder(dim_feat=dim_feat, num_pc=num_pc, num_p0=num_p0, radius=radius, up_factors=up_factors)
        self.PFdropBF = PFdropBatchFeature()
        self.conv1 = torch.nn.Conv1d(3,1,1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self,  input_point, upinput,return_P0=False):
        pcd_bnc = input_point

        # pcd =
        # point_cloud1024 = torch.cat([pcd,input_point],1)

        point_cloud512 = input_point.permute(0, 2, 1).contiguous()
        point_cloud1024 = fps_subsample(upinput, 1024).permute(0, 2, 1).contiguous()
        point_cloud1536 = upinput.permute(0, 2, 1).contiguous()

        #
        F_10241, F_512 = self.feat_extractor1(point_cloud512)  # (B,1024) (B,512)
        F_10242, F_512 = self.feat_extractor2(point_cloud1024)
        F_10243, F_512 = self.feat_extractor3(point_cloud1536)

        F_10241 = torch.unsqueeze(F_10241, 2)  # 1
        F_10242 = torch.unsqueeze(F_10242, 2)  # 1
        F_10243 = torch.unsqueeze(F_10243, 2)  # 1

        latentfeature = torch.cat([F_10241, F_10242,F_10243], 2)
        latentfeature = latentfeature.transpose(1, 2)
        latentfeature = F.relu(self.bn1(self.conv1(latentfeature)))
        F_1024 = latentfeature.transpose(1, 2)
        F_1024 = torch.squeeze(F_1024, 2)  # 1


        # 
        pre_drop64, pre_drop128,pre_drop1536 = self.PFdropBF(F_1024)  # （16，64，1） （16，512，3）

        return pre_drop64, pre_drop128,pre_drop1536 # ,  pre_drop




class SnowflakeNet(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256, num_p0=512, radius=1, up_factors=None):
        super(SnowflakeNet, self).__init__()

        self.my_netG_1536=_netG_1536()

        self.my_ppp= PPP()

    def forward(self, input_point, upinput): # (B,2048,3)  fps   (B,XX,3)
        pre_drop64, pre_drop128,pre_drop1536 =self.my_netG_1536(input_point, upinput)
        whole=torch.cat([upinput,pre_drop1536],1)
        whole512,whole1024,whole2048=self.my_ppp(whole)

        return whole512,whole1024,whole2048,pre_drop64, pre_drop128,pre_drop1536

