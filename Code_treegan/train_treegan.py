import torch
import torch.nn as nn
import torch.optim as optim

from Code_treegan.data.CRN_dataset import CRNShapeNet
from Code_treegan.model.treegan_network import Generator, Discriminator
from Code_treegan.model.gradient_penalty import GradientPenalty
from Code_treegan.evaluation.FPD import calculate_fpd

import time
import numpy as np
from Code_treegan.loss import *
from Code_treegan.metrics import *

import os.path as osp
from Code_treegan.eval_treegan import checkpoint_eval


import os




def InitTreeGan(args,device):
    ### Model
    #  NewModel网络 +  New优化器 + 惩罚梯度？
    G_treegan = Generator(features=args.G_FEAT, degrees=args.DEGREE, support=args.support, args=args).to(device)
    D_treegan = Discriminator(features=args.D_FEAT).to(device)

    optimizerG = optim.Adam(G_treegan.parameters(), lr=args.lr, betas=(0, 0.99))
    optimizerD = optim.Adam(D_treegan.parameters(), lr=args.lr, betas=(0, 0.99))
    GP = GradientPenalty(args.lambdaGP, gamma=1, device=device)  #


    G_treegan = nn.DataParallel(G_treegan, device_ids=args.gpu_list)  # @inchar 分配到多卡上
    D_treegan = nn.DataParallel(D_treegan, device_ids=args.gpu_list)

    ######################################################################
    ### uniform losses  统一
    if args.expansion_penality:  # MSN   default=False
        expansion = expansionPenaltyModule()

    if args.krepul_loss:  # PU-net      default=False
        krepul_loss = kNNRepulsionLoss(k=args.krepul_k, n_seeds=args.krepul_n_seeds, h=args.krepul_h)

    if args.knn_loss:  # PatchVariance   default=True
        knn_loss = kNNLoss(k=args.knn_k,n_seeds=args.knn_n_seeds)  # args.knn_k default=30   #args.knn_n_seeds default=100
    ######################################################################

    if len(args.w_train_ls) == 1:                # default=[1]
        w_train_ls = args.w_train_ls * 4
    else:
        w_train_ls = args.w_train_ls
    ######################################################################



    return  G_treegan,D_treegan,optimizerG,optimizerD,GP,knn_loss,w_train_ls

#####################################
### 输入 FPdrop_gt  FPdrop_fake
### 输出

def RunOnebatch(args,G_treegan,D_treegan,optimizerG,optimizerD,GP,knn_loss,w_train,FPdrop_gt,FPdrop_fake,epoch_d_loss,epoch_g_loss):

    #####################################
    ## 变量 本土适应化

    z   = FPdrop_fake
    point = FPdrop_gt

    #####################################
    # -------------------- Discriminator -------------------- #
    tree = [z]
    with torch.no_grad():
        fake_point = G_treegan(tree)
    D_real, _ = D_treegan(point)
    D_fake, _ = D_treegan(fake_point)
    gp_loss = GP(D_treegan, point.data, fake_point.data)

    # compute D loss
    D_realm = D_real.mean()
    D_fakem = D_fake.mean()
    d_loss = -D_realm + D_fakem
    d_loss_gp = d_loss + gp_loss
    # times weight before backward
    d_loss *= w_train
    d_loss_gp.backward()
    optimizerD.step()

    epoch_d_loss.append(d_loss.item())

    #####################################
    # ---------------------- Generator ---------------------- #
    G_treegan.zero_grad()
    tree = [z]

    fake_point = G_treegan(tree)
    G_fake, _ = D_treegan(fake_point)

    # compute G loss
    G_fakem = G_fake.mean()
    g_loss = -G_fakem

    if args.knn_loss:
        knn_loss = knn_loss(fake_point)
        g_loss = -G_fakem + args.knn_scalar * knn_loss

    g_loss *= w_train
    g_loss.backward()
    optimizerG.step()

    epoch_g_loss.append(g_loss.item())



    return FPdrop_gt,FPdrop_fake,fake_point,epoch_d_loss,epoch_g_loss










