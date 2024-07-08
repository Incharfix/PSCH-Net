# -*- coding: utf-8 -*-
# @Author: XP

import logging
import torch
import utils.data_loaders
import utils.helpers
from tqdm import tqdm

from utils.average_meter import AverageMeter
# from utils.metrics import Metrics
from utils.loss_utils import get_loss
from models.model import SnowflakeNet as Model
import os.path
import numpy as np
from  A3_FPS_Npoint2Npoint import sample_and_group as FPS
from shape_utils import random_occlude_pointcloud as crop_shape
from models.utils import fps_subsample
from utils_F.metrics import Metrics
from Chamfer3D_L1L2.loss_utils import chamfer_3DDist
from Chamfer3D_L1L2.loss_utils import get_loss  as get_lossl1l2
import sys
sys.path.append("./emd/")
import emd_module as emd



EMD = emd.emdModule()

TxtLine_testloss= './testloss.txt'

def savetxt(idx_test,partial,pcds):

    savepath = './output_draw'
    if (os.path.exists(savepath) == False):
        os.makedirs(savepath)
    partial_temp = torch.squeeze(partial.cpu())
    np.savetxt(savepath + '/'+ str(idx_test) + '_1partial.txt', partial_temp.numpy(),fmt='%1.5f')

    pcds_temp = pcds
    pcds_temp_1 = torch.squeeze(pcds_temp[3].cpu())
    np.savetxt(savepath + '/'+ str(idx_test) + '_2fake.txt', pcds_temp_1.numpy(),fmt='%1.5f')







def val_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None):
    # Enable the inbuilt cudnn auto-tuner pre_whole512,pre_whole1024,pre_whole2048,pre_drop64, pre_drop128,pre_drop1536=to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        # Set up data loader
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetSubset.VAL),
                                                       batch_size=1,
                                                       collate_fn=utils.data_loaders.collate_fn,
                                                       pin_memory=True,
                                                       shuffle=False)

    # Setup networks and initialize networks
    if model is None:
        model = Model(dim_feat=512, up_factors=[2, 2], radius=1)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])

    # Switch models to evaluation mode
    model.eval()

    n_samples = len(test_data_loader)
    test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()



    # Testing loop
    idx_test = 0
    Dloss_L_CD_true = []
    Dloss_L_CD_false = []
    Closs_Fscore = []
    Closs_EMD = []


    with tqdm(test_data_loader) as t:
        for model_idx, (taxonomy_id, model_id, data) in enumerate(t):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            model_id = model_id[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                    #
                # partial
                partial_dense = data['partial_dense']  # (B,512,3)
                partial_input = data['partial_input']  # (B,128,3)
                partial_bilinear = data['partial_bilinear']  # (B,512,3)

                # drop
                gt_drop1536 = data['drop_dense']  # (B,1536,3)

                # whole
                gt = torch.cat([partial_dense, gt_drop1536], 1)  # (B,2048,3)

                    ########################################################
                    # 没有gan时，完整的
                pre_whole512,pre_whole1024,pre_whole2048,pre_drop64, pre_drop128,pre_drop1536 = model(partial_input,
                                                                                                  partial_bilinear)  # (B,512,3)   (B,1024,3)


                loss_whole2048 = get_lossl1l2(pre_whole2048, gt, sqrt=False)  # L1= True  L2 = False = CD
                loss_drop512 = get_lossl1l2(pre_drop1536, gt_drop1536, sqrt=False)  # L1= True  L2 = False = CD

                Dloss_L_CD_true.append(float(loss_drop512))
                Dloss_L_CD_false.append(float(loss_whole2048))


                # WriteTxtLine(TxtLine_testloss, str(epoch_idx) + " " +str(loss_whole2048))

            # ###############################################
                # # ############################
                # ###  F_score
                # for batch_i in range(gt.shape[0]):
                #     _metrics = Metrics.get((pcds_pred[3].detach())[batch_i], (gt.detach())[batch_i])
                #     Closs_Fscore.append(_metrics[0])
                # ############################
                # ###  EMD
                # dist, _ = EMD(pcds_pred[3], gt, 0.002, 10000)
                # # dist, _ = EMD(pre_gt, gt, 0.005, 50)
                # emd1 = torch.sqrt(dist).mean()
                # Closs_EMD.append(emd1)
                # #######################################################################################
                # # # ## 测试的是时候， 才会使用, choose one file

                # if idx_test< 2 :
                #     savetxt(model_id, partial, pcds_pred)
                # idx_test = idx_test + 1
                ########################################################################################






    avgepoch_L_CD_true = sum(Dloss_L_CD_true) / len(Dloss_L_CD_true)  # 所有的损失 / 摊平到每一个上
    avgepoch_L_CD_false = sum(Dloss_L_CD_false) / len(Dloss_L_CD_false)  # 所有的损失 / 摊平到每一个上

    # avgCloss_Fscore = sum(Closs_Fscore) / len(Closs_Fscore)  # 所有的损失 / 摊平到每一个上
    # avgCloss_EMD = sum(Closs_EMD) / len(Closs_EMD)  # 所有的损失 / 摊平到每一个上
    # print(avgepoch_L_CD_true, avgepoch_L_CD_false, 'Fscore', avgCloss_Fscore, 'EMD', avgCloss_EMD)

    return avgepoch_L_CD_true,avgepoch_L_CD_false
