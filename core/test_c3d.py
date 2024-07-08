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

TxtLine_testloss= './output_draw/testL2.txt'

def  savepoint(epoch,name, data):
    savepath = './output_draw/'
    if(os.path.exists(savepath) == False):
        os.makedirs(savepath)

    if len(data.shape) == 4:
        data = data.squeeze(1)

    Newdata = data[0].cpu().detach().numpy()
    np.savetxt(savepath + str(epoch) + name + '.txt', Newdata, fmt="%0.3f")


def WriteTxtLine(filename , context):

    with open(filename, 'a+',encoding='utf-8') as fp:
        fp.write('\n'+ context )
        fp.close()



def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None):
    # Enable the inbuilt cudnn auto-tuner pre_whole512,pre_whole1024,pre_whole2048,pre_drop64, pre_drop128,pre_drop1536=to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        # Set up data loader
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetSubset.TEST),
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
    loss_L1_arr = []
    loss_L2_arr = []
    loss_Fscore_arr = []
    loss_EMD_arr = []


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

                loss_L1 = get_lossl1l2(pre_whole2048, gt, sqrt=True)  # L1= True  L2 = False = CD
                loss_L2 = get_lossl1l2(pre_whole2048, gt, sqrt=False)  # L1= True  L2 = False = CD
                loss_L1_arr.append(float(loss_L1))
                loss_L2_arr.append(float(loss_L2))


                ###############################################
                # ############################
                ###  F_score
                for batch_i in range(gt.shape[0]):
                    _metrics = Metrics.get((pre_whole2048.detach())[batch_i], (gt.detach())[batch_i])
                    loss_Fscore_arr.append(_metrics[0])
                ############################
                ###  EMD
                dist, _ = EMD(pre_whole2048, gt, 0.005, 50)
                # dist, _ = EMD(pre_gt, gt, 0.005, 50)
                emd1 = torch.sqrt(dist).mean()
                loss_EMD_arr.append(emd1)
                #######################################################################################
                # # ## 测试的是时候， 才会使用, choose one file
                if model_idx <= 30 :
                    savepoint(model_id,'_gt_input', partial_input)
                    savepoint(model_id, '_bilinear',partial_bilinear)
                    savepoint(model_id, '_gt', gt)
                    savepoint(model_id, '_gt_drop1536',gt_drop1536)
                    savepoint(model_id, '_pregt',pre_whole2048)
                WriteTxtLine(TxtLine_testloss, model_id + " " +str(loss_L2))
                #######################################################################################


    loss_L1_arravg = sum(loss_L1_arr) / len(loss_L1_arr)  # 所有的损失 / 摊平到每一个上
    loss_L2_arravg = sum(loss_L2_arr) / len(loss_L2_arr)  # 所有的损失 / 摊平到每一个上

    loss_Fscore_arravg = sum(loss_Fscore_arr) / len(loss_Fscore_arr)  # 所有的损失 / 摊平到每一个上
    loss_EMD_arravg = sum(loss_EMD_arr) / len(loss_EMD_arr)  # 所有的损失 / 摊平到每一个上
    print(loss_L1_arravg, loss_L2_arravg, 'Fscore', loss_Fscore_arravg, 'EMD', loss_EMD_arravg)


