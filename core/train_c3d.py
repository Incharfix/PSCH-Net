# -*- coding: utf-8 -*-
# @Author: XP

import logging
import os
import numpy as np
import torch
import utils.data_loaders
import utils.helpers
from datetime import datetime
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter
from core.val_c3d import val_net
from utils.average_meter import AverageMeter
from torch.optim.lr_scheduler import StepLR
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils import get_loss

from models.model import SnowflakeNet as Model
import arguments
from Chamfer3D_L1L2.loss_utils import get_loss  as get_lossl1l2
import Code_treegan.train_treegan as file_treegan
from models.utils import fps_subsample
from shape_utils import random_occlude_pointcloud as crop_shape






TxtLine_trainloss= './trainloss.txt'

def WriteTxtLine(filename , context):

    with open(filename, 'a+',encoding='utf-8') as fp:
        fp.write('\n'+ context )
        fp.close()

def  saveFPpoint(epoch,name, data):
    savepath = './output_train/'

    if len(data.shape) == 4:
        data = data.squeeze(1)

    Newdata = data[0].cpu().detach().numpy()
    np.savetxt(savepath + str(epoch) + name + '.txt', Newdata, fmt="%0.3f")


def find_nearby_points(points, reference_point, num_neighbors=5):
    distances = torch.norm(points - reference_point, dim=1)
    nearest_indices = torch.topk(distances, num_neighbors + 1, largest=False).indices
    return points[nearest_indices[1:]]  # Exclude the reference point itself
def add_new_points(reference_point, nearby_points, num_new_points=3):
    mean_position = torch.mean(nearby_points, dim=0)
    direction_vector = mean_position - reference_point
    new_points = [reference_point + i * direction_vector for i in range(1, num_new_points + 1)]
    return torch.stack(new_points)

def add_noise_to_point_cloud(points, num_neighbors=2, num_new_points=1):
    all_new_points = []
    for reference_point in points:
        nearby_points = find_nearby_points(points, reference_point, num_neighbors)
        new_points = add_new_points(reference_point, nearby_points, num_new_points)
        all_new_points.append(new_points)
    return torch.cat(all_new_points)


def addTorchNoise(Batchpoint_cloud):  # （B，512，3）  ---》 （B,1536,3）

    arraytorch = []
    for i in range(Batchpoint_cloud.shape[0]):

        noisy_p1 = add_noise_to_point_cloud(Batchpoint_cloud[i],num_neighbors=2, num_new_points=1)
        noisy_p2 = add_noise_to_point_cloud(Batchpoint_cloud[i], num_neighbors=3, num_new_points=1)
        noisy_p3 = add_noise_to_point_cloud(Batchpoint_cloud[i], num_neighbors=4, num_new_points=1)

        onenoise=torch.cat([noisy_p1,noisy_p2,noisy_p3],0)
        onenoise=onenoise.unsqueeze(0)
        arraytorch.append(onenoise)

    all_noise=torch.cat(arraytorch, 0)
    all_noise_1024 = fps_subsample(all_noise, 1024)
    end =torch.cat([all_noise_1024,Batchpoint_cloud] ,1)

    return end



def splitpoint(gt):
    ##########################################################
    ####################################################
    # start_time = time.time()
    batch_size=gt.shape[0]

    num_holes = 1
    crop_point_num = 512
    context_point_num = 512
    N = 2048
    points = torch.squeeze(gt, 1)
    points = points.cpu()  # tensor  CPU 形式
    partials = []
    fine_gts, interm_gts = [], []
    N_partial_points = N - (crop_point_num * num_holes)
    centroids = np.asarray(
        [[1, 0, 0], [0, 0, 1], [1, 0, 1], [-1, 0, 0], [-1, 1, 0]])

    for m in range(batch_size):
        # partial, fine_gt, interm_gt = crop_shape(
        partial, fine_gt = crop_shape(
            points[m],
            centroids=centroids,
            scales=[crop_point_num, (crop_point_num + context_point_num)],
            n_c=num_holes
        )

        if partial.shape[0] > N_partial_points:
            assert num_holes > 1
            # sampling without replacement
            choice = torch.randperm(partial.size(0))[:N_partial_points]
            partial = partial[choice]

        partials.append(partial)
        fine_gts.append(fine_gt)
        # interm_gts.append(interm_gt)

    gt_crop_dense = partials = torch.stack(partials).cuda()  # [B,  N-512 ，3,]
    gt_drop = fine_gts = torch.stack(fine_gts).cuda()  # [B, 512, 3]
    # interm_gts = torch.stack(interm_gts).to(device)  # [B, 1024, 3]  # 暂时 不用
    gt = gt.cuda()
    return gt,gt_crop_dense,gt_drop



def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN),
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    collate_fn=utils.data_loaders.collate_fn,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=False)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.VAL),
                                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                                  num_workers=cfg.CONST.NUM_WORKERS//2,
                                                  collate_fn=utils.data_loaders.collate_fn,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up folders for logs and checkpoints
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat())
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    model = Model()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # Create the optimizers
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                       lr=cfg.TRAIN.LEARNING_RATE,
                                       weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                       betas=cfg.TRAIN.BETAS)

    # lr scheduler
    scheduler_steplr = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
                                          after_scheduler=scheduler_steplr)

    init_epoch = 0
    best_metrics = float('inf')
    steps = 0

    #if 'WEIGHTS' in cfg.CONST:
        #logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        #checkpoint = torch.load(cfg.CONST.WEIGHTS)
        #best_metrics = checkpoint['best_metrics']
        #model.load_state_dict(checkpoint['model'])
        #logging.info('Recover complete. Current epoch = #%d; best metrics = %s.' % (init_epoch, best_metrics))

    _args = arguments.parse_args()
    print(_args)

    if _args.gpu_list[0] >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = _args.gpu_environ
        strTemp = 'cuda:' + str(_args.gpu_list[0])
        strTemp = 'cuda:0'
        device = torch.device(strTemp) if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')

    G_treegan,D_treegan,optimizerG_treegan,optimizerD_treegan,GP,knn_loss,w_train_ls = file_treegan.InitTreeGan(_args,device)




    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        model.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)

        w_train = w_train_ls[min(3, int(epoch_idx / 500))]

        with tqdm(train_data_loader) as t:
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):
                data_time.update(time() - batch_end_time)
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # partial
                partial_dense = data['partial_dense']         # (B,512,3)
                partial_input = data['partial_input']         # (B,128,3)
                partial_bilinear = data['partial_bilinear']   #  (B,512,3)

                # drop
                gt_drop1536 = data['drop_dense']                     #  (B,1536,3)
                gt_drop128 = fps_subsample(gt_drop1536, 128) #  (B,384,3)
                gt_drop64 = fps_subsample(gt_drop1536, 64)  # (B,384,3)

                # whole
                gt = torch.cat([partial_dense, gt_drop1536], 1)   #  (B,2048,3)
                gt_whole1024 = data["gt_half"]                               #  (B,1024,3)
                gt_whole512 = fps_subsample(gt, 512)   #  (B,512,3)


                # gt, gt_crop_dense, gt_drop512 = splitpoint(gt)  # # gt = gt_crop_dense + gt_drop

                ########################################################
                # 没有gan时，完整的
                pre_whole512,pre_whole1024,pre_whole2048,pre_drop64, pre_drop128,pre_drop1536= model(partial_input,partial_bilinear)  # (B,512,3)   (B,1024,3)

                loss_whole512 = get_lossl1l2(pre_whole512,    gt_whole512, sqrt=False)  # False True
                loss_whole1024 = get_lossl1l2(pre_whole1024,  gt_whole1024, sqrt=False)  # False True
                loss_whole2048 = get_lossl1l2(pre_whole2048,  gt,           sqrt=False)  # False True
                loss_drop64 = get_lossl1l2(pre_drop64,          gt_drop64, sqrt=False)  # False True
                loss_drop128 = get_lossl1l2(pre_drop128,       gt_drop128, sqrt=False)  # False True
                loss_drop512 = get_lossl1l2(pre_drop1536,     gt_drop1536, sqrt=False)  # False True



                loss = loss_drop128  + loss_drop64 + loss_drop512*2 + loss_whole512 + loss_whole1024 + loss_whole2048

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

               ############################################################

                if steps <= cfg.TRAIN.WARMUP_STEPS:
                    lr_scheduler.step()
                    steps += 1



        lr_scheduler.step()

        # Validate the current model
        avgepoch_L_CD_true,cd_eval = val_net(cfg, epoch_idx, val_data_loader, val_writer, model)
        WriteTxtLine(TxtLine_trainloss, str(epoch_idx) + " " + str(cd_eval))
        print(epoch_idx, 'Loss  drop512,whole:', avgepoch_L_CD_true,cd_eval)


        if  epoch_idx % 20 == 0 :
            saveFPpoint(epoch_idx, '_input', partial_input)
            saveFPpoint(epoch_idx, '_input_dense', partial_dense)
            saveFPpoint(epoch_idx, '_input_bilinear', partial_bilinear)
            saveFPpoint(epoch_idx, '_gt_drop1536', gt_drop1536)
            saveFPpoint(epoch_idx, '_predrop1536', pre_drop1536)
            saveFPpoint(epoch_idx, '_pregt', pre_whole2048)



        # Save checkpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
            file_name = 'ckpt-best.pth' if cd_eval < best_metrics else 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': best_metrics,
                'model': model.state_dict()
            }, output_path)

            logging.info('Saved checkpoint to %s ...' % output_path)
            if cd_eval < best_metrics:
                best_metrics = cd_eval



    train_writer.close()
    val_writer.close()
