
import argparse

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--dataroot', default='../data_our_all/TrainPoint', help='path to dataset')
    parser.add_argument('--gpu_environ', default='0', type=str, metavar='DEVICE', help='GPU to use, ')
    parser.add_argument('--gpu_list', default=[0], type=list, metavar='DEVICE', help='GPU to use, ')  # gpu_list  gpu_ids
    parser.add_argument('--niter', type=int, default=301, help='number of epochs to train for')  # default=201
    parser.add_argument('--batchSize', type=int, default=24, help='size')  

    parser.add_argument('--pnum', type=int, default=2048, help=' 送到网络中数量/原始文件中点数  2048 4096')
    parser.add_argument('--point_scales_list', type=list, default=[2048, 1024, 512],
                        help='number of points in each scales')



    parser.add_argument('--crop_point_num', type=int, default=1536, help='0 means do not use else use with this weight')
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--TxtLineA', default='./output_train/WriteTxtALoss.txt', type=str, help='损失 画曲线的地方')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')

    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--cuda', type=bool, default=False, help='enables cuda')
    parser.add_argument('--D_choose', type=int, default=1, help='0 not use D-net,1 use D-net')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--num_scales', type=int, default=3, help='number of scales')

    parser.add_argument('--each_scales_size', type=int, default=1, help='each scales size')
    parser.add_argument('--wtl2', type=float, default=0.95, help='0 means do not use else use with this weight')
    parser.add_argument('--cropmethod', default='random_center', help='random|center|random_center')





    ##############################################
    ###  tree gan
    ##############################################

    ### general training related

    parser.add_argument('--lr', type=float, default=1e-4, help='Float value for learning rate.')
    parser.add_argument('--lambdaGP', type=int, default=10, help='Lambda for GP term.')

    parser.add_argument('--D_iter', type=int, default=3, help='Number of iterations for discriminator.')
    parser.add_argument('--w_train_ls', type=float, default=[1], nargs='+', help='train loss weightage')

    ### uniform losses related
    # PatchVariance
    parser.add_argument('--knn_loss', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--knn_k', type=int, default=30)
    parser.add_argument('--knn_n_seeds', type=int, default=100)
    parser.add_argument('--knn_scalar', type=float, default=0.2)
    # PU-Net's uniform loss
    parser.add_argument('--krepul_loss', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--krepul_k', type=int, default=10)
    parser.add_argument('--krepul_n_seeds', type=int, default=20)
    parser.add_argument('--krepul_scalar', type=float, default=1)
    parser.add_argument('--krepul_h', type=float, default=0.01)
    # MSN's Expansion-Penalty
    parser.add_argument('--expansion_penality', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--expan_primitive_size', type=int, default=2)
    parser.add_argument('--expan_alpha', type=float, default=1.5)
    parser.add_argument('--expan_scalar', type=float, default=0.1)


    ### TreeGAN architecture related
    parser.add_argument('--DEGREE', type=int, default=[1, 2, 2, 2, 2, 2, 64], nargs='+',
                              help='Upsample degrees for generator.')
    parser.add_argument('--G_FEAT', type=int, default=[96, 256, 256, 256, 128, 128, 128, 3], nargs='+',
                              help='Features for generator.')
    parser.add_argument('--D_FEAT', type=int, default=[3, 64, 128, 256, 256, 512], nargs='+',
                              help='Features for discriminator.')
    parser.add_argument('--support', type=int, default=10, help='Support value for TreeGCN loop term.')
    parser.add_argument('--loop_non_linear', default=False, type=lambda x: (str(x).lower() == 'true'))



    ### ohters
    parser.add_argument('--ckpt_path0', type=str, default='./checkpoints', help='Checkpoint path.')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/chair', help='Checkpoint path.')
    parser.add_argument('--ckpt_save', type=str, default='tree_ckpt_', help='Checkpoint name to save.')
    parser.add_argument('--eval_every_n_epoch', type=int, default=25, help='0 means never eval')
    parser.add_argument('--save_every_n_epoch', type=int, default=100, help='save models every n epochs')




    return parser.parse_args()