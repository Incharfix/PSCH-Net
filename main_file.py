#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Xiang

import argparse
import logging
import os
import numpy as np
import sys
import torch
from pprint import pprint
from config_c3d import cfg
from core.train_c3d import train_net
from core.test_c3d import test_net
from core.val_c3d import val_net


#  测试集
'''
--test
'''


# 验证集 (在训练集中使用， 一般不手动添加)
'''
--val
'''




TxtLine_trainloss= './trainloss.txt'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of SnowflakeNet')
    parser.add_argument('--val', dest='val', help='Inference for benchmark', action='store_true')
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')

    args = parser.parse_args()

    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()
    print('cuda available ', torch.cuda.is_available())

    # Print config
    print('Use config:')
    pprint(cfg)

    if not args.test and not args.val:
        train_net(cfg)
    else:
        if cfg.CONST.WEIGHTS is None:
            raise Exception('Please specify the path to checkpoint in the configuration file!')

        if args.val:
            val_net(cfg)
        else:
            test_net(cfg)

if __name__ == '__main__':
    # Check python version
    seed = 1
    set_seed(seed)
    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)
    main()
