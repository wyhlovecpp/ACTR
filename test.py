r'''
    modified test script of CATs
    https://github.com/SunghwanHong/Cost-Aggregation-transformers
'''

import argparse
import os
import pickle
import random
import time
from os import path as osp
import numpy as np
import torch
import torch.nn as nn
from termcolor import colored
from torch.utils.data import DataLoader
from models.ACTR import ACTR
import utils_training.optimize as optimize
from utils_training.evaluation import Evaluator
from utils_training.utils import log_args
from data import download


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.multiprocessing.set_sharing_strategy('file_system')
    # Argument parsing
    parser = argparse.ArgumentParser(description='CATs Test Script')
    # Paths
    parser.add_argument('--name_exp', type=str,
                        default=time.strftime('%Y_%m_%d_%H_%M'),
                        help='name of the experiment to save')
    parser.add_argument('--snapshots1', type=str, default='./eval')
    parser.add_argument('--pretrained', dest='pretrained',
                       help='path to pre-trained model')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=0,
                        help='number of parallel threads for dataloaders')
    parser.add_argument('--seed', type=int, default=2021,
                        help='Pseudo-RNG seed')
    parser.add_argument('--datapath', type=str, default='./SC_Dataset')
    parser.add_argument('--benchmark', type=str, choices=['pfpascal', 'spair', 'pfwillow'])
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--ibot_ckp_file', type=str, default='./checkpoint.pth')

    # Seed
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(args.alpha)

    # Initialize Evaluator
    Evaluator.initialize(args.benchmark, args.alpha)
    
    with open(osp.join(args.pretrained, 'args.pkl'), 'rb') as f:
        args_model = pickle.load(f)
    log_args(args_model)
    
    # Dataloader
    download.download_dataset(args.datapath, args.benchmark)
    test_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'test', False, args_model.feature_size)
    test_dataloader = DataLoader(test_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_threads,
        shuffle=False)

    model = ACTR(
        ibot_ckp_file=args.ibot_ckp_file, feature_size=args_model.feature_size, depth=args_model.depth, num_heads=args_model.num_heads
        , mlp_ratio=args_model.mlp_ratio, freeze=True)
    if args.pretrained:
        checkpoint = torch.load(osp.join(args.pretrained, 'model_best.pth'))
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise NotImplementedError()
    # create summary writer

    # model = nn.DataParallel(model)
    model = model.to(device)

    train_started = time.time()

    val_loss_grid, val_mean_pck = optimize.test_epoch(model,
                                                    test_dataloader,
                                                    device,
                                                    epoch=0)
    print(colored('==> ', 'blue') + 'Test average grid loss :',
            val_loss_grid)
    print('mean PCK is {}'.format(val_mean_pck))

    print(args.seed, 'Test took:', time.time()-train_started, 'seconds')
