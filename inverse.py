r'''
    modified test script of CATs
    https://github.com/SunghwanHong/Cost-Aggregation-transformers
'''

import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
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
import matplotlib.pyplot as plt
from PIL import Image
transform = transforms.Compose([transforms.Resize((512,512)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])
inv_transform = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                         std=[1/0.229, 1/0.224, 1/0.225]),
    # transforms.ToPILImage()
])
def estimate_transform(source_points, target_points):
    # 将原图的坐标和目标图的坐标转换为齐次坐标
    source_points = np.hstack((source_points, np.ones((len(source_points), 1))))
    target_points = np.hstack((target_points, np.ones((len(target_points), 1))))

    # 使用最小二乘法求解变换矩阵
    transform_matrix, _ = np.linalg.lstsq(source_points, target_points, rcond=None)[:2]

    return transform_matrix
if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # # Argument parsing
    # parser = argparse.ArgumentParser(description='CATs Test Script')
    # # Paths
    # parser.add_argument('--name_exp', type=str,
    #                     default=time.strftime('%Y_%m_%d_%H_%M'),
    #                     help='name of the experiment to save')
    # parser.add_argument('--snapshots1', type=str, default='./eval')
    # parser.add_argument('--pretrained', dest='pretrained',
    #                     help='path to pre-trained model')
    # parser.add_argument('--batch-size', type=int, default=16,
    #                     help='training batch size')
    # parser.add_argument('--n_threads', type=int, default=0,
    #                     help='number of parallel threads for dataloaders')
    # parser.add_argument('--seed', type=int, default=2021,
    #                     help='Pseudo-RNG seed')
    # parser.add_argument('--datapath', type=str, default='./SC_Dataset')
    # parser.add_argument('--benchmark', type=str, choices=['pfpascal', 'spair', 'pfwillow'])
    # parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    # parser.add_argument('--alpha', type=float, default=0.1)
    # parser.add_argument('--ibot_ckp_file', type=str, default='./checkpoint.pth')
    #
    # args = parser.parse_args()
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # print(args.alpha)
    #
    # # Initialize Evaluator
    # Evaluator.initialize(args.benchmark, args.alpha)
    #
    # with open(osp.join(args.pretrained, 'args.pkl'), 'rb') as f:
    #     args_model = pickle.load(f)
    # log_args(args_model)
    #
    # # Dataloader
    # download.download_dataset(args.datapath, args.benchmark)
    # test_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'test', False,
    #                                      args_model.feature_size)
    # test_dataloader = DataLoader(test_dataset,
    #                              batch_size=args.batch_size,
    #                              num_workers=args.n_threads,
    #                              shuffle=False)
    #
    # model = ACTR(
    #     ibot_ckp_file=args.ibot_ckp_file, feature_size=args_model.feature_size, depth=args_model.depth,
    #     num_heads=args_model.num_heads
    #     , mlp_ratio=args_model.mlp_ratio, freeze=True)
    # if args.pretrained:
    #     checkpoint = torch.load(osp.join(args.pretrained, 'model_best.pth'))
    #     model.load_state_dict(checkpoint['state_dict'])
    # else:
    #     raise NotImplementedError()
    # # create summary writer
    #
    # # model = nn.DataParallel(model)
    # model = model.to(device)
    #
    # train_started = time.time()

    source_path = './test_data/eg1.png'
    target_path = './test_data/eg2.png'
    source_path = Image.open(source_path).convert('RGB')
    target_path = Image.open(target_path).convert('RGB')
    source = transform(source_path)
    target = transform(target_path)
    path = './vis_test/kp.npy'
    kp = np.load(path)
    image = torch.zeros_like(source)
    import cv2
    # for i in range(kp.shape[1]):
    #     image[:,kp[0][i],kp[1][i]] = source[:,int(i/512),i%512]
    tensor = torch.zeros(2, 512 * 512)
    tensor = np.zeros_like(tensor)
    for i in range(512):
        for j in range(512):
            tensor[0][i * 512 + j] = i
            tensor[1][i * 512 + j] = j
    source_data = np.zeros((512*512,1,2))
    target_data = np.zeros((512*512,1,2))
    for i in range(512*512):
        source_data[i,0,0] = tensor[0][i]
        source_data[i,0,1] = tensor[1][i]
        target_data[i,0,0] = kp[0][i]
        target_data[i,0,1] = kp[1][i]
    source = source.numpy().transpose(1,2,0)
    target = target.numpy().transpose(1,2,0)
    transform_matrix, _ = cv2.findHomography(source_data, target_data, cv2.RANSAC, 5.0)
    transformed = cv2.warpPerspective(target, transform_matrix, (target.shape[1], target.shape[0]))
    import torch.nn.functional as F

    # 将转换后的图像覆盖到目标图像上
    result = transformed
    result = torch.from_numpy(result)

    result = result.permute(2,0,1).unsqueeze(0)
    result = F.interpolate(result.to('cuda'), 512, None, 'bilinear', False)
    print(result.shape)
    result  = result.squeeze(0)
    result = inv_transform(result)
    result = transforms.ToPILImage()(result)
    result.save('eg10.png')


    # print(transform_matrix)
    # image = inv_transform(image)
    # image.save('eg3.png')
    # tensor = torch.arange(512)
    # tensor = torch.cat((tensor, tensor), dim=0).resize(2, 512)
    # image = torch.zeros_like(source)
    # for j in range(512):
    #     image[:,j,j]

    # print(kp)
    # print(source_points.shape)
    #增加一维
    # source = source.unsqueeze(0)
    # target = target.unsqueeze(0)
    # optimize.test_epoch_1(model,
    #                     source,
    #                     target,
    #                       device,
    #                       epoch=0)


    # # 将结果转换成单通道图片并保存
    # source = source.numpy()
    # source = np.transpose(source, (1, 2, 0))
    # source = inv_transform(source)
    # source.save('eg1.png')
    # target = target.numpy()
    # target = np.transpose(target, (1, 2, 0))
    # target = inv_transform(target)
    # target.save('eg2.png')


    # source.save('eg1.png')
    #
    #
    #
    #
    #
    # print(source.shape)

