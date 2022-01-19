import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import importlib
from utils import AverageMeter
from tensorboardX import SummaryWriter

from data import get_dataloader
from model import get_model

import sys
sys.path.append('systems')

import argparse

np.random.seed(1024)

parser = argparse.ArgumentParser(description="")
parser.add_argument('--system', type=str,
                        default='jetengine')
parser.add_argument('--lambda', dest='_lambda', type=float, default=0.03)
parser.add_argument('--alpha', dest='alpha', type=float, default=0.001)
parser.add_argument('--N_X0', type=int, default=100)
parser.add_argument('--N_x0', type=int, default=10)
parser.add_argument('--N_t', type=int, default=100)
parser.add_argument('--layer1', type=int, default=64)
parser.add_argument('--layer2', type=int, default=64)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.01)
parser.add_argument('--data_file_train', default='train.pklz', type=str)
parser.add_argument('--data_file_eval', default='eval.pklz', type=str)
parser.add_argument('--log', type=str)

parser.add_argument('--bs', dest='batch_size', type=int, default=256)
parser.add_argument('--no_cuda', dest='use_cuda', action='store_false')
parser.set_defaults(use_cuda=True)
parser.add_argument('--num_test', type=int, default=10)
parser.add_argument('--eps', dest='eps', type=float, default=0.01)
parser.add_argument('--lr_step', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

os.system('mkdir '+args.log)
os.system('echo "%s" > %s/cmd.txt'%(' '.join(sys.argv), args.log))
os.system('cp *.py '+args.log)
os.system('cp -r systems/ '+args.log)
os.system('cp -r ODEs/ '+args.log)

np.random.seed(1024)

config = importlib.import_module('system_'+args.system)

ACC = 0.97

def PWD(normalized_dis, t):
    T = np.sort(list(set(t.tolist())))
    num_t = len(T)
    DIS = np.zeros(num_t)
    for idx_t in range(num_t):
        idx = np.where(t==T[idx_t])[0]
        dis = normalized_dis[idx]
        dis = np.sort(dis)
        idx = int(len(dis)*ACC)
        if idx == len(dis):
            idx -= 1
        DIS[idx_t] = dis[idx]

    # import ipdb;ipdb.set_trace()

    T = np.array([0,] + T.tolist())
    DIS = np.array([1, ] + DIS.tolist())
    y = np.log(DIS)
    K = 1
    y = y-np.log(K)
    dy = y[1:] - y[:-1]
    dt = T[1:] - T[:-1]

    gamma = dy / dt

    return gamma, T

train_loader, val_loader = get_dataloader(config, args)

normalized_dis = []
t = []
for (X0, T, ref, xt) in train_loader:
    DXi = (xt - ref).cpu().detach().numpy()
    dis = np.sqrt((DXi**2).sum(axis=1)).reshape(-1)
    R = X0[:,-1].cpu().detach().numpy().reshape(-1)
    T = T.cpu().detach().numpy().reshape(-1)
    normalized_dis.append(dis/R)
    t.append(T)

normalized_dis = np.concatenate(normalized_dis)
t = np.concatenate(t)

gammas, t = PWD(normalized_dis, t)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    filename = args.log + '/' + filename
    torch.save(state, filename)

save_checkpoint({'state_dict': [gammas, t]})
