import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import importlib
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool

import sys
sys.path.append('systems')
sys.path.append('.')

from model import get_model
from model_dryvr import get_model as get_model_dryvr

def mute():
    sys.stdout = open(os.devnull, 'w')

import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('--system', type=str,
                        default='jetengine')
parser.add_argument('--no_cuda', dest='use_cuda', action='store_false')
parser.set_defaults(use_cuda=True)
parser.add_argument('--layer1', type=int, default=64)
parser.add_argument('--layer2', type=int, default=64)
parser.add_argument('--pretrained', type=str)
parser.add_argument('--output', type=str, default='test.txt')
parser.add_argument('--seed', type=int, default=1024)

args = parser.parse_args()

config = importlib.import_module('system_'+args.system)
use_dryvr = 'dryvr' in args.pretrained
if use_dryvr:
    model, forward = get_model_dryvr(len(config.sample_X0())+1, config.simulate(config.get_init_center(config.sample_X0())).shape[1]-1)
else:
    model, forward = get_model(len(config.sample_X0())+1, config.simulate(config.get_init_center(config.sample_X0())).shape[1]-1, config, args)
    if args.use_cuda:
        model = model.cuda()
    else:
        model = model.cpu()

model.load_state_dict(torch.load(args.pretrained)['state_dict'])

def calc_volume(Ps):
    vol = 0.
    for P in Ps:
        tmp = np.sqrt(1 / np.linalg.det(P.T.dot(P)))
        if P.shape[0] == 3:
            tmp *= np.pi * 4 / 3
        elif P.shape[0] == 2:
            tmp *= np.pi
        else:
            raise ValueError('wrong shape')
        vol += tmp
    return vol

def calc_acc(sampled_traces, Ps, ref):
    ref = np.array(ref) # T x n
    trj = np.array(sampled_traces)[:,:,1:].transpose([1,2,0]) # T x n x N
    trj = trj - np.expand_dims(ref[:,1:], -1)
    Ps = np.array(Ps) # T x n x n
    Px = np.matmul(Ps,trj) # T x n x N
    Pxn = (Px**2).sum(axis=1).reshape(-1)
    return (Pxn<=1).sum()/len(Pxn)

np.random.seed(args.seed)

X0 = config.sample_X0()

ref = config.simulate(config.get_init_center(X0))#[::20]
sampled_inits = [config.sample_x0_uniform(X0) for _ in range(100)]
num_proc = min([1, multiprocessing.cpu_count()-3])
sampled_trajs = list(tqdm(map(config.simulate, sampled_inits), total=len(sampled_inits)))

benchmark_name = args.system

reachsets = []

X0_mean, X0_std = config.get_X0_normalization_factor()
X0 = (X0 - X0_mean) / X0_std

for idx_t in range(1, ref.shape[0]):
    s = time.time()
    tmp = torch.tensor(X0.tolist()+[ref[idx_t, 0],]).view(1,-1).float()
    if args.use_cuda:
        tmp = tmp.cuda()
    P = forward(tmp)
    e = time.time()
    P = P.squeeze(0)
    reachsets.append([ref[idx_t, 1:], P.cpu().detach().numpy()])

vol = calc_volume([r[1] for r in reachsets])

acc = calc_acc(np.array(sampled_trajs)[:, 1:, :], [r[1] for r in reachsets], ref[1:,:])
print(vol, acc)
with open(args.output, 'a') as f:
    print(vol, acc, file=f)
