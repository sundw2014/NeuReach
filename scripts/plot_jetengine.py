import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import importlib
from tqdm import tqdm
from matplotlib import pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 13
HUGE_SIZE = 25

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=HUGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)

plt.subplots_adjust(
    top=0.92,
    bottom=0.15,
    left=0.11,
    right=1.0,
    hspace=0.2,
    wspace=0.2)

import sys
sys.path.append('systems')
sys.path.append('.')

from model import get_model as get_model_ours
from model_dryvr import get_model as get_model_dryvr

import argparse

np.random.seed(1024)

parser = argparse.ArgumentParser(description="")
parser.add_argument('--system', type=str,
                        default='jetengine')
parser.add_argument('--no_cuda', dest='use_cuda', action='store_false')
parser.set_defaults(use_cuda=True)
parser.add_argument('--layer1', type=int, default=64)
parser.add_argument('--layer2', type=int, default=64)
parser.add_argument('--pretrained_ours', type=str)
parser.add_argument('--pretrained_dryvr', type=str)

args = parser.parse_args()

config = importlib.import_module('system_'+args.system)
model_ours, forward_ours = get_model_ours(len(config.sample_X0())+1, config.simulate(config.get_init_center(config.sample_X0())).shape[1]-1,  config, args)
if args.use_cuda:
    model_ours = model_ours.cuda()
else:
    model_ours = model_ours.cpu()
model_ours.load_state_dict(torch.load(args.pretrained_ours)['state_dict'])

model_dryvr, forward_dryvr = get_model_dryvr(len(config.sample_X0())+1, config.simulate(config.get_init_center(config.sample_X0())).shape[1]-1)
model_dryvr.load_state_dict(torch.load(args.pretrained_dryvr)['state_dict'])

def ellipsoid_surface_2D(P):
    K = 100
    thetas = np.linspace(0, 2 * np.pi, K)
    points = []
    for i, theta in enumerate(thetas):
        point = np.array([np.cos(theta), np.sin(theta)])
        points.append(point)
    points = np.array(points)
    points = np.linalg.inv(P).dot(points.T)
    return points[0,:], points[1,:]

benchmark_name = args.system

center = np.array([1.,1.])
r = 0.3
print(center, r)

traces = []
# ref trace
ref = config.simulate(center)
traces.append(np.array(ref))

# calculate the reachset using the trained model
reachsets_ours = []
reachsets_dryvr = []
for idx_t in tqdm(range(1, ref.shape[0])):
    tmp = torch.tensor(center.tolist()+[r, ref[idx_t, 0],]).view(1,-1).float()
    if args.use_cuda:
        tmp = tmp.cuda()
    P = forward_ours(tmp)
    P = P.squeeze(0)
    reachsets_ours.append([ref[idx_t, 1:], P.cpu().detach().numpy()])

    P = forward_dryvr(tmp)
    P = P.squeeze(0)
    reachsets_dryvr.append([ref[idx_t, 1:], P])

# plot the ref trace
plt.plot(ref[:,1], ref[:,2], 'r-')#, label='ref')

# plot ellipsoids for each time step
for reachset_ours, reachset_dryvr in zip(reachsets_ours[::10], reachsets_dryvr[::10]):
    label = reachset_ours is reachsets_ours[0]
    c = reachset_ours[0]
    x,y = ellipsoid_surface_2D(reachset_ours[1])
    plt.plot(x+c[0], y+c[1], 'g-', markersize=1, label='NeuReach' if label else None)
    x,y = ellipsoid_surface_2D(reachset_dryvr[1])
    plt.plot(x+c[0], y+c[1], 'y-', markersize=1, label='DryVR' if label else None)

sampled_traces = []

# randomly sample some traces
for _ in range(100):
    n = len(center)
    direction = np.random.randn(n)
    direction = direction / np.linalg.norm(direction)

    dist = np.random.rand()
    x0 = center + direction * dist * r

    _trace = config.simulate(x0)[:,1:]
    sampled_traces.append(_trace)

_traces = np.array(sampled_traces)[:,1:,:]
plt.plot(_traces[:,::10,0], _traces[:,::10,1], 'kx', markersize=1)

plt.xlim(-2, 1)
plt.ylim(-3, 0)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend(loc='upper left')
plt.title('Reachsets of JetEngine')
# plt.show()
plt.savefig('jetEngine.pdf')
