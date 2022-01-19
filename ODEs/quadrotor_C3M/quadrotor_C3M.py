import torch
from torch import nn
import numpy as np
from .utils import loadpklz
from .ol_dynamics import f

effective_dim_start = 3
effective_dim_end = 8
n = 8
m = 3
dim = effective_dim_end - effective_dim_start # 8 - 3. Do not use the positions
c = 3 * n

def odeint(f, x0, t, args=()):
    x = [np.array(x0),]
    for idx in range(len(t)-1):
        dot_x = f(x[-1], t[idx], *args)
        x.append(x[-1] + dot_x*(t[idx+1]-t[idx]))
    return np.array(x)

class C3M(object):
    def __init__(self, model_file):
        super(C3M, self).__init__()
        self.model_u_w1 = torch.nn.Sequential(
            torch.nn.Linear(2*dim, 128, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(128, c*n, bias=True))

        self.model_u_w2 = torch.nn.Sequential(
            torch.nn.Linear(2*dim, 128, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(128, m*c, bias=True))

        ck = torch.load(model_file, map_location='cpu')
        self.model_u_w1.load_state_dict(ck['model_u_w1'])
        self.model_u_w2.load_state_dict(ck['model_u_w2'])

    def __call__(self, x, xe, uref):
        x = torch.from_numpy(x).float().view(1,-1,1)
        xe = torch.from_numpy(xe).float().view(1,-1,1)
        uref = torch.from_numpy(uref).float().view(1,-1,1)

        # x: B x n x 1
        # u: B x m x 1
        bs = x.shape[0]

        w1 = self.model_u_w1(torch.cat([x[:,effective_dim_start:effective_dim_end,:],(x-xe)[:,effective_dim_start:effective_dim_end,:]],dim=1).squeeze(-1)).reshape(bs, -1, n)
        w2 = self.model_u_w2(torch.cat([x[:,effective_dim_start:effective_dim_end,:],(x-xe)[:,effective_dim_start:effective_dim_end,:]],dim=1).squeeze(-1)).reshape(bs, m, -1)
        u = w2.matmul(torch.tanh(w1.matmul(xe))) + uref

        u = u.squeeze(0).detach().numpy()
        return u
import os

class TC_Simulate(object):
    def __init__(self):
        super(TC_Simulate, self).__init__()
        self.C3M = C3M(os.path.dirname(os.path.abspath(__file__)) + '/data/model_best.pth.tar')
        ref = loadpklz(os.path.dirname(os.path.abspath(__file__)) + '/data/ref.pklz')
        self.xref = ref['xref']
        self.uref = ref['uref']
        self.t = ref['t']
        self.time_step = self.t[1] - self.t[0]
        # print('time_step', self.time_step)
        # print('t_max', self.t[-1])

    def __call__(self, Mode, initialCondition, time_bound):
        def u(x, t):
            idx = int(t//self.time_step)
            xref = self.xref[idx,:]
            uref = self.uref[idx,:]
            xe = x - xref
            u = self.C3M(x, xe, uref)
            return u

        noise = np.zeros([len(self.t), 8])
        noise_level = 0.1
        # noise_level = 0.
        num_segs = 20
        segments = np.random.permutation(len(self.t))[:num_segs-1]
        segments.sort()
        segments = [0,] + segments.tolist() + [-1,]
        # print(segments)

        for i in range(num_segs):
            n = np.random.randn(8) * noise_level
            noise[segments[i]:segments[i+1],:] = n.reshape(1,-1)

        def n(t):
            # noise = np.zeros(8)
            # if np.random.rand() < 10/len(self.t):
            #     noise[0:3] = 10.
            # return noise
            idx = int(t//self.time_step)
            _n = noise[idx,:]
            return _n

        def cl_dynamics(x, t, u, noise):
            # closed-loop dynamics. u should be a function
            x = np.array(x)
            dot_x = f(x, u(x, t)) + noise(t)
            return dot_x
        X0 = np.array(initialCondition)
        x_nl = odeint(cl_dynamics, X0, self.t, args=(u,n))
        return np.concatenate([self.t.reshape(-1,1), x_nl], axis=1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import argparse
    import time

    ######################## plot #######################
    fig = plt.figure(figsize=(20, 10))
    track = fig.add_subplot(1, 1, 1, projection="3d")

    simulator = TC_Simulate()

    for i in range(10):
        # x_nl = TC_Simulate("waypoints", np.random.randn(12), 10)
        # for w in waypoints:
        #     track.plot(w[0:1], w[1:2], w[2:3], 'ro', markersize=10.)
        np.random.seed(int(time.time()*1000)%(2**32-1))
        x_nl = simulator("None", 2*(np.random.rand(8) - 0.5), 10)
        x_nl = x_nl[:,1:]
        track.plot(x_nl[:, 0], x_nl[:, 1], x_nl[:, 2], color="g")
    x_nl = simulator.xref
    track.plot(x_nl[:, 0], x_nl[:, 1], x_nl[:, 2], color="r")

    # track.text(x_nl[0,0], x_nl[0,2], x_nl[0,4], "start", color='red')
    # track.text(x_nl[-1,0], x_nl[-1,2], x_nl[-1,4], "finish", color='red')
    track.set_xlabel('x')
    track.set_ylabel('y')
    track.set_zlabel('z')

    plt.show()
