import sys
sys.path.append('..')
from ODEs.jet_engine import TC_Simulate
import numpy as np

TMAX = 10.
dt = 0.05

# range of initial states
lower = np.array([0.3, 0.3])
higher = np.array([1.3, 1.3])
X0_center_range = np.array([lower, higher]).T
X0_r_max = 0.5

def sample_X0():
    center = X0_center_range[:,0] + np.random.rand(X0_center_range.shape[0]) * (X0_center_range[:,1]-X0_center_range[:,0])
    r = np.random.rand()*X0_r_max
    X0 = np.concatenate([center, np.array(r).reshape(-1)])
    return X0

def sample_t():
    return (np.random.randint(int(TMAX/dt))+1) * dt

def sample_x0(X0):
    center = X0[:-1]
    r = X0[-1]

    n = len(center)
    direction = np.random.randn(n)
    direction = direction / np.linalg.norm(direction)

    x0 = center + direction * r
    x0[x0>X0_center_range[:,1]] = X0_center_range[x0>X0_center_range[:,1],1]
    x0[x0<X0_center_range[:,0]] = X0_center_range[x0<X0_center_range[:,0],0]
    return x0

def sample_x0_uniform(X0):
    center = X0[:-1]
    r = X0[-1]

    n = len(center)
    direction = np.random.randn(n)
    direction = direction / np.linalg.norm(direction)

    dist = np.random.rand()
    x0 = center + direction * dist * r
    # x0 = center + direction * r
    x0[x0>X0_center_range[:,1]] = X0_center_range[x0>X0_center_range[:,1],1]
    x0[x0<X0_center_range[:,0]] = X0_center_range[x0<X0_center_range[:,0],0]
    return x0

def simulate(x0):
    return np.array(TC_Simulate("Default", x0, TMAX))

def get_init_center(X0):
    center = X0[:-1]
    return center

def get_X0_normalization_factor():
    mean = np.zeros(len(sample_X0()))
    std = np.ones(len(sample_X0()))
    return [mean, std]
