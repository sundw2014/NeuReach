import math
import sys
import numpy as np
from numpy import deg2rad
import sys
sys.path.append('ODEs')
from aerobench.run_f16_sim import run_f16_sim
from aerobench.examples.gcas.gcas_autopilot import GcasAutopilot

TMAX = 20.
dt = 1/30.

lower = np.array([560, -0.1, 0,       -np.pi/4, -0.1, 70])
higher = np.array([600,  0.1, np.pi/4, np.pi/4,  0.1, 80])
X0_center_range = np.array([lower, higher]).T
half_width = (higher - lower) / 2
center = (higher + lower) / 2
r_max = 0.5

def unnormalize(x):
    return x * half_width + center

def sample_X0():
    center = np.random.rand(X0_center_range.shape[0]) * 2 - 1
    r = np.random.rand()*r_max
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
    x0[x0>1] = 1
    x0[x0<-1] = -1
    return x0

def sample_x0_uniform(X0):
    center = X0[:-1]
    r = X0[-1]

    n = len(center)
    direction = np.random.randn(n)
    direction = direction / np.linalg.norm(direction)
    dist = np.random.rand()

    x0 = center + direction * dist * r
    x0[x0>1] = 1
    x0[x0<-1] = -1
    return x0

def simulate(x0):
    x0 = unnormalize(x0)
    ### Initial Conditions ###
    power = 9 # engine power level (0-10)
    # Default alpha & beta
    # alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)
    # Initial Attitude
    # alt = x0[0]        # altitude (ft)
    # vt = x0[1]          # initial velocity (ft/sec)
    # phi = 0           # Roll angle from wings level (rad)
    theta = (-math.pi/2)*0.7         # Pitch angle from nose level (rad)
    # psi = 0.8 * math.pi   # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [x0[0], x0[1], beta, x0[2], theta, x0[3], 0, x0[4], 0, 0, 0, x0[5]*100, power]
    tmax = TMAX # simulation time

    # ### Initial Conditions ###
    # power = 9 # engine power level (0-10)
    # # Default alpha & beta
    # alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    # beta = 0                # Side slip angle (rad)
    # # Initial Attitude
    # alt = x0[-1]        # altitude (ft)
    # vt = x0[0]          # initial velocity (ft/sec)
    # phi = 0           # Roll angle from wings level (rad)
    # theta = (-math.pi/2)*0.7         # Pitch angle from nose level (rad)
    # psi = 0.8 * math.pi   # Yaw angle from North (rad)

    # # Build Initial Condition Vectors
    # # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    # init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    # tmax = TMAX # simulation time

    ap = GcasAutopilot(init_mode='waiting', stdout=True)

    ap.waiting_time = 5
    ap.waiting_cmd[1] = 2.2 # ps command

    # custom gains
    ap.cfg_k_prop = 1.4
    ap.cfg_k_der = 0
    ap.cfg_eps_p = deg2rad(20)
    ap.cfg_eps_phi = deg2rad(15)

    step = 1/30
    res = run_f16_sim(init, tmax, ap, step=step, extended_states=True, integrator_str='rk45')
    states = res['states']
    traj = np.concatenate([np.array(res['times']).reshape(-1,1), states[:,[0,11]]], axis=1)
    traj[:,-1] /= 100.
    return traj

def get_init_center(X0):
    center = X0[:-1]
    return center

def get_X0_normalization_factor():
    mean = np.zeros(len(sample_X0()))
    std = np.ones(len(sample_X0()))
    return [mean, std]
