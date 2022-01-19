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
X0_r_max = np.array([10,   0.1, np.pi/16, np.pi/8,  0.1, 1])

def sample_X0():
    center = X0_center_range[:,0] + np.random.rand(X0_center_range.shape[0]) * (X0_center_range[:,1]-X0_center_range[:,0])
    r = np.random.rand(len(X0_r_max))*X0_r_max
    return np.concatenate([center, r])

def sample_t():
    return (np.random.randint(int(TMAX/dt))+1) * dt

def sample_x0(X0):
    n = len(X0_r_max)
    center = X0[:n]
    r = X0[n:]

    dist = (np.random.rand(n)-0.5)*2

    idx = np.random.randint(n)
    sign = np.random.choice([-1,1])
    dist[idx] = sign
    x0 = center + dist * r
    x0[x0>X0_center_range[:,1]] = X0_center_range[x0>X0_center_range[:,1],1]
    x0[x0<X0_center_range[:,0]] = X0_center_range[x0<X0_center_range[:,0],0]
    return x0

def sample_x0_uniform(X0):
    n = len(X0_r_max)
    center = X0[:n]
    r = X0[n:]

    dist = (np.random.rand(n)-0.5)*2

    x0 = center + dist * r
    x0[x0>X0_center_range[:,1]] = X0_center_range[x0>X0_center_range[:,1],1]
    x0[x0<X0_center_range[:,0]] = X0_center_range[x0<X0_center_range[:,0],0]
    return x0

def simulate(x0):
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
    n = len(X0_r_max)
    center = X0[:n]
    return center

def get_X0_normalization_factor():
    mean = np.concatenate([X0_center_range.mean(axis=1), X0_r_max/2])
    std = np.concatenate([(X0_center_range[:,1]-X0_center_range[:,0])/2, X0_r_max/2])
    return [mean, std]
