# 3D Control of Quadcopter
# based on https://github.com/juanmed/quadrotor_sim/blob/master/3D_Quadrotor/3D_control_with_body_drag.py
# The dynamics is from pp. 17, Eq. (2.22). https://www.kth.se/polopoly_fs/1.588039.1550155544!/Thesis%20KTH%20-%20Francesco%20Sabatino.pdf
# The linearization is from Different Linearization Control Techniques for
# a Quadrotor System (many typos)

import numpy as np
import scipy
import scipy.linalg
# from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .nonlinear_dynamics import g, m, Ix, Iy, Iz, f
import contextlib

waypoints = [[1, 1, 1], [1, 1, 2], [0, 0, 0]]

def odeint(f, x0, t, args=()):
    x = [np.array(x0),]
    for idx in range(len(t)-1):
        dot_x = f(x[-1], t[idx], *args)
        x.append(x[-1] + dot_x*(t[idx+1]-t[idx]))
    return np.array(x)

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u
    """
    # http://www.mwm.im/lqr-controllers-with-python/
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R) * (B.T * X))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return np.asarray(K), np.asarray(X), np.asarray(eigVals)




# The control can be done in a decentralized style
# The linearized system is divided into four decoupled subsystems

# X-subsystem
# The state variables are x, dot_x, pitch, dot_pitch
Ax = np.array(
    [[0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, g, 0.0],
     [0.0, 0.0, 0.0, 1.0],
     [0.0, 0.0, 0.0, 0.0]])
Bx = np.array(
    [[0.0],
     [0.0],
     [0.0],
     [1 / Ix]])

# Y-subsystem
# The state variables are y, dot_y, roll, dot_roll
Ay = np.array(
    [[0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, -g, 0.0],
     [0.0, 0.0, 0.0, 1.0],
     [0.0, 0.0, 0.0, 0.0]])
By = np.array(
    [[0.0],
     [0.0],
     [0.0],
     [1 / Iy]])

# Z-subsystem
# The state variables are z, dot_z
Az = np.array(
    [[0.0, 1.0],
     [0.0, 0.0]])
Bz = np.array(
    [[0.0],
     [1 / m]])

# Yaw-subsystem
# The state variables are yaw, dot_yaw
Ayaw = np.array(
    [[0.0, 1.0],
     [0.0, 0.0]])
Byaw = np.array(
    [[0.0],
     [1 / Iz]])

####################### solve LQR #######################
Ks = []  # feedback gain matrices K for each subsystem
for A, B in ((Ax, Bx), (Ay, By), (Az, Bz), (Ayaw, Byaw)):
    n = A.shape[0]
    m = B.shape[1]
    Q = np.eye(n)
    Q[0, 0] = 10.  # The first state variable is the one we care about.
    R = np.diag([1., ])
    K, _, _ = lqr(A, B, Q, R)
    Ks.append(K)

def TC_Simulate(Mode,initialCondition,time_bound):
    ######################## simulate #######################
    # time instants for simulation
    t_max = time_bound
    t = np.arange(0., t_max, 0.01)


    def cl_nonlinear(x, t, u):
        x = np.array(x)
        dot_x = f(x, u(x, t) + np.array([m * g, 0, 0, 0]))
        noise = np.zeros(12)
        # noise[[0,2,4]] = 1e-1*np.random.randn(3)
        return dot_x + 0.1*np.random.randn(12)#+ noise


    if Mode == 'waypoints':
        # waypoints = [[1, 1, 1], [1, 1, 2], [0, 0, 0]]
        # follow waypoints
        signal = np.zeros([len(t), 3])
        num_w = len(waypoints)
        for i, w in enumerate(waypoints):
            assert len(w) == 3
            signal[len(t) // num_w * i:len(t) // num_w *
                   (i + 1), :] = np.array(w).reshape(1, -1)
        # X0 = np.zeros(12)
        signalx = signal[:, 0]
        signaly = signal[:, 1]
        signalz = signal[:, 2]
    else:
        # Create an random signal to track
        num_dim = 3
        freqs = np.arange(0.1, 2., 0.1)
        with temp_seed(0):
            weights = np.random.randn(len(freqs), num_dim)  # F x n
        weights = weights / \
            np.sqrt((weights**2).sum(axis=0, keepdims=True))  # F x n
        signal_AC = np.sin(freqs.reshape(1, -1) * t.reshape(-1, 1)
                           ).dot(weights)  # T x F * F x n = T x n
        with temp_seed(0):
            signal_DC = np.random.randn(num_dim).reshape(1, -1)  # offset
        signal = signal_AC + signal_DC
        signalx = signal[:, 0]
        signaly = signal[:, 1]
        signalz = 0.1 * t
        # initial state
        # _X0 = 0.1 * np.random.randn(num_dim) + signal_DC.reshape(-1)
        # X0 = np.zeros(12)
        # X0[[0, 2, 4]] = _X0

    signalyaw = np.zeros_like(signalz)  # we do not care about yaw


    def u(x, _t):
        # the controller
        dis = _t - t
        dis[dis < 0] = np.inf
        idx = dis.argmin()
        UX = Ks[0].dot(np.array([signalx[idx], 0, 0, 0]) - x[[0, 1, 8, 9]])[0]
        UY = Ks[1].dot(np.array([signaly[idx], 0, 0, 0]) - x[[2, 3, 6, 7]])[0]
        UZ = Ks[2].dot(np.array([signalz[idx], 0]) - x[[4, 5]])[0]
        UYaw = Ks[3].dot(np.array([signalyaw[idx], 0]) - x[[10, 11]])[0]
        return np.array([UZ, UY, UX, UYaw])

    X0 = np.array(initialCondition)
    # simulate
    x_nl = odeint(cl_nonlinear, X0, t, args=(u,))
    return np.concatenate([t.reshape(-1,1), x_nl], axis=1)

if __name__ == '__main__':
    import argparse
    import time

    ######################## plot #######################
    fig = plt.figure(figsize=(20, 10))
    track = fig.add_subplot(1, 1, 1, projection="3d")

    for i in range(10):
        # x_nl = TC_Simulate("waypoints", np.random.randn(12), 10)
        # for w in waypoints:
        #     track.plot(w[0:1], w[1:2], w[2:3], 'ro', markersize=10.)
        np.random.seed(int(time.time()*1000)%(2**32-1))
        x_nl = TC_Simulate("random", 2*(np.random.rand(12) - 0.5), 10)
        x_nl = x_nl[:,1:]
        track.plot(x_nl[:, 0], x_nl[:, 2], x_nl[:, 4], color="g")
    # track.text(x_nl[0,0], x_nl[0,2], x_nl[0,4], "start", color='red')
    # track.text(x_nl[-1,0], x_nl[-1,2], x_nl[-1,4], "finish", color='red')
    track.set_xlabel('x')
    track.set_ylabel('y')
    track.set_zlabel('z')

    plt.show()
