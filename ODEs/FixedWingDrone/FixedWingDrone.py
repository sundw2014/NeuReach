from scipy.integrate import odeint
import numpy as np
import math
from typing import Optional, List, Tuple

# function to return derivatives of state to be integrated
def position(p: np.array, t: float, cd1: float, vc: float, m: float,
             k1: float, k2: float, x_n: float, y_n: float) -> np.array:
    # get initial conditions
    if p.shape != (4,):
        print("bad shape:", p.shape)
        raise ValueError("p must be a length 4 array for drone")

    x: float = p[0]  # x_i
    y: float = p[1]  # y_i
    s: float = p[2]  # psi
    v: float = p[3]  # velocity
    # constants
    G: float = 32.2  # ft/sec

    # compute dvdt
    # vc = math.sqrt((x - x_n)*(x - x_n)+(y - y_n)*(y - y_n)) / ts
    D: float = cd1 * v ** 2  # drag
    T: float = k1 * m * (vc - v)  # thrust
    dvdt: float = (T - D) / m  # v'

    dot: float = (y_n - y)  # dot product between [x1, y1] and [x2, y2]
    det: float = (x_n - x)  # determinant
    sh: float = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    h: float = (k2 * vc / G) * (sh - s)
    dsdt: float = (G / v) * np.sin(h)  # * (sh - s) #

    dxdt: float = v * np.sin(s)
    dydt: float = v * np.cos(s)
    return np.array((dxdt, dydt, dsdt, dvdt))


# function to provide traces of the system
def TC_Simulate(initial_point: np.array, time_bound: float, mode_parameters=[5,5]) -> np.array:
    time_step = 0.05
    mode = 'follow_waypoint'
    if mode == 'follow_waypoint':
        # mode parameters for this is the waypoint center
        t = np.arange(0, time_bound + time_step, time_step)
        assert isinstance(mode_parameters, list) and (len(mode_parameters) == 2) and (isinstance(
            mode_parameters[0], float) or isinstance(
            mode_parameters[0], int)), "must give length 2 list as params to follow_waypoint mode of fixedwing_drone"
        red_args = (0.002, 0.6, 1, 1.5, 0.6, mode_parameters[0], mode_parameters[1])
        sol = odeint(position, initial_point, t, args=red_args, hmax=time_step)
        trace = np.column_stack((t, sol))
        return trace
    else:
        raise ValueError("Mode: ", mode, "is not defined for the Fixedwing Drone")

if __name__ == "__main__":
    set_lower = np.array([0-0.125, 0-0.125, 0, 1-0.125])
    set_higher = np.array([0+0.125, 0+0.125, 2 * np.pi, 1+0.125])

    # import ipdb; ipdb.set_trace()
    # from IPython import embed; embed()
    goal = [5,5]
    traces = []
    for _ in range(100):
        initial_state = np.random.rand(len(set_lower)) * (set_higher - set_lower) + set_lower
        trace = TC_Simulate(initial_state, 13., goal)
        traces.append(trace)

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams['legend.fontsize'] = 10

    for trace in traces:
        plt.plot(trace[:,1], trace[:,2], color='b')#, label='trace')

    plt.scatter(goal[0], goal[1], marker='o', color='r', s=12)
    plt.show()
    # dimensions = len(trace[0])
    # init_delta_array = [0.5,0.5,0.5] + [0.1] * (dimensions - 4)
    # k = [1] * (dimensions - 1)
    # gamma = [0] * (dimensions - 1)
    # tube = bloatToTube(k, gamma, init_delta_array, trace, dimensions)
    # gazebotube = tube[:][1:4]
    # gazebotrace = trace[:][1:4]
    # print(tube)
    # plt.plot(trace[:,1], trace[:,3])
    # safety, reach = _verify_reach_tube(np.zeros((9,)), "[2; 2; 5]", 2.5, [])
    #print("reach: ", reach.tube)
    # plt.show()
