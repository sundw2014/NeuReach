from scipy.integrate import odeint
import numpy as np
import math as m

mass = 20
G = 9.81

def compute_angle(psi, vec):
    unit = np.array([1.0, 0.0])
    c, s = m.cos(psi), m.sin(psi)
    R = np.array(((c, -s), (s, c)))

    heading = np.matmul(R, unit)
    diff = np.arctan2(np.linalg.det([heading, vec]), np.dot(heading, vec))
    return psi + diff


# function to return derivatives of state to be integrated
def dynamics(state, time, action):
    # Variables
    (_, _, _, vx, vy, vz, phi, theta, psi) = state[:9]
    (fz, w1, w2, w3) = action[:]

    # Derivatives
    dvx = (m.cos(phi) * m.sin(theta) * m.cos(psi) + m.sin(phi) * m.sin(psi)) * fz / mass
    dvy = (m.cos(phi) * m.sin(theta) * m.sin(psi) - m.sin(phi) * m.cos(psi)) * fz / mass
    dvz = m.cos(phi) * m.cos(theta) * fz / mass + G

    dphi = w1 + m.sin(phi) * m.tan(theta) + w2 + m.cos(phi) * m.tan(theta) * w3
    dtheta = m.cos(phi) * w2 - m.sin(phi) * w3
    dpsi = m.sin(phi) * (1 / m.cos(theta)) * w2 + m.cos(phi) * (1 / m.cos(theta)) * w3

    return np.array([vx, vy, vz, dvx, dvy, dvz, dphi, dtheta, dpsi])


def control(state, desired, action):
    # Constants
    Kp, Kp_bar, Kd = 0.9, 0.1, 0.1

    # Variables
    (x, y, z, vx, vy, vz, phi, theta, psi) = state[:9]
    (x_d, y_d, z_d, vx_d, vy_d, vz_d, phi_d, theta_d, psi_d,
     ax_d, ay_d, az_d, dphi_d, dtheta_d, dpsi_d) = desired

    # Derivative of state
    (_, _, _, _, _, _, dphi, dtheta, dpsi) = dynamics(state[:9], 1.0, action)

    # Feedforward control
    f_ff = -mass * m.sqrt(ax_d ** 2 + ay_d ** 2 + (az_d - G) ** 2)
    w1_ff = dphi_d - m.sin(theta_d) * dpsi_d
    w2_ff = m.cos(phi_d) * dtheta_d + m.sin(phi_d) * m.cos(theta_d) * dpsi_d
    w3_ff = -m.sin(phi_d) * dtheta_d + m.cos(phi_d) * m.cos(theta_d) * dpsi_d

    # Feedback control
    f_fbx = ((m.cos(phi) * m.sin(theta) * m.cos(psi) + m.sin(phi) * m.sin(psi)) *
             ((x_d - x) * Kp + (vx_d - vx) * Kd))
    f_fby = ((m.cos(phi) * m.sin(theta) * m.sin(psi) - m.sin(phi) * m.cos(psi)) *
             ((y_d - y) * Kp + (vy_d - vy) * Kd))
    f_fbz = m.cos(phi) * m.cos(theta) * ((z_d - z) * Kp + (vz_d - vz) * Kd)
    f_fb = f_fbx + f_fby + f_fbz

    w1_fb = Kp * (phi_d - phi) + Kd * (dphi_d - dphi) + Kp_bar * (y_d - y)
    w2_fb = Kp * (theta_d - theta) + Kd * (dtheta_d - dtheta) + Kp_bar * (x_d - x)
    w3_fb = Kp * (psi_d - psi) + Kd * (dpsi_d - dpsi)

    return [f_ff + f_fb, w1_ff + w1_fb, w2_ff + w2_fb, w3_ff + w3_fb]


def compute_desired_state(state, goal, time_step):
    # State variables
    (x, y, z, vx, vy, vz, phi, theta, psi) = state[:9]
    # Compute distance and angle to goal
    v2 = np.array([goal[0] - x, goal[1] - y])
    dist = np.linalg.norm(v2)
    angle = compute_angle(psi, v2)
    # Compute desired state
    if np.linalg.norm(v2) >= 1:
        con = v2 / dist
    else:
        con = v2
    x_d = con[0] * time_step + x
    y_d = con[1] * time_step + y
    z_d = (goal[2] - z) * time_step + z
    vx_d = 0.5 * con[0]
    vy_d = 0.5 * con[1]
    vz_d = goal[2] - z
    ax_d = (vx_d - vx)
    ay_d = (vy_d - vy)
    az_d = vz_d - vz

    psi_d = angle
    beta_a = -ax_d * m.cos(psi_d) - ay_d * m.sin(psi_d)
    beta_b = -az_d + G
    beta_c = -ax_d * m.sin(psi_d) + ay_d * m.cos(psi_d)
    theta_d = m.atan2(beta_a, beta_b)
    phi_d = m.atan2(beta_c, m.sqrt(beta_a ** 2 + beta_b ** 2))
    dphi_d = -phi
    dtheta_d = -theta
    dpsi_d = psi_d - psi
    return [x_d, y_d, z_d, vx_d, vy_d, vz_d, phi_d, theta_d, psi_d, ax_d, ay_d, az_d, dphi_d, dtheta_d, dpsi_d]


# function to provide traces of the system
def TC_Simulate(initial_condition, time_bound, mode=[0,0,0]):
    action = [0.0, 0.0, 0.0, 0.0]
    time_step = 0.01
    number_points = int(np.ceil(time_bound / time_step))
    time = [i * time_step for i in range(0, number_points)]
    if time[-1] != time_bound:
        time.append(time_bound)
    time_seq = np.arange(0.0, time_step, time_step / 10)
    # Simulate the system
    state = list(initial_condition)
    trace = [[0.] + list(state[:9]),]
    for i, t in enumerate(time[1:]):
        desired_state = compute_desired_state(state, mode, time_step)
        action = control(state, desired_state, action)
        out = odeint(dynamics, state, time_seq, args=(action,))
        state = out[-1]
        # Construct trace
        trace.append([t] + list(state[:9]))
    return np.array(trace)


if __name__ == "__main__":
    set_lower = np.array([1-0.125, 1-0.125, 4.95, -0.1, -0.1, -0.01, -0.01, -0.01, 0.0])
    set_higher = np.array([1+0.125, 1+0.125, 5.05, 0.1, 0.1, 0.01, 0.01, 0.01, 2 * np.pi - 1e-6])

    # import ipdb; ipdb.set_trace()
    # from IPython import embed; embed()
    goal = [2,2,6]
    traces = []
    for _ in range(100):
        initial_state = np.random.rand(len(set_lower)) * (set_higher - set_lower) + set_lower
        trace = TC_Simulate(goal, initial_state, 30.)
        traces.append(trace)

    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import matplotlib.pyplot as plt

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for trace in traces:
        ax.plot(trace[:,1], trace[:,2], trace[:,3], color='b')#, label='trace')

    # ax.legend()
    ax.scatter(goal[0], goal[1], goal[2], marker='o', color='r', s=12)
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
