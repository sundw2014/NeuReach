import torch
import torch.nn.functional as F
import numpy as np

class Model(object):
    def __init__(self, num_dim_output):
        super(Model, self).__init__()
        self.num_dim_output = num_dim_output

    def __call__(self, x):
        x = x.cpu().detach().numpy().reshape(-1)
        r = x[-2]
        t = x[-1]
        K = 1
        dt = t - self.t
        dt[dt < 0] = np.inf
        idx = dt.argmin()
        exp = (self.dt[:idx] * self.gammas[:idx]).sum() if idx > 0 else 0
        exp += (t - self.t[idx]) * (self.gammas[idx] if idx < len(self.gammas) else 0.)
        dis = r*K*np.exp(exp)
        return torch.from_numpy(1/dis*np.eye(self.num_dim_output)).unsqueeze(0)

    def load_state_dict(self, state_dict):
        self.gammas = state_dict[0]
        self.t = state_dict[1]
        self.dt = self.t[1:] - self.t[:-1]

def get_model(num_dim_input, num_dim_output):
    model = Model(num_dim_output)
    return model, model
