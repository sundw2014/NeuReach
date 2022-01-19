class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def getAxisAlignedCircumscribedRectangleOfEllipsoid(P):
    import numpy as np
    # f(x) = x^T M x - 1
    assert P.ndim == 2
    assert P.shape[0] == P.shape[1]
    n = P.shape[0]
    M = P.T.dot(P)
    assert np.linalg.eig(M)[0].min() > 0

    # print(np.linalg.eig(M)[0].min())
    bounds = []
    for i in range(n):
        _M = M.copy()
        row_i = _M[i,:].copy()
        row_0 = _M[0,:].copy()
        _M[i,:] = row_0
        _M[0,:] = row_i

        col_i = _M[:,i].copy()
        col_0 = _M[:,0].copy()
        _M[:,i] = col_0
        _M[:,0] = col_i

        a1 = _M[0,0]
        a = _M[1:,0]
        b = _M[0,1:].T
        A_bot = _M[1:,1:]
        _x1_sq = 1 / (a1-b.T.dot(np.linalg.inv(A_bot)).dot(a))
        # print(_x1_sq)
        assert _x1_sq >= 0
        _x1 = np.sqrt(_x1_sq)
        # _xbot = -_x1 * b.T.dot(np.linalg.inv(A_bot))
        # _x = np.array([_x1] + _xbot.tolist()).reshape(-1,1)
        # _x.T.dot(M)
        # _x.T.dot(M).dot(_x)
        # import ipdb; ipdb.set_trace()
        bounds.append(_x1)
    return np.array(bounds)

def ellipsoid2AArectangle(P, center):
    import numpy as np
    bounds = getAxisAlignedCircumscribedRectangleOfEllipsoid(P)
    return np.array([center - bounds, center + bounds]).T.reshape(-1)

def loadTrainedModel(path):
    import torch
    import torch.nn.functional as F
    import numpy as np
    import time

    from model import get_model

    from config import num_dim_observable

    num_dim = 9
    model, forward = get_model(num_dim, num_dim_observable)
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    torch.backends.cudnn.benchmark = True
    return forward

def get_tube(initCond, initDelta, waypoint, TC_Simulate, beta):
    initCond[:3] -= waypoint

    from config import normalize, num_dim_observable, observe
    import numpy as np
    import torch
    # initCond: n array
    # initDelta: n array
    # beta = loadTrainedModel()

    T_MAX = 10.0

    # find circumscribed ball
    r = np.sqrt(((normalize(initCond) - normalize(initCond+initDelta))**2).sum())
    center = initCond
    ref_trace = TC_Simulate(center, T_MAX).tolist()
    ellipsoids = []
    reachsets = [waypoint.repeat(2) + np.array([initCond-initDelta, initCond+initDelta]).T[:3,:].reshape(-1), ]

    # for point in tqdm(trace[1::]):
    for point in ref_trace[1::]:
        P = beta(torch.tensor(center.tolist() + [r, point[0]]).view(1,-1).cuda())
        P = P.view(num_dim_observable,num_dim_observable)
        reachsets.append(waypoint.repeat(2) + ellipsoid2AArectangle(P.cpu().detach().numpy(), observe(np.array(point[1::]))))
        ellipsoids.append([observe(np.array(point[1::])), P.cpu().detach().numpy()])
    return ellipsoids, reachsets

def samplePointsOnAARectangle(bounds, K=100):
    import numpy as np
    bounds = bounds.reshape(-1, 2)
    n = bounds.shape[0]
    points = []
    for i in range(n):
        _b = bounds.copy()
        _b[i,:] = bounds[i,0]
        points.append((_b[:,1]-_b[:,0]).reshape(1,-1) * np.random.rand(K,n) + _b[:,0].reshape(1,-1))
        _b[i,:] = bounds[i,1]
        points.append((_b[:,1]-_b[:,0]).reshape(1,-1) * np.random.rand(K,n) + _b[:,0].reshape(1,-1))

    return np.concatenate(points, axis=0)

import gzip
import pickle
def savepklz(data_to_dump, dump_file_full_name):
    ''' Saves a pickle object and gzip it '''

    with gzip.open(dump_file_full_name, 'wb') as out_file:
        pickle.dump(data_to_dump, out_file)


def loadpklz(dump_file_full_name):
    ''' Loads a gziped pickle object '''

    with gzip.open(dump_file_full_name, 'rb') as in_file:
        dump_data = pickle.load(in_file)

    return dump_data
