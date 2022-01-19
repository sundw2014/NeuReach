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

import contextlib
import numpy as np
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
