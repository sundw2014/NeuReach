import numpy as np
import sys
a = np.loadtxt(sys.argv[1])
print(sys.argv[2].split('_')[-1] + '   Volume: %f   Error: %f'%(a[:,0].mean(), 1-a[:,1].mean()))
