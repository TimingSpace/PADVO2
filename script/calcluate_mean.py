import numpy as np
import sys
data = np.loadtxt(sys.argv[1])
if (len(data.shape)==2):
    mean = np.mean(data[-6:,:],0)
    print(mean[0],mean[1]*180/np.pi)
else:
    mean = np.mean(data[-100:])
    print(mean)

