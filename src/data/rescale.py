import numpy as np
from transformation import *
import sys
import matplotlib.pyplot as plt
import evaluate

scale = float(sys.argv[3])
pose_data = np.loadtxt(sys.argv[1])
ground    = np.loadtxt(sys.argv[2])
#plt.plot(pose_data[:,3],pose_data[:,11])
#plt.plot(ground[:,3],ground[:,11])
motion = pose2motion(pose_data)
motion[:,3:12:4] = motion[:,3:12:4]*scale

pose_update = motion2pose(motion)
res = evaluate.evaluate(ground[:pose_update.shape[0],:],pose_update)
print(np.mean(res,1)*100)
print(180*np.mean(res,1)/np.pi)
np.savetxt(sys.argv[1]+'.rescale_'+str(scale),pose_update)

#plt.plot(pose_update[:,3],pose_update[:,11])
#plt.show()

