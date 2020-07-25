import egomotionprediction as ep
import numpy as np

quat = []
print(type(quat))
tran = []
for i in range(0,10):
    quat.append([0,0,np.sin(i*0.1),np.cos(i*0.1)])
    tran.append([0,0,0])


epo = ep.EgomotionPrediction()
res = epo.predict_patch(quat,tran)
print(np.array(res))

