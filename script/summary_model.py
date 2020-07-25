import torch
import sys

model = torch.load(sys.argv[1])
print(model['vonet.0.0.weight'].shape)
print(model['vonet.1.0.weight'].shape)
print(model['vonet.2.0.weight'].shape)
print(model['vonet.3.weight'].shape)
