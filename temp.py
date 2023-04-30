import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


p_dims = [200, 600, 20000]
q_dims = p_dims[::-1]
dims = q_dims + p_dims[1:]
dims = [200, 600, 20000, 20000, 600, 200]
# layers = torch.nn.ModuleList([torch.nn.Linear(d_in, d_out) for d_in, d_out in zip(dims[:-1], dims[1:])])

temp_q_dims = q_dims[:-1] + [q_dims[-1] * 2]

q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(p_dims[:-1], p_dims[1:])])
print(temp_q_dims)
print(q_layers)
print(p_layers)