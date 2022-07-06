import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import tensorly as tl
import numpy as np


#TFNN用のレイヤ
#TuckerTFFLayer: Tensor Feed Forward層
#TensorOutputLayer: 出力層

class TuckerTFFLayer(nn.Module):
    def __init__(self, input_rank, output_rank, device='cpu'):
        if len(input_rank) != len(output_rank):
            raise Exception(f'input and output dimensions must be same!: input dim = {len(input_rank)}, output dim = {len(output_rank)}')
        
        super().__init__()
        self.device = device
        np.random.seed(1111)
        self.factors = nn.ParameterList([nn.Parameter(torch.FloatTensor(np.random.standard_normal(size=(i, o))).T) for (i, o) in zip(input_rank, output_rank)])
    
    def forward(self, x):
        core = tl.tenalg.multi_mode_dot(x, self.factors, modes=list(np.arange(1, x.dim())))
        return core

class TensorOutputLayer(nn.Module):
    def __init__(self, core_shape, num_classes=1000):
        super().__init__()
        
        np.random.seed(2222)
        weights_shape = [num_classes] + list(core_shape)
        self.weights = nn.Parameter(torch.FloatTensor(np.random.standard_normal(size=weights_shape)))
        self.ndim = len(core_shape)
    
    def forward(self, core):
        a = torch.tensor([]).requires_grad_().to(self.device)
        for i in range(len(core)):
            a = torch.cat((a, tl.tenalg.inner(self.weights, core[i], n_modes=self.ndim)), 0)
        
        a = a.reshape(len(core), -1)
        return a