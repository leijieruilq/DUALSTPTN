import math
import torch
import torch.nn as nn
import numpy as np
    
class DataEmbedding(nn.Module):
    def __init__(self):
        super(DataEmbedding, self).__init__()
        self.hyperparam()
        self.poly_fc_x = Multi_Linear(self.inp_len, self.pred_len, self.c_in, self.n_nodes, order=self.linear_order)
        self.poly_fc_y = Multi_Linear(self.inp_len, self.pred_len, self.c_in, self.n_nodes, order=self.linear_order)
        self.per_emb = FDEmbedding(self.c_in, self.c_date, self.n_nodes, self.embed_period, self.embed_heads)
        self.channel_emb = nn.Conv2d(2*self.c_in, self.c_model, (1,1))
        
        self.IN = nn.InstanceNorm2d(self.c_model)
        
    def hyperparam(self):
        self.c_date = 5 # channels of date
        self.c_in = 1 # channels of date, speed, flow...
        self.n_nodes = 207    # nodes
        self.c_model = 32    # dim of model, c_model = c_in for directly prediction
        self.embed_heads = 8 # Period nums
        self.embed_period = [0,0.1,0.5,2,3,5,7] # pre defined period
        self.linear_order = 8 # polynomial regression order
        self.device = 'cuda' # device
        self.inp_len = 12 # length in time
        self.pred_len = 12 # prediction length
        self.batch_size = 32 # batch size

    def forward(self, x, x_mark, y_mark):
        # period embedding
        x_poly = self.poly_fc_x(x)
        y_poly = self.poly_fc_y(x)
        x_per = self.per_emb(x_mark)
        y_per = self.per_emb(y_mark)
        
        x_token = self.channel_emb(torch.cat((x_poly, x_per), dim=1))
        y_token = self.channel_emb(torch.cat((y_poly, y_per), dim=1))

        x_token = self.IN(x_token)
        y_token = self.IN(y_token)
        return x_token, y_token

class Multi_Linear(nn.Module):
    def __init__(self, c_in, c_out, channels, nodes, order=1):
        nn.Module.__init__(self)
        order = order
        self.scale = c_out/(order*c_in)
        self.order = order
        self.res_weight = nn.Parameter(torch.randn((channels, nodes, c_in, c_out)))
        self.res_bias = nn.Parameter(torch.randn((1, channels, nodes, c_out, 1)))
        self.order_mlp = nn.Linear(order,1)

    def forward(self, x):
        '''
        There should be a Max Min scaler:
        x \in [0,1]
        '''
        h = [x.unsqueeze(-1)]
        x0 = x
        for _ in range(1,self.order):
            x0 = x0*x
            h.append(x0.unsqueeze(-1))
        h = torch.cat(h,dim=-1)
        y = torch.einsum('BCNID,CNIO->BCNOD',(h, self.res_weight)) + self.res_bias
        y = y*self.scale
        y = self.order_mlp(y).squeeze(-1)
        return y

class FDEmbedding(nn.Module):
    # Fourier Decomposition Embedding.
    def __init__(self, channels, c_date, n_nodes, embed_period, heads=8, sigma=1.0):
        nn.Module.__init__(self)
        self.channels = channels
        self.c_date = c_date
        self.n_nodes = n_nodes
        self.heads = heads
        self.period = embed_period
        self.sigma = sigma
        self.projection_init()

    def projection_init(self):
        self.phase_mlp = nn.Linear(1, 2*self.channels*self.n_nodes*self.heads)
        # init weight
        period = self.period[:self.heads] # only heads length
        if len(period) < self.heads:
            period = period + [1 for _ in range(self.heads-len(period))]
        period = torch.tensor(period).reshape(1,1,self.heads,1)
        weight = torch.randn((self.channels,self.n_nodes,self.heads,2))
        weight = 2*np.pi*period + self.sigma*weight
        self.phase_mlp.weight.data = weight.reshape(-1,1)
        # init bias
        self.phase_mlp.bias.data = 2*np.pi*torch.rand_like(self.phase_mlp.bias.data)
        # heads and date info fusion
        self.fusion_heads = nn.Linear(2*self.heads,1)
        self.fusion_dim = nn.Linear(self.c_date,1)
    
    def forward(self, x):
        '''
        There should be a Max Min scaler:
        x \in [0,1]
        '''
        B,S,D = x.size()
        phases = self.phase_mlp(x.unsqueeze(-1)).reshape(B,S,D,self.channels,self.n_nodes,self.heads,-1).permute(0,3,4,1,2,5,6)
        v_sin = torch.sin(phases[...,0])
        v_cos = torch.cos(phases[...,1])
        values = torch.cat((v_sin,v_cos),dim=-1)
        y = self.fusion_heads(values).squeeze(-1)
        y = self.fusion_dim(y).squeeze(-1)
        return y