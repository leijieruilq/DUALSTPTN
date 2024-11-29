import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class FDEmbedding(nn.Module):
    '''
    Fourier Decomposition Embedding.
    '''
    def __init__(self, channels, c_date, n_nodes, embed_period, heads=8, sigma=0.1):
        nn.Module.__init__(self)
        self.channels = channels
        self.c_date = c_date
        self.n_nodes = n_nodes
        self.heads = heads
        self.period = embed_period
        self.sigma = sigma
        self.gate_mlp = nn.Linear(96,96)
        self.direct_mlp = nn.Linear(5,self.channels)
        self.projection_init()

    def projection_init(self):
        self.phase_mlp = nn.Linear(1, 2*self.channels*self.n_nodes*self.heads) #(1,2*c*1*n) ，用来增加(channels,2,n)三个维度的信息
        # init weight
        period = self.period[:self.heads] # only heads length
        if len(period) < self.heads:
            period = period + [1 for _ in range(self.heads-len(period))] #不够就补1
        period = torch.tensor(period).reshape(1,1,self.heads,1) #(1,1,n,1)
        weight = torch.randn((self.channels,self.n_nodes,self.heads,2)) #(c,1,n,2)
        weight = 2*np.pi*period + self.sigma*weight #(c,1,n,2)          #角频率：(2*pi*t+ceta*eps)
        self.phase_mlp.weight.data = weight.reshape(-1,1)  #(c*1*n*2,1)
        # init bias
        self.phase_mlp.bias.data = 2*np.pi*torch.rand_like(self.phase_mlp.bias.data)  #初始相位：(2*pi)
        # heads and date info fusion
        self.fusion = nn.Parameter(torch.ones((1,1,1,1,self.c_date,2*self.heads))) #振幅和偏差
        # self.mlp = nn.Linear(self.c_date*2*self.heads,1)
        self.scale = self.c_date*2*self.heads
        self.dp2d = nn.Dropout2d(0.3)
        self.dp = nn.Dropout(0.3)

    def forward(self, x, series, deseason):
        B,S,D = x.size()
        phases = self.phase_mlp(x.unsqueeze(-1)).reshape(B,S,D,self.channels,self.n_nodes,self.heads,-1).permute(0,3,4,1,2,5,6) 
        #增加了(channels,2,n)三个维度信息 -> (batchsize,channels,1,in_len,c_date,n,2)
        v_sin = torch.sin(phases[...,0])  #(b,c,1,in_len,c_date,n)
        v_cos = torch.cos(phases[...,1])  #(b,c,1,in_len,c_date,n)
        values = torch.cat((v_sin,v_cos),dim=-1)  #(b,c,1,in_len,c_date,2*n)
        """
        # y = self.mlp(values.reshape(B,self.channels,self.n_nodes,-1,self.scale)).squeeze(-1)/self.scale
        """
        y = self.fusion * values #(b,c,1,in_len,c_date,2*n)
        """
        # y = self.dp2d(y.reshape(B,S*self.channels,self.c_date,-1))
        # y = y.mean((2,3)).reshape(B,self.channels,1,-1)
        """
        y = y.mean((4,5)) / self.scale
        if series is not None:
            deseason = series - y #减去傅里叶趋势项


        gate = self.gate_mlp(deseason) 
        gate = F.sigmoid(gate) #门控机制得到权重
        
        y2 = self.direct_mlp(x)
        """
        # y2 = self.dp(y2).reshape(B,-1,1,self.channels).transpose(1,3)
        """
        y2 = y2.reshape(B,-1,1,self.channels).transpose(1,3)
        y = gate * y + (1-gate) * y2 #将傅里叶信息和MLP信息利用门控机制进行融合在一起
        # else:
        #     gate = self.gate_mlp(deseason)
        #     gate = F.sigmoid(gate)
        return y
    

class Multi_Linear(nn.Module):
    '''
    Polynomial regress MLP.
    '''
    def __init__(self, c_in, c_out, channels, nodes, order=1):
        nn.Module.__init__(self)
        order = order
        self.scale = c_out/(order*c_in)
        self.order = order
        self.res_weight = nn.Parameter(torch.randn((channels, nodes, c_in, c_out, order)))
        self.res_bias = nn.Parameter(torch.randn((1, channels, nodes, 1, 1))) # batch. channels, nodes, c_out, order
        nn.init.xavier_uniform_(self.res_weight)
        nn.init.xavier_uniform_(self.res_bias)
        self.order_mlp = nn.Linear(order,1)
        self.bn = nn.ModuleList()
        for _ in range(order-1):
            self.bn.append(nn.BatchNorm2d(channels))
        self.gate_mlp1 = nn.Linear(c_in,c_out)
        self.gate_mlp2 = nn.Linear(c_in,c_out)
        self.dropout = nn.Dropout(0.3)
        # self.mlp = nn.Linear(c_in, c_out)
        self.mlp = nn.Sequential(
                    nn.Linear(c_in, c_out),
                    nn.ReLU(),
                    nn.Linear(c_out,c_out)
        )

    def forward(self, x):
        # param share
        x_share = self.mlp(x) #inp_len->pred_len  (b,c,n,pred_len)
        # x_share = self.dropout(x_share)
        # do not share
        h = [x.unsqueeze(-1)] #(b,c,n,seq_len,1) #(b,c,n,t,p)
        x0 = x
        for i in range(1, self.order):
            x0 = x0*x
            x1 = self.bn[i-1](x0)
            # x1 = x0
            h.append(x1.unsqueeze(-1))
        h = torch.cat(h,dim=-1) #(b,c,n,t,p)
        x_uniq = torch.einsum('BCNID,CNIOD->BCNOD',(h, self.res_weight)) + self.res_bias
        x_uniq = self.order_mlp(x_uniq).squeeze(-1)
        x_uniq = x_uniq# * self.scale/4 # "/4": bias added in 2 mlp; "scale": order -> 1, c_in -> c_out
        # x_uniq = self.dropout(x_uniq)
        # cat
        gate1 = self.gate_mlp1(x)
        gate1 = F.sigmoid(gate1)
        gate2 = self.gate_mlp2(x)
        gate2 = F.sigmoid(gate2)
        y = gate1*(x_uniq) + gate2*(x_share) #+ x 
        return y