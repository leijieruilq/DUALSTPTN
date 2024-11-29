import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .utils import AR_Conv_share, AR_Conv_unique
from .block import FDEmbedding, Multi_Linear

class Model(nn.Module):
    def __init__(self, configs, **args):
        nn.Module.__init__(self)
        self.params(configs)

        
        self.trend_smooth = AR_Conv_unique(channels=self.c_in, kernel_size=self.move_step, sym_pad=None, softmax=True)

        self.period_mlp = FDEmbedding(self.c_in, self.c_date, self.n_nodes, self.embed_period)
        self.poly_mlp = Multi_Linear(self.inp_len, self.pred_len, self.c_in, self.n_nodes, self.poly_order)
        self.res_proj = res_projection(seq_in=self.inp_len, seq_out=self.pred_len,
                                       channels=self.c_in, nodes=self.n_nodes,
                                       adj_heads=self.adj_heads, layers=self.res_layers,
                                       hid_dim=self.channels_emb_dim,device=configs['device'])
        if not self.use_dft:
            self.AR_conv = AR_Conv_unique(channels=self.c_in, kernel_size=self.move_step, sym_pad=True, softmax=True)
    
    def dft_choose(self,x,split_point):
        freq = torch.fft.rfft(x)
        low_freq = freq[:]
        low_freq[...,split_point:] = 0.0
        x_low = torch.fft.irfft(low_freq)
        return x_low

    def params(self, configs):
        self.c_in = configs['c_in']
        self.c_out = configs['c_out']
        self.c_date = configs['c_date']
        self.dp_rate = configs['dropout']
        self.n_nodes = configs['n_nodes']
        self.inp_len = configs['inp_len']
        self.pred_len = configs['pred_len']
        self.adj_heads = configs['adj_heads']
        self.move_step = configs['move_step']
        self.poly_order = configs['poly_order']
        self.res_layers = configs['res_layers']
        self.embed_period = configs['embed_period']
        self.period_heads = configs['period_heads']
        self.channels_emb_dim = configs['channels_emb_dim']
        self.use_dft = configs['use_dft']
        self.split_point = configs['split_point']
        
    def forward(self, x, x_mark, y_mark, **args):
        # decomposition(分解)
        if self.use_dft:
            x_trend = self.dft_choose(x,split_point=self.split_point)
        else:
            x_trend = self.AR_conv(x) # 先分离出趋势项
        x_detrend = x - x_trend
        x_season = self.period_mlp(x_mark, series=x, deseason=None) # 计算季节项
        x_res = x_detrend - x_season # 计算残差项
        # Embedding
        trend_emb = None
        res_emb = None
        
        # composition(组合)
        y_trend = self.poly_mlp(x_trend) # 多项式网络+趋势项
        trend = torch.cat((x_trend[...,-self.move_step+1:],y_trend),dim=-1)
        y_trend = self.trend_smooth(trend) #再加上ar_season-->未来趋势项
        y_res = self.res_proj(x_res) #残差项-->未来残差项
        y_deseason = y_trend + y_res
        y_season = self.period_mlp(y_mark, series=None, deseason=y_deseason) #未来季节项
        # output
        y = y_trend + y_season + y_res
        return y


class res_projection(nn.Module):
    '''
    residual part mapping of series.
    '''
    def __init__(self, seq_in, seq_out, channels, nodes, adj_heads, layers=4, hid_dim=10, device='cpu'):
        nn.Module.__init__(self)
        self.mlp = nn.Linear(seq_in, seq_out)        
        self.time_proj = nn.ModuleList()
        self.chan_proj = nn.ModuleList()
        self.Layers = layers
        for _ in range(self.Layers):
            self.time_proj.append(gated_mlp(seq_in, seq_out))
            self.chan_proj.append(gated_gcn(seq_in, seq_out, channels, hid_dim, adj_heads, device=device))
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x = self.mlp(x)
        for i in range(self.Layers):
            res = x
            x = self.time_proj[i](x)
            x = F.tanh(x)
            x = self.chan_proj[i](x)
            x = res + F.tanh(x)
        y = x
        return y
    

class gated_mlp(nn.Module):
    def __init__(self, seq_in, seq_out):
        nn.Module.__init__(self)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.gate = nn.Linear(seq_in, seq_out)
        self.update = nn.Linear(seq_in, seq_out)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # value
        value = self.update(x)
        # value = self.tanh(value)
        # value = self.dropout(value)
        # gate and update
        gate = self.gate(x)
        gate = self.sigmoid(gate)
        y =  gate*value + (1-gate)*x
        return y


class gated_gcn(nn.Module):
    def __init__(self, seq_in, seq_out, channels, hid_dim, heads=3, device='cpu'):
        nn.Module.__init__(self)
        self.neg_inf = -1e9 * torch.eye(channels, device=device).unsqueeze(0)
        self.nodes1 = nn.Parameter(torch.randn((heads, channels, hid_dim)))
        self.nodes2 = nn.Parameter(torch.randn((heads, channels, hid_dim)))
        self.gate_proj = nn.Linear(seq_in, seq_out)
        self.update_proj = nn.Linear(heads*seq_in, seq_out)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # channels adjacency matrix
        B,C,N,T = x.size()
        adj = torch.matmul(self.nodes1, self.nodes2.transpose(1,2))
        adj = adj + self.neg_inf
        adj = self.softmax(adj)
        # sptial convolutional operate
        g = torch.einsum('HOI,BINS->BHONS', (adj,x)).permute(0,2,3,1,4).reshape(B,C,N,-1)
        g = self.update_proj(g)
        # g = self.tanh(g)
        # g = self.dropout(g)
        # gate and update
        gate = self.gate_proj(x)
        gate = self.sigmoid(gate)
        y = gate*g + (1-gate)*x
        return y

