import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .utils import AR_Conv_share, AR_Conv_unique

class Model(nn.Module):
    def __init__(self, configs, **args):
        nn.Module.__init__(self)
        self.params(configs)
        self.ptn_block = nn.ModuleList()
        self.layers = 1
        for i in range(self.layers):
            self.ptn_block.append(Block(self.c_in, self.c_out, self.c_date, self.inp_len, self.pred_len, 
                               self.move_step, self.embed_period, self.poly_order, self.adj_heads, 
                               self.res_layers, self.channels_emb_dim, self.n_nodes, configs['device']))
            
        self.period_mlp = FDEmbedding(self.c_in, self.c_date, self.n_nodes, self.embed_period)

        # self.AR_conv = AR_Conv_share(channels=self.c_in, kernel_size=self.move_step, sym_pad=True, softmax=True)
        self.season_smooth = AR_Conv_unique(channels=self.c_in, kernel_size=self.move_step, sym_pad=True, softmax=True)
        
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
    
    def forward(self, x, x_mark, y_mark, **args):
        x_season = self.period_mlp(x_mark)
        y_season = self.period_mlp(y_mark)
        x_deseason = x - x_season

        
        for i in range(self.layers):
            y_trend, y_res = self.ptn_block[i](x_deseason)
            x_deseason = y_trend + y_res

        y = y_trend + y_season + y_res
        return y

class Block(nn.Module):
    def __init__(self, c_in, c_out, c_date, inp_len, pred_len, move_step, embed_period, poly_order, adj_heads, res_layers, channels_emb_dim, n_nodes, device):
        nn.Module.__init__(self)
        # self.period_mlp = FDEmbedding(c_in, c_date, n_nodes, embed_period)
        self.poly_mlp = Multi_Linear(inp_len, pred_len, c_in, n_nodes, poly_order)
        self.res_proj = res_projection(seq_in=inp_len, seq_out=pred_len,
                                       channels=c_in, nodes=n_nodes,
                                       adj_heads=adj_heads, layers=res_layers,
                                       hid_dim=channels_emb_dim,device=device)
        
        self.AR_conv = AR_Conv_share(channels=c_in, kernel_size=move_step, sym_pad=True, softmax=True)

    def decomp(self,x):
        x_trend = x
        x_trend = self.AR_conv(x_trend)
        x_detrend = x - x_trend
        return x_trend, x_detrend
    
    def forward(self, x_deseason):
        '''Series decomposition'''
        x_trend, x_res = self.decomp(x_deseason)
        
        '''Composition'''
        y_trend = self.poly_mlp(x_trend)
        
        y_res = self.res_proj(x_res)

        return y_trend, y_res

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
            x = self.chan_proj[i](x)
            x = res + self.tanh(x)
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
        value = self.tanh(value)
        value = self.dropout(value)
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
        g = self.tanh(g)
        g = self.dropout(g)
        # gate and update
        gate = self.gate_proj(x)
        gate = self.sigmoid(gate)
        y = gate*g + (1-gate)*x
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
        self.mlp = nn.Linear(c_in, c_out)
        self.gate_mlp1 = nn.Linear(c_in,c_out)
        self.gate_mlp2 = nn.Linear(c_in,c_out)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # param share
        x_share = self.mlp(x)
        x_share = self.dropout(x_share)
        # do not share
        h = [x.unsqueeze(-1)]
        x0 = x
        for i in range(1, self.order):
            x0 = x0*x
            x1 = self.bn[i-1](x0)
            # x1 = x0
            h.append(x1.unsqueeze(-1))
        h = torch.cat(h,dim=-1)
        x_uniq = torch.einsum('BCNID,CNIOD->BCNOD',(h, self.res_weight)) + self.res_bias
        x_uniq = self.order_mlp(x_uniq).squeeze(-1)
        x_uniq = x_uniq * self.scale/4 # "/4": bias added in 2 mlp; "scale": order -> 1, c_in -> c_out
        x_uniq = self.dropout(x_uniq)
        # cat
        gate1 = self.gate_mlp1(x)
        gate = F.sigmoid(gate1)
        # gate2 = self.gate_mlp2(x)
        # gate2 = F.sigmoid(gate2)
        y = gate*(x_uniq) + (1-gate)*(x_share) + x
        # y = x + x_share + x_uniq
        return y


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
        self.fusion = nn.Parameter(torch.randn((1,1,1,1,self.c_date,2*self.heads)))

    def forward(self, x):
        B,S,D = x.size()
        phases = self.phase_mlp(x.unsqueeze(-1)).reshape(B,S,D,self.channels,self.n_nodes,self.heads,-1).permute(0,3,4,1,2,5,6)
        v_sin = torch.sin(phases[...,0])
        v_cos = torch.cos(phases[...,1])
        values = torch.cat((v_sin,v_cos),dim=-1)
        y = self.fusion * values
        y = y.mean((4,5))
        return y