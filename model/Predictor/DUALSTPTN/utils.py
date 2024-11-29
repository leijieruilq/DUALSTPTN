import torch
import torch.nn as nn
import torch.nn.functional as F

def My_Padding(x, kernel_size, sym_pad):
    if sym_pad is None:
        pass
    elif sym_pad:
        pad1 = (kernel_size - 1) // 2
        pad2 = (kernel_size - 1) - pad1
        front = x[..., 0:1].repeat(1, 1, 1, pad1)
        end = x[..., -1:].repeat(1, 1, 1, pad2)
        x = torch.cat([front, x, end], dim=-1)
    else:
        pad = (kernel_size - 1)
        front = x[..., 0:1].repeat(1, 1, 1, pad)
        x = torch.cat([front, x], dim=-1)
    return x

class AR_Conv_share(nn.Module):
    def __init__(self, channels, kernel_size, sym_pad=True, softmax=True):
        nn.Module.__init__(self)
        self.sym_pad = sym_pad
        self.softmax = softmax
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.ones((1,1,kernel_size)))

    def forward(self,x):
        x = My_Padding(x, self.kernel_size, self.sym_pad)
        B,C,N,T_new = x.size()
        if self.softmax:
            weight = F.softmax(self.weight,dim=-1)
        else:
            weight = self.weight
        y = F.conv1d(x.reshape(-1,N,T_new), weight=weight, bias=None, stride=1, padding=0,
                              dilation=1, groups=1).reshape(B,C,N,-1)
        return y
    
class AR_Conv_unique(nn.Module):
    def __init__(self, channels, kernel_size, sym_pad=True, softmax=True):
        nn.Module.__init__(self)
        self.sym_pad = sym_pad
        self.softmax = softmax
        self.kernel_size = kernel_size
        self.groups = channels
        self.weight = nn.Parameter(torch.ones((channels,channels//channels,1,kernel_size)))

    def forward(self,x):
        # padding
        x = My_Padding(x,self.kernel_size,self.sym_pad)
        # activation
        if self.softmax:
            weight = F.softmax(self.weight,dim=-1)
        else:
            weight = self.weight
        # convolutional operation
        y = F.conv2d(x, weight=weight, bias=None, stride=1, padding=0,
                              dilation=1, groups=self.groups) #(手动定义卷积通道,7个)
        return y
    
class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool2d(kernel_size=(1,kernel_size), stride=(1,stride), padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        pad1 = (self.kernel_size - 1) // 2
        pad2 = (self.kernel_size - 1) - pad1
        front = x[..., 0:1].repeat(1, 1, 1, pad1)
        end = x[..., -1:].repeat(1, 1, 1, pad2)
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x