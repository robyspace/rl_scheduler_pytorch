import torch, sys
import torch.nn as nn

class SelfSpatialAttn(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(SelfSpatialAttn, self).__init__()

        self.k_size = output_channel
        self.keyLayer = nn.Conv2d(input_channel, output_channel, 1) # Key
        self.queryLayer = nn.Conv2d(input_channel, output_channel, 1) # Query
        self.valueLayer = nn.Conv2d(input_channel, output_channel, 1) # Value
        self.softmax = nn.Softmax(dim = 2)  # dim = -1 equal to dim = 2, bur TensorRT don't support -1.
        self.dropout = nn.Dropout(0.4)
        self.gamma = nn.Parameter(torch.Tensor([0.]))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( N x C x H x W)
            returns :
                out   : self attention value + input feature
                attention: N x C x H x W
            Note:
                The original formula is : Similarity = softmax( (QxK^t) / Dk^1/2 )
                Weighted sum = Similarity x V

                I change the formula to : Similarity = softmax( (Q^txK) / Dk^1/2 )
                Weighted sum = V x Similarity

                Because use the original formula to compute attention weighted sum will lead to local minima.
        """
        # input x: [N, C, H, W]
        n, c, h, w = x.size()
        # print('Attention input size ', x.size())

        key   = self.keyLayer(x)   # B x C x H x W
        query = self.queryLayer(x) # B x C x H x W
        value = self.valueLayer(x) # B x C x H x W

        # compute similarity
        similarity = torch.matmul(key.permute(0, 1, 3, 2), query)
        k_size = self.k_size**0.5
        similarity = torch.div(similarity, k_size)

        # compute softmax_value
        softmax_value = self.softmax(similarity)

        # weighted sum
        output = torch.matmul(value, softmax_value)

        # shortcut
        output = self.gamma*output + x
        return output


class ChannelAttention(nn.Module):
    r'''
    Channel Attention from CBAM.
    Input size = (N, C, W, H)
    Output size = (N, C, W, H)
    Avg & Max output size = (N, C, 1, 1)
    '''
    def __init__(self, in_channels, reduction = 16):
        super(ChannelAttention,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size = 1),
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input size = (N, C, W, H)
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        weight = self.sigmoid(avgout + maxout) #通道权重,   size = (N, C, 1, 1)

        # same as weight.expand_as(x), but TensorRT doesn't support expand_as() or expand()
        output = x * weight #返回通道注意力后的值,  size = (N, C, W, H)
        return output


class SpatialAttention(nn.Module):
    r'''
    Spatial Attention from CBAM.
    Input size = (N, C, H, W)
    Output size = (N, C, H, W)
    Avg & Max output size = (N, C, 1, 1)

    Note : Original paper said that kernel size 7 is better than 3.
    '''
    def __init__(self, kernel_size = 7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('Input to Spatial Attn size ', x.size())
        avgout = torch.mean(x, dim=1, keepdim=True) #size = (batch,1,w,h), 1 is because the keepdim=True
        maxout, _ = torch.max(x, dim=1, keepdim=True) #size = (batch,1,w,h), 1 is because the keepdim=True
        x1 = torch.cat([avgout, maxout], dim=1) #size = (batch,2,w,h), dim=x -> 沿著x的維度拼接
        x1 = self.conv(x1)    #size = (batch,1,w,h)
        weight = self.sigmoid(x1)   #size = (batch,1,w,h)
        output = x * weight  #size = (batch,channels,w,h)
        return  output


class SimpleSelfAttention(nn.Module):
    def __init__(self, n_in:int, ks=1):#, n_out:int):
        super().__init__()
        self.conv = nn.Conv1d(n_in, n_in, ks, padding=ks//2, bias=False)
        self.gamma = nn.Parameter(torch.Tensor([0.]))
        self.n_in = n_in

    def forward(self,x):
        size = x.size()
        x = x.view(*size[:2],-1)   # (C, N)

        convx = self.conv(x)   # (C, C) * (C, N) = (C, N)   => O(N, C^2)
        xxT = torch.matmul(x, x.permute(0,2,1).contiguous())   # (C, N) * (N, C) = (C, C)   => O(N, C^2)
        o = torch.matmul(xxT, convx)   # (C, C) * (C, N) = (C, N)   => O(N, C^2)
        o = self.gamma * o + x
        return o.view(*size).contiguous()


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio = 16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelAttention(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialAttention()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out    