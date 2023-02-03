from torch import nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class FCMultiViewFusion(nn.Module):
    def __init__(self, channels: int, num_views: int, num_fusion_layers: int = 3):
        super().__init__()
        if num_fusion_layers > 1:
            c = channels * num_views
            step = int((c - channels) / num_fusion_layers)
            self.fusion = []
            for i in range(num_fusion_layers - 1):
                self.fusion.append(nn.Linear(c, c-step))
                c -= step

            self.fusion.append(nn.Linear(c, channels))
            self.fusion = nn.Sequential(*self.fusion)
        else:
            self.fusion = nn.Linear(channels * num_views, channels)

    def forward(self, x):
        x = x.flatten(1)
        return self.fusion(x)


class ConvMultiViewFusion(nn.Module):
    def __init__(self, num_views: int, num_fusion_layers: int = 3):
        super().__init__()
        if num_fusion_layers > 1:
            c = num_views
            step = int(num_views / num_fusion_layers)
            self.fusion = []
            for i in range(num_fusion_layers-1):
                self.fusion.append(nn.Conv2d(c, c-step, kernel_size=1))
                c -= step

            self.fusion.append(nn.Conv2d(c, 1, kernel_size=1))
            self.fusion = nn.Sequential(*self.fusion)
        else:
            self.fusion = nn.Conv2d(num_views, 1, kernel_size=1)

    def forward(self, x):
        return self.fusion(x.unsqueeze(-1)).flatten(1)


class FCSqueezeAndExcitation(nn.Module):
    '''
    Inspired by:
    Copied from https://github.com/TUI-NICR/ESANet
    paper title: Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis
    authros: Seichter, Daniel and K{\"o}hler, Mona and Lewandowski, Benjamin and Wengefeld, Tim and Gross, Horst-Michael
    '''
    def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super(FCSqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            activation,
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = self.fc(x)
        y = x * weighting
        return y


class MultiViewSqueezeAndExciteFusionAdd(nn.Module):
    '''
        Copied from https://github.com/TUI-NICR/ESANet
        paper title: Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis
        authros: Seichter, Daniel and K{\"o}hler, Mona and Lewandowski, Benjamin and Wengefeld, Tim and Gross, Horst-Michael
    '''
    def __init__(self, channels, num_views: int, activation=nn.ReLU(inplace=True)):
        super(MultiViewSqueezeAndExciteFusionAdd, self).__init__()
        self.fuse = nn.ModuleList(
            [FCSqueezeAndExcitation(channels, activation=activation) for _ in range(num_views)])

    def forward(self, x):
        views = x.shape[1]
        out = []
        for i in range(views):
            out.append(self.fuse[i](x[:, i]))
        return sum(out)


class MultiViewSharedSqueezeAndExciteFusionAdd(nn.Module):
    '''
        Copied from https://github.com/TUI-NICR/ESANet
        paper title: Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis
        authros: Seichter, Daniel and K{\"o}hler, Mona and Lewandowski, Benjamin and Wengefeld, Tim and Gross, Horst-Michael
    '''
    def __init__(self, channels, activation=nn.ReLU(inplace=True)):
        super(MultiViewSharedSqueezeAndExciteFusionAdd, self).__init__()
        self.fuse = FCSqueezeAndExcitation(channels, activation=activation)

    def forward(self, x):
        views = x.shape[1]
        out = []
        for i in range(views):
            out.append(self.fuse(x[:, i]))
        return sum(out)


class ConvDepthFusion(nn.Module):
    def __init__(self, channels: int, num_fusion_layers: int = 3):
        super().__init__()
        if num_fusion_layers > 1:
            c = channels * 2
            step = int(channels / num_fusion_layers)
            self.fusion = []
            for i in range(num_fusion_layers-1):
                self.fusion.append(nn.Conv2d(c, c-step, kernel_size=1))
                c -= step

            self.fusion.append(nn.Conv2d(c, channels, kernel_size=1))
            self.fusion = nn.Sequential(*self.fusion)
        else:
            self.fusion = nn.Conv2d(channels*2, channels, kernel_size=1)

    def forward(self, x, depth_x):
        return self.fusion(torch.cat((x, depth_x), dim=1))


class SqueezeAndExcitation(nn.Module):
    '''
    Copied from https://github.com/TUI-NICR/ESANet
    paper title: Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis
    authros: Seichter, Daniel and K{\"o}hler, Mona and Lewandowski, Benjamin and Wengefeld, Tim and Gross, Horst-Michael
    '''
    def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class SqueezeAndExciteFusionAdd(nn.Module):
    '''
        Copied from https://github.com/TUI-NICR/ESANet
        paper title: Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis
        authros: Seichter, Daniel and K{\"o}hler, Mona and Lewandowski, Benjamin and Wengefeld, Tim and Gross, Horst-Michael
    '''
    def __init__(self, channels, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels, activation=activation)
        self.se_depth = SqueezeAndExcitation(channels, activation=activation)

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = rgb + depth
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class TransformerMultiViewHead(nn.Module):
    def __init__(self, channels, layers=1, heads=8):
        super().__init__()
        self.tf = Transformer(channels, layers, heads, channels, channels)

    def __call__(self, x):
        return self.tf(x)


if __name__ == '__main__':
    c = 512
    m = TransformerMultiViewHead(c)
    x = torch.rand((1, 5, c))
    o = m(x)
    print(o.shape)



