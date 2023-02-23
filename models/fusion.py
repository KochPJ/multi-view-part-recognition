from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.transformer import TransformerEncoderDecoder
import math
from utils.stuff import load_fitting_state_dict



class WeightNet(nn.Module):
    def __init__(self, num_classes: int, out_channels: int, pc_embed_channels: int = 64, pc_scale=200*math.pi,
                 with_fc=True, pc_temp=2000):
        super(WeightNet, self).__init__()
        self.weight_fusion = nn.Sequential(
            nn.Linear(pc_embed_channels, pc_embed_channels * 2),
            nn.Linear(pc_embed_channels * 2, pc_embed_channels * 4),
            nn.Linear(pc_embed_channels * 4, pc_embed_channels * 8),
            nn.Linear(pc_embed_channels * 8, pc_embed_channels * 8),
            nn.Linear(pc_embed_channels * 8, out_channels),
        )

        self.with_fc = with_fc
        if self.with_fc:
            self.fc = nn.Linear(out_channels, num_classes)
            self.drop = nn.Dropout(0.5)
            self.out_channels = num_classes
        else:
            self.drop = None
            self.fc = None
            self.out_channels = out_channels

        self.pc_embed_channels = pc_embed_channels
        self.pc_scale = pc_scale
        self.pc_temp = pc_temp

    def forward(self, weight):
        weight = get_pos_embed(weight, self.pc_embed_channels, scale=self.pc_scale, temperature=self.pc_temp).squeeze(1)
        weight = self.weight_fusion(weight)
        if self.fc is not None:
            if self.training:
                weight = self.drop(weight)
            weight = self.fc(weight)
        return weight


class MaxPool(nn.Module):
    def __init__(self, num_views: int):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=num_views)

    def __call__(self, x):
        return self.pool(x.permute(0, 2, 1)).squeeze(-1)


class AveragePool(nn.Module):
    def __init__(self, num_views: int):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=num_views)

    def __call__(self, x):
        return self.pool(x.permute(0, 2, 1)).squeeze(-1)


class Mean(nn.Module):
    def __init__(self, num_views: int):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=num_views)

    def __call__(self, x):
        return torch.mean(x, dim=1)


class FCMultiViewFusion(nn.Module):
    def __init__(self, channels: int, num_views: int, num_fusion_layers: int = 3, pc_embed_channels=64,
                 pc_scale=200*math.pi, pc_temp=2000):
        self.pc_embed_channels = pc_embed_channels
        self.channels = channels
        self.num_views = num_views
        self.pc_scale = pc_scale
        self.pc_temp = pc_temp
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

    def forward(self, x, weight=None):
        x = x.flatten(1)
        if weight is not None:
            weight_ = torch.zeros((len(x), self.channels * self.num_views))
            weight_[:, :self.pc_embed_channels] = get_pos_embed(weight, self.pc_embed_channels,
                                                                scale=self.pc_scale, temperature=self.pc_temp
                                                                ).squeeze(1)
            x = x + weight_.to(x.device)
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


class SqueezeAndExciteFusionAddBiDirectional(nn.Module):
    '''
        Copied from https://github.com/TUI-NICR/ESANet
        paper title: Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis
        authros: Seichter, Daniel and K{\"o}hler, Mona and Lewandowski, Benjamin and Wengefeld, Tim and Gross, Horst-Michael
    '''
    def __init__(self, channels, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAddBiDirectional, self).__init__()
        self.se_rgb = SqueezeAndExcitation(channels, activation=activation)
        self.se_depth = SqueezeAndExcitation(channels, activation=activation)
        self.se_rgb2 = SqueezeAndExcitation(channels, activation=activation)
        self.se_depth2 = SqueezeAndExcitation(channels, activation=activation)

    def forward(self, rgb, depth):
        return self.se_rgb(rgb) + self.se_depth(depth), self.se_rgb2(rgb) + self.se_depth2(depth)


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
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

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
    def __init__(self, channels, num_views, layers=1, heads=8, with_positional_encoding=False, learnable_pe=False,
                 pc_embed_channels=64, pc_scale=200*math.pi, pc_temp=2000):
        super().__init__()
        self.num_views = num_views
        self.with_positional_encoding = with_positional_encoding
        self.learnable_pe = learnable_pe
        self.channels = channels
        self.pc_embed_channels = pc_embed_channels
        self.pc_temp = pc_temp
        self.pc_scale = pc_scale
        self.tf = Transformer(channels, layers, heads, channels, channels)
        if self.with_positional_encoding:
            if self.learnable_pe:
                self.pos_emb = nn.Embedding(self.num_views, self.channels)
            else:
                self.pos_emb = torch.zeros((self.num_views, self.channels))
                self.pos_emb[:, :self.pc_embed_channels] = get_pos_embed(torch.arange(0, self.num_views).unsqueeze(0),
                                                                         pc_embed_channels,
                                                                         scale=self.pc_scale,
                                                                         temperature=self.pc_temp).squeeze(0)
        else:
            self.pos_emb = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.with_positional_encoding and self.learnable_pe:
            nn.init.uniform_(self.pos_emb.weight)

    def __call__(self, x):
        if self.pos_emb is not None:
            if self.learnable_pe:
                pos_embed = self.pos_emb.weight
            else:
                pos_embed = self.pos_emb.to(x.device)
            x = x + pos_embed
        return self.tf(x)


class TransformerMultiViewHeadDecoder(nn.Module):
    def __init__(self, channels, num_views, layers=1, heads=8, with_positional_encoding=False, learnable_pe=False,
                 pc_embed_channels=16, pc_scale=200*math.pi, pc_temp=2000):
        super().__init__()
        self.num_views = num_views + 1
        self.with_positional_encoding = with_positional_encoding
        self.learnable_pe = learnable_pe
        self.channels = channels
        self.pc_embed_channels = pc_embed_channels
        self.pc_scale = pc_scale
        self.pc_temp = pc_temp
        self.tf = Transformer(channels, layers, heads, channels, channels)
        if self.with_positional_encoding:
            if self.learnable_pe:
                self.pos_emb = nn.Embedding(self.num_views, self.channels)
            else:
                self.pos_emb = torch.zeros((self.num_views, self.channels))
                self.pos_emb[:, :self.pc_embed_channels] = get_pos_embed(torch.arange(0, self.num_views).unsqueeze(0),
                                                                         pc_embed_channels,
                                                                         scale=self.pc_scale,
                                                                         temperature=self.pc_temp).squeeze(0)
        else:
            self.pos_emb = None

        self.query_emb = nn.Embedding(1, self.channels)

        self.reset_parameters()

    def reset_parameters(self):
        if self.with_positional_encoding and self.learnable_pe:
            nn.init.uniform_(self.pos_emb.weight)

    def __call__(self, x, weight=None):
        query = self.query_emb.weight.repeat(len(x), 1)
        if weight is not None:
            weight_ = torch.zeros((len(x), self.channels))
            weight_[:, :self.pc_embed_channels] = get_pos_embed(weight, self.pc_embed_channels,
                                                                scale=self.pc_scale,
                                                                temperature=self.pc_temp).squeeze(1)
            query = query + weight_.to(query.device)

        x = torch.hstack([query.unsqueeze(1), x])
        if self.pos_emb is not None:
            if self.learnable_pe:
                pos_embed = self.pos_emb.weight
            else:
                pos_embed = self.pos_emb.to(x.device)
            x = x + pos_embed
        return self.tf(x)[:, 0]


class TransfomerEncoderDecoderMultiViewHead(nn.Module):
    def __init__(self, channels, num_views, layers=1, heads=8, dim_feedforward=2048, activation="relu",
                 normalize_before=False, return_intermediate_dec=False, with_positional_encoding=True,
                 pc_embed_channels=64, learnable_pe=False, use_weightnet=False, pc_temp=2000, pc_scale=200*math.pi,
                 freeze_weightnet=False):
        super().__init__()
        self.tf = TransformerEncoderDecoder(d_model=channels, nhead=heads, num_encoder_layers=layers,
                                            num_decoder_layers=layers, dim_feedforward=dim_feedforward,
                                            activation=activation, normalize_before=normalize_before,
                                            return_intermediate_dec=return_intermediate_dec)
        self.query_embed = nn.Embedding(1, channels)

        self.with_positional_encoding = with_positional_encoding
        self.learnable_pe = learnable_pe
        self.pc_embed_channels = pc_embed_channels
        self.num_views = num_views
        self.channels = channels
        self.pc_scale = pc_scale
        self.pc_temp = pc_temp
        if self.with_positional_encoding:
            if self.learnable_pe:
                self.pos_emb = nn.Embedding(self.num_views, self.channels)
            else:
                self.pos_emb = torch.zeros((self.num_views, self.channels))
                self.pos_emb[:, :self.pc_embed_channels] = get_pos_embed(torch.arange(0, self.num_views).unsqueeze(0),
                                                                         pc_embed_channels,
                                                                         scale=self.pc_scale,
                                                                         temperature=self.pc_temp).squeeze(0)

        else:
            self.pos_emb = None

        self.use_weightnet = use_weightnet
        self.freeze_weightnet = freeze_weightnet
        if self.use_weightnet:
            weightet_path = '/home/kochpaul/git/multi-view-part-recognition/results/run100/WeightNet10000/WeightNet10000_best.ckpt'
            sd = torch.load(weightet_path, map_location='cpu')['state_dict']
            #print(sd.keys())

            self.weightNet = WeightNet(0, channels, with_fc=False, pc_scale=self.pc_scale, pc_temp=self.pc_temp,
                                       pc_embed_channels=self.pc_embed_channels)
            self.weightNet = load_fitting_state_dict(self.weightNet, sd)

            if self.freeze_weightnet:
                for param in self.weightNet.parameters():
                    param.requires_grad = False
                #for param in self.weightNet.parameters():
                #    print(param.requires_grad)

            self.query_emb = None
        else:
            self.weightNet = None
            self.query_emb = nn.Embedding(1, self.channels)
        self.reset_parameters()

    def reset_parameters(self):
        if self.with_positional_encoding and self.learnable_pe:
            nn.init.uniform_(self.pos_emb.weight)

    def __call__(self, x, weight=None):

        if self.pos_emb is not None:
            if self.learnable_pe:
                pos_embed = self.pos_emb.weight
            else:
                pos_embed = self.pos_emb.to(x.device)
        else:
            pos_embed = None

        if weight is not None:
            if self.weightNet is not None:
                query = self.weightNet(weight)
            else:
                weight = get_pos_embed(weight, self.pc_embed_channels,
                                       scale=self.pc_scale, temperature=self.pc_temp).squeeze(1)
                query = self.query_embed.weight.repeat(len(x), 1)
                query[:, :self.pc_embed_channels] = weight
        else:
            query = self.query_embed.weight.repeat(len(x), 1)


        out, _ = self.tf(x, None, query, pos_embed)
        return out.squeeze(1)

#def get_pos_embed(x, num_pos_feats=64, temperature=2000, scale=200*math.pi):
def get_pos_embed(x, num_pos_feats=16, temperature=2000, scale=100*math.pi):
    if scale is None:
        scale = 2 * math.pi
        x = x * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = (2 * torch.div(dim_t, 2, rounding_mode='trunc')) / num_pos_feats
    dim_t = temperature ** dim_t  # dim_t // 2
    dim_t = dim_t.to(x.device)
    x = x[:, :, None] / dim_t
    x = torch.stack((x[:, :, 0::2].sin(),
                     x[:, :, 1::2].cos()), dim=3).flatten(2)
    return x


if __name__ == '__main__':

    c = 1024
    v = 3
    m = TransfomerEncoderDecoderMultiViewHead(c, v, with_positional_encoding=False, learnable_pe=True,
                                              use_weightnet=True)
    #m = m.to('cuda:0')
    bs = 4
    weight = torch.rand((bs, 1)) * 5
    #print(weight)
    #weight = None

    x = torch.rand((bs, v, c))
    out = m(x, weight)#, weight.to('cuda:0')) #
    print(out.shape)#.to('cuda:0')



