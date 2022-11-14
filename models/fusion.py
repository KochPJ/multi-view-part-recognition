from torch import nn
import torch
import torch.nn.functional as F


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
            step = int(num_views/ num_fusion_layers)
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