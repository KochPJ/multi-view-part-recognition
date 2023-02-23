from torch import nn
import models.fusion as fusion
import inspect
from models.encoders import get_encoder
import math


__fusion__ = {
    'Squeeze&Excite': fusion.MultiViewSqueezeAndExciteFusionAdd,
    'SharedSqueeze&Excite': fusion.MultiViewSharedSqueezeAndExciteFusionAdd,
    'FC': fusion.FCMultiViewFusion,
    'Conv': fusion.ConvMultiViewFusion,
    'Transformer': fusion.TransformerMultiViewHead,
    'max-pool': fusion.MaxPool,
    'average-pool': fusion.AveragePool,
    'mean': fusion.Mean,
    'TransfomerEncoderDecoderMultiViewHead': fusion.TransfomerEncoderDecoderMultiViewHead,
    'TransformerMultiViewHeadDecoder': fusion.TransformerMultiViewHeadDecoder
}

__allows_weight__ = [fusion.TransfomerEncoderDecoderMultiViewHead, fusion.FCMultiViewFusion,
                     fusion.TransformerMultiViewHeadDecoder]
__keeps_shape__ = [fusion.TransformerMultiViewHead]


def get_model(args):
    model = get_encoder(args)
    if args.multiview:
        f = __fusion__.get(args.fusion)
        if f is None:
            raise ValueError('Fusion "" does not exist, {} are implemented'.format(args.fusion, __fusion__.keys()))
        num_views = len(args.views.split('-'))
        if 'depth' in args.input_keys:
            model = MultiViewRGBD(model, args.num_classes, num_views, f, args.tf_layers,
                                  multi_head_classification=args.multi_head_classification,
                                  rgbd_wise_multi_head=args.rgbd_wise_multi_head,
                                  with_positional_encoding=args.with_positional_encoding,
                                  learnable_pe=args.learnable_pe,
                                  pc_embed_channels=args.pc_embed_channels,
                                  rgbd_wise_mv_fusion=args.rgbd_wise_mv_fusion,
                                  use_weightnet=args.use_weightNet,
                                  pc_scale=args.pc_scale,
                                  pc_temp=args.pc_temp,
                                  freeze_weightnet=args.freeze_weightnet)
        elif 'x' in args.input_keys:
            model = MultiView(model, args.num_classes, num_views, f, args.tf_layers,
                              multi_head_classification=args.multi_head_classification,
                              with_positional_encoding=args.with_positional_encoding,
                              learnable_pe=args.learnable_pe,
                              pc_embed_channels=args.pc_embed_channels,
                              use_weightnet=args.use_weightNet,
                              pc_scale=args.pc_scale,
                              pc_temp=args.pc_temp,
                              freeze_weightnet=args.freeze_weightnet)
        elif 'weight' in args.input_keys:
            print('Using no multi view  model for input keys: {}'.format(args.input_keys))
            pass
        else:
            raise ValueError('Multiview model not implemented')

    return model

#class MultiViewDepth(nn.Module):


class MultiView(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, num_views: int, fusion: nn.Module,
                 tf_layers: int = 0, dropout: float = 0.5, multi_head_classification=False,
                 with_positional_encoding=False, learnable_pe=False, pc_embed_channels=64, cat_weight=True,
                 pc_temp=2000, pc_scale=200*math.pi, use_weightnet=False, freeze_weightnet=False):
        super(MultiView, self).__init__()
        self.encoder = encoder
        print('Multi view rgb multi_head_classification: {}'.format(multi_head_classification))


        args = [arg.name for arg in inspect.signature(fusion).parameters.values()]
        arg_dict = {'channels': encoder.out_channels,
                    'num_views': num_views,
                    'with_positional_encoding': with_positional_encoding,
                    'learnable_pe': learnable_pe,
                    'pc_embed_channels': pc_embed_channels,
                    'pc_temp': pc_temp,
                    'pc_scale': pc_scale,
                    'use_weightnet': use_weightnet,
                    'freeze_weightnet': freeze_weightnet}

        args = {arg: arg_dict[arg] for arg in args if arg in arg_dict}
        self.multiViewFusion = fusion(**args)


        if tf_layers > 0:
            self.tf = __fusion__['Transformer'](encoder.out_channels, tf_layers)
        else:
            self.tf = None

        args = [arg.name for arg in inspect.signature(self.multiViewFusion).parameters.values()]
        if 'weight' in args:
            self.allows_weight = True
        else:
            self.allows_weight = False
            print('allows_weight', self.allows_weight)

        self.cat_weight = cat_weight
        self.pc_embed_channels = pc_embed_channels

        if not self.allows_weight and self.cat_weight:
            self.weight_fusion1 = nn.Linear(encoder.out_channels + pc_embed_channels, encoder.out_channels)
            self.weight_fusion2 = nn.Linear(encoder.out_channels, encoder.out_channels)
        else:
            self.weight_fusion1 = None
            self.weight_fusion2 = None

        self.fc = nn.Linear(encoder.out_channels, num_classes)

        self.drop_out = nn.Dropout(p=dropout)
        self.drop_out2 = nn.Dropout(p=dropout)
        self.norm = nn.BatchNorm1d(encoder.out_channels)
        self.multi_head_classification = multi_head_classification
        self.view_fc = None
        if self.multi_head_classification:
            self.view_fc = nn.ModuleList([nn.Linear(encoder.out_channels, num_classes) for _ in range(num_views)])

    def forward(self, x, weight=None):
        bs, i, c, h, w = x.shape
        x = x.view(bs * i, c, h, w)
        x = self.encoder(x)

        x = self.norm(x)
        if self.training:
            x = self.drop_out(x)

        # get shape back again and flatten for each set
        x = x.view(bs, i, -1)

        # transformer encoder head for view fusion
        if self.multi_head_classification:
            multi = {}
            for j in range(i):
                multi[str(j)] = self.view_fc[j](x[:, j])
        else:
            multi = None
        if self.tf is not None:
            x = self.tf(x)

        # fuse and classify
        if weight is not None and self.allows_weight:
            x = self.multiViewFusion(x, weight)
        else:
            x = self.multiViewFusion(x)

        if self.training:
            x = self.drop_out2(x)

        if weight is not None and self.cat_weight and not self.allows_weight:
            weight = fusion.get_pos_embed(weight, self.pc_embed_channels).squeeze(1)
            #print(x.shape, weight.shape)
            x = torch.cat([weight, x], dim=1)
            #print(x.shape)
            x = self.weight_fusion2(self.weight_fusion1(x))

        x = self.fc(x)

        if self.multi_head_classification:
            multi['out'] = x
            x = multi
        return x


class MultiViewRGBD(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, num_views: int, fusion: nn.Module,
                 tf_layers: int = 0, dropout: float = 0.5, multi_head_classification=False, rgbd_wise_multi_head=False,
                 with_positional_encoding=False, learnable_pe=False, pc_embed_channels=64, rgbd_wise_mv_fusion=False,
                 pc_temp=2000, pc_scale=200*math.pi, use_weightnet=False, freeze_weightnet=False):
        super(MultiViewRGBD, self).__init__()
        print('Multi view RGBD multi_head_classification: {}, rgbd_wise_multi_head: {}'.format(
            multi_head_classification, rgbd_wise_multi_head))
        self.encoder = encoder

        self.allows_weight = False
        for mv_fusion in __allows_weight__:
            if isinstance(fusion, mv_fusion):
                self.allows_weight = True
                break

        if tf_layers > 0:
            self.tf = __fusion__['Transformer'](encoder.out_channels, tf_layers)
        else:
            self.tf = None
        if self.encoder.out_channels == num_classes:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(encoder.out_channels, num_classes)
        self.drop_out = nn.Dropout(p=dropout)
        self.drop_out2 = nn.Dropout(p=dropout)
        self.norm = nn.BatchNorm1d(encoder.out_channels)
        self.multi_head_classification = multi_head_classification
        self.rgbd_wise_multi_head = rgbd_wise_multi_head
        self.rgbd_wise_mv_fusion = rgbd_wise_mv_fusion if self.rgbd_wise_multi_head else False

        args = [arg.name for arg in inspect.signature(fusion).parameters.values()]
        arg_dict = {'channels': encoder.out_channels,
                    'num_views': num_views,
                    'with_positional_encoding': with_positional_encoding,
                    'learnable_pe': learnable_pe,
                    'pc_embed_channels': pc_embed_channels,
                    'pc_temp': pc_temp,
                    'pc_scale': pc_scale,
                    'use_weightnet': use_weightnet,
                    'freeze_weightnet': freeze_weightnet}

        if self.rgbd_wise_mv_fusion:
            arg_dict['num_views'] = num_views * 3

        args = {arg: arg_dict[arg] for arg in args if arg in arg_dict}
        self.multiViewFusion = fusion(**args)

        self.view_fc = None
        self.view_fc_color = None
        self.view_fc_depth = None
        self.norm_color = None
        self.norm_depth = None
        if self.multi_head_classification:
            self.view_fc = nn.Linear(encoder.out_channels, num_classes)
            if self.rgbd_wise_multi_head:
                self.view_fc_color = nn.Linear(encoder.out_channels, num_classes)
                self.view_fc_depth = nn.Linear(encoder.out_channels, num_classes)
                self.norm_color = nn.BatchNorm1d(encoder.out_channels)
                self.norm_depth = nn.BatchNorm1d(encoder.out_channels)

    def forward(self, x=None, depth=None, weight=None):
        if depth is not None:
            bs, i, c, h, w = depth.shape
            depth = depth.view(bs * i, c, h, w)
        if x is not None:
            bs, i, c, h, w = x.shape
            x = x.view(bs * i, c, h, w)
        x = self.encoder(x, depth)

        if isinstance(x, dict):
            x, xc, xd = x['out'], x['x'], x['xd']

        x = self.norm(x)
        if self.training:
            x = self.drop_out(x)

        # get shape back again and flatten for each set
        x = x.view(bs, i, -1)

        if self.multi_head_classification:
            multi = {}
            for j in range(i):
                multi[str(j)] = self.view_fc(x[:, j])
            if self.rgbd_wise_multi_head:
                xc = self.norm_color(xc)
                xd = self.norm_color(xd)
                if self.training:
                    xc = self.drop_out(xc)
                    xd = self.drop_out(xd)
                # get shape back again and flatten for each set
                xc = xc.view(bs, i, -1)
                xd = xd.view(bs, i, -1)
                if self.rgbd_wise_mv_fusion:
                    x = torch.cat([x, xc, xd], dim=1)

                for j in range(i):
                    multi[str(j)+'c'] = self.view_fc_color(xc[:, j])
                for j in range(i):
                    multi[str(j)+'xd'] = self.view_fc_depth(xd[:, j])
        else:
            multi = None

        # transformer encoder head for view fusion
        if self.tf is not None:
            x = self.tf(x)

        # fuse and classify
        if weight is not None:
            x = self.multiViewFusion(x, weight)
        else:
            x = self.multiViewFusion(x)

        if self.training:
            x = self.drop_out2(x)

        x = self.fc(x)
        if self.multi_head_classification:
            multi['out'] = x
            x = multi
        return x


if __name__ == '__main__':
    from models.encoders import ResNet, ResNetRGBD
    from models.encoders import __fusion__ as fn
    import torch
    f = 'mean'
    c = 512
    cls = 308
    v = 3
    d = 1
    m = MultiView(ResNet('50', c), num_classes=cls, num_views=v, fusion=__fusion__[f], tf_layers=1,
                  multi_head_classification=True)

    bs = 2
    x = torch.rand((bs, v, 3, 512, 512))

    o = m(x)
    if isinstance(o, dict):
        for key, val in o.items():
            if isinstance(val, torch.Tensor):
                print(key, val.shape)
            else:

                print(key, [val_.shape for val_ in val])
                from models.fusion import GCNNLoss
                crit = GCNNLoss()
                y = torch.randint(0, cls, (bs, 1)).long().squeeze(-1)
                loss = crit(val, y)
                print(loss)
    else:
        print(o.shape)

    m = MultiViewRGBD(ResNetRGBD('50', c, fusion=fn['Squeeze&Excite']), num_classes=c, num_views=v,
                      fusion=__fusion__[f], tf_layers=1, multi_head_classification=True,
                      rgbd_wise_multi_head=False)

    x = torch.rand((1, v, 3, 512, 512))
    xd = torch.rand((1, v, d, 512, 512))

    o = m(x, xd)
    if isinstance(o, dict):
        for key, val in o.items():
            print(key, val.shape)
    else:
        print(o.shape)

    m = MultiViewRGBD(ResNetRGBD('50', c, fusion=fn['Squeeze&Excite'], rgbd_wise_multi_head=True), num_classes=c, num_views=v,
                      fusion=__fusion__[f], tf_layers=1, multi_head_classification=True,
                      rgbd_wise_multi_head=True, rgbd_wise_mv_fusion=True)

    x = torch.rand((1, v, 3, 512, 512))
    xd = torch.rand((1, v, d, 512, 512))

    o = m(x, xd)
    if isinstance(o, dict):
        for key, val in o.items():
            print(key, val.shape)
    else:
        print(o.shape)
