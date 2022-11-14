from torch import nn
import models.fusion as fusion
import inspect
from models.encoders import get_encoder


__fusion__ = {
    'Squeeze&Excite': fusion.MultiViewSqueezeAndExciteFusionAdd,
    'SharedSqueeze&Excite': fusion.MultiViewSharedSqueezeAndExciteFusionAdd,
    'FC': fusion.FCMultiViewFusion,
    'Conv': fusion.ConvMultiViewFusion,
}


def get_model(args):
    model = get_encoder(args)
    if args.multiview:
        f = __fusion__.get(args.fusion)
        if f is None:
            raise ValueError('Fusion "" does not exist, {} are implemented'.format(args.fusion, __fusion__.keys()))
        num_views = len(args.views.split('-'))
        if 'depth' in args.input_keys:
            model = MultiViewRGBD(model, args.num_classes, num_views, f)
        else:
            model = MultiView(model, args.num_classes, num_views, f)

    return model


class MultiView(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, num_views: int, fusion: nn.Module):
        super(MultiView, self).__init__()
        self.encoder = encoder

        args = [arg.name for arg in inspect.signature(fusion).parameters.values()]
        arg_dict = {'channels': encoder.out_channels, 'num_views': num_views}
        args = {arg: arg_dict[arg] for arg in args if arg in arg_dict}
        self.multiViewFusion = fusion(**args)
        self.fc = nn.Linear(encoder.out_channels, num_classes)

    def forward(self, x):
        bs, i, c, h, w = x.shape
        x = x.view(bs * i, c, h, w)
        x = self.encoder(x)

        # get shape back again and flatten for each set
        x = x.view(bs, i, -1)

        # fuse and classify
        x = self.multiViewFusion(x)
        x = self.fc(x)
        return x


class MultiViewRGBD(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, num_views: int, fusion: nn.Module):
        super(MultiViewRGBD, self).__init__()
        self.encoder = encoder
        args = [arg.name for arg in inspect.signature(fusion).parameters.values()]
        arg_dict = {'channels': encoder.out_channels, 'num_views': num_views}
        args = {arg: arg_dict[arg] for arg in args if arg in arg_dict}
        self.multiViewFusion = fusion(**args)

        if self.encoder.out_channels == num_classes:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(encoder.out_channels, num_classes)


    def forward(self, x, depth):
        bs, i, c, h, w = depth.shape
        depth = depth.view(bs * i, c, h, w)
        bs, i, c, h, w = x.shape
        x = x.view(bs * i, c, h, w)
        x = self.encoder(x, depth)

        # get shape back again and flatten for each set
        x = x.view(bs, i, -1)

        # fuse and classify
        x = self.multiViewFusion(x)
        x = self.fc(x)

        return x

