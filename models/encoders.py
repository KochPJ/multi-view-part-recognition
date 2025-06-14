import os.path
import torch.nn as nn
import torchvision.models as models
from utils.stuff import load_fitting_state_dict
import models.fusion as dfusion
import torch
import inspect
import os
import math


__resnets__ = {
    '18': models.resnet18,
    '34': models.resnet34,
    '50': models.resnet50,
    '101': models.resnet101,
    '152': models.resnet152,
}

__resnets_channels__ = {
    '18': (64, 64, 128, 256, 512),
    '34': (64, 64, 128, 256, 512),
    '50': (64, 256, 512, 1024, 2048),
    '101': (64, 256, 512, 1024, 2048),
    '152': (64, 256, 512, 1024, 2048),
}

__efficientNets__ = {
    'B0': models.efficientnet_b0,
    'B1': models.efficientnet_b1,
    'B2': models.efficientnet_b2,
    'B3': models.efficientnet_b3,
    'B4': models.efficientnet_b4,
    'B5': models.efficientnet_b5,
    'B6': models.efficientnet_b6,
    'B7': models.efficientnet_b7
}

__efficient_channels__ = {
    'B0': (16, 24, 40, 112, 1280),
    'B1': (16, 24, 40, 112, 1280),
    'B2': (16, 24, 48, 120, 1408),
    'B3': (24, 32, 48, 136, 1536),
    'B4': (24, 32, 56, 160, 1792),
    'B5': (24, 40, 64, 176, 2048),
    'B6': (32, 40, 72, 200, 2304),
    'B7': (32, 48, 80, 224, 2560)
}


__fusion__ = {
    'Squeeze&Excite': dfusion.SqueezeAndExciteFusionAdd,
    'Conv': dfusion.ConvDepthFusion,
    'Squeeze&Excite-Bi-Directional': dfusion.SqueezeAndExciteFusionAddBiDirectional
}


def check_encoder(args, d):
    #encoder = d.get(args.model_version)
    if args.model_version not in d:
        raise ValueError('Model version "{}" does not exist for model "{}", {} are implemented'.format(
            args.model_version, args.model_name, d.keys()
        ))

def load_encoder_weights(args, encoder):
    print('args.encoder_path', args.encoder_path)
    if args.encoder_path:
        if os.path.exists(args.encoder_path):
            weights = torch.load(args.encoder_path)['state_dict']

            encoder = load_fitting_state_dict(encoder, weights)
            #del weights['arch.fc.weight']
            #del weights['arch.fc.bias']
            #encoder.load_state_dict(weights, strict=False)
            print('pre trained weights loaded from {}'.format(args.encoder_path))
        else:
            print('pre trained weights: {} do not exist'.format(args.encoder_path))
    return encoder


def get_encoder(args):
    if args.multiview or 'weight' in args.input_keys:
        out_channels = args.hidden_channels
    else:
        out_channels = args.num_classes

    if args.model_name == 'ResNet':
        check_encoder(args, __resnets__)
        if 'depth' in args.input_keys:
            fusion = __fusion__.get(args.depth_fusion)
            if fusion is None:
                raise ValueError('Depth fusion {} does not exist. {} are implemented'.format(
                    args.depth_fusion, __fusion__.keys()
                ))
            if 'x' not in args.input_keys:
                encoder = ResNetDepth(args.model_version, out_channels, False,
                                      depth_channels=1 if not args.depth2hha else 3)
            elif args.rgbd_version == 'v1':
                encoder = ResNetRGBD(args.model_version, out_channels, args.pretrained, fusion=fusion,
                                     depth_channels=1 if not args.depth2hha else 3, fuse_layers=args.fuse_layers)
            elif args.rgbd_version == 'v2':
                encoder = ResNetRGBDv2(args.model_version, out_channels, args.pretrained, fusion=fusion,
                                     depth_channels=1 if not args.depth2hha else 3, fuse_layers=args.fuse_layers)
            elif args.rgbd_version == 'v3':
                encoder = ResNetRGBDv3(args.model_version, out_channels, args.pretrained, fusion=fusion,
                                     depth_channels=1 if not args.depth2hha else 3, fuse_layers=args.fuse_layers)
            else:
                raise ValueError('version {} for ResNet RGBD ist not Implemented'.format(args.rgbd_version))

            if args.with_rednet_pretrained:
                encoder = load_red_net_pretrained(args.rednet_pretrained_path, args.overwrite_imagenet, encoder)
        elif 'x' in args.input_keys:
            encoder = ResNet(args.model_version, out_channels, args.pretrained)
        elif 'size' in args.input_keys:
            encoder = dfusion.PropertyNet(args.num_classes, out_channels, pc_embed_channels=args.pc_embed_channels,
                                          pc_scale=args.pc_scale, pc_temp=args.pc_temp)
        elif 'weight' in args.input_keys:
            encoder = dfusion.WeightNet(args.num_classes, out_channels, pc_embed_channels=args.pc_embed_channels,
                                        pc_scale=args.pc_scale, pc_temp=args.pc_temp)
        else:
            raise ValueError('No model implemented for the input keys {}'.format(args.input_keys))

    elif args.model_name == 'EffNet':
        check_encoder(args, __efficientNets__)
        if 'depth' in args.input_keys:
            raise NotImplementedError('Depth ist not implemented for EffNets')
        encoder = EfficientNet(args.model_version, out_channels, args.pretrained)
    else:
        raise ValueError('Model Name "{}" unknown'.format(args.model_name))

    encoder = load_encoder_weights(args, encoder)

    return encoder

def load_red_net_pretrained(path, overwrite, encoder):
    sd_ = encoder.state_dict()
    sd = torch.load(path, map_location='cpu')['state_dict']
    new_sd = {}
    counter = 1
    for key, v in sd.items():
        if '_d' in key:
            a = key.split('.')
            nkey = 'depth_encoder.arch.{}.{}'.format('_'.join(a[0].split('_')[:-1]), '.'.join(a[1:]))
        else:
            if not overwrite:
                continue
            nkey = 'color_encoder.arch.{}'.format(key)

        if nkey in sd_:
            counter += 1
            new_sd[nkey] = v

    missing = []
    for key in sd_.keys():
        if key not in new_sd:
            if 'num_batches_tracked' not in key and 'fc.' not in key and 'fusion' not in key:
                missing.append(key)

    #encoder.load_state_dict(new_sd, strict=False)
    encoder = load_fitting_state_dict(encoder, new_sd)
    print('loaded {}/{} for {} weights  from {}'.format(len(new_sd), len(sd), len(sd_), path))
    return encoder

class ResNet(nn.Module):
    def __init__(self, encoder: str, num_classes: int, pretrained: bool = True):
        super(ResNet, self).__init__()
        self.arch = __resnets__[encoder](pretrained=pretrained)
        if num_classes > 0:
            self.arch.fc = nn.Linear(self.arch.fc.in_features, num_classes, bias=True)
            self.out_channels = num_classes
        else:
            self.out_channels = self.arch.fc.in_features
            self.arch.fc = nn.Identity()

    def forward(self, x):
        return self.arch(x)


class ResNetDepth(nn.Module):
    def __init__(self, encoder: str, num_classes: int, pretrained: bool = False, depth_channels: int = 1):
        super(ResNetDepth, self).__init__()
        self.depth_encoder = ResNet(encoder, num_classes, pretrained=pretrained)
        self.depth_channels = depth_channels
        if depth_channels != 3:
            self.depth_encoder.arch.conv1 = nn.Conv2d(depth_channels,
                                                      64,
                                                      kernel_size=tuple(self.depth_encoder.arch.conv1.kernel_size),
                                                      stride=tuple(self.depth_encoder.arch.conv1.stride),
                                                      padding=tuple(self.depth_encoder.arch.conv1.padding),
                                                      bias=bool(self.depth_encoder.arch.conv1.bias))
        self.out_channels = self.depth_encoder.out_channels

    def forward(self, x=None, depth=None):
        return self.depth_encoder(depth)


class EfficientNet(nn.Module):
    def __init__(self, encoder: str, num_classes: int, pretrained: bool = False):
        super(EfficientNet, self).__init__()
        self.arch = __efficientNets__[encoder](pretrained=pretrained)
        if num_classes > 0:
            self.arch.classifier[1] = nn.Linear(self.arch.classifier[1].in_features, num_classes, bias=True)
            self.out_channels = num_classes
        else:
            self.out_channels = self.arch.classifier[1].in_features
            self.arch.classifier = nn.Identity()

    def forward(self, x):
        return self.arch(x)


class ResNetRGBD(nn.Module):
    def __init__(self, encoder: str, num_classes: int, pretrained: bool = False, fusion=None,
                 fuse_layers: bool = True, return_layers: bool = False, with_fc: bool = True,
                 num_fusion_layers: int = 3, depth_channels: int = 1, with_depth: bool = True,
                 rgbd_wise_multi_head=True):
        super(ResNetRGBD, self).__init__()
        print('ResNetRGBD: v1')

        self.num_classes = num_classes
        self.encoder = encoder
        self.return_layers = return_layers
        self.fuse_layers = fuse_layers
        self.with_fc = with_fc if num_classes > 0 else False
        c = num_classes if with_fc else 0
        self.color_encoder = ResNet(encoder, c, pretrained=pretrained)
        self.with_depth = with_depth
        self.out_channels = self.color_encoder.out_channels
        self.fusion_layer0 = None
        self.fusion_layer1 = None
        self.fusion_layer2 = None
        self.fusion_layer3 = None
        self.fusion_layer4 = None
        self.rgbd_wise_multi_head = rgbd_wise_multi_head
        if self.rgbd_wise_multi_head:
            self.color_fc = nn.Linear(self.color_encoder.arch.fc.in_features, self.out_channels)
            self.depth_fc = nn.Linear(self.color_encoder.arch.fc.in_features, self.out_channels)
        else:
            self.color_fc = None
            self.depth_fc = None

        if with_depth:
            self.depth_encoder = ResNet(encoder, 0, pretrained=False)
            if depth_channels != 3:
                self.depth_encoder.arch.conv1 = nn.Conv2d(depth_channels,
                                                          64,
                                                          kernel_size=tuple(self.depth_encoder.arch.conv1.kernel_size),
                                                          stride=tuple(self.depth_encoder.arch.conv1.stride),
                                                          padding=tuple(self.depth_encoder.arch.conv1.padding),
                                                          bias=bool(self.depth_encoder.arch.conv1.bias))

            self.fuse = True if fusion is not None else False
            channels = __resnets_channels__[encoder]
            args = [arg.name for arg in inspect.signature(fusion).parameters.values()]
            arg_dict = {'channels': channels[0], 'num_fusion_layers': num_fusion_layers}
            if self.fuse:
                self.fusion_layer0 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict}) \
                    if fuse_layers else None
                arg_dict['channels'] = channels[1]
                self.fusion_layer1 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict}) \
                    if fuse_layers else None
                arg_dict['channels'] = channels[2]
                self.fusion_layer2 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict}) \
                    if fuse_layers else None
                arg_dict['channels'] = channels[3]
                self.fusion_layer3 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict}) \
                    if fuse_layers else None
                arg_dict['channels'] = channels[4]
            self.fusion_layer4 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})

    def forward(self, x, depth):
        xs = {}
        if self.return_layers:
            xs['input'] = x
        x = self.color_encoder.arch.conv1(x)
        x = self.color_encoder.arch.bn1(x)
        x = self.color_encoder.arch.relu(x)

        if depth is not None and self.with_depth:
            depth = self.depth_encoder.arch.conv1(depth)
            depth = self.depth_encoder.arch.bn1(depth)
            depth = self.depth_encoder.arch.relu(depth)

        if self.fuse_layers and depth is not None and self.with_depth:
            x = self.fusion_layer0(x, depth)

        if self.return_layers:
            xs['layer0'] = x

        # first layer
        x = self.color_encoder.arch.maxpool(x)
        x = self.color_encoder.arch.layer1(x)

        if depth is not None and self.with_depth:
            depth = self.depth_encoder.arch.maxpool(depth)
            depth = self.depth_encoder.arch.layer1(depth)

        if self.fuse_layers and depth is not None and self.with_depth:
            x = self.fusion_layer1(x, depth)

        if self.return_layers:
            xs['layer1'] = x

        # second layer
        x = self.color_encoder.arch.layer2(x)

        if depth is not None and self.with_depth:
            depth = self.depth_encoder.arch.layer2(depth)

            if self.fuse_layers:
                x = self.fusion_layer2(x, depth)

        if self.return_layers:
            xs['layer2'] = x

        # third layer
        x = self.color_encoder.arch.layer3(x)

        if depth is not None and self.with_depth:
            depth = self.depth_encoder.arch.layer3(depth)

            if self.fuse_layers:
                x = self.fusion_layer3(x, depth)

        if self.return_layers:
            xs['layer3'] = x

        # fourth layer
        x = self.color_encoder.arch.layer4(x)

        if self.rgbd_wise_multi_head:
            xs['x'] = self.color_encoder.arch.avgpool(x)
            xs['x'] = torch.flatten(xs['x'], 1)
            xs['x'] = self.color_fc(xs['x'])

        if depth is not None and self.with_depth:
            depth = self.depth_encoder.arch.layer4(depth)
            if self.rgbd_wise_multi_head:
                xs['xd'] = self.depth_encoder.arch.avgpool(x)
                xs['xd'] = torch.flatten(xs['xd'], 1)
                xs['xd'] = self.depth_fc(xs['xd'])
            x = self.fusion_layer4(x, depth)

        if self.return_layers:
            xs['layer4'] = x

        if self.with_fc:
            x = self.color_encoder.arch.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.color_encoder.arch.fc(x)
            if self.return_layers:
                xs['out'] = x

        if self.rgbd_wise_multi_head:
            x = {'out': x, 'x': xs['x'], 'xd': xs['xd']}

        if self.return_layers:
            return list(xs.values())
        else:
            return x


class ResNetRGBDv2(nn.Module):
    def __init__(self, encoder: str, num_classes: int, pretrained: bool = False, fusion=None,
                 fuse_layers: bool = True, return_layers: bool = False, with_fc: bool = True,
                 num_fusion_layers: int = 3, depth_channels: int = 1, with_depth: bool = True,
                 rgbd_wise_multi_head=True):
        super(ResNetRGBDv2, self).__init__()
        print('ResNetRGBD: v2')

        self.num_classes = num_classes
        self.encoder = encoder
        self.return_layers = return_layers
        self.fuse_layers = fuse_layers
        self.with_fc = with_fc if num_classes > 0 else False
        c = num_classes if with_fc else 0
        self.color_encoder = ResNet(encoder, c, pretrained=pretrained)
        self.with_depth = with_depth
        self.out_channels = self.color_encoder.out_channels
        self.fusion_layer0 = None
        self.fusion_layer1 = None
        self.fusion_layer2 = None
        self.fusion_layer3 = None
        self.fusion_layer4 = None
        self.rgbd_wise_multi_head = rgbd_wise_multi_head
        if self.rgbd_wise_multi_head:
            self.color_fc = nn.Linear(self.color_encoder.arch.fc.in_features, self.out_channels)
            self.depth_fc = nn.Linear(self.color_encoder.arch.fc.in_features, self.out_channels)
        else:
            self.color_fc = None
            self.depth_fc = None

        if with_depth:
            self.depth_encoder = ResNet(encoder, 0, pretrained=False)
            if depth_channels != 3:
                self.depth_encoder.arch.conv1 = nn.Conv2d(depth_channels,
                                                          64,
                                                          kernel_size=tuple(self.depth_encoder.arch.conv1.kernel_size),
                                                          stride=tuple(self.depth_encoder.arch.conv1.stride),
                                                          padding=tuple(self.depth_encoder.arch.conv1.padding),
                                                          bias=bool(self.depth_encoder.arch.conv1.bias))

            self.fuse = True if fusion is not None else False
            channels = __resnets_channels__[encoder]
            args = [arg.name for arg in inspect.signature(fusion).parameters.values()]
            arg_dict = {'channels': channels[0], 'num_fusion_layers': num_fusion_layers}
            if self.fuse:
                self.fusion_layer0 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict}) \
                    if fuse_layers else None
                arg_dict['channels'] = channels[1]
                self.fusion_layer1 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict}) \
                    if fuse_layers else None
                arg_dict['channels'] = channels[2]
                self.fusion_layer2 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict}) \
                    if fuse_layers else None
                arg_dict['channels'] = channels[3]
                self.fusion_layer3 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict}) \
                    if fuse_layers else None
                arg_dict['channels'] = channels[4]
            self.fusion_layer4 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})

    def forward(self, x, depth):
        xs = {}
        if self.return_layers:
            xs['input'] = x
        x = self.color_encoder.arch.conv1(x)
        x = self.color_encoder.arch.bn1(x)
        x = self.color_encoder.arch.relu(x)

        if depth is not None and self.with_depth:
            depth = self.depth_encoder.arch.conv1(depth)
            depth = self.depth_encoder.arch.bn1(depth)
            depth = self.depth_encoder.arch.relu(depth)

        if self.fuse_layers and depth is not None and self.with_depth:
            depth = self.fusion_layer0(x, depth)

        if self.return_layers:
            xs['layer0'] = x

        # first layer
        x = self.color_encoder.arch.maxpool(x)
        x = self.color_encoder.arch.layer1(x)

        if depth is not None and self.with_depth:
            depth = self.depth_encoder.arch.maxpool(depth)
            depth = self.depth_encoder.arch.layer1(depth)

        if self.fuse_layers and depth is not None and self.with_depth:
            depth = self.fusion_layer1(x, depth)

        if self.return_layers:
            xs['layer1'] = x

        # second layer
        x = self.color_encoder.arch.layer2(x)

        if depth is not None and self.with_depth:
            depth = self.depth_encoder.arch.layer2(depth)

            if self.fuse_layers:
                depth = self.fusion_layer2(x, depth)

        if self.return_layers:
            xs['layer2'] = x

        # third layer
        x = self.color_encoder.arch.layer3(x)

        if depth is not None and self.with_depth:
            depth = self.depth_encoder.arch.layer3(depth)

            if self.fuse_layers:
                depth = self.fusion_layer3(x, depth)

        if self.return_layers:
            xs['layer3'] = x

        # fourth layer
        x = self.color_encoder.arch.layer4(x)

        if self.rgbd_wise_multi_head:
            xs['x'] = self.color_encoder.arch.avgpool(x)
            xs['x'] = torch.flatten(xs['x'], 1)
            xs['x'] = self.color_fc(xs['x'])

        if depth is not None and self.with_depth:
            depth = self.depth_encoder.arch.layer4(depth)
            if self.rgbd_wise_multi_head:
                xs['xd'] = self.depth_encoder.arch.avgpool(x)
                xs['xd'] = torch.flatten(xs['xd'], 1)
                xs['xd'] = self.depth_fc(xs['xd'])
            x = self.fusion_layer4(x, depth)

        if self.return_layers:
            xs['layer4'] = x

        if self.with_fc:
            x = self.color_encoder.arch.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.color_encoder.arch.fc(x)
            if self.return_layers:
                xs['out'] = x

        if self.rgbd_wise_multi_head:
            x = {'out': x, 'x': xs['x'], 'xd': xs['xd']}

        if self.return_layers:
            return list(xs.values())
        else:
            return x


class ResNetRGBDv3(nn.Module):
    def __init__(self, encoder: str, num_classes: int, pretrained: bool = False, fusion=None,
                 fuse_layers: bool = True, return_layers: bool = False, with_fc: bool = True,
                 num_fusion_layers: int = 3, depth_channels: int = 1, with_depth: bool = True,
                 rgbd_wise_multi_head=True):
        super(ResNetRGBDv3, self).__init__()
        print('ResNetRGBD: v3')
        self.num_classes = num_classes
        self.encoder = encoder
        self.return_layers = return_layers
        self.fuse_layers = fuse_layers
        self.with_fc = with_fc if num_classes > 0 else False
        c = num_classes if with_fc else 0
        self.color_encoder = ResNet(encoder, c, pretrained=pretrained)
        self.with_depth = with_depth
        self.out_channels = self.color_encoder.out_channels
        self.fusion_layer0 = None
        self.fusion_layer1 = None
        self.fusion_layer2 = None
        self.fusion_layer3 = None
        self.fusion_layer4 = None
        self.rgbd_wise_multi_head = rgbd_wise_multi_head
        if self.rgbd_wise_multi_head:
            self.color_fc = nn.Linear(self.color_encoder.arch.fc.in_features, self.out_channels)
            self.depth_fc = nn.Linear(self.color_encoder.arch.fc.in_features, self.out_channels)
        else:
            self.color_fc = None
            self.depth_fc = None

        if with_depth:
            self.depth_encoder = ResNet(encoder, 0, pretrained=False)
            if depth_channels != 3:
                self.depth_encoder.arch.conv1 = nn.Conv2d(depth_channels,
                                                          64,
                                                          kernel_size=tuple(self.depth_encoder.arch.conv1.kernel_size),
                                                          stride=tuple(self.depth_encoder.arch.conv1.stride),
                                                          padding=tuple(self.depth_encoder.arch.conv1.padding),
                                                          bias=bool(self.depth_encoder.arch.conv1.bias))

            self.fuse = True if fusion is not None else False
            channels = __resnets_channels__[encoder]
            args = [arg.name for arg in inspect.signature(fusion).parameters.values()]
            arg_dict = {'channels': channels[0], 'num_fusion_layers': num_fusion_layers}
            if self.fuse:
                self.fusion_layer0 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict}) \
                    if fuse_layers else None
                arg_dict['channels'] = channels[1]
                self.fusion_layer1 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict}) \
                    if fuse_layers else None
                arg_dict['channels'] = channels[2]
                self.fusion_layer2 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict}) \
                    if fuse_layers else None
                arg_dict['channels'] = channels[3]
                self.fusion_layer3 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict}) \
                    if fuse_layers else None
                arg_dict['channels'] = channels[4]
            self.fusion_layer4 = fusion(**{arg: arg_dict[arg] for arg in args if arg in arg_dict})

    def forward(self, x, depth):
        xs = {}
        if self.return_layers:
            xs['input'] = x
        x = self.color_encoder.arch.conv1(x)
        x = self.color_encoder.arch.bn1(x)
        x = self.color_encoder.arch.relu(x)

        if depth is not None and self.with_depth:
            depth = self.depth_encoder.arch.conv1(depth)
            depth = self.depth_encoder.arch.bn1(depth)
            depth = self.depth_encoder.arch.relu(depth)

        if self.fuse_layers and depth is not None and self.with_depth:
            x, depth = self.fusion_layer0(x, depth)

        if self.return_layers:
            xs['layer0'] = x

        # first layer
        x = self.color_encoder.arch.maxpool(x)
        x = self.color_encoder.arch.layer1(x)

        if depth is not None and self.with_depth:
            depth = self.depth_encoder.arch.maxpool(depth)
            depth = self.depth_encoder.arch.layer1(depth)

        if self.fuse_layers and depth is not None and self.with_depth:
            x, depth = self.fusion_layer1(x, depth)

        if self.return_layers:
            xs['layer1'] = x

        # second layer
        x = self.color_encoder.arch.layer2(x)

        if depth is not None and self.with_depth:
            depth = self.depth_encoder.arch.layer2(depth)

            if self.fuse_layers:
                x, depth = self.fusion_layer2(x, depth)

        if self.return_layers:
            xs['layer2'] = x

        # third layer
        x = self.color_encoder.arch.layer3(x)

        if depth is not None and self.with_depth:
            depth = self.depth_encoder.arch.layer3(depth)

            if self.fuse_layers:
                x, depth = self.fusion_layer3(x, depth)

        if self.return_layers:
            xs['layer3'] = x

        # fourth layer
        x = self.color_encoder.arch.layer4(x)

        if self.rgbd_wise_multi_head:
            xs['x'] = self.color_encoder.arch.avgpool(x)
            xs['x'] = torch.flatten(xs['x'], 1)
            xs['x'] = self.color_fc(xs['x'])

        if depth is not None and self.with_depth:
            depth = self.depth_encoder.arch.layer4(depth)
            if self.rgbd_wise_multi_head:
                xs['xd'] = self.depth_encoder.arch.avgpool(x)
                xs['xd'] = torch.flatten(xs['xd'], 1)
                xs['xd'] = self.depth_fc(xs['xd'])
            x, depth = self.fusion_layer4(x, depth)

        if self.return_layers:
            xs['layer4'] = x

        if self.with_fc:
            x = self.color_encoder.arch.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.color_encoder.arch.fc(x)
            if self.return_layers:
                xs['out'] = x

        if self.rgbd_wise_multi_head:
            x = {'out': x, 'x': xs['x'], 'xd': xs['xd']}

        if self.return_layers:
            return list(xs.values())
        else:
            return x


if __name__ == '__main__':
    path = '/home/kochpaul/Downloads/rednet_ckpt.pth'
    over = False
    m = ResNetRGBD('50', 1, fusion=__fusion__['Squeeze&Excite'])
    load_red_net_pretrained(path, over, m)
    m = ResNetRGBDv2('50', 1, fusion=__fusion__['Squeeze&Excite'])
    load_red_net_pretrained(path, over, m)
    m = ResNetRGBDv3('50', 1, fusion=__fusion__['Squeeze&Excite'])
    load_red_net_pretrained(path, over, m)
