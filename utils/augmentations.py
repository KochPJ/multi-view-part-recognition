import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from utils.stuff import random_flip
import random
import numpy as np
import torch
from PIL import Image

class RandomRotation:
    def __init__(self, min_degree=-180,  max_degree=180, uniform: bool = False):
        self.min_degree = min_degree
        self.max_degree = max_degree
        self.uniform = uniform

    def __call__(self, sample):
        if self.uniform:
            angle = random.uniform(self.min_degree, self.max_degree)
            sample['x'] = [F.rotate(x, angle) for x in sample.get('x')]
            if 'depth' in sample:
                sample['depth'] = [F.rotate(x, angle) for x in sample.get('depth')]
            if 'mask' in sample:
                sample['mask'] = [F.rotate(x, angle) for x in sample.get('mask')]
        else:
            for i, x in enumerate(sample['x']):
                angle = random.uniform(self.min_degree, self.max_degree)
                sample['x'][i] = F.rotate(x, angle)
                if 'depth' in sample:
                    sample['depth'][i] = F.rotate(sample['depth'][i], angle)
                if 'mask' in sample:
                    sample['mask'][i] = F.rotate(sample['mask'][i], angle)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'min_degree={0}'.format(self.min_degree)
        format_string += ', max_degree={0}'.format(self.max_degree)
        format_string += ', uniform={0})'.format(self.uniform)
        return format_string


class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.005):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        self.tf = transforms.ColorJitter(brightness=self.brightness,
                                         contrast=self.contrast,
                                         saturation=self.saturation,
                                         hue=self.hue)

    def __call__(self, sample):
        sample['x'] = [self.tf(x) for x in sample['x']]
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string



class DepthNoise:
    def __init__(self, max_noise=5.0):
        self.max_noise = max_noise

    def __call__(self, sample):
        if 'depth' in sample:
            for i in range(len(sample['depth'])):
                sample['depth'][i] = np.array(sample['depth'][i])
                noise = (torch.rand(sample['depth'][i].shape).numpy() * 2) - 1
                noise = np.array(noise * self.max_noise, dtype=np.int32)
                sample['depth'][i][sample['depth'][i] > self.max_noise] += noise[sample['depth'][i] > self.max_noise]
                sample['depth'][i] = Image.fromarray(sample['depth'][i])

        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class RandomFlip:
    def __init__(self, p: float = 0.5, horizontal: bool = True, vertical: bool = True, diagonal: bool = True,
                 uniform: bool = False):
        self.p = {'p': p}
        self.horizontal = horizontal
        self.vertical = vertical
        self.diagonal = diagonal
        self.n = 0
        self.uniform = uniform

        if self.horizontal:
            self.n += p
        self.p['h'] = self.n
        if self.vertical:
            self.n += 1
        self.p['v'] = self.n
        if self.diagonal:
            self.n += 1
        self.p['d'] = self.n

    def __call__(self, sample):

        x = sample.get('x')
        m = sample.get('mask')
        d = sample.get('depth')
        l = None
        for s in [x, m, d]:
            if s is not None:
                l = [None for _ in range(len(s))]
                break
        x = sample.get('x', l)
        m = sample.get('mask', l)
        d = sample.get('depth', l)

        if self.uniform:
            x, m, d = random_flip([x, m, d], self.p, self.n, self.horizontal, self.vertical, self.diagonal)
        else:
            out = [random_flip([x_, m_, d_], self.p, self.n, self.horizontal, self.vertical, self.diagonal)
                   for x_, m_, d_ in zip(x, m, d)]
            x = [o[0] for o in out]
            m = [o[1] for o in out]
            d = [o[2] for o in out]

        if x[0] is not None:
            sample['x'] = x
        if m[0] is not None:
            sample['mask'] = m
        if d[0] is not None:
            sample['depth'] = d

        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'p={0}'.format(self.p['p'])
        format_string += ', horizontal={0}'.format(self.horizontal)
        format_string += ', vertical={0}'.format(self.vertical)
        format_string += ', diagonal={0}'.format(self.diagonal)
        format_string += ', uniform={0})'.format(self.uniform)
        return format_string