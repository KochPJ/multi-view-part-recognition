import copy

import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from utils.stuff import random_flip
import random
import numpy as np
import torch
from PIL import Image
from utils.perlin_noise import random_binary_perlin_noise


class RandomRotation:
    def __init__(self, min_degree=-180,  max_degree=180, uniform: bool = False):
        self.min_degree = min_degree
        self.max_degree = max_degree
        self.uniform = uniform

    def __call__(self, sample):
        if self.uniform:
            angle = random.uniform(self.min_degree, self.max_degree)
            if 'x' in sample:
                sample['x'] = [F.rotate(x, angle) for x in sample.get('x')]
            if 'depth' in sample:
                sample['depth'] = [F.rotate(x, angle) for x in sample.get('depth')]
            if 'mask' in sample:
                sample['mask'] = [F.rotate(x, angle) for x in sample.get('mask')]
        else:
            key = 'x' if 'x' in sample else 'depth'
            for i, x in enumerate(sample.get(key, [])):
                angle = random.uniform(self.min_degree, self.max_degree)
                if 'x' in sample:
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
        if 'x' in sample:
            sample['x'] = [self.tf(x) for x in sample['x']]
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class RandomNoise:
    def __init__(self, p=0.01, disable=0.0, const=0.020):
        self.disable = disable
        self.p = p
        self.const = const
        self.rand = self.disable + ((1-self.disable) / 2)

    def __call__(self, sample):
        if 'weight' in sample:
            r = np.random.rand()
            if r < self.disable:
                if np.random.rand() > 0.5:
                    sample['weight'] = torch.Tensor([0.0])
                else:
                    sample['weight'] = torch.Tensor([float(np.random.rand()*10)])

            elif r < self.rand:
                sample['weight'] += (torch.rand([1]) * (self.p * 2)) - self.p
            else:
                sample['weight'] += (torch.rand([1]) * (self.const * 2)) - self.const

            if sample['weight'] < 0:
                sample['weight'] = torch.Tensor([0.0])

        if 'size' in sample:
            r = np.random.rand()
            if r < self.disable:
                if np.random.rand() > 0.5:
                    sample['size'][0] = torch.Tensor([
                        0
                        if np.random.rand() < 0.33 else s
                        for s in sample['size'][0]
                    ])
                else:
                    sample['size'][0] = torch.Tensor([
                        np.random.rand() * 0.4
                        if np.random.rand() < 0.33 else s
                        for s in sample['size'][0]
                    ])

            elif r < self.rand:
                sample['size'][0] = torch.Tensor([
                    s+(torch.rand([1]) * (self.p * 2)) - self.p
                    if np.random.rand() > 0.5 else s
                    for s in sample['size'][0]
                ])
            else:
                sample['size'][0] = torch.Tensor([
                    s + (torch.rand([1]) * (self.const * 2)) - self.const
                    if np.random.rand() > 0.5 else s
                    for s in sample['size'][0]
                ])

            sample['size'][sample['size'] < 0] = 0
            sample['size'] = torch.sort(sample['size']).values

        return sample


    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'disable={0}'.format(self.disable)
        format_string += ', p={0}'.format(self.p)
        format_string += ', const={0}'.format(self.const)
        format_string += ', rand={0})'.format(self.rand)
        return format_string


class DepthNoise:
    def __init__(self, max_noise: float = 3.0, to_meter=False, random_offset_range: float = 300,
                 perlin_noise_p: float = 0.5, zero_p: float = 0.0, zero_max: float = 0.2,
                 offset_p: float = 0.9, noise_color_p: float = 0.0, noise_color_max: float = 0.2,
                 p_random_color_noise: float = 0.33, p_random_color_gray: float = 0.33):
        if to_meter:
            max_noise = max_noise / 1000
            random_offset_range = random_offset_range / 1000

        self.max_noise = max_noise
        self.random_offset_range = random_offset_range
        self.offset_p = offset_p
        if self.offset_p > 0:
            self.apply_offset = True
        else:
            self.apply_offset = True

        self.perlin_noise_p = perlin_noise_p

        self.zero_max = zero_max
        self.zero_p = zero_p
        if self.zero_p > 0:
            self.apply_zeros = True
        else:
            self.apply_zeros = False

        self.noise_color_p = noise_color_p
        self.noise_color_max = noise_color_max
        if self.noise_color_p > 0:
            self.apply_noise_color = True
        else:
            self.apply_noise_color = False
        self.p_random_color_noise = p_random_color_noise
        self.p_random_color_gray = p_random_color_gray

    def __call__(self, sample):
        if 'depth' in sample:
            do_color = True if 'x' in sample and self.apply_noise_color else False
            if isinstance(sample['depth'], list):
                for i in range(len(sample['depth'])):
                    sample['depth'][i], depth_zeros = self.apply_depth_noise(sample['depth'][i])
                    if do_color and not depth_zeros:
                        sample['x'][i] = self.apply_color_noise(sample['x'][i])
            else:
                sample['depth'], depth_zeros = self.apply_depth_noise(sample['depth'])
                if do_color and not depth_zeros:
                    sample['x'] = self.apply_color_noise(sample['x'])
        return sample

    def apply_depth_noise(self, depth):
        depth = np.array(depth)
        depth_zeros = False
        if self.apply_zeros:
            if np.random.rand() < self.zero_p:
                h, w = depth.shape
                t = float(np.random.rand()) * self.zero_max

                if np.random.rand() < self.perlin_noise_p:
                    mask = random_binary_perlin_noise(shape=(h, w), t=t)
                else:
                    mask = torch.rand((h, w))
                    mask = mask < t

                depth[mask] = 0
                depth_zeros = True

        if self.apply_offset:
            if np.random.rand() < self.offset_p:
                offset = float((np.random.rand() - 0.5) * 2) * self.random_offset_range
                mask = depth > 0
                depth[mask] = depth[mask] + offset
                depth[depth < 0] = 0

        #noise = (torch.rand(depth.shape).numpy() * 2) - 1
        #noise = np.array(noise * self.max_noise, dtype=np.int32)
        #depth[depth > self.max_noise] += noise[depth > self.max_noise]

        #noise = (torch.rand(depth.shape).numpy() * 2) - 1
        #noise = noise * self.max_noise

        noise = torch.normal(0, self.max_noise, depth.shape).numpy()
        depth[depth > 0] = noise[depth > 0] + depth[depth > 0]
        depth[depth < 0] = 0

        depth = Image.fromarray(depth)
        return depth, depth_zeros

    def apply_color_noise(self, color):
        if np.random.rand() < self.noise_color_p:
            w, h = color.size
            color = np.array(color, dtype=np.uint8)
            t = float(np.random.rand()) * self.noise_color_max
            if np.random.rand() < self.perlin_noise_p:
                mask = random_binary_perlin_noise(shape=(h, w), t=t)
            else:
                mask = torch.rand((h, w))
                mask = mask < t

            r = float(np.random.rand())
            if r < self.p_random_color_noise:
                noise = np.array(torch.rand(color.shape).numpy() * 255, dtype=np.uint8)
                color[mask] = noise[mask]
            elif r < self.p_random_color_gray + self.p_random_color_noise:
                color[mask] = int(np.random.randint(100, 150))
            else:
                color[mask] = 0

            color = Image.fromarray(color)
        return color

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

        if x is None and d is None:
            return sample

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



if __name__ == '__main__':
    r = RandomNoise()
    weight = {'weight': torch.Tensor([1.0])}
    for _ in range(100):
        print(r( {'weight': torch.Tensor([1.0])}))