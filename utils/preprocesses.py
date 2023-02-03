from utils.stuff import get_wh, get_center, get_mask_extrems, resize_keep_ratio
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import torch
import random


class Normalize:
    def __init__(self, mean=None, std=None, depth_mean=None, depth_std=None):
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        self.mean = mean
        self.std = std
        if isinstance(depth_mean, str):
            depth_mean = None
        if isinstance(depth_std, str):
            depth_std = None
        self.depth_mean = depth_mean
        self.depth_std = depth_std

    def __call__(self, sample):
        for i in range(sample['x'].shape[0]):
            sample['x'][i] = F.normalize(sample['x'][i], self.mean, self.std)

        if self.depth_mean is not None and self.depth_std is not None and 'depth' in sample:
            for i in range(len(sample['depth'])):
                sample['depth'][i] = F.normalize(sample['depth'][i].unsqueeze(0), self.depth_mean, self.depth_std)
            #print(sample['depth'].shape, 'tensor')
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'mean={0}'.format(self.mean)
        format_string += ', std={0}'.format(self.std)
        format_string += ', depth_mean={0}'.format(self.depth_mean)
        format_string += ', depth_std={0})'.format(self.depth_std)
        return format_string


class ToTensor:
    def __call__(self, sample):

        sample['x'] = torch.cat([F.to_tensor(x).unsqueeze(0) for x in sample['x']])
        if 'depth' in sample:
            sample['depth'] = torch.cat([torch.as_tensor(np.array(x), dtype=torch.float32).unsqueeze(0)
                                         for x in sample['depth']]).unsqueeze(1)
            #print(sample['depth'].shape, 'tensor')
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RoiCrop:
    def __init__(self, roi_inflation: float = 0.1,
                 train_random_zoom_max: float = 0.1, train_random_zoom_min: float = -0.1, zoom_mean: float = 0.0,
                 train_center_jitter: bool = True, train_no_crop: float = 0.0, zoom_std: float = 0.33,
                 p_min: float = 0.1, distribution: str = 'uni'):

        self.roi_inflation = roi_inflation + 1
        self.zoom_min = train_random_zoom_min
        self.zoom_max = train_random_zoom_max
        self.zoom_dist = self.zoom_max - self.zoom_min
        self.p_min = p_min
        self.zoom_mean = zoom_mean
        self.zoom_std = zoom_std
        self.distribution = distribution
        self.min_to_mean_zoom = self.zoom_mean - self.zoom_min
        self.max_to_mean_zoom = self.zoom_max - self.zoom_mean

        self.center_jitter = train_center_jitter
        self.train_no_crop = train_no_crop

    def get_normal_zoom(self):
        a = np.random.normal(0, self.zoom_std)
        if a > 1:
            a = 1
        elif a < -1:
            a = -1

        if a < 0:
            a = self.zoom_mean + (a * self.min_to_mean_zoom)
        else:
            a = self.zoom_mean + (a * self.max_to_mean_zoom)

        return a

    def get_p_zoom(self):
        r = np.random.rand()
        a = np.random.rand()
        if r < self.p_min:
            a = self.zoom_mean - (a * self.min_to_mean_zoom)
        else:
            a = self.zoom_mean + (a * self.max_to_mean_zoom)

        return a

    def get_balanced_zoom(self):
        a = (np.random.rand() - 0.5) * 2
        if a < 0:
            a = self.zoom_mean + (a * self.min_to_mean_zoom)
        else:
            a = self.zoom_mean + (a * self.max_to_mean_zoom)

        return a

    def get_uni_zoom(self):
        a = (np.random.rand() * self.zoom_dist) + self.zoom_min
        return a

    def __call__(self, sample):
        if 'mask' in sample and not sample.get('disable_roi-crop', False):
            for i, mask in enumerate(sample['mask']):

                if sample.get('mode') != 'train' or np.random.rand() > self.train_no_crop:
                    mask = np.array(mask, dtype=np.uint8)
                    crop = get_mask_extrems(mask)
                    wc, hc = get_wh(crop)
                    cx, cy = get_center(crop)
                    m = int(max(wc, hc) * self.roi_inflation)
                    if sample.get('mode') == 'train':
                        if self.distribution == 'uni':
                            a = self.get_uni_zoom()
                        elif self.distribution == 'normal':
                            a = self.get_normal_zoom()
                        elif self.distribution == 'p_min':
                            a = self.get_p_zoom()
                        elif self.distribution == 'balanced':
                            a = self.get_balanced_zoom()
                        else:
                            a = self.get_uni_zoom()
                        a = int(m * a)
                        m += a
                        if self.center_jitter and a > 0:
                            center_offset_x = int((np.random.rand() * a) - (a // 2))
                            center_offset_y = int((np.random.rand() * a) - (a // 2))
                            cx += center_offset_x
                            cy += center_offset_y

                    m = int(m // 2)
                    crop = (cx - m, cy - m, cx + m, cy + m)
                    sample['x'][i] = sample['x'][i].crop(crop)
                    sample['mask'][i] = sample['mask'][i].crop(crop)
                    if 'depth' in sample:
                        sample['depth'][i] = sample['depth'][i].crop(crop)
        return sample


class Pad:
    def __init__(self, size_divisor: int = 32):
        self.size_divisor = size_divisor

    def __call__(self, sample):
        for i in range(len(sample['x'])):
            w, h = sample['x'][i].size
            w_ = w // self.size_divisor
            h_ = h // self.size_divisor
            skip = True
            if w % self.size_divisor != 0:
                w_ += 1
                skip = False
            if h % self.size_divisor != 0:
                h_ += 1
                skip = False
            if skip:
                continue
            w_ = self.size_divisor * w_
            h_ = self.size_divisor * h_
            w1 = int((w_ - w) // 2)
            h1 = int((h_ - h) // 2)
            x = np.zeros((h_, w_, 3), dtype=np.uint8)
            x[h1:h1 + h, w1:w1 + w] = np.array(sample['x'][i], dtype=np.uint8)
            sample['x'][i] = Image.fromarray(x)
            if 'depth' in sample:
                x = np.zeros((h_, w_), dtype=np.float32)
                x[h1:h1 + h, w1:w1 + w] = sample['depth'][i]
                sample['depth'][i] = Image.fromarray(x)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'size_divisor={0})'.format(self.size_divisor)
        return format_string


class Resize:
    def __init__(self, height: int = 224, width: int = 224, keep_ratio: bool = True, fill: int = 0,
                 padding_mode: str = 'constant', multi_scale_training: bool = True, training_scale_low: float = 0.25,
                 training_scale_high: float = 0.1):
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.height = height
        self.width = width
        self.ratio = height / width
        self.keep_ratio = keep_ratio
        self.fill = fill
        self.padding_mode = padding_mode

        self.multi_scale_training = multi_scale_training
        self.training_scale_high = training_scale_high
        self.training_scale_low = training_scale_low

        if 0 < self.training_scale_low < 0.9 and 0 < self.training_scale_high < 0.9 and self.multi_scale_training:
            self.scale_range = (min(int(self.width * (1 - self.training_scale_low)),
                                    int(self.height * (1 - self.training_scale_low))),
                                max(int(self.width * (1 + self.training_scale_high)),
                                    int(self.height * (1 + self.training_scale_high))))
        else:
            self.scale_range = None

        self.call_counter = 0
        self.set_max = False
        self.img_scale = (self.width, self.height)
        self.init_resize(default=True)

    def init_resize(self, default=False, set_max=False):
        if self.scale_range is not None and not default and self.multi_scale_training:
            if set_max:
                self.img_scale = (self.scale_range[1], self.scale_range[1])
            else:
                r = random.randint(self.scale_range[0], self.scale_range[1])
                self.img_scale = (r, r)
        else:
            self.img_scale = (self.width, self.height)

    def __call__(self, sample):
        if self.scale_range is not None:
            if sample.get('mode') == 'train':
                if sample.get('checking_batch_size'):
                    if not self.set_max:
                        self.init_resize(set_max=True)
                        self.set_max = True
                    self.call_counter += 1
                elif (self.call_counter == sample.get('batch_size') or self.call_counter == 0):
                    self.init_resize()
                    self.call_counter = 1
                    self.set_max = False
                else:
                    self.call_counter += 1
                    self.set_max = False

        size = sample.get('resize_size', self.img_scale) # (self.width, self.height)

        if self.keep_ratio:
            for i in range(len(sample['x'])):
                sample['x'][i] = resize_keep_ratio(sample['x'][i], size[0], size[1], self.fill, self.padding_mode)

                if 'depth' in sample:
                    sample['depth'][i] = resize_keep_ratio(sample['depth'][i],
                                                           size[0], size[1], self.fill, self.padding_mode,
                                                           interpolation=Image.NEAREST)
                if 'mask' in sample:
                    sample['mask'][i] = resize_keep_ratio(sample['mask'][i],
                                                           size[0], size[1], self.fill, self.padding_mode,
                                                           interpolation=Image.NEAREST)
        else:
            for i in range(len(sample['x'])):
                sample['x'][i] = sample['x'][i].resize(size=size)
                if 'depth' in sample:
                    sample['depth'][i] = sample['depth'][i].resize(size=size, resample=Image.NEAREST)
                if 'mask' in sample:
                    sample['mask'][i] = sample['mask'][i].resize(size=size, resample=Image.NEAREST)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'height={0}, '.format(self.height)
        format_string += 'width={0}, '.format(self.width)
        format_string += 'ratio={0}, '.format(self.ratio)
        format_string += 'keep_ratio={0}, '.format(self.keep_ratio)
        format_string += 'fill={0}, '.format(self.fill)
        format_string += 'padding_mode={0})'.format(self.padding_mode)
        return format_string