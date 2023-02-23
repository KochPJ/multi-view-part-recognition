import numpy as np
import sys
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import torch
import random


def bar_progress(mode, epoch, step, steps, lr, loss, acc, old_loss, old_acc, times, topk):

    progress = (epoch-1) * steps.get('train', 0) + (epoch-1) * steps.get('valid',0 )
    if mode == 'train':
        progress += step
        d_valid_steps = steps.get('valid', 0)
        d_train_steps = steps.get('train', 0) - step
        d_test_steps = steps.get('test', 0)
    elif mode == 'valid':
        progress += steps.get('train', 0) + step
        d_valid_steps = steps.get('valid', 0) - step
        d_train_steps = 0
        d_test_steps = steps.get('test', 0)
    else:
        progress += steps.get('train', 0) + steps.get('valid', 0) + step
        d_valid_steps = 0
        d_train_steps = 0
        d_test_steps = steps.get('test', 0) - step

    progress = progress / (steps.get('epochs', 0) * steps.get('train', 0) +
                           steps.get('epochs', 0) * steps.get('valid', 0) + steps.get('test', 0))

    loss_delta = float(np.round(loss - old_loss, 4)) if old_loss is not None else None
    acc_delta = float(np.round(acc-old_acc, 4)) if old_acc is not None else None

    a = step
    b = steps[mode]
    c = '{} step'.format(mode)

    d_epoch = steps.get('epochs', 0) - epoch

    tepoch = np.mean(times.get('epoch', [0])) if len(times.get('epoch', [])) > 0 else None
    ttrain = np.mean(times.get('train', [0]))
    tvalid = np.mean(times.get('valid', [0])) if len(times.get('valid', [])) > 0 else None
    ttest = np.mean(times.get('test', [0])) if len(times.get('test', [])) > 0 else None

    if tvalid is None:
        tvalid = ttrain
    if ttest is None:
        ttest = tvalid
    if tepoch is None:
        tepoch = ttrain * steps.get('train', 0) + tvalid * steps.get('valid', 0)

    eta = d_epoch * tepoch + d_train_steps * ttrain + d_valid_steps * tvalid + ttest * d_test_steps

    hours = int(eta / 3600)
    if hours > 24:
        days = int(hours / 24)
        hours -= days * 24
        eta -= (days * 24) * 3600
    else:
        days = 0
    eta -= hours * 3600
    minutes = int(eta / 60)
    eta -= minutes * 60
    sec = int(eta)

    eta = ''
    if days > 0:
        eta += str(days) + ':'
    eta += '{}:{}:{}'.format(str(hours) if len(str(hours)) > 2 else str(hours).zfill(2),
                             str(minutes).zfill(2), str(sec).zfill(2))

    if lr is not None:
        if isinstance(lr, list):
            lr = [float(np.round(v, 10)) for v in lr]
        else:
            lr = float(np.round(lr, 10))

    progress_message = "eta: {} | Progress {}% [{}/{}] epochs | {}% [{}/{}] {} | lr: {} | mean loss: {} |" \
                       "loss delta: {} | top {}: {}% | top {} delta: {}%".format(
        eta,
        float(np.round(progress * 100, 4)),
        epoch,
        steps.get('epochs'),
        float(np.round(a / b * 100, 2)),
        a,
        b,
        c,
        lr,
        float(np.round(loss, 4)),
        loss_delta,
        topk,
        float(np.round(acc, 4)),
        topk,
        acc_delta)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()




def resize_keep_ratio(image, wt, ht, fill=0, padding_mode='constant', interpolation=Image.Resampling.BICUBIC):
    if isinstance(image, Image.Image):
        w, h = image.size
    else:
        if len(image.shape) == 2:
            h, w = image.shape
        elif len(image.shape) == 3:
            h, w, c = image.shape

    scale = wt / w
    if h * scale > ht:
        scale = ht / h

    new_w = int(np.round(w * scale, 0))
    new_h = int(np.round(h * scale, 0))

    if isinstance(image, Image.Image):
        image = image.resize((new_w, new_h), interpolation)
    else:
        image = F.resize(image,
                         size=(new_h, new_w),
                         interpolation=interpolation)
    w_padding = (wt - new_w) / 2
    h_padding = (ht - new_h) / 2

    l_pad = w_padding if w_padding % 1 == 0 else w_padding + 0.5
    t_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    r_pad = w_padding if w_padding % 1 == 0 else w_padding - 0.5
    b_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5

    padding = [int(l_pad), int(t_pad), int(r_pad), int(b_pad)]

    image = F.pad(image, padding, fill, padding_mode)

    return image


def get_mask_extrems(mask):
    mask = np.array(mask)
    ys, xs = np.where(mask != 0)
    if len(ys) > 3 and len(xs) > 3:
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
    else:
        h, w = mask.shape[:2]
        if w == h:
            return (0, 0, w, h)
        else:
            l = min(h, w)
            h1 = int((h - l) // 2)
            w1 = int((w - l) // 2)
            return (w1, h1, l + w1, l + h1)


def get_center(bbox):
    return (int(bbox[0] + ((bbox[2]-bbox[0])/2)), int(bbox[1] + ((bbox[3]-bbox[1])/2)))


def get_wh(bbox):
    return (bbox[2]-bbox[0], bbox[3]-bbox[1])


def random_flip(x, p: dict, n: int, horizontal: bool, vertical: bool, diagonal: bool):
    if random.random() < p['p']:
        f = random.random()
        if horizontal and f <= p['h'] / n:
            if isinstance(x, list):
                for i, s in enumerate(x):
                    if s is not None:
                        x[i] = F.hflip(s)

                        #x = [F.hflip(s) for s in x if s i not None]
            else:
                if x is not None:
                    x = F.hflip(x)
        elif vertical and f <= p['v'] / n:
            if isinstance(x, list):
                #if x[0] is not None:
                #    x = [F.vflip(s) for s in x]
                for i, s in enumerate(x):
                    if s is not None:
                        x[i] = F.vflip(s)
            else:
                if x is not None:
                    x = F.vflip(x)
        elif diagonal and f <= p['d'] / n:
            if random.random() < 0.5:
                if isinstance(x, list):
                    for i, s in enumerate(x):
                        if s is not None:
                            x[i] = F.hflip(s)
                            x[i] = F.vflip(s)
                        #else:
                        #x = [F.hflip(s) for s in x]
                        #x = [F.vflip(s) for s in x]
                else:
                    if x is not None:
                        x = F.hflip(x)
                        x = F.vflip(x)
            else:
                if isinstance(x, list):
                    #if x[0] is not None:
                    for i, s in enumerate(x):
                        if s is not None:
                            #x = [F.vflip(s) for s in x]
                            #x = [F.hflip(s) for s in x]
                            x[i] = F.vflip(s)
                            x[i] = F.hflip(s)
                else:
                    if x is not None:
                        x = F.vflip(x)
                        x = F.hflip(x)
    return x


def lookup(n):
    """
            Args:
                n (int): number of plots
            returns:
                x, y (int): matplotlib grid x, y
            """
    if n == 1:
        x, y = 1, 1
    elif n == 2:
        x, y = 1, 2
    elif n == 3:
        x, y = 1, 3
    elif n == 4:
        x, y = 2, 2
    elif n in [5, 6]:
        x, y = 2, 3
    elif n in [7, 8]:
        x, y = 2, 4
    elif n == 9:
        x, y = 3, 3
    elif n in [10, 11, 12]:
        x, y = 3, 4
    elif n in [13, 14, 15]:
        x, y = 3, 5
    elif n == 16:
        x, y = 4, 4
    elif n in [17, 18, 19, 20]:
        x, y = 4, 5
    elif n in [21, 22, 23, 24, 25]:
        x, y = 5, 5
    elif n in [26, 27, 28, 29, 30]:
        x, y = 5, 6
    elif n in [31, 32, 33, 34, 35, 36]:
        x, y = 6, 6
    elif n in [37, 38, 39, 40, 41, 42]:
        x, y = 6, 7
    elif n in [43, 44, 45, 46, 47, 48, 49]:
        x, y = 7, 7
    elif n in [50, 51, 52, 53, 54, 55, 56]:
        x, y = 7, 8
    elif n in [57, 58, 59, 60, 61, 62, 63, 64]:
        x, y = 8, 8
    elif n in [65, 66, 67, 68, 69, 70, 71, 72]:
        x, y = 8, 9
    else:
        raise NotImplementedError
    return int(x), int(y)


def load_fitting_state_dict(arch, state_dict):
    keys = list(arch.state_dict().keys())
    wrong_key = 0
    wrong_shape = 0
    okay = 0
    for key, v in state_dict.items():
        if key not in keys:
            wrong_key += 1
            continue
        sd = {key: v}
        try:
            arch.load_state_dict(sd, strict=False)
        except Exception as e:
            #print(e)
            wrong_shape += 1
            continue
        okay += 1
    msg = 'Loaded {}/{} weights, wrong key: {}, wrong shape: {}, missing: {}'.format(
        okay, len(keys), wrong_key, wrong_shape, len(keys) - (okay+wrong_key+wrong_shape) )
    print(msg)
    return arch