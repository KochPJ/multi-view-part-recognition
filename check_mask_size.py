import os
from PIL import Image
import numpy as np

if __name__ == '__main__':
    path = '/mnt/share/more_datasets/MVIP/sets'

    hs = []
    ws = []
    cxs = []
    cys = []
    x0s = []
    x1s = []
    y0s = []
    y1s = []
    i = 0
    I = 0
    failed = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if '_rgb_mask_gen.png' in file:
                I += 1

    for root, dirs, files in os.walk(path):
        for file in files:
            if '_rgb_mask_gen.png' in file:
                i+= 1
                p = os.path.join(root, file)
                img = np.array(Image.open(p))
                #print(img.shape)
                ys, xs = np.where(img > 0)
                if len(ys) == 0 or len(xs) == 0:
                    failed.append(p)
                    print('failed {} | {}'.format(len(failed), failed))
                    continue

                x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
                w = x1 - x0
                h = y1 - y0
                cx = x0 + w / 2
                cy = y0 + h / 2

                hs.append(h)
                ws.append(w)
                cxs.append(cx)
                cys.append(cy)
                x0s.append(x0)
                x1s.append(x1)
                y0s.append(y0)
                y1s.append(y1)

                print('{}/{}'.format(i, I))
        #if i == 10:
        #    break
    print('w | mean = {}, std = {}'.format(np.mean(ws), np.std(ws)))
    print('h | mean = {}, std = {}'.format(np.mean(hs), np.std(hs)))
    print('x0 | mean = {}, std = {}'.format(np.mean(x0s), np.std(x0s)))
    print('x1 | mean = {}, std = {}'.format(np.mean(x1s), np.std(x1s)))
    print('y0 | mean = {}, std = {}'.format(np.mean(y0s), np.std(y0s)))
    print('y1 | mean = {}, std = {}'.format(np.mean(y1s), np.std(y1s)))
    print('cx | mean = {}, std = {}'.format(np.mean(cxs), np.std(cxs)))
    print('cy | mean = {}, std = {}'.format(np.mean(cys), np.std(cys)))

    print('failed {} | {}'.format(len(failed), failed))




    print(hha, rgb, hha/rgb * 100)