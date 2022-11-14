from utils.preprocesses import RoiCrop, Resize
import os
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

if __name__ == '__main__':

    roi = RoiCrop()
    m = 512
    kh = 50
    kw = 10

    dpi = 600
    resize = Resize(height=m, width=m)
    root = '/mnt/HDD/industrial_part_recognition'
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    print(classes)


    for count, cls in enumerate(classes):
        path = '/home/kochpaul/git/multi-view-part-recognition/industrial_part_recognition/{}/train_data/0/0'.format(cls)
        out = '/home/kochpaul/images'
        print('{}/{} | {}'.format(count+1, len(classes), cls))

        ns = [10]
        views_ = list(os.listdir(path))
        random.shuffle(views_)

        for n in ns:

            h = m + (kh * (n-1))
            w = m + (kw * (n-1))
            img = np.ones((h, w, 3), dtype=np.uint8) * 255
            depth = np.zeros((h, w), dtype=np.uint8)
            print(h, w)
            j = 0
            views = views_[:n]
            for view in views:
                # print(view)
                _, nr = view.split('_')
                s = {'x': [Image.open(os.path.join(path, view, '{}_rgb.png'.format(nr)))],
                     'depth': [Image.open(os.path.join(path, view, '{}_depth.png'.format(nr)))],
                     'mask': [Image.open(os.path.join(path, view, '{}_rgb_mask.png'.format(nr)))],
                     'mode': 'test'}
                s = resize(roi(s))
                print(j * kh, j * kh + m, j * kw, j * kw + m, np.array(s['depth'][0]).shape)
                img[j * kh:j * kh + m, j * kw:j * kw + m, :] = np.array(s['x'][0])
                depth[j * kh:j * kh + m, j * kw:j * kw + m] = np.array(s['depth'][0])

                j += 1


            k = 0
            op = os.path.join(out, '{}_{}.png'.format(cls, k))
            opd = os.path.join(out, '{}_{}_depth.png'.format(cls, k))
            while os.path.exists(op):
                k += 1
                op = os.path.join(out, '{}_{}.png'.format(cls, k))
                opd = os.path.join(out, '{}_{}_depth.png'.format(cls, k))

            plt.imshow(img)
            plt.axis('off')
            plt.savefig(op, bbox_inches='tight', pad_inches=0, dpi=dpi)

            #plt.show()

            plt.imshow(depth)
            plt.axis('off')
            plt.savefig(op, bbox_inches='tight', pad_inches=0, dpi=dpi)


            input()







        xx = [1, 2]
        yy = [10, 2]

        for x, y in zip(xx, yy):

            n = x*y
            i = 0
            j = 0

            w = y * m
            h = x * m

            img = np.zeros((h, w, 3), dtype=np.uint8)

            #print(img.shape)
            views = views_[:n]
            for view in views:
                #print(view)
                _, nr = view.split('_')
                s = {'x': [Image.open(os.path.join(path, view, '{}_rgb.png'.format(nr)))],
                     'mask': [Image.open(os.path.join(path, view, '{}_rgb_mask.png'.format(nr)))],
                     'mode': 'test'}
                s = resize(roi(s))
                #samples.append(s)

                #print(i*m, (i+1)*m, j*m,(j+1)*m, np.array(s['x'][0]).shape)

                img[i*m:(i+1)*m, j*m:(j+1)*m, :] = np.array(s['x'][0])


                j +=1
                if j == y:
                    j = 0
                    i += 1


            plt.imshow(img)
            plt.axis('off')
            #plt.show()

            k = 0
            op = os.path.join(out, '{}_{}.png'.format(cls, k))
            while os.path.exists(op):
                k += 1
                op = os.path.join(out, '{}_{}.png'.format(cls, k))

            plt.savefig(op, bbox_inches='tight', pad_inches=0, dpi=dpi)








