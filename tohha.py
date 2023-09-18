from utils.hha import getHHA
from PIL import Image
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing



def hhasets(name, path, sets):
    for s in sets:
        p = os.path.join(path, s)
        hha(p, name)

def hha(path, name=None):
    counter = 0
    times = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if 'depth.png' in file:
                try:
                    t = time.time()
                    cam_id = file.split('_')[0]
                    hha_out = os.path.join(root, '{}_hha.png'.format(cam_id))
                    if os.path.exists(hha_out):
                        # os.remove(hha_out)
                        # print('remove', hha_out)
                        continue

                    with open(os.path.join(root, '{}_meta.json'.format(cam_id))) as f:
                        meta = json.load(f)
                    cammat = np.array(meta['cameraMatrix']).reshape((3, 3))
                    d = np.array(Image.open(os.path.join(root, file)))
                    # print(np.mean(d))
                    # print(cammat)
                    hha = Image.fromarray(np.array(getHHA(cammat, d, d), dtype=np.uint8))

                    hha.save(hha_out)
                    if False:
                        print('hha_out', hha_out)
                        plt.subplot(1, 2, 1)
                        plt.imshow(d)
                        plt.subplot(1, 2, 2)
                        plt.imshow(hha)
                        plt.show()
                    counter += 1
                    times.append(time.time() - t)
                    if len(times) > 1000:
                        times = times[1:]
                    print('{} | counter: {}, tmean = {}s'.format(name, counter, np.mean(times)))


                except Exception as e:
                    print(e)
                    print('Broken at', name, root, file)



if __name__ == '__main__':
    path = '/mnt/share/more_datasets/MVIP/sets'
    s = list(os.listdir(path))
    print('s, ', len(s))
    p = 32
    mp = {}

    multiprocessing.set_start_method('spawn')

    n = len(s) // p
    sets = {}
    for k, cls in enumerate(s):
        i = k%p
        name = str(i).zfill(2)
        if name not in sets:
            sets[name] = []
        sets[name].append(cls)

    for name, s in sets.items():
        print(name, len(s))
        mp[name] = multiprocessing.Process(target=hhasets, args=(name, path, s))
        mp[name].daemon = False
        mp[name].start()
        
    for key, t in mp.items():
        t.join()
        print('join', key)


