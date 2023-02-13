import os
import numpy as np
import json

if __name__ == '__main__':
    root = './MVIP/sets/'
    run = 'run003'
    name = 'ResNet_50_nr3_1-6-9_fuse_SharedSqueeze&Excite_no_TrFo-multihead'
    path = './results/{}/{}/{}_test_logs.log'.format(run, name, name)

    classes = list(os.listdir(root))


    super_classes = {}
    super_lut = {}

    for cls in classes:
        with open(os.path.join(root, cls, 'meta.json')) as f:
            meta = json.load(f)

            super_lut[cls] = meta['super_class']
            for s in meta['super_class']:
                if s not in super_classes:
                    super_classes[s] = {}


    print('super_lut', super_lut)
    print('super_classes', super_classes)

    with open(path) as f:
        logs = json.load(f)

    for cls, mat in logs['topk_cls_test'].items():
        for k, v in mat.items():
            for s in super_lut[cls]:
                if k not in super_classes[s]:
                    super_classes[s][k] = []
                super_classes[s][k].append(v)

    for s, r in super_classes.items():
        print('--- {} ---'.format(s))
        for k, v in r.items():
            print('     {}: {}'.format(k, np.mean(v)))



