import os
import json
import numpy as np


if __name__ == '__main__':
    dir = './results/run000'
    #dir = './new2'
    #dir = './results_lr_fuse'
    ignore_keys = ['name', 'outdir']

    arg_keys = ['multiview', 'roicrop', 'shuf_views', 'shuf_views_cw', 'shuf_views_vw', 'fusion',
                'depth_fusion',
                'input_keys',
                'load_keys',
                'views',
                'data_views',
                'encoder_path',
                'rotations',
                'rotation_aug',
                'flip_aug',
                'random_view_order']
    accs = []

    args = None
    for i, name in enumerate(sorted(list(os.listdir(dir)))):

        if 'new_res' not in name:
            continue

        with open(os.path.join(dir, name, '{}_logs.log'.format(name))) as f:
            logs = json.load(f)
        #print(logs.keys())
        if logs['acc_test'] is not None:
            print('{}, name: {}:\n                           '
                  'test acc: {}, valid acc: {}, train acc: {}, test loss: {}, valid loss: {}, train loss: {}'.format(
                i+1, name,
                np.round(logs['acc_test'], 3),
                np.round(logs['best_acc_valid'][0], 3),
                np.round(logs['best_acc_train'][0], 3),
                np.round(logs['loss_test'], 7),
                np.round(logs['best_loss_valid'][0], 7),
                np.round(logs['best_loss_train'][0], 7)
            ))
            accs.append(logs['acc_test'])
            if args is None:
                args = logs['args']
                for key in arg_keys:
                    print('         {} | {}'.format(key, args.get(key)))


            else:
                for key in args:
                    if args[key] != logs['args'][key] and key not in ignore_keys:
                        print('         {} | {}'.format(key, logs['args'].get(key)))

    print('{} | {}'.format(len(accs), accs))
    print('3v-rgb mean: {}, std={}, min: {}, max: {}, dis: {}'.format(
        np.mean(accs), np.std(accs), np.min(accs), np.max(accs), np.max(accs) - np.min(accs)))
