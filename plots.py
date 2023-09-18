import os.path
import numpy as np
import matplotlib.pyplot as plt
import json

if __name__ == '__main__':
    root = './results/run000'


    nets = ['ResNet_34_nr3-c_1-6-9',
            'ResNet_34_nr3-e_1-6-9',
            'ResNet_34_nr3-d_1-6-9',
            'ResNet_34_nr3-f_1-6-9',
            'ResNet_34_nr3-g_1-6-9',
            'ResNet_34_nr3-h_1-6-9',
            'ResNet_34_nr3-i_1-6-9',
            #'ResNet_34_nr3-j_1-6-9',
            'ResNet_34_nr3_1-6-9_fuse_Conv',
            'ResNet_34_nr3_1-6-9',
            'ResNet_34_nr3_1-6-9_rot0-1-2-3-4-5-6-7-8-9-10-11'
            ]
    accs = []
    for i, name in enumerate(nets):
        if not os.path.exists(os.path.join(root, name, '{}_logs.log'.format(name))):
            print(name)
            continue
        with open(os.path.join(root, name, '{}_logs.log'.format(name))) as f:
            logs = json.load(f)

        acc = logs['acc_test']
        if acc is None:
            acc = logs['best_acc_valid'][0]
            print(acc, name)

        if acc is not None:
            accs.append(logs['acc_test'])
        else:
            print(name, 'uff')

    print('{} | {}'.format(len(accs), accs))
    print('3v-rgb mean: {}, std={}, min: {}, max: {}, dis: {}'.format(
        np.mean(accs), np.std(accs), np.min(accs), np.max(accs), np.max(accs)-np.min(accs)))


    fusion = [['ResNet_34_nr3_1-6-9_fuse_Conv', 'RGB (Conv)'],
              ['ResNet_34_nr3_1-6-9_fuse_FC', 'RGB (FC)'],
              ['ResNet_34_nr3_1-6-9_fuse_Squeeze&Excite', 'RGB (Squeeze&Excite)'],
              ['ResNet_34_nr3_1-6-9_fuse_SharedSqueeze&Excite', 'RGB (SharedSqueeze&Excite)'],
              ['ResNet_34_nr3_1-6-9_no_roicrop', 'RGB (Conv, No ROI Crop)'],
              ['ResNet_34_nr3_1-6-9', 'RGB (Conv, Shuff_views)'],
              ['ResNet_34_nr3_1-6-9_shuf_views_cw', 'RGB (Conv, Shuff_views cw)'],
              ['ResNet_34_nr3_1-6-9_shuf_views_vw', 'RGB (Conv, Shuff_views vw)'],
              ['ResNet_34_nr3_1-6-9_shuf_views_cw_shuf_views_vw', 'RGB (Conv, Shuff_views vw & cw)'],
              #['ResNet_34_nr3_1-6-9_fuse_Conv', 'RGB'],
              ['ResNet_34_nr3_1-6-9_dfuse_Squeeze&Excite', 'RGBD (Squeeze&Excite, Conv)'],
              ['ResNet_34_nr3_1-6-9_dfuse_Conv', 'RGBD (Conv, Conv)']]

    #fig, axs = plt.subplots(1, 1, constrained_layout=True)
    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    fig.suptitle('Fusion & Aug Exp', fontsize=16)
    accs, titles = [], []
    for i, (name, title) in enumerate(fusion):
        if not os.path.exists(os.path.join(root, name, '{}_logs.log'.format(name))):
            continue
        with open(os.path.join(root, name, '{}_logs.log'.format(name))) as f:
            logs = json.load(f)
        accs.append(logs['acc_test'])
        titles.append(title)

    print('###### {} results ####'.format('Fusion & Aug Exp'))
    for a, b in zip(titles, accs):
        print('{} | {}'.format(a, b))
    axs.bar(titles, accs)
    plt.show()

    views = [['ResNet_34_nr1-a_9', '1 View (data 1 view)'],
             #['ResNet_34_nr1-b_9', '1 View (data 3 views)'],
             #['ResNet_34_nr1-c_9', '1 View (data all views)'],
             ['ResNet_34_nr2_1-10', '2 Views'],
             ['ResNet_34_nr3_1-6-9_fuse_Conv', '3 Views'],
             ['ResNet_34_nr4_1-3-8-10', '4 Views'],
             ['ResNet_34_nr5_1-3-4-7-9', '5 Views'],
             ['ResNet_34_nr6_1-3-2-4-7-9', '6 Views'],
             ['ResNet_34_nr7_1-2-3-4-7-8-10', '7 Views'],
             ['ResNet_34_nr8_1-2-3-4-7-8-9-10', '8 Views'],
             ['ResNet_34_nr9_1-3-4-5-6-7-8-9-10', '9 Views'],
             ['ResNet_34_nr10_1-2-3-4-5-6-7-8-9-10', '10 Views']]


    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    fig.suptitle('Views Exp', fontsize=16)
    accs_views, titles = [], []
    for i, (name, title) in enumerate(views):
        if not os.path.exists(os.path.join(root, name, '{}_logs.log'.format(name))):
            print('missing', name)
            continue
        with open(os.path.join(root, name, '{}_logs.log'.format(name))) as f:
            logs = json.load(f)
        if logs['acc_test'] is None:
            continue
        accs_views.append(logs['acc_test'])
        titles.append(title)

    print('###### {} results ####'.format('Views Exp'))
    for a, b in zip(titles, accs_views):
        print('{} | {}'.format(a, b))

    axs.bar(titles, accs_views)
    plt.show()
    plt.show()

    views = [
             ['ResNet_34_nr3_1-6-9_fuse_Conv', '3 Views'],
             ['ResNet_34_nr3-b_1-6-9', '3 Views (data all views randomly sampled)'],
             ['ResNet_34_nr3_1-6-9_pretrained_1-a', '3 Views (data 1 view pretrained)'],
             ['ResNet_34_nr3_1-6-9_pretrained_1-b', '3 Views (data 3 view pretrained)'],
             ['ResNet_34_nr3_1-6-9_pretrained_1-c', '3 Views (data all view pretrained)'],
             ['ResNet_34_nr3_1-6-9_no_rot', '3 Views (no rotation aug)'],
             ['ResNet_34_nr3_1-6-9_no_flip', '3 Views (no flip aug)'],
             ['ResNet_34_nr3_1-6-9_no_flip_no_rot', '3 Views (no flip/rot aug)']]

    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    fig.suptitle('Views Aug Exp', fontsize=16)
    accs, titles = [], []
    for i, (name, title) in enumerate(views):
        if not os.path.exists(os.path.join(root, name, '{}_logs.log'.format(name))):
            continue
        with open(os.path.join(root, name, '{}_logs.log'.format(name))) as f:
            logs = json.load(f)
        accs.append(logs['acc_test'])
        titles.append(title)

    print('###### {} results ####'.format('Views Aug Exp'))
    for a, b in zip(titles, accs):
        print('{} | {}'.format(a, b))

    axs.bar(titles, accs)
    plt.show()

    rots = ['0', '0-5', '0-4-8', '0-3-6-9', '0-2-5-8-10', '0-2-4-6-8-10', '0-2-4-5-6-8-10', '0-1-3-5-6-8-9-10',
            '0-1-2-3-5-6-8-9-10', '0-1-2-3-5-6-8-9-10-11', '0-1-2-3-4-5-6-8-9-10-11',
            '0-1-2-3-4-5-6-7-8-9-10-11']

    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    fig.suptitle('Rot Exp', fontsize=16)
    accs_rots, titles = [], []
    for i, rot in enumerate(rots):
        name = 'ResNet_34_nr3_1-6-9_rot{}'.format(rot)
        title = '{} Rots'.format(len(rot.split('-')))
        if not os.path.exists(os.path.join(root, name, '{}_logs.log'.format(name))):
            continue
        with open(os.path.join(root, name, '{}_logs.log'.format(name))) as f:
            logs = json.load(f)
        accs_rots.append(logs['acc_test'])
        titles.append(title)

    print('###### {} results ####'.format('Rot Exp'))
    for a, b in zip(titles, accs_rots):
        print('{} | {}'.format(a, b))

    axs.bar(titles, accs_rots)
    plt.show()



    print('accs_rots {} | {}'.format(len(accs_rots), accs_rots))
    print('accs_views {} | {}'.format(len(accs_views), accs_views))

    plt.plot(list(range(1, len(accs_rots)+1)), accs_rots)
    plt.plot(list(range(1, len(accs_views)+1)), accs_views)
    plt.legend(['Image-sets', 'Views'])
    plt.ylabel('ACC [%]')
    #plt.xlabel('# off ...')
    plt.savefig('/home/kochpaul/plot.png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()


