import argparse
import copy
import os
import numpy as np
import json
import matplotlib.pyplot as plt



root = './results'
ks = ['1', '3', '5']
args = ['tf_layers', 'multi_head_classification', 'fusion', 'data_views']
res = {}
min_run = 8 # 8
runs = sorted([d for d in os.listdir(root) if 'run' in d])
for run in runs:
    if int(run[3:]) < min_run:
        continue
    for exp in os.listdir(os.path.join(root, run)):
        try:
            with open(os.path.join(root, run, exp, exp+'_logs.log')) as f:
                logs = json.load(f)
        except Exception as e:
            print(exp, e)
            continue

        if 'ResNet_50_nr3_1-6-9_fuse_' in exp and False:
            print(exp[len('ResNet_50_nr3_1-6-9_fuse_'):])
            for arg in args:
                print('     ',arg, logs['args'][arg])
        if 'normal' in exp and False:
            print('exp', exp)
            for key, arg in logs['args'].items():
                print('     ', key, arg)

        if logs['topk_test']['1'] is None:
            print('exp: {}, {} not finished'.format(run, exp))
            continue

        if 'Conv' in exp and 'Conv_' not in exp:
            print(exp, run, logs['args']['multi_head_classification'])

        if exp not in res:
            res[exp] = {
                'best_acc_valid': [],
                'best_acc_train': [],
                'best_loss_valid': [],
                'best_loss_train': [],
                'runs': [],
                'losses_train': [],
                'losses_valid': [],
                'loss_test': [],
                'topk_test': {k: [] for k in ks},
                'topk_valid': {k: [] for k in ks},
                'topk_train': {k: [] for k in ks},
                'args': logs['args']
            }
        res[exp]['runs'].append(run)


        for key, v in logs.items():
            if key in res[exp]:
                if 'topk' in key:
                    for k in v.keys():
                        res[exp][key][k].append(v[k])
                elif key != 'args':
                    if 'best_acc' in key or 'best_loss' in key:
                        v = v[0]
                    res[exp][key].append(v)

for i, (exp, v) in enumerate(res.items()):
    for key, r in v.items():
        if 'topk' in key:
            for k in r.keys():


                res[exp][key][k] = np.array(res[exp][key][k])
                res[exp][key][k] = {'m': np.mean(np.array(res[exp][key][k]), axis=0),
                                    's': np.std(np.array(res[exp][key][k]), axis=0),
                                    'max': np.max(np.array(res[exp][key][k]), axis=0),
                                    'min': np.min(np.array(res[exp][key][k]), axis=0),
                                    'l': len(res[exp][key][k]),
                                    'all': np.array(res[exp][key][k])}
                #try:
                #except Exception as e:
                #    #print(exp, key, res[exp][key][k].shape, np.array(res[exp][key][k]))
                #    raise e
        elif key not in ['runs', 'args']:

            try:
                res[exp][key] = np.array(res[exp][key])
                #print(exp, key, res[exp][key].shape, np.array(res[exp][key]))
                res[exp][key] = {'m': np.mean(np.array(res[exp][key]), axis=0),
                                 's': np.std(np.array(res[exp][key]), axis=0),
                                 'max': np.max(np.array(res[exp][key]), axis=0),
                                 'min': np.min(np.array(res[exp][key]), axis=0),
                                 'l': len(res[exp][key])}
            except:
                print(key, exp)
                for ii, l in enumerate(res[exp][key]):
                    if isinstance(l, list):
                        print('     ', ii, len(l), 'list', res[exp]['runs'][ii])
                    else:
                        print('     ', ii, l, 'list', res[exp]['runs'][ii])
                input()
            #print(key, res[exp][key]['m'].shape, res[exp][key]['s'].shape)
            #print(key, res[exp][key]['m'], res[exp][key]['s'])
            #try:

            #except:
            #    #print(exp, key, res[exp][key].shape, np.array(res[exp][key]))
            #    pass
    print(i+1, exp, len(v['runs']), v['runs'])



test_names = ['ResNet_50_nr3_1-6-9_dfuse_TFED_RGBDMVF_v1',
              'ResNet_50_nr3_1-6-9_dfuse_TFED_RGBDMVF_v2',
              'ResNet_50_nr3_1-6-9_dfuse_TFED_RGBDMVF_v3',
              'ResNet_50_nr3_1-6-9_dfuse_TFED_RGBD_v1',
              'ResNet_50_nr3_1-6-9_dfuse_TFED_RGBDM_v2',
              'ResNet_50_nr3_1-6-9_dfuse_TFED_RGBD_v3']


def norm(l):
    m = sum(l)
    l = [float(v) / m for v in l]
    return l

def norm_act(acts, keys=None):
    if keys is None:
        keys = ['mean', 'std']

    for key in keys:
        for k, l in acts[key].items():
            if k == 'uni':
                assert len(l) == 3
                continue
            acts[key][k] = norm(l)



    return acts


plt.style.use('seaborn-v0_8-whitegrid')
exps = []
plot_names = []
lines = []
fig, ax = plt.subplots()

for test_dir in test_names:

    gathered = {'mean': {'counts': [], 'max': []},
                'std': {'counts': [], 'max': []},
                'topk': {'mean': [], 'std': []}}

    for run in res[test_dir]['runs']:
        r = './results/{}'.format(run)
        with open(os.path.join(root, run, test_dir, test_dir + '_test_logs.log')) as f:
            logs = json.load(f)
        if 'activations' not in logs:
            continue

        #print(test_dir, run, logs.keys())
        logs['activations'] = norm_act(logs['activations'])
        #print(logs['activations'].keys())
        #print(logs['activations']['mean'])
        #print(logs['activations']['std'])
        #print(logs['activations']['topk'].keys())

        for key in ['mean', 'std', 'topk']:
            for k in ['counts', 'max', 'mean', 'std']:
                if k not in gathered[key]:
                    continue
                gathered[key][k].append(logs['activations'][key][k])

        keys = [key for key in logs.keys() if 'topk_test' in key]
        s = 'topk test: '
        print(test_dir, run)
        com = None
        for key in keys:
            if '_' in key and 'd' not in key and 'c'  not in key and len(key) > len('topk_test') and 'out' not in key:
                #print(key)
                if com is None:
                    com = logs[key]
                    for key_ in com.keys():

                        com[key_] = [com[key_]]
                else:
                    for key_, r in logs[key].items():
                        com[key_].append(r)

        print('fused top1: {}'.format(np.mean(com['1'])))
        for key in keys:
            if '_' in key and 'd' in key and 'c'  not in key and len(key) > len('topk_test') and 'out' not in key:
                if com is None:
                    com = logs[key]
                    for key_ in com.keys():

                        com[key_] = [com[key_]]
                else:
                    for key_, r in logs[key].items():
                        com[key_].append(r)
        if com is not None:
            print('depth top1: {}'.format(np.mean(com['1'])))

        for key in keys:
            if '_' in key and 'd' not in key and 'c' in key and len(key) > len('topk_test') and 'out' not in key:
                if com is None:
                    com = logs[key]
                    for key_ in com.keys():

                        com[key_] = [com[key_]]
                else:
                    for key_, r in logs[key].items():
                        com[key_].append(r)
        if com is not None:
            print('color top1: {}'.format(np.mean(com['1'])))

    for key in ['mean', 'std', 'topk']:
        for k in ['counts', 'max', 'mean', 'std']:
            if k not in gathered[key]:
                continue

            #print(key, k, np.array(gathered[key][k]).shape)

            gathered[key][k+'_avg'] = np.round(np.std(np.array(gathered[key][k]), axis=0) * 100, 1)

            gathered[key][k] = np.round(np.mean(np.array(gathered[key][k]), axis=0)*100, 1)

            #print(key, k, gathered[key][k].shape)


    #plt.plot(gathered['topk']['mean'][1])
    #plt.plot(gathered['topk']['mean'][2])

    name = test_dir[len('ResNet_50_nr3_1-6-9_fuse_'):]
    name = name.replace('multihead', 'w. MultiHead')
    name = name.replace('max-pool', 'Max Pool')
    name = name.replace('Conv_Tr', 'Tr+Conv')

    name = name.replace('-', ' ')
    name = name.replace('_', ' ')
    if 'MV' not in name:
        continue
    #plot_names.append(name)
    max_index = np.argmax(gathered['mean']['max'])

    #plot_names.append(test_dir[len('ResNet_50_nr3_1-6-9_fuse_'):] + '_2')
    #plot_names.append(test_dir[len('ResNet_50_nr3_1-6-9_fuse_'):] + '_3')

    length = len(gathered['topk']['mean'][0])
    length = np.arange(length) / length

    for i, c in enumerate(['Color', 'Fused', 'Depth']):

        #if c not in ['Color']:
        #    continue

        #if i < 2:
        #    continue
        line = np.array(gathered['topk']['mean'][i*3:+i*3+3])
        #print(c, sum(gathered['mean']['max'][i*3:+i*3+3]))
        #print(line.shape)
        line = np.mean(line, axis=0)
        line = line/np.max(line)
        #print(line.shape)
        plot_names.append(name + ' {}'.format(c))

        line_, = ax.plot(length, line, label=name+ ' {}'.format(c))
        lines.append(line_)

    alpha = .3
    # plt.fill_between(xd, acc_d - std_d, acc_d + std_d, alpha=alpha)
    #ax.fill_between(length, gathered['topk']['mean'][0] - gathered['topk']['mean_avg'][0],
    #                        gathered['topk']['mean'][0] + gathered['topk']['mean_avg'][0], alpha=alpha)
    #print(line_)
    #print(test_dir)
    #print(test_dir)
    #print('mean', gathered['mean'])
    #print(gathered['mean'])
    m_id, m_w = np.argmax(gathered['mean']['counts']), np.max(gathered['mean']['counts'])
    ps = ['{} \pm {}'.format(a, b) for a, b in zip(gathered['mean']['max'], gathered['mean']['max_avg'])]
    ps = '~|~'.join(ps)
    #print(ps)
    #print('{} ({} \pm {})| {}'.format(m_id+1, m_w, gathered['mean']['counts_avg'][m_id], ps))

    #print('std', gathered['std'])

ax.legend(handles=lines)
ax.set_ylabel('Absolute Activation')
ax.set_xlabel('% of Sorted View Embeddings')

plt.show()
