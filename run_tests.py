from test import main, get_args_parser
import argparse
import copy
import os
import numpy as np
import json


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


parser = argparse.ArgumentParser('MultiView testing script', parents=[get_args_parser()])
args = parser.parse_args()



test_names = [
    'ResNet_50_nr3_1-6-9_dfuse_TFED_RGBDMVF_v1',
    'ResNet_50_nr3_1-6-9_dfuse_TFED_RGBDMVF_v2',
    'ResNet_50_nr3_1-6-9_dfuse_TFED_RGBDMVF_v3',
    'ResNet_50_nr3_1-6-9_dfuse_TFED_RGBD_v1',
    'ResNet_50_nr3_1-6-9_dfuse_TFED_RGBDM_v2',
    'ResNet_50_nr3_1-6-9_dfuse_TFED_RGBD_v3'


    #'ResNet_50_nr3_1-6-9_long_pretrained_1-c', 'ResNet_50_nr3_1-6-9_long_pretrained_1-a'
    #['ResNet_50_nr1-a_9_long'], ['ResNet_50_nr1-b_9_long'], ['ResNet_50_nr1-c_9_long']
    #'ResNet_50_nr3_1-6-9_fuse_max-pool', 'ResNet_50_nr3_1-6-9_fuse_max-pool-multihead',
    #'ResNet_50_nr3_1-6-9_fuse_Conv', 'ResNet_50_nr3_1-6-9_fuse_Conv-multihead',
    #'ResNet_50_nr3_1-6-9_fuse_Conv_Tr', 'ResNet_50_nr3_1-6-9_fuse_Conv_Tr-multihead'
    #'ResNet_50_nr3_1-6-9_dfuse_TFED_RGBD_v1', 'ResNet_50_nr3_1-6-9_dfuse_TFED_RGBDMVF_v1',
    #'ResNet_50_nr3_1-6-9_dfuse_TFED_RGBDMVF_v2',
    #'ResNet_50_nr3_1-6-9_dfuse_TFED_RGBD_v3', 'ResNet_50_nr3_1-6-9_dfuse_TFED_RGBDMVF_v3'
]


exps = []
for test_dir in test_names:
    if isinstance(test_dir, list):
        test_dir = test_dir[0]
        encoder = True
    else:
        encoder = False


    for run in res[test_dir]['runs']:
        ignore_sets = []
        new_args = copy.deepcopy(args)
        if encoder:
            ignore_sets = ['views', 'encoder_path', 'data_views', 'multiview']
            new_args.name = test_dir
            #new_args.encoder_path ='./results/{}/{}/{}_best.ckpt'.format(run, test_dir,test_dir)
            new_args.fusion = 'max-pool'
            new_args.data_views = '1-6-9'
            new_args.multiview = True
            new_args.toogle_newview_model = True

            #new_args.hidden_channels = 2048
            ignore_sets.append('fusion')

            #ignore_sets.append('hidden_channels')
        else:
            new_args.name = test_dir
        new_args.outdir = './results/{}'.format(run)
        exps.append([new_args, ignore_sets])

arg_keys = ['name', 'outdir']
start_exp = 0
l = len(exps)
execute = True
fails = []
for i, (args, ignore_sets) in enumerate(exps[start_exp:]):
    print('################################')
    print('---- Running Exp {}/{} ----'.format(i + 1 + start_exp, l))
    for key in arg_keys:
        print('     {} | {}'.format(key, getattr(args, key)))
    print('################################')
    try:
        if execute:
            main(args, ignore_sets)
    except Exception as e:
        print(e)
        fails.append(args['name'])
        print('{} fails | {}'.format(len(fails), fails))

print('{} fails | {}'.format(len(fails), fails))
