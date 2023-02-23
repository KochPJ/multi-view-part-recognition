import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_res(epxs, extr: '',  cut: '', exp_name=''):
    if len(epxs) > 0:
        # fig, axs = plt.subplots(1, 1, constrained_layout=True)
        accs, titles, stds, mins, maxs = [], [], [], [], []
        train_loss_m = []
        train_loss_s = []
        valid_loss_m = []
        valid_loss_s = []
        valid_top1_m = []
        valid_top1_s = []
        valid_top3_m = []
        valid_top3_s = []
        valid_top5_m = []
        valid_top5_s = []
        train_top1_m = []
        train_top1_s = []
        train_top3_m = []
        train_top3_s = []
        train_top5_m = []
        train_top5_s = []
        for i, name in enumerate(sorted(epxs.keys())):
            v = epxs[name]
            accs.append(v['topk_test']['1']['m'])
            maxs.append(v['topk_test']['1']['max'])
            mins.append(v['topk_test']['1']['min'])
            stds.append(v['topk_test']['1']['s'])

            title = extr+name[len(cut):]
            titles.append(title)
            r = '{} | '.format(title)
            for k, acc in v['topk_test']['1'].items():
                r += '{}: {},'.format(k, acc)
            #print(r)
            train_loss_m.append(v['losses_train']['m'])
            train_loss_s.append(v['losses_train']['s'])
            valid_loss_m.append(v['losses_valid']['m'])
            valid_loss_s.append(v['losses_valid']['s'])
            train_top1_m.append(v['topk_train']['1']['m'])
            train_top1_s.append(v['topk_train']['1']['s'])
            train_top3_m.append(v['topk_train']['3']['m'])
            train_top3_s.append(v['topk_train']['3']['s'])
            train_top5_m.append(v['topk_train']['5']['m'])
            train_top5_s.append(v['topk_train']['5']['s'])
            valid_top1_m.append(v['topk_valid']['1']['m'])
            valid_top1_s.append(v['topk_valid']['1']['s'])
            valid_top3_m.append(v['topk_valid']['3']['m'])
            valid_top3_s.append(v['topk_valid']['3']['s'])
            valid_top5_m.append(v['topk_valid']['5']['m'])
            valid_top5_s.append(v['topk_valid']['5']['s'])

        print('###### {} results ####'.format('{} Exp results'.format(exp_name)))
        for a, b, c, d, e in zip(titles, accs, stds, mins, maxs):
            print('{} | mean: {}, std: {}, min: {}, max: {}'.format(a, b, c, d, e))
        fig, axs = plt.subplots(1, 1, constrained_layout=True)
        fig.suptitle(exp_name, fontsize=16)
        axs.bar(titles, maxs)
        plt.xticks(rotation=45)
        plt.ylim(int(min(maxs)*0.9), 100)

        fig, axs = plt.subplots(2, 4, constrained_layout=True)
        fig.suptitle('{} curves'.format(exp_name), fontsize=16)

        for i in range(len(train_loss_m)):
            # plt.fill_between(x, mean - std, mean + std, alpha=alpha)
            axs[0,0].plot(train_loss_m[i])
            axs[0,1].plot(train_top1_m[i])
            axs[0,2].plot(train_top3_m[i])
            axs[0,3].plot(train_top5_m[i])
            axs[1,0].plot(valid_loss_m[i])
            axs[1,1].plot(valid_top1_m[i])
            axs[1,2].plot(valid_top3_m[i])
            axs[1,3].plot(valid_top5_m[i])
        for i in range(8):
            j = i//4
            n = i%4
            t = axis_titles[i]
            axs[j, n].set_title(t)
            axs[j, n].set_ylabel('CrossEntropy' if 'Loss' in t else 'Acc[%]')
            axs[j, n].set_xlabel('Epochs')
        plt.legend(titles)


if __name__ == '__main__':
    root = './results'
    ks = ['1', '3', '5']
    args = ['tf_layers', 'multi_head_classification', 'fusion', 'data_views']
    res = {}
    min_run = 7
    runs = sorted(list(os.listdir(root)))
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
                print('exp: {} not finished'.format(exp))
                continue

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
                                        'min': np.min(np.array(res[exp][key][k]), axis=0)}
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
                                     'min': np.min(np.array(res[exp][key]), axis=0)}
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


    depth_epxs = {}
    fusion_exps = {}
    single_exps = {}
    augs_exps = {}
    transformer_exps = {}
    v3 = {}
    allv = {}
    weight_exps = {}
    rot_exps = {}
    view_exps = {}
    weightnet_exps = {}
    dataview_exps = {}


    axis_titles = ['Train Loss', 'Train Top 1', 'Train Top 3', 'Train Top 5',
                   'Valid Loss', 'Valid Top 1', 'Valid Top 3', 'Valid Top 5']

    for exp, v in res.items():
        if '_rota_exp' in exp:
            rot_exps[exp] = v
        if '_dataviews_exp' in exp:
            dataview_exps[exp] = v

        if 'WeightNet' in exp:
            weightnet_exps[exp] = v

        if '_nr3_' not in exp:
            view_exps[exp] = v

        if '_dfuse_' in exp:
            depth_epxs[exp] = v

        if '_fuse_' in exp:
            fusion_exps[exp] = v

        if '1-a' in exp or '1-b' in exp or '1-c' in exp or 'normal' in exp:
            single_exps[exp] = v

        if 'aug' in exp:
            augs_exps[exp] = v

        if 'EDTF' in exp or 'DTF' in exp:
            transformer_exps[exp] = v

        if 'Weight' in exp:
            weight_exps[exp] = v

        viewl = len(v['args']['data_views'].split('-'))
        if viewl == 3:
            v3[exp] = v
        elif viewl == 10:
            allv[exp] = v



    plot_res(depth_epxs, extr='', cut='ResNet_50_nr3_1-6-9_dfuse_', exp_name='Depth Fusion')
    plot_res(fusion_exps, extr='', cut='ResNet_50_nr3_1-6-9_fuse_', exp_name='MV Fusion')
    plot_res(augs_exps, extr='', cut='ResNet_50_nr3_1-6-9_', exp_name='Aug')
    plot_res(transformer_exps, extr='', cut='ResNet_50_nr', exp_name='Transformer')
    plot_res(single_exps, extr='', cut='ResNet_50_nr', exp_name='SingeView & Pretrained v3')
    #plot_res(v3, extr='', cut='ResNet_50_nr3_1-6-9_', exp_name='3V data views')
    #plot_res(allv, extr='', cut='ResNet_50_nr', exp_name='All Data Views')
    plot_res(weight_exps, extr= '',  cut= 'ResNet_50_nr3_1-6-9_', exp_name='Weight')

    plot_res(view_exps, extr='', cut='ResNet_50_nr', exp_name='View')
    print(len(rot_exps))
    plot_res(rot_exps, extr= '',  cut='ResNet_50_nr', exp_name='Rotation')
    plot_res(weightnet_exps, extr= '',  cut='', exp_name='WeightNet')
    plot_res(dataview_exps, extr= '',  cut='ResNet_50_nr3_1-6-9_datatviews_', exp_name='DataViews')




    plt.show()
