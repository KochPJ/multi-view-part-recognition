import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_res(epxs, extr: '',  cut: '', exp_name='', lenght_wise=False, return_mean=False, lut=None):
    if len(epxs) > 0:

        keys = sorted(epxs.keys())
        if lenght_wise:
            keys = sorted(keys, key=len)

        # fig, axs = plt.subplots(1, 1, constrained_layout=True)
        accs, titles, stds, mins, maxs, ls, alls = [], [], [], [], [], [], []
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
        for i, name in enumerate(keys):
            v = epxs[name]
            title = extr+name[len(cut):]
            if lut is not None:
                if title not in lut:
                    continue
                title = lut.get(title)

            titles.append(title)

            selk = '1'
            accs.append(v['topk_test'][selk]['m'])
            maxs.append(v['topk_test'][selk]['max'])
            mins.append(v['topk_test'][selk]['min'])
            stds.append(v['topk_test'][selk]['s'])
            ls.append(v['topk_test'][selk]['l'])
            alls.append(v['topk_test'][selk]['all'])

            r = '{} | '.format(title)
            for k, acc in v['topk_test'][selk].items():
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
        for a, b, c, d, e, l, f in zip(titles, accs, stds, mins, maxs, ls, alls):
            print('{} | {} | mean: {}, std: {}, min: {}, max: {}, all: {}'.format(l, a, b, c, d, e, f))
        fig, axs = plt.subplots(1, 1, constrained_layout=True)
        fig.suptitle(exp_name, fontsize=16)
        axs.bar(titles, maxs)
        plt.xticks(rotation=45)
        plt.ylim(int(min(maxs)*0.9), 100)

        fig, axs = plt.subplots(2, 2, constrained_layout=True)
        fig.suptitle('{} curves'.format(exp_name), fontsize=16)

        for i in range(len(train_loss_m)):
            # plt.fill_between(x, mean - std, mean + std, alpha=alpha)
            axs[0,0].plot(train_loss_m[i])
            axs[0,1].plot(train_top1_m[i])
            #axs[0,2].plot(train_top3_m[i])
            #axs[0,3].plot(train_top5_m[i])
            axs[1,0].plot(valid_loss_m[i])
            axs[1,1].plot(valid_top1_m[i])
            #axs[1,2].plot(valid_top3_m[i])
            #axs[1,3].plot(valid_top5_m[i])
        for i in range(8):
            try:
                j = i//4
                n = i%4
                t = axis_titles[i]
                axs[j, n].set_title(t)
                axs[j, n].set_ylabel('CrossEntropy' if 'Loss' in t else 'Acc[%]')
                axs[j, n].set_xlabel('Epochs')
                axs[j, n].legend(titles)

                #plt.legend(titles)
            except:
                pass
        return np.array(accs), np.array(stds)


if __name__ == '__main__':
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


    plt.style.use('seaborn-v0_8-whitegrid')
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

        if '_nr3_' not in exp and '_fuse_' not in exp and '-b_' not in exp and '3-b_1' not in exp and \
                'Weight' not in exp and '1-c_9' not in exp:
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
        elif viewl != 3:
            allv[str(viewl)+'_'+exp] = v



    #plot_res(depth_epxs, extr='', cut='ResNet_50_nr3_1-6-9_dfuse_', exp_name='Depth Fusion')
    fusion_lut = {
        'Conv_no_TrFo': 'Conv',
        'Conv_Tr': 'T + Conv',
        'FC_no_TrFo': 'MLP',
        'SharedSqueeze&Excite_no_TrFo': 'S.S.&E.',
        'Squeeze&Excite_no_TrFo': 'S.&E.',
        'maxl-pool_no_TrFo': 'MaxPool',
        'mean_no_TrFo': 'Mean',
        'TransfomerEncoderDecoderMultiViewHead_no_TrFo': 'Transformer Decoder',
        'TransformerMultiViewHeadDecoder_no_TrFo': 'Transformer Encoder',
    }

    plot_res(fusion_exps, extr='', cut='ResNet_50_nr3_1-6-9_fuse_', exp_name='MV Fusion', lut=fusion_lut)
    plot_res(augs_exps, extr='', cut='ResNet_50_nr3_1-6-9_', exp_name='Aug')
    plot_res(transformer_exps, extr='', cut='ResNet_50_nr', exp_name='Transformer')
    plot_res(single_exps, extr='', cut='ResNet_50_nr', exp_name='SingeView & Pretrained v3')
    plot_res(v3, extr='', cut='ResNet_50_nr3_1-6-9_', exp_name='3V data views')
    plot_res(allv, extr='', cut='ResNet_50_nr', exp_name='All Data Views')
    plot_res(weight_exps, extr= '',  cut= 'ResNet_50_nr3_1-6-9_', exp_name='Weight')

    acc_v, std_v = plot_res(view_exps, extr='', cut='ResNet_50_nr', exp_name='View', lenght_wise=True)
    acc_r, std_r = plot_res(rot_exps, extr= '',  cut='ResNet_50_nr', exp_name='Rotation')
    plot_res(weightnet_exps, extr= '',  cut='', exp_name='WeightNet')
    acc_d, std_d = plot_res(dataview_exps, extr= '',  cut='ResNet_50_nr3_1-6-9_datatviews_', exp_name='DataViews')

    alpha=0.5

    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    fig.suptitle('Number of res', fontsize=16)

    xv = [i+1 for i in range(len(acc_v))]
    plt.plot(xv, acc_v)

    xd = [i+1 for i in range(len(acc_d))]
    plt.plot(xd, acc_d)
    xr = [i+1 for i in range(len(acc_r))]
    plt.plot(xr, acc_r)

    plt.legend(['Inference Views', 'Training Views', 'Object Rotations'])
    plt.fill_between(xv, acc_v - std_v, acc_v + std_v, alpha=alpha)
    plt.fill_between(xd, acc_d - std_d, acc_d + std_d, alpha=alpha)
    plt.fill_between(xr, acc_r - std_r, acc_r + std_r, alpha=alpha)
    plt.ylabel('ACC  [%]')
    plt.xlabel('# number of')


    print('-----not 3views---')
    for key in allv.keys():
        print(key)




    plt.show()
