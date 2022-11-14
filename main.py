import torch
import numpy as np
import json
import os
import argparse
from utils.dataset import get_dataset, Dataset
import torch.nn as nn
from models.multiview import get_model
import torch.optim.lr_scheduler as schedulers
from utils.metric import TopKAccuracy
from utils.stuff import bar_progress, lookup
import time
import random


def get_args_parser():
    parser = argparse.ArgumentParser('Set MultiView', add_help=False)

    # training
    parser.add_argument('--name', default='ResNet_34_nr3_1_6_9-res512-ms', type=str)
    parser.add_argument('--outdir', default='./results', type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--multi_gpu', default=True, type=bool)

    # optimizer
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--weight_decay', default=0.0, type=float)

    # loss
    parser.add_argument('--lr_group_wise', default=False, type=bool)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_encoder', default=1e-4, type=float)
    parser.add_argument('--lr_fusion', default=1e-4, type=float)

    # metrics
    parser.add_argument('--topk', default='1-3-5', type=str)

    # dataset
    parser.add_argument('--path', default='./industrial_part_recognition', type=str)
    parser.add_argument('--roicrop', default=True, type=bool)
    parser.add_argument('--shuf_views', default=False, type=bool)
    parser.add_argument('--shuf_views_cw', default=True, type=bool)
    parser.add_argument('--shuf_views_vw', default=True, type=bool)
    parser.add_argument('--p_shuf_cw', default=0.3, type=float)
    parser.add_argument('--p_shuf_vw', default=0.3, type=float)
    parser.add_argument('--data_views', default='1-6-9', type=str) #1-6-9
    parser.add_argument('--views', default='1-6-9', type=str) #1-6-9
    parser.add_argument('--random_view_order', default=False, type=bool)
    parser.add_argument('--rotations', default='0-1-2-3-4-5-6-7-8-9-10-11', type=str)
    parser.add_argument('--input_keys', default='x', type=str)
    parser.add_argument('--load_keys', default='x-mask', type=str)
    parser.add_argument('--num_classes', default=-1, type=int)
    parser.add_argument('--visualize_samples', default=False, type=bool)
    parser.add_argument('--show_axis', default=True, type=bool)
    parser.add_argument('--depth_mean', default=897.8720890284345, type=float)
    parser.add_argument('--depth_std', default=859.9051176853665, type=float)
    parser.add_argument('--rotation_aug', default=True, type=bool)
    parser.add_argument('--flip_aug', default=True, type=bool)
    parser.add_argument('--width', default=512, type=int)
    parser.add_argument('--height', default=512, type=int)
    parser.add_argument('--multi_scale_training', default=False, type=bool)
    parser.add_argument('--training_scale_low', default=0.1, type=float)
    parser.add_argument('--training_scale_high', default=0.1, type=float)

    # model
    parser.add_argument('--multiview', default=True, type=bool)
    parser.add_argument('--hidden_channels', default=512, type=int)
    parser.add_argument('--model_name', default='ResNet', type=str)
    parser.add_argument('--model_version', default='34', type=str)
    parser.add_argument('--fusion', default='Conv', type=str) #['Squeeze&Excite', 'SharedSqueeze&Excite', 'FC', 'Conv']
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--encoder_path', default='', type=str)
    parser.add_argument('--depth_fusion', default='Squeeze&Excite', type=str) #['Squeeze&Excite',  'Conv']

    # scheduler
    parser.add_argument('--one_cycle', default=True, type=bool)
    parser.add_argument('--pct_start', default=0.5, type=float)
    parser.add_argument('--div_factor', default=10.0, type=float)
    parser.add_argument('--final_div_factor', default=100.0, type=float)

    # misc
    parser.add_argument('--time_max', default=1000, type=int)
    return parser


def main(args):
    args.outdir = os.path.join(args.outdir, args.name)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    current_dir = os.path.join(args.outdir, '{}_current.ckpt'.format(args.name))
    best_dir = os.path.join(args.outdir, '{}_best.ckpt'.format(args.name))
    logs_dir = os.path.join(args.outdir, '{}_logs.log'.format(args.name))
    if os.path.exists(current_dir):
        cp = torch.load(current_dir)
    else:
        cp = None

    ds = get_dataset(args)

    if args.shuf_views and not args.shuf_views_cw and args.p_shuf_cw == -1:
        args.p_shuf_cw = float(1.0/len(args.views.split('-')))
        print('Shuffle class wise with p={}'.format(args.p_shuf_cw))

    if args.shuf_views and not args.shuf_views_vw and args.p_shuf_vw == -1:
        args.p_shuf_vw = float(1 / len(args.views.split('-')))
        print('Shuffle view wise with p={}'.format(args.p_shuf_vw))

    modes = ['train', 'valid', 'test']
    datasets = {mode: Dataset(args, ds[mode], ds['meta'], mode) for mode in modes}
    args.num_classes = datasets['train'].num_classes
    print('number of classes: {}'.format(args.num_classes))

    if args.visualize_samples:
        import matplotlib.pyplot as plt
        while True:
            for mode in modes:
                index = random.randint(0, len(datasets[mode]))
                x, y = datasets[mode].__getitem__(index)
                n = sum([len(v) for v in x.values()])
                h, w = lookup(n)
                keys = list(x.keys())
                K = n//len(keys)
                print(n, K, keys, len(x['x']))
                j = 0
                i = 0
                n = 1
                fig, axs = plt.subplots(h, w, constrained_layout=True)
                fig.suptitle('label: {}'.format(y), fontsize=16)
                for k in range(K):
                    for key in keys:
                        if h > 1:
                            if key == 'depth':
                                axs[i, j].imshow(np.array(x[key][k]))
                            else:
                                axs[i, j].imshow(x[key][k])
                            axs[i, j].set_title('v{} | {}'.format(k+1, key))
                        else:
                            if key == 'depth':
                                axs[i].imshow(np.array(x[key][k]))
                            else:
                                axs[i].imshow(x[key][k])
                            axs[j].set_title('v{} | {}'.format(k+1, key))

                        if not args.show_axis:
                            if h > 1:
                                axs[i, j].get_xaxis().set_visible(False)
                                axs[j, i].get_yaxis().set_visible(False)
                            else:
                                axs[j].get_xaxis().set_visible(False)
                                axs[i].get_yaxis().set_visible(False)

                        if n == w:
                            i += 1
                            j = 0
                            n = 1
                        else:
                            j += 1
                            n += 1

                plt.show()

    dataloader = {mode: torch.utils.data.DataLoader(dataset=datasets[mode],
                                                    batch_size=args.batch_size,
                                                    shuffle=True if mode == 'train' else False,
                                                    num_workers=args.num_workers,
                                                    drop_last=False) for mode in modes}

    print('##############')
    steps = {'epochs': args.epochs}
    for mode in modes:
        print('------ {} ------'.format(mode))
        print('     classes: {}'.format(len(ds[mode])))
        print('     samples: {}'.format(sum([len(ds[mode][cls]) for cls in ds[mode]])))
        print('     dataset len: {}'.format(len(datasets[mode])))
        print('     dataloader len: {}'.format(len(dataloader[mode])))
        steps[mode] = len(dataloader[mode])

        #if mode in ['valid', 'test']:
        #    for cls in ds[mode]:
        #        if len(ds[mode][cls]) != 5:
        #            print(mode, cls, len(ds[mode][cls]))

    if not args.shuf_views_cw and args.shuf_views:
        print('Using BCEWithLogitsLoss')
        criterion = nn.BCEWithLogitsLoss()
    else:
        print('Using CrossEntropyLoss')
        criterion = nn.CrossEntropyLoss()

    # set device, build model, push model to device and load a checkpoint if given
    device = torch.device(args.device)
    model = get_model(args)
    if cp is not None:
        print('loading model state dict')
        model.load_state_dict(cp['state_dict'])

    if args.multi_gpu and torch.cuda.device_count() > 1 and args.device != 'cpu':
        model = nn.DataParallel(model)
    model = model.to(device)

    if args.lr_group_wise:
        param_dicts = [
            {
                "params":
                    [p for n, p in model.named_parameters()
                     if 'encoder' in n and p.requires_grad],
                "lr": args.lr_encoder,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if 'encoder' not in n and p.requires_grad],
                "lr": args.lr_fusion,
            }
        ]
        max_lr = [args.lr_encoder, args.lr_fusion]

        print('##############')
        print('Distribution of {} Paramgroups:'.format(
            sum([len(param_dicts[i]['params']) for i in range(len(param_dicts))])))
        print('     Encoder: {} with lr: {}'.format(len(param_dicts[0]['params']), args.lr_encoder))
        print('     Fusion: {} with lr: {}'.format(len(param_dicts[1]['params']), args.lr_fusion))
    else:
        param_dicts = [
            {
                "params":
                    [p for n, p in model.named_parameters()],
                "lr": args.lr,
            }]
        max_lr = args.lr
        print('Distribution of {} Paramgroups:'.format(
            sum([len(param_dicts[i]['params']) for i in range(len(param_dicts))])))
        print('     Model: {} with lr: {}'.format(len(param_dicts[0]['params']), args.lr))

    # init the optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=param_dicts, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params=param_dicts, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_dicts, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError('the optimizer {} is not implemented'.format(args.optimizer))

    scheduler = schedulers.OneCycleLR(optimizer=optimizer,
                                      max_lr=max_lr,
                                      total_steps=args.epochs * len(dataloader['train']),
                                      pct_start=args.pct_start, div_factor=args.div_factor,
                                      final_div_factor=args.final_div_factor)
    metrics = {k: TopKAccuracy(int(k)) for k in args.topk.split('-')}

    if cp is not None:
        optimizer.load_state_dict(cp['optimizer'])
        scheduler.load_state_dict(cp['scheduler'])
        logs = cp['logs']
        args.start_epoch = logs['epoch']
    else:
        logs = {
            'best_loss_train': (np.Inf, None),
            'best_loss_valid': (np.Inf, None),
            'best_acc_valid': (0, None),
            'best_acc_train': (0, None),
            'acc_test': None,
            'loss_test': None,
            'epoch': 0,
            'losses_train': [None],
            'losses_valid': [None],
            'topk_train': {k: [None] for k in args.topk.split('-')},
            'topk_valid': {k: [None] for k in args.topk.split('-')},
            'topk_test': {k: None for k in args.topk.split('-')},
            'args': vars(args)
        }

    bestk = sorted(list(args.topk.split('-')))[0]
    del cp

    times = {
        'epoch': [],
        'train': [],
        'valid': [],
        'test': []

    }
    first_epoch = True
    print('##### Start Training #####')
    t_epoch = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if first_epoch and args.multi_scale_training:
            dataloader['train'].dataset.checking_batch_size = True
            first_epoch = False
        else:
            dataloader['train'].dataset.checking_batch_size = False

        logs['epoch'] = epoch
        losses = []

        t_step = time.time()
        for step, (x, y) in enumerate(dataloader['train']):
            for key in x:
                x[key] = x[key].to(device)
            y = y.to(device)
            pred = model(**x)

            optimizer.zero_grad()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))
            for k, m in metrics.items():
                m.add(pred, y)
            scheduler.step()
            t = time.time()
            times['train'].append(t - t_step)
            t_step = t
            if len(times['train']) > args.time_max:
                times['train'] = times['train'][1:]

            bar_progress('train', epoch+1, step+1, steps, scheduler.get_last_lr(), float(loss.item()),
                         metrics[bestk].result(),
                         logs['losses_train'][-1],
                         logs['topk_train'][bestk][-1],
                         times, bestk)
        print('\n')

        for k, m in metrics.items():
            acc = m.result()
            if epoch == 0:
                logs['topk_train'][k][0] = acc
            else:
                logs['topk_train'][k].append(acc)
            if k == bestk and acc > logs['best_acc_train'][0]:
                logs['best_acc_train'] = (acc, epoch)

            m.reset()

        if epoch == 0:
            logs['losses_train'][0] = float(np.mean(losses))
        else:
            logs['losses_train'].append(float(np.mean(losses)))

        if logs['losses_train'][-1] < logs['best_loss_train'][0]:
            logs['best_loss_train'] = (logs['losses_train'][-1], epoch)

        losses = []
        for step, (x, y) in enumerate(dataloader['valid']):
            for key in x:
                x[key] = x[key].to(device)
            y = y.to(device)

            with torch.no_grad():
                pred = model(**x)

            loss = criterion(pred, y)
            losses.append(float(loss.item()))
            for k, m in metrics.items():
                m.add(pred, y)

            t = time.time()
            times['valid'].append(t - t_step)
            t_step = t
            if len(times['valid']) > args.time_max:
                times['valid'] = times['valid'][1:]

            bar_progress('valid', epoch+1, step+1, steps, None, float(loss.item()), metrics[bestk].result(),
                         logs['losses_valid'][-1],
                         logs['topk_valid'][bestk][-1],
                         times, bestk)

        print('\n')

        for k, m in metrics.items():
            acc = m.result()
            if epoch == 0:
                logs['topk_valid'][k][0] = acc
            else:
                logs['topk_valid'][k].append(acc)

            if k == bestk and acc > logs['best_acc_valid'][0]:
                logs['best_acc_valid'] = (acc, epoch)
            m.reset()

        if epoch == 0:
            logs['losses_valid'][0] = float(np.mean(losses))
        else:
            logs['losses_valid'].append(float(np.mean(losses)))

        # make the current checkpoint
        ckpt = {
            'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'logs': logs}
        torch.save(ckpt, current_dir)

        if logs['losses_valid'][-1] < logs['best_loss_valid'][0]:
            logs['best_loss_valid'] = (logs['losses_valid'][-1], epoch)
            torch.save(ckpt, best_dir)

        # save the current checkpoint and also if it is the best at the valid AUC or loss
        with open(logs_dir, 'w') as f:
            json.dump(logs, f)

        # remove the ckpt dict out of the RAM during the next epoch
        del ckpt

        dataloader['train'].dataset.shuffle_data()

        t = time.time()
        times['epoch'].append(t-t_epoch)
        t_epoch = t

    print('##### Run Test #####')
    losses = []
    t_step = time.time()
    for step, (x, y) in enumerate(dataloader['test']):
        for key in x:
            x[key] = x[key].to(device)
        y = y.to(device)

        with torch.no_grad():
            pred = model(**x)

        loss = criterion(pred, y)
        losses.append(float(loss.item()))
        for k, m in metrics.items():
            m.add(pred, y)

        t = time.time()
        times['test'].append(t - t_step)
        t_step = t
        if len(times['test']) > args.time_max:
            times['test'] = times['test'][1:]

        bar_progress('test', args.epochs, step + 1, steps, None, float(loss.item()),
                     metrics[bestk].result(),
                     None,
                     None,
                     times, bestk)
    print('\n')

    print('##### Test Results ######')
    for k, m in metrics.items():
        logs['topk_test'][k] = m.result()
        print('Test top {}: {}%'.format(k, float(np.round(logs['topk_test'][k], 3))))
    logs['acc_test'] = metrics[bestk].result()
    logs['loss_test'] = float(np.mean(losses))
    print('Test loss: {}'.format(logs['loss_test']))

    # save the current checkpoint and also if it is the best at the valid AUC or loss
    with open(logs_dir, 'w') as f:
        json.dump(logs, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MultiView training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

