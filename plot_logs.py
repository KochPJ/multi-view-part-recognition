import argparse
import matplotlib.pyplot as plt
import json
import os
from utils.stuff import lookup

def get_args_parser():
    parser = argparse.ArgumentParser('Plot multi view results', add_help=False)

    parser.add_argument('--outdir', default='./results', type=str)
    # training
    #parser.add_argument('--name', default='ResNet_34_nr3_1-6-9_fuse_Squeeze&Excite', type=str)
    #parser.add_argument('--name', default='ResNet_34_nr3_1-6-9_fuse_Conv', type=str)
    #parser.add_argument('--name', default='ResNet_34_nr3_1-6-9_fuse_FC', type=str)
    parser.add_argument('--name', default='ResNet_34_nr3_1-6-9_pretrained_1-a', type=str)

    #parser.add_argument('--name', default='ResNet_34_nr3_1-6-9_fuse_SharedSqueeze&Excite', type=str)


    return parser

def main(args):
    with open(os.path.join(args.outdir, args.name, '{}_logs.log'.format(args.name))) as f:
        logs = json.load(f)


    print(logs.keys())

    print('best_loss_train: {}'.format(logs['best_loss_train']))
    print('best_loss_valid: {}'.format(logs['best_loss_valid']))
    print('best_acc_valid: {}'.format(logs['best_acc_valid']))
    print('best_acc_train: {}'.format(logs['best_acc_train']))
    print('acc_test: {}'.format(logs['acc_test']))
    print('loss_test: {}'.format(logs['loss_test']))
    print('epoch: {}'.format(logs['epoch']))
    #for k, acc in logs['topk_valid'].items():
    #    print('top {} valid: {}'.format(k, max(acc)))
    for k, acc in logs['topk_test'].items():
        print('top {} test: {}'.format(k, acc))

    keys = [['Train Loss', logs['losses_train'], 'loss', 'b'], ['Valid Loss', logs['losses_valid'], 'loss', 'b']]
    keys += [['Train Top {}'.format(k), l, 'Acc[%]', 'r'] for k, l in logs['topk_train'].items()]
    keys += [['Valid Top {} '.format(k), l, 'Acc[%]', 'r'] for k, l in logs['topk_valid'].items()]

    h, w = lookup(len(keys))
    j = 0
    i = 0
    n = 1
    fig, axs = plt.subplots(h, w, constrained_layout=True)
    fig.suptitle('Results: {}'.format(args.name), fontsize=16)
    for key in keys:
        if h > 1:
            axs[i, j].plot(key[1], c=key[3])
            axs[i, j].set_title(key[0])
            axs[i, j].set_ylabel(key[2])
            axs[i, j].set_xlabel('Epochs')
        else:
            axs[j].plot(key[1], c=key[3])
            axs[j].set_title(key[0])

            axs[j].set_ylabel(key[2])
            axs[j].set_xlabel('Epochs')

        if n == w:
            i += 1
            j = 0
            n = 1
        else:
            j += 1
            n += 1

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Multi View plot script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
