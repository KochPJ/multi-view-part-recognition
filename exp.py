import copy
import os

from main import get_args_parser, main
import argparse


if __name__ == '__main__':
    views_lut = {
        '1-a': ['9', '9'],
        '1-b': ['9', '1-6-9'],
        '1-c': ['9', '1-2-3-4-5-6-7-8-9-10'],
        '2': '1-10',
        '3': '1-6-9',
        '3-b': ['1-6-9', '1-2-3-4-5-6-7-8-9-10'],
        '4': '1-3-8-10',
        '5': '1-3-4-7-9',
        '6': '1-3-2-4-7-9',
        '7': '1-2-3-4-7-8-10',
        '8': '1-2-3-4-7-8-9-10',
        '9': '1-3-4-5-6-7-8-9-10',
        '10': '1-2-3-4-5-6-7-8-9-10'
    }

    #views_lut = {
    #    '3': '1-6-9'
    #}

    runs = 10
    main_view = '3'

    parser = argparse.ArgumentParser('MultiView training script', parents=[get_args_parser()])
    args = parser.parse_args()

    add_shuffle_exp = True
    add_fusion_exp = True
    add_roi_crop_exp = True
    add_view_exp = True
    add_depth_exp = True
    shuffle_all_views_exp = True
    add_pretrain_exp = False
    add_aug_exps = False
    add_rotation_exps = False
    add_random_view_order_exp = False
    add_resize_exp = True
    add_lr_exp = True
    add_models_exp = True

    execute = True

    rots = ['0', '0-5', '0-4-8', '0-3-6-9', '0-2-5-8-10', '0-2-4-6-8-10', '0-2-4-5-6-8-10', '0-1-3-5-6-8-9-10',
            '0-1-2-3-5-6-8-9-10', '0-1-2-3-5-6-8-9-10-11', '0-1-2-3-4-5-6-8-9-10-11',
            '0-1-2-3-4-5-6-7-8-9-10-11']
    lrs = [1e-5, 1e-6, [1e-5, 1e-4], [1e-4, 1e-5]]
    models = [['ResNet', '50']]
    outdir = './results'
    exps = []
    new_res = (512, 512)
    res_runs = 10
    runs = 1
    run_start = 0
    for run in range(run_start, runs):
        outdir_ = os.path.join(outdir, 'run{}'.format(str(run).zfill(3)))
        pretrain_path = [[os.path.join(outdir_,
                                       '{}_{}_nr1-a_9/{}_{}_nr1-a_9_best.ckpt'.format(args.model_name,
                                                                                      args.model_version,
                                                                                      args.model_name,
                                                                                      args.model_version)), '1-a'],
                         [os.path.join(outdir_,
                                       '{}_{}_nr1-b_9/{}_{}_nr1-b_9_best.ckpt'.format(args.model_name,
                                                                                      args.model_version,
                                                                                      args.model_name,
                                                                                      args.model_version)), '1-b'],
                         [os.path.join(outdir_,
                                       '{}_{}_nr1-c_9/{}_{}_nr1-c_9_best.ckpt'.format(args.model_name,
                                                                                      args.model_version,
                                                                                      args.model_name,
                                                                                      args.model_version)), '1-c']]

        for nr, views in views_lut.items():
            n = nr.split('-')[0]
            if isinstance(views, list):
                views, data_views = views
            else:
                data_views = views
            #if not add_view_exp and n != main_view:
            #    continue

            a = copy.deepcopy(args)
            if n == '1':
                setattr(a, 'multiview', False)
            else:
                setattr(a, 'multiview', True)
            setattr(a, 'views', views)
            setattr(a, 'data_views', data_views)
            setattr(a, 'name', '{}_{}_nr{}_{}'.format(getattr(a, 'model_name'), getattr(a, 'model_version'),
                                                      nr, views))
            setattr(a, 'outdir', outdir_)

            if nr != main_view:
                if add_view_exp:
                    exps.append(copy.deepcopy(a))
                if shuffle_all_views_exp and n != 1:
                    b = copy.deepcopy(a)
                    setattr(b, 'name', '{}_{}'.format(getattr(b, 'name'), 'shuffle'))
                    setattr(b, 'shuf_views', True)
                    exps.append(b)
            elif nr == main_view:
                if add_resize_exp:
                    for res_r in range(res_runs):
                        b = copy.deepcopy(a)
                        setattr(b, 'name', '{}_{}_{}'.format(getattr(b, 'name'), 'new_res', res_r))
                        setattr(b, 'width', new_res[0])
                        setattr(b, 'height', new_res[0])
                        exps.append(copy.deepcopy(b))

                    b = copy.deepcopy(a)
                    setattr(b, 'name', '{}_{}'.format(getattr(b, 'name'), 'new_res_ms'))
                    setattr(b, 'width', new_res[0])
                    setattr(b, 'height', new_res[0])
                    setattr(b, 'multi_scale_training', True)
                    exps.append(copy.deepcopy(b))

                if add_models_exp:
                    for mn, mv in models:
                        b = copy.deepcopy(a)
                        new_name = getattr(b, 'name')
                        new_name = new_name.split('_')
                        new_name[0] = mn
                        new_name[1] = mv
                        setattr(b, 'name', '_'.join(new_name))
                        setattr(b, 'model_name', mn)
                        setattr(b, 'model_version', mv)
                        exps.append(copy.deepcopy(b))

                if add_lr_exp:
                    for lr in lrs:
                        b = copy.deepcopy(a)
                        if isinstance(lr, list):
                            setattr(b, 'lr_group_wise', True)
                            setattr(b, 'lr_encoder', lr[0])
                            setattr(b, 'lr_fusion', lr[1])
                            setattr(b, 'name', '{}_{}_{}_{}'.format(getattr(b, 'name'), 'lr_gw', lr[0], lr[1]))
                        else:
                            setattr(b, 'lr', lr)
                            setattr(b, 'name', '{}_{}_{}'.format(getattr(b, 'name'), 'lr', lr))
                        exps.append(copy.deepcopy(b))

                if add_pretrain_exp:
                    for path in pretrain_path:
                        b = copy.deepcopy(a)
                        setattr(b, 'name', '{}_{}'.format(getattr(b, 'name'), 'pretrained_{}'.format(path[1])))
                        setattr(b, 'encoder_path', path[0])
                        exps.append(copy.deepcopy(b))

                if add_random_view_order_exp:
                    b = copy.deepcopy(a)
                    setattr(b, 'name', '{}_{}'.format(getattr(b, 'name'), 'random_view_order'))
                    setattr(b, 'random_view_order', True)
                    exps.append(copy.deepcopy(b))

                if add_rotation_exps:
                    for rot in rots:
                        b = copy.deepcopy(a)
                        setattr(b, 'name', '{}_{}'.format(getattr(b, 'name'), 'rot{}'.format(rot)))
                        setattr(b, 'rotations', rot)
                        exps.append(copy.deepcopy(b))

                if add_aug_exps:
                    b = copy.deepcopy(a)
                    setattr(b, 'name', '{}_{}'.format(getattr(b, 'name'), 'no_rot'))
                    setattr(b, 'rotation_aug', False)
                    exps.append(b)

                    b = copy.deepcopy(a)
                    setattr(b, 'name', '{}_{}'.format(getattr(b, 'name'), 'no_flip'))
                    setattr(b, 'flip_aug', False)
                    exps.append(b)

                    b = copy.deepcopy(a)
                    setattr(b, 'name', '{}_{}'.format(getattr(b, 'name'), 'no_flip_no_rot'))
                    setattr(b, 'flip_aug', False)
                    setattr(b, 'rotation_aug', False)
                    exps.append(b)

                if add_fusion_exp:
                    for fusion in ['Squeeze&Excite', 'SharedSqueeze&Excite', 'FC', 'Conv']:
                        b = copy.deepcopy(a)
                        setattr(b, 'fusion', fusion)
                        setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'), 'fuse_{}'.format(fusion)))
                        exps.append(copy.deepcopy(b))

                if add_depth_exp:
                    for fusion in ['Squeeze&Excite', 'Conv']:
                        b = copy.deepcopy(a)
                        setattr(b, 'input_keys', 'x-depth')
                        setattr(b, 'load_keys', 'x-mask-depth')
                        setattr(b, 'depth_fusion', fusion)
                        setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'), 'dfuse_{}'.format(fusion)))
                        exps.append(copy.deepcopy(b))

                if add_roi_crop_exp:
                    b = copy.deepcopy(a)
                    setattr(b, 'roicrop', False)
                    setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'), 'no_roicrop'))
                    exps.append(copy.deepcopy(b))

                if add_shuffle_exp:
                    for keys in [[], ['shuf_views_cw'], ['shuf_views_vw'], ['shuf_views_cw', 'shuf_views_vw']]:
                        b = copy.deepcopy(a)
                        setattr(b, 'shuf_views', True)
                        setattr(b, 'name', '{}_{}'.format(getattr(b, 'name'), 'sv'))
                        for key in keys:
                            setattr(b, key, False)
                            setattr(b, 'name', '{}_{}'.format(getattr(b, 'name'), key))
                        exps.append(copy.deepcopy(b))

    arg_keys = ['name', 'multiview', 'roicrop', 'shuf_views', 'shuf_views_cw', 'shuf_views_vw', 'fusion',
                'depth_fusion',
                'input_keys',
                'load_keys',
                'views',
                'data_views',
                'encoder_path',
                'rotations',
                'rotation_aug',
                'flip_aug',
                'outdir',
                'random_view_order',
                'lr', 'lr_fusion', 'lr_encoder', 'lr_group_wise']

    l = len(exps)
    start_exp = 0
    fails = []
    for i, args in enumerate(exps[start_exp:]):
        print('################################')
        print('---- Running Exp {}/{} ----'.format(i+1 + start_exp, l))
        for key in arg_keys:
            print('     {} | {}'.format(key, getattr(args, key)))
        print('################################')
        try:
            if execute:
                main(args)
        except Exception as e:
            print(e)
            fails.append(args['name'])
            print('{} fails | {}'.format(len(fails), fails))










