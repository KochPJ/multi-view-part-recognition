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
        '3': ['1-6-9', '1-2-3-4-5-6-7-8-9-10'],#, '1-6-9',
        #'3-b': '1-6-9',
        #'3': '1-6-9',
        '4': ['1-3-8-10', '1-2-3-4-5-6-7-8-9-10'],
        '4-b': '1-3-8-10',
        '5': ['1-3-4-7-9', '1-2-3-4-5-6-7-8-9-10'],
        '5-b': '1-3-4-7-9',
        '6': ['1-3-2-4-7-9', '1-2-3-4-5-6-7-8-9-10'],
        '6-b': '1-3-2-4-7-9',
        '7': ['1-2-3-4-7-8-10', '1-2-3-4-5-6-7-8-9-10'],
        '7-b': '1-2-3-4-7-8-10',
        '8': ['1-2-3-4-7-8-9-10', '1-2-3-4-5-6-7-8-9-10'],
        '8-b': '1-2-3-4-7-8-9-10',
        '9': ['1-3-4-5-6-7-8-9-10', '1-2-3-4-5-6-7-8-9-10'],
        '9-b': '1-3-4-5-6-7-8-9-10',
        '10': '1-2-3-4-5-6-7-8-9-10'
    }
    views_lut = {
        '1-a': ['9', '9'],
        '1-b': ['9', '1-6-9'],
        '1-c': ['9', '1-2-3-4-5-6-7-8-9-10'],
        '3': '1-6-9',
    }
    #views_lut = {
    #    '3': '1-6-9'
    #}

    runs = 5
    main_view = '3'

    parser = argparse.ArgumentParser('MultiView training script', parents=[get_args_parser()])
    args = parser.parse_args()

    add_shuffle_exp = True#True
    add_fusion_exp = True#True
    add_encoder_decoder_tf = True#True
    add_PE_exp = True#True
    add_roi_crop_exp = True#True
    add_view_exp = True#True
    add_depth_exp = False
    shuffle_all_views_exp = False
    add_pretrain_exp = True#True
    add_aug_exps = True#True
    add_rotation_exps = False#True
    add_random_view_order_exp = True#True
    add_resize_exp = True#True
    add_lr_exp = False
    add_models_exp = False
    execute = True

    rots = ['0', '0-5', '0-4-8', '0-3-6-9', '0-2-5-8-10', '0-2-4-6-8-10', '0-2-4-5-6-8-10', '0-1-3-5-6-8-9-10',
            '0-1-2-3-5-6-8-9-10', '0-1-2-3-5-6-8-9-10-11', '0-1-2-3-4-5-6-8-9-10-11',
            '0-1-2-3-4-5-6-7-8-9-10-11']
    lrs = [1e-5, 1e-6, [1e-5, 1e-4], [1e-4, 1e-5]]
    models = [['ResNet', '50']]
    outdir = './results'
    exps = []
    new_res = (512, 512)

    depth_epoch_multiplier = 1.5
    tf_layers = 1
    runs = 5
    run_start = 4 #3
    start_exp = 6
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
                    b = copy.deepcopy(a)
                    setattr(b, 'name', '{}_{}'.format(getattr(b, 'name'), 'aug_ms'))
                    setattr(b, 'multi_scale_training', True)
                    exps.append(copy.deepcopy(b))

                    b = copy.deepcopy(a)
                    setattr(b, 'name', '{}_{}'.format(getattr(b, 'name'), 'aug_new_res'))
                    setattr(b, 'width', new_res[0])
                    setattr(b, 'height', new_res[0])
                    setattr(b, 'multi_scale_training', False)
                    #setattr(b, 'batch_size', 14)
                    exps.append(copy.deepcopy(b))

                    b = copy.deepcopy(a)
                    setattr(b, 'name', '{}_{}'.format(getattr(b, 'name'), 'aug_new_res_ms'))
                    setattr(b, 'width', new_res[0])
                    setattr(b, 'height', new_res[0])
                    setattr(b, 'multi_scale_training', True)
                    #setattr(b, 'batch_size', 14)
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
                        setattr(b, 'multi_head_classification', False)
                        exps.append(copy.deepcopy(b))

                if add_random_view_order_exp:
                    b = copy.deepcopy(a)
                    setattr(b, 'name', '{}_{}'.format(getattr(b, 'name'), 'aug_random_view_order'))
                    setattr(b, 'random_view_order', True)
                    exps.append(copy.deepcopy(b))

                if add_rotation_exps:
                    for rot in rots:
                        b = copy.deepcopy(a)
                        setattr(b, 'name', '{}_{}'.format(getattr(b, 'name'), 'rotation_exp{}'.format(rot)))
                        setattr(b, 'rotations', rot)
                        exps.append(copy.deepcopy(b))

                if add_aug_exps:
                    b = copy.deepcopy(a)
                    setattr(b, 'name', '{}_{}'.format(getattr(b, 'name'), 'aug_no_rot'))
                    setattr(b, 'rotation_aug', False)
                    exps.append(b)

                    b = copy.deepcopy(a)
                    setattr(b, 'name', '{}_{}'.format(getattr(b, 'name'), 'aug_no_flip'))
                    setattr(b, 'flip_aug', False)
                    exps.append(b)

                    b = copy.deepcopy(a)
                    setattr(b, 'name', '{}_{}'.format(getattr(b, 'name'), 'aug_no_flip_no_rot'))
                    setattr(b, 'flip_aug', False)
                    setattr(b, 'rotation_aug', False)
                    exps.append(b)

                if add_fusion_exp:
                    for fusion in ['Conv']: #, 'Conv','FC', 'max-pool', 'mean', 'Squeeze&Excite', 'SharedSqueeze&Excite'
                        b = copy.deepcopy(a)
                        setattr(b, 'data_views', '1-2-3-4-5-6-7-8-9-10')
                        setattr(b, 'fusion', fusion)
                        #setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'), 'fuse_{}'.format(fusion)))
                        #setattr(b, 'multi_head_classification', False)
                        #setattr(b, 'tf_layers', tf_layers)
                        #exps.append(copy.deepcopy(b))
                        #setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'), 'fuse_{}-multiead'.format(fusion)))
                        #setattr(b, 'multi_head_classification', True)
                        #setattr(b, 'tf_layers', tf_layers)
                        #exps.append(copy.deepcopy(b))
                        #setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'), 'fuse_{}-2multiead'.format(fusion)))
                        #setattr(b, 'multi_head_classification', True)
                        #setattr(b, 'tf_layers', tf_layers*2)
                        #exps.append(copy.deepcopy(b))
                        #setattr(b, 'tf_layers', 0)
                        #setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'), 'fuse_{}_no_TrFo'.format(fusion)))
                        #setattr(b, 'multi_head_classification', False)
                        #exps.append(copy.deepcopy(b))
                        setattr(b, 'tf_layers', 0)
                        setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'), 'fuse_{}_no_TrFo-multihead'.format(fusion)))
                        setattr(b, 'multi_head_classification', True)
                        exps.append(copy.deepcopy(b))

                if add_encoder_decoder_tf:
                    b = copy.deepcopy(a)
                    setattr(b, 'data_views', '1-2-3-4-5-6-7-8-9-10')
                    setattr(b, 'fusion', 'TransfomerEncoderDecoderMultiViewHead')
                    setattr(b, 'tf_layers', 0)
                    setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'), 'EDTF'))
                    setattr(b, 'multi_head_classification', True)
                    setattr(b, 'with_positional_encoding', False)
                    setattr(b, 'learnable_pe', False)
                    exps.append(copy.deepcopy(b))
                    setattr(b, 'data_views', '1-2-3-4-5-6-7-8-9-10')
                    setattr(b, 'fusion', 'TransfomerEncoderDecoderMultiViewHead')
                    setattr(b, 'tf_layers', 0)
                    setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'), 'EDTF_PE'))
                    setattr(b, 'multi_head_classification', True)
                    setattr(b, 'with_positional_encoding', True)
                    setattr(b, 'learnable_pe', False)
                    exps.append(copy.deepcopy(b))
                    setattr(b, 'data_views', '1-2-3-4-5-6-7-8-9-10')
                    setattr(b, 'fusion', 'TransfomerEncoderDecoderMultiViewHead')
                    setattr(b, 'tf_layers', 0)
                    setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'), 'EDTF_learnPE'))
                    setattr(b, 'multi_head_classification', True)
                    setattr(b, 'with_positional_encoding', True)
                    setattr(b, 'learnable_pe', True)
                    exps.append(copy.deepcopy(b))
                    setattr(b, 'data_views', '1-2-3-4-5-6-7-8-9-10')
                    setattr(b, 'fusion', 'TransformerMultiViewHeadDecoder')
                    setattr(b, 'tf_layers', 0)
                    setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'), 'DTF'))
                    setattr(b, 'multi_head_classification', True)
                    setattr(b, 'with_positional_encoding', False)
                    setattr(b, 'learnable_pe', False)
                    exps.append(copy.deepcopy(b))
                    setattr(b, 'data_views', '1-2-3-4-5-6-7-8-9-10')
                    setattr(b, 'fusion', 'TransformerMultiViewHeadDecoder')
                    setattr(b, 'tf_layers', 0)
                    setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'), 'DTF_PE'))
                    setattr(b, 'multi_head_classification', True)
                    setattr(b, 'with_positional_encoding', True)
                    setattr(b, 'learnable_pe', False)
                    exps.append(copy.deepcopy(b))
                    setattr(b, 'data_views', '1-2-3-4-5-6-7-8-9-10')
                    setattr(b, 'fusion', 'TransformerMultiViewHeadDecoder')
                    setattr(b, 'tf_layers', 0)
                    setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'), 'DTF_learnPE'))
                    setattr(b, 'multi_head_classification', True)
                    setattr(b, 'with_positional_encoding', True)
                    setattr(b, 'learnable_pe', True)
                    exps.append(copy.deepcopy(b))

                if add_PE_exp:
                    b = copy.deepcopy(a)
                    setattr(b, 'data_views', '1-6-9')
                    setattr(b, 'fusion', 'Conv')
                    setattr(b, 'tf_layers', 1)
                    setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'), 'PE-RV'.format(fusion)))
                    setattr(b, 'multi_head_classification', True)
                    setattr(b, 'random_view_order', True)
                    exps.append(copy.deepcopy(b))
                    setattr(b, 'data_views', '1-6-9')
                    setattr(b, 'fusion', 'Conv')
                    setattr(b, 'tf_layers', 1)
                    setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'), 'PE-SV'.format(fusion)))
                    setattr(b, 'multi_head_classification', True)
                    setattr(b, 'random_view_order', False)
                    exps.append(copy.deepcopy(b))
                    setattr(b, 'data_views', '1-2-3-4-5-6-7-8-9-10')
                    setattr(b, 'fusion', 'Conv')
                    setattr(b, 'tf_layers', 1)
                    setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'), 'PE-RV-all'.format(fusion)))
                    setattr(b, 'multi_head_classification', True)
                    setattr(b, 'random_view_order', True)
                    exps.append(copy.deepcopy(b))

                if add_depth_exp:
                    for fusion in ['Squeeze&Excite']: #Conv
                        b = copy.deepcopy(a)
                        setattr(b, 'epochs', int(getattr(a, 'epochs') * depth_epoch_multiplier))
                        setattr(b, 'input_keys', 'x-depth')
                        setattr(b, 'load_keys', 'x-mask-depth')
                        setattr(b, 'depth_fusion', fusion)
                        #setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'), 'dfuse_{}'.format(fusion)))
                        #setattr(b, 'multi_head_classification', False)
                        #setattr(b, 'rgbd_wise_multi_head', False)
                        #setattr(b, 'rgbd_version', 'v1')
                        #setattr(b, 'tf_layers', tf_layers)
                        #setattr(b, 'with_rednet_pretrained', False)
                        #setattr(b, 'overwrite_imagenet', False)
                        #exps.append(copy.deepcopy(b))
                        #setattr(b, 'depth_fusion', fusion)
                        #setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'), 'dfuse_{}_multihead'.format(fusion)))
                        #setattr(b, 'multi_head_classification', True)
                        #setattr(b, 'rgbd_wise_multi_head', False)
                        #setattr(b, 'rgbd_version', 'v1')
                        #setattr(b, 'tf_layers', tf_layers)
                        #setattr(b, 'with_rednet_pretrained', False)
                        #setattr(b, 'overwrite_imagenet', False)
                        #exps.append(copy.deepcopy(b))
                        #setattr(b, 'depth_fusion', fusion)
                        #setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'),
                        # 'dfuse_{}_multihead-RGBDwise'.format(fusion)))
                        #setattr(b, 'multi_head_classification', True)
                        #setattr(b, 'rgbd_wise_multi_head', True)
                        #setattr(b, 'rgbd_version', 'v1')
                        #setattr(b, 'tf_layers', tf_layers)
                        #setattr(b, 'with_rednet_pretrained', False)
                        #setattr(b, 'overwrite_imagenet', False)
                        #exps.append(copy.deepcopy(b))
                        '''
                        setattr(b, 'depth_fusion', fusion)
                        setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'),
                                                          'dfuse_{}_multihead-RGBDwise-notf'.format(fusion)))
                        setattr(b, 'multi_head_classification', True)
                        setattr(b, 'rgbd_wise_multi_head', True)
                        setattr(b, 'rgbd_version', 'v1')
                        setattr(b, 'tf_layers', 0)
                        setattr(b, 'with_rednet_pretrained', False)
                        setattr(b, 'overwrite_imagenet', False)
                        exps.append(copy.deepcopy(b))
                        setattr(b, 'depth_fusion', fusion)
                        setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'),
                                                          'dfuse_{}_multihead-RGBDwise-v2'.format(fusion)))
                        setattr(b, 'multi_head_classification', True)
                        setattr(b, 'rgbd_wise_multi_head', True)
                        setattr(b, 'rgbd_version', 'v2')
                        setattr(b, 'tf_layers', tf_layers)
                        setattr(b, 'with_rednet_pretrained', False)
                        setattr(b, 'overwrite_imagenet', False)
                        exps.append(copy.deepcopy(b))
                        setattr(b, 'depth_fusion', 'Squeeze&Excite-Bi-Directional')
                        setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'),
                                                          'dfuse_{}_multihead-RGBDwise-v3'.format(fusion)))
                        setattr(b, 'multi_head_classification', True)
                        setattr(b, 'rgbd_wise_multi_head', True)
                        setattr(b, 'rgbd_version', 'v3')
                        setattr(b, 'tf_layers', tf_layers)
                        setattr(b, 'with_rednet_pretrained', False)
                        setattr(b, 'overwrite_imagenet', False)
                        exps.append(copy.deepcopy(b))
                        setattr(b, 'depth_fusion', fusion)
                        setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'),
                                                          'dfuse_{}_multihead-RGBDwise-v2-notf'.format(fusion)))
                        setattr(b, 'multi_head_classification', True)
                        setattr(b, 'rgbd_wise_multi_head', True)
                        setattr(b, 'rgbd_version', 'v2')
                        setattr(b, 'tf_layers', 0)
                        setattr(b, 'with_rednet_pretrained', False)
                        setattr(b, 'overwrite_imagenet', False)
                        exps.append(copy.deepcopy(b))
                        setattr(b, 'depth_fusion', 'Squeeze&Excite-Bi-Directional')
                        setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'),
                                                          'dfuse_{}_multihead-RGBDwise-v3-notf'.format(fusion)))
                        setattr(b, 'multi_head_classification', True)
                        setattr(b, 'rgbd_wise_multi_head', True)
                        setattr(b, 'rgbd_version', 'v3')
                        setattr(b, 'tf_layers', 0)
                        setattr(b, 'with_rednet_pretrained', False)
                        setattr(b, 'overwrite_imagenet', False)

                        #rednet
                        exps.append(copy.deepcopy(b))
                        setattr(b, 'depth_fusion', fusion)
                        setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'),
                                                          'dfuse_{}_multihead-RGBDwise-rednet'.format(fusion)))
                        setattr(b, 'multi_head_classification', True)
                        setattr(b, 'rgbd_wise_multi_head', True)
                        setattr(b, 'rgbd_version', 'v1')
                        setattr(b, 'tf_layers', tf_layers)
                        setattr(b, 'with_rednet_pretrained', True)
                        setattr(b, 'overwrite_imagenet', True)
                        exps.append(copy.deepcopy(b))
                        setattr(b, 'depth_fusion', fusion)
                        setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'),
                                                          'dfuse_{}_multihead-RGBDwise-v2-rednet'.format(fusion)))
                        setattr(b, 'multi_head_classification', True)
                        setattr(b, 'rgbd_wise_multi_head', True)
                        setattr(b, 'rgbd_version', 'v2')
                        setattr(b, 'tf_layers', tf_layers)
                        setattr(b, 'with_rednet_pretrained', True)
                        setattr(b, 'overwrite_imagenet', True)
                        exps.append(copy.deepcopy(b))
                        setattr(b, 'depth_fusion', 'Squeeze&Excite-Bi-Directional')
                        setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'),
                                                          'dfuse_{}_multihead-RGBDwise-v3-rednet'.format(fusion)))
                        setattr(b, 'multi_head_classification', True)
                        setattr(b, 'rgbd_wise_multi_head', True)
                        setattr(b, 'rgbd_version', 'v3')
                        setattr(b, 'tf_layers', tf_layers)
                        setattr(b, 'with_rednet_pretrained', True)
                        setattr(b, 'overwrite_imagenet', True)
                        exps.append(copy.deepcopy(b))
                        setattr(b, 'depth_fusion', fusion)
                        setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'),
                                                          'dfuse_{}_multihead-RGBDwise-rednet-mix'.format(fusion)))
                        setattr(b, 'multi_head_classification', True)
                        setattr(b, 'rgbd_wise_multi_head', True)
                        setattr(b, 'rgbd_version', 'v1')
                        setattr(b, 'tf_layers', tf_layers)
                        setattr(b, 'with_rednet_pretrained', True)
                        setattr(b, 'overwrite_imagenet', False)
                        exps.append(copy.deepcopy(b))
                        setattr(b, 'depth_fusion', fusion)
                        setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'),
                                                          'dfuse_{}_multihead-RGBDwise-v2-rednet-mix'.format(fusion)))
                        setattr(b, 'multi_head_classification', True)
                        setattr(b, 'rgbd_wise_multi_head', True)
                        setattr(b, 'rgbd_version', 'v2')
                        setattr(b, 'tf_layers', tf_layers)
                        setattr(b, 'with_rednet_pretrained', True)
                        setattr(b, 'overwrite_imagenet', False)
                        exps.append(copy.deepcopy(b))
                        setattr(b, 'depth_fusion', 'Squeeze&Excite-Bi-Directional')
                        setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'),
                                                          'dfuse_{}_multihead-RGBDwise-v3-rednet-mix'.format(fusion)))
                        setattr(b, 'multi_head_classification', True)
                        setattr(b, 'rgbd_wise_multi_head', True)
                        setattr(b, 'rgbd_version', 'v3')
                        setattr(b, 'tf_layers', tf_layers)
                        setattr(b, 'with_rednet_pretrained', True)
                        setattr(b, 'overwrite_imagenet', False)
                        exps.append(copy.deepcopy(b))
                        
                        '''


                if add_roi_crop_exp:
                    b = copy.deepcopy(a)
                    setattr(b, 'roicrop', False)
                    setattr(b, 'name', '{}_{}'.format(getattr(a, 'name'), 'no_roicrop'))
                    exps.append(copy.deepcopy(b))

                if add_shuffle_exp:
                    for keys in [[], ['shuf_views_cw'], ['shuf_views_vw'], ['shuf_views_cw', 'shuf_views_vw']]:
                        b = copy.deepcopy(a)
                        setattr(b, 'shuf_views', True)
                        setattr(b, 'name', '{}_{}'.format(getattr(b, 'name'), 'aug_sv'))
                        for key in keys:
                            setattr(b, key, False)
                            setattr(b, 'name', '{}_{}'.format(getattr(b, 'name'), key))
                        exps.append(copy.deepcopy(b))

    arg_keys = ['outdir', 'multi_head_classification', 'rgbd_wise_multi_head', 'tf_layers', 'width', 'height',
                'name', 'multiview', 'roicrop', 'shuf_views', 'shuf_views_cw', 'shuf_views_vw', 'fusion',
                'depth_fusion',
                'input_keys',
                'load_keys',
                'views',
                'data_views',
                'encoder_path',
                'rotations',
                'rotation_aug',
                'flip_aug',
                'random_view_order',
                'lr', 'lr_fusion', 'lr_encoder', 'lr_group_wise',
                'epochs']

    l = len(exps)
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










