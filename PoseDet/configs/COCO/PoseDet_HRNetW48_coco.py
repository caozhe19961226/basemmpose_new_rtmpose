_base_ = [
    '../_base_/dataset_coco.py',
    '../_base_/PoseDet_DLA34.py',
    '../_base_/train_schedule.py',
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadKeypoints'),
    dict(type='LoadAnnotations', with_bbox=True),    
    dict(
        type='CenterRandomCropXiao',
        scale_factor=0.5,
        rot_factor=0,
        patch_width=512,
        patch_height=512),
    dict(type='Resize',         
        img_scale=[
                    (800,512),
                    # (1000,640),
                    ], 
        multiscale_mode='value', 
        keep_ratio=True,
        with_keypoints=True),
    dict(type='RandomFlip', flip_ratio=0.5, with_keypoints=True, gt_num_keypoints=17),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='GaussianMap', strides=[8,16,32], num_keypoints=17, sigma=2),
    dict(type='FormatBundleKeypoints'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 
                                'gt_keypoints', 'gt_num_keypoints',
                                'heatmap', 'heatmap_weight']),
    ]
    
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(800,512),
        img_scale=(1000,640),
        # multi-scale test
        # img_scale=[(520, 300), (666, 400), (833, 500), (1000, 600), (1167, 700), (1333, 800)]
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

channels = 256
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    pretrained='pretrained/hrnetv2_w48_imagenet_pretrained.pth',
    backbone=dict(
        _delete_=True,
        type='HRNet',
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384)))),

    #####
    # neck=dict(
    #     type='DiteHRNet',
    #     in_channels=4,
    #     with_cp=False,
    #     extra=dict(
    #         stem=dict(stem_channels=32, out_channels=32),
    #         num_stages=3,
    #         stages_spec=dict(
    #             num_modules=(2, 4, 2),
    #             num_branches=(2, 3, 4),
    #             num_blocks=(2, 2, 2),
    #             with_fuse=(True, True, True),
    #             num_channels=(
    #                 (40, 80),
    #                 (40, 80, 160),
    #                 (40, 80, 160, 320),
    #             )),
    #         with_head=True,
    #     )),
    neck=dict(
        _delete_=True,
        type='HRFPN',
        in_channels=[48, 96, 192, 384],
        out_channels=channels,             #输出通道是256
        stride=2,
        num_outs=4,                        #输出3级
        norm_cfg=norm_cfg),


    bbox_head=dict(
# type='PoseDetHead',
        type='PoseDetHeadHeatMapMl',
        norm_cfg=norm_cfg,
        num_classes=1,
        in_channels=channels,
        feat_channels=channels,
        embedding_feat_channels=channels,
        init_convs=3,
        refine_convs=2,
        cls_convs=2,
        gradient_mul=0.1,
        dcn_kernel=(1,17),
        refine_num=1,
        point_strides=[8, 16, 32, 64],
        point_base_scale=4,
        num_keypoints=17,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_keypoints_init=dict(type='KeypointsLoss',
                                d_type='L2',
                                weight=.1,
                                stage='init',
                                normalize_factor=1,
                                ),
        loss_keypoints_refine=dict(type='KeypointsLoss',
                                d_type='L2',
                                weight=.2,
                                stage='refine',
                                normalize_factor=1,
                                ),
        loss_heatmap=dict(type='HeatmapLoss', weight=.1, with_sigmas=False),
        # point_strides=[8, 16, 32],
        # in_channels=channels,
        # feat_channels=channels,
        # embedding_feat_channels=channels,
        ),
    ) 

exp_name = 'PoseDet_HRNetW48_coco'
work_dir = './output/' + exp_name