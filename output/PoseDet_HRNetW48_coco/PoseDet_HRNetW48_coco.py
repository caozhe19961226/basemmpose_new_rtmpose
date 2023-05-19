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
    dict(
        type='Resize',
        img_scale=[(800, 512)],
        multiscale_mode='value',
        keep_ratio=True,
        with_keypoints=True),
    dict(
        type='RandomFlip',
        flip_ratio=0.5,
        with_keypoints=True,
        gt_num_keypoints=17),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='GaussianMap', strides=[8, 16, 32], num_keypoints=17, sigma=2),
    dict(type='FormatBundleKeypoints'),
    dict(
        type='Collect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_keypoints',
            'gt_num_keypoints', 'heatmap', 'heatmap_weight'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
dataset_type = 'CocoKeypoints'
data_root = '/mnt/data/tcy/coco/'
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=4,
    train=dict(
        type='CocoKeypoints',
        ann_file=
        '/mnt/data/tcy/coco/annotations/person_keypoints_train2017.json',
        img_prefix='/mnt/data/tcy/coco/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadKeypoints'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='CenterRandomCropXiao',
                scale_factor=0.5,
                rot_factor=0,
                patch_width=512,
                patch_height=512),
            dict(
                type='Resize',
                img_scale=[(800, 512)],
                multiscale_mode='value',
                keep_ratio=True,
                with_keypoints=True),
            dict(
                type='RandomFlip',
                flip_ratio=0.5,
                with_keypoints=True,
                gt_num_keypoints=17),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='GaussianMap',
                strides=[8, 16, 32, 64],
                num_keypoints=17,
                sigma=2),
            dict(type='FormatBundleKeypoints'),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_keypoints',
                    'gt_num_keypoints', 'heatmap', 'heatmap_weight'
                ])
        ]),
    val=dict(
        type='CocoKeypoints',
        ann_file='/mnt/data/tcy/coco/annotations/person_keypoints_val2017.json',
        img_prefix='/mnt/data/tcy/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoKeypoints',
        ann_file=
        '/mnt/data/tcy/coco/annotations/person_keypoints_val2017_small500.json',
        img_prefix='/mnt/data/tcy/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='PoseDetDetector',
    pretrained='pretrained/hrnetv2_w48_imagenet_pretrained.pth',
    backbone=dict(
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
    neck=dict(
        type='HRFPN',
        in_channels=[48, 96, 192, 384],
        out_channels=256,
        stride=2,
        num_outs=3,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    bbox_head=dict(
        type='PoseDetHeadHeatMapMl',
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        embedding_feat_channels=256,
        init_convs=3,
        refine_convs=2,
        cls_convs=2,
        gradient_mul=0.1,
        dcn_kernel=(1, 17),
        refine_num=1,
        point_strides=[8, 16, 32],
        point_base_scale=4,
        num_keypoints=17,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_keypoints_init=dict(
            type='KeypointsLoss',
            d_type='L2',
            weight=0.1,
            stage='init',
            normalize_factor=1),
        loss_keypoints_refine=dict(
            type='KeypointsLoss',
            d_type='L2',
            weight=0.2,
            stage='refine',
            normalize_factor=1),
        loss_heatmap=dict(type='HeatmapLoss', weight=0.1, with_sigmas=False)))
train_cfg = dict(
    init=dict(
        assigner=dict(
            type='KeypointsAssigner',
            scale=4,
            pos_num=1,
            number_keypoints_thr=3,
            num_keypoints=17,
            center_type='keypoints'),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    refine=dict(
        assigner=dict(
            type='OksAssigner',
            pos_PD_thr=0.7,
            neg_PD_thr=0.7,
            min_pos_iou=0.52,
            ignore_iof_thr=-1,
            match_low_quality=True,
            num_keypoints=17,
            number_keypoints_thr=3),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    cls=dict(
        assigner=dict(
            type='OksAssigner',
            pos_PD_thr=0.6,
            neg_PD_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1,
            match_low_quality=False,
            num_keypoints=17,
            number_keypoints_thr=3),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    nms_pre=500,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='keypoints_nms', iou_thr=0.2),
    max_per_img=100)
checkpoint_config = dict(interval=5)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
evaluation = dict(interval=5, metric='bbox')
optimizer = dict(type='Adam', lr=0.0001, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.3333333333333333,
    step=[180, 200])
total_epochs = 210
channels = 256
exp_name = 'PoseDet_HRNetW48_coco'
work_dir = './output/PoseDet_HRNetW48_coco'
find_unused_parameters = True
gpu_ids = range(0, 1)
