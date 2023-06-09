#data loading
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
    dict(type='GaussianMap', strides=[8,16,32,64], num_keypoints=17, sigma=2),
    dict(type='FormatBundleKeypoints'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 
                                'gt_keypoints', 'gt_num_keypoints',
                                'heatmap', 'heatmap_weight']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800,512),
        # img_scale=(1000,640),
        # multi-scale test
        # img_scale=[(520, 300), (666, 400), (833, 500), (1000, 600), (1167, 700), (1333, 800)] 64115 Epoch
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
dataset_type = 'CocoKeypoints'
data_root = '/root/PoseDet/data/coco/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/person_keypoints_train2017.json',
        img_prefix=data_root + 'train2017/',
        # ann_file=data_root + 'annotations/keypoints_train.json',
        # img_prefix=data_root + 'train/',
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/keypoints_train.json',
        # img_prefix=data_root + 'val/',
        ann_file=data_root + 'annotations/person_keypoints_train2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/person_keypoints_val2017.json',
        img_prefix=data_root + 'val2017/',
        # img_prefix=data_root + 'test2017/',
        pipeline=test_pipeline))

print(data)