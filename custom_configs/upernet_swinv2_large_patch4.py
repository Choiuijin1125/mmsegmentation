norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SwinTransformerV2',
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False),
    decode_head=dict(
    type="UPerHead",
    in_channels=[192,384,768,1536],
    in_index=[0, 1, 2, 3],
    pool_scales=(1, 2, 3, 6),
    channels=512,
    dropout_ratio=0.1,
    num_classes=6,
    norm_cfg=norm_cfg,
    align_corners=False,
    loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, 
    loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset settings
size = 224*8
dataset_type = 'CustomDataset'
data_root = f'../input/{size}x{size}/'
classes = ['background', 'kidney', 'prostate', 'largeintestine', 'spleen', 'lung']
palette = [[0,0,0], [255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255]]
img_norm_cfg = dict(mean=[196.869, 190.186, 194.802], std=[63.010, 66.765, 65.745], to_rgb=True)


fold=1

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    #dict(type='Resize', img_scale=(size, size), keep_ratio=True),
    #dict(type='RandomCrop', crop_size=(10, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(size, size), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(size, size),
        flip=False,
        transforms=[
            #dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train',
        ann_dir='masks',
        img_suffix=".png",
        seg_map_suffix='.png',        
        split=f"../folds/fold_{fold}_{size}.txt",
        classes=classes,
        palette=palette,
        pipeline=train_pipeline),
    
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train',
        ann_dir='masks',
        img_suffix=".png",
        seg_map_suffix='.png',        
        split=f"../folds/valid_{fold}_{size}.txt",
        classes=classes,
        palette=palette,
        pipeline=test_pipeline),
    
    test=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        img_dir='train',
        ann_dir='masks',
        img_suffix=".png",
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))

log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

cudnn_benchmark = True

optimizer = dict(
    type='AdamW',
    lr=1e-03,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=15000)
checkpoint_config = dict(by_epoch=False, interval=500, save_optimizer=False)
evaluation = dict(interval=500, metric='mDice', pre_eval=True)
work_dir = f'./work_dirs/upernet_swinv2_large_patch4_fold_{fold}'
#gpu_ids = range(0, 4)
auto_resume = False
