exp_name = 'ttvsr_reds4'

# model settings
model = dict(
    type='TTVSR',
    generator=dict(
        type='TTVSRNet', mid_channels=64, num_blocks=60 ,stride=4,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(fix_iter=5000)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)

# dataset settings
train_dataset_type = 'SRREDSMultipleGTDataset'
val_dataset_type = 'SRREDSMultipleGTDataset'
dataset_root = './dataset/REDS/'

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='TemporalReverse', keys=['lq_path',"gt_path"], reverse_ratio=0.5),
    dict(type='LoadImageFromFileList', io_backend='disk', key='lq', channel_order='rgb'),
    dict(type='LoadImageFromFileList', io_backend='disk', key='gt', channel_order='rgb'),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

test_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFileList', io_backend='disk', key='lq', channel_order='rgb'),
    dict(type='LoadImageFromFileList', io_backend='disk', key='gt', channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path', 'key'])
]

data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=2, drop_last=True),  # 2 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder=dataset_root+'train/train_sharp_bicubic/X4',
            gt_folder=dataset_root+'train/train_sharp',
            num_input_frames=50,
            pipeline=train_pipeline,
            scale=4,
            val_partition='REDS4',
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        lq_folder=dataset_root+'train/train_sharp_bicubic/X4',
        gt_folder=dataset_root+'train/train_sharp',
        num_input_frames=100,
        pipeline=test_pipeline,
        scale=4,
        val_partition='REDS4',
        test_mode=True),
    # test
    test=dict(
        type=val_dataset_type,
        lq_folder=dataset_root+'train/train_sharp_bicubic/X4',
        gt_folder=dataset_root+'train/train_sharp',
        num_input_frames=100,
        pipeline=test_pipeline,
        scale=4,
        val_partition='REDS4',
        test_mode=True),
)
# optimizer
optimizers = dict(
    generator=dict(type='Adam', lr=2e-4, betas=(0.9, 0.99), paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.125)})))

# learning policy
total_iters = 400000
lr_config = dict(policy='CosineRestart', by_epoch=False, periods=[400000], restart_weights=[1], min_lr=1e-7)
checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False, create_symlink=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=10000, save_image=False, gpu_collect=True)
log_config = dict(interval=1000, hooks=[dict(type='TextLoggerHook', by_epoch=False, interval_exp_name=400000),])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir =  'ttvsr_reds4'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True