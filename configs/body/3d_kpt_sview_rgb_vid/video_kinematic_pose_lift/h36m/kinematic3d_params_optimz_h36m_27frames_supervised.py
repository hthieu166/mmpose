_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/h36m.py'
]
evaluation = dict(interval=10, metric=['angle_l1', 'mpjpe', 'p-mpjpe'], save_best='MPJPE')

# optimizer settings
optimizer = dict(
    type='Adam',
    lr=1e-3,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='exp',
    by_epoch=True,
    gamma=0.975,
)

total_epochs = 160

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    ])

# model settings
model = dict(
    type='KinematicModelPredictor',
    pretrained=None,
    backbone=dict(
        type='TCN',
        in_channels=2 * 17,
        stem_channels=1024,
        num_blocks=2,
        kernel_sizes=(3, 3, 3),
        dropout=0.25,
        use_stride_conv=True),
    # Replace the temporal regression head with the human kinematic layer
    params_pred_head=dict(
        type='HumanKinematicParamsRegressionHead', #Using human kinematic layer for regression head
        in_channels=1024,
        dataset = 'h36m',
        mode='3d',
        kinematic_layer = dict(type="Human3DKinematicLayerV2"),
        loss_kinematic_params=dict(type='L1Loss')),
    train_cfg=dict(),
    test_cfg=dict(restore_global_position=True))

# data settings
data_root = '/mnt/data0-nfs/shared-datasets/human36m/processed'
data_cfg = dict(
    num_joints=17,
    seq_len=27,
    seq_frame_interval=1,
    causal=False,
    temporal_padding=True,
    joint_2d_src='gt',
    need_camera_param=True,
    camera_param_file=f'{data_root}/annotation_body3d/cameras.pkl',
)

train_pipeline = [
    dict(
        type='GetRootCenteredPose',
        item='target',
        visible_item='target_visible',
        root_index=0,
        root_name='root_position',
        remove_root=False),
    dict(type='ImageCoordinateNormalization', item='input_2d'),
    # No-flipping augmentation
    # dict( 
    #     type='RelativeJointRandomFlip',
    #     item=['input_2d', 'target'],
    #     flip_cfg=[
    #         dict(center_mode='static', center_x=0.),
    #         dict(center_mode='root', center_index=0)
    #     ],
    #     visible_item=['input_2d_visible', 'target_visible'],
    #     flip_prob=0.5),
    dict(
        type='GenSkeKinematicFeat', 
        dataset='h36m', 
        mode='3d',
        feats=['dirc_vector', 'bone_length'], 
        ),
    dict(type='PoseSequenceToTensor', item='input_2d'),
    dict(
        type='Collect',
        keys=[('input_2d', 'input'), 'target', 'dirc_vector', 'bone_length'],
        meta_name='metas',
        meta_keys=['target_image_path', 'flip_pairs', 'root_position'])
]

val_pipeline = [
    dict(
        type='GetRootCenteredPose',
        item='target',
        visible_item='target_visible',
        root_index=0,
        root_name='root_position',
        remove_root=False),
    dict(type='ImageCoordinateNormalization', item='input_2d'),
    dict(
        type='GenSkeKinematicFeat', 
        dataset='h36m', 
        mode='3d',
        feats=['dirc_vector', 'bone_length'], 
        ),
    dict(type='PoseSequenceToTensor', item='input_2d'),
    dict(
        type='Collect',
        keys=[('input_2d', 'input'), 'target', 'dirc_vector', 'bone_length'],
        meta_name='metas',
        meta_keys=['target_image_path', 'flip_pairs', 'root_position'])
]

test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=128),
    test_dataloader=dict(samples_per_gpu=128),
    train=dict(
        type='Body3DH36MKinematicDataset',
        ann_file=f'{data_root}/annotation_body3d/fps50/h36m_train.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='Body3DH36MKinematicDataset',
        ann_file=f'{data_root}/annotation_body3d/fps50/h36m_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='Body3DH36MKinematicDataset',
        ann_file=f'{data_root}/annotation_body3d/fps50/h36m_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
