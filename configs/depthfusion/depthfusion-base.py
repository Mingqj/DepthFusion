_base_ = [
    '../_base_/datasets/nus-3d.py',
    '../_base_/schedules/cyclic_20e.py',
    '../_base_/default_runtime.py'
]
'''

'''
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-54.0, -54.0, -3.0, 54.0, 54.0, 5.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

data_config = {
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 5,
    'input_size': (320, 800),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.44),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'random_crop_height': True,
    'vflip':True,
    'resize_test': 0.04,

    'pmd': dict(
        brightness_delta=32,
        contrast_lower=0.5,
        contrast_upper=1.5,
        saturation_lower=0.5,
        saturation_upper=1.5,
        hue_delta=18,
        rate=0.5
    )
}

grid_config = {
    'x': [-54.0, 54.0, 0.6],
    'y': [-54.0, 54.0, 0.6],
    'z': [-3, 5, 8],
    'depth': [1.0, 60.0, 0.5],
}


# Model
voxel_size = [0.075, 0.075, 0.2]

feat_bev_img_dim = 32
img_feat_dim = 128
model = dict(
    type='DepthFusion',
    use_grid_mask=True,
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=False,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[512, 1024, 2048],
        out_channels=img_feat_dim,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=img_feat_dim,
        out_channels=feat_bev_img_dim,
        downsample=8,
        with_depth_from_lidar=True),
    pts_voxel_layer=dict(
        max_num_points=10, voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_voxels=(120000, 160000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        base_channels=24,
        sparse_shape=[41, 1440, 1440],
        output_channels=192,
        order=('conv', 'norm', 'act'),
        encoder_channels=((24, 24, 48),
                          (48, 48, 96),
                          (96, 96, 192),
                          (192, 192)),
        encoder_paddings=((0, 0, 1),
                          (0, 0, 1),
                          (0, 0, [0, 1, 1]),
                          (0, 0)),
        block_type='basicblock'),

    pts_backbone=dict(
        type='SECOND',
        in_channels=384,
        out_channels=[192, 384],
        layer_nums=[8, 8],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[192, 384],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='DepthFusionTinyHead',
        feat_bev_img_dim=feat_bev_img_dim,
        img_feat_dim=img_feat_dim,
        sparse_fuse_layers=2,
        dense_fuse_layers=2,
        instance_attn=False,
        num_proposals=200,
        in_channels=512,
        hidden_channel=128,
        num_classes=10,
        num_decoder_layers=1,
        num_heads=8,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        auxiliary=True,
        common_heads0=dict(
            center0=[2, 2],
            height0=[1, 2],
            dim0=[3, 2],
            rot0=[2, 2],
            vel0=[2, 2]),
        common_heads=dict(
            center=[2, 2],
            height=[1, 2],
            dim=[3, 2],
            rot=[2, 2],
            vel=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=10),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0 * 1.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25 * 1.0),
        loss_heatmap=dict(
            type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_heatmap_sparse=dict(
            type='GaussianFocalLoss', reduction='mean', loss_weight=1.0 * 1.0),
        loss_cls0=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_bbox0=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        ),
    train_cfg=dict(pts=dict(
        dataset='nuScenes',
        point_cloud_range=point_cloud_range,
        grid_size=[1440, 1440, 40],
        voxel_size=voxel_size,
        out_size_factor=8,
        gaussian_overlap=0.1,
        min_radius=2,
        pos_weight=-1,
        code_weights=[
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ],
        assigner=dict(
            type='HungarianAssigner3D',
            iou_calculator=dict(
                type='BboxOverlaps3D', coordinate='lidar'),
            cls_cost=dict(
                type='FocalLossCost',
                gamma=2.0,
                alpha=0.25,
                weight=0.15),
            reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
            iou_cost=dict(type='IoU3DCost', weight=0.25)))),
    test_cfg=dict(pts=dict(
        dataset='nuScenes',
        grid_size=[1440, 1440, 40],
        img_feat_downsample=8,
        out_size_factor=8,
        voxel_size=voxel_size[:2],
        pc_range=point_cloud_range[:2],
        nms_type=None,
        use_rotate_nms=True, 
        max_num=500,
        nms_thr=0.1)),
)

dataset_type = 'NuScenesDataset'
data_root = './data/nuscenes/'

file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-22.5 * 2, 22.5 * 2),
    scale_lim=(0.9, 1.1),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    tran_lim=[0.5, 0.5, 0.5]
)

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'bevdetv3-nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True, opencv_pp=True,
        data_config=data_config),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='ToEgo'),
    dict(type='LoadAnnotations'),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='VelocityAug'),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(type='PointToMultiViewDepthFusion', downsample=1,
         grid_config=grid_config),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d',
                                 'img_inputs', 'gt_depth',
                                 'gt_bboxes_ignore'
                                 ])
]

test_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=False, opencv_pp=True,
        data_config=data_config),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='ToEgo'),
    dict(type='LoadAnnotations'),

    # # nuScenes-C
    # dict(
    #     type='CorruptionMethods',
    #     corruption_severity_dict=
    #         {
    #             # 'background_noise':3,
    #             # 'snow_sim_image': 3,
    #             # 'snow_sim_lidar': 3,
    #             # 'rain_sim_image': 3,
    #             # 'fog_sim_image': 3,
    #             # 'fog_sim_lidar': 3,
    #             # 'sun_sim': 3,
    #             # 'scene_glare_noise': 3
    #             # 'density_dec_global': 3
    #             # 'cutout_local': 3
    #             # 'lidar_crosstalk_noise': 3
    #             # 'fov_filter': 3
    #             # 'gaussian_noise': 3
    #             # 'uniform_noise': 3
    #             # 'impulse_noise': 3
    #             # 'gauss_sim': 3
    #             # 'uniform_sim': 3
    #             # 'impulse_sim': 3
    #             # 'fulltrajectory_noise': 3
    #             # 'moving_noise_bbox': 3
    #             # 'density_dec_bbox' : 3
    #             # 'cutout_bbox': 3
    #             # 'gaussian_noise_bbox': 3
    #             # 'uniform_noise_bbox': 3
    #             # 'impulse_noise_bbox': 3
    #             # 'shear_bbox': 3
    #             # 'scale_bbox': 3
    #             # 'rotation_bbox': 3
    #             # 'spatial_alignment_noise': 3
    #             # 'temporal_alignment_noise': 3
    #         },
    # ),
    
    dict(type='BEVAug',
         bda_aug_conf=bda_aug_conf,
         classes=class_names,
         is_train=False),
    dict(type='PointToMultiViewDepthFusion', downsample=1,
         grid_config=grid_config),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1, 
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs', 'gt_depth'])
        ])
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'bevdetv3-nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            modality=input_modality,
            img_info_prototype='bevdet',
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(pipeline=test_pipeline, classes=class_names,
             modality=input_modality,
             ann_file=data_root + 'bevdetv3-nuscenes_infos_val.pkl',
             img_info_prototype='bevdet'),
    test=dict(pipeline=test_pipeline, classes=class_names,
              modality=input_modality,
              ann_file=data_root + 'bevdetv3-nuscenes_infos_val.pkl',
              img_info_prototype='bevdet'))

evaluation = dict(interval=20, pipeline=test_pipeline)
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)  # for 64 total batch size
two_stage = True
runner = dict(type='TwoStageRunner', max_epochs=20)
num_proposals_test = 300
find_unused_parameters=True
