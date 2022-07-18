import os
import os.path as osp
import argparse
from unittest import result
from matplotlib import pyplot as plt
from mmcv.parallel import MMDataParallel
import numpy as np
import torch

from mmcv import Config
from mmcv import build_from_cfg, Registry

from mmpose.datasets import build_dataset, build_dataloader
from mmpose.datasets.builder import PIPELINES
from mmpose.datasets import DatasetInfo
from mmpose.models import   build_posenet
from mmpose.models.kinematic import Human3DKinematicLayer, Human3DKinematicLayerV2
from mmpose.models.utils.human_kinematic_utils import HumanKinematic
from mmpose.apis   import  vis_3d_pose_result, init_pose_model, inference_pose_lifter_model, single_gpu_test

PIPELINES = Registry('pipeline')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="../configs/body/3d_kpt_sview_rgb_vid/video_kinematic_pose_lift/h36m/kinematic3d_params_optimz_h36m_27frames_supervised.py")
    parser.add_argument('--checkpoint', 
        default="../tools/work_dirs/videopose3d_h36m_27frames_fullconv_supervised/latest.pth")
    return parser.parse_args()

def process_3dkpts(keypoints_3d, rebase_keypoint_height = True):
    keypoints_3d = keypoints_3d[..., [0, 2, 1]]
    keypoints_3d[..., 0] = -keypoints_3d[..., 0]
    keypoints_3d[..., 2] = -keypoints_3d[..., 2]
    # rebase height (z-axis)
    if rebase_keypoint_height:
        keypoints_3d[..., 2] -= np.min(
            keypoints_3d[..., 2], axis=-1, keepdims=True)
    return keypoints_3d
def process_result(result):
    poses_3d = result['preds']
    if poses_3d.shape[-1] != 4:
        assert poses_3d.shape[-1] == 3
        dummy_score = np.ones(
            poses_3d.shape[:-1] + (1, ), dtype=poses_3d.dtype)
        poses_3d = np.concatenate((poses_3d, dummy_score), axis=-1)
    
    pose_lift_results = []
    for pose_3d in poses_3d:
        pose_result = {}
        pose_result['keypoints_3d'] = pose_3d
        pose_lift_results.append(pose_result)

    # Pose processing
    pose_lift_results_vis = []
    for idx, res in enumerate(pose_lift_results):
        res['keypoints_3d'] = process_3dkpts(res['keypoints_3d'])
        # add title
        # det_res = pose_det_results[idx]
        # instance_id = det_res['track_id']
        # res['title'] = f'Prediction ({instance_id})'
        # only visualize the target frame
        # res['keypoints'] = det_res['keypoints']
        # res['bbox'] = det_res['bbox']
        # res['track_id'] = instance_id
        pose_lift_results_vis.append(res)
    return pose_lift_results_vis

def get_dataloader(cfg, dataset_cfg):
    # step 0: build dataset
    dataset = build_dataset(dataset_cfg, dict(test_mode=True))
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(seed=cfg.get('seed'), drop_last=False, dist=False),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'seed',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }
    # step2: cfg.data.test_dataloader has higher priority
    test_loader_cfg = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **dict(workers_per_gpu=cfg.data.get('workers_per_gpu', 1)),
        **dict(samples_per_gpu=cfg.data.get('samples_per_gpu', 1)),
        **cfg.data.get('test_dataloader', {})
    }
    return build_dataloader(dataset, **test_loader_cfg)

def main():
    args = parse_args()
    cfg  = Config.fromfile(args.config)
    # Load checkpoint
    if args.checkpoint is None:
        args.checkpoint = osp.join('../tools/work_dirs',
            osp.splitext(osp.basename(args.config))[0], "latest.pth")
        
    # Build dataset loader
    data_cfg  = cfg.data.train #select dataset config
    pose_lift_dataloader = get_dataloader(cfg, data_cfg)
    dataset_info = DatasetInfo(data_cfg.dataset_info)
    
    # Prepare pose model (for inference)
    pose_lift_model = build_posenet(cfg.model)
    # pose_lift_model  = init_pose_model(cfg, args.checkpoint)
    
    # [DEBUG] Prepare kinematic layer 
    kinematic_layout = HumanKinematic.get_layout('h36m', '3d')
    pose_kinematic_layer = Human3DKinematicLayerV2(
        layout=kinematic_layout
    ) 

    # Run 2D-3D pose lifting
    # for i, data in enumerate(pose_lift_dataloader):
    #     print(i, data['input'].shape)
    data = next(iter(pose_lift_dataloader))

    # pose_lift_model = MMDataParallel(pose_lift_model, device_ids=[0])
    # with torch.no_grad():
        # result = pose_lift_model(return_loss=False, **data)

    est_pose_from_kinematic_layer = pose_kinematic_layer(data['dirc_vector'], data['bone_length'],) 
    print(est_pose_from_kinematic_layer.shape)
    # Process pose lifting result
    
    # Visualize dataset
    # pose_lift_results_vis = process_result(result)
    # Visualize gt
    pose_lift_gt_vis = [dict(
        keypoints_3d = process_3dkpts(data['target'][i].cpu().numpy())) for i in range(len(data['target']))]
    # Visualize prediction from kinematic layer
    pose_kinematic_vis= [dict(
        keypoints_3d = process_3dkpts(est_pose_from_kinematic_layer[i].cpu().numpy())) for i in range(len(est_pose_from_kinematic_layer))
    ]
    
    img_gt = vis_3d_pose_result(
        model  = pose_lift_model,
        result = pose_lift_gt_vis,
        dataset= "Body3DH36MDataset",
        dataset_info=  dataset_info,
        out_file="gt.png",
        num_instances=1)
    
    img_kinematic = vis_3d_pose_result(
        model  = pose_lift_model,
        result = pose_kinematic_vis,
        dataset= "Body3DH36MDataset",
        dataset_info=  dataset_info,
        out_file="kinematic.png",
        num_instances=1)
    
    # img_preds = vis_3d_pose_result(
    #     model  = pose_lift_model,
    #     result = pose_lift_results_vis,
    #     dataset= "Body3DH36MDataset",
    #     dataset_info=  dataset_info,
    #     out_file="preds.png",
    #     num_instances=1)
    
if __name__ == '__main__':
    main()