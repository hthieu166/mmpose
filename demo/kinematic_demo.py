from demo_utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="../configs/body/3d_kpt_sview_rgb_vid/video_kinematic_pose_lift/h36m/kinematic3d_params_optimz_h36m_27frames_supervised.py")
    parser.add_argument('--checkpoint', 
        default="../tools/work_dirs/kinematic3d_params_optimz_h36m_27frames_supervised/best_MPJPE_epoch_110.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = Config.fromfile(args.config)
    # Load checkpoint
    if args.checkpoint is None:
        args.checkpoint = osp.join('../tools/work_dirs',
            osp.splitext(osp.basename(args.config))[0], "latest.pth")
        
    # Build dataset loader
    data_cfg  = cfg.data.test #select dataset config
    pose_lift_dataloader = get_dataloader(cfg, data_cfg)
    dataset_info = DatasetInfo(data_cfg.dataset_info)
    
    # Prepare pose model (for inference)
    # pose_lift_model = build_posenet(cfg.model)
    pose_lift_model  = init_pose_model(cfg, args.checkpoint)
    
    # [DEBUG] Prepare kinematic layer 
    kinematic_layout = HumanKinematic.get_layout('h36m', '3d')
    pose_kinematic_layer = Human3DKinematicLayerV2(
        layout=kinematic_layout
    ) 

    # Run 2D-3D pose lifting
    # for i, data in enumerate(pose_lift_dataloader):
    #     print(i, data['input'].shape)
    data = next(iter(pose_lift_dataloader))
    
    pose_lift_model = MMDataParallel(pose_lift_model, device_ids=[0])
    pose_lift_model.eval()
    with torch.no_grad():
        result = pose_lift_model(return_loss=False, **data)

    est_pose_from_kinematic_layer = pose_kinematic_layer(result['preds_dirc_vector'].cpu(), data['bone_length'].cpu(),) 
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