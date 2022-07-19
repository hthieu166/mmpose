from ntpath import join
import os
import os.path as osp
import argparse
from matplotlib import pyplot as plt
import numpy as np
import torch as T

from mmcv import Config
from mmcv.parallel import MMDataParallel

from mmpose.datasets import build_dataset, build_dataloader
from mmpose.datasets import DatasetInfo
from mmpose.models import   build_posenet
from mmpose.models.kinematic import Human3DKinematicLayer, Human3DKinematicLayerV2
from mmpose.models.utils.human_kinematic_utils import HumanKinematic
from mmpose.apis   import  vis_3d_pose_result, init_pose_model, inference_pose_lifter_model, single_gpu_test
from mmpose.datasets.pipelines.kinematic_transform import KinematicBoneLength, KinematicDirectionVector
import json
import pickle as pkl
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="../configs/body/3d_kpt_sview_rgb_vid/video_kinematic_pose_lift/h36m/kinematic3d_params_optimz_h36m_27frames_supervised.py")
    parser.add_argument('--checkpoint', 
        default="../tools/work_dirs/videopose3d_h36m_27frames_fullconv_supervised/latest.pth")
    return parser.parse_args()

def get_dataloader(cfg, dataset_cfg):
    # step 0: build dataset
    dataset = build_dataset(dataset_cfg, dict(test_mode=True))
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(seed=cfg.get('seed'), drop_last=False, dist=False),
        **({} if T.__version__ != 'parrots' else dict(
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


METAS_FILE="h36m_test_set_metas.pkl"
H36M_TEST_SET_KINEMATIC_FILE="h36m_test_set_kinematic.npz"
def preprocess():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = parse_args()
    cfg  = Config.fromfile(args.config)
    data_cfg  = cfg.data.test #select dataset config
    bone_length = []
    dirc_vector = []
    metas = []
    pose_lift_dataloader = get_dataloader(cfg, data_cfg)
    # dataset_info = DatasetInfo(data_cfg.dataset_info)
    for i, data in tqdm(enumerate(pose_lift_dataloader)):
        bone_length.append(data['bone_length'])
        dirc_vector.append(data['dirc_vector'])
        metas.extend(data['metas'].data[0])
    bone_length = T.cat(bone_length).cpu().numpy()
    dirc_vector = T.cat(dirc_vector).cpu().numpy()
    # save to npz for future processing
    np.savez(H36M_TEST_SET_KINEMATIC_FILE, bone_length = bone_length, dirc_vector = dirc_vector)
    with open(METAS_FILE, "wb") as fo:
        pkl.dump(metas, fo)

PROCESSED_FILE = "h36m_kinematic_processed.npz"
def process_data():
    with np.load(H36M_TEST_SET_KINEMATIC_FILE) as data:
        bone_length = data['bone_length']
        dirc_vector = data['dirc_vector']
    with open(METAS_FILE, "rb") as fi:
        metas = pkl.load(fi)

    bone_length_by_subject = {}
    euler_angles = HumanKinematic.compute_euler_angle_from_vector(dirc_vector)
    np.savez(PROCESSED_FILE, euler_angles=euler_angles)
    print("saved!")
    for i in tqdm(range(len(metas))):
        subj, action = osp.basename(metas[i]['target_image_path']).split('_')[:2]
        if subj not in bone_length_by_subject:
            bone_length_by_subject[subj] = []
        bone_length_by_subject[subj].append(bone_length[i,:][None,:])
    
    subject_kinematic_info = {} 
    for subj in bone_length_by_subject:
        bone_length_by_subject[subj] = np.concatenate(bone_length_by_subject[subj])
        if subj not in subject_kinematic_info:
            subject_kinematic_info[subj] = {}
        subject_kinematic_info[subj]['bone_length_mean'] = bone_length_by_subject[subj].mean(axis=0)
        subject_kinematic_info[subj]['bone_length_std'] = bone_length_by_subject[subj].std(axis=0)

JSON_TO_NPY_CACHE_FILE="test_kpts_pred.np"
TEST_KINEMATIC_CACHE_FILE="test_preds_kinematic.npz"
def analyze_test_result():
    kpts_file = "../tools/work_dirs/kinematic3d_params_optimz_h36m_27frames_supervised/result_keypoints.json"
    test_kpts_pred = []
    with open(kpts_file, "r") as fi:
        data = json.load(fi)
        for i in tqdm(range(len(data))):
            test_kpts_pred.append(np.array(data[i]['keypoints'])[None, :])
    test_kpts_pred = np.concatenate(test_kpts_pred)
    with open(JSON_TO_NPY_CACHE_FILE, "wb") as fo:
        np.save(fo, test_kpts_pred)

def compute_kinematic_params():
    kpts = np.load(JSON_TO_NPY_CACHE_FILE)
    layout = HumanKinematic.get_layout(dataset="h36m", mode="3d")
    compute_dirc_vector = KinematicDirectionVector(layout, source="3d_kpts",target="dirc_vector")
    compute_bone_length = KinematicBoneLength(layout, source="3d_kpts", target="bone_length")
    dirc_vector = []
    bone_length = []
    for i in tqdm(range(len(kpts))):
        r = compute_dirc_vector({"3d_kpts": kpts[i]})
        r = compute_bone_length(r)
        dirc_vector.append(r['dirc_vector'][None, ...])
        bone_length.append(r['bone_length'][None, ...])
    dirc_vector = np.concatenate(dirc_vector)
    bone_length = np.concatenate(bone_length)
    np.savez(TEST_KINEMATIC_CACHE_FILE, bone_length = bone_length, dirc_vector = dirc_vector)

def _load_data(file):
    with np.load(file) as data:
        bone_length = data['bone_length']
        dirc_vector = data['dirc_vector']
    return bone_length, dirc_vector
def visualize_data():
    bone_length_1, series_1 = _load_data(H36M_TEST_SET_KINEMATIC_FILE)
    bone_length_2, series_2 = _load_data(TEST_KINEMATIC_CACHE_FILE)
    # fig = plt.figure()
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(30,30), subplot_kw=dict(projection='3d'))
    # ax = fig.add_subplot(111, projection='3d')
    
    skip = 300
    for i in range(4):
        for j in range(4):
            joint_idx = i*4+j
            ax = axs[i,j]
            # plot series_1
            kpts3d = series_1[::skip,joint_idx,:]
            ax.scatter(kpts3d[:,0], kpts3d[:,1], kpts3d[:,2], alpha=0.5)
            # plot series_2
            kpts3d = series_2[::skip,joint_idx,:]
            ax.scatter(kpts3d[:,0], kpts3d[:,1], kpts3d[:,2])
            ax.set_title("Joint %d" % (joint_idx + 1))
    plt.savefig("3d_plot.png", bbox_inches='tight')


def group_bone_length_by_subject():
    with open(METAS_FILE, "rb") as fi:
        metas = pkl.load(fi)
    subj_indcs = {}
    for i in tqdm(range(len(metas))):
        subj, action = osp.basename(metas[i]['target_image_path']).split('_')[:2]
        if subj not in subj_indcs:
            subj_indcs[subj] = []
        subj_indcs[subj].append(i)
    for subj in subj_indcs:
        subj_indcs[subj] = np.array(subj_indcs[subj])
    return subj_indcs

def visualize_bone_length():
    bone_length_1, series_1 = _load_data(H36M_TEST_SET_KINEMATIC_FILE)
    bone_length_2, series_2 = _load_data(TEST_KINEMATIC_CACHE_FILE)
    subj_idcs = group_bone_length_by_subject()
    subject_id= "S11"
    idcs = subj_idcs[subject_id]
    gt_mean   = bone_length_1[idcs].mean(axis=0)[:,0]
    gt_std    = bone_length_1[idcs].std(axis=0)[:,0]
    preds_mean   = bone_length_2[idcs].mean(axis=0)[:,0]
    preds_std    = bone_length_2[idcs].std(axis=0)[:,0]
    labels = ['B%d' % i for i in range(len(gt_mean))]
    x = np.arange(len(gt_mean))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(15, 7))
    rects1 = ax.bar(x - width/2, gt_mean, width, label='gt')
    rects2 = ax.bar(x + width/2, preds_mean, width, label='preds')
    ax.errorbar(x + width/2, preds_mean, yerr=preds_std, ls='none', color ='black')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Length')
    ax.set_title('Bones idx')
    ax.set_xticks(x, labels)
    ax.legend()
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    plt.savefig("bone_length.png", bbox_inches='tight')

def compute_mpjpe():
    from mmpose.core.evaluation import keypoint_mpjpe
    args = parse_args()
    cfg  = Config.fromfile(args.config)
    dataset_cfg  = cfg.data.test
    dataset = build_dataset(dataset_cfg, dict(test_mode=True))
    gts = dataset.data_info['joints_3d']
    preds = np.load(JSON_TO_NPY_CACHE_FILE)
    import ipdb; ipdb.set_trace()
    error = keypoint_mpjpe(preds, gts[...,:3], gts[...,3])
    print("mpjpe: ", error)
    import ipdb; ipdb.set_trace()
if __name__ == "__main__":
    # preprocess()
    # process_data()
    # analyze_test_result()
    # compute_kinematic_params()
    # visualize_bone_length()
    # visualize_data()

    compute_mpjpe()
    