# Copyright (c) OpenMMLab. All rights reserved.
import imp
import warnings

import mmcv
import numpy as np
from mmcv.utils.misc import deprecated_api_warning

from mmpose.core import imshow_bboxes, imshow_keypoints, imshow_keypoints_3d
from .. import builder
from ..builder import POSENETS
from .base import BasePose

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class KinematicModelPredictor(BasePose):
    """Predicting Human Body Kinematic Parameters.

    The basic model is a pose model that predicts root-relative pose. If
    traj_head is not None, a trajectory model that predicts absolute root joint
    position is also built.

    Args:
        backbone (dict): Config for the backbone of pose model.
        neck (dict|None): Config for the neck of pose model.
        keypoint_head (dict|None): Config for the head of pose model.
        traj_backbone (dict|None): Config for the backbone of trajectory model.
            If traj_backbone is None and traj_head is not None, trajectory
            model will share backbone with pose model.
        traj_neck (dict|None): Config for the neck of trajectory model.
        traj_head (dict|None): Config for the head of trajectory model.
        loss_semi (dict|None): Config for semi-supervision loss.
        train_cfg (dict|None): Config for keypoint head during training.
        test_cfg (dict|None): Config for keypoint head during testing.
        pretrained (str|None): Path to pretrained weights.
    """

    def __init__(self,
                 backbone,
                 params_pred_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        self.fp16_enabled = False

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # pose model
        self.backbone = builder.build_backbone(backbone)

        # params prediction head
        self.params_pred_head = builder.build_head(params_pred_head)
        self.pretrained = pretrained
        self.init_weights()

    
    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        if pretrained is not None:
            self.pretrained = pretrained
        # Init backbone
        self.backbone.init_weights(self.pretrained)
        # Init params prediction head
        self.params_pred_head.init_weights()
        
    @auto_fp16(apply_to=('input', ))
    def forward(self,
                input,
                target=None,
                dirc_vector = None,
                target_weight=None,
                metas=None,
                return_loss=True,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note:
            - batch_size: N
            - num_input_keypoints: Ki
            - input_keypoint_dim: Ci
            - input_sequence_len: Ti
            - num_output_keypoints: Ko
            - output_keypoint_dim: Co
            - input_sequence_len: To

        Args:
            input (torch.Tensor[NxKixCixTi]): Input keypoint coordinates.
            target (torch.Tensor[NxKoxCoxTo]): Output keypoint coordinates.
                Defaults to None.
            target_weight (torch.Tensor[NxKox1]): Weights across different
                joint types. Defaults to None.
            metas (list(dict)): Information about data augmentation
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.

        Returns:
            dict|Tensor: If `reutrn_loss` is true, return losses. \
                Otherwise return predicted poses.
        """
        if return_loss:
            return self.forward_train(input, dirc_vector, target, target_weight, metas, **kwargs)
        else:
            return self.forward_test(input, dirc_vector, metas, target,  **kwargs)
    
    def _compose_kinematic_params(self, output, kwargs):
        assert 'bone_length' in kwargs
        return {
            "bone_length": kwargs['bone_length'],
            "dirc_vector": output
        }
    def forward_train(self, input, dirc_vector, target, target_weight, metas, **kwargs):
        """Defines the computation performed at every call when training."""
        assert input.size(0) == len(metas)
        # supervised learning
        # pose model
        features = self.backbone(input)
        # prediction head
        output = self.params_pred_head(features)
        # compute loss
        losses = dict()
        # Update kinematic model's parameters loss
        params_losses = self.params_pred_head.get_loss(output, dirc_vector)
        losses.update(params_losses)
        # Compute keypoint accuracy
        if self.params_pred_head.with_kinematic_layer:
            kinematic_params = self._compose_kinematic_params(output, kwargs)
            keypoint_accuracy = self.params_pred_head.get_accuracy(
                    kinematic_params, target, target_weight, metas)
            losses.update(keypoint_accuracy)
        return losses

    def forward_test(self, input, dirc_vector, metas, target, **kwargs):
        """Defines the computation performed at every call when training."""
        if type(metas) != list:
            metas = metas.data[0]
        assert input.size(0) == len(metas)

        results  = {}
        features = self.backbone(input)
        output   = self.params_pred_head(features)
        results.update({
            "preds_dirc_vector" : output,
            "gt_dirc_vector": dirc_vector
        })
        # If the kinematic layer is available:
        if self.params_pred_head.with_kinematic_layer:
            kinematic_params = self._compose_kinematic_params(output, kwargs)
            # Compute keypoints from kinematic params
            output = self.params_pred_head.infer_kinematic_layer(**kinematic_params)
            output = output.detach().cpu().numpy()
            keypoint_result = self.params_pred_head.decode(metas, output)
            # Update results
            results.update(keypoint_result)

        import ipdb; ipdb.set_trace()
        return results

    def forward_dummy(self, input):
        """Used for computing network FLOPs. See ``tools/get_flops.py``.

        Args:
            input (torch.Tensor): Input pose

        Returns:
            Tensor: Model output
        """
        raise NotImplementedError

    @deprecated_api_warning({'pose_limb_color': 'pose_link_color'},
                            cls_name='PoseLifter')
    def show_result(self,
                    result,
                    img=None,
                    skeleton=None,
                    pose_kpt_color=None,
                    pose_link_color=None,
                    radius=8,
                    thickness=2,
                    vis_height=400,
                    num_instances=-1,
                    axis_azimuth=70,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Visualize 3D pose estimation results.

        Args:
            result (list[dict]): The pose estimation results containing:

                - "keypoints_3d" ([K,4]): 3D keypoints
                - "keypoints" ([K,3] or [T,K,3]): Optional for visualizing
                    2D inputs. If a sequence is given, only the last frame
                    will be used for visualization
                - "bbox" ([4,] or [T,4]): Optional for visualizing 2D inputs
                - "title" (str): title for the subplot
            img (str or Tensor): Optional. The image to visualize 2D inputs on.
            skeleton (list of [idx_i,idx_j]): Skeleton described by a list of
                links, each is a pair of joint indices.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            vis_height (int): The image height of the visualization. The width
                will be N*vis_height depending on the number of visualized
                items.
            num_instances (int): Number of instances to be shown in 3D. If
                smaller than 0, all the instances in the result will be shown.
                Otherwise, pad or truncate the result to a length of
                num_instances.
            axis_azimuth (float): axis azimuth angle for 3D visualizations.
            win_name (str): The window name.
            show (bool): Whether to directly show the visualization.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """
        if num_instances < 0:
            assert len(result) > 0
        result = sorted(result, key=lambda x: x.get('track_id', 1e4))

        # draw image and input 2d poses
        if img is not None:
            img = mmcv.imread(img)

            bbox_result = []
            pose_input_2d = []
            for res in result:
                if 'bbox' in res:
                    bbox = np.array(res['bbox'])
                    if bbox.ndim != 1:
                        assert bbox.ndim == 2
                        bbox = bbox[-1]  # Get bbox from the last frame
                    bbox_result.append(bbox)
                if 'keypoints' in res:
                    kpts = np.array(res['keypoints'])
                    if kpts.ndim != 2:
                        assert kpts.ndim == 3
                        kpts = kpts[-1]  # Get 2D keypoints from the last frame
                    pose_input_2d.append(kpts)

            if len(bbox_result) > 0:
                bboxes = np.vstack(bbox_result)
                imshow_bboxes(
                    img,
                    bboxes,
                    colors='green',
                    thickness=thickness,
                    show=False)
            if len(pose_input_2d) > 0:
                imshow_keypoints(
                    img,
                    pose_input_2d,
                    skeleton,
                    kpt_score_thr=0.3,
                    pose_kpt_color=pose_kpt_color,
                    pose_link_color=pose_link_color,
                    radius=radius,
                    thickness=thickness)
            img = mmcv.imrescale(img, scale=vis_height / img.shape[0])

        img_vis = imshow_keypoints_3d(
            result,
            img,
            skeleton,
            pose_kpt_color,
            pose_link_color,
            vis_height,
            num_instances=num_instances,
            axis_azimuth=axis_azimuth,
        )

        if show:
            mmcv.visualization.imshow(img_vis, win_name, wait_time)

        if out_file is not None:
            mmcv.imwrite(img_vis, out_file)

        return img_vis
