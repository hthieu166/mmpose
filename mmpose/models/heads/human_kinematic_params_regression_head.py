# Implemented by Hieu Hoang.
import numpy as np
import torch.nn as nn
import torch as T
from mmcv.cnn import build_conv_layer, constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmpose.core import (WeightNormClipHook, compute_similarity_transform,
                         fliplr_regression)

from mmpose.models.utils.human_kinematic_utils import HumanKinematic
from mmpose.models.kinematic.kinematic_layer import Human3DKinematicLayer
from mmpose.models.builder import HEADS, build_kinematic_layer, build_loss


@HEADS.register_module()
class HumanKinematicParamsRegressionHead(nn.Module):
    """Regression head using human kinematic layer

    Args:
        in_channels (int): Number of input channels
        num_joints (int): Number of joints
        loss_kinematic_params (dict): Config for kinematic params loss. Default: None.
        max_norm (float|None): if not None, the weight of convolution layers
            will be clipped to have a maximum norm of max_norm.
        is_trajectory (bool): If the model only predicts root joint
            position, then this arg should be set to True. In this case,
            traj_loss will be calculated. Otherwise, it should be set to
            False. Default: False.
    """

    def __init__(self,
                 in_channels,
                 dataset,
                 mode,
                 kinematic_layer = None,
                 max_norm=None,
                 loss_kinematic_params=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.layout = HumanKinematic.get_layout(dataset, mode)
        self.in_channels = in_channels
        self.num_joints  = self.layout['n_joints']
        self.num_angles  = len(self.layout['angles'])
        self.num_bones   = len(self.layout['bones'])
        
        self.max_norm = max_norm
        self.loss = build_loss(loss_kinematic_params)
        
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        
        # Predicting direction vector for the human kinematic model
        self.conv = build_conv_layer(
            dict(type='Conv1d'), in_channels, self.num_bones * 3, 1)
        
        # Initialize kinematic layer
        if kinematic_layer is not None:
            kinematic_layer['layout'] = self.layout
            self.kinematic_layer = build_kinematic_layer(kinematic_layer)
        
        if self.max_norm is not None:
            # Apply weight norm clip to conv layers
            weight_clip = WeightNormClipHook(self.max_norm)
            for module in self.modules():
                if isinstance(module, nn.modules.conv._ConvNd):
                    weight_clip.register(module)

    @property
    def with_kinematic_layer(self):
        """Check if this layer has a kinematic layer."""
        return hasattr(self, 'kinematic_layer')

    @staticmethod
    def _transform_inputs(x):
        """Transform inputs for decoder.

        Args:
            inputs (tuple or list of Tensor | Tensor): multi-level features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(x, (list, tuple)):
            return x

        assert len(x) > 0

        # return the top-level feature of the 1D feature pyramid
        return x[-1]

    def infer_kinematic_layer(self, **kinematic_params):
        assert self.kinematic_layer is not None
        return  self.kinematic_layer(**kinematic_params)

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)

        assert x.ndim == 3 and x.shape[2] == 1, f'Invalid shape {x.shape}'
        
        # Kinematic parameters regression
        output = self.conv(x)
        N = output.shape[0]
        output = output.reshape(N, self.num_bones, 3)
        # Normalizing to unit-vector
        output = output/T.linalg.norm(output, dim=-1)[...,None]
        return output

    def get_loss(self, output, dirc_vector):
        """Calculate keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 3]): Output keypoints.
            target (torch.Tensor[N, K, 3]): Target keypoints.
            target_weight (torch.Tensor[N, K, 3]):
                Weights across different joint types.
                If self.is_trajectory is True and target_weight is None,
                target_weight will be set inversely proportional to joint
                depth.
        """
        losses = dict()
        assert not isinstance(self.loss, nn.Sequential)
        losses['reg_loss'] = self.loss(output, dirc_vector)
        return losses

    def get_accuracy(self, kinematic_params, target, target_weight, metas):
        """Calculate accuracy for keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 3]): Output keypoints.
            target (torch.Tensor[N, K, 3]): Target keypoints.
            target_weight (torch.Tensor[N, K, 3]):
                Weights across different joint types.
            metas (list(dict)): Information about data augmentation including:

                - target_image_path (str): Optional, path to the image file
                - target_mean (float): Optional, normalization parameter of
                    the target pose.
                - target_std (float): Optional, normalization parameter of the
                    target pose.
                - root_position (np.ndarray[3,1]): Optional, global
                    position of the root joint.
                - root_index (torch.ndarray[1,]): Optional, original index of
                    the root joint before root-centering.
        """
        assert self.with_kinematic_layer
        output = self.infer_kinematic_layer(**kinematic_params)
        accuracy = dict()

        N = output.shape[0]
        output_ = output.detach().cpu().numpy()
        target_ = target.detach().cpu().numpy()
        # Denormalize the predicted pose
        if 'target_mean' in metas[0] and 'target_std' in metas[0]:
            target_mean = np.stack([m['target_mean'] for m in metas])
            target_std = np.stack([m['target_std'] for m in metas])
            output_ = self._denormalize_joints(output_, target_mean,
                                               target_std)
            target_ = self._denormalize_joints(target_, target_mean,
                                               target_std)

        # Restore global position
        if self.test_cfg.get('restore_global_position', False):
            root_pos = np.stack([m['root_position'] for m in metas])
            root_idx = metas[0].get('root_position_index', None)
            output_ = self._restore_global_position(output_, root_pos,
                                                    root_idx)
            target_ = self._restore_global_position(target_, root_pos,
                                                    root_idx)
        # Get target weight
        if target_weight is None:
            target_weight_ = np.ones_like(target_)
        else:
            target_weight_ = target_weight.detach().cpu().numpy()
            if self.test_cfg.get('restore_global_position', False):
                root_idx = metas[0].get('root_position_index', None)
                root_weight = metas[0].get('root_joint_weight', 1.0)
                target_weight_ = self._restore_root_target_weight(
                    target_weight_, root_weight, root_idx)

        mpjpe = np.mean(
            np.linalg.norm((output_ - target_) * target_weight_, axis=-1))

        transformed_output = np.zeros_like(output_)
        for i in range(N):
            transformed_output[i, :, :] = compute_similarity_transform(
                output_[i, :, :], target_[i, :, :])
        p_mpjpe = np.mean(
            np.linalg.norm(
                (transformed_output - target_) * target_weight_, axis=-1))

        accuracy['mpjpe'] = output.new_tensor(mpjpe)
        accuracy['p_mpjpe'] = output.new_tensor(p_mpjpe)

        return accuracy

    def inference_model(self, x):
        """Inference function.

        Returns:
            output_regression (np.ndarray): Output regression.

        Args:
            x (torch.Tensor[N, K, 2]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)
        output_regression = output.detach().cpu().numpy()
        return output_regression

    def decode(self, metas, output):
        """Decode the keypoints from output regression.

        Args:
            metas (list(dict)): Information about data augmentation.
                By default this includes:

                - "target_image_path": path to the image file
            output (np.ndarray[N, K, 3]): predicted regression vector.
            metas (list(dict)): Information about data augmentation including:

                - target_image_path (str): Optional, path to the image file
                - target_mean (float): Optional, normalization parameter of
                    the target pose.
                - target_std (float): Optional, normalization parameter of the
                    target pose.
                - root_position (np.ndarray[3,1]): Optional, global
                    position of the root joint.
                - root_index (torch.ndarray[1,]): Optional, original index of
                    the root joint before root-centering.
        """

        # Denormalize the predicted pose
        if 'target_mean' in metas[0] and 'target_std' in metas[0]:
            target_mean = np.stack([m['target_mean'] for m in metas])
            target_std = np.stack([m['target_std'] for m in metas])
            output = self._denormalize_joints(output, target_mean, target_std)

        # Restore global position
        if self.test_cfg.get('restore_global_position', False):
            root_pos = np.stack([m['root_position'] for m in metas])
            root_idx = metas[0].get('root_position_index', None)
            output = self._restore_global_position(output, root_pos, root_idx)

        target_image_paths = [m.get('target_image_path', None) for m in metas]
        result = {'preds': output, 'target_image_paths': target_image_paths}

        return result

    @staticmethod
    def _denormalize_joints(x, mean, std):
        """Denormalize joint coordinates with given statistics mean and std.

        Args:
            x (np.ndarray[N, K, 3]): Normalized joint coordinates.
            mean (np.ndarray[K, 3]): Mean value.
            std (np.ndarray[K, 3]): Std value.
        """
        assert x.ndim == 3
        assert x.shape == mean.shape == std.shape

        return x * std + mean

    @staticmethod
    def _restore_global_position(x, root_pos, root_idx=None):
        """Restore global position of the root-centered joints.

        Args:
            x (np.ndarray[N, K, 3]): root-centered joint coordinates
            root_pos (np.ndarray[N,1,3]): The global position of the
                root joint.
            root_idx (int|None): If not none, the root joint will be inserted
                back to the pose at the given index.
        """
        x = x + root_pos
        if root_idx is not None:
            x = np.insert(x, root_idx, root_pos.squeeze(1), axis=1)
        return x

    @staticmethod
    def _restore_root_target_weight(target_weight, root_weight, root_idx=None):
        """Restore the target weight of the root joint after the restoration of
        the global position.

        Args:
            target_weight (np.ndarray[N, K, 1]): Target weight of relativized
                joints.
            root_weight (float): The target weight value of the root joint.
            root_idx (int|None): If not none, the root joint weight will be
                inserted back to the target weight at the given index.
        """
        if root_idx is not None:
            root_weight = np.full(
                target_weight.shape[0], root_weight, dtype=target_weight.dtype)
            target_weight = np.insert(
                target_weight, root_idx, root_weight[:, None], axis=1)
        return target_weight

    def init_weights(self):
        """Initialize the weights."""
        for m in self.modules():
            if isinstance(m, nn.modules.conv._ConvNd):
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)
