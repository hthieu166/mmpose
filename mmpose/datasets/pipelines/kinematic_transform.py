from mmpose.datasets.builder import PIPELINES
from mmpose.models.utils.human_kinematic_utils import HumanKinematic
from .shared_transform import Compose
import numpy as np
import abc

class KinematicParams(abc.ABC):
    def __init__(self, layout, source, target):
        self.target = target
        self.layout = layout
        self.source = source
    
    def __call__(self, results):
        raise NotImplementedError

@PIPELINES.register_module()
class KinematicEulerAngle(KinematicParams):
    def __init__(self, layout, source, target):
        super().__init__(layout, source, target)
        
    def __call__(self, results):
        results[self.target] = HumanKinematic.compute_euler_angle_from_joints_position(
            results[self.source], self.layout)
        return results

@PIPELINES.register_module()
class KinematicBoneLength(KinematicParams):
    def __init__(self, layout, source, target):
        super().__init__(layout, source, target)

    def __call__(self, results):
        results[self.target] = HumanKinematic.compute_bone_length(
            results[self.source], self.layout)
        return results

@PIPELINES.register_module()
class KinematicDirectionVector(KinematicParams):
    def __init__(self, layout, source, target):
        super().__init__(layout, source, target) 
        self.j0 = []
        self.j1 = []
        for b in self.layout['bones']:
            self.j0.append(b[0])
            self.j1.append(b[1])

    def __call__(self, results):
        keypoints = results[self.source]
        assert len(keypoints.shape) == 2
        vect = keypoints[self.j1,:] - keypoints[self.j0,:]
        vect = HumanKinematic.normalized_vector(vect)
        results[self.target] = vect
        return results

@PIPELINES.register_module()
class GenSkeKinematicFeat:
    def __init__(self, dataset='h36m', mode='3d', feats=['euler_angle']):
        self.dataset = dataset
        self.feats = feats
        assert mode in ['3d']
        if mode == '3d':
            source_key = 'target'
        
        # Get kinematic layout:
        layout = HumanKinematic.get_layout(dataset, mode)
        
        ops = []
        if 'bone_length' in feats:
            ops.append(KinematicBoneLength(
                layout=layout, 
                source=source_key, 
                target='bone_length'))
        
        if 'euler_angle' in feats:
            ops.append(KinematicEulerAngle(
                layout=layout, 
                source=source_key, 
                target='euler_angle'))

        if 'dirc_vector' in feats:
            ops.append(KinematicDirectionVector(
                layout= layout,
                source= source_key,
                target='dirc_vector'
            ))
        # ops.append(MergeSkeFeat(feat_list=feats, axis=axis, target='kinematic'))
        self.ops = Compose(ops)
    
    def __call__(self, results):
        return self.ops(results)