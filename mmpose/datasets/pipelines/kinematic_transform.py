from mmpose.datasets.builder import PIPELINES
from mmpose.models.utils.human_kinematic_utils import HumanKinematic
from .shared_transform import Compose
@PIPELINES.register_module()
class KinematicAngle3Axis:
    def __init__(self, layout, source, target):
        self.target = target
        self.source = source
        self.layout = layout
    def __call__(self, results):
        results[self.target] = HumanKinematic.compute_euler_angle_from_joints_position(
            results[self.source], self.layout)
        return results

@PIPELINES.register_module()
class KinematicAngleBoneLength:
    def __init__(self, layout, source, target):
        self.target = target
        self.source = source
        self.layout = layout
    def __call__(self, results):
        results[self.target] = HumanKinematic.compute_bone_length(
            results[self.source], self.layout)
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
            ops.append(KinematicAngleBoneLength(
                layout=layout, 
                source=source_key, 
                target='bone_length'))
        
        if 'euler_angle' in feats:
            ops.append(KinematicAngle3Axis(
                layout=layout, 
                source=source_key, 
                target='euler_angle'))

        # ops.append(MergeSkeFeat(feat_list=feats, axis=axis, target='kinematic'))
        self.ops = Compose(ops)
    
    def __call__(self, results):
        return self.ops(results)