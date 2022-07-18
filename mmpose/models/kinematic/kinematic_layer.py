from turtle import forward
import torch as T
from torch import nn
from pytorch3d import transforms
from mmpose.models.builder import HEADS, KINEMATIC_LAYER, build_kinematic_layer

class KinematicLayer3D(nn.Module):
    def __init__(self, layout, init_pos = None):
        super().__init__()
        self.layout   = layout
        self.init_pos = init_pos if init_pos is not None else T.zeros(3)
        self.n_joints = layout['n_joints']
        self.n_angles = len(layout['angles'])
        self.n_bones  = len(layout['bones'])
        self.j_dim    = 3
    def forward(self, x):
        raise NotImplementedError

@HEADS.register_module()
class Human3DKinematicLayer(KinematicLayer3D):    
    def __init__(self, layout, init_pos = None):
        super(Human3DKinematicLayer, self).__init__(layout, init_pos)
        
    def forward(self, euler_angle: T.Tensor, bone_length: T.Tensor, **kwargs):
        assert euler_angle.shape[0] == bone_length.shape[0]
        assert euler_angle.shape[1] == self.n_angles
        assert bone_length.shape[1] == self.n_bones

        device = euler_angle.device
        N = euler_angle.shape[0] #batch size
        est_pose  = T.zeros(N, self.n_joints, self.j_dim).to(device)
        rot_trsf    = {}
        rot_trsf[self.layout['root_joint']] = transforms.Rotate(T.eye(3).repeat(N, 1, 1), device=device)
        rot_mat     = transforms.euler_angles_to_matrix(euler_angle, "XYZ")

        for i, (b0,b1) in enumerate(self.layout['angles']):
            j0, j1 = self.layout['bones'][b1]
            rot_trsf[j1]       = rot_trsf[j0].rotate(rot_mat[:,i])
            est_pose[:, j1, 2] = bone_length[:,b1, 0] #set the z-axis equal to the bone length 
            j1_est       = rot_trsf[j1].transform_points(est_pose[:, j1]) + est_pose[:, j0]
            est_pose[:, j1] = j1_est[:,0]   
        return est_pose

@HEADS.register_module()
class Human3DKinematicLayerV2(KinematicLayer3D):
    def __init__(self, layout, init_pos = None):
        super().__init__(layout, init_pos)
    
    def forward(self, dirc_vector: T.Tensor, bone_length: T.Tensor):
        N = dirc_vector.shape[0] #batch size
        est_pose = T.zeros(N, self.n_joints, self.j_dim)
        est_pose[:,1:] = dirc_vector * bone_length
        for b in self.layout['bones']:
            est_pose[:,b[1]] = est_pose[:,b[1]] + est_pose[:,b[0]]
        return est_pose