import torch as T
from torch import nn
from pytorch3d import transforms

class Human3DKinematicLayer(nn.Module):    
    def __init__(self, layout, init_pos = None):
        super(Human3DKinematicLayer, self).__init__()
        self.layout   = layout
        self.init_pos = init_pos if init_pos is not None else T.zeros(3)
        self.n_joints = layout['n_joints']
        self.n_angles = len(layout['angles'])
        self.n_bones  = len(layout['bones'])
        self.j_dim    = 3

    def __translate_z(self, bone_len: T.Tensor):
        N = len(bone_len)
        translate_by     = T.zeros(N, 3)
        translate_by[:,2]= bone_len[:,0]
        return translate_by
    
    def forward(self, euler_angle: T.Tensor, bone_length: T.Tensor, **kwargs):
        assert euler_angle.shape[0] == bone_length.shape[0]
        assert euler_angle.shape[1] == self.n_angles
        assert bone_length.shape[1] == self.n_bones

        N = euler_angle.shape[0] #batch size
        est_pose  = T.zeros(N, self.n_joints, self.j_dim).to(euler_angle.device)

        rot_trsf    = {}
        rot_trsf[self.layout['root_joint']] = transforms.Rotate(T.eye(3).repeat(N, 1, 1))
        
        for i, (b0,b1) in enumerate(self.layout['angles']):
            j0, j1 = self.layout['bones'][b1]
            rot_trsf[j1] = rot_trsf[j0].rotate(transforms.euler_angles_to_matrix(euler_angle[:,i], "XYZ"))
            j1_est       = rot_trsf[j1].transform_points(self.__translate_z(bone_length[:,b1])).to(euler_angle.device) + est_pose[:, j0]
            est_pose[:, j1] = j1_est[:,0]   
        return est_pose
