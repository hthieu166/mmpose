import numpy as np
from scipy.spatial.transform import Rotation

class HumanKinematic():
    UP_VECTOR_3D = np.array([[0.0, 0.0, 1.0]])
    UP_VECTOR_2D = np.array([[0.0, 1.0]])
    JOINTS_3D_KINEMATIC_NTU = dict(
        n_joints = 25,
        bones = [
            (1,0), 
            (0,12), (12,13), (13,14), (14,15),
            (0,16), (16,17), (17,18), (18,19),
            (1,20), (20, 4), (4,5), (5,6), (6,7), (7,22), (22,21),
            (20,8), (8,9), (9,10), (10,11), (11,24), (24,23),
            (20,2), (2,3)
            ],
        angles = [
            ('up', 0), # 0
            (0,1),  (1,2),  (2,3), (3,4), # left leg
            (0,5),  (5,6),  (6,7), (7,8), # right leg
            ('up', 9), #9
            (9,10), (10,11), (11,12), (12,13), (13,14), (14,15),
            (9,16), (16,17),(17,18), (18,19), (19,20), (20,21),
            (9,22), (22,23)
        ],
        root_joint = 1
    )

    JOINTS_3D_KINEMATIC_H36M = dict(
        n_joints = 17,
        bones = [
            (0,1), (1,2), (2,3),
            (0,4), (4,5), (5,6),
            (0,7), (7,8), (8,9), (9,10),
            (8,11), (11,12), (12,13),
            (8,14), (14,15), (15,16)
        ],
        angles = [
            ('up', 0), (0,1), (1,2),
            ('up', 3), (3,4), (4,5),
            ('up', 6), (6,7), (7,8), (8,9),
            (7,10), (10,11), (11,12),
            (7,13), (13,14), (14,15)
        ],
        root_joint = 0
    )

    def rotation_matrix(axis, theta):
        """ Codes adapted from https://github.com/lshiwjx/2s-AGCN
        Return the rotation matrix associated with counterclockwise rotation
        about the given axis by theta radians."""
        N = theta.shape[0]
        rot_mat = np.zeros((N, 3, 3))

        axis = axis / (np.linalg.norm(axis, axis=1) + 1e-6) [:, None]
        
        a = np.cos(theta / 2.0)
        v = -axis * np.sin(theta / 2.0)[:, None]
        b, c, d = v[:,0], v[:,1], v[:,2]
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        
        tmp = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
        rot_mat = tmp.transpose(2,0,1)
        rot_mat[
            np.where(np.logical_or(
                np.abs(axis).sum(axis=1) < 1e-6,
                np.abs(theta) < 1e-6
            ))[0]
        ] = np.eye(3)
        return rot_mat

    def vector_angle(uo, vo):
        """Return the angle between two vectors using the dot product"""
        if len(uo.shape) == 2:
            u  = uo/(np.linalg.norm(uo, axis=1) + 1e-6)[:,None]
            v  = vo/(np.linalg.norm(vo, axis=1) + 1e-6)[:,None]
            dp = np.einsum('ij,ij->i',  u, v)
        else:
            u  = uo/(np.linalg.norm(uo) + 1e-6)
            v  = vo/(np.linalg.norm(vo) + 1e-6)
            dp = np.dot(u,v)
        ag = np.arccos(np.clip(dp, -1.0, 1.0))
        return ag

    def get_vector(keypoint, layout, p):
        assert len(keypoint.shape) == 3
        N, nj, d = keypoint.shape
        if   d == 3 and p == 'up':
            return HumanKinematic.UP_VECTOR_3D.repeat(N, axis=0)
        elif d == 2 and p == 'up':
            return HumanKinematic.UP_VECTOR_2D.repeat(N, axis=0)
        else:
            v = layout['bones'][p]
            return keypoint[:, v[1]] - keypoint[:, v[0]]
        
    @staticmethod
    def get_layout(dataset: str, mode :str):
        dataset, mode = dataset.lower(), mode.lower()
        assert dataset in ['ntu', 'h36m']
        assert mode in ['3d', '2d']
        if dataset == 'ntu':
            if mode == '3d':
                return HumanKinematic.JOINTS_3D_KINEMATIC_NTU
            else:
                return HumanKinematic.JOINTS_2D_10_ANGLES_COCO
        elif dataset == 'h36m':
            if mode == '3d':
                return HumanKinematic.JOINTS_3D_KINEMATIC_H36M
        raise Exception("Does not support {mode} layout for dataset {dataset}")

    @staticmethod
    def compute_bone_length(keypoint, layout):
        nj, nd = keypoint.shape
        assert 'bones' in layout

        bone_length= np.zeros((len(layout['bones']), 1))
        for i, (j0, j1) in enumerate(layout['bones']):
            bone_length[i] = np.linalg.norm(keypoint[j1,:] - keypoint[j0,:], axis=0)
        return bone_length

    @staticmethod
    def compute_rotation_matrix_from_joints_position(keypoint, layout):
        keypoint = keypoint[None, ...]
        nb, nj, nd = keypoint.shape
        keypoint = keypoint.reshape(nb, nj, nd)
        rot_mat  = np.zeros((nb, len(layout['angles']), nd, nd))
        for i, (p1,p2) in enumerate(layout['angles']):
            v1 = HumanKinematic.get_vector(keypoint, layout, p1) 
            v2 = HumanKinematic.get_vector(keypoint, layout, p2)
            rot_axis = np.cross(v2, v1)
            rot_theta= HumanKinematic.vector_angle(v1, v2)
            rot_mat[:,i,...] = HumanKinematic.rotation_matrix(rot_axis, rot_theta)
        rot_mat = rot_mat.reshape(nb, len(layout['angles']), nd, nd)
        return rot_mat[0]

    @staticmethod
    def compute_x_from_rotation_matrix(rot_mat, target = 'euler'):
        if len(rot_mat.shape) == 3:
            rot_mat = rot_mat[None, ...]
        
        nb, n_ang, nd1, nd2 = rot_mat.shape
        assert nd1 == 3 and nd2 == 3
        rot_mat   = rot_mat.reshape(nb*n_ang, 3, 3) 
        r = Rotation.from_matrix(rot_mat)
        if target == 'euler':
            res = r.as_euler('XYZ')
            res = res.reshape(nb, n_ang, 3)
            assert res.shape == (nb, n_ang, 3)
            return res.squeeze()
        else:
            raise NotImplementedError
    
    @staticmethod
    def compute_euler_angle_from_joints_position(keypoint: np.array, layout):
        rot_mat = HumanKinematic.compute_rotation_matrix_from_joints_position(keypoint, layout)
        euler_ang = HumanKinematic.compute_x_from_rotation_matrix(rot_mat)
        return euler_ang