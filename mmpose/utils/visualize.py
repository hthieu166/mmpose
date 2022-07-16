import io

import cv2
import decord
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
from mmcv import load
from tqdm import tqdm
from pyskl.utils import joint_angle

class Vis3DPose:

    def __init__(self, item, layout='nturgb+d', fps=12, angle=(30, 45), fig_size=(8, 8), dpi=80):
        kp = item['keypoint']
        self.kp = kp
        assert self.kp.shape[-1] == 3
        self.layout = layout
        self.fps = fps
        self.angle = angle  # For 3D data only
        self.colors = ('#3498db', '#000000', '#e74c3c')  # l, m, r
        self.fig_size = fig_size
        self.dpi = dpi

        assert layout in ['nturgb+d']
        if self.layout == 'nturgb+d':
            self.num_joint = 25
            self.links = np.array([
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
                (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
                (19, 18), (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)], dtype=np.int) - 1
            self.left = np.array([5, 6, 7, 8, 13, 14, 15, 16, 22, 23], dtype=np.int) - 1
            self.right = np.array([9, 10, 11, 12, 17, 18, 19, 20, 24, 25], dtype=np.int) - 1
            self.num_link = len(self.links)
        self.limb_tag = [1] * self.num_link

        for i, link in enumerate(self.links):
            if link[0] in self.left or link[1] in self.left:
                self.limb_tag[i] = 0
            elif link[0] in self.right or link[1] in self.right:
                self.limb_tag[i] = 2

        assert len(kp.shape) == 4 and kp.shape[3] == 3 and kp.shape[2] == self.num_joint
        x, y, z = kp[..., 0], kp[..., 1], kp[..., 2]

        min_x, max_x = min(x[x != 0]), max(x[x != 0])
        min_y, max_y = min(y[y != 0]), max(y[y != 0])
        min_z, max_z = min(z[z != 0]), max(z[z != 0])
        max_axis = max(max_x - min_x, max_y - min_y, max_z - min_z)
        mid_x, mid_y, mid_z = (min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2
        self.min_x, self.max_x = mid_x - max_axis / 2, mid_x + max_axis / 2
        self.min_y, self.max_y = mid_y - max_axis / 2, mid_y + max_axis / 2
        self.min_z, self.max_z = mid_z - max_axis / 2, mid_z + max_axis / 2

        self.images = []

    def get_img(self, dpi=80):
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        return cv2.imdecode(img, -1)

    def vis_frame(self, ax, t):
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')
        ax.set_xlim3d([self.min_x, self.max_x])
        ax.set_ylim3d([self.min_y, self.max_y])
        ax.set_zlim3d([self.min_z, self.max_z])
        ax.view_init(*self.angle)
        ax.set_aspect('auto')
        for i in range(self.num_link):
            for m in range(self.kp.shape[0]):
                link = self.links[i]
                color = self.colors[self.limb_tag[i]]
                j1, j2 = self.kp[m, t, link[0]], self.kp[m, t, link[1]]
                if not ((np.allclose(j1, 0) or np.allclose(j2, 0)) and link[0] != 1 and link[1] != 1):
                    ax.plot([j1[0], j2[0]], [j1[1], j2[1]], [j1[2], j2[2]], lw=1, c=color)

    def vis(self):
        self.images = []
        plt.figure(figsize=self.fig_size)
        for t in range(self.kp.shape[1]):
            ax = plt.gca(projection='3d')
            self.vis_frame(ax, t)            
            self.images.append(self.get_img(dpi=self.dpi))
            ax.cla()
        return mpy.ImageSequenceClip(self.images, fps=self.fps)

