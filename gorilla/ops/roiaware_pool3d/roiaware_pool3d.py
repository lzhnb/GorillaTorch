import torch
from torch import nn as nn
from torch.autograd import Function

from . import roiaware_pool3d_ext


class RoIAwarePool3d(nn.Module):

    def __init__(self, out_size, max_pts_per_voxel=128, mode='max'):
        super().__init__()
        """RoIAwarePool3d module

        Args:
            out_size (int or tuple): n or [n1, n2, n3]
            max_pts_per_voxel (int): m
            mode (str): 'max' or 'avg'
        """
        self.out_size = out_size
        self.max_pts_per_voxel = max_pts_per_voxel
        assert mode in ['max', 'avg']
        pool_method_map = {'max': 0, 'avg': 1}
        self.mode = pool_method_map[mode]

    def forward(self, rois, pts, pts_feature):
        """RoIAwarePool3d module forward.

        Args:
            rois (torch.Tensor): [N, 7],in LiDAR coordinate,
                (x, y, z) is the bottom center of rois
            pts (torch.Tensor): [npoints, 3]
            pts_feature (torch.Tensor): [npoints, C]

        Returns:
            pooled_features (torch.Tensor): [N, out_x, out_y, out_z, C]
        """

        return RoIAwarePool3dFunction.apply(rois, pts, pts_feature,
                                            self.out_size,
                                            self.max_pts_per_voxel, self.mode)

if __name__ == '__main__':
    pass
