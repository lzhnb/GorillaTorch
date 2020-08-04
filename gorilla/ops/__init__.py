from .context_block import ContextBlock
# TODO: fix Registry bug KeyError: 'ConvWS is already registered in conv layer'
# from .conv_ws import ConvWS2d, conv_ws_2d
from .corner_pool import CornerPool
from .dcn import (DeformConv, DeformConvPack, DeformRoIPooling,
                  DeformRoIPoolingPack, ModulatedDeformConv,
                  ModulatedDeformConvPack, ModulatedDeformRoIPoolingPack,
                  deform_conv, deform_roi_pooling, modulated_deform_conv)
from .generalized_attention import GeneralizedAttention
from .masked_conv import MaskedConv2d
from .nms import batched_nms, nms, nms_match, soft_nms
from .non_local import NonLocal2D
from .plugin import build_plugin_layer
from .point_sample import (SimpleRoIAlign, point_sample,
                           rel_roi_point_to_rel_img_point)
from .roi_align import RoIAlign, roi_align
from .roi_pool import RoIPool, roi_pool
# fix import error
# from .saconv import SAConv2d
# from .sigmoid_focal_loss import SigmoidFocalLoss, sigmoid_focal_loss
from .utils import get_compiler_version, get_compiling_cuda_version
from .wrappers import Conv2d, ConvTranspose2d, Linear, MaxPool2d

from .pointnet_modules import PointFPModule, PointSAModule, PointSAModuleMSG
from .ball_query import ball_query
from .furthest_point_sample import furthest_point_sample
from .gather_points import gather_points
from .group_points import (GroupAll, QueryAndGroup, group_points,
                           grouping_operation)
from .interpolate import three_interpolate, three_nn
from .roiaware_pool3d import (RoIAwarePool3d, points_in_boxes_batch,
                              points_in_boxes_cpu, points_in_boxes_gpu)
from .utils import get_compiler_version, get_compiling_cuda_version
from .voxel import DynamicScatter, Voxelization, dynamic_scatter, voxelization
from .sparse_block import (SparseBasicBlock, SparseBottleneck, make_sparse_convmodule)

__all__ = [
    # NOTE: # modify from mmdetection3s https://github.com/open-mmlab/mmdetection/mmdet/ops
    'nms', 'soft_nms', 'RoIAlign', 'roi_align', 'RoIPool', 'roi_pool',
    'DeformConv', 'DeformConvPack', 'DeformRoIPooling', 'DeformRoIPoolingPack',
    'ModulatedDeformRoIPoolingPack', 'ModulatedDeformConv',
    'ModulatedDeformConvPack', 'deform_conv', 'modulated_deform_conv',
    'deform_roi_pooling', 'SigmoidFocalLoss', 'sigmoid_focal_loss',
    'MaskedConv2d', 'ContextBlock', 'GeneralizedAttention', 'NonLocal2D',
    'get_compiler_version', 'get_compiling_cuda_version', 'ConvWS2d',
    'conv_ws_2d', 'build_plugin_layer', 'batched_nms', 'Conv2d',
    'ConvTranspose2d', 'MaxPool2d', 'Linear', 'nms_match', 'CornerPool',
    'point_sample', 'rel_roi_point_to_rel_img_point', 'SimpleRoIAlign',
    'SAConv2d',

    # NOTE: from mmdetection3s https://github.com/open-mmlab/mmdetection3d/mmdet3d/ops
    "soft_nms", "get_compiler_version",
    "batched_nms", "Voxelization", "voxelization",
    "dynamic_scatter", "DynamicScatter",
    "RoIAwarePool3d", "points_in_boxes_gpu", "points_in_boxes_cpu",
    "ball_query", "furthest_point_sample",
    "SparseBasicBlock", "SparseBottleneck", "make_sparse_convmodule"
    "three_interpolate", "three_nn", "gather_points", "grouping_operation",
    "group_points", "GroupAll", "QueryAndGroup", "PointSAModule",
    "PointSAModuleMSG", "points_in_boxes_batch",
    "get_compiler_version", "get_compiling_cuda_version"
]
