# -*- coding: utf-8 -*-
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Zhihao Liang
## Gorilla Lab, South China University of Technology
## Email: mszhihaoliang@mail.scut.edu.cn
## Copyright (c) 2020
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from .base import convert_into_torch_tensor

__all__ = [
    "get_rotation_matrix_from_quaternion", "batch_get_rotation_matrix_from_quaternion",
    "get_quaternion_from_rotation_matrix", "batch_get_quaternion_from_rotation_matrix",
    "broadcast_rotation_array_multiplication", "broadcast_quaternion_array_multiplication",
    "quaternion_multiplication",
    "get_world_matrix", "batch_get_world_matrix"
]


# rotation operation
def get_rotation_matrix_from_quaternion(quaternion, first_w=True) -> torch.Tensor:
    r"""Get rotation matrix from quaternion

    Args:
        quaternion (list, tuple, np.ndarray, torch.Tensor): [4] quaternion vector.
        first_w (bool, optional): the w of quaternion are the first or last. Defaults to True.

    Returns:
        torch.Tensor: a [3, 3] rotation matrix.
    """
    quaternion = convert_into_torch_tensor(quaternion)
    assert len(quaternion.shape) == 1 and quaternion.shape[0] == 4, "quaternion must be a 4-length vector."
    quaternion = F.normalize(quaternion, dim=-1)
    if not first_w:
        quaternion = quaternion[[3, 0, 1, 2]]
    
    # get rotation matrix
    rotation = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
    rotation = torch.Tensor(rotation)
    return rotation

def batch_get_rotation_matrix_from_quaternion(quaternion, first_w=True) -> torch.Tensor:
    r"""Batch version of get rotation matrix from quaternion

    Args:
        quaternion (list, tuple, np.ndarray, torch.Tensor): a [N, 4] array contains N quaternion.
        first_w (bool, optional): the w of quaternion are the first or last. Defaults to True.

    Returns:
        torch.Tensor: a [N, 3, 3] array contains N rotation matrix.
    """
    quaternion = convert_into_torch_tensor(quaternion)
    assert len(quaternion.shape) == 2 and quaternion.shape[1] == 4, "quaternion must be a [N, 4] array."
    quaternion = F.normalize(quaternion, dim=-1)

    if not first_w:
        quaternion = quaternion[:, [3, 0, 1, 2]]
    sw = quaternion[:, 0] * quaternion[:, 0]
    sx = quaternion[:, 1] * quaternion[:, 1]
    sy = quaternion[:, 2] * quaternion[:, 2]
    sz = quaternion[:, 3] * quaternion[:, 3]

    m00 = (sx - sy - sz + sw)
    m11 = (-sx + sy - sz + sw)
    m22 = (-sx - sy + sz + sw)

    tmp1 = quaternion[:, 1] * quaternion[:, 2]
    tmp2 = quaternion[:, 3] * quaternion[:, 0]
    m10 = 2.0 * (tmp1 + tmp2)
    m01 = 2.0 * (tmp1 - tmp2)

    tmp1 = quaternion[:, 1] * quaternion[:, 3]
    tmp2 = quaternion[:, 2] * quaternion[:, 0]
    m20 = 2.0 * (tmp1 - tmp2)
    m02 = 2.0 * (tmp1 + tmp2)

    tmp1 = quaternion[:, 2] * quaternion[:, 3]
    tmp2 = quaternion[:, 1] * quaternion[:, 0]
    m21 = 2.0 * (tmp1 + tmp2)
    m12 = 2.0 * (tmp1 - tmp2)

    return torch.stack([m00, m01, m02, m10, m11, m12, m20, m21, m22], dim=1).view(-1, 3, 3).contiguous()


def get_quaternion_from_rotation_matrix(rotation, first_w=True) -> torch.Tensor:
    r"""Get quaternion from rotation matrix

    Args:
        rotation (list, tuple, np.ndarray, torch.Tensor): a [3, 3] rotation matrix.
        first_w (bool, optional): the w of quaternion are the first or last. Defaults to True.

    Returns:
        torch.Tensor: a 4-length vector.
    """
    rotation = convert_into_torch_tensor(rotation)
    assert len(rotation.shape) == 2 and list(rotation.shape) == [3, 3], "rotation must be a 3x3 matrix"

    t0 = rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
    t1 = rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
    t2 = rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
    t3 = rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
    temp = torch.Tensor([t0, t1, t2, t3])
    biggest_val, biggest_index = torch.max(temp)
    factor = 0.25 / biggest_val
    
    temp0 = biggest_val
    temp1 = (rotation[2, 1] - rotation[1, 2]) * factor
    temp2 = (rotation[0, 2] - rotation[2, 0]) * factor
    temp3 = (rotation[1, 0] - rotation[0, 1]) * factor
    temp4 = (rotation[2, 1] + rotation[1, 2]) * factor
    temp5 = (rotation[0, 2] + rotation[2, 0]) * factor
    temp6 = (rotation[1, 0] + rotation[0, 1]) * factor

    if biggest_index == 0:
        quaternion = torch.Tensor([temp0, temp1, temp2, temp3])
    elif biggest_index == 1:
        quaternion = torch.Tensor([temp1, temp0, temp6, temp5])
    elif biggest_index == 2:
        quaternion = torch.Tensor([temp2, temp6, temp0, temp4])
    elif biggest_index == 3:
        quaternion = torch.Tensor([temp3, temp5, temp4, temp0])

    if not first_w:
        quaternion = quaternion[[1, 2, 3, 0]]
    return quaternion

def batch_get_quaternion_from_rotation_matrix(rotation, first_w=True) -> torch.Tensor:
    r"""Batch version of get quaternion from matrix

    Args:
        rotation (list, tuple, np.ndarray, torch.Tensor): a [N, 3, 3] array contains N rotation matrix.
        first_w (bool, optional): the w of quaternion are the first or last. Defaults to True.

    Returns:
        torch.Tensor: a [N, 4] array contains N quaternion.
    """
    rotation = convert_into_torch_tensor(rotation)
    assert len(rotation.shape) == 3 and list(rotation.shape[1:]) == [3, 3], "rotation must be a [N, 3, 3] array"

    t0 = rotation[:, 0, 0] + rotation[:, 1, 1] + rotation[:, 2, 2]
    t1 = rotation[:, 0, 0] - rotation[:, 1, 1] - rotation[:, 2, 2]
    t2 = rotation[:, 1, 1] - rotation[:, 0, 0] - rotation[:, 2, 2]
    t3 = rotation[:, 2, 2] - rotation[:, 0, 0] - rotation[:, 1, 1]
    temp = torch.stack([t0, t1, t2, t3], dim=1)
    biggest_val, biggest_index = torch.max(temp, dim=1)
    factor = 0.25 / biggest_val

    temp0 = biggest_val
    temp1 = (rotation[:, 2, 1] - rotation[:, 1, 2]) * factor
    temp2 = (rotation[:, 0, 2] - rotation[:, 2, 0]) * factor
    temp3 = (rotation[:, 1, 0] - rotation[:, 0, 1]) * factor
    temp4 = (rotation[:, 2, 1] + rotation[:, 1, 2]) * factor
    temp5 = (rotation[:, 0, 2] + rotation[:, 2, 0]) * factor
    temp6 = (rotation[:, 1, 0] + rotation[:, 0, 1]) * factor

    quaternion = torch.empty([rotation.shape[0], 4], dtype=torch.float)
    quaternion_index0 = torch.copy(torch.stack([temp0, temp1, temp2, temp3], dim=1))
    quaternion_index1 = torch.copy(torch.stack([temp1, temp0, temp6, temp5], dim=1))
    quaternion_index2 = torch.copy(torch.stack([temp2, temp6, temp0, temp4], dim=1))
    quaternion_index3 = torch.copy(torch.stack([temp3, temp5, temp4, temp0], dim=1))

    biggest_index1_map = (biggest_index == 1)[:, None].expand_as(quaternion)
    biggest_index2_map = (biggest_index == 2)[:, None].expand_as(quaternion)
    biggest_index3_map = (biggest_index == 3)[:, None].expand_as(quaternion)

    quaternion = quaternion_index0
    quaternion = torch.where(biggest_index1_map, quaternion_index1, quaternion)
    quaternion = torch.where(biggest_index2_map, quaternion_index2, quaternion)
    quaternion = torch.where(biggest_index3_map, quaternion_index3, quaternion)

    if not first_w:
        quaternion = quaternion[:, [1, 2, 3, 0]]
    return quaternion


def rotation_array_multiplication(rotation_array1, rotation_array2) -> torch.Tensor:
    r"""Rotation array multiplication

    Args:
        rotation_array1 (list, tuple, np.ndarray, torch.Tensor): a [N, 3, 3] array contains N rotation matrix.
        rotation_array2 (list, tuple, np.ndarray, torch.Tensor): a [N, 3, 3] array contains N rotation matrix.

    Returns:
        torch.Tensor: a [N, 3, 3] array contains N rotation matrix after matrix multiplication.
    """
    assert len(rotation_array1) == len(rotation_array2), "length of both rotation_array must be same"
    rotation_array1 = convert_into_torch_tensor(rotation_array1)
    rotation_array2 = convert_into_torch_tensor(rotation_array2)
    assert len(rotation_array1.shape) == 3 and list(rotation_array1.shape[1:]) == [3, 3], "rotation_array1 must be a [N, 3, 3] array"
    assert len(rotation_array2.shape) == 3 and list(rotation_array2.shape[1:]) == [3, 3], "rotation_array2 must be a [N, 3, 3] array"

    rotation_array = torch.zeros_like(rotation_array1)
    for i in range(3):
        row = rotation_array1[:, i, :]
        for j in range(3):
            col = rotation_array2[:, :, j]
            mul = (row * col).sum(dim=1)
            rotation_array[:, i, j] = mul

    return rotation_array

def broadcast_rotation_array_multiplication(rotation_array, rot_matrix=torch.eye(3), type="left") -> torch.Tensor:
    r"""Broadcat rotation matrix multiplication

    Args:
        rotation_array (list, tuple, np.ndarray, torch.Tensor): a [N, 3, 3] array contains N rotation matrix.
        rot_matrix (list, tuple, np.ndarray, torch.Tensor, optional):  a [3, 3] rotation matrix. Defaults to torch.eye(3).
        type (str, optional): left multiplication or right multiplication (rot_matrix in left or right). Defaults to "left".

    Returns:
        torch.Tensor: a [N, 3, 3] array contains N rotation matrix after broadcast matrix multiplication.
    """
    assert type in ["left", "right"], "'left' or 'right' multiplication"
    rotation_array  = convert_into_torch_tensor(rotation_array)
    rot_matrix = convert_into_torch_tensor(rot_matrix)
    assert len(rotation_array.shape) == 3 and list(rotation_array.shape[1:]) == [3, 3], "rotation_array must be a [N, 3, 3] array"
    assert len(rot_matrix.shape) == 2 and list(rot_matrix.shape) == [3, 3], "rot_matrix must be a [3, 3] array"
    rotation = rot_matrix[None, :, :].expand_as(rotation_array)

    if type == "left":
        rotation_array = rotation_array_multiplication(rotation, rotation_array)
    else:
        rotation_array = rotation_array_multiplication(rotation_array, rotation)

    return rotation_array


def quaternion_multiplication(quaternion1, quaternion2, first_w=False) -> torch.Tensor:
    r"""Quaternion multiplication

    Args:
        quaternion1 (list, tuple, np.ndarray, torch.Tensor): a [4] quaternion vector.
        quaternion2 (list, tuple, np.ndarray, torch.Tensor): a [4] quaternion vector.
        first_w (bool, optional): the w of quaternion are the first or last. Defaults to True.

    Returns:
        torch.Tensor: a [4] quaternion vector.
    """
    quaternion1 = convert_into_torch_tensor(quaternion1)
    quaternion2 = convert_into_torch_tensor(quaternion2)
    assert len(quaternion1.shape) == 1 and quaternion1.shape[0] == 4, "quaternion1 must be a 4-length vector."
    assert len(quaternion2.shape) == 1 and quaternion2.shape[0] == 4, "quaternion2 must be a 4-length vector."
    quaternion1 = F.normalize(quaternion1, dim=-1)
    quaternion2 = F.normalize(quaternion2, dim=-1)

    if first_w:
        w1, w2 = quaternion1[0], quaternion2[0]
        x1, x2 = quaternion1[1], quaternion2[1]
        y1, y2 = quaternion1[2], quaternion2[2]
        z1, z2 = quaternion1[3], quaternion2[3]
    else:
        w1, w2 = quaternion1[3], quaternion2[3]
        x1, x2 = quaternion1[0], quaternion2[0]
        y1, y2 = quaternion1[1], quaternion2[1]
        z1, z2 = quaternion1[2], quaternion2[2]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    if first_w:
        return torch.stack([w, x, y, z])
    else:
        return torch.stack([x, y, z, w])

def quaternion_array_multiplication(quaternion_array1, quaternion_array2, first_w=False) -> torch.Tensor:
    r"""Quaternion array multiplication

    Args:
        quaternion_array1 (list, tuple, np.ndarray, torch.Tensor): a [N, 4] array contains N quaternion.
        quaternion_array2 (list, tuple, np.ndarray, torch.Tensor): a [N, 4] array contains N quaternion.
        first_w (bool, optional): the w of quaternion are the first or last. Defaults to True.

    Returns:
        torch.Tensor: a [N, 4] array contains N quaternion.
    """
    quaternion_array1 = convert_into_torch_tensor(quaternion_array1)
    quaternion_array2 = convert_into_torch_tensor(quaternion_array2)
    assert len(quaternion_array1.shape) == 2 and quaternion_array1.shape[1] == 4, "quaternion_array1 must be a [N, 4] array."
    assert len(quaternion_array2.shape) == 2 and quaternion_array2.shape[1] == 4, "quaternion_array2 must be a [N, 4] array."
    quaternion_array1 = F.normalize(quaternion_array1, dim=-1)
    quaternion_array2 = F.normalize(quaternion_array2, dim=-1)

    if first_w:
        w1, w2 = quaternion_array1[:, 0], quaternion_array2[:, 0]
        x1, x2 = quaternion_array1[:, 1], quaternion_array2[:, 1]
        y1, y2 = quaternion_array1[:, 2], quaternion_array2[:, 2]
        z1, z2 = quaternion_array1[:, 3], quaternion_array2[:, 3]
    else:
        w1, w2 = quaternion_array1[:, 3], quaternion_array2[:, 3]
        x1, x2 = quaternion_array1[:, 0], quaternion_array2[:, 0]
        y1, y2 = quaternion_array1[:, 1], quaternion_array2[:, 1]
        z1, z2 = quaternion_array1[:, 2], quaternion_array2[:, 2]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    if first_w:
        return torch.stack([w, x, y, z], dim=1)
    else:
        return torch.stack([x, y, z, w], dim=1)

def broadcast_quaternion_array_multiplication(quaternion_array, quater_vector=[1, 0, 0, 0], type="left", first_w=True) -> torch.Tensor:
    r"""Broadcast quaternion array multiplication

    Args:
        quaternion_array (list, tuple, np.ndarray, torch.Tensor): a [N, 4] array contains N quaternion.
        quater_vector (list, tuple, np.ndarray, torch.Tensor, optional):  a [4] quaternion vector. Defaults to [1, 0, 0, 0].
        type (str, optional): left multiplication or right multiplication (quater in left or right). Defaults to "left".
        first_w (bool, optional): the w of quaternion are the first or last. Defaults to True.

    Returns:
        torch.Tensor: a [N, 4] array contains N quaternion.
    """
    assert type in ["left", "right"], "'left' or 'right' multiplication"
    quaternion_array = convert_into_torch_tensor(quaternion_array)
    quater_vector = convert_into_torch_tensor(quater_vector)
    assert len(quaternion_array.shape) == 2 and list(quaternion_array.shape[1:]) == [4], "quaternion_array must be a [N, 4] array"
    assert len(quater_vector.shape) == 1 and quater_vector == 4, "quater_vector must be a 4-length vector"

    quaternion = quater_vector[None, :].expand_as(quaternion_array)

    if type == "left":
        quaternion_array = quaternion_array_multiplication(quaternion, quaternion_array, first_w=first_w)
    else:
        quaternion_array = quaternion_array_multiplication(quaternion_array, quaternion, first_w=first_w)

    return quaternion_array


def get_world_matrix(translation=[0, 0, 0], rotation=[1, 0, 0, 0], first_w=False) -> torch.Tensor:
    """Get world matrix according to translation and rotation

    Args:
        translation (list, tuple, np.ndarray, torch.Tensor, optional): a 3-length translation vector. Defaults to [0, 0, 0].
        rotation (list, tuple, np.ndarray, torch.Tensor, optional): quaternion or rotation matrix. Defaults to [1, 0, 0, 0].
        first_w (bool, optional): the w of quaternion are the first or last, used if rotation is quaternion. Defaults to True.

    Raises:
        ValueError: rotation is not a 4-length quaternion vector or a [3, 3] rotation matrix.

    Returns:
        torch.Tensor: a [4, 4] world matrix
    """
    translation = convert_into_torch_tensor(translation)
    assert len(translation.shape) == 1 and translation.shape[0] == 3, "translation must be a 3-length vector"
    rotation = convert_into_torch_tensor(rotation)
    if len(rotation.shape) == 1 and rotation.shape[0] == 4: # quaternion
        rotation = F.normalize(rotation, dim=-1)
        rotation = get_rotation_matrix_from_quaternion(rotation, first_w)
    elif len(rotation.shape) == 2 and list(rotation.shape) == [3, 3]: # rotation matrix
        pass
    else:
        print("rotation error:\n", rotation)
        raise ValueError("rotation must be a quaternion vector(a 4-length vector) or a [3, 3] rotation matrix")

    world_matrix = torch.eye(4, dtype=torch.float)
    world_matrix[:3, :3] = rotation
    world_matrix[:3, 3]  = translation
    return world_matrix

def batch_get_world_matrix(translation_array=[[0, 0, 0]], rotation_array=[[1, 0, 0, 0]], first_w=False) -> torch.Tensor:
    """Get world matrix according to translation and rotation

    Args:
        translation_array (list, tuple, np.ndarray, torch.Tensor, optional): a [N, 3] translation array. Defaults to [[0, 0, 0]].
        rotation_array (list, tuple, np.ndarray, torch.Tensor, optional):
            a [N, 4] quaternion array or a [N, 3, 3] rotation array. Defaults to [[1, 0, 0, 0]].
        first_w (bool, optional): the w of quaternion are the first or last, used if rotation_array is quaternion array Defaults to True.

    Raises:
        ValueError: rotation array is not a [N, 4] quaternion array or a [N, 3, 3] rotation array.

    Returns:
        torch.Tensor: a [N, 4, 4] world matrix
    """
    assert len(translation_array) == len(rotation_array), "length of translation and rotation must be same"

    translation_array = convert_into_torch_tensor(translation_array)
    assert len(translation_array.shape) == 2 and translation_array.shape[-1] == 3, "translation must be a [n, 3] array"
    rotation_array = convert_into_torch_tensor(rotation_array)
    if len(rotation_array.shape) == 2 and rotation_array.shape[-1] == 4: # quaternion array
        rotation_array = F.normalize(rotation_array, dim=-1)
        rotation_array = batch_get_rotation_matrix_from_quaternion(rotation_array, first_w)
    elif len(rotation_array.shape) == 3 and list(rotation_array.shape[1:]) == [3, 3]: # rotation array
        pass
    else:
        print("rotation_array error:\n", rotation_array)
        raise ValueError("rotation_array must be a [N, 4] quaternion array or a [N, 3, 3] rotation array")
    
    world_matrix = F.pad(rotation_array, (0, 1, 0, 1), "constant", 0)
    world_matrix[:, :3, 3] = translation_array
    world_matrix[:, -1, -1] = 1
    return world_matrix

