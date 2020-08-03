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
import numpy as np


def convert_into_torch_tensor(array) -> torch.Tensor:
    r"""Convert other type array into torch.Tensor

    Args:
        array (list, tuple, np.ndarray, torch.Tensor): Input array

    Returns:
        torch.Tensor: Processed array
    """
    if not isinstance(array, torch.Tensor):
        array = torch.Tensor(array)
    array = array.squeeze().float()
    return array
