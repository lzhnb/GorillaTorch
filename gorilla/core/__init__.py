# -*- coding: utf-8 -*-
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Zhihao Liang
## Gorilla Lab, South China University of Technology
## Email: mszhihaoliang@mail.scut.edu.cn
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from .registry import Registry, build_from_cfg


__all__ = [
    "Registry", "build_from_cfg"
]

