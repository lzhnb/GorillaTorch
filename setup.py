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

#!/usr/bin/env python

import os
import sys
import glob
import torch
import os.path as osp
from setuptools import dist, setup, find_packages, Extension
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension

# run the version-script and get the "__version__" arg
# exec(open(osp.join("gorilla", "version.py")).read())

requirements = [
    "addict",
    "yapf",
    "numpy",
    "tqdm",
    "open3d",
    "torch>=1.0.0",
    "torchvision>=0.5.0",
    "gpustat",
    "pynvml",
    "numba",
    "pyyaml",
    "terminaltables",
    "lyft_dataset_sdk",
    "pycocotools",
    "nuscenes-devkit",
]

def get_version():
    version_file = osp.join("gorilla", "version.py")
    with open(version_file, "r", encoding="utf-8") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]

def get_sources(module, surfix="*.c*"):
    src_dir = osp.join(*module.split("."), "src")
    return glob.glob(osp.join(src_dir, surfix))

def make_extension(name, module):
    if not torch.cuda.is_available(): return
    extersion = CUDAExtension
    return extersion(
        name=".".join([module, name]),
        sources=get_sources(module),
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        },
        define_macros=[("WITH_CUDA", None)]
    )

def get_extensions():
    extensions = []

    if torch.cuda.is_available():
        extension = CUDAExtension
    
        extensions = [
            make_extension(
                name="compiling_info",
                module="gorilla.ops.utils"),
            make_extension(
                name="nms_ext",
                module="gorilla.ops.nms"),
            make_extension(
                name="iou3d_cuda",
                module="gorilla.ops.iou3d"),
            make_extension(
                name="voxel_layer",
                module="gorilla.ops.voxel"),
            make_extension(
                name="roiaware_pool3d_ext",
                module="gorilla.ops.roiaware_pool3d"),
            make_extension(
                name="ball_query_ext",
                module="gorilla.ops.ball_query"),
            make_extension(
                name="group_points_ext",
                module="gorilla.ops.group_points"),
            make_extension(
                name="interpolate_ext",
                module="gorilla.ops.interpolate"),
            make_extension(
                name="furthest_point_sample_ext",
                module="gorilla.ops.furthest_point_sample"),
            make_extension(
                name="gather_points_ext",
                module="gorilla.ops.gather_points"),
            make_extension(
                name='roi_align_ext',
                module='gorilla.ops.roi_align'),
            make_extension(
                name='roi_pool_ext',
                module='gorilla.ops.roi_pool'),
            # TODO: fix compile bug
            # make_extension(
            #     name='deform_conv_ext',
            #     module='gorilla.ops.dcn'),
            # make_extension(
            #     name='deform_pool_ext',
            #     module='gorilla.ops.dcn'),
            make_extension(
                name='sigmoid_focal_loss_ext',
                module='gorilla.ops.sigmoid_focal_loss'),
            make_extension(
                name='masked_conv2d_ext',
                module='gorilla.ops.masked_conv'),
            # make_extension(
            #     name='carafe_ext',
            #     module='gorilla.ops.carafe'),
            # make_extension(
            #     name='carafe_naive_ext',
            #     module='gorilla.ops.carafe'),
            make_extension(
                name='corner_pool_ext',
                module='gorilla.ops.corner_pool'),
        ]
    return extensions

if __name__ == "__main__":
    setup(
        name = "gorilla",
        version = get_version(),
        author = "Zhihao Liang",
        author_mail = "mszhihaoliang@mail.scut.edu.cn",
        url = "https://github.com/lzhnb/GorillaTorch",
        description="Utils Package for 3D learning task using PyTorch",
        long_description=open("README.md").read(),
        license="MIT",
        install_requires=requirements,
        packages=find_packages(exclude=["tests"]),
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False
    )