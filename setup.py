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

#!/usr/bin/env python

import os, sys, glob
import torch
import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension

# run the version-script and get the '__version__' arg
exec(open(osp.join("gorilla", "version.py")).read())

requirements = [
    "numpy",
    "tqdm",
    "open3d",
    "torch>=1.0.0",
    "torchvision>=0.5.0",
    "gpustat",
    "pynvml"
]

def get_extensions():
    extensions = []
    ext_name = "gorilla._ext"
    if torch.cuda.is_available():
        op_files = glob.glob(osp.join(".", "gorilla", "lib", "gpu", "*"))
        extension = CUDAExtension
    else:
        print(f"Compiling {ext_name} without CUDA")
        op_files = glob.glob(osp.join(".", "gorilla", "lib", "cpu", "*"))
        extension = CppExtension

        ext_ops = extension(
            name=ext_name,
            sources=op_files,
            extra_compile_args={
                "cxx": ["-g"],
                "nvcc": ["-O2"]
            }
        )
        extensions.append(ext_ops)
    return extension

if __name__ == "__main__":
    setup(
        name = "gorilla",
        version = __version__,
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
    )