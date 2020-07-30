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

from gpustat import GPUStatCollection
import gc, time
import datetime
import pynvml

import torch
import numpy as np

__all__ = []

def get_free_gpu(mode="memory", memory_need=10000) -> list:
    r"""Get free gpu according to mode (process-free or memory-free).

    Args:
        mode (str, optional): memory-free or process-free. Defaults to "memory".
        memory_need (int): The memory you need, used if mode=='memory'. Defaults to 10000.

    Returns:
        list: free gpu ids
    """
    assert mode in ["memory", "process"], "mode must be 'memory' or 'process'"
    if mode == "memory":
        assert memory_need is not None, "'memory_need' if None, 'memory' mode must give the free memory you want to apply for"
        memory_need = int(memory_need)
        assert memory_need > 0, "'memory_need' you want must be positive"
    gpu_stats = GPUStatCollection.new_query()
    gpu_free_id_list = []

    for idx, gpu_stat in enumerate(gpu_stats):
        if gpu_check_condition(gpu_stat, mode, memory_need):
            gpu_free_id_list.append(idx)
            print("gpu[{}]: {}MB".format(i, gpu_stat.memory_free))
    
    return gpu_free_id_list

def gpu_check_condition(gpu_stat, mode, memory_need) -> bool:
    r"""Check gpu is free or not.

    Args:
        gpu_stat (gpustat.core): gpustat to check
        mode (str): memory-free or process-free.
        memory_need (int): The memory you need, used if mode=='memory'

    Returns:
        bool: gpu is free or not
    """
    if mode == "memory":
        return gpu_stat.memory_free > memory_need
    elif mode == "process":
        for process in gpu_stat.processes:
            if process["command"] == "python": return False
        return True
    else: return False

def supervise_gpu(num_gpu=1, memory_need=10000, mode="memory") -> list:
    r"""Supervise gpu for you need

    Args:
        num_gpu (int, optional): The number of gpu you need. Defaults to 1.
        memory_need (int, optional): The memory you need, used if mode=='memory'. Defaults to 10000.
        mode (str, optional): memory-free or process-free. Defaults to "memory".

    Returns:
        list: free gpu ids
    """
    gpu_free_ids = []
    while len(gpu_free_ids) < num_gpu:
        time.sleep(2)
        gpu_free_ids = get_free_gpu(memory_need, mode)
    return gpu_free_ids[:num_gpu]


# change from https://github.com/Oldpan/Pytorch-Memory-Utils/blob/master/gpu_mem_track.py
class MemTracker(object):
    r"""Class used to track PyTorch memory usage
    Args:
        frame: a frame to detect current py-file runtime
        detail(bool, default True): whether the function shows the detail gpu memory usage
        path(str): where to save log file
        verbose(bool, default False): whether show the trivial exception
        device(int): GPU number, default is 0
    """
    def __init__(self, frame, detail=True, path="", verbose=False, device=0):
        self.frame = frame
        self.print_detail = detail
        self.last_tensor_sizes = set()
        self.gpu_profile_fn = path + f"{datetime.datetime.now():%d-%b-%y-%H:%M:%S}-gpu_mem_track.txt"
        self.verbose = verbose
        self.begin = True
        self.device = device

        self.func_name = frame.f_code.co_name
        self.filename = frame.f_globals["__file__"]
        if (self.filename.endswith(".pyc") or
                self.filename.endswith(".pyo")):
            self.filename = self.filename[:-1]
        self.module_name = self.frame.f_globals["__name__"]
        self.curr_line = self.frame.f_lineno

    def get_tensors(self):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                    tensor = obj
                else:
                    continue
                if tensor.is_cuda:
                    yield tensor
            except Exception as e:
                if self.verbose:
                    print("A trivial exception occured: {}".format(e))

    def track(self):
        r"""
        Track the GPU memory usage
        """
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        self.curr_line = self.frame.f_lineno
        where_str = self.module_name + " " + self.func_name + ":" + " line " + str(self.curr_line)

        with open(self.gpu_profile_fn, "a+") as f:

            if self.begin:
                f.write(f"GPU Memory Track | {datetime.datetime.now():%d-%b-%y-%H:%M:%S} |"
                        f" Total Used Memory:{meminfo.used/1000**2:<7.1f}Mb\n\n")
                self.begin = False

            if self.print_detail is True:
                ts_list = [tensor.size() for tensor in self.get_tensors()]
                new_tensor_sizes = {(type(x), tuple(x.size()), ts_list.count(x.size()), np.prod(np.array(x.size()))*4/1000**2)
                                    for x in self.get_tensors()}
                for t, s, n, m in new_tensor_sizes - self.last_tensor_sizes:
                    f.write(f"+ | {str(n)} * Size:{str(s):<20} | Memory: {str(m*n)[:6]} M | {str(t):<20}\n")
                for t, s, n, m in self.last_tensor_sizes - new_tensor_sizes:
                    f.write(f"- | {str(n)} * Size:{str(s):<20} | Memory: {str(m*n)[:6]} M | {str(t):<20} \n")
                self.last_tensor_sizes = new_tensor_sizes

            f.write(f"\nAt {where_str:<50}"
                    f"Total Used Memory:{meminfo.used/1000**2:<7.1f}Mb\n\n")

        pynvml.nvmlShutdown()





