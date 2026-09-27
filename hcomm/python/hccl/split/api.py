#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

## @package hccl.split.api
# HCCL group管理API

import ctypes
import os

## group名称最大长度
MAX_GROUP_NAME_LEN = 127
## The max value of uint32
MAX_VALUE_UINT32 = 4294967295

def load_lib():
    """ load libhcomm.so file."""
    try:
        hccl_lib = ctypes.CDLL('libhcomm.so')
    except Exception as e:
        raise ValueError('load hccl lib error')

    return hccl_lib

## hccl动态库
HCCL_LIB_CTYPES = load_lib()

def c_str(string):
    """Convert a python string to C string."""
    return ctypes.c_char_p(string.encode('utf-8'))

def c_array(ctype, values):
    """Create ctypes array from a python array."""
    return (ctype * len(values))(*values)

## 基于梯度的索引id，在集合通信group内设置反向梯度切分策略
#  @param idxList list类型，梯度的索引id列表；
#  @param group string类型，group名称，可以为用户自定义group或者"hccl_world_group";
#  @return none
#  @see set_split_strategy_by_idx()
def set_split_strategy_by_idx(idxList, group="hccl_world_group"):
    # 判断入参类型和长度信息是否正确
    if isinstance(group, (str)):
        if len(group) > MAX_GROUP_NAME_LEN:
            raise ValueError('group name len[{}] too long,'.format(len(group))
                + ' Max len[{}].'.format(MAX_GROUP_NAME_LEN))
        if len(group) == 0:
            raise ValueError('group name is empty.')
    else:
        raise ValueError('group must be a python str')
    if isinstance(idxList, (list)):
        if(len(idxList) == 0):
            raise ValueError('idxList length is 0')
    else:
        raise ValueError('idxList must be a python list')

    # 判断入参idxList中的数据类型和数据范围是否正确
    for idx in idxList:
        if not isinstance(idx, (int)):
            raise ValueError('idx val[{}] in idxList is not python int type.'.format(idx))
        if idx < 0 or idx > MAX_VALUE_UINT32:
            raise ValueError('idx val[{}] in idxList is an out-of-range value,'
                ' the correct value range is 0 to {}'.format(idx, MAX_VALUE_UINT32))

    # 判断入参idxList中的数据逻辑(按升序排列)是否正确
    if not all([idxList[idx] < idxList[idx + 1]
               for idx in range(len(idxList) - 1)]):
        raise ValueError('idx in idxList is not ascending')

    c_array_idxList = c_array(ctypes.c_uint, idxList)
    c_idx_num = ctypes.c_uint(len(idxList))
    c_group = c_str(group)
    ret = HCCL_LIB_CTYPES.HcomSetGradFusionByIndex(c_group, c_idx_num, c_array_idxList)
    if ret != 0:
        raise ValueError('split error. ret[{}]'.format(ret))

## 基于梯度的索引id，在集合通信group内设置反向梯度切分策略
#  @param dataSizeList list类型，梯度参数数据量百分比列表；
#   梯度数据量序列总百分比之和必须为100，比如模型总共有150M梯度数据量，需要切分90M，30M，30M三段，则可以设置为[60,20,20]
#  @param group string类型，group名称，可以为用户自定义group或者"hccl_world_group";
#  @return none
#  @see set_split_strategy_by_size()
def set_split_strategy_by_size(dataSizeList, group="hccl_world_group"):
    # 判断入参类型和长度信息是否正确
    if isinstance(group, (str)):
        if len(group) > MAX_GROUP_NAME_LEN:
            raise ValueError('group name len[{}] too long,'.format(len(group))
                + ' Max len[{}].'.format({MAX_GROUP_NAME_LEN}))
        if len(group) == 0:
            raise ValueError('group name is empty.')
    else:
        raise ValueError('group must be a python str')

    if isinstance(dataSizeList, (list)):
        if len(dataSizeList) == 0:
            raise ValueError('dataSizeList length is 0')
    else:
        raise ValueError('dataSizeList must be a python list')

    # 判断入参dataSizeList中的数据类型和数据范围是否正确
    for dataSize in dataSizeList:
        if not isinstance(dataSize, (int, float)):
            raise ValueError('dataSize val[{}] in dataSizeList is not python int or float type.'.format(dataSize))
        if dataSize < 0:
            raise ValueError('dataSize val[{}] in dataSizeList cannot be a negative number.'.format(dataSize))

    # 判断入参dataSizeList中的数据逻辑(总数据量百分比为100%)是否正确
    if sum(dataSizeList) != 100:
        raise ValueError('size percentage list sum is not 100%')

    c_array_sizeList = c_array(ctypes.c_float, dataSizeList)
    c_size_num = ctypes.c_uint(len(dataSizeList))
    c_group = c_str(group)
    ret = HCCL_LIB_CTYPES.HcomSetGradFusionBySize(c_group, c_size_num, c_array_sizeList)
    if ret != 0:
        raise ValueError('split error, ret[{}]'.format(ret))



