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

import ctypes
import os

## group名称最大长度
MAX_GROUP_NAME_LEN = 127

def check_group(group):
    """A function that check if a collection
    communication group is legal.If not raise error.
    Returns:
        None
    """
    if isinstance(group, (str)):
        if len(group) > MAX_GROUP_NAME_LEN:
            raise ValueError('group name is invalid. group: ' + group[0:MAX_GROUP_NAME_LEN])
        if len(group) == 0:
            raise ValueError('group name is empty.')
    else:
        raise ValueError('group must be a python str')

def check_rank_num(rank_num):
    """A function that check if a collection
    communication rank number is legal.If not raise error.
    Returns:
        None
    """
    if isinstance(rank_num, (int)):
        if rank_num <= 0:
            raise ValueError('rank_num[{}] is less than 0 or equal to 0'.format(rank_num))
    else:
        raise ValueError('rank_num must be a python int')

def check_rank_id(rank_id):
    """A function that check if a collection
    communication rank id is legal.If not raise error.
    Returns:
        None
    """
    if isinstance(rank_id, (int)):
        if rank_id < 0:
            raise ValueError('rank_id[{}] is less than 0'.format(rank_id))
    else:
        raise ValueError('rank_id must be a python int')


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

## 创建以group为名字的集合通信group
#  @param group string类型，集合通信group的标识，group_name作为字符串最大长度为128字节，含结束符；
#  @param rank_num int类型，组成该group的rank数量；
#  @param rank_ids list类型，组成该group的world_rank_id列表；
#  @return none
#  @see destroy_group()
def create_group(group, rank_num, rank_ids):
    check_group(group)
    check_rank_num(rank_num)
    if isinstance(rank_ids, (list)):
        if rank_num != len(rank_ids):
            raise ValueError('rank_num[{}]'.format(rank_num) + ' not equal to rank_ids len[{}].'.format(len(rank_ids)))
        for rank_id in rank_ids:
            check_rank_id(rank_id)
        c_array_rank_ids = c_array(ctypes.c_uint, rank_ids)
        c_rank_num = ctypes.c_uint(rank_num)
        c_group = c_str(group)
        ret = HCCL_LIB_CTYPES.HcomCreateGroup(c_group, c_rank_num, c_array_rank_ids)
        if ret != 0:
            raise ValueError('create group error:' + group)
    else:
        raise ValueError('rank_ids must be a python list')

## 销毁group
#  @param group string类型，集合通信group的标识，group_name作为字符串最大长度为128字节，含结束符；
#  @return none
#  @see create_group()
def destroy_group(group):
    check_group(group)
    c_group = c_str(group)
    ret = HCCL_LIB_CTYPES.HcomDestroyGroup(c_group)
    if ret != 0:
        raise ValueError('destroy group error :' + group)

## 获取group内rank数量
#  @param group string类型，group名称，可以为用户自定义group，默认为"hccl_world_group"
#  @return int类型，返回group内rank数量
def get_rank_size(group="hccl_world_group"):
    check_group(group)
    c_group = c_str(group)
    c_rank_size = ctypes.c_uint()
    ret = HCCL_LIB_CTYPES.HcomGetRankSize(c_group, ctypes.byref(c_rank_size))
    if ret != 0:
        raise ValueError('get rank size error. ret[{}]'.format(ret))

    return c_rank_size.value

## 获取device在group中对应的rank序号
#  @param group string类型，group名称，可以为用户自定义group，默认为"hccl_world_group"
#  @return int类型，返回device所在group的rank id
def get_rank_id(group="hccl_world_group"):
    check_group(group)
    c_group = c_str(group)
    c_rank_id = ctypes.c_uint()
    ret = HCCL_LIB_CTYPES.HcomGetRankId(c_group, ctypes.byref(c_rank_id))
    if (ret != 0):
        raise ValueError('get rank id error. ret[{}]'.format(ret))

    return c_rank_id.value


## 获取group内device所在服务器内的local rank数量
#  @param group string类型，group名称，可以为用户自定义group，默认为"hccl_world_group"
#  @return int类型，返回device所在服务器内的local rank数量
def get_local_rank_size(group="hccl_world_group"):
    check_group(group)
    c_group = c_str(group)
    c_local_rank_size = ctypes.c_uint()
    ret = HCCL_LIB_CTYPES.HcomGetLocalRankSize(c_group, ctypes.byref(c_local_rank_size))
    if (ret != 0):
        raise ValueError('get local rank size error. ret[{}]'.format(ret))

    return c_local_rank_size.value

## 获取group内device所在服务器内的local rank序号
#  @param group string类型，group名称，可以为用户自定义group，默认为"hccl_world_group"
#  @return int类型，返回device所在服务器内的local rank id序号
def get_local_rank_id(group="hccl_world_group"):
    check_group(group)
    c_group = c_str(group)
    c_local_rank_id = ctypes.c_uint()
    ret = HCCL_LIB_CTYPES.HcomGetLocalRankId(c_group, ctypes.byref(c_local_rank_id))
    if (ret != 0):
        raise ValueError('get local rank id error. ret[{}]'.format(ret))

    return c_local_rank_id.value

## 从group rank id，获取该进程对应的world rank id
#  @param group string类型，group名称
#  @group_rank_id int类型，进程在group中的rank id；
#  @return int类型，进程在"hccl_world_group"中的rank id
def get_world_rank_from_group_rank(group, group_rank_id):
    check_group(group)
    check_rank_id(group_rank_id)
    c_group = c_str(group)
    c_group_rank_id = ctypes.c_uint(group_rank_id)
    c_world_rank_id = ctypes.c_uint()
    ret = HCCL_LIB_CTYPES.HcomGetWorldRankFromGroupRank(c_group, c_group_rank_id, ctypes.byref(c_world_rank_id))
    if (ret != 0):
        raise ValueError('get world rank from group rank error. ret[{}]'.format(ret))

    return c_world_rank_id.value

## 从world rank id，获取该进程在group中的group rank id
#  @param world_rank_id int类型，进程在"hccl_world_group"中的rank id；
#  @param group string类型，group名称
#  @return int类型，进程在"hccl_world_group"中的rank id
def get_group_rank_from_world_rank(world_rank_id, group):
    check_group(group)
    check_rank_id(world_rank_id)
    c_group = c_str(group)
    c_world_rank_id = ctypes.c_uint(world_rank_id)
    c_group_rank_id = ctypes.c_uint()
    ret = HCCL_LIB_CTYPES.HcomGetGroupRankFromWorldRank(c_world_rank_id, c_group, ctypes.byref(c_group_rank_id))
    if (ret != 0):
        raise ValueError('get group rank from world rank error. ret[{}]'.format(ret))
    return c_group_rank_id.value