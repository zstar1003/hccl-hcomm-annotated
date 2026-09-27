/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __RANK_INFO_H__
#define __RANK_INFO_H__

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 获取topo addr info的大小，用于提前分配内存
 * param[in] phyId   NPU物理ID
 * param[out] size   rankinfo的大小, 注意不是精确大小，能保证装下rankinfo
 */
int TopoAddrInfoGetSize(int phyId, size_t* size);

/**
 * 获取拓扑文件的路径
 * param[in] phyId   NPU物理ID
 * param[out] filePath  拓扑文件的路径
 * param[out] bufSize   拓扑文件路径的最大长度
 */

int TopoAddrInfoGetTopoFilePath(int phyId, char* filePath, size_t bufSize);
/**
 * 获取rankinfo的内容，用于集合通信感知每层组网的地址信息
 * param[in] phyId   NPU物理ID
 * param[out] rankInfo   rankinfo的内容，为json格式的字符串
 * param[out] bufSize   实际大小
 */
int TopoAddrInfoGet(int phyId, char* rankInfo, size_t* bufSize);

#ifdef __cplusplus
}
#endif

#endif // __RANK_INFO_H__
