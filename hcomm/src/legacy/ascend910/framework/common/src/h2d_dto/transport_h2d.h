/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H2D_DTO_TRANSPORT_H2D_H
#define H2D_DTO_TRANSPORT_H2D_H

#include "hccl/base.h"

// 全部 Transport QP/Mem 信息
struct HcclAiRMAInfo {
    u32 curRankId; // 当前rankId
    u32 rankNum;   // rank数量
    u32 qpNum;     // 单个Transport的QP数量

    u32 sizeOfAiRMAWQ;  // sizeof(HcclAiRMAWQ)
    u32 sizeOfAiRMACQ;  // sizeof(HcclAiRMACQ)
    u32 sizeOfAiRMAMem; // sizeof(HcclAiRMAMemInfo)

    // HcclAiRMAWQ二维数组首地址
    // QP个数: rankNum * qpNum
    // 计算偏移获取SQ指针：sqPtr + dstRankId * qpNum * sizeOfAiRMAWQ + qpIndex
    void* sqPtr;

    // HcclAiRMACQ二维数组首地址
    // QP个数: rankNum * qpNum
    // 计算偏移获取SCQ指针：scqPtr + dstRankId * qpNum * sizeOfAiRMACQ + qpIndex
    void* scqPtr;

    // HcclAiRMAWQ二维数组首地址
    // QP个数: rankNum * qpNum
    // 计算偏移获取RQ指针：rqPtr + dstRankId * qpNum * sizeOfAiRMAWQ + qpIndex
    void* rqPtr;

    // HcclAiRMACQ二维数组首地址
    // QP个数: rankNum * qpNum
    // 计算偏移获取RCQ指针: rcqPtr + dstRankId * qpNum * sizeOfAiRMACQ + qpIndex
    void* rcqPtr;

    // HcclAivMemInfo一维数组
    // 内存信息个数: rankNum
    // 计算偏移获取内存信息指针: memPtr + rankId * sizeOfAiRMAMem
    // srcRankId 获取自身内存信息，dstRankId 获取 Transport 内存信息
    void* memPtr;

    // 可往后追加字段

    HcclAiRMAInfo()
        : curRankId(0),
          rankNum(0),
          qpNum(0),
          sizeOfAiRMAWQ(0),
          sizeOfAiRMACQ(0),
          sizeOfAiRMAMem(0),
          sqPtr(nullptr),
          scqPtr(nullptr),
          rqPtr(nullptr),
          rcqPtr(nullptr),
          memPtr(nullptr)
    {}
};

#endif
