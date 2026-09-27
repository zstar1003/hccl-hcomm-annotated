/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARAM_CHECK_PUB_V2_H
#define PARAM_CHECK_PUB_V2_H

#include <string>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "hccl_common_v2.h"

// reduction op合法性检测
HcclResult HcomCheckReductionOpV2(const HcclReduceOp op);
// Reduce data type合法性检测
HcclResult HcomCheckReduceDataTypeV2(const HcclDataType dataType, const HcclReduceOp op);

HcclResult HcomCheckDataTypeV2(const HcclDataType dataType);

// tag合法性检测
HcclResult HcomCheckTagV2(const char* tag);

// group name合法性检测
HcclResult HcomCheckGroupNameV2(const char* group = nullptr);

// op param合法性检测
HcclResult HcomCheckOpParamV2(
    const char* tag, const u64 count, const HcclDataType dataType, const char* group, const void* stream);

HcclResult HcomCheckOpParamV2(const u64 count, const HcclDataType dataType, const char* group);

// pytorch 通信域适配 参数合法性检测
HcclResult HcomCheckOpParamV2(const char* tag, const u64 count, const HcclDataType dataType, const void* stream);

HcclResult HcomCheckOpParamV2(const char* tag, const u64 count, const HcclDataType dataType);

HcclResult HcomCheckOpParamV2(const u64 count, const HcclDataType dataType);
HcclResult HcomCheckCountV2(const u64 count);

std::string GetDataTypeEnumStrV2(HcclDataType dataType);
std::string GetReduceOpEnumStrV2(HcclReduceOp reduceOp);

// AlltoAllV count和buff合法性检测
HcclResult HcomCheckAlltoAllVExternalMemV2(
    const void* sendBuf, const void* sendCounts, const void* recvBuf, const void* recvCounts, u32 rankSize);

// AlltoAllVC count和buff合法性检测
HcclResult HcomCheckAlltoAllVCExternalMemV2(
    const void* sendBuf, const void* sendCountMatrix, const void* recvBuf, u32 rankSize, u32 rank);
// AlltoAllVC Matrix传输值是否为空
HcclResult HcomCheckAlltoAllVCEmptyV2(
    const void* sendBuf, const void* sendCountMatrix, const void* recvBuf, u32 rankSize, bool& isEmpty);

HcclResult HcomCheckUserRankV2(const u32 totalRanks, const u32 userRank);

// 读取RankTable文件 并进行一些必要校验
HcclResult HcomLoadRankTableFileV2(const char* clusterInfo, std::string& rankTableM);

HcclResult HcomCheckVOpParamV2(u32 rankId, u32 rankSize, u64 count, void* inCounts);

// 打印sendCountMatrix信息
void HcomGetHashFromSendCountMatrixV2(
    u64& sendCountMatrixHash, const void* sendCountMatrix, u64 rankSize, const std::string& tag);
#endif
