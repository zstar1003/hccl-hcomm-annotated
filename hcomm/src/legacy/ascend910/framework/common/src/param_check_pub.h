/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARAM_CHECK_PUB_H
#define PARAM_CHECK_PUB_H

#include <string>
#include <hccl/hccl_types.h>

#include "hccl/base.h"
#include "comm.h"
#include "topoinfo_struct.h"

// ranktable文件路径合法性检测
HcclResult HcomGetRanktableRealPath(const char* rankTable, std::string& realFilePath);

// ranktable磁盘文件读取到内存中
HcclResult HcomReadRanktable(const char* rankTable, std::string& rankTableM);

// ranktable内存文件合法性检测
HcclResult HcomCheckRankTable(const char* rankTableM, u32& rankTableSize);

// ranktable磁盘文件处理
HcclResult HcomLoadRanktableFile(const char* rankTablePath, std::string& rankTableM, std::string& realFilePath);

// ranktableCRC计算
HcclResult HcomCalcCRC(hccl::HcclCommParams& params, const char* rankTable);

// identify合法性检测
HcclResult HcomCheckIdentify(const char* identify);

// device id合法性检测
HcclResult HcomCheckDeviceId(const u32 device_id);

// tag合法性检测
HcclResult HcomCheckTag(const char* tag);

// count合法性检测
HcclResult HcomCheckCount(const u64 count);

// AlltoAllV count和buff合法性检测
HcclResult HcomCheckAlltoAllVExternalMem(
    const void* sendBuf, const void* sendCounts, const void* recvBuf, const void* recvCounts, u32 rankSize);

// AlltoAllVC count和buff合法性检测
HcclResult HcomCheckAlltoAllVCExternalMem(
    const void* sendBuf, const void* sendCountMatrix, const void* recvBuf, u32 rankSize, u32 rank);

// 打印sendCountMatrix信息
void HcomGetHashFromSendCountMatrix(
    u64& sendCountMatrixHash, const void* sendCountMatrix, u64 rankSize, const std::string& tag);

// data type合法性检测
HcclResult HcomCheckDataType(const HcclDataType dataType);

// group name合法性检测
HcclResult HcomCheckGroupName(const char* group = nullptr);

// reduction op合法性检测
HcclResult HcomCheckReductionOp(const std::string& callerOpName, const HcclReduceOp op);

// Reduce data type合法性检测
HcclResult HcomCheckReduceDataType(const HcclDataType dataType, const HcclReduceOp op, DevType deviceType);

// user rank合法性检测
HcclResult HcomCheckUserRank(const u32 totalRanks, const u32 userRank);

// op param合法性检测
HcclResult
HcomCheckOpParam(const char* tag, const u64 count, const HcclDataType dataType, const char* group, const void* stream);

// pytorch 通信域适配 参数合法性检测
HcclResult HcomCheckOpParam(const char* tag, const u64 count, const HcclDataType dataType, const void* stream);

HcclResult HcomCheckOpParam(const char* tag, const u64 count, const HcclDataType dataType);

HcclResult HcclParseRanktable(
    const std::string& rankTableM, const std::string& identify, hccl::HcclCommParams& params,
    hccl::RankTable_t& rankTable);

#include "param_check_basic_v2.h"
bool IsSupportHCCLV2(const char* socNamePtr);

#endif // PARAM_CHECK_PUB_H
