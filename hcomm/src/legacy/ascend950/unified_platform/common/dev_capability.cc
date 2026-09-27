/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dev_capability.h"
#include "not_support_exception.h"
#include "string_util.h"
namespace Hccl {
using namespace std;
const u64 RDMA_SEND_MAX_SIZE = 0x80000000;  // 节点间RDMA发送数据单个WQE支持的最大数据量
const u64 SDMA_SEND_MAX_SIZE = 0x100000000; // 节点内单个SDMA任务发送数据支持的最大数据量
const map<DataType, bool> CAP_INLINE_REDUCE_DATATYPE_910A = {
    {DataType::INT8, false},   {DataType::INT16, false},  {DataType::INT32, false},  {DataType::FP16, false},
    {DataType::FP32, true},    {DataType::INT64, false},  {DataType::UINT64, false}, {DataType::UINT8, false},
    {DataType::UINT16, false}, {DataType::UINT32, false}, {DataType::FP64, false},   {DataType::BFP16, false},
    {DataType::INT128, false},
};
const map<ReduceOp, bool> CAP_INLINE_REDUCE_OP_910A
    = {{ReduceOp::SUM, true}, {ReduceOp::PROD, false}, {ReduceOp::MAX, false}, {ReduceOp::MIN, false}};
const u32 CAP_NOTIFY_SIZE_910A = 8;
const u32 CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_910A = 128;
const map<DataType, bool> CAP_INLINE_REDUCE_DATATYPE_910A3 = {
    {DataType::INT8, true},    {DataType::INT16, true},   {DataType::INT32, true},   {DataType::FP16, true},
    {DataType::FP32, true},    {DataType::INT64, false},  {DataType::UINT64, false}, {DataType::UINT8, false},
    {DataType::UINT16, false}, {DataType::UINT32, false}, {DataType::FP64, false},   {DataType::BFP16, true},
    {DataType::INT128, false},
};
const map<ReduceOp, bool> CAP_INLINE_REDUCE_OP_910A3
    = {{ReduceOp::SUM, true}, {ReduceOp::PROD, false}, {ReduceOp::MAX, true}, {ReduceOp::MIN, true}};
const u32 CAP_NOTIFY_SIZE_910A3 = 4;
const u32 CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_910A3 = 32;

const map<DataType, bool> CAP_INLINE_REDUCE_DATATYPE_V82 = {
    {DataType::INT8, true},    {DataType::INT16, true},    {DataType::INT32, true},   {DataType::FP16, true},
    {DataType::FP32, true},    {DataType::INT64, false},   {DataType::UINT64, false}, {DataType::UINT8, true},
    {DataType::UINT16, true},  {DataType::UINT32, true},   {DataType::FP64, false},   {DataType::BFP16, true},
    {DataType::INT128, false}, {DataType::BF16_SAT, true},
};
const map<ReduceOp, bool> CAP_INLINE_REDUCE_OP_V82
    = {{ReduceOp::SUM, true},
       {ReduceOp::PROD, false},
       {ReduceOp::MAX, true},
       {ReduceOp::MIN, true},
       {ReduceOp::EQUAL, true}};
const u32 CAP_NOTIFY_SIZE_V82 = 8;
const u32 CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_V82 = 32;

DevCapability& DevCapability::GetInstance()
{
    static DevCapability devCapability;
    return devCapability;
}

DevCapability::DevCapability() {}

void DevCapability::Init(DevType givenDevType)
{
    if (isInit) {
        return;
    }
    isInit = true;
    devType = givenDevType;
    if (devType == DevType::DEV_TYPE_910A) {
        Load910ACap();
    } else if (devType == DevType::DEV_TYPE_910A3 || devType == DevType::DEV_TYPE_910A2) {
        Load910A3Cap();
    } else if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        LoadV82Cap();
    } else {
        throw NotSupportException(StringFormat("we don't support %s now.", devType.Describe().c_str()));
    }
}

void DevCapability::Reset() { isInit = false; }

void DevCapability::Load910A910A3CommonCap()
{
    isSupportWriteWithNotify = false;
    isSupportStarsPollNetCq = false;
    sdmaSendMaxSize = SDMA_SEND_MAX_SIZE;
    rdmaSendMaxSize = RDMA_SEND_MAX_SIZE;
}

void DevCapability::Load910A3Cap()
{
    Load910A910A3CommonCap();
    inlineReduceDataTypeMap = CAP_INLINE_REDUCE_DATATYPE_910A3;
    inlineReduceOpMap = CAP_INLINE_REDUCE_OP_910A3;
    sdmaInlineReduceAlignBytes = CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_910A3;
    notifySize = CAP_NOTIFY_SIZE_910A3;
    isSupportDevNetInlineReduce = true;
}

void DevCapability::Load910ACap()
{
    Load910A910A3CommonCap();
    inlineReduceDataTypeMap = CAP_INLINE_REDUCE_DATATYPE_910A;
    inlineReduceOpMap = CAP_INLINE_REDUCE_OP_910A;
    sdmaInlineReduceAlignBytes = CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_910A;
    notifySize = CAP_NOTIFY_SIZE_910A;
    isSupportDevNetInlineReduce = false;
}

void DevCapability::LoadV82Cap()
{
    isSupportWriteWithNotify = true;
    isSupportStarsPollNetCq = true;
    sdmaSendMaxSize = SDMA_SEND_MAX_SIZE;
    rdmaSendMaxSize = RDMA_SEND_MAX_SIZE;

    inlineReduceDataTypeMap = CAP_INLINE_REDUCE_DATATYPE_V82;
    inlineReduceOpMap = CAP_INLINE_REDUCE_OP_V82;
    sdmaInlineReduceAlignBytes = CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_V82;
    notifySize = CAP_NOTIFY_SIZE_V82;
    isSupportDevNetInlineReduce = true;
}

} // namespace Hccl
