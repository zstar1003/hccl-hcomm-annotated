/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "type_conversion.h"

#include <cstdint>

namespace HcclSim {
std::map<DataType, HcclDataType> g_DataType2CheckerDataType_aicpu = {
    {DataType::INT8, HcclDataType::HCCL_DATA_TYPE_INT8},
    {DataType::INT16, HcclDataType::HCCL_DATA_TYPE_INT16},
    {DataType::INT32, HcclDataType::HCCL_DATA_TYPE_INT32},
    {DataType::FP16, HcclDataType::HCCL_DATA_TYPE_FP16},
    {DataType::FP32, HcclDataType::HCCL_DATA_TYPE_FP32},
    {DataType::UINT64, HcclDataType::HCCL_DATA_TYPE_UINT64},
    {DataType::UINT8, HcclDataType::HCCL_DATA_TYPE_UINT8},
    {DataType::UINT16, HcclDataType::HCCL_DATA_TYPE_UINT16},
    {DataType::UINT32, HcclDataType::HCCL_DATA_TYPE_UINT32},
    {DataType::FP64, HcclDataType::HCCL_DATA_TYPE_FP64},
    {DataType::BFP16, HcclDataType::HCCL_DATA_TYPE_BFP16},
    {DataType::INT128, HcclDataType::HCCL_DATA_TYPE_INT128},
    {DataType::INT64, HcclDataType::HCCL_DATA_TYPE_INT64},
    {DataType::HIF8, HcclDataType::HCCL_DATA_TYPE_HIF8},
    {DataType::FP8E4M3, HcclDataType::HCCL_DATA_TYPE_FP8E4M3},
    {DataType::FP8E5M2, HcclDataType::HCCL_DATA_TYPE_FP8E5M2},
};

std::map<uint16_t, HcclReduceOp> g_ReduceOp2CheckerReduceOp_ccu
    = {{10, HcclReduceOp::HCCL_REDUCE_SUM}, {9, HcclReduceOp::HCCL_REDUCE_MIN}, {8, HcclReduceOp::HCCL_REDUCE_MAX}};

} // namespace HcclSim
