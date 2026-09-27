/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _SIM_HCCL_PROXY_COMMON_H_
#define _SIM_HCCL_PROXY_COMMON_H_

#include "hccl/hccl_types.h"
#include <string>
#include <map>

#include <cstdint>
#include <map>

namespace sim {
inline const std::map<HcclDataType, uint32_t> DATA_TYPE_SIZE_MAP
    = {{HcclDataType::HCCL_DATA_TYPE_INT8, 1},    {HcclDataType::HCCL_DATA_TYPE_INT16, 2},
       {HcclDataType::HCCL_DATA_TYPE_INT32, 4},   {HcclDataType::HCCL_DATA_TYPE_FP16, 2},
       {HcclDataType::HCCL_DATA_TYPE_FP32, 4},    {HcclDataType::HCCL_DATA_TYPE_INT64, 8},
       {HcclDataType::HCCL_DATA_TYPE_UINT64, 8},  {HcclDataType::HCCL_DATA_TYPE_UINT8, 1},
       {HcclDataType::HCCL_DATA_TYPE_UINT16, 2},  {HcclDataType::HCCL_DATA_TYPE_UINT32, 4},
       {HcclDataType::HCCL_DATA_TYPE_FP64, 8},    {HcclDataType::HCCL_DATA_TYPE_BFP16, 2},
       {HcclDataType::HCCL_DATA_TYPE_INT128, 16}, {HcclDataType::HCCL_DATA_TYPE_HIF8, 1},
       {HcclDataType::HCCL_DATA_TYPE_FP8E4M3, 1}, {HcclDataType::HCCL_DATA_TYPE_FP8E5M2, 1},
       {HcclDataType::HCCL_DATA_TYPE_FP8E8M0, 1}};

inline int GetDataTypeSize(HcclDataType dataType, uint32_t& size)
{
    auto iter = DATA_TYPE_SIZE_MAP.find(dataType);
    if (iter == DATA_TYPE_SIZE_MAP.end()) {
        return 1;
    }
    size = iter->second;
    return 0;
}

bool IsDeviceAddress(void* addr);

bool ParseKernelJson(const std::string& jsonPath, std::map<std::string, std::string>& out);

} // namespace sim
#endif
