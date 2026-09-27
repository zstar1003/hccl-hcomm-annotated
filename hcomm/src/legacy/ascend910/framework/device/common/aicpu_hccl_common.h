/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_HCCL_COMMON_H__
#define __AICPU_HCCL_COMMON_H__

#include "profiling_manager_device.h"
#include "dtype_common.h"
#include "log.h"
#include "sal_pub.h"

inline uint32_t DataUnitSize(HcclDataType dataType)
{
    if (dataType >= HCCL_DATA_TYPE_RESERVED) {
        HCCL_ERROR(
            "[dataUnitSize]data type[%s] out of range[%d, %d]", GetDataTypeEnumStr(dataType).c_str(),
            HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_RESERVED - 1);
        return 0;
    }

    return SIZE_TABLE[dataType];
}

#define NSEC_PER_SEC 1000000000U

inline u64 GetCurCpuTimestamp(bool isProfTime = false)
{
#ifndef CCL_LLT
    if (isProfTime && dfx::ProfilingManager::GetProfL1State()) {
        uint64_t cntvct;
        AsmCntvc(cntvct);
        return cntvct;
    } else {
        struct timespec timestamp;
        (void)clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp);
        return static_cast<u64>((timestamp.tv_sec * NSEC_PER_SEC) + (timestamp.tv_nsec));
    }
#else
    struct timespec timestamp;
    (void)clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp);
    return static_cast<u64>((timestamp.tv_sec * NSEC_PER_SEC) + (timestamp.tv_nsec));
#endif
}

#endif // __AICPU_HCCL_COMMON_HPP__
