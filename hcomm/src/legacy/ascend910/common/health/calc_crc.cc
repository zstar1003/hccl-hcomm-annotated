/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "calc_crc.h"
#include <string>
#include <cstdlib>
#include <iostream>
#include <ios>
#include <fstream>
#include <mutex>
#include "hccl_common.h"

namespace hccl {
constexpr int CRC_TABLE_LENGTH = 256;
constexpr u32 CRC_DEFAULT_VALUE = 0xEDB88320;
constexpr u32 CRC_CALC_8 = 8;
constexpr u32 CRC_CALC_10 = 10;
constexpr s32 STRING_MAX_LENGTH = 40 * 1024 * 1024;
u32 g_crcCalcTable[CRC_TABLE_LENGTH];

HcclResult CalcCrc::HcclCalcCrc(const char* data, u64 length, u32& crcValue)
{
    CHK_PTR_NULL(data);

    CHK_PRT_RET(
        length <= 0 || length > STRING_MAX_LENGTH,
        HCCL_ERROR(
            "[Calc][StringCrc]String length[%llu] is empty or greater than %d bytes.", length, STRING_MAX_LENGTH),
        HCCL_E_PARA);
    HCCL_DEBUG("data[%s], length[%llu]", data, length);

    // 计算并设置CRC值
    u32 ret = INVALID_UINT;
    for (u64 i = 0; i < length; i++) {
        ret = g_crcCalcTable[((ret & 0xFF) ^ static_cast<u8>(data[i]))] ^ (ret >> CRC_CALC_8);
    }

    crcValue = ~ret;
    return HCCL_SUCCESS;
}

__attribute__((constructor)) void InitTable()
{
    for (u32 i = 0; i < CRC_TABLE_LENGTH; i++) {
        u32 crc = i;
        for (u32 j = 0; j < CRC_CALC_8; j++) {
            if ((crc & 1) != 0) {
                crc = (crc >> 1) ^ CRC_DEFAULT_VALUE;
            } else {
                crc = crc >> 1;
            }
        }
        g_crcCalcTable[i] = crc;
    }
}
} // namespace hccl
