/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INC_CRC_CHECK_H
#define HCCL_INC_CRC_CHECK_H

#include <vector>
#include <hccl/hccl_types.h>
#include "hccl/base.h"

namespace Hccl {
constexpr int CRC_TABLE_LENTH = 256;
constexpr u32 CRC_DEFAULT_VALUE = 0xEDB88320;
constexpr u32 CRC_CALC_8 = 8;

class CheckCrc {
public:
    explicit CheckCrc();

    virtual ~CheckCrc();

    // 增加一个CRC值
    HcclResult AddCrc(u32 crcValue);

    // 获取num个CRC值
    HcclResult GetCrc(u32 num, u32* crcAddr);

    // 获取CRC的数目
    HcclResult GetCrcNum(u32* num);

    // 根据输入的字符串指针和长度计算CRC值
    HcclResult Calc32Crc(const char* data, u64 length, u32* crcValue);

    // 根据输入的String内容，计算CRC值
    HcclResult CalcStringCrc(const char* str, u32* crcValue);

    // 将本地的CRC值和NUM组成字符串，格式为CRC_NUM CRC1 CRC2 CRC3...
    std::string GetString(void);

    // 单次模型结束后将所有CRC信息删除
    HcclResult ClearCrcInfo(void);

private:
    void InitTable(void);

private:
    bool initFlag_;
    u32 crcCalcTable[CRC_TABLE_LENTH];
    std::vector<u32> crcTable_;
};
} // namespace Hccl

#endif // HCCL_SRC_CRC_CHECK_H
