/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "checkcrc.h"
#include "log.h"
#include <string>
#include <cstdlib>
#include <iostream>
#include <ios>
#include <fstream>
#include "const_val.h"

namespace Hccl {
constexpr s32 FILE_MAX_LENGTH = 40 * 1024 * 1024; // file max length 40*1024*1024=40M.
CheckCrc::CheckCrc() : initFlag_(0), crcCalcTable{0}, crcTable_(0) {}

CheckCrc::~CheckCrc() {}

HcclResult CheckCrc::AddCrc(u32 crcValue)
{
    HCCL_INFO("crcValue[%u]", crcValue);
    crcTable_.push_back(crcValue);
    HCCL_INFO("num[%llu]", crcTable_.size());
    return HCCL_SUCCESS;
}

HcclResult CheckCrc::GetCrcNum(u32* num)
{
    CHK_PTR_NULL(num);
    HCCL_INFO("num[%u]", *num);
    *num = crcTable_.size();
    return HCCL_SUCCESS;
}

HcclResult CheckCrc::GetCrc(u32 num, u32* crcAddr)
{
    CHK_PTR_NULL(crcAddr);
    HCCL_INFO("num[%u], crc[%u]", num, *crcAddr);

    if (num == 0) {
        HCCL_ERROR("[Get][Crc]errNo[0x%016llx] In get crc the value of num is 0", HCCL_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    if (num != crcTable_.size()) {
        HCCL_ERROR(
            "[Get][Crc]errNo[0x%016llx] num error inputNum[%u], localNum[%llu]", HCCL_ERROR_CODE(HCCL_E_INTERNAL), num,
            crcTable_.size());
        return HCCL_E_INTERNAL;
    }

    for (u32 i = 0; i < num; i++) {
        crcAddr[i] = crcTable_[i];
    }
    return HCCL_SUCCESS;
}

void CheckCrc::InitTable(void)
{
    for (u32 i = 0; i < CRC_TABLE_LENTH; i++) {
        u32 crc = i;
        for (u32 j = 0; j < CRC_CALC_8; j++) {
            if ((crc & 1) != 0) {
                crc = (crc >> 1) ^ CRC_DEFAULT_VALUE;
            } else {
                crc = crc >> 1;
            }
        }
        crcCalcTable[i] = crc;
    }
}

// 构造CRC字符串，按照CRC_NUM CRC1 CRC2 CRC3....,每个数字中间使用空格隔开
std::string CheckCrc::GetString(void)
{
    u32 num = crcTable_.size();
    std::string str;
    str += std::to_string(num);
    str += " ";
    if (num != 0) {
        for (u32 i = 0; i < num; i++) {
            str += std::to_string(crcTable_[i]);
            str += " ";
        }
    }
    HCCL_INFO("str[%s]", str.c_str());
    return str;
}

HcclResult CheckCrc::Calc32Crc(const char* data, u64 length, u32* crcValue)
{
    CHK_PTR_NULL(data);
    CHK_PTR_NULL(crcValue);
    HCCL_INFO("length[%llu], crcValue[%u]", length, *crcValue);
    u32 ret = INVALID_U32;
    if (initFlag_ == 0) {
        InitTable();
        initFlag_ = 1;
    }
    for (u64 i = 0; i < length; i++) {
        ret = crcCalcTable[((ret & 0xFF) ^ static_cast<u8>(data[i]))] ^ (ret >> CRC_CALC_8);
    }

    *crcValue = ~ret;
    return HCCL_SUCCESS;
}

HcclResult CheckCrc::CalcStringCrc(const char* str, u32* crcValue)
{
    CHK_PTR_NULL(str);
    CHK_PTR_NULL(crcValue);
    s32 strLength;

    strLength = strlen(str);
    CHK_PRT_RET(
        strLength <= 0, HCCL_ERROR("[Calc][StringCrc]String is empty, String length[%d].", strLength), HCCL_E_PARA);

    // 计算并设置CRC值
    CHK_RET(this->Calc32Crc(str, static_cast<u64>(strLength), crcValue));

    return HCCL_SUCCESS;
}

HcclResult CheckCrc::ClearCrcInfo(void)
{
    this->crcTable_.clear();
    if ((this->crcTable_.size()) != 0) {
        HCCL_ERROR("[Clear][CrcInfo]errNo[0x%016llx] clear crcTable_ is failed", HCCL_ERROR_CODE(HCCL_E_INTERNAL));
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}
} // namespace Hccl
