/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "recover_info.h"
#include "string_util.h"
#include "invalid_params_exception.h"

namespace Hccl {

static void ReportRecoverInfoCheckFailed(const std::string& paraName, uint32_t localPara, uint32_t remotePara)
{
    THROW<InvalidParamsException>(StringFormat(
        "[recover_info][ReportOpInfoCheckFailed]recover information %s check fail. local[%u], remote[%u]",
        paraName.c_str(), localPara, remotePara));
}

RecoverInfo::RecoverInfo(const RecoverInfoData& recoverInfoData, RankId myRank)
    : recoverInfoData(recoverInfoData),
      myRank(myRank)
{}

RecoverInfo::RecoverInfo(const std::vector<char>& v)
{
    size_t recoverInfoDataSize = sizeof(RecoverInfoData);
    if (v.size() != recoverInfoDataSize) {
        THROW<InternalException>(StringFormat("Vector size does not match RecoverInfoData size"));
    }

    int ret = memcpy_s(&this->recoverInfoData, recoverInfoDataSize, &v[0], v.size());
    if (ret != 0) {
        THROW<InternalException>(StringFormat("Vector size does not match RecoverInfoData size"));
    }
}

std::string RecoverInfo::Describe() const { return recoverInfoData.Describe(); }

std::vector<char> RecoverInfo::GetUniqueId() const
{
    std::vector<char> byteVector = ReCoverCustomTypeToCharVector<RecoverInfoData>(recoverInfoData);

    return byteVector;
}

void RecoverInfo::SetCrcValue(u32 crcValue) { recoverInfoData.crcValue = crcValue; }

// 功能说明：一致性校验
// 输入说明：const std::vector<char> &rmtUniqueId：对端数据
void RecoverInfo::Check(const std::vector<char>& rmtUniqueId) const
{
    RecoverInfoData rmtRecoverInfoData = RecoverCharVectorToCustomType<RecoverInfoData>(rmtUniqueId);
    CompareRecoverInfo(rmtRecoverInfoData);
}

void RecoverInfo::CompareRecoverInfo(const RecoverInfoData& otherRecoverInfoData) const
{
    if (recoverInfoData.collOpIndex != otherRecoverInfoData.collOpIndex) {
        ReportRecoverInfoCheckFailed("collOpIndex", recoverInfoData.collOpIndex, otherRecoverInfoData.collOpIndex);
    } else if (recoverInfoData.crcValue != otherRecoverInfoData.crcValue) {
        ReportRecoverInfoCheckFailed("crcValue", recoverInfoData.crcValue, otherRecoverInfoData.crcValue);
    } else if (recoverInfoData.step != otherRecoverInfoData.step) {
        ReportRecoverInfoCheckFailed("step", recoverInfoData.step, otherRecoverInfoData.step);
    }
    return;
}

} // namespace Hccl
