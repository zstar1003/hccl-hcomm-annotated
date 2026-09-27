/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RECOVER_INFO_H
#define HCCLV2_RECOVER_INFO_H

#include <string>
#include <vector>
#include "types.h"
#include "string_util.h"
#include "exception_util.h"
#include "internal_exception.h"
namespace Hccl {

const u32 RECOVER_OP_TAG_MAX_LEN = 191; // 最大的tag长度

template <typename T>
inline std::vector<char> ReCoverCustomTypeToCharVector(const T& data)
{
    std::vector<char> v;
    v.resize(sizeof(T));
    int ret = memcpy_s(v.data(), v.size(), &data, sizeof(T));
    if (ret != 0) {
        THROW<InternalException>(StringFormat("CustomTypeToCharVector copy dwqe failed, ret=%d", ret));
    }
    return v;
}

template <typename T>
inline T RecoverCharVectorToCustomType(const std::vector<char>& v)
{
    T result;
    int ret = memcpy_s(&result, sizeof(result), &v[0], v.size());
    if (ret != 0) {
        THROW<InternalException>(StringFormat("VectorByteToCustomType copy dwqe failed, ret=%d", ret));
    }
    return result;
}

struct RecoverInfoData {
    // 通信算子数目
    u32 collOpIndex{0};
    // RankTable CRC值
    u32 crcValue{0};
    // 通信步骤
    u32 step{0};

    RecoverInfoData() {}

    RecoverInfoData(u32 collOpIndex, u32 crcValue, u32 step) : collOpIndex(collOpIndex), crcValue(crcValue), step(step)
    {
        this->collOpIndex = collOpIndex;
        this->crcValue = crcValue;
        this->step = step;
    }
    std::string Describe() const
    {
        return StringFormat("RecoverInfo[CollOpIndex=%u,CrcValue=%u,Step=%u]", collOpIndex, crcValue, step);
    }
};

class RecoverInfo {
public:
    explicit RecoverInfo(const RecoverInfoData& recoverInfoData, RankId myRank);

    explicit RecoverInfo(const std::vector<char>& v);

    std::string Describe() const;

    // 获取当前RecoverInfoData的唯一标识
    std::vector<char> GetUniqueId() const;

    void SetCrcValue(u32 crcValue);

    void Check(const std::vector<char>& rmtUniqueId) const;

    void CompareRecoverInfo(const RecoverInfoData& otherRecoverInfoData) const;

private:
    RecoverInfoData recoverInfoData;
    RankId myRank{0};
};

} // namespace Hccl

#endif // HCCLV2_RECOVER_INFO_H
