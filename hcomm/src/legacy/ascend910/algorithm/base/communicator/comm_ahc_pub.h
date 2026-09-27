/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_AHC_PUB_H
#define COMM_AHC_PUB_H

#include <cmath>
#include <algorithm>
#include "device_capacity.h"

namespace hccl {

// AHC算法相关
enum class AHCLevel { AHC_LEVEL_0 = 0, AHC_LEVEL_1, AHC_LEVEL_MAX };

enum class ConcType { CONC_INTRA = 0, CONC_INTER, CONC_RESERVED };

enum class AHCOpType {
    AHC_OP_TYPE_ALLGATHER = 0,
    AHC_OP_TYPE_ALLREDUCE,
    AHC_OP_TYPE_REDUCE_SCATTER,
    AHC_OP_TYPE_RESERVED
};

enum class AHCTemplateType {
    // AHC 内置template类型
    AHC_TEMPLATE_NB = 0,
    AHC_TEMPLATE_RING,
    AHC_TEMPLATE_NHR,
    AHC_TEMPLATE_RESERVED
};

// AHC prepare 扩展参数定义
using OXCPreparePara = struct OXCPrepareParaDef {
    u32 netPlaneId;
    u32 netPlaneNum;
    OXCPrepareParaDef(u32 netPlaneId, u32 netPlaneNum) : netPlaneId(netPlaneId), netPlaneNum(netPlaneNum) {}
};

union AHCExtendPreparePara {
    OXCPreparePara oxcPreparePara;
    AHCExtendPreparePara() {}
};

using AHCConcOpType = struct AHCConcOpTypeDef {
    AHCLevel ahcLevel;
    ConcType concType;
    AHCOpType ahcOpType;

    AHCConcOpTypeDef()
        : ahcLevel(AHCLevel::AHC_LEVEL_0),
          concType(ConcType::CONC_INTRA),
          ahcOpType(AHCOpType::AHC_OP_TYPE_RESERVED)
    {}

    AHCConcOpTypeDef(AHCLevel ahcLevel, ConcType concType, AHCOpType ahcOpType)
        : ahcLevel(ahcLevel),
          concType(concType),
          ahcOpType(ahcOpType)
    {}

    // 重载 == 运算符，用于比较两个 MyKey 对象是否相等
    bool operator==(const AHCConcOpTypeDef& other) const
    {
        return ahcLevel == other.ahcLevel && concType == other.concType && ahcOpType == other.ahcOpType;
    }

    // 重载 < 运算符，用于排序
    bool operator<(const AHCConcOpTypeDef& other) const
    {
        if (ahcLevel != other.ahcLevel) {
            return ahcLevel < other.ahcLevel;
        }
        if (concType != other.concType) {
            return concType < other.concType;
        }
        return ahcOpType < other.ahcOpType;
    }
};
constexpr double AHC_SYM_THRESHOLD = 0.05;
using AHCAlgSelectParam = struct AHCAlgSelectParamDef {
    bool enableOXC;
    bool enableAlgAutoSelect;
    bool enableSubGroupsSplit;
    u64 dataSize;
    AHCOpType opType;
    float symThreshold;

    AHCAlgSelectParamDef()
        : enableOXC(false),
          enableAlgAutoSelect(true),
          enableSubGroupsSplit(true),
          dataSize(0),
          opType(AHCOpType::AHC_OP_TYPE_RESERVED),
          symThreshold(AHC_SYM_THRESHOLD)
    {}
};

} // namespace hccl

#endif /* COMM_AHC_PUB_H */
