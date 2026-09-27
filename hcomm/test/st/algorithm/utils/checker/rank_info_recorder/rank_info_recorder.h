/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV1_RANK_INFO_RECORDER_H
#define HCCLV1_RANK_INFO_RECORDER_H
#include <map>
#include "llt_common.h"
#include "topo_meta.h"
#include "checker_def.h"

using namespace hccl;

namespace checker {

class RankInfoRecorder {
public:
    static RankInfoRecorder* Global();
    void Reset();

    void SetRankId(RankId rankId);
    RankId GetRankId();
    void SetDevType(CheckerDevType devType);
    CheckerDevType GetDevType();
    u32 GetRankSize();

    void InitRankInfo(TopoMeta topoMeta, CheckerDevType uniDevType);

    RankId curRankId = 0;
    CheckerDevType curDevType = CheckerDevType::DEV_TYPE_NOSOC;

    std::map<u32, u32> rankId2phyId;
    std::map<u32, u32> rankId2serverId;
    std::map<u32, u32> rankId2superpodId;
    u32 rankSize_ = 0;
};

} // namespace checker

#endif
