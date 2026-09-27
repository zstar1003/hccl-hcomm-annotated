/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_info_recorder.h"

namespace checker {
RankInfoRecorder* RankInfoRecorder::Global()
{
    static RankInfoRecorder* rankInfoRecorder = new RankInfoRecorder;
    return rankInfoRecorder;
}

void RankInfoRecorder::Reset()
{
    rankId2phyId.clear();
    rankId2serverId.clear();
    rankId2superpodId.clear();
    return;
}

void RankInfoRecorder::SetRankId(RankId rankId)
{
    curRankId = rankId;
    return;
}

RankId RankInfoRecorder::GetRankId() { return curRankId; }

void RankInfoRecorder::SetDevType(CheckerDevType devType)
{
    curDevType = devType;
    return;
}

CheckerDevType RankInfoRecorder::GetDevType() { return curDevType; }

u32 RankInfoRecorder::GetRankSize() { return rankSize_; }

void RankInfoRecorder::InitRankInfo(TopoMeta topoMeta, CheckerDevType uniDevType)
{
    u32 myRankId = 0;
    for (int i = 0; i < topoMeta.size(); i++) {
        for (int j = 0; j < topoMeta[i].size(); j++) {
            for (int k = 0; k < topoMeta[i][j].size(); k++) {
                rankId2phyId[myRankId] = topoMeta[i][j][k];
                rankId2serverId[myRankId] = j;
                rankId2superpodId[myRankId] = i;
                myRankId++;
            }
        }
    }
    rankSize_ = myRankId;
}

} // namespace checker
