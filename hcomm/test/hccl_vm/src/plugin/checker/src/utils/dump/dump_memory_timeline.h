/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_VM_DUMP_MEMORY_TIMELINE_H
#define HCCL_VM_DUMP_MEMORY_TIMELINE_H

#include <cstdint>
#include <map>
#include <nlohmann_json/json.hpp>
#include <unordered_map>
#include <vector>

#include "task_check_op_semantics.h"

namespace HcclSim {
class MemoryTimelineDumpStream {
public:
    HcclResult Initialize(
        const std::vector<RankId>& rankIds, const std::vector<TimelineEvent>& timelineEvents,
        const std::map<u32, u32>& globalStepToEventId);
    HcclResult AppendSnapshot(
        RankId rankId, u32 logicalStep, const std::vector<std::string>& memoryTaskIds,
        const RankMemorySemantics& fullLayout);
    HcclResult Finalize();

private:
    struct RankDumpState {
        size_t nextChunkId = 0;
        size_t totalSnapshotCount = 0;
        size_t openChunkSnapshotCount = 0;
        u32 openChunkBeginStep = 0;
        u32 openChunkEndStep = 0;
        std::vector<uint8_t> openChunkSnapshotEntries;
        nlohmann::json manifestChunks = nlohmann::json::array();
    };

    HcclResult FlushOpenChunk(RankId rankId, RankDumpState& rankDumpState);

    bool m_enabled = false;
    std::unordered_map<u32, const TimelineEvent*> m_globalStepToEvent;
    std::map<RankId, RankDumpState> m_rankDumpStates;
};
} // namespace HcclSim

#endif // HCCL_VM_DUMP_MEMORY_TIMELINE_H
