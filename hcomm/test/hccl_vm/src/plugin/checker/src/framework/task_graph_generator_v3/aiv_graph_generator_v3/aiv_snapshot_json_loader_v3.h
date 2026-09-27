/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECKER_TASK_GRAPH_GENERATOR_V3_AIV_SNAPSHOT_JSON_LOADER_V3_H
#define CHECKER_TASK_GRAPH_GENERATOR_V3_AIV_SNAPSHOT_JSON_LOADER_V3_H

#include <cstdint>
#include <string>
#include <vector>

#include "hccl_types.h"
#include "../task_def_v3.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {

    enum class AivRuntimeTaskTypeV3 : uint32_t {
        MEM_COPY = 0,
        REDUCE,
        SET_FLAG,
        WAIT_FLAG,
        PIPE_BARRIER,
        SYNC_ALL,
        SEND_FLAG,
        RECV_FLAG,
        INVALID_TYPE,
    };

    enum class AivBufferTypeV3 : uint32_t {
        INPUT = 0,
        OUTPUT,
        CCL,
        UB,
        AIV_COMM,
        INVALID,
    };

    struct AivDataSliceV3 {
        AivBufferTypeV3 type{AivBufferTypeV3::INVALID};
        uint64_t offset{0};
        uint64_t size{0};
    };

    struct AivRuntimeTaskV3 {
        AivRuntimeTaskTypeV3 taskType{AivRuntimeTaskTypeV3::INVALID_TYPE};
        uint32_t taskId{0};
        RankId rankId{INVALID_RANK_ID};
        uint32_t blockId{0};
        uint32_t curPipe{0};

        RankId srcRank{INVALID_RANK_ID};
        AivDataSliceV3 src;
        RankId dstRank{INVALID_RANK_ID};
        AivDataSliceV3 dst;
        uint32_t dataType{0};
        uint32_t reduceOp{0};

        uint32_t srcPipe{0};
        uint32_t dstPipe{0};
        int32_t eventId{0};

        uint32_t pipeType{0};
        std::vector<uint32_t> barrierGroupTaskIds;
        uint32_t syncRound{0};

        RankId flagOwnerRank{INVALID_RANK_ID};
        uint64_t commInfoOffset{0};
        int32_t flagValue{0};
    };

    struct AivRuntimeBlockSnapshotV3 {
        uint32_t blockIdx{0};
        std::vector<AivRuntimeTaskV3> scalarTasks;
        std::vector<AivRuntimeTaskV3> mte2Tasks;
        std::vector<AivRuntimeTaskV3> mte3Tasks;
    };

    struct AivRuntimeTaskSnapshotV3 {
        RankId rankId{INVALID_RANK_ID};
        uint64_t launchIndex{0};
        uint32_t rankSize{0};
        uint64_t inBufferSize{0};
        uint64_t outBufferSize{0};
        uint64_t cclBufferSize{0};
        uint64_t aivCommInfoSize{0};
        uint64_t ubBufferSize{0};
        std::string filePath;
        std::vector<AivRuntimeBlockSnapshotV3> blocks;
    };

    class AivSnapshotJsonLoaderV3 {
    public:
        AivSnapshotJsonLoaderV3() = default;
        ~AivSnapshotJsonLoaderV3() = default;

        HcclResult LoadByRankAndLaunch(
            RankId rankId, uint64_t launchIndex, AivRuntimeTaskSnapshotV3& snapshot, std::string& errorMessage) const;
    };

} // namespace TaskGraphGeneratorV3
} // namespace HcclSim

#endif // CHECKER_TASK_GRAPH_GENERATOR_V3_AIV_SNAPSHOT_JSON_LOADER_V3_H
