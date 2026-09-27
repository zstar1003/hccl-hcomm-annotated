/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allgather_semantics_checker.h"

#include <map>

#include "check_utils.h"
#include "sim_common.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
HcclResult TaskCheckAllGatherSemantics(std::map<RankId, RankMemorySemantics>& allRankMemSemantics, u64 dataSize)
{
    u32 rankSize = allRankMemSemantics.size();

    for (RankId rankId = 0; rankId < rankSize; rankId++) {
        // 对应的rank不存在需要报错
        if (allRankMemSemantics.count(rankId) == 0) {
            HCCL_VM_ERROR(
                "{} AllGather produced no result data for rank {}, but this rank is "
                "expected to receive data from all {} participating ranks with an expected total "
                "result size of 0x{:x} bytes.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, rankSize, dataSize * rankSize);
            return HcclResult::HCCL_E_PARA;
        }

        u64 totalSize = 0;
        RankId curRankId = 0;
        u64 curDataSize = 0;
        for (auto& ele : allRankMemSemantics[rankId][BufferType::OUTPUT]) {
            const u64 rangeEnd = ele.startAddr + ele.size;
            if (ele.startAddr != totalSize) {
                HCCL_VM_ERROR(
                    "{} AllGather output for rank {} should continue at 0x{:x}, "
                    "but the next actual range starts at 0x{:x} (actual range: [0x{:x},0x{:x}))."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, totalSize, ele.startAddr,
                    ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != 1) {
                HCCL_VM_ERROR(
                    "{} AllGather output range [0x{:x},0x{:x}) for rank {} should come "
                    "from exactly one source, but it actually comes from {} sources."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, rankId,
                    ele.srcBufs.size(), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            const auto& srcBuf = *ele.srcBufs.begin();

            if (srcBuf.rankId != curRankId) {
                HCCL_VM_ERROR(
                    "{} AllGather output range [0x{:x},0x{:x}) for rank {} should come from "
                    "rank {}, but it actually comes from rank {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, rankId, curRankId,
                    srcBuf.rankId, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.bufType != BufferType::INPUT) {
                HCCL_VM_ERROR(
                    "{} AllGather output range [0x{:x},0x{:x}) for rank {} should come from "
                    "INPUT, but it actually comes from rank {} with buffer type {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, rankId,
                    srcBuf.rankId, BufferTypeToString(srcBuf.bufType), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.srcAddr != curDataSize) {
                HCCL_VM_ERROR(
                    "{} AllGather output range [0x{:x},0x{:x}) for rank {} should come from "
                    "rank {} at source address 0x{:x}, but the actual source address is 0x{:x}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, rankId,
                    srcBuf.rankId, curDataSize, srcBuf.srcAddr, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            curDataSize += ele.size;
            if (curDataSize == dataSize) {
                curDataSize = 0;
                curRankId++;
            } else if (curDataSize > dataSize) {
                HCCL_VM_ERROR(
                    "{} AllGather data collected from rank {} for rank {} becomes larger "
                    "than expected after outputRange [0x{:x},0x{:x}). The accumulated size is 0x{:x}, "
                    "but the expected size from this source rank is 0x{:x}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), curRankId, rankId, ele.startAddr, rangeEnd,
                    curDataSize, dataSize, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += ele.size;
        }
        if (totalSize != dataSize * rankSize) {
            HCCL_VM_ERROR(
                "{} AllGather output for rank {} ends too early: the checker validated "
                "0x{:x} bytes in total, but the expected total result size is 0x{:x} bytes.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, totalSize, dataSize * rankSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
