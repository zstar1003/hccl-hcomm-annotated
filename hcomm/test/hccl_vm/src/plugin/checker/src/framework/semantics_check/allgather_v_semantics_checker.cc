/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allgather_v_semantics_checker.h"

#include <map>

#include "checker_def.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
HcclResult
TaskCheckAllGatherVSemantics(std::map<RankId, RankMemorySemantics>& allRankMemSemantics, VDataDesTagInner& vDataDes)
{
    u32 rankSize = allRankMemSemantics.size();

    u64 outputSize = 0;
    // AllGatherV 输入不等长
    for (u32 i = 0; i < rankSize; i++) {
        u64 curCounts = vDataDes.counts[i];
        u64 curLength = curCounts * CHECK_SIZE_TABLE[vDataDes.dataType];
        outputSize += curLength;
    }

    for (RankId rankId = 0; rankId < rankSize; rankId++) {
        // 对应的rank不存在需要报错
        if (allRankMemSemantics.count(rankId) == 0) {
            HCCL_VM_ERROR(
                "{} AllGatherV produced no result data for rank {}, but this rank is "
                "expected to receive data from all {} participating ranks with an expected total "
                "result size of 0x{:x} bytes.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, rankSize, outputSize);
            return HcclResult::HCCL_E_PARA;
        }

        u64 totalSize = 0;
        RankId curRankId = 0;
        u64 curDataSize = 0;
        for (auto& ele : allRankMemSemantics[rankId][BufferType::OUTPUT]) {
            while (curRankId < rankSize && vDataDes.counts[curRankId] == 0) {
                ++curRankId;
            }
            if (curRankId >= rankSize) {
                HCCL_VM_ERROR(
                    "{} AllGatherV output for rank {} contains unexpected data after all "
                    "expected source ranks have been consumed."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), rankId, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            const u64 inputSize = vDataDes.counts[curRankId] * CHECK_SIZE_TABLE[vDataDes.dataType];
            const u64 rangeEnd = ele.startAddr + ele.size;

            if (ele.startAddr != totalSize) {
                HCCL_VM_ERROR(
                    "{} AllGatherV output for rank {} should continue at 0x{:x}, "
                    "but the next actual range starts at 0x{:x} (actual range: [0x{:x},0x{:x}))."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, totalSize, ele.startAddr,
                    ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != 1) {
                HCCL_VM_ERROR(
                    "{} AllGatherV output range [0x{:x},0x{:x}) for rank {} should come "
                    "from exactly one source, but it actually comes from {} sources."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, rankId,
                    ele.srcBufs.size(), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            const auto& srcBuf = *ele.srcBufs.begin();

            if (srcBuf.rankId != curRankId) {
                HCCL_VM_ERROR(
                    "{} AllGatherV output range [0x{:x},0x{:x}) for rank {} should come from "
                    "rank {}, but it actually comes from rank {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, rankId, curRankId,
                    srcBuf.rankId, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.bufType != BufferType::INPUT) {
                HCCL_VM_ERROR(
                    "{} AllGatherV output range [0x{:x},0x{:x}) for rank {} should come from "
                    "INPUT, but it actually comes from rank {} with buffer type {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, rankId,
                    srcBuf.rankId, BufferTypeToString(srcBuf.bufType), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.srcAddr != curDataSize) {
                HCCL_VM_ERROR(
                    "{} AllGatherV output range [0x{:x},0x{:x}) for rank {} should come from "
                    "rank {} at source address 0x{:x}, but the actual source address is 0x{:x}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, rankId,
                    srcBuf.rankId, curDataSize, srcBuf.srcAddr, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            curDataSize += ele.size;

            if (curDataSize == inputSize) {
                curDataSize = 0;
                curRankId++;
            } else if (curDataSize > inputSize) {
                HCCL_VM_ERROR(
                    "{} AllGatherV data collected from rank {} for rank {} becomes larger "
                    "than expected after outputRange [0x{:x},0x{:x}). The accumulated size is 0x{:x}, "
                    "but the expected size from this source rank is 0x{:x}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), curRankId, rankId, ele.startAddr, rangeEnd,
                    curDataSize, inputSize, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += ele.size;
        }
        if (totalSize != outputSize) {
            HCCL_VM_ERROR(
                "{} AllGatherV output for rank {} ends too early: the checker validated "
                "0x{:x} bytes in total, but the expected total result size is 0x{:x} bytes.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, totalSize, outputSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
