/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_v_semantics_checker.h"

#include <map>

#include "base.h"
#include "checker_def.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
HcclResult TaskCheckReduceScatterVSemantics(
    std::map<RankId, RankMemorySemantics>& allRankMemSemantics, HcclReduceOp reduceType, VDataDesTagInner& vDataDes)
{
    u32 rankSize = allRankMemSemantics.size();
    u64 outputSize = 0;

    for (u32 rankId = 0; rankId < rankSize; rankId++) {
        // 对应的rank不存在需要报错
        if (allRankMemSemantics.count(rankId) == 0) {
            HCCL_VM_ERROR(
                "{} ReduceScatterV produced no result data for rank {}, but this rank is "
                "expected to have one reduced shard (expected {} source ranks, expected result size 0x{:x}).",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, rankSize,
                vDataDes.counts[rankId] * CHECK_SIZE_TABLE[vDataDes.dataType]);
            return HcclResult::HCCL_E_PARA;
        }

        u64 totalSize = 0;
        for (auto& ele : allRankMemSemantics[rankId][BufferType::OUTPUT]) {
            const u64 rangeEnd = ele.startAddr + ele.size;
            if (ele.startAddr != totalSize) {
                HCCL_VM_ERROR(
                    "{} ReduceScatterV output for rank {} should continue at 0x{:x}, "
                    "but the next actual range starts at 0x{:x} (actual range: [0x{:x},0x{:x}))."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, totalSize, ele.startAddr,
                    ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() > 1 && ele.reduceType != reduceType) {
                HCCL_VM_ERROR(
                    "{} ReduceScatterV output range [0x{:x},0x{:x}) for rank {} uses reduce "
                    "mode {}, but the operator is set to reduce mode {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, rankId,
                    DumpReduceOpToString(ele.reduceType), DumpReduceOpToString(reduceType), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != rankSize) {
                HCCL_VM_ERROR(
                    "{} ReduceScatterV output range [0x{:x},0x{:x}) for rank {} expected "
                    "{} source ranks but got {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, rankId,
                    rankSize, ele.srcBufs.size(), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            if (ele.srcBufs.begin()->rankId != 0 or ele.srcBufs.rbegin()->rankId != static_cast<int>(rankSize - 1)) {
                HCCL_VM_ERROR(
                    "{} ReduceScatterV output range [0x{:x},0x{:x}) for rank {} expected "
                    "source ranks [0,{}] but got [{},{}]."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, rankId,
                    rankSize - 1, ele.srcBufs.begin()->rankId, ele.srcBufs.rbegin()->rankId, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            for (auto& srcBuf : ele.srcBufs) {
                if (srcBuf.bufType != BufferType::INPUT) {
                    HCCL_VM_ERROR(
                        "{} ReduceScatterV output range [0x{:x},0x{:x}) for rank {} comes "
                        "from rank {} with buffer type {}, but it should come from INPUT."
                        "\nCurrent result range detail:\n{}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, rankId,
                        srcBuf.rankId, BufferTypeToString(srcBuf.bufType), ele.Describe());
                    return HcclResult::HCCL_E_PARA;
                }

                const u64 outputOffset = vDataDes.displs[rankId] * CHECK_SIZE_TABLE[vDataDes.dataType];
                if (srcBuf.srcAddr != outputOffset + totalSize) {
                    HCCL_VM_ERROR(
                        "{} ReduceScatterV output range [0x{:x},0x{:x}) for rank {} should "
                        "come from rank {}.INPUT at 0x{:x}, but it actually comes from rank {}.INPUT "
                        "at 0x{:x}.\nCurrent result range detail:\n{}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, rankId,
                        srcBuf.rankId, outputOffset + totalSize, srcBuf.rankId, srcBuf.srcAddr, ele.Describe());
                    return HcclResult::HCCL_E_PARA;
                }
            }
            totalSize += ele.size;
        }
        outputSize = vDataDes.counts[rankId] * CHECK_SIZE_TABLE[vDataDes.dataType];

        if (totalSize != outputSize) {
            HCCL_VM_ERROR(
                "{} ReduceScatterV output for rank {} ends too early: the checker "
                "validated 0x{:x} bytes in total, but the expected result size is 0x{:x}.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, totalSize, outputSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
