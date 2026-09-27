/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allreduce_semantics_checker.h"

#include <map>

#include "base.h"
#include "check_utils.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
HcclResult TaskCheckAllReduceSemantics(
    std::map<RankId, RankMemorySemantics>& allRankMemSemantics, u64 dataSize, HcclReduceOp reduceType)
{
    u32 rankSize = allRankMemSemantics.size();

    for (RankId rankId = 0; rankId < rankSize; rankId++) {
        // 对应的rank不存在需要报错
        if (allRankMemSemantics.count(rankId) == 0) {
            HCCL_VM_ERROR(
                "{} AllReduce produced no result data for rank {}, but this rank is expected to hold "
                "a full reduced result of 0x{:x} bytes from all {} participating ranks.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, dataSize, rankSize);
            return HcclResult::HCCL_E_PARA;
        }

        u64 totalSize = 0;
        for (auto& ele : allRankMemSemantics[rankId][BufferType::OUTPUT]) {
            const u64 rangeEnd = ele.startAddr + ele.size;
            if (ele.startAddr != totalSize) {
                HCCL_VM_ERROR(
                    "{} AllReduce output for rank {} should continue at 0x{:x}, "
                    "but the next actual range starts at 0x{:x} (actual range: [0x{:x},0x{:x}))."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, totalSize, ele.startAddr,
                    ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() > 1 && ele.reduceType != reduceType) {
                HCCL_VM_ERROR(
                    "{} AllReduce result range [0x{:x},0x{:x}) for rank {} was reduced with mode {}, "
                    "but the operator expects reduce mode {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, rankId,
                    DumpReduceOpToString(ele.reduceType), DumpReduceOpToString(reduceType), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != rankSize) {
                HCCL_VM_ERROR(
                    "{} AllReduce result range [0x{:x},0x{:x}) for rank {} should combine inputs "
                    "from {} source ranks, but it actually combines {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, rankId,
                    rankSize, ele.srcBufs.size(), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->rankId != 0 or ele.srcBufs.rbegin()->rankId != (rankSize - 1)) {
                HCCL_VM_ERROR(
                    "{} AllReduce result range [0x{:x},0x{:x}) for rank {} should combine "
                    "source ranks [0,{}], but the actual source ranks only span [{},{}]."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, rankId,
                    rankSize - 1, ele.srcBufs.begin()->rankId, ele.srcBufs.rbegin()->rankId, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            for (auto& srcBuf : ele.srcBufs) {
                if (srcBuf.bufType != BufferType::INPUT) {
                    HCCL_VM_ERROR(
                        "{} AllReduce result range [0x{:x},0x{:x}) for rank {} should come from INPUT, "
                        "but source rank {} actually provides buffer type {}."
                        "\nCurrent result range detail:\n{}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, rankId,
                        srcBuf.rankId, BufferTypeToString(srcBuf.bufType), ele.Describe());
                    return HcclResult::HCCL_E_PARA;
                }

                if (srcBuf.srcAddr != totalSize) {
                    HCCL_VM_ERROR(
                        "{} AllReduce result range [0x{:x},0x{:x}) for rank {} should read "
                        "source rank {} at 0x{:x}, but the actual source address is 0x{:x}."
                        "\nCurrent result range detail:\n{}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, rankId,
                        srcBuf.rankId, totalSize, srcBuf.srcAddr, ele.Describe());
                    return HcclResult::HCCL_E_PARA;
                }
            }
            totalSize += ele.size;
        }
        if (totalSize != dataSize) {
            HCCL_VM_ERROR(
                "{} AllReduce result for rank {} ends at 0x{:x} bytes, "
                "but the expected total size is 0x{:x}.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, totalSize, dataSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
