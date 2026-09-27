/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "batchsendrecv_semantics_checker.h"

#include <map>

#include "base.h"
#include "check_utils.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
HcclResult TaskCheckBatchSendRecvSemantics(
    std::map<RankId, RankMemorySemantics>& allRankMemSemantics, u32 expectedRankSize, u64 dataSize)
{
    if (expectedRankSize == 0 || allRankMemSemantics.size() != expectedRankSize) {
        HCCL_VM_ERROR(
            "{} BatchSendRecv rank set size mismatch: expected {}, actual {}.",
            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), expectedRankSize, allRankMemSemantics.size());
        return HcclResult::HCCL_E_PARA;
    }

    for (RankId rankId = 0; rankId < expectedRankSize; rankId++) {
        // 对应的rank不存在需要报错
        if (allRankMemSemantics.count(rankId) == 0) {
            HCCL_VM_ERROR(
                "{} BatchSendRecv produced no result data for rank {}, but this rank is "
                "expected to receive one segment from each of the {} participating ranks "
                "(expected total size 0x{:x}).",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, expectedRankSize,
                dataSize * expectedRankSize);
            return HcclResult::HCCL_E_PARA;
        }

        u64 totalSize = 0;
        RankId curRankId = 0;
        u64 curDataSize = 0;
        for (auto& ele : allRankMemSemantics[rankId][BufferType::OUTPUT]) {
            const u64 rangeEnd = ele.startAddr + ele.size;
            if (ele.startAddr != totalSize) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv output for rank {} should continue at 0x{:x}, "
                    "but the next actual range starts at 0x{:x} (actual range: [0x{:x},0x{:x}))."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, totalSize, ele.startAddr,
                    ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != 1) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv output range [0x{:x},0x{:x}) for rank {} should "
                    "come from exactly one source, but it actually comes from {} sources."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, rankId,
                    ele.srcBufs.size(), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            const auto& srcBuf = *ele.srcBufs.begin();
            if (srcBuf.rankId != curRankId) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv output range [0x{:x},0x{:x}) for rank {} should come "
                    "from rank {}, but it actually comes from rank {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, rankId, curRankId,
                    srcBuf.rankId, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.bufType != BufferType::INPUT) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv output range [0x{:x},0x{:x}) for rank {} should come "
                    "from rank{}.INPUT, but it actually comes from rank{}.{}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, rankId, curRankId,
                    srcBuf.rankId, BufferTypeToString(srcBuf.bufType), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.srcAddr != dataSize * rankId + curDataSize) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv output range [0x{:x},0x{:x}) for rank {} should take "
                    "data from source rank {} at input address 0x{:x}, but it actually takes data "
                    "from source rank {} at input address 0x{:x}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, rankId, curRankId,
                    dataSize * rankId + curDataSize, srcBuf.rankId, srcBuf.srcAddr, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            curDataSize += ele.size;
            if (curDataSize == dataSize) {
                curDataSize = 0;
                curRankId++;
            } else if (curDataSize > dataSize) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv data collected from rank{} for rank {} becomes "
                    "larger than expected after outputRange [0x{:x},0x{:x}). The accumulated size is "
                    "0x{:x}, but the expected size from this source rank is 0x{:x}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), curRankId, rankId, ele.startAddr, rangeEnd,
                    curDataSize, dataSize, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += ele.size;
        }
        // 如果curRankId等于rankSize，表示已经接受到其他所有rank的数据
        if (curRankId != expectedRankSize) {
            HCCL_VM_ERROR(
                "{} BatchSendRecv output for rank {} ends too early. The checker has "
                "validated 0x{:x} bytes in total, but the expected total size is 0x{:x}.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, totalSize, dataSize * expectedRankSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskCheckBatchSendRecvRingSemantics(
    std::map<RankId, RankMemorySemantics>& allRankMemSemantics, u32 expectedRankSize, u64 dataSize)
{
    if (expectedRankSize < 2 || allRankMemSemantics.size() != expectedRankSize) {
        HCCL_VM_ERROR(
            "{} BatchSendRecv ring rank set size mismatch: expected {}, actual {}.",
            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), expectedRankSize, allRankMemSemantics.size());
        return HcclResult::HCCL_E_PARA;
    }

    for (RankId rankId = 0; rankId < expectedRankSize; ++rankId) {
        const auto rankIt = allRankMemSemantics.find(rankId);
        if (rankIt == allRankMemSemantics.end()) {
            HCCL_VM_ERROR(
                "{} BatchSendRecv ring produced no result data for rank {}.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId);
            return HcclResult::HCCL_E_PARA;
        }

        const RankId expectedSrcRank = (rankId + expectedRankSize - 1U) % expectedRankSize;
        const auto outputIt = rankIt->second.find(BufferType::OUTPUT);
        if (dataSize == 0) {
            if (outputIt != rankIt->second.end()) {
                for (const auto& output : outputIt->second) {
                    if (output.size != 0) {
                        HCCL_VM_ERROR(
                            "{} BatchSendRecv ring rank {} should have an empty output, but range "
                            "[0x{:x},0x{:x}) is present.",
                            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), rankId, output.startAddr,
                            output.startAddr + output.size);
                        return HcclResult::HCCL_E_PARA;
                    }
                }
            }
            continue;
        }
        if (outputIt == rankIt->second.end() || outputIt->second.empty()) {
            HCCL_VM_ERROR(
                "{} BatchSendRecv ring output is missing for rank {}, expected size 0x{:x}.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, dataSize);
            return HcclResult::HCCL_E_PARA;
        }

        u64 totalSize = 0;
        for (const auto& output : outputIt->second) {
            const u64 rangeEnd = output.startAddr + output.size;
            if (rangeEnd < output.startAddr || output.startAddr != totalSize || output.size > dataSize - totalSize) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv ring output for rank {} is not a contiguous range of size "
                    "0x{:x}; next actual range is [0x{:x},0x{:x}).\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), rankId, dataSize, output.startAddr,
                    rangeEnd, output.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            if (output.srcBufs.size() != 1) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv ring output range [0x{:x},0x{:x}) for rank {} should "
                    "come from exactly one source, but it comes from {} sources.\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), output.startAddr, rangeEnd, rankId,
                    output.srcBufs.size(), output.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            const auto& srcBuf = *output.srcBufs.begin();
            if (srcBuf.rankId != expectedSrcRank || srcBuf.bufType != BufferType::INPUT
                || srcBuf.srcAddr != output.startAddr) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv ring output range [0x{:x},0x{:x}) for rank {} should "
                    "come from rank{}.INPUT at address 0x{:x}, but it comes from rank{}.{} at address "
                    "0x{:x}.\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), output.startAddr, rangeEnd, rankId,
                    expectedSrcRank, output.startAddr, srcBuf.rankId, BufferTypeToString(srcBuf.bufType),
                    srcBuf.srcAddr, output.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += output.size;
        }
        if (totalSize != dataSize) {
            HCCL_VM_ERROR(
                "{} BatchSendRecv ring output for rank {} has size 0x{:x}, expected 0x{:x}.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), rankId, totalSize, dataSize);
            return HcclResult::HCCL_E_PARA;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
