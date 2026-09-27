/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "send_recv_semantics_checker.h"

#include <map>

#include "base.h"
#include "check_utils.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
HcclResult TaskCheckSendRecvSemantics(
    std::map<RankId, RankMemorySemantics>& allRankMemSemantics, u64 dataSize, RankId srcRank, RankId dstRank)
{
    if (allRankMemSemantics.count(srcRank) == 0 || allRankMemSemantics.count(dstRank) == 0) {
        HCCL_VM_ERROR(
            "{} Send/Recv pair references a missing rank (sourceRank={}, targetRank={}).",
            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_CHECK_FAILED), srcRank, dstRank);
        return HcclResult::HCCL_E_PARA;
    }

    u64 totalSize = 0;
    for (auto& ele : allRankMemSemantics[dstRank][BufferType::OUTPUT]) {
        const u64 rangeEnd = ele.startAddr + ele.size;
        if (ele.startAddr != totalSize) {
            HCCL_VM_ERROR(
                "{} Send/Recv output for rank {} should continue at 0x{:x}, "
                "but the next actual range starts at 0x{:x} (actual range: [0x{:x},0x{:x}))."
                "\nCurrent result range detail:\n{}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), dstRank, totalSize, ele.startAddr, ele.startAddr,
                rangeEnd, ele.Describe());
            return HcclResult::HCCL_E_PARA;
        }

        if (ele.srcBufs.size() != 1) {
            HCCL_VM_ERROR(
                "{} Send/Recv output range [0x{:x},0x{:x}) for rank {} should come from "
                "exactly one source, but it actually comes from {} sources."
                "\nCurrent result range detail:\n{}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, dstRank,
                ele.srcBufs.size(), ele.Describe());
            return HcclResult::HCCL_E_PARA;
        }

        const auto& srcBuf = *ele.srcBufs.begin();
        if (srcBuf.rankId != srcRank) {
            HCCL_VM_ERROR(
                "{} Send/Recv output range [0x{:x},0x{:x}) for rank {} should come from "
                "rank {}, but it actually comes from rank {}."
                "\nCurrent result range detail:\n{}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, dstRank, srcRank,
                srcBuf.rankId, ele.Describe());
            return HcclResult::HCCL_E_PARA;
        }

        if (srcBuf.bufType != BufferType::INPUT) {
            HCCL_VM_ERROR(
                "{} Send/Recv output range [0x{:x},0x{:x}) for rank {} should come from "
                "rank{}.INPUT, but it actually comes from rank{}.{}."
                "\nCurrent result range detail:\n{}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, dstRank, srcRank,
                srcBuf.rankId, BufferTypeToString(srcBuf.bufType), ele.Describe());
            return HcclResult::HCCL_E_PARA;
        }
        if (srcBuf.srcAddr != totalSize) {
            HCCL_VM_ERROR(
                "{} Send/Recv output range [0x{:x},0x{:x}) for rank {} should take data "
                "from source rank {} at input address 0x{:x}, but it actually takes data from "
                "source rank {} at input address 0x{:x}."
                "\nCurrent result range detail:\n{}",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, dstRank, srcRank,
                totalSize, srcBuf.rankId, srcBuf.srcAddr, ele.Describe());
            return HcclResult::HCCL_E_PARA;
        }
        totalSize += ele.size;
    }
    if (totalSize != dataSize) {
        HCCL_VM_ERROR(
            "{} Send/Recv output for rank {} ends too early. The checker has "
            "validated 0x{:x} bytes in total, but the expected total size is 0x{:x}.",
            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), dstRank, totalSize, dataSize);
        return HcclResult::HCCL_E_PARA;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskCheckSendRecvGroupSemantics(
    std::map<RankId, RankMemorySemantics>& allRankMemSemantics, u64 dataSize,
    const std::vector<SendRecvPairParam>& pairs)
{
    for (const auto& pair : pairs) {
        const HcclResult ret = TaskCheckSendRecvSemantics(allRankMemSemantics, dataSize, pair.srcRank, pair.dstRank);
        if (ret != HcclResult::HCCL_SUCCESS) {
            return ret;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
