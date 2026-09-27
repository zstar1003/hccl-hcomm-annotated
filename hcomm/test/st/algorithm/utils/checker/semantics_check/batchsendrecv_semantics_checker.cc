/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "batchsendrecv_semantics_checker.h"
#include "data_dumper.h"
#include "analysis_result.pb.h"
#include "semantics_utils.h"

#include <map>
#include "base.h"
#include "log.h"

namespace checker {

HcclResult TaskCheckBatchSendRecvSemantics(std::map<RankId, RankMemorySemantics>& allRankMemSemantics, u64 dataSize)
{
    u32 rankSize = allRankMemSemantics.size();

    for (RankId rankId = 0; rankId < rankSize; rankId++) {
        // 对应的rank不存在需要报错
        if (allRankMemSemantics.count(rankId) == 0) {
            DataDumper::Global()->AddMissingSemantic(rankId, BufferType::OUTPUT, 0);
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_MISSING_SEMANTIC);
            DUMP_AND_ERROR("Missing rank %d mem semantics", rankId);
            return HcclResult::HCCL_E_PARA;
        }

        u64 totalSize = 0;
        RankId curRankId = 0;
        u64 curDataSize = 0;
        for (auto& ele : allRankMemSemantics[rankId][BufferType::OUTPUT]) {
            if (ele.startAddr != totalSize) {
                DataDumper::Global()->AddMissingSemantic(rankId, BufferType::OUTPUT, totalSize);
                DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_MISSING_SEMANTIC);
                DUMP_AND_ERROR(
                    "[rankId:%u]Missing buffer semantic: "
                    "expected startAddr is %llu, while cur buffer semantic startAddr is %llu, cur buffer semantic is "
                    "%s",
                    rankId, totalSize, ele.startAddr, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != 1) {
                DataDumper::Global()->MarkInvalidSemantic(rankId, BufferType::OUTPUT, ele);
                DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
                DUMP_AND_ERROR(
                    "[rankId:%u]Cur buffer semantic should not be reduce, which mean srcBufs size should be 1, "
                    "while cur buffer semantic is %s",
                    rankId, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->rankId != curRankId) {
                DataDumper::Global()->MarkInvalidSemantic(rankId, BufferType::OUTPUT, ele);
                DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
                DUMP_AND_ERROR(
                    "[rankId:%u]Cur buffer semantic should come from rank %u, while it come from rank %u, "
                    "cur buffer semantic is %s",
                    rankId, curRankId, ele.srcBufs.begin()->rankId, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->bufType != BufferType::INPUT) {
                DataDumper::Global()->MarkInvalidSemantic(rankId, BufferType::OUTPUT, ele);
                DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
                DUMP_AND_ERROR(
                    "[rankId:%u]Cur buffer semantic srcBufs bufType is not INPUT, cur buffer semantic is %s", rankId,
                    ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->srcAddr != dataSize * rankId + curDataSize) {
                DataDumper::Global()->MarkInvalidSemantic(rankId, BufferType::OUTPUT, ele);
                DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
                DUMP_AND_ERROR(
                    "[rankId:%u]Cur buffer semantic srcBufs srcAddr should be %llu, while it is %llu, "
                    "cur buffer semantic is %s",
                    rankId, dataSize * rankId + curDataSize, ele.srcBufs.begin()->srcAddr, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }
            curDataSize += ele.size;
            if (curDataSize == dataSize) {
                curDataSize = 0;
                curRankId++;
            } else if (curDataSize > dataSize) {
                DataDumper::Global()->MarkInvalidSemantic(rankId, BufferType::OUTPUT, ele);
                DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
                DUMP_AND_ERROR(
                    "[rankId:%u]Accumulated semantic size from rank %u is %llu, greater than expected %llu", rankId,
                    curRankId, curDataSize, dataSize);
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += ele.size;
        }
        // 如果curRankId等于rankSize，表示已经接受到其他所有rank的数据
        if (curRankId != rankSize) {
            DataDumper::Global()->AddMissingSemantic(rankId, BufferType::OUTPUT, totalSize);
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_MISSING_SEMANTIC);
            DUMP_AND_ERROR(
                "[rankId:%u]Missing buffer semantics in tail: already checked total size is %llu, "
                "accumulated semantic size from rank %u is %llu, while rankSize is %u",
                rankId, totalSize, curRankId, curDataSize, rankSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

} // namespace checker
