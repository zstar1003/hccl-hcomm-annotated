/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <regex>
#include "ins_to_sqe_rule.h"

namespace Hccl {

using InsToSqeRule = std::function<std::vector<std::unique_ptr<HcclSqe>>(
    const Instruction& ins, const u32 streamId, ResMgrFetcher* resMgrFetcher)>;

template <class InsType>
InsToSqeRule Rules()
{
    return [](const Instruction& ins, const u32 streamId, ResMgrFetcher* resMgrFetcher) {
        return Interpret(static_cast<const InsType&>(ins), streamId, resMgrFetcher);
    };
}

const std::map<InstructionType, InsToSqeRule> ruleMap{
    {InstructionType::LOCAL_COPY, Rules<InsLocalCopy>()},
    {InstructionType::LOCAL_POST_TO, Rules<InsLocalPostTo>()},
    {InstructionType::LOCAL_WAIT_FROM, Rules<InsLocalWaitFrom>()},
    {InstructionType::WAIT_READY, Rules<InsWaitReady>()},
    {InstructionType::POST_READY, Rules<InsPostReady>()},
    {InstructionType::WAIT_FIN, Rules<InsWaitFin>()},
    {InstructionType::POST_FIN, Rules<InsPostFin>()},
    {InstructionType::WRITE, Rules<InsWaitFinAck>()},
    {InstructionType::WRITE_REDUCE, Rules<InsPostFinAck>()},
    {InstructionType::READ, Rules<InsWaitFinAck>()},
    {InstructionType::READ_REDUCE, Rules<InsPostFinAck>()},
};

std::vector<std::unique_ptr<HcclSqe>>
Interpret(const Instruction& ins, const u32 streamId, ResMgrFetcher* resMgrFetcher)
{
    if (ruleMap.find(ins.GetType()) != ruleMap.end()) {
        return ruleMap.at(ins.GetType())(ins, streamId, resMgrFetcher);
    }
    return vector<std::unique_ptr<HcclSqe>>();
}

std::vector<std::unique_ptr<HcclSqe>>
Interpret(const InsLocalCopy& ins, const u32 streamId, ResMgrFetcher* resMgrFetcher)
{
    std::vector<std::unique_ptr<HcclSqe>> res(1);

    u64 srcAddr = resMgrFetcher->GetLocAddr(ins.GetSrcSlice().GetType()) + ins.GetSrcSlice().GetOffset();
    u64 dstAddr = resMgrFetcher->GetLocAddr(ins.GetDstSlice().GetType()) + ins.GetDstSlice().GetOffset();
    auto sqe = std::make_unique<HcclSdmaSqe>();
    sqe->Config(
        streamId, RTSQ_TASK_ID, srcAddr, ins.GetSrcSlice().GetSize(), RtDataType::RT_DATA_TYPE_END,
        RtReduceKind::RT_REDUCE_KIND_END, dstAddr, RTSQ_PART_ID);
    res[0] = std::move(sqe);
    return res;
}

std::vector<std::unique_ptr<HcclSqe>>
Interpret(const InsWriteReduce& ins, const u32 streamId, ResMgrFetcher* resMgrFetcher)
{
    (void)ins;
    (void)streamId;
    (void)resMgrFetcher;
    std::vector<std::unique_ptr<HcclSqe>> res;
    return res;
}

std::vector<std::unique_ptr<HcclSqe>>
Interpret(const InsLocalPostTo& ins, const u32 streamId, ResMgrFetcher* resMgrFetcher)
{
    (void)ins;
    (void)streamId;
    (void)resMgrFetcher;
    return std::vector<std::unique_ptr<HcclSqe>>(0);
}
} // namespace Hccl
