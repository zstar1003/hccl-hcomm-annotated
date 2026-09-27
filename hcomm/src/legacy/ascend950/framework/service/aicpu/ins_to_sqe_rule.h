/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INS_RULE_H
#define HCCL_INS_RULE_H

#include <vector>
#include "hccl_sqe.h"
#include "instruction.h"
#include "lite_res_mgr_fetcher.h"
namespace Hccl {
constexpr u32 RTSQ_TASK_ID = 0;
constexpr u32 RTSQ_PART_ID = 0;

std::vector<std::unique_ptr<HcclSqe>>
Interpret(const Instruction& ins, const u32 streamId, ResMgrFetcher* resMgrFetcher);
std::vector<std::unique_ptr<HcclSqe>>
Interpret(const InsLocalCopy& ins, const u32 streamId, ResMgrFetcher* resMgrFetcher);
std::vector<std::unique_ptr<HcclSqe>>
Interpret(const InsWriteReduce& ins, const u32 streamId, ResMgrFetcher* resMgrFetcher);
std::vector<std::unique_ptr<HcclSqe>>
Interpret(const InsLocalPostTo& ins, const u32 streamId, ResMgrFetcher* resMgrFetcher);

void Interpret(const Instruction& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);

void Interpret(const InsLocalPostTo& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsLocalWaitFrom& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsLocalCopy& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsLocalCopyExtend& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsLocalReduce& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);

void Interpret(const InsLocalWaitGroup& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsLocalBcastPost& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);

void Interpret(const InsPostReady& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsWaitReady& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsPostFin& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsWaitFin& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsPostFinAck& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsWaitFinAck& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);

void Interpret(const InsRead& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsReadReduce& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsBatchRead& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsReadExtend& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsBatchWrite& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsWrite& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsWriteExtend& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsWriteWithFin& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsWriteWithFinExtend& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsWriteReduce& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsWriteReduceWithFin& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsBatchOneSidedRead& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsBatchOneSidedWrite& ins, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsStreamSync& insStreamSync, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsPreStreamSync& insPreStreamSync, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
void Interpret(const InsAicpuReduce& insAicpuReduce, const StreamLite& stream, ResMgrFetcher* resMgrFetcher);
SendRecvItemTokenInfo
ConvertDataAddrRange(DataBuffer insDataBuffer, std::unordered_map<DataBuffer, SendRecvItemTokenInfo>& tokenInfos);
SendRecvItemTokenInfo
ConvertDataAddrRange(DataBuffer insDataBuffer, std::unordered_map<DataBuffer, SendRecvItemTokenInfo>& tokenInfos);
} // namespace Hccl
#endif // HCCL_INS_RULE_H
