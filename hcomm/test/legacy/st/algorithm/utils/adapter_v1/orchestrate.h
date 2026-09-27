/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ADAPTER_V1_ORCHESTRATE_H
#define HCCL_ADAPTER_V1_ORCHESTRATE_H

#include <map>
#include <unordered_map>
#include <vector>

#include "transformer.h"
#include "hccl_types.h"
#include "checker_def.h"
#include "topo_meta.h"
#include "transport.h"
#include "comm.h"
#include "communicator_stub.h"

using namespace checker;
namespace hccl {

extern std::unordered_map<RankId, std::vector<std::shared_ptr<TransportCompared>>> AllTransport_;
extern std::unordered_map<Transport*, std::shared_ptr<TransportCompared>> links2TransportCompare_;
extern map<TransportType, unordered_map<RankId, unordered_map<RankId, std::shared_ptr<Transport>>>> CreatedLinksDict_;
HcclResult CheckTransportLink();
HcclResult InitCommParams(HcclCommParams& params, RankTable_t& rankTable, RankId myRank);
void InitOpParam(
    OpParam& opParam, CheckerOpParam& checkerOpParam, RankId myRank, u32 rankSize, bool initStream, bool isIOSameAddr);
HcclResult GenRankTable(hccl::RankTable_t& rankTable, TopoMeta topoMate);
HcclResult OrchestraTask(
    CheckerOpParam& checkerOpParam, RankTable_t& rankTable, u32 rankNum, bool isRunning,
    std::vector<std::shared_ptr<hccl::HcclCommunicator>>& communicators, bool isIOSameAddr);

} // namespace hccl

#endif
