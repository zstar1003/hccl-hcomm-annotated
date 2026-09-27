/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_SERVICE_DEFAULT_IMPL_H
#define HCCLV2_COLL_SERVICE_DEFAULT_IMPL_H

#include <set>
#include <vector>
#include <string>
#include "coll_service_base.h"
#include "hccl_params_pub.h"
#include "coll_operator.h"
#include "stream.h"
#include "virtual_topo.h"
#include "prim_queue.h"
#include "ins_queue.h"
#include "instruction.h"
#include "prim_translator.h"
#include "interpreter.h"
#include "connections_builder.h"
#include "mask_event.h"
#include "ub_ci_updater_manager.h"
#include "trace.h"
namespace Hccl {

class CollServiceDefaultImpl : public CollServiceBase {
public:
    explicit CollServiceDefaultImpl(CommunicatorImpl* comm) : CollServiceBase(comm) {};

    void Init() override;

    void LoadWithOpBasedMode(CollOperator& op, unique_ptr<Stream> stream) override;

    void LoadWithOffloadMode(CollOperator& op, std::unique_ptr<Stream> stream) override;

    void RecoverTransport(vector<LinkData>& links, vector<std::pair<LinkGroup, u32>> linkGroupPair) override;

    void ReLoadWithOpBasedMode(CollOperator& op) override;

    void ReLoadWithOffloadMode(CollOperator& op) override;

private:
    shared_ptr<PrimQueue> OrchestrateWithPrim(const CollAlgOperator& op) const;

    shared_ptr<InsQueue> OrchestrateWithIns(const CollAlgOperator& op) const;

    void AllocNotifies(const vector<LinkData>& links);

    void AllocOneLocCntNotify(const Instruction& ins) const;

    void AllocLocCntNotifies(const InsQueue& insQueue) const;

    void AddNop(const std::string& opTag, const vector<LinkData>& linkDataVec) const;

    void UpdateUbCiIfNeed(const std::string& opTag);

    void LoadWithOpBasedModeNoRegister(CollOperator& op);

    void LoadWithOffloadModeNoRegister(CollOperator& op);

    unique_ptr<PrimTranslator> primTranslator;

    // 创建rmaConnections
    unordered_map<std::string, unique_ptr<ConnectionsBuilder>> connectionsBuilders;

    std::set<LinkData> availableLinks;

    std::unique_ptr<UbCiUpdaterManager> ubCiUpdaterMgr;

    std::unique_ptr<MaskEvent> updatingUbCiEvent;
};
} // namespace Hccl

#endif // HCCLV2_COLL_SERVICE_H
