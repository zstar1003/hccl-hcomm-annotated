/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_SERVICE_STUB_H
#define HCCLV2_COLL_SERVICE_STUB_H

#include "checker_def.h"
#include "dma_mode.h"
#include "topo_meta.h"
#include "ccu_ins_preprocessor.h"
#include "david_alg_config.h"
#include "coll_operator.h"
#include "coll_alg_component.h"

using namespace checker;

namespace Hccl {

class CommunicatorImpl;
class CollServiceStub {
public:
    explicit CollServiceStub(CommunicatorImpl* comm) : comm_(comm), ccuInsPreprocessor_(comm) {}
    HcclResult GenRankSize(TopoMetaParam& topoMetaParam, TopoMeta& topoMate);
    HcclResult Orchestrate(CheckerOpParam& checkerOpParam, TopoMeta& topoMeta, DavidAlgConfig& config);
    shared_ptr<InsQueue> Orchestrate(CollAlgOperator& op, std::string& algName);

    void LoadWithOpBasedMode(CollOperator& op, std::string& algName);
    void LoadWithOffloadMode(CollOperator& op);
    void Init();
    void CollAlgComponentInit();
    CcuInsPreprocessor* GetCcuInsPreprocessor();
    void TransformTask();

private:
    shared_ptr<InsQueue> insQue_;
    CommunicatorImpl* comm_;
    CcuInsPreprocessor ccuInsPreprocessor_;
    shared_ptr<CollAlgComponent> collAlgComponent_;
};

} // namespace Hccl
#endif
