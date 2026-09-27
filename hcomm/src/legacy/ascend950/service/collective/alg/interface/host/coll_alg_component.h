/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_ALG_COMPONENT
#define HCCLV2_COLL_ALG_COMPONENT

#include "coll_alg_params.h"
#include "coll_operator.h"
#include "prim_queue.h"
#include "ins_queue.h"
#include "virtual_topo.h"
#include "execute_selector.h"
#include "mc2_selector.h"
#include "coll_alg_registry.h"
#include "ins_coll_alg_registry.h"
#include "env_func.h"
#include "env_config_v2.h"

namespace Hccl {

using PrimQuePtr = std::shared_ptr<PrimQueue>;
using InsQuePtr = std::shared_ptr<InsQueue>;

MAKE_ENUM(OrchestMode, PRIMITIVE, INSTRUCTION)

class CollAlgComponent {
public:
    CollAlgComponent(RankGraph* rankGraph, DevType devType, u32 myRank, u32 rankSize);

    void EnableDetour(bool enableDetour);
    void EnableDataAllign(bool enableAllign);
    void SetAllignSize(u64 allignSize);
    void SetMaxQueue(u32 maxQueue);
    void SetMaxLink(u32 maxLink);
    void SetMaxDepQueuePairs(u32 maxDepQueuePairs);
    void SetDmaMode(const DmaMode dmaMode);
    std::vector<char> GetPackedData() const;
    HcclResult ExecAlgSelect(
        const CollAlgOperator& op, const CollAlgParams& params, std::string& algName, OpExecuteConfig& opExecuteConfig);
    AlgorithmType GetAlgorithmTypeForMC2CCU(const std::string& name);
    // Host
    virtual HcclResult Orchestrate(
        const CollAlgOperator& op, const CollAlgParams& params, const string& algName,
        PrimQuePtr queue); // Primitive based
    virtual HcclResult Orchestrate(
        const CollAlgOperator& op, const CollAlgParams& params, const string& algName,
        InsQuePtr queue); // Instruction based
    virtual HcclResult CalcResOffload(
        const OpType& opType, const u64& dataSize, const HcclDataType& dataType, const OpExecuteConfig& opExecuteConfig,
        CollOffloadOpResReq& resReq);
    HcclResult CalcTaskNum(OpType opType, DataType dataType, u32 count, u32& taskNum);

    HcclResult CalNumBlocks(u32& numBlocks, u64 dataSize, OpType opType, string& algName, u32 numBlocksLimit) const;

    // for AICPU
    virtual std::vector<std::string>
    GetOpAlgNames(const OpType& opType, const OrchestMode& orchestMode = OrchestMode::PRIMITIVE);
    virtual CollAlgResReq GetCollAlgResReqByName(
        const OpType& opType, const std::string& algName, const OrchestMode& orchestMode = OrchestMode::PRIMITIVE);
    virtual CollAlgOpReq GetCollAlgOpReq(const CollAlgOperator& op, const std::string& collAlgName);

protected:
    HcclResult SetCollAlgExecutor(std::shared_ptr<CollAlgBase> collAlgExecutor) const;
    HcclResult SetInsCollAlgExecutor(std::shared_ptr<InsCollAlgBase> insCollAlgExecutor) const;

    RankGraph* rankGraph_ = nullptr;
    DevType devType_ = DevType::DEV_TYPE_NOSOC;
    u32 myRank_ = INVALID_RANKID;
    u32 rankSize_ = 0;

    bool enableDetour_ = false;
    bool enableAllign_ = false;
    u64 allignSize_ = 0;

    u32 maxQueue_ = 0;
    u32 maxLink_ = 0;
    u32 maxDepQueuePairs_ = 0;
    DmaMode dmaMode_ = DmaMode::DEFAULT;
    std::map<std::string, CollAlgResReq> algName2Res;
    std::shared_ptr<ExecuteSelector> collAlgSelector_;

private:
    void GetNHRStepNum(u32& nSteps) const;
    HcclResult CalcTaskNumMesh(OpType opType, u64 dataSize, u64 scratchBufSize, u32& taskNum);
    HcclResult CalcTaskNumNHR(OpType opType, u32& taskNum) const;
    void GetRoundByBufferSize(OpType opType, u64 dataSize, u64 scratchBufSize, u32& roundNum, u32& extraNum) const;
};

using CollAlgComponentPtr = std::shared_ptr<CollAlgComponent>;
} // namespace Hccl
#endif
