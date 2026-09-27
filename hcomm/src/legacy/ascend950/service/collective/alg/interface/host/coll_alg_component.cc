/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <string>
#include <map>

#include "log.h"
#include "execute_selector.h"
#include "coll_alg_component.h"
#include "data_type.h"

namespace Hccl {
CollAlgComponent::CollAlgComponent(RankGraph* rankGraph, DevType devType, u32 myRank, u32 rankSize)
    : rankGraph_(rankGraph),
      devType_(devType),
      myRank_(myRank),
      rankSize_(rankSize)

{
    collAlgSelector_ = std::make_shared<ExecuteSelector>(
        ExecuteSelector().SetVirtualTopo(rankGraph).SetRankSize(rankSize).SetMyRank(myRank));
}

constexpr u64 HCCLV2_DEFAULT_TASK_NUM = 30;
constexpr u32 ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE = 8;
constexpr u64 SMALL_COUNT_512KB = 512 * 1024;
constexpr u64 TASK_NUM_CONST_TWO = 2;
constexpr u64 TASK_NUM_CONST_THREE = 3;
constexpr u64 TASK_NUM_CONST_FOUR = 4;
constexpr u64 TASK_NUM_CONST_FIVE = 5;
constexpr u64 TASK_NUM_CONST_SIX = 6;

void CollAlgComponent::EnableDetour(bool enableDetour)
{
    enableDetour_ = enableDetour;
    return;
}

void CollAlgComponent::EnableDataAllign(bool enableAllign)
{
    enableAllign_ = enableAllign;
    return;
}

void CollAlgComponent::SetAllignSize(u64 allignSize)
{
    allignSize_ = allignSize;
    return;
}

void CollAlgComponent::SetMaxQueue(u32 maxQueue)
{
    maxQueue_ = maxQueue;
    return;
}

void CollAlgComponent::SetMaxLink(u32 maxLink)
{
    maxLink_ = maxLink;
    return;
}

void CollAlgComponent::SetMaxDepQueuePairs(u32 maxDepQueuePairs)
{
    maxDepQueuePairs_ = maxDepQueuePairs;
    return;
}

void CollAlgComponent::SetDmaMode(const DmaMode dmaMode)
{
    dmaMode_ = dmaMode;
    return;
}

AlgorithmType CollAlgComponent::GetAlgorithmTypeForMC2CCU(const std::string& name)
{
    return collAlgSelector_->GetAlgorithmTypeForMC2CCU(name);
}

HcclResult CollAlgComponent::ExecAlgSelect(
    const CollAlgOperator& op, const CollAlgParams& params, std::string& algName, OpExecuteConfig& opExecuteConfig)
{
    HCCL_INFO("CollAlgComponent::ExecAlgSelect currentCollOperator dataType[%s]", op.dataType.Describe().c_str());
    CollAlgParams paramsTmp = params;
    paramsTmp.dataSize = op.dataCount * DataTypeSizeGet(op.dataType);
    CHK_RET(collAlgSelector_->Run(op, paramsTmp, algName));
    opExecuteConfig = paramsTmp.opExecuteConfig;
    return HcclResult::HCCL_SUCCESS;
}

// 临时函数：由于资源回退时无法重新申请资源，所以暂时统一按照最大资源需求量申请资源
HcclResult TmpStubCalcResOffload(CollOffloadOpResReq& resReq)
{
    u64 stubRequiredSubQueNum = 16;
    u64 stubRequiredScratchMemSize = 256 * 1024 * 1024; // 256 * 1024 * 1024 = 256 M

    HCCL_INFO(
        "[TmpStubCalcResOffload] original requiredSubQueNum is [%llu], stubRequiredSubQueNum[%llu]",
        resReq.requiredSubQueNum, stubRequiredSubQueNum);
    HCCL_INFO(
        "[TmpStubCalcResOffload] original requiredScratchMemSize is [%llu], stubRequiredScratchMemSize[%llu]",
        resReq.requiredScratchMemSize, stubRequiredScratchMemSize);

    resReq.requiredSubQueNum = max(stubRequiredSubQueNum, resReq.requiredSubQueNum);
    resReq.requiredScratchMemSize = max(stubRequiredScratchMemSize, resReq.requiredScratchMemSize);

    HCCL_INFO(
        "[TmpStubCalcResOffload] updated requiredSubQueNum[%llu], requiredScratchMemSize[%llu]",
        resReq.requiredSubQueNum, resReq.requiredScratchMemSize);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CollAlgComponent::CalcResOffload(
    const OpType& opType, const u64& dataSize, const HcclDataType& dataType, const OpExecuteConfig& opExecuteConfig,
    CollOffloadOpResReq& resReq)
{
    bool isAlltoAll = (opType == OpType::ALLTOALL) || (opType == OpType::ALLTOALLV) || (opType == OpType::ALLTOALLVC);
    if ((rankSize_ == 1) && (!isAlltoAll)) {
        resReq.requiredScratchMemSize = 0;
        resReq.requiredSubQueNum = 0;
        HCCL_INFO("[CollAlgComponent] rankSize = 1, requiredSubQueNum and requiredScratchMemSize set to [0].");
        return HcclResult::HCCL_SUCCESS;
    }

    CollAlgOperator op;
    op.opType = opType;
    op.dataType = HcclDataTypeToDataType(dataType);
    op.dataCount = dataSize / DataTypeSizeGet(op.dataType);
    CollAlgParams params;
    params.opExecuteConfig = opExecuteConfig;
    params.opMode = OpMode::OFFLOAD;
    params.dataSize = dataSize;
    std::string collAlgName;
    CHK_RET(collAlgSelector_->Run(op, params, collAlgName));
    CHK_PRT_RET(
        collAlgName.empty(), HCCL_ERROR("[CollAlgComponent] Please assign a collAlgName by env variable!"),
        HcclResult::HCCL_E_PARA);

    std::shared_ptr<InsCollAlgBase> insGenFunc = InsCollAlgRegistry::Global()->GetAlgImpl(opType, collAlgName);
    CHK_PTR_NULL(insGenFunc);

    CHK_PRT_RET(
        SetInsCollAlgExecutor(insGenFunc) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[CollAlgComponent] Unable to Set InsCollAlgExecutor, please check params!"),
        HcclResult::HCCL_E_PARA);
    CHK_RET(insGenFunc->CalcResOffload(rankGraph_, dataSize, resReq));
    if (opExecuteConfig.accState == AcceleratorState::CCU_MS
        || opExecuteConfig.accState == AcceleratorState::CCU_SCHED) {
        CHK_RET(TmpStubCalcResOffload(resReq));
    }

    HCCL_INFO(
        "[CollAlgComponent][CalcResOffload] requiredSubQueNum[%llu], requiredScratchMemSize[%llu]",
        resReq.requiredSubQueNum, resReq.requiredScratchMemSize);
    return HcclResult::HCCL_SUCCESS;
}

std::vector<std::string> CollAlgComponent::GetOpAlgNames(const OpType& opType, const OrchestMode& orchestMode)
{
    if (orchestMode == OrchestMode::INSTRUCTION) {
        return (InsCollAlgRegistry::Global()->GetAvailAlgs()).at(opType);
    }

    return (CollAlgRegistry::Global()->GetAvailAlgs()).at(opType);
}

CollAlgResReq CollAlgComponent::GetCollAlgResReqByName(
    const OpType& opType, const std::string& algName, const OrchestMode& orchestMode)
{
    if (algName2Res.find(algName) != algName2Res.end()) {
        return algName2Res[algName];
    }
    CollAlgResReq algResReq;
    if (orchestMode == OrchestMode::PRIMITIVE) {
        HCCL_DEBUG("[CollAlgComponent] Primitive based algorithm.");
        std::shared_ptr<CollAlgBase> primGenFunc = CollAlgRegistry::Global()->GetAlgImpl(opType, algName);
        if (primGenFunc == nullptr) {
            return algResReq;
        }
        SetCollAlgExecutor(primGenFunc);
        primGenFunc->CalcRes(rankGraph_, algResReq);
        algName2Res[algName] = algResReq;
    } else if (orchestMode == OrchestMode::INSTRUCTION) {
        HCCL_DEBUG("[CollAlgComponent] Instruction based algorithm.");
        std::shared_ptr<InsCollAlgBase> insGenFunc = InsCollAlgRegistry::Global()->GetAlgImpl(opType, algName);
        if (insGenFunc == nullptr) {
            return algResReq;
        }
        SetInsCollAlgExecutor(insGenFunc);
        insGenFunc->CalcRes(rankGraph_, algResReq);
        algName2Res[algName] = algResReq;
    }

    HCCL_DEBUG("[CollAlgComponent] Finish CollAlgComponent::CalcRes for AICPU Mode.");
    return algResReq;
}

CollAlgOpReq CollAlgComponent::GetCollAlgOpReq(const CollAlgOperator& op, const std::string& collAlgName)
{
    CollAlgOpReq collAlgOpReq;

    collAlgOpReq.algName = collAlgName;
    if (algName2Res.find(collAlgName) != algName2Res.end() && op.opType != OpType::BATCHSENDRECV
        && op.opType != OpType::SEND && op.opType != OpType::RECV) {
        collAlgOpReq.resReq = algName2Res[collAlgName];
        return collAlgOpReq;
    }

    CHK_PRT_RET(
        collAlgOpReq.algName.empty(), HCCL_WARNING("[CollAlgComponent] Please assign a collAlgName by env variable!"),
        collAlgOpReq);

    std::shared_ptr<InsCollAlgBase> insGenFunc
        = InsCollAlgRegistry::Global()->GetAlgImpl(op.opType, collAlgOpReq.algName);
    if (insGenFunc == nullptr) {
        return collAlgOpReq;
    }

    SetInsCollAlgExecutor(insGenFunc);
    insGenFunc->SetOp(op);
    insGenFunc->SetSendRecvRemoteRank(op.sendRecvRemoteRank);
    auto req = insGenFunc->CalcRes(rankGraph_, collAlgOpReq.resReq);
    if (req != HcclResult::HCCL_SUCCESS) {
        THROW<InvalidParamsException>(StringFormat("CollAlgComponent::CalcRes failed"));
    }
    algName2Res[collAlgOpReq.algName] = collAlgOpReq.resReq;

    if (rankSize_ == 1) {
        collAlgOpReq.resReq.primQueueNum = 1;
        HCCL_DEBUG("[CollAlgComponent] rankSize = 1, algName %s.", collAlgOpReq.algName.c_str());
    }

    HCCL_DEBUG("[CollAlgComponent] Finish CollAlgComponent::CalcRes for AICPU Mode.");
    return collAlgOpReq;
}

std::vector<char> CollAlgComponent::GetPackedData() const
{
    BinaryStream binaryStream;
    binaryStream << dmaMode_;
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

HcclResult CollAlgComponent::Orchestrate(
    const CollAlgOperator& op, const CollAlgParams& params, const string& algName, PrimQuePtr queue)
{
    HCCL_DEBUG("[CollAlgComponent] Primitive based algorithm.");

    CHK_PRT_RET(
        algName.empty(), HCCL_ERROR("[CollAlgComponent] Empty collAlgName, please check envVar settings."),
        HcclResult::HCCL_E_PARA);
    std::shared_ptr<CollAlgBase> primGenFunc = CollAlgRegistry::Global()->GetAlgImpl(op.opType, algName);
    if (primGenFunc == nullptr) {
        HCCL_ERROR("[CollAlgComponent] Invalid opType and invalid collAlgName, [%s].", algName.c_str());
        return HcclResult::HCCL_E_PARA;
    }

    CHK_PRT_RET(
        enableDetour_
            && ((algName != "AllGatherMesh") && (algName != "ReduceScatterMesh") && (algName != "AllReduceMesh")),
        HCCL_ERROR("[CollAlgComponent] Current algorithm can not support detouring, please check!"),
        HcclResult::HCCL_E_NOT_SUPPORT);

    if (rankSize_ == 1) {
        u64 dataSize = op.dataCount * DataTypeSizeGet(op.dataType);
        DataSlice usrInSlice = DataSlice(BufferType::INPUT, 0, dataSize);
        DataSlice usrOutSlice = DataSlice(BufferType::OUTPUT, 0, dataSize);
        std::unique_ptr<Primitive> primLocalCopy = std::make_unique<PrimLocalCopy>(usrInSlice, usrOutSlice);
        queue->Append(std::move(primLocalCopy));

        HCCL_DEBUG("[CollAlgComponent] rankSize = 1.");
    } else {
        CHK_PRT_RET(
            SetCollAlgExecutor(primGenFunc) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CollAlgComponent] Unable to Set CollAlgExecutor, please check params!"),
            HcclResult::HCCL_E_PARA);
        primGenFunc->GenPrimQues(rankGraph_, op, params, queue);
    }

    HCCL_DEBUG("[CollAlgComponent] Primitive based algorithm: finish CollAlgComponent::Orchestrate.");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CollAlgComponent::SetCollAlgExecutor(std::shared_ptr<CollAlgBase> collAlgExecutor) const
{
    if (collAlgExecutor == nullptr) {
        HCCL_ERROR("CollAlgComponent::SetCollAlgExecutor ptr is null");
        return HcclResult::HCCL_E_PTR;
    }
    collAlgExecutor->SetMyRank(myRank_);
    collAlgExecutor->SetRankSize(rankSize_);
    collAlgExecutor->EnableDetour(enableDetour_);
    collAlgExecutor->EnableDataAllign(enableAllign_);
    collAlgExecutor->SetAllignSize(allignSize_);
    collAlgExecutor->SetDmaMode(dmaMode_);
    collAlgExecutor->SetDevType(devType_);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CollAlgComponent::CalNumBlocks(u32& numBlocks, u64 dataSize, OpType opType, string& algName, u32 numBlocksLimit) const
{
    std::string insCollAlgName;

    if (algName.empty()) {
        HCCL_ERROR("[CollAlgComponent] algName is empty");
        return HcclResult::HCCL_E_INTERNAL;
    } else {
        // 上层测试用例指定算法名字
        insCollAlgName = algName;
    }
    std::shared_ptr<InsCollAlgBase> insGenFunc = InsCollAlgRegistry::Global()->GetAlgImpl(opType, insCollAlgName);
    CHK_RET(insGenFunc->CalNumBlocks(numBlocks, dataSize, numBlocksLimit));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CollAlgComponent::Orchestrate(
    const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue)
{
    HCCL_DEBUG("[CollAlgComponent] Instruction based algorithm.");

    std::string insCollAlgName;

    if (algName.empty()) {
        HCCL_ERROR("[CollAlgComponent] algName is empty");
        return HcclResult::HCCL_E_INTERNAL;
    } else {
        // 上层测试用例指定算法名字
        insCollAlgName = algName;
    }
    std::shared_ptr<InsCollAlgBase> insGenFunc = InsCollAlgRegistry::Global()->GetAlgImpl(op.opType, insCollAlgName);

    if (insGenFunc == nullptr) {
        HCCL_ERROR("[CollAlgComponent] Invalid opType and invalid insCollAlgName, [%s].", algName.c_str());
        return HcclResult::HCCL_E_PARA;
    }

    bool isAlltoAll
        = (op.opType == OpType::ALLTOALL) || (op.opType == OpType::ALLTOALLV) || (op.opType == OpType::ALLTOALLVC);
    if ((rankSize_ == 1) && (op.inputMem == nullptr || op.outputMem == nullptr)) {
        HCCL_INFO("[CollAlgComponent] rankSize = 1 and inputMem or outputMem is nullptr. Do nothing.");
        return HcclResult::HCCL_SUCCESS;
    } else if ((rankSize_ == 1) && (!isAlltoAll)) {
        HCCL_INFO("[CollAlgComponent] rankSize = 1, copy from input to output.");
        u64 dataSize = op.dataCount * DataTypeSizeGet(op.dataType);
        u64 inputOffset = 0;
        u64 outputOffset = 0;
        if (op.opType == OpType::ALLGATHERV) {
            CHK_PTR_NULL(op.vDataDes.displs);
            outputOffset = static_cast<u64*>(op.vDataDes.displs)[0];
        } else if (op.opType == OpType::REDUCESCATTERV) {
            CHK_PTR_NULL(op.vDataDes.displs);
            inputOffset = static_cast<u64*>(op.vDataDes.displs)[0];
        }
        DataSlice usrInSlice = DataSlice(BufferType::INPUT, inputOffset, dataSize);
        DataSlice usrOutSlice = DataSlice(BufferType::OUTPUT, outputOffset, dataSize);
        std::unique_ptr<Instruction> insLocalCopy = std::make_unique<InsLocalCopy>(usrInSlice, usrOutSlice);
        queue->Append(std::move(insLocalCopy));
    } else {
        HCCL_INFO(
            "[CollAlgComponent] Orchestrate, opType[%s], rankSize[%llu].", op.opType.Describe().c_str(), rankSize_);
        CHK_PRT_RET(
            SetInsCollAlgExecutor(insGenFunc) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CollAlgComponent] Unable to Set InsCollAlgExecutor, please check params!"),
            HcclResult::HCCL_E_PARA);
        CHK_RET(insGenFunc->Orchestrate(rankGraph_, op, params, queue));
    }
    HCCL_DEBUG("[CollAlgComponent] Instruction based algorithm: finish CollAlgComponent::Orchestrate.");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CollAlgComponent::SetInsCollAlgExecutor(std::shared_ptr<InsCollAlgBase> insCollAlgExecutor) const
{
    if (insCollAlgExecutor == nullptr) {
        THROW<NullPtrException>(StringFormat("CollAlgComponent::SetInsCollAlgExecutor ptr is null"));
    }
    insCollAlgExecutor->SetMyRank(myRank_);
    insCollAlgExecutor->SetRankSize(rankSize_);
    insCollAlgExecutor->EnableDetour(enableDetour_);
    insCollAlgExecutor->EnableDataAllign(enableAllign_);
    insCollAlgExecutor->SetAllignSize(allignSize_);
    insCollAlgExecutor->SetDmaMode(dmaMode_);
    insCollAlgExecutor->SetDevType(devType_);

    return HcclResult::HCCL_SUCCESS;
}

void CollAlgComponent::GetNHRStepNum(u32& nSteps) const
{
    for (u32 tmp = rankSize_ - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    return;
}

void CollAlgComponent::GetRoundByBufferSize(
    OpType opType, u64 dataSize, u64 scratchBufSize, u32& roundNum, u32& extraNum) const
{
    if (opType == OpType::ALLREDUCE || opType == OpType::REDUCE || opType == OpType::BROADCAST) {
        roundNum = (dataSize + scratchBufSize - 1) / scratchBufSize;
        extraNum = 0;
    } else if (opType == OpType::ALLGATHER || opType == OpType::REDUCESCATTER) {
        u32 oneSliceSize = scratchBufSize / rankSize_;
        roundNum = (dataSize + oneSliceSize - 1) / oneSliceSize;
        extraNum = (rankSize_ - 1) * roundNum;
    } else if (opType == OpType::SCATTER) {
        u32 oneSliceSize = scratchBufSize / rankSize_;
        roundNum = (dataSize + oneSliceSize - 1) / oneSliceSize;
        extraNum = 0;
    } else {
        roundNum = 1;
        extraNum = 0;
    }
    return;
}

HcclResult CollAlgComponent::CalcTaskNumMesh(OpType opType, u64 dataSize, u64 scratchBufSize, u32& taskNum)
{
    if (opType == OpType::ALLGATHER) {
        taskNum += TASK_NUM_CONST_FIVE * (rankSize_ - 1) + TASK_NUM_CONST_FOUR * (rankSize_ - TASK_NUM_CONST_TWO)
                   + rankSize_; // 每个对端5次同步+拷贝，每个queue 4次同步，ranksize个localCopy
    } else if (opType == OpType::ALLREDUCE) {
        if (dataSize < SMALL_COUNT_512KB) {
            taskNum += TASK_NUM_CONST_FIVE * (rankSize_ - 1) + TASK_NUM_CONST_FOUR * (rankSize_ - TASK_NUM_CONST_TWO)
                       + rankSize_; // 每个对端5次同步+拷贝，每个queue 4次同步
        } else {
            taskNum += TASK_NUM_CONST_TWO * TASK_NUM_CONST_FIVE * (rankSize_ - 1)
                       + TASK_NUM_CONST_TWO * TASK_NUM_CONST_FOUR * (rankSize_ - TASK_NUM_CONST_TWO)
                       + rankSize_; // 每个对端5次同步+拷贝，每个queue 4次同步
        }
    } else if (opType == OpType::REDUCESCATTER) {
        taskNum += TASK_NUM_CONST_FIVE * (rankSize_ - 1) + TASK_NUM_CONST_FOUR * (rankSize_ - TASK_NUM_CONST_TWO)
                   + rankSize_; // 每个对端5次同步+拷贝，每个queue 4次同步，ranksize个localCopy、localReduce
    } else if (opType == OpType::ALLTOALL || opType == OpType::ALLTOALLV) {
        u32 numSubStep = (dataSize + scratchBufSize - 1) / scratchBufSize;
        u32 concurrentSendRecvNum = (rankSize_ > ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE) ?
                                        ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE :
                                        rankSize_;
        u64 commLoops = (rankSize_ + concurrentSendRecvNum - 1) / concurrentSendRecvNum;
        taskNum += numSubStep * commLoops * (6 * concurrentSendRecvNum); // 每步6次同步拷贝task
    } else if (opType == OpType::BROADCAST) {
        if (dataSize < SMALL_COUNT_512KB) {
            taskNum
                += TASK_NUM_CONST_THREE * (rankSize_ - 1)
                   + TASK_NUM_CONST_FOUR * (rankSize_ - TASK_NUM_CONST_TWO); // 每个对端3次同步+拷贝，每个queue 4次同步
        } else {
            taskNum
                += TASK_NUM_CONST_SIX * (rankSize_ - TASK_NUM_CONST_TWO)
                   + TASK_NUM_CONST_FOUR * (rankSize_ - TASK_NUM_CONST_TWO); // 每个对端6次同步+拷贝，每个queue 4次同步
        }
    } else if (opType == OpType::SCATTER) {
        taskNum += TASK_NUM_CONST_THREE * (rankSize_ - 1)
                   + TASK_NUM_CONST_FOUR * (rankSize_ - TASK_NUM_CONST_TWO); // 每片数据3个Task，每个que同步4个Task
    } else {
        taskNum += HCCLV2_DEFAULT_TASK_NUM;
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlgComponent::CalcTaskNumNHR(OpType opType, u32& taskNum) const
{
    u32 nSteps = 0;
    GetNHRStepNum(nSteps);
    if (opType == OpType::ALLGATHER) {
        taskNum += 4 * nSteps + (1LL << nSteps) + 1; // 每步4个卡间同步task
    } else if (opType == OpType::ALLREDUCE) {
        taskNum += 4 * nSteps + (1LL << nSteps) + 1; // AllGather, 每步4个卡间同步task
        taskNum += 4 * nSteps + (1LL << nSteps) + 1; // ReduceScatter, 每步4个卡间同步task
    } else if (opType == OpType::REDUCESCATTER) {
        taskNum += 4 * nSteps + (1LL << nSteps) + 1; // 每步4个卡间同步, task+数据搬运
    } else if (opType == OpType::BROADCAST) {
        // scatter + allgather
        taskNum += TASK_NUM_CONST_TWO * nSteps + (rankSize_ - 1) + (rankSize_ + 1);
        taskNum += 4 * nSteps + (1LL << nSteps) + 1; // 每步4个卡间同步task
    } else if (opType == OpType::SCATTER) {
        taskNum += TASK_NUM_CONST_TWO * nSteps + (rankSize_ - 1)
                   + (rankSize_ + 1); // 同步+分片数据拷贝，rankSize + 1次localCopy
    } else if (opType == OpType::REDUCE) {
        taskNum += 4 * nSteps + (1LL << nSteps) + 1; // 每步4个卡间同步task
    } else {
        taskNum += HCCLV2_DEFAULT_TASK_NUM;
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlgComponent::CalcTaskNum(OpType opType, DataType dataType, u32 count, u32& taskNum)
{
    if (rankSize_ == 0) {
        HCCL_ERROR("[CalcTaskNum]errNo[0x%016llx], invalid rankSize zero", HCCL_ERROR_CODE(HCCL_E_INTERNAL));
        return HCCL_E_INTERNAL;
    }
    std::map<OpType, std::vector<HcclAlgoType>> configAlgMap = EnvConfig::GetInstance().GetAlgoConfig().GetAlgoConfig();
    std::vector<HcclAlgoType> algos
        = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opType);
    if (it != configAlgMap.end()) {
        algos = it->second;
    }
    u32 dataSizePerVolume = DataTypeSizeGet(dataType);
    u64 dataSize = dataSizePerVolume * count;
    u64 scratchBufSize = EnvConfig::GetInstance().GetAlgoConfig().GetBuffSize();
    HCCL_DEBUG("[CollAlgComponent][CalcTaskNum] dataSize[%llu], scratchBufSize[%llu]", dataSize, scratchBufSize);
    if (algos[0] == HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH) {
        CalcTaskNumMesh(opType, dataSize, scratchBufSize, taskNum);
    } else {
        CalcTaskNumNHR(opType, taskNum);
    }

    u32 roundNum = 0;
    u32 extraNum = 0;
    GetRoundByBufferSize(opType, dataSize, scratchBufSize, roundNum, extraNum);
    taskNum = roundNum * taskNum + extraNum;
    HCCL_DEBUG("[CollAlgComponent][CalcTaskNum] taskNum is %llu", taskNum);
    return HCCL_SUCCESS;
}
} // namespace Hccl
