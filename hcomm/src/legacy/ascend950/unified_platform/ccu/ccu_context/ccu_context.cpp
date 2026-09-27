/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_ctx.h"
#include "ccu_context_resource.h"
#include "ccu_assist.h"
#include "ccu_microcode.h"

#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_device_manager.h"
#include "ccu_rep_type.h"

namespace Hccl {

constexpr u32 DATAT_SIZE_U32 = 32;
constexpr u32 TOKEN_VALUE_INDEX = 2;

CcuContext::CcuContext(
    const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& transportGroup)
    : transports(transports),
      transportGroup(&transportGroup)
{
    HCCL_INFO("Construct CcuContext: %s", arg.GetCtxSignature().GetData().c_str());
    if (transports.size() == 0 || transports[0] == nullptr) {
        HCCL_WARNING("No valid transport in CcuContext, Use Die0");
        SetDieId(0);
    } else {
        SetDieId(transports[0]->GetDieId());
    }

    // 生成SQE粒度profiling信息
    AddSqeProfiling(arg);
}

CcuContext::~CcuContext() { HCCL_DEBUG("~CcuContext"); }

HcclResult CcuContext::Init()
{
    TRY_CATCH_RETURN(Algorithm());
    return HCCL_SUCCESS;
}

HcclResult CcuContext::GeneTaskParam(const CcuTaskArg& arg, std::vector<CcuTaskParam>& taskParams)
{
    auto args = GeneArgs(arg);
    auto argsNum = args.size();
    if (argsNum != loadArgIndex) {
        HCCL_ERROR(
            "Args number does not match the Load instruction, argsNum = %lu, loadArgInstr= %u", argsNum, loadArgIndex);
        return HCCL_E_PARA;
    }

    // 如果args数量超过sqe arg的最大数量，则返回多个TaskParam，前面几个只从sqe中加载args;
    // args数量大于等于0、小于等于最大值时，返回1个TaskParam
    uint32_t seqNum
        = (argsNum / CCU_SQE_ARGS_LEN) + ((argsNum % CCU_SQE_ARGS_LEN) == 0 ? 0 : 1) + (argsNum == 0 ? 1 : 0);
    taskParams.resize(seqNum);
    for (uint32_t index = 0; index < seqNum; index++) {
        taskParams[index].dieId = GetDieId();
        taskParams[index].missionId = GetMissionId();
        taskParams[index].instStartId = instrInfo.missionStartInstrId + index * CCU_SQE_ARGS_LEN;
        taskParams[index].key = GetMissionKey();
        taskParams[index].argSize = CCU_SQE_ARGS_LEN;
        if (index == seqNum - 1) {
            taskParams[index].instCnt = instrInfo.missionInstrCount - index * CCU_SQE_ARGS_LEN;
            std::copy(std::begin(args) + index * CCU_SQE_ARGS_LEN, std::end(args), std::begin(taskParams[index].args));
        } else {
            taskParams[index].instCnt = CCU_SQE_ARGS_LEN;
            std::copy(
                std::begin(args) + index * CCU_SQE_ARGS_LEN, std::begin(args) + (index + 1) * CCU_SQE_ARGS_LEN,
                std::begin(taskParams[index].args));
        }

        HCCL_INFO(
            "[GeneTaskParam]task Param, dieId[%u] missionId[%u] instStartId[%u] instCnt[%u], argSize[%u]",
            taskParams[index].dieId, taskParams[index].missionId, taskParams[index].instStartId,
            taskParams[index].instCnt, taskParams[index].argSize);
    }
    return HCCL_SUCCESS;
}

void CcuContext::AllocGoResource(uint32_t parallelDim, uint32_t msPerLoop)
{
    if (moConfig.loopCount != 0xFFFFFFFF && moConfig.msInterleave != 0xFFFFFFFF
        && moConfig.memSlice != 0xFFFFFFFFFFFFFFFF) {
        // 已经配置过，略过
        return;
    } else {
        // 采用默认配置
        moConfig = {CcuRep::CCU_MS_INTERLEAVE, CcuRep::CCU_MS_DEFAULT_LOOP_COUNT, CcuRep::CCU_MS_SIZE};
    }
    // 算法配置的loop数覆盖默认配置，parallelDim默认为CCU_MS_DEFAULT_LOOP_COUNT
    moConfig.loopCount = parallelDim;
    // 算法配置的msPerLoop * CcuRep::CCU_MS_SIZE覆盖默认配置，msPerLoop默认为1
    moConfig.memSlice = msPerLoop * CcuRep::CCU_MS_SIZE;

    HCCL_INFO(
        "[AllocGoResource]moConfig: loopCount = %u, msInterleave = %u", moConfig.loopCount, moConfig.msInterleave);

    // 简单实现,只需要申请一次资源
    if (moRes.executor.size() == 0) {
        moRes.executor = CreateBlockExecutor(moConfig.loopCount);
        moRes.maskSignal = CreateBlockMaskSignal(moConfig.loopCount);
        moRes.ccuBuffer = CreateBlockCcuBuffer(moConfig.loopCount * moConfig.msInterleave);
    }

    constexpr size_t minMaskSignalCount = 2;
    if (moRes.maskSignal.size() < minMaskSignalCount) {
        THROW<CcuApiException>("MaskSignal is not enough, maskSignal = %lu", moRes.maskSignal.size());
    }
}

std::vector<uint64_t> CcuContext::CalGoSize(uint64_t size) { return CalGoSizeStatic(size, moConfig); }

std::vector<uint64_t> CcuContext::CalGoSizeStatic(uint64_t size, GroupOpConfig& moCfg)
{
    uint64_t offset = 0;
    uint64_t loopIterNum = 0;
    uint64_t loopExtendNum = 0;
    uint64_t tailSize = 0;

    uint64_t loopSize = moCfg.loopCount * moCfg.memSlice;
    uint64_t maxSize = loopSize * (CcuRep::GetMaxLoopIterNum() + 1);

    if (moCfg.loopCount == 0 || moCfg.memSlice == 0) {
        THROW<CcuApiException>(
            "Please Check Configure, loopCount = %u, memSlice = %u", moCfg.loopCount, moCfg.memSlice);
    }

    if (size > maxSize) {
        THROW<CcuApiException>("Too Large Size, size = %llu, maxSize = %llu", size, maxSize);
    }

    uint64_t m = size / loopSize;
    uint64_t n = (size - m * loopSize) / moCfg.memSlice;
    uint64_t p = size - m * loopSize - n * moCfg.memSlice;

    if (size == maxSize) {
        m = CcuRep::GetMaxLoopIterNum();
        n = moCfg.loopCount - 1;
        p = moCfg.memSlice;
    }

    HCCL_INFO(
        "[CalGoSizeStatic] moCfg.memSlice[%llu], moCfg.loopCount[%u], moCfg.msInterleave[%u]", moCfg.memSlice,
        moCfg.loopCount, moCfg.msInterleave);
    HCCL_INFO("Ccu Slice Split: m = %llu, n = %llu, p = %llu", m, n, p);

    // 数据量 < 256K, 跳过LoopGroup0
    // 此时loopIterNum == 0
    // 可以以此做为跳过LoopGroup0的条件
    offset = moCfg.memSlice * moCfg.loopCount * m;
    // 未实现, 这里可以只传入m, 在内部通过加法获得完整的参数
    loopIterNum = m;

    if (n == 0 && p == 0) {
        // 数据量为256K的整数倍，跳过LoopGroup1
        // 此时tailSize = 0，可以依次做为跳过LoopGroup1的条件
        loopExtendNum = 0; // loopExtendNum 赋值
        tailSize = 0;      // tailSize 赋值
    } else if (n != 0 && p == 0) {
        // 数据量为256K * m + 4K * n
        // 因为p == 0, 所以只需要使用第一个Loop, 数据量4K, 展开成n次
        loopExtendNum = CcuRep::GetParallelParam(n - 1, 0, 1); // loopExtendNum 赋值
        tailSize = moCfg.memSlice;                             // tailSize 赋值
    } else if (n == 0 && p != 0) {
        // 数据量为256K * m + p
        // 因为n == 0, 所以只需要使用第一个Loop, 数据量p, 不展开
        loopExtendNum = CcuRep::GetParallelParam(0, 0, 1); // loopExtendNum 赋值
        tailSize = p;                                      // tailSize 赋值
    } else {
        loopExtendNum = CcuRep::GetParallelParam(n - 1, 1, 2); // loopExtendNum 赋值, 为2
        tailSize = p;                                          // tailSize 赋值
    }

    HCCL_INFO(
        "offset = %lu, loopIterNum = %lu, loopExtendNum = %lu, tailSize = %lu", offset, loopIterNum, loopExtendNum,
        tailSize);

    return {offset, loopIterNum, loopExtendNum, tailSize};
}

CcuRep::Variable CcuContext::CreateVariable(const CcuTransport& transport, uint32_t varIndex) const
{
    CcuRep::Variable var;
    var.Reset(transport.GetLocXnByIndex(varIndex), transport.GetDieId());
    return var;
}

CcuRep::Variable CcuContext::ImportVariable(const std::string& tag)
{
    CcuRep::Variable var;
    importRes.sharedVar.insert({tag, var});
    return var;
}

void CcuContext::ExportVariable(const CcuRep::Variable& var, const std::string& tag)
{
    exportRes.sharedVar.insert({tag, var});
}

CcuRep::MaskSignal CcuContext::ImportMaskSignal(const std::string& tag)
{
    CcuRep::MaskSignal sig;
    importRes.sharedSig.insert({tag, sig});
    return sig;
}

void CcuContext::ExportMaskSignal(const CcuRep::MaskSignal& sig, const std::string& tag)
{
    exportRes.sharedSig.insert({tag, sig});
}

CcuSharedResource& CcuContext::GetExportRes() { return exportRes; }

CcuSharedResource& CcuContext::GetImportRes() { return importRes; }

CcuRepResource& CcuContext::GetResource() { return res; }

CcuResReq CcuContext::GetResourceRequest()
{
    CcuResReq req;
    uint32_t dieId = GetDieId();
    req.msReq[dieId] = res.ccubuffers[dieId].size();
    req.blockMsReq[dieId] = res.blockCcubuffers[dieId].size();
    req.ckeReq[dieId] = res.maskSignal[dieId].size();
    req.blockCkeReq[dieId] = res.blockMaskSignal[dieId].size();
    req.loopEngineReq[dieId] = res.executor[dieId].size();
    req.blockLoopEngineReq[dieId] = res.blockExecutor[dieId].size();
    req.gsaReq[dieId] = res.address[dieId].size();
    req.blockGsaReq[dieId] = res.blockAddress[dieId].size();
    req.xnReq[dieId] = res.variable[dieId].size();
    req.blockXnReq[dieId] = res.continuousVariable[dieId].size();

    req.missionReq.reqType = MissionReqType::FUSION_MULTIPLE_DIE;
    req.missionReq.req[dieId] = 1;

    auto info = StringFormat(
        "resource request: dieId[%u], ms[%u], blockMs[%u], cke[%u], blockCke[%u], "
        "loopEngine[%u], blockLoopEngine[%u], gsa[%u], blockGsa[%u], xn[%u], block xn[%u], missionId[%u]",
        dieId, req.msReq[dieId], req.blockMsReq[dieId], req.ckeReq[dieId], req.blockCkeReq[dieId],
        req.loopEngineReq[dieId], req.blockLoopEngineReq[dieId], req.gsaReq[dieId], req.blockGsaReq[dieId],
        req.xnReq[dieId], req.blockXnReq[dieId], req.missionReq.req[dieId]);

    HCCL_INFO("%s", info.c_str());

    return req;
}

void CcuContext::Load(const CcuRep::Variable& var)
{
    // 记录goSize相关变量对应的task argIndex
    auto loadArgRep = std::make_shared<CcuRep::CcuRepLoadArg>(var, loadArgIndex % CCU_SQE_ARGS_LEN);
    GetLGProfilingInfo().loadRep2ArgIdxMap[loadArgRep] = loadArgIndex;
    Append(loadArgRep);
    loadArgIndex++;
}

void CcuContext::LoadVariable(uint64_t addr, const CcuRep::Variable& var)
{
    Append(std::make_shared<CcuRep::CcuRepLoad>(addr, var));
}

void CcuContext::LoadVariable(uint64_t addr, const CcuRep::Variable& var, uint32_t num)
{
    Append(std::make_shared<CcuRep::CcuRepLoad>(addr, var, num));
}

void CcuContext::StoreVariable(const CcuRep::Variable& var, uint64_t addr)
{
    Append(std::make_shared<CcuRep::CcuRepStore>(var, addr));
}

void CcuContext::LoadVariable(const CcuRep::Variable& src, const CcuRep::Variable& var, uint32_t num)
{
    Append(std::make_shared<CcuRep::CcuRepLoadVar>(src, var, num));
}

void CcuContext::StoreVariable(const CcuRep::Variable& var, const CcuRep::Variable& src)
{
    Append(std::make_shared<CcuRep::CcuRepStoreVar>(src, var));
}

void CcuContext::Load(GroupOpSize moSize)
{
    Load(moSize.addrOffset);
    Load(moSize.loopParam);
    Load(moSize.parallelParam);
    Load(moSize.residual);
}

void CcuContext::LocalCtxPost(const CcuRep::MaskSignal& sig, uint32_t mask)
{
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        THROW<CcuApiException>("LocalCtxPost is not allowed in LoopBlock");
    }
    Append(std::make_shared<CcuRep::CcuRepPostSharedSem>(sig, mask));
}

void CcuContext::LocalCtxPostVar(
    const CcuRep::Variable& srcVar, const CcuRep::Variable& dstVar, const CcuRep::MaskSignal& sig, uint32_t mask)
{
    Append(std::make_shared<CcuRep::CcuRepPostSharedVar>(srcVar, dstVar, sig, mask));
}

void CcuContext::LocalPost(const CcuRep::MaskSignal& sig, uint32_t mask)
{
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        THROW<CcuApiException>("LocalPost is not allowed in LoopBlock");
    }
    auto rep = std::make_shared<CcuRep::CcuRepLocPostSem>(sig, mask);
    Append(rep);
    SetDependencyInfo(sig.Id(), mask, rep);
}

void CcuContext::LocalWait(const CcuRep::MaskSignal& sig, uint32_t mask)
{
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        Append(std::make_shared<CcuRep::CcuRepLocWaitSem>(sig, mask, false));
    } else {
        auto rep = std::make_shared<CcuRep::CcuRepLocWaitSem>(sig, mask, true);
        AddProfiling("LocalWait", mask);
        rep->SetDependencyInfo(GetDependencyInfo(sig.Id()));
        ClearDependencyInfo();
        Append(rep);
    }
}

void CcuContext::RemotePost(const CcuTransport& transport, uint32_t signalIndex, uint32_t mask, bool single)
{
    Append(std::make_shared<CcuRep::CcuRepRemPostSem>(transport, signalIndex, mask, single));
}

void CcuContext::WriteVariableWithSignal(
    const CcuTransport& transport, const CcuRep::Variable& var, uint32_t varIndex, uint32_t signalIndex, uint32_t mask)
{
    Append(std::make_shared<CcuRep::CcuRepRemPostVar>(var, transport, varIndex, signalIndex, mask));
}

void CcuContext::RemoteWait(const CcuTransport& transport, uint32_t signalIndex, uint32_t mask)
{
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        Append(std::make_shared<CcuRep::CcuRepRemWaitSem>(transport, signalIndex, mask, false));
    } else {
        auto rep = std::make_shared<CcuRep::CcuRepRemWaitSem>(transport, signalIndex, mask, true);
        AddProfiling(transport, "RemoteWait", signalIndex, mask);
        Append(rep);
    }
}

void CcuContext::GroupWait(const CcuTransportGroup& transportGroup, uint32_t signalIndex, uint32_t mask)
{
    if (CurrentBlock()->Type() == CcuRep::CcuRepType::LOOP_BLOCK) {
        Append(std::make_shared<CcuRep::CcuRepWaitGroup>(transportGroup, signalIndex, mask, false));
    } else {
        auto rep = std::make_shared<CcuRep::CcuRepWaitGroup>(transportGroup, signalIndex, mask, true);
        AddProfiling(transportGroup, "GroupWait", signalIndex, mask);
        Append(rep);
    }
}

void CcuContext::Read(
    const CcuTransport& transport, const CcuRep::CcuBuffer& loc, const CcuRep::Memory& rem, const CcuRep::Variable& len,
    const CcuRep::MaskSignal& locSig, uint32_t mask)
{
    auto rep = std::make_shared<CcuRep::CcuRepBufRead>(transport, rem, loc, len, locSig, mask);
    Append(rep);
    SetDependencyInfo(locSig.Id(), mask, rep);
}

void CcuContext::Write(
    const CcuTransport& transport, const CcuRep::Memory& rem, const CcuRep::CcuBuffer& loc, const CcuRep::Variable& len,
    const CcuRep::MaskSignal& locSig, uint32_t mask)
{
    auto rep = std::make_shared<CcuRep::CcuRepBufWrite>(transport, loc, rem, len, locSig, mask);
    Append(rep);
    SetDependencyInfo(locSig.Id(), mask, rep);
}

static bool isLowPrecisionIn(DataType dataType)
{
    return dataType == DataType::INT8 || dataType == DataType::HIF8 || dataType == DataType::FP8E4M3
           || dataType == DataType::FP8E5M2;
}

static bool isLowPrecisionOut(DataType dataType)
{
    return dataType == DataType::FP16 || dataType == DataType::BFP16 || dataType == DataType::FP32;
}

void CcuContext::LocalReduce(
    const std::vector<CcuRep::CcuBuffer>& bufs, uint32_t count, DataType dataType, DataType outputDataType,
    ReduceOp opType, const CcuRep::MaskSignal& locSig, const CcuRep::Variable& len, uint32_t mask)
{
    if ((opType == ReduceOp::SUM && isLowPrecisionIn(dataType) && !isLowPrecisionOut(outputDataType))
        || (opType == ReduceOp::SUM && !isLowPrecisionIn(dataType) && dataType != outputDataType)
        || (opType != ReduceOp::SUM && dataType != outputDataType)) {
        THROW<CcuApiException>(
            "Unsupported inputDataType[%s], outputDataType[%s] for reduceOp[%s]", dataType.Describe().c_str(),
            outputDataType.Describe().c_str(), opType.Describe().c_str());
    }

    auto rep = std::make_shared<CcuRep::CcuRepBufReduce>(
        bufs, count, CcuRep::GetCcuDataType(dataType, opType), CcuRep::GetCcuDataType(outputDataType, opType),
        CcuRep::GetCcuReduceType(opType), locSig, len, mask);
    Append(rep);
    SetDependencyInfo(locSig.Id(), mask, rep);
}

void CcuContext::Read(
    const CcuTransport& transport, const CcuRep::Memory& loc, const CcuRep::Memory& rem, const CcuRep::Variable& len,
    const CcuRep::MaskSignal& locSig, uint32_t mask)
{
    auto rep = std::make_shared<CcuRep::CcuRepRead>(transport, loc, rem, len, locSig, mask);
    Append(rep);
    SetDependencyInfo(locSig.Id(), mask, rep);
}

void CcuContext::ReadReduce(
    const CcuTransport& transport, const CcuRep::Memory& loc, const CcuRep::Memory& rem, const CcuRep::Variable& len,
    DataType dataType, ReduceOp opType, const CcuRep::MaskSignal& locSig, uint32_t mask)
{
    auto rep = std::make_shared<CcuRep::CcuRepRead>(
        transport, loc, rem, len, CcuRep::GetUBDataType(dataType), CcuRep::GetUBReduceType(opType), locSig, mask);
    Append(rep);
    SetDependencyInfo(locSig.Id(), mask, rep);
}

void CcuContext::Write(
    const CcuTransport& transport, const CcuRep::Memory& rem, const CcuRep::Memory& loc, const CcuRep::Variable& len,
    const CcuRep::MaskSignal& locSig, uint32_t mask)
{
    auto rep = std::make_shared<CcuRep::CcuRepWrite>(transport, rem, loc, len, locSig, mask);
    Append(rep);
    SetDependencyInfo(locSig.Id(), mask, rep);
}

void CcuContext::WriteReduce(
    const CcuTransport& transport, const CcuRep::Memory& rem, const CcuRep::Memory& loc, const CcuRep::Variable& len,
    DataType dataType, ReduceOp opType, const CcuRep::MaskSignal& locSig, uint32_t mask)
{
    auto rep = std::make_shared<CcuRep::CcuRepWrite>(
        transport, rem, loc, len, CcuRep::GetUBDataType(dataType), CcuRep::GetUBReduceType(opType), locSig, mask);
    Append(rep);
    SetDependencyInfo(locSig.Id(), mask, rep);
}

void CcuContext::LocalCopy(
    const CcuRep::Memory& dst, const CcuRep::Memory& src, const CcuRep::Variable& len, const CcuRep::MaskSignal& locSig,
    uint32_t mask)
{
    auto rep = std::make_shared<CcuRep::CcuRepLocCpy>(dst, src, len, locSig, mask);
    Append(rep);
    SetDependencyInfo(locSig.Id(), mask, rep);
}

void CcuContext::LocalCopy(
    const CcuRep::CcuBuffer& dst, const CcuRep::Memory& src, const CcuRep::Variable& len,
    const CcuRep::MaskSignal& locSig, uint32_t mask)
{
    auto rep = std::make_shared<CcuRep::CcuRepBufLocRead>(src, dst, len, locSig, mask);
    Append(rep);
    SetDependencyInfo(locSig.Id(), mask, rep);
}

void CcuContext::LocalCopy(
    const CcuRep::Memory& dst, const CcuRep::CcuBuffer& src, const CcuRep::Variable& len,
    const CcuRep::MaskSignal& locSig, uint32_t mask)
{
    auto rep = std::make_shared<CcuRep::CcuRepBufLocWrite>(src, dst, len, locSig, mask);
    Append(rep);
    SetDependencyInfo(locSig.Id(), mask, rep);
}

void CcuContext::LocalReduce(
    const CcuRep::Memory& dst, const CcuRep::Memory& src, const CcuRep::Variable& len, DataType dataType,
    ReduceOp opType, const CcuRep::MaskSignal& locSig, uint32_t mask)
{
    auto rep = std::make_shared<CcuRep::CcuRepLocCpy>(
        dst, src, len, CcuRep::GetUBDataType(dataType), CcuRep::GetUBReduceType(opType), locSig, mask);
    Append(rep);
    SetDependencyInfo(locSig.Id(), mask, rep);
}

void CcuContext::CreateMultiOpCopy()
{
    AllocGoResource(CCU_MS_LOCAL_COPY_LOOP_COUNT, LOCAL_COPY_MS_PER_LOOP);
    std::string loopType = "localcopy";
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }

    uint32_t usedBufNum = moConfig.memSlice / CcuRep::CCU_MS_SIZE;

    for (uint32_t index = 0; index < 2; index++) { // 需要实现化2个Loop
        CcuRep::Memory src = CreateMemory();
        CcuRep::Memory dst = CreateMemory();
        CcuRep::Variable len = CreateVariable();
        CcuRep::LoopBlock lb(this, loopType + "_loop_" + std::to_string(index));
        lb(src, dst, len);

        CcuRep::MaskSignal sem = moRes.maskSignal[index];

        std::vector<CcuRep::CcuBuffer> bufs
            = {moRes.ccuBuffer.begin() + index * moConfig.msInterleave,
               moRes.ccuBuffer.begin() + index * moConfig.msInterleave + usedBufNum};

        LocalCopy(bufs[0], src, len, sem);
        LocalWait(sem);
        LocalCopy(dst, bufs[0], len, sem);
        LocalWait(sem);
    }

    registeredLoop.insert(loopType);
    return;
}

void CcuContext::GroupCopy(CcuRep::Memory dst, CcuRep::Memory src, GroupOpSize goSize)
{
    CcuRep::Memory tmpDst = CreateMemory();
    tmpDst = dst;
    CcuRep::Memory tmpSrc = CreateMemory();
    tmpSrc = src;

    CreateMultiOpCopy();
    CCU_IF(goSize.addrOffset != 0)
    {
        CcuRep::Variable loopParam = CreateVariable();
        loopParam = CcuRep::GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += goSize.loopParam;

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        auto lc = Loop("localcopy_loop_0")(tmpSrc, tmpDst, sliceSize);

        CcuRep::Variable paraCfg = CreateVariable();
        paraCfg = CcuRep::GetParallelParam(moConfig.loopCount - 1, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }

    CCU_IF(goSize.parallelParam != 0)
    {
        CcuRep::Condition cond(this, goSize.parallelParam != 0);

        tmpSrc.addr += goSize.addrOffset;
        tmpDst.addr += goSize.addrOffset;
        auto lc0 = Loop("localcopy_loop_0")(tmpSrc, tmpDst, goSize.residual);

        tmpSrc.addr += goSize.residual;
        tmpDst.addr += goSize.residual;
        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        auto lc1 = Loop("localcopy_loop_1")(tmpSrc, tmpDst, sliceSize);

        CcuRep::Variable loopCfg0 = CreateVariable();
        loopCfg0 = CcuRep::GetLoopParam(0, 0, 1);
        CcuRep::Variable loopCfg1 = CreateVariable();
        loopCfg1 = CcuRep::GetLoopParam(0, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
        LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);
    }
}

void CcuContext::CreateMultiOpBroadcast(const std::vector<CcuTransport*>& transports)
{
    AllocGoResource();

    std::string loopType = "broadcast";
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }

    uint32_t size = transports.size() + 1;

    for (uint32_t index = 0; index < 2; index++) { // 需要实现化2个Loop
        CcuRep::Memory src = CreateMemory();
        std::vector<CcuRep::Memory> dst;
        for (uint32_t i = 0; i < size; i++) {
            dst.emplace_back(CreateMemory());
        }
        CcuRep::Variable len = CreateVariable();
        CcuRep::LoopBlock lb(this, loopType + "_loop_" + std::to_string(index));
        lb(src, dst, len);

        CcuRep::CcuBuffer buf = moRes.ccuBuffer[index * moConfig.msInterleave];
        CcuRep::MaskSignal sem = moRes.maskSignal[index];

        LocalCopy(buf, src, len, sem);
        LocalWait(sem);

        for (uint32_t i = 0; i < transports.size(); i++) {
            if (transports[i] == nullptr) {
                THROW<CcuApiException>("transport is nullptr");
            }
            Write(*transports[i], dst[i], buf, len, sem, 1 << i);
        }
        LocalCopy(dst[size - 1], buf, len, sem, 1 << (size - 1));
        LocalWait(sem, (1 << size) - 1);
    }

    registeredLoop.insert(loopType);
}

void CcuContext::CreateMultiOpBroadcastWithoutMyRank(const std::vector<CcuTransport*>& ccuTransports)
{
    AllocGoResource();

    std::string loopType = "broadcast";
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }

    uint32_t size = ccuTransports.size() + 1;

    for (uint32_t index = 0; index < 2; index++) { // 需要实现化2个Loop
        CcuRep::Memory src = CreateMemory();
        std::vector<CcuRep::Memory> dst;
        for (uint32_t i = 0; i < size; i++) {
            dst.emplace_back(CreateMemory());
        }
        CcuRep::Variable len = CreateVariable();
        CcuRep::LoopBlock lb(this, loopType + "_loop_" + std::to_string(index));
        lb(src, dst, len);

        CcuRep::CcuBuffer buf = moRes.ccuBuffer[index * moConfig.msInterleave];
        CcuRep::MaskSignal sem = moRes.maskSignal[index];

        LocalCopy(buf, src, len, sem);
        LocalWait(sem);

        for (uint32_t i = 0; i < ccuTransports.size(); i++) {
            if (ccuTransports[i] == nullptr) {
                THROW<CcuApiException>("transport is nullptr");
            }
            Write(*ccuTransports[i], dst[i], buf, len, sem, 1 << i);
        }
        LocalWait(sem, (1 << ccuTransports.size()) - 1);
    }

    registeredLoop.insert(loopType);
}

void CcuContext::GroupBroadcastWithoutMyRank(
    const std::vector<CcuTransport*>& ccuTransports, std::vector<CcuRep::Memory> dst, CcuRep::Memory src,
    GroupOpSize goSize)
{
    CreateMultiOpBroadcastWithoutMyRank(ccuTransports);

    uint32_t size = ccuTransports.size() + 1;

    CCU_IF(goSize.addrOffset != 0)
    {
        CcuRep::Variable loopParam = CreateVariable();
        loopParam = CcuRep::GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += goSize.loopParam;

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        auto lc = Loop("broadcast_loop_0")(src, dst, sliceSize);

        CcuRep::Variable paraCfg = CreateVariable();
        paraCfg = CcuRep::GetParallelParam(moConfig.loopCount - 1, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
        AddCcuProfiling(goSize, ccuTransports);
    }

    CCU_IF(goSize.parallelParam != 0)
    {
        src.addr += goSize.addrOffset;
        for (uint32_t i = 0; i < size; i++) {
            dst[i].addr += goSize.addrOffset;
        }

        auto lc0 = Loop("broadcast_loop_0")(src, dst, goSize.residual);

        src.addr += goSize.residual;
        for (uint32_t i = 0; i < size; i++) {
            dst[i].addr += goSize.residual;
        }

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        auto lc1 = Loop("broadcast_loop_1")(src, dst, sliceSize);

        CcuRep::Variable loopCfg0 = CreateVariable();
        loopCfg0 = CcuRep::GetLoopParam(0, 0, 1);
        CcuRep::Variable loopCfg1 = CreateVariable();
        loopCfg1 = CcuRep::GetLoopParam(0, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);
        AddCcuProfiling(goSize, ccuTransports);
    }
}

void CcuContext::CreateMultiOpReduceWithoutMyRank(
    const std::vector<CcuTransport*>& ccuTransports, DataType dataType, DataType outputDataType, ReduceOp opType)
{
    AllocGoResource();

    std::string loopType = CcuRep::GetReduceTypeStr(dataType, opType);
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }

    uint32_t size = ccuTransports.size();
    uint32_t expansionNum = CcuRep::GetReduceExpansionNum(opType, dataType, outputDataType);
    uint32_t usedBufNum = size > expansionNum ? size : expansionNum;

    for (int32_t index = 0; index < 2; index++) { // 需要实现化2个Loop
        std::vector<CcuRep::Memory> src;
        for (uint32_t i = 0; i < size; i++) {
            src.emplace_back(CreateMemory());
        }
        CcuRep::Memory dst = CreateMemory();
        CcuRep::Variable len = CreateVariable();
        CcuRep::Variable lenForExpansion = CreateVariable();
        CcuRep::LoopBlock lb(this, loopType + "_loop_" + std::to_string(index));
        lb(src, dst, len, lenForExpansion);

        std::vector<CcuRep::CcuBuffer> bufs
            = {moRes.ccuBuffer.begin() + index * moConfig.msInterleave,
               moRes.ccuBuffer.begin() + index * moConfig.msInterleave + usedBufNum};
        CcuRep::MaskSignal sem = moRes.maskSignal[index];
        for (uint32_t i = 0; i < ccuTransports.size(); i++) {
            if (ccuTransports[i] == nullptr) {
                THROW<CcuApiException>("transport is nullptr");
            }
            Read(*ccuTransports[i], bufs[i], src[i], len, sem, 1 << i);
        }
        LocalWait(sem, (1 << size) - 1);

        if (size > 1) {
            LocalReduce(bufs, size, dataType, outputDataType, opType, sem, len);
            LocalWait(sem);
        }

        LocalCopy(dst, bufs[0], lenForExpansion, sem);

        LocalWait(sem);
    }

    registeredLoop.insert(loopType);
}

void CcuContext::GroupReduceWithoutMyRank(
    const std::vector<CcuTransport*>& ccuTransports, CcuRep::Memory& dst, std::vector<CcuRep::Memory>& src,
    GroupOpSize& goSize, DataType dataType, DataType outputDataType, ReduceOp opType)
{
    CreateMultiOpReduceWithoutMyRank(ccuTransports, dataType, outputDataType, opType);

    uint32_t size = src.size();
    uint32_t expansionNum = CcuRep::GetReduceExpansionNum(opType, dataType, outputDataType);
    CcuRep::Variable sliceSizeExpansion = CreateVariable();

    if (expansionNum != 1) {
        CcuRep::Variable tmp = CreateVariable();
        tmp = CcuRep::GetExpansionParam(expansionNum);
        dst.token += tmp;
    }

    CCU_IF(goSize.loopParam != 0)
    {
        CcuRep::Variable loopParam = CreateVariable();
        loopParam = CcuRep::GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += goSize.loopParam;

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        sliceSizeExpansion = moConfig.memSlice * expansionNum;

        auto lc = Loop(CcuRep::GetReduceTypeStr(dataType, opType) + "_loop_0")(src, dst, sliceSize, sliceSizeExpansion);

        CcuRep::Variable paraCfg = CreateVariable();
        paraCfg = CcuRep::GetParallelParam(moConfig.loopCount - 1, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
        AddCcuProfiling(goSize, ccuTransports, dataType, outputDataType, opType);
    }

    CCU_IF(goSize.parallelParam != 0)
    {
        for (uint32_t i = 0; i < size; i++) {
            src[i].addr += goSize.addrOffset;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.addrOffset;
        }

        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion += goSize.residual;
        }

        auto lc0 = Loop(CcuRep::GetReduceTypeStr(dataType, opType) + "_loop_0")(
            src, dst, goSize.residual, sliceSizeExpansion);

        for (uint32_t i = 0; i < size; i++) {
            src[i].addr += goSize.residual;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.residual;
        }

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        sliceSizeExpansion = moConfig.memSlice * expansionNum;

        auto lc1
            = Loop(CcuRep::GetReduceTypeStr(dataType, opType) + "_loop_1")(src, dst, sliceSize, sliceSizeExpansion);

        CcuRep::Variable loopCfg0 = CreateVariable();
        loopCfg0 = CcuRep::GetLoopParam(0, 0, 1);
        CcuRep::Variable loopCfg1 = CreateVariable();
        loopCfg1 = CcuRep::GetLoopParam(0, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);
        AddCcuProfiling(goSize, ccuTransports, dataType, outputDataType, opType);
    }
}

void CcuContext::GroupBroadcast(
    const std::vector<CcuTransport*>& transports, std::vector<CcuRep::Memory> dst, CcuRep::Memory src,
    GroupOpSize goSize)
{
    CreateMultiOpBroadcast(transports);

    uint32_t size = transports.size() + 1;

    CCU_IF(goSize.addrOffset != 0)
    {
        CcuRep::Variable loopParam = CreateVariable();
        loopParam = CcuRep::GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += goSize.loopParam;

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        auto lc = Loop("broadcast_loop_0")(src, dst, sliceSize);

        CcuRep::Variable paraCfg = CreateVariable();
        paraCfg = CcuRep::GetParallelParam(moConfig.loopCount - 1, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
        AddCcuProfiling(goSize, transports);
    }

    CCU_IF(goSize.parallelParam != 0)
    {
        src.addr += goSize.addrOffset;
        for (uint32_t i = 0; i < size; i++) {
            dst[i].addr += goSize.addrOffset;
        }

        auto lc0 = Loop("broadcast_loop_0")(src, dst, goSize.residual);

        src.addr += goSize.residual;
        for (uint32_t i = 0; i < size; i++) {
            dst[i].addr += goSize.residual;
        }

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        auto lc1 = Loop("broadcast_loop_1")(src, dst, sliceSize);

        CcuRep::Variable loopCfg0 = CreateVariable();
        loopCfg0 = CcuRep::GetLoopParam(0, 0, 1);
        CcuRep::Variable loopCfg1 = CreateVariable();
        loopCfg1 = CcuRep::GetLoopParam(0, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);
        AddCcuProfiling(goSize, transports);
    }
}

void CcuContext::CreateMultiOpReduce(
    const std::vector<CcuTransport*>& transports, DataType dataType, DataType outputDataType, ReduceOp opType)
{
    AllocGoResource();

    std::string loopType = CcuRep::GetReduceTypeStr(dataType, opType);
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }

    uint32_t size = transports.size() + 1;
    uint32_t expansionNum = CcuRep::GetReduceExpansionNum(opType, dataType, outputDataType);
    uint32_t usedBufNum = size > expansionNum ? size : expansionNum;

    for (int32_t index = 0; index < 2; index++) { // 需要实现化2个Loop
        std::vector<CcuRep::Memory> src;
        for (uint32_t i = 0; i < size; i++) {
            src.emplace_back(CreateMemory());
        }
        CcuRep::Memory dst = CreateMemory();
        CcuRep::Variable len = CreateVariable();
        CcuRep::Variable lenForExpansion = CreateVariable();
        CcuRep::LoopBlock lb(this, loopType + "_loop_" + std::to_string(index));
        lb(src, dst, len, lenForExpansion);

        std::vector<CcuRep::CcuBuffer> bufs
            = {moRes.ccuBuffer.begin() + index * moConfig.msInterleave,
               moRes.ccuBuffer.begin() + index * moConfig.msInterleave + usedBufNum};
        CcuRep::MaskSignal sem = moRes.maskSignal[index];
        for (uint32_t i = 0; i < transports.size(); i++) {
            if (transports[i] == nullptr) {
                THROW<CcuApiException>("transport is nullptr");
            }
            Read(*transports[i], bufs[i], src[i], len, sem, 1 << i);
        }
        if (size > DATAT_SIZE_U32) {
            THROW<CcuApiException>("CcuContext::CreateMultiOpReduce size is invalid ,size[%u]", size);
        }
        LocalCopy(bufs[size - 1], src[size - 1], len, sem, 1 << (size - 1));
        LocalWait(sem, (1 << size) - 1);

        if (size > 1) {
            LocalReduce(bufs, size, dataType, outputDataType, opType, sem, len);
            LocalWait(sem);
        }

        LocalCopy(dst, bufs[0], lenForExpansion, sem);

        LocalWait(sem);
    }

    registeredLoop.insert(loopType);
}

void CcuContext::GroupReduce(
    const std::vector<CcuTransport*>& transports, CcuRep::Memory dst, std::vector<CcuRep::Memory> src,
    GroupOpSize goSize, DataType dataType, DataType outputDataType, ReduceOp opType)
{
    CreateMultiOpReduce(transports, dataType, outputDataType, opType);

    uint32_t size = transports.size() + 1;
    uint32_t expansionNum = CcuRep::GetReduceExpansionNum(opType, dataType, outputDataType);
    CcuRep::Variable sliceSizeExpansion = CreateVariable();

    if (expansionNum != 1) {
        CcuRep::Variable tmp = CreateVariable();
        tmp = CcuRep::GetExpansionParam(expansionNum);
        dst.token += tmp;
    }

    CCU_IF(goSize.loopParam != 0)
    {
        CcuRep::Variable loopParam = CreateVariable();
        loopParam = CcuRep::GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += goSize.loopParam;

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        sliceSizeExpansion = moConfig.memSlice * expansionNum;

        auto lc = Loop(CcuRep::GetReduceTypeStr(dataType, opType) + "_loop_0")(src, dst, sliceSize, sliceSizeExpansion);

        CcuRep::Variable paraCfg = CreateVariable();
        paraCfg = CcuRep::GetParallelParam(moConfig.loopCount - 1, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
        AddCcuProfiling(goSize, transports, dataType, outputDataType, opType);
    }

    CCU_IF(goSize.parallelParam != 0)
    {
        for (uint32_t i = 0; i < size; i++) {
            src[i].addr += goSize.addrOffset;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.addrOffset;
        }

        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion += goSize.residual;
        }

        auto lc0 = Loop(CcuRep::GetReduceTypeStr(dataType, opType) + "_loop_0")(
            src, dst, goSize.residual, sliceSizeExpansion);

        for (uint32_t i = 0; i < size; i++) {
            src[i].addr += goSize.residual;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.residual;
        }

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize = moConfig.memSlice;
        sliceSizeExpansion = moConfig.memSlice * expansionNum;

        auto lc1
            = Loop(CcuRep::GetReduceTypeStr(dataType, opType) + "_loop_1")(src, dst, sliceSize, sliceSizeExpansion);

        CcuRep::Variable loopCfg0 = CreateVariable();
        loopCfg0 = CcuRep::GetLoopParam(0, 0, 1);
        CcuRep::Variable loopCfg1 = CreateVariable();
        loopCfg1 = CcuRep::GetLoopParam(0, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = CcuRep::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);
        AddCcuProfiling(goSize, transports, dataType, outputDataType, opType);
    }
}

CcuRep::FuncCall CcuContext::Func(const std::string& label) { return CcuRep::FuncCall(this, label); }

CcuRep::FuncCall CcuContext::Func(const CcuRep::Variable& funcAddr) { return CcuRep::FuncCall(this, funcAddr); }

CcuRep::LoopCall CcuContext::Loop(const std::string& label) { return CcuRep::LoopCall(this, label); }

void CcuContext::LoopGroup(
    const std::vector<CcuRep::LoopCall>& loops, const std::vector<CcuRep::Variable>& loopCfg,
    const CcuRep::Variable& paraCfg, const CcuRep::Variable& offsetCfg)
{
    auto lgc = CcuRep::LoopGroupCall(this);
    std::vector<CcuRep::Executor> executors;
    for (size_t i = 0; i < loops.size(); i++) {
        executors.push_back(moRes.executor[i]);
    }
    lgc.Run(loops, loopCfg, executors, paraCfg, offsetCfg);
}

void CcuContext::SetResPack(CcuResPack& resPack) { resPack_ = &resPack; }

CcuResPack* CcuContext::GetResPack() const { return resPack_; }

void CcuContext::SetInstrId(uint32_t instrId)
{
    HCCL_INFO("[SetInstrId] Input params: instrId[%u]", instrId);
    instrInfo.startInstrId = instrId;
}

uint32_t CcuContext::GetInstrId() const { return instrInfo.startInstrId; }

uint32_t CcuContext::GetInstrCount()
{
    uint32_t instrCount = 0;
    for (const auto& rep : GetRepSequence()) {
        instrCount += rep->InstrCount();
    }
    instrInfo.instrCount = instrCount;
    HCCL_INFO("Ctx inst %u", instrCount);
    return instrCount;
}

void CcuContext::SetCcuInstrInfo(const CcuRep::CcuInstrInfo& instrInfo)
{
    HCCL_INFO(
        "[SetCcuInstrInfo] Input params: instrVec size[%u], startInstrId[%u], instrCount[%u], missionStartInstrId[%u], "
        "missionInstrCount[%u]",
        instrInfo.instrVec.size(), instrInfo.startInstrId, instrInfo.instrCount, instrInfo.missionStartInstrId,
        instrInfo.missionInstrCount);
    this->instrInfo = instrInfo;
}

template <typename T>
T CcuContext::CreateResAssist(std::array<std::vector<T>, MAX_CCU_IODIE_NUM>& resRecord)
{
    // 获取DieId
    uint32_t dieId = GetDieId();
    // 检查DieId是否越界
    if (dieId >= MAX_CCU_IODIE_NUM) {
        THROW<CcuApiException>("dieId[%u] out of range[0, %u]", dieId, MAX_CCU_IODIE_NUM - 1);
    }
    resRecord[dieId].emplace_back(this);

    auto& item = resRecord[dieId].back();
    item.Reset(resRecord[dieId].size(), dieId);
    return item;
}

CcuRep::Variable CcuContext::CreateVariable() { return CreateResAssist(res.continuousVariable); }

CcuRep::Variable CcuContext::CreateContinuousVariable() { return CreateResAssist(res.continuousVariable); }

CcuRep::Address CcuContext::CreateAddress() { return CreateResAssist(res.blockAddress); }

CcuRep::MaskSignal CcuContext::CreateMaskSignal() { return CreateResAssist(res.blockMaskSignal); }

CcuRep::CcuBuffer CcuContext::CreateCcuBuffer() { return CreateResAssist(res.blockCcubuffers); }

CcuRep::Executor CcuContext::CreateExecutor() { return CreateResAssist(res.blockExecutor); }

CcuRep::Memory CcuContext::CreateMemory() { return CcuRep::Memory(CreateAddress(), CreateVariable()); }

CcuRep::Memory CcuContext::GetRmtBuffer(const CcuTransport& transport, uint32_t index)
{
    (void)index;
    auto mem = CcuRep::Memory(CreateAddress(), CreateVariable());
    Append(std::make_shared<CcuRep::CcuRepRemMem>(transport, mem));
    return mem;
}

CcuRep::Memory CcuContext::CreateMemory(const CcuRep::Variable& token)
{
    return CcuRep::Memory(CreateAddress(), token);
}

CcuContext::GroupOpSize CcuContext::CreateGroupOpSize()
{
    return GroupOpSize{CreateVariable(), CreateVariable(), CreateVariable(), CreateVariable()};
}

template <typename T>
std::vector<T>
CcuContext::CreateBlockResAssist(uint32_t count, std::array<std::vector<T>, MAX_CCU_IODIE_NUM>& resRecord)
{
    std::vector<T> block;
    // 获取DieId
    uint32_t dieId = GetDieId();
    // 检查DieId是否越界
    if (dieId >= MAX_CCU_IODIE_NUM) {
        THROW<CcuApiException>("dieId[%u] out of range[0, %u]", dieId, MAX_CCU_IODIE_NUM - 1);
    }
    block.reserve(count);
    for (size_t i = 0; i < count; i++) {
        block.emplace_back(this);
        block.back().Reset(0x1000 + resRecord[dieId].size() + i, dieId); // 0x1000分割Block资源和离散资源
    }
    resRecord[dieId].insert(resRecord[dieId].end(), block.begin(), block.end());
    return block;
}

std::vector<CcuRep::CcuBuffer> CcuContext::CreateBlockCcuBuffer(uint32_t count)
{
    return CreateBlockResAssist(count, res.blockCcubuffers);
}

std::vector<CcuRep::Executor> CcuContext::CreateBlockExecutor(uint32_t count)
{
    return CreateBlockResAssist(count, res.blockExecutor);
}

std::vector<CcuRep::MaskSignal> CcuContext::CreateBlockMaskSignal(uint32_t count)
{
    return CreateBlockResAssist(count, res.blockMaskSignal);
}

/*
 * 功能描述：通过goSize varId获取其对应的task arg index。当前仅支持两种场景：
 * 场景1：goSize var直接通过LoadArg赋值得到；
 * 场景2：goSize var经过LoadArg和若干Assign(varB, varA)操作得到。
 */
uint64_t CcuContext::GetArgIndex(
    const std::unordered_map<uint16_t, uint16_t>& varId2VarIdMap,
    const std::unordered_map<uint16_t, uint32_t>& varId2ArgIndexMap, const std::vector<uint64_t>& taskArgs,
    uint16_t varId) const
{
    HCCL_INFO("[GetArgIndex] Enter varId(%u)", varId);
    auto item = varId2ArgIndexMap.find(varId);
    if (item == varId2ArgIndexMap.end()) {
        string msg = StringFormat("Invalid goSize variable id(%u).", varId);
        uint16_t oriVarId = varId;
        auto iter = varId2VarIdMap.find(varId);
        while (iter != varId2VarIdMap.end()) { // 循环查找中间assign Rep，找到起始varId
            oriVarId = iter->second;
            iter = varId2VarIdMap.find(oriVarId);
        }
        if (oriVarId != varId) { // 起始varId预期通过LoadArg赋值
            item = varId2ArgIndexMap.find(oriVarId);
            if (item == varId2ArgIndexMap.end()) {
                THROW<CcuApiException>(msg);
            }
        } else {
            THROW<CcuApiException>(msg);
        }
    }
    HCCL_INFO("[GetArgIndex] find end");
    if (item->second >= taskArgs.size()) {
        string msg = StringFormat("Invalid goSize variable index(%u).", item->second);
        THROW<CcuApiException>(msg);
    }
    HCCL_INFO(
        "GetArgIndex success: varId(%u) varId2VarIdMapSize(%u) varId2ArgIndexMapSize(%u) taskArgsSize(%u)", varId,
        varId2VarIdMap.size(), varId2ArgIndexMap.size(), taskArgs.size());
    return taskArgs[item->second];
}

void CcuContext::AddCcuProfiling(GroupOpSize goSize, const std::vector<CcuTransport*>& transportsIn)
{
    AddProfiling(transportsIn);
    groupOpSizeInfo.push_back(goSize);
}

void CcuContext::AddCcuProfiling(
    GroupOpSize goSize, const std::vector<CcuTransport*>& transportsIn, DataType dataType, DataType outputDataType,
    ReduceOp opType)
{
    AddProfiling(transportsIn, dataType, outputDataType, opType);
    groupOpSizeInfo.push_back(goSize);
}

/*
 * variable/maskSignal等资源变量Id，一定要在获取ccu profiling时才获取；
 * 原因：在创建context Rep时，其资源Id属于虚拟资源；翻译时，才会绑定固定的物理资源。
 */
HcclResult CcuContext::GetCcuProfilingInfo(const CcuTaskArg& arg, std::vector<CcuProfilingInfo>& allCcuProfilingInfo)
{
    HCCL_INFO("[GetCcuProfilingInfo] Enter.");
    std::vector<CcuProfilingInfo> allCcuProfilingInfos;
    auto& ccuProfilingCache = GetProfilingInfo();

    auto taskArgs = GeneArgs(arg);
    uint32_t count = 0;
    HCCL_INFO("[GetCcuProfilingInfo] Process sqe&waitcke profiling info start.");
    for (auto& profInfo : ccuProfilingCache) {
        profInfo.missionId = GetMissionId();
        if (profInfo.type == CcuProfilinType::CCU_TASK_PROFILING) {
            profInfo.instrId = GetInstrId();
            allCcuProfilingInfos.push_back(profInfo);
            continue;
        }
        if (count >= GetWaiteCkeProfilingReps().size()) {
            HCCL_ERROR(
                "count[%u] out of range[0, %u], cache size(%u).", count, GetWaiteCkeProfilingReps().size(),
                ccuProfilingCache.size());
            return HCCL_E_INTERNAL;
        }
        auto waitCkeRep = GetWaiteCkeProfilingReps()[count];
        profInfo.instrId = waitCkeRep->StartInstrId();
        if (profInfo.ckeId == INVALID_CKE_ID) { // localWait Rep
            if (waitCkeRep.get() == nullptr) {
                HCCL_ERROR("[GetCcuProfilingInfo] localWaitRep is nullptr.");
                return HCCL_E_PTR;
            }
            auto localWaitRep = dynamic_cast<CcuRep::CcuRepLocWaitSem*>(waitCkeRep.get());
            profInfo.ckeId = localWaitRep->GetSemId();
        }
        allCcuProfilingInfos.push_back(profInfo);
        count++;
    }

    // loopGroup
    auto& lgProfInfo = GetLGProfilingInfo();
    HCCL_INFO("[GetCcuProfilingInfo] create varId2ArgIndexMap start. size=%lu", lgProfInfo.loadRep2ArgIdxMap.size());
    std::unordered_map<uint16_t, uint32_t> varId2ArgIndexMap;
    for (auto& iter : lgProfInfo.loadRep2ArgIdxMap) {
        if (iter.first.get() == nullptr) {
            HCCL_ERROR("[GetCcuProfilingInfo] loadRep is nullptr.");
            return HCCL_E_PTR;
        }
        auto loadRep = dynamic_cast<CcuRep::CcuRepLoadArg*>(iter.first.get());
        varId2ArgIndexMap[loadRep->GetVarId()] = iter.second;
    }

    HCCL_INFO("[GetCcuProfilingInfo] create varId2VarIdMap start. size=%lu", lgProfInfo.assignProfilingReps.size());
    std::unordered_map<uint16_t, uint16_t> varId2VarIdMap;
    for (auto& iter : lgProfInfo.assignProfilingReps) {
        if (iter.get() == nullptr) {
            HCCL_ERROR("[GetCcuProfilingInfo] assignRep is nullptr.");
            return HCCL_E_PTR;
        }
        auto assignRep = dynamic_cast<CcuRep::CcuRepAssign*>(iter.get());
        varId2VarIdMap[assignRep->varB.Id()] = assignRep->varA.Id();
    }

    HCCL_INFO(
        "[GetCcuProfilingInfo] process loop group profiling start: lgsize(%lu), goSize(%lu)",
        lgProfInfo.lgProfilingReps.size(), groupOpSizeInfo.size());
    for (uint32_t i = 0; i < lgProfInfo.lgProfilingReps.size();
         i += 2) { // 2: 一个goSize对应一个CcuProfilingInfo，对应1个loopGroup Rep
        if (taskArgs.empty() || varId2ArgIndexMap.empty()) {
            continue;
        }
        uint64_t loopParam
            = GetArgIndex(varId2VarIdMap, varId2ArgIndexMap, taskArgs, groupOpSizeInfo[i].loopParam.Id());
        uint64_t parallelParam
            = GetArgIndex(varId2VarIdMap, varId2ArgIndexMap, taskArgs, groupOpSizeInfo[i].parallelParam.Id());
        HCCL_INFO(
            "Collect loopgroup profiling info: repSize[%u], index[%u], loopParam[%llu], parallelParam[%llu].",
            lgProfInfo.lgProfilingReps.size(), i, loopParam, parallelParam);

        if (loopParam != 0) {
            lgProfInfo.ccuProfilingInfos[i].dataSize = loopParam * moConfig.loopCount * moConfig.memSlice;
            lgProfInfo.ccuProfilingInfos[i].instrId
                = dynamic_cast<CcuRep::CcuRepLoopGroup*>(lgProfInfo.lgProfilingReps[i].get())->StartInstrId();
            allCcuProfilingInfos.push_back(lgProfInfo.ccuProfilingInfos[i]);
        }

        if (parallelParam != 0) {
            HCCL_INFO("[GetCcuProfilingInfo] collect lg, residual start i=%lu", i);
            uint64_t residual
                = GetArgIndex(varId2VarIdMap, varId2ArgIndexMap, taskArgs, groupOpSizeInfo[i].residual.Id());
            uint64_t repeatNum = CcuRep::ParseRepeatNumFromParallelParam(parallelParam);
            lgProfInfo.ccuProfilingInfos[i].dataSize = repeatNum * moConfig.memSlice + residual;
            lgProfInfo.ccuProfilingInfos[i].instrId
                = dynamic_cast<CcuRep::CcuRepLoopGroup*>(lgProfInfo.lgProfilingReps[i + 1].get())->StartInstrId();
            allCcuProfilingInfos.push_back(lgProfInfo.ccuProfilingInfos[i]);
        }
    }
    DumpCcuProfilingInfo(allCcuProfilingInfos);
    allCcuProfilingInfo = allCcuProfilingInfos;
    return HCCL_SUCCESS;
}

void CcuContext::DumpCcuProfilingInfo(const std::vector<CcuProfilingInfo>& ccuProfilingInfo) const
{
    auto dumpLinkInfo = [](const CcuProfilingInfo& info) -> void {
        for (int i = 0; i < CCU_MAX_CHANNEL_NUM; i++) {
            if (info.channelId[i] == INVALID_VALUE_CHANNELID) {
                continue;
            }
            HCCL_INFO("channelId(%u), remoteRankId(%u).", info.channelId[i], info.remoteRankId[i]);
        }
    };

    for (const auto& profInfo : ccuProfilingInfo) {
        if (profInfo.type == CcuProfilinType::CCU_TASK_PROFILING) {
            HCCL_INFO(
                "Dump CCU Profiling Info:SQE Profiling Info: ctxSignature(%s), "
                "dieId(%d), missionId(%d), instrId(%d).",
                profInfo.name.c_str(), static_cast<int>(profInfo.dieId), static_cast<int>(profInfo.missionId),
                static_cast<int>(profInfo.instrId));
        } else if (profInfo.type == CcuProfilinType::CCU_WAITCKE_PROFILING) {
            HCCL_INFO(
                "Microcode WaitCKE Profiling Info: name(%s), "
                "dieId(%d), missionId(%d), instrId(%d), ckeId(%u), mask(%u).",
                profInfo.name.c_str(), static_cast<int>(profInfo.dieId), static_cast<int>(profInfo.missionId),
                static_cast<int>(profInfo.instrId), profInfo.ckeId, profInfo.mask);
            dumpLinkInfo(profInfo);
        } else if (profInfo.type == CcuProfilinType::CCU_LOOPGROUP_PROFILING) {
            HCCL_INFO(
                "Microcode LoopGroup Profiling Info: name(%s), "
                "dieId(%d), missionId(%d), instrId(%d), reduceOpType(%d), inputDataType(%d), "
                "outputDataType(%d), dataSize(%llu).",
                profInfo.name.c_str(), static_cast<int>(profInfo.dieId), static_cast<int>(profInfo.missionId),
                static_cast<int>(profInfo.instrId), static_cast<int>(profInfo.reduceOpType),
                static_cast<int>(profInfo.inputDataType), static_cast<int>(profInfo.outputDataType), profInfo.dataSize);
            dumpLinkInfo(profInfo);
        }
    }
}

}; // namespace Hccl
