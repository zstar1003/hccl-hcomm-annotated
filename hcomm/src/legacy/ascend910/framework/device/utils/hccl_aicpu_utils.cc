/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "utils/hccl_aicpu_utils.h"
#include <sstream>
#include <dlog_pub.h>

#include "algorithm/task_orchestrator.h"
#include "common/aicpu_sqe_context.h"
#include "common/aicpu_hccl_common.h"
#include "profiling_manager_device.h"
#include "framework/aicpu_communicator.h"
#include "log.h"
#include "hccl_types.h"
#include "transport_pub.h"

int32_t HcclAicpuUtils::GetCpuId()
{
    static thread_local int32_t curCpu = -1;
    if (curCpu < 0) {
        curCpu = sched_getcpu();
    }
    return curCpu;
}

int32_t HcclAicpuUtils::GetCurClusterId() { return !!(GetCpuId() & (AICPU_CNT / CLUSTER_CNT)); }

void HcclAicpuUtils::PrintHcclCombinOpParam(const HccCommResParamTask& commParam)
{
    if (!HcclCheckLogLevel(HCCL_LOG_INFO)) {
        return;
    }
    HCCL_INFO("HccCommResParamTask.workSpace %p", commParam.mc2WorkSpace.workSpace);
    HCCL_INFO("HccCommResParamTask.workSpaceSize %lu", commParam.mc2WorkSpace.workSpaceSize);
    HCCL_INFO("HccCommResParamTask.rankId %u", commParam.rankId);
    HCCL_INFO("HccCommResParamTask.rankNum %u", commParam.rankNum);
    HCCL_INFO("HccCommResParamTask.winSize %lu", commParam.winSize);
    for (uint32_t i = 0; i < AC_MAX_RANK_NUM; i++) {
        HCCL_INFO(
            "HccCommResParamTask.windowsIn[%u] %p, windowsOut[%u] %p", i, commParam.windowsIn[i], i,
            commParam.windowsOut[i]);
    }
    for (uint32_t i = 0; i < AC_MAX_RANK_NUM; i++) {
        const HcclStreamInfo& sinfo = commParam.streamInfo[i];
        HCCL_INFO(
            "HccCommResParamTask.streamInfo[%u] streamId %d, sqId %u, cqId %lu logicCqid %u", i, sinfo.streamIds,
            sinfo.sqIds, sinfo.cqIds, sinfo.logicCqids);
    }
    for (uint32_t i = 0; i < AC_MAX_RANK_NUM * 2; i++) { // 2 is number of noIpcNotify
        const HcclSignalInfo& sinfo = commParam.signalInfo.noIpcNotifys[i];
        HCCL_INFO(
            "HccCommResParamTask.noIpcNotifys[%u] resId %lu, addr %p, devId %u, tsId %u, rankId %u", i, sinfo.resId,
            sinfo.addr, sinfo.devId, sinfo.tsId, sinfo.rankId);
    }
    for (uint32_t i = 0; i < AC_MAX_RANK_NUM * 4; i++) { // 4 is number of ipcNotifys
        const HcclSignalInfo& sinfo = commParam.signalInfo.ipcNotifys[i];
        HCCL_INFO(
            "HccCommResParamTask.ipcNotifys[%u] resId %lu, addr %p, devId %u, tsId %u, rankId %u", i, sinfo.resId,
            sinfo.addr, sinfo.devId, sinfo.tsId, sinfo.rankId);
    }
    for (uint32_t i = 0; i < AC_MAX_RANK_NUM; i++) {
        const HcclSignalInfo& sinfo = commParam.signalInfo.noIpcEvents[i];
        HCCL_INFO(
            "HccCommResParamTask.noIpcEvents[%u] resId %lu, addr %p, devId %u, tsId %u, rankId %u", i, sinfo.resId,
            sinfo.addr, sinfo.devId, sinfo.tsId, sinfo.rankId);
    }

    for (uint32_t i = 0; i < AICPU_OP_NOTIFY_NUM; i++) {
        const HcclSignalInfo& sinfo = commParam.signalInfo.aicpuOpNotify[i];
        HCCL_INFO(
            "HccCommResParamTask.aicpuOpNotify[%u] resId %lu, addr %p, devId %u, tsId %u, rankId %u", i, sinfo.resId,
            sinfo.addr, sinfo.devId, sinfo.tsId, sinfo.rankId);
    }
    const auto& sigInfo = commParam.signalInfo.aicpuNotify;
    HCCL_INFO(
        "HccCommResParamTask.aicpuNotify resId %lu, addr %p, devId %u, tsId %u, rankId %u", sigInfo.resId, sigInfo.addr,
        sigInfo.devId, sigInfo.tsId, sigInfo.rankId);
    HCCL_INFO("HccCommResParamTask.determinism %u", commParam.config.deterministic);
    HCCL_INFO("HccCommResParamTask.overflowAddr %p", commParam.overFlowAddr);
    HCCL_INFO("HccCommResParamTask.retryParams: retryEnable %u", commParam.config.retryEnable);
}

void HcclAicpuUtils::PrintHcclOpResParam(const HcclOpResParam* resParam)
{
    if (resParam == nullptr) {
        HCCL_ERROR("[%s] resParam is nullptr", __func__);
        return;
    }
    HCCL_INFO("HcclOpResParam.rankId %u", resParam->localUsrRankId);
    HCCL_INFO("HcclOpResParam.rankNum %u", resParam->rankSize);

    HCCL_INFO("HcclOpResParam.windowSize %lu", resParam->winSize);
    HCCL_INFO("HcclOpResParam.workSpaceAddr %p", resParam->mc2WorkSpace.workSpace);
    HCCL_INFO("HcclOpResParam.workSpaceSize %lu", resParam->mc2WorkSpace.workSpaceSize);
    for (uint32_t i = 0; i < resParam->localRes.streamNum; i++) {
        const auto streamInfo = resParam->localRes.streamParam[i].streamInfo;
        HCCL_INFO(
            "HcclOpResParam.streamInfo[%u] streamId %d sqId %u cqId %u logicCqid %u", i, streamInfo.streamIds,
            streamInfo.sqIds, streamInfo.cqIds, streamInfo.logicCqids);
    }
    const auto mainStreamInfo = resParam->localRes.mainStreamParam.streamInfo;
    HCCL_INFO(
        "HcclOpResParam.mainStreamInfo streamId %d sqId %u cqId %u logicCqid %u", mainStreamInfo.streamIds,
        mainStreamInfo.sqIds, mainStreamInfo.cqIds, mainStreamInfo.logicCqids);
    for (uint32_t i = 0; i < resParam->localRes.signalNum; i++) {
        const auto& signals = resParam->localRes.localSignals[i];
        HCCL_INFO(
            "HcclOpResParam.localSignals[%u] resId %p addr %p devId %u tsId %u rankId %u", i, signals.resId,
            signals.addr, signals.devId, signals.tsId, signals.rankId);
    }

    for (uint32_t i = 0; i < AICPU_OP_NOTIFY_MAX_NUM; i++) {
        const auto& aicpuOpNotify = resParam->localRes.aicpuOpNotify[i];
        HCCL_INFO(
            "HcclOpResParam.aicpuOpNotify[%u] resId %p addr %p devId %u tsId %u rankId %u", i, aicpuOpNotify.resId,
            aicpuOpNotify.addr, aicpuOpNotify.devId, aicpuOpNotify.tsId, aicpuOpNotify.rankId);
    }
    HCCL_INFO("HcclOpResParam.determinism %u", resParam->config.deterministic);
}

HcclResult HcclAicpuUtils::Getkey(
    const AicpuComContext& ctx, u32 remoteRankId, const void* userAddr, u64 length, u32& outKey, int32_t keyType)
{
    CHK_PTR_NULL(userAddr);
    HCCL_INFO("[HcclAicpuUtils][Getkey] addr[%p] len[%llu]", userAddr, length);
    u64 inAddr = reinterpret_cast<u64>(userAddr);
    MemDetails inputMem
        = (keyType == LOCAL) ? ctx.ibversData[remoteRankId].localInputMem : ctx.ibversData[remoteRankId].remoteInputMem;
    MemDetails outputMem = (keyType == LOCAL) ? ctx.ibversData[remoteRankId].localOutputMem :
                                                ctx.ibversData[remoteRankId].remoteOutputMem;

    u64 inputStartAddr = inputMem.addr;
    u64 inputCCLSize = inputMem.size;
    u32 inputKey = inputMem.key;

    u64 outputStartAddr = outputMem.addr;
    u64 outputCCLSize = outputMem.size;
    u32 outputKey = outputMem.key;
    if (inAddr >= inputStartAddr && inAddr < inputStartAddr + inputCCLSize) {
        outKey = inputKey;
    } else if (inAddr >= outputStartAddr && inAddr <= outputStartAddr + outputCCLSize) {
        outKey = outputKey;
    } else {
        HCCL_ERROR(
            "[HcclAicpuUtils][Getkey]src_ptr=%p is out of range, inputmem src[%p], size[%llu];"
            " outputmem src[%p] size[%llu]",
            userAddr, inputStartAddr, inputCCLSize, outputStartAddr, outputCCLSize);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO(
        "[HcclAicpuUtils][Getkey] addr[%p] length[%llu] outKey[%u], keyType:[%s]", userAddr, length, outKey,
        (keyType == LOCAL) ? "local" : "remote");

    return HCCL_SUCCESS;
}

std::mutex g_mtxForDoorbell;
HcclResult HcclAicpuUtils::PostSend(
    const AicpuComContext& ctx, u32 remoteRankId, struct std::vector<hccl::Transport::Buffer>& remoteBuf,
    struct std::vector<hccl::Transport::Buffer>& localBuf, bool isWrite)
{
    if (UNLIKELY(remoteRankId >= ctx.rankNum)) {
        HCCL_ERROR("[AicpuIbverbs][PostSend] remoteRankId %u is out of range, ranknum %u", remoteRankId, ctx.rankNum);
        return HCCL_E_PARA;
    }
    CHK_PRT_RET(
        remoteBuf.size() != localBuf.size(),
        HCCL_ERROR(
            "[AicpuIbverbs][PostSend] remoteBuf list size %u is not equal localBuffer list size %u ", remoteBuf.size(),
            localBuf.size()),
        HCCL_E_PARA);

    uint32_t len = remoteBuf.size();
    const uint32_t MAX_MEM_NUM = 8;
    CHK_PRT_RET(
        len > MAX_MEM_NUM,
        HCCL_ERROR("[AicpuIbverbs][PostSend] buffer size is:%u over MAX_MEM_NUM: %u", len, MAX_MEM_NUM), HCCL_E_PARA);

    MemDetails localMems[MAX_MEM_NUM];
    MemDetails remoteMems[MAX_MEM_NUM];
    u32 lkey = 0;
    u32 rkey = 0;
    for (uint32_t index = 0; index < len; index++) {
        u64 remBuffSize = remoteBuf[index].size;
        u64 locBuffSize = localBuf[index].size;
        CHK_PRT_RET(
            remBuffSize != locBuffSize,
            HCCL_ERROR(
                "[AicpuIbverbs][PostSend] remoteBuf size %u is not equal localBuffer size %u ", remBuffSize,
                locBuffSize),
            HCCL_E_PARA);
        // 获取WR的lkey和rkey
        CHK_RET(Getkey(ctx, remoteRankId, localBuf[index].addr, locBuffSize, lkey, LOCAL));
        CHK_RET(Getkey(ctx, remoteRankId, remoteBuf[index].addr, remBuffSize, rkey, REMOTE));
        // 设置MemDetails
        localMems[index].addr = reinterpret_cast<u64>(localBuf[index].addr);
        localMems[index].size = locBuffSize;
        localMems[index].key = lkey;

        remoteMems[index].addr = reinterpret_cast<u64>(remoteBuf[index].addr);
        remoteMems[index].size = remBuffSize;
        remoteMems[index].key = rkey;
    }

    u64 db_info = 0;
    u32 memNum = (ctx.ibversData[remoteRankId].qpMode != QPMode::NORMAL) ? 1 : len;
    CHK_RET(
        LIKELY(isWrite) ? hccl::Transport::HcclBatchWrite(
                              ctx.ibversData[remoteRankId], &localMems[0], &remoteMems[0], memNum, db_info) :
                          hccl::Transport::HcclBatchRead(
                              ctx.ibversData[remoteRankId], &localMems[0], &remoteMems[0], memNum, db_info));

    if (UNLIKELY(ctx.ibversData[remoteRankId].qpMode != QPMode::NORMAL)) {
        for (u32 i = 1; i < len; i++) {
            CHK_RET(
                LIKELY(isWrite) ? hccl::Transport::HcclBatchWrite(
                                      ctx.ibversData[remoteRankId], &localMems[i], &remoteMems[i], 1, db_info) :
                                  hccl::Transport::HcclBatchRead(
                                      ctx.ibversData[remoteRankId], &localMems[i], &remoteMems[i], 1, db_info));
        }
        u64 roceBaseAddr = 0x2000000000ULL;
        u64 roceVfDbCfg0Reg = 0x230ULL;
        u64 chipAddrOffset = 0x80000000000ULL;
        u64 dieAddrOffset = 0x10000000000ULL;
        u64 dbDieIdMask = 0x00ff0000;
        u64 dbDieIdShift = 16; // 16 is dbDieIdShift
        u64 dbAddr = roceBaseAddr + roceVfDbCfg0Reg + chipAddrOffset * ctx.chipId
                     + dieAddrOffset * ((ctx.ibversData[remoteRankId].qpInfo.dbIndex & dbDieIdMask) >> dbDieIdShift);
        HCCL_DEBUG("chipId : %llu", ctx.chipId);
        std::lock_guard<std::mutex> lock(g_mtxForDoorbell);
        CHK_RET(AicpuDispatcher::RdmaSend(0, db_info, dbAddr, remoteRankId));
        CHK_RET(AicpuDispatcher::LaunchTask(0));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclAicpuUtils::PostSend(
    const u32 lKey, const u32 rKey, const struct HcclQpInfoV2& qpInfo, const struct hccl::Transport::Buffer& remoteBuf,
    const struct hccl::Transport::Buffer& localBuf, const bool isWrite)
{
    MemDetails localMems;
    MemDetails remoteMems;
    u64 remainDataSize = localBuf.size;
    u64 remBuffSize = remainDataSize;
    u64 locBuffSize = remainDataSize;

    // 设置MemDetails
    localMems.addr = reinterpret_cast<u64>(localBuf.addr);
    localMems.size = locBuffSize;
    localMems.key = lKey;

    remoteMems.addr = reinterpret_cast<u64>(remoteBuf.addr);
    remoteMems.size = remBuffSize;
    remoteMems.key = rKey;

    u64 db_info = 0;
    u32 memNum = 1;
    struct hccl::TransportDeviceNormalData ibversDataforRemoteRank;
    ibversDataforRemoteRank.qpInfo = qpInfo;
    HCCL_INFO(
        "remBuffSize is [%u], locBuffSize is [%u], localMems.addr is [%p], localMems.size is [%u],"
        "localMems.key is [%u], remoteMems.addr is [%p], remoteMems.size is [%u], remoteMems.key is [%u],"
        "ibversDataforRemoteRank.qpInfo.qpPtr is [%p]",
        remBuffSize, locBuffSize, localMems.addr, localMems.size, localMems.key, remoteMems.addr, remoteMems.size,
        remoteMems.key, ibversDataforRemoteRank.qpInfo.qpPtr);
    while (remainDataSize > 0) {
        u64 chunkBytes = (remainDataSize > MAX_RDMA_WQE_SIZE) ? MAX_RDMA_WQE_SIZE : remainDataSize;
        localMems.size = chunkBytes;
        remoteMems.size = chunkBytes;
        CHK_RET(
            LIKELY(isWrite) ?
                hccl::Transport::HcclBatchWrite(ibversDataforRemoteRank, &localMems, &remoteMems, memNum, db_info) :
                hccl::Transport::HcclBatchRead(ibversDataforRemoteRank, &localMems, &remoteMems, memNum, db_info));
        localMems.addr += chunkBytes;
        remoteMems.addr += chunkBytes;
        remainDataSize -= chunkBytes;
    }
    return HCCL_SUCCESS;
}

u32 HcclAicpuUtils::GetBlockNum(u32 defaultVal)
{
    if (AicpuGetBlockNum != nullptr) {
        return AicpuGetBlockNum();
    } else if (aicpu::GetBlockNum != nullptr) {
        return aicpu::GetBlockNum();
    } else {
        return defaultVal;
    }
}

u32 HcclAicpuUtils::GetBlockIdx()
{
    u32 res = 0U;
    if (AicpuGetBlockIdx != nullptr) {
        res = AicpuGetBlockIdx();
    } else if (aicpu::GetBlockIdx != nullptr) {
        res = aicpu::GetBlockIdx();
    }
    return res;
}
