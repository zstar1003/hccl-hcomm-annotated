/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include <string>
#include <dlog_pub.h>
#include <hccl/hccl_types.h>
#include "log.h"
#include "securec.h"
#include "aicpu_communicator.h"
#include "common/aicpu_hccl_common.h"
#include "common/aicpu_sqe_context.h"
#include "executor_tracer.h"
#include "utils/hccl_aicpu_utils.h"
#include "hccl_types.h"
#include "framework/aicpu_hccl_process.h"
#include "dtype_common.h"
#include "aicpu_one_side_service.h"
#include "coll_comm_aicpu_mgr.h"

using namespace hccl;
using namespace HcclApi;
namespace {
AicpuComContext g_comContext[CLUSTER_CNT];
DevType g_devType = DevType::DEV_TYPE_COUNT;
} // namespace

DevType AicpuHcclProcess::AicpuGetInnerDevType() { return g_devType; }

static constexpr uint32_t ALLTOALLV_INFO_INDEX_2 = 2;
static constexpr uint32_t ALLTOALLV_INFO_INDEX_3 = 3;
static constexpr uint32_t ALLTOALLV_INFO_INDEX_4 = 4;

AicpuComContext* AicpuGetComContext()
{
    auto clusterId = HcclAicpuUtils::GetCurClusterId();
    return &g_comContext[clusterId];
}

void AicpuGetAllComContext(AicpuComContext*& contextBase, uint32_t& contextNum)
{
    contextBase = &g_comContext[0];
    contextNum = CLUSTER_CNT;
    return;
}

void AicpuHcclProcess::CallMC2MaintenanceThread(AicpuComContext* ctx)
{
    if (!IsSupportStartMC2MaintenanceThread()) {
        return;
    }

    if (ctx->commOpenStatus && (ctx->devType == DevType::DEV_TYPE_310P1 || ctx->devType == DevType::DEV_TYPE_310P3)) {
        return;
    }

    HCCL_INFO(
        "Register back ground func on device type %u, status %u.", static_cast<u32>(ctx->devType),
        static_cast<u32>(ctx->commOpenStatus));
    hrtHalStartMC2MaintenanceThread(
        dfx_tracer::ExecutorTracer::BackGroundDfx, ctx, dfx_tracer::ExecutorTracer::StopBackGroundDfx, ctx);
}

HcclResult AicpuHcclProcess::CalcDataSize(
    HcclCMDType op, HcclDataType type, u64 count, u32 rankSize, u64& inputSize, u64& outputSize)
{
    u32 perDataSize = DataUnitSize(type);
    if (perDataSize == 0) {
        HCCL_ERROR("[AicpuHcclProcess][CalcDataSize] type [%u] DataUnitSize is 0", type);
        return HCCL_E_PARA;
    }

    switch (op) {
        case HcclCMDType::HCCL_CMD_ALLGATHER:
            inputSize = count * perDataSize;
            outputSize = rankSize * count * perDataSize;
            break;
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
        case HcclCMDType::HCCL_CMD_SCATTER:
            inputSize = rankSize * count * perDataSize;
            outputSize = count * perDataSize;
            break;
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
        case HcclCMDType::HCCL_CMD_BROADCAST:
        case HcclCMDType::HCCL_CMD_REDUCE:
        case HcclCMDType::HCCL_CMD_SEND:
        case HcclCMDType::HCCL_CMD_RECEIVE:
        default:
            inputSize = count * perDataSize;
            outputSize = count * perDataSize;
            break;
    }

    HCCL_DEBUG(
        "[AicpuHcclProcess][CalcDataSize] perDataSize %u count %lu rankSize %u input %lu output %lu", perDataSize,
        count, rankSize, inputSize, outputSize);
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::CalcDataSizeV(hccl::OpParam& param, u32 rankSize)
{
    const HcclDataType type = param.GetDataType();
    u32 perDataSize = DataUnitSize(type);
    if (perDataSize == 0) {
        HCCL_ERROR("[AicpuHcclProcess][CalcDataSizeV] type[%u] DataUnitSize is 0", type);
        return HCCL_E_PARA;
    }

    const u32 rankId = param.srcRank;
    if (rankId >= rankSize) {
        HCCL_ERROR(
            "[AicpuHcclProcess][CalcDataSizeV] rankId[%u] should not be bigger than rankSize[%u]", rankId, rankSize);
        return HCCL_E_PARA;
    }

    const HcclCMDType op = param.opType;
    switch (op) {
        case HcclCMDType::HCCL_CMD_ALLGATHER_V: {
            const u64* counts = static_cast<u64*>(param.VDataDes.counts);
            const u64 totalSize = std::accumulate(counts, counts + rankSize, 0ULL) * perDataSize;
            param.inputSize = param.GetDataCount(rankId) * perDataSize;
            param.outputSize = totalSize;
            break;
        }
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V: {
            const u64* counts = static_cast<u64*>(param.VDataDes.counts);
            const u64 totalSize = std::accumulate(counts, counts + rankSize, 0ULL) * perDataSize;
            param.inputSize = totalSize;
            param.outputSize = param.GetDataCount(rankId) * perDataSize;
            break;
        }
        default:
            HCCL_ERROR("[AicpuHcclProcess][CalcDataSizeV] op[%u] is not supported", op);
            return HCCL_E_PARA;
    }

    HCCL_DEBUG(
        "[AicpuHcclProcess][CalcDataSizeV] opType[%u] perDataSize[%u] rankId[%u] rankSize[%u] input[%llu] "
        "output[%llu]",
        op, perDataSize, rankId, rankSize, param.inputSize, param.outputSize);
    return HCCL_SUCCESS;
}

u64 AicpuHcclProcess::CalcOpTilingVDataDesVDataLen(u32 rankSize)
{
    const u32 vFactor = 2; // counts和displs 2个变长数组
    return vFactor * rankSize * sizeof(u64);
}

u32 AicpuHcclProcess::AicpuRpcResInitV2(HcclOpResParam* commParam, bool isCustom)
{
    HCCL_DEBUG("[AicpuHcclProcess][AicpuRpcResInitV2]Entry AicpuRpcResInitV2 process-------");
    hccl::HcclCommAicpu* commAicpu = nullptr;
    HcclResult ret = HCCL_SUCCESS;
    std::string group = commParam->hcomId;
    CHK_RET(AcquireAicpuComm(group, &commAicpu));
    if (commAicpu == nullptr) {
        HCCL_ERROR("[AicpuHcclProcess][AicpuRpcResInitV2]commAicpu is null group[%s]", group.c_str());
        return 1U;
    }
    ret = commAicpu->Init(commParam, isCustom);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AicpuHcclProcess][AicpuRpcResInitV2]errNo[0x%016llx] Failed to init comm resource group[%s]",
            HCCL_ERROR_CODE(ret), group.c_str()),
        ret);
    HCCL_DEBUG("[AicpuHcclProcess][AicpuRpcResInitV2]AicpuRpcResInitV2 process end-------");
    if (g_devType == DevType::DEV_TYPE_COUNT) {
        CHK_RET(hrtHalGetDeviceType(commAicpu->GetDevId(), g_devType));
        HCCL_RUN_INFO(
            "[AicpuHcclProcess][AicpuRpcResInitV2] devId %u, get devType %u", commAicpu->GetDevId(),
            static_cast<u32>(g_devType));
    }
    HCCL_INFO(
        "[AicpuHcclProcess][AicpuRpcResInitV2] get PlatformVersion %u, %u", static_cast<u32>(g_devType),
        commAicpu->GetDevId());
    AicpuComContext* ctx = AicpuGetComContext();
    CallMC2MaintenanceThread(ctx);

    return 0;
}

HcclResult AicpuHcclProcess::AcquireAicpuComm(const std::string& group, HcclCommAicpu** aicpuCommPtr)
{
    // 通过 CollCommAicpuMgr 单例管理通信域，init 路径不标记使用中
    CollCommAicpu* collComm = nullptr;
    CHK_RET(CollCommAicpuMgr::GetInstance().AcquireAndCreateComm(group, &collComm));
    if (collComm == nullptr) {
        HCCL_ERROR("[%s]Failed to acquire comm group[%s]", __func__, group.c_str());
        return HCCL_E_PTR;
    }

    HcclCommAicpu* legacyComm = collComm->GetLegacy910CollComm();
    if (legacyComm == nullptr) {
        std::shared_ptr<HcclCommAicpu> newComm;
        try {
            newComm = std::make_shared<HcclCommAicpu>();
        } catch (std::exception& e) {
            HCCL_ERROR("[%s]Failed, exception caught:%s", __func__, e.what());
            return HCCL_E_PTR;
        }
        if (UNLIKELY(!newComm)) {
            HCCL_ERROR("[%s]errNo[0x%016llx] aicpuComm is nullptr", __func__, HCCL_ERROR_CODE(HCCL_E_PTR));
            return HCCL_E_PTR;
        }
        collComm->SetLegacy910CollComm(newComm);
        legacyComm = newComm.get();
        HCCL_INFO("[%s]Created new legacy comm group [%s]", __func__, group.c_str());
    } else {
        HCCL_INFO("[%s]Reuse existing legacy comm group [%s]", __func__, group.c_str());
    }

    *aicpuCommPtr = legacyComm;
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::AicpuIndOpCommInit(CommAicpuParam* commAicpuParam)
{
    hccl::HcclCommAicpu* commAicpu = nullptr;
    HcclResult ret = HCCL_SUCCESS;
    std::string group = commAicpuParam->hcomId;
    CHK_RET(AcquireAicpuComm(group, &commAicpu));
    if (commAicpu == nullptr) {
        HCCL_ERROR("[AicpuHcclProcess][AicpuIndOpCommInit]commAicpu is null group[%s]", group.c_str());
        return HCCL_E_PTR;
    }
    ret = commAicpu->InitAicpuIndOp(commAicpuParam);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AicpuHcclProcess][AicpuIndOpCommInit]errNo[0x%016llx] Failed to init independent op comm group[%s]",
            HCCL_ERROR_CODE(ret), group.c_str()),
        ret);
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::AicpuRegOpInfo(void* opInfo, u32 size)
{
    CollCommAicpu* collComm = CollCommAicpuMgr::GetInstance().GetCurrentComm();
    CHK_PTR_NULL(collComm);
    HcclCommAicpu* legacyComm = collComm->GetLegacy910CollComm();
    CHK_PTR_NULL(legacyComm);
    CHK_RET(legacyComm->RegisterOpInfo(opInfo, size));
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::AicpuRegOpTaskException(HcommGetOpInfoCallback callback)
{
    CollCommAicpu* collComm = CollCommAicpuMgr::GetInstance().GetCurrentComm();
    CHK_PTR_NULL(collComm);
    HcclCommAicpu* legacyComm = collComm->GetLegacy910CollComm();
    CHK_PTR_NULL(legacyComm);
    CHK_RET(legacyComm->RegOpTaskException(callback));
    return HCCL_SUCCESS;
}

std::shared_mutex& AicpuHcclProcess::AicpuGetCommMutex() { return CollCommAicpuMgr::GetInstance().GetMutex(); }

hccl::HcclCommAicpu* AicpuHcclProcess::AicpuGetCommbyGroup(const std::string& group)
{
    CollCommAicpu* collComm = CollCommAicpuMgr::GetInstance().AcquireCommForUse(group);
    if (collComm == nullptr) {
        HCCL_ERROR("[AicpuHcclProcess][%s] group[%s] not found", __func__, group.c_str());
        return nullptr;
    }
    collComm->SetLegacy910CollCommBusy(true);
    return collComm->GetLegacy910CollComm();
}

bool AicpuHcclProcess::GetCommExecStatus(const std::string& group)
{
    CollCommAicpu* collComm = CollCommAicpuMgr::GetInstance().FindCommByGroup(group);
    if (collComm != nullptr) {
        return collComm->IsLegacy910CollCommBusy();
    }
    return false;
}

void AicpuHcclProcess::AicpuReleaseCommbyGroup(const std::string& group)
{
    CollCommAicpu* collComm = CollCommAicpuMgr::GetInstance().FindCommByGroup(group);
    if (collComm != nullptr) {
        collComm->SetLegacy910CollCommBusy(false);
    }
    CollCommAicpuMgr::GetInstance().ReleaseComm(group);
}

u32 AicpuHcclProcess::AicpuRpcClearOpRes(const struct HcclKfcClearOpResTilingData* tilingData)
{
    if (tilingData == nullptr) {
        HCCL_ERROR("[AicpuRpcClearOpRes] tilingData is null");
        return HCCL_E_PARA;
    }
    if (tilingData->magic != HCCL_KFC_CLEAR_OP_RES_MAGIC) {
        HCCL_ERROR(
            "[AicpuRpcClearOpRes] magic mismatch: expect[0x%x] actual[0x%x], reject as misroute",
            HCCL_KFC_CLEAR_OP_RES_MAGIC, tilingData->magic);
        return HCCL_E_PARA;
    }
    if (tilingData->tagCount == 0 || tilingData->tagCount > HCCL_KFC_CLEAR_OP_RES_MAX_BATCH) {
        HCCL_ERROR(
            "[AicpuRpcClearOpRes] invalid tagCount[%u], must be in [1, %u]", tilingData->tagCount,
            HCCL_KFC_CLEAR_OP_RES_MAX_BATCH);
        return HCCL_E_PARA;
    }

    // group 来自 host 端共享 HBM；按 C 字符串语义截断，确保不越界
    char group[HCOMID_MAX_LENGTH + 1] = {0};
    if (memcpy_s(group, sizeof(group) - 1, tilingData->group, HCOMID_MAX_LENGTH) != EOK) {
        HCCL_ERROR("[AicpuRpcClearOpRes] memcpy_s group failed");
        return HCCL_E_MEMORY;
    }
    const std::string groupStr(group);

    // GetCommbyGroup 独占占用槽位（与 ExecOp 互斥），一次覆盖整批 tag 避免循环内反复抢占；超时 10ms 返回 nullptr
    hccl::HcclCommAicpu* commAicpu = AicpuHcclProcess::AicpuGetCommbyGroup(groupStr);
    if (commAicpu == nullptr) {
        HCCL_WARNING(
            "[AicpuRpcClearOpRes] group[%s] not found or busy, skip batch; tagCount[%u]", groupStr.c_str(),
            tilingData->tagCount);
        return HCCL_SUCCESS;
    }

    HcclResult lastErr = HCCL_SUCCESS;
    for (u32 i = 0; i < tilingData->tagCount; ++i) {
        char tag[TAG_MAX_LENGTH + 1] = {0};
        if (memcpy_s(tag, sizeof(tag) - 1, tilingData->tags[i], TAG_MAX_LENGTH) != EOK) {
            HCCL_ERROR("[AicpuRpcClearOpRes] memcpy_s tag failed, group[%s] idx[%u]", groupStr.c_str(), i);
            lastErr = HCCL_E_MEMORY;
            continue;
        }
        const std::string tagStr(tag);
        HcclResult ret = commAicpu->ClearOpResource(tagStr);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[AicpuRpcClearOpRes] ClearOpResource fail, group[%s] tag[%s] ret[%d]", groupStr.c_str(),
                tagStr.c_str(), ret);
            lastErr = ret;
            // 单 tag 失败不影响后续 tag, 尽量清干净
        }
    }
    AicpuHcclProcess::AicpuReleaseCommbyGroup(groupStr);
    HCCL_INFO(
        "[AicpuRpcClearOpRes] group[%s] processed batch tagCount[%u] lastErr[%d]", groupStr.c_str(),
        tilingData->tagCount, lastErr);
    return static_cast<u32>(lastErr);
}

HcclResult AicpuHcclProcess::AicpuGetCommAll(std::vector<std::pair<std::string, HcclCommAicpu*>>& aicpuCommInfo)
{
    std::vector<std::pair<std::string, CollCommAicpu*>> commInfo;
    CollCommAicpuMgr::GetInstance().GetAllComms(commInfo);
    for (auto& kv : commInfo) {
        HcclCommAicpu* legacy = kv.second->GetLegacy910CollComm();
        if (legacy != nullptr) {
            aicpuCommInfo.push_back({kv.first, legacy});
        }
    }
    return HCCL_SUCCESS;
}

void AicpuHcclProcess::AicpuDestoryCommbyGroup(const std::string& group)
{
    CollCommAicpuMgr::GetInstance().DestroyComm(group);
}

HcclResult AicpuHcclProcess::HandleOneSideService(const OpTilingData* tilingData)
{
    return HcclOneSideServiceAicpu::Process(tilingData);
}

HcclResult AicpuHcclProcess::AicpuRunRpcServerV2(
    hccl::HcclCommAicpu* hcclCommAicpu, OpTilingData* tilingData, HcclOpResParam* commParam)
{
    std::string algName = tilingData->algName;
    std::string tag = reinterpret_cast<char*>(tilingData->tag);
    std::string newTag = reinterpret_cast<char*>(tilingData->newTag);

    HCCL_DEBUG(
        "[AicpuHcclProcess][AicpuRunRpcServerV2]Entry AicpuRunRpcServerV2, group[%s], tag[%s], newTag[%s]",
        hcclCommAicpu->GetGroupName().c_str(), tag.c_str(), newTag.c_str());

    HCCL_DEBUG(
        "[AicpuHcclProcess][AicpuRunRpcServerV2]Entry AicpuRunRpcServerV2, algName[%s], algtype[%llu],"
        "floatOverflowMode[%u], dumpDebug[%u], debugMode[%u], inputPtr[%p], outputPtr[%p].",
        algName.c_str(), tilingData->algType, tilingData->floatOverflowMode, tilingData->dumpDebug,
        tilingData->debugMode, tilingData->inputPtr, tilingData->outputPtr);

    HCCL_DEBUG(
        "[AicpuHcclProcess][AicpuRunRpcServerV2]Entry AicpuRunRpcServerV2, reduceType[%u], syncMode[%u],"
        "root[%u], dstRank[%u], srcRank[%u], opType[%u], index[%u], length[%llu].",
        tilingData->reduceType, tilingData->syncMode, tilingData->root, tilingData->dstRank, tilingData->srcRank,
        tilingData->opType, tilingData->index, tilingData->length);

    HCCL_DEBUG(
        "[AicpuHcclProcess][AicpuRunRpcServerV2]Entry AicpuRunRpcServerV2, aicpuCacheEnable[%u]",
        tilingData->aicpuCacheEnable);

    CHK_RET(hcclCommAicpu->RecordHostOrder(commParam, tag, tilingData->orderLaunchMode));

    hccl::OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = reinterpret_cast<void*>(tilingData->inputPtr);
    opParam.outputPtr = reinterpret_cast<void*>(tilingData->outputPtr);
    opParam.reduceType = static_cast<HcclReduceOp>(tilingData->reduceType);
    opParam.stream = hcclCommAicpu->GetMainStream();
    opParam.syncMode = static_cast<SyncMode>(tilingData->syncMode);
    opParam.inputSymWindow = reinterpret_cast<void*>(tilingData->inputSymWindow);
    opParam.inputOffset = tilingData->inputOffset;
    opParam.outputSymWindow = reinterpret_cast<void*>(tilingData->outputSymWindow);
    opParam.outputOffset = tilingData->outputOffset;

    hcclCommAicpu->UpdateNotifyWaitTimeOut(opParam.syncMode, commParam->config.notifyWaitTime);

    opParam.opBaseAtraceInfo = nullptr;
    opParam.root = tilingData->root;
    opParam.dstRank = tilingData->dstRank;
    opParam.srcRank = tilingData->srcRank;
    opParam.opType = static_cast<HcclCMDType>(tilingData->opType);
    opParam.isZeroCopy = tilingData->isZeroCopy;
    opParam.supportSymmetricMemory = tilingData->isSymmetricMemory;
    opParam.index = tilingData->index;
    opParam.isCapture = tilingData->isCapture;
    opParam.needIncreLink = tilingData->needIncreLink;
    opParam.aicpuCacheEnable = tilingData->aicpuCacheEnable;
    opParam.aicpuUnfoldMode = tilingData->aicpuUnfoldMode;
    hcclCommAicpu->PrepareOpRetryHandler(
        tilingData->inplaceSupportRetry, tilingData->retryEnable, tilingData->inPlaceSupportRetryStatus,
        tilingData->isInplacePreSync, tilingData->isPostSync);
    u8* dynamicDataPtr = reinterpret_cast<u8*>(tilingData) + sizeof(struct OpTilingData);
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (opParam.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        struct OpTilingBatchSendRecvDataDes* batchSendRecvDataPtr
            = reinterpret_cast<struct OpTilingBatchSendRecvDataDes*>(dynamicDataPtr);
        opParam.BatchSendRecvDataDes.itemNum = batchSendRecvDataPtr->itemNum;
        opParam.BatchSendRecvDataDes.sendRecvItemsPtr = batchSendRecvDataPtr->batchSendRecvItem;
        opParam.BatchSendRecvDataDes.isDirectRemoteRank
            = reinterpret_cast<u8*>(batchSendRecvDataPtr->batchSendRecvItem + batchSendRecvDataPtr->itemNum);
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U, "tag:%s", opParam.tag.c_str());
        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", opParam.tag.c_str()));
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        struct OpTilingAllToAllDataDes* a2ADataPtr = reinterpret_cast<struct OpTilingAllToAllDataDes*>(dynamicDataPtr);
        opParam.All2AllDataDes.sendType = static_cast<HcclDataType>(a2ADataPtr->sendType);
        opParam.All2AllDataDes.recvType = static_cast<HcclDataType>(a2ADataPtr->recvType);
        opParam.All2AllDataDes.sendCount = a2ADataPtr->sendCount;
        HCCL_DEBUG(
            "[AicpuHcclProcess][AicpuRunRpcServerV2] AllToAll aicpu, sendCounts[%llu] rankSize[%u]",
            opParam.All2AllDataDes.sendCount, commParam->rankSize);
        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U, "tag:%s,ct:%llu,dt:%u", opParam.tag.c_str(),
            opParam.All2AllDataDes.sendCount, opParam.All2AllDataDes.sendType);
        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", opParam.tag.c_str()));
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        struct OpTilingAlltoallvDataDes* alltoallvDataPtr
            = reinterpret_cast<struct OpTilingAlltoallvDataDes*>(dynamicDataPtr);
        opParam.All2AllDataDes.sendType = static_cast<HcclDataType>(alltoallvDataPtr->sendType);
        opParam.All2AllDataDes.recvType = static_cast<HcclDataType>(alltoallvDataPtr->recvType);
        u64 rankSize = commParam->rankSize;
        opParam.All2AllDataDes.sendCounts = static_cast<void*>(alltoallvDataPtr->sendRecvInfos);
        opParam.All2AllDataDes.recvCounts
            = static_cast<void*>(static_cast<u64*>(alltoallvDataPtr->sendRecvInfos) + rankSize);
        opParam.All2AllDataDes.sdispls = static_cast<void*>(
            static_cast<u64*>(alltoallvDataPtr->sendRecvInfos) + ALLTOALLV_INFO_INDEX_2 * rankSize);
        opParam.All2AllDataDes.rdispls = static_cast<void*>(
            static_cast<u64*>(alltoallvDataPtr->sendRecvInfos) + ALLTOALLV_INFO_INDEX_3 * rankSize);
        if (algName == "RunAlltoAllVTwoLevelPipeline") {
            hcclCommAicpu->SetSendRecvInfoPtr(static_cast<void*>(
                static_cast<u64*>(alltoallvDataPtr->sendRecvInfos) + ALLTOALLV_INFO_INDEX_4 * rankSize));
        }
        HCCL_DEBUG(
            "[AicpuHcclProcess][AicpuRunRpcServerV2] sendCountsPtr[%p], recvCountsPtr[%p], sdisplsPtr[%p], "
            "rdisplsPtr[%p].",
            opParam.All2AllDataDes.sendCounts, opParam.All2AllDataDes.recvCounts, opParam.All2AllDataDes.sdispls,
            opParam.All2AllDataDes.rdispls);
        for (u32 i = 0; i < rankSize; i++) {
            HCCL_DEBUG(
                "[AicpuHcclProcess][AicpuRunRpcServerV2] sendCounts[%llu], recvCounts[%llu].",
                *(static_cast<const u64*>(opParam.All2AllDataDes.sendCounts) + i),
                *(static_cast<const u64*>(opParam.All2AllDataDes.recvCounts) + i));
        }
        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U, "tag:%s,ct:%llu,dt:%u", opParam.tag.c_str(),
            opParam.All2AllDataDes.sendCounts, opParam.All2AllDataDes.sendType);
        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", opParam.tag.c_str()));
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        struct OpTilingAlltoallvcDataDes* alltoallvcDataPtr
            = reinterpret_cast<struct OpTilingAlltoallvcDataDes*>(dynamicDataPtr);
        opParam.All2AllDataDes.sendType = static_cast<HcclDataType>(alltoallvcDataPtr->sendType);
        opParam.All2AllDataDes.recvType = static_cast<HcclDataType>(alltoallvcDataPtr->recvType);
        opParam.All2AllDataDes.sendCountMatrix = static_cast<void*>(alltoallvcDataPtr->sendCountMatrix);
        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U, "tag:%s,ct:%llu,dt:%u", opParam.tag.c_str(),
            opParam.All2AllDataDes.sendCountMatrix, opParam.All2AllDataDes.sendType);
        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", opParam.tag.c_str()));
    } else if (
        opParam.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V
        || opParam.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        const u64 vStructSize = sizeof(struct OpTilingVDataDes);
        CHK_PRT_RET(
            tilingData->length < vStructSize,
            HCCL_ERROR(
                "[AicpuHcclProcess][AicpuRunRpcServerV2] TilingData "
                "corrupt, length[%llu], expect[%llu]",
                tilingData->length, vStructSize),
            HCCL_E_PARA);
        const u32 rankSize = commParam->rankSize;
        struct OpTilingVDataDes* vDataPtr = reinterpret_cast<struct OpTilingVDataDes*>(dynamicDataPtr);
        const u64 vDataLen = CalcOpTilingVDataDesVDataLen(rankSize);
        CHK_PRT_RET(
            vDataPtr->vDataLen != vDataLen,
            HCCL_ERROR(
                "[AicpuHcclProcess][AicpuRunRpcServerV2] TilingVDataDes "
                "corrupt, length[%llu], expect[%llu], rankSize[%u]",
                vDataPtr->vDataLen, vDataLen, rankSize),
            HCCL_E_PARA);
        opParam.VDataDes.dataType = static_cast<HcclDataType>(vDataPtr->dataType);
        opParam.VDataDes.counts = static_cast<void*>(static_cast<u64*>(vDataPtr->vData));
        opParam.VDataDes.displs = static_cast<void*>(static_cast<u64*>(vDataPtr->vData) + rankSize);
        CHK_RET(CalcDataSizeV(opParam, rankSize));
        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U, "tag:%s,ct:%llu,ds:%llu,dt:%u", opParam.tag.c_str(),
            opParam.VDataDes.counts, opParam.VDataDes.displs, opParam.VDataDes.dataType);
        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", opParam.tag.c_str()));
    } else {
        struct OpTilingDataDes* opDataDesPtr = reinterpret_cast<struct OpTilingDataDes*>(dynamicDataPtr);
        opParam.DataDes.count = opDataDesPtr->count;
        opParam.DataDes.dataType = static_cast<HcclDataType>(opDataDesPtr->dataType);
        CHK_RET(CalcDataSize(
            opParam.opType, static_cast<HcclDataType>(opDataDesPtr->dataType), opDataDesPtr->count,
            hcclCommAicpu->GetRankSize(), opParam.inputSize, opParam.outputSize));
        HCCL_DEBUG(
            "[AicpuHcclProcess][AicpuRunRpcServerV2] Entry AicpuRunRpcServerV2, "
            "count[%llu], dataType[%u] inputSize[%lu] outputSize[%lu].",
            opDataDesPtr->count, opDataDesPtr->dataType, opParam.inputSize, opParam.outputSize);
        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U, "tag:%s,ct:%llu,dt:%u", opParam.tag.c_str(),
            opParam.DataDes.count, opParam.DataDes.dataType);
        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", opParam.tag.c_str()));
    }

    CHK_RET(hrtSetLocalDeviceSatMode(static_cast<aclrtFloatOverflowMode>(tilingData->floatOverflowMode)));
    hcclCommAicpu->SetDumpDebug(tilingData->dumpDebug);
    hcclCommAicpu->SetAlgType(tilingData->algType);
    hcclCommAicpu->SetDebugMode(tilingData->debugMode);
    hcclCommAicpu->SetIsDeviceMode(false);
    hcclCommAicpu->SetUserStreamId(tilingData->userStreamId);
    CHK_RET(hcclCommAicpu->ParseHierarchicalAlgOption(tilingData->ahcConfInfo));
    /* 接口交互信息日志 */
    std::string logInfo = std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclCommAicpu->SaveTraceInfo(logInfo), opParam.tag.c_str());

    HcclUs startut = TIME_NOW();
    HcclResult ret = hcclCommAicpu->ExecOp(newTag, algName, opParam, commParam);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[AicpuHcclProcess][AicpuRunRpcServerV2] newTag[%s] algName[%s]", newTag.c_str(), algName.c_str()),
        ret);
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo
        = "AicpuRunRpcServerV2:success,take time: " + std::to_string(DURATION_US(endut - startut).count()) + " us";
    CHK_RET_AND_PRINT_IDE(hcclCommAicpu->SaveTraceInfo(endInfo), opParam.tag.c_str());

    HCCL_INFO("[AicpuHcclProcess][AicpuRunRpcServerV2]AicpuRunRpcServerV2 process end-------");
    return HCCL_SUCCESS;
}

void AicpuHcclProcess::CopyCtxInfo(AicpuComContext* ctx)
{
    auto otherClusterId = CLUSTER_CNT - ctx->clusterId - 1;
    auto otherCluster = &g_comContext[otherClusterId];
    *otherCluster = *ctx;
    otherCluster->clusterId = otherClusterId;
    HCCL_DEBUG("curClusterId = %d, otherClusterId = %d, copy finished", ctx->clusterId, otherCluster->clusterId);
}

void AicpuHcclProcess::CopyCtxForBackGroundDfx(const AicpuComContext* ctx)
{
    auto otherClusterId = CLUSTER_CNT - ctx->clusterId - 1;
    AicpuComContext* otherCluster = &g_comContext[otherClusterId];
    otherCluster->workSpaceAddr = ctx->workSpaceAddr;
    otherCluster->notifyOff = ctx->notifyOff;
    otherCluster->notifyBeginCnt = ctx->notifyBeginCnt;
    otherCluster->totalTurnCnt = ctx->totalTurnCnt;
    AicpuSqeContext::SaveVariable();
}

HcclResult AicpuHcclProcess::WaitAsyncFlag(
    hccl::Transport::Buffer* localFlagBufforCheck, const uint32_t flagValue, uint64_t timeOut)
{
    // 轮询等待flag
    if (flagValue < FLAG_OFFSET) {
        HCCL_ERROR("[AicpuHcclProcess][WaitAsyncFlag] flagValue must > 0,now flagValue is [%u]", flagValue);
        return HCCL_E_PARA;
    }
    uint32_t index = flagValue - FLAG_OFFSET;
    bool isTimeout = true;
    u64 startTime = GetCurCpuTimestamp();
    CHK_PTR_NULL(localFlagBufforCheck);
    CHK_PTR_NULL(localFlagBufforCheck[index].addr);
    uint32_t* waitPtr = const_cast<uint32_t*>(static_cast<const uint32_t*>(localFlagBufforCheck[index].addr));
    while ((GetCurCpuTimestamp() - startTime) < static_cast<unsigned long long>(NSEC_PER_SEC * timeOut)) {
        if (*waitPtr == flagValue) {
            isTimeout = false;
            *waitPtr = 0;
            break; // 当前位置的值等于flag时，说明对端写过来了，可以退出
        }
    }
    CHK_PRT_RET(
        isTimeout,
        HCCL_ERROR(
            "[AicpuHcclProcess][WaitAsyncFlag]Kernel Run TimeOut %llus, now Operation is CheckFlag, "
            "localFlagBufforCheck0.addr is [%p], localFlagBufforCheck0.value is [%u],"
            "localFlagBufforCheck1.addr is [%p], localFlagBufforCheck1.value is [%u],"
            "localFlagBufforCheck2.addr is [%p], localFlagBufforCheck2.value is [%u]",
            timeOut, localFlagBufforCheck[0].addr, *(const uint32_t*)(localFlagBufforCheck[0].addr),
            localFlagBufforCheck[1].addr, *(const uint32_t*)(localFlagBufforCheck[1].addr),
            localFlagBufforCheck[2].addr, *(const uint32_t*)(localFlagBufforCheck[2].addr)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::InitAsyncFlag(
    const uint32_t* lFlagAddr, const uint32_t* rFlagAddr, hccl::Transport::Buffer* localFlagBufforCheck,
    hccl::Transport::Buffer* localFlagBufforWrite, hccl::Transport::Buffer* remoteFlagBuf)
{
    CHK_PTR_NULL(lFlagAddr);
    CHK_PTR_NULL(rFlagAddr);
    CHK_PTR_NULL(remoteFlagBuf);
    for (uint32_t i = 0; i < POST_SEND_FLAG_COUNT; i++) {
        localFlagBufforCheck[i].addr = lFlagAddr + i * FLAG_INTERVAL; // 接收对端给本端写的flag
        localFlagBufforCheck[i].size = sizeof(uint32_t);
        localFlagBufforWrite[i].addr = lFlagAddr + i * FLAG_INTERVAL + FLAG_OFFSET; // 存放本端给对端写的flag
        localFlagBufforWrite[i].size = sizeof(uint32_t);
        remoteFlagBuf[i].addr = rFlagAddr + i * FLAG_INTERVAL;
        remoteFlagBuf[i].size = sizeof(uint32_t);
        uint32_t* waitPtr = const_cast<uint32_t*>(static_cast<const uint32_t*>(localFlagBufforWrite[i].addr));
        *waitPtr = i + FLAG_OFFSET; // 置flag
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::AicpuIndOpThreadInit(ThreadMgrAicpuParam* param)
{
    std::string group = param->hcomId;
    hccl::HcclCommAicpu* hcclCommAicpu = AicpuHcclProcess::AicpuGetCommbyGroup(group);
    CHK_PRT_RET(!hcclCommAicpu, HCCL_ERROR("%s hcclCommAicpu is null, group[%s]", __func__, group.c_str()), HCCL_E_PTR);
    HcclResult ret = hcclCommAicpu->InitThreads(param);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AicpuHcclProcess][AicpuIndOpThreadInit]errNo[0x%016llx] Failed to init threads group[%s]",
            HCCL_ERROR_CODE(ret), group.c_str()),
        ret);
    AicpuReleaseCommbyGroup(group);
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::AicpuIndOpChannelInit(HcclIndOpChannelRemoteResV3* commParam)
{
    std::string group = commParam->hcomId;
    hccl::HcclCommAicpu* hcclCommAicpu = AicpuHcclProcess::AicpuGetCommbyGroup(group);
    CHK_PRT_RET(!hcclCommAicpu, HCCL_ERROR("%s hcclCommAicpu is null, group[%s]", __func__, group.c_str()), HCCL_E_PTR);
    HcclResult ret = hcclCommAicpu->AllocChannelResource(commParam);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AicpuHcclProcess][AicpuIndOpChannelInit]errNo[0x%016llx] Failed to init channels group[%s]",
            HCCL_ERROR_CODE(ret), group.c_str()),
        ret);
    AicpuReleaseCommbyGroup(group);
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::AicpuIndOpNotifyInit(NotifyMgrAicpuParam* param)
{
    CHK_PTR_NULL(param);
    std::string group = param->hcomId;
    hccl::HcclCommAicpu* hcclCommAicpu = AicpuHcclProcess::AicpuGetCommbyGroup(group);
    CHK_PRT_RET(!hcclCommAicpu, HCCL_ERROR("%s hcclCommAicpu is null, group[%s]", __func__, group.c_str()), HCCL_E_PTR);

    HcclResult ret = HCCL_E_INTERNAL;
    if (param->freeFlag) {
        ret = hcclCommAicpu->NotifyFree(param);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[AicpuHcclProcess][%s]errNo[0x%016llx] Failed to free notify group[%s]", __func__,
                HCCL_ERROR_CODE(ret), group.c_str()),
            ret);
    } else {
        ret = hcclCommAicpu->NotifyAlloc(param);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[AicpuHcclProcess][%s]errNo[0x%016llx] Failed to alloc notify group[%s]", __func__,
                HCCL_ERROR_CODE(ret), group.c_str()),
            ret);
    }

    HCCL_INFO(
        "[AicpuHcclProcess][%s] comm identifier[%s], notify op[%u] success, num[%u]", __func__, group.c_str(),
        param->freeFlag, param->notifyNum);
    AicpuReleaseCommbyGroup(group);
    return HCCL_SUCCESS;
}
