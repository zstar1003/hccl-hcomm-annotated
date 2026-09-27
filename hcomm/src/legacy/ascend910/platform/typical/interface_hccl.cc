/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * 该特性代码不涉及开源
 */

#include "interface_hccl.h"
#include "typical_param_check.h"
#include "hccl_common.h"
#include "externalinput.h"
#include "rdma_resource_manager.h"
#include "typical_mr_manager.h"
#include "typical_sync_mem.h"
#include "typical_window_mem.h"
#include "typical_qp_manager.h"
#include "send_recv_executor.h"
#include "adapter_rts.h"

using namespace hccl;
constexpr u32 DEVISOR_VALUE_FOUR = 4;
constexpr u32 MAX_WQE_PER_DOORBELL = 300;
constexpr u32 QP_QUEUE_DEPTH_MAX = 32768;
constexpr u32 QP_QUEUE_DEPTH_MIN = 64;
#define HCCN_RESV_MEM_TYPE_PDCCL (0)

struct MrInfoT AscendMrInfo2MrInfo(AscendMrInfo* ascendMrInfo)
{
    struct MrInfoT innerMrInfo = {};
    innerMrInfo.addr = reinterpret_cast<void*>(ascendMrInfo->addr);
    innerMrInfo.size = ascendMrInfo->size;
    innerMrInfo.lkey = ascendMrInfo->key;
    return innerMrInfo;
}

HcclResult hcclCreateAscendQP(AscendQPInfo* ascendQPInfo)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PTR_NULL(ascendQPInfo);
    struct TypicalQp qpInfo = {};
    CHK_RET(TypicalQpManager::GetInstance().CreateQp(qpInfo));
    ascendQPInfo->qpn = qpInfo.qpn;
    ascendQPInfo->gidIdx = qpInfo.gidIdx;
    for (uint32_t i = 0; i < GID_LENGTH; i++) {
        ascendQPInfo->gid[i] = qpInfo.gid[i];
    }
    ascendQPInfo->psn = qpInfo.psn;
    HCCL_INFO(
        "hcclCreateAscendQP success! qpn %u, gid index %u, psn %u", ascendQPInfo->qpn, ascendQPInfo->gidIdx,
        ascendQPInfo->psn);

    return HCCL_SUCCESS;
}

HcclResult CheckDepth(uint32_t depth)
{
    if (depth >= QP_QUEUE_DEPTH_MIN && ((depth & (depth - 1)) == 0) && depth <= QP_QUEUE_DEPTH_MAX) {
        return HCCL_SUCCESS;
    }
    HCCL_ERROR(
        "[CheckDepth]depth[%u] is invalid, depth should be power of 2 and in [%u, %u]", depth, QP_QUEUE_DEPTH_MIN,
        QP_QUEUE_DEPTH_MAX);
    return HCCL_E_PARA;
}

HcclResult hcclCreateAscendQPWithAttr(AscendQPInfo* ascendQPInfo)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PTR_NULL(ascendQPInfo);
    CHK_RET(CheckDepth(ascendQPInfo->sq_depth));
    CHK_RET(CheckDepth(ascendQPInfo->scq_depth));
    CHK_RET(CheckDepth(ascendQPInfo->rq_depth));
    CHK_RET(CheckDepth(ascendQPInfo->rcq_depth));
    struct TypicalQp qpInfo;
    QpConfigInfo qpConfigInfo{
        ascendQPInfo->sq_depth, ascendQPInfo->rq_depth, ascendQPInfo->scq_depth, ascendQPInfo->rcq_depth, 0, 0};
    u32 poolId;
    if (HCCL_SUCCESS == RdmaResourceManager::GetInstance().GetResvMemPoolIdByType(HCCN_RESV_MEM_TYPE_PDCCL, poolId)) {
        qpConfigInfo.use_resv_mem = 1;
        qpConfigInfo.resv_mem_pool_id = poolId;
    }
    CHK_RET(TypicalQpManager::GetInstance().CreateQp(qpInfo, qpConfigInfo));
    ascendQPInfo->qpn = qpInfo.qpn;
    ascendQPInfo->gidIdx = qpInfo.gidIdx;
    for (uint32_t i = 0; i < GID_LENGTH; i++) {
        ascendQPInfo->gid[i] = qpInfo.gid[i];
    }
    ascendQPInfo->psn = qpInfo.psn;
    HCCL_INFO(
        "hcclCreateAscendQP success! qpn[%u], gid index[%u], psn[%u], sq_depth[%u], rq_depth[%u], scq_depth[%u], "
        "rcq_depth[%u] ",
        ascendQPInfo->qpn, ascendQPInfo->gidIdx, ascendQPInfo->psn, ascendQPInfo->sq_depth, ascendQPInfo->rq_depth,
        ascendQPInfo->scq_depth, ascendQPInfo->rcq_depth);

    return HCCL_SUCCESS;
}

HcclResult hcclCreateAscendCQWithAttr(AscendCQInfo* ascendCQInfo)
{
    CHK_PTR_NULL(ascendCQInfo);
    if (ascendCQInfo->cqDepth == 0) {
        ascendCQInfo->cqDepth = HOST_SQ_CQ_DEPTH;
    }
    CHK_RET(CheckDepth(ascendCQInfo->cqDepth));

    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_RET(TypicalQpManager::GetInstance().CreateCq(*ascendCQInfo));
    HCCL_INFO("hcclCreateAscendCQWithAttr success! cqn[%u], cqDepth[%u]", ascendCQInfo->cqn, ascendCQInfo->cqDepth);
    return HCCL_SUCCESS;
}

HcclResult hcclDestroyAscendCQ(AscendCQInfo* ascendCQInfo)
{
    CHK_PTR_NULL(ascendCQInfo);
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_RET(TypicalQpManager::GetInstance().DestroyCq(ascendCQInfo->cqn));
    HCCL_INFO("hcclDestroyAscendCQ success! cqn[%u]", ascendCQInfo->cqn);
    return HCCL_SUCCESS;
}

HcclResult hcclPollAscendCQ(AscendCQInfo* cqInfo, uint32_t num, uint32_t* polledNum, struct AscendWc* wc)
{
    CHK_PTR_NULL(cqInfo);
    CHK_PTR_NULL(polledNum);
    CHK_PTR_NULL(wc);

    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    void* cqHandle = nullptr;
    CHK_RET(TypicalQpManager::GetInstance().GetCqHandle(cqInfo->cqn, cqHandle));

    s32 ret = HrtRaPollTypicalCq(cqHandle, num, (void*)wc);
    if (ret < 0) {
        HCCL_ERROR("[hcclPollAscendCQ] Poll typical cq failed, cqn[%u], ret[%d]", cqInfo->cqn, ret);
        return HCCL_E_INTERNAL;
    }
    *polledNum = (uint32_t)ret;
    return HCCL_SUCCESS;
}

HcclResult hcclCreateAscendQPWithCQWithAttr(
    AscendCQInfo* ascendSendCQInfo, AscendCQInfo* ascendRecvCQInfo, AscendVerbsQPInfo* ascendQPInfo)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PTR_NULL(ascendSendCQInfo);
    CHK_PTR_NULL(ascendRecvCQInfo);
    CHK_PTR_NULL(ascendQPInfo);

    CHK_RET(TypicalQpManager::GetInstance().ValidateCq(ascendSendCQInfo->cqn));
    CHK_RET(TypicalQpManager::GetInstance().ValidateCq(ascendRecvCQInfo->cqn));

    struct TypicalQp qpInfo {};
    QpConfigWithCQInfo qpConfigInfo{};
    qpConfigInfo.sendCqn = ascendSendCQInfo->cqn;
    qpConfigInfo.recvCqn = ascendRecvCQInfo->cqn;

    if (ascendQPInfo->cap.maxSendWr == 0) {
        qpConfigInfo.sq_depth = DEFAULT_OPBASE_MAX_SEND_WR;
    } else {
        CHK_RET(CheckDepth(ascendQPInfo->cap.maxSendWr));
        qpConfigInfo.sq_depth = ascendQPInfo->cap.maxSendWr;
    }
    if (ascendQPInfo->cap.maxRecvWr == 0) {
        qpConfigInfo.rq_depth = DEFAULT_MAX_RECV_WR;
    } else {
        CHK_RET(CheckDepth(ascendQPInfo->cap.maxRecvWr));
        qpConfigInfo.rq_depth = ascendQPInfo->cap.maxRecvWr;
    }

    CHK_RET(TypicalQpManager::GetInstance().GetCqDepth(ascendSendCQInfo->cqn, qpConfigInfo.scq_depth));
    CHK_RET(TypicalQpManager::GetInstance().GetCqDepth(ascendRecvCQInfo->cqn, qpConfigInfo.rcq_depth));

    if (ascendQPInfo->qp_type == 0) {
        ascendQPInfo->qp_type = ASCEND_QPT_RC;
    }

    qpConfigInfo.sq_sig_all = ascendQPInfo->sqSigAll;

    if (ascendQPInfo->cap.maxSendSge == 0) {
        qpConfigInfo.max_send_sge = DEFAULT_MAX_SEND_SGE;
    } else {
        qpConfigInfo.max_send_sge = ascendQPInfo->cap.maxSendSge;
    }
    if (ascendQPInfo->cap.maxRecvSge == 0) {
        qpConfigInfo.max_recv_sge = DEFAULT_MAX_RECV_SGE;
    } else {
        qpConfigInfo.max_recv_sge = ascendQPInfo->cap.maxRecvSge;
    }
    if (ascendQPInfo->cap.maxInlineData == 0) {
        qpConfigInfo.max_inline_data = DEFAULT_MAX_INLINE_DATA;
    } else {
        qpConfigInfo.max_inline_data = ascendQPInfo->cap.maxInlineData;
    }

    u32 poolId;
    if (HCCL_SUCCESS == RdmaResourceManager::GetInstance().GetResvMemPoolIdByType(HCCN_RESV_MEM_TYPE_PDCCL, poolId)) {
        qpConfigInfo.use_resv_mem = 1;
        qpConfigInfo.resv_mem_pool_id = poolId;
    }
    CHK_RET(TypicalQpManager::GetInstance().CreateQpWithCQ(qpInfo, qpConfigInfo));
    ascendQPInfo->qpn = qpInfo.qpn;
    ascendQPInfo->gidIdx = qpInfo.gidIdx;
    for (uint32_t i = 0; i < GID_LENGTH; i++) {
        ascendQPInfo->gid[i] = qpInfo.gid[i];
    }
    ascendQPInfo->psn = qpInfo.psn;
    HCCL_INFO(
        "hcclCreateAscendQPWithCQWithAttr success! qpn[%u], gid index[%u], psn[%u], sendCqn[%u], recvCqn[%u]",
        ascendQPInfo->qpn, ascendQPInfo->gidIdx, ascendQPInfo->psn, ascendSendCQInfo->cqn, ascendRecvCQInfo->cqn);

    return HCCL_SUCCESS;
}

HcclResult hcclDestroyAscendVerbsQP(AscendVerbsQPInfo* ascendQPVerbsInfo)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PTR_NULL(ascendQPVerbsInfo);
    struct TypicalQp qpInfo;
    qpInfo.qpn = ascendQPVerbsInfo->qpn;
    qpInfo.gidIdx = ascendQPVerbsInfo->gidIdx;
    for (uint32_t i = 0; i < GID_LENGTH; i++) {
        qpInfo.gid[i] = ascendQPVerbsInfo->gid[i];
    }
    qpInfo.psn = ascendQPVerbsInfo->psn;
    CHK_RET(TypicalQpManager::GetInstance().DestroyQpWithoutCQ(qpInfo));
    HCCL_INFO("hcclDestroyAscendVerbsQP success! qpn[%u]", ascendQPVerbsInfo->qpn);
    return HCCL_SUCCESS;
}

HcclResult hcclModifyAscendQP(AscendQPInfo* localQPInfo, AscendQPInfo* remoteQPInfo)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    AscendQPQos qpQos;
    qpQos.sl = GetExternalInputRdmaServerLevel();
    qpQos.tc = GetExternalInputRdmaTrafficClass();
    CHK_RET(hcclModifyAscendQPEx(localQPInfo, remoteQPInfo, &qpQos));
    return HCCL_SUCCESS;
}

HcclResult hcclModifyAscendQPEx(AscendQPInfo* localQPInfo, AscendQPInfo* remoteQPInfo, AscendQPQos* qpQos)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PTR_NULL(localQPInfo);
    CHK_PTR_NULL(remoteQPInfo);
    CHK_PTR_NULL(qpQos);
    CHK_PRT_RET(
        (qpQos->sl > HCCL_RDMA_SL_MAX),
        HCCL_ERROR(
            "[hcclModifyAscendQPEx]The value of sl[%u] is invalid. except: [%u, %u].", qpQos->sl, HCCL_RDMA_SL_MIN,
            HCCL_RDMA_SL_MAX),
        HCCL_E_PARA);
    CHK_PRT_RET(
        (qpQos->tc > HCCL_RDMA_TC_MAX),
        HCCL_ERROR(
            "[hcclModifyAscendQPEx]The value of tc[%u] is invalid. except: [%u, %u].", qpQos->tc, HCCL_RDMA_TC_MIN,
            HCCL_RDMA_TC_MAX),
        HCCL_E_PARA);

    // 设置的RDMATrafficClass需要是4的整数倍, 否则报错
    CHK_PRT_RET(
        qpQos->tc % DEVISOR_VALUE_FOUR != 0,
        HCCL_ERROR("[hcclModifyAscendQPEx]The value of tc[%u] is not a multiple of 4.", qpQos->tc), HCCL_E_PARA);

    struct TypicalQp localQp;
    localQp.qpn = localQPInfo->qpn;
    localQp.gidIdx = localQPInfo->gidIdx;
    for (uint32_t i = 0; i < GID_LENGTH; i++) {
        localQp.gid[i] = localQPInfo->gid[i];
    }
    localQp.psn = localQPInfo->psn;
    localQp.sl = qpQos->sl;
    localQp.tc = qpQos->tc;

    struct TypicalQp remoteQp = {};
    remoteQp.qpn = remoteQPInfo->qpn;
    remoteQp.gidIdx = remoteQPInfo->gidIdx;
    for (uint32_t i = 0; i < GID_LENGTH; i++) {
        remoteQp.gid[i] = remoteQPInfo->gid[i];
    }
    remoteQp.psn = remoteQPInfo->psn;
    CHK_RET(TypicalQpManager::GetInstance().ModifyQp(localQp, remoteQp));
    return HCCL_SUCCESS;
}

HcclResult hcclModifyAscendVerbsQP(AscendVerbsQPInfo* localQPInfo, AscendVerbsQPInfo* remoteQPInfo)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    AscendQPQos qpQos;
    qpQos.sl = GetExternalInputRdmaServerLevel();
    qpQos.tc = GetExternalInputRdmaTrafficClass();
    CHK_RET(hcclModifyAscendVerbsQPEx(localQPInfo, remoteQPInfo, &qpQos));
    return HCCL_SUCCESS;
}

HcclResult
hcclModifyAscendVerbsQPEx(AscendVerbsQPInfo* localQPInfo, AscendVerbsQPInfo* remoteQPInfo, AscendQPQos* qpQos)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PTR_NULL(localQPInfo);
    CHK_PTR_NULL(remoteQPInfo);
    CHK_PTR_NULL(qpQos);
    CHK_PRT_RET(
        (qpQos->sl > HCCL_RDMA_SL_MAX),
        HCCL_ERROR(
            "[hcclModifyAscendVerbsQPEx]The value of sl[%u] is invalid. except: [%u, %u].", qpQos->sl, HCCL_RDMA_SL_MIN,
            HCCL_RDMA_SL_MAX),
        HCCL_E_PARA);
    CHK_PRT_RET(
        (qpQos->tc > HCCL_RDMA_TC_MAX),
        HCCL_ERROR(
            "[hcclModifyAscendVerbsQPEx]The value of tc[%u] is invalid. except: [%u, %u].", qpQos->tc, HCCL_RDMA_TC_MIN,
            HCCL_RDMA_TC_MAX),
        HCCL_E_PARA);

    CHK_PRT_RET(
        qpQos->tc % DEVISOR_VALUE_FOUR != 0,
        HCCL_ERROR("[hcclModifyAscendVerbsQPEx]The value of tc[%u] is not a multiple of 4.", qpQos->tc), HCCL_E_PARA);

    struct TypicalQp localQp;
    localQp.qpn = localQPInfo->qpn;
    localQp.gidIdx = localQPInfo->gidIdx;
    for (uint32_t i = 0; i < GID_LENGTH; i++) {
        localQp.gid[i] = localQPInfo->gid[i];
    }
    localQp.psn = localQPInfo->psn;
    localQp.sl = qpQos->sl;
    localQp.tc = qpQos->tc;

    struct TypicalQp remoteQp = {};
    remoteQp.qpn = remoteQPInfo->qpn;
    remoteQp.gidIdx = remoteQPInfo->gidIdx;
    for (uint32_t i = 0; i < GID_LENGTH; i++) {
        remoteQp.gid[i] = remoteQPInfo->gid[i];
    }
    remoteQp.psn = remoteQPInfo->psn;
    CHK_RET(TypicalQpManager::GetInstance().ModifyVerbsQp(localQp, remoteQp));
    return HCCL_SUCCESS;
}

HcclResult hcclDestroyAscendQP(AscendQPInfo* ascendQPInfo)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PTR_NULL(ascendQPInfo);
    struct TypicalQp qpInfo;
    qpInfo.qpn = ascendQPInfo->qpn;
    qpInfo.gidIdx = ascendQPInfo->gidIdx;
    for (uint32_t i = 0; i < GID_LENGTH; i++) {
        qpInfo.gid[i] = ascendQPInfo->gid[i];
    }
    qpInfo.psn = ascendQPInfo->psn;
    CHK_RET(TypicalQpManager::GetInstance().DestroyQp(qpInfo));
    return HCCL_SUCCESS;
}

HcclResult hcclAllocWindowMem(void** ptr, size_t len)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    return TypicalWindowMem::GetInstance().AllocWindowMem(ptr, len);
}

HcclResult hcclFreeWindowMem(void* ptr)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    return TypicalWindowMem::GetInstance().FreeWindowMem(ptr);
}

HcclResult hcclAllocSyncMem(int32_t** ptr)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    return TypicalSyncMem::GetInstance().AllocSyncMem(ptr);
}

HcclResult hcclFreeSyncMem(int32_t* ptr)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    return TypicalSyncMem::GetInstance().FreeSyncMem(ptr);
}

HcclResult hcclRegisterMem(AscendMrInfo* memInfo)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PTR_NULL(memInfo);
    MrInfoT mrInfo = {};
    mrInfo.addr = reinterpret_cast<void*>(static_cast<uintptr_t>(memInfo->addr));
    mrInfo.size = memInfo->size;
    mrInfo.access = RA_ACCESS_REMOTE_WRITE | RA_ACCESS_LOCAL_WRITE | RA_ACCESS_REMOTE_READ;
    CHK_RET(TypicalMrManager::GetInstance().RegisterMem(mrInfo));
    memInfo->key = mrInfo.lkey;
    HCCL_RUN_INFO(
        "[hcclRegisterMem] Register WindowMem addr[%p], size[%llu], key[%u].", mrInfo.addr, mrInfo.size, mrInfo.lkey);
    return HCCL_SUCCESS;
}

HcclResult hcclDeRegisterMem(AscendMrInfo* memInfo)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PTR_NULL(memInfo);
    MrInfoT mrInfo = {};
    mrInfo.addr = reinterpret_cast<void*>(static_cast<uintptr_t>(memInfo->addr));
    mrInfo.size = memInfo->size;
    mrInfo.access = RA_ACCESS_REMOTE_WRITE | RA_ACCESS_LOCAL_WRITE | RA_ACCESS_REMOTE_READ;
    mrInfo.lkey = memInfo->key;
    CHK_RET(TypicalMrManager::GetInstance().DeRegisterMem(mrInfo));
    HCCL_RUN_INFO(
        "[hcclDeRegisterMem] DeRegister WindowMem addr[%p], size[%llu], key[%u].", mrInfo.addr, mrInfo.size,
        mrInfo.lkey);
    return HCCL_SUCCESS;
}

HcclResult hcclRdmaMemRegister(AscendMrAttr* memInfo)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PTR_NULL(memInfo);
    MrInfoT mrInfo = {};
    mrInfo.addr = reinterpret_cast<void*>(static_cast<uintptr_t>(memInfo->addr));
    mrInfo.size = memInfo->size;
    mrInfo.access = RA_ACCESS_REMOTE_WRITE | RA_ACCESS_LOCAL_WRITE | RA_ACCESS_REMOTE_READ;
    CHK_RET(TypicalMrManager::GetInstance().RegisterMem(mrInfo));
    memInfo->lkey = mrInfo.lkey;
    memInfo->rkey = mrInfo.rkey;
    HCCL_RUN_INFO(
        "[hcclRdmaMemRegister] Register MR addr[%p], size[%llu], lkey[%u], rkey[%u].", mrInfo.addr, mrInfo.size,
        mrInfo.lkey, mrInfo.rkey);
    return HCCL_SUCCESS;
}

HcclResult hcclRdmaMemDeRegister(AscendMrAttr* memInfo)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PTR_NULL(memInfo);
    MrInfoT mrInfo = {};
    mrInfo.addr = reinterpret_cast<void*>(static_cast<uintptr_t>(memInfo->addr));
    mrInfo.size = memInfo->size;
    mrInfo.access = RA_ACCESS_REMOTE_WRITE | RA_ACCESS_LOCAL_WRITE | RA_ACCESS_REMOTE_READ;
    mrInfo.lkey = memInfo->lkey;
    CHK_RET(TypicalMrManager::GetInstance().DeRegisterMem(mrInfo));
    HCCL_RUN_INFO(
        "[hcclRdmaMemDeRegister] DeRegister MR addr[%p], size[%llu], lkey[%u].", mrInfo.addr, mrInfo.size, mrInfo.lkey);
    return HCCL_SUCCESS;
}

HcclResult hcclAscendPostSend(
    AscendVerbsQPInfo* qpInfo, struct AscendSendWr* sendWr, aclrtStream stream, struct AscendSendWr** badWr)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PTR_NULL(qpInfo);
    CHK_PTR_NULL(sendWr);
    CHK_PTR_NULL(badWr);
    CHK_PTR_NULL(stream);

    QpHandle qpHandle;
    CHK_RET(TypicalQpManager::GetInstance().GetVerbsQpHandleByQpn(qpInfo->qpn, qpHandle));

    constexpr u32 MAX_SGLIST_VERBS = 16;
    struct AscendSendWr* curWr = sendWr;
    while (curWr != nullptr) {
        CHK_PRT_RET(
            static_cast<u32>(curWr->numSge) > MAX_SGLIST_VERBS,
            HCCL_ERROR("[hcclAscendPostSend] numSge[%d] exceeds MAX_SGLIST_VERBS[%u]", curWr->numSge, MAX_SGLIST_VERBS),
            HCCL_E_PARA);
        CHK_PTR_NULL(curWr->sgList);

        struct SgList sgeList[MAX_SGLIST_VERBS];
        for (int i = 0; i < curWr->numSge; i++) {
            sgeList[i].addr = curWr->sgList[i].addr;
            sgeList[i].len = curWr->sgList[i].len;
            sgeList[i].lkey = curWr->sgList[i].lkey;
        }

        struct SendWrVerbs wr = {};
        struct SendWrRsp opRsp = {};
        wr.wrId = curWr->wrId;
        wr.sgList = sgeList;
        wr.numSge = curWr->numSge;
        wr.remoteAddr = curWr->wr.rdma.remoteAddr;
        wr.rkey = curWr->wr.rdma.rkey;
        wr.opcode = static_cast<uint32_t>(curWr->opcode);
        wr.sendFlags = static_cast<int>(curWr->sendFlags);
        wr.immData = curWr->immData;

        CHK_RET(HrtRaSendWrVerbs(qpHandle, &wr, &opRsp));

        if (curWr->next == nullptr) {
            CHK_RET(hrtRDMADBSend(opRsp.db.dbIndex, opRsp.db.dbInfo, stream));
        }

        curWr = curWr->next;
    }

    *badWr = nullptr;
    return HCCL_SUCCESS;
}

HcclResult hcclAscendPostRecv(
    AscendVerbsQPInfo* qpInfo, struct AscendRecvWr* recvWr, aclrtStream stream, struct AscendRecvWr** badWr)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PTR_NULL(qpInfo);
    CHK_PTR_NULL(recvWr);
    CHK_PTR_NULL(badWr);
    CHK_PTR_NULL(stream);

    QpHandle qpHandle;
    CHK_RET(TypicalQpManager::GetInstance().GetVerbsQpHandleByQpn(qpInfo->qpn, qpHandle));

    constexpr u32 MAX_SGLIST_RECV = 16;
    struct AscendRecvWr* curWr = recvWr;
    while (curWr != nullptr) {
        CHK_PRT_RET(
            static_cast<u32>(curWr->numSge) > MAX_SGLIST_RECV,
            HCCL_ERROR("[hcclAscendPostRecv] numSge[%d] exceeds MAX_SGLIST_RECV[%u]", curWr->numSge, MAX_SGLIST_RECV),
            HCCL_E_PARA);
        CHK_PTR_NULL(curWr->sgList);

        struct SgList sgeList[MAX_SGLIST_RECV];
        for (int i = 0; i < curWr->numSge; i++) {
            sgeList[i].addr = curWr->sgList[i].addr;
            sgeList[i].len = curWr->sgList[i].len;
            sgeList[i].lkey = curWr->sgList[i].lkey;
        }

        struct RecvWrVerbs wr = {};
        wr.wrId = curWr->wrId;
        wr.sgList = sgeList;
        wr.numSge = curWr->numSge;

        CHK_RET(HrtRaRecvWrVerbs(qpHandle, &wr));

        curWr = curWr->next;
    }

    *badWr = nullptr;
    return HCCL_SUCCESS;
}

HcclResult HcclSendByAscendQP(
    void* sendBuf, uint64_t count, HcclDataType dataType, AscendSendRecvInfo* sendRecvInfo, aclrtStream stream)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_RET(CheckParam(sendBuf, count, dataType, stream));
    CHK_RET(CheckSendRecvInfo(sendRecvInfo));
    QpHandle qpHandle;
    CHK_RET(TypicalQpManager::GetInstance().GetQpHandleByQpn(sendRecvInfo->localQPinfo->qpn, qpHandle));
    SendRecvExecutor executor(
        stream, qpHandle, AscendMrInfo2MrInfo(sendRecvInfo->localWindowMem),
        AscendMrInfo2MrInfo(sendRecvInfo->remoteWindowMem), AscendMrInfo2MrInfo(sendRecvInfo->localSyncMemPrepare),
        AscendMrInfo2MrInfo(sendRecvInfo->localSyncMemDone), AscendMrInfo2MrInfo(sendRecvInfo->localSyncMemAck),
        AscendMrInfo2MrInfo(sendRecvInfo->remoteSyncMemPrepare), AscendMrInfo2MrInfo(sendRecvInfo->remoteSyncMemDone),
        AscendMrInfo2MrInfo(sendRecvInfo->remoteSyncMemAck), sendRecvInfo->immData);
    CHK_RET(executor.Init());
    CHK_RET(executor.Send(sendBuf, count, dataType));
    return HCCL_SUCCESS;
}

HcclResult HcclRecvByAscendQP(
    void* recvBuf, uint64_t count, HcclDataType dataType, AscendSendRecvInfo* sendRecvInfo, aclrtStream stream)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_RET(CheckParam(recvBuf, count, dataType, stream));
    CHK_RET(CheckSendRecvInfo(sendRecvInfo));
    QpHandle qpHandle;
    CHK_RET(TypicalQpManager::GetInstance().GetQpHandleByQpn(sendRecvInfo->localQPinfo->qpn, qpHandle));
    SendRecvExecutor executor(
        stream, qpHandle, AscendMrInfo2MrInfo(sendRecvInfo->localWindowMem),
        AscendMrInfo2MrInfo(sendRecvInfo->remoteWindowMem), AscendMrInfo2MrInfo(sendRecvInfo->localSyncMemPrepare),
        AscendMrInfo2MrInfo(sendRecvInfo->localSyncMemDone), AscendMrInfo2MrInfo(sendRecvInfo->localSyncMemAck),
        AscendMrInfo2MrInfo(sendRecvInfo->remoteSyncMemPrepare), AscendMrInfo2MrInfo(sendRecvInfo->remoteSyncMemDone),
        AscendMrInfo2MrInfo(sendRecvInfo->remoteSyncMemAck), sendRecvInfo->immData);
    CHK_RET(executor.Init());
    CHK_RET(executor.Receive(recvBuf, count, dataType));
    return HCCL_SUCCESS;
}

HcclResult hcclAscendRdmaInit()
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_RET(RdmaResourceManager::GetInstance().Init());
    return HCCL_SUCCESS;
}
HcclResult hcclAscendRdmaInitV2()
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    RdmaResourceManager::GetInstance().SetDisableLiteThread(true);
    CHK_RET(RdmaResourceManager::GetInstance().Init());
    return HCCL_SUCCESS;
}
HcclResult hcclAscendRdmaDeInit()
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_RET(RdmaResourceManager::GetInstance().DeInit());
    return HCCL_SUCCESS;
}

HcclResult HcclGetCqeErrInfoList(struct HcclErrCqeInfo* infoList, uint32_t* num)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PTR_NULL(infoList);
    CHK_PTR_NULL(num);
    u32 arrLen = *num;
    struct CqeErrInfo errCqeList[arrLen];
    CHK_RET(RdmaResourceManager::GetInstance().GetCqeErrInfo(errCqeList, num));

    CHK_PRT_RET(
        *num > arrLen,
        HCCL_ERROR(
            "[HcclGetCqeErrInfoList] GetCqeErrInfo num[%u] is larger than "
            "infoList user given[%u].",
            num, arrLen),
        HCCL_E_INTERNAL);
    for (u32 i = 0; i < *num; i++) {
        infoList[i].status = errCqeList[i].status;
        infoList[i].qpn = errCqeList[i].qpn;
        infoList[i].time = errCqeList[i].time;
        time_t tmpt = static_cast<time_t>(errCqeList[i].time.tv_sec);
        struct tm errTime;
        localtime_r(&tmpt, &errTime);
        HCCL_INFO(
            "[HcclGetCqeErrInfoList] Err Cqe status[%d], qpn[%d], time[%04u-%02d-%02d %02d:%0d:%02d.%06u]",
            errCqeList[i].status, errCqeList[i].qpn, errTime.tm_year + TIME_FROM_1900, errTime.tm_mon + 1,
            errTime.tm_mday, errTime.tm_hour, errTime.tm_min, errTime.tm_sec,
            static_cast<u32>(errCqeList[i].time.tv_usec));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetCqeErrInfoListByQpn(uint32_t qpn, struct HcclErrCqeInfo* infoList, uint32_t* num)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PTR_NULL(infoList);
    CHK_PTR_NULL(num);
    CHK_RET(RdmaResourceManager::GetInstance().GetCqeErrInfoByQpn(qpn, infoList, num));
    return HCCL_SUCCESS;
}

HcclResult HcclPutByAscendQP(
    void* putBuf, uint64_t count, HcclDataType dataType, AscendSendRecvInfo* sendRecvInfo, aclrtStream stream)
{
    CHK_PTR_NULL(stream);
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_RET(CheckParam(putBuf, count, dataType, stream));
    QpHandle qpHandle;
    CHK_RET(TypicalQpManager::GetInstance().GetQpHandleByQpn(sendRecvInfo->localQPinfo->qpn, qpHandle));
    SendRecvExecutor executor(
        stream, qpHandle, AscendMrInfo2MrInfo(sendRecvInfo->localWindowMem),
        AscendMrInfo2MrInfo(sendRecvInfo->remoteWindowMem), AscendMrInfo2MrInfo(sendRecvInfo->localSyncMemPrepare),
        AscendMrInfo2MrInfo(sendRecvInfo->localSyncMemDone), AscendMrInfo2MrInfo(sendRecvInfo->localSyncMemAck),
        AscendMrInfo2MrInfo(sendRecvInfo->remoteSyncMemPrepare), AscendMrInfo2MrInfo(sendRecvInfo->remoteSyncMemDone),
        AscendMrInfo2MrInfo(sendRecvInfo->remoteSyncMemAck), sendRecvInfo->immData);
    CHK_RET(executor.Init());
    CHK_RET(executor.Put(putBuf, count, dataType));
    return HCCL_SUCCESS;
}

HcclResult HcclBatchPutMRByAscendQP(
    unsigned int num, AscendMrInfo* putMRList, AscendMrInfo* remoteMRList, AscendSendRecvLinkInfo* sendRecvLinkInfo,
    aclrtStream stream)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PRT_RET(
        num == 0, HCCL_INFO("[HcclBatchPutMRByAscendQP] mr list len is 0. No need to send data."), HCCL_SUCCESS);
    CHK_PTR_NULL(putMRList);
    CHK_PTR_NULL(remoteMRList);
    CHK_PTR_NULL(sendRecvLinkInfo);
    CHK_PTR_NULL(stream);
    CHK_RET(CheckSendRecvLinkInfo(sendRecvLinkInfo));
    CHK_PRT_RET(
        sendRecvLinkInfo->wqePerDoorbell == 0 || sendRecvLinkInfo->wqePerDoorbell > MAX_WQE_PER_DOORBELL,
        HCCL_ERROR(
            "[HcclBatchPutMRByAscendQP] wqePerDoorbell[%u] is invalid, should be greater than 0 and not exceed "
            "MAX_WQE_PER_DOORBELL[%u].",
            sendRecvLinkInfo->wqePerDoorbell, MAX_WQE_PER_DOORBELL),
        HCCL_E_PARA);
    QpHandle qpHandle;
    CHK_RET(TypicalQpManager::GetInstance().GetQpHandleByQpn(sendRecvLinkInfo->localQPinfo->qpn, qpHandle));

    SendRecvExecutor executor(stream, qpHandle, sendRecvLinkInfo);
    CHK_RET(executor.Init());
    CHK_RET(executor.BatchPutMR(num, putMRList, remoteMRList));

    return HCCL_SUCCESS;
}

HcclResult HcclWaitPutMRByAscendQP(AscendSendRecvLinkInfo* sendRecvLinkInfo, aclrtStream stream)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PTR_NULL(sendRecvLinkInfo);
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(sendRecvLinkInfo->remoteSyncMemAck);
    CHK_PTR_NULL(sendRecvLinkInfo->localSyncMemDone);
    QpHandle qpHandle;
    CHK_RET(TypicalQpManager::GetInstance().GetQpHandleByQpn(sendRecvLinkInfo->localQPinfo->qpn, qpHandle));

    SendRecvExecutor executor(stream, qpHandle, sendRecvLinkInfo->localSyncMemDone, sendRecvLinkInfo->remoteSyncMemAck);
    CHK_RET(executor.WaitPutInit());
    CHK_RET(executor.WaitPutMR());
    return HCCL_SUCCESS;
}

HcclResult HcclWaitPutMRDoWait(AscendSendRecvLinkInfo* sendRecvLinkInfo, aclrtStream stream)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PTR_NULL(sendRecvLinkInfo);
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(sendRecvLinkInfo->remoteSyncMemAck);
    CHK_PTR_NULL(sendRecvLinkInfo->localSyncMemDone);
    QpHandle qpHandle;
    CHK_RET(TypicalQpManager::GetInstance().GetQpHandleByQpn(sendRecvLinkInfo->localQPinfo->qpn, qpHandle));

    SendRecvExecutor executor(stream, qpHandle, sendRecvLinkInfo->localSyncMemDone, sendRecvLinkInfo->remoteSyncMemAck);
    CHK_RET(executor.WaitPutInit());
    CHK_RET(executor.WaitPutMROnlyWait());
    return HCCL_SUCCESS;
}

HcclResult HcclWaitPutMRDoRecord(AscendSendRecvLinkInfo* sendRecvLinkInfo, aclrtStream stream)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PTR_NULL(sendRecvLinkInfo);
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(sendRecvLinkInfo->remoteSyncMemAck);
    CHK_PTR_NULL(sendRecvLinkInfo->localSyncMemDone);
    QpHandle qpHandle;
    CHK_RET(TypicalQpManager::GetInstance().GetQpHandleByQpn(sendRecvLinkInfo->localQPinfo->qpn, qpHandle));

    SendRecvExecutor executor(stream, qpHandle, sendRecvLinkInfo->localSyncMemDone, sendRecvLinkInfo->remoteSyncMemAck);
    CHK_RET(executor.WaitPutInit());
    CHK_RET(executor.WaitPutMROnlyRecord());
    return HCCL_SUCCESS;
}

HcclResult hcclGetSyncMemRegKey(AscendMrInfo* memInfo)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PTR_NULL(memInfo);
    struct MrInfoT mrInfo {};
    CHK_RET(RdmaResourceManager::GetInstance().GetNotifyMrInfo(mrInfo));
    memInfo->key = mrInfo.lkey;
    HCCL_RUN_INFO(
        "[hcclGetSyncMemRegKey] SyncMem addr[%p], size[%llu], key[%u].", memInfo->addr, memInfo->size, memInfo->key);
    return HCCL_SUCCESS;
}

HcclResult HcclOneSideBatchPutByAscendQP(
    unsigned int num, AscendMrInfo* putMRList, AscendMrInfo* remoteMRList, AscendSendLinkInfo* sendlinkInfo,
    aclrtStream stream)
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PRT_RET(
        num == 0, HCCL_INFO("[HcclOneSideBatchPutByAscendQP] mr list len is 0. No need to send data."), HCCL_SUCCESS);
    CHK_PTR_NULL(putMRList);
    CHK_PTR_NULL(remoteMRList);
    CHK_PTR_NULL(sendlinkInfo);
    CHK_PTR_NULL(stream);
    CHK_RET(CheckSendLinkInfo(sendlinkInfo));
    CHK_PRT_RET(
        sendlinkInfo->wqePerDoorbell == 0 || sendlinkInfo->wqePerDoorbell > MAX_WQE_PER_DOORBELL,
        HCCL_ERROR(
            "[HcclOneSideBatchPutByAscendQP] wqePerDoorbell[%u] is invalid, should be greater than 0 and not exceed "
            "MAX_WQE_PER_DOORBELL[%u].",
            sendlinkInfo->wqePerDoorbell, MAX_WQE_PER_DOORBELL),
        HCCL_E_PARA);
    QpHandle qpHandle;
    CHK_RET(TypicalQpManager::GetInstance().GetQpHandleByQpn(sendlinkInfo->localQPinfo->qpn, qpHandle));

    SendRecvExecutor executor(stream, qpHandle, sendlinkInfo);
    CHK_RET(executor.OneSideBatchPutMR(num, putMRList, remoteMRList));

    return HCCL_SUCCESS;
}
