/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_unfold_cache.h"

#include "aicpu_hccl_sqcq.h"
#include "aicpu_hccl_sqcqv1.h"
#include "aicpu_hccl_sqcqv2.h"
#include "config_plf_log.h"
#include "log.h"

namespace hccl {

constexpr size_t OP_UNFOLD_CACHE_CAPACITY = 500; // Cache entry数目的上限

OpUnfoldCache::OpUnfoldCache()
{
    HCCL_INFO("[OpUnfoldCache][OpUnfoldCache] create a cache at 0x%016llx for operator unfolding", this);
}

OpUnfoldCache::~OpUnfoldCache()
{
    HCCL_INFO(
        "[OpUnfoldCache][~OpUnfoldCache] release dynamically allocated entries from the cache at 0x%016llx", this);

    for (CacheHashMap::const_iterator constIter = cacheHashMap_.cbegin(); constIter != cacheHashMap_.cend();
         ++constIter) {
        OpUnfoldCacheEntry* entryPtr = constIter->second;

        // 不能使用CHK_PTR_NULL，因为会return HcclResult
        if (UNLIKELY(entryPtr == nullptr)) {
            HCCL_ERROR("[OpUnfoldCache][OpUnfoldCache] entryPtr is nullptr");
            continue;
        }

        // 释放当前cache entry
        delete entryPtr;
        entryPtr = nullptr;
    }
}

bool OpUnfoldCache::IsCacheFull() const
{
    const size_t curSize = cacheHashMap_.size();
    if (curSize >= OP_UNFOLD_CACHE_CAPACITY) {
        HCCL_INFO(
            "[OpUnfoldCache][IsCacheFull] cacheHashMap_.size[%u] >= capacity[%u]", cacheHashMap_.size(),
            OP_UNFOLD_CACHE_CAPACITY);
        return true;
    }
    return false;
}

HcclResult OpUnfoldCache::FindEntry(const OpUnfoldKey& key, OpUnfoldCacheEntry** entryPtrPtr) const
{
    // 检查入参
    CHK_PTRPTR_NULL(entryPtrPtr); // 检查指针, entryPtrPtr不应该是null, 但*entryPtrPtr应该是null

    CacheHashMap::const_iterator constIter = cacheHashMap_.find(key);
    if (constIter != cacheHashMap_.cend()) {
        // 检查cache entry pointer
        *entryPtrPtr = constIter->second;
        CHK_PTR_NULL(*entryPtrPtr);

        HCCL_INFO("[OpUnfoldCache][FindEntry] find cache entry for key[%s]", key.GetKeyString().c_str());
    } else {
        *entryPtrPtr = nullptr;

        HCCL_INFO("[OpUnfoldCache][FindEntry] not find cache entry for key[%s]", key.GetKeyString().c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult OpUnfoldCache::AddEntry(
    const OpUnfoldKey& key, const std::vector<OpUnfoldMemRange>& userInputMemRanges,
    const std::vector<OpUnfoldMemRange>& userOutputMemRanges, OpUnfoldCacheEntry** entryPtrPtr)
{
    // 检查入参
    CHK_PTRPTR_NULL(entryPtrPtr); // 检查指针, entryPtrPtr不应该是null, 但*entryPtrPtr应该是null

    CacheHashMap::iterator iter = cacheHashMap_.find(key);

    // key不应该存在
    CHK_PRT_RET(
        iter != cacheHashMap_.end(),
        HCCL_ERROR(
            "[OpUnfoldCache][AddEntry] cache entry for key[%s] exists at 0x%016llx", key.GetKeyString().c_str(), this),
        HCCL_E_INTERNAL);

    // 创建一个新的cache entry
    OpUnfoldCacheEntry* newCacheEntryPtr
        = (new (std::nothrow) OpUnfoldCacheEntry(userInputMemRanges, userOutputMemRanges));
    CHK_PTR_NULL(newCacheEntryPtr);

    // 插入新的cache entry
    std::pair<CacheHashMap::iterator, bool> insertResult = cacheHashMap_.emplace(key, newCacheEntryPtr);
    if (UNLIKELY(!(insertResult.second))) {
        HCCL_ERROR(
            "[OpUnfoldCache][AddEntry] fail to insert a new cache entry for key[%s]", key.GetKeyString().c_str());

        delete newCacheEntryPtr;
        newCacheEntryPtr = nullptr;

        return HCCL_E_INTERNAL;
    } else {
        HCCL_INFO(
            "[OpUnfoldCache][AddEntry] add a new cache entry for key[%s] cacheHashMap_.size[%u]",
            key.GetKeyString().c_str(), cacheHashMap_.size());

        iter = insertResult.first;
        *entryPtrPtr = iter->second;
        CHK_PTR_NULL(*entryPtrPtr);
    }

    return HCCL_SUCCESS;
}

HcclResult OpUnfoldCache::ClearEntry(const OpUnfoldKey& key)
{
    CacheHashMap::iterator iter = cacheHashMap_.find(key);
    if (iter != cacheHashMap_.end()) { // cache entry存在
        HCCL_INFO("[OpUnfoldCache][ClearEntry] clear cache entry for key[%s]", key.GetKeyString().c_str());

        // 释放cache entry对应的内存
        OpUnfoldCacheEntry* entryPtr = iter->second;
        CHK_PTR_NULL(entryPtr);
        delete entryPtr;
        entryPtr = nullptr;
        iter->second = nullptr;

        // 清理cache entry
        cacheHashMap_.erase(iter);
    } else { // cache entry不存在
        HCCL_INFO("[OpUnfoldCache][ClearEntry] not find cache entry for key [%s]", key.GetKeyString().c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult OpUnfoldCache::ClearEntryForAlltoallv()
{
    // 清理所有与alltoallv类算子相关的cache entry
    for (CacheHashMap::iterator iter = cacheHashMap_.begin(); iter != cacheHashMap_.end();) {
        // 先备份iter的下一个位置 (使用std::next避免修改iter本身)
        CacheHashMap::iterator nextIter = std::next(iter);

        // 清理alltoallv类算子的cache entry
        const HcclCMDType opType = iter->first.opType;
        if (opType == HcclCMDType::HCCL_CMD_ALLTOALLV || opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
            CHK_RET(ClearEntry(iter->first));
        }

        // 注意: ClearEntry后不能再访问iter (已经从cacheHashMap_中被erase了)

        // 推进iter到下一个位置
        iter = nextIter;
    }

    return HCCL_SUCCESS;
}

HcclResult OpUnfoldCache::DumpSqeContent(const uint8_t* sqePtr, const uint8_t sqeType)
{
    // 根据sqe type打印debug信息
    // 设置HCCL_DEBUG_CONFIG="task", 或者设置ASCEND_GLOBAL_LOG_LEVEL=0
    if ((UNLIKELY(GetExternalInputDebugConfig() & PLF_TASK)) || UNLIKELY(HcclCheckLogLevel(HCCL_LOG_DEBUG))) {
        CHK_PTR_NULL(sqePtr);

        switch (sqeType) {
            case SqeType::NOTIFY_SQE: {
                const rtStarsNotifySqeV1_t* notifySqePtr = reinterpret_cast<const rtStarsNotifySqeV1_t*>(sqePtr);
                PLF_CONFIG_DEBUG(PLF_TASK, "[OpUnfoldCache][DumpSqeContent] rtStarsNotifySqeV1_t sqeType[%u]", sqeType);
                CHK_RET(DumpSqeHeader(notifySqePtr->header));
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] notify_id[%u] res2[%u] res3[%u] kernel_credit[%u] res4[%u] "
                    "timeout[%u]",
                    notifySqePtr->notify_id, notifySqePtr->res2, notifySqePtr->res3, notifySqePtr->kernel_credit,
                    notifySqePtr->res4, notifySqePtr->timeout);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] res5[0][%u] res5[1][%u] res5[2][%u] res5[3][%u] res5[4][%u] "
                    "res5[5][%u]",
                    notifySqePtr->res5[0], notifySqePtr->res5[1], notifySqePtr->res5[2], notifySqePtr->res5[3],
                    notifySqePtr->res5[4], notifySqePtr->res5[5]);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] res5[6][%u] res5[7][%u] res5[8][%u] res5[9][%u] res5[10][%u]",
                    notifySqePtr->res5[6], notifySqePtr->res5[7], notifySqePtr->res5[8], notifySqePtr->res5[9],
                    notifySqePtr->res5[10]);
                break;
            }
            case SqeType::WRITE_VALUE_SQE:
            case SqeType::RDMA_DB_SEND_SQE: {
                const rtStarsWriteValueSqe_t* writeValueSqePtr
                    = reinterpret_cast<const rtStarsWriteValueSqe_t*>(sqePtr);
                PLF_CONFIG_DEBUG(
                    PLF_TASK, "[OpUnfoldCache][DumpSqeContent] rtStarsWriteValueSqe_t sqeType[%u]", sqeType);
                CHK_RET(DumpSqeHeader(writeValueSqePtr->header));
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] res3[%u] res4[%u] kernel_credit[%u] res5[%u] "
                    "write_addr_low[0x%08x] write_addr_high[0x%08x]",
                    writeValueSqePtr->res3, writeValueSqePtr->res4, writeValueSqePtr->kernel_credit,
                    writeValueSqePtr->res5, writeValueSqePtr->write_addr_low, writeValueSqePtr->write_addr_high);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] res6[%u] awsize[%u] snoop[%u] awcache[%u] awprot[%u] va[%u] "
                    "res7[%u]",
                    writeValueSqePtr->res6, writeValueSqePtr->awsize, writeValueSqePtr->snoop,
                    writeValueSqePtr->awcache, writeValueSqePtr->awprot, writeValueSqePtr->va, writeValueSqePtr->res7);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] sub_type[%u] write_value_part0[%u] write_value_part1[%u] "
                    "rdmaWrLenth[%u] rdmaType[%u]",
                    writeValueSqePtr->sub_type, writeValueSqePtr->write_value_part0,
                    writeValueSqePtr->write_value_part1, writeValueSqePtr->rdmaWrLenth, writeValueSqePtr->rdmaType);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] write_value_part4[%u] write_value_part5[%u] write_value_part6[%u] "
                    "write_value_part7[%u]",
                    writeValueSqePtr->write_value_part4, writeValueSqePtr->write_value_part5,
                    writeValueSqePtr->write_value_part6, writeValueSqePtr->write_value_part7);
                break;
            }
            case SqeType::EVENT_SQE: {
                const rtStarsEventSqe_t* eventSqePtr = reinterpret_cast<const rtStarsEventSqe_t*>(sqePtr);
                PLF_CONFIG_DEBUG(PLF_TASK, "[OpUnfoldCache][DumpSqeContent] rtStarsEventSqe_t sqeType[%u]", sqeType);
                CHK_RET(DumpSqeHeader(eventSqePtr->header));
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] eventId[%u] res2[%u] res3[%u] kernel_credit[%u] res4[%u] "
                    "exe_result[%u] timeout[%u]",
                    eventSqePtr->eventId, eventSqePtr->res2, eventSqePtr->res3, eventSqePtr->kernel_credit,
                    eventSqePtr->res4, eventSqePtr->exe_result, eventSqePtr->timeout);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] res5[0][%u] res5[1][%u] res5[2][%u] res5[3][%u] res5[4][%u]",
                    eventSqePtr->res5[0], eventSqePtr->res5[1], eventSqePtr->res5[2], eventSqePtr->res5[3],
                    eventSqePtr->res5[4]);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] res5[5][%u] res5[6][%u] res5[7][%u] res5[8][%u] res5[9][%u]",
                    eventSqePtr->res5[5], eventSqePtr->res5[6], eventSqePtr->res5[7], eventSqePtr->res5[8],
                    eventSqePtr->res5[9]);
                break;
            }
            case SqeType::MEMCPY_ASYNC_SQE: {
                const rtStarsMemcpyAsyncSqe_t* memcpyAsyncSqePtr
                    = reinterpret_cast<const rtStarsMemcpyAsyncSqe_t*>(sqePtr);
                PLF_CONFIG_DEBUG(
                    PLF_TASK, "[OpUnfoldCache][DumpSqeContent] rtStarsMemcpyAsyncSqe_t sqeType[%u]", sqeType);
                CHK_RET(DumpSqeHeader(memcpyAsyncSqePtr->header));
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] res3[%u] res4[%u] kernel_credit[%u] ptr_mode[%u] res5[%u] "
                    "opcode[%u] ie2[%u]",
                    memcpyAsyncSqePtr->res3, memcpyAsyncSqePtr->res4, memcpyAsyncSqePtr->kernel_credit,
                    memcpyAsyncSqePtr->ptr_mode, memcpyAsyncSqePtr->res5, memcpyAsyncSqePtr->opcode,
                    memcpyAsyncSqePtr->ie2);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] sssv[%u] dssv[%u] sns[%u] dns[%u] qos[%u] sro[%u] dro[%u] "
                    "partid[%u] mpam[%u]",
                    memcpyAsyncSqePtr->sssv, memcpyAsyncSqePtr->dssv, memcpyAsyncSqePtr->sns, memcpyAsyncSqePtr->dns,
                    memcpyAsyncSqePtr->qos, memcpyAsyncSqePtr->sro, memcpyAsyncSqePtr->dro, memcpyAsyncSqePtr->partid,
                    memcpyAsyncSqePtr->mpam);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] res6[%u] src_streamid[%u] src_sub_streamid[%u] dst_streamid[%u] "
                    "dst_sub_streamid[%u]",
                    memcpyAsyncSqePtr->res6, memcpyAsyncSqePtr->src_streamid, memcpyAsyncSqePtr->src_sub_streamid,
                    memcpyAsyncSqePtr->dst_streamid, memcpyAsyncSqePtr->dst_sub_streamid);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] length[%u] src_addr_low[0x%08x] src_addr_high[0x%08x] "
                    "dst_addr_low[0x%08x] dst_addr_high[0x%08x]",
                    memcpyAsyncSqePtr->length, memcpyAsyncSqePtr->src_addr_low, memcpyAsyncSqePtr->src_addr_high,
                    memcpyAsyncSqePtr->dst_addr_low, memcpyAsyncSqePtr->dst_addr_high);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] linkType[%u] reserved[0][%u] reserved[1][%u] reserved[2][%u] "
                    "reslast[0][%u] reslast[1][%u] reslast[2][%u]",
                    memcpyAsyncSqePtr->linkType, memcpyAsyncSqePtr->resvered[0], memcpyAsyncSqePtr->resvered[1],
                    memcpyAsyncSqePtr->resvered[2], memcpyAsyncSqePtr->reslast[0], memcpyAsyncSqePtr->reslast[1],
                    memcpyAsyncSqePtr->reslast[2]);
                break;
            }
            case SqeType::CCORE_WAIT_START_SQE: {
                const rtStarsCcoreWaitStartSqe_t* ccoreWaitStartSqePtr
                    = reinterpret_cast<const rtStarsCcoreWaitStartSqe_t*>(sqePtr);
                PLF_CONFIG_DEBUG(
                    PLF_TASK, "[OpUnfoldCache][DumpSqeContent] rtStarsCcoreWaitStartSqe_t sqeType[%u]", sqeType);
                CHK_RET(DumpSqeHeader(ccoreWaitStartSqePtr->sqeHeader));
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] reserved0[%u] reserved1[%u] kernel_credit[%u] reserved2[%u] "
                    "csc[%u]",
                    ccoreWaitStartSqePtr->reserved0, ccoreWaitStartSqePtr->reserved1,
                    ccoreWaitStartSqePtr->kernel_credit, ccoreWaitStartSqePtr->reserved2, ccoreWaitStartSqePtr->csc);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] ldrImm1: opCode[%u] rd[%u] reserved[%u] func3[%u] "
                    "immdAddrHigh[0x%08x] immdAddrLow[0x%08x]",
                    ccoreWaitStartSqePtr->ldrImm1.opCode, ccoreWaitStartSqePtr->ldrImm1.rd,
                    ccoreWaitStartSqePtr->ldrImm1.reserved, ccoreWaitStartSqePtr->ldrImm1.func3,
                    ccoreWaitStartSqePtr->ldrImm1.immdAddrHigh, ccoreWaitStartSqePtr->ldrImm1.immdAddrLow);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] ldrImm2: opCode[%u] rd[%u] reserved[%u] func3[%u] "
                    "immdAddrHigh[0x%08x] immdAddrLow[0x%08x]",
                    ccoreWaitStartSqePtr->ldrImm2.opCode, ccoreWaitStartSqePtr->ldrImm2.rd,
                    ccoreWaitStartSqePtr->ldrImm2.reserved, ccoreWaitStartSqePtr->ldrImm2.func3,
                    ccoreWaitStartSqePtr->ldrImm2.immdAddrHigh, ccoreWaitStartSqePtr->ldrImm2.immdAddrLow);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] beq: opCode[%u] jumpInstrOffset[%u] rsvd[%u] func3[%u] rs1[%u] "
                    "rsvd1[%u] rs2[%u] rsvd2[%u] rsvd3[%u]",
                    ccoreWaitStartSqePtr->beq.opCode, ccoreWaitStartSqePtr->beq.jumpInstrOffset,
                    ccoreWaitStartSqePtr->beq.rsvd, ccoreWaitStartSqePtr->beq.func3, ccoreWaitStartSqePtr->beq.rs1,
                    ccoreWaitStartSqePtr->beq.rsvd1, ccoreWaitStartSqePtr->beq.rs2, ccoreWaitStartSqePtr->beq.rsvd2,
                    ccoreWaitStartSqePtr->beq.rsvd3);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] clear.llwi1: opCode[%u] rd[%u] reserved0[%u] func3[%u] "
                    "immdHigh[%u] immdLow[%u]",
                    ccoreWaitStartSqePtr->clear.llwi1.opCode, ccoreWaitStartSqePtr->clear.llwi1.rd,
                    ccoreWaitStartSqePtr->clear.llwi1.reserved0, ccoreWaitStartSqePtr->clear.llwi1.func3,
                    ccoreWaitStartSqePtr->clear.llwi1.immdHigh, ccoreWaitStartSqePtr->clear.llwi1.immdLow);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] clear.lhwi1: opCode[%u] rd[%u] reserved0[%u] func3[%u] "
                    "reserved1[%u] immd[%u]",
                    ccoreWaitStartSqePtr->clear.lhwi1.opCode, ccoreWaitStartSqePtr->clear.lhwi1.rd,
                    ccoreWaitStartSqePtr->clear.lhwi1.reserved0, ccoreWaitStartSqePtr->clear.lhwi1.func3,
                    ccoreWaitStartSqePtr->clear.lhwi1.reserved1, ccoreWaitStartSqePtr->clear.lhwi1.immd);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] clear.sw: opCode[%u] immdLow[%u] func3[%u] rs1[%u] reserved1[%u] "
                    "rs2[%u] reserved2[%u] immdHigh[%u]",
                    ccoreWaitStartSqePtr->clear.sw.opCode, ccoreWaitStartSqePtr->clear.sw.immdLow,
                    ccoreWaitStartSqePtr->clear.sw.func3, ccoreWaitStartSqePtr->clear.sw.rs1,
                    ccoreWaitStartSqePtr->clear.sw.reserved1, ccoreWaitStartSqePtr->clear.sw.rs2,
                    ccoreWaitStartSqePtr->clear.sw.reserved2, ccoreWaitStartSqePtr->clear.sw.immdHigh);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] clear.nop[0]: opCode[%u] rd[%u] reserved0[%u] func3[%u] rs1[%u] "
                    "reserved1[%u] immd[%u]",
                    ccoreWaitStartSqePtr->clear.nop[0].opCode, ccoreWaitStartSqePtr->clear.nop[0].rd,
                    ccoreWaitStartSqePtr->clear.nop[0].reserved0, ccoreWaitStartSqePtr->clear.nop[0].func3,
                    ccoreWaitStartSqePtr->clear.nop[0].rs1, ccoreWaitStartSqePtr->clear.nop[0].reserved1,
                    ccoreWaitStartSqePtr->clear.nop[0].immd);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] clear.nop[1]: opCode[%u] rd[%u] reserved0[%u] func3[%u] rs1[%u] "
                    "reserved1[%u] immd[%u]",
                    ccoreWaitStartSqePtr->clear.nop[1].opCode, ccoreWaitStartSqePtr->clear.nop[1].rd,
                    ccoreWaitStartSqePtr->clear.nop[1].reserved0, ccoreWaitStartSqePtr->clear.nop[1].func3,
                    ccoreWaitStartSqePtr->clear.nop[1].rs1, ccoreWaitStartSqePtr->clear.nop[1].reserved1,
                    ccoreWaitStartSqePtr->clear.nop[1].immd);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] clear.nop[2]: opCode[%u] rd[%u] reserved0[%u] func3[%u] rs1[%u] "
                    "reserved1[%u] immd[%u]",
                    ccoreWaitStartSqePtr->clear.nop[2].opCode, ccoreWaitStartSqePtr->clear.nop[2].rd,
                    ccoreWaitStartSqePtr->clear.nop[2].reserved0, ccoreWaitStartSqePtr->clear.nop[2].func3,
                    ccoreWaitStartSqePtr->clear.nop[2].rs1, ccoreWaitStartSqePtr->clear.nop[2].reserved1,
                    ccoreWaitStartSqePtr->clear.nop[2].immd);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] nop[0]: opCode[%u] rd[%u] reserved0[%u] func3[%u] rs1[%u] "
                    "reserved1[%u] immd[%u]",
                    ccoreWaitStartSqePtr->nop[0].opCode, ccoreWaitStartSqePtr->nop[0].rd,
                    ccoreWaitStartSqePtr->nop[0].reserved0, ccoreWaitStartSqePtr->nop[0].func3,
                    ccoreWaitStartSqePtr->nop[0].rs1, ccoreWaitStartSqePtr->nop[0].reserved1,
                    ccoreWaitStartSqePtr->nop[0].immd);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] nop[1]: opCode[%u] rd[%u] reserved0[%u] func3[%u] rs1[%u] "
                    "reserved1[%u] immd[%u]",
                    ccoreWaitStartSqePtr->nop[1].opCode, ccoreWaitStartSqePtr->nop[1].rd,
                    ccoreWaitStartSqePtr->nop[1].reserved0, ccoreWaitStartSqePtr->nop[1].func3,
                    ccoreWaitStartSqePtr->nop[1].rs1, ccoreWaitStartSqePtr->nop[1].reserved1,
                    ccoreWaitStartSqePtr->nop[1].immd);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] nop[2]: opCode[%u] rd[%u] reserved0[%u] func3[%u] rs1[%u] "
                    "reserved1[%u] immd[%u]",
                    ccoreWaitStartSqePtr->nop[2].opCode, ccoreWaitStartSqePtr->nop[2].rd,
                    ccoreWaitStartSqePtr->nop[2].reserved0, ccoreWaitStartSqePtr->nop[2].func3,
                    ccoreWaitStartSqePtr->nop[2].rs1, ccoreWaitStartSqePtr->nop[2].reserved1,
                    ccoreWaitStartSqePtr->nop[2].immd);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] nop[3]: opCode[%u] rd[%u] reserved0[%u] func3[%u] rs1[%u] "
                    "reserved1[%u] immd[%u]",
                    ccoreWaitStartSqePtr->nop[3].opCode, ccoreWaitStartSqePtr->nop[3].rd,
                    ccoreWaitStartSqePtr->nop[3].reserved0, ccoreWaitStartSqePtr->nop[3].func3,
                    ccoreWaitStartSqePtr->nop[3].rs1, ccoreWaitStartSqePtr->nop[3].reserved1,
                    ccoreWaitStartSqePtr->nop[3].immd);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] nop[4]: opCode[%u] rd[%u] reserved0[%u] func3[%u] rs1[%u] "
                    "reserved1[%u] immd[%u]",
                    ccoreWaitStartSqePtr->nop[4].opCode, ccoreWaitStartSqePtr->nop[4].rd,
                    ccoreWaitStartSqePtr->nop[4].reserved0, ccoreWaitStartSqePtr->nop[4].func3,
                    ccoreWaitStartSqePtr->nop[4].rs1, ccoreWaitStartSqePtr->nop[4].reserved1,
                    ccoreWaitStartSqePtr->nop[4].immd);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] nop[5]: opCode[%u] rd[%u] reserved0[%u] func3[%u] rs1[%u] "
                    "reserved1[%u] immd[%u]",
                    ccoreWaitStartSqePtr->nop[5].opCode, ccoreWaitStartSqePtr->nop[5].rd,
                    ccoreWaitStartSqePtr->nop[5].reserved0, ccoreWaitStartSqePtr->nop[5].func3,
                    ccoreWaitStartSqePtr->nop[5].rs1, ccoreWaitStartSqePtr->nop[5].reserved1,
                    ccoreWaitStartSqePtr->nop[5].immd);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] nop[6]: opCode[%u] rd[%u] reserved0[%u] func3[%u] rs1[%u] "
                    "reserved1[%u] immd[%u]",
                    ccoreWaitStartSqePtr->nop[6].opCode, ccoreWaitStartSqePtr->nop[6].rd,
                    ccoreWaitStartSqePtr->nop[6].reserved0, ccoreWaitStartSqePtr->nop[6].func3,
                    ccoreWaitStartSqePtr->nop[6].rs1, ccoreWaitStartSqePtr->nop[6].reserved1,
                    ccoreWaitStartSqePtr->nop[6].immd);
                break;
            }
            case SqeType::CCORE_WRITE_VALUE_SQE: {
                const rtStarsCcoreWriteValueSqe_t* ccoreWriteValueSqePtr
                    = reinterpret_cast<const rtStarsCcoreWriteValueSqe_t*>(sqePtr);
                PLF_CONFIG_DEBUG(
                    PLF_TASK, "[OpUnfoldCache][DumpSqeContent] rtStarsCcoreWriteValueSqe_t sqeType[%u]", sqeType);
                CHK_RET(DumpSqeHeader(ccoreWriteValueSqePtr->sqeHeader));
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] reserved0[%u] reserved1[%u] kernel_credit[%u] reserved2[%u] "
                    "csc[%u]",
                    ccoreWriteValueSqePtr->reserved0, ccoreWriteValueSqePtr->reserved1,
                    ccoreWriteValueSqePtr->kernel_credit, ccoreWriteValueSqePtr->reserved2, ccoreWriteValueSqePtr->csc);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] ldrImm: opCode[%u] rd[%u] reserved[%u] func3[%u] "
                    "immdAddrHigh[0x%08x] immdAddrLow[0x%08x]",
                    ccoreWriteValueSqePtr->ldrImm.opCode, ccoreWriteValueSqePtr->ldrImm.rd,
                    ccoreWriteValueSqePtr->ldrImm.reserved, ccoreWriteValueSqePtr->ldrImm.func3,
                    ccoreWriteValueSqePtr->ldrImm.immdAddrHigh, ccoreWriteValueSqePtr->ldrImm.immdAddrLow);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] llwi1: opCode[%u] rd[%u] reserved0[%u] func3[%u] immdHigh[%u] "
                    "immdLow[%u]",
                    ccoreWriteValueSqePtr->llwi1.opCode, ccoreWriteValueSqePtr->llwi1.rd,
                    ccoreWriteValueSqePtr->llwi1.reserved0, ccoreWriteValueSqePtr->llwi1.func3,
                    ccoreWriteValueSqePtr->llwi1.immdHigh, ccoreWriteValueSqePtr->llwi1.immdLow);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] lhwi1: opCode[%u] rd[%u] reserved0[%u] func3[%u] reserved1[%u] "
                    "immd[%u]",
                    ccoreWriteValueSqePtr->lhwi1.opCode, ccoreWriteValueSqePtr->lhwi1.rd,
                    ccoreWriteValueSqePtr->lhwi1.reserved0, ccoreWriteValueSqePtr->lhwi1.func3,
                    ccoreWriteValueSqePtr->lhwi1.reserved1, ccoreWriteValueSqePtr->lhwi1.immd);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] sw: opCode[%u] immdLow[%u] func3[%u] rs1[%u] reserved1[%u] "
                    "rs2[%u] reserved2[%u] immdHigh[%u]",
                    ccoreWriteValueSqePtr->sw.opCode, ccoreWriteValueSqePtr->sw.immdLow,
                    ccoreWriteValueSqePtr->sw.func3, ccoreWriteValueSqePtr->sw.rs1, ccoreWriteValueSqePtr->sw.reserved1,
                    ccoreWriteValueSqePtr->sw.rs2, ccoreWriteValueSqePtr->sw.reserved2,
                    ccoreWriteValueSqePtr->sw.immdHigh);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] nop[0]: opCode[%u] rd[%u] reserved0[%u] func3[%u] rs1[%u] "
                    "reserved1[%u] immd[%u]",
                    ccoreWriteValueSqePtr->nop[0].opCode, ccoreWriteValueSqePtr->nop[0].rd,
                    ccoreWriteValueSqePtr->nop[0].reserved0, ccoreWriteValueSqePtr->nop[0].func3,
                    ccoreWriteValueSqePtr->nop[0].rs1, ccoreWriteValueSqePtr->nop[0].reserved1,
                    ccoreWriteValueSqePtr->nop[0].immd);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] nop[1]: opCode[%u] rd[%u] reserved0[%u] func3[%u] rs1[%u] "
                    "reserved1[%u] immd[%u]",
                    ccoreWriteValueSqePtr->nop[1].opCode, ccoreWriteValueSqePtr->nop[1].rd,
                    ccoreWriteValueSqePtr->nop[1].reserved0, ccoreWriteValueSqePtr->nop[1].func3,
                    ccoreWriteValueSqePtr->nop[1].rs1, ccoreWriteValueSqePtr->nop[1].reserved1,
                    ccoreWriteValueSqePtr->nop[1].immd);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] nop[2]: opCode[%u] rd[%u] reserved0[%u] func3[%u] rs1[%u] "
                    "reserved1[%u] immd[%u]",
                    ccoreWriteValueSqePtr->nop[2].opCode, ccoreWriteValueSqePtr->nop[2].rd,
                    ccoreWriteValueSqePtr->nop[2].reserved0, ccoreWriteValueSqePtr->nop[2].func3,
                    ccoreWriteValueSqePtr->nop[2].rs1, ccoreWriteValueSqePtr->nop[2].reserved1,
                    ccoreWriteValueSqePtr->nop[2].immd);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] nop[3]: opCode[%u] rd[%u] reserved0[%u] func3[%u] rs1[%u] "
                    "reserved1[%u] immd[%u]",
                    ccoreWriteValueSqePtr->nop[3].opCode, ccoreWriteValueSqePtr->nop[3].rd,
                    ccoreWriteValueSqePtr->nop[3].reserved0, ccoreWriteValueSqePtr->nop[3].func3,
                    ccoreWriteValueSqePtr->nop[3].rs1, ccoreWriteValueSqePtr->nop[3].reserved1,
                    ccoreWriteValueSqePtr->nop[3].immd);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] nop[4]: opCode[%u] rd[%u] reserved0[%u] func3[%u] rs1[%u] "
                    "reserved1[%u] immd[%u]",
                    ccoreWriteValueSqePtr->nop[4].opCode, ccoreWriteValueSqePtr->nop[4].rd,
                    ccoreWriteValueSqePtr->nop[4].reserved0, ccoreWriteValueSqePtr->nop[4].func3,
                    ccoreWriteValueSqePtr->nop[4].rs1, ccoreWriteValueSqePtr->nop[4].reserved1,
                    ccoreWriteValueSqePtr->nop[4].immd);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] nop[5]: opCode[%u] rd[%u] reserved0[%u] func3[%u] rs1[%u] "
                    "reserved1[%u] immd[%u]",
                    ccoreWriteValueSqePtr->nop[5].opCode, ccoreWriteValueSqePtr->nop[5].rd,
                    ccoreWriteValueSqePtr->nop[5].reserved0, ccoreWriteValueSqePtr->nop[5].func3,
                    ccoreWriteValueSqePtr->nop[5].rs1, ccoreWriteValueSqePtr->nop[5].reserved1,
                    ccoreWriteValueSqePtr->nop[5].immd);
                break;
            }
            case SqeType::NOTIFY_SQE_V2: {
                const rtStarsNotifySqeV2_t* notifyV2SqePtr = reinterpret_cast<const rtStarsNotifySqeV2_t*>(sqePtr);
                PLF_CONFIG_DEBUG(PLF_TASK, "[OpUnfoldCache][DumpSqeContent] rtStarsNotifySqeV2_t sqeType[%u]", sqeType);
                CHK_RET(DumpSqeHeader(notifyV2SqePtr->header));
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] notify_id[%u] res2[%u] res3[%u] kernel_credit[%u] res4[%u]",
                    notifyV2SqePtr->notify_id, notifyV2SqePtr->res2, notifyV2SqePtr->res3,
                    notifyV2SqePtr->kernel_credit, notifyV2SqePtr->res);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] res[0][%u] res[1][%u] res[2][%u] res[3][%u] res[4][%u] res[5][%u]",
                    notifyV2SqePtr->res[0], notifyV2SqePtr->res[1], notifyV2SqePtr->res[2], notifyV2SqePtr->res[3],
                    notifyV2SqePtr->res[4], notifyV2SqePtr->res[5]);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] res[6][%u] res[7][%u] res[8][%u] res[9][%u] res[10][%u] "
                    "res[11][%u]",
                    notifyV2SqePtr->res[6], notifyV2SqePtr->res[7], notifyV2SqePtr->res[8], notifyV2SqePtr->res[9],
                    notifyV2SqePtr->res[10], notifyV2SqePtr->res[11]);
                break;
            }
            case SqeType::WRITE_VALUE_SQE_V2: {
                const rtStarsWriteValueSqeV2_t* writeValueV2SqePtr
                    = reinterpret_cast<const rtStarsWriteValueSqeV2_t*>(sqePtr);
                PLF_CONFIG_DEBUG(
                    PLF_TASK, "[OpUnfoldCache][DumpSqeContent] rtStarsWriteValueSqeV2_t sqeType[%u]", sqeType);
                CHK_RET(DumpSqeHeader(writeValueV2SqePtr->header));
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] reg_addr_low[0x%08x] reg_addr_high[0x%08x] awsize[%u] snoop[%u] "
                    "res2[%u] awcache[%u] awprot[%u] VA[%u]",
                    writeValueV2SqePtr->reg_addr_low, writeValueV2SqePtr->reg_addr_high, writeValueV2SqePtr->awsize,
                    writeValueV2SqePtr->snoop, writeValueV2SqePtr->res2, writeValueV2SqePtr->awcache,
                    writeValueV2SqePtr->awprot, writeValueV2SqePtr->VA);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] write_val[0][%u] write_val[1][%u] write_val[2][%u] "
                    "write_val[3][%u] write_val[4][%u] write_val[5][%u] write_val[6][%u] write_val[7][%u]",
                    writeValueV2SqePtr->write_val[0], writeValueV2SqePtr->write_val[1],
                    writeValueV2SqePtr->write_val[2], writeValueV2SqePtr->write_val[3],
                    writeValueV2SqePtr->write_val[4], writeValueV2SqePtr->write_val[5],
                    writeValueV2SqePtr->write_val[6], writeValueV2SqePtr->write_val[7]);
                PLF_CONFIG_DEBUG(
                    PLF_TASK, "[OpUnfoldCache][DumpSqeContent] res[0][%u] res[1][%u] res[2][%u] res[3][%u]",
                    writeValueV2SqePtr->res[0], writeValueV2SqePtr->res[1], writeValueV2SqePtr->res[2],
                    writeValueV2SqePtr->res[3]);
                break;
            }
            case SqeType::EVENT_SQE_V2: {
                const rtStarsEventSqeV2_t* eventV2SqePtr = reinterpret_cast<const rtStarsEventSqeV2_t*>(sqePtr);
                PLF_CONFIG_DEBUG(PLF_TASK, "[OpUnfoldCache][DumpSqeContent] rtStarsEventSqeV2_t sqeType[%u]", sqeType);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] type[%u] ie[%u] pre_p[%u] post_p[%u] wr_cqe[%u] res0[%u] flag[%u] "
                    "rt_stream_id[%u] task_id[%u] event_id[%u]",
                    eventV2SqePtr->type, eventV2SqePtr->ie, eventV2SqePtr->pre_p, eventV2SqePtr->post_p,
                    eventV2SqePtr->wr_cqe, eventV2SqePtr->res0, eventV2SqePtr->flag, eventV2SqePtr->rt_stream_id,
                    eventV2SqePtr->task_id, eventV2SqePtr->event_id);
                PLF_CONFIG_DEBUG(
                    PLF_TASK, "[OpUnfoldCache][DumpSqeContent] res2[%u] res3[%u] kernel_credit[%u] res4[%u] offset[%u]",
                    eventV2SqePtr->res2, eventV2SqePtr->res3, eventV2SqePtr->kernel_credit, eventV2SqePtr->res4,
                    eventV2SqePtr->offset);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] res6[0][%u] res6[1][%u] res6[2][%u] res6[3][%u] res6[4][%u] "
                    "res6[5][%u] res6[6][%u] res6[7][%u] res6[8][%u] res6[9][%u] res6[10][%u] res6[11][%u]",
                    eventV2SqePtr->res6[0], eventV2SqePtr->res6[1], eventV2SqePtr->res6[2], eventV2SqePtr->res6[3],
                    eventV2SqePtr->res6[4], eventV2SqePtr->res6[5], eventV2SqePtr->res6[6], eventV2SqePtr->res6[7],
                    eventV2SqePtr->res6[8], eventV2SqePtr->res6[9], eventV2SqePtr->res6[10], eventV2SqePtr->res6[11]);
                break;
            }
            case SqeType::MEMCPY_ASYNC_SQE_V2: {
                const rtStarsMemcpyAsyncSqeV2_t* memcpyAsyncV2SqePtr
                    = reinterpret_cast<const rtStarsMemcpyAsyncSqeV2_t*>(sqePtr);
                PLF_CONFIG_DEBUG(
                    PLF_TASK, "[OpUnfoldCache][DumpSqeContent] rtStarsMemcpyAsyncSqeV2_t sqeType[%u]", sqeType);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] type[%u] ie[%u] pre_p[%u] post_p[%u] wr_cqe[%u] res0[%u] "
                    "l2_lock[%u] l2_unlock[%u] res1[%u] rt_stream_id[%u] task_id[%u]",
                    memcpyAsyncV2SqePtr->type, memcpyAsyncV2SqePtr->ie, memcpyAsyncV2SqePtr->pre_p,
                    memcpyAsyncV2SqePtr->post_p, memcpyAsyncV2SqePtr->wr_cqe, memcpyAsyncV2SqePtr->res0,
                    memcpyAsyncV2SqePtr->l2_lock, memcpyAsyncV2SqePtr->l2_unlock, memcpyAsyncV2SqePtr->res1,
                    memcpyAsyncV2SqePtr->rt_stream_id, memcpyAsyncV2SqePtr->task_id);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] res2[%u] res3[%u] kernel_credit[%u] res4[%u] opcode[%u] "
                    "ie_dma[%u] sssv[%u] dssv[%u] sns[%u] dns[%u] qos[%u] sro[%u] dro[%u] overflow_en[%u] res5[%u]",
                    memcpyAsyncV2SqePtr->res2, memcpyAsyncV2SqePtr->res3, memcpyAsyncV2SqePtr->kernel_credit,
                    memcpyAsyncV2SqePtr->res4, memcpyAsyncV2SqePtr->opcode, memcpyAsyncV2SqePtr->ie_dma,
                    memcpyAsyncV2SqePtr->sssv, memcpyAsyncV2SqePtr->dssv, memcpyAsyncV2SqePtr->sns,
                    memcpyAsyncV2SqePtr->dns, memcpyAsyncV2SqePtr->qos, memcpyAsyncV2SqePtr->sro,
                    memcpyAsyncV2SqePtr->dro, memcpyAsyncV2SqePtr->overflow_en, memcpyAsyncV2SqePtr->res5);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] src_streamid[%u] src_substreamid[%u] dst_streamid[%u] "
                    "dst_substreamid[%u] length[%u]",
                    memcpyAsyncV2SqePtr->src_streamid, memcpyAsyncV2SqePtr->src_substreamid,
                    memcpyAsyncV2SqePtr->dst_streamid, memcpyAsyncV2SqePtr->dst_substreamid,
                    memcpyAsyncV2SqePtr->length);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] src_addr_low[0x%08x] src_addr_high[0x%08x] dst_addr_low[0x%08x] "
                    "dst_addr_high[0x%08x] overflow_addr_low[0x%08x] overflow_addr_high[0x%08x]",
                    memcpyAsyncV2SqePtr->src_addr_low, memcpyAsyncV2SqePtr->src_addr_high,
                    memcpyAsyncV2SqePtr->dst_addr_low, memcpyAsyncV2SqePtr->dst_addr_high,
                    memcpyAsyncV2SqePtr->overflow_addr_low, memcpyAsyncV2SqePtr->overflow_addr_high);
                break;
            }
            case SqeType::FLIP_PLACEHOLDER_SQE: {
                const rtStarsPlaceHolderSqe_t* placeholderSqePtr
                    = reinterpret_cast<const rtStarsPlaceHolderSqe_t*>(sqePtr);
                PLF_CONFIG_DEBUG(
                    PLF_TASK, "[OpUnfoldCache][DumpSqeContent] rtStarsPlaceHolderSqe_t sqeType[%u]", sqeType);
                CHK_RET(DumpSqeHeader(placeholderSqePtr->header));
                PLF_CONFIG_DEBUG(
                    PLF_TASK, "[OpUnfoldCache][DumpSqeContent] res1[%u] res2[%u] kernel_credit[%u] res3[%u]",
                    placeholderSqePtr->res1, placeholderSqePtr->res2, placeholderSqePtr->kernel_credit,
                    placeholderSqePtr->res3);
                PLF_CONFIG_DEBUG(
                    PLF_TASK, "[OpUnfoldCache][DumpSqeContent] u.flip_task_info: flipNumReport[%u]",
                    placeholderSqePtr->u.flip_task_info.flipNumReport);
                // 注意: 无需打印u.flip_task_info中的reserved数组
                // 注意: 无需打印union中的resv数组
                break;
            }
            case SqeType::CACHE_MEMCPY_PLACEHOLDER_SQE: {
                const rtStarsPlaceHolderSqe_t* placeholderSqePtr
                    = reinterpret_cast<const rtStarsPlaceHolderSqe_t*>(sqePtr);
                PLF_CONFIG_DEBUG(
                    PLF_TASK, "[OpUnfoldCache][DumpSqeContent] rtStarsPlaceHolderSqe_t sqeType[%u]", sqeType);
                CHK_RET(DumpSqeHeader(placeholderSqePtr->header));
                PLF_CONFIG_DEBUG(
                    PLF_TASK, "[OpUnfoldCache][DumpSqeContent] res1[%u] res2[%u] kernel_credit[%u] res3[%u]",
                    placeholderSqePtr->res1, placeholderSqePtr->res2, placeholderSqePtr->kernel_credit,
                    placeholderSqePtr->res3);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] u.cache_memcpy_task_info: src_addr_low[0x%08x] "
                    "src_addr_high[0x%08x] dst_addr_low[0x%08x] dst_addr_high[0x%08x]",
                    placeholderSqePtr->u.cache_memcpy_task_info.src_addr_low,
                    placeholderSqePtr->u.cache_memcpy_task_info.src_addr_high,
                    placeholderSqePtr->u.cache_memcpy_task_info.dst_addr_low,
                    placeholderSqePtr->u.cache_memcpy_task_info.dst_addr_high);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] u.cache_memcpy_task_info: kernel_credit[%u] linkType[%u]",
                    placeholderSqePtr->u.cache_memcpy_task_info.kernel_credit,
                    placeholderSqePtr->u.cache_memcpy_task_info.linkType);
                // 注意: 无需打印u.cache_memcpy_task_info中的reserved数组
                // 注意: 无需打印union中的resv数组
                break;
            }
            case SqeType::CACHE_NOTIFY_PLACEHOLDER_SQE: {
                const rtStarsPlaceHolderSqe_t* placeholderSqePtr
                    = reinterpret_cast<const rtStarsPlaceHolderSqe_t*>(sqePtr);
                PLF_CONFIG_DEBUG(
                    PLF_TASK, "[OpUnfoldCache][DumpSqeContent] rtStarsPlaceHolderSqe_t sqeType[%u]", sqeType);
                CHK_RET(DumpSqeHeader(placeholderSqePtr->header));
                PLF_CONFIG_DEBUG(
                    PLF_TASK, "[OpUnfoldCache][DumpSqeContent] res1[%u] res2[%u] kernel_credit[%u] res3[%u]",
                    placeholderSqePtr->res1, placeholderSqePtr->res2, placeholderSqePtr->kernel_credit,
                    placeholderSqePtr->res3);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] u.cache_notify_task_info: is_wait[%u] kernel_credit[%u] "
                    "timeout[%u] notify_id[%u]",
                    placeholderSqePtr->u.cache_notify_task_info.is_wait,
                    placeholderSqePtr->u.cache_notify_task_info.kernel_credit,
                    placeholderSqePtr->u.cache_notify_task_info.timeout,
                    placeholderSqePtr->u.cache_notify_task_info.notify_id);
                // 注意: 无需打印u.cache_notify_task_info中的reserved数组
                // 注意: 无需打印union中的resv数组
                break;
            }
            case SqeType::CACHE_WRITE_VALUE_PLACEHOLDER_SQE: {
                const rtStarsPlaceHolderSqe_t* placeholderSqePtr
                    = reinterpret_cast<const rtStarsPlaceHolderSqe_t*>(sqePtr);
                PLF_CONFIG_DEBUG(
                    PLF_TASK, "[OpUnfoldCache][DumpSqeContent] rtStarsPlaceHolderSqe_t sqeType[%u]", sqeType);
                CHK_RET(DumpSqeHeader(placeholderSqePtr->header));
                PLF_CONFIG_DEBUG(
                    PLF_TASK, "[OpUnfoldCache][DumpSqeContent] res1[%u] res2[%u] kernel_credit[%u] res3[%u]",
                    placeholderSqePtr->res1, placeholderSqePtr->res2, placeholderSqePtr->kernel_credit,
                    placeholderSqePtr->res3);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] u.cache_write_value_task_info: write_addr_low[0x%08x] "
                    "write_addr_high[0x%08x]",
                    placeholderSqePtr->u.cache_write_value_task_info.write_addr_low,
                    placeholderSqePtr->u.cache_write_value_task_info.write_addr_high);
                // 注意: 无需打印u.cache_write_value_task_info中的reserved数组
                // 注意: 无需打印union中的resv数组
                break;
            }
            case SqeType::CACHE_MEMCPY_RECORD_PLACEHOLDER_SQE: {
                const rtStarsPlaceHolderSqe_t* placeholderSqePtr
                    = reinterpret_cast<const rtStarsPlaceHolderSqe_t*>(sqePtr);
                PLF_CONFIG_DEBUG(
                    PLF_TASK, "[OpUnfoldCache][DumpSqeContent] rtStarsPlaceHolderSqe_t sqeType[%u]", sqeType);
                CHK_RET(DumpSqeHeader(placeholderSqePtr->header));
                PLF_CONFIG_DEBUG(
                    PLF_TASK, "[OpUnfoldCache][DumpSqeContent] res1[%u] res2[%u] kernel_credit[%u] res3[%u]",
                    placeholderSqePtr->res1, placeholderSqePtr->res2, placeholderSqePtr->kernel_credit,
                    placeholderSqePtr->res3);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] u.cache_memcpy_record_task_info: length[%u] "
                    "src_addr_low[0x%08x] src_addr_high[0x%08x] dst_addr_low[0x%08x] dst_addr_high[0x%08x]",
                    placeholderSqePtr->u.cache_memcpy_record_task_info.length,
                    placeholderSqePtr->u.cache_memcpy_record_task_info.src_addr_low,
                    placeholderSqePtr->u.cache_memcpy_record_task_info.src_addr_high,
                    placeholderSqePtr->u.cache_memcpy_record_task_info.dst_addr_low,
                    placeholderSqePtr->u.cache_memcpy_record_task_info.dst_addr_high);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[OpUnfoldCache][DumpSqeContent] u.cache_memcpy_record_task_info: opcode[%u] "
                    "partid[%u] kernel_credit[%u] linkType[%u]",
                    placeholderSqePtr->u.cache_memcpy_record_task_info.opcode,
                    placeholderSqePtr->u.cache_memcpy_record_task_info.partid,
                    placeholderSqePtr->u.cache_memcpy_record_task_info.kernel_credit,
                    placeholderSqePtr->u.cache_memcpy_record_task_info.linkType);
                // 注意: 无需打印u.cache_memcpy_record_task_info中的reserved数组
                // 注意: 无需打印union中的resv数组
                break;
            }
            default: {
                HCCL_WARNING("[OpUnfoldCache][DumpSqeContent] sqeType %u is unsupported", sqeType);
                break; // This function only for debug, no need to return HCCL_E_NOT_SUPPORT (否则LLT会失败)
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult OpUnfoldCache::DumpSqeHeader(const rtStarsSqeHeader_t& sqeHeader)
{
    PLF_CONFIG_DEBUG(
        PLF_TASK,
        "[OpUnfoldCache][DumpSqeHeader] SQE header: type[%u] l1Lock[%u] l1Unlock[%u] ie[%u] preP[%u] postP[%u] "
        "wrCqe[%u]",
        sqeHeader.type, sqeHeader.l1Lock, sqeHeader.l1Unlock, sqeHeader.ie, sqeHeader.preP, sqeHeader.postP,
        sqeHeader.wrCqe);
    PLF_CONFIG_DEBUG(
        PLF_TASK, "[OpUnfoldCache][DumpSqeHeader] SQE header: reserved[%u] numBlocks[%u] rtStreamId[%u] taskId[%u]",
        sqeHeader.reserved, sqeHeader.blockDim, sqeHeader.rtStreamId, sqeHeader.taskId);

    return HCCL_SUCCESS;
}

HcclResult OpUnfoldCache::DumpSqeHeader(const rtStarsSqeHeaderV2_t& sqeHeader)
{
    PLF_CONFIG_DEBUG(
        PLF_TASK,
        "[OpUnfoldCache][DumpSqeHeader] SQE header V2: type[%u] graph_lock[%u] graph_unlock[%u] ie[%u] pre_p[%u] "
        "post_p[%u] wr_cqe[%u]",
        sqeHeader.type, sqeHeader.graph_lock, sqeHeader.graph_unlock, sqeHeader.ie, sqeHeader.pre_p, sqeHeader.post_p,
        sqeHeader.wr_cqe);
    PLF_CONFIG_DEBUG(
        PLF_TASK,
        "[OpUnfoldCache][DumpSqeHeader] SQE header V2: res0[%u] l2_lock[%u] l2_unlock[%u] res1[%u] rt_stream_id[%u] "
        "task_id[%u]",
        sqeHeader.res0, sqeHeader.l2_lock, sqeHeader.l2_unlock, sqeHeader.res1, sqeHeader.rt_stream_id,
        sqeHeader.task_id);

    return HCCL_SUCCESS;
}
} // namespace hccl
