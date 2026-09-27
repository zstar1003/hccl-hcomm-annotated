/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "sqe_mgr.h"
#include <chrono>
#include "stl_util.h"
#include "exception_util.h"
#include "null_ptr_exception.h"
#include "internal_exception.h"
#include "string_util.h"
#include "drv_api_exception.h"

namespace Hccl {

constexpr u64 COMMIT_TIMEOUT = 30;

HcclResult SqeMgr::Begin(u32 sqId)
{
    HCCL_INFO("SqeMgr::%s start, sqId %u", __func__, sqId);
    if (Contain(sqInfos, sqId)) {
        if (sqInfos[sqId]->sqeCnt != 0) {
            // called Begin twice in the same round
            HCCL_ERROR("sqInfos[sqId]->sqeCnt != 0");
            return HcclResult::HCCL_E_INTERNAL;
        }
        return HcclResult::HCCL_SUCCESS;
    }
    std::unique_ptr<SqInfo> sqInfo = std::make_unique<SqInfo>();
    sqInfo->sqeCnt = 0;
    sqInfo->sqDepth = QuerySqDepth(sqId);
    sqInfo->sqTail = QuerySqTail(sqId);
    sqInfo->sqHead = QuerySqHead(sqId);
    sqInfo->sqBaseAddr = QuerySqBaseAddr(sqId);
    s32 ret = memset_s(sqInfo->sqeBuffer, sizeof(sqInfo->sqeBuffer), 0, sizeof(sqInfo->sqeBuffer));
    if (ret != EOK) {
        std::string formatStr = StringFormat("SqeMgr::%s memset_s failed. ret[%d]", __func__, ret);
        HCCL_ERROR("%s", formatStr.c_str());
        THROW<InternalException>(formatStr);
    }

    sqInfos[sqId] = std::move(sqInfo);
    if (sqInfos[sqId] == nullptr) {
        HCCL_ERROR("SqeMgr::Begin sqInfos[sqId] == nullptr, sq_id=%u", sqId);
        return HcclResult::HCCL_E_INTERNAL;
    }
    HCCL_INFO("SqeMgr::%s end", __func__);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult SqeMgr::Add(u32 sqId, HcclSqe* sqe)
{
    if (sqe == nullptr) {
        HCCL_ERROR("SqeMgr::Add HcclSqe *sqe is nullptr");
        return HcclResult::HCCL_E_PTR;
    }
    HCCL_INFO("SqeMgr::%s start sq_id=%u", __func__, sqId);
    if (!Contain(sqInfos, sqId)) {
        HCCL_ERROR("no available buffer found for sq_id=%u", sqId);
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    SqInfo* sqInfo = sqInfos[sqId].get();
    if (sqInfo == nullptr) {
        HCCL_ERROR("SqeMgr::add failed sqInfo is null sq_id=%u", sqId);
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    void* nextBufferAddr = static_cast<u8*>(sqInfo->sqeBuffer) + sqInfo->sqeCnt * AC_SQE_SIZE;

    HCCL_INFO("SqeMgr::%s sqe->GetSqe() %llu", __func__, sqe->GetSqe());

    AddSqeToBuffer(nextBufferAddr, ReinterpretAs<void*>(sqe->GetSqe()));
    sqInfo->sqeCnt++;
    HCCL_INFO("SqeMgr::%s end sqInfo->sqeCnt[%u]", __func__, sqInfo->sqeCnt);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult SqeMgr::Commit(u32 sqId)
{
    HCCL_INFO("SqeMgr::%s start sq_id=%u", __func__, sqId);
    if (!Contain(sqInfos, sqId)) {
        HCCL_ERROR("no available buffer found for sq_id=%u", sqId);
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    SqInfo* sqInfo = sqInfos[sqId].get();
    CHECK_NULLPTR(sqInfo, "[SqeMgr::Commit] sqInfo is nullptr!");
    u32 availableSpace = GetTailToHeadDist(sqId, sqInfo->sqHead, sqInfo->sqTail);
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(COMMIT_TIMEOUT);
    HCCL_INFO("SqeMgr::%s start", __func__);
    while (availableSpace <= sqInfo->sqeCnt) {
        HCCL_INFO(
            "SqeMgr::%s while loop availableSpace %u <= sqInfo->sqeCnt %u", __func__, availableSpace, sqInfo->sqeCnt);
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            HCCL_ERROR("Rtsq full, timeout 30s. cur head: %u, sqId: %u", sqInfo->sqHead, sqId);
            return HcclResult::HCCL_E_TIMEOUT;
        }
        sqInfo->sqHead = QuerySqHead(sqId);
        availableSpace = GetTailToHeadDist(sqId, sqInfo->sqHead, sqInfo->sqTail);
    }
    u32 depthLeft = sqInfo->sqDepth - sqInfo->sqTail;
    if (sqInfo->sqeCnt <= depthLeft) { // 没有回绕
        HCCL_INFO(
            "SqeMgr::%s copy sqe from sqe buffer, cur tail: %u, size: %u", __func__, sqInfo->sqTail, sqInfo->sqeCnt);
        int ret = memcpy_s(
            ReinterpretAs<u8*>(sqInfo->sqBaseAddr) + sqInfo->sqTail * AC_SQE_SIZE, sqInfo->sqeCnt * AC_SQE_SIZE,
            sqInfo->sqeBuffer, sqInfo->sqeCnt * AC_SQE_SIZE);
        if (ret != 0) {
            THROW<InternalException>(StringFormat("SqeMgr::%s sqe memcpy_s failed, ret = %d", __func__, ret));
        }
    } else {
        HCCL_INFO(
            "SqeMgr::%s copy sqe twice, cnt: %u, tail: %u, depth remain: %u", __func__, sqInfo->sqeCnt, sqInfo->sqTail,
            depthLeft);
        // 先拷贝rtsq里剩余空间大小
        int ret = memcpy_s(
            ReinterpretAs<u8*>(sqInfo->sqBaseAddr) + sqInfo->sqTail * AC_SQE_SIZE, depthLeft * AC_SQE_SIZE,
            sqInfo->sqeBuffer, depthLeft * AC_SQE_SIZE);
        if (ret != 0) {
            THROW<InternalException>(
                StringFormat("SqeMgr::%s rtsq remaining space memcpy_s failed, ret = %d", __func__, ret));
        }
        // 拷贝剩余sqe
        ret = memcpy_s(
            ReinterpretAs<u8*>(sqInfo->sqBaseAddr), sqInfo->sqHead * AC_SQE_SIZE,
            sqInfo->sqeBuffer + depthLeft * AC_SQE_SIZE, (sqInfo->sqeCnt - depthLeft) * AC_SQE_SIZE);
        if (ret != 0) {
            THROW<InternalException>(StringFormat("SqeMgr::%s remaining sqe memcpy_s failed, ret = %d", __func__, ret));
        }
    }

    // set new tail position
    u32 newTail = static_cast<u32>((static_cast<u64>(sqInfo->sqTail) + sqInfo->sqeCnt) % sqInfo->sqDepth);
    ConfigSqTail(sqId, newTail);
    sqInfo->sqTail = newTail;
    // clear sqe buffer
    auto sRet = memset_s(sqInfo->sqeBuffer, sizeof(sqInfo->sqeBuffer), 0, sqInfo->sqeCnt * AC_SQE_SIZE);
    if (sRet != 0) {
        THROW<InternalException>(StringFormat("SqeMgr::%s remaining sqe memcpy_s failed, ret = %d", __func__, sRet));
    }
    sqInfo->sqeCnt = 0;
    HCCL_INFO("SqeMgr::%s end", __func__);
    return HcclResult::HCCL_SUCCESS;
}

u32 SqeMgr::QuerySqHead(u32 sqId)
{
    HCCL_INFO("SqeMgr::%s", __func__);
    return QuerySqStatusByType(sqId, DRV_SQCQ_PROP_SQ_HEAD);
}

u32 SqeMgr::QuerySqTail(u32 sqId)
{
    HCCL_INFO("SqeMgr::%s", __func__);
    return QuerySqStatusByType(sqId, DRV_SQCQ_PROP_SQ_TAIL);
}

u32 SqeMgr::QuerySqDepth(u32 sqId)
{
    HCCL_INFO("SqeMgr::%s", __func__);
    return QuerySqStatusByType(sqId, DRV_SQCQ_PROP_SQ_DEPTH);
}

u64 SqeMgr::QuerySqBaseAddr(u32 sqId)
{
    halSqCqQueryInfo queryInfo;
    queryInfo.tsId = 0;
    queryInfo.sqId = sqId;
    queryInfo.cqId = 0;
    queryInfo.type = DRV_NORMAL_TYPE;
    queryInfo.prop = DRV_SQCQ_PROP_SQ_BASE;

    HCCL_INFO("SqeMgr::%s begin", __func__);
    drvError_t ret = halSqCqQuery(devPhyId, &queryInfo);
    if (ret != 0) {
        std::string formatStr
            = StringFormat("SqeMgr::%s call halSqCqQuery failed, devPhyId %d, ret %d", __func__, devPhyId, ret);
        HCCL_ERROR("%s", formatStr.c_str());
        THROW<DrvApiException>(formatStr);
    }
    HCCL_INFO("SqeMgr::%s end", __func__);

    return ((static_cast<u64>(queryInfo.value[1])) << UINT32_BIT_NUM) | queryInfo.value[0];
}

u32 SqeMgr::QuerySqStatusByType(u32 sqId, drvSqCqPropType_t type) const
{
    halSqCqQueryInfo queryInfo;
    queryInfo.tsId = 0;
    queryInfo.sqId = sqId;
    queryInfo.cqId = 0;
    queryInfo.type = DRV_NORMAL_TYPE;
    queryInfo.prop = type;
    HCCL_INFO("%s::halSqCqQuery begin, type %d sqId %u", __func__, type, sqId);
    drvError_t ret = halSqCqQuery(devPhyId, &queryInfo);
    if (ret != 0) {
        std::string formatStr
            = StringFormat("SqeMgr::%s call halSqCqQuery failed, devPhyId %d, ret %d", __func__, devPhyId, ret);
        HCCL_ERROR("%s", formatStr.c_str());
        THROW<DrvApiException>(formatStr);
    }
    HCCL_INFO("%s::halSqCqQuery end", __func__);

    return queryInfo.value[0];
}

void SqeMgr::ConfigSqTail(u32 sqId, u32 value)
{
    HCCL_INFO("SqeMgr::%s sqId %u, value %u", __func__, sqId, value);
    ConfigSqStatusByType(sqId, DRV_SQCQ_PROP_SQ_TAIL, value);
}

void SqeMgr::ConfigSqStatusByType(u32 sqId, drvSqCqPropType_t type, u32 value) const
{
    halSqCqConfigInfo configInfo;
    configInfo.tsId = 0;
    configInfo.sqId = sqId;
    configInfo.cqId = 0;
    configInfo.type = DRV_NORMAL_TYPE;
    configInfo.prop = type;
    configInfo.value[0] = value;

    HCCL_INFO("SqeMgr::%s start", __func__);
    drvError_t ret = halSqCqConfig(devPhyId, &configInfo);
    if (ret != 0) {
        HCCL_ERROR("call halSqCqConfig failed, ret %d", ret);
    }
    HCCL_INFO("SqeMgr::%s end", __func__);
}

u32 SqeMgr::GetTailToHeadDist(u32 sqId, u32 head, u32 tail)
{
    if (head == tail) {
        return sqInfos[sqId]->sqDepth;
    }
    return (tail < head) ? head - tail : sqInfos[sqId]->sqDepth - (tail - head);
}

void SqeMgr::AddSqeToBuffer(void* bufferAddr, void* sqeAddr) const
{
    if (bufferAddr == nullptr || sqeAddr == nullptr) {
        std::string formatStr = StringFormat("SqeMgr::%s bufferAddr[%u], sqeAddr[%u]", __func__, bufferAddr, sqeAddr);
        HCCL_ERROR("%s", formatStr.c_str());
        THROW<NullPtrException>(formatStr);
    }
    s32 ret = memcpy_s(bufferAddr, AC_SQE_SIZE, sqeAddr, AC_SQE_SIZE);
    if (ret != 0) {
        std::string formatStr = StringFormat("SqeMgr::%s memcpy_s failed, ret = %d", __func__, ret);
        HCCL_ERROR("%s", formatStr.c_str());
        THROW<InternalException>(formatStr);
    }
}

SqeMgr::SqeMgr(u32 devPhysicalId) : devPhyId(devPhysicalId) {}

} // namespace Hccl
