/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rtsq_base.h"
#include "log.h"
#include "drv_api_exception.h"
#include "exception_util.h"
#include "internal_exception.h"
#include "sqe_v82.h"
#include <unordered_map>
namespace Hccl {
RtsqBase::RtsqBase(u32 devPhyId, u32 streamId, u32 sqId) : devPhyId_(devPhyId), streamId_(streamId), sqId_(sqId)
{
    auto ret = drvGetLocalDevIDByHostDevID(devPhyId_, &localDevId_);
    if (ret != DRV_ERROR_NONE) {
        std::string formatStr = StringFormat(
            "RtsqBase::%s call drvGetLocalDevIDByHostDevID failed, devPhyId %u, ret %d", __func__, devPhyId_, ret);
        THROW<DrvApiException>(formatStr);
    }

    sqHead_ = QuerySqHead();
    sqTail_ = QuerySqTail();
    sqDepth_ = QuerySqDepth();
    sqBaseAddr_ = QuerySqBaseAddr();

    if (sqDepth_ == 0) {
        THROW<InternalException>("sqDepth_ cannot be zero.");
    }
    HCCL_INFO("%s, %s", __func__, GetHwSqDescribe().c_str());
}

void RtsqBase::Reset(bool reset)
{
    sqHead_ = QuerySqHead();
    sqTail_ = QuerySqTail();

    if (reset) {
        CHK_PRT_CONT(
            sqHead_ != 0 || sqTail_ != 0,
            HCCL_ERROR("RtsqBase::%s, sqHead_=%u, sqTail_=%u", __func__, sqHead_, sqTail_));
    }

    sqDepth_ = QuerySqDepth();
    sqBaseAddr_ = QuerySqBaseAddr();
    SetTaskIdBySqeId();
    HCCL_INFO("%s, %s", __func__, GetHwSqDescribe().c_str());
}

std::string RtsqBase::GetHwSqDescribe() const
{
    return StringFormat(
        "devPhyId=%u, localDevId=%u, streamId=%u, sqId=%u, sqDepth=%u, sqBaseAddr=0x%llx, "
        "currentHead=%u, currentTail=%u, cqeStatus=%u, taskId=%u",
        devPhyId_, localDevId_, streamId_, sqId_, sqDepth_, sqBaseAddr_, QuerySqHead(), QuerySqTail(), QueryCqeStatus(),
        taskId_);
}

u32 RtsqBase::QuerySqStatusByType(drvSqCqPropType_t givenType) const
{
    halSqCqQueryInfo queryInfo;

    queryInfo.tsId = 0;
    queryInfo.sqId = sqId_;
    queryInfo.cqId = 0;
    queryInfo.type = DRV_NORMAL_TYPE;
    queryInfo.prop = givenType;
    drvError_t ret = halSqCqQuery(localDevId_, &queryInfo);
    if (ret != 0) {
        std::string formatStr = StringFormat(
            "RtsqBase::%s call halSqCqQuery failed, localDevId %u, ret %d, givenType=%u", __func__, localDevId_, ret,
            givenType);
        THROW<DrvApiException>(formatStr);
    }

    return queryInfo.value[0];
}

u64 RtsqBase::QuerySqBaseAddr() const
{
    halSqCqQueryInfo queryInfo;
    queryInfo.tsId = 0;
    queryInfo.sqId = sqId_;
    queryInfo.cqId = 0;
    queryInfo.type = DRV_NORMAL_TYPE;
    queryInfo.prop = DRV_SQCQ_PROP_SQ_BASE;
    drvError_t ret = halSqCqQuery(localDevId_, &queryInfo);
    if (ret != 0) {
        std::string formatStr
            = StringFormat("RtsqBase::%s call halSqCqQuery failed, localDevId %u, ret %d", __func__, localDevId_, ret);
        THROW<DrvApiException>(formatStr);
    }
    HCCL_INFO("RtsqBase::%s end", __func__);

    // 参照 driver API，BaseAddress为64bit，由两个32bit拼接而成，高32bit为 value[1], 低32bit为value[0]
    return ((static_cast<u64>(queryInfo.value[1])) << 32) | queryInfo.value[0];
}

u32 RtsqBase::QuerySqHead() const { return QuerySqStatusByType(drvSqCqPropType_t::DRV_SQCQ_PROP_SQ_HEAD); }
u32 RtsqBase::QuerySqTail() const { return QuerySqStatusByType(drvSqCqPropType_t::DRV_SQCQ_PROP_SQ_TAIL); }
u32 RtsqBase::QuerySqDepth() const { return QuerySqStatusByType(drvSqCqPropType_t::DRV_SQCQ_PROP_SQ_DEPTH); }
u32 RtsqBase::QueryCqeStatus() const { return QuerySqStatusByType(drvSqCqPropType_t::DRV_SQCQ_PROP_SQ_CQE_STATUS); }

void RtsqBase::ConfigSqStatusByType(drvSqCqPropType_t givenType, u32 value) const
{
    halSqCqConfigInfo configInfo;
    configInfo.tsId = 0;
    configInfo.sqId = sqId_;
    configInfo.cqId = 0;
    configInfo.type = DRV_NORMAL_TYPE;
    configInfo.prop = givenType;
    configInfo.value[0] = value;

    drvError_t ret = halSqCqConfig(localDevId_, &configInfo);
    if (UNLIKELY(ret != 0)) {
        std::string formatStr
            = StringFormat("RtsqBase::%s call halSqCqConfig failed, localDevId %u, ret %d", __func__, localDevId_, ret);
        THROW<DrvApiException>(formatStr);
    }
}

void RtsqBase::ConfigSqTail(u32 value) const
{
    HCCL_INFO("RtsqBase::%s, value=%u", __func__, value);
    ConfigSqStatusByType(drvSqCqPropType_t::DRV_SQCQ_PROP_SQ_TAIL, value);
}
void RtsqBase::ConfigDisableToEnable(u32 value) const
{
    HCCL_INFO("RtsqBase::%s, value=%u", __func__, value);
    ConfigSqStatusByType(drvSqCqPropType_t::DRV_SQCQ_PROP_SQ_DISABLE_TO_ENABLE, value);
}

HcclResult RtsqBase::GetStreamIdAndTaskIdBySqIdx(u32 sqIdx, uint16_t& streamId, uint16_t& taskId) const
{
    if (sqBaseAddr_ == 0 || sqIdx >= sqDepth_) {
        HCCL_ERROR("[%s]fail, sqBaseAddr_[0x%llu], sqIdx[%u]", __func__, sqBaseAddr_, sqIdx);
        return HCCL_E_PARA;
    }

    Rt91095StarsNotifySqe* sqe = (Rt91095StarsNotifySqe*)(sqBaseAddr_ + sqIdx * RTSQ_SQE_SIZE);
    streamId = sqe->header.rtStreamId;
    taskId = sqe->header.taskId;
    HCCL_INFO("[%s]sqId:%u, streamId:%u, taskId:%u", __func__, sqId_, streamId, taskId);
    return HCCL_SUCCESS;
}
} // namespace Hccl
