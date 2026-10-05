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

// [中文导读] [AllReduce逐行 S65] RtsqBase::QuerySqStatusByType的接口声明：待查询/配置的SQ属性类型；这些参数属于本函数调用边界。
u32 RtsqBase::QuerySqStatusByType(drvSqCqPropType_t givenType) const
// [中文导读] [AllReduce逐行 S66] 进入RtsqBase::QuerySqStatusByType函数体：调用 halSqCqQuery 查询指定硬件 SQ 属性；失败抛异常。
{
    // [中文导读] [AllReduce逐行 S67] 准备驱动SQ/CQ查询参数与结果的局部存储/结构描述，初始化方式以本行声明为准。
    halSqCqQueryInfo queryInfo;

    // [中文导读] [AllReduce逐行 S69] 设置驱动SQ/CQ查询参数与结果的tsId字段为/按`0`。
    queryInfo.tsId = 0;
    // [中文导读] [AllReduce逐行 S70] 设置驱动SQ/CQ查询参数与结果的sqId字段为/按`sqId_`（硬件SQ编号）。
    queryInfo.sqId = sqId_;
    // [中文导读] [AllReduce逐行 S71] 设置驱动SQ/CQ查询参数与结果的cqId字段为/按`0`。
    queryInfo.cqId = 0;
    // [中文导读] [AllReduce逐行 S72] 设置驱动SQ/CQ查询参数与结果的type字段为/按`DRV_NORMAL_TYPE`。
    queryInfo.type = DRV_NORMAL_TYPE;
    // [中文导读] [AllReduce逐行 S73] 设置驱动SQ/CQ查询参数与结果的prop字段为/按`givenType`（待查询/配置的SQ属性类型）。
    queryInfo.prop = givenType;
    // [中文导读] [AllReduce逐行 S74] 设置当前调用状态为/按`halSqCqQuery(localDevId_, &queryInfo)`（驱动使用的本地设备编号、驱动SQ/CQ查询参数与结果）；外部驱动边界：读SQ/CQ属性，检查驱动错误。
    drvError_t ret = halSqCqQuery(localDevId_, &queryInfo);
    // [中文导读] [AllReduce逐行 S75] 仅当`(ret != 0)`（当前调用状态）成立时进入此分支。
    if (ret != 0) {
        // [中文导读] [AllReduce逐行 S76] 设置std::string formatStr为/按`StringFormat(`；组装带上下文的错误或状态文本。
        std::string formatStr = StringFormat(
            // [中文导读] [AllReduce逐行 S77] 为当前RtsqBase::QuerySqStatusByType诊断/异常表达式提供格式文本，将报告驱动使用的本地设备编号、当前调用状态；这一物理行没有数据搬运副作用。
            "RtsqBase::%s call halSqCqQuery failed, localDevId %u, ret %d, givenType=%u", __func__, localDevId_, ret,
            // [中文导读] [AllReduce逐行 S78] 为组装带上下文的错误或状态文本补入`givenType)`（待查询/配置的SQ属性类型）；本行是参数/结构化初始化续行。
            givenType);
        // [中文导读] [AllReduce逐行 S79] 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。
        THROW<DrvApiException>(formatStr);
    // [中文导读] [AllReduce逐行 S80] 结束`if (ret != 0)`（当前调用状态）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S82] 直接返回`queryInfo.value[0]`（驱动SQ/CQ查询参数与结果的value字段）；将当前查询结果/句柄交给调用者。
    return queryInfo.value[0];
// [中文导读] [AllReduce逐行 S83] 结束RtsqBase::QuerySqStatusByType函数体；控制流返回外层。
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

// [中文导读] [AllReduce逐行 S105] 本行定义并直接执行RtsqBase::QuerySqHead：查询硬件 SQ 当前头位置；组装并执行指定SQ属性的驱动查询。
u32 RtsqBase::QuerySqHead() const { return QuerySqStatusByType(drvSqCqPropType_t::DRV_SQCQ_PROP_SQ_HEAD); }
// [中文导读] [AllReduce逐行 S106] 本行定义并直接执行RtsqBase::QuerySqTail：查询硬件 SQ 当前尾位置；组装并执行指定SQ属性的驱动查询。
u32 RtsqBase::QuerySqTail() const { return QuerySqStatusByType(drvSqCqPropType_t::DRV_SQCQ_PROP_SQ_TAIL); }
u32 RtsqBase::QuerySqDepth() const { return QuerySqStatusByType(drvSqCqPropType_t::DRV_SQCQ_PROP_SQ_DEPTH); }
u32 RtsqBase::QueryCqeStatus() const { return QuerySqStatusByType(drvSqCqPropType_t::DRV_SQCQ_PROP_SQ_CQE_STATUS); }

// [中文导读] [AllReduce逐行 S110] RtsqBase::ConfigSqStatusByType的接口声明：待查询/配置的SQ属性类型、写给驱动的SQ属性值；这些参数属于本函数调用边界。
void RtsqBase::ConfigSqStatusByType(drvSqCqPropType_t givenType, u32 value) const
// [中文导读] [AllReduce逐行 S111] 进入RtsqBase::ConfigSqStatusByType函数体：构造驱动 SQ 属性配置请求并调用 halSqCqConfig；失败抛异常。
{
    // [中文导读] [AllReduce逐行 S112] 准备驱动SQ/CQ配置参数的局部存储/结构描述，初始化方式以本行声明为准。
    halSqCqConfigInfo configInfo;
    // [中文导读] [AllReduce逐行 S113] 设置驱动SQ/CQ配置参数的tsId字段为/按`0`。
    configInfo.tsId = 0;
    // [中文导读] [AllReduce逐行 S114] 设置驱动SQ/CQ配置参数的sqId字段为/按`sqId_`（硬件SQ编号）。
    configInfo.sqId = sqId_;
    // [中文导读] [AllReduce逐行 S115] 设置驱动SQ/CQ配置参数的cqId字段为/按`0`。
    configInfo.cqId = 0;
    // [中文导读] [AllReduce逐行 S116] 设置驱动SQ/CQ配置参数的type字段为/按`DRV_NORMAL_TYPE`。
    configInfo.type = DRV_NORMAL_TYPE;
    // [中文导读] [AllReduce逐行 S117] 设置驱动SQ/CQ配置参数的prop字段为/按`givenType`（待查询/配置的SQ属性类型）。
    configInfo.prop = givenType;
    // [中文导读] [AllReduce逐行 S118] 设置驱动SQ/CQ配置参数的value字段为/按`value`（写给驱动的SQ属性值）。
    configInfo.value[0] = value;

    // [中文导读] [AllReduce逐行 S120] 设置当前调用状态为/按`halSqCqConfig(localDevId_, &configInfo)`（驱动使用的本地设备编号、驱动SQ/CQ配置参数）；外部驱动边界：写SQ/CQ属性，检查驱动错误。
    drvError_t ret = halSqCqConfig(localDevId_, &configInfo);
    // [中文导读] [AllReduce逐行 S121] 仅当`(UNLIKELY(ret != 0))`（当前调用状态）成立时进入此分支。
    if (UNLIKELY(ret != 0)) {
        // [中文导读] [AllReduce逐行 S122] 准备`std::string formatStr`的局部存储/结构描述，初始化方式以本行声明为准。
        std::string formatStr
            // [中文导读] [AllReduce逐行 S123] 组装带上下文的错误或状态文本；传入/处理驱动使用的本地设备编号、当前调用状态。
            = StringFormat("RtsqBase::%s call halSqCqConfig failed, localDevId %u, ret %d", __func__, localDevId_, ret);
        // [中文导读] [AllReduce逐行 S124] 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。
        THROW<DrvApiException>(formatStr);
    // [中文导读] [AllReduce逐行 S125] 结束`if (UNLIKELY(ret != 0))`（当前调用状态）分支/循环；控制流返回外层。
    }
// [中文导读] [AllReduce逐行 S126] 结束RtsqBase::ConfigSqStatusByType函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S128] RtsqBase::ConfigSqTail的接口声明：写给驱动的SQ属性值；这些参数属于本函数调用边界。
void RtsqBase::ConfigSqTail(u32 value) const
// [中文导读] [AllReduce逐行 S129] 进入RtsqBase::ConfigSqTail函数体：把新 SQ 尾通过驱动配置包装写入硬件。
{
    // [中文导读] [AllReduce逐行 S130] 记录RtsqBase::ConfigSqTail的状态/性能诊断，字段包含写给驱动的SQ属性值；日志本身不执行传输。
    HCCL_INFO("RtsqBase::%s, value=%u", __func__, value);
    // [中文导读] [AllReduce逐行 S131] 组装并执行指定SQ属性的驱动配置；传入/处理写给驱动的SQ属性值。
    ConfigSqStatusByType(drvSqCqPropType_t::DRV_SQCQ_PROP_SQ_TAIL, value);
// [中文导读] [AllReduce逐行 S132] 结束RtsqBase::ConfigSqTail函数体；控制流返回外层。
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
