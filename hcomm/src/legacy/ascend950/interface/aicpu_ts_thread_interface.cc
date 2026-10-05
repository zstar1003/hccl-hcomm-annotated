/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_ts_thread_interface.h"

#include <memory>

#include "stream_lite.h"
#include "sqe_build_a5.h"

namespace Hccl {

namespace { // make the definitions file-scoped

    std::unordered_map<uint32_t, ReduceOp> mapU32ToReduceOp
        = {{0, ReduceOp::SUM}, {1, ReduceOp::PROD}, {2, ReduceOp::MAX}, {3, ReduceOp::MIN}};

    std::unordered_map<uint32_t, DataType> mapU32ToDataType
        = {{0, DataType::INT8},    {1, DataType::INT16},  {2, DataType::INT32},    {3, DataType::FP16},
           {4, DataType::FP32},    {5, DataType::INT64},  {6, DataType::UINT64},   {7, DataType::UINT8},
           {8, DataType::UINT16},  {9, DataType::UINT32}, {10, DataType::FP64},    {11, DataType::BFP16},
           {12, DataType::INT128}, {14, DataType::HIF8},  {15, DataType::FP8E4M3}, {16, DataType::FP8E5M2},
           {17, DataType::FP8E8M0}};

    inline HcclResult CheckDataTypeAndReduceOp(uint32_t dataType, uint32_t reduceOp)
    {
        if (mapU32ToDataType.find(dataType) == mapU32ToDataType.end()) {
            HCCL_ERROR("[IAicpuTsThread][%s] type[%u] is not supported.", __func__, dataType);
            return HCCL_E_PARA;
        }
        if (mapU32ToReduceOp.find(reduceOp) == mapU32ToReduceOp.end()) {
            HCCL_ERROR("[IAicpuTsThread][%s] op[%u] is not supported.", __func__, reduceOp);
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

} // namespace

// 此处失败的原因只可能是内存分配失败，所以可以直接抛出标准异常
IAicpuTsThread::IAicpuTsThread(uint32_t id, uint32_t sqIds, uint32_t phyId, uint32_t logicCqids)
{
    StreamLite* streamLitePtr = new StreamLite(id, sqIds, phyId, logicCqids, true);
    if (streamLitePtr == nullptr) {
        HCCL_ERROR(
            "[IAicpuTsThread::%s] new StreamLite failed, id [%u], sqIds [%u], phyId [%u], logicCqids [%u]", __func__,
            id, sqIds, phyId, logicCqids);
        throw std::bad_alloc();
    }
    streamLiteVoidPtr_ = static_cast<void*>(streamLitePtr);
}

IAicpuTsThread::~IAicpuTsThread()
{
    StreamLite* streamLitePtr = static_cast<StreamLite*>(streamLiteVoidPtr_);
    if (streamLitePtr != nullptr) {
        delete streamLitePtr;
        streamLiteVoidPtr_ = nullptr;
    }
}

// [中文导读] [AllReduce逐行 S69] IAicpuTsThread::LaunchTask的接口声明：从 StreamLite 取得 RTSQ 并调用具体队列发射；这些参数属于本函数调用边界。
void IAicpuTsThread::LaunchTask() const
// [中文导读] [AllReduce逐行 S70] 进入IAicpuTsThread::LaunchTask函数体：从 StreamLite 取得 RTSQ 并调用具体队列发射。
{
    // [中文导读] [AllReduce逐行 S71] 设置具体RTSQ执行队列为/按`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq()`（接口层保存的StreamLite地址）；返回当前StreamLite持有的具体执行队列。
    RtsqBase* rtsqA5 = static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq();

    // [中文导读] [AllReduce逐行 S73] 记录IAicpuTsThread::LaunchTask的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S74] 为当前IAicpuTsThread::LaunchTask诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "[IAicpuTsThread::%s] Launch Task at Stream id [%u]", __func__,
        // [中文导读] [AllReduce逐行 S75] 为读取执行流/通知资源的实际ID补入`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId())`（接口层保存的StreamLite地址）；本行是参数/结构化初始化续行。
        static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId());

    // [中文导读] [AllReduce逐行 S77] 将已生成的任务提交到具体RTSQ队列，不等同于全部任务完成；传入/处理具体RTSQ执行队列的LaunchTask字段。
    rtsqA5->LaunchTask();
    // [中文导读] [AllReduce逐行 S78] 结束本次void调用，当前路径不再继续下发后续操作。
    return;
// [中文导读] [AllReduce逐行 S79] 结束IAicpuTsThread::LaunchTask函数体；控制流返回外层。
}

void IAicpuTsThread::TryLaunchTask() const
{
    HCCL_DEBUG(
        "[IAicpuTsThread::%s] TryLaunch Task at Stream id [%u]", __func__,
        static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId());

    RtsqBase* rtsqA5 = static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq();
    if (rtsqA5 != nullptr) {
        rtsqA5->TryLaunchTask();
    }
    return;
}

HcclResult IAicpuTsThread::NotifyWait(uint32_t notifyId) const
{
    return NotifyWait(notifyId, GetKernelExecTimeoutFromEnvConfig());
}

// [中文导读] [AllReduce逐行 S99] IAicpuTsThread::NotifyWait的接口声明：硬件通知ID、超时秒数；这些参数属于本函数调用边界。
HcclResult IAicpuTsThread::NotifyWait(uint32_t notifyId, uint32_t timeout) const
// [中文导读] [AllReduce逐行 S100] 进入IAicpuTsThread::NotifyWait函数体：将硬件通知ID与超时交给具体 RTSQ 等待任务。
{
    // [中文导读] [AllReduce逐行 S101] 设置具体RTSQ执行队列为/按`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq()`（接口层保存的StreamLite地址）；返回当前StreamLite持有的具体执行队列。
    RtsqBase* rtsqA5 = static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq();

    // [中文导读] [AllReduce逐行 S103] 记录IAicpuTsThread::NotifyWait的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S104] 提供通知等待日志格式；原格式写ms，但timeout未经换算交RTSQ，等待链实际按秒解释。
        "[IAicpuTsThread::%s] at Stream id [%u], notifyId [%u], timeout [%u ms]", __func__,
        // [中文导读] [AllReduce逐行 S105] 为读取执行流/通知资源的实际ID补入`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId(), notifyId, timeout)`（接口层保存的StreamLite地址、硬件通知ID、超时秒数）；本行是参数/结构化初始化续行。
        static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId(), notifyId, timeout);

    // [中文导读] [AllReduce逐行 S107] 生成指定硬件通知ID的等待SQE；传入/处理具体RTSQ执行队列的NotifyWait字段、硬件通知ID、超时秒数。
    rtsqA5->NotifyWait(notifyId, timeout);

    // [中文导读] [AllReduce逐行 S109] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S110] 结束IAicpuTsThread::NotifyWait函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S112] IAicpuTsThread::NotifyRecordLoc的接口声明：硬件通知ID；这些参数属于本函数调用边界。
HcclResult IAicpuTsThread::NotifyRecordLoc(uint32_t notifyId) const
// [中文导读] [AllReduce逐行 S113] 进入IAicpuTsThread::NotifyRecordLoc函数体：从 StreamLite 取得 RTSQ 并生成本地通知记录。
{
    // [中文导读] [AllReduce逐行 S114] 设置具体RTSQ执行队列为/按`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq()`（接口层保存的StreamLite地址）；返回当前StreamLite持有的具体执行队列。
    RtsqBase* rtsqA5 = static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq();

    // [中文导读] [AllReduce逐行 S116] 记录IAicpuTsThread::NotifyRecordLoc的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S117] 为当前IAicpuTsThread::NotifyRecordLoc诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "[IAicpuTsThread::%s] at Stream id [%u], notifyId [%u]", __func__,
        // [中文导读] [AllReduce逐行 S118] 为读取执行流/通知资源的实际ID补入`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId(), notifyId)`（接口层保存的StreamLite地址、硬件通知ID）；本行是参数/结构化初始化续行。
        static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId(), notifyId);

    // [中文导读] [AllReduce逐行 S120] 生成指定硬件ID的本地通知记录SQE；传入/处理具体RTSQ执行队列的NotifyRecordLoc字段、硬件通知ID。
    rtsqA5->NotifyRecordLoc(notifyId);

    // [中文导读] [AllReduce逐行 S122] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S123] 结束IAicpuTsThread::NotifyRecordLoc函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S125] IAicpuTsThread::SdmaCopy的接口声明：目标地址整数表示、源地址整数表示、SDMA字节长度；这些参数属于本函数调用边界。
HcclResult IAicpuTsThread::SdmaCopy(uint64_t dstAddr, uint64_t srcAddr, uint64_t sizeByte) const
// [中文导读] [AllReduce逐行 S126] 进入IAicpuTsThread::SdmaCopy函数体：检查单次 SDMA 字节范围并转换为32位长度，转换源/目标顺序交给 RTSQ。
{
    // SDMA单个任务最大支持4GB的数据量，超过4GB需要分多次提交
    // 为了避免不必要的依赖和复杂性，这里不直接使用DeviceCapacity中定义的SDMA_SEND_MAX_SIZE，而是直接使用4GB的值
    // [中文导读] [AllReduce逐行 S129] 仅当`(sizeByte > 0x100000000ULL)`（SDMA字节长度）成立时进入此分支。
    if (sizeByte > 0x100000000ULL) {
        // [中文导读] [AllReduce逐行 S130] 记录IAicpuTsThread::SdmaCopy的错误诊断，字段包含SDMA字节长度；日志本身不执行传输。
        HCCL_ERROR("[%s] sizeByte [%llu] exceeds 4GB", __func__, static_cast<unsigned long long>(sizeByte));
        // [中文导读] [AllReduce逐行 S131] 返回HCCL_E_PARA，表示参数不满足此分支要求；此路径停止本函数的后续处理。
        return HCCL_E_PARA;
    // [中文导读] [AllReduce逐行 S132] 结束`if (sizeByte > 0x100000000ULL)`（SDMA字节长度）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S134] 设置具体RTSQ执行队列为/按`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq()`（接口层保存的StreamLite地址）；返回当前StreamLite持有的具体执行队列。
    RtsqBase* rtsqA5 = static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq();

    // [中文导读] [AllReduce逐行 S136] 把SDMA字节长度窄化为32位；前面仅拒绝大于4GiB，等于4GiB仍按源码发生窄化。
    uint32_t sizeByteNarrowed = static_cast<uint32_t>(sizeByte);

    // [中文导读] [AllReduce逐行 S138] 记录IAicpuTsThread::SdmaCopy的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S139] 为当前IAicpuTsThread::SdmaCopy诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "[IAicpuTsThread::%s] at Stream id [%u], dstAddr [%llx], srcAddr [%llx], sizeByteNarrowed [%u]", __func__,
        // [中文导读] [AllReduce逐行 S140] 为读取执行流/通知资源的实际ID补入`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId(), static_cast<unsigned long long>(dstAddr),`（接口层保存的StreamLite地址、目标地址整数表示）；本行是参数/结构化初始化续行。
        static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId(), static_cast<unsigned long long>(dstAddr),
        // [中文导读] [AllReduce逐行 S141] 为读取执行流/通知资源的实际ID补入`static_cast<unsigned long long>(srcAddr), sizeByteNarrowed)`（源地址整数表示）；本行是参数/结构化初始化续行。
        static_cast<unsigned long long>(srcAddr), sizeByteNarrowed);

    // [中文导读] [AllReduce逐行 S143] 按本端地址/字节长度生成SDMA复制SQE；传入/处理具体RTSQ执行队列的SdmaCopy字段、源地址整数表示、目标地址整数表示。
    rtsqA5->SdmaCopy(srcAddr, dstAddr, sizeByteNarrowed, 0);

    // [中文导读] [AllReduce逐行 S145] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S146] 结束IAicpuTsThread::SdmaCopy函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S148] IAicpuTsThread::SdmaReduce的接口声明：检查字节长度与归约映射并构造 ReduceIn，交给 RTSQ SDMA归约；这些参数属于本函数调用边界。
HcclResult IAicpuTsThread::SdmaReduce(
    // [中文导读] [AllReduce逐行 S149] IAicpuTsThread::SdmaReduce的接口声明：目标地址整数表示、源地址整数表示、SDMA字节长度、接口数据类型原始枚举、接口归约操作原始枚举；这些参数属于本函数调用边界。
    uint64_t dstAddr, uint64_t srcAddr, uint64_t sizeByte, uint32_t dataTypeRaw, uint32_t reduceOpRaw) const
// [中文导读] [AllReduce逐行 S150] 进入IAicpuTsThread::SdmaReduce函数体：检查字节长度与归约映射并构造 ReduceIn，交给 RTSQ SDMA归约。
{
    // SDMA单个任务最大支持4GB的数据量，超过4GB需要分多次提交
    // 为了避免不必要的依赖和复杂性，这里不直接使用DeviceCapacity中定义的SDMA_SEND_MAX_SIZE，而是直接使用4GB的值
    // [中文导读] [AllReduce逐行 S153] 仅当`(sizeByte > 0x100000000ULL)`（SDMA字节长度）成立时进入此分支。
    if (sizeByte > 0x100000000ULL) {
        // [中文导读] [AllReduce逐行 S154] 记录IAicpuTsThread::SdmaReduce的错误诊断，字段包含SDMA字节长度；日志本身不执行传输。
        HCCL_ERROR("[%s] sizeByte [%llu] exceeds 4GB", __func__, static_cast<unsigned long long>(sizeByte));
        // [中文导读] [AllReduce逐行 S155] 返回HCCL_E_PARA，表示参数不满足此分支要求；此路径停止本函数的后续处理。
        return HCCL_E_PARA;
    // [中文导读] [AllReduce逐行 S156] 结束`if (sizeByte > 0x100000000ULL)`（SDMA字节长度）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S158] 设置具体RTSQ执行队列为/按`static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq()`（接口层保存的StreamLite地址）；返回当前StreamLite持有的具体执行队列。
    RtsqBase* rtsqA5 = static_cast<StreamLite*>(streamLiteVoidPtr_)->GetRtsq();

    // [中文导读] [AllReduce逐行 S160] 检查公开归约类型/操作是否存在底层支持映射；返回非成功时由检查宏立即向上传递。
    CHK_RET(CheckDataTypeAndReduceOp(dataTypeRaw, reduceOpRaw));
    // [中文导读] [AllReduce逐行 S161] 设置元素数据类型为/按`mapU32ToDataType.at(dataTypeRaw)`（接口数据类型原始枚举）；调用at，使用接口数据类型原始枚举。
    DataType dataType = mapU32ToDataType.at(dataTypeRaw);
    // [中文导读] [AllReduce逐行 S162] 设置归约操作为/按`mapU32ToReduceOp.at(reduceOpRaw)`（接口归约操作原始枚举）；调用at，使用接口归约操作原始枚举。
    ReduceOp reduceOp = mapU32ToReduceOp.at(reduceOpRaw);
    // [中文导读] [AllReduce逐行 S163] 准备底层归约类型/操作描述、元素数据类型、归约操作的局部存储/结构描述，初始化方式以本行声明为准。
    ReduceIn reduceIn{dataType, reduceOp};

    // [中文导读] [AllReduce逐行 S165] 把SDMA字节长度窄化为32位；前面仅拒绝大于4GiB，等于4GiB仍按源码发生窄化。
    uint32_t sizeByteNarrowed = static_cast<uint32_t>(sizeByte);

    // [中文导读] [AllReduce逐行 S167] 记录IAicpuTsThread::SdmaReduce的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S168] 为当前IAicpuTsThread::SdmaReduce诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "[IAicpuTsThread::%s] at Stream id [%u], dstAddr [%llx], srcAddr [%llx], sizeByteNarrowed [%u], dataType "
        // [中文导读] [AllReduce逐行 S169] 为当前IAicpuTsThread::SdmaReduce诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "[%u][%s], reduceOp [%u][%s]",
        // [中文导读] [AllReduce逐行 S170] 为读取执行流/通知资源的实际ID补入`__func__, static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId(), static_cast<unsigned long long>(dstAddr),`（接口层保存的StreamLite地址、目标地址整数表示）；本行是参数/结构化初始化续行。
        __func__, static_cast<StreamLite*>(streamLiteVoidPtr_)->GetId(), static_cast<unsigned long long>(dstAddr),
        // [中文导读] [AllReduce逐行 S171] 为读取执行流/通知资源的实际ID；取得对象诊断文本用于日志补入`static_cast<unsigned long long>(srcAddr), sizeByteNarrowed, dataTypeRaw, dataType.Describe().c_str(),`（源地址整数表示、接口数据类型原始枚举、元素数据类型的Describe字段）；本行是参数/结构化初始化续行。
        static_cast<unsigned long long>(srcAddr), sizeByteNarrowed, dataTypeRaw, dataType.Describe().c_str(),
        // [中文导读] [AllReduce逐行 S172] 为读取执行流/通知资源的实际ID；取得对象诊断文本用于日志补入`reduceOpRaw, reduceOp.Describe().c_str())`（接口归约操作原始枚举、归约操作的Describe字段）；本行是参数/结构化初始化续行。
        reduceOpRaw, reduceOp.Describe().c_str());

    // [中文导读] [AllReduce逐行 S174] 按本端地址/字节长度和归约方式生成SDMA归约SQE；传入/处理具体RTSQ执行队列的SdmaReduce字段、源地址整数表示、目标地址整数表示、底层归约类型/操作描述。
    rtsqA5->SdmaReduce(srcAddr, dstAddr, sizeByteNarrowed, 0, reduceIn);

    // [中文导读] [AllReduce逐行 S176] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S177] 结束IAicpuTsThread::SdmaReduce函数体；控制流返回外层。
}

} // namespace Hccl
