/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_SQE_BUILD_A5_H
#define HCCLV2_SQE_BUILD_A5_H
#include "types.h"
#include "ub_jetty_lite.h"
#include "sqe_v82.h"
#include "log.h"

namespace Hccl {

constexpr u32 LOW_BITS = 16;

u32 GetKernelExecTimeoutFromEnvConfig();

extern thread_local uint8_t g_sqeProfBit;
void SetSqeProfilingEnabled(bool isEnabled);

// [中文导读] [AllReduce逐行 S27] SetSqeHeaderTaskFields的接口声明：当前UB WQE结构、当前任务编号；这些参数属于本函数调用边界。
inline void SetSqeHeaderTaskFields(void* sqe, u32 taskId)
// [中文导读] [AllReduce逐行 S28] 进入SetSqeHeaderTaskFields函数体：把32位taskId拆低/高16位写入SQE头两个任务标识字段。
{
    // [中文导读] [AllReduce逐行 S29] 设置SQE公共头为/按`reinterpret_cast<Rt91095StarsSqeHeader*>(sqe)`（当前UB WQE结构）。
    auto header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqe);
    // [中文导读] [AllReduce逐行 S30] 设置SQE公共头的rtStreamId字段为/按`static_cast<uint16_t>(taskId)`（当前任务编号）。
    header->rtStreamId = static_cast<uint16_t>(taskId);
    // [中文导读] [AllReduce逐行 S31] 设置SQE公共头的taskId字段为/按`static_cast<uint16_t>(taskId >> LOW_BITS)`（当前任务编号）。
    header->taskId = static_cast<uint16_t>(taskId >> LOW_BITS);
// [中文导读] [AllReduce逐行 S32] 结束SetSqeHeaderTaskFields函数体；控制流返回外层。
}

inline void BuildA5SqeNotifyWait(u32 streamId, u32 taskId, u32 notifyId, uint8_t* const sqeIn)
{
    (void)streamId;
    Rt91095StarsNotifySqe* sqe = (Rt91095StarsNotifySqe*)sqeIn;
    SetSqeHeaderTaskFields(sqe, taskId);

    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    sqe->header.wrCqe = 0U;

    sqe->cntFlag = false;
    sqe->clrFlag = true;
    sqe->subType = static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_SINGLE_NOTIFY_WAIT);
    sqe->kernelCredit
        = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT; // wait任务需要设置为0xff，否则会触发rtsq的超时机制，即使record了也wait不到
    sqe->notifyId = notifyId;
    sqe->timeout = GetKernelExecTimeoutFromEnvConfig();
    sqe->header.sqeProf = g_sqeProfBit;
}

// [中文导读] [AllReduce逐行 S53] BuildA5SqeNotifyWait的接口声明：当前任务编号、硬件通知ID、超时秒数、当前RTSQ本地SQE缓存地址；这些参数属于本函数调用边界。
inline void BuildA5SqeNotifyWait(u32 streamId, u32 taskId, u32 notifyId, u32 timeout, uint8_t* const sqeIn)
// [中文导读] [AllReduce逐行 S54] 进入BuildA5SqeNotifyWait函数体：填写本端硬件通知等待SQE，超时原值写入timeout并开启消费清除。
{
    // [中文导读] [AllReduce逐行 S55] 显式忽略`streamId`，该接口参数/调用结果在此实现中未参与后续计算。
    (void)streamId;
    // [中文导读] [AllReduce逐行 S56] 把当前RTSQ本地缓存地址按该硬件SQE结构解释；以下字段写入缓存，等待RefreshInfo/LaunchTask提交。
    Rt91095StarsNotifySqe* sqe = (Rt91095StarsNotifySqe*)sqeIn;
    // [中文导读] [AllReduce逐行 S57] 把taskId低16位/高16位写入SQE头rtStreamId/taskId；传入/处理当前UB WQE结构、当前任务编号。
    SetSqeHeaderTaskFields(sqe, taskId);

    // [中文导读] [AllReduce逐行 S59] 设置当前UB WQE结构的header.type字段为/按`static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT)`。
    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    // [中文导读] [AllReduce逐行 S60] 设置该RTSQ任务的wrCqe为0；此字段与UB数据WQE的cqe使能是不同层的配置。
    sqe->header.wrCqe = 0U;

    // [中文导读] [AllReduce逐行 S62] 设置当前UB WQE结构的kernelCredit字段为/按`RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT`。
    sqe->kernelCredit = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT;
    // [中文导读] [AllReduce逐行 S63] 关闭计数通知模式，本任务按单次通知等待语义编码。
    sqe->cntFlag = false;
    // [中文导读] [AllReduce逐行 S64] 设置等待消费后清除通知的clrFlag，供后续轮次重新使用该通知槽。
    sqe->clrFlag = true;
    // [中文导读] [AllReduce逐行 S65] 设置当前UB WQE结构的subType字段为/按`static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_SINGLE_NOTIFY_WAIT)`。
    sqe->subType = static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_SINGLE_NOTIFY_WAIT);
    // [中文导读] [AllReduce逐行 S66] 设置当前UB WQE结构的notifyId字段为/按`notifyId`（硬件通知ID）。
    sqe->notifyId = notifyId;
    // [中文导读] [AllReduce逐行 S67] 将调用链传入的秒级timeout原值写入硬件通知等待SQE；本行没有单位换算。
    sqe->timeout = timeout;
    // [中文导读] [AllReduce逐行 S68] 把当前线程SQE性能采集标志写入头字段，保留任务观测配置。
    sqe->header.sqeProf = g_sqeProfBit;
// [中文导读] [AllReduce逐行 S69] 结束BuildA5SqeNotifyWait函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S71] BuildA5SqeNotifyRecord的接口声明：当前任务编号、硬件通知ID、当前RTSQ本地SQE缓存地址；这些参数属于本函数调用边界。
inline void BuildA5SqeNotifyRecord(u32 streamId, u32 taskId, u32 notifyId, uint8_t* const sqeIn)
// [中文导读] [AllReduce逐行 S72] 进入BuildA5SqeNotifyRecord函数体：填写本端硬件通知记录SQE，设置通知ID与单通知子类型。
{
    // [中文导读] [AllReduce逐行 S73] 显式忽略`streamId`，该接口参数/调用结果在此实现中未参与后续计算。
    (void)streamId;
    // [中文导读] [AllReduce逐行 S74] 把当前RTSQ本地缓存地址按该硬件SQE结构解释；以下字段写入缓存，等待RefreshInfo/LaunchTask提交。
    Rt91095StarsNotifySqe* sqe = (Rt91095StarsNotifySqe*)sqeIn;
    // [中文导读] [AllReduce逐行 S75] 把taskId低16位/高16位写入SQE头rtStreamId/taskId；传入/处理当前UB WQE结构、当前任务编号。
    SetSqeHeaderTaskFields(sqe, taskId);

    // [中文导读] [AllReduce逐行 S77] 设置当前UB WQE结构的header.type字段为/按`static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD)`。
    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
    // [中文导读] [AllReduce逐行 S78] 设置该RTSQ任务的wrCqe为0；此字段与UB数据WQE的cqe使能是不同层的配置。
    sqe->header.wrCqe = 0U;

    // [中文导读] [AllReduce逐行 S80] 设置当前UB WQE结构的kernelCredit字段为/按`RT_STARS_DEFAULT_KERNEL_CREDIT`。
    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    // [中文导读] [AllReduce逐行 S81] 设置当前UB WQE结构的subType字段为/按`static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_SINGLE_NOTIFY_RECORD)`。
    sqe->subType = static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_SINGLE_NOTIFY_RECORD);
    // [中文导读] [AllReduce逐行 S82] 设置当前UB WQE结构的notifyId字段为/按`notifyId`（硬件通知ID）。
    sqe->notifyId = notifyId;
    // [中文导读] [AllReduce逐行 S83] 把当前线程SQE性能采集标志写入头字段，保留任务观测配置。
    sqe->header.sqeProf = g_sqeProfBit;
// [中文导读] [AllReduce逐行 S84] 结束BuildA5SqeNotifyRecord函数体；控制流返回外层。
}

inline void BuildA5SqeCnt1toNNotifyRecord(u32 streamId, u32 taskId, u32 notifyId, u32 cntValue, uint8_t* const sqeIn)
{
    (void)streamId;
    Rt91095StarsNotifySqe* sqe = (Rt91095StarsNotifySqe*)sqeIn;
    SetSqeHeaderTaskFields(sqe, taskId);

    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
    sqe->header.wrCqe = 0U;

    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->clrFlag = false;
    sqe->cntFlag = true;
    sqe->recordModeBit = 0x0U;
    sqe->subType = static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_COUNT_NOTIFY_RECORD);
    sqe->notifyId = notifyId;
    sqe->cntValue = cntValue;
    sqe->header.sqeProf = g_sqeProfBit;
}

inline void BuildA5SqeCnt1toNNotifyWait(u32 streamId, u32 taskId, u32 notifyId, u32 cntValue, uint8_t* const sqeIn)
{
    (void)streamId;
    Rt91095StarsNotifySqe* sqe = (Rt91095StarsNotifySqe*)sqeIn;
    SetSqeHeaderTaskFields(sqe, taskId);

    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    sqe->header.wrCqe = 0U;

    sqe->kernelCredit = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT;
    sqe->cntFlag = true;
    sqe->clrFlag = true;
    sqe->bitmap = 1U;
    sqe->subType = static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_COUNT_NOTIFY_WAIT);
    sqe->notifyId = notifyId;
    sqe->cntValue = cntValue;
    sqe->header.sqeProf = g_sqeProfBit;
}

inline void BuildA5SqeCntNto1NotifyRecord(u32 streamId, u32 taskId, u32 notifyId, u32 cntValue, uint8_t* const sqeIn)
{
    (void)streamId;
    Rt91095StarsNotifySqe* sqe = (Rt91095StarsNotifySqe*)sqeIn;
    SetSqeHeaderTaskFields(sqe, taskId);

    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
    sqe->header.wrCqe = 0U;
    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;

    sqe->clrFlag = false;
    sqe->cntFlag = true;
    sqe->recordModeBit = 0x2U;
    sqe->subType = static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_COUNT_NOTIFY_RECORD);
    sqe->notifyId = notifyId;
    sqe->cntValue = cntValue;
    sqe->header.sqeProf = g_sqeProfBit;
}

inline void BuildA5SqeCntNto1NotifyWait(u32 streamId, u32 taskId, u32 notifyId, u32 cntValue, uint8_t* const sqeIn)
{
    (void)streamId;
    Rt91095StarsNotifySqe* sqe = (Rt91095StarsNotifySqe*)sqeIn;
    SetSqeHeaderTaskFields(sqe, taskId);

    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    sqe->header.wrCqe = 0U;

    sqe->kernelCredit = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT;
    sqe->cntFlag = true;
    sqe->clrFlag = true;
    sqe->waitModeBit = 0x1U;
    sqe->subType = static_cast<uint16_t>(Rt91095NotifySubType::NOTIFY_SUB_TYPE_COUNT_NOTIFY_WAIT);
    sqe->notifyId = notifyId;
    sqe->cntValue = cntValue;
    sqe->header.sqeProf = g_sqeProfBit;
}

// [中文导读] [AllReduce逐行 S162] BuildA5SqeSdmaCopy的接口声明：按字节长度/地址/归约opcode填写本地SDMA SQE；这些参数属于本函数调用边界。
inline void BuildA5SqeSdmaCopy(
    // [中文导读] [AllReduce逐行 S163] BuildA5SqeSdmaCopy的接口声明：当前任务编号、目标地址整数表示、源地址整数表示、字节容量或单片字节数、SDMA分区标识、SDMA复制/归约操作编码、当前RTSQ本地SQE缓存地址；这些参数属于本函数调用边界。
    u32 streamId, u32 taskId, u64 dstAddr, u64 srcAddr, u32 size, u32 partId, u32 opcode, uint8_t* const sqeIn)
// [中文导读] [AllReduce逐行 S164] 进入BuildA5SqeSdmaCopy函数体：按字节长度/地址/归约opcode填写本地SDMA SQE。
{
    // [中文导读] [AllReduce逐行 S165] 显式忽略`streamId`，该接口参数/调用结果在此实现中未参与后续计算。
    (void)streamId;
    // [中文导读] [AllReduce逐行 S166] 把当前RTSQ本地缓存地址按该硬件SQE结构解释；以下字段写入缓存，等待RefreshInfo/LaunchTask提交。
    Rt91095StarsMemcpySqe* sqe = (Rt91095StarsMemcpySqe*)sqeIn;
    // [中文导读] [AllReduce逐行 S167] 把taskId低16位/高16位写入SQE头rtStreamId/taskId；传入/处理当前UB WQE结构、当前任务编号。
    SetSqeHeaderTaskFields(sqe, taskId);

    // [中文导读] [AllReduce逐行 S169] 设置当前UB WQE结构的header.type字段为/按`static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA)`。
    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
    // [中文导读] [AllReduce逐行 S170] 设置该RTSQ任务的wrCqe为0；此字段与UB数据WQE的cqe使能是不同层的配置。
    sqe->header.wrCqe = 0U;

    // [中文导读] [AllReduce逐行 S172] 设置当前UB WQE结构的opcode字段为/按`opcode`（SDMA复制/归约操作编码）。
    sqe->opcode = opcode;
    // [中文导读] [AllReduce逐行 S173] 设置当前UB WQE结构的kernelCredit字段为/按`RT_STARS_DEFAULT_KERNEL_CREDIT`。
    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    // [中文导读] [AllReduce逐行 S174] 设置当前UB WQE结构的sssv字段为/按`1U`。
    sqe->sssv = 1U;
    // [中文导读] [AllReduce逐行 S175] 设置当前UB WQE结构的dssv字段为/按`1U`。
    sqe->dssv = 1U;
    // [中文导读] [AllReduce逐行 S176] 设置当前UB WQE结构的sns字段为/按`1U`。
    sqe->sns = 1U;
    // [中文导读] [AllReduce逐行 S177] 设置当前UB WQE结构的dns字段为/按`1U`。
    sqe->dns = 1U;
    // [中文导读] [AllReduce逐行 S178] 设置当前UB WQE结构的mapamPartId字段为/按`partId`（SDMA分区标识）。
    sqe->mapamPartId = partId;

    // [中文导读] [AllReduce逐行 S180] 将本次SDMA字节长度写入strideMode0.lengthMove，归约操作亦通过该长度字段提交。
    sqe->u.strideMode0.lengthMove = size;
    // [中文导读] [AllReduce逐行 S181] 设置当前UB WQE结构的u.strideMode0.srcAddrLow字段为/按`static_cast<uint32_t>(srcAddr & 0x00000000ffffffffU)`（源地址整数表示）。
    sqe->u.strideMode0.srcAddrLow = static_cast<uint32_t>(srcAddr & 0x00000000ffffffffU);
    // [中文导读] [AllReduce逐行 S182] 设置当前UB WQE结构的u.strideMode0.srcAddrHigh字段为/按`static_cast<uint32_t>((srcAddr & 0xffffffff00000000U) >> 32)`（源地址整数表示）。
    sqe->u.strideMode0.srcAddrHigh = static_cast<uint32_t>((srcAddr & 0xffffffff00000000U) >> 32);
    // [中文导读] [AllReduce逐行 S183] 设置当前UB WQE结构的u.strideMode0.dstAddrLow字段为/按`static_cast<uint32_t>(dstAddr & 0x00000000ffffffffU)`（目标地址整数表示）。
    sqe->u.strideMode0.dstAddrLow = static_cast<uint32_t>(dstAddr & 0x00000000ffffffffU);
    // [中文导读] [AllReduce逐行 S184] 设置当前UB WQE结构的u.strideMode0.dstAddrHigh字段为/按`static_cast<uint32_t>((dstAddr & 0xffffffff00000000U) >> 32)`（目标地址整数表示）。
    sqe->u.strideMode0.dstAddrHigh = static_cast<uint32_t>((dstAddr & 0xffffffff00000000U) >> 32);
    // [中文导读] [AllReduce逐行 S185] 把当前线程SQE性能采集标志写入头字段，保留任务观测配置。
    sqe->header.sqeProf = g_sqeProfBit;
// [中文导读] [AllReduce逐行 S186] 结束BuildA5SqeSdmaCopy函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S188] BuildA5SqeUbDbSend的接口声明：把jetty/die/function标识与16位UB PI编码到RTSQ Doorbell SQE；这些参数属于本函数调用边界。
inline void
// [中文导读] [AllReduce逐行 S189] BuildA5SqeUbDbSend的接口声明：当前任务编号、UB jetty的die/function/jetty标识、16位UB jetty生产指针、当前RTSQ本地SQE缓存地址；这些参数属于本函数调用边界。
BuildA5SqeUbDbSend(u32 streamId, u32 taskId, const UbJettyLiteId& jettyLiteId, u16 piValue, uint8_t* const sqeIn)
// [中文导读] [AllReduce逐行 S190] 进入BuildA5SqeUbDbSend函数体：把jetty/die/function标识与16位UB PI编码到RTSQ Doorbell SQE。
{
    // [中文导读] [AllReduce逐行 S191] 显式忽略`streamId`，该接口参数/调用结果在此实现中未参与后续计算。
    (void)streamId;
    // [中文导读] [AllReduce逐行 S192] 把当前RTSQ本地缓存地址按该硬件SQE结构解释；以下字段写入缓存，等待RefreshInfo/LaunchTask提交。
    Rt91095StarsUbdmaDBmodeSqe* sqe = (Rt91095StarsUbdmaDBmodeSqe*)sqeIn;
    // [中文导读] [AllReduce逐行 S193] 把taskId低16位/高16位写入SQE头rtStreamId/taskId；传入/处理当前UB WQE结构、当前任务编号。
    SetSqeHeaderTaskFields(sqe, taskId);

    // [中文导读] [AllReduce逐行 S195] 设置当前UB WQE结构的header.type字段为/按`static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA)`。
    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA);

    // [中文导读] [AllReduce逐行 S197] 设置当前UB WQE结构的mode字段为/按`Rt91095UbDmaSqeMode::RT_91095_SQE_DOORBELL_MODE`。
    sqe->mode = Rt91095UbDmaSqeMode::RT_91095_SQE_DOORBELL_MODE;
    // [中文导读] [AllReduce逐行 S198] 设置当前UB WQE结构的kernelCredit字段为/按`RT_STARS_DEFAULT_KERNEL_CREDIT`。
    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    // [中文导读] [AllReduce逐行 S199] 设置当前UB WQE结构的doorbellNum字段为/按`1U`。
    sqe->doorbellNum = 1U;
    // [中文导读] [AllReduce逐行 S200] 设置当前UB WQE结构的jettyId1字段为/按`jettyLiteId.GetJettyId()`（UB jetty的die/function/jetty标识的GetJettyId字段）；读取UB jetty编号以填写Doorbell目标。
    sqe->jettyId1 = jettyLiteId.GetJettyId();
    // [中文导读] [AllReduce逐行 S201] 设置当前UB WQE结构的funcId1字段为/按`jettyLiteId.GetFuncId()`（UB jetty的die/function/jetty标识的GetFuncId字段）；读取UB功能编号以填写Doorbell目标。
    sqe->funcId1 = jettyLiteId.GetFuncId();
    // [中文导读] [AllReduce逐行 S202] 把16位UB生产指针写入Doorbell条目，通知指定jetty处理此前已写入UB SQ的WQE。
    sqe->piValue1 = piValue;
    // [中文导读] [AllReduce逐行 S203] 设置当前UB WQE结构的dieId1字段为/按`jettyLiteId.GetDieId()`（UB jetty的die/function/jetty标识的GetDieId字段）；读取UB die编号以填写Doorbell目标。
    sqe->dieId1 = jettyLiteId.GetDieId();
    // [中文导读] [AllReduce逐行 S204] 把当前线程SQE性能采集标志写入头字段，保留任务观测配置。
    sqe->header.sqeProf = g_sqeProfBit;
// [中文导读] [AllReduce逐行 S205] 结束BuildA5SqeUbDbSend函数体；控制流返回外层。
}

inline void BuildA5SqeP2pWriteValue(u32 streamId, u32 taskId, u64 remoteAddr, u32 writeValue, uint8_t* const sqeIn)
{
    (void)streamId;
    Rt91095StarsWriteValueSqe* sqe = reinterpret_cast<Rt91095StarsWriteValueSqe*>(sqeIn);
    SetSqeHeaderTaskFields(sqe, taskId);

    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_WRITE_VALUE);

    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->writeAddrLow = remoteAddr & MASK_32_BIT;
    sqe->writeAddrHigh = (remoteAddr >> UINT32_BIT_NUM) & MASK_17_BIT;
    sqe->awsize = RtStarsWriteValueSizeType::RT_STARS_WRITE_VALUE_SIZE_TYPE_32BIT;
    sqe->writeValuePart[0] = writeValue;
    sqe->va = 1;
    sqe->header.sqeProf = g_sqeProfBit;
}

u32 GetKernelExecTimeoutFromEnvConfig();

void BuildA5SqeCCoreNotifyWait(u32 streamId, u32 taskId, u64 waitAddr, u64 actAddr, bool last, uint8_t* const sqeIn);

void BuildA5SqeCCoreNotifyRecord(u32 streamId, u32 taskId, u64 writeAddr, u64 valueAddr, uint8_t* const sqeIn);

void BuildA5SqeRdmaDbSend(u32 streamId, u32 taskId, u64 dbAddr, u64 dbValue, uint8_t* const sqeIn);

} // namespace Hccl

#endif
