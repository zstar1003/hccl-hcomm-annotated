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
#include <chrono>
#include <unordered_map>
#include "rtsq_a5.h"
#include "log.h"
#include "exception_util.h"
#include "internal_exception.h"
#include "sqe_build_a5.h"
#include "sqe.h"
#ifdef CCL_KERNEL_AICPU
#include "aicpu_ts_primitives_c_adpt.h"
#endif
#include "aicpu_task_utils.h"

namespace Hccl {
using namespace std;
constexpr u32 RTSQ_A5_PART_ID = 0;
constexpr u32 PRINT_INTERVAL = 30;

RtsqA5::RtsqA5(u32 devPhyId, u32 streamId, u32 sqId) : RtsqBase(devPhyId, streamId, sqId) { SetTaskIdBySqeId(); }

RtsqA5::RtsqA5(u32 devPhyId, u32 streamId, u32 sqId, bool launchFlag) : RtsqBase(devPhyId, streamId, sqId)
{
    SetTaskIdBySqeId();
    launchFlag_ = launchFlag;
}

void RtsqA5::Reset(bool reset)
{
    RtsqBase::Reset(reset);
    pendingSqeCnt = 0;
    s32 sRet = memset_s(locBuf, RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT, 0, RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT);
    if (UNLIKELY(sRet != EOK)) {
        auto msg = StringFormat("[RtsqA5][Reset] locBuf memset fail. errorno[%d]", sRet);
        THROW<InternalException>(msg);
    }
    HCCL_INFO("[NsRecovery]RtsqA5::%s success", __func__);
}

// 计算head和tail之间的距离
u32 RtsqA5::GetTailToHeadDist() const
{
    if (UNLIKELY(sqHead_ == sqTail_)) { // 头尾相同，则距离大小为sq深度
        return sqDepth_;
    }
    return (sqTail_ < sqHead_) ? (sqHead_ - sqTail_) : (sqDepth_ - (sqTail_ - sqHead_));
}

// [中文导读] [AllReduce逐行 S59] RtsqA5::MakeSureAvailableSpace的接口声明：在 SQ 反压时读取头位置等待空间，检查超时/域状态并尝试其他流发射；这些参数属于本函数调用边界。
void RtsqA5::MakeSureAvailableSpace()
// [中文导读] [AllReduce逐行 S60] 进入RtsqA5::MakeSureAvailableSpace函数体：在 SQ 反压时读取头位置等待空间，检查超时/域状态并尝试其他流发射。
{
    // [中文导读] [AllReduce逐行 S61] 设置RTSQ可用槽位数为/按`GetTailToHeadDist()`；根据RTSQ头尾位置计算可用距离。
    u32 availableSpace = GetTailToHeadDist();
    // [中文导读] [AllReduce逐行 S62] 设置单调时钟开始时刻为/按`std::chrono::steady_clock::now()`；调用std::chrono::steady_clock::now。
    auto startTime = std::chrono::steady_clock::now();
// [中文导读] [AllReduce逐行 S63] 编译条件`ifdef CCL_KERNEL_AICPU`限定后续实现，区分Host/设备或构建能力分支。
#ifdef CCL_KERNEL_AICPU
    // [中文导读] [AllReduce逐行 S64] 设置SQ反压等待超时秒数为/按`GetSqFullTimeOut()`；调用GetSqFullTimeOut。
    sqFullTimeout_ = GetSqFullTimeOut();
// [中文导读] [AllReduce逐行 S65] 结束前述编译条件控制的实现片段。
#endif
    // [中文导读] [AllReduce逐行 S66] 调用printInterval，使用状态日志打印间隔；对象涉及状态日志打印间隔。
    const std::chrono::seconds printInterval(PRINT_INTERVAL); // 打印间隔30s
    // [中文导读] [AllReduce逐行 S67] 设置上次状态打印时刻为/按`std::chrono::steady_clock::now() - printInterval`（状态日志打印间隔）；调用std::chrono::steady_clock::now，使用状态日志打印间隔。
    auto lastPrintTime = std::chrono::steady_clock::now() - printInterval;
    // [中文导读] [AllReduce逐行 S68] 记录RtsqA5::MakeSureAvailableSpace的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S69] 为当前RtsqA5::MakeSureAvailableSpace诊断/异常表达式提供格式文本，将报告硬件SQ编号、SQ反压等待超时秒数；这一物理行没有数据搬运副作用。
        "[%s]sqId:%u, sqFullTimeout_: %u s, sqHead:%u, sqTail:%u, pendingSqeCnt:%u", __func__, sqId_, sqFullTimeout_,
        // [中文导读] [AllReduce逐行 S70] 为前述多行表达式补入`sqHead_, sqTail_, pendingSqeCnt)`（软件保存的RTSQ头槽位、软件保存的RTSQ尾槽位、本地待提交SQE条数）；本行是参数/结构化初始化续行。
        sqHead_, sqTail_, pendingSqeCnt);

    // [中文导读] [AllReduce逐行 S72] 在`(availableSpace <= pendingSqeCnt)`（RTSQ可用槽位数、本地待提交SQE条数）条件下重复执行后续等待或分片处理。
    while (availableSpace <= pendingSqeCnt) {
        // [中文导读] [AllReduce逐行 S73] 设置软件保存的RTSQ头槽位为/按`QuerySqHead()`；通过驱动查询SQ头，判断是否追上固定尾位置。
        sqHead_ = QuerySqHead();
        // [中文导读] [AllReduce逐行 S74] 设置RTSQ可用槽位数为/按`GetTailToHeadDist()`；根据RTSQ头尾位置计算可用距离。
        availableSpace = GetTailToHeadDist();
        // [中文导读] [AllReduce逐行 S75] 仅当`(availableSpace > pendingSqeCnt)`（RTSQ可用槽位数、本地待提交SQE条数）成立时进入此分支。
        if (availableSpace > pendingSqeCnt) {
            // [中文导读] [AllReduce逐行 S76] 退出当前循环或switch路径，继续其后处理；不代表其他执行流已经完成。
            break; // 避免head没更新导致假反压
        // [中文导读] [AllReduce逐行 S77] 结束`if (availableSpace > pendingSqeCnt)`（RTSQ可用槽位数、本地待提交SQE条数）分支/循环；控制流返回外层。
        }

        // [中文导读] [AllReduce逐行 S79] 设置当前单调时钟时刻为/按`std::chrono::steady_clock::now()`；调用std::chrono::steady_clock::now。
        auto curTime = std::chrono::steady_clock::now();
        // [中文导读] [AllReduce逐行 S80] 仅当`(UNLIKELY(curTime - lastPrintTime >= printInterval))`（当前单调时钟时刻、上次状态打印时刻、状态日志打印间隔）成立时进入此分支。
        if (UNLIKELY(curTime - lastPrintTime >= printInterval)) {
            // [中文导读] [AllReduce逐行 S81] 记录RtsqA5::MakeSureAvailableSpace的状态/性能诊断；日志本身不执行传输。
            HCCL_RUN_INFO(
                // [中文导读] [AllReduce逐行 S82] 为当前RtsqA5::MakeSureAvailableSpace诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "[%s]while loop, sqId:%u, sqHead:%u, sqTail:%u, availableSpace:%u, pendingSqeCnt:%u, "
                // [中文导读] [AllReduce逐行 S83] 为当前RtsqA5::MakeSureAvailableSpace诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "sqFullTimeout_:%u s",
                // [中文导读] [AllReduce逐行 S84] 为前述多行表达式补入`__func__, sqId_, sqHead_, sqTail_, availableSpace, pendingSqeCnt, sqFullTimeout_)`（硬件SQ编号、软件保存的RTSQ头槽位、软件保存的RTSQ尾槽位、RTSQ可用槽位数、本地待提交SQE条数、SQ反压等待超时秒数）；本行是参数/结构化初始化续行。
                __func__, sqId_, sqHead_, sqTail_, availableSpace, pendingSqeCnt, sqFullTimeout_);
            // [中文导读] [AllReduce逐行 S85] 设置上次状态打印时刻为/按`curTime`（当前单调时钟时刻）。
            lastPrintTime = curTime;
        // [中文导读] [AllReduce逐行 S86] 结束`if (UNLIKELY(curTime - lastPrintTime >= printInterval))`（当前单调时钟时刻、上次状态打印时刻、状态日志打印间隔）分支/循环；控制流返回外层。
        }

        // [中文导读] [AllReduce逐行 S88] 检查SQ等待超时与域挂起/不可用状态；传入/处理单调时钟开始时刻、当前单调时钟时刻。
        CheckLaunchTaskStatus(startTime, curTime);
// [中文导读] [AllReduce逐行 S89] 编译条件`ifdef CCL_KERNEL_AICPU`限定后续实现，区分Host/设备或构建能力分支。
#ifdef CCL_KERNEL_AICPU
        // [中文导读] [AllReduce逐行 S90] 设置当前调用状态为/按`HandleDispatchAllStreams()`；在当前流反压时尝试发射上下文内其他执行流。
        HcclResult ret = HandleDispatchAllStreams();
        // [中文导读] [AllReduce逐行 S91] 仅当`(UNLIKELY(ret != HCCL_SUCCESS))`（当前调用状态）成立时进入此分支。
        if (UNLIKELY(ret != HCCL_SUCCESS)) {
            // [中文导读] [AllReduce逐行 S92] 准备`auto msg`的局部存储/结构描述，初始化方式以本行声明为准。
            auto msg
                // [中文导读] [AllReduce逐行 S93] 设置当前调用状态为/按`%d, sqId:%u, ", __func__, ret, sqId_)`（硬件SQ编号、当前调用状态）。
                = StringFormat("RtsqA5::%s HandleDispatchAllStreams failed, ret = %d, sqId:%u, ", __func__, ret, sqId_);
            // [中文导读] [AllReduce逐行 S94] 记录RtsqA5::MakeSureAvailableSpace的错误诊断；日志本身不执行传输。
            HCCL_ERROR("%s", msg.c_str());
            // [中文导读] [AllReduce逐行 S95] 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。
            THROW<InternalException>(msg);
        // [中文导读] [AllReduce逐行 S96] 结束`if (UNLIKELY(ret != HCCL_SUCCESS))`（当前调用状态）分支/循环；控制流返回外层。
        }
// [中文导读] [AllReduce逐行 S97] 结束前述编译条件控制的实现片段。
#endif
        // [中文导读] [AllReduce逐行 S98] 仅当`(checkOpExecStatusCallback_ != nullptr)`成立时进入此分支。
        if (checkOpExecStatusCallback_ != nullptr) {
            // [中文导读] [AllReduce逐行 S99] 调用通信域执行状态检查回调。
            checkOpExecStatusCallback_();
        // [中文导读] [AllReduce逐行 S100] 结束`if (checkOpExecStatusCallback_ != nullptr)`分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S101] 结束`while (availableSpace <= pendingSqeCnt)`（RTSQ可用槽位数、本地待提交SQE条数）分支/循环；控制流返回外层。
    }
// [中文导读] [AllReduce逐行 S102] 结束RtsqA5::MakeSureAvailableSpace函数体；控制流返回外层。
}

void RtsqA5::CheckLaunchTaskStatus(
    const std::chrono::steady_clock::time_point& startTime, const std::chrono::steady_clock::time_point& curTime)
{
    bool isTimeout = (sqFullTimeout_ == 0) ? false : ((curTime - startTime) >= std::chrono::seconds(sqFullTimeout_));
    // step1 检测是否launch超时，如果超时打印rtsq full的ERROR日志
    if (UNLIKELY(isTimeout)) {
        HCCL_ERROR(
            "Rtsq full, sqFullTimeout_:[%u s]. sqId:[%u], sqHead:[%u], sqTail:[%u], pendingSqeCnt:[%u]", sqFullTimeout_,
            sqId_, sqHead_, sqTail_, pendingSqeCnt);
    }

    HcclResult checkRet = (checkExecStatusCallback_ != nullptr) ? checkExecStatusCallback_(isTimeout) : HCCL_SUCCESS;
    // step2 通信域状态为HCCL_COMM_STATUS_SUSPENDING状态，则终止launch不抛异
    if (UNLIKELY(checkRet == HCCL_E_SUSPENDING)) {
        pendingSqeCnt = 0;
        return;
    }
    // step3 调用回调检查执行状态：1、如果超时，打印taskException；2、如果通信域不可用，终止launch
    if (UNLIKELY(isTimeout || checkRet != HCCL_SUCCESS)) {
        THROW<InternalException>(
            StringFormat("[%s]stop launch Task, isTimeout[%d], checkRet[%d]", __func__, isTimeout, checkRet));
    }
}

// [中文导读] [AllReduce逐行 S128] RtsqA5::CopySqeBufToSq的接口声明：待复制的SQE源缓存；这些参数属于本函数调用边界。
void RtsqA5::CopySqeBufToSq(u8* sqeBuf) const
// [中文导读] [AllReduce逐行 S129] 进入RtsqA5::CopySqeBufToSq函数体：按环队列是否回绕一次或两次复制 SQE 缓存到 RTSQ VA。
{
    // [中文导读] [AllReduce逐行 S130] 设置本次RTSQ目标地址为/按`ReinterpretAs<u8*>(sqBaseAddr_) + sqTail_ * RTSQ_SQE_SIZE`（RTSQ映射基地址、软件保存的RTSQ尾槽位）。
    u8* sqCurrAddr = ReinterpretAs<u8*>(sqBaseAddr_) + sqTail_ * RTSQ_SQE_SIZE;
    // [中文导读] [AllReduce逐行 S131] 仅当`(sqTail_ >= sqHead_)`（软件保存的RTSQ尾槽位、软件保存的RTSQ头槽位）成立时进入此分支。
    if (sqTail_ >= sqHead_) {
        // [中文导读] [AllReduce逐行 S132] 设置RTSQ尾部到数组末尾的剩余槽数为/按`sqDepth_ - sqTail_`（RTSQ或UB SQ深度、软件保存的RTSQ尾槽位）。
        u32 depthLeft = sqDepth_ - sqTail_;
        // [中文导读] [AllReduce逐行 S133] 仅当`(pendingSqeCnt <= depthLeft)`（本地待提交SQE条数、RTSQ尾部到数组末尾的剩余槽数）成立时进入此分支。
        if (pendingSqeCnt <= depthLeft) { // 没有回绕
            // [中文导读] [AllReduce逐行 S134] 记录RtsqA5::CopySqeBufToSq的状态/性能诊断；日志本身不执行传输。
            HCCL_INFO(
                // [中文导读] [AllReduce逐行 S135] 为当前RtsqA5::CopySqeBufToSq诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "RtsqA5::%s copy sqe from sqe buffer, sqId_: %u, streamId_: %u, cur head: %u, cur tail: %u, size: %u, "
                // [中文导读] [AllReduce逐行 S136] 为当前RtsqA5::CopySqeBufToSq诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "depth remain: %u",
                // [中文导读] [AllReduce逐行 S137] 为前述多行表达式补入`__func__, sqId_, streamId_, sqHead_, sqTail_, pendingSqeCnt, depthLeft)`（硬件SQ编号、运行时执行流编号、软件保存的RTSQ头槽位、软件保存的RTSQ尾槽位、本地待提交SQE条数、RTSQ尾部到数组末尾的剩余槽数）；本行是参数/结构化初始化续行。
                __func__, sqId_, streamId_, sqHead_, sqTail_, pendingSqeCnt, depthLeft);
            // [中文导读] [AllReduce逐行 S138] 设置当前调用状态为/按`memcpy_sp(sqCurrAddr, pendingSqeCnt * AC_SQE_SIZE, sqeBuf, pendingSqeCnt * RTSQ_SQE_SIZE)`（本次RTSQ目标地址、本地待提交SQE条数、待复制的SQE源缓存）；把WQE/SQE数据复制到设备映射队列内存，失败抛异常。
            int ret = memcpy_sp(sqCurrAddr, pendingSqeCnt * AC_SQE_SIZE, sqeBuf, pendingSqeCnt * RTSQ_SQE_SIZE);
            // [中文导读] [AllReduce逐行 S139] 仅当`(UNLIKELY(ret != 0))`（当前调用状态）成立时进入此分支。
            if (UNLIKELY(ret != 0)) {
                // [中文导读] [AllReduce逐行 S140] 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。
                THROW<InternalException>(StringFormat("RtsqA5::%s sqe memcpy_sp failed, ret = %d", __func__, ret));
            // [中文导读] [AllReduce逐行 S141] 结束`if (UNLIKELY(ret != 0))`（当前调用状态）分支/循环；控制流返回外层。
            }
        // [中文导读] [AllReduce逐行 S142] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
        } else {
            // [中文导读] [AllReduce逐行 S143] 记录RtsqA5::CopySqeBufToSq的状态/性能诊断；日志本身不执行传输。
            HCCL_INFO(
                // [中文导读] [AllReduce逐行 S144] 为当前RtsqA5::CopySqeBufToSq诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "RtsqA5::%s copy sqe twice, sqId_: %u, streamId_: %u, cur head: %u, cur tail: %u, cnt: %u, depth "
                // [中文导读] [AllReduce逐行 S145] 为当前RtsqA5::CopySqeBufToSq诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "remain: %u",
                // [中文导读] [AllReduce逐行 S146] 为前述多行表达式补入`__func__, sqId_, streamId_, sqHead_, sqTail_, pendingSqeCnt, depthLeft)`（硬件SQ编号、运行时执行流编号、软件保存的RTSQ头槽位、软件保存的RTSQ尾槽位、本地待提交SQE条数、RTSQ尾部到数组末尾的剩余槽数）；本行是参数/结构化初始化续行。
                __func__, sqId_, streamId_, sqHead_, sqTail_, pendingSqeCnt, depthLeft);
            // 先拷贝rtsq里剩余空间大小
            // [中文导读] [AllReduce逐行 S148] 设置当前调用状态为/按`memcpy_sp(sqCurrAddr, depthLeft * AC_SQE_SIZE, sqeBuf, depthLeft * RTSQ_SQE_SIZE)`（本次RTSQ目标地址、RTSQ尾部到数组末尾的剩余槽数、待复制的SQE源缓存）；把WQE/SQE数据复制到设备映射队列内存，失败抛异常。
            int ret = memcpy_sp(sqCurrAddr, depthLeft * AC_SQE_SIZE, sqeBuf, depthLeft * RTSQ_SQE_SIZE);
            // [中文导读] [AllReduce逐行 S149] 仅当`(ret != 0)`（当前调用状态）成立时进入此分支。
            if (ret != 0) {
                // [中文导读] [AllReduce逐行 S150] 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。
                THROW<InternalException>(
                    // [中文导读] [AllReduce逐行 S151] 为组装带上下文的错误或状态文本补入`StringFormat("RtsqA5::%s rtsq remaining space memcpy_sp failed, ret = %d", __func__, ret))`（当前调用状态）；本行是参数/结构化初始化续行。
                    StringFormat("RtsqA5::%s rtsq remaining space memcpy_sp failed, ret = %d", __func__, ret));
            // [中文导读] [AllReduce逐行 S152] 结束`if (ret != 0)`（当前调用状态）分支/循环；控制流返回外层。
            }
            // 拷贝剩余sqe
            // [中文导读] [AllReduce逐行 S154] 设置当前调用状态为/按`memcpy_sp(`；把WQE/SQE数据复制到设备映射队列内存，失败抛异常。
            ret = memcpy_sp(
                // [中文导读] [AllReduce逐行 S155] 为把WQE/SQE数据复制到设备映射队列内存，失败抛异常补入`ReinterpretAs<u8*>(sqBaseAddr_), sqHead_ * RTSQ_SQE_SIZE, sqeBuf + depthLeft * RTSQ_SQE_SIZE,`（RTSQ映射基地址、软件保存的RTSQ头槽位、待复制的SQE源缓存、RTSQ尾部到数组末尾的剩余槽数）；本行是参数/结构化初始化续行。
                ReinterpretAs<u8*>(sqBaseAddr_), sqHead_ * RTSQ_SQE_SIZE, sqeBuf + depthLeft * RTSQ_SQE_SIZE,
                // [中文导读] [AllReduce逐行 S156] 为把WQE/SQE数据复制到设备映射队列内存，失败抛异常补入`(pendingSqeCnt - depthLeft) * AC_SQE_SIZE)`（本地待提交SQE条数、RTSQ尾部到数组末尾的剩余槽数）；本行是参数/结构化初始化续行。
                (pendingSqeCnt - depthLeft) * AC_SQE_SIZE);
            // [中文导读] [AllReduce逐行 S157] 仅当`(UNLIKELY(ret != 0))`（当前调用状态）成立时进入此分支。
            if (UNLIKELY(ret != 0)) {
                // [中文导读] [AllReduce逐行 S158] 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。
                THROW<InternalException>(
                    // [中文导读] [AllReduce逐行 S159] 为组装带上下文的错误或状态文本补入`StringFormat("RtsqA5::%s remaining sqe memcpy_sp failed, ret = %d", __func__, ret))`（当前调用状态）；本行是参数/结构化初始化续行。
                    StringFormat("RtsqA5::%s remaining sqe memcpy_sp failed, ret = %d", __func__, ret));
            // [中文导读] [AllReduce逐行 S160] 结束`if (UNLIKELY(ret != 0))`（当前调用状态）分支/循环；控制流返回外层。
            }
        // [中文导读] [AllReduce逐行 S161] 结束`if (pendingSqeCnt <= depthLeft)`（本地待提交SQE条数、RTSQ尾部到数组末尾的剩余槽数）分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S162] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] [AllReduce逐行 S163] 记录RtsqA5::CopySqeBufToSq的状态/性能诊断；日志本身不执行传输。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S164] 为当前RtsqA5::CopySqeBufToSq诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
            "RtsqA5::%s copy sqe from sqe buffer, tail < head, sqId_: %u, streamId_: %u, cur head: %u, cur tail: %u, "
            // [中文导读] [AllReduce逐行 S165] 为当前RtsqA5::CopySqeBufToSq诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
            "size: %u",
            // [中文导读] [AllReduce逐行 S166] 为前述多行表达式补入`__func__, sqId_, streamId_, sqHead_, sqTail_, pendingSqeCnt)`（硬件SQ编号、运行时执行流编号、软件保存的RTSQ头槽位、软件保存的RTSQ尾槽位、本地待提交SQE条数）；本行是参数/结构化初始化续行。
            __func__, sqId_, streamId_, sqHead_, sqTail_, pendingSqeCnt);
        // [中文导读] [AllReduce逐行 S167] 设置当前调用状态为/按`memcpy_sp(sqCurrAddr, pendingSqeCnt * AC_SQE_SIZE, sqeBuf, pendingSqeCnt * RTSQ_SQE_SIZE)`（本次RTSQ目标地址、本地待提交SQE条数、待复制的SQE源缓存）；把WQE/SQE数据复制到设备映射队列内存，失败抛异常。
        int ret = memcpy_sp(sqCurrAddr, pendingSqeCnt * AC_SQE_SIZE, sqeBuf, pendingSqeCnt * RTSQ_SQE_SIZE);
        // [中文导读] [AllReduce逐行 S168] 仅当`(UNLIKELY(ret != 0))`（当前调用状态）成立时进入此分支。
        if (UNLIKELY(ret != 0)) {
            // [中文导读] [AllReduce逐行 S169] 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。
            THROW<InternalException>(StringFormat("RtsqA5::%s sqe memcpy_sp failed, ret = %d", __func__, ret));
        // [中文导读] [AllReduce逐行 S170] 结束`if (UNLIKELY(ret != 0))`（当前调用状态）分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S171] 结束`if (sqTail_ >= sqHead_)`（软件保存的RTSQ尾槽位、软件保存的RTSQ头槽位）分支/循环；控制流返回外层。
    }
// [中文导读] [AllReduce逐行 S172] 结束RtsqA5::CopySqeBufToSq函数体；控制流返回外层。
}

void RtsqA5::PreLaunchSqeForCache(bool& needCacheTask)
{
    // 校验needCacheTaskCallback_
    // 注意: A5新流程下needCacheTaskCallback_一定非空; 但A5老流程下不支持aicpu task cache, needCacheTaskCallback_为空;
    //     为避免A5老流程报错, 这里为空时跳过执行而非报错
    needCacheTask = false;
    if (UNLIKELY(needCacheTaskCallback_ == nullptr)) {
        HCCL_WARNING("[RtsqA5][PreLaunchSqeForCache] needCacheTaskCallback_ is null, keep needCacheTask as false");
    } else {
        needCacheTask = needCacheTaskCallback_();
    }
}

void RtsqA5::PostLaunchSqeForCache()
{
    // 注意: 只有needCacheTask为true时才调用PostLaunchSqeForCache, 此时一定是A5新流程, 因此addSqeArrayCallback_一定非空
    if (UNLIKELY(aicpuTsThreadPtr_ == nullptr)) {
        THROW<InternalException>("[RtsqA5][PostLaunchSqeForCache] aicpuTsThreadPtr_ is null");
    }
    if (UNLIKELY(addSqeArrayCallback_ == nullptr)) {
        THROW<InternalException>("[RtsqA5][PostLaunchSqeForCache] addSqeArrayCallback_ is null");
    }
    HcclResult ret = addSqeArrayCallback_(this, aicpuTsThreadPtr_, pendingSqeCnt, locBuf, streamId_);
    if (UNLIKELY(ret != HCCL_SUCCESS)) {
        THROW<InternalException>("[RtsqA5][PostLaunchSqeForCache] addSqeArrayCallback_ failed, ret %d", ret);
    }
}

// 向芯片RTSQ VA中写入 SQE，并触发芯片执行
// [中文导读] [AllReduce逐行 S203] RtsqA5::LaunchTask的接口声明：确保空间并复制本地 SQE 到设备 RTSQ，更新硬件 SQ 尾触发执行，按需保存任务缓存；这些参数属于本函数调用边界。
void RtsqA5::LaunchTask()
// [中文导读] [AllReduce逐行 S204] 进入RtsqA5::LaunchTask函数体：确保空间并复制本地 SQE 到设备 RTSQ，更新硬件 SQ 尾触发执行，按需保存任务缓存。
{
    // [中文导读] [AllReduce逐行 S205] 记录RtsqA5::LaunchTask的状态/性能诊断，字段包含本地待提交SQE条数；日志本身不执行传输。
    HCCL_INFO("RtsqA5::%s: START, pendingSqeCnt[%u]", __func__, pendingSqeCnt);
    // [中文导读] [AllReduce逐行 S206] 仅当`(pendingSqeCnt == 0)`（本地待提交SQE条数）成立时进入此分支。
    if (pendingSqeCnt == 0) { // 没有SQE ，直接返回
        // [中文导读] [AllReduce逐行 S207] 记录RtsqA5::LaunchTask的状态/性能诊断，字段包含本地待提交SQE条数；日志本身不执行传输。
        HCCL_INFO("RtsqA5::%s: pendingSqeCnt is %u, return", __func__, pendingSqeCnt);
        // [中文导读] [AllReduce逐行 S208] 结束本次void调用，当前路径不再继续下发后续操作。
        return;
    // [中文导读] [AllReduce逐行 S209] 结束`if (pendingSqeCnt == 0)`（本地待提交SQE条数）分支/循环；控制流返回外层。
    }
    // 确保 rtsq 有足够空间放pending SQE
    // [中文导读] [AllReduce逐行 S211] 在反压时等待RTSQ可用空间并检查超时/域状态。
    MakeSureAvailableSpace();

    // [中文导读] [AllReduce逐行 S213] 仅当`(pendingSqeCnt == 0)`（本地待提交SQE条数）成立时进入此分支。
    if (pendingSqeCnt == 0) {
        // [中文导读] [AllReduce逐行 S214] 结束本次void调用，当前路径不再继续下发后续操作。
        return;
    // [中文导读] [AllReduce逐行 S215] 结束`if (pendingSqeCnt == 0)`（本地待提交SQE条数）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S217] 设置是否记录本轮任务缓存为/按`false`。
    bool needCacheTask = false;
    // [中文导读] [AllReduce逐行 S218] 查询本轮是否需要保存SQE缓存；传入/处理是否记录本轮任务缓存。
    PreLaunchSqeForCache(needCacheTask);
    // localBuffer拷贝到 RTSQ
    // [中文导读] [AllReduce逐行 S220] 按RTSQ环形槽位复制本地SQE缓存到硬件映射内存；传入/处理本地SQE待提交缓存。
    CopySqeBufToSq(locBuf);

    // 正常展开按需打印SQE
    // [中文导读] [AllReduce逐行 S223] 仅当`((UNLIKELY(GetPlfDebugConfigValue() & PLF_TASK)) || UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO)))`成立时进入此分支；调用GetPlfDebugConfigValue, HcclCheckLogLevel。
    if ((UNLIKELY(GetPlfDebugConfigValue() & PLF_TASK)) || UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
        // [中文导读] [AllReduce逐行 S224] 记录RtsqA5::LaunchTask的状态/性能诊断；日志本身不执行传输。
        PLF_CONFIG_INFO(
            // [中文导读] [AllReduce逐行 S225] 为前述多行表达式补入`PLF_TASK, "[RtsqA5][LaunchTask] dump %llu generated SQEs in stream[%u]", pendingSqeCnt, streamId_)`（本地待提交SQE条数、运行时执行流编号）；本行是参数/结构化初始化续行。
            PLF_TASK, "[RtsqA5][LaunchTask] dump %llu generated SQEs in stream[%u]", pendingSqeCnt, streamId_);

        // [中文导读] [AllReduce逐行 S227] 设置当前调用状态为/按`HCCL_SUCCESS`。
        int ret = HCCL_SUCCESS;
        // [中文导读] [AllReduce逐行 S228] 设置待打印的SQE指针为/按`locBuf`（本地SQE待提交缓存）。
        uint8_t* sqePtr = locBuf;
        // [中文导读] [AllReduce逐行 S229] 按`(size_t sqeIdx = 0; sqeIdx < pendingSqeCnt; sqeIdx++)`（当前SQE下标、本地待提交SQE条数）遍历本批条目/分片；各次处理保持数组对应关系。
        for (size_t sqeIdx = 0; sqeIdx < pendingSqeCnt; sqeIdx++) {
            // [中文导读] [AllReduce逐行 S230] 记录RtsqA5::LaunchTask的状态/性能诊断，字段包含当前SQE下标、运行时执行流编号；日志本身不执行传输。
            PLF_CONFIG_INFO(PLF_TASK, "[RtsqA5][LaunchTask] %uth generated SQE in stream[%u]", sqeIdx, streamId_);
            // [中文导读] [AllReduce逐行 S231] 设置当前调用状态为/按`hcomm::AicpuTaskUtils::DumpSqeContent(sqePtr)`（待打印的SQE指针）；调用hcomm::AicpuTaskUtils::DumpSqeContent，使用待打印的SQE指针。
            ret = hcomm::AicpuTaskUtils::DumpSqeContent(sqePtr);
            // [中文导读] [AllReduce逐行 S232] 仅当`(UNLIKELY(ret != HCCL_SUCCESS))`（当前调用状态）成立时进入此分支。
            if (UNLIKELY(ret != HCCL_SUCCESS)) {
                // [中文导读] [AllReduce逐行 S233] 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。
                THROW<InternalException>(StringFormat("RtsqA5::%s DumpSqeContent failed, ret = %d", __func__, ret));
            // [中文导读] [AllReduce逐行 S234] 结束`if (UNLIKELY(ret != HCCL_SUCCESS))`（当前调用状态）分支/循环；控制流返回外层。
            }

            // [中文导读] [AllReduce逐行 S236] 增加待打印的SQE指针为/按`RTSQ_SQE_SIZE`。
            sqePtr += RTSQ_SQE_SIZE;
        // [中文导读] [AllReduce逐行 S237] 结束`for (size_t sqeIdx = 0; sqeIdx < pendingSqeCnt; sqeIdx++)`（当前SQE下标、本地待提交SQE条数）分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S238] 结束`if ((UNLIKELY(GetPlfDebugConfigValue() & PLF_TASK)) || UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO)))`分支/循环；控制流返回外层。
    }

    // 更新tail，触发芯片执行
    // [中文导读] [AllReduce逐行 S241] 设置提交后新的RTSQ尾位置为/按`(sqTail_ + pendingSqeCnt) % sqDepth_`（软件保存的RTSQ尾槽位、本地待提交SQE条数、RTSQ或UB SQ深度）。
    u32 newTail = (sqTail_ + pendingSqeCnt) % sqDepth_;
    // [中文导读] [AllReduce逐行 S242] 向驱动配置新的SQ尾，触发芯片读取提交任务；传入/处理提交后新的RTSQ尾位置。
    ConfigSqTail(newTail);
    // [中文导读] [AllReduce逐行 S243] 设置软件保存的RTSQ尾槽位为/按`newTail`（提交后新的RTSQ尾位置）。
    sqTail_ = newTail;

    // 缓存sqe
    // [中文导读] [AllReduce逐行 S246] 仅当`(needCacheTask)`（是否记录本轮任务缓存）成立时进入此分支。
    if (needCacheTask) {
        // [中文导读] [AllReduce逐行 S247] 在本轮提交后把生成的SQE数组登记到任务缓存。
        PostLaunchSqeForCache();
    // [中文导读] [AllReduce逐行 S248] 结束`if (needCacheTask)`（是否记录本轮任务缓存）分支/循环；控制流返回外层。
    }
    // 清空本地的locBuffer和sqeCnt数目
    // [中文导读] [AllReduce逐行 S250] 记录RtsqA5::LaunchTask的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S251] 为当前RtsqA5::LaunchTask诊断/异常表达式提供格式文本，将报告本地待提交SQE条数、运行时执行流编号；这一物理行没有数据搬运副作用。
        "RtsqA5::%s: END, pendingSqeCnt[%u], streamId_[%u] sqHead_[%u] sqTail_[%u]", __func__, pendingSqeCnt, streamId_,
        // [中文导读] [AllReduce逐行 S252] 为前述多行表达式补入`sqHead_, sqTail_)`（软件保存的RTSQ头槽位、软件保存的RTSQ尾槽位）；本行是参数/结构化初始化续行。
        sqHead_, sqTail_);
    // [中文导读] [AllReduce逐行 S253] 设置本地待提交SQE条数为/按`0`。
    pendingSqeCnt = 0;
    // [中文导读] [AllReduce逐行 S254] 清零本地待提交SQE缓存；显式丢弃memset_s返回值，已提交队列的执行完成由后续通知/Join观察。
    (void)memset_s(locBuf, RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT, 0, RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT); // locBuffer清零
// [中文导读] [AllReduce逐行 S255] 结束RtsqA5::LaunchTask函数体；控制流返回外层。
}

void RtsqA5::RefreshSqeHeaderTaskField(Rt91095StarsSqeHeader* sqeHeaderPtr)
{
    SetSqeHeaderTaskFields(sqeHeaderPtr, taskId_);
    SetTaskIdBySqeId();
}

// 向芯片RTSQ VA中写入aicpu task cache SQE，并触发芯片执行
void RtsqA5::LaunchNewTask(uint8_t* sqeArray, uint32_t sqeCount)
{
    // 注意: cache命中时才会调用LaunchNewTask, 此时一定不存在pending SQE
    if (UNLIKELY(pendingSqeCnt > 0)) {
        THROW<InternalException>(StringFormat(
            "RtsqA5::%s: pendingSqeCnt[%u] should be 0 when aicpu task cache hits!", __func__, pendingSqeCnt));
    }

    // 临时设置pendingSqeCnt, 用于MakeSureAvailableSpace
    pendingSqeCnt = sqeCount;

    // 确保 rtsq 有足够空间放pending SQE
    MakeSureAvailableSpace();

    // sqeArray拷贝到 RTSQ
    CopySqeBufToSq(sqeArray);

    // 更新tail，触发芯片执行
    u32 newTail = (sqTail_ + pendingSqeCnt) % sqDepth_;
    ConfigSqTail(newTail);
    sqTail_ = newTail;

    HCCL_INFO(
        "RtsqA5::%s: END, pendingSqeCnt[%u], streamId_[%u] sqHead_[%u] sqTail_[%u]", __func__, pendingSqeCnt, streamId_,
        sqHead_, sqTail_);

    // 重置pendingSqeCnt
    pendingSqeCnt = 0;
}

void RtsqA5::TryLaunchTask()
{
    if (pendingSqeCnt == 0) {
        return;
    }

    sqHead_ = QuerySqHead();
    u32 availableSpace = GetTailToHeadDist();
    if (availableSpace <= pendingSqeCnt) {
        return;
    }

    bool needCacheTask = false;
    PreLaunchSqeForCache(needCacheTask);

    CopySqeBufToSq(locBuf);

    u32 newTail = (sqTail_ + pendingSqeCnt) % sqDepth_;
    ConfigSqTail(newTail);
    sqTail_ = newTail;

    // 缓存sqe
    if (needCacheTask) {
        PostLaunchSqeForCache();
    }
    pendingSqeCnt = 0;
    (void)memset_s(locBuf, RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT, 0, RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT);
    HCCL_INFO(
        "RtsqA5::%s: END, pendingSqeCnt[%u], streamId_[%u] sqHead_[%u] sqTail_[%u]", __func__, pendingSqeCnt, streamId_,
        sqHead_, sqTail_);
}

// [中文导读] [AllReduce逐行 S326] RtsqA5::GetCurrSqeBuffer的接口声明：计算当前 SQE 的实际环队列地址并返回本地待提交缓存位置；这些参数属于本函数调用边界。
u8* RtsqA5::GetCurrSqeBuffer()
// [中文导读] [AllReduce逐行 S327] 进入RtsqA5::GetCurrSqeBuffer函数体：计算当前 SQE 的实际环队列地址并返回本地待提交缓存位置。
{
    // [中文导读] [AllReduce逐行 S328] 设置lastSqeAddr_为/按`sqBaseAddr_ + static_cast<u64>((sqTail_ + pendingSqeCnt) % sqDepth_) * RTSQ_SQE_SIZE`（RTSQ映射基地址、软件保存的RTSQ尾槽位、本地待提交SQE条数、RTSQ或UB SQ深度）。
    lastSqeAddr_ = sqBaseAddr_ + static_cast<u64>((sqTail_ + pendingSqeCnt) % sqDepth_) * RTSQ_SQE_SIZE;
    // [中文导读] [AllReduce逐行 S329] 直接返回`locBuf + pendingSqeCnt * RTSQ_SQE_SIZE`（本地SQE待提交缓存、本地待提交SQE条数）；将当前查询结果/句柄交给调用者。
    return locBuf + pendingSqeCnt * RTSQ_SQE_SIZE;
// [中文导读] [AllReduce逐行 S330] 结束RtsqA5::GetCurrSqeBuffer函数体；控制流返回外层。
}

u64 RtsqA5::GetSqeAddr() const { return lastSqeAddr_; }

// [中文导读] [AllReduce逐行 S334] RtsqA5::RefreshInfo的接口声明：推进 SQE/Task计数：EAGER可立即提交，BATCH 达内部阈值仍会提交；这些参数属于本函数调用边界。
void RtsqA5::RefreshInfo()
// [中文导读] [AllReduce逐行 S335] 进入RtsqA5::RefreshInfo函数体：推进 SQE/Task计数：EAGER可立即提交，BATCH 达内部阈值仍会提交。
{
    // [中文导读] [AllReduce逐行 S336] 按当前SQE位置刷新taskId。
    SetTaskIdBySqeId();
    // [中文导读] [AllReduce逐行 S337] 推进/回退`pendingSqeCnt++`（本地待提交SQE条数），更新当前分片、槽位或状态重试的计数。
    pendingSqeCnt++;

// [中文导读] [AllReduce逐行 S339] 编译条件`ifdef CCL_KERNEL_AICPU`限定后续实现，区分Host/设备或构建能力分支。
#ifdef CCL_KERNEL_AICPU
    // [中文导读] [AllReduce逐行 S340] 仅当`(launchFlag_ && !IsBatchLaunchMode())`成立时进入此分支；调用IsBatchLaunchMode。
    if (launchFlag_ && !IsBatchLaunchMode()) {
        // [中文导读] [AllReduce逐行 S341] 将已生成的任务提交到具体RTSQ队列，不等同于全部任务完成。
        LaunchTask();
        // [中文导读] [AllReduce逐行 S342] 结束本次void调用，当前路径不再继续下发后续操作。
        return;
    // [中文导读] [AllReduce逐行 S343] 结束`if (launchFlag_ && !IsBatchLaunchMode())`分支/循环；控制流返回外层。
    }
// [中文导读] [AllReduce逐行 S344] 结束前述编译条件控制的实现片段。
#endif

    // [中文导读] [AllReduce逐行 S346] 仅当`(pendingSqeCnt != PER_LAUNCH_SQE_CNT)`（本地待提交SQE条数）成立时进入此分支。
    if (pendingSqeCnt != PER_LAUNCH_SQE_CNT) {
        // [中文导读] [AllReduce逐行 S347] 结束本次void调用，当前路径不再继续下发后续操作。
        return;
    // [中文导读] [AllReduce逐行 S348] 结束`if (pendingSqeCnt != PER_LAUNCH_SQE_CNT)`（本地待提交SQE条数）分支/循环；控制流返回外层。
    }
    // 挂起的sqe数量为128个，则需要向芯片RTSQ中写入task
    // [中文导读] [AllReduce逐行 S350] 将已生成的任务提交到具体RTSQ队列，不等同于全部任务完成。
    LaunchTask();
// [中文导读] [AllReduce逐行 S351] 结束RtsqA5::RefreshInfo函数体；控制流返回外层。
}

void RtsqA5::NotifyWait(u32 notifyId) { NotifyWait(notifyId, GetKernelExecTimeoutFromEnvConfig()); }

// [中文导读] [AllReduce逐行 S355] RtsqA5::NotifyWait的接口声明：硬件通知ID、超时秒数；这些参数属于本函数调用边界。
void RtsqA5::NotifyWait(u32 notifyId, u32 timeout)
// [中文导读] [AllReduce逐行 S356] 进入RtsqA5::NotifyWait函数体：构造有秒级超时的通知等待 SQE，再更新待提交计数。
{
    // [中文导读] [AllReduce逐行 S357] 在本地SQE缓存编码通知ID、超时及流/任务ID；取得下一个本地SQE缓存位置并记录对应RTSQ目标地址；传入/处理运行时执行流编号、队列当前任务编号、硬件通知ID、超时秒数。
    BuildA5SqeNotifyWait(streamId_, taskId_, notifyId, timeout, GetCurrSqeBuffer());
    // [中文导读] [AllReduce逐行 S358] 记录RtsqA5::NotifyWait的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S359] 为当前RtsqA5::NotifyWait诊断/异常表达式提供格式文本，将报告运行时执行流编号、队列当前任务编号、硬件通知ID；这一物理行没有数据搬运副作用。
        "RtsqA5::NotifyWait: streamId %u, taskId %u, notifyId %u, timeout[%u s]", streamId_, taskId_, notifyId,
        // [中文导读] [AllReduce逐行 S360] 为前述多行表达式补入`timeout)`（超时秒数）；本行是参数/结构化初始化续行。
        timeout);
    // [中文导读] [AllReduce逐行 S361] 推进任务编号和待提交数，EAGER立即提交或达到内部阈值提交。
    RefreshInfo();
// [中文导读] [AllReduce逐行 S362] 结束RtsqA5::NotifyWait函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S364] RtsqA5::NotifyRecordLoc的接口声明：硬件通知ID；这些参数属于本函数调用边界。
void RtsqA5::NotifyRecordLoc(u32 notifyId)
// [中文导读] [AllReduce逐行 S365] 进入RtsqA5::NotifyRecordLoc函数体：构造本地通知记录 SQE并更新提交状态。
{
    // [中文导读] [AllReduce逐行 S366] 在本地SQE缓存编码本地通知记录；取得下一个本地SQE缓存位置并记录对应RTSQ目标地址；传入/处理运行时执行流编号、队列当前任务编号、硬件通知ID。
    BuildA5SqeNotifyRecord(streamId_, taskId_, notifyId, GetCurrSqeBuffer());
    // [中文导读] [AllReduce逐行 S367] 记录RtsqA5::NotifyRecordLoc的状态/性能诊断，字段包含运行时执行流编号、队列当前任务编号、硬件通知ID；日志本身不执行传输。
    HCCL_INFO("RtsqA5::NotifyRecordLoc: streamId %u, taskId %u, notifyId %u", streamId_, taskId_, notifyId);
    // [中文导读] [AllReduce逐行 S368] 推进任务编号和待提交数，EAGER立即提交或达到内部阈值提交。
    RefreshInfo();
// [中文导读] [AllReduce逐行 S369] 结束RtsqA5::NotifyRecordLoc函数体；控制流返回外层。
}

void RtsqA5::Cnt1toNNotifyWait(u32 notifyId, u32 value)
{
    BuildA5SqeCnt1toNNotifyWait(streamId_, taskId_, notifyId, value, GetCurrSqeBuffer());
    HCCL_INFO("RtsqA5::Cnt1toNNotifyWait: streamId %u, taskId %u, notifyId %u", streamId_, taskId_, notifyId);
    RefreshInfo();
}

void RtsqA5::Cnt1toNNotifyRecord(u32 notifyId, u32 value)
{
    BuildA5SqeCnt1toNNotifyRecord(streamId_, taskId_, notifyId, value, GetCurrSqeBuffer());
    HCCL_INFO("RtsqA5::Cnt1toNNotifyRecord: streamId %u, taskId %u, notifyId %u", streamId_, taskId_, notifyId);
    RefreshInfo();
}

void RtsqA5::CntNto1NotifyWait(u32 notifyId, u32 value)
{
    BuildA5SqeCntNto1NotifyWait(streamId_, taskId_, notifyId, value, GetCurrSqeBuffer());
    HCCL_INFO("RtsqA5::CntNto1NotifyWait: streamId %u, taskId %u, notifyId %u", streamId_, taskId_, notifyId);
    RefreshInfo();
}

void RtsqA5::CntNto1NotifyRecord(u32 notifyId, u32 value)
{
    BuildA5SqeCntNto1NotifyRecord(streamId_, taskId_, notifyId, value, GetCurrSqeBuffer());
    HCCL_INFO("RtsqA5::CntNto1NotifyRecord: streamId %u, taskId %u, notifyId %u", streamId_, taskId_, notifyId);
    RefreshInfo();
}

// [中文导读] [AllReduce逐行 S399] RtsqA5::SdmaCopy的接口声明：源地址整数表示、目标地址整数表示、字节容量或单片字节数、SDMA分区标识；这些参数属于本函数调用边界。
void RtsqA5::SdmaCopy(u64 srcAddr, u64 dstAddr, u32 size, u32 partId)
// [中文导读] [AllReduce逐行 S400] 进入RtsqA5::SdmaCopy函数体：构造无归约 SDMA SQE并更新提交状态。
{
    // 不带reduce的拷贝，opcode填0
    // [中文导读] [AllReduce逐行 S402] 显式忽略`partId`（SDMA分区标识），该接口参数/调用结果在此实现中未参与后续计算。
    (void)partId;
    // [中文导读] [AllReduce逐行 S403] 在本地SQE缓存编码SDMA源/目标、字节长度与归约码；取得下一个本地SQE缓存位置并记录对应RTSQ目标地址；传入/处理运行时执行流编号、队列当前任务编号、目标地址整数表示、源地址整数表示、字节容量或单片字节数。
    BuildA5SqeSdmaCopy(streamId_, taskId_, dstAddr, srcAddr, size, RTSQ_A5_PART_ID, 0, GetCurrSqeBuffer());
    // [中文导读] [AllReduce逐行 S404] 记录RtsqA5::SdmaCopy的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S405] 为当前RtsqA5::SdmaCopy诊断/异常表达式提供格式文本，将报告运行时执行流编号、队列当前任务编号；这一物理行没有数据搬运副作用。
        "RtsqA5::SdmaCopy: streamId %u, taskId %u, srcAddr 0x%llx, dstAddr 0x%llx, size %u", streamId_, taskId_,
        // [中文导读] [AllReduce逐行 S406] 为前述多行表达式补入`srcAddr, dstAddr, size)`（源地址整数表示、目标地址整数表示、字节容量或单片字节数）；本行是参数/结构化初始化续行。
        srcAddr, dstAddr, size);
    // [中文导读] [AllReduce逐行 S407] 推进任务编号和待提交数，EAGER立即提交或达到内部阈值提交。
    RefreshInfo();
// [中文导读] [AllReduce逐行 S408] 结束RtsqA5::SdmaCopy函数体；控制流返回外层。
}

const std::unordered_map<ReduceOp, RtStarsMemcpyAsyncOperationKind, EnumClassHash> ReduceOpToStarsOpKindMap
    = {{ReduceOp::SUM, RtStarsMemcpyAsyncOperationKind::RT_STARS_MEMCPY_ASYNC_OP_KIND_ADD},
       {ReduceOp::MAX, RtStarsMemcpyAsyncOperationKind::RT_STARS_MEMCPY_ASYNC_OP_KIND_MAX},
       {ReduceOp::MIN, RtStarsMemcpyAsyncOperationKind::RT_STARS_MEMCPY_ASYNC_OP_KIND_MIN},
       {ReduceOp::EQUAL, RtStarsMemcpyAsyncOperationKind::RT_STARS_MEMCPY_ASYNC_OP_KIND_EQUAL}};

const std::unordered_map<DataType, RtStarsMemcpyAsyncDataType, EnumClassHash> DataTypeToStarsDataTypeMap
    = {{DataType::INT8, RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT8},
       {DataType::INT16, RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT16},
       {DataType::INT32, RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT32},
       {DataType::FP16, RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_FP16},
       {DataType::FP32, RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_FP32},
       {DataType::BFP16, RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_BFP16}};

// [中文导读] [AllReduce逐行 S424] RtsqA5::SdmaReduce的接口声明：源地址整数表示、目标地址整数表示、字节容量或单片字节数、SDMA分区标识、底层归约类型/操作描述；这些参数属于本函数调用边界。
void RtsqA5::SdmaReduce(u64 srcAddr, u64 dstAddr, u32 size, u32 partId, const ReduceIn& reduceIn)
// [中文导读] [AllReduce逐行 S425] 进入RtsqA5::SdmaReduce函数体：转换归约/类型编码后构造 SDMA SQE并更新提交状态。
{
    // [中文导读] [AllReduce逐行 S426] 显式忽略`partId`（SDMA分区标识），该接口参数/调用结果在此实现中未参与后续计算。
    (void)partId;
    // [中文导读] [AllReduce逐行 S427] 仅当`(UNLIKELY(`成立时进入此分支。
    if (UNLIKELY(
            // [中文导读] [AllReduce逐行 S428] 补全本分支/循环判断的`ReduceOpToStarsOpKindMap.find(reduceIn.reduceOp) == ReduceOpToStarsOpKindMap.end()`（底层归约类型/操作描述的reduceOp字段），和前面条件共同决定是否进入后续路径。
            ReduceOpToStarsOpKindMap.find(reduceIn.reduceOp) == ReduceOpToStarsOpKindMap.end()
            // [中文导读] [AllReduce逐行 S429] 补全本分支/循环判断的`|| DataTypeToStarsDataTypeMap.find(reduceIn.dataType) == DataTypeToStarsDataTypeMap.end()))`（底层归约类型/操作描述的dataType字段），和前面条件共同决定是否进入后续路径。
            || DataTypeToStarsDataTypeMap.find(reduceIn.dataType) == DataTypeToStarsDataTypeMap.end())) {
        // [中文导读] [AllReduce逐行 S430] 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。
        THROW<InternalException>(StringFormat(
            // [中文导读] [AllReduce逐行 S431] 为当前RtsqA5::SdmaReduce诊断/异常表达式提供格式文本，将报告底层归约类型/操作描述的reduceOp.Describe字段；这一物理行没有数据搬运副作用。
            "Sdma does not support reduceOp %s dataType %s", reduceIn.reduceOp.Describe().c_str(),
            // [中文导读] [AllReduce逐行 S432] 为组装带上下文的错误或状态文本；取得对象诊断文本用于日志补入`reduceIn.dataType.Describe().c_str()))`（底层归约类型/操作描述的dataType.Describe字段）；本行是参数/结构化初始化续行。
            reduceIn.dataType.Describe().c_str()));
    // [中文导读] [AllReduce逐行 S433] 结束当前局部作用域；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S435] 设置u8 op为/按`static_cast<u8>(ReduceOpToStarsOpKindMap.at(reduceIn.reduceOp))`（底层归约类型/操作描述的reduceOp字段）；调用at，使用底层归约类型/操作描述的reduceOp字段。
    u8 op = static_cast<u8>(ReduceOpToStarsOpKindMap.at(reduceIn.reduceOp));
    // [中文导读] [AllReduce逐行 S436] 设置Thread类型为/按`static_cast<u8>(DataTypeToStarsDataTypeMap.at(reduceIn.dataType))`（底层归约类型/操作描述的dataType字段）；调用at，使用底层归约类型/操作描述的dataType字段。
    u8 type = static_cast<u8>(DataTypeToStarsDataTypeMap.at(reduceIn.dataType));

    // [中文导读] [AllReduce逐行 S438] 在本地SQE缓存编码SDMA源/目标、字节长度与归约码；取得下一个本地SQE缓存位置并记录对应RTSQ目标地址；传入/处理运行时执行流编号、队列当前任务编号、目标地址整数表示、源地址整数表示、字节容量或单片字节数、Thread类型。
    BuildA5SqeSdmaCopy(streamId_, taskId_, dstAddr, srcAddr, size, RTSQ_A5_PART_ID, (op | type), GetCurrSqeBuffer());
    // [中文导读] [AllReduce逐行 S439] 记录RtsqA5::SdmaReduce的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S440] 为当前RtsqA5::SdmaReduce诊断/异常表达式提供格式文本，将报告运行时执行流编号、队列当前任务编号；这一物理行没有数据搬运副作用。
        "RtsqA5::SdmaReduce: streamId %u, taskId %u, srcAddr 0x%llx, dstAddr 0x%llx, size %u", streamId_, taskId_,
        // [中文导读] [AllReduce逐行 S441] 为前述多行表达式补入`srcAddr, dstAddr, size)`（源地址整数表示、目标地址整数表示、字节容量或单片字节数）；本行是参数/结构化初始化续行。
        srcAddr, dstAddr, size);
    // [中文导读] [AllReduce逐行 S442] 推进任务编号和待提交数，EAGER立即提交或达到内部阈值提交。
    RefreshInfo();
// [中文导读] [AllReduce逐行 S443] 结束RtsqA5::SdmaReduce函数体；控制流返回外层。
}

bool RtsqA5::IsRtsqQueueSpaceSufficient()
{
    // 判断逻辑与rtsq内部保持一致，rtsq剩余空间需要大于（rtsq挂起的任务数量+本次任务）
    u32 availableSpace = GetTailToHeadDist();
    if (availableSpace > pendingSqeCnt + 1) {
        return true;
    }

    // 否则的话，需要再次查询一次head，确认是否是因为head没有更新导致空间不足，如果查询后空间仍然不足，则返回false
    sqHead_ = QuerySqHead();
    availableSpace = GetTailToHeadDist();

    return (availableSpace > pendingSqeCnt + 1);
}

HcclResult RtsqA5::SetPreStreamSyncReady()
{
    isPreStreamSync = true;
    return HCCL_SUCCESS;
}

HcclResult RtsqA5::SetPreStreamSyncFin()
{
    isPreStreamSync = false;
    return HCCL_SUCCESS;
}

bool RtsqA5::GetPreStreamSyncStatus() { return isPreStreamSync; }

// [中文导读] [AllReduce逐行 S474] RtsqA5::UbDbSend的接口声明：UB jetty的die/function/jetty标识、16位UB jetty生产指针；这些参数属于本函数调用边界。
void RtsqA5::UbDbSend(const UbJettyLiteId& jettyLiteId, u16 piValue)
// [中文导读] [AllReduce逐行 S475] 进入RtsqA5::UbDbSend函数体：把 jetty 与16位 UB PI编码成 RTSQ Doorbell SQE，再更新提交状态。
{
    // piValue需要使用u16数据类型，保证自然增长，用于判断是否翻转
    // [中文导读] [AllReduce逐行 S477] 在本地SQE缓存编码UB jetty与生产指针Doorbell；取得下一个本地SQE缓存位置并记录对应RTSQ目标地址；传入/处理运行时执行流编号、队列当前任务编号、UB jetty的die/function/jetty标识、16位UB jetty生产指针。
    BuildA5SqeUbDbSend(streamId_, taskId_, jettyLiteId, piValue, GetCurrSqeBuffer());
    // [中文导读] [AllReduce逐行 S478] 记录RtsqA5::UbDbSend的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S479] 为当前RtsqA5::UbDbSend诊断/异常表达式提供格式文本，将报告运行时执行流编号、队列当前任务编号、16位UB jetty生产指针；这一物理行没有数据搬运副作用。
        "RtsqA5::UbDbSend: streamId %u, taskId %u, piValue(UbPi):%u, SqTail(Rtsq Pi):%u", streamId_, taskId_, piValue,
        // [中文导读] [AllReduce逐行 S480] 为前述多行表达式补入`sqTail_)`（软件保存的RTSQ尾槽位）；本行是参数/结构化初始化续行。
        sqTail_);
    // [中文导读] [AllReduce逐行 S481] 推进任务编号和待提交数，EAGER立即提交或达到内部阈值提交。
    RefreshInfo();
// [中文导读] [AllReduce逐行 S482] 结束RtsqA5::UbDbSend函数体；控制流返回外层。
}

void RtsqA5::RdmaDbSend(const uint64_t& dbAddr, const uint64_t& dbValue)
{
    BuildA5SqeRdmaDbSend(streamId_, taskId_, dbAddr, dbValue, GetCurrSqeBuffer());
    HCCL_INFO(
        "RtsqA5::RdmaDbSend: RdmaDbSend streamId %u, taskId %u, Sqe: %s, dbAddr:0x%llx, dbValue:0x%llx, SqTail(Rtsq "
        "Pi):%u",
        streamId_, taskId_, Bytes2hex(GetCurrSqeBuffer(), RTSQ_SQE_SIZE).c_str(), dbAddr, dbValue, sqTail_);
    RefreshInfo();
}

void RtsqA5::CCoreNotifyWait(u64 waitAddr, u64 curTurnCntAddr, bool last)
{
    BuildA5SqeCCoreNotifyWait(streamId_, taskId_, waitAddr, curTurnCntAddr, last, GetCurrSqeBuffer());
    HCCL_INFO(
        "RtsqA5::CCoreNotifyWait: streamId %u, taskId %u, waitAddr %llu, curTurnCntAddr %llu, last %d", streamId_,
        taskId_, waitAddr, curTurnCntAddr, last);
    RefreshInfo();
}

void RtsqA5::CCoreNotifyRecord(u64 recordAddr, u64 curTurnCntAddr)
{
    BuildA5SqeCCoreNotifyRecord(streamId_, taskId_, recordAddr, curTurnCntAddr, GetCurrSqeBuffer());
    HCCL_INFO(
        "RtsqA5::CCoreNotifyRecord: streamId %u, taskId %u, recordAddr %llu, curTurnCntAddr %llu", streamId_, taskId_,
        recordAddr, curTurnCntAddr);
    RefreshInfo();
}

void RtsqA5::P2PWriteValue(u64 remoteAddr, u32 writeValue)
{
    BuildA5SqeP2pWriteValue(streamId_, taskId_, remoteAddr, writeValue, GetCurrSqeBuffer());
    HCCL_INFO(
        "RtsqA5::P2PWriteValue: streamId %u, taskId %u, remoteAddr %llu, writeValue %u", streamId_, taskId_, remoteAddr,
        writeValue);
    RefreshInfo();
}

HcclResult RtsqA5::GetLastStreamIdAndTaskId(uint16_t& streamId, uint16_t& taskId) const
{
    if (pendingSqeCnt > 0) {
        const u8* lastSqe = locBuf + (pendingSqeCnt - 1U) * RTSQ_SQE_SIZE;
        auto* sqe = ReinterpretAs<const Rt91095StarsNotifySqe*>(lastSqe);
        streamId = sqe->header.rtStreamId;
        taskId = sqe->header.taskId;
        HCCL_INFO(
            "[%s] from pending, pendingSqeCnt[%u], sqId[%u], streamId[%u], taskId[%u].", __func__, pendingSqeCnt, sqId_,
            streamId, taskId);
        return HCCL_SUCCESS;
    }
    const u32 lastIdx = (sqTail_ + sqDepth_ - 1U) % sqDepth_;
    HCCL_INFO(
        "[%s] from rtsq, sqId[%u], sqTail[%u], sqDepth[%u], lastIdx[%u].", __func__, sqId_, sqTail_, sqDepth_, lastIdx);
    return GetStreamIdAndTaskIdBySqIdx(lastIdx, streamId, taskId);
}
} // namespace Hccl
