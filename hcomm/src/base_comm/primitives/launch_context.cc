/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "launch_context.h"
#include "new/hccl_primitive_local.h"

constexpr u32 THREAD_VECTOR_DEFAULT_SIZE = 128; // 设置vector初始长度，避免频繁扩容
constexpr u32 NOTIFY_WAIT_TIMEOUT_OFFSET = 27;  // AICPU device侧notify等待超时偏移量

extern HcclResult CommTaskLaunch(ThreadHandle* threads, uint32_t threadNum); // host ffts+或aicpu stars使用"
extern HcclResult CommTaskPrepare(char* key, uint32_t keyLen);               // host ffts+使用
extern HcclResult DispatchAllStreams(const ThreadHandle* threads, uint32_t threadNum);

LaunchContext::LaunchContext() { threadVec_.reserve(THREAD_VECTOR_DEFAULT_SIZE); }

// [中文导读] [AllReduce逐行 S23] LaunchContext::HandleEagerMode的接口声明：分别提交 tag 匹配的 Thread 集合与当前非 tag Thread 列表；这些参数属于本函数调用边界。
HcclResult LaunchContext::HandleEagerMode()
// [中文导读] [AllReduce逐行 S24] 进入LaunchContext::HandleEagerMode函数体：分别提交 tag 匹配的 Thread 集合与当前非 tag Thread 列表。
{
    // 带launchTag部分
    // [中文导读] [AllReduce逐行 S26] 仅当`(!launchModeMap_.empty())`（按提交标签登记的Thread集合的empty字段）成立时进入此分支；检查容器是否没有登记项。
    if (!launchModeMap_.empty()) {
        // [中文导读] [AllReduce逐行 S27] 设置auto it为/按`launchModeMap_.find(launchTag_)`（按提交标签登记的Thread集合的find字段、当前提交标签）；调用find，使用按提交标签登记的Thread集合的find字段、当前提交标签。
        auto it = launchModeMap_.find(launchTag_);
        // [中文导读] [AllReduce逐行 S28] 仅当`(it != launchModeMap_.end())`（按提交标签登记的Thread集合的end字段）成立时进入此分支；调用end，使用按提交标签登记的Thread集合的end字段。
        if (it != launchModeMap_.end()) {
            // [中文导读] [AllReduce逐行 S29] 调用threadVec, begin, end；保持声明的局部对象用于后续处理。
            std::vector<ThreadHandle> threadVec(it->second.begin(), it->second.end());
            // [中文导读] [AllReduce逐行 S30] 提交参与批次的执行Thread任务；读取容器登记项数；返回非成功时由检查宏立即向上传递。
            CHK_RET(CommTaskLaunch(threadVec.data(), threadVec.size()));
            // [中文导读] [AllReduce逐行 S31] 记录LaunchContext::HandleEagerMode的状态/性能诊断，字段包含当前提交标签的c_str字段；日志本身不执行传输。
            HCCL_INFO("[%s] success, launchTag[%s], size[%zu]", __func__, launchTag_.c_str(), threadVec.size());
        // [中文导读] [AllReduce逐行 S32] 结束`if (it != launchModeMap_.end())`（按提交标签登记的Thread集合的end字段）分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S33] 结束`if (!launchModeMap_.empty())`（按提交标签登记的Thread集合的empty字段）分支/循环；控制流返回外层。
    }

    // 不带launchTag部分
    // [中文导读] [AllReduce逐行 S36] 仅当`(!threadVec_.empty())`（本轮BATCH参与Thread列表的empty字段）成立时进入此分支；检查容器是否没有登记项。
    if (!threadVec_.empty()) {
        // [中文导读] [AllReduce逐行 S37] 提交参与批次的执行Thread任务；读取容器登记项数；返回非成功时由检查宏立即向上传递。
        CHK_RET(CommTaskLaunch(threadVec_.data(), threadVec_.size()));
        // [中文导读] [AllReduce逐行 S38] 记录LaunchContext::HandleEagerMode的状态/性能诊断，字段包含本轮BATCH参与Thread列表的size字段；日志本身不执行传输。
        HCCL_INFO("[%s] success, size[%zu]", __func__, threadVec_.size());
    // [中文导读] [AllReduce逐行 S39] 结束`if (!threadVec_.empty())`（本轮BATCH参与Thread列表的empty字段）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S40] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S41] 结束LaunchContext::HandleEagerMode函数体；控制流返回外层。
}

HcclResult LaunchContext::HandleDispatchAllStreams()
{
    // 带launchTag部分
    if (!launchModeMap_.empty()) {
        auto it = launchModeMap_.find(launchTag_);
        if (it != launchModeMap_.end()) {
            std::vector<ThreadHandle> threadVec(it->second.begin(), it->second.end());
            CHK_RET(DispatchAllStreams(threadVec.data(), threadVec.size()));
        }
    }

    // 不带launchTag部分
    if (!threadVec_.empty()) {
        CHK_RET(DispatchAllStreams(threadVec_.data(), threadVec_.size()));
    }
    return HCCL_SUCCESS;
}

// [中文导读] [AllReduce逐行 S61] LaunchContext::HandleClear的接口声明：清除当前批的 Thread/tag 登记；非 950/960 还清除兼容任务缓存；这些参数属于本函数调用边界。
HcclResult LaunchContext::HandleClear()
// [中文导读] [AllReduce逐行 S62] 进入LaunchContext::HandleClear函数体：清除当前批的 Thread/tag 登记；非 950/960 还清除兼容任务缓存。
{
    // [中文导读] [AllReduce逐行 S63] 调用clear，使用本轮BATCH参与Thread列表的clear字段；传入/处理本轮BATCH参与Thread列表的clear字段。
    threadVec_.clear();
    // [中文导读] [AllReduce逐行 S64] 仅当`(!launchModeMap_.empty())`（按提交标签登记的Thread集合的empty字段）成立时进入此分支；检查容器是否没有登记项。
    if (!launchModeMap_.empty()) {
        // [中文导读] [AllReduce逐行 S65] 调用erase，使用按提交标签登记的Thread集合的erase字段、当前提交标签；传入/处理按提交标签登记的Thread集合的erase字段、当前提交标签。
        launchModeMap_.erase(launchTag_);
    // [中文导读] [AllReduce逐行 S66] 结束`if (!launchModeMap_.empty())`（按提交标签登记的Thread集合的empty字段）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S67] 记录LaunchContext::HandleClear的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S68] 为当前LaunchContext::HandleClear诊断/异常表达式提供格式文本，将报告当前提交标签的c_str字段、当前提交模式；这一物理行没有数据搬运副作用。
        "[%s] begin clear, launchTag[%s], launchMode[%d].", __func__, launchTag_.c_str(), static_cast<int32_t>(mode_));

    // [中文导读] [AllReduce逐行 S70] 设置设备型号为/按`DevType::DEV_TYPE_COUNT`。
    DevType devType = DevType::DEV_TYPE_COUNT;
    // [中文导读] [AllReduce逐行 S71] 读取设备型号用于新旧/协议分支选择；传入/处理设备型号。
    hrtGetDeviceType(devType);
    // [中文导读] [AllReduce逐行 S72] 仅当`(devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960)`（设备型号）成立时进入此分支。
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        // [中文导读] [AllReduce逐行 S73] 记录LaunchContext::HandleClear的状态/性能诊断；日志本身不执行传输。
        HCCL_INFO("[%s] Running on A5/A6, HcclTaskClear skipped.", __func__);
        // [中文导读] [AllReduce逐行 S74] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S75] 结束`if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960)`（设备型号）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S76] 直接返回`HcclTaskClear(launchTag_)`（当前提交标签）；清除兼容任务缓存登记。
    return HcclTaskClear(launchTag_);
// [中文导读] [AllReduce逐行 S77] 结束LaunchContext::HandleClear函数体；控制流返回外层。
}

HcclResult LaunchContext::SetNotifyWaitTimeOut(uint32_t timeout)
{
    notifyWaitTimeoutConfig_.notifyWaitTimeout = timeout;
    notifyWaitTimeoutConfig_.isSet = true;
    return HCCL_SUCCESS;
}

HcclResult LaunchContext::GetNotifyWaitTimeOut(uint32_t& timeout)
{
    timeout = notifyWaitTimeoutConfig_.notifyWaitTimeout;
#ifndef CCL_KERNEL_AICPU
    if (!notifyWaitTimeoutConfig_.isSet) {
        timeout = timeout + NOTIFY_WAIT_TIMEOUT_OFFSET;
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult LaunchContext::SetSqFullTimeOut(uint32_t timeout)
{
    sqFullTimeoutConfig_.sqFullTimeout = timeout;
    sqFullTimeoutConfig_.isSet = true;
    return HCCL_SUCCESS;
}

uint32_t LaunchContext::GetSqFullTimeOut() { return sqFullTimeoutConfig_.sqFullTimeout; }

/*
    1 AICPU_TS模式
    AICPU上执行
    告知后面的CommWrite等任务进入批量模式，（只写任务的SQE，但是不触发执行）
    举例：
    HcommSetLaunchMode("abc", HCOMM_LAUNCH_MODE_BATCH);
    HcommAclrtNotifyWaitOnThread(thread, notifyId, 0);
    HcommAclrtNotifyRecordOnThread(thread, notifyId);
    HcommSetLaunchMode("abc", HCOMM_LAUNCH_MODE_EAGER);

    2 CPU_TS模式
    FFTS+子图，最后批量提交。在HOST CPU上执行
    告知后面的CommWrite等任务进入批量模式（开始ffts+子图）

    1）复用task子图缓存
    增加 launchTag 的原因，进入批量模式之后，缓存要执行的一些task，最后提交。缓
    存的标识采用launchTag。在第二次执行想要复用子图执行时，只需要拿着相同的
    launchTag，调用 HcommSetLaunchMode接口，传入HCOMM_LAUNCH_MODE_EAGER参数，即可复用执行。
    比如下面的： HcommSetLaunchMode ("abc", HCOMM_LAUNCH_MODE_EAGER);
    执行之前缓存到"abc"下的几个数据面操作。

    2）清理
    如果不需要"abc"标识的这个子图的task 缓存了，可以采用如下方式清理该子图内容：
    HcommSetLaunchMode ("abc", HCOMM_LAUNCH_MODE_RESERVED)

    3）缺省 launchTag
    launchTag 如果为 nullptr，表示缺省值，标识不需要缓存到 FFTS+子图。
 */
// [中文导读] [AllReduce逐行 S134] LaunchContext::SetLaunchMode的接口声明：任务提交标签、请求的提交模式；这些参数属于本函数调用边界。
HcclResult LaunchContext::SetLaunchMode(const char* launchTag, HcommLaunchMode mode)
// [中文导读] [AllReduce逐行 S135] 进入LaunchContext::SetLaunchMode函数体：保存模式与 tag，按 BATCH/EAGER/RESERVED 准备、提交或清理任务。
{
    // [中文导读] [AllReduce逐行 S136] 设置当前提交模式为/按`mode`（请求的提交模式）。
    mode_ = mode;
    // 统一处理 launchTag
    // [中文导读] [AllReduce逐行 S138] 设置是否未指定提交标签为/按`(launchTag == nullptr)`（任务提交标签）。
    bool defaultTag = (launchTag == nullptr);
    // [中文导读] [AllReduce逐行 S139] 设置当前提交标签为/按`defaultTag ? "" : std::string(launchTag)`（是否未指定提交标签、任务提交标签）；调用std::string，使用是否未指定提交标签、任务提交标签。
    launchTag_ = defaultTag ? "" : std::string(launchTag);
    // [中文导读] [AllReduce逐行 S140] 记录LaunchContext::SetLaunchMode的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S141] 为当前LaunchContext::SetLaunchMode诊断/异常表达式提供格式文本，将报告当前提交标签的c_str字段；这一物理行没有数据搬运副作用。
        "[%s] SetLaunchMode begin, launchTag[%s], launchMode[%d].", __func__, launchTag_.c_str(),
        // [中文导读] [AllReduce逐行 S142] 为调用c_str，使用当前提交标签的c_str字段、请求的提交模式补入`static_cast<int32_t>(mode))`（请求的提交模式）；本行是参数/结构化初始化续行。
        static_cast<int32_t>(mode));

// [中文导读] [AllReduce逐行 S144] 编译条件`ifndef CCL_KERNEL_AICPU`限定后续实现，区分Host/设备或构建能力分支。
#ifndef CCL_KERNEL_AICPU
    // [中文导读] [AllReduce逐行 S145] 设置设备型号为/按`DevType::DEV_TYPE_COUNT`。
    DevType devType = DevType::DEV_TYPE_COUNT;
// [中文导读] [AllReduce逐行 S146] 结束前述编译条件控制的实现片段。
#endif
    // [中文导读] [AllReduce逐行 S147] 以`mode_`（当前提交模式）选择后续互斥处理路径。
    switch (mode_) {
        // [中文导读] [AllReduce逐行 S148] 匹配`HCOMM_LAUNCH_MODE_BATCH`的枚举路径；继续执行本case中的操作。
        case HCOMM_LAUNCH_MODE_BATCH:
// [中文导读] [AllReduce逐行 S149] 编译条件`ifndef CCL_KERNEL_AICPU`限定后续实现，区分Host/设备或构建能力分支。
#ifndef CCL_KERNEL_AICPU
            // [中文导读] [AllReduce逐行 S150] 读取设备型号用于新旧/协议分支选择；传入/处理设备型号。
            hrtGetDeviceType(devType);
            // [中文导读] [AllReduce逐行 S151] 仅当`(devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960)`（设备型号）成立时进入此分支。
            if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
                // [中文导读] [AllReduce逐行 S152] 记录LaunchContext::SetLaunchMode的状态/性能诊断；日志本身不执行传输。
                HCCL_INFO("[%s] Running on A5, CommTaskPrepare skipped.", __func__);
                // [中文导读] [AllReduce逐行 S153] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
                return HCCL_SUCCESS;
            // [中文导读] [AllReduce逐行 S154] 结束`if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960)`（设备型号）分支/循环；控制流返回外层。
            }
            // [中文导读] [AllReduce逐行 S155] 记录LaunchContext::SetLaunchMode的状态/性能诊断；日志本身不执行传输。
            HCCL_INFO("[%s] host mode, need CommTaskPrepare", __func__);
            // [中文导读] [AllReduce逐行 S156] 仅当`(!defaultTag)`（是否未指定提交标签）成立时进入此分支。
            if (!defaultTag) {
                // 仅非缺省 tag 需要准备任务缓存
                // [中文导读] [AllReduce逐行 S158] 直接返回`CommTaskPrepare(const_cast<char*>(launchTag_.c_str()), launchTag_.length())`（当前提交标签的c_str字段、当前提交标签的length字段）；兼容Host按tag准备任务缓存/发射上下文。
                return CommTaskPrepare(const_cast<char*>(launchTag_.c_str()), launchTag_.length());
            // [中文导读] [AllReduce逐行 S159] 结束`if (!defaultTag)`（是否未指定提交标签）分支/循环；控制流返回外层。
            }
// [中文导读] [AllReduce逐行 S160] 结束前述编译条件控制的实现片段。
#endif
            // [中文导读] [AllReduce逐行 S161] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
            return HCCL_SUCCESS;
        // [中文导读] [AllReduce逐行 S162] 匹配`HCOMM_LAUNCH_MODE_EAGER`的枚举路径；继续执行本case中的操作。
        case HCOMM_LAUNCH_MODE_EAGER:
            // [中文导读] [AllReduce逐行 S163] 提交当前tag与非tag参与Thread集合；返回非成功时由检查宏立即向上传递。
            CHK_RET(HandleEagerMode());
            // 缺省 tag 模式下清理缓存
            // [中文导读] [AllReduce逐行 S165] 直接返回`HandleClear()`；清除当前批登记状态，必要时清兼容任务缓存。
            return HandleClear();
        // [中文导读] [AllReduce逐行 S166] 匹配`HCOMM_LAUNCH_MODE_RESERVED`的枚举路径；继续执行本case中的操作。
        case HCOMM_LAUNCH_MODE_RESERVED:
            // [中文导读] [AllReduce逐行 S167] 仅当`(!defaultTag)`（是否未指定提交标签）成立时进入此分支。
            if (!defaultTag) {
                // [中文导读] [AllReduce逐行 S168] 直接返回`HandleClear()`；清除当前批登记状态，必要时清兼容任务缓存。
                return HandleClear();
            // [中文导读] [AllReduce逐行 S169] 结束`if (!defaultTag)`（是否未指定提交标签）分支/循环；控制流返回外层。
            }
            // [中文导读] [AllReduce逐行 S170] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
            return HCCL_SUCCESS;
        // [中文导读] [AllReduce逐行 S171] 处理switch中没有明确匹配的枚举值，具体返回/回退行为由以下代码决定。
        default:
            // [中文导读] [AllReduce逐行 S172] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
            return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S173] 结束`switch (mode_)`（当前提交模式）分支/循环；控制流返回外层。
    }
// [中文导读] [AllReduce逐行 S174] 结束LaunchContext::SetLaunchMode函数体；控制流返回外层。
}
