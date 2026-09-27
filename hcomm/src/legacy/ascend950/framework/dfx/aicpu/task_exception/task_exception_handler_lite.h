/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TASK_EXCEPTION_HANDLER_LITE_H
#define HCCL_TASK_EXCEPTION_HANDLER_LITE_H

#include "hccl/base.h"
#include "global_mirror_tasks.h"
#include "task_exception_func.h"
#include "dlhal_function_v2.h"
#include "communicator_impl_lite.h"

namespace Hccl {
class TaskExceptionHandlerLite {
public:
    // 获取单例实例的方法
    static TaskExceptionHandlerLite& GetInstance();
    // task exception处理逻辑
    static void Process(CommunicatorImplLite* aicpuComm, rtLogicCqReport_t* exceptionInfo);
    static std::string GetGroupRankInfo(const TaskInfo& taskInfo);
    static void PrintTaskContextInfo(CommunicatorImplLite* aicpuComm, uint32_t sqId, uint32_t taskId);

private:
    // 私有构造函数
    TaskExceptionHandlerLite();

    // 私有析构函数
    ~TaskExceptionHandlerLite();

    // 私有拷贝构造函数和赋值运算符，防止拷贝
    TaskExceptionHandlerLite(const TaskExceptionHandlerLite&) = delete;
    TaskExceptionHandlerLite& operator=(const TaskExceptionHandlerLite&) = delete;

    // 注册方法
    void Register() const;
};

uint16_t SwitchUBCqeErrCodeToTsErrCode(u32 cqeErrCode);
uint16_t SwitchSdmaCqeErrCodeToTsErrCode(u32 cqeErrCode);

} // namespace Hccl

#endif // HCCL_TASK_EXCEPTION_HANDLER_LITE_H
