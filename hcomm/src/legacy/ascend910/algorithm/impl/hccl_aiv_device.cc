/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_aiv.h"

using namespace std;

namespace hccl {

HcclResult ReadBinFile(const string& fileName, string& buffer)
{
    (void)fileName;
    (void)buffer;
    return HCCL_SUCCESS;
}

// Kernel注册入口，全局只需要初始化一次
HcclResult RegisterKernel(DevType deviceType)
{
    (void)deviceType;
    return HCCL_SUCCESS;
}

HcclResult UnRegisterAivKernel() { return HCCL_SUCCESS; }

// KernelLaunch内部接口
HcclResult ExecuteKernelLaunchInner(
    const AivOpArgs& opArgs, const AivTopoArgs& topoArgs, const AivResourceArgs& resourceArgs,
    const AivAlgArgs& algArgs, void* args, u32 argsSize, AivProfilingInfo& aivProfilingInfo)
{
    (void)algArgs;
    (void)args;
    (void)argsSize;
    (void)topoArgs;
    (void)aivProfilingInfo;
    (void)opArgs;
    (void)resourceArgs;
    return HCCL_SUCCESS;
}

// Kernel单次调用Launch外部接口
HcclResult ExecuteKernelLaunch(
    const AivOpArgs& opArgs, const AivTopoArgs& topoArgs, const AivResourceArgs& resourceArgs,
    const AivAlgArgs& algArgs, AivProfilingInfo& aivProfilingInfo)
{
    (void)algArgs;
    (void)topoArgs;
    (void)aivProfilingInfo;
    (void)opArgs;
    (void)resourceArgs;
    return HCCL_SUCCESS;
}

// Kernel单次调用Launch外部接口
HcclResult ExecuteKernelLaunch(
    const AivOpArgs& opArgs, const AivTopoArgs& topoArgs, const AivResourceArgs& resourceArgs,
    const AivAlgArgs& algArgs, const ExtraArgs& extraArgs, AivProfilingInfo& aivProfilingInfo)
{
    (void)algArgs;
    (void)topoArgs;
    (void)extraArgs;
    (void)opArgs;
    (void)resourceArgs;
    (void)aivProfilingInfo;
    return HCCL_SUCCESS;
}

// Kernel单次调用Launch外部接口
HcclResult ExecuteKernelLaunch(
    const AivOpArgs& opArgs, const AivTopoArgs& topoArgs, const AivResourceArgs& resourceArgs,
    const AivAlgArgs& algArgs, const ExtraArgsV2& extraArgs, AivProfilingInfo& aivProfilingInfo)
{
    (void)algArgs;
    (void)topoArgs;
    (void)extraArgs;
    (void)aivProfilingInfo;
    (void)opArgs;
    (void)resourceArgs;
    return HCCL_SUCCESS;
}

void SetAivProfilingInfoBeginTime(AivProfilingInfo& aivProfilingInfo)
{
    (void)aivProfilingInfo;
    return;
}

void SetAivProfilingInfoBeginTime(uint64_t& beginTime)
{
    (void)beginTime;
    return;
}

} // namespace hccl
