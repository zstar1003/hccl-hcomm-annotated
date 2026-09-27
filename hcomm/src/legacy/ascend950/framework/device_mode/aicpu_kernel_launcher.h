/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_AICPU_KERNEL_LAUNCHER
#define HCCLV2_AICPU_KERNEL_LAUNCHER

#include <unordered_map>
#include "stream.h"
#include "kernel_param_lite.h"
#include "coll_operator.h"
#include "communicator_impl.h"

namespace Hccl {

class AicpuKernelLauncher {
public:
    explicit AicpuKernelLauncher(const CommunicatorImpl& comm) : comm(&comm) {}

    void AicpuKernelLaunch(const Stream& stream, const string& algName) const;

private:
    const CommunicatorImpl* comm;

    void SetOpbaseBufferParam(HcclKernelLaunchParam& param, CollOperator& op) const;
    void SetOffloadBufferParam(HcclKernelLaunchParam& param, CollOperator& op) const;
    void SetHcclKernelLaunchParam(HcclKernelLaunchParam& param) const;
    void AddPostToUserStream(const Stream& stream) const;
    void AddWaitToUserStream(const Stream& stream) const;
};

} // namespace Hccl

#endif
