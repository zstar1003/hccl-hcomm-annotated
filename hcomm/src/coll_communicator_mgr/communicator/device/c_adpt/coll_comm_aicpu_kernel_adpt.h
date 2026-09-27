/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_COMM_AICPU_KERNEL_ADPT_H
#define COLL_COMM_AICPU_KERNEL_ADPT_H

#include "common.h"
#include "aicpu_launch_manager.h"
#include "channel_param.h"
#include "aicpu_init_param.h"

// kernel 入口 C++ 适配层 — 供 kernel 入口调用，封装 Acquire→操作→Release 流程。extern "C" 接口见
// coll_comm_aicpu_kernel.h。 内部委托到 CollCommAicpuMgr 单例，完成 Acquire → 操作 → Release 流程。

HcclResult CollCommAicpuKernelAdptInitThreads(ThreadMgrAicpuParam* param);
HcclResult CollCommAicpuKernelAdptInitChannel(HcclChannelUrmaRes* commParam);
HcclResult CollCommAicpuKernelAdptUpdateChannel(HcclChannelUrmaRes* commParam);
HcclResult CollCommAicpuKernelAdptInitNotify(NotifyMgrAicpuParam* param);

#endif // COLL_COMM_AICPU_KERNEL_ADPT_H
