/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_GROUP_H
#define HCCL_GROUP_H

#include "hccl_group_utils.h"

extern thread_local s32 hcclGroupDepth; // depth of HcclGroupStart nesting

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcclLegacyGroupStart();
HcclResult HcclLegacyGroupEnd();
HcclResult HcclLegacyAsyncJobLaunch();

#ifdef __cplusplus
}
#endif
namespace hccl {

#ifdef __cplusplus
extern "C" {
#endif

HcclResult
commInitTaskAppend(std::shared_ptr<struct hcclAsyncJob> job, HcclResult (*func)(struct hcclAsyncJob*), HcclComm* comm);

HcclResult taskAppend(HcclComm comm, hcclOpInfo& info);

#ifdef __cplusplus
}
#endif

} // namespace hccl

u64 calcOpDataVolume(const hccl::hcclOpInfo& info);
std::vector<hccl::hcclOpInfo> sortGroupTasks(const std::deque<hccl::hcclOpInfo>& tasks);

#endif
