/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INC_ADAPTER_TDT_H
#define HCCL_INC_ADAPTER_TDT_H

#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "dltdt_function.h"
#include "acl/acl_base.h"
#include "rt_external.h"

#ifdef __cplusplus
extern "C" {
#endif
HcclResult hrtOpenTsd();
HcclResult hrtOpenNetService(rtNetServiceOpenArgs* openArgs);
HcclResult hrtCloseNetService();
HcclResult hrtTsdCapabilityGet(uint32_t deviceLogicId, int32_t type, uint64_t ptr);
#ifdef __cplusplus
} // extern "C"
#endif
#endif
