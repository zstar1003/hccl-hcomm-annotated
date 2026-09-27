/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef WORKFLOW_H
#define WORKFLOW_H

#include <hccl/hccl_types.h>

enum class HcclWorkflowMode {
    HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB = 0,
    HCCL_WORKFLOW_MODE_OP_BASE = 1,
    HCCL_WORKFLOW_MODE_RESERVED = 255
};

HcclResult InitWorkflowMode(HcclWorkflowMode mode);
HcclResult SetWorkflowMode(HcclWorkflowMode mode);
HcclWorkflowMode GetWorkflowMode();
#endif // WORKFLOW_H
