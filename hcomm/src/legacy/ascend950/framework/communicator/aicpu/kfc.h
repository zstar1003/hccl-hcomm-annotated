/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_KFC_H
#define HCCLV2_KFC_H

#include "enum_factory.h"
#include "error_message_v2.h"

namespace Hccl {
MAKE_ENUM(KfcCommand, NONE, NS_STOP_LAUNCH, NS_CLEAN, DESTROY_AICPU_COMM)
MAKE_ENUM(KfcStatus, NONE, STOP_LAUNCH_DONE, CLEAN_DONE, DESTROY_AICPU_COMM_DONE, ERROR)
MAKE_ENUM(KfcErrType, NONE, TIMEOUT, EXEC)

#pragma pack(push)
#pragma pack(1)
// aicpu向host提供的状态查询
struct KfcExecStatus {
    KfcStatus kfcStatus; // KFC状态
    KfcErrType kfcError; // KFC错误码
    ErrorMessageReport errorMessageReport;
};
#pragma pack(pop)
} // namespace Hccl
#endif
