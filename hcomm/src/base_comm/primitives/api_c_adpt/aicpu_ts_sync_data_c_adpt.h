/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_PRIMITIVES_INNER_H
#define HCOMM_PRIMITIVES_INNER_H

#include "hcomm_primitives.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MSG_HANDLE_DEFINED
#define MSG_HANDLE_DEFINED
typedef void* MsgHandle;
#endif

extern int32_t
HcommSendRequest(MsgHandle handle, const char* msgTag, const void* src, size_t sizeByte, uint32_t* msgId);
extern int32_t HcommWaitResponse(MsgHandle handle, void* dst, size_t sizeByte, uint32_t* msgId);
extern int32_t HcommThreadSynchronize(ThreadHandle thread);

#ifdef __cplusplus
}
#endif

#endif // HCOMM_PRIMITIVES_INNER_H
