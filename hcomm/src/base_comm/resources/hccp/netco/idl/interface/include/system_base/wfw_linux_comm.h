/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef WFW_LINUX_COMM_H
#define WFW_LINUX_COMM_H

#include "bkf_comm.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#define WFW_PIPE_FDR_IDX (0)
#define WFW_PIPE_FDW_IDX (1)

typedef void *(*F_PTHREAD_ROUTINE)(void *arg);

#define WFW_RET_RETRY_NOW(ret, errno_) (((ret) == -1) && ((errno_) == EINTR))
#define WFW_RET_RETRY_LATER(ret, errno_) (((ret) == -1) && (((errno_) == EAGAIN) || ((errno_) == EWOULDBLOCK)))

#define WFW_RET_IN_CONNING(ret, errno_) (((ret) == -1) && ((errno_) == EINPROGRESS))
#define WFW_RET_HAS_CONNED(ret, errno_) (((ret) == 0) || (((ret) == -1) && ((errno_) == EISCONN)))

int WfwSetFdNonBlock(int fd);

char *WfwGetBackTraceStr(uint8_t *buf, int32_t bufLen);
int WfwGetFuncCallStackDepth(void);

#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
