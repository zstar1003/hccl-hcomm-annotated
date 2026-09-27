/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_V2_STEST_ST_CTX_H
#define HCCL_V2_STEST_ST_CTX_H

#include "../hccl_st_situation.h"
#include "hccl_communicator.h"
// situation是公用的配置， context是执行起来之后各个线程拿到的上下文
struct ThreadContext {
    Situation situation;
    int serverId;
    int deviceId;
    int myRank;
    std::string commId;
    void* sendBuf;
    void* recvBuf;
    void* expectedResBuf;
    Hccl::HcclCommunicator* comm;

    virtual ~ThreadContext();
};

ThreadContext* GetCurrentThreadContext();

void SetCurrentThreadContext(ThreadContext* ctx);

#endif // HCCL_V2_STEST_ST_CTX_H
