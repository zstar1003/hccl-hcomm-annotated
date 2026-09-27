/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_comm_socket_c_adpt.h"
#include "adapter_rts_common.h"
#include "socket_process.h"

namespace {
constexpr s32 kHostResourceId = 0;

HcclResult GetRuntimeSocketProcessId(s32& socketProcessId)
{
    uint32_t deviceCount = 0;
    HcclResult ret = hrtGetDeviceCount(&deviceCount);
    if (ret != HCCL_SUCCESS || deviceCount == 0) {
        HCCL_WARNING(
            "[%s] get device count ret[%d], count[%u], use host resource id[%d].", __func__, ret, deviceCount,
            kHostResourceId);
        socketProcessId = kHostResourceId;
        return HCCL_SUCCESS;
    }

    CHK_RET(hrtGetDevice(&socketProcessId));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult SocketCreate(SocketDesc* socketDesc, SocketHandler* socketHandle)
{
    CHK_PTR_NULL(socketDesc);
    CHK_PTR_NULL(socketHandle);

    s32 socketProcessId = 0;
    CHK_RET(GetRuntimeSocketProcessId(socketProcessId));
    return hcomm::SocketProcess::GetInstance(socketProcessId).GetSocket(socketDesc, *socketHandle);
}

HcclResult SocketDestroy(SocketHandler socketHandle)
{
    CHK_PTR_NULL(socketHandle);
    s32 socketProcessId = 0;
    CHK_RET(GetRuntimeSocketProcessId(socketProcessId));
    return hcomm::SocketProcess::GetInstance(socketProcessId).DestroySocketHandle(socketHandle);
}

HcclResult SocketGetStatus(SocketHandler socketHandle, SocketStates* status)
{
    CHK_PTR_NULL(socketHandle);
    CHK_PTR_NULL(status);
    s32 socketProcessId = 0;
    CHK_RET(GetRuntimeSocketProcessId(socketProcessId));
    return hcomm::SocketProcess::GetInstance(socketProcessId).GetStatus(socketHandle, *status);
}

HcclResult SocketSendNb(SocketHandler socketHandle, void* sendbuffer, uint64_t sendSize, uint64_t* sentSize)
{
    CHK_PTR_NULL(socketHandle);
    CHK_PTR_NULL(sendbuffer);
    CHK_PTR_NULL(sentSize);
    s32 socketProcessId = 0;
    CHK_RET(GetRuntimeSocketProcessId(socketProcessId));
    return hcomm::SocketProcess::GetInstance(socketProcessId)
        .SendNoBlock(socketHandle, sendbuffer, sendSize, reinterpret_cast<u64*&>(sentSize));
}

HcclResult SocketRecvNb(SocketHandler socketHandle, void* recvBuffer, uint64_t recvSize, uint64_t* recvedSize)
{
    CHK_PTR_NULL(socketHandle);
    CHK_PTR_NULL(recvBuffer);
    CHK_PTR_NULL(recvedSize);
    s32 socketProcessId = 0;
    CHK_RET(GetRuntimeSocketProcessId(socketProcessId));
    return hcomm::SocketProcess::GetInstance(socketProcessId)
        .RecvNoBlock(socketHandle, recvBuffer, recvSize, reinterpret_cast<u64*&>(recvedSize));
}
