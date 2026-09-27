/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMM_SOCKET_C_ADPT_H
#define HCCL_COMM_SOCKET_C_ADPT_H

#include "hcomm_res.h"
#include "hccl/hccl_res.h"
#include "mem_host_pub.h"
#include "hccl_diag.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#define HCCL_SOCKET_TAG_LEN 192

/**
 * @brief socket通道描述参数
 */
typedef struct {
    CommAbiHeader header;
    EndpointDesc localEndpoint;    // 本地网络设备端侧描述
    EndpointDesc remoteEndpoint;   // 远端网络设备端侧描述
    char tag[HCCL_SOCKET_TAG_LEN]; // 用户自定义标签,must end with '\0'
    HcommSocketRole role;          // socket角色：服务端或客户端
    uint16_t listenPort;           // 本地监听端口，仅server端有效
    char reserved[256];            // 预留字段
} SocketDesc;                      // SocketDesc

typedef void* SocketHandler;
// 此处前缀改一下 取消掉HcclComm前缀
enum SocketStates { SOCKET_OK, SOCKET_CONNECTING, SOCKET_TIMEOUT };

HcclResult
SocketCreate(SocketDesc* socketDesc, SocketHandler* socketHandle); // HcclCommSocketCreate接口，创建socket连接

HcclResult SocketDestroy(SocketHandler socketHandle);

HcclResult SocketGetStatus(SocketHandler socketHandle, SocketStates* status);

HcclResult SocketSendNb(SocketHandler socketHandle, void* sendbuffer, uint64_t sendSize, uint64_t* sentSize);

HcclResult SocketRecvNb(SocketHandler socketHandle, void* recvBuffer, uint64_t recvSize, uint64_t* recvedSize);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HCCL_COMM_SOCKET_C_ADPT_H
