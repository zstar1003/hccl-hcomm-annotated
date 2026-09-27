/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_AICPU_PIPE_MSG_H
#define SIM_AICPU_PIPE_MSG_H

#include <cstdint>
#include <cstring>

constexpr uint16_t KERNEL_NAME_LEN_MAX = 64;
constexpr uint16_t KERNEL_SO_NAME_LEN_MAX = 64;
constexpr uint16_t DEVICE_MEMORY_NAME_LEN_MAX = 100;

// Host-Device管道通信消息定义
constexpr uint8_t PIPE_CMD_SHUTDOWN = 0x00;
constexpr uint8_t PIPE_CMD_EXEC_KERNEL = 0x01;
constexpr uint8_t PIPE_CMD_GET_DEV_PTR = 0x02;
constexpr uint8_t PIPE_CMD_SET_DEV_ID = 0x03;
constexpr uint8_t PIPE_CMD_FREE_DEV_PTR = 0x04;
constexpr uint8_t PIPE_RSP_SHUTDOWN_ACK = 0x80;
constexpr uint8_t PIPE_RSP_EXEC_KERNEL = 0x81;
constexpr uint8_t PIPE_RSP_GET_DEV_PTR = 0x82;
constexpr uint8_t PIPE_RSP_SET_DEV_ID = 0x83;
constexpr uint8_t PIPE_RSP_FREE_DEV_PTR = 0x84;
constexpr uint8_t PIPE_RSP_READY = 0x85;
constexpr uint8_t PIPE_RSP_ERROR = 0xFF;
constexpr uint8_t PAYLOAD_LEN_MAX = 253;

#pragma pack(push, 1)

struct PipeMessage {
    uint8_t cmd;
    uint16_t bufLen;
    uint8_t payload[PAYLOAD_LEN_MAX];
};

typedef struct {
    char kernelName[KERNEL_NAME_LEN_MAX];
    char soName[KERNEL_SO_NAME_LEN_MAX];
    uint64_t args;
} ExecKernelPayload;

typedef struct {
    int32_t status;
} RspExecKernelPayload;

typedef struct {
    char memName[DEVICE_MEMORY_NAME_LEN_MAX];
} DevMemOpPayload;

typedef struct {
    uint64_t ptr;
} RspGetDevPtrPayload;

typedef struct {
    int32_t status;
} RspFreeDevPtrPayload;

typedef struct {
    uint64_t rankId;
    uint64_t deviceKey;
} SetDevIdPayload;

#pragma pack(pop)

#endif
