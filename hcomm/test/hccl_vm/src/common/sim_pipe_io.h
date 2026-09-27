/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_PIPE_IO_H
#define SIM_PIPE_IO_H

#include <cstddef>
#include <cstdint>
#include "sim_aicpu_pipe_msg.h"

namespace sim {

struct PipePair {
    int readFd;
    int writeFd;
};

int PipeBlockRead(int fd, void* buf, size_t len);

int PipeBlockWrite(int fd, const void* buf, size_t len);

int PipeSendMsg(int fd, uint8_t cmd, const void* data, uint32_t len);

int PipeRecvMsg(int fd, uint8_t& outCmd, void* outData, uint32_t maxLen, uint32_t& outLen);

int PipeCreate(PipePair& pair);

void PipeClose(int& fd);

int PipeChildSetup(PipePair& h2d, PipePair& d2h, int targetReadFd, int targetWriteFd);

} // namespace sim

#endif // SIM_PIPE_IO_H
