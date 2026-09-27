/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sim_pipe_io.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

#include "sim_log.h"

namespace sim {

int PipeBlockRead(int fd, void* buf, size_t len)
{
    uint8_t* ptr = static_cast<uint8_t*>(buf);
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = read(fd, ptr, remaining);
        if (n > 0) {
            ptr += n;
            remaining -= n;
        } else if (n == 0) {
            HCCL_VM_INFO("read EOF, peer closed.");
            return -1;
        } else if (errno == EINTR) {
            continue;
        } else {
            HCCL_VM_ERROR("read failed: {} (errno={}).", strerror(errno), errno);
            return -1;
        }
    }

    return 0;
}

int PipeBlockWrite(int fd, const void* buf, size_t len)
{
    const uint8_t* ptr = static_cast<const uint8_t*>(buf);
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = write(fd, ptr, remaining);
        if (n > 0) {
            ptr += n;
            remaining -= n;
        } else if (n == 0) {
            HCCL_VM_INFO("write returned 0.");
            return -1;
        } else if (errno == EINTR) {
            continue;
        } else if (errno == EPIPE || errno == EAGAIN) {
            HCCL_VM_INFO("peer closed (errno={}).", errno);
            return -1;
        } else {
            HCCL_VM_ERROR("write failed: {} (errno={}).", strerror(errno), errno);
            return -1;
        }
    }

    return 0;
}

int PipeSendMsg(int fd, uint8_t cmd, const void* data, uint32_t len)
{
    PipeMessage msg{};
    msg.cmd = cmd;
    msg.bufLen = static_cast<uint16_t>(len);
    if (len > 0 && data != nullptr) {
        std::memcpy(msg.payload, data, len);
    }

    return PipeBlockWrite(fd, &msg, sizeof(msg));
}

int PipeRecvMsg(int fd, uint8_t& outCmd, void* outData, uint32_t maxLen, uint32_t& outLen)
{
    PipeMessage msg{};
    if (PipeBlockRead(fd, &msg, sizeof(msg)) != 0) {
        outCmd = 0;
        outLen = 0;
        return -1;
    }

    outCmd = msg.cmd;
    outLen = msg.bufLen;

    if (msg.bufLen > maxLen) {
        HCCL_VM_ERROR("payload too large: {} > {}", msg.bufLen, maxLen);
        outLen = 0;
        return -1;
    }

    if (msg.bufLen > 0 && outData != nullptr) {
        std::memcpy(outData, msg.payload, msg.bufLen);
    }

    return 0;
}

int PipeCreate(PipePair& pair)
{
    int fds[2] = {-1, -1};
    if (pipe2(fds, O_CLOEXEC) == -1) {
        HCCL_VM_ERROR("pipe2() failed: {}", strerror(errno));
        return -1;
    }

    pair.readFd = fds[0];
    pair.writeFd = fds[1];
    return 0;
}

void PipeClose(int& fd)
{
    if (fd >= 0) {
        close(fd);
        fd = -1;
    }
}

int PipeChildSetup(PipePair& h2d, PipePair& d2h, int targetReadFd, int targetWriteFd)
{
    // 子进程不写 h2d，不读 d2h
    close(h2d.writeFd);
    close(d2h.readFd);

    // 重定向到固定 fd
    dup2(h2d.readFd, targetReadFd);
    dup2(d2h.writeFd, targetWriteFd);

    // 清除 CLOEXEC 标记（execvp 后保留）
    fcntl(targetReadFd, F_SETFD, 0);
    fcntl(targetWriteFd, F_SETFD, 0);

    // 关闭原始 fd（如果与目标 fd 不同）
    if (h2d.readFd != targetReadFd) {
        close(h2d.readFd);
    }
    if (d2h.writeFd != targetWriteFd) {
        close(d2h.writeFd);
    }

    return 0;
}

} // namespace sim
