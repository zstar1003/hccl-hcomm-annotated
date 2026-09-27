/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <cstring>
#include <sstream>

#include "hccp_peer_manager.h"
#include "host_buffer.h"
#include "host_socket_handle_manager.h"
#include "orion_adapter_hccp.h"
#include "orion_adapter_rts.h"
#include "../../../src/legacy/ascend950/framework/env_config/env_func.h"
#include "../../../src/legacy/ascend950/framework/topo/rank_info_detect/preempt_port_manager_v2.h"
#include "socket.h"
#include "socket_manager.h"
#include "topo_addr_info.h"

namespace {
const char* GetStubRootInfo()
{
    return R"({"version":"2.0","topo_file_path":"/tmp","rank_count":1,"rank_list":[{"device_id":0,"local_id":0,"level_list":[{"net_layer":0,"net_instance_id":"0","rank_addr_list":[]}]}]})";
}
} // namespace

extern "C" {
int TopoAddrInfoGetSize(int phyId, size_t* size)
{
    (void)phyId;
    if (size == nullptr) {
        return -1;
    }
    *size = std::strlen(GetStubRootInfo()) + 1U;
    return 0;
}

int TopoAddrInfoGetTopoFilePath(int phyId, char* filePath, size_t bufSize)
{
    (void)phyId;
    constexpr const char* stubTopoFilePath = "/tmp";
    const size_t requiredSize = std::strlen(stubTopoFilePath) + 1U;
    if (filePath == nullptr || bufSize < requiredSize) {
        return -1;
    }
    std::memcpy(filePath, stubTopoFilePath, requiredSize);
    return 0;
}

int TopoAddrInfoGet(int phyId, char* rankInfo, size_t* bufSize)
{
    (void)phyId;
    if (rankInfo == nullptr || bufSize == nullptr) {
        return -1;
    }

    const char* stubRootInfo = GetStubRootInfo();
    const size_t jsonSize = std::strlen(stubRootInfo);
    const size_t requiredSize = jsonSize + 1U;
    if (*bufSize < requiredSize) {
        *bufSize = requiredSize;
        return -1;
    }

    std::memcpy(rankInfo, stubRootInfo, requiredSize);
    *bufSize = jsonSize;
    return 0;
}
}

namespace Hccl {
void HrtSetDevice(s32 deviceLogicId) { (void)deviceLogicId; }

u32 HrtGetDeviceCount() { return 8U; }

std::vector<std::pair<std::string, IpAddress>> HrtGetHostIf(u32 devPhyId)
{
    (void)devPhyId;
    return {{"lo", IpAddress("127.0.0.1")}};
}

void HrtRaSocketSetWhiteListStatus(u32 enable) { (void)enable; }

void HrtRaSocketListenOneStart(RaSocketListenParam& in, HrtNetworkMode netMode)
{
    (void)in;
    (void)netMode;
}

bool HrtRaSocketTryListenOneStart(RaSocketListenParam& in, HrtNetworkMode netMode)
{
    (void)netMode;
    if (in.port == 0U) {
        in.port = 43210U;
    }
    return true;
}

void HrtRaSocketBlockSend(const FdHandle fdHandle, const void* data, u32 sendSize)
{
    (void)fdHandle;
    (void)data;
    (void)sendSize;
}

bool HrtRaSocketNonBlockSend(const FdHandle fdHandle, void* data, u64 size, u64* sentSize)
{
    (void)fdHandle;
    (void)data;
    if (sentSize != nullptr) {
        *sentSize = size;
    }
    return true;
}

HcclResult HrtRaWaitEventHandle(
    int eventHandle, std::vector<SocketEventInfo>& eventInfos, int timeout, unsigned int maxEvents, u32& eventsNum)
{
    (void)eventHandle;
    (void)eventInfos;
    (void)timeout;
    (void)maxEvents;
    eventsNum = 0U;
    return HCCL_SUCCESS;
}

void* HrtMallocHost(u64 size) { return std::malloc(static_cast<std::size_t>(size)); }

void HrtFreeHost(void* hostPtr) { std::free(hostPtr); }

void HccpPeerManager::DeInit(s32 deviceLogicId) { (void)deviceLogicId; }

void Socket::Close() {}

void Socket::StopListen() {}

PreemptPortManager& PreemptPortManager::GetInstance(s32 deviceLogicId)
{
    (void)deviceLogicId;
    alignas(PreemptPortManager) static unsigned char storage[sizeof(PreemptPortManager)] = {};
    return *reinterpret_cast<PreemptPortManager*>(storage);
}

void PreemptPortManager::ListenPreempt(
    const std::shared_ptr<Socket>& listenSocket, const std::vector<SocketPortRange>& portRange, u32& usePort)
{
    CHK_SMART_PTR_RET_NULL(listenSocket);
    for (const auto& range : portRange) {
        for (u32 port = range.min; port <= range.max; ++port) {
            if (listenSocket->Listen(port)) {
                usePort = port;
                return;
            }
        }
    }
    THROW<InvalidParamsException>("No available port to listen");
}

void PreemptPortManager::Release(const std::shared_ptr<Socket>& listenSocket) { (void)listenSocket; }

HostBuffer::HostBuffer(uintptr_t devAddr, std::size_t devSize) : Buffer(devSize), selfOwned(false)
{
    addr_ = devAddr;
    size_ = devSize;
}

HostBuffer::HostBuffer(std::size_t allocSize) : Buffer(allocSize), selfOwned(true)
{
    if (allocSize == 0U) {
        THROW<InternalException>("allocSize should not be 0.");
    }
    addr_ = reinterpret_cast<uintptr_t>(std::malloc(allocSize));
    if (addr_ == 0U) {
        THROW<InternalException>("alloc host buffer failed.");
    }
}

HostBuffer::~HostBuffer()
{
    if (selfOwned) {
        std::free(reinterpret_cast<void*>(addr_));
    }
}

std::string HostBuffer::Describe() const
{
    std::ostringstream oss;
    oss << "HostBuffer[addr=0x" << std::hex << addr_ << ", size=0x" << size_ << ", selfOwned=" << selfOwned << "]";
    return oss.str();
}

bool HostBuffer::GetSelfOwned() const { return selfOwned; }

void SocketManager::ServerInitAll(NewRankInfo& rankInfo) { (void)rankInfo; }

void HostSocketHandleManager::Destroy(DevId devicePhyId, const IpAddress& hostIp)
{
    (void)devicePhyId;
    (void)hostIp;
}
} // namespace Hccl
