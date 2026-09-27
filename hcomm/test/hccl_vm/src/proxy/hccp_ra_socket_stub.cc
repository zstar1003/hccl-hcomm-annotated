/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "RASOCKET_STUB"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "sim_log.h"
#include "hccp_common.h"
#include "sim_ip_address.h"
#include "runtime/base.h"
#include "store_sim_comm_memory_manager.h"
#include "db_sim_runner_common.h"
#include "db_sim_runner_ops.h"
#include "db_sim_runner_ops.h"

extern uint64_t g_cur_server_key;

constexpr uint32_t RA_SOCKET_BUF_SIZE = (64 * 1024);

constexpr const char* RA_SOCK_BUF_PREFIX = "ra_sock_";

#define RS_FD_CLIENT 0x01U
#define RS_FD_SERVER 0x00U
#define MAKE_FD(pair_id, role) (((uint64_t)(role) << 32) | (uint64_t)(pair_id))
#define FD_PAIR_ID(fd_) ((fd_) & 0xFFFFFFFF)
#define FD_ROLE(fd_) ((fd_) >> 32)

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

struct RaSocketBuff {
    uint64_t size{0};
    uint32_t sendBytes{0};
    uint32_t recvBytes{0};
    char data[RA_SOCKET_BUF_SIZE];
};

static uint64_t ComputeTagHash(const char* tag)
{
    return static_cast<uint64_t>(std::hash<std::string>{}(std::string(tag)));
}

static bool GenRaSocketBufKeyByPairId(uint64_t pairId)
{
    std::string c2s = RA_SOCK_BUF_PREFIX + std::to_string(pairId) + "_c2s";
    void* ptrC2s = sim::CommunicationMemoryManager::GetInstance().AllocCommMem(c2s.data());
    if (ptrC2s == nullptr) {
        HCCL_VM_ERROR(" alloc sock name {} mem failed", c2s.data());
        return false;
    }

    std::string s2c = RA_SOCK_BUF_PREFIX + std::to_string(pairId) + "_s2c";
    void* ptrS2c = sim::CommunicationMemoryManager::GetInstance().AllocCommMem(s2c.data());
    if (ptrS2c == nullptr) {
        sim::CommunicationMemoryManager::GetInstance().ReleaseCommMem(c2s.data());
        HCCL_VM_ERROR(" alloc sock name {} mem failed", s2c.data());
        return false;
    }
    return true;
}

static bool AcquireRaSocketBufKeyByPairId(uint64_t pairId)
{
    std::string c2s = RA_SOCK_BUF_PREFIX + std::to_string(pairId) + "_c2s";
    void* ptrC2s = sim::CommunicationMemoryManager::GetInstance().AcquireCommMem(c2s.data());
    if (ptrC2s == nullptr) {
        HCCL_VM_ERROR(" acquire sock name {} mem failed", c2s.data());
        return false;
    }

    std::string s2c = RA_SOCK_BUF_PREFIX + std::to_string(pairId) + "_s2c";
    void* ptrS2c = sim::CommunicationMemoryManager::GetInstance().AcquireCommMem(s2c.data());
    if (ptrS2c == nullptr) {
        sim::CommunicationMemoryManager::GetInstance().ReleaseCommMem(c2s.data());
        HCCL_VM_ERROR(" acquire sock name {} mem failed", s2c.data());
        return false;
    }
    return true;
}

static bool DestoryRaSocketBufKeyByPairId(uint64_t pairId)
{
    std::string c2s = RA_SOCK_BUF_PREFIX + std::to_string(pairId) + "_c2s";
    std::string s2c = RA_SOCK_BUF_PREFIX + std::to_string(pairId) + "_s2c";
    sim::CommunicationMemoryManager::GetInstance().ReleaseCommMem(c2s.data());
    sim::CommunicationMemoryManager::GetInstance().ReleaseCommMem(s2c.data());
    return true;
}

static void GetRaSendSocketkeyByFd(uint64_t socketFd, std::string& key)
{
    uint64_t pairId = FD_PAIR_ID(socketFd);
    uint32_t role = FD_ROLE(socketFd);
    key = (role == RS_FD_CLIENT) ? RA_SOCK_BUF_PREFIX + std::to_string(pairId) + "_c2s" :
                                   RA_SOCK_BUF_PREFIX + std::to_string(pairId) + "_s2c";
}

static void GetRaRecvSocketkeyByFd(uint64_t socketFd, std::string& key)
{
    uint64_t pairId = FD_PAIR_ID(socketFd);
    uint32_t role = FD_ROLE(socketFd);
    key = (role == RS_FD_CLIENT) ? RA_SOCK_BUF_PREFIX + std::to_string(pairId) + "_s2c" :
                                   RA_SOCK_BUF_PREFIX + std::to_string(pairId) + "_c2s";
}

int RaSocketInit(int mode, struct rdev rdevInfo, void** socketHandle)
{
    (void)mode;
    sim::Device device{};
    if (GetDeviceByPhysicalId(rdevInfo.phyId, device) != ACL_SUCCESS) {
        HCCL_VM_ERROR(" get device by phy id {} failed.", rdevInfo.phyId);
        return -1;
    }

    BinaryAddr ba{};
    memcpy(&ba, &rdevInfo.localIp, sizeof(BinaryAddr));
    auto ipAddr = IpAddress(ba, AF_INET6).GetIpStr().substr(2);

    sim::EndPoint endPoint{};
    if (GetEndPointByIpAddr(ipAddr, endPoint) != 0) {
        HCCL_VM_ERROR(" cannot find remote ip {} ", ipAddr);
        return -1;
    }

    auto deviceIdx = endPoint.device_id;
    auto endpointId = endPoint.id;

    auto ret = RunnerDB::GetOneByPred<sim::RaSocket>([deviceIdx, endpointId](const sim::RaSocket& so) {
        return (so.device_id == deviceIdx && so.endpoint_id == endpointId);
    });
    if (ret.second) {
        *socketHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(ret.first.id));
        HCCL_VM_INFO(" Found socket socketFd:{:d}", ret.first.id);
        return 0;
    }

    sim::RaSocket socket{};
    socket.device_id = deviceIdx;
    socket.endpoint_id = endPoint.id;
    auto socketfd = RunnerDB::Add<sim::RaSocket>(socket);

    if (socketfd == 0) {
        auto retry = RunnerDB::GetOneByPred<sim::RaSocket>([deviceIdx, endpointId](const sim::RaSocket& so) {
            return (so.device_id == deviceIdx && so.endpoint_id == endpointId);
        });
        if (retry.second) {
            *socketHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(retry.first.id));
            HCCL_VM_WARN(" Insert conflict, reused existing socketFd:{:d}", retry.first.id);
            return 0;
        }
        HCCL_VM_ERROR(" Insert and re-lookup both failed for device:{:d} endpointId:{:d}", deviceIdx, endpointId);
        return -1;
    }

    *socketHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(socketfd));
    HCCL_VM_INFO(" add socketHandle:{:d} device:{:d} ip:{} endpointId:{:d}", socketfd, deviceIdx, ipAddr, endpointId);
    return 0;
}

int RaSocketInitV1(int mode, struct SocketInitInfoT socketInit, void** socketHandle)
{
    (void)mode;
    (void)socketInit;
    struct rdev rdevInfo {};
    return RaSocketInit(mode, rdevInfo, socketHandle);
}

int RaSocketDeinit(void* socketHandle)
{
    uint64_t localFd = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(socketHandle));
    RunnerDB::Delete<sim::RaSocket>(localFd);

    HCCL_VM_INFO(" delete socketHandle:{:d}", localFd);
    return 0;
}

int RaSocketListenStart(struct SocketListenInfoT conn[], uint32_t num)
{
    for (uint32_t i = 0; i < num; i++) {
        uint64_t socketHandle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(conn[i].socketHandle));
        auto socketRes = RunnerDB::GetById<sim::RaSocket>(socketHandle);
        if (!socketRes.has_value()) {
            HCCL_VM_ERROR(" can not get Socket:{:d}", socketHandle);
            return -1;
        }
        RunnerDB::Update<sim::RaSocket>(socketHandle, [](sim::RaSocket& s) {
            s.state = 1;
        });
        HCCL_VM_INFO(" socket {:d} port:{:d}", socketHandle, conn[i].port);
    }
    return 0;
}

int RaSocketListenStop(struct SocketListenInfoT conn[], unsigned int num)
{
    for (uint32_t i = 0; i < num; i++) {
        uint64_t socketHandle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(conn[i].socketHandle));
        auto socketRes = RunnerDB::GetById<sim::RaSocket>(socketHandle);
        if (!socketRes.has_value()) {
            HCCL_VM_ERROR(" can not get Socket:{:d}", socketHandle);
            return -1;
        }
        RunnerDB::Update<sim::RaSocket>(socketHandle, [](sim::RaSocket& s) {
            s.state = 0;
        });
        HCCL_VM_INFO(" socket {:d} port:{:d}", socketHandle, conn[i].port);
    }
    return 0;
}

int RaSocketBatchConnect(struct SocketConnectInfoT conn[], unsigned int num)
{
    for (int i = 0; i < num; i++) {
        uint64_t localFd = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(conn[i].socketHandle));
        auto localSock = RunnerDB::GetById<sim::RaSocket>(localFd);
        if (!localSock.has_value()) {
            HCCL_VM_ERROR(" can not get Local Ra Socket:{:d}", localFd);
            return -1;
        }

        BinaryAddr ba{};
        memcpy(&ba, &conn[i].remoteIp, sizeof(BinaryAddr));
        auto ipAddr = IpAddress(ba, AF_INET6).GetIpStr().substr(2);
        sim::EndPoint endPoint{};
        if (GetEndPointByIpAddr(ipAddr, endPoint) != 0) {
            HCCL_VM_ERROR(" cannot find remote ip:{}", ipAddr);
            return -1;
        }

        auto remoteDevId = endPoint.device_id;
        auto remoteEndPointId = endPoint.id;
        uint64_t tagHash = ComputeTagHash(conn[i].tag);
        uint32_t remotePort = conn[i].port;

        HCCL_VM_INFO(
            " dev:{} sock:{} connect dev:{} ip addr:{} tag:{}", localSock->device_id, localFd, remoteDevId, ipAddr,
            conn[i].tag);
        uint32_t count = 0;
        while (true) {
            if (count++ >= 600) { // 超时60s
                HCCL_VM_ERROR(
                    " can not get break dev:{} sock:{} connect dev:{} ip addr:{} tag:{}", localSock->device_id, localFd,
                    remoteDevId, ipAddr, conn[i].tag);
                break;
            }
            auto remoteSockRes
                = RunnerDB::GetOneByPred<sim::RaSocket>([remoteDevId, remoteEndPointId](const sim::RaSocket& socket) {
                      return socket.device_id == remoteDevId && socket.endpoint_id == remoteEndPointId;
                  });
            if (!remoteSockRes.second) {
                if (count % 100 == 0) {
                    HCCL_VM_WARN(" can not find remote dev:{}, endpoint:{}", remoteDevId, remoteEndPointId);
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            uint64_t peerFd = remoteSockRes.first.id;

            auto existPairRes = RunnerDB::GetOneByPred<sim::RaSocketPair>(
                [localFd, peerFd, remotePort, tagHash](const sim::RaSocketPair& p) {
                    return ((p.client_id == localFd && p.server_id == peerFd)
                            || (p.client_id == peerFd && p.server_id == localFd))
                           && p.port == remotePort && p.tag_hash == tagHash;
                });

            uint64_t pairId = 0;
            if (existPairRes.second) {
                pairId = existPairRes.first.id;
                HCCL_VM_INFO(" pair already exists id:{:d}", pairId);
            } else {
                sim::RaSocketPair socketpair{};
                socketpair.client_id = localFd;
                socketpair.server_id = peerFd;
                socketpair.ref_cnt = 0;
                socketpair.port = remotePort;
                socketpair.tag_hash = tagHash;
                socketpair.buf_status = 0;
                pairId = RunnerDB::Add<sim::RaSocketPair>(socketpair);
                if (pairId == 0) {
                    HCCL_VM_WARN(" add failed retry");
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
                // 创建链路buffer
                if (!GenRaSocketBufKeyByPairId(pairId)) {
                    RunnerDB::Delete<sim::RaSocketPair>(pairId);
                    return -1;
                }
                RunnerDB::Update<sim::RaSocketPair>(pairId, [](sim::RaSocketPair& pair) {
                    pair.buf_status = 1;
                });
                std::string sendBuf, recvBuf;
                GetRaSendSocketkeyByFd(pairId, sendBuf);
                GetRaRecvSocketkeyByFd(pairId, recvBuf);
                HCCL_VM_INFO(
                    " add pair id:{:d} local:{:d} peer:{:d} port:{:d} sendBuf:{}, recvBuf:{}", pairId, localFd, peerFd,
                    remotePort, sendBuf, recvBuf);
            }
            break;
        }
    }
    return 0;
}

int RaGetSockets(unsigned int role, struct SocketInfoT conn[], unsigned int num, unsigned int* connectedNum)
{
    *connectedNum = 0;
    for (int i = 0; i < num; i++) {
        uint64_t loadFd = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(conn[i].socketHandle));
        auto localSock = RunnerDB::GetById<sim::RaSocket>(loadFd);
        if (!localSock.has_value()) {
            HCCL_VM_ERROR(" can not get local socket fd:{:d}", loadFd);
            return -1;
        }

        BinaryAddr ba{};
        memcpy(&ba, &conn[i].remoteIp, sizeof(BinaryAddr));
        auto ipAddr = IpAddress(ba, AF_INET6).GetIpStr().substr(2);
        sim::EndPoint endPoint{};
        if (GetEndPointByIpAddr(ipAddr, endPoint) != 0) {
            HCCL_VM_ERROR(" cannot find remote ip:{}", ipAddr);
            continue;
        }

        auto remoteDevId = endPoint.device_id;
        auto remoteEndpointId = endPoint.id;
        uint64_t tagHash = ComputeTagHash(conn[i].tag);

        HCCL_VM_INFO(
            " dev:{} sock:{} get remote dev:{} addr:{} tag:{}", localSock->device_id, loadFd, remoteDevId, ipAddr,
            conn[i].tag);

        uint32_t count = 0;
        bool found = false;
        while (count++ < 180) {
            auto remoteSockRes
                = RunnerDB::GetOneByPred<sim::RaSocket>([remoteDevId, remoteEndpointId](const sim::RaSocket& socket) {
                      return socket.device_id == remoteDevId && socket.endpoint_id == remoteEndpointId;
                  });
            if (!remoteSockRes.second) {
                if (count % 60 == 0) {
                    HCCL_VM_WARN(
                        " waiting for remote socket dev:{}, endpoint:{}, attempt:{}/180", remoteDevId, remoteEndpointId,
                        count);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }
            uint64_t peerFd = remoteSockRes.first.id;

            auto pairRes
                = RunnerDB::GetOneByPred<sim::RaSocketPair>([loadFd, peerFd, tagHash](const sim::RaSocketPair& p) {
                      return ((p.client_id == loadFd && p.server_id == peerFd)
                              || (p.client_id == peerFd && p.server_id == loadFd))
                             && (p.tag_hash == tagHash) && (p.buf_status == 1);
                  });

            if (!pairRes.second) {
                if (count % 60 == 0) {
                    HCCL_VM_WARN(
                        " waiting for socket pair local:{:d} peer:{:d} tag:{}, attempt:{}/180", loadFd, peerFd,
                        conn[i].tag, count);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }

            auto pairId = pairRes.first.id;
            if (!AcquireRaSocketBufKeyByPairId(pairId)) {
                continue;
            }

            RunnerDB::Update<sim::RaSocketPair>(pairId, [](sim::RaSocketPair& pair) {
                pair.ref_cnt += 1;
            });

            uint32_t fdRole = (role == 0) ? RS_FD_SERVER : RS_FD_CLIENT;
            conn[i].fdHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(MAKE_FD(pairId, fdRole)));
            conn[i].status = 1;
            *connectedNum += 1;
            found = true;

            HCCL_VM_INFO(" get socket local:{:d} peer:{:d} pairId:{:d} role:{:d}", loadFd, peerFd, pairId, role);
            break;
        }

        if (!found) {
            HCCL_VM_ERROR("get socket failed tag:{}", conn[i].tag);
        }
    }
    return 0;
}

int RaSocketBatchClose(struct SocketCloseInfoT conn[], unsigned int num)
{
    for (int i = 0; i < num; i++) {
        uint64_t socketFd = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(conn[i].fdHandle));
        uint64_t pairId = FD_PAIR_ID(socketFd);

        auto pairRes = RunnerDB::GetById<sim::RaSocketPair>(pairId);
        if (!pairRes.has_value()) {
            HCCL_VM_WARN(" pair {:d} not found during close", pairId);
            continue;
        }

        std::string sendBuf, recvBuf;
        GetRaSendSocketkeyByFd(pairId, sendBuf);
        GetRaRecvSocketkeyByFd(pairId, recvBuf);

        if (pairRes->ref_cnt > 1) {
            RunnerDB::Update<sim::RaSocketPair>(pairId, [](sim::RaSocketPair& pair) {
                pair.ref_cnt -= 1;
            });
            HCCL_VM_INFO(
                " reduce pair:{:d} ref_cnt:{:d} sendBuf:{}, recvBuf:{}", pairId, pairRes->ref_cnt, sendBuf, recvBuf);
        } else {
            RunnerDB::Delete<sim::RaSocketPair>(pairId);
            HCCL_VM_INFO(
                " delete pair:{:d} ref_cnt:{:d} sendBuf:{}, recvBuf:{}", pairId, pairRes->ref_cnt, sendBuf, recvBuf);
        }

        DestoryRaSocketBufKeyByPairId(pairId);
    }
    return 0;
}

int RaSocketBatchAbort(struct SocketConnectInfoT conn[], unsigned int num)
{
    (void)conn;
    (void)num;
    return 0;
}

int RaSocketSend(const void* fdHandle, const void* data, unsigned long long size, unsigned long long* sentSize)
{
    uint64_t socketFd = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fdHandle));
    std::string socketKey;
    GetRaSendSocketkeyByFd(socketFd, socketKey);

    int64_t ret = sim::CommunicationMemoryManager::GetInstance().WriteCommMem(socketKey.data(), data, size);
    if (ret == -1) {
        HCCL_VM_ERROR(" cannot pair socket:{:d} role:{:d}, key={}", FD_PAIR_ID(socketFd), FD_ROLE(socketFd), socketKey);
        return -1;
    }

    *sentSize = ret;
    HCCL_VM_INFO(
        " pair socket:{:d} role:{:d} key={} Send:{:d}", FD_PAIR_ID(socketFd), FD_ROLE(socketFd), socketKey.data(),
        size);

    return 0;
}

int RaSocketRecv(const void* fdHandle, void* data, unsigned long long size, unsigned long long* receivedSize)
{
    uint64_t socketFd = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fdHandle));
    std::string socketKey;
    GetRaRecvSocketkeyByFd(socketFd, socketKey);

    int ret = sim::CommunicationMemoryManager::GetInstance().ReadCommMem(socketKey.data(), data, size);
    if (ret == 0) {
        HCCL_VM_WARN(
            " socket pair:{:d} role:{:d}, key={} read try again", FD_PAIR_ID(socketFd), FD_ROLE(socketFd), socketKey);
        return SOCK_EAGAIN;
    } else if (ret == -1) {
        HCCL_VM_ERROR(
            " socket pair:{:d} role:{:d}, key={} recv failed", FD_PAIR_ID(socketFd), FD_ROLE(socketFd), socketKey);
        return -1;
    } else if (ret > 0) {
        HCCL_VM_INFO(
            " socket pair:{:d} role:{:d}, key={} recv bytes:{:d} ", FD_PAIR_ID(socketFd), FD_ROLE(socketFd), socketKey,
            ret);
    }
    *receivedSize = ret;
    return 0;
}

int RaEpollCtlAdd(const void* fdHandle, enum RaEpollEvent event)
{
    (void)fdHandle;
    (void)event;
    return 0;
}

int RaEpollCtlMod(const void* fdHandle, enum RaEpollEvent event)
{
    (void)fdHandle;
    (void)event;
    return 0;
}

int RaEpollCtlDel(const void* fdHandle)
{
    (void)fdHandle;
    return 0;
}

int RaSetTcpRecvCallback(const void* socketHandle, const void* callback)
{
    (void)socketHandle;
    (void)callback;
    return 0;
}

/////////////////////////////////async/////////////////////////////
int RaGetAsyncReqResult(void* reqHandle, int* reqResult)
{
    (void)reqHandle;
    (void)reqResult;
    return 0;
}

int RaSocketBatchConnectAsync(struct SocketConnectInfoT conn[], unsigned int num, void** reqHandle)
{
    *reqHandle = reinterpret_cast<void*>(0x12345678);
    return RaSocketBatchConnect(conn, num);
}

int RaSocketListenStartAsync(struct SocketListenInfoT conn[], unsigned int num, void** reqHandle)
{
    (void)conn;
    (void)num;
    (void)reqHandle;
    return -1;
}

int RaSocketListenStopAsync(struct SocketListenInfoT conn[], unsigned int num, void** reqHandle)
{
    (void)conn;
    (void)num;
    *reqHandle = reinterpret_cast<void*>(0x12345678);
    return 0;
}

int RaSocketBatchCloseAsync(struct SocketCloseInfoT conn[], unsigned int num, void** reqHandle)
{
    (void)reqHandle;
    HCCL_VM_INFO(" close socket async");
    return RaSocketBatchClose(conn, num);
}

int RaSocketSendAsync(
    const void* fdHandle, const void* data, unsigned long long size, unsigned long long* sentSize, void** reqHandle)
{
    *reqHandle = reinterpret_cast<void*>(0x12345678);
    return RaSocketSend(fdHandle, data, size, sentSize);
}

int RaSocketRecvAsync(
    const void* fdHandle, void* data, unsigned long long size, unsigned long long* receivedSize, void** reqHandle)
{
    *reqHandle = reinterpret_cast<void*>(0x12345678);

    uint64_t socketFd = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fdHandle));
    std::string socketKey;
    GetRaRecvSocketkeyByFd(socketFd, socketKey);

    int ret = sim::CommunicationMemoryManager::GetInstance().ReadCommMem(socketKey.data(), data, size);
    if (ret == 0) {
        HCCL_VM_WARN(
            " socket pair:{:d} role:{:d} key:{} try again", FD_PAIR_ID(socketFd), FD_ROLE(socketFd), socketKey);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        *receivedSize = 0;
        return 0;
    } else if (ret == -1) {
        HCCL_VM_ERROR(
            " socket pair:{:d} role:{:d} key:{} recv failed", FD_PAIR_ID(socketFd), FD_ROLE(socketFd), socketKey);
        return -1;
    }

    *receivedSize = ret;
    HCCL_VM_INFO(
        " socket pair:{:d} role:{:d} key:{} recv bytes:{:d} ", FD_PAIR_ID(socketFd), FD_ROLE(socketFd), socketKey, ret);
    return 0;
}

#ifdef __cplusplus
}
#endif // __cplusplus
