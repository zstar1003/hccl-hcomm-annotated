/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "preempt_port_manager_v2.h"
#include <sstream>
#include "orion_adapter_hccp.h"
#include "hccl_common_v2.h"
#include "adapter_error_manager_pub.h"

namespace Hccl {

constexpr s32 HOST_DEVICE_ID = -1; // device id 无效值

bool PreemptPortManager::initialized = false;

PreemptPortManager::PreemptPortManager()
{
    // 根据host or device nic区分
    IpPortRef hostPortRef;
    preemptSockets_.emplace(HrtNetworkMode::PEER, hostPortRef);
    IpPortRef devPortRef;
    preemptSockets_.emplace(HrtNetworkMode::HDC, devPortRef);
    initialized = true;
}

PreemptPortManager::~PreemptPortManager()
{
    preemptSockets_.clear();
    initialized = false;
}

PreemptPortManager& PreemptPortManager::GetInstance(s32 deviceLogicId)
{
    static PreemptPortManager instance[MAX_MODULE_DEVICE_NUM];
    if (deviceLogicId == HOST_DEVICE_ID) {
        HCCL_INFO("[GetInstance] deviceLogicId[-1] is HOST_DEVICE_ID");
        return instance[0];
    }
    CHK_PRT_RET(
        (static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM || deviceLogicId < 0),
        HCCL_WARNING("[PreemptPortManager::%s] deviceLogicId[%d] is invalid", __func__, deviceLogicId), instance[0]);

    return instance[deviceLogicId];
}

void PreemptPortManager::ListenPreempt(
    const std::shared_ptr<Socket>& listenSocket, const std::vector<SocketPortRange>& portRange, u32& usePort)
{
    CHK_PRT_RET(
        !initialized,
        HCCL_ERROR("[PreemptPortManager::%s] preempt port manager has already been released.", __func__), );

    CHK_SMART_PTR_RET_NULL(listenSocket);
    NicType nicType = listenSocket->GetNicType();
    HrtNetworkMode netMode = nicType == NicType::HOST_NIC_TYPE ? HrtNetworkMode::PEER : HrtNetworkMode::HDC;
    std::lock_guard<std::mutex> lock(preemptMutex_);
    PreemptPortInRange(listenSocket, netMode, portRange, usePort);
    HCCL_INFO("[PreemptPortManager::%s] listening on port[%u] for nicType[%u] success.", __func__, usePort, nicType);
}

void PreemptPortManager::Release(const std::shared_ptr<Socket>& listenSocket)
{
    CHK_PRT_RET(
        !initialized,
        HCCL_WARNING("[PreemptPortManager::%s] preempt port manager has already been released.", __func__), );

    CHK_SMART_PTR_RET_NULL(listenSocket);
    NicType nicType = listenSocket->GetNicType();
    HrtNetworkMode netMode = nicType == NicType::HOST_NIC_TYPE ? HrtNetworkMode::PEER : HrtNetworkMode::HDC;

    std::lock_guard<std::mutex> lock(preemptMutex_);
    ReleasePreempt(preemptSockets_[netMode], listenSocket, netMode);
    HCCL_INFO("[PreemptPortManager::%s] release socket of type[%u] success.", __func__, nicType);
}

void PreemptPortManager::PreemptPortInRange(
    const std::shared_ptr<Socket>& listenSocket, HrtNetworkMode netMode, const std::vector<SocketPortRange>& portRange,
    u32& usePort)
{
    IpPortRef& portRef = preemptSockets_[netMode];
    std::string ipAddr(listenSocket->GetLocalIp().GetIpStr());
    if (portRef.find(ipAddr) != portRef.end()) {
        // 如果在这个IP上已经有已经抢占的port，则复用这个port
        usePort = portRef[ipAddr].first;
        bool ret = listenSocket->Listen(usePort);
        CHK_PRT_THROW(
            !ret, HCCL_ERROR("[PreemptPortManager::%s] usePort[%u] listen failed.", __func__, usePort),
            InvalidParamsException, "socket listen failed");
        portRef[ipAddr].second.Ref();
        HCCL_INFO(
            "[PreemptPortManager::%s] socket has already been listened, ref count[%d].", __func__,
            portRef[ipAddr].second.Count());
        return;
    }
    // 如果这个IP上没有抢占过的port，则轮询输入的端口范围，找到一个可用的端口
    for (auto& range : portRange) {
        for (u32 port = range.min; port <= range.max; ++port) {
            if (listenSocket->Listen(port)) {
                // 抢占端口成功，将端口记录到计数器中，并作为出参返回
                usePort = port;
                portRef[ipAddr].first = usePort;
                portRef[ipAddr].second.Ref();
                HCCL_INFO(
                    "[PreemptPortManager::%s] listen on ip[%s] and port[%u] success.", __func__, ipAddr.c_str(),
                    usePort);
                return;
            }

            // 当前端口已被占用，尝试抢占下一个端口
            HCCL_INFO("[PreemptPortManager::%s] could not listen on ip[%s], port[%u].", __func__, ipAddr.c_str(), port);
        }
    }
    // 所有端口范围内的端口都已经被占用，没有可用的端口，抢占监听失败
    std::string errormessage
        = "The IP address " + ipAddr + " and port " + std::to_string(usePort) + " have already been bound.";
    NicType nicType = listenSocket->GetNicType();
    if (nicType == NicType::HOST_NIC_TYPE) {
        RPT_INPUT_ERR(true, "EI0019", std::vector<std::string>({"reason"}), std::vector<std::string>({errormessage}));
        HCCL_ERROR(
            "[%s][%s] socket type[2], %s Please check the port status and whether the port is being used by other "
            "process.",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), errormessage.c_str());
    } else {
        RPT_INPUT_ERR(true, "EI0020", std::vector<std::string>({"reason"}), std::vector<std::string>({errormessage}));
        HCCL_ERROR(
            "[%s][%s] socket type[2], %s Please check the port status and whether the port is being used by other "
            "process.",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(), errormessage.c_str());
    }
    std::string portRangeStr = GetRangeStr(portRange);
    HCCL_ERROR("[PreemptPortManager::%s] Complete polling of socket port range:%s", __func__, portRangeStr.c_str());
    HCCL_ERROR(
        "[PreemptPortManager::%s] All ports in socket port range are bound already. "
        "no available port to listen. Please check the ports status, or change the port range to listen on.",
        __func__);
    std::string envName
        = nicType == NicType::HOST_NIC_TYPE ? "HCCL_HOST_SOCKET_PORT_RANGE" : "HCCL_NPU_SOCKET_PORT_RANGE";
    HCCL_ERROR(
        "NOTICE: Users need to make sure ports in %s are available for HCCL."
        "Please double check whether the ports are used by other unexpected processes. "
        "The port ranges size should also be enough when running multi-process HCCL.",
        envName.c_str());
    HCCL_ERROR("NOTICE: The host port range size is not suggested to be smaller than the process number"
               " on current rank.");
    THROW<InvalidParamsException>("No available port to listen");
}

void PreemptPortManager::ReleasePreempt(
    IpPortRef& portRef, const std::shared_ptr<Socket>& listenSocket, [[maybe_unused]] HrtNetworkMode netMode)
{
    std::string ipAddr(listenSocket->GetLocalIp().GetIpStr());
    u32 port = listenSocket->GetListenPort();
    HCCL_INFO("[PreemptPortManager::%s] releasing socket, ip[%s], port[%u].", __func__, ipAddr.c_str(), port);

    bool isListening = IsAlreadyListening(portRef, ipAddr, port);
    // 释放的端口并非正在抢占的端口
    CHK_PRT_RET(
        !isListening,
        HCCL_WARNING(
            "[PreemptPortManager::%s] socket ip[%s], port[%u] is not preempted or has already been released.", __func__,
            ipAddr.c_str(), port), );

    // 释放的端口计数异常
    Referenced& ref = portRef[ipAddr].second;
    CHK_PRT_THROW(
        ref.Count() <= 0,
        HCCL_ERROR(
            "[PreemptPortManager::%s] ref[%d], ip[%s] port[%u] has already been released.", __func__, ref.Count(),
            ipAddr.c_str(), port),
        InvalidParamsException, "socket port duplicate release");

    // 释放绑定端口的Socket
    listenSocket->StopListen();
    int count = ref.Unref();
    CHK_PRT_RET(
        count > 0,
        HCCL_INFO(
            "[PreemptPortManager::%s] release a socket on ip[%s], port[%u], ref[%d].", __func__, ipAddr.c_str(), port,
            count), );

    // 如果端口的计数归零，则不再抢占该端口
    portRef.erase(ipAddr);
    HCCL_INFO(
        "[PreemptPortManager::%s] release preemption of socket on ip[%s], port[%u].", __func__, ipAddr.c_str(), port);
}

bool PreemptPortManager::IsAlreadyListening(const IpPortRef& ipPortRef, const std::string& ipAddr, const u32 port)
{
    auto iterPortRef = ipPortRef.find(ipAddr);
    return iterPortRef != ipPortRef.end() && iterPortRef->second.first == port
           && iterPortRef->second.second.Count() > 0;
}

std::string PreemptPortManager::GetRangeStr(const std::vector<SocketPortRange>& portRangeVec)
{
    std::ostringstream portRangeOss;
    for (auto range : portRangeVec) {
        portRangeOss << " [" << std::to_string(range.min) << ", " << std::to_string(range.max) << "]";
    }
    return portRangeOss.str();
}
} // namespace Hccl
