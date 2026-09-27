/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "preempt_port_manager.h"
#include <sstream>
#include "adapter_error_manager_pub.h"
#include "hccl_common.h"

namespace hccl {

bool PreemptPortManager::initialized = false;

PreemptPortManager::PreemptPortManager()
{
    // 根据host or device nic区分
    IpPortRef hostPortRef;
    preemptSockets_.emplace(NICDeployment::NIC_DEPLOYMENT_HOST, hostPortRef);
    IpPortRef devPortRef;
    preemptSockets_.emplace(NICDeployment::NIC_DEPLOYMENT_DEVICE, devPortRef);
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
        HCCL_RUN_WARNING("[PreemptPortManager][GetInstance]deviceLogicId[%d] is invalid", deviceLogicId), instance[0]);

    return instance[deviceLogicId];
}

HcclResult PreemptPortManager::ListenPreempt(
    const std::shared_ptr<HcclSocket>& listenSocket, const std::vector<HcclSocketPortRange>& portRange, u32& usePort)
{
    CHK_PRT_RET(
        !initialized, HCCL_ERROR("[PreemptPortManager][ListenPreempt] preempt port manager has already been released."),
        HCCL_E_INTERNAL);

    CHK_SMART_PTR_NULL(listenSocket);
    NicType socketType = listenSocket->GetSocketType();
    NICDeployment nicDeploy = socketType == NicType::HOST_NIC_TYPE ? NICDeployment::NIC_DEPLOYMENT_HOST :
                                                                     NICDeployment::NIC_DEPLOYMENT_DEVICE;
    std::lock_guard<std::mutex> lock(preemptMutex_);
    HcclResult ret = PreemptPortInRange(preemptSockets_[nicDeploy], listenSocket, nicDeploy, portRange, usePort);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[PreemptPortManager][ListenPreempt] listen preempt fail, socketType[%u]", socketType), ret);
    HCCL_INFO(
        "[PreemptPortManager][ListenPreempt] listening on port[%u] for socketType[%u] success.", usePort, socketType);
    return HCCL_SUCCESS;
}

HcclResult PreemptPortManager::Release(const std::shared_ptr<HcclSocket>& listenSocket)
{
    CHK_PRT_RET(
        !initialized, HCCL_RUN_WARNING("[PreemptPortManager][Release] preempt port manager has already been released."),
        HCCL_SUCCESS);

    CHK_SMART_PTR_NULL(listenSocket);
    NicType socketType = listenSocket->GetSocketType();
    NICDeployment nicDeploy = socketType == NicType::HOST_NIC_TYPE ? NICDeployment::NIC_DEPLOYMENT_HOST :
                                                                     NICDeployment::NIC_DEPLOYMENT_DEVICE;

    std::lock_guard<std::mutex> lock(preemptMutex_);
    HcclResult ret = ReleasePreempt(preemptSockets_[nicDeploy], listenSocket, nicDeploy);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[PreemptPortManager][Release] release fail, socketType[%u]", socketType), ret);
    HCCL_INFO("[PreemptPortManager][Release] release socket of type[%u] success.", socketType);
    return HCCL_SUCCESS;
}

HcclResult PreemptPortManager::PreemptPortInRange(
    IpPortRef& portRef, const std::shared_ptr<HcclSocket>& listenSocket, [[maybe_unused]] NICDeployment nicDeploy,
    const std::vector<HcclSocketPortRange>& portRange, u32& usePort)
{
    std::string ipAddr(listenSocket->GetLocalIp().GetReadableAddress());
    if (portRef.find(ipAddr) != portRef.end()) {
        // 如果在这个IP上已经有已经抢占的port，则复用这个port
        usePort = portRef[ipAddr].first;
        CHK_RET(listenSocket->Listen(usePort));
        portRef[ipAddr].second.Ref();
        HCCL_INFO(
            "[PreemptPortManager][PreemptPortInRange] for ip[%s], port[%u] has already been listened, "
            "ref count[%u].",
            ipAddr.c_str(), usePort, portRef[ipAddr].second.Count());
        return HCCL_SUCCESS;
    }

    // 如果不设置portRange，不走端口抢占逻辑
    bool isPreempt = true;
    if (!GetExternalInputHostPortSwitch()) {
        isPreempt = false;
    }

    // 如果这个IP上没有抢占过的port，则轮询输入的端口范围，找到一个可用的端口
    for (auto& range : portRange) {
        for (u32 port = range.min; port <= range.max; ++port) {
            HcclResult ret = listenSocket->Listen(port);
            if (ret == HCCL_SUCCESS) {
                usePort = listenSocket->GetLocalPort();
                if (isPreempt) {
                    // 抢占端口成功，将端口记录到计数器中，并作为出参返回
                    portRef[ipAddr].first = usePort;
                    portRef[ipAddr].second.Ref();
                }
                HCCL_INFO(
                    "[PreemptPortManager][PreemptPortInRange] listen on ip[%s] and port[%u] success.", ipAddr.c_str(),
                    usePort);
                return HCCL_SUCCESS;
            }
            // 非已占用的错误，直接报错退出
            CHK_PRT_RET(
                ret != HCCL_E_UNAVAIL,
                HCCL_ERROR(
                    "[PreemptPortManager][PreemptPortInRange] attempt to listen on port[%u] for ip[%s] fail."
                    " some unexpected error occurs, errNo[0x%016llx]. attempt is stopped.",
                    port, ipAddr.c_str(), ret),
                ret);
            // 当前端口已被占用，尝试抢占下一个端口
            HCCL_INFO(
                "[PreemptPortManager][PreemptPortInRange] could not listen on ip[%s], port[%u].", ipAddr.c_str(), port);
        }
    }
    // 所有端口范围内的端口都已经被占用，没有可用的端口，抢占监听失败
    std::string portRangeStr = GetRangeStr(portRange);
    std::string errormessage;
    if (usePort == HCCL_INVALID_PORT) {
        errormessage = "The IP address " + ipAddr + " has no available port in range" + portRangeStr
                       + " (all ports in the range have already been bound).";
    } else {
        errormessage
            = "The IP address " + ipAddr + " and port " + std::to_string(usePort) + " have already been bound.";
    }
    NicType socketType = listenSocket->GetSocketType();
    if (socketType == NicType::HOST_NIC_TYPE) {
        RPT_INPUT_ERR(true, "EI0019", std::vector<std::string>({"reason"}), std::vector<std::string>({errormessage}));
    } else {
        RPT_INPUT_ERR(true, "EI0020", std::vector<std::string>({"reason"}), std::vector<std::string>({errormessage}));
    }
    HCCL_ERROR(
        "[PreemptPortManager][PreemptPortInRange] Complete polling of socket port range:%s", portRangeStr.c_str());
    HCCL_ERROR("[PreemptPortManager][PreemptPortInRange] All ports in socket port range are bound already. "
               "no available port to listen. Please check the ports status, or change the port range to listen on.");
    HCCL_ERROR("NOTICE: Users need to make sure ports in HCCL_HOST_SOCKET_PORT_RANGE and HCCL_NPU_SOCKET_PORT_RANGE "
               "are available for HCCL. Please double check whether the ports are used by other unexpected processes. "
               "The port ranges size should also be enough when running multi-process HCCL.");
    HCCL_ERROR("NOTICE: The host/npu port range size is not suggested to be smaller than the process number"
               " on current rank.");
    return HCCL_E_UNAVAIL;
}

HcclResult PreemptPortManager::ReleasePreempt(
    IpPortRef& portRef, const std::shared_ptr<HcclSocket>& listenSocket, [[maybe_unused]] NICDeployment nicDeploy)
{
    std::string ipAddr(listenSocket->GetLocalIp().GetReadableAddress());
    u32 port = listenSocket->GetLocalPort();
    HCCL_INFO("[PreemptPortManager][ReleasePreempt] releasing socket, ip[%s], port[%u].", ipAddr.c_str(), port);

    bool isListening = IsAlreadyListening(portRef, ipAddr, port);
    // 释放的端口并非正在抢占的端口
    CHK_PRT_RET(
        !isListening,
        HCCL_RUN_WARNING(
            "[PreemptPortManager][ReleasePreempt] "
            "socket ip[%u], port[%u] is not preempted or has already been released.",
            ipAddr.c_str(), port),
        HCCL_SUCCESS);

    // 释放的端口计数异常
    Referenced& ref = portRef[ipAddr].second;
    CHK_PRT_RET(
        ref.Count() <= 0,
        HCCL_ERROR(
            "[PreemptPortManager][ReleasePreempt] ref[%u], ip[%s] port[%u] has already been released. "
            "Please do not duplicate release.",
            ref.Count(), ipAddr.c_str(), port),
        HCCL_E_INTERNAL);

    // 释放绑定端口的Socket
    CHK_RET(listenSocket->DeInit());
    int count = ref.Unref();
    CHK_PRT_RET(
        count > 0,
        HCCL_INFO(
            "[PreemptPortManager][ReleasePreempt] release a socket on ip[%s], port[%u], ref[%u].", ipAddr.c_str(), port,
            count),
        HCCL_SUCCESS);

    // 如果端口的计数归零，则不再抢占该端口
    portRef.erase(ipAddr);
    HCCL_INFO(
        "[PreemptPortManager][ReleasePreempt] release preemption of socket on ip[%s], port[%u].", ipAddr.c_str(), port);
    return HCCL_SUCCESS;
}

bool PreemptPortManager::IsAlreadyListening(const IpPortRef& ipPortRef, const std::string& ipAddr, const u32 port)
{
    auto iterPortRef = ipPortRef.find(ipAddr);
    return iterPortRef != ipPortRef.end() && iterPortRef->second.first == port
           && iterPortRef->second.second.Count() > 0;
}

std::string PreemptPortManager::GetRangeStr(const std::vector<HcclSocketPortRange>& portRangeVec)
{
    std::ostringstream portRangeOss;
    for (auto range : portRangeVec) {
        portRangeOss << " [" << std::to_string(range.min) << ", " << std::to_string(range.max) << "]";
    }
    return portRangeOss.str();
}
} // namespace hccl
