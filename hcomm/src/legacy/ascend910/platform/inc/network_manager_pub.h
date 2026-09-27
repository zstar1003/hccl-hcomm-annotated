/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NETWORK_MANAGER_PUB_H
#define NETWORK_MANAGER_PUB_H

#include <map>
#include <mutex>
#include <atomic>
#include "hccl/base.h"
#include "hccl_common.h"
#include "hccl_inner_common.h"
#include "hccl_ip_address.h"
#include "adapter_hccp.h"
#include "hccl_net_dev.h"

namespace hccl {
enum class SocketRole {
    SOCKET_ROLE_SERVER = 0, /* server的角色 */
    SOCKET_ROLE_CLIENT = 1, /* client的角色 */
    SOCKET_ROLE_RESERVED    /* 不参与socket连接 */
};

typedef enum {
    TSD_CAPABILITY_MUTIPLE_HCCP = 6,
} TsdCapabilityType;

constexpr s32 CONNECT_FAIL = 0; /* 连接失败 */
constexpr s32 CONNECT_OK = 1;   /* 连接成功 */
constexpr u32 DEFAULT_PHY_ID = INVALID_UINT;
constexpr u32 NO_LISTEN_PORT = INVALID_UINT;
constexpr s32 FIRST_LISTEN = 1; /* 首次监听 */
constexpr s32 LAST_RELEASE = 0; /* 最后释放 */
constexpr u64 TSD_OPEN_EXT_PARA_NUM = 2UL;
constexpr s32 BACKUP_DEVICE_LOG_DEV_VERSION = 0x72318; // MAJOR:0x07, MINOR:0x23, PATCH:0x18

class NetworkManager {
public:
    class InitTool;
    static NetworkManager& GetInstance(s32 deviceLogicID);
    // 初始化网卡，enableWhitelistFlag决定是否感知白名单disable配置
    HcclResult Init(
        NICDeployment nicDeploy, bool enableWhitelistFlag = false, u32 devicePhyId = DEFAULT_PHY_ID,
        bool isHostUseDevNic = false, bool hasBackup = false);
    HcclResult DeInit(NICDeployment nicDeploy, bool resetDeviceFlag = false, bool hasBackup = false);
    HcclResult HeterogInit(u32 devId, const HcclIpAddress& ipAddr, u32 port);
    HcclResult HeterogDeinit(u32 devId, const HcclIpAddress& ipAddr, u32 port);
    HcclResult StartVnic(HcclIpAddress localIp, u32& port);
    HcclResult StopVnic(const HcclIpAddress& localIp, u32 port);
    // port值为无效值0xFFFFFFFF时, 只初始化nic网卡，不启动监听
    HcclResult
    StartNic(const HcclIpAddress& ipAddr, u32& port, bool rdmaFlag, HcclIpAddress ipAddrBackup = HcclIpAddress(0));
    void SetDisableLiteThread(bool disable);
    HcclResult StopNic(const HcclIpAddress& ipAddr, u32 port);
    HcclResult StartHostNetAndListen(const HcclIpAddress& ipAddr, SocketHandle& socketHandle, u32& port, bool rdmaFlag);
    HcclResult StopHostNetAndListen(SocketHandle socketHandle, const HcclIpAddress& ipAddr, u32 port);
    HcclResult StartHostNet(const HcclIpAddress& ipAddr, SocketHandle& socketHandle);
    HcclResult StopHostNet(SocketHandle socketHandle, const HcclIpAddress& ipAddr);
    HcclResult GetRaResourceInfo(RaResourceInfo& raResourceInfo);
    HcclResult Destroy();
    HcclResult HeterogStartListen(const HcclIpAddress& ipAddr, u32 port);
    HcclResult HeterogStopListen(const HcclIpAddress& ipAddr, u32 port, bool isNeedDeinit = false);
    HcclResult PsWorkerRaInit(
        u32 devId, const HcclIpAddress& ipAddr, u32 port, bool isHostUseDevNic = false, bool remoteIsHdc = false,
        bool isBoardVersion = false);
    HcclResult PsWorkerRaDeinit(u32 devId, const HcclIpAddress& ipAddr, u32 port);
    HcclResult InitRdmaHandle(
        u32 devId, const HcclIpAddress& ipAddr, bool disabledLiteThread = false, bool enable910ALite = false);
    HcclResult PingMeshRaPingInit(u32 devLogicId, u32 devPhyId, RaInitConfig* config);
    HcclResult PingMeshRaPingDeinit();
    bool GetRdmaLiteStatus();
    HcclResult GetRdmaHandleByIpAddr(const HcclIpAddress& ipAddr, RdmaHandle& rdmaHandle);
    HcclResult GetNicHandleByIpAddr(const HcclIpAddress& ipAddr, SocketHandle& nicHandle);

    // 重构后的版本
    HcclResult
    InitV2(NICDeployment nicDeploy, bool isBackup, u32 devicePhyId = DEFAULT_PHY_ID, bool isHostUseDevNic = false);
    HcclResult DeInitV2(NICDeployment nicDeploy, bool isBackup, bool resetDeviceFlag = false);

    HcclResult CreateVnicSocketHandle(HcclIpAddress localIp);
    HcclResult StopVnicSocketHandle(const HcclIpAddress& localIp);

    HcclResult CreateNicSocketHandle(const HcclIpAddress& ipAddr);
    HcclResult StopNicSocketHandle(const HcclIpAddress& ipAddr);

    HcclResult CreateRdmaHandle(
        const HcclIpAddress& ipAddr, bool isBackup, NetworkMode netMode, NotifyTypeT notifyType,
        HcclNetDevDeployment netDevDeployment);
    HcclResult StopRdmaHandle(const HcclIpAddress& ipAddr, HcclNetDevDeployment netDevDeployment);

    HcclResult CreateHostSocketHandle(const HcclIpAddress& ipAddr, SocketHandle& socketHandle);
    HcclResult StopHostSocketHandle(const HcclIpAddress& ipAddr);

    // 地址转换
    HcclResult HcclIpAddressConvertHcclAddr(HcclAddress* hccladdr, HcclIpAddress* hcclIP);
    // 创建rdma时 需要获取Mode类型和type类型
    HcclResult GetNotifyType(NotifyTypeT& notifyType) const;
    void GetNetworkMode(NetworkMode& netMode) const;
    HcclResult OccupyIp(const HcclIpAddress& ipAddr, std::map<hccl::HcclIpAddress, IpSocket>& socketMap);
    HcclResult GetNicIp(uint32_t devicePhyId, HcclAddress** addr, uint32_t* len);

private:
    friend InitTool;
    NetworkManager();
    ~NetworkManager();
    HcclResult TsdCapabilityGet(bool& supportMultiProcHCCP);
    HcclResult TsdProcessOpen(bool hasBackup);
    HcclResult InitHostSocket(const HcclIpAddress& addr, SocketHandle& socketHandle) const;
    HcclResult InitDeviceSocket(u32 devicePhysicID, const HcclIpAddress& ipAddr, SocketHandle& socketHandle);
    HcclResult InitRDMA(
        u32 devicePhysicID, const HcclIpAddress& ipAddr, NetworkMode netMode, NotifyTypeT notifyType,
        RdmaHandle& rdmaHandle, bool disabledLiteThread = false, bool enable910ALite = false,
        HcclIpAddress ipAddrBackup = HcclIpAddress(0));
    HcclResult StartListenSocket(const SocketHandle socketHandle, u32& port) const;
    HcclResult StopListenSocket(const SocketHandle socketHandle, u32 port) const;
    HcclResult
    CheckSocketInfo(const SocketHandle socketHandle, const HcclIpAddress& ipAddr, u32 port = NO_LISTEN_PORT) const;
    HcclResult StopNicsSocketListen(const HcclIpAddress& localIp, u32 port);
    HcclResult StopNicsSocket(const HcclIpAddress& ipAddr);
    HcclResult StopAllDeviceNicSockets();
    HcclResult StopAllDeviceVnicSockets();
    HcclResult StopAllHostNicSockets();

    HcclResult CheckAutoListenVersion(bool isAutoPort);

    HcclResult CloseHccpSubProc();
    HcclResult CloseHccpProcess();
    void GetDeviceRaInitConfig(RaInitConfig& config);

    // 重构后的版本
    HcclResult PrepareInit(NICDeployment nicDeploy, u32 devicePhyId, s32& ref);
    HcclResult GetTsdOpen(NICDeployment nicDeploy, bool hasBackup, bool& supportMultiProcHCCP);
    HcclResult GetConfigAndRaInit(struct RaInitConfig& config, bool isHdcV2, NICDeployment nicDeploy);

    HcclResult PrepareDeInit(s32& ref, NICDeployment nicDeploy);
    HcclResult
    GetConfigAndRaDeinit(struct RaInitConfig& config, NICDeployment nicDeploy, bool& isMultiProc, bool hasBackup);

    // device的rdma区分是否是rdmalite的模式, host不需要
    HcclResult RdmaSupportLite(RdmaHandle rdmaHandle);

    s32 deviceLogicId_;
    u32 devicePhyId_;
    bool isHostUseDevNic_ = false;
    bool isRdmaLiteEn_ = false;
    bool isRaInitRepeated_ = false;
    bool isRaDeInit_ = false;
    bool isEnableHdcAsync_ = false;
    bool isDisableLiteThread_ = false;
    RaResourceInfo raResourceInfo_;
    Referenced deviceNicInitRef_;
    Referenced hostNicInitRef_;
    std::map<HcclIpAddress, Referenced> hostNicSocketClientRef_;
    std::map<HcclIpAddress, std::map<u32, Referenced>> IPPortListenRefMapHost_;
    std::map<HcclIpAddress, std::map<u32, Referenced>> IPPortListenRefMapDevice_;
    std::map<HcclIpAddress, std::map<u32, Referenced>> IPPortListenRefMapVnicDevice_;
    std::mutex raLock_;
    std::mutex hccpProcInfoMutex_;
    NotifyTypeT notifyType_;
    static NetworkManager* nmInstance[MAX_DEV_NUM];
    pid_t subPid_{0};
    bool isTsdOpened_{false};
    std::mutex memResMutex_;
    std::vector<HcclAddress> nicIpAddrs_;
    bool isTsdProcessOpen_ = false;
};

class NetworkManager::InitTool {
public:
    InitTool();
    ~InitTool();

private:
    static std::atomic<unsigned> initCount;
};

static NetworkManager::InitTool g_networkManagerInit; // 关键变量

} // namespace hccl

#endif
