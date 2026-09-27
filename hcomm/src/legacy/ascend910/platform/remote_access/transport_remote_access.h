/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_REMOTE_ACCESS_H
#define TRANSPORT_REMOTE_ACCESS_H

#include "transport_ibverbs.h"
#include "dispatcher_pub.h"

namespace hccl {
constexpr u32 MAX_HDC_CHANEL_NUM = 1024;
constexpr u32 NOTIFY_BUFFER_SIZE = 8;
constexpr u32 SEND_WRLIST_MAX_COUNT = 100;

using RemoteAccessPara = struct TagRemoteAccessParaS {
public:
    u32 role{0};                          // server:0 client:1
    HcclIpAddress localIp;                // 本端rank ip
    u32 localRank{INVALID_VALUE_RANKID};  // 本端rankuser rank
    u32 remoteRank{INVALID_VALUE_RANKID}; // 对端rankuser rank
    FdHandle socketFdhandle{nullptr};
    SocketHandle nicSocketHandle{nullptr};
    RdmaHandle nicRdmaHandle{nullptr};
    RaResourceInfo raResourceInfo;

    TagRemoteAccessParaS() {}

    TagRemoteAccessParaS(const struct TagRemoteAccessParaS& that)
    {
        role = (that.role);
        localIp = (that.localIp);
        localRank = (that.localRank);
        remoteRank = (that.remoteRank);
        socketFdhandle = (that.socketFdhandle);
        nicSocketHandle = (that.nicSocketHandle);
        nicRdmaHandle = (that.nicRdmaHandle);
        raResourceInfo = (that.raResourceInfo);
    }

    struct TagRemoteAccessParaS& operator=(struct TagRemoteAccessParaS& that)
    {
        if (&that != this) {
            role = (that.role);
            localIp = (that.localIp);
            localRank = (that.localRank);
            remoteRank = (that.remoteRank);
            socketFdhandle = (that.socketFdhandle);
            nicSocketHandle = (that.nicSocketHandle);
            nicRdmaHandle = (that.nicRdmaHandle);
            raResourceInfo = (that.raResourceInfo);
        }

        return *this;
    }
};

struct NotifyMsg {
    void* addr{nullptr};
    u64 len{0};
    s32 mrRegFlag{0};
    u64 offset{0};
};

class TransportRemoteAccess {
public:
    explicit TransportRemoteAccess(
        const std::string tag, const HcclDispatcher dispatcher, const std::unique_ptr<NotifyPool>& notifyPool,
        const RemoteAccessPara& remoteAccessPara, const std::vector<MemRegisterAddr>& memRegistInfos,
        s32 deviceLogicId);
    ~TransportRemoteAccess();
    HcclResult Init();
    HcclResult RemoteRead(const std::vector<HcomRemoteAccessAddrInfo>& addrInfos, Stream& stream);
    HcclResult RemoteWrite(const std::vector<HcomRemoteAccessAddrInfo>& addrInfos, Stream& stream);

private:
    HcclResult CreateQp();
    HcclResult MrRegister();
    HcclResult NotifyRegister();
    HcclResult GetRemoteNotifyInfo();
    HcclResult SetLocalNotify();
    HcclResult CreateNotify();
    HcclResult CreateNotifyValueBuffer();
    HcclResult RdmaDataTransport(const std::vector<HcomRemoteAccessAddrInfo>& addrInfos, s32 rdmaOp);
    HcclResult ReadRemoteNotifyBuffer();
    HcclResult ConnectQp();
    const HcclDispatcher dispatcher_;
    const std::unique_ptr<NotifyPool>& notifyPool_;
    const std::vector<MemRegisterAddr> MemRegistInfos_; // 本端注册地址
    RemoteAccessPara RemoteAccessPara_;
    QpHandle handle_; // QP handle
    std::shared_ptr<LocalIpcNotify> ackNotify_;
    s32 access_;                     // rdma 访问权限，支持本端/远端写数据、远端读数据
    u32 notifySize_;                 // notify buffer size，8字节
    std::vector<void*> localRegMem_; // 本端注册成功的地址信息，用于解注册
    NotifyMsg ackNotifyMsg_;         // 本端notify的信息
    NotifyMsg remoteNotifyDataMsg_;
    static std::array<DeviceMem, MAX_MODULE_DEVICE_NUM> notifyValueMem_;
    static std::array<std::mutex, MAX_MODULE_DEVICE_NUM> notifyValueMutex_;
    const u64 notifyValueSize_{LARGE_PAGE_MEMORY_MIN_SIZE};            // 避免申请小页内存。最小2*1024*1024
    static std::array<Referenced, MAX_MODULE_DEVICE_NUM> instanceRef_; // 实例计数，用于释放静态资源
    const std::string tag_;
    const std::chrono::seconds timeout_;
    s32 deviceLogicId_;
};
} // namespace hccl

#endif // TRANSPORT_REMOTE_ACCESS_H
