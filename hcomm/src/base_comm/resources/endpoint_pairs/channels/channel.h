/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHANNEL_H
#define CHANNEL_H

#include <memory>
#include <vector>
#include <unordered_map>
#include "hccl/hccl_res.h"
#include "hccl/hccl_types.h"
#include "hcomm_res_defs.h"
#include "hccl_mem_defs.h"
#include <string>
#include <unordered_map>
#include <vector>
#include "enum_factory.h"
#include "hcomm_nic_plugin.h"
#include "hcomm_channel.h"
#include "mem_device_pub.h"

// Orion
#include "transport_status.h"
#include "ip_address.h"
#include "topo_common_types.h"
#include "virtual_topo.h"

class AicpuTsChannelHelper;

namespace hcomm {

MAKE_ENUM(ChannelStatus, INIT, SOCKET_OK, SOCKET_TIMEOUT, READY, FAILED, RES_LOC_UNAVAIL, RES_RMT_UNAVAIL)

/**
 * @brief 通道种类（与 HcommChannelRes.channelTypeList 中 u32 数值一致；由 CommEngine + CommProtocol 推导）。
 */
enum class HcommChannelKind : uint32_t {
    INVALID = 0U,
    AICPU_TS_URMA = 1U,
    AICPU_TS_ROCE = 2U,
    AICPU_TS_HCCS = 3U,
    CPU_ROCE = 4U,
    AIV_UB_MEM = 5U,
    AICPU_TS_UBOE = 6U,
    AICPU_TS_ROCE_V2 = 7U,
    AIV_URMA = 8U,
    AICPU_TS_UB_RTP = 9U,
    AICPU_TS_PCIE = 10U,
};

inline const char* HcommChannelKindToString(HcommChannelKind kind)
{
    switch (kind) {
        case HcommChannelKind::INVALID:
            return "INVALID";
        case HcommChannelKind::AICPU_TS_URMA:
            return "AICPU_TS_URMA";
        case HcommChannelKind::AICPU_TS_ROCE:
            return "AICPU_TS_ROCE";
        case HcommChannelKind::AICPU_TS_HCCS:
            return "AICPU_TS_HCCS";
        case HcommChannelKind::CPU_ROCE:
            return "CPU_ROCE";
        case HcommChannelKind::AIV_UB_MEM:
            return "AIV_UB_MEM";
        case HcommChannelKind::AICPU_TS_UBOE:
            return "AICPU_TS_UBOE";
        case HcommChannelKind::AICPU_TS_ROCE_V2:
            return "AICPU_TS_ROCE_V2";
        case HcommChannelKind::AIV_URMA:
            return "AIV_URMA";
        case HcommChannelKind::AICPU_TS_UB_RTP:
            return "AICPU_TS_UB_RTP";
        case HcommChannelKind::AICPU_TS_PCIE:
            return "AICPU_TS_PCIE";
        default:
            return "UNKNOWN";
    }
}

/**
 * @note 职责：一个EndPointPair上的建立的通信通道的C++抽象接口类声明。
 * 管理该通信通道Channel对上的同步信号Notify、通信队列（如qp、jetty等）等资源管理，负责建立连接，以及注册内存、同步信号等的交换。
 */
class Channel {
public:
    Channel() {};
    virtual ~Channel() = default;

    // 禁拷贝（避免切片/资源重复释放等）
    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;

    // 视需要决定是否允许移动；很多资源类也会禁移动
    Channel(Channel&&) = default;
    Channel& operator=(Channel&&) = default;

    // ------------------ 控制面接口 ------------------
    virtual HcclResult Init() = 0;
    virtual HcclResult GetNotifyNum(uint32_t* notifyNum) const = 0;
    virtual HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) = 0;
    virtual ChannelStatus GetStatus() = 0;
    virtual HcclResult UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum);

    virtual HcclResult Clean() = 0;
    virtual HcclResult Resume() = 0;

    virtual HcommChannelKind GetChannelKind() const;
    CommEngine GetEngine() const { return engine_; }
    virtual const HcommChannelDesc& GetChannelDesc() const = 0;
    virtual HcclResult Serialize(std::shared_ptr<hccl::DeviceMem>& out);
    virtual void AddPtrArrayDevMem(std::shared_ptr<hccl::DeviceMem> ptrArrayMem);
    // ------------------ 数据面接口 ------------------
    virtual HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) = 0;
    virtual HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) = 0;
    virtual HcclResult WriteWithNotify(void* dst, const void* src, const uint64_t len, uint32_t remoteNotifyIdx) = 0;
    virtual HcclResult Write(void* dst, const void* src, uint64_t len) = 0;
    virtual HcclResult Read(void* dst, const void* src, uint64_t len) = 0;
    virtual HcclResult ChannelFence() = 0;
    virtual HcclResult ChannelDrain() { return HCCL_E_NOT_SUPPORT; }

    // ------------------ NIC插件相关 ------------------
    void SetNicChannelCtx(HcommNicChannelOps* nicOps, void* nicCtx)
    {
        nicOps_ = nicOps;
        nicCtx_ = nicCtx;
    }
    HcommNicChannelOps* GetNicOps() const { return nicOps_; }
    void* GetNicCtx() const { return nicCtx_; }

    // ------------------ 工具方法 ------------------
    static ChannelStatus TransportStatusToChannelStatus(
        Hccl::TransportStatus ts, const EndpointDesc& localEp, const HcommChannelDesc& channelDesc);

    // ------------------ 共享 Jetty 模式 ------------------
    // 由 CreateChannelsLoop 根据本次调用是否共享（HcommChannelConfig.isSharedQueue）设置；
    // BuildConnection 据此决定走复用路径还是自建独立 jetty，避免 Endpoint 全局标记跨调用串扰。
    void SetSharedJetty(bool enable) { isSharedJetty_ = enable; }
    bool IsSharedJetty() const { return isSharedJetty_; }

    // ------------------ 工厂 ------------------
    static HcclResult CreateChannel(
        EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc channelDesc, std::shared_ptr<Channel>& out,
        bool isSharedQueue = false);

    virtual AicpuTsChannelHelper* GetAicpuTsHelper() { return nullptr; }

    bool IsDeviceEntityReady() const { return deviceEntityReady_; }
    void SetDeviceEntityReady() { deviceEntityReady_ = true; }

protected:
    void ReleasePtrArrayDevMems();

    HcommChannelKind channelKind_{HcommChannelKind::INVALID};
    CommEngine engine_{COMM_ENGINE_RESERVED};
    std::vector<std::shared_ptr<hccl::DeviceMem>> ptrArrayDevMems_{};
    bool deviceEntityReady_{false};
    bool isSharedJetty_{false};

    HcommNicChannelOps* nicOps_{nullptr};
    void* nicCtx_{nullptr};
};

} // namespace hcomm
#endif // CHANNEL_H
