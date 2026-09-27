/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TS_ROCE_CHANNEL_V2_H
#define AICPU_TS_ROCE_CHANNEL_V2_H

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../channel.h"
#include "aicpu_ts_channel_helper.h"
#include "enum_factory.h"
#include "hccl_common.h"
#include "mem_device_pub.h"
#include <mutex>
#include "hcomm/hcomm_res_entity_defs.h"

// Orion
#include "../../../../../../legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "../../../../../../legacy/ascend950/unified_platform/resource/buffer/local_rdma_rma_buffer_v2.h"
#include "remote_rma_buffer.h"
#include "./dev_rdma_connection_v2.h"
#include "rdma_local_notify.h"
#include "dev_buffer.h"

namespace hcomm {
/**
 * @note 职责：Channel的AicpuTs通信引擎、RoCE协议的类派生
 */
constexpr u32 RDMA_NOTIFY_NUM = 3;

using EndpointDeviceLoc = decltype(EndpointLoc::device);

class AicpuTsRoceChannelV2 final : public Channel {
public:
    MAKE_ENUM(RdmaStatus, INIT, SOCKET_OK, QP_CREATED, DATA_EXCHANGE, QP_MODIFIED, CONN_OK)

    AicpuTsRoceChannelV2(EndpointHandle endpointHandle, HcommChannelDesc channelDesc, CommEngine engine);
    ~AicpuTsRoceChannelV2() override;

    HcclResult Init() override;
    HcclResult GetNotifyNum(uint32_t* notifyNum) const override;
    HcclResult GetBufferNum(uint32_t* bufferNum) const;
    HcclResult GetQpNum(uint32_t* qpNum) const;
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override;
    ChannelStatus GetStatus() override;
    HcclResult GetStatus(ChannelStatus& status);
    HcclResult ProcessStatus();
    HcommChannelKind GetChannelKind() const override;
    const HcommChannelDesc& GetChannelDesc() const override { return channelDesc_; }

    std::string Describe() const;

    CommEngine GetCommEngine() const { return engine_; }
    CommProtocol GetCommProtocol() const { return channelDesc_.remoteEndpoint.protocol; }

    HcclResult BuildAndGetDevChannelEntity(uint64_t* devChannelEntityPtr);
    HcclResult PreAllocDevChannelEntity(uint64_t* devChannelEntityPtr);
    HcclResult FillDevChannelEntity();

    HcclResult H2DResPack(std::vector<char>& buffer) const;

    HcclResult Serialize(std::shared_ptr<hccl::DeviceMem>& out) override;

    HcclResult Clean() override;
    HcclResult Resume() override;

    HcclResult NotifyRecord([[maybe_unused]] const uint32_t remoteNotifyIdx) override { return HCCL_SUCCESS; }
    HcclResult
    NotifyWait([[maybe_unused]] const uint32_t localNotifyIdx, [[maybe_unused]] const uint32_t timeout) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult WriteWithNotify(
        [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] const uint64_t len,
        [[maybe_unused]] uint32_t remoteNotifyIdx) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult
    Write([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult
    Read([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult ChannelFence() override { return HCCL_SUCCESS; }

    AicpuTsChannelHelper* GetAicpuTsHelper() override { return &aicpuTsHelper_; }

private:
    AicpuTsChannelHelper aicpuTsHelper_;
    HcclResult ParseInputParam();
    HcclResult BuildConnection();
    HcclResult BuildNotify();
    HcclResult BuildBuffer();
    HcclResult BuildNotifyValueBuffer();
    HcclResult BuildSocket();
    HcclResult StartListen();

    HcclResult CheckSocketStatus();
    HcclResult CreateQp();
    HcclResult ExchangeData();
    HcclResult ModifyQp();

    void NotifyVecPack(Hccl::BinaryStream& binaryStream);
    HcclResult BufferVecPack(Hccl::BinaryStream& binaryStream);
    HcclResult ConnVecPack(Hccl::BinaryStream& binaryStream);

    HcclResult NotifyVecUnpack(Hccl::BinaryStream& binaryStream);
    HcclResult RmtBufferVecUnpackProc(Hccl::BinaryStream& binaryStream);
    HcclResult ConnVecUnpackProc(Hccl::BinaryStream& binaryStream);

    HcclResult BuildAndGetLocNotifyInfo(RegedNotifyEntity** notify) const;
    HcclResult BuildAndGetRmtNotifyInfo(RegedNotifyEntity** notify) const;
    HcclResult BuildAndGetRmtBufInfo(std::vector<RegedBufferEntity>& bufList, RegedBufferEntity** bufferEntityPtr);
    HcclResult BuildAndGetLocBufInfo(std::vector<RegedBufferEntity>& bufList, RegedBufferEntity** bufferEntityPtr);
    HcclResult BuildAndGetSqContext(std::vector<SqContext>& sqList, SqContext** sqContextPtr);
    HcclResult BuildAndGetCqContext(std::vector<CqContext>& cqList, CqContext** cqContextPtr);
    HcclResult BuildHostEntity(
        ChannelEntity& hostEntity, std::vector<RegedBufferEntity>& locBufList,
        std::vector<RegedBufferEntity>& rmtBufList, std::vector<SqContext>& sqList, std::vector<CqContext>& cqList);

    void FreeDeviceMemories();
    void ReleaseDeviceEntitySlab();

    std::vector<char> GetLocalNotifyUniqueIds() const;
    std::vector<char> GetRemoteNotifyUniqueIds() const;
    std::vector<char> GetLocBufferUniqueIds() const;
    std::vector<char> GetRmtBufferUniqueIds() const;
    std::vector<char> GetNotifyValueBufferUniqueIds() const;
    std::vector<char> GetSingleRmaBufferUniqueId(u64 addr, u64 size, u32 key) const;
    std::vector<char> GetConnUniqueIds() const;
    std::vector<char> GetUniqueId() const;
    HcclResult PackOpData(std::vector<char>& data) const;

    // 入参
    EndpointHandle endpointHandle_;
    HcommChannelDesc channelDesc_;
    CommEngine engine_;

    // 转换参数
    EndpointDesc localEp_;
    EndpointDesc remoteEp_;
    uint32_t notifyNum_{0};
    Hccl::Socket* socket_{nullptr};
    const Hccl::SocketConfig* socketConfig_{nullptr};
    RdmaHandle rdmaHandle_{nullptr};
    uint32_t devicePhyId_{};

    std::vector<std::unique_ptr<DevRdmaConnectionV2>> connections_{};
    std::vector<Hccl::LocalRdmaRmaBuffer*> localRmaBuffers_{};
    std::vector<std::unique_ptr<Hccl::RdmaLocalNotify>> localNotifies_{};
    std::shared_ptr<Hccl::DevBuffer> notifyValueMem_{nullptr};
    std::unique_ptr<Hccl::LocalRdmaRmaBuffer> notifyValueBuffer_{nullptr};
    uint32_t bufferNum_{0};
    uint32_t connNum_{0};
    ChannelStatus channelStatus_{ChannelStatus::INIT};
    RdmaStatus rdmaStatus_{RdmaStatus::INIT};
    std::vector<std::unique_ptr<Hccl::RemoteRdmaRmaBuffer>> remoteNotifies_{};
    std::vector<std::unique_ptr<Hccl::RemoteRdmaRmaBuffer>> rmtRmaBuffers_{};
    ExchangeRdmaConnDto rmtConnDto_;
    std::mutex remoteMemsMutex_{};
    std::vector<CommMem> remoteUserMems_{};
    std::vector<std::string> memInfoCopies_{};
    std::vector<char*> memInfoPointers_{};
    bool cacheValid_{false};

    void* devChannelEntitySlab_{nullptr};
    size_t devChannelEntitySlabSize_{0};
};

} // namespace hcomm

#endif // AICPU_TS_ROCE_CHANNEL_V2_H
