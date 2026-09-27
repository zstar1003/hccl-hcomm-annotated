/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_EXPERIMENTAL_HOST_CPU_ROCE_CHANNEL_H
#define HCOMM_EXPERIMENTAL_HOST_CPU_ROCE_CHANNEL_H

#include <mutex>

#include "channel.h"
#include "enum_factory.h"
#include "hccl_common.h"
#include "socket_mgr.h"
#include "infiniband/verbs.h"

// Orion
#include "socket.h"
#include "local_rdma_rma_buffer_v2.h"
#include "remote_rma_buffer.h"
#include "host_rdma_connection.h"
#include "task_param.h"

#include "exchange_data_format.h"
#include "private_types.h"

namespace hcomm_experimental {

class HostCpuRoceChannel final : public Channel {
public:
    MAKE_ENUM(RdmaStatus, INIT, SOCKET_OK, CAP_EXCHANGED, QP_CREATED, DATA_EXCHANGE, QP_MODIFIED, CONN_OK)

    HostCpuRoceChannel(EndpointHandle endpointHandle, HcommChannelDesc channelDesc);
    ~HostCpuRoceChannel();

    HcclResult Init() override;
    HcclResult GetNotifyNum(uint32_t* notifyNum) const override;
    HcclResult GetRemoteMem(HcclMem** remoteMem, uint32_t* memNum, char** memTags) override;
    HcclResult GetUserRemoteMem(CommMem** remoteMem, char*** memTag, uint32_t* memNum) override;
    ChannelStatus GetStatus() override;
    HcclResult GetStatus(ChannelStatus& status);
    HcclResult ProcessStatus();

    std::string Describe() const;

    HcclResult SetDfxCallback(std::function<HcclResult(const Hccl::TaskParam&, u64)> callback);

    // 数据面调用verbs接口
    HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override;
    HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override;
    HcclResult WriteWithNotify(void* dst, const void* src, const uint64_t len, uint32_t remoteNotifyIdx) override;
    HcclResult Write(void* dst, const void* src, uint64_t len) override;
    HcclResult Read(void* dst, const void* src, uint64_t len) override;
    HcclResult ChannelFence() override;
    HcclResult GetHcclBuffer(void*& addr, uint64_t& size);

private:
    HcclResult WaitForFenceCompletion();

    virtual HcclResult Clean() override;
    virtual HcclResult Resume() override;
    HcclResult ExchangeCapability();
    HcclResult ExchangeDataHybird();
    HcclResult GetRemoteAddrHybird(hccl::MemType memType, u8*& data, u64& size);
    HcclResult ParseRecvExchangeDataHybird();
    HcclResult BuildExchangeDataHybird();
    HcclResult BuildExchangeDataLengthHybird();

    HcclResult RegisterUserMemHybird();
    HcclResult BuildNotifyWrHybird(const uint32_t remoteNotifyIdx, struct ibv_send_wr& notifRecordWr);
    HcclResult WriteWithNotifyHybrid(void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx);
    HcclResult NotifyWaitHybrid(uint32_t localNotifyIdx, uint32_t timeout);

    HcclResult CreateNotifyHybird(hccl::MemType notifyType, uint32_t notifyId);
    HcclResult CreateNotifyValueBufferHybird();
    HcclResult CreateNotifyBufferHybird(hccl::MemType notifyType, uint32_t notifyId, u8*& data, u64& size);
    hccl::MemType NotifyIdToMemtypeHybird(uint32_t remoteNotifyIdx);
    HcclResult ConnectSingleQpHybrid(std::function<bool()> needStop);

private:
    HcclResult ParseInputParam();
    HcclResult StartListen();
    HcclResult BuildSocket();
    HcclResult BuildConnection();
    HcclResult BuildNotify();
    HcclResult BuildBuffer();

    HcclResult CheckSocketStatus();
    HcclResult CreateQp();
    HcclResult ExchangeData();
    HcclResult ModifyQp();

    void NotifyVecPack(Hccl::BinaryStream& binaryStream);
    HcclResult BufferVecPack(Hccl::BinaryStream& binaryStream);
    HcclResult ConnVecPack(Hccl::BinaryStream& binaryStream);
    // void HandshakeMsgPack(Hccl::BinaryStream &binaryStream);

    // HcclResult HandshakeMsgUnpack(Hccl::BinaryStream &binaryStream);
    HcclResult NotifyVecUnpack(Hccl::BinaryStream& binaryStream);
    HcclResult RmtBufferVecUnpackProc(Hccl::BinaryStream& binaryStream);
    HcclResult ConnVecUnpackProc(Hccl::BinaryStream& binaryStream);

    std::vector<Hccl::QpInfo> GetQpInfos() const; // in Connection

    HcclResult IbvPostRecv() const;
    HcclResult PrepareNotifyWrResource(
        uint32_t qpIdx, const uint64_t len, const uint32_t remoteNotifyIdx, struct ibv_send_wr& notifyRecordWr,
        Hccl::TaskParam& taskParam) const;
    HcclResult PrepareWriteWrResource(
        const void* dst, const void* src, const uint64_t len, const uint32_t remoteNotifyIdx,
        struct ibv_send_wr& writeWithNotifyWr, Hccl::TaskParam& taskParam) const;

    HcclResult
    PostRdmaOp(const char* caller, ibv_wr_opcode opcode, void* localAddr, const void* remoteAddr, uint64_t len);
    void BuildRdmaWr(
        const char* caller, ibv_wr_opcode opcode, void* localAddr, const void* remoteAddr, uint64_t len,
        size_t localIdx, size_t rmtIdx, struct ibv_send_wr& wr, struct ibv_sge& sg) const;
    HcclResult PostAndCheckSend(struct ibv_qp* qp, const uint32_t qpIdx, const char* caller, struct ibv_send_wr& wr);
    HcclResult FindLocalBuffer(const uint64_t addr, const uint64_t len, size_t& targetIdx) const;
    HcclResult FindRemoteBuffer(const uint64_t addr, const uint64_t len, size_t& targetIdx) const;
    HcclResult ReportWcStatusError(enum ibv_wc_status status);

    // Wrapper for stub
    int IbvPollCq(ibv_cq* sendCq, uint32_t numEntries, ibv_wc* wc) const { return ibv_poll_cq(sendCq, numEntries, wc); }

    // 入参
    EndpointHandle endpointHandle_;
    HcommChannelDesc channelDesc_;

    // 转换参数
    EndpointDesc localEp_;
    EndpointDesc remoteEp_;
    uint32_t notifyNum_{0};
    Hccl::Socket* socket_{nullptr};
    const Hccl::SocketConfig* socketConfig_{nullptr};
    RdmaHandle rdmaHandle_{nullptr};

    std::vector<std::unique_ptr<HostRdmaConnection>> connections_{};
    std::vector<Hccl::LocalRdmaRmaBuffer*> localRmaBuffers_{};
    std::vector<uint32_t> localDpuNotifyIds_{};
    uint32_t bufferNum_{0};
    uint32_t connNum_{0};
    // Hccl::BaseMemTransport::Attribution attr_;
    ChannelStatus channelStatus_{ChannelStatus::INIT};
    RdmaStatus rdmaStatus_{RdmaStatus::INIT};
    std::vector<uint32_t> remoteDpuNotifyIds_;
    std::vector<std::unique_ptr<Hccl::RemoteRdmaRmaBuffer>> rmtRmaBuffers_{};
    std::vector<ExchangeRdmaConnDto> rmtConnDtos_;
    std::vector<std::unique_ptr<HcclMem>> remoteMems{};
    std::vector<int> wqeNums_;
    bool fenceFlag_{false};
    std::mutex remoteMemsMutex_; // 远端内存列表互斥锁
    std::unique_ptr<HcclMem[]> remoteMemsPtr_;

    // GetUserRemoteMem 缓存
    bool userRemoteMemCacheValid_{false};
    std::vector<CommMem> userRemoteMems_;
    std::vector<std::string> tagCopies_;
    std::vector<char*> tagPointers_;

    uint64_t maxMsgSize_{0};
    uint32_t lbMax_{0}; // 多QP负载均衡

    std::function<HcclResult(const Hccl::TaskParam&, u64)> dfxCallback_;

    std::mutex cq_mutex;
    std::mutex sendCq_mutex;

    // ========== 混合模式（RoCE Cross-Mode）成员变量 ==========
    // 1. 能力协商结果
    RoCECapability remoteCap_;  // 对端能力
    bool isHybridMode_ = false; // 是否为混合模式

    uint32_t localNotifySize_;
    uint32_t localNotifyAccess_;

    std::array<hccl::MemMsg, static_cast<u32>(hccl::MemType::MEM_TYPE_RESERVED)> localMemMsg_;
    std::array<hccl::MemMsg, static_cast<u32>(hccl::MemType::MEM_TYPE_RESERVED)> remoteMemMsg_;
    uint64_t exchangeDataTotalSize_;
    std::vector<uint8_t> exchangeDataForSend_;
    std::vector<uint8_t> exchangeDataForRecv_;

    uint32_t devicePhyId_{};
};

} // namespace hcomm_experimental

#endif // HCOMM_EXPERIMENTAL_HOST_CPU_ROCE_CHANNEL_H
