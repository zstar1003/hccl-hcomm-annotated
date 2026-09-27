/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TS_UBOE_UB_RTP_CHANNEL_HELPER_H
#define AICPU_TS_UBOE_UB_RTP_CHANNEL_HELPER_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "channel.h"
#include "aicpu_ts_channel_helper.h"

// Orion
#include "../../../../../../src/legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "rma_connection.h"
#include "ub_mem_transport.h"
#include "dev_ub_connection.h"
#include "ub_local_notify.h"
#include "hcomm_adapter_hccp.h"

namespace hcomm {

constexpr u32 FINISH_MSG_SIZE = 128;
constexpr char_t FINISH_MSG[FINISH_MSG_SIZE] = "Uboe Comm Pipe ready!";

class AicpuTsUboeUbRtpChannelHelper : public Channel {
public:
    AicpuTsUboeUbRtpChannelHelper(EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc);
    ~AicpuTsUboeUbRtpChannelHelper() override;

    HcclResult GetNotifyNum(uint32_t* notifyNum) const override;
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override;

    HcclResult H2DResPack(std::vector<char>& buffer);
    const HcommChannelDesc& GetChannelDesc() const override { return channelDesc_; }
    HcclResult Clean() override;
    HcclResult Resume() override;

    // 数据面接口
    HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override;
    HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override;
    HcclResult WriteWithNotify(void* dst, const void* src, const uint64_t len, uint32_t remoteNotifyIdx) override;
    HcclResult Write(void* dst, const void* src, uint64_t len) override;
    HcclResult Read(void* dst, const void* src, uint64_t len) override;
    HcclResult ChannelFence() override;

    AicpuTsChannelHelper* GetAicpuTsHelper() override { return &aicpuTsHelper_; }

protected:
    static constexpr u64 NORMAL_NOTIFY_VAL = 1;

    MAKE_ENUM(UboeRmtBufType, NOTIFY, BUFFER)
    using RemoteBufferVec = std::vector<std::unique_ptr<Hccl::RemoteUbRmaBuffer>>;
    using LocalBufferVec = std::vector<Hccl::LocalUbRmaBuffer*>;

    virtual HcclResult BuildConnection() = 0;
    virtual void SendFinish() = 0;
    virtual void RecvFinish() = 0;

    HcclResult ParseInputParam();
    HcclResult BuildNotify();
    HcclResult BuildDrainResource();
    HcclResult BuildSocket();
    void BuildConn();

    bool IsSocketReady();
    bool IsResReady();
    bool IsConnsReady();
    bool RecvDataProcess();
    void SendDataSize();
    void RecvDataSize();
    void SendExchangeData();
    void RecvExchangeData();

    void NotifyVecPack(Hccl::BinaryStream& binaryStream);
    void BufferVecPack(Hccl::BinaryStream& binaryStream, std::vector<Hccl::LocalRmaBuffer*>& bufferVec) const;
    void DrainBufferPack(Hccl::BinaryStream& binaryStream);
    void ConnVecPack(Hccl::BinaryStream& binaryStream);
    void RmtBufferVecUnpackProc(
        u32 locNum, Hccl::BinaryStream& binaryStream, RemoteBufferVec& bufferVec, UboeRmtBufType type);
    void RmtDrainBufferUnpackProc(Hccl::BinaryStream& binaryStream);
    bool ConnVecUnpackProc(Hccl::BinaryStream& binaryStream);

    std::vector<char> GetUniqueIdV2();
    std::vector<char> GetNotifyUniqueIds();
    std::vector<char> GetRmtBufferUniqueIds(RemoteBufferVec& bufferVec, UboeRmtBufType type) const;
    std::vector<char> GetLocBufferUniqueIds(LocalBufferVec& bufferVec, UboeRmtBufType type) const;
    std::vector<char> GetSingleRmtBufferUniqueId(u64 addr, u64 size, u32 tokenId, u32 tokenValue, u32 notifyId) const;
    std::vector<char> GetSingleLocBufferUniqueId(u64 addr, u64 size, u32 tokenId, u32 tokenValue) const;
    std::vector<char> GetDrainUniqueIds() const;
    std::vector<char> GetConnUniqueIds();

    // --------------------- 入参 ---------------------
    EndpointHandle endpointHandle_;
    HcommChannelDesc channelDesc_;

    // --------------------- 转换参数 ---------------------
    EndpointDesc localEp_{};
    EndpointDesc remoteEp_{};
    uint32_t notifyNum_{0};

    // --------------------- 具体成员 ---------------------
    Hccl::Socket* socket_{nullptr};
    RdmaHandle rdmaHandle_{nullptr};
    std::unique_ptr<Hccl::UbMemTransport> memTransport_{nullptr};
    Hccl::BaseMemTransport::CommonLocRes commonRes_{};
    std::vector<std::unique_ptr<Hccl::DevUbConnection>> connections_{};
    std::vector<std::unique_ptr<Hccl::UbLocalNotify>> localNotifies_{};
    std::unique_ptr<Hccl::Socket> serverSocket_;

    std::unique_ptr<Hccl::UbLocalNotify> drainNotify_;        // 本端drain notify
    std::unique_ptr<Hccl::LocalUbRmaBuffer> drainBuffer_;     // 本端常量1 buffer
    std::unique_ptr<Hccl::RemoteUbRmaBuffer> rmtDrainBuffer_; // 对端常量1 buffer

    ChannelStatus channelStatus{ChannelStatus::INIT};

    u32 type_{1}; // 默认为UB
    u32 bufferNum_{0};
    u32 connNum_{0};
    u32 recvDataSize_{0};

    RemoteBufferVec rmtNotifyVec_;
    RemoteBufferVec rmtBufferVec_;
    LocalBufferVec locBufferVec_;
    std::vector<char> recvData_{};
    std::vector<char> recvFinishMsg_{};
    std::vector<char> sendData_{};
    std::vector<char> sendFinishMsg_{};
    bool isRecvFirst_{false};

    Hccl::IpAddress locAddr_;
    Hccl::IpAddress rmtAddr_;

    std::mutex remoteMemsMutex_;
    bool cacheValid_ = false;
    std::vector<CommMem> remoteUserMems_;
    std::vector<std::string> memInfoCopies_;
    std::vector<char*> memInfoPointers_;
    const Hccl::SocketConfig* socketConfig_{nullptr};
    DevBaseAttr devBaseAttr_{};
    uint32_t devicePhyId_{};

private:
    AicpuTsChannelHelper aicpuTsHelper_;
};

} // namespace hcomm

#endif // AICPU_TS_UBOE_UB_RTP_CHANNEL_HELPER_H
