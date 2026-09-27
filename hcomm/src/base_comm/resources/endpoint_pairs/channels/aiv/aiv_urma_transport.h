/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_URMA_TRANSPORT_H
#define AIV_URMA_TRANSPORT_H
#include "base_mem_transport.h"
#include "task_param.h"
#include "orion_adapter_hccp.h"
#include "dev_ub_connection.h"
#include "hcomm/hcomm_res_entity_defs.h"
#include "mem_device_pub.h"
namespace Hccl {

class AivUrmaTransport {
public:
    AivUrmaTransport(
        BaseMemTransport::CommonLocRes& commonLocRes, BaseMemTransport::Attribution& attr, const LinkData& linkData,
        const Socket& socket, RdmaHandle rdmaHandle);

    TransportStatus GetStatus();
    std::string Describe() const;
    void GetEntityCountsForLayout(uint32_t& bufNum, uint32_t& connNum) const;
    void PrepareHostChannelEntity(ChannelEntity* channelEntitiesHost);
    void SetQueueIndexDeviceMem(void* sqPiMem, void* sqCiMem, void* cqPiMem, void* cqCiMem, size_t memSize);
    void GetHostChannelEntity(ChannelEntity* channelEntitiesHost);
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos);
    HcclResult CheckSocketStatus(std::string socketOperator) const;
    HcclResult UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum);

private:
    MAKE_ENUM(UrmaStatus, INIT, SOCKET_OK, SEND_DATA, RECV_DATA, SEND_FIN, RECV_FIN, PROCESS_DATA, CONN_OK)
    UrmaStatus urmaStatus_{UrmaStatus::INIT};

    // 获取ChannelEntity
    void GetSqContext();
    void GetCqContext();
    void GetProtectionInfo();
    void ResolveLocalBufferTokens();
    void EnsureQueueIndexDeviceMem();

    struct BufferTokenInfo {
        uint32_t tokenId{0};
        uint32_t tokenValue{0};
        bool useProcessToken{false};
    };

    // 建链过程
    using RemoteBufferVec = std::vector<std::unique_ptr<RemoteUbRmaBuffer>>;
    using LocalBufferVec = std::vector<std::unique_ptr<LocalUbRmaBuffer>>;

    void SendExchangeData();
    void RecvExchangeData();
    void SendFinish();
    void RecvFinish();
    void RmtBufferVecUnpackProc(uint32_t locNum, BinaryStream& binaryStream, RemoteBufferVec& bufferVec);
    bool ConnVecUnpackProc(BinaryStream& binaryStream);
    bool IsResReady();
    bool IsConnsReady();
    bool RecvDataProcess();
    std::string GetLinkDescInfo();
    void CheckLocBuffer(BaseMemTransport::CommonLocRes& res);
    void CheckLocConn(BaseMemTransport::CommonLocRes& res);
    void CheckCommonLocRes(BaseMemTransport::CommonLocRes& res);
    void HandshakeMsgPack(BinaryStream& binaryStream);
    void BufferVecPack(BinaryStream& binaryStream);
    void ConnVecPack(BinaryStream& binaryStream);
    void HandshakeMsgUnpack(BinaryStream& binaryStream);
    bool IsSocketReady();
    bool PrepareGetStatus();
    void ProcessUrmaStatus();

    BaseMemTransport::CommonLocRes commonLocRes_{};
    BaseMemTransport::Attribution attr_{};
    LinkData linkData_;
    Socket* socket_{nullptr};
    TransportType transportType_{TransportType::UB};
    RdmaHandle rdmaHandle_{nullptr};

    vector<char> sendData_{};
    vector<char> recvData_{};
    size_t exchangeDataSize_{0};
    vector<char> sendFinishMsg_{};
    vector<char> recvFinishMsg_{};
    TransportStatus transportStatus_{TransportStatus::INIT};

    RemoteBufferVec rmtBufferVec_{};  // 远端 buffer
    std::mutex remoteMemsMutex_;      // 远端内存列表互斥锁
    LocalBufferVec localBuffers_{};   // 本地 buffer
    vector<char> rmtHandshakeMsg_{0}; // 远端握手消息
    AcceleratorState rmtOpAcceState_{AcceleratorState::AIV};

    std::vector<RegedBufferEntity> localBufferInfo_{};
    std::vector<RegedBufferEntity> remoteBufferInfo_{};
    std::vector<BufferTokenInfo> localBufferTokens_{};
    bool localBufferTokensResolved_{false};
    std::vector<SqContext> sqContextVec_{};
    std::vector<CqContext> cqContextVec_{};
    hccl::DeviceMem sqPiMem_{};
    hccl::DeviceMem sqCiMem_{};
    hccl::DeviceMem cqPiMem_{};
    hccl::DeviceMem cqCiMem_{};
    uint32_t connNum_{0};
    bool cacheValid_{false};                   // 缓存是否有效
    std::vector<CommMem> remoteUserMems_{};    // 内存基本信息缓存
    std::vector<std::string> memInfoCopies_{}; // 储存 Tag 字符串副本
    std::vector<char*> memInfoPointers_{};     // Tag 缓存
};
} // namespace Hccl
#endif
