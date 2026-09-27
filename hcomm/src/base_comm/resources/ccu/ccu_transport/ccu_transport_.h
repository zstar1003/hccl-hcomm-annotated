/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_TRANSPORT_H
#define HCOMM_CCU_TRANSPORT_H

#include <memory>
#include <vector>
#include <shared_mutex>

#include "../../../../../legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "env_config.h"
#include "op_mode.h"
#include "binary_stream.h"
#include "ccu_conn.h"
#include "remote_rma_buffer.h"

namespace hcomm {

class CcuTransport {
public:
    // 缩减channel预留的cke、xn量，避免与ccu instance需求量冲突
    static constexpr uint32_t INIT_CKE_NUM = 4;
    static constexpr uint32_t INIT_XN_NUM = 4;
    MAKE_ENUM(
        TransStatus, INIT, SEND_DATA_SIZE, RECV_DATA_SIZE, SEND_ALL_INFO, RECV_ALL_INFO, SEND_TRANS_RES, RECV_TRANS_RES,
        SEND_FIN, RECV_FIN, RECVING_FIN, RECVING_TRANS_RES, READY, CONNECT_FAILED, SOCKET_TIMEOUT)

    enum class CcuResStatus : uint8_t {
        RES_UNKNOWN = 0, // 待确认（初始值）
        RES_OK = 1,      // 资源充足
        RES_UNAVAIL = 2, // 资源不足
        RES_FAILED = 3,  // 资源创建硬失败
    };

    struct CclBufferInfo {
        uint64_t addr{0};
        uint32_t size{0};
        uint32_t tokenId{0};
        uint32_t tokenValue{0};
        CommMemType type{COMM_MEM_TYPE_INVALID};
        std::array<char, HCCL_RES_TAG_MAX_LEN> memInfo{};

        explicit CclBufferInfo() = default;
        CclBufferInfo(const uint64_t addr, const uint32_t size, const uint32_t tokenId, const uint32_t tokenValue)
            : addr(addr),
              size(size),
              tokenId(tokenId),
              tokenValue(tokenValue)
        {}

        CclBufferInfo(
            const uint64_t addr, const uint32_t size, const uint32_t tokenId, const uint32_t tokenValue,
            const CommMemType type, const std::array<char, HCCL_RES_TAG_MAX_LEN>& memInfo)
            : addr(addr),
              size(size),
              tokenId(tokenId),
              tokenValue(tokenValue),
              type(type),
              memInfo(memInfo)
        {}

        void Pack(Hccl::BinaryStream& binaryStream) const
        {
            binaryStream << addr << size << tokenId << tokenValue << type;
            // 逐个字节传输
            for (uint32_t i = 0; i < HCCL_RES_TAG_MAX_LEN; ++i) {
                binaryStream << static_cast<u8>(memInfo[i]);
            }
            HCCL_INFO("Pack Ccl Buffer Info: addr[%llu] size[%u] memInfo[%s]", addr, size, memInfo.data());
        }

        void Unpack(Hccl::BinaryStream& binaryStream)
        {
            binaryStream >> addr >> size >> tokenId >> tokenValue >> type;
            for (uint32_t i = 0; i < HCCL_RES_TAG_MAX_LEN; ++i) {
                u8 byte;
                binaryStream >> byte;
                memInfo[i] = static_cast<char>(byte);
            }
            HCCL_INFO("Unpack Ccl Buffer Info: addr[%llu] size[%u] memInfo[%s]", addr, size, memInfo.data());
        }
    };

    MAKE_ENUM(CcuConnectionType, UBC_TP, UB_CTP, UBC_CTP = UB_CTP);
    struct CcuConnectionInfo {
        CcuConnectionType type{CcuConnectionType::UBC_TP};
        CommAddr locAddr{};
        CommAddr rmtAddr{};
        CcuChannelInfo channelInfo{};
        std::vector<CcuJetty*> ccuJettys{};
        uint32_t qos{EnvConfig::UB_QOS_DEFAULT};

        explicit CcuConnectionInfo() = default;
        CcuConnectionInfo(
            const CcuConnectionType type, const CommAddr& locAddr, const CommAddr& rmtAddr,
            const CcuChannelInfo& channelInfo, const std::vector<CcuJetty*>& ccuJettys,
            uint32_t qos = EnvConfig::UB_QOS_DEFAULT)
            : type(type),
              locAddr(locAddr),
              rmtAddr(rmtAddr),
              channelInfo(channelInfo),
              ccuJettys(ccuJettys),
              qos(qos)
        {}
    };

    CcuTransport(Hccl::Socket* socket, std::unique_ptr<CcuConnection>&& connection, const CclBufferInfo& locCclBufInfo);
    CcuTransport(
        Hccl::Socket* socket, std::unique_ptr<CcuConnection>&& connection,
        const std::vector<CclBufferInfo>& bufferInfos);
    CcuTransport(const CcuTransport& that) = delete;
    CcuTransport& operator=(const CcuTransport& other) = delete;
    ~CcuTransport();
    HcclResult Init();
    TransStatus GetStatus();
    void Clean();
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos);
    HcclResult CheckSocketStatus();
    HcclResult UpdateMemInfo(std::vector<CcuTransport::CclBufferInfo>& bufferVecTemp);
    HcclResult CcuGetRmtMemToken(uint64_t srcVa, uint64_t& tokenInfo);

    CcuResStatus GetLocResStatus() const { return locResStatus_; }
    void SetLocResStatus(CcuResStatus status) { locResStatus_ = status; }
    bool IsLocResUnavailable() const { return locResStatus_ == CcuResStatus::RES_UNAVAIL; }
    bool IsRmtResUnavailable() const { return rmtResStatus_ == CcuResStatus::RES_UNAVAIL; }
    bool IsBothEndResSufficient() const
    {
        return locResStatus_ == CcuResStatus::RES_OK && rmtResStatus_ == CcuResStatus::RES_OK;
    }
    static HcclResult
    ConstructMsgOnlyTransport(Hccl::Socket* socket, std::unique_ptr<CcuTransport>& impl, CcuResStatus status);

    // 下面接口为平台层接口，不能在框架层使用
    uint32_t GetDieId() const;
    uint32_t GetChannelId() const;
    HcclResult GetLocCkeByIndex(const uint32_t index, uint32_t& locCkeId) const;
    HcclResult GetLocXnByIndex(const uint32_t index, uint32_t& locXnId) const;
    HcclResult GetRmtCkeByIndex(const uint32_t index, uint32_t& rmtCkeId) const;
    HcclResult GetRmtXnByIndex(const uint32_t index, uint32_t& rmtXnId) const;
    HcclResult GetLocBuffer(CclBufferInfo& bufferInfo, const uint32_t& bufNum) const;
    HcclResult GetRmtBuffer(CclBufferInfo& bufferInfo, const uint32_t& bufNum) const;
    HcclResult GetCkeNum(uint32_t& ckeNum) const;
    HcclResult GetRmtVarAddrByIndex(uint32_t index, uint64_t& rmtXnAddr) const;
    HcclResult GetRmtSignalAddrByIndex(uint32_t index, uint64_t& rmtCkeAddr) const;
    HcclResult GetRmtCcuBufferTokenInfo(uint32_t& rmtTokenId, uint32_t& rmtTokenValue) const;
    std::string Describe() const;
    HcclResult Describe(std::string& dfxMsg);

public:
    struct Attribution {
        Hccl::OpMode opMode{Hccl::OpMode::OPBASE};
        u32 devicePhyId{0};
        std::vector<char> handshakeMsg{};
        std::string Describe() const
        {
            return Hccl::StringFormat(
                "CcuTransportAttribution[opMode=%s, devicePhyId=%u, handshakeMsg=%s]", opMode.Describe().c_str(),
                devicePhyId, Hccl::Bytes2hex(handshakeMsg.data(), handshakeMsg.size()).c_str());
        }
    };

    std::vector<char>& GetRmtHandshakeMsg() // 返回握手消息
    {
        return rmtHandshakeMsg_;
    }

    std::vector<char>& GetLocalHandshakeMsg() // 返回握手消息
    {
        return attr_.handshakeMsg;
    }

    void SetHandshakeMsg(const std::vector<char>& handshakeMsg) { attr_.handshakeMsg = handshakeMsg; }

private:
    HcclResult StatusMachine();
    HcclResult AppendCkes(uint32_t ckesNum);
    HcclResult AppendXns(uint32_t xnsNum);
    HcclResult SendFinish();
    HcclResult RecvFinish();
    HcclResult CheckFinish();
    HcclResult RecvDataProcess();
    HcclResult RecvTransInfoProcess();
    HcclResult ReleaseTransRes();
    HcclResult SendConnAndTransInfo();
    HcclResult RecvConnAndTransInfo();
    HcclResult SendDataSize();
    HcclResult RecvDataSize();
    HcclResult SendTransInfo();
    HcclResult RecvTransInfo();
    HcclResult HandshakeMsgPack(Hccl::BinaryStream& binaryStream);
    HcclResult ConnInfoPack(Hccl::BinaryStream& binaryStream) const;
    HcclResult TransResPack(Hccl::BinaryStream& binaryStream);
    HcclResult BufferInfoPack(Hccl::BinaryStream& binaryStream, std::vector<CclBufferInfo>& bufferVec) const;
    HcclResult HandshakeMsgUnpack(Hccl::BinaryStream& binaryStream);
    HcclResult ConnInfoUnpackProc(Hccl::BinaryStream& binaryStream) const;
    HcclResult TransResUnpackProc(Hccl::BinaryStream& binaryStream);
    HcclResult BufferInfoUnpack(Hccl::BinaryStream& binaryStream);
    HcclResult GetRmtVarAddrByXnId(const uint32_t rmtXnId, uint64_t& rmtXnAddr) const;

    HcclResult ReturnErrorStatus(const std::string& funcName);

private:
    // 保存transport中需要使用的cke，xn等ccu资源
    struct TransRes {
        std::vector<uint32_t> ckes{};
        std::vector<uint32_t> xns{};
    };

    uint32_t dieId_{0};
    int32_t userDevId_{0};
    Attribution attr_{};
    std::vector<char> rmtHandshakeMsg_{0}; // 远端握手消息
    Hccl::Socket* socket_{nullptr};
    std::unique_ptr<CcuConnection> ccuConnection_;
    TransRes locRes_{};
    TransRes rmtRes_{};
    TransStatus transStatus_{TransStatus::INVALID};
    CcuResStatus locResStatus_{CcuResStatus::RES_UNKNOWN};
    CcuResStatus rmtResStatus_{CcuResStatus::RES_UNKNOWN};
    std::vector<std::vector<ResInfo>> ckesRes_{};
    std::vector<std::vector<ResInfo>> xnsRes_{};
    std::vector<CclBufferInfo> locBufferInfos_{};
    CclBufferInfo rmtHcclBufferInfo_{};
    std::vector<std::unique_ptr<Hccl::RemoteUbRmaBuffer>> rmtBufferVec_{};
    uint32_t exchangeDataSize_{0};
    std::vector<char> recvData_{};
    std::vector<char> recvTrans_{};
    std::vector<char> sendData_{};
    std::vector<char> sendTrans_{};
    std::vector<char> recvFinishMsg_{};
    std::vector<char> sendFinishMsg_{};
    bool cacheValid_ = false;                // GetUserRemoteMem 的缓存标识
    std::mutex remoteMemsMutex_;             // 远端内存列表互斥锁
    std::vector<CommMem> remoteUserMems_;    // 内存基本信息缓存
    std::vector<std::string> memInfoCopies_; // 储存 Tag 字符串副本
    std::vector<char*> memInfoPointers_;     // Tag 缓存
};

HcclResult BuildCcuConnection(
    const CcuTransport::CcuConnectionInfo& ccuConnectionInfo, std::unique_ptr<CcuConnection>& ccuConnection);

HcclResult CcuCreateTransport(
    Hccl::Socket* socket, const CcuTransport::CcuConnectionInfo& ccuConnectionInfo,
    const CcuTransport::CclBufferInfo& cclBufferInfo, std::unique_ptr<CcuTransport>& ccuTransport);

HcclResult CcuCreateTransport(
    Hccl::Socket* socket, const CcuTransport::CcuConnectionInfo& ccuConnectionInfo,
    const std::vector<CcuTransport::CclBufferInfo>& bufferInfos, std::unique_ptr<CcuTransport>& ccuTransport);
} // namespace hcomm
#endif // HCOMM_CCU_TRANSPORT_H
