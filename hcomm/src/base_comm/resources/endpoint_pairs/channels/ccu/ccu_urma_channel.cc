/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_urma_channel.h"

#include "hcomm_c_adpt.h"

#include "orion_adpt_utils.h"

#include "exception_handler.h"
#include "comm_mems.h"

#include "config_log.h"
#include "config_plf_log_v2.h"

// 暂时引入orion
#include "local_ub_rma_buffer.h"

namespace hcomm {
using Hccl::PLF_CHANNEL;

CcuUrmaChannel::CcuUrmaChannel(const EndpointHandle locEndpointHandle, const HcommChannelDesc& channelDesc)
    : locEndpointHandle_(locEndpointHandle),
      channelDesc_(channelDesc)
{}

CcuUrmaChannel::~CcuUrmaChannel()
{
    // 先释放 transport/connection（CKE/XN、TP、远端 jetty QP unimport），
    // 再向 endpoint 的 CcuChannelCtxPool 归还 channel ctx / jetty ctx / wqeBB。
    // 顺序必须如此：CcuConnection 持有 batch 内 CcuJetty 裸指针。
    impl_.reset();
    if (ccuEndpoint_ != nullptr) {
        Hccl::LinkData linkData = BuildDefaultLinkData();
        HcclResult keyRet
            = EndpointDescPairToLinkData(ccuEndpoint_->GetEndpointDesc(), channelDesc_.remoteEndpoint, linkData);
        if (keyRet != HCCL_SUCCESS) {
            HCCL_ERROR("[CcuUrmaChannel][%s] failed to build link data for release, ret[%d].", __func__, keyRet);
            return;
        }
        auto* pool = ccuEndpoint_->GetCcuChannelCtxPool();
        if (pool != nullptr) {
            HcclResult releaseRet = pool->ReleaseChannel(linkData);
            if (releaseRet != HCCL_SUCCESS) {
                HCCL_WARNING("[CcuUrmaChannel][%s] release channel to pool failed, ret[%d].", __func__, releaseRet);
            }
        }
    }
}

HcclResult BuildBufferInfos(
    HcommMemHandle* memHandles, uint32_t memHandleNum, std::vector<CcuTransport::CclBufferInfo>& bufferInfos)
{
    for (uint32_t i = 0; i < memHandleNum; ++i) {
        auto localRmaBuffer = reinterpret_cast<Hccl::LocalUbRmaBuffer*>(memHandles[i]);
        CHK_PTR_NULL(localRmaBuffer);
        auto buf = localRmaBuffer->GetBuf();
        CHK_PTR_NULL(buf);

        std::array<char, HCCL_RES_TAG_MAX_LEN> memInfo{};
        std::string tag = buf->GetMemInfo();
        if (UNLIKELY(tag.size() >= HCCL_RES_TAG_MAX_LEN)) {
            HCCL_ERROR("[BuildBufferInfos] tagSize exceeds limit[%u]", HCCL_RES_TAG_MAX_LEN);
            return HCCL_E_PARA;
        }
        CHK_SAFETY_FUNC_RET(memcpy_s(memInfo.data(), memInfo.size(), tag.c_str(), tag.size()));
        bufferInfos.emplace_back(
            localRmaBuffer->GetAddr(), static_cast<uint32_t>(localRmaBuffer->GetSize()), localRmaBuffer->GetTokenId(),
            localRmaBuffer->GetTokenValue(), hccl::ConvertHcclToCommMemType(buf->GetMemType()), memInfo);
    }
    return HCCL_SUCCESS;
}

static HcclResult CreateCcuTransport(
    UrmaEndpoint* ccuEndpoint, const Hccl::LinkData& linkData, Hccl::Socket* socket, HcommMemHandle* memHandles,
    uint32_t memHandleNum, uint32_t qos, uint32_t sqSize, std::unique_ptr<CcuTransport>& impl)
{
    HCCL_INFO("[CcuUrmaChannel][%s] begin, sqSize[%u]", __func__, sqSize);
    // 当前ccu channel不支持按需申请cke
    CHK_PTR_NULL(ccuEndpoint);
    CHK_PTR_NULL(socket);
    CHK_PTR_NULL(memHandles);

    auto ret = HcclResult::HCCL_SUCCESS;
    auto* channelCtxPool = ccuEndpoint->GetCcuChannelCtxPool();
    CHK_PTR_NULL(channelCtxPool);
    // 申请ccu channel ctx， jetty ctx，wqebb，可能资源不足，需要回退
    ret = channelCtxPool->PrepareCreate({linkData}, sqSize);
    if (ret == HCCL_E_UNAVAIL) {
        HCCL_WARNING(
            "[CcuUrmaChannel][%s] prepare ccu channel ctx failed, "
            "ccu resources unavailable.",
            __func__);
        return ret;
    }
    CHK_RET(ret);

    CcuChannelCtxPool::CcuChannelCtx channelCtx{};
    CHK_RET(channelCtxPool->GetChannelCtx(linkData, channelCtx));
    const auto& channelInfo = channelCtx.first;
    const auto& ccuJettys = channelCtx.second;

    const auto& locAddr_ = linkData.GetLocalAddr();
    const auto& rmtAddr_ = linkData.GetRemoteAddr();

    CommAddr locAddr{}, rmtAddr{};
    CHK_RET(IpAddressToCommAddr(locAddr_, locAddr));
    CHK_RET(IpAddressToCommAddr(rmtAddr_, rmtAddr));

    CcuTransport::CcuConnectionType type_ = linkData.GetLinkProtocol() == Hccl::LinkProtocol::UB_CTP ?
                                                CcuTransport::CcuConnectionType::UB_CTP :
                                                CcuTransport::CcuConnectionType::UBC_TP;

    CcuTransport::CcuConnectionInfo connectionInfo{type_, locAddr, rmtAddr, channelInfo, ccuJettys, qos};

    std::vector<CcuTransport::CclBufferInfo> bufferInfos{};
    CHK_RET(BuildBufferInfos(memHandles, memHandleNum, bufferInfos));

    // 调用底层的创建函数 (CcuCreateTransport 通常是全局函数或静态函数)
    // 申请 xn cke可能失败，需要回退
    ret = CcuCreateTransport(socket, connectionInfo, bufferInfos, impl);
    if (ret == HCCL_E_UNAVAIL) {
        HCCL_WARNING("[CcuUrmaChannel][%s] failed, ccu resources unavailable.", __func__);
        return ret;
    }
    CHK_RET(ret);

    PLF_CONFIG_INFO(
        PLF_CHANNEL, "[CcuUrmaChannel] transport created, sqSize[%u], linkData[%s].", sqSize,
        linkData.Describe().c_str());
    return HCCL_SUCCESS;
}

static HcclResult CheckEndpointDesc(const EndpointDesc& locDesc, const EndpointDesc& rmtDesc)
{
    if (locDesc.protocol != rmtDesc.protocol) {
        HCCL_ERROR(
            "[CcuUrmaChannel][%s] failed, endpoints protocols are not same, "
            "loc[%d] rmt[%d].",
            __func__, locDesc.protocol, rmtDesc.protocol);
        return HcclResult::HCCL_E_PARA;
    }

    if (locDesc.protocol != COMM_PROTOCOL_UB_CTP && locDesc.protocol != COMM_PROTOCOL_UBC_TP) {
        HCCL_ERROR("[CcuUrmaChannel][%s] failed, protocol[%d] are not supported in ccu.", __func__, locDesc.protocol);
        return HcclResult::HCCL_E_PARA;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::Init()
{
    EXCEPTION_HANDLE_BEGIN
    CHK_PTR_NULL(channelDesc_.socket);
    socket_ = static_cast<Hccl::Socket*>(channelDesc_.socket);
    // 当前socket在外部统一触发connect，建议之后改为异步建链流程内触发

    CHK_PTR_NULL(locEndpointHandle_);
    void* endpoint{nullptr};
    CHK_RET(static_cast<HcclResult>(HcommEndpointGet(locEndpointHandle_, &endpoint)));
    ccuEndpoint_ = dynamic_cast<UrmaEndpoint*>(static_cast<Endpoint*>(endpoint));
    CHK_PTR_NULL(ccuEndpoint_);
    const auto& locEndpointDesc = ccuEndpoint_->GetEndpointDesc();

    CHK_RET(CheckEndpointDesc(locEndpointDesc, channelDesc_.remoteEndpoint));

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(locEndpointDesc, channelDesc_.remoteEndpoint, linkData));

    if (channelDesc_.memHandleNum == 0) {
        HCCL_ERROR("[CcuUrmaChannel][%s] failed, unsupported memHandleNum[%u].", __func__, channelDesc_.memHandleNum);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    CHK_PTR_NULL(channelDesc_.memHandles);

    memHandles_.assign(channelDesc_.memHandles, channelDesc_.memHandles + channelDesc_.memHandleNum);

    // 当前建链不支持资源扩容，CCU资源默认固定为8
    HCCL_WARNING("[CcuUrmaChannel][%s] now only support notify num is 8.", __func__);
    HCCL_WARNING("[CcuUrmaChannel][%s] now only support to exchange hccl buffer.", __func__);

    channelStatus_ = ChannelStatus::INIT;
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

ChannelStatus CcuUrmaChannel::TryPrepareAndConstruct()
{
    Hccl::LinkData linkData = BuildDefaultLinkData();
    HcclResult keyRet
        = EndpointDescPairToLinkData(ccuEndpoint_->GetEndpointDesc(), channelDesc_.remoteEndpoint, linkData);
    if (keyRet != HCCL_SUCCESS) {
        HCCL_ERROR("[CcuUrmaChannel][%s] failed to build link data, ret[%d].", __func__, keyRet);
        return ChannelStatus::FAILED;
    }

    HcclResult ret = CreateCcuTransport(
        ccuEndpoint_, linkData, socket_, memHandles_.data(), static_cast<uint32_t>(memHandles_.size()),
        channelDesc_.qos, channelDesc_.ubAttr.sqDepth, impl_);
    if (ret == HCCL_SUCCESS) {
        impl_->SetLocResStatus(CcuTransport::CcuResStatus::RES_OK);
        return ChannelStatus::INIT;
    }
    // 失败(资源不足或硬失败): 构造 msg-only transport 通知对端快速失败, 避免对端 120s 超时
    CcuTransport::CcuResStatus failStatus
        = (ret == HCCL_E_UNAVAIL) ? CcuTransport::CcuResStatus::RES_UNAVAIL : CcuTransport::CcuResStatus::RES_FAILED;
    HCCL_RUN_WARNING(
        "[CcuUrmaChannel][%s] CreateCcuTransport failed[%d], construct msg-only transport.", __func__, ret);
    HcclResult transRet = CcuTransport::ConstructMsgOnlyTransport(socket_, impl_, failStatus);
    if (transRet != HCCL_SUCCESS) {
        HCCL_ERROR("[CcuUrmaChannel][%s] ConstructMsgOnlyTransport failed[%d].", __func__, transRet);
        return ChannelStatus::FAILED;
    }
    return ChannelStatus::INIT;
}

// 状态机终态 -> 通道状态映射（含 LOC/RMT 资源不足优先级）
static ChannelStatus MapTransStatusToChannelStatus(CcuTransport::TransStatus status, const CcuTransport& impl)
{
    switch (status) {
        case CcuTransport::TransStatus::READY:
            return ChannelStatus::READY;
        case CcuTransport::TransStatus::SOCKET_TIMEOUT:
            HCCL_ERROR("[CcuUrmaChannel][%s] error status[%s].", __func__, status.Describe().c_str());
            return ChannelStatus::SOCKET_TIMEOUT;
        case CcuTransport::TransStatus::CONNECT_FAILED:
            // 资源不足终态映射(LOC 优先): 本端不足报 LOC, 否则若对端不足报 RMT, 否则普通 FAILED
            if (impl.IsLocResUnavailable()) {
                return ChannelStatus::RES_LOC_UNAVAIL;
            }
            if (impl.IsRmtResUnavailable()) {
                return ChannelStatus::RES_RMT_UNAVAIL;
            }
            HCCL_ERROR("[CcuUrmaChannel][%s] error status[%s].", __func__, status.Describe().c_str());
            return ChannelStatus::FAILED;
        default:
            return ChannelStatus::INIT;
    }
}

ChannelStatus CcuUrmaChannel::GetStatus()
{
    std::lock_guard<std::mutex> lock(statusMtx_);

    if (ccuEndpoint_ == nullptr || socket_ == nullptr || memHandles_.empty()) {
        HCCL_ERROR(
            "[CcuUrmaChannel][%s] endpoint[%p], socket[%p], memHandleNum[%zu], Init() may not have succeeded.",
            __func__, ccuEndpoint_, socket_, memHandles_.size());
        channelStatus_ = ChannelStatus::FAILED;
        return channelStatus_;
    }
    // INIT 态内: locResStatus_==RES_UNKNOWN(或 impl_ 为空) 则构造 transport + 申请资源(仅一次)
    if (channelStatus_ == ChannelStatus::INIT) {
        if (!impl_ || impl_->GetLocResStatus() == CcuTransport::CcuResStatus::RES_UNKNOWN) {
            channelStatus_ = TryPrepareAndConstruct();
            return channelStatus_;
        }
    }

    if (!impl_) {
        HCCL_ERROR("[CcuUrmaChannel][%s] failed, impl is nullptr.", __func__);
        return ChannelStatus::FAILED;
    }

    CcuTransport::TransStatus status = impl_->GetStatus();
    ChannelStatus out = MapTransStatusToChannelStatus(status, *impl_);
    channelStatus_ = out;

    if (isFirstPrintChannelInfo_ && out == ChannelStatus::READY) {
        std::string channelInfo = "create channel info:channel handle[";
        channelInfo.append(std::to_string(reinterpret_cast<uint64_t>(this)));
        channelInfo.append("] ");
        HcclResult ret = impl_->Describe(channelInfo);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[CcuUrmaChannel][%s] Describe channel info failed, ret=%d", __func__, ret);
            out = ChannelStatus::FAILED;
            channelStatus_ = out;
        } else {
            channelInfo.append(" TA[RM]"); // 目前TA只支持RM
            HCCL_CONFIG_DEBUG(hccl::HCCL_RES, "%s", channelInfo.c_str());
        }
        isFirstPrintChannelInfo_ = false;
    }
    return channelStatus_; // todo: AICPU 重新定义基类的状态后，需要修改为CONNECTING
}

uint32_t CcuUrmaChannel::GetDieId() const
{
    if (!impl_) {
        return UINT32_MAX;
    }

    return impl_->GetDieId();
}

uint32_t CcuUrmaChannel::GetChannelId() const
{
    if (!impl_) {
        return UINT32_MAX;
    }
    return impl_->GetChannelId();
}

HcclResult CcuUrmaChannel::GetRmtSignalAddrByIndex(uint32_t index, uint64_t& rmtCkeAddr) const
{
    CHK_PTR_NULL(impl_);
    CHK_RET(impl_->GetRmtSignalAddrByIndex(index, rmtCkeAddr));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::GetRmtVarAddrByIndex(uint32_t index, uint64_t& rmtXnAddr) const
{
    CHK_PTR_NULL(impl_);
    CHK_RET(impl_->GetRmtVarAddrByIndex(index, rmtXnAddr));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::GetRmtCcuBufferTokenInfo(uint32_t& rmtTokenId, uint32_t& rmtTokenValue) const
{
    CHK_PTR_NULL(impl_);
    CHK_RET(impl_->GetRmtCcuBufferTokenInfo(rmtTokenId, rmtTokenValue));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::GetLocCkeByIndex(const uint32_t index, uint32_t& locCkeId) const
{
    CHK_PTR_NULL(impl_);
    CHK_RET(impl_->GetLocCkeByIndex(index, locCkeId));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::GetLocXnByIndex(const uint32_t index, uint32_t& locXnId) const
{
    CHK_PTR_NULL(impl_);
    CHK_RET(impl_->GetLocXnByIndex(index, locXnId));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::GetRmtCkeByIndex(const uint32_t index, uint32_t& rmtCkeId) const
{
    CHK_PTR_NULL(impl_);
    CHK_RET(impl_->GetRmtCkeByIndex(index, rmtCkeId));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::GetRmtXnByIndex(const uint32_t index, uint32_t& rmtXnId) const
{
    CHK_PTR_NULL(impl_);
    CHK_RET(impl_->GetRmtXnByIndex(index, rmtXnId));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::GetRmtBuffer(uint64_t& addr, uint32_t& size, uint32_t& tokenId, uint32_t& tokenValue) const
{
    CHK_PTR_NULL(impl_);
    CcuTransport::CclBufferInfo bufInfo{};
    constexpr uint32_t bufNum = 0; // 当前不支持
    CHK_RET(impl_->GetRmtBuffer(bufInfo, bufNum));

    addr = bufInfo.addr;
    size = bufInfo.size;
    tokenId = bufInfo.tokenId;
    tokenValue = bufInfo.tokenValue;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::GetNotifyNum(uint32_t* notifyNum) const
{
    CHK_PTR_NULL(impl_);
    CHK_RET(impl_->GetCkeNum(*notifyNum));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    CHK_PTR_NULL(impl_);
    return impl_->GetRemoteMems(memNum, remoteMem, memInfos);
}

HcclResult CcuUrmaChannel::Clean()
{
    CHK_PTR_NULL(impl_);
    PLF_CONFIG_INFO(PLF_CHANNEL, "[CcuUrmaChannel] clean channel resource.");
    impl_->Clean();
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::Resume() { return HCCL_SUCCESS; }

HcclResult CcuUrmaChannel::UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum)
{
    CHK_PTR_NULL(impl_);
    std::vector<CcuTransport::CclBufferInfo> bufferVecTemp{};
    CHK_RET(BuildBufferInfos(memHandles, memHandleNum, bufferVecTemp));
    return impl_->UpdateMemInfo(bufferVecTemp);
}
HcclResult CcuUrmaChannel::CcuGetRmtMemToken(uint64_t srcVa, uint64_t& tokenInfo)
{
    return impl_->CcuGetRmtMemToken(srcVa, tokenInfo);
}

HcclResult CcuUrmaChannel::NotifyRecord([[maybe_unused]] const uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[CcuUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
CcuUrmaChannel::NotifyWait([[maybe_unused]] const uint32_t localNotifyIdx, [[maybe_unused]] const uint32_t timeout)
{
    HCCL_INFO("[CcuUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CcuUrmaChannel::WriteWithNotify(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] const uint64_t len,
    [[maybe_unused]] uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[CcuUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
CcuUrmaChannel::Write([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len)
{
    HCCL_INFO("[CcuUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
CcuUrmaChannel::Read([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len)
{
    HCCL_INFO("[CcuUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CcuUrmaChannel::ChannelFence()
{
    HCCL_INFO("[CcuUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

} // namespace hcomm
