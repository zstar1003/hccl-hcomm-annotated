/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_urma_channel.h"
#include "endpoint.h"
#include "../sockets/socket_mgr.h"
#include "comm_queue_context/jetty_context.h"
#include "orion_adpt_utils.h"
#include "acl_device_slab_guard.h"
#include "shared_jetty_channel_helper.h"

#include "hcomm_c_adpt.h"
#include "hcomm_res_mgr.h"
#include "config_plf_log_v2.h"

// Orion
#include "topo_common_types.h"
#include "virtual_topo.h"
#include "makebufs_helper.h"
#include "orion_adapter_hccp.h"
#include "adapter_rts.h"
#include "acl/acl_rt.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace hcomm {
using Hccl::PLF_CHANNEL;
constexpr uint16_t DEFAULT_LISTENING_PORT = 60001;

namespace {
    constexpr size_t AIV_URMA_ENTITY_ALIGN_SIZE = 64;
    constexpr size_t QUEUE_INDEX_MEM_UNIT_SIZE = sizeof(void*);

    struct DeviceEntitySection {
        size_t offset{0};
        size_t size{0};
    };

    struct DeviceChannelEntityLayout {
        DeviceEntitySection entitySection{0, sizeof(ChannelEntity)};
        DeviceEntitySection localNotifySection;
        DeviceEntitySection remoteNotifySection;
        DeviceEntitySection localBufferSection;
        DeviceEntitySection remoteBufferSection;
        DeviceEntitySection sqContextSection;
        DeviceEntitySection cqContextSection;
        DeviceEntitySection sqPiSection;
        DeviceEntitySection sqCiSection;
        DeviceEntitySection cqPiSection;
        DeviceEntitySection cqCiSection;
        size_t slabSize{0};
    };

    HcclResult SecureMemset(void* dest, size_t destMax, int value, size_t count, const char* fieldName)
    {
        if (dest == nullptr) {
            HCCL_ERROR("[SecureMemset] dest is nullptr, field[%s]", fieldName);
            return HCCL_E_PTR;
        }
        if (count > destMax) {
            HCCL_ERROR("[SecureMemset] invalid size, field[%s], count[%zu], destMax[%zu]", fieldName, count, destMax);
            return HCCL_E_PARA;
        }

        errno_t ret = memset_s(dest, destMax, value, count);
        if (ret != EOK) {
            HCCL_ERROR(
                "[SecureMemset] memset_s failed, field[%s], ret[%d], count[%zu], destMax[%zu]", fieldName, ret, count,
                destMax);
            return HCCL_E_MEMORY;
        }
        return HCCL_SUCCESS;
    }

    HcclResult GetAllMemHandles(EndpointHandle endpointHandle, void** memHandles, uint32_t* memHandleNum)
    {
        return static_cast<HcclResult>(HcommMemGetAllMemHandles(endpointHandle, memHandles, memHandleNum));
    }

    size_t AlignUp(size_t value, size_t alignment) { return (value + alignment - 1) / alignment * alignment; }

    HcclResult AddDeviceEntitySection(
        size_t elemSize, uint32_t elemNum, size_t& offset, DeviceEntitySection& section, const char* sectionName)
    {
        section.offset = AlignUp(offset, AIV_URMA_ENTITY_ALIGN_SIZE);
        if (elemNum == 0) {
            section.size = 0;
            offset = section.offset;
            return HCCL_SUCCESS;
        }
        CHK_PRT_RET(
            elemSize != 0 && elemNum > (SIZE_MAX / elemSize),
            HCCL_ERROR(
                "[AivUrmaChannel::AddDeviceEntitySection] %s size overflow, elemSize[%zu], elemNum[%u]", sectionName,
                elemSize, elemNum),
            HCCL_E_PARA);
        section.size = elemSize * static_cast<size_t>(elemNum);
        CHK_PRT_RET(
            section.offset > (SIZE_MAX - section.size),
            HCCL_ERROR(
                "[AivUrmaChannel::AddDeviceEntitySection] %s offset overflow, offset[%zu], size[%zu]", sectionName,
                section.offset, section.size),
            HCCL_E_PARA);
        offset = section.offset + section.size;
        return HCCL_SUCCESS;
    }

    void* GetSlabPtr(void* base, const DeviceEntitySection& section)
    {
        if (section.size == 0) {
            return nullptr;
        }
        return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) + section.offset);
    }

    template <typename T>
    HcclResult CopyArrayToSlab(
        void* slabBase, const T* hostArray, uint32_t arrayNum, const DeviceEntitySection& section, T** deviceArrayPtr,
        const char* arrayName)
    {
        CHK_PTR_NULL(deviceArrayPtr);
        if (arrayNum == 0 || hostArray == nullptr) {
            CHK_PRT_RET(
                arrayNum != 0,
                HCCL_ERROR("[AivUrmaChannel::CopyArrayToSlab] %s hostArray is nullptr, num[%u]", arrayName, arrayNum),
                HCCL_E_PTR);
            *deviceArrayPtr = nullptr;
            return HCCL_SUCCESS;
        }
        CHK_PRT_RET(
            section.size != static_cast<size_t>(arrayNum) * sizeof(T),
            HCCL_ERROR(
                "[AivUrmaChannel::CopyArrayToSlab] %s size mismatch, sectionSize[%zu], expect[%zu]", arrayName,
                section.size, static_cast<size_t>(arrayNum) * sizeof(T)),
            HCCL_E_PARA);
        void* sectionPtr = GetSlabPtr(slabBase, section);
        CHK_PTR_NULL(sectionPtr);
        Hccl::HrtMemcpy(
            sectionPtr, section.size, hostArray, section.size, Hccl::tagRtMemcpyKind::RT_MEMCPY_HOST_TO_DEVICE);
        *deviceArrayPtr = reinterpret_cast<T*>(sectionPtr);
        HCCL_INFO(
            "[AivUrmaChannel::CopyArrayToSlab] %s: host[%p] -> dev[%p], num[%u], size[%zu]", arrayName, hostArray,
            sectionPtr, arrayNum, section.size);
        return HCCL_SUCCESS;
    }

    HcclResult BuildDeviceChannelEntityLayout(const ChannelEntity& hostChannel, DeviceChannelEntityLayout& layout)
    {
        layout.slabSize = AlignUp(sizeof(ChannelEntity), AIV_URMA_ENTITY_ALIGN_SIZE);
        CHK_RET(AddDeviceEntitySection(
            sizeof(RegedNotifyEntity), hostChannel.localNotifyNum, layout.slabSize, layout.localNotifySection,
            "localNotifyAddr"));
        CHK_RET(AddDeviceEntitySection(
            sizeof(RegedNotifyEntity), hostChannel.remoteNotifyNum, layout.slabSize, layout.remoteNotifySection,
            "remoteNotifyAddr"));
        CHK_RET(AddDeviceEntitySection(
            sizeof(RegedBufferEntity), hostChannel.localBufferNum, layout.slabSize, layout.localBufferSection,
            "localBufferAddr"));
        CHK_RET(AddDeviceEntitySection(
            sizeof(RegedBufferEntity), hostChannel.remoteBufferNum, layout.slabSize, layout.remoteBufferSection,
            "remoteBufferAddr"));
        CHK_RET(AddDeviceEntitySection(
            sizeof(SqContext), hostChannel.sqNum, layout.slabSize, layout.sqContextSection, "sqContextAddr"));
        CHK_RET(AddDeviceEntitySection(
            sizeof(CqContext), hostChannel.cqNum, layout.slabSize, layout.cqContextSection, "cqContextAddr"));
        CHK_RET(AddDeviceEntitySection(
            QUEUE_INDEX_MEM_UNIT_SIZE, hostChannel.sqNum, layout.slabSize, layout.sqPiSection, "sqPiAddr"));
        CHK_RET(AddDeviceEntitySection(
            QUEUE_INDEX_MEM_UNIT_SIZE, hostChannel.sqNum, layout.slabSize, layout.sqCiSection, "sqCiAddr"));
        CHK_RET(AddDeviceEntitySection(
            QUEUE_INDEX_MEM_UNIT_SIZE, hostChannel.cqNum, layout.slabSize, layout.cqPiSection, "cqPiAddr"));
        CHK_RET(AddDeviceEntitySection(
            QUEUE_INDEX_MEM_UNIT_SIZE, hostChannel.cqNum, layout.slabSize, layout.cqCiSection, "cqCiAddr"));
        layout.slabSize = AlignUp(layout.slabSize, AIV_URMA_ENTITY_ALIGN_SIZE);
        return HCCL_SUCCESS;
    }

    HcclResult AllocDeviceEntitySlab(size_t slabSize, AclDeviceSlabGuard& slabGuard, void*& slabPtr)
    {
        HcclResult ret = hrtMalloc(&slabPtr, slabSize);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS || slabPtr == nullptr,
            HCCL_ERROR("[AivUrmaChannel::%s] hrtMalloc slab failed, ret[%d], size[%zu]", __func__, ret, slabSize),
            HCCL_E_MEMORY);
        slabGuard.Reset(slabPtr, slabSize);
        return HCCL_SUCCESS;
    }

    HcclResult ZeroQueueIndexSection(
        void* slabPtr, const std::vector<uint8_t>& zeroQueueIndexMem, const DeviceEntitySection& section)
    {
        if (section.size == 0) {
            return HCCL_SUCCESS;
        }
        void* sectionPtr = GetSlabPtr(slabPtr, section);
        CHK_PTR_NULL(sectionPtr);
        Hccl::HrtMemcpy(
            sectionPtr, section.size, zeroQueueIndexMem.data(), section.size,
            Hccl::tagRtMemcpyKind::RT_MEMCPY_HOST_TO_DEVICE);
        return HCCL_SUCCESS;
    }

    HcclResult InitQueueIndexSections(void* slabPtr, const DeviceChannelEntityLayout& layout, uint32_t queueNum)
    {
        std::vector<uint8_t> zeroQueueIndexMem(QUEUE_INDEX_MEM_UNIT_SIZE * queueNum, 0);
        CHK_RET(ZeroQueueIndexSection(slabPtr, zeroQueueIndexMem, layout.sqPiSection));
        CHK_RET(ZeroQueueIndexSection(slabPtr, zeroQueueIndexMem, layout.sqCiSection));
        CHK_RET(ZeroQueueIndexSection(slabPtr, zeroQueueIndexMem, layout.cqPiSection));
        CHK_RET(ZeroQueueIndexSection(slabPtr, zeroQueueIndexMem, layout.cqCiSection));
        return HCCL_SUCCESS;
    }

    void SetQueueIndexDeviceMem(
        Hccl::AivUrmaTransport& transport, void* slabPtr, const DeviceChannelEntityLayout& layout, uint32_t queueNum)
    {
        transport.SetQueueIndexDeviceMem(
            GetSlabPtr(slabPtr, layout.sqPiSection), GetSlabPtr(slabPtr, layout.sqCiSection),
            GetSlabPtr(slabPtr, layout.cqPiSection), GetSlabPtr(slabPtr, layout.cqCiSection),
            queueNum * QUEUE_INDEX_MEM_UNIT_SIZE);
    }

    HcclResult CopyChannelEntityToSlab(
        void* slabPtr, const ChannelEntity& hostChannel, const DeviceChannelEntityLayout& layout,
        ChannelEntity& devChannel)
    {
        devChannel = hostChannel;
        CHK_RET(CopyArrayToSlab(
            slabPtr, hostChannel.localNotifyAddr, hostChannel.localNotifyNum, layout.localNotifySection,
            &devChannel.localNotifyAddr, "localNotifyAddr"));
        CHK_RET(CopyArrayToSlab(
            slabPtr, hostChannel.remoteNotifyAddr, hostChannel.remoteNotifyNum, layout.remoteNotifySection,
            &devChannel.remoteNotifyAddr, "remoteNotifyAddr"));
        CHK_RET(CopyArrayToSlab(
            slabPtr, hostChannel.localBufferAddr, hostChannel.localBufferNum, layout.localBufferSection,
            &devChannel.localBufferAddr, "localBufferAddr"));
        CHK_RET(CopyArrayToSlab(
            slabPtr, hostChannel.remoteBufferAddr, hostChannel.remoteBufferNum, layout.remoteBufferSection,
            &devChannel.remoteBufferAddr, "remoteBufferAddr"));
        CHK_RET(CopyArrayToSlab(
            slabPtr, hostChannel.sqContextAddr, hostChannel.sqNum, layout.sqContextSection, &devChannel.sqContextAddr,
            "sqContextAddr"));
        CHK_RET(CopyArrayToSlab(
            slabPtr, hostChannel.cqContextAddr, hostChannel.cqNum, layout.cqContextSection, &devChannel.cqContextAddr,
            "cqContextAddr"));
        return HCCL_SUCCESS;
    }

    HcclResult CopyChannelEntityHeaderToSlab(
        void* slabPtr, const DeviceChannelEntityLayout& layout, const ChannelEntity& devChannel, void*& entityDevPtr)
    {
        entityDevPtr = GetSlabPtr(slabPtr, layout.entitySection);
        CHK_PTR_NULL(entityDevPtr);
        Hccl::HrtMemcpy(
            entityDevPtr, sizeof(ChannelEntity), &devChannel, sizeof(ChannelEntity),
            Hccl::tagRtMemcpyKind::RT_MEMCPY_HOST_TO_DEVICE);
        return HCCL_SUCCESS;
    }
} // namespace

AivUrmaChannel::AivUrmaChannel(EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc)
    : endpointHandle_(endpointHandle),
      channelDesc_(channelDesc)
{
    channelKind_ = HcommChannelKind::AIV_URMA;
}

AivUrmaChannel::~AivUrmaChannel()
{
    PutSocketIfNeeded();
    ReleaseDeviceChannelEntity();
}

void AivUrmaChannel::PutSocketIfNeeded()
{
    if (socket_ == nullptr) {
        return;
    }
    if (socketConfig_ == nullptr) {
        socket_ = nullptr;
        return;
    }
    if (channelDesc_.socket == nullptr && socket_ != nullptr) {
        (void)SocketMgr::GetInstance(devicePhyId_).PutSocket(socketConfig_, socket_);
    }
    socket_ = nullptr;
}

void AivUrmaChannel::ReleaseDeviceChannelEntity()
{
    if (devChannelEntitySlab_ != nullptr) {
        HcclResult ret = hrtFree(devChannelEntitySlab_);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[AivUrmaChannel::%s] hrtFree devChannelEntitySlab failed, ptr[%p], size[%zu], ret[%d]", __func__,
                devChannelEntitySlab_, devChannelEntitySlabSize_, ret);
        }
        devChannelEntitySlab_ = nullptr;
        devChannelEntitySlabSize_ = 0;
    }
    deviceMemories_.clear();
    devChannelEntity_ = nullptr;
}

HcclResult AivUrmaChannel::ParseInputParam()
{
    // 1. 从 endpointHandle_，获得 localEp_ 和 rdmaHandle_
    Endpoint* localEpPtr = static_cast<Endpoint*>(endpointHandle_);
    CHK_PTR_NULL(localEpPtr);
    localEp_ = localEpPtr->GetEndpointDesc();
    rdmaHandle_ = localEpPtr->GetRdmaHandle();
    devicePhyId_ = localEp_.loc.device.devPhyId;

    socket_ = static_cast<Hccl::Socket*>(channelDesc_.socket);
    remoteEp_ = channelDesc_.remoteEndpoint;
    notifyNum_ = channelDesc_.notifyNum;
    commonRes_.bufferVec.clear();
    if (channelDesc_.exchangeAllMems) {
        HCCL_INFO("[AivUrmaChannel][%s] exchangeAllMems == true. Get memHandles from endpoint.", __func__);
        std::shared_ptr<Hccl::LocalUbRmaBuffer>* memHandles = nullptr;
        uint32_t memHandleNum = 0;
        CHK_RET(GetAllMemHandles(endpointHandle_, reinterpret_cast<void**>(&memHandles), &memHandleNum));
        HCCL_INFO("[AivUrmaChannel][%s] Got memHandleNum[%u].", __func__, memHandleNum);
        for (uint32_t i = 0; i < memHandleNum; ++i) {
            std::shared_ptr<Hccl::LocalUbRmaBuffer>& localUbRmaBuffer = memHandles[i];
            CHK_SMART_PTR_NULL(localUbRmaBuffer);
            Hccl::Buffer* buf = localUbRmaBuffer->GetBuf();
            CHK_PTR_NULL(buf);
            HCCL_INFO(
                "[AivUrmaChannel][%s] Got memHandle No.%u: addr[0x%llx], size[0x%llx], memInfo[%s].", __func__, i,
                localUbRmaBuffer->GetAddr(), localUbRmaBuffer->GetSize(), buf->GetMemInfo().c_str());
            commonRes_.bufferVec.push_back(localUbRmaBuffer.get());
        }
    } else {
        HCCL_INFO("[AivUrmaChannel][%s] exchangeAllMems == false. Get memHandles from channelDesc.", __func__);
        CHK_RET(MakeRmaBufferVecFromMemHandles(
            channelDesc_.memHandles, channelDesc_.memHandleNum, commonRes_.bufferVec, "AivUrmaChannel"));
    }

    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::BuildSocket()
{
    if (socket_ != nullptr) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[AivUrmaChannel][%s] socket ptr is NULL, rebuildSocket", __func__);

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));
    HCCL_INFO("[AivUrmaChannel][%s] built linkData: %s", __func__, linkData.Describe().c_str());
    uint16_t port = channelDesc_.port;
    if (port == 0) {
        port = DEFAULT_LISTENING_PORT;
        HCCL_INFO("[AivUrmaChannel::%s] channelDesc port is 0, use default port [%u]", __func__, port);
    }
    std::string socketTag
        = (channelDesc_.channelName != nullptr) ? std::string(channelDesc_.channelName) : "AUTOMATIC_SOCKET_TAG";
    if (channelDesc_.role == HCOMM_SOCKET_ROLE_RESERVED) {
        EXCEPTION_CATCH(
            socketConfigHolder_ = std::make_unique<Hccl::SocketConfig>(linkData, port, socketTag), return HCCL_E_PTR);
        socketConfigHolder_->noRankId = true;
    } else {
        bool isServer = (channelDesc_.role == HCOMM_SOCKET_ROLE_SERVER);
        EXCEPTION_CATCH(
            socketConfigHolder_ = std::make_unique<Hccl::SocketConfig>(linkData, port, socketTag, isServer),
            return HCCL_E_PTR);
    }
    socketConfig_ = socketConfigHolder_.get();
    CHK_RET(SocketMgr::GetInstance(devicePhyId_).GetSocket(*socketConfigHolder_, socket_));
    HCCL_INFO("[AivUrmaChannel::%s] SUCCESS. port[%u].", __func__, port);
    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::StartListen()
{
    if (channelDesc_.role != HCOMM_SOCKET_ROLE_SERVER) {
        return HCCL_SUCCESS;
    }

    uint16_t port = channelDesc_.port;
    HCCL_INFO("[AivUrmaChannel::%s] Start. EndpointHandle[%p], port[%u]", __func__, endpointHandle_, port);
    if (port == 0) {
        port = DEFAULT_LISTENING_PORT;
        HCCL_INFO("[AivUrmaChannel::%s] channelDesc port is 0, use default port [%u]", __func__, port);
    }
    CHK_RET(static_cast<HcclResult>(HcommEndpointStartListen(endpointHandle_, port, nullptr)));
    HCCL_INFO("[AivUrmaChannel::%s] SUCCESS. port[%u].", __func__, port);
    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::BuildAttr()
{
    attr_.devicePhyId = localEp_.loc.device.devPhyId;
    attr_.opMode = Hccl::OpMode::OPBASE;
    attr_.opAcceState = Hccl::AcceleratorState::AIV;
    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::CreateUbConnectionByProtocol(
    const UbConnBuildContext& ctx, std::unique_ptr<Hccl::DevUbConnection>& ubConn,
    Hccl::DevUbConnection::JettyMode jettyMode)
{
    Hccl::OpMode opMode = Hccl::OpMode::OPBASE;
    bool devUsed = true;
    Hccl::HrtUbJfcMode jfcMode = Hccl::HrtUbJfcMode::USER_CTL;
    // UB_CTP → HCOMM_TA_CTP_UB_TIMEOUT，UB_TP/UB_RTP → HCOMM_TA_RTP_UB_TIMEOUT
    u8 taTimeOut = 0;
    uint32_t taTimeOutValue = 0;
    if (ctx.protocol == Hccl::LinkProtocol::UB_CTP) {
        CHK_RET(hcomm::HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(taTimeOutValue));
    } else {
        CHK_RET(hcomm::HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(taTimeOutValue));
    }
    taTimeOut = static_cast<u8>(taTimeOutValue);
    switch (ctx.protocol) {
        case Hccl::LinkProtocol::UB_TP:
            EXCEPTION_CATCH(
                ubConn = std::make_unique<Hccl::DevUbTpConnection>(
                    rdmaHandle_, ctx.locAddr, ctx.rmtAddr, opMode, devUsed, jfcMode, Hccl::IpAddress(),
                    Hccl::IpAddress(), ctx.qosPre, taTimeOut, COMM_ENGINE_AIV, ctx.sqDepth, jettyMode),
                return HCCL_E_PTR);
            break;
        case Hccl::LinkProtocol::UB_CTP:
            EXCEPTION_CATCH(
                ubConn = std::make_unique<Hccl::DevUbCtpConnection>(
                    rdmaHandle_, ctx.locAddr, ctx.rmtAddr, opMode, devUsed, jfcMode, Hccl::IpAddress(),
                    Hccl::IpAddress(), ctx.qosPre, taTimeOut, COMM_ENGINE_AIV, ctx.sqDepth, jettyMode),
                return HCCL_E_PTR);
            break;
        case Hccl::LinkProtocol::UB_RTP:
            EXCEPTION_CATCH(
                ubConn = std::make_unique<Hccl::DevUbRtpConnection>(
                    rdmaHandle_, ctx.locAddr, ctx.rmtAddr, opMode, devUsed, jfcMode, ctx.locAddr, ctx.rmtAddr,
                    ctx.qosPre, taTimeOut, COMM_ENGINE_AIV, ctx.sqDepth, jettyMode),
                return HCCL_E_PTR);
            break;
        default:
            HCCL_ERROR("%s No LinkProtocol to match", __func__);
            break;
    }
    return HCCL_SUCCESS;
}

HcclResult
AivUrmaChannel::AcquireSharedJettyInBuildConnection(const UbConnBuildContext& ctx, Hccl::DevUbConnection* connection)
{
    // 共享 jetty 模式：复用同 Endpoint 下已创建的 jetty
    Endpoint* endpoint = static_cast<Endpoint*>(endpointHandle_);
    // UB_CTP → HCOMM_TA_CTP_UB_TIMEOUT，UB_TP/UB_RTP → HCOMM_TA_RTP_UB_TIMEOUT
    u8 taTimeOut = 0;
    uint32_t taTimeOutValue = 0;
    if (ctx.protocol == Hccl::LinkProtocol::UB_CTP) {
        CHK_RET(hcomm::HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(taTimeOutValue));
    } else {
        CHK_RET(hcomm::HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(taTimeOutValue));
    }
    taTimeOut = static_cast<u8>(taTimeOutValue);
    auto tempFactory = [rdmaHandle = rdmaHandle_, &ctxLoc = ctx.locAddr, &ctxRmt = ctx.rmtAddr, qosPre = ctx.qosPre,
                        protocol = ctx.protocol, sqDepth = ctx.sqDepth,
                        taTimeOut = taTimeOut]() -> std::unique_ptr<Hccl::DevUbConnection> {
        return CreateSharedJettyConnection(rdmaHandle, ctxLoc, ctxRmt, qosPre, protocol, taTimeOut, sqDepth);
    };
    JettyContext::Ctx sharedCtx{};
    CHK_RET(hcomm::AcquireSharedJettyForChannel(endpoint, connection, tempFactory, sharedCtx));
    // 保存共享 PI/CI 指针，供 BuildChannelEntityToDevice 绑给 transport
    sharedSqPiPtr_ = sharedCtx.sqPiPtr;
    sharedSqCiPtr_ = sharedCtx.sqCiPtr;
    sharedCqPiPtr_ = sharedCtx.cqPiPtr;
    sharedCqCiPtr_ = sharedCtx.cqCiPtr;
    return HCCL_SUCCESS;
}

std::unique_ptr<Hccl::DevUbConnection> AivUrmaChannel::CreateSharedJettyConnection(
    Hccl::RdmaHandle rdmaHandle, const Hccl::IpAddress& locAddr, const Hccl::IpAddress& rmtAddr, u8 qosPre,
    Hccl::LinkProtocol protocol, u8 taTimeOut, u32 sqDepth)
{
    switch (protocol) {
        case Hccl::LinkProtocol::UB_TP: {
            std::unique_ptr<Hccl::DevUbConnection> conn;
            EXCEPTION_CATCH(
                conn = std::make_unique<Hccl::DevUbTpConnection>(
                    rdmaHandle, locAddr, rmtAddr, Hccl::OpMode::OPBASE, true, Hccl::HrtUbJfcMode::USER_CTL,
                    Hccl::IpAddress(), Hccl::IpAddress(), qosPre, taTimeOut, COMM_ENGINE_AIV, sqDepth),
                return nullptr);
            return conn;
        }
        case Hccl::LinkProtocol::UB_CTP: {
            std::unique_ptr<Hccl::DevUbConnection> conn;
            EXCEPTION_CATCH(
                conn = std::make_unique<Hccl::DevUbCtpConnection>(
                    rdmaHandle, locAddr, rmtAddr, Hccl::OpMode::OPBASE, true, Hccl::HrtUbJfcMode::USER_CTL,
                    Hccl::IpAddress(), Hccl::IpAddress(), qosPre, taTimeOut, COMM_ENGINE_AIV, sqDepth),
                return nullptr);
            return conn;
        }
        case Hccl::LinkProtocol::UB_RTP: {
            std::unique_ptr<Hccl::DevUbConnection> conn;
            EXCEPTION_CATCH(
                conn = std::make_unique<Hccl::DevUbRtpConnection>(
                    rdmaHandle, locAddr, rmtAddr, Hccl::OpMode::OPBASE, true, Hccl::HrtUbJfcMode::USER_CTL, locAddr,
                    rmtAddr, qosPre, taTimeOut, COMM_ENGINE_AIV, sqDepth),
                return nullptr);
            return conn;
        }
        default:
            HCCL_ERROR(
                "[AivUrmaChannel][CreateSharedJettyConnection] unsupported protocol[%s], return nullptr.",
                protocol.Describe().c_str());
            return nullptr;
    }
}

HcclResult AivUrmaChannel::BuildConnection()
{
    UbConnBuildContext ctx;
    CHK_RET(PrepareUbConnBuildContext(localEp_, remoteEp_, channelDesc_, ctx));
    CHK_RET(CheckUbSqDepth(ctx, devBaseAttr_));

    // 共享 jetty 模式：主 connection 构造时传 EXTERNAL_INJECT，跳过建 JFC/jetty，等 SetSharedJettyFields 填充
    auto jettyMode = IsSharedJetty() ? Hccl::DevUbConnection::JettyMode::EXTERNAL_INJECT :
                                       Hccl::DevUbConnection::JettyMode::SELF_CREATE;
    std::unique_ptr<Hccl::DevUbConnection> ubConn = nullptr;
    CHK_RET(CreateUbConnectionByProtocol(ctx, ubConn, jettyMode));
    CHK_SMART_PTR_NULL(ubConn);

    // 共享 jetty 模式：复用同 Endpoint 下已创建的 jetty。
    // 必须在 push_back(move(ubConn)) 之前调用：AcquireSharedJetty 失败时 ubConn 仍为局部变量，
    // 函数返回时自动析构，不会在 connections_/connVec 中残留不完整 connection。
    if (IsSharedJetty()) {
        CHK_RET(AcquireSharedJettyInBuildConnection(ctx, ubConn.get()));
    }

    commonRes_.connVec.clear();
    connections_.clear();
    commonRes_.connVec.emplace_back(ubConn.get());
    connections_.push_back(std::move(ubConn));
    PLF_CONFIG_INFO(
        PLF_CHANNEL, "[AivUrmaChannel] build DevUbConnection, protocol[%s] sqDepth[%u] sharedJetty[%d].",
        ctx.protocol.Describe().c_str(), ctx.sqDepth, static_cast<int>(IsSharedJetty()));

    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::BuildAivUrmaTransport()
{
    const Hccl::Socket& socket = *socket_;

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));

    // make_unique / make_shared / release 包一层抛异常的宏
    EXCEPTION_CATCH(
        transport_ = std::make_unique<Hccl::AivUrmaTransport>(
            commonRes_, attr_, linkData, socket, rdmaHandle_), // 这里区分是否是优先recv
        return HCCL_E_PTR);
    PLF_CONFIG_INFO(
        PLF_CHANNEL, "[AivUrmaChannel] create AivUrmaTransport, socket[%s], linkData[%s].", socket_->Describe().c_str(),
        linkData.Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::BuildChannelEntityToDevice(void** devChannelPtr)
{
    if (devChannelPtr == nullptr) {
        HCCL_ERROR("[AivUrmaChannel] BuildChannelEntityToDevice devChannelPtr is nullptr");
        return HCCL_E_PTR;
    }

    CHK_PTR_NULL(transport_.get());

    if (devChannelEntitySlab_ != nullptr) {
        CHK_RET(FillChannelEntityToDevice());
        *devChannelPtr = devChannelEntity_;
        return HCCL_SUCCESS;
    }

    ChannelEntity hostChannel;
    CHK_RET(SecureMemset(&hostChannel, sizeof(ChannelEntity), 0, sizeof(ChannelEntity), "hostChannel"));

    transport_->PrepareHostChannelEntity(&hostChannel);

    DeviceChannelEntityLayout layout;
    CHK_RET(BuildDeviceChannelEntityLayout(hostChannel, layout));
    void* slabPtr = nullptr;
    AclDeviceSlabGuard slabGuard;
    CHK_RET(AllocDeviceEntitySlab(layout.slabSize, slabGuard, slabPtr));
    uint32_t queueNum = std::max(hostChannel.sqNum, hostChannel.cqNum);
    if (IsSharedJetty() && sharedSqPiPtr_ != nullptr) {
        // 共享 jetty：PI/CI 用同 endpoint 下多 channel 共享的 device 内存，slab 内 PI/CI 段闲置不用。
        // 共享内存在首次 AcquireSharedJettyForChannel 时已分配并清零，此处直接绑给 transport。
        transport_->SetQueueIndexDeviceMem(
            sharedSqPiPtr_, sharedSqCiPtr_, sharedCqPiPtr_, sharedCqCiPtr_, queueNum * QUEUE_INDEX_MEM_UNIT_SIZE);
    } else {
        CHK_RET(InitQueueIndexSections(slabPtr, layout, queueNum));
        SetQueueIndexDeviceMem(*transport_, slabPtr, layout, queueNum);
    }

    CHK_RET(SecureMemset(&hostChannel, sizeof(ChannelEntity), 0, sizeof(ChannelEntity), "hostChannel"));
    transport_->GetHostChannelEntity(&hostChannel);
    hostChannel.abiHeader = channelDesc_.header;
    hostChannel.engine = COMM_ENGINE_AIV;
    hostChannel.protocol = channelDesc_.remoteEndpoint.protocol;

    ChannelEntity devChannel;
    CHK_RET(CopyChannelEntityToSlab(slabPtr, hostChannel, layout, devChannel));
    void* entityDevPtr = nullptr;
    CHK_RET(CopyChannelEntityHeaderToSlab(slabPtr, layout, devChannel, entityDevPtr));
    ReleaseDeviceChannelEntity();
    devChannelEntitySlab_ = slabGuard.Release();
    devChannelEntitySlabSize_ = layout.slabSize;
    devChannelEntity_ = entityDevPtr;
    *devChannelPtr = devChannelEntity_;
    HCCL_INFO(
        "[AivUrmaChannel] Build channel entity to device success, devPtr[%p], slabPtr[%p], slabSize[%zu]",
        devChannelEntity_, devChannelEntitySlab_, devChannelEntitySlabSize_);
    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::PreAllocChannelEntityToDevice(void** devChannelPtr)
{
    if (devChannelPtr == nullptr) {
        HCCL_ERROR("[AivUrmaChannel::%s] devChannelPtr is nullptr", __func__);
        return HCCL_E_PTR;
    }
    CHK_PTR_NULL(transport_.get());

    if (devChannelEntitySlab_ != nullptr) {
        *devChannelPtr = devChannelEntity_;
        HCCL_INFO("[AivUrmaChannel::%s] already built, return cached devPtr[%p]", __func__, devChannelEntity_);
        return HCCL_SUCCESS;
    }

    uint32_t bufNum = 0;
    uint32_t connNum = 0;
    transport_->GetEntityCountsForLayout(bufNum, connNum);

    ChannelEntity tmp{};
    tmp.localBufferNum = bufNum;
    tmp.remoteBufferNum = bufNum;
    tmp.sqNum = connNum;
    tmp.cqNum = connNum;

    DeviceChannelEntityLayout layout;
    CHK_RET(BuildDeviceChannelEntityLayout(tmp, layout));

    void* slabPtr = nullptr;
    AclDeviceSlabGuard slabGuard;
    CHK_RET(AllocDeviceEntitySlab(layout.slabSize, slabGuard, slabPtr));

    uint32_t queueNum = std::max(tmp.sqNum, tmp.cqNum);
    if (IsSharedJetty() && sharedSqPiPtr_ != nullptr) {
        // 共享 jetty：PI/CI 用共享 device 内存，slab 内 PI/CI 段闲置不用（已在首次分配时清零）
        transport_->SetQueueIndexDeviceMem(
            sharedSqPiPtr_, sharedSqCiPtr_, sharedCqPiPtr_, sharedCqCiPtr_, queueNum * QUEUE_INDEX_MEM_UNIT_SIZE);
    } else {
        CHK_RET(InitQueueIndexSections(slabPtr, layout, queueNum));
    }

    devChannelEntitySlab_ = slabGuard.Release();
    devChannelEntitySlabSize_ = layout.slabSize;
    devChannelEntity_ = GetSlabPtr(devChannelEntitySlab_, layout.entitySection);
    if (!IsSharedJetty() || sharedSqPiPtr_ == nullptr) {
        SetQueueIndexDeviceMem(*transport_, devChannelEntitySlab_, layout, queueNum);
    }
    *devChannelPtr = devChannelEntity_;

    HCCL_INFO(
        "[AivUrmaChannel::%s] pre-alloc success, devPtr[%p], slabPtr[%p], slabSize[%zu]", __func__, devChannelEntity_,
        devChannelEntitySlab_, devChannelEntitySlabSize_);
    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::FillChannelEntityToDevice()
{
    if (devChannelEntitySlab_ == nullptr) {
        HCCL_ERROR("[AivUrmaChannel::%s] devChannelEntitySlab_ is nullptr, not pre-allocated.", __func__);
        return HCCL_E_INTERNAL;
    }
    CHK_PTR_NULL(transport_.get());

    ChannelEntity hostChannel;
    CHK_RET(SecureMemset(&hostChannel, sizeof(ChannelEntity), 0, sizeof(ChannelEntity), "hostChannel"));
    transport_->GetHostChannelEntity(&hostChannel);
    hostChannel.abiHeader = channelDesc_.header;
    hostChannel.engine = COMM_ENGINE_AIV;
    hostChannel.protocol = channelDesc_.remoteEndpoint.protocol;

    DeviceChannelEntityLayout layout;
    CHK_RET(BuildDeviceChannelEntityLayout(hostChannel, layout));
    if (layout.slabSize > devChannelEntitySlabSize_) {
        HCCL_ERROR(
            "[AivUrmaChannel::%s] slabSize[%zu] > preAllocSize[%zu]", __func__, layout.slabSize,
            devChannelEntitySlabSize_);
        return HCCL_E_INTERNAL;
    }

    ChannelEntity devChannel;
    CHK_RET(CopyChannelEntityToSlab(devChannelEntitySlab_, hostChannel, layout, devChannel));
    void* entityDevPtr = nullptr;
    CHK_RET(CopyChannelEntityHeaderToSlab(devChannelEntitySlab_, layout, devChannel, entityDevPtr));

    devChannelEntity_ = entityDevPtr;
    HCCL_INFO("[AivUrmaChannel::%s] fill success, devPtr[%p]", __func__, devChannelEntity_);
    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::GetNotifyNum([[maybe_unused]] uint32_t* notifyNum) const
{
    HCCL_INFO("AivUrmaChannel GetNotifyNum is not supported.");
    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    return transport_->GetRemoteMems(memNum, remoteMem, memInfos);
}

HcclResult AivUrmaChannel::UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum)
{
    CHK_PTR_NULL(transport_.get());
    return transport_->UpdateMemInfo(memHandles, memHandleNum);
}

HcclResult AivUrmaChannel::Clean()
{
    PLF_CONFIG_INFO(PLF_CHANNEL, "[AivUrmaChannel] clean channel resource.");
    ReleaseDeviceChannelEntity();
    ReleasePtrArrayDevMems();
    transport_.reset();
    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::Resume()
{
    BuildConnection();
    BuildAivUrmaTransport();
    return HCCL_SUCCESS;
}

HcclResult AivUrmaChannel::NotifyRecord([[maybe_unused]] const uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AivUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AivUrmaChannel::NotifyWait([[maybe_unused]] const uint32_t localNotifyIdx, [[maybe_unused]] const uint32_t timeout)
{
    HCCL_INFO("[AivUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AivUrmaChannel::WriteWithNotify(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] const uint64_t len,
    [[maybe_unused]] uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AivUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AivUrmaChannel::Write([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len)
{
    HCCL_INFO("[AivUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AivUrmaChannel::Read([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len)
{
    HCCL_INFO("[AivUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AivUrmaChannel::ChannelFence()
{
    HCCL_INFO("[AivUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AivUrmaChannel::Init()
{
    /*
        Argue result: make_unique 配合一场捕获的宏 EXCEPTION CATCH
        Attention: const 和引用
    */
    CHK_RET(ParseInputParam());
    CHK_RET(StartListen());
    CHK_RET(BuildSocket());
    CHK_RET(BuildAttr());

    CHK_RET(HccpRaGetDevBaseAttr(rdmaHandle_, &devBaseAttr_));
    CHK_RET(BuildConnection());
    CHK_RET(BuildAivUrmaTransport());
    return HCCL_SUCCESS;
}

ChannelStatus AivUrmaChannel::GetStatus()
{
    Hccl::TransportStatus transportStatus = transport_->GetStatus();
    ChannelStatus out = ChannelStatus::INIT;
    switch (transportStatus) {
        case Hccl::TransportStatus::INIT:
            out = ChannelStatus::INIT;
            break;
        case Hccl::TransportStatus::SOCKET_OK:
            out = ChannelStatus::SOCKET_OK;
            break;
        case Hccl::TransportStatus::SOCKET_TIMEOUT:
            out = ChannelStatus::SOCKET_TIMEOUT;
            break;
        case Hccl::TransportStatus::READY:
            out = ChannelStatus::READY;
            break;
        default:
            HCCL_ERROR("[AivUrmaChannel][%s] Invalid TransportStatus[%d]", __func__, transportStatus);
            out = ChannelStatus::INVALID;
            break;
    }
    return out;
}

} // namespace hcomm
