/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <arpa/inet.h>
#include <securec.h>
#include <string>
#include "network/hccp_common.h"
#include "device_capacity.h"
#include "network_manager_pub.h"
#include "adapter_rts.h"
#include "externalinput_pub.h"
#include "hccl_network.h"
#include "externalinput.h"
#include "../host/transport_ibverbs.h"

// 混合模式（RoCE Cross-Mode）公共类型定义
#include "../../../../base_comm/resources/endpoint_pairs/channels/host/exchange_data_format.h"

using namespace std;
constexpr u32 RDMA_QP_EXPECT_STATUS_PAUSE = 5;
constexpr u32 RDMA_QP_EXPECT_STATUS_CONNECTED = 1;

namespace hccl {
std::array<DeviceMem, MAX_MODULE_DEVICE_NUM> TransportIbverbs::notifyValueMem_;
std::array<std::mutex, MAX_MODULE_DEVICE_NUM> TransportIbverbs::notifyValueMutex_;
std::array<Referenced, MAX_MODULE_DEVICE_NUM> TransportIbverbs::instanceRef_;
UniversalConcurrentMap<u64, TransportIbverbs*> TransportIbverbs::g_qpn2IbversLinkMap_;
bool TransportIbverbs::g_flag = false;
bool TransportIbverbs::g_isSupCqeErrInfoListConfig = false;
u32 TransportIbverbs::cqeErrQpn_ = 0;

constexpr u32 CQE_ARRAY_SIZE = 128;
constexpr u32 DEV_PHY_ID_BIT = 32;

constexpr u32 WQE_RESERVE_LENGTH = 4;

constexpr u32 NOTIFY_VA_ALIGN_EIGHT = 8; // notifyVa地址8byte对齐

TransportIbverbs::TransportIbverbs(
    DispatcherPub* dispatcher, const std::unique_ptr<NotifyPool>& notifyPool, MachinePara& machinePara,
    std::chrono::milliseconds timeout)
    : TransportNet(dispatcher, notifyPool, machinePara, timeout),
      qpsPerConnection_(1),
      notifySize_(0),
      ackNotify_(nullptr),
      dataAckNotify_(nullptr),
      access_(RA_ACCESS_LOCAL_WRITE | RA_ACCESS_REMOTE_WRITE | RA_ACCESS_REMOTE_READ),
      workFlowMode_(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB),
      sqeCounter_(0),
      currentQP_(0),
      qpMode_(machinePara.qpMode)
{
    dataNotify_ = nullptr;
    if (machinePara_.deviceLogicId >= 0 && (static_cast<u32>(machinePara_.deviceLogicId) < MAX_MODULE_DEVICE_NUM)) {
        instanceRef_[machinePara_.deviceLogicId].Ref();
    }
}

TransportIbverbs::~TransportIbverbs()
{
    HCCL_DEBUG("~TransportIbverbs Enter!");

    (void)DeInit();

    if (machinePara_.deviceLogicId >= 0 && (static_cast<u32>(machinePara_.deviceLogicId) < MAX_MODULE_DEVICE_NUM)) {
        if (instanceRef_[machinePara_.deviceLogicId].Unref() == 0) {
            std::unique_lock<std::mutex> lock(notifyValueMutex_[machinePara_.deviceLogicId]);
            notifyValueMem_[machinePara_.deviceLogicId].free();
        }
    }
    HCCL_DEBUG("~TransportIbverbs Success!");
}

HcclResult TransportIbverbs::DeInit()
{
    (void)DeRegMR();

    (void)DestroySignal();

    (void)DestroyQP();

    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::DeRegOneMR(QpHandle& qpHandle, MemMsg& memMsg)
{
    struct MrInfoT mrInfo = {};
    mrInfo.addr = memMsg.addr;
    HcclResult ret = HrtRaMrDereg(qpHandle, &mrInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("errNo[0x%016llx] in link lbv, In lbv exp deconstruct, mr dereg failed.", HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

void TransportIbverbs::DeRegMRForQPhandles(MemMsg& memMsg)
{
    for (u32 j = 0; j < combineQpHandles_.size(); j++) {
        if (combineQpHandles_[j].qpHandle == nullptr) {
            continue;
        }
        (void)DeRegOneMR(combineQpHandles_[j].qpHandle, memMsg);
    }
    for (u32 j = 0; j < multiCombineQpHandles_.size(); j++) {
        if (multiCombineQpHandles_[j].qpHandle == nullptr) {
            continue;
        }
        (void)DeRegOneMR(multiCombineQpHandles_[j].qpHandle, memMsg);
    }
}

HcclResult TransportIbverbs::DeRegMR()
{
    /* 销毁mr */
    std::map<uintptr_t, s32> addrMap;
    for (s32 i = 0; i < static_cast<s32>(MemType::MEM_TYPE_RESERVED); i++) {
        if (memMsg_[i].mrRegFlag == REG_VALID) {
            std::pair<std::map<uintptr_t, s32>::iterator, bool> res
                = addrMap.insert(std::pair<uintptr_t, s32>(reinterpret_cast<uintptr_t>(memMsg_[i].addr), 0));
            if (res.second) {
                DeRegMRForQPhandles(memMsg_[i]);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::DestroyQP(QpHandle& qpHandle)
{
    if (qpHandle != nullptr) {
        struct QpAttr attr {};
        CHK_RET(hrtRaGetQpAttr(qpHandle, &attr));

        g_qpn2IbversLinkMap_.Erase(((static_cast<u64>(machinePara_.localDeviceId) << DEV_PHY_ID_BIT) | attr.qpn));

        HcclResult ret = HrtRaQpDestroy(qpHandle);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("errNo[0x%016llx] in link lbv, lbv exp deconstruct, qp destroy failed.", HCCL_ERROR_CODE(ret));
        }
        qpHandle = nullptr;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::DestroyQP()
{
    for (u32 i = 0; i < combineQpHandles_.size(); i++) {
        CHK_RET(DestroyQP(combineQpHandles_[i].qpHandle));
    }
    for (u32 i = 0; i < multiCombineQpHandles_.size(); i++) {
        CHK_RET(DestroyQP(multiCombineQpHandles_[i].qpHandle));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::Stop()
{
    HcclResult ret = hrtRaQpBatchModify(
        nicRdmaHandle_, &combineQpHandles_[0].qpHandle, combineQpHandles_.size(), RDMA_QP_EXPECT_STATUS_PAUSE);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("errNo[0x%016llx] in link lbv, ra qp modify stop fail.", HCCL_ERROR_CODE(ret));
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::Resume()
{
    HcclResult ret = hrtRaQpBatchModify(
        nicRdmaHandle_, &combineQpHandles_[0].qpHandle, combineQpHandles_.size(), RDMA_QP_EXPECT_STATUS_CONNECTED);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("errNo[0x%016llx] in link lbv, ra qp modify resume fail.", HCCL_ERROR_CODE(ret));
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::Init()
{
    HCCL_INFO(
        "machineType=[%d], serverId=[%s], localDeviceId=[%d], remoteDeviceId=[%d], "
        "localRank=[%u], localUserRank=[%u], remoteRank=[%u], remoteUserrank=[%u], "
        "deviceType=[%d], inputMem=[%p], outputMem=[%p], isAicpuModeEn[%d], notifyNum[%u], "
        "isIndOp[%d], custom exchange data size [%llu], drainEnable[%d]",
        machinePara_.machineType, machinePara_.serverId.c_str(), machinePara_.localDeviceId,
        machinePara_.remoteDeviceId, machinePara_.localUserrank, machinePara_.localWorldRank,
        machinePara_.remoteUserrank, machinePara_.remoteWorldRank, machinePara_.deviceType, machinePara_.inputMem.ptr(),
        machinePara_.outputMem.ptr(), machinePara_.isAicpuModeEn, machinePara_.notifyNum, machinePara_.isIndOp,
        machinePara_.exchangeInfo.size(), machinePara_.drainEnable);
    HcclUs startut = TIME_NOW();

    if (machinePara_.userMemEnable) {
        CHK_SMART_PTR_NULL(machinePara_.inputMem);
        CHK_SMART_PTR_NULL(machinePara_.outputMem);
        CHK_SMART_PTR_NULL(notifyPool_);
    }

    if (machinePara_.drainEnable) {
        CHK_SMART_PTR_NULL(notifyPool_);
    }

    CHK_PTR_NULL(dispatcher_);
    CHK_RET(CheckDeviceId());
    CHK_RET(CheckExchangeData());

    // 上层初始化时保证 machinePara_.sockets 非空
    if (machinePara_.sockets.size() == 0) {
        HCCL_ERROR("machinePara sockets is empty.");
        return HCCL_E_INTERNAL;
    }
    defaultSocket_ = machinePara_.sockets[0];
    CHK_PTR_NULL(defaultSocket_);

    CHK_RET(hrtGetDeviceType(localDeviceType));
    HCCL_INFO("localDeviceType=[%d], remoteDeviceType=[%d]", localDeviceType, machinePara_.deviceType);
    CHK_RET(GetNicHandle());

    // 设置linkType
    transportAttr_.linkType = hccl::LinkType::LINK_ROCE;

    /* 获取当前的连接模式，offline模式或者op base模式 */
    workFlowMode_ = GetWorkflowMode();
    HCCL_INFO("current work mode is [%d]", workFlowMode_);

    CHK_RET(GetNotifySize());

    /* 创建QP连接 */
    CHK_RET(InitQpConnect());

    HCCL_INFO("linkexp initialization success,Time:%lld us", DURATION_US(TIME_NOW() - startut));

    CHK_RET(GetQpAttr());
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::GetQpAttr()
{
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(
        stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "communicator[%s], local rank[%u], ip[%s], remote rank[%u], ip[%s], transporttype[%s]",
        machinePara_.tag.c_str(), machinePara_.localUserrank, machinePara_.localIpAddr.GetReadableAddress(),
        machinePara_.remoteUserrank, machinePara_.remoteIpAddr.GetReadableAddress(),
        GetLinkTypeEnumStr(GetLinkType()).c_str());
    CHK_PRT_RET(
        ret == -1, HCCL_ERROR("[GetQpAttr]errNo[0x%016llx] sal snprintf_s error", HCCL_ERROR_CODE(HCCL_E_INTERNAL)),
        HCCL_E_INTERNAL);
    std::string logInfo = "create hccl transport:" + std::string(stackLogBuffer);
    for (u32 i = 0; i < combineQpHandles_.size(); i++) {
        struct QpAttr attr {};
        CHK_RET(hrtRaGetQpAttr(combineQpHandles_[i].qpHandle, &attr));
        HCCL_USER_CRITICAL_LOG(
            "%s, rdma qpn[%u], rdma qp sport[%u], rdma TC[%u], rdma SL[%u]", logInfo.c_str(), attr.qpn, attr.udpSport,
            machinePara_.tc, machinePara_.sl);
    }
    if (UseMultiQp()) {
        for (u32 i = 0; i < multiCombineQpHandles_.size(); i++) {
            struct QpAttr attr {};
            CHK_RET(hrtRaGetQpAttr(multiCombineQpHandles_[i].qpHandle, &attr));
            HCCL_USER_CRITICAL_LOG(
                "%s, rdma qpn[%u], rdma qp sport[%u], rdma TC[%u], rdma SL[%u]", logInfo.c_str(), attr.qpn,
                attr.udpSport, machinePara_.tc, machinePara_.sl);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::GetNotifySize()
{
    u32 notifySize = 0;
    CHK_RET(hrtGetNotifySize(notifySize));
    notifySize_ = notifySize;
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::IsUseQpCreateWithAttrs(bool& isUseQpCreateWithAttrs, s32 qpMode)
{
    isUseQpCreateWithAttrs = false;
    if (machinePara_.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        bool is910Bor91093
            = machinePara_.deviceType == DevType::DEV_TYPE_910B || machinePara_.deviceType == DevType::DEV_TYPE_910_93;
        if (is910Bor91093 && (qpMode == OFFLINE_QP_MODE_EXT || qpMode == OPBASE_QP_MODE_EXT)) {
            isUseQpCreateWithAttrs = true;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::FillExchangeDataTotalSize()
{
    exchangeDataTotalSize_ = 0;
    exchangeDataTotalSize_ += sizeof(u32); // 首个内容放qp数量
    if (UseMultiQp()) {
        exchangeDataTotalSize_ += sizeof(u32); // 再放个MultiQpThreshold
    }
    if (machinePara_.userMemEnable) {
        exchangeDataTotalSize_ += sizeof(MemMsg) * 2; // 2: output and input mem
        exchangeDataTotalSize_ += sizeof(MemMsg) * 3; // 3: dataNotify_\ackNotify_\dataAckNotify_
        if (UseMultiQp()) {
            exchangeDataTotalSize_ += qpsPerConnection_ * sizeof(MemMsg); // 多QP下新增协商内容
        }
    }
    if (machinePara_.drainEnable) {
        exchangeDataTotalSize_ += sizeof(MemMsg); // Drain用到了dataNotify_
        exchangeDataTotalSize_ += sizeof(MemMsg); // 本端notify srcmem, 用于Drain
    }
    if (!isHybridMode_) {
        exchangeDataTotalSize_ += machinePara_.exchangeInfo.size();
    }

    // 4.新增notify资源统计
    // 单qp notify资源大小：1*notifyNum， 多qp notify资源大小：qpNum*notifyNum
    exchangeDataTotalSize_ += qpsPerConnection_ * sizeof(MemMsg) * notifyNum_;

    // 5.新增和对端协商atomic write是否使能
    exchangeDataTotalSize_ += sizeof(u8);

    if (machinePara_.isIndOp) {
        // 6. userDeviceMem数量\userDeviceMem\userHostMem数量\userHostMem
        exchangeDataTotalSize_ += sizeof(u32);
        exchangeDataTotalSize_ += sizeof(MemMsg) * machinePara_.userDeviceMem.size();
        exchangeDataTotalSize_ += sizeof(u32);
        exchangeDataTotalSize_ += sizeof(MemMsg) * machinePara_.userHostMem.size();
    }

    HCCL_DEBUG("[TransportIbverbs][FillExchangeDataTotalSize] exchangeDataTotalSize[%llu]", exchangeDataTotalSize_);
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::ConstructExchangeForSend()
{
    exchangeDataForSend_.resize(exchangeDataTotalSize_);
    u8* exchangeDataPtr = exchangeDataForSend_.data();
    u64 exchangeDataBlankSize = exchangeDataTotalSize_;
    // 把qp对数量放在最前头，第一个做检验
    u32 qpNum = UseMultiQp() ? qpsPerConnection_ : 1;
    s32 sRet = memcpy_s(exchangeDataPtr, sizeof(u32), reinterpret_cast<void*>(&qpNum), sizeof(u32));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[Set][LocalMem]errNo[0x%016llx] memory copy failed. errorno[%d], params:dstMaxSize[%zu],cnt[%zu]",
            HCCL_ERROR_CODE(HCCL_E_MEMORY), sRet, sizeof(u32), sizeof(u32)),
        HCCL_E_MEMORY);
    exchangeDataPtr += sizeof(u32);
    exchangeDataBlankSize -= sizeof(u32);

    if (UseMultiQp()) {
        // 把multiQpThreshold放第二个，第二个做检验
        u32 multiQpThreshold = GetExternalInputMultiQpThreshold();
        sRet = memcpy_s(exchangeDataPtr, sizeof(u32), reinterpret_cast<void*>(&multiQpThreshold), sizeof(u32));
        CHK_PRT_RET(
            sRet != EOK,
            HCCL_ERROR(
                "[Set][LocalMem]errNo[0x%016llx] memory copy failed. errorno[%d], params:dstMaxSize[%zu],cnt[%zu]",
                HCCL_ERROR_CODE(HCCL_E_MEMORY), sRet, sizeof(u32), sizeof(u32)),
            HCCL_E_MEMORY);
        exchangeDataPtr += sizeof(u32);
        exchangeDataBlankSize -= sizeof(u32);
    }

    if (machinePara_.userMemEnable) {
        CHK_RET(RegUserMem(MemType::USER_OUTPUT_MEM, exchangeDataPtr, exchangeDataBlankSize));
        CHK_RET(RegUserMem(MemType::USER_INPUT_MEM, exchangeDataPtr, exchangeDataBlankSize));
        if (machinePara_.isAicpuModeEn) {
            CHK_RET(CreateNotifyBuffer(
                dataNotify_, MemType::DATA_NOTIFY_MEM, exchangeDataPtr, exchangeDataBlankSize,
                NotifyLoadType::DEVICE_NOTIFY));
            CHK_RET(CreateNotifyBuffer(
                ackNotify_, MemType::ACK_NOTIFY_MEM, exchangeDataPtr, exchangeDataBlankSize,
                NotifyLoadType::DEVICE_NOTIFY));
            CHK_RET(CreateNotifyBuffer(
                dataAckNotify_, MemType::DATA_ACK_NOTIFY_MEM, exchangeDataPtr, exchangeDataBlankSize,
                NotifyLoadType::DEVICE_NOTIFY));
        } else {
            CHK_RET(CreateNotifyBuffer(dataNotify_, MemType::DATA_NOTIFY_MEM, exchangeDataPtr, exchangeDataBlankSize));
            CHK_RET(CreateNotifyBuffer(ackNotify_, MemType::ACK_NOTIFY_MEM, exchangeDataPtr, exchangeDataBlankSize));
            CHK_RET(CreateNotifyBuffer(
                dataAckNotify_, MemType::DATA_ACK_NOTIFY_MEM, exchangeDataPtr, exchangeDataBlankSize));
        }
    }
    if (!machinePara_.userMemEnable && machinePara_.drainEnable) {
        if (machinePara_.isAicpuModeEn) {
            CHK_RET(CreateNotifyBuffer(
                dataNotify_, MemType::DATA_NOTIFY_MEM, exchangeDataPtr, exchangeDataBlankSize,
                NotifyLoadType::DEVICE_NOTIFY));
        } else {
            CHK_RET(CreateNotifyBuffer(dataNotify_, MemType::DATA_NOTIFY_MEM, exchangeDataPtr, exchangeDataBlankSize));
        }
        // Drain操作需读取对端srcmem
        CHK_SAFETY_FUNC_RET(memcpy_s(
            exchangeDataPtr, exchangeDataBlankSize,
            reinterpret_cast<void*>(&memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)]), sizeof(MemMsg)));
        exchangeDataPtr += sizeof(MemMsg);
        exchangeDataBlankSize -= sizeof(MemMsg);
    }
    if (!isHybridMode_) {
        CHK_RET(ConstructExchangeDataForSend(exchangeDataPtr, exchangeDataBlankSize));
    }
    if (machinePara_.userMemEnable && UseMultiQp()) {
        for (u32 i = 0; i < qpsPerConnection_; i++) {
            std::shared_ptr<LocalIpcNotify> oneNotify;
            if (machinePara_.isAicpuModeEn) {
                CHK_RET(CreateNotifyBuffer(
                    oneNotify, MemType::MULTI_QP_DATA_NOTIFY_MEM, exchangeDataPtr, exchangeDataBlankSize,
                    NotifyLoadType::DEVICE_NOTIFY));
            } else {
                CHK_RET(CreateNotifyBuffer(
                    oneNotify, MemType::MULTI_QP_DATA_NOTIFY_MEM, exchangeDataPtr, exchangeDataBlankSize));
            }
            multiQpDataNotify_.push_back(std::move(oneNotify));
        }
    }

    // 创建notify pool资源
    // 单qp创建1*notifyNum个，多qp创建qpNum*notifyNum个
    for (u32 i = 0; i < qpsPerConnection_; i++) {
        std::vector<std::shared_ptr<LocalIpcNotify>> notifyVec;
        CHK_RET(CreateNotifyVectorBuffer(notifyVec, exchangeDataPtr, exchangeDataBlankSize));
        userMultiQpLocalNotify_.push_back(std::move(notifyVec));
    }

    u8 localEnableAtomicWrite = machinePara_.enableAtomicWrite ? 1 : 0;
    sRet = memcpy_s(exchangeDataPtr, sizeof(u8), reinterpret_cast<void*>(&localEnableAtomicWrite), sizeof(u8));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[Set][LocalMem]errNo[0x%016llx] memory copy failed. errorno[%d], params:dstMaxSize[%zu], cnt[%zu]",
            HCCL_ERROR_CODE(HCCL_E_MEMORY), sRet, sizeof(bool), sizeof(bool)),
        HCCL_E_MEMORY);
    exchangeDataPtr += sizeof(u8);
    exchangeDataBlankSize -= sizeof(u8);

    if (machinePara_.isIndOp) {
        CHK_RET(RegCustomUserMem(exchangeDataPtr, exchangeDataBlankSize));
    }

    if (exchangeDataBlankSize != 0) {
        HCCL_ERROR(
            "[TransportIbverbs][ConstructExchangeForSend] failed to construct exchange Data "
            "exchangeDataBlankSize[%llu]",
            exchangeDataBlankSize);
        return HCCL_E_INTERNAL;
    }

    HCCL_DEBUG("[TransportIbverbs] ConstructExchangeForSend finished.");
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::ParseReceivedExchangeData()
{
    u8* exchangeDataPtr = exchangeDataForRecv_.data();
    u64 exchangeDataBlankSize = exchangeDataTotalSize_;

    // 首先解析qp对数量，并作一致性校验
    u32 localQpNum = UseMultiQp() ? qpsPerConnection_ : 1;
    u32 remoteQpNum = 0;
    s32 sRet = memcpy_s(reinterpret_cast<void*>(&remoteQpNum), sizeof(u32), exchangeDataPtr, sizeof(u32));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[Get][RemoteMem]errNo[0x%016llx] memory copy failed. errorno[%d], params:dstMaxSize[%zu],cnt[%zu]",
            HCCL_ERROR_CODE(HCCL_E_MEMORY), sRet, sizeof(u32), sizeof(u32)),
        HCCL_E_MEMORY);
    CHK_PRT_RET(
        localQpNum != remoteQpNum,
        HCCL_ERROR(
            "[TransportIbverbs][ParseReceivedExchangeData]"
            "local qps[%u] not equal to remote qps[%u], rank:local[%u],remote[%u]",
            localQpNum, remoteQpNum, machinePara_.localUserrank, machinePara_.remoteUserrank),
        HCCL_E_INTERNAL);
    exchangeDataPtr += sizeof(u32);
    exchangeDataBlankSize -= sizeof(u32);

    if (UseMultiQp()) {
        // 再解析multiQpThreshold，并作一致性校验
        u32 localmultiQpThreshold = GetExternalInputMultiQpThreshold();
        u32 remotemultiQpThreshold = 0;
        sRet = memcpy_s(reinterpret_cast<void*>(&remotemultiQpThreshold), sizeof(u32), exchangeDataPtr, sizeof(u32));
        CHK_PRT_RET(
            sRet != EOK,
            HCCL_ERROR(
                "[Get][RemoteMem]errNo[0x%016llx] memory copy failed. errorno[%d], params:dstMaxSize[%zu],cnt[%zu]",
                HCCL_ERROR_CODE(HCCL_E_MEMORY), sRet, sizeof(u32), sizeof(u32)),
            HCCL_E_MEMORY);
        CHK_PRT_RET(
            localmultiQpThreshold != remotemultiQpThreshold,
            HCCL_ERROR(
                "[TransportIbverbs][ParseReceivedExchangeData]"
                "local env HCCL_MULTI_QP_THRESHOLD[%u] not equal to remote env HCCL_MULTI_QP_THRESHOLD[%u], "
                "rank:local[%u],remote[%u]",
                localmultiQpThreshold, remotemultiQpThreshold, machinePara_.localUserrank, machinePara_.remoteUserrank),
            HCCL_E_INTERNAL);
        exchangeDataPtr += sizeof(u32);
        exchangeDataBlankSize -= sizeof(u32);
    }

    if (machinePara_.userMemEnable) {
        CHK_RET(GetRemoteAddr(MemType::USER_OUTPUT_MEM, exchangeDataPtr, exchangeDataBlankSize));
        CHK_RET(GetRemoteAddr(MemType::USER_INPUT_MEM, exchangeDataPtr, exchangeDataBlankSize));
        CHK_RET(GetRemoteAddr(MemType::DATA_NOTIFY_MEM, exchangeDataPtr, exchangeDataBlankSize));
        s32 sret = memcpy_s(
            &remoteDataNotifyMsg_, sizeof(MemMsg), &remoteMemMsg_[static_cast<u32>(MemType::DATA_NOTIFY_MEM)],
            sizeof(MemMsg));
        CHK_PRT_RET(
            sret != EOK,
            HCCL_ERROR(
                "[Get][RemoteMem]errNo[0x%016llx] In lbv exp init, memory copy failed. errorno[%d], "
                "params:destMaxSize[%zu],count[%zu]",
                HCCL_ERROR_CODE(HCCL_E_MEMORY), sret, sizeof(MemMsg), sizeof(MemMsg)),
            HCCL_E_MEMORY);
        CHK_RET(GetRemoteAddr(MemType::ACK_NOTIFY_MEM, exchangeDataPtr, exchangeDataBlankSize));
        CHK_RET(GetRemoteAddr(MemType::DATA_ACK_NOTIFY_MEM, exchangeDataPtr, exchangeDataBlankSize));
    }
    if (!machinePara_.userMemEnable && machinePara_.drainEnable) {
        CHK_RET(GetRemoteAddr(MemType::DATA_NOTIFY_MEM, exchangeDataPtr, exchangeDataBlankSize));
        CHK_RET(GetRemoteAddr(MemType::NOTIFY_SRC_MEM, exchangeDataPtr, exchangeDataBlankSize));
    }

    if (!isHybridMode_) {
        CHK_RET(ParseExchangeData(exchangeDataPtr, exchangeDataBlankSize));
    }

    if (machinePara_.userMemEnable && UseMultiQp()) {
        for (u32 i = 0; i < qpsPerConnection_; i++) {
            CHK_RET(GetRemoteAddr(MemType::MULTI_QP_DATA_NOTIFY_MEM, exchangeDataPtr, exchangeDataBlankSize));
        }
    }

    // 解析远端新增的notify资源，二维vec大小：qpNum*notifyNum
    userMultiQpRemoteNotifyMsg_.resize(qpsPerConnection_);
    for (u32 i = 0; i < qpsPerConnection_; i++) {
        userMultiQpRemoteNotifyMsg_[i].resize(notifyNum_);
        for (u32 j = 0; j < notifyNum_; j++) {
            CHK_RET(GetRemoteNotifyAddr(exchangeDataPtr, exchangeDataBlankSize, userMultiQpRemoteNotifyMsg_[i][j]));
        }
    }

    // 解析远端是否都支持atomic write，仅本端和对端都支持时才使能atomic write
    u8 remoteEnableAtomicWrite = 0;
    sRet = memcpy_s(reinterpret_cast<void*>(&remoteEnableAtomicWrite), sizeof(u8), exchangeDataPtr, sizeof(u8));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[Get][RemoteMem]errNo[0x%016llx] memory copy failed. errorno[%d], params:dstMaxSize[%zu], cnt[%zu]",
            HCCL_ERROR_CODE(HCCL_E_MEMORY), sRet, sizeof(bool), sizeof(bool)),
        HCCL_E_MEMORY);
    useAtomicWrite_ = machinePara_.enableAtomicWrite && (remoteEnableAtomicWrite != 0);
    HCCL_INFO(
        "atomic write enable only when both the local and remote support, local[%d], remote[%d], result[%d]",
        machinePara_.enableAtomicWrite, remoteEnableAtomicWrite, useAtomicWrite_);
    exchangeDataPtr += sizeof(u8);
    exchangeDataBlankSize -= sizeof(u8);

    if (machinePara_.isIndOp) {
        CHK_RET(GetIndOpRemoteAddr(exchangeDataPtr, exchangeDataBlankSize));
    }

    if (exchangeDataBlankSize != 0) {
        HCCL_ERROR(
            "[TransportIbverbs][ParseReceivedExchangeData] failed to Parse exchange Data "
            "exchangeDataBlankSize[%llu]",
            exchangeDataBlankSize);
        return HCCL_E_INTERNAL;
    }
    HCCL_DEBUG("Parse Received ExchangeData success!");
    return HCCL_SUCCESS;
}

void TransportIbverbs::ModifyAtomicWriteAfterReduce(u32& preWrOpcode, u64 wqeType, u32& opcode, u32& immData)
{
    bool isNotifyWqe = wqeType == static_cast<u64>(WqeType::WQE_TYPE_DATA_NOTIFY)
                       || wqeType == static_cast<u64>(WqeType::WQE_TYPE_ACK_NOTIFY)
                       || wqeType == static_cast<u64>(WqeType::WQE_TYPE_DATA_ACK_NOTIFY);
    if (useAtomicWrite_ && preWrOpcode == RA_WR_RDMA_REDUCE_WRITE && isNotifyWqe) {
        opcode = RA_WR_RDMA_ATOMIC_WRITE;
        immData = machinePara_.isAicpuModeEn ? htobe32(0x1) : 0x1; // aicpu展开时，HCCL直调RoCE驱动，需进行字节序转换
    }
    HCCL_DEBUG(
        "%s preWrOpcode[%u] useAtomicWrite[%d] wqeType[%d] opcode[0x%x] immdata[%u]", __func__, preWrOpcode,
        useAtomicWrite_, wqeType, opcode, immData);
    preWrOpcode = opcode;
}

u32 TransportIbverbs::GetQpsPerConnection()
{
    u32 externalQps = std::max(static_cast<u32>(machinePara_.srcPorts.size()), 1U);
    s32 qpMode = GetQpMode();
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        && externalQps != HCCL_QPS_PER_CONNECTION_DEFAULT) {
        HCCL_RUN_INFO(
            "HCCL_RDMA_QPS_PER_CONNECTION is set to [%u] but it is not effective in offline mode.", externalQps);
    } else if (qpMode != OPBASE_QP_MODE_EXT && externalQps > 1) {
        HCCL_RUN_INFO(
            "HCCL_RDMA_QPS_PER_CONNECTION is set to [%u] but current devType[%d] does not support multi-QP.",
            externalQps, machinePara_.deviceType);
        return 1; // 非单算子模式仅支持单QP， QPS = 1
    }
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        return externalQps; // only work for opbase mode
    }
    return 1; // 非单算子模式仅支持单QP， QPS = 1
}

HcclResult TransportIbverbs::GetNicHandle()
{
    RaResourceInfo raResourceInfo;
    CHK_RET(NetworkManager::GetInstance(machinePara_.deviceLogicId).GetRaResourceInfo(raResourceInfo));
    std::map<HcclIpAddress, IpSocket>& tmpSocketMap = machinePara_.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE ?
                                                          raResourceInfo.nicSocketMap :
                                                          raResourceInfo.hostNetSocketMap;

    HcclIpAddress localIpAddr = machinePara_.localIpAddr;

    // 获取 nicRdmaHandle
    auto itSocket = tmpSocketMap.find(localIpAddr);
    if (itSocket == tmpSocketMap.end()) {
        HCCL_ERROR(
            "[Get][NicHandle]In get nic handle, can not find socket handle, handle size[%u], "
            "local ip[%s]",
            tmpSocketMap.size(), localIpAddr.GetReadableAddress());
        return HCCL_E_PARA;
    }

    nicRdmaHandle_ = itSocket->second.nicRdmaHandle;
    CHK_PTR_NULL(nicRdmaHandle_);

    return HCCL_SUCCESS;
}

inline static void MultiQpAdjustQpCapacity(struct QpExtAttrs& attrs)
{
    constexpr int multiQpCapacityRatio = 2;
    attrs.qpAttr.cap.max_send_wr /= multiQpCapacityRatio;
    attrs.cqAttr.sendCqDepth /= multiQpCapacityRatio;
}

// 创建一个QP
HcclResult TransportIbverbs::CreateOneQp(
    s32 qpMode, u32 qpsPerConnection, QpHandle& qpHandle, AiQpInfo& aiQpInfo, bool useAicpu, u32 udpSport)
{
    bool isUseQpCreateWithAttrs = false;
    CHK_RET(IsUseQpCreateWithAttrs(isUseQpCreateWithAttrs, qpMode));
    HcclResult ret;
    std::string useAicpuTitle = useAicpu ? std::string("aicpu ") : std::string("");
    std::string qpInfo = useAicpuTitle + std::string("rank:") + std::to_string(machinePara_.localWorldRank)
                         + std::string(",localUserrank:") + std::to_string(machinePara_.localUserrank)
                         + std::string(",localIpAddr: ") + std::string(machinePara_.localIpAddr.GetReadableAddress())
                         + std::string(",deviceLogicId:") + std::to_string(machinePara_.deviceLogicId);
    struct QpExtAttrs attrs {};
    // 判断是否为NORMALQP需要使用qpMode_; hostnic场景的qpmode也是NORMALQP
    if (useAicpu || qpMode_ == QPMode::NORMAL) {
        bool isWorkFlowLib = (workFlowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
        CHK_RET(ConstructQpAttrs(qpMode, attrs, machinePara_.queueDepthAttr, isWorkFlowLib));
        bool isA3Aicpu = machinePara_.isAicpuModeEn && (machinePara_.deviceType == DevType::DEV_TYPE_910_93);
        if (isA3Aicpu && workFlowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
            && qpMode == OPBASE_QP_MODE_EXT) {
            attrs.qpAttr.cap.max_send_wr = machinePara_.queueDepthAttr.sqDepth == INVALID_UINT ?
                                               AICPU_SQ_CQ_DEPTH :
                                               attrs.qpAttr.cap.max_send_wr;
            attrs.cqAttr.sendCqDepth = machinePara_.queueDepthAttr.sendCqDepth == INVALID_UINT ?
                                           AICPU_SQ_CQ_DEPTH :
                                           attrs.cqAttr.sendCqDepth;
        }
        // A3 aicpu图模式使用单个qp, qp深度为socket数量*128
        bool isAicpuLib = machinePara_.isAicpuModeEn && (machinePara_.deviceType == DevType::DEV_TYPE_910_93)
                          && (workFlowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
        if (!UseMultiQp() && isAicpuLib) {
            attrs.qpAttr.cap.max_send_wr = machinePara_.sockets.size() * DEFAULT_OFFLINE_MAX_SEND_WR;
        } else if (UseMultiQp()) {
            MultiQpAdjustQpCapacity(attrs);
        }
        HCCL_DEBUG(
            "qp set max_send_wr %u, socket size %u, isWorkFlowLib %d", attrs.qpAttr.cap.max_send_wr,
            machinePara_.sockets.size(), isWorkFlowLib);

        attrs.udpSport = udpSport;
        ret = hrtRaAiQpCreate(machinePara_.localDeviceId, nicRdmaHandle_, &attrs, &aiQpInfo, qpHandle);
        HCCL_DEBUG("aiQpAddr:%llu db_index:%u, sq_index=%u", aiQpInfo.aiQpAddr, aiQpInfo.dbIndex, aiQpInfo.sqIndex);
        qpInfo = qpInfo + std::string(",sendCqDepth:") + std::to_string(attrs.cqAttr.sendCqDepth);
    } else if (!isUseQpCreateWithAttrs && qpsPerConnection == HCCL_QPS_PER_CONNECTION_DEFAULT) {
        ret = HrtRaQpCreate(nicRdmaHandle_, QP_FLAG_RC, qpMode, qpHandle);
    } else if (!isUseQpCreateWithAttrs && qpsPerConnection != HCCL_QPS_PER_CONNECTION_DEFAULT) {
        HCCL_ERROR("qpsPerConnection[%u] is set but qpMode[%d] is not supported", qpsPerConnection, qpMode);
        return HCCL_E_PARA;
    } else {
        CHK_RET(ConstructQpAttrs(qpMode, attrs, machinePara_.queueDepthAttr));
        if (machinePara_.deviceType == DevType::DEV_TYPE_910_93 && !machinePara_.isAicpuModeEn
            && workFlowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && qpMode == OPBASE_QP_MODE_EXT) {
            attrs.qpAttr.cap.max_send_wr
                = machinePara_.queueDepthAttr.sqDepth == INVALID_UINT ? HOST_SQ_CQ_DEPTH : attrs.qpAttr.cap.max_send_wr;
            attrs.cqAttr.sendCqDepth
                = machinePara_.queueDepthAttr.sendCqDepth == INVALID_UINT ? HOST_SQ_CQ_DEPTH : attrs.cqAttr.sendCqDepth;
        }
        if (UseMultiQp()) {
            MultiQpAdjustQpCapacity(attrs);
        }
        attrs.udpSport = udpSport;
        ret = hrtRaQpCreateWithAttrs(nicRdmaHandle_, &attrs, qpHandle);
        qpInfo = qpInfo + std::string(",sendCqDepth:") + std::to_string(attrs.cqAttr.sendCqDepth);
    }

    RPT_ENV_ERR(
        ret != 0 || (qpHandle == nullptr), "EI0007", vector<string>({"resource_type", "resource_info"}),
        vector<string>({"qp", "CreateOneQp"}));

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]create qp failed, localDeviceId[%d], qpMode[%d]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RESOURCE.c_str(), machinePara_.localDeviceId, qpMode),
        HCCL_E_ROCE_CONNECT);

    // 表示没有通过config配置，则使用环境变量配置
    CHK_RET(SetQpAttrQos(qpHandle, machinePara_.tc, machinePara_.sl));
    // 配置RDMA Timeout时间
    CHK_RET(SetQpAttrTimeOut(qpHandle, machinePara_.retryInterval));
    // 配置RDMA Retry Cnt重传次数
    CHK_RET(SetQpAttrRetryCnt(qpHandle, machinePara_.retryCnt));
    // qpn map 插入
    struct QpAttr attr {};
    CHK_RET(hrtRaGetQpAttr(qpHandle, &attr));

    g_qpn2IbversLinkMap_.Emplace(((static_cast<u64>(machinePara_.localDeviceId) << DEV_PHY_ID_BIT) | attr.qpn), this);

    HCCL_DEBUG("ra qp create success, use input udpSport[%u].", udpSport);
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::CreateSingleQp(s32 qpMode) // 根据socket个数创建QP（下沉模板不够用多QP）
{
    u32 socketNum = 1;
    // A3 aicpu图模式只使用1个qp，qp深度为socketNum*128，最大不超过32K
    bool isAicpuLib = machinePara_.isAicpuModeEn && (machinePara_.deviceType == DevType::DEV_TYPE_910_93)
                      && (workFlowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    if (!UseMultiQp() && !isAicpuLib) {
        socketNum = machinePara_.sockets.size();
    }
    // 原来是 machinePara_.socketFdHandles 换成 machinePara_.sockets
    for (u32 i = 0; i < socketNum; i++) {
        QpHandle qpHandle = nullptr;
        u32 udpSport = machinePara_.srcPorts.empty() ? 0 : machinePara_.srcPorts[0];
        CHK_RET(CreateOneQp(
            qpMode, HCCL_QPS_PER_CONNECTION_DEFAULT, qpHandle, combineAiQpInfo_.aiQpInfo, machinePara_.isAicpuModeEn,
            udpSport));
        CombineQpHandle tmpCombineQpHandle;
        tmpCombineQpHandle.qpHandle = qpHandle;
        combineQpHandles_.push_back(tmpCombineQpHandle);
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::CreateMultiQp(s32 qpMode, u32 qpsPerConnection)
{
    // 配置了多qp源端口号时的处理流程
    if (machinePara_.srcPorts.size() > 0) {
        HCCL_DEBUG("[TransportIbverbs][CreateMultiQp]use Multi qp create qps.");
        // 创建qp
        for (const auto& port : machinePara_.srcPorts) {
            QpHandle qpHandle = nullptr;
            AiQpInfo tmpAiQpInfo{};
            CHK_RET(CreateOneQp(qpMode, qpsPerConnection, qpHandle, tmpAiQpInfo, machinePara_.isAicpuModeEn, port));
            multiCombineQpHandles_.push_back(CombineQpHandle(qpHandle));
            combineAiQpInfos_.push_back(CombineQpInfo(tmpAiQpInfo));
        }
    }
    HCCL_DEBUG("ra multi-qp creation success.");
    return HCCL_SUCCESS;
}

s32 TransportIbverbs::GetQpMode()
{
    s32 qpMode = NORMAL_QP_MODE;

    if (qpMode_ == QPMode::NORMAL) {
        return qpMode;
    }

    if (machinePara_.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        if (workFlowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            qpMode = (machinePara_.deviceType == DevType::DEV_TYPE_910B
                      || machinePara_.deviceType == DevType::DEV_TYPE_910_93) ?
                         OPBASE_QP_MODE_EXT :
                         OPBASE_QP_MODE;
            // isCapture需要创建下沉QP
            qpMode = (qpMode == OPBASE_QP_MODE_EXT && qpMode_ == QPMode::OFFLOAD) ? OFFLINE_QP_MODE_EXT : qpMode;
            isCapture_ = (qpMode == OFFLINE_QP_MODE_EXT) ? true : false;
        } else {
            qpMode = (machinePara_.deviceType == DevType::DEV_TYPE_910B
                      || machinePara_.deviceType == DevType::DEV_TYPE_910_93) ?
                         OFFLINE_QP_MODE_EXT :
                         OFFLINE_QP_MODE;
        }
    }
    if (machinePara_.isAicpuModeEn) {
        qpMode = (machinePara_.deviceType == DevType::DEV_TYPE_910B
                  || machinePara_.deviceType == DevType::DEV_TYPE_910_93) ?
                     OPBASE_QP_MODE_EXT :
                     OPBASE_QP_MODE;
    }
    return qpMode;
}

bool TransportIbverbs::UseMultiQp()
{
    s32 qpMode = GetQpMode();
    HCCL_DEBUG("[TransportIbverbs]UseMultiQp qpsPerConnection[%u]", qpsPerConnection_);
    if (qpMode == OPBASE_QP_MODE_EXT && qpsPerConnection_ > 1) {
        return true;
    }
    return false;
}

HcclResult TransportIbverbs::CreateQp()
{
    s32 qpMode = GetQpMode();
    HCCL_DEBUG("[TransportIbverbs][CreateQp] QpMode[%u]", qpMode);
    CHK_RET(CreateSingleQp(qpMode));
    if (UseMultiQp()) {
        HCCL_DEBUG("[TransportIbverbs]Create MultiQP begin");
        CHK_RET(CreateMultiQp(qpMode, qpsPerConnection_));
    }
    HCCL_DEBUG("ra qp create %u qp success.", combineQpHandles_.size());
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::InitQpConnect()
{
    /* 创建QP操作句柄 */
    qpsPerConnection_ = GetQpsPerConnection();

    CHK_RET(ExchangeCapabilityHybrid());

    CHK_RET(CreateQp());

    CHK_RET(FillExchangeDataTotalSize());

    // 注册notify 内存信息
    CHK_RET(CreateNotifyValueBuffer());

    CHK_RET(ConstructExchangeForSend());
    HCCL_DEBUG("[TransportIbverbs] resource create done exchangeDataTotalSize_[%llu]", exchangeDataTotalSize_);

    HcclResult ret = defaultSocket_->Send(exchangeDataForSend_.data(), exchangeDataTotalSize_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TransportIbverbs][InitQpConnect] failed to send exchangeData exchangeDataTotalSize[%llu], "
            "custom exchange data size [%llu].",
            exchangeDataTotalSize_, machinePara_.exchangeInfo.size()),
        ret);
    HCCL_DEBUG("[TransportIbverbs]Socket Send finished, exchangeDataTotalSize[%llu]", exchangeDataTotalSize_);

    exchangeDataForRecv_.resize(exchangeDataTotalSize_);
    ret = defaultSocket_->Recv(exchangeDataForRecv_.data(), exchangeDataTotalSize_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TransportIbverbs][InitQpConnect] failed to recv exchangeData exchangeDataTotalSize[%llu], "
            "custom exchange data size [%llu].",
            exchangeDataTotalSize_, machinePara_.exchangeInfo.size()),
        ret);

    HCCL_DEBUG("[TransportIbverbs][Init] Socket Data Recved");

    CHK_RET(ParseReceivedExchangeData());

    // 连接Qp
    CHK_RET(ConnectQp());
    HCCL_INFO("In link ibv, qp status has ready");
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::ConnectSingleQp(std::function<bool()> needStop)
{
    // QP建链
    for (u32 i = 0; i < combineQpHandles_.size(); i++) {
        CHK_RET(HrtRaQpConnectAsync(combineQpHandles_[i].qpHandle, machinePara_.sockets[i]->GetFdHandle(), needStop));
    }
    // 查询QP建链是否成功
    s32 qpStatus = 0;
    s32 raRet = 0;
    auto startTime = std::chrono::steady_clock::now();
    HCCL_INFO("In link ibv, waiting for qp status ready...");
    for (u32 i = 0; i < combineQpHandles_.size(); i++) {
        while (true) {
            CHK_PRT_RET(needStop(), HCCL_ERROR("Terminating operation due to external request"), HCCL_E_INTERNAL);

            if ((std::chrono::steady_clock::now() - startTime) >= timeout_) {
                HCCL_ERROR("[Connect][Qp]get qp status timeout_=[%lld ms], qp_status=[%d]", timeout_, qpStatus);
                return HCCL_E_TIMEOUT;
            }
            raRet = hrtGetRaQpStatus(combineQpHandles_[i].qpHandle, &qpStatus);
            if ((raRet == 0) && (qpStatus == 1)) { // 为1时，qp 建链成功
                HCCL_INFO("In link ibv, %u of %u QP get status success.", (i + 1), combineQpHandles_.size());
                break;
            } else {
                // qp建链需要时间，获取qp状态直至超时
                SaluSleep(WAIT_US_COUNT);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::ConnectMultiQp(u32 qpsPerConnection, std::function<bool()> needStop)
{
    // 多QP下，复用同一个socket handle来modify QP, 此时需要串行创建
    if (machinePara_.sockets.size() != 2) { // 2：多QP下需要一个额外的Socket来做QP状态迁移同步
        return HCCL_E_INTERNAL;
    }
    for (u32 i = 0; i < qpsPerConnection; i++) {
        // QP建链
        u8 localQpConnectReady = 1;
        u8 remoteQpConnectReady = 0;
        CHK_RET(machinePara_.sockets[1]->Send(&localQpConnectReady, 1));
        CHK_RET(machinePara_.sockets[1]->Recv(&remoteQpConnectReady, 1));
        std::string aicpu = machinePara_.isAicpuModeEn ? "aicpu" : "";
        CHK_PRT_RET(
            remoteQpConnectReady != localQpConnectReady,
            HCCL_ERROR(
                "[TransportIbverbs] %s multi Qp Connected checking failed! %u of %u QP", aicpu.c_str(), (i + 1),
                qpsPerConnection),
            HCCL_E_NETWORK);
        CHK_RET(
            HrtRaQpConnectAsync(multiCombineQpHandles_[i].qpHandle, machinePara_.sockets[0]->GetFdHandle(), needStop));
        // 查询QP建链是否成功
        s32 qpStatus = 0;
        s32 raRet = 0;
        auto startTime = std::chrono::steady_clock::now();
        HCCL_INFO(
            "In link ibv, waiting for qp status ready... %u of %u %s QP", (i + 1), multiCombineQpHandles_.size(),
            aicpu.c_str());
        while (true) {
            if ((std::chrono::steady_clock::now() - startTime) >= timeout_) {
                HCCL_ERROR(
                    "[Connect][Qp]get qp status timeout_=[%lld ms], qp_status=[%d], index[%u]", timeout_, qpStatus, i);
                return HCCL_E_TIMEOUT;
            }
            raRet = hrtGetRaQpStatus(multiCombineQpHandles_[i].qpHandle, &qpStatus);
            if ((raRet == 0) && (qpStatus == 1)) { // 为1时，qp 建链成功
                HCCL_INFO(
                    "In link ibv, %u of %u %s QP get status success.", (i + 1), multiCombineQpHandles_.size(),
                    aicpu.c_str());
                break;
            } else {
                // qp建链需要时间，获取qp状态直至超时
                SaluSleep(WAIT_US_COUNT);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::ConnectQp()
{
    CHK_RET(ConnectSingleQp([this]() -> bool {
        return this->GetStopFlag();
    }));
    if (UseMultiQp()) {
        CHK_RET(ConnectMultiQp(qpsPerConnection_, [this]() -> bool {
            return this->GetStopFlag();
        }));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::Fence()
{
    fence_ = true;
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::AddWqeList(
    void* dstMemPtr, const void* srcMemPtr, u64 srcMemSize, WqeType wqeType, WrAuxInfo& aux,
    std::vector<WqeInfo>& wqeInfoVec)
{
    WqeInfo wqeInfoTmp;

    wqeInfoTmp.wqeData.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(dstMemPtr));
    wqeInfoTmp.wqeData.sendFlags = fence_ ? (RA_SEND_SIGNALED | RA_SEND_FENCE) : RA_SEND_SIGNALED;
    fence_ = false;
    wqeInfoTmp.wqeData.memList.addr = static_cast<u64>(reinterpret_cast<uintptr_t>(srcMemPtr));
    wqeInfoTmp.wqeData.memList.len = srcMemSize;
    wqeInfoTmp.wqeData.memList.lkey = 0;

    switch (wqeType) {
        case WqeType::WQE_TYPE_DATA:
        case WqeType::WQE_TYPE_DATA_NOTIFY:
        case WqeType::WQE_TYPE_ACK_NOTIFY:
        case WqeType::WQE_TYPE_DATA_ACK_NOTIFY:
        case WqeType::WQE_TYPE_DATA_WITH_NOTIFY:
            wqeInfoTmp.wqeData.op = RA_WR_RDMA_WRITE;
            wqeInfoTmp.wqeType = static_cast<u64>(wqeType);
            break;
        case WqeType::WQE_TYPE_DATA_WITH_REDUCE:
            wqeInfoTmp.wqeData.op = RA_WR_RDMA_REDUCE_WRITE;
            wqeInfoTmp.wqeData.aux = aux;
            // REDUCE WRITE 作为特殊的DATA
            wqeInfoTmp.wqeType = static_cast<u64>(WqeType::WQE_TYPE_DATA);
            break;
        case WqeType::WQE_TYPE_READ_DATA:
            wqeInfoTmp.wqeData.op = RA_WR_RDMA_READ;
            wqeInfoTmp.wqeType = static_cast<u64>(wqeType);
            break;
        default:
            HCCL_ERROR("error wqeType[%d]", wqeType);
            return HCCL_E_INTERNAL;
    }
    CHK_RET(GetWqeDataOffsetAndNotifyId(wqeType, wqeInfoTmp.wqeDataOffset, wqeInfoTmp.notifyId));

    wqeInfoVec.push_back(wqeInfoTmp);
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::ConstructPayLoadWqe(
    void* dstMemPtr, const void* src, u64 len, WqeType wqeType, WrAuxInfo& aux, std::vector<WqeInfo>& wqeInfoVec,
    u32 txSendDataTimes)
{
    HcclResult ret;
    // 发送数据Wqe
    for (u32 txSendDataIdx = 0; txSendDataIdx < txSendDataTimes; txSendDataIdx++) {
        u64 txSendDataOffset = txSendDataIdx * RDMA_SEND_MAX_SIZE;
        u64 txSendDataSize = (txSendDataIdx == (txSendDataTimes - 1)) ? len - txSendDataOffset : RDMA_SEND_MAX_SIZE;

        void* txdstMemPtr = reinterpret_cast<void*>(reinterpret_cast<char*>(dstMemPtr) + txSendDataOffset);

        const void* txsrcMemPtr = reinterpret_cast<const void*>(reinterpret_cast<const char*>(src) + txSendDataOffset);
        ret = AddWqeList(txdstMemPtr, txsrcMemPtr, txSendDataSize, wqeType, aux, wqeInfoVec);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[TransportIbverbs][TxAsync]errNo[0x%016llx] In lbv exp, add wqe list failed."
                "srcMemSize[%llu Byte]",
                HCCL_ERROR_CODE(ret), txSendDataSize),
            ret);
    }

    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::TxPayLoad(
    UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, WqeType wqeType, WrAuxInfo& aux,
    std::vector<WqeInfo>& wqeInfoVec)
{
    void* dstMemPtr = nullptr;
    u64 dstMemSize = 0;
    // 为保证单算子下不同数据量下子图的结构相同，zero byte message 时也需要下发task
    u32 txSendDataTimes = (len == 0) ? 1 : (len + RDMA_SEND_MAX_SIZE - 1) / RDMA_SEND_MAX_SIZE;
    CHK_RET(GetMemInfo(dstMemType, &dstMemPtr, &dstMemSize));

    if (dstOffset > dstMemSize) {
        HCCL_ERROR(
            "[TransportIbverbs][TxAsync]dst_mem_type=%d, dst_mem_ptr=%p, dst_offset=%llu, dst_mem_size=%llu Byte",
            dstMemType, dstMemPtr, dstOffset, dstMemSize);
        return HCCL_E_INTERNAL;
    }

    dstMemPtr = reinterpret_cast<void*>(reinterpret_cast<char*>(dstMemPtr) + dstOffset);
    CHK_RET(ConstructPayLoadWqe(dstMemPtr, src, len, wqeType, aux, wqeInfoVec, txSendDataTimes));

    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::TxAsync(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream)
{
    std::vector<WqeInfo> wqeInfoVec;
    wqeInfoVec.reserve(WQE_RESERVE_LENGTH);
    struct WrAuxInfo aux = {};
    HCCL_DEBUG("TX src[%p] len[%llu] dstOffset[%llu]", src, len, dstOffset);

    if (src != nullptr) {
        CHK_RET(TxPayLoad(dstMemType, dstOffset, src, len, WqeType::WQE_TYPE_DATA, aux, wqeInfoVec));
    }

    CHK_RET(TxSendDataAndNotify(wqeInfoVec, stream, GetUseOneDoorbellValue()));
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::TxWithReduce(
    UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, const HcclDataType datatype, HcclReduceOp redOp,
    Stream& stream)
{
    std::vector<WqeInfo> wqeInfoVec;
    wqeInfoVec.reserve(WQE_RESERVE_LENGTH);
    struct WrAuxInfo aux = {};
    aux.dataType = RDMA_REDUCE_DATA_TYPE_TABLE[datatype];
    aux.reduceType = RDMA_REDUCE_OP_TYPE_TABLE[redOp];
    if (aux.dataType == static_cast<uint8_t>(RdmaReduceDataType::RDMA_REDUCE_DATA_INVALID)
        || aux.reduceType == static_cast<uint8_t>(RdmaReduceOpType::RDMA_REDUCE_OP_INVALID)) {
        HCCL_ERROR(
            "unsupported data type [%s] or Reduce type [%s]", GetDataTypeEnumStr(datatype).c_str(),
            GetReduceOpEnumStr(redOp).c_str());
        return HCCL_E_INTERNAL;
    }

    CHK_PTR_NULL(src);
    CHK_RET(TxPayLoad(dstMemType, dstOffset, src, len, WqeType::WQE_TYPE_DATA_WITH_REDUCE, aux, wqeInfoVec));

    CHK_RET(TxSendDataAndNotify(wqeInfoVec, stream, GetUseOneDoorbellValue()));
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::TxWithReduce(
    const std::vector<TxMemoryInfo>& txWithReduceMems, const HcclDataType datatype, HcclReduceOp redOp, Stream& stream)
{
    std::vector<WqeInfo> wqeInfoVec;
    wqeInfoVec.reserve(WQE_RESERVE_LENGTH);
    struct WrAuxInfo aux = {};
    aux.dataType = RDMA_REDUCE_DATA_TYPE_TABLE[datatype];
    aux.reduceType = RDMA_REDUCE_OP_TYPE_TABLE[redOp];
    if (aux.dataType == static_cast<uint8_t>(RdmaReduceDataType::RDMA_REDUCE_DATA_INVALID)
        || aux.reduceType == static_cast<uint8_t>(RdmaReduceOpType::RDMA_REDUCE_OP_INVALID)) {
        HCCL_ERROR(
            "unsupported data type [%s] or Reduce type [%s]", GetDataTypeEnumStr(datatype).c_str(),
            GetReduceOpEnumStr(redOp).c_str());
        return HCCL_E_INTERNAL;
    }

    for (const TxMemoryInfo& txWithReduceMem : txWithReduceMems) {
        CHK_RET(TxPayLoad(
            txWithReduceMem.dstMemType, txWithReduceMem.dstOffset, txWithReduceMem.src, txWithReduceMem.len,
            WqeType::WQE_TYPE_DATA_WITH_REDUCE, aux, wqeInfoVec));
    }

    CHK_RET(TxSendDataAndNotify(wqeInfoVec, stream, GetUseOneDoorbellValue()));
    return HCCL_SUCCESS;
}

bool TransportIbverbs::IsSupportTransportWithReduce()
{
    if (machinePara_.deviceType == DevType::DEV_TYPE_910B || machinePara_.deviceType == DevType::DEV_TYPE_910_93) {
        return true;
    } else {
        return false;
    }
}

// 910A1 不支持write with notify;
// 910A2 由于PCIE through 和 write with notify 冲突，所以不支持write with Notify;
bool TransportIbverbs::IsSupportRdmaNotify() { return false; }

bool TransportIbverbs::IsTemplateMode()
{
    if (workFlowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        || machinePara_.deviceType == DevType::DEV_TYPE_910B || machinePara_.deviceType == DevType::DEV_TYPE_910_93) {
        return false;
    } else {
        return true;
    }
}

HcclResult TransportIbverbs::GetIndOpRemoteMemDetails(MemDetails** remoteMem, uint32_t* memNum, HcclMemType memType)
{
    CHK_PRT_RET(remoteMem == nullptr, HCCL_ERROR("[%s] remoteMem is nullptr", __func__), HCCL_E_PARA);
    CHK_PRT_RET(memNum == nullptr, HCCL_ERROR("[%s] memNum is nullptr", __func__), HCCL_E_PARA);

    *remoteMem = nullptr;
    *memNum = 0;
    uint32_t memCount;
    if (memType == HcclMemType::HCCL_MEM_TYPE_DEVICE) {
        memCount = remoteUserDeviceMemMsg_.size();
    } else if (memType == HcclMemType::HCCL_MEM_TYPE_HOST) {
        memCount = remoteUserHostMemMsg_.size();
    } else {
        HCCL_ERROR("[%s] not support memType[%d]", __func__, memType);
        return HCCL_E_INTERNAL;
    }
    if (memCount == 0) {
        HCCL_DEBUG("[%s] No remote memory regions available", __func__);
        return HCCL_SUCCESS;
    }
    // 外部需要手动释放内存
    MemDetails* remoteMemDetails = static_cast<MemDetails*>(malloc(memCount * sizeof(MemDetails)));
    CHK_PTR_NULL(remoteMemDetails);
    uint32_t index = 0;
    if (memType == HcclMemType::HCCL_MEM_TYPE_DEVICE) {
        for (const auto& msg : remoteUserDeviceMemMsg_) {
            remoteMemDetails[index].addr = reinterpret_cast<u64>(msg.addr);
            remoteMemDetails[index].size = msg.len;
            remoteMemDetails[index].key = msg.lkey;
            index++;
        }
    } else if (memType == HcclMemType::HCCL_MEM_TYPE_HOST) {
        for (const auto& msg : remoteUserHostMemMsg_) {
            remoteMemDetails[index].addr = reinterpret_cast<u64>(msg.addr);
            remoteMemDetails[index].size = msg.len;
            remoteMemDetails[index].key = msg.lkey;
            index++;
        }
    }
    *memNum = memCount;
    *remoteMem = remoteMemDetails;

    HCCL_DEBUG("[%s] Successfully returned %u remote memory regions", __func__, index);
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::GetIndOpRemoteMem(HcclMem** remoteMem, uint32_t* memNum)
{
    CHK_PRT_RET(remoteMem == nullptr, HCCL_ERROR("[GetIndOpRemoteMem] remoteMem is nullptr"), HCCL_E_PARA);
    CHK_PRT_RET(memNum == nullptr, HCCL_ERROR("[GetIndOpRemoteMem] memNum is nullptr"), HCCL_E_PARA);

    *remoteMem = nullptr;
    *memNum = 0;

    std::lock_guard<std::mutex> lock(remoteMemsMutex_);

    if (!remoteMemsPtr_) {
        uint32_t totalCount = remoteUserDeviceMemMsg_.size() + remoteUserHostMemMsg_.size();
        if (totalCount == 0) {
            HCCL_INFO("[GetIndOpRemoteMem] No remote memory regions available");
            return HCCL_SUCCESS;
        }
        remoteMemsPtr_ = std::make_unique<HcclMem[]>(totalCount);
        CHK_PTR_NULL(remoteMemsPtr_);
        uint32_t index = 0;
        for (const auto& msg : remoteUserDeviceMemMsg_) {
            remoteMemsPtr_[index].type = HcclMemType::HCCL_MEM_TYPE_DEVICE;
            remoteMemsPtr_[index].addr = msg.addr;
            remoteMemsPtr_[index].size = msg.len;
            index++;
        }
        for (const auto& msg : remoteUserHostMemMsg_) {
            remoteMemsPtr_[index].type = HcclMemType::HCCL_MEM_TYPE_HOST;
            remoteMemsPtr_[index].addr = msg.addr;
            remoteMemsPtr_[index].size = msg.len;
            index++;
        }
        remoteMemsNum_ = totalCount;
    }

    *memNum = remoteMemsNum_;
    *remoteMem = remoteMemsPtr_.get();

    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::GetRemoteMem(UserMemType memType, void** remotePtr)
{
    switch (memType) {
        case UserMemType::INPUT_MEM:
        case UserMemType::OUTPUT_MEM:
            *remotePtr = remoteMemMsg_[static_cast<u32>(memType)].addr;
            break;

        default:
            HCCL_ERROR("[Get][RemoteMem]not support dst_mem_type=%d", memType);
            return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::GetRemoteMemSize(UserMemType memType, u64& size)
{
    switch (memType) {
        case UserMemType::INPUT_MEM:
        case UserMemType::OUTPUT_MEM:
            size = remoteMemMsg_[static_cast<u32>(memType)].len;
            break;

        default:
            HCCL_ERROR("[Get][RemoteMem]not support dst_mem_type=%d", memType);
            return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult
TransportIbverbs::TxSendDataAndNotifyWithSingleQP(std::vector<WqeInfo>& wqeInfoVec, Stream& stream, bool useOneDoorbell)
{
    if (IsSupportRdmaNotify() && wqeInfoVec.size() > 0) {
        // 支持RDMA NOTIFY时, 修改wqeInfoVec中最后一个wqe, 使其附带Notify信息
        u32 offset = static_cast<u32>(remoteDataNotifyMsg_.offset);
        if (wqeInfoVec.back().wqeData.op == RA_WR_RDMA_REDUCE_WRITE) {
            wqeInfoVec.back().wqeData.op = RA_WR_RDMA_REDUCE_WRITE_WITH_NOTIFY;
        } else {
            wqeInfoVec.back().wqeData.op = RA_WR_RDMA_WRITE_WITH_NOTIFY;
        }
        wqeInfoVec.back().wqeData.aux.notifyOffset = offset;
    } else {
        // 发送data notify同步信息
        struct WrAuxInfo aux = {};
        void* remoteNotifyaddr = remoteDataNotifyMsg_.addr;
        CHK_RET(AddWqeList(
            remoteNotifyaddr, notifyValueMem_[machinePara_.deviceLogicId].ptr(), notifySize_,
            WqeType::WQE_TYPE_DATA_NOTIFY, aux, wqeInfoVec));
    }

    if (machinePara_.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        CHK_RET(RdmaSendAsync(wqeInfoVec, stream, useOneDoorbell));
    } else {
        CHK_RET(RdmaSendAsyncHostNIC(wqeInfoVec, stream));
    }
    return HCCL_SUCCESS;
}

u32 TransportIbverbs::GetActualQpNum(u32 maxLength)
{
    u32 actualMultiQpNum = 1;
    const u32 KByteToByte = 1024; // 1024 多QP阈值单位是KB
    if (maxLength / qpsPerConnection_ >= GetExternalInputMultiQpThreshold() * KByteToByte) {
        actualMultiQpNum = qpsPerConnection_;
    } else {
        u32 quotient = maxLength / (GetExternalInputMultiQpThreshold() * KByteToByte);
        u32 remainder = maxLength % (GetExternalInputMultiQpThreshold() * KByteToByte);
        actualMultiQpNum = quotient + (remainder != 0 ? 1 : 0);
    }

    return actualMultiQpNum;
}

HcclResult TransportIbverbs::TxSendDataAndNotify(std::vector<WqeInfo>& wqeInfoVec, Stream& stream, bool useOneDoorbell)
{
    u32 maxLength = 0;
    for (u32 i = 0; i < wqeInfoVec.size(); i++) {
        if (wqeInfoVec[i].wqeData.memList.len > maxLength) {
            maxLength = wqeInfoVec[i].wqeData.memList.len;
        }
    }

    u32 actualMultiQpNum = GetActualQpNum(maxLength);

    HCCL_DEBUG(
        "[TransportIbverbs][TxSendDataAndNotify] UseMultiQp[%d] MultiQpNum[%u] actualMultiQpNum[%u] maxLength[%u]",
        UseMultiQp(), qpsPerConnection_, actualMultiQpNum, maxLength);
    if (UseMultiQp() && actualMultiQpNum != 1 && actualMultiQpNum <= qpsPerConnection_ && maxLength != 0) {
        CHK_RET(TxSendDataAndNotifyWithMultiQP(wqeInfoVec, actualMultiQpNum, stream, useOneDoorbell));
    } else {
        CHK_RET(TxSendDataAndNotifyWithSingleQP(wqeInfoVec, stream, useOneDoorbell));
    }
    return HCCL_SUCCESS;
}

std::vector<u32> TransportIbverbs::RdmaLengthSplit(u32 length, u32 splitNum)
{
    // step 1, 先计算有多少个128 Byte
    u32 alignNum = length / RDMA_ADDR_ALIGNMENT;
    u32 tailBytes = length % RDMA_ADDR_ALIGNMENT; // 尾块简单处理，放在最后一个切分出来的块后面
    // step 2, 将这128 Byte再分成 splitNum 分，每一份有多少个 128Byte
    u32 alignNumPerSplit = alignNum / splitNum;
    u32 tailAlignNum = alignNum % splitNum; // 尾块简单处理，放在最后一个切分出来的块后面
    std::vector<u32> vctSplittedLength(splitNum, 0);
    for (u32 i = 0; i < splitNum; i++) {
        u32 lengthTmp = alignNumPerSplit * RDMA_ADDR_ALIGNMENT;
        vctSplittedLength[i] = lengthTmp;
    }
    vctSplittedLength[splitNum - 1] += tailAlignNum * RDMA_ADDR_ALIGNMENT + tailBytes;
    return vctSplittedLength;
}

HcclResult TransportIbverbs::TxSendDataAndNotifyWithMultiQP(
    std::vector<WqeInfo>& wqeInfoVec, u32 actualMultiQpNum, Stream& stream, [[maybe_unused]] bool useOneDoorbell)
{
    // vector<WqeInfo> 是一个vector的原因是 单个wqe只能发2GB数据，如果超过2GB，就拆分到多个WqeInfo中了
    // 多QP下，对每个WqeInfo都进行多QP切分，然后在收发每一个QP的数据
    std::vector<std::vector<WqeInfo>> multiQpWqeInfoVct(actualMultiQpNum, wqeInfoVec);
    for (u32 i = 0; i < wqeInfoVec.size(); i++) {
        WqeInfo tmpWqeInfo = wqeInfoVec[i];
        u32 curLen = tmpWqeInfo.wqeData.memList.len;
        std::vector<u32> splittedLen = RdmaLengthSplit(curLen, actualMultiQpNum);
        uint64_t curSrcAddr = tmpWqeInfo.wqeData.memList.addr;
        uint64_t curDstAddr = tmpWqeInfo.wqeData.dstAddr;
        for (u32 qpIndex = 0; qpIndex < actualMultiQpNum; qpIndex++) {
            multiQpWqeInfoVct[qpIndex][i].wqeData.memList.len = splittedLen[qpIndex];
            multiQpWqeInfoVct[qpIndex][i].wqeData.memList.addr = curSrcAddr;
            multiQpWqeInfoVct[qpIndex][i].wqeData.dstAddr = curDstAddr;
            curSrcAddr += splittedLen[qpIndex];
            curDstAddr += splittedLen[qpIndex];
        }
    }
    // 给每个QP最后增加一个属于该QP的DataNotify
    for (u32 qpIndex = 0; qpIndex < actualMultiQpNum; qpIndex++) {
        struct WrAuxInfo aux = {};
        void* remoteNotifyaddr = multiQpDataNotifyRemoteMemMsg_[qpIndex].addr;
        CHK_RET(AddWqeList(
            remoteNotifyaddr, notifyValueMem_[machinePara_.deviceLogicId].ptr(), notifySize_,
            WqeType::WQE_TYPE_DATA_NOTIFY, aux, multiQpWqeInfoVct[qpIndex]));
    }
    // useOneDoorbell 配置成true。最后一个payload去按doorbell
    for (u32 qpIndex = 0; qpIndex < actualMultiQpNum; qpIndex++) {
        CHK_RET(
            RdmaSendAsync(multiQpWqeInfoVct[qpIndex], stream, true, qpIndex)); // 多QP使用同一个stream异步doorbell触发
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::TxAsync(std::vector<TxMemoryInfo>& txMems, Stream& stream)
{
    std::vector<WqeInfo> wqeInfoVec;
    wqeInfoVec.reserve(WQE_RESERVE_LENGTH);
    struct WrAuxInfo aux = {};

    for (auto& mem : txMems) {
        HCCL_DEBUG("TX src[%p] len[%llu] dstOffset[%llu]", mem.src, mem.len, mem.dstOffset);
        CHK_PTR_NULL(mem.src);
        CHK_RET(TxPayLoad(mem.dstMemType, mem.dstOffset, mem.src, mem.len, WqeType::WQE_TYPE_DATA, aux, wqeInfoVec));
    }

    CHK_RET(TxSendDataAndNotify(wqeInfoVec, stream, GetUseOneDoorbellValue()));
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::TxWqeList(
    std::vector<WqeInfo>& wqeInfoVec, Stream& stream, std::vector<struct SendWrRsp>& opRspVec, u32 multiQpIndex)
{
    (void)stream;
    if (!IsTemplateMode()) {
        currentQP_ = 0;
    } else {
        if (sqeCounter_ < (HCCP_SQ_TEMPLATE_CAPACITY + 1)
            && (sqeCounter_ + wqeInfoVec.size()) >= (HCCP_SQ_TEMPLATE_CAPACITY + 1)) {
            currentQP_++;
            sqeCounter_ = wqeInfoVec.size();
        } else {
            sqeCounter_ += wqeInfoVec.size();
        }
    }
    CHK_PRT_RET(
        currentQP_ >= combineQpHandles_.size(),
        HCCL_ERROR(
            "[TransportIbverbs][TxWqeList]errNo[0x%016llx] In lbv "
            "exp, qp idx[%u] is invalid.",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), currentQP_),
        HCCL_E_INTERNAL);

    HCCL_DEBUG("rdma tx send wqes: ra qp sqe counter:%u, current qp idx:%u", sqeCounter_, currentQP_);

    std::vector<SendWrlistDataExt> wqelisDatatVec;
    for (u32 index = 0; index < wqeInfoVec.size(); index++) {
        // 使能atomic write场景下，reduce的下一个notify的opcode要设置为atomic write
        u32& preWrOpcode = multiQpIndex == RDMA_INVALID_QP_INDEX ? combineQpHandles_[currentQP_].preWrOpcode :
                                                                   multiCombineQpHandles_[multiQpIndex].preWrOpcode;
        ModifyAtomicWriteAfterReduce(
            preWrOpcode, wqeInfoVec[index].wqeType, wqeInfoVec[index].wqeData.op,
            wqeInfoVec[index].wqeData.ext.immData);

        wqelisDatatVec.push_back(wqeInfoVec[index].wqeData);
    }

    u32 totalWqeCount = wqelisDatatVec.size();
    struct SendWrlistDataExt* wqelist = wqelisDatatVec.data();
    struct SendWrRsp* opRsp = opRspVec.data();

    // HCCP会校验 zero byte messages 的内存地址是否已注册MR。对于 zero byte messages 不下发WR，将opRsp设置为特殊值。
    // 下发rdmasend task时检查该特殊值，如果zero byte message则不下发rdmasend task。
    bool batchSendWr = true;
    for (u32 i = 0; i < totalWqeCount; i++) {
        if (wqelisDatatVec[i].memList.len == 0) {
            batchSendWr = false;
            break;
        }
    }
    QpHandle currentQp;
    if (multiQpIndex == RDMA_INVALID_QP_INDEX) {
        currentQp = combineQpHandles_[currentQP_].qpHandle;
    } else {
        currentQp = multiCombineQpHandles_[multiQpIndex].qpHandle;
    }
    if (batchSendWr) {
        CHK_RET(SendWqeList(currentQp, totalWqeCount, wqelist, opRsp));
    } else {
        for (u32 i = 0; i < totalWqeCount; i++) {
            if (wqelisDatatVec[i].memList.len > 0) {
                CHK_RET(SendWqeList(currentQp, 1U, &wqelist[i], &opRsp[i]));
            } else {
                opRsp[i].wqeTmp.sqIndex = INVALID_UINT;
                opRsp[i].wqeTmp.wqeIndex = INVALID_UINT;
                opRsp[i].db.dbIndex = INVALID_UINT;
                opRsp[i].db.dbInfo = INVALID_U64;
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult
TransportIbverbs::SendWqeList(QpHandle qpHandle, u32 wqeNum, struct SendWrlistDataExt* wqelist, struct SendWrRsp* opRsp)
{
    unsigned int completeNum = 0;
    HcclResult ret = HrtRaSendWrlistExt(qpHandle, wqelist, opRsp, wqeNum, &completeNum);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[TransportIbverbs][TxWqeList]In ibv send wq list, HrtRaSendWrlist failed.ret[%d]", ret),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult
TransportIbverbs::RdmaSendAsync(std::vector<WqeInfo>& wqeInfoVec, Stream& stream, bool useOneDoorbell, u32 multiQpIndex)
{
    HcclResult ret;

    std::vector<struct SendWrRsp> opRspVec(wqeInfoVec.size());
    CHK_RET(TxWqeList(wqeInfoVec, stream, opRspVec, multiQpIndex));

    std::vector<SendWrlistDataExt> wqelistVec;
    for (u32 index = 0; index < wqeInfoVec.size(); index++) {
        wqelistVec.push_back(wqeInfoVec[index].wqeData);
    }

    struct SendWr wr = {};
    wr.bufNum = 1;
    wr.op = 0;
    wr.sendFlag = fence_ ? (RA_SEND_SIGNALED | RA_SEND_FENCE) : RA_SEND_SIGNALED;
    fence_ = false;

    if (useOneDoorbell && !IsTemplateMode()) {
        // 内存块不连续时，只敲最后一次doorbell
        wr.bufList = &wqelistVec.back().memList;
        wr.dstAddr = static_cast<u64>(wqelistVec.back().dstAddr);
        // 只敲一次doorbell时，len为所有非连续内存块长度总和+notify(4Bytes)
        wr.bufList[0].len = std::accumulate(wqelistVec.begin(), wqelistVec.end(), 0llu, [](u64 acc, auto wqelist) {
            return acc + wqelist.memList.len;
        });

        const u32 dbIndex = static_cast<u32>(opRspVec.back().db.dbIndex);
        const u64 dbInfo = static_cast<u64>(opRspVec.back().db.dbInfo);
        ret = dispatcher_->RdmaSend(dbIndex, dbInfo, wr, stream, machinePara_.remoteWorldRank, isCapture_);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[TransportIbverbs][RdmaSendAsync][useOneDoorbell]errNo[0x%016llx] In lbv exp op base mode, "
                "rdma send failed. dbIndex[%u] dbInfo[%llu]",
                HCCL_ERROR_CODE(ret), dbIndex, dbInfo),
            ret);
        HCCL_INFO("[TransportIbverbs][RdmaSendAsync][useOneDoorbell] db_index[%u], db_info[%llu]", dbIndex, dbInfo);
        return HCCL_SUCCESS;
    }

    for (u32 i = 0; i < wqeInfoVec.size(); i++) {
        wr.bufList = &wqelistVec[i].memList;
        wr.dstAddr = static_cast<u64>(wqelistVec[i].dstAddr);

        if (!IsTemplateMode()) {
            u32 dbIndex = static_cast<u32>(opRspVec[i].db.dbIndex);
            u64 dbInfo = static_cast<u64>(opRspVec[i].db.dbInfo);

            // op base 模式下的发送接口
            if (wqeInfoVec[i].wqeType == static_cast<u64>(WqeType::WQE_TYPE_DATA)) {
                ret = dispatcher_->RdmaSend(dbIndex, dbInfo, wr, stream, machinePara_.remoteWorldRank, isCapture_);
            } else {
                ret = dispatcher_->RdmaSend(
                    dbIndex, dbInfo, wr, stream, machinePara_.remoteWorldRank, wqeInfoVec[i].wqeDataOffset, isCapture_);
            }
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[TransportIbverbs][RdmaSendAsync]errNo[0x%016llx] In lbv exp op base mode, "
                    "rdma send failed. dbIndex[%u] dbInfo[%llu] wqe type[%llu] offset[%llu]",
                    HCCL_ERROR_CODE(ret), dbIndex, dbInfo, wqeInfoVec[i].wqeType, wqeInfoVec[i].wqeDataOffset),
                ret);
        } else { // offline mode
            // 下沉模式
            if (wqeInfoVec[i].wqeType == static_cast<u64>(WqeType::WQE_TYPE_DATA)) {
                ret = dispatcher_->RdmaSend(
                    opRspVec[i].wqeTmp.sqIndex, opRspVec[i].wqeTmp.wqeIndex, wr, stream, machinePara_.remoteWorldRank);
            } else {
                ret = dispatcher_->RdmaSend(
                    opRspVec[i].wqeTmp.sqIndex, opRspVec[i].wqeTmp.wqeIndex, wr, stream, machinePara_.remoteWorldRank,
                    wqeInfoVec[i].wqeDataOffset);
            }
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[TransportIbverbs][RdmaSendAsync]errNo[0x%016llx] In lbv exp offline mode, "
                    "rdma send failed. sq_index[%u] wqe_index[%u], offset[%llu]",
                    HCCL_ERROR_CODE(ret), opRspVec[i].wqeTmp.sqIndex, opRspVec[i].wqeTmp.wqeIndex,
                    wqeInfoVec[i].wqeDataOffset),
                ret);
        }
    }
    return HCCL_SUCCESS;
}
HcclResult TransportIbverbs::RdmaSendAsyncHostNIC(std::vector<WqeInfo>& wqeInfoVec, Stream& stream)
{
    HcclResult ret;

    std::vector<SendWrlistDataExt> wqelistVec;
    for (u32 index = 0; index < wqeInfoVec.size(); index++) {
        wqelistVec.push_back(wqeInfoVec[index].wqeData);
    }
    struct SendWrRsp opRsp = {};
    struct SendWrlistDataExt wr = {};
    for (u32 i = 0; i < wqeInfoVec.size(); i++) {
        wr.memList.addr = wqelistVec[i].memList.addr;
        wr.memList.len = wqelistVec[i].memList.len;
        wr.dstAddr = static_cast<u64>(wqelistVec[i].dstAddr);
        wr.op = 0;
        wr.sendFlags = fence_ ? (RA_SEND_SIGNALED | RA_SEND_FENCE) : RA_SEND_SIGNALED;
        fence_ = false;

        if (wqeInfoVec[i].wqeType == static_cast<u64>(WqeType::WQE_TYPE_DATA)) {
            ret = dispatcher_->HostNicRdmaSend(
                combineQpHandles_[0].qpHandle, wr, opRsp, stream, machinePara_.remoteWorldRank);
        } else {
            ret = dispatcher_->HostNicRdmaSend(
                combineQpHandles_[0].qpHandle, wr, opRsp, stream, machinePara_.remoteWorldRank,
                wqeInfoVec[i].wqeDataOffset);
        }
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[TransportIbverbs][RdmaSendAsyncHostNIC]errNo[0x%016llx] In lbv exp offline "
                "mode, rdma send failed. sq_index[%u] wqe_index[%u], offset[%llu]",
                HCCL_ERROR_CODE(ret), opRsp.wqeTmp.sqIndex, opRsp.wqeTmp.wqeIndex, wqeInfoVec[i].wqeDataOffset),
            ret);
    }
    return HCCL_SUCCESS;
}
HcclResult TransportIbverbs::RdmaSendAsync(
    struct SendWr& wr, Stream& stream, WqeType wqeType, u64 notifyOffset, [[maybe_unused]] u32 notifyId)
{
    HcclResult ret;
    struct SendWrRsp opRsp = {};
    if (!IsTemplateMode()) {
        currentQP_ = 0;
    } else {
        if (sqeCounter_ == HCCP_SQ_TEMPLATE_CAPACITY) {
            currentQP_++;
            sqeCounter_ = 1;
        } else {
            sqeCounter_++;
        }
    }
    CHK_PRT_RET(
        currentQP_ >= combineQpHandles_.size(),
        HCCL_ERROR(
            "[TransportIbverbs][RdmaSendAsync]errNo[0x%016llx] In "
            "lbv exp, qp idx[%u] is invalid.",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), currentQP_),
        HCCL_E_INTERNAL);
    HCCL_DEBUG("rdma send async: ra qp sqe counter:%u, current qp idx:%u.", sqeCounter_, currentQP_);

    CHK_RET(HrtRaSendWr(combineQpHandles_[currentQP_].qpHandle, &wr, &opRsp));

    if (!IsTemplateMode()) {
        u32 dbIndex = static_cast<u32>(opRsp.db.dbIndex);
        u64 dbInfo = static_cast<u64>(opRsp.db.dbInfo);
        if (wqeType == WqeType::WQE_TYPE_DATA) {
            ret = dispatcher_->RdmaSend(dbIndex, dbInfo, wr, stream, machinePara_.remoteWorldRank, isCapture_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[TransportIbverbs][RdmaSendAsync]errNo[0x%016llx] In lbv exp op base mode, "
                    "rdma send failed. dbIndex[%u] dbInfo[%llu]",
                    HCCL_ERROR_CODE(ret), dbIndex, dbInfo),
                ret);
        } else {
            ret = dispatcher_->RdmaSend(
                dbIndex, dbInfo, wr, stream, machinePara_.remoteWorldRank, notifyOffset, isCapture_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[TransportIbverbs][RdmaSendAsync]errNo[0x%016llx] In lbv exp op base mode, "
                    "rdma send failed. dbIndex[%u] dbInfo[%llu], offset[%llu]",
                    HCCL_ERROR_CODE(ret), dbIndex, dbInfo, notifyOffset),
                ret);
        }
    } else { // offline mode
        if (wqeType == WqeType::WQE_TYPE_DATA) {
            ret = dispatcher_->RdmaSend(
                opRsp.wqeTmp.sqIndex, opRsp.wqeTmp.wqeIndex, wr, stream, machinePara_.remoteWorldRank);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[TransportIbverbs][RdmaSendAsync]errNo[0x%016llx] In lbv exp offline mode, "
                    "rdma send failed. sq_index[%u] wqe_index[%u]",
                    HCCL_ERROR_CODE(ret), opRsp.wqeTmp.sqIndex, opRsp.wqeTmp.wqeIndex),
                ret);
        } else {
            ret = dispatcher_->RdmaSend(
                opRsp.wqeTmp.sqIndex, opRsp.wqeTmp.wqeIndex, wr, stream, machinePara_.remoteWorldRank, notifyOffset);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[TransportIbverbs][RdmaSendAsync]errNo[0x%016llx] In lbv exp offline mode, "
                    "rdma send failed. sq_index[%u] wqe_index[%u], offset[%llu]",
                    HCCL_ERROR_CODE(ret), opRsp.wqeTmp.sqIndex, opRsp.wqeTmp.wqeIndex, notifyOffset),
                ret);
        }
    }
    return HCCL_SUCCESS;
}
HcclResult
TransportIbverbs::RdmaSendAsyncHostNIC(struct SendWrlistDataExt& wr, Stream& stream, WqeType wqeType, u64 notifyOffset)
{
    HcclResult ret;
    struct SendWrRsp opRsp = {};
    if (wqeType == WqeType::WQE_TYPE_DATA) {
        ret = dispatcher_->HostNicRdmaSend(
            combineQpHandles_[0].qpHandle, wr, opRsp, stream, machinePara_.remoteWorldRank);
    } else {
        ret = dispatcher_->HostNicRdmaSend(
            combineQpHandles_[0].qpHandle, wr, opRsp, stream, machinePara_.remoteWorldRank, notifyOffset);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TransportIbverbs][RdmaSendAsyncHostNIC]errNo[0x%016llx] In lbv exp offline mode, "
            "rdma send failed. sq_index[%u] wqe_index[%u], offset[%llu]",
            HCCL_ERROR_CODE(ret), opRsp.wqeTmp.sqIndex, opRsp.wqeTmp.wqeIndex, notifyOffset),
        ret);

    return HCCL_SUCCESS;
}
HcclResult TransportIbverbs::GetWqeDataOffsetAndNotifyId(WqeType wqeType, u64& wqeDataOffset, u32& notifyId)
{
    switch (wqeType) {
        case WqeType::WQE_TYPE_DATA:
        case WqeType::WQE_TYPE_DATA_WITH_NOTIFY:
        case WqeType::WQE_TYPE_DATA_WITH_REDUCE:
        case WqeType::WQE_TYPE_READ_DATA:
            wqeDataOffset = 0;
            notifyId = INVALID_UINT;
            break;
        case WqeType::WQE_TYPE_DATA_NOTIFY:
            wqeDataOffset = remoteDataNotifyMsg_.offset;
            notifyId = remoteDataNotifyMsg_.notifyId;
            break;
        case WqeType::WQE_TYPE_ACK_NOTIFY:
            wqeDataOffset = remoteMemMsg_[static_cast<u32>(MemType::ACK_NOTIFY_MEM)].offset;
            notifyId = remoteMemMsg_[static_cast<u32>(MemType::ACK_NOTIFY_MEM)].notifyId;
            break;
        case WqeType::WQE_TYPE_DATA_ACK_NOTIFY:
            wqeDataOffset = remoteMemMsg_[static_cast<u32>(MemType::DATA_ACK_NOTIFY_MEM)].offset;
            notifyId = remoteMemMsg_[static_cast<u32>(MemType::DATA_ACK_NOTIFY_MEM)].notifyId;
            break;
        default:
            HCCL_ERROR("[Get][WqeDataOffset]error wqeType[%d]", wqeType);
            return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult
TransportIbverbs::TxSendWqe(void* dstMemPtr, const void* srcMemPtr, u64 srcMemSize, Stream& stream, WqeType wqeType)
{
    if (machinePara_.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE && !useAtomicWrite_) {
        struct SgList list = {};
        struct SendWr wr = {};
        // 构造wr信息
        list.addr = static_cast<u64>(reinterpret_cast<uintptr_t>(srcMemPtr));
        list.len = srcMemSize;

        wr.bufList = &list;
        wr.bufNum = 1; /* 此处list只有一个，设置为1 */
        wr.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(dstMemPtr));
        wr.op = 0; /* RDMA_WRITE: 0 */
        wr.sendFlag = fence_ ? (RA_SEND_SIGNALED | RA_SEND_FENCE) : RA_SEND_SIGNALED;
        fence_ = false;

        // 获取notify偏移地址，对于发送数据时，偏移地址为0
        u32 notifyId = INVALID_UINT;
        u64 wqeDataOffset = 0;
        CHK_RET(GetWqeDataOffsetAndNotifyId(wqeType, wqeDataOffset, notifyId));

        // RDMA异步发送
        CHK_RET(RdmaSendAsync(wr, stream, wqeType, wqeDataOffset, notifyId));
    } else if (machinePara_.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE && useAtomicWrite_) {
        struct WrAuxInfo aux = {};
        std::vector<WqeInfo> wqeInfoVec;
        CHK_RET(AddWqeList(dstMemPtr, srcMemPtr, srcMemSize, wqeType, aux, wqeInfoVec));
        CHK_RET(RdmaSendAsync(wqeInfoVec, stream, false));
        HCCL_DEBUG("TxSendWqe useAtomicWrite[%d]", useAtomicWrite_);
    } else {
        struct SendWrlistDataExt wr = {};
        // 构造wr信息
        wr.memList.addr = static_cast<u64>(reinterpret_cast<uintptr_t>(srcMemPtr));
        wr.memList.len = srcMemSize;

        wr.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(dstMemPtr));
        wr.op = 0; /* RDMA_WRITE: 0 */
        wr.sendFlags = fence_ ? (RA_SEND_SIGNALED | RA_SEND_FENCE) : RA_SEND_SIGNALED;
        fence_ = false;

        // 获取notify偏移地址，对于发送数据时，偏移地址为0
        u32 notifyId = INVALID_UINT;
        u64 wqeDataOffset = 0;
        CHK_RET(GetWqeDataOffsetAndNotifyId(wqeType, wqeDataOffset, notifyId));

        // RDMA异步发送
        CHK_RET(RdmaSendAsyncHostNIC(wr, stream, wqeType, wqeDataOffset));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::TxSendNotifyWqe(MemMsg& memMsg, const void* srcMemPtr, u64 srcMemSize, Stream& stream)
{
    if (machinePara_.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE && !useAtomicWrite_) {
        struct SgList list = {};
        struct SendWr wr = {};
        // 构造wr信息
        list.addr = static_cast<u64>(reinterpret_cast<uintptr_t>(srcMemPtr));
        list.len = srcMemSize;
        wr.bufList = &list;
        wr.bufNum = 1; /* 此处list只有一个，设置为1 */
        wr.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(memMsg.addr));
        wr.op = 0; /* RDMA_WRITE: 0 */
        wr.sendFlag = fence_ ? (RA_SEND_SIGNALED | RA_SEND_FENCE) : RA_SEND_SIGNALED;
        fence_ = false;

        // RDMA异步发送
        CHK_RET(RdmaSendAsync(wr, stream, WqeType::WQE_TYPE_ACK_NOTIFY, memMsg.offset, memMsg.notifyId));
    } else if (machinePara_.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE && useAtomicWrite_) {
        struct WrAuxInfo aux = {};
        std::vector<WqeInfo> wqeInfoVec;
        CHK_RET(AddWqeList(memMsg.addr, srcMemPtr, srcMemSize, WqeType::WQE_TYPE_ACK_NOTIFY, aux, wqeInfoVec));
        CHK_RET(RdmaSendAsync(wqeInfoVec, stream, false));
        HCCL_DEBUG("TxSendNotifyWqe useAtomicWrite[%d]", useAtomicWrite_);
    } else {
        struct SendWrlistDataExt wr = {};
        // 构造wr信息
        wr.memList.addr = static_cast<u64>(reinterpret_cast<uintptr_t>(srcMemPtr));
        wr.memList.len = srcMemSize;
        wr.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(memMsg.addr));
        wr.op = 0; /* RDMA_WRITE: 0 */
        wr.sendFlags = fence_ ? (RA_SEND_SIGNALED | RA_SEND_FENCE) : RA_SEND_SIGNALED;
        fence_ = false;

        // RDMA异步发送
        CHK_RET(RdmaSendAsyncHostNIC(wr, stream, WqeType::WQE_TYPE_ACK_NOTIFY, memMsg.offset));
    }
    return HCCL_SUCCESS;
}

HcclResult
TransportIbverbs::RxAsync([[maybe_unused]] UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream)
{
    u32 actualMultiQpNum = 1;
    const u32 KByteToByte = 1024; // 1024 多QP阈值单位是KB
    if (len / qpsPerConnection_ > GetExternalInputMultiQpThreshold() * KByteToByte) {
        actualMultiQpNum = qpsPerConnection_;
    } else {
        u32 quotient = len / (GetExternalInputMultiQpThreshold() * KByteToByte);
        u32 remainder = len % (GetExternalInputMultiQpThreshold() * KByteToByte);
        actualMultiQpNum = quotient + (remainder != 0 ? 1 : 0);
    }
    // 等待TS把任务处理完成
    HCCL_DEBUG(
        "[TransportIbverbs][RxAsync] UseMultiQp[%d] actualMultiQpNum[%u], RX dst[%p] len[%llu] srcOffset[%llu]",
        UseMultiQp(), actualMultiQpNum, dst, len, srcOffset);
    if (UseMultiQp() && actualMultiQpNum != 1 && actualMultiQpNum <= qpsPerConnection_ && len != 0) {
        for (u32 i = 0; i < actualMultiQpNum; i++) {
            CHK_RET(LocalIpcNotify::Wait(
                stream, dispatcher_, multiQpDataNotify_[i], INVALID_VALUE_STAGE, NOTIFY_INVALID_WAIT_TIME,
                machinePara_.localUserrank, machinePara_.remoteWorldRank));
        }
    } else {
        CHK_RET(LocalIpcNotify::Wait(
            stream, dispatcher_, dataNotify_, INVALID_VALUE_STAGE, NOTIFY_INVALID_WAIT_TIME, machinePara_.localUserrank,
            machinePara_.remoteWorldRank));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::RxAsync(std::vector<RxMemoryInfo>& rxMems, Stream& stream)
{
    CHK_PRT_RET(rxMems.size() == 0, HCCL_ERROR("Invalid rxMem size[%u]", rxMems.size()), HCCL_E_PARA);
    for (auto& mem : rxMems) {
        HCCL_DEBUG("RX dst[%p] len[%llu] dstOffset[%llu]", mem.dst, mem.len, mem.srcOffset);
    }
    u32 maxLength = 0;
    for (u32 i = 0; i < rxMems.size(); i++) {
        if (rxMems[i].len > maxLength) {
            maxLength = rxMems[i].len;
        }
    }

    CHK_RET(RxAsync(rxMems[0].srcMemType, rxMems[0].srcOffset, rxMems[0].dst, maxLength, stream));
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::DataReceivedAck(Stream& stream)
{
    CHK_RET(PostFinAck(stream));
    CHK_RET(WaitFinAck(stream));

    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::TxWaitDone([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }

/* 发送ack消息(同步模式) */
HcclResult TransportIbverbs::TxAck(Stream& stream)
{
    CHK_RET(TxSendWqe(
        remoteMemMsg_[static_cast<u32>(MemType::ACK_NOTIFY_MEM)].addr,
        notifyValueMem_[machinePara_.deviceLogicId].ptr(), notifySize_, stream, WqeType::WQE_TYPE_ACK_NOTIFY));
    return HCCL_SUCCESS;
}

/* 接收ack消息(同步模式) */
HcclResult TransportIbverbs::RxAck(Stream& stream)
{
    HcclResult ret = LocalIpcNotify::Wait(
        stream, dispatcher_, ackNotify_, INVALID_VALUE_STAGE, NOTIFY_INVALID_WAIT_TIME, machinePara_.localUserrank,
        machinePara_.remoteWorldRank);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TransportIbverbs][RxAck]errNo[0x%016llx] In lbv exp rx ack, signal wait failed. ", HCCL_ERROR_CODE(ret)),
        ret);

    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::TxDataSignal(Stream& stream)
{
    // 发送data notify同步信息
    void* remoteNotifyaddr = remoteDataNotifyMsg_.addr;
    HcclResult ret = TxSendWqe(
        remoteNotifyaddr, notifyValueMem_[machinePara_.deviceLogicId].ptr(), notifySize_, stream,
        WqeType::WQE_TYPE_DATA_NOTIFY);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TransportIbverbs][TxDataSignal]errNo[0x%016llx] In ibv tx data signal, send notify "
            "wqe failed. dstMemPtr[%p], srcMemPtr[%p], srcMemSize[%llu Byte]",
            HCCL_ERROR_CODE(ret), remoteNotifyaddr, notifyValueMem_[machinePara_.deviceLogicId].ptr(), notifySize_),
        ret);
    // 每发送一个data notify wqe, count 自增
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::RxDataSignal(Stream& stream)
{
    /* 等待send_ready_event事件 */
    CHK_RET(LocalIpcNotify::Wait(
        stream, dispatcher_, dataNotify_, INVALID_VALUE_STAGE, NOTIFY_INVALID_WAIT_TIME, machinePara_.localUserrank,
        machinePara_.remoteWorldRank));
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::CreateNotifyVectorBuffer(
    std::vector<std::shared_ptr<LocalIpcNotify>>& notifyVector, u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    NotifyLoadType notifyLoadType
        = machinePara_.isAicpuModeEn ? NotifyLoadType::DEVICE_NOTIFY : NotifyLoadType::HOST_NOTIFY;
    for (u32 i = 0; i < notifyNum_; i++) {
        std::shared_ptr<LocalIpcNotify> oneNotify;
        CHK_RET(CreateNotifyBuffer(
            oneNotify, MemType::MUILT_NOTIFY_MEM, exchangeDataPtr, exchangeDataBlankSize, notifyLoadType));
        notifyVector.push_back(std::move(oneNotify));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::CreateNotifyBuffer(
    std::shared_ptr<LocalIpcNotify>& localNotify, MemType notifyType, u8*& exchangeDataPtr, u64& exchangeDataBlankSize,
    NotifyLoadType notifyLoadType)
{
    u64 offset = 0;
    u64 notifyBaseVa = 0; // notify寄存器虚拟地址
    u64 notifyTotalSize = 0;
    u32 notifyKey = 0;

    HcclRtNotify notify = nullptr;

    /* 获取notify寄存器虚拟基地址、大小, 物理地址回传值为空 */
    struct MrInfoT mrInfo = {};
    if (machinePara_.isAicpuModeEn
        || (machinePara_.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE
            && machinePara_.deviceType != localDeviceType)
        || isHybridMode_) {
        CHK_RET(HrtRaGetNotifyMrInfo(machinePara_.localDeviceId, nicRdmaHandle_, &mrInfo));
        notifyBaseVa = reinterpret_cast<u64>(mrInfo.addr);
        notifyTotalSize = mrInfo.size;
        notifyKey = mrInfo.lkey;
    } else {
        u64 notifyBaseVaTmp = 0;
        notifyBaseVaTmp = notifyBaseVa;
        CHK_RET(HrtRaGetNotifyBaseAddr(nicRdmaHandle_, &notifyBaseVa, &notifyTotalSize, [this]() -> bool {
            return this->GetStopFlag();
        }));
        CHK_PRT_RET(
            ((notifyBaseVaTmp != 0) && (notifyBaseVaTmp != notifyBaseVa)),
            HCCL_ERROR("[Create][NotifyBuffer]In lbv exp init, get base addr failed. notify base va has changed."),
            HCCL_E_INTERNAL);
    }

    /* 申请Notify Group ID */
    RemoteRankInfo info(machinePara_.localDeviceId, machinePara_.remoteUserrank);
    CHK_RET(SalGetBareTgid(&info.remotePid)); // 当前进程id

    // atomic write使能场景下，要求写入wr的notify地址是8byte对齐的
    u32 offsetAlignSize = INVALID_UINT;
    if (machinePara_.enableAtomicWrite) {
        // 映射的MR基地址必定是8字节对齐的
        CHK_PRT_RET(
            notifyBaseVa % NOTIFY_VA_ALIGN_EIGHT != 0,
            HCCL_ERROR("%s notifyBaseVa[0x%llx] not %u aligned", __func__, notifyBaseVa, NOTIFY_VA_ALIGN_EIGHT),
            HCCL_E_INTERNAL);
        offsetAlignSize = NOTIFY_VA_ALIGN_EIGHT;
    }
    CHK_RET(notifyPool_->Alloc(machinePara_.tag, info, localNotify, notifyLoadType, offsetAlignSize));
    // 设置remote id
    s64 recvId = 0xFFFFFFFF00000000 | (static_cast<s64>(info.remotePid) & 0xFFFFFFFF);
    CHK_RET(localNotify->Grant(recvId));

    /* 获取notify虚拟地址 */
    CHK_RET(localNotify->GetNotifyOffset(offset));

    // notify寄存器的虚拟地址与物理地址偏移相同，所以虚拟地址为虚拟基地址加偏移
    u64 notifyVa = notifyBaseVa + offset;
    CHK_PRT_RET(
        machinePara_.enableAtomicWrite && (notifyVa % NOTIFY_VA_ALIGN_EIGHT != 0),
        HCCL_ERROR(
            "%s notifyVa[0x%llx] not %u aligned, notifyBaseVa[0x%llx], offset[0x%llx], enableAtomicWrite[%d]", __func__,
            notifyVa, NOTIFY_VA_ALIGN_EIGHT, notifyBaseVa, offset, machinePara_.enableAtomicWrite),
        HCCL_E_INTERNAL);

    HCCL_INFO(
        "%s notifyBaseVa=0x%llx, notifyTotalSize=0x%x, offset=0x%llx, notifyVa=0x%llx machineType=%d, "
        "notify=%p, notifyType=%d, notifyId=%u, offsetAlignSize[%u]",
        __func__, notifyBaseVa, notifyTotalSize, offset, notifyVa, machinePara_.machineType, notify, notifyType,
        localNotify->notifyId_, offsetAlignSize);

    if (notifyType != MULTI_QP_DATA_NOTIFY_MEM) {
        /* notify地址注册为mr, 在roce驱动中注册 */
        memMsg_[static_cast<u32>(notifyType)].mrRegFlag = 0; // mem注册给网卡标志位
        // 本端notify地址交换给对端
        memMsg_[static_cast<u32>(notifyType)].addr = reinterpret_cast<void*>(static_cast<uintptr_t>(notifyVa));
        memMsg_[static_cast<u32>(notifyType)].len = notifySize_;
        memMsg_[static_cast<u32>(notifyType)].memType = notifyType;
        memMsg_[static_cast<u32>(notifyType)].offset = offset;
        memMsg_[static_cast<u32>(notifyType)].lkey = mrInfo.lkey;
        memMsg_[static_cast<u32>(notifyType)].lkey = notifyKey;
        memMsg_[static_cast<u32>(notifyType)].notifyId = localNotify->notifyId_;

        /* 拼接要发送的数据 */
        CHK_SAFETY_FUNC_RET(memcpy_s(
            exchangeDataPtr, exchangeDataBlankSize, reinterpret_cast<void*>(&memMsg_[static_cast<u32>(notifyType)]),
            sizeof(MemMsg)));
    } else {
        MemMsg memMsg;
        memMsg.mrRegFlag = 0;
        memMsg.addr = reinterpret_cast<void*>(static_cast<uintptr_t>(notifyVa));
        memMsg.len = notifySize_;
        memMsg.memType = notifyType;
        memMsg.offset = offset;
        memMsg.notifyId = localNotify->notifyId_;
        multiQpDataNotifyMemMsg_.push_back(std::move(memMsg));
        CHK_SAFETY_FUNC_RET(
            memcpy_s(exchangeDataPtr, exchangeDataBlankSize, reinterpret_cast<void*>(&memMsg), sizeof(MemMsg)));
    }

    exchangeDataPtr += sizeof(MemMsg);
    exchangeDataBlankSize -= sizeof(MemMsg);

    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::RegUserMem(MemType memType, u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    void* memPtr = nullptr;
    u64 memSize;
    switch (memType) {
        case MemType::USER_INPUT_MEM: {
            memPtr = machinePara_.inputMem.ptr();
            memSize = machinePara_.inputMem.size();
            break;
        }

        case MemType::USER_OUTPUT_MEM: {
            memPtr = machinePara_.outputMem.ptr();
            memSize = machinePara_.outputMem.size();
            break;
        }

        default: {
            HCCL_ERROR("[Reg][UserMem]not support dst_mem_type=%d", memType);
            return HCCL_E_NOT_SUPPORT;
        }
    }
    struct MrInfoT mrInfo = {};
    mrInfo.addr = memPtr;
    mrInfo.size = memSize;
    mrInfo.access = access_;
    if (mrInfo.size != 0) {
        for (u32 i = 0; i < combineQpHandles_.size(); i++) {
            CHK_RET(HrtRaMrReg(combineQpHandles_[i].qpHandle, &mrInfo));
        }

        if (UseMultiQp()) {
            for (u32 i = 0; i < qpsPerConnection_; i++) {
                CHK_RET(HrtRaMrReg(multiCombineQpHandles_[i].qpHandle, &mrInfo));
            }
        }
        memMsg_[static_cast<u32>(memType)].mrRegFlag = REG_VALID;
    }

    memMsg_[static_cast<u32>(memType)].addr = memPtr;
    memMsg_[static_cast<u32>(memType)].len = memSize;
    memMsg_[static_cast<u32>(memType)].memType = memType;
    memMsg_[static_cast<u32>(memType)].lkey = mrInfo.lkey;

    CHK_SAFETY_FUNC_RET(memcpy_s(
        exchangeDataPtr, exchangeDataBlankSize, reinterpret_cast<void*>(&memMsg_[static_cast<u32>(memType)]),
        sizeof(MemMsg)));

    exchangeDataPtr += sizeof(MemMsg);
    exchangeDataBlankSize -= sizeof(MemMsg);

    HCCL_DEBUG("memType=%d mem_ptr=%p mem_size=%llu Byte, key = %u", memType, memPtr, memSize, mrInfo.lkey);

    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::RegCustomUserMemWithMsg(
    void* addr, u64 size, MemMsg& memMsg, u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    struct MrInfoT mrInfo = {};
    mrInfo.addr = addr;
    mrInfo.size = size;
    mrInfo.access = access_;
    for (u32 i = 0; i < combineQpHandles_.size(); i++) {
        CHK_RET(HrtRaMrReg(combineQpHandles_[i].qpHandle, &mrInfo));
    }

    if (UseMultiQp()) {
        for (u32 i = 0; i < qpsPerConnection_; i++) {
            CHK_RET(HrtRaMrReg(multiCombineQpHandles_[i].qpHandle, &mrInfo));
        }
    }

    memMsg.mrRegFlag = REG_VALID;
    memMsg.addr = addr;
    memMsg.len = size;
    memMsg.lkey = mrInfo.lkey;

    CHK_SAFETY_FUNC_RET(
        memcpy_s(exchangeDataPtr, exchangeDataBlankSize, reinterpret_cast<void*>(&memMsg), sizeof(MemMsg)));

    exchangeDataPtr += sizeof(MemMsg);
    exchangeDataBlankSize -= sizeof(MemMsg);

    HCCL_DEBUG("mem_ptr=%p mem_size=%llu Byte, key = %u", addr, size, mrInfo.lkey);

    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::RegCustomUserMem(u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    u32 deviceMemNum = machinePara_.userDeviceMem.size();
    s32 sRet = memcpy_s(exchangeDataPtr, sizeof(u32), reinterpret_cast<void*>(&deviceMemNum), sizeof(u32));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[Set][LocalMem]errNo[0x%016llx] memory copy failed. errorno[%d], params:dstMaxSize[%zu],cnt[%zu]",
            HCCL_ERROR_CODE(HCCL_E_MEMORY), sRet, sizeof(u32), sizeof(u32)),
        HCCL_E_MEMORY);
    exchangeDataPtr += sizeof(u32);
    exchangeDataBlankSize -= sizeof(u32);

    userDeviceMemMsg_.resize(deviceMemNum);
    for (u32 i = 0; i < deviceMemNum; i++) {
        RegCustomUserMemWithMsg(
            machinePara_.userDeviceMem[i].ptr(), machinePara_.userDeviceMem[i].size(), userDeviceMemMsg_[i],
            exchangeDataPtr, exchangeDataBlankSize);
    }

    u32 hostMemNum = machinePara_.userHostMem.size();
    sRet = memcpy_s(exchangeDataPtr, sizeof(u32), reinterpret_cast<void*>(&hostMemNum), sizeof(u32));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[Set][LocalMem]errNo[0x%016llx] memory copy failed. errorno[%d], params:dstMaxSize[%zu],cnt[%zu]",
            HCCL_ERROR_CODE(HCCL_E_MEMORY), sRet, sizeof(u32), sizeof(u32)),
        HCCL_E_MEMORY);
    exchangeDataPtr += sizeof(u32);
    exchangeDataBlankSize -= sizeof(u32);

    userHostMemMsg_.resize(hostMemNum);
    for (u32 i = 0; i < hostMemNum; i++) {
        RegCustomUserMemWithMsg(
            machinePara_.userHostMem[i].ptr(), machinePara_.userHostMem[i].size(), userHostMemMsg_[i], exchangeDataPtr,
            exchangeDataBlankSize);
    }

    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::GetMemInfo(UserMemType memType, void** dstMemPtr, u64* dstMemSize)
{
    switch (memType) {
        case UserMemType::INPUT_MEM: {
            *dstMemPtr = remoteMemMsg_[static_cast<u32>(MemType::USER_INPUT_MEM)].addr;
            *dstMemSize = remoteMemMsg_[static_cast<u32>(MemType::USER_INPUT_MEM)].len;
            break;
        }

        case UserMemType::OUTPUT_MEM: {
            *dstMemPtr = remoteMemMsg_[static_cast<u32>(MemType::USER_OUTPUT_MEM)].addr;
            *dstMemSize = remoteMemMsg_[static_cast<u32>(MemType::USER_OUTPUT_MEM)].len;
            break;
        }

        default: {
            HCCL_ERROR("[Get][MemInfo]not support dst_mem_type=%d", memType);
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::GetRemoteAddr(MemType memType, u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    if (memType != MULTI_QP_DATA_NOTIFY_MEM) {
        s32 sRet = memcpy_s(&remoteMemMsg_[static_cast<u32>(memType)], sizeof(MemMsg), exchangeDataPtr, sizeof(MemMsg));
        CHK_PRT_RET(
            sRet != EOK,
            HCCL_ERROR(
                "[Get][RemoteAddr]errNo[0x%016llx] In lbv exp get remote addr, "
                "memcpy failed. errorno[%d], params:destMaxSize[%zu],count[%zu]",
                HCCL_ERROR_CODE(HCCL_E_MEMORY), sRet, sizeof(MemMsg), sizeof(MemMsg)),
            HCCL_E_MEMORY);
        CHK_PTR_NULL(remoteMemMsg_[static_cast<u32>(memType)].addr);
    } else {
        MemMsg memMsg;
        s32 sRet = memcpy_s(&memMsg, sizeof(MemMsg), exchangeDataPtr, sizeof(MemMsg));
        CHK_PRT_RET(
            sRet != EOK,
            HCCL_ERROR(
                "[Get][RemoteAddr]errNo[0x%016llx] In lbv exp get remote addr, "
                "memcpy failed. errorno[%d], params:destMaxSize[%zu],count[%zu]",
                HCCL_ERROR_CODE(HCCL_E_MEMORY), sRet, sizeof(MemMsg), sizeof(MemMsg)),
            HCCL_E_MEMORY);
        CHK_PTR_NULL(memMsg.addr);
        multiQpDataNotifyRemoteMemMsg_.push_back(std::move(memMsg));
    }

    exchangeDataPtr += sizeof(MemMsg);
    exchangeDataBlankSize -= sizeof(MemMsg);
    HCCL_INFO(
        "GetRemoteAddr success: memType=%d, addr=%p len=%llu, notifyId=%u", static_cast<int32_t>(memType),
        remoteMemMsg_[static_cast<u32>(memType)].addr, remoteMemMsg_[static_cast<u32>(memType)].len,
        remoteMemMsg_[static_cast<u32>(memType)].notifyId);
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::GetIndOpRemoteAddr(u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    u32 remoteDmemNum = 0;
    s32 sRet = memcpy_s(reinterpret_cast<void*>(&remoteDmemNum), sizeof(u32), exchangeDataPtr, sizeof(u32));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[Get][RemoteMem]errNo[0x%016llx] memory copy failed. errorno[%d], params:dstMaxSize[%zu],cnt[%zu]",
            HCCL_ERROR_CODE(HCCL_E_MEMORY), sRet, sizeof(u32), sizeof(u32)),
        HCCL_E_MEMORY);
    exchangeDataPtr += sizeof(u32);
    exchangeDataBlankSize -= sizeof(u32);

    remoteUserDeviceMemMsg_.resize(remoteDmemNum);
    for (u32 i = 0; i < remoteDmemNum; i++) {
        sRet = memcpy_s(&remoteUserDeviceMemMsg_[i], sizeof(MemMsg), exchangeDataPtr, sizeof(MemMsg));
        CHK_PRT_RET(
            sRet != EOK,
            HCCL_ERROR(
                "[Get][GetCustomRemoteAddr]errNo[0x%016llx] In lbv exp get remote addr, "
                "memcpy failed. errorno[%d], params:destMaxSize[%zu],count[%zu]",
                HCCL_ERROR_CODE(HCCL_E_MEMORY), sRet, sizeof(MemMsg), sizeof(MemMsg)),
            HCCL_E_MEMORY);
        exchangeDataPtr += sizeof(MemMsg);
        exchangeDataBlankSize -= sizeof(MemMsg);
        CHK_PTR_NULL(remoteUserDeviceMemMsg_[i].addr);
    }

    u32 remoteHmemNum = 0;
    sRet = memcpy_s(reinterpret_cast<void*>(&remoteHmemNum), sizeof(u32), exchangeDataPtr, sizeof(u32));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[Get][RemoteMem]errNo[0x%016llx] memory copy failed. errorno[%d], params:dstMaxSize[%zu],cnt[%zu]",
            HCCL_ERROR_CODE(HCCL_E_MEMORY), sRet, sizeof(u32), sizeof(u32)),
        HCCL_E_MEMORY);
    exchangeDataPtr += sizeof(u32);
    exchangeDataBlankSize -= sizeof(u32);

    remoteUserHostMemMsg_.resize(remoteHmemNum);
    for (u32 i = 0; i < remoteHmemNum; i++) {
        sRet = memcpy_s(&remoteUserHostMemMsg_[i], sizeof(MemMsg), exchangeDataPtr, sizeof(MemMsg));
        CHK_PRT_RET(
            sRet != EOK,
            HCCL_ERROR(
                "[Get][GetCustomRemoteAddr]errNo[0x%016llx] In lbv exp get remote addr, "
                "memcpy failed. errorno[%d], params:destMaxSize[%zu],count[%zu]",
                HCCL_ERROR_CODE(HCCL_E_MEMORY), sRet, sizeof(MemMsg), sizeof(MemMsg)),
            HCCL_E_MEMORY);
        exchangeDataPtr += sizeof(MemMsg);
        exchangeDataBlankSize -= sizeof(MemMsg);
        CHK_PTR_NULL(remoteUserHostMemMsg_[i].addr);
    }

    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::GetRemoteNotifyAddr(u8*& exchangeDataPtr, u64& exchangeDataBlankSize, MemMsg& memMsg)
{
    s32 sRet = memcpy_s(&memMsg, sizeof(MemMsg), exchangeDataPtr, sizeof(MemMsg));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[Get][GetRemoteNotifyAddr]errNo[0x%016llx] In lbv exp get remote addr, "
            "memcpy failed. errorno[%d], params:destMaxSize[%zu],count[%zu]",
            HCCL_ERROR_CODE(HCCL_E_MEMORY), sRet, sizeof(MemMsg), sizeof(MemMsg)),
        HCCL_E_MEMORY);

    exchangeDataPtr += sizeof(MemMsg);
    exchangeDataBlankSize -= sizeof(MemMsg);
    CHK_PTR_NULL(memMsg.addr);
    HCCL_INFO("GetRemoteNotifyAddr success: addr=%p len=%llu", memMsg.addr, memMsg.len);
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::CreateNotifyValueBuffer()
{
    if (!machinePara_.userMemEnable && !machinePara_.drainEnable) {
        HCCL_INFO("userMemEnable is false, no need to create notify value buffer");
        return HCCL_SUCCESS;
    }
    std::unique_lock<std::mutex> lock(notifyValueMutex_[machinePara_.deviceLogicId]);
    if (notifyValueMem_[machinePara_.deviceLogicId].ptr() == nullptr) {
        u64 notifyVaule = 1; // notify值写1表示record
        CHK_RET(DeviceMem::alloc(notifyValueMem_[machinePara_.deviceLogicId], notifyValueSize_));
        HCCL_DEBUG(
            "create notify value buffer[%p], size[%u]", notifyValueMem_[machinePara_.deviceLogicId].ptr(), notifySize_);

        CHK_RET(hrtMemSyncCopy(
            notifyValueMem_[machinePara_.deviceLogicId].ptr(), notifyValueMem_[machinePara_.deviceLogicId].size(),
            &notifyVaule, notifySize_, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    }
    lock.unlock();

    struct MrInfoT mrInfo = {};
    mrInfo.addr = notifyValueMem_[machinePara_.deviceLogicId].ptr();
    mrInfo.size = notifySize_;
    mrInfo.access = access_;

    for (u32 i = 0; i < combineQpHandles_.size(); i++) {
        CHK_RET(HrtRaMrReg(combineQpHandles_[i].qpHandle, &mrInfo));
    }

    if (UseMultiQp()) {
        for (u32 i = 0; i < qpsPerConnection_; i++) {
            CHK_RET(HrtRaMrReg(multiCombineQpHandles_[i].qpHandle, &mrInfo));
        }
    }
    memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].mrRegFlag = REG_VALID;
    memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].addr = notifyValueMem_[machinePara_.deviceLogicId].ptr();
    memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].len = notifySize_;
    memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].memType = MemType::NOTIFY_SRC_MEM;
    memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].lkey = mrInfo.lkey;

    HCCL_DEBUG("notifyValueMem_=%p", notifyValueMem_[machinePara_.deviceLogicId].ptr());

    return HCCL_SUCCESS;
}
void TransportIbverbs::DestroySignal()
{
    dataNotify_ = nullptr;

    ackNotify_ = nullptr;

    dataAckNotify_ = nullptr;

    multiQpDataNotify_.clear();
}

/* 发送ack消息(同步模式) */
HcclResult TransportIbverbs::TxPrepare(Stream& stream)
{
    HcclResult ret = LocalIpcNotify::Wait(
        stream, dispatcher_, ackNotify_, INVALID_VALUE_STAGE, NOTIFY_INVALID_WAIT_TIME, machinePara_.localUserrank,
        machinePara_.remoteWorldRank);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TransportIbverbs][TxPrepare]errNo[0x%016llx] In lbv exp rx ack, signal wait failed. ",
            HCCL_ERROR_CODE(ret)),
        ret);

    return HCCL_SUCCESS;
}

/* 接收ack消息(同步模式) */
HcclResult TransportIbverbs::RxPrepare(Stream& stream)
{
    CHK_RET(TxSendWqe(
        remoteMemMsg_[static_cast<u32>(MemType::ACK_NOTIFY_MEM)].addr,
        notifyValueMem_[machinePara_.deviceLogicId].ptr(), notifySize_, stream, WqeType::WQE_TYPE_ACK_NOTIFY));
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::TxData(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream)
{
    std::vector<WqeInfo> wqeInfoVec;
    struct WrAuxInfo aux = {};
    HCCL_DEBUG("TX src[%p] len[%llu] dstOffset[%llu]", src, len, dstOffset);

    if (src != nullptr) {
        CHK_RET(TxPayLoad(dstMemType, dstOffset, src, len, WqeType::WQE_TYPE_DATA, aux, wqeInfoVec));
    }

    if (machinePara_.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        CHK_RET(RdmaSendAsync(wqeInfoVec, stream, false));
    } else {
        CHK_RET(RdmaSendAsyncHostNIC(wqeInfoVec, stream));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::RxData(
    [[maybe_unused]] UserMemType srcMemType, [[maybe_unused]] u64 srcOffset, [[maybe_unused]] void* dst,
    [[maybe_unused]] u64 len, [[maybe_unused]] Stream& stream)
{
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::TxDone(Stream& stream)
{
    // 发送数据接收确认notify
    CHK_RET(TxSendWqe(
        remoteMemMsg_[static_cast<u32>(MemType::DATA_NOTIFY_MEM)].addr,
        notifyValueMem_[machinePara_.deviceLogicId].ptr(), notifySize_, stream, WqeType::WQE_TYPE_DATA_NOTIFY));
    // 接收数据接收确认notify
    CHK_RET(LocalIpcNotify::Wait(
        stream, dispatcher_, dataAckNotify_, INVALID_VALUE_STAGE, NOTIFY_INVALID_WAIT_TIME, machinePara_.localUserrank,
        machinePara_.remoteWorldRank));
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::RxDone(Stream& stream)
{
    // 接收数据接收确认notify
    CHK_RET(LocalIpcNotify::Wait(
        stream, dispatcher_, dataNotify_, INVALID_VALUE_STAGE, NOTIFY_INVALID_WAIT_TIME, machinePara_.localUserrank,
        machinePara_.remoteWorldRank));

    // 发送数据接收确认notify
    CHK_RET(TxSendWqe(
        remoteMemMsg_[static_cast<u32>(MemType::DATA_ACK_NOTIFY_MEM)].addr,
        notifyValueMem_[machinePara_.deviceLogicId].ptr(), notifySize_, stream, WqeType::WQE_TYPE_DATA_ACK_NOTIFY));
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::PostReady(Stream& stream)
{
    CHK_RET(TxSendWqe(
        remoteMemMsg_[static_cast<u32>(MemType::ACK_NOTIFY_MEM)].addr,
        notifyValueMem_[machinePara_.deviceLogicId].ptr(), notifySize_, stream, WqeType::WQE_TYPE_ACK_NOTIFY));
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::WaitReady(Stream& stream)
{
    CHK_RET(LocalIpcNotify::Wait(
        stream, dispatcher_, ackNotify_, INVALID_VALUE_STAGE, NOTIFY_INVALID_WAIT_TIME, machinePara_.localUserrank,
        machinePara_.remoteWorldRank));
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::PostFin(Stream& stream)
{
    // 发送data notify同步信息
    void* remoteNotifyaddr = remoteDataNotifyMsg_.addr;
    HcclResult ret = TxSendWqe(
        remoteNotifyaddr, notifyValueMem_[machinePara_.deviceLogicId].ptr(), notifySize_, stream,
        WqeType::WQE_TYPE_DATA_NOTIFY);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TransportIbverbs][PostFin]errNo[0x%016llx] In ibv tx data signal, send notify "
            "wqe failed. dstMemPtr[%p], srcMemPtr[%p], srcMemSize[%llu]",
            HCCL_ERROR_CODE(ret), remoteNotifyaddr, notifyValueMem_[machinePara_.deviceLogicId].ptr(), notifySize_),
        ret);
    // 每发送一个data notify wqe, count 自增
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::WaitFin(Stream& stream)
{
    CHK_RET(LocalIpcNotify::Wait(
        stream, dispatcher_, dataNotify_, INVALID_VALUE_STAGE, NOTIFY_INVALID_WAIT_TIME, machinePara_.localUserrank,
        machinePara_.remoteWorldRank));
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::PostFinAck(Stream& stream)
{
    CHK_RET(TxSendWqe(
        remoteMemMsg_[static_cast<u32>(MemType::DATA_ACK_NOTIFY_MEM)].addr,
        notifyValueMem_[machinePara_.deviceLogicId].ptr(), notifySize_, stream, WqeType::WQE_TYPE_DATA_ACK_NOTIFY));
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::WaitFinAck(Stream& stream)
{
    CHK_RET(LocalIpcNotify::Wait(
        stream, dispatcher_, dataAckNotify_, INVALID_VALUE_STAGE, NOTIFY_INVALID_WAIT_TIME, machinePara_.localUserrank,
        machinePara_.remoteWorldRank));
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::Post(u32 notifyIdx, Stream& stream)
{
    // 校验notifyIdx有效性
    bool bRet = (notifyIdx >= notifyNum_);
    CHK_PRT_RET(
        bRet,
        HCCL_ERROR(
            "[TransportIbverbs][Post]notifyNum[%u], notifyIdx[%u] out of range[0, %u]", notifyNum_, notifyIdx,
            notifyNum_ - 1),
        HCCL_E_INTERNAL);

    // 每个QP发送一个指定idx的notify
    for (u32 i = 0; i < qpsPerConnection_; i++) {
        CHK_RET(TxSendNotifyWqe(
            userMultiQpRemoteNotifyMsg_[i][notifyIdx], notifyValueMem_[machinePara_.deviceLogicId].ptr(), notifySize_,
            stream));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::Wait(u32 notifyIdx, Stream& stream, const u32 timeOut)
{
    // 校验notifyIdx有效性
    bool bRet = (notifyIdx >= notifyNum_);
    CHK_PRT_RET(
        bRet,
        HCCL_ERROR(
            "[TransportIbverbs][Wait]notifyNum[%u], notifyIdx[%u] out of range[0, %u]", notifyNum_, notifyIdx,
            notifyNum_ - 1),
        HCCL_E_INTERNAL);

    // 每个qp接收一个指定idx的notify
    for (u32 i = 0; i < qpsPerConnection_; i++) {
        CHK_RET(LocalIpcNotify::Wait(
            stream, dispatcher_, userMultiQpLocalNotify_[i][notifyIdx], INVALID_VALUE_STAGE, timeOut,
            machinePara_.localUserrank, machinePara_.remoteWorldRank));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::GetLocalRdmaNotify(std::vector<HcclSignalInfo>& rdmaNotify)
{
    HcclSignalInfo signalInfo;
    CHK_SMART_PTR_NULL(ackNotify_);
    CHK_RET(ackNotify_->GetNotifyData(signalInfo));
    rdmaNotify.push_back(signalInfo);
    CHK_SMART_PTR_NULL(ackNotify_);
    CHK_RET(dataNotify_->GetNotifyData(signalInfo));
    rdmaNotify.push_back(signalInfo);
    CHK_SMART_PTR_NULL(ackNotify_);
    CHK_RET(dataAckNotify_->GetNotifyData(signalInfo));
    rdmaNotify.push_back(signalInfo);
    // 提取新增的notify资源
    for (u32 i = 0; i < qpsPerConnection_; i++) {
        for (u32 j = 0; j < notifyNum_; j++) {
            CHK_SMART_PTR_NULL(userMultiQpLocalNotify_[i][j]);
            CHK_RET(userMultiQpLocalNotify_[i][j]->GetNotifyData(signalInfo));
            rdmaNotify.push_back(signalInfo);
        }
        if (qpsPerConnection_ > 1) {
            CHK_SMART_PTR_NULL(multiQpDataNotify_[i]);
            CHK_RET(multiQpDataNotify_[i]->GetNotifyData(signalInfo));
            rdmaNotify.push_back(signalInfo);
        }
        HCCL_DEBUG(
            "[TransportIbverbs][GetLocalRdmaNotify] resId[%llu] addr[%llu]", rdmaNotify.back().resId,
            rdmaNotify.back().addr);
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::GetRemoteRdmaNotifyAddrKey(std::vector<AddrKey>& rdmaNotify)
{
    if (remoteMemMsg_[static_cast<u32>(MemType::ACK_NOTIFY_MEM)].addr == nullptr) {
        HCCL_ERROR("[TransportIbverbs][GetRemoteRdmaNotifyAddrKey] ackNotify is null!");
        return HCCL_E_PTR;
    } else if (remoteMemMsg_[static_cast<u32>(MemType::DATA_NOTIFY_MEM)].addr == nullptr) {
        HCCL_ERROR("[TransportIbverbs][GetRemoteRdmaNotifyAddrKey] dataNotify is null!");
        return HCCL_E_PTR;
    } else if (remoteMemMsg_[static_cast<u32>(MemType::DATA_ACK_NOTIFY_MEM)].addr == nullptr) {
        HCCL_ERROR("[TransportIbverbs][GetRemoteRdmaNotifyAddrKey] dataAckNotify is null!");
        return HCCL_E_PTR;
    }
    AddrKey notifyDetails;
    notifyDetails.addr = reinterpret_cast<u64>(remoteMemMsg_[static_cast<u32>(MemType::ACK_NOTIFY_MEM)].addr);
    notifyDetails.key = reinterpret_cast<u32>(remoteMemMsg_[static_cast<u32>(MemType::ACK_NOTIFY_MEM)].lkey);
    notifyDetails.notifyId = remoteMemMsg_[static_cast<u32>(MemType::ACK_NOTIFY_MEM)].notifyId;
    rdmaNotify.push_back(notifyDetails);
    notifyDetails.addr = reinterpret_cast<u64>(remoteMemMsg_[static_cast<u32>(MemType::DATA_NOTIFY_MEM)].addr);
    notifyDetails.key = reinterpret_cast<u32>(remoteMemMsg_[static_cast<u32>(MemType::DATA_NOTIFY_MEM)].lkey);
    notifyDetails.notifyId = remoteMemMsg_[static_cast<u32>(MemType::DATA_NOTIFY_MEM)].notifyId;
    rdmaNotify.push_back(notifyDetails);
    notifyDetails.addr = reinterpret_cast<u64>(remoteMemMsg_[static_cast<u32>(MemType::DATA_ACK_NOTIFY_MEM)].addr);
    notifyDetails.key = reinterpret_cast<u32>(remoteMemMsg_[static_cast<u32>(MemType::DATA_ACK_NOTIFY_MEM)].lkey);
    notifyDetails.notifyId = remoteMemMsg_[static_cast<u32>(MemType::DATA_ACK_NOTIFY_MEM)].notifyId;
    rdmaNotify.push_back(notifyDetails);

    // 获取新增的多notify资源
    for (u32 i = 0; i < qpsPerConnection_; i++) {
        for (u32 j = 0; j < notifyNum_; j++) {
            notifyDetails.addr = reinterpret_cast<u64>(userMultiQpRemoteNotifyMsg_[i][j].addr);
            notifyDetails.key = reinterpret_cast<u32>(userMultiQpRemoteNotifyMsg_[i][j].lkey);
            notifyDetails.notifyId = userMultiQpRemoteNotifyMsg_[i][j].notifyId;
            rdmaNotify.push_back(notifyDetails);
        }
        if (qpsPerConnection_ > 1) {
            notifyDetails.addr = reinterpret_cast<u64>(multiQpDataNotifyRemoteMemMsg_[i].addr);
            notifyDetails.key = reinterpret_cast<u32>(multiQpDataNotifyRemoteMemMsg_[i].lkey);
            notifyDetails.notifyId = multiQpDataNotifyRemoteMemMsg_[i].notifyId;
            rdmaNotify.push_back(notifyDetails);
        }
        HCCL_DEBUG(
            "[TransportIbverbs][GetRemoteRdmaNotifyAddrKey]remote addr[0x%llx], key[%lu], notifyId[%u]",
            rdmaNotify.back().addr, rdmaNotify.back().key, rdmaNotify.back().notifyId);
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::GetLocalNotifyValueAddrKey(std::vector<AddrKey>& notifyValue)
{
    if (memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].addr == nullptr) {
        HCCL_ERROR("[TransportIbverbs][GetLocalNotifyValueAddrKey] notifyValue is null!");
        return HCCL_E_PTR;
    }

    AddrKey notifyDetails;
    notifyDetails.addr = reinterpret_cast<u64>(memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].addr);
    notifyDetails.key = reinterpret_cast<u32>(memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].lkey);
    notifyValue.push_back(notifyDetails);
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::GetRemoteMemKey(UserMemType memType, uint32_t* remoteMemKey)
{
    switch (memType) {
        case UserMemType::INPUT_MEM:
        case UserMemType::OUTPUT_MEM:
            *remoteMemKey = remoteMemMsg_[static_cast<u32>(memType)].lkey;
            break;

        default:
            HCCL_ERROR("[Get][RemoteMemKey]not support dst_mem_type=%d", memType);
            return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::GetLocalMemDetails(UserMemType memType, MemDetails& memDetails)
{
    switch (memType) {
        case UserMemType::INPUT_MEM:
        case UserMemType::OUTPUT_MEM:
            memDetails.addr = reinterpret_cast<u64>(memMsg_[static_cast<u32>(memType)].addr);
            memDetails.size = memMsg_[static_cast<u32>(memType)].len;
            memDetails.key = memMsg_[static_cast<u32>(memType)].lkey;
            break;

        default:
            HCCL_ERROR("[Get][LocalMemDetails]not support dst_mem_type=%d", memType);
            return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::GetAiQpInfo(std::vector<HcclQpInfoV2>& aiQpInfo)
{
    aiQpInfo.resize(combineAiQpInfos_.size() + 1);

    aiQpInfo[0].qpPtr = combineAiQpInfo_.aiQpInfo.aiQpAddr;
    aiQpInfo[0].sqIndex = combineAiQpInfo_.aiQpInfo.sqIndex;
    aiQpInfo[0].dbIndex = combineAiQpInfo_.aiQpInfo.dbIndex;
    HCCL_DEBUG(
        "[TransportIbverbs][GetAiQpInfo] i[0] qpPtr[%llu] sqIndex[%u] dbIndex[%u]", aiQpInfo[0].qpPtr,
        aiQpInfo[0].sqIndex, aiQpInfo[0].dbIndex);
    for (u32 i = 1, j = 0; i < aiQpInfo.size(); i++, j++) {
        aiQpInfo[i].qpPtr = combineAiQpInfos_[j].aiQpInfo.aiQpAddr;
        aiQpInfo[i].sqIndex = combineAiQpInfos_[j].aiQpInfo.sqIndex;
        aiQpInfo[i].dbIndex = combineAiQpInfos_[j].aiQpInfo.dbIndex;
        HCCL_DEBUG(
            "[TransportIbverbs][GetAiQpInfo] i[%u] qpPtr[%llu] sqIndex[%u] dbIndex[%u]", i, aiQpInfo[i].qpPtr,
            aiQpInfo[i].sqIndex, aiQpInfo[i].dbIndex);
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::GetAiRMAQueueInfo(std::vector<HcclAiRMAQueueInfo>& aiRMAQueueInfo)
{
    bool isSupport = false;
    CHK_RET(IsSupportAIVNormalQP(machinePara_.localDeviceId, isSupport));
    CHK_PRT_RET(
        isSupport == false,
        HCCL_ERROR(
            "[IsSupportCQCoverNormalQP]"
            "devicePhyId[%u] not support",
            machinePara_.localDeviceId),
        HCCL_E_NOT_SUPPORT);

    u32 sl = GetExternalInputRdmaServerLevel();
    if (machinePara_.sl != HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET) {
        sl = machinePara_.sl;
    }
    HCCL_INFO(
        "[TransportIbverbs][GetAiRMAQueueInfo] localUserRank[%u], remoteUserrank[%u], sl[%u]",
        machinePara_.localUserrank, machinePara_.remoteUserrank, sl);
    aiRMAQueueInfo.resize(combineAiQpInfos_.size() + 1);
    CopyAiWQInfo(aiRMAQueueInfo[0].sq, combineAiQpInfo_.aiQpInfo.dataPlaneInfo.sq, DBMode::HW_DB, sl);
    CopyAiWQInfo(aiRMAQueueInfo[0].rq, combineAiQpInfo_.aiQpInfo.dataPlaneInfo.rq, DBMode::SW_DB, sl);
    CopyAiCQInfo(aiRMAQueueInfo[0].scq, combineAiQpInfo_.aiQpInfo.dataPlaneInfo.scq, DBMode::SW_DB);
    CopyAiCQInfo(aiRMAQueueInfo[0].rcq, combineAiQpInfo_.aiQpInfo.dataPlaneInfo.rcq, DBMode::SW_DB);

    // 预留多QP的能力; 当前主要是单QP场景
    for (u32 i = 1, j = 0; i < aiRMAQueueInfo.size(); i++, j++) {
        CopyAiWQInfo(aiRMAQueueInfo[i].sq, combineAiQpInfos_[j].aiQpInfo.dataPlaneInfo.sq, DBMode::HW_DB, sl);
        CopyAiWQInfo(aiRMAQueueInfo[i].rq, combineAiQpInfos_[j].aiQpInfo.dataPlaneInfo.rq, DBMode::SW_DB, sl);
        CopyAiCQInfo(aiRMAQueueInfo[i].scq, combineAiQpInfos_[j].aiQpInfo.dataPlaneInfo.scq, DBMode::SW_DB);
        CopyAiCQInfo(aiRMAQueueInfo[i].rcq, combineAiQpInfos_[j].aiQpInfo.dataPlaneInfo.rcq, DBMode::SW_DB);
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::WriteCommon(
    const void* remoteAddr, const void* localAddr, u64 length, Stream& stream, WqeType wqeType, struct WrAuxInfo& aux)
{
    // 单qp & 多qp
    std::vector<WqeInfo> wqeInfoVec;
    wqeInfoVec.reserve(WQE_RESERVE_LENGTH);
    HCCL_DEBUG("write localAddr[%p] remoteAddr[%p] len[%llu] remoteOffset[%llu]", localAddr, remoteAddr, length);

    if (localAddr != nullptr) {
        // 为保证单算子下不同数据量下子图的结构相同，zero byte message 时也需要下发task
        u32 txSendDataTimes = (length == 0) ? 1 : (length + RDMA_SEND_MAX_SIZE - 1) / RDMA_SEND_MAX_SIZE;
        CHK_RET(ConstructPayLoadWqe(
            const_cast<void*>(remoteAddr), const_cast<void*>(localAddr), length, wqeType, aux, wqeInfoVec,
            txSendDataTimes));
    }

    u32 maxLength = 0;
    for (u32 i = 0; i < wqeInfoVec.size(); i++) {
        if (wqeInfoVec[i].wqeData.memList.len > maxLength) {
            maxLength = wqeInfoVec[i].wqeData.memList.len;
        }
    }

    u32 actualMultiQpNum = GetActualQpNum(maxLength);

    HCCL_DEBUG(
        "[TransportIbverbs][TxSendDataAndNotify] UseMultiQp[%d] MultiQpNum[%u] actualMultiQpNum[%u] maxLength[%u]",
        UseMultiQp(), qpsPerConnection_, actualMultiQpNum, maxLength);
    if (UseMultiQp() && actualMultiQpNum != 1 && actualMultiQpNum <= qpsPerConnection_ && maxLength != 0) {
        std::vector<std::vector<WqeInfo>> multiQpWqeInfoVct(actualMultiQpNum, wqeInfoVec);
        for (u32 i = 0; i < wqeInfoVec.size(); i++) {
            WqeInfo tmpWqeInfo = wqeInfoVec[i];
            u32 curLen = tmpWqeInfo.wqeData.memList.len;
            std::vector<u32> splittedLen = RdmaLengthSplit(curLen, actualMultiQpNum);
            uint64_t curSrcAddr = tmpWqeInfo.wqeData.memList.addr;
            uint64_t curDstAddr = tmpWqeInfo.wqeData.dstAddr;
            for (u32 qpIndex = 0; qpIndex < actualMultiQpNum; qpIndex++) {
                multiQpWqeInfoVct[qpIndex][i].wqeData.memList.len = splittedLen[qpIndex];
                multiQpWqeInfoVct[qpIndex][i].wqeData.memList.addr = curSrcAddr;
                multiQpWqeInfoVct[qpIndex][i].wqeData.dstAddr = curDstAddr;
                curSrcAddr += splittedLen[qpIndex];
                curDstAddr += splittedLen[qpIndex];
            }
        }

        // useOneDoorbell 配置成true。最后一个payload去按doorbell
        for (u32 qpIndex = 0; qpIndex < actualMultiQpNum; qpIndex++) {
            CHK_RET(RdmaSendAsync(
                multiQpWqeInfoVct[qpIndex], stream, true, qpIndex)); // 多QP使用同一个stream异步doorbell触发
        }
    } else {
        if (machinePara_.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
            CHK_RET(RdmaSendAsync(wqeInfoVec, stream, GetUseOneDoorbellValue()));
        } else {
            CHK_RET(RdmaSendAsyncHostNIC(wqeInfoVec, stream));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult
TransportIbverbs::WriteAsync(struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, Stream& stream)
{
    struct WrAuxInfo aux = {};
    return WriteCommon(remoteBuf.addr, localBuf.addr, remoteBuf.size, stream, WqeType::WQE_TYPE_DATA, aux);
}

HcclResult TransportIbverbs::WriteReduceAsync(
    struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, const HcclDataType datatype,
    HcclReduceOp redOp, Stream& stream)
{
    struct WrAuxInfo aux = {};
    aux.dataType = RDMA_REDUCE_DATA_TYPE_TABLE[datatype];
    aux.reduceType = RDMA_REDUCE_OP_TYPE_TABLE[redOp];
    if (aux.dataType == static_cast<uint8_t>(RdmaReduceDataType::RDMA_REDUCE_DATA_INVALID)
        || aux.reduceType == static_cast<uint8_t>(RdmaReduceOpType::RDMA_REDUCE_OP_INVALID)) {
        HCCL_ERROR(
            "unsupported data type [%s] or Reduce type [%s]", GetDataTypeEnumStr(datatype).c_str(),
            GetReduceOpEnumStr(redOp).c_str());
        return HCCL_E_INTERNAL;
    }

    return WriteCommon(remoteBuf.addr, localBuf.addr, remoteBuf.size, stream, WqeType::WQE_TYPE_DATA_WITH_REDUCE, aux);
}

HcclResult TransportIbverbs::WriteSync(
    [[maybe_unused]] struct Transport::Buffer& remoteBuf, [[maybe_unused]] struct Transport::Buffer& localBuf,
    [[maybe_unused]] Stream& stream)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
TransportIbverbs::ReadAsync(struct Transport::Buffer& localBuf, struct Transport::Buffer& remoteBuf, Stream& stream)
{
    struct WrAuxInfo aux = {};
    return WriteCommon(remoteBuf.addr, localBuf.addr, remoteBuf.size, stream, WqeType::WQE_TYPE_READ_DATA, aux);
}

HcclResult TransportIbverbs::ReadSync(
    [[maybe_unused]] struct Transport::Buffer& localBuf, [[maybe_unused]] struct Transport::Buffer& remoteBuf,
    [[maybe_unused]] Stream& stream)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportIbverbs::GetLocalNotify(std::vector<HcclSignalInfo>& localNotify)
{
    HcclSignalInfo notifyInfo;
    CHK_SMART_PTR_NULL(dataNotify_);
    CHK_RET(dataNotify_->GetNotifyData(notifyInfo));
    localNotify.push_back(notifyInfo);

    CHK_SMART_PTR_NULL(ackNotify_);
    CHK_RET(ackNotify_->GetNotifyData(notifyInfo));
    localNotify.push_back(notifyInfo);

    CHK_SMART_PTR_NULL(dataAckNotify_);
    CHK_RET(dataAckNotify_->GetNotifyData(notifyInfo));
    localNotify.push_back(notifyInfo);

    // 提取新增的notify资源
    for (u32 i = 0; i < qpsPerConnection_; i++) {
        for (u32 j = 0; j < notifyNum_; j++) {
            CHK_SMART_PTR_NULL(userMultiQpLocalNotify_[i][j]);
            CHK_RET(userMultiQpLocalNotify_[i][j]->GetNotifyData(notifyInfo));
            localNotify.push_back(notifyInfo);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::GetTransportErrorCqe(
    const HcclNetDevCtx netDevCtx, std::vector<std::pair<TransportBase*, CqeInfo>>& infos, u32& num)
{
    if (g_qpn2IbversLinkMap_.Size() == 0) {
        num = 0;
        return HCCL_SUCCESS;
    }

    if (UNLIKELY(!g_flag)) {
        CHK_RET(IsSuppCqeErrInfoListConfig(g_isSupCqeErrInfoListConfig));
        g_flag = true;
    }

    CHK_PTR_NULL(netDevCtx);
    s32 deviceLogicId = (static_cast<NetDevContext*>(netDevCtx))->GetLogicId();
    s32 devicePhyId = (static_cast<NetDevContext*>(netDevCtx))->GetPhyId();
    HcclIpAddress localIp = (static_cast<NetDevContext*>(netDevCtx))->GetLocalIp();
    NicType nicType = (static_cast<NetDevContext*>(netDevCtx))->GetNicType();
    CHK_PRT_RET(
        nicType == NicType::HOST_NIC_TYPE,
        HCCL_WARNING("[TransportIbverbs][GetTransportErrorCqe] nicType[%d] not support", nicType), HCCL_SUCCESS);
    RaResourceInfo raResourceInfo;
    CHK_RET(NetworkManager::GetInstance(deviceLogicId).GetRaResourceInfo(raResourceInfo));
    RdmaHandle rdmaHandle = raResourceInfo.nicSocketMap[localIp].nicRdmaHandle;
    CHK_PTR_NULL(rdmaHandle);

    if (g_isSupCqeErrInfoListConfig) {
        u32 loop = 0;
        if (num > CQE_ARRAY_SIZE) {
            loop = (num % CQE_ARRAY_SIZE != 0) ? (num / CQE_ARRAY_SIZE) : ((num / CQE_ARRAY_SIZE) - 1);
        }

        struct CqeErrInfo infolist[CQE_ARRAY_SIZE] = {};
        u32 cqeNum = CQE_ARRAY_SIZE;
        for (u32 index = 0; index <= loop; index++) {
            cqeNum = (index == loop) ? (num - index * CQE_ARRAY_SIZE) : CQE_ARRAY_SIZE;
            u32 temNum = cqeNum;
            CHK_RET(hrtRaGetCqeErrInfoList(rdmaHandle, infolist, &temNum));
            ProcessCqeInfo(devicePhyId, infolist, temNum, infos);
            if (temNum < cqeNum) {
                break;
            }
        }
    } else {
        struct CqeErrInfo infolist[1] = {};
        CHK_RET(hrtRaGetCqeErrInfo(devicePhyId, &infolist[0]));
        if (infolist[0].status == 0) {
            num = 0;
            return HCCL_SUCCESS;
        }
        u32 cqeNum = 1;
        ProcessCqeInfo(devicePhyId, infolist, cqeNum, infos);
    }

    num = infos.size();

    return HCCL_SUCCESS;
}

void TransportIbverbs::ProcessCqeInfo(
    const s32 deviceId, const struct CqeErrInfo* infolist, const u32 cqeNum,
    std::vector<std::pair<TransportBase*, CqeInfo>>& infos)
{
    for (u32 i = 0; i < cqeNum; i++) {
        // localPhyId + qpn
        auto it = g_qpn2IbversLinkMap_.Find(((static_cast<u64>(deviceId) << DEV_PHY_ID_BIT) | infolist[i].qpn));
        if (it.second) {
            TransportBase* ptr = reinterpret_cast<TransportBase*>(it.first->second);
            infos.push_back(
                std::make_pair(ptr, CqeInfo(infolist[i].time, infolist[i].status, it.first->second->GetRemoteIp())));
            cqeErrQpn_ = infolist[i].qpn;
        } else {
            HCCL_RUN_WARNING("[GetTransportErrorCqe]get err failed, transport was not found.");
        }
    }
    return;
}

HcclIpAddress& TransportIbverbs::GetRemoteIp() { return machinePara_.remoteIpAddr; }

HcclResult TransportIbverbs::GetTransportId(u32& id)
{
    id = cqeErrQpn_;
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::ExchangeCapabilityHybrid()
{
    HCCL_INFO("[Hybrid][TransportIbverbs] Starting capability exchange");

    // 1. 构造本地能力信息（使用公共头文件中的默认值）
    using namespace hcomm;
    RoCECapability localCap;
    localCap.InitDefaults();
    localCap.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    localCap.commStack = CommStackType::COMM_STACK_TRANSPORT_IBVERBS;

    // 2. 发送本地能力（合并为单次发送：totalLength已包含结构体大小）
    CHK_RET(defaultSocket_->Send(&localCap, sizeof(localCap)));
    HCCL_INFO("[Hybrid][TransportIbverbs] Sent capability, version=%u", localCap.version);

    // 3. 接收对端能力（单次接收）
    RoCECapability recvCap;
    CHK_RET(defaultSocket_->Recv(&recvCap, sizeof(recvCap)));

    // 4. 先检查魔数，如果不对可能是旧版本，需要回退
    if (!RoCECapability::CheckMagic(reinterpret_cast<uint8_t*>(&recvCap), sizeof(recvCap))) {
        HCCL_WARNING("[Hybrid][TransportIbverbs] Magic mismatch, peer may be old version. "
                     "Falling back to native mode.");
        // 回退到原生模式
        isHybridMode_ = false;
        return HCCL_SUCCESS;
    }

    // 5. 魔数正确，解析对端能力
    RoCECapability remoteCap;
    if (!remoteCap.Deserialize(reinterpret_cast<uint8_t*>(&recvCap), sizeof(recvCap))) {
        HCCL_ERROR("[Hybrid][TransportIbverbs] Failed to deserialize capability");
        return HCCL_E_PARA;
    }

    // 6. 校验字段有效性
    if (!remoteCap.Validate()) {
        HCCL_ERROR("[Hybrid][TransportIbverbs] Capability validation failed");
        return HCCL_E_INTERNAL;
    }

    // 7. 版本兼容性处理（高版本兼容低版本）
    if (remoteCap.version > ROCE_CAPABILITY_VERSION) {
        // 对端版本更高，使用本地版本的功能集（最小公分母）
        HCCL_INFO(
            "[Hybrid][TransportIbverbs] Remote version %u > local %u, using local version features", remoteCap.version,
            ROCE_CAPABILITY_VERSION);
    } else if (remoteCap.version < ROCE_CAPABILITY_VERSION) {
        // 对端版本更低，使用对端版本的功能集（向下兼容）
        HCCL_INFO(
            "[Hybrid][TransportIbverbs] Remote version %u < local %u, using remote version features", remoteCap.version,
            ROCE_CAPABILITY_VERSION);
    }

    isHybridMode_ = (remoteCap.commStack == CommStackType::COMM_STACK_HOST_CPU_ROCE) ? true : false;
    HCCL_INFO(
        "[Hybrid][TransportIbverbs]Exchange mode success, Remote is %s",
        isHybridMode_ ? "HostCpuRoceChannel" : "TransportIbverbs");

    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::GetDrainRemSrcMem(void*& remoteAddr, uint32_t& remoteKey, uint32_t& size)
{
    auto opType = static_cast<u32>(MemType::NOTIFY_SRC_MEM);
    remoteAddr = remoteMemMsg_[opType].addr;
    remoteKey = remoteMemMsg_[opType].lkey;
    size = remoteMemMsg_[opType].len;
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::Drain(Stream& stream)
{
    auto remoteType = static_cast<u32>(MemType::NOTIFY_SRC_MEM);
    auto localType = static_cast<u32>(MemType::DATA_NOTIFY_MEM);
    CHK_PTR_NULL(memMsg_[localType].addr);
    CHK_PTR_NULL(remoteMemMsg_[remoteType].addr);
    CHK_RET(Fence());
    std::vector<WqeInfo> wqeInfoVec;
    WrAuxInfo aux = {};
    CHK_RET(AddWqeList(
        remoteMemMsg_[remoteType].addr, memMsg_[localType].addr, memMsg_[localType].len, WqeType::WQE_TYPE_READ_DATA,
        aux, wqeInfoVec));
    CHK_RET(RdmaSendAsync(wqeInfoVec, stream, false));
    CHK_SMART_PTR_NULL(dataNotify_);
    CHK_RET(LocalIpcNotify::Wait(
        stream, dispatcher_, dataNotify_, INVALID_VALUE_STAGE, NOTIFY_INVALID_WAIT_TIME, machinePara_.localUserrank,
        machinePara_.remoteWorldRank));
    return HCCL_SUCCESS;
}

HcclResult TransportIbverbs::GetDrainLocalDataNotify(void*& localAddr, uint32_t& lkey, HcclSignalInfo& dataNotify)
{
    localAddr = memMsg_[DATA_NOTIFY_MEM].addr;
    lkey = memMsg_[DATA_NOTIFY_MEM].lkey;
    CHK_SMART_PTR_NULL(dataNotify_);
    CHK_RET(dataNotify_->GetNotifyData(dataNotify));
    return HCCL_SUCCESS;
}
} // namespace hccl
