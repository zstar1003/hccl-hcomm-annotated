/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include "hccl_common.h"
#include "stream_pub.h"
#include "aicpu/aicpu_hccl_sqcqv1.h"
#include "aicpu/aicpu_hccl_sqcqv2.h"
#include "config_plf_log.h"
#include "dlhal_function.h"
#include "adapter_hal_pub.h"
#include "adapter_rts.h"
#include "dispatcher_aicpu.h"

namespace hccl {
constexpr uint64_t NANOSECOND_TO_SECOND = 1000000000U;
constexpr uint16_t TURN_LEFT_SHIFT_BIT = 16;
constexpr uint32_t HCCL_PER_LAUNCH_SQE_CNT = 128U; // 每编排N个SQE，做一次launchtask

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
HcclResult
HcclDispatcherAicpuInit(HcclDispatcher* dispatcher, const u32 devPhyId, uint32_t hcclQos, DispatcherType type)
{
    CHK_PTR_NULL(dispatcher);
    DispatcherPub* pDispatcher = nullptr;
    if (type == DispatcherType::DISPATCHER_AICPU) {
        pDispatcher = new (std::nothrow) DispatcherAiCpu(devPhyId);
        pDispatcher->SetHcclQos(hcclQos);
        pDispatcher->SetMpamid(0);
        HCCL_INFO("HcclDispatcherAicpuInit hcclQos = %u", pDispatcher->GetHcclQos());
    } else {
        HCCL_ERROR("[HcclCommAicpu][HcclDispatcherInit] Not support the dispatcher type[%d]", type);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PTR_NULL(pDispatcher);
    HcclResult ret = pDispatcher->Init();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclCommAicpu][HcclDispatcherInit] Dispatcher init failed, type[%d]", type);
        delete pDispatcher;
        pDispatcher = nullptr;
        return ret;
    }
    *dispatcher = pDispatcher;
    return HCCL_SUCCESS;
}
#ifdef __cplusplus
}
#endif // __cplusplus

DispatcherAiCpu::DispatcherAiCpu(const u32 devPhyId) // deprecated
    : DispatcherPub(INVALID_INT)
{
    aicpuInfo_.devId = devPhyId;
}

DispatcherAiCpu::~DispatcherAiCpu() {}

HcclResult DispatcherAiCpu::Init()
{
    CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());
    CHK_RET(hrtHalGetDeviceType(aicpuInfo_.devId, aicpuInfo_.devType));
    CHK_RET(hrtHalGetDeviceInfo(
        aicpuInfo_.devId, MODULE_TYPE_SYSTEM, INFO_TYPE_PHY_CHIP_ID, reinterpret_cast<int64_t*>(&aicpuInfo_.chipId)));

    if (aicpuInfo_.devType == DevType::DEV_TYPE_310P1 || aicpuInfo_.devType == DevType::DEV_TYPE_310P3) {
        addOneNotifyWaitSqe_ = AddOneNotifyWaitSqeV2;
        addOneRecordSqe_ = AddOneRecordSqeV2;
        addOneWriteValueRecordSqe_ = AddOneWriteValueRecordSqeV2;
        addOneMemcpySqe_ = AddOneMemcpySqeV2;
        addOneEventResetSqe_ = AddOneEventResetSqeV2;
        addOneEventRecordSqe_ = AddOneEventRecordSqeV2;
        addOneEventWaitSqe_ = AddOneEventWaitSqeV2;
    } else {
        addOneNotifyWaitSqe_ = AddOneNotifyWaitSqeV1;
        addOneRecordSqe_ = AddOneRecordSqeV1;
        addOneWriteValueRecordSqe_ = AddOneWriteValueRecordSqeV1;
        addOneMemcpySqe_ = AddOneMemcpySqeV1;
        addOneEventResetSqe_ = AddOneEventResetSqeV1;
        addOneEventRecordSqe_ = AddOneEventRecordSqeV1;
        addOneEventWaitSqe_ = AddOneEventWaitSqeV1;
        addOneRdmaDbSendSqe_ = AddOneRdmaDbSendSqeV1;
        addOneFlipPlaceHolderSqe_ = AddOneFlipPlaceHolderSqeV1;
        addOneCacheMemcpyPlaceHolderSqe_ = AddOneCacheMemcpyPlaceHolderSqeV1;
        addOneCacheNotifyWaitPlaceholderSqe_ = AddOneCacheNotifyWaitPlaceholderSqeV1;
        addOneCacheNotifyRecordPlaceholderSqe_ = AddOneCacheNotifyRecordPlaceholderSqeV1;
        addOneCacheWriteValuePlaceholderSqe_ = AddOneCacheWriteValuePlaceholderSqeV1;
        addOneCacheMemcpyRecordPlaceholderSqe_ = AddOneCacheMemcpyRecordPlaceholderSqeV1;
        CHK_PTR_NULL(addOneRdmaDbSendSqe_);
        CHK_PTR_NULL(addOneFlipPlaceHolderSqe_);
        CHK_PTR_NULL(addOneCacheMemcpyPlaceHolderSqe_);
        CHK_PTR_NULL(addOneCacheNotifyWaitPlaceholderSqe_);
        CHK_PTR_NULL(addOneCacheNotifyRecordPlaceholderSqe_);
        CHK_PTR_NULL(addOneCacheWriteValuePlaceholderSqe_);
        CHK_PTR_NULL(addOneCacheMemcpyRecordPlaceholderSqe_);
    }

    CHK_PTR_NULL(addOneNotifyWaitSqe_);
    CHK_PTR_NULL(addOneRecordSqe_);
    CHK_PTR_NULL(addOneWriteValueRecordSqe_);
    CHK_PTR_NULL(addOneMemcpySqe_);
    CHK_PTR_NULL(addOneEventResetSqe_);
    CHK_PTR_NULL(addOneEventRecordSqe_);
    CHK_PTR_NULL(addOneEventWaitSqe_);

    CHK_RET(GetNotifyMaxWaitTime());
    // 和hrtGetNotifySize接口保持一致，910B和910_93的notify寄存器大小为4字节，其他芯片为8字节
    notifySize_ = (aicpuInfo_.devType == DevType::DEV_TYPE_910B || aicpuInfo_.devType == DevType::DEV_TYPE_910_93) ?
                      NOTIFY_SIZE_BYTES_910B :
                      NOTIFY_SIZE_BYTES_910A;
    InitTimeOutConfig();
    HCCL_INFO(
        "%s success, devId:%u, devType:%d, chipId:%lld", __func__, aicpuInfo_.devId, aicpuInfo_.devType,
        aicpuInfo_.chipId);
    return HCCL_SUCCESS;
}

HcclResult DispatcherAiCpu::WaitValue(hccl::Stream& stream, u64 waitAddr, u64 valueAddr, bool reset)
{
    u32 turnNum = *(reinterpret_cast<u32*>(static_cast<uintptr_t>(valueAddr)));
    HCCL_DEBUG("[DispatcherAiCpu][WaitValue] turnNum %u", turnNum);
    uint8_t* sqeBuffer = nullptr;
    uint8_t* sqeTypeAddr = nullptr;
    uint8_t* sqeDfxInfoAddr = nullptr;
    uint16_t taskId = 0U;
    CHK_RET(stream.GetNextSqeBufferAddr(sqeBuffer, sqeTypeAddr, sqeDfxInfoAddr, taskId));
    const HcclComStreamInfo* streamInfo;
    CHK_RET(stream.GetStreamInfo(streamInfo));
    AddOneWaitStartSqe(
        streamInfo->actualStreamId, taskId, waitAddr, valueAddr, reset,
        reinterpret_cast<rtStarsCcoreWaitStartSqe_t*>(sqeBuffer), sqeTypeAddr);
    HcclSqeContext* sqeCtx = stream.GetSqeContextPtr();
    if (sqeCtx == nullptr) {
        HCCL_ERROR("[DispatcherAiCpu][WaitValue] AddCcoreWait sqeCtx is nullptr");
        return HCCL_E_INTERNAL;
    }
    sqeCtx->buffer.addInfo[taskId % hccl::HCCL_SQE_MAX_CNT]
        = ((turnNum << TURN_LEFT_SHIFT_BIT) + static_cast<uint32_t>(reset));
    return HCCL_SUCCESS;
}

HcclResult DispatcherAiCpu::WriteValue(hccl::Stream& stream, u64 writeAddr, u64 valueAddr)
{
    u32 turnNum = *(reinterpret_cast<u32*>(static_cast<uintptr_t>(valueAddr)));
    HCCL_DEBUG("[DispatcherAiCpu][WriteValue] turnNum %u", turnNum);
    uint8_t* sqeBuffer = nullptr;
    uint8_t* sqeTypeAddr = nullptr;
    uint8_t* sqeDfxInfoAddr = nullptr;
    uint16_t taskId = 0U;
    CHK_RET(stream.GetNextSqeBufferAddr(sqeBuffer, sqeTypeAddr, sqeDfxInfoAddr, taskId));
    const HcclComStreamInfo* streamInfo;
    CHK_RET(stream.GetStreamInfo(streamInfo));
    AddOneWriteValueStartSqe(
        streamInfo->actualStreamId, taskId, writeAddr, valueAddr,
        reinterpret_cast<rtStarsCcoreWriteValueSqe_t*>(sqeBuffer), sqeTypeAddr);
    HcclSqeContext* sqeCtx = stream.GetSqeContextPtr();
    if (sqeCtx == nullptr) {
        HCCL_ERROR("[DispatcherAiCpu][WriteValue] AddCcoreNotify sqeCtx is nullptr");
        return HCCL_E_INTERNAL;
    }
    sqeCtx->buffer.addInfo[taskId % hccl::HCCL_SQE_MAX_CNT] = turnNum;
    return HCCL_SUCCESS;
}

HcclResult DispatcherAiCpu::SignalRecord(
    [[maybe_unused]] HcclRtNotify signal, hccl::Stream& stream, u32 userRank, [[maybe_unused]] u64 offset,
    [[maybe_unused]] s32 stage, bool inchip, u64 signalAddr, u32 notifyId)
{
    const HcclComStreamInfo& streamInfo = stream.GetHcclStreamInfo();
    uint8_t* sqeBuffer = nullptr;
    uint8_t* sqeTypeAddr = nullptr;
    uint8_t* sqeDfxInfoAddr = nullptr;
    uint16_t taskId = 0U;
    CHK_RET(GetStreamSqeBufferAddr(stream, sqeBuffer, sqeTypeAddr, sqeDfxInfoAddr, taskId));
    AicpuDfxInfo* const dfxInfo = (AicpuDfxInfo*)sqeDfxInfoAddr;
    dfxInfo->opRingBufferIdx = opRingBufferIdx_;
    dfxInfo->remoteRank = userRank;
    dfxInfo->notifyId = notifyId;

    // 检查是否需要生成cache-write placeholder
    if (isPlaceholder_) {
        // 只有使能alltoallv类算子的aicpu cache才需要cache-write placeholder
        CHK_PRT_RET(
            !(needAddSqe_ && isAlltoallv_),
            HCCL_ERROR(
                "[DispatcherAiCpu][SignalRecord] isPlaceholder_[%u] "
                "needAddSqe_[%u] isAlltoallv_[%u]",
                isPlaceholder_, needAddSqe_, isAlltoallv_),
            HCCL_E_INTERNAL);

        // 只有p2p的SignalRecord才需要cache-write placeholder
        // 不需要为inchip场景下LocalNotify::Post的SignalRecord调用addOneCacheNotifyRecordPlaceholderSqe_生成cache-notify
        // placeholder
        CHK_PRT_RET(
            inchip,
            HCCL_ERROR("[DispatcherAiCpu][SignalRecord] isPlaceholder_[%u], inchip[%u]", isPlaceholder_, inchip),
            HCCL_E_INTERNAL);

        addOneCacheWriteValuePlaceholderSqe_(streamInfo.actualStreamId, taskId, signalAddr, sqeBuffer, sqeTypeAddr);
        HCCL_INFO("%s generate cache-write placeholder for signal record", __func__);
    } else { // 正常生成SQE
        if (inchip) {
            addOneRecordSqe_(streamInfo.actualStreamId, taskId, notifyId, sqeBuffer, sqeTypeAddr);
        } else {
            addOneWriteValueRecordSqe_(streamInfo.actualStreamId, taskId, signalAddr, sqeBuffer, sqeTypeAddr);
        }
    }

    PLF_CONFIG_INFO(
        PLF_TASK, "%s para: streamId[%d] remoteRank[%u] inchip[%d] devType[%d] notifyId[%u]", __func__,
        streamInfo.actualStreamId, userRank, inchip, aicpuInfo_.devType, notifyId);
    return HCCL_SUCCESS;
}

HcclResult DispatcherAiCpu::SignalRecord(
    hccl::DeviceMem& dst, hccl::DeviceMem& src, hccl::Stream& stream, u32 remoteUserRank, hccl::LinkType inLinkType,
    u32 notifyId)
{
    // 参数有效性检查
    CHK_PRT_RET(src.size() == 0, HCCL_INFO("%s src size is 0, not need copy", __func__), HCCL_SUCCESS);
    CHK_PRT_RET(src == dst, HCCL_INFO("%s src and dst is same, not need copy", __func__), HCCL_SUCCESS);
    CHK_PRT_RET(
        dst.size() < src.size(),
        HCCL_ERROR(
            "%s The size of dst is smaller than that of src. dst addr:%p, dst size:%llu, src addr:%p, "
            "src size:%llu",
            __func__, dst.ptr(), dst.size(), src.ptr(), src.size()),
        HCCL_E_PARA);
    CHK_PRT_RET(
        src.size() != notifySize_,
        HCCL_ERROR(
            "%s src size[%llu] should be %llu in devType[%d]", __func__, src.size(), notifySize_, aicpuInfo_.devType),
        HCCL_E_PARA);

    const HcclComStreamInfo& streamInfo = stream.GetHcclStreamInfo();
    uint8_t* sqeBuffer = nullptr;
    uint8_t* sqeTypeAddr = nullptr;
    uint8_t* sqeDfxInfoAddr = nullptr;
    uint16_t taskId = 0U;
    aclrtReduceKind rtReduceOp = RK_MAP_TABLE[HCCL_REDUCE_RESERVED];
    uint8_t linkType = static_cast<uint8_t>(inLinkType);

    CHK_RET(GetStreamSqeBufferAddr(stream, sqeBuffer, sqeTypeAddr, sqeDfxInfoAddr, taskId));
    AicpuDfxInfo* const dfxInfo = reinterpret_cast<AicpuDfxInfo*>(sqeDfxInfoAddr);
    dfxInfo->opRingBufferIdx = opRingBufferIdx_;
    dfxInfo->remoteRank = remoteUserRank;
    dfxInfo->notifyId = notifyId;

    // 检查是否需要生成cache-memcpy-record placeholder
    if (isPlaceholder_) {
        // 只有使能alltoallv类算子的aicpu cache才需要cache-memcpy-record placeholder
        CHK_PRT_RET(
            !(needAddSqe_ && isAlltoallv_),
            HCCL_ERROR(
                "[DispatcherAiCpu][SignalRecord] isPlaceholder_[%u] "
                "needAddSqe_[%u] isAlltoallv_[%u]",
                isPlaceholder_, needAddSqe_, isAlltoallv_),
            HCCL_E_INTERNAL);

        addOneCacheMemcpyRecordPlaceholderSqe_(
            streamInfo.actualStreamId, taskId, src.ptr(), src.size(), ACL_FLOAT, rtReduceOp, dst.ptr(), 0,
            aicpuInfo_.ssid, aicpuInfo_.devId, aicpuInfo_.overflowAddr, linkType, sqeBuffer, sqeTypeAddr,
            SDMA_QOS_DEFAULT);
        HCCL_INFO("%s generate cache-memcpy-record placeholder for signal record", __func__);
    } else { // 正常生成memcpy record SQE
        addOneMemcpySqe_(
            streamInfo.actualStreamId, taskId, src.ptr(), src.size(), ACL_FLOAT, rtReduceOp, dst.ptr(), 0,
            aicpuInfo_.ssid, aicpuInfo_.devId, aicpuInfo_.overflowAddr, linkType, sqeBuffer, sqeTypeAddr,
            SDMA_QOS_DEFAULT);
    }

    PLF_CONFIG_INFO(
        PLF_TASK,
        "%s para: linkType[%u] srcPtr[%p] srcSize[%llu] dstPtr[%p] taskId[%u] streamId[%u] remoteRank[%u] notifyId[%u] "
        "hcclQos[%u]",
        __func__, linkType, src.ptr(), src.size(), dst.ptr(), taskId, streamInfo.actualStreamId, remoteUserRank,
        notifyId, SDMA_QOS_DEFAULT);
    return HCCL_SUCCESS;
}

HcclResult DispatcherAiCpu::SignalWait(
    [[maybe_unused]] HcclRtNotify signal, Stream& stream, u32 userRank, u32 remoteUserRank, [[maybe_unused]] s32 stage,
    bool inchip, u32 notifyId, u32 timeOut)
{
    const HcclComStreamInfo& streamInfo = stream.GetHcclStreamInfo();
    uint8_t* sqeBuffer = nullptr;
    uint8_t* sqeTypeAddr = nullptr;
    uint8_t* sqeDfxInfoAddr = nullptr;
    uint16_t taskId = 0U;
    CHK_RET(GetStreamSqeBufferAddr(stream, sqeBuffer, sqeTypeAddr, sqeDfxInfoAddr, taskId));
    AicpuDfxInfo* const dfxInfo = (AicpuDfxInfo*)sqeDfxInfoAddr;
    dfxInfo->opRingBufferIdx = opRingBufferIdx_;
    dfxInfo->remoteRank = remoteUserRank;
    dfxInfo->notifyId = notifyId;

    dfxTimeOutConfig_.sqeTimeOutTimeOut = timeOut < notifyMaxWaitTime_ ? timeOut : dfxTimeOutConfig_.sqeTimeOutTimeOut;
    // 检查是否需要生成cache-notify placeholder
    if (isPlaceholder_) {
        // 只有使能alltoallv类算子的aicpu cache才需要cache-notify placeholder
        CHK_PRT_RET(
            !(needAddSqe_ && isAlltoallv_),
            HCCL_ERROR(
                "[DispatcherAiCpu][SignalWait] isPlaceholder_[%u] "
                "needAddSqe_[%u] isAlltoallv_[%u]",
                isPlaceholder_, needAddSqe_, isAlltoallv_),
            HCCL_E_INTERNAL);

        // 只有A3下才会使能alltoallv aicpu cache并需要cache-notify placeholder
        CHK_PRT_RET(
            !(inchip
              || (aicpuInfo_.devType != DevType::DEV_TYPE_310P1 && aicpuInfo_.devType != DevType::DEV_TYPE_310P3)),
            HCCL_ERROR(
                "[DispatcherAiCpu][SignalWait] isPlaceholder_[%u] inchip[%u] devType[%u]", isPlaceholder_, inchip,
                aicpuInfo_.devType),
            HCCL_E_INTERNAL);

        addOneCacheNotifyWaitPlaceholderSqe_(
            streamInfo.actualStreamId, taskId, notifyId, sqeBuffer, sqeTypeAddr, dfxTimeOutConfig_);
        HCCL_INFO("%s generate cache-notify placeholder for signal wait", __func__);
    } else { // 正常生成SQE
        if (inchip
            || (aicpuInfo_.devType != DevType::DEV_TYPE_310P1 && aicpuInfo_.devType != DevType::DEV_TYPE_310P3)) {
            addOneNotifyWaitSqe_(
                streamInfo.actualStreamId, taskId, notifyId, sqeBuffer, sqeTypeAddr, dfxTimeOutConfig_);
        } else {
            u32 notifyRevisedOffset = 15U; // eventid偏移15位后为1
            u32 notifyGetEventId = 0x3FFU; // 取低15位
            if ((notifyId >> notifyRevisedOffset) != 0) {
                addOneEventWaitSqe_(
                    streamInfo.actualStreamId, (notifyId & notifyGetEventId), taskId, sqeBuffer, sqeTypeAddr);

                uint8_t* sqeBuffer1 = nullptr;
                uint8_t* sqeTypeAddr1 = nullptr;
                uint8_t* sqeDfxInfoAddr1 = nullptr;
                CHK_RET(GetStreamSqeBufferAddr(stream, sqeBuffer1, sqeTypeAddr1, sqeDfxInfoAddr1, taskId));
                AicpuDfxInfo* const dfxInfo = (AicpuDfxInfo*)sqeDfxInfoAddr1;
                dfxInfo->opRingBufferIdx = opRingBufferIdx_;
                dfxInfo->remoteRank = INVALID_VALUE_RANKID;

                u64 addr = 0;
                addOneEventResetSqe_(
                    streamInfo.actualStreamId, (notifyId & notifyGetEventId), taskId, aicpuInfo_.devId, 0, addr,
                    sqeBuffer1, sqeTypeAddr1);
            } else {
                HCCL_WARNING("%s SignalWait id is not event, please check %d", __func__, notifyId);
            }
        }
    }

    PLF_CONFIG_INFO(
        PLF_TASK,
        "%s para: streamId[%u] userRank[%u] remoteRank[%u] inchip[%d] devType[%d] notifyId[%u] sqeTimeOutTimeOut[%llu]",
        __func__, streamInfo.actualStreamId, userRank, remoteUserRank, inchip, aicpuInfo_.devType, notifyId,
        dfxTimeOutConfig_.sqeTimeOutTimeOut);
    return HCCL_SUCCESS;
}

HcclResult DispatcherAiCpu::MemcpyAsync(
    hccl::DeviceMem& dst, const hccl::DeviceMem& src, hccl::Stream& stream, u32 remoteUserRank,
    hccl::LinkType inLinkType)
{
    // 检查是否需要生成cache-memcpy placeholder
    if (isPlaceholder_) {
        // 只有零长拷贝才需要cache-memcpy placeholder
        CHK_PRT_RET(
            src.size() != 0,
            HCCL_ERROR("[DispatcherAiCpu][MemcpyAsync] isPlaceholder_[%u] src.size[%llu]", isPlaceholder_, src.size()),
            HCCL_E_INTERNAL);

        // 只有使能alltoallv类算子的aicpu cache才需要cache-memcpy placeholder
        CHK_PRT_RET(
            !(needAddSqe_ && isAlltoallv_),
            HCCL_ERROR(
                "[DispatcherAiCpu][MemcpyAsync] isPlaceholder_[%u] "
                "needAddSqe_[%u] isAlltoallv_[%u]",
                isPlaceholder_, needAddSqe_, isAlltoallv_),
            HCCL_E_INTERNAL);

        // 准备SQE, sqeType, dfxInfo
        uint8_t* sqeBuffer = nullptr;
        uint8_t* sqeTypeAddr = nullptr;
        uint8_t* sqeDfxInfoAddr = nullptr;
        uint16_t taskId = 0U;
        CHK_RET(GetStreamSqeBufferAddr(stream, sqeBuffer, sqeTypeAddr, sqeDfxInfoAddr, taskId));

        // 生成cache-memcpy placeholder SQE, 并设置sqeType
        CHK_PTR_NULL(addOneCacheMemcpyPlaceHolderSqe_);
        const HcclComStreamInfo& streamInfo = stream.GetHcclStreamInfo();
        addOneCacheMemcpyPlaceHolderSqe_(
            streamInfo.actualStreamId, taskId, src.ptr(), dst.ptr(), static_cast<uint8_t>(inLinkType), sqeBuffer,
            sqeTypeAddr, hcclQos_);

        // 设置dfxInfo
        AicpuDfxInfo* const dfxInfo = (AicpuDfxInfo*)sqeDfxInfoAddr;
        dfxInfo->opRingBufferIdx = opRingBufferIdx_;
        dfxInfo->remoteRank = remoteUserRank;
        dfxInfo->notifyId = INVALID_VALUE_RANKID;

        HCCL_INFO("%s capture zero-len memcpy for alltoallv, generate cache-memcpy placeholder sqe", __func__);

        return HCCL_SUCCESS;
    }

    // 参数有效性检查
    if (src.size() == 0) {
        HCCL_INFO("%s src memory size is 0, not need copy.", __func__);
        return HCCL_SUCCESS;
    }

    if (src == dst) {
        HCCL_INFO("%s src memory and dst memory is same, not need copy.", __func__);
        return HCCL_SUCCESS;
    }

    if (dst.size() < src.size()) {
        HCCL_ERROR(
            "%s The size of dst is smaller than that of src. dst addr[%p], dst size[%llu], "
            "src addr[%p], src size[%llu]",
            __func__, dst.ptr(), dst.size(), src.ptr(), src.size());
        return HCCL_E_PTR;
    }
    const HcclComStreamInfo& streamInfo = stream.GetHcclStreamInfo();

    // 将数据按4GB切分循环处理
    uint64_t spiltLoop = 0;
    uint64_t addrOffset = 0;
    uint64_t countSplit = 0;
    uint64_t countSize = src.size();
    uint8_t* sqeBuffer = nullptr;
    uint8_t* sqeTypeAddr = nullptr;
    uint8_t* sqeDfxInfoAddr = nullptr;
    uint16_t taskId = 0U;
    HcclReduceOp redOp = HCCL_REDUCE_RESERVED;
    aclrtReduceKind rtReduceOp = RK_MAP_TABLE[redOp];

    if (countSize > HCCL_SDMA_MAX_COUNT_4GB) {
        spiltLoop = (countSize % HCCL_SDMA_MAX_COUNT_4GB) ? (countSize / HCCL_SDMA_MAX_COUNT_4GB) :
                                                            ((countSize / HCCL_SDMA_MAX_COUNT_4GB) - 1);
        HCCL_INFO(
            "%s MemcpyAsync SDMA task countSize is bigger than 4GB"
            " and do segmentation splitloop[%llu]",
            __func__, spiltLoop);
    }
    uint8_t linkType = static_cast<uint8_t>(inLinkType);
    for (uint64_t index = 0; index <= spiltLoop; index++) {
        addrOffset = index * HCCL_SDMA_MAX_COUNT_4GB;
        countSplit = (index == spiltLoop) ? (countSize - index * HCCL_SDMA_MAX_COUNT_4GB) : (HCCL_SDMA_MAX_COUNT_4GB);
        void* srcSplit = static_cast<void*>(static_cast<char*>(const_cast<void*>(src.ptr())) + addrOffset);
        void* dstSplit = static_cast<void*>(static_cast<char*>(dst.ptr()) + addrOffset);

        CHK_RET(GetStreamSqeBufferAddr(stream, sqeBuffer, sqeTypeAddr, sqeDfxInfoAddr, taskId));
        AicpuDfxInfo* const dfxInfo = (AicpuDfxInfo*)sqeDfxInfoAddr;
        dfxInfo->opRingBufferIdx = opRingBufferIdx_;
        dfxInfo->remoteRank = remoteUserRank;
        dfxInfo->notifyId = INVALID_VALUE_RANKID;
        addOneMemcpySqe_(
            streamInfo.actualStreamId, taskId, srcSplit, countSplit, ACL_FLOAT, rtReduceOp, dstSplit, 0,
            aicpuInfo_.ssid, aicpuInfo_.devId, aicpuInfo_.overflowAddr, linkType, sqeBuffer, sqeTypeAddr, hcclQos_);

        PLF_CONFIG_INFO(
            PLF_TASK,
            "%s para: linkType[%u] srcSplit[%p] dstSplit[%p] countSplit[%llu] taskId[%u] streamId[%u] remoteRank[%u]",
            __func__, linkType, srcSplit, dstSplit, countSplit, taskId, streamInfo.actualStreamId, remoteUserRank);
    }
    return HCCL_SUCCESS;
}

HcclResult DispatcherAiCpu::ClearLaunchContext()
{
    HCCL_INFO("[DispatcherAiCpu][ClearLaunchContext] clear launch context");

    key_ = OpUnfoldKey();
    cachePtr_ = nullptr;
    userInputMemRanges_.clear();
    userOutputMemRanges_.clear();
    isAlltoallv_ = false;
    alltoallvMetadataPtr_ = nullptr;
    needAddSqe_ = false;

    return HCCL_SUCCESS;
}

HcclResult DispatcherAiCpu::SetLaunchContext(
    const OpUnfoldKey& key, OpUnfoldCache* cachePtr, const std::vector<OpUnfoldMemRange>& userInputMemRanges,
    const std::vector<OpUnfoldMemRange>& userOutputMemRanges, const bool isAlltoallv,
    const AlltoallvMetadata* alltoallvMetadataPtr)
{
    CHK_PTR_NULL(cachePtr);

    HCCL_INFO("[DispatcherAiCpu][SetLaunchContext] set launch context for key %s", key.GetKeyString().c_str());

    if (isAlltoallv) {
        CHK_PTR_NULL(alltoallvMetadataPtr);
        CHK_RET(alltoallvMetadataPtr->Check(false));
    }

    key_ = key;
    cachePtr_ = cachePtr;
    userInputMemRanges_ = userInputMemRanges;
    userOutputMemRanges_ = userOutputMemRanges;
    isAlltoallv_ = isAlltoallv;
    alltoallvMetadataPtr_ = alltoallvMetadataPtr;
    needAddSqe_ = true;

    return HCCL_SUCCESS;
}

HcclResult DispatcherAiCpu::LaunchNewTask(
    OpUnfoldCacheEntry* entryPtr, const std::vector<OpUnfoldMemRange>& userInputMemRanges,
    const std::vector<OpUnfoldMemRange>& userOutputMemRanges, Stream& mainStream, std::vector<Stream>& slaveStreams,
    const bool profL1Enable, const bool isAlltoallv, const AlltoallvMetadata& alltoallvMetadata,
    const AlltoallvSendRecvInfo& alltoallvSendRecvInfo)
{
    // 校验入参
    CHK_PTR_NULL(entryPtr);
    if (isAlltoallv) { // 注意: LaunchNewTask是在缓存命中时调用, 此时无launch context,
                       // 所以不能直接通过key_.opType来判断是否为alltoallv算子, 需要框架侧传入
        CHK_RET(alltoallvMetadata.Check(true));
        CHK_RET(alltoallvSendRecvInfo.Check());
    }

    // 准备SQE刷新需要的变量
    size_t sqeCount = 0;
    uint8_t* sqeArray = nullptr;
    uint8_t* sqeTypeArray = nullptr;
    AicpuDfxInfo* sqeDfxInfoArray = nullptr;
    Stream* streamPtr = nullptr;
    std::vector<FlipInfo> flipInfos;      // taskid==0且flipnum!=0的SQE索引, 即它们前面需要添加placeholder
    std::vector<uint64_t> profTimestamps; // 只有当profiling L1 enable时, 才需要记录各SQE的刷新时间

    // 准备placeholder需要的变量
    uint8_t placeholderSqe[HCCL_SQE_SIZE]; // placeholder SQE
    uint8_t placeholderSqeType;            // placeholder SQE type
    AicpuDfxInfo placeholderSqeDfxInfo;    // placeholder DfxInfo
    placeholderSqeDfxInfo.opRingBufferIdx = opRingBufferIdx_;
    placeholderSqeDfxInfo.remoteRank = INVALID_VALUE_RANKID;
    placeholderSqeDfxInfo.notifyId = INVALID_VALUE_RANKID;
    constexpr uint16_t flipPlaceholderTaskId = 0; // FlipPlaceholder的taskId一定为0

    // 下发多段SQE数组，SQE刷新与下发异步执行
    size_t sqeArrayCount = 0;
    CHK_RET(entryPtr->GetSqeArrayCount(sqeArrayCount));
    HCCL_INFO(
        "[DispatcherAiCpu][LaunchNewTask] launch new task for sqeArrayCount[%u] in the cache entry at 0x%016llx",
        sqeArrayCount, entryPtr);
    for (size_t arrayIdx = 0; arrayIdx < sqeArrayCount; ++arrayIdx) {
        // 刷新并获得对应信息 (之前下发到RTSQ的SQE正在异步被消费)
        CHK_RET(entryPtr->UpdateAndGetSqeArray(
            arrayIdx, userInputMemRanges, userOutputMemRanges, mainStream, slaveStreams, opRingBufferIdx_, sqeCount,
            &sqeArray, &sqeTypeArray, &sqeDfxInfoArray, &streamPtr, flipInfos, profL1Enable, profTimestamps,
            isAlltoallv, alltoallvMetadata, alltoallvSendRecvInfo));

        // Profiling timestamp的个数应该等于缓存的SQE个数 + flip placeholder个数
        CHK_PRT_RET(
            (!profL1Enable) && (profTimestamps.size() != 0),
            HCCL_ERROR(
                "[DispatcherAiCpu][LaunchNewTask] profL1Enable[%u] profTimestamps.size[%u]", profL1Enable,
                profTimestamps.size()),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(
            profL1Enable && (profTimestamps.size() != (sqeCount + flipInfos.size())),
            HCCL_ERROR(
                "[DispatcherAiCpu][LaunchNewTask] profL1Enable[%u] profTimestamps.size[%u] sqeCount[%u] "
                "flipInfos.size[%u]",
                profL1Enable, profTimestamps.size(), sqeCount, flipInfos.size()),
            HCCL_E_INTERNAL);

        // 打印缓存并下发的SQE内容for debug
        // 设置HCCL_DEBUG_CONFIG="task", 或者设置ASCEND_GLOBAL_LOG_LEVEL=0
        int32_t streamId = streamPtr->GetHcclStreamInfo().actualStreamId;
        if ((UNLIKELY(GetExternalInputDebugConfig() & PLF_TASK)) || UNLIKELY(HcclCheckLogLevel(HCCL_LOG_DEBUG))) {
            PLF_CONFIG_DEBUG(
                PLF_TASK,
                "[DispatcherAicpu][LaunchNewTask] dump content of %uth cached SQE array with %u cached SQEs and stream "
                "id %u",
                arrayIdx, sqeCount, streamId);
            for (size_t sqeIdx = 0; sqeIdx < sqeCount; ++sqeIdx) {
                uint8_t* sqePtr = sqeArray + sqeIdx * HCCL_SQE_SIZE;
                const uint8_t sqeType = sqeTypeArray[sqeIdx];
                PLF_CONFIG_DEBUG(PLF_TASK, "[DispatcherAicpu][LaunchNewTask] %uth cached SQE", sqeIdx);
                CHK_RET(OpUnfoldCache::DumpSqeContent(sqePtr, sqeType));

                const AicpuDfxInfo& dfxinfo = sqeDfxInfoArray[sqeIdx];
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[DispatcherAicpu][LaunchNewTask] AicpuDfxInfo: remoteRank[%u] opRingBufferIdx[%u] notifyId[%u]",
                    dfxinfo.remoteRank, dfxinfo.opRingBufferIdx, dfxinfo.notifyId);
            }
        }

        HCCL_INFO("[DispatcherAiCpu][LaunchNewTask] arrayIdx[%u] flipInfos.size[%u]", arrayIdx, flipInfos.size());

        // 分段下发
        size_t sqeStartIdx = 0;           // 要拷贝的SQE在sqeArray中的起始索引
        size_t profTimestampStartIdx = 0; // 要拷贝的profiling timestamp在profTimestamps中的起始索引
        for (size_t i = 0; i < flipInfos.size(); ++i) {
            // Copy [sqeStartIdx, curZeroTaskidSqeIdx) + placeholder into RTSQ
            const size_t curZeroTaskidSqeIdx = flipInfos[i].first;
            const size_t curSqeCount = curZeroTaskidSqeIdx - sqeStartIdx + 1;
            CHK_PRT_RET(
                curZeroTaskidSqeIdx < sqeStartIdx,
                HCCL_ERROR(
                    "[DispatcherAiCpu][LaunchNewTask] curZeroTaskidSqeIdx[%u] < sqeStartIdx[%u]", curZeroTaskidSqeIdx,
                    sqeStartIdx),
                HCCL_E_INTERNAL);

            // Wait RTSQ for curSqeCount SQE (including placeholder) space
            HCCL_INFO("[DispatcherAiCpu][LaunchNewTask] wait rtsq for %u sqe space", curSqeCount);
            CHK_RET(WaitRtsq(*streamPtr, curSqeCount, true));

            // 下发sqeArray[sqeStartIdx, curZeroTaskidSqeIdx)到RTSQ中 (excluding placeholder)
            if (curZeroTaskidSqeIdx > sqeStartIdx) { // 需要下发的cached SQE数量 > 0
                HCCL_INFO(
                    "[DispatcherAiCpu][LaunchNewTask] launch %uth sqeArray[%u:%u)", arrayIdx, sqeStartIdx,
                    curZeroTaskidSqeIdx);
                CHK_RET(MemcpyRtsq(
                    *streamPtr, curSqeCount - 1, sqeArray + sqeStartIdx * HCCL_SQE_SIZE, sqeTypeArray + sqeStartIdx,
                    sqeDfxInfoArray + sqeStartIdx, profL1Enable, profTimestamps, profTimestampStartIdx));
                if (profL1Enable) {
                    profTimestampStartIdx += (curSqeCount - 1); // Cached SQEs
                }
            }

            // 根据具体SQE下发信息更新placeholder
            // 参考AddFlipTask, 设置placeholder SQE (streamId和stream相关, flipNum和SQE下发相关)
            // 注意: 由于flipPlaceholder DfxInfo在当前算子下不变, 提前设置, 后续只需要刷新placeholder即可
            const uint16_t curFlipNum = flipInfos[i].second;
            CHK_PRT_RET(
                curFlipNum == 0,
                HCCL_ERROR(
                    "[DispatcherAiCpu][LaunchNewTask] invalid flipNum[%u]: flipInfoIdx[%u] arrayIdx[%u] "
                    "zeroTaskidSqeIdx[%u] streamId[%u]",
                    curFlipNum, i, arrayIdx, curZeroTaskidSqeIdx, streamId),
                HCCL_E_INTERNAL);
            CHK_PTR_NULL(addOneFlipPlaceHolderSqe_);
            addOneFlipPlaceHolderSqe_(streamId, curFlipNum, flipPlaceholderTaskId, placeholderSqe, &placeholderSqeType);
            HCCL_INFO(
                "[DispatcherAiCpu][LaunchNewTask] flip placeholder SQE with flipnum[%u] and streamid[%u]", curFlipNum,
                streamId);

            // 下发placeholder SQE
            HCCL_INFO(
                "[DispatcherAiCpu][LaunchNewTask] launch placeholder SQE after %uth sqeArray[%u:%u)", arrayIdx,
                sqeStartIdx, curZeroTaskidSqeIdx);
            CHK_RET(MemcpyRtsq(
                *streamPtr, 1, placeholderSqe, &placeholderSqeType, &placeholderSqeDfxInfo, profL1Enable,
                profTimestamps, profTimestampStartIdx));
            if (profL1Enable) {
                profTimestampStartIdx += 1; // Flip placeholder
            }

            sqeStartIdx = curZeroTaskidSqeIdx;
        }

        // 按需下发剩余SQE
        if (sqeStartIdx < sqeCount) {
            // Copy [sqeStartIdx, sqeCount - 1] into RTSQ
            const size_t remainSqeCount = sqeCount - sqeStartIdx;

            // Wait RTSQ for remainSqeCount SQE space
            HCCL_INFO("[DispatcherAiCpu][LaunchNewTask] wait rtsq for %u sqe space", remainSqeCount);
            CHK_RET(WaitRtsq(*streamPtr, remainSqeCount, true));

            // 下发sqeArray[sqeStartIdx, sqeCount - 1]到RTSQ中
            HCCL_INFO(
                "[DispatcherAiCpu][LaunchNewTask] launch %uth sqeArray[%u:%u]", arrayIdx, sqeStartIdx, sqeCount - 1);
            CHK_RET(MemcpyRtsq(
                *streamPtr, remainSqeCount, sqeArray + sqeStartIdx * HCCL_SQE_SIZE, sqeTypeArray + sqeStartIdx,
                sqeDfxInfoArray + sqeStartIdx, profL1Enable, profTimestamps, profTimestampStartIdx));
            if (profL1Enable) {
                profTimestampStartIdx += remainSqeCount; // Remaining cached SQEs
            }
        }

        // 为下一段SQE数组的刷新清理变量
        sqeCount = 0;
        sqeArray = nullptr;
        sqeTypeArray = nullptr;
        sqeDfxInfoArray = nullptr;
        streamPtr = nullptr;
        flipInfos.clear();
    }

    // 下发完当前cache entry中所有SQE数组后, 更新input/output memory ranges, 与SQE中in-place update的addr-related
    // fields保持一直
    CHK_RET(entryPtr->SetInputOutputMemRanges(userInputMemRanges, userOutputMemRanges));

    return HCCL_SUCCESS;
}

HcclResult DispatcherAiCpu::LaunchTask(Stream& stream, bool isBlockLaunch)
{
    const HcclComStreamInfo& streamInfo = stream.GetHcclStreamInfo();
    HcclSqeContext* sqeContext = stream.GetSqeContextPtr();
    CHK_PTR_NULL(sqeContext);
    SqeRingBuffer* sqeContextBuffer = &(sqeContext->buffer);
    CHK_PTR_NULL(sqeContextBuffer);
    const auto cnt = sqeContextBuffer->sqeCnt;
    if (cnt == 0) {
        CHK_PRT_CONT(
            isBlockLaunch, HCCL_DEBUG("no sqe, streamId:%d, sqId:%u", streamInfo.actualStreamId, streamInfo.sqId));
        return HCCL_SUCCESS;
    } else if (cnt > streamInfo.sqDepth) {
        HCCL_ERROR("LaunchTask fail, cnt:%u should be less than sqDepth:%u", cnt, streamInfo.sqDepth);
        return HCCL_E_PTR;
    }

    auto& head = sqeContextBuffer->sqHead;
    auto& tail = sqeContextBuffer->sqTail;
    u32 newTail = (tail + cnt) % streamInfo.sqDepth;
    // 仅在阻塞下发场景打印，避免非阻塞场景调用时刷屏
    CHK_PRT_CONT(
        isBlockLaunch,
        HCCL_INFO(
            "Before send sqid:%d cnt:%u head:%u curtail:%u newTail:%u", streamInfo.sqId, cnt, head, tail, newTail));

    u64 startUsec = GetCurAicpuTimestamp();
    u64 lastUsec = startUsec;
    while (((tail < head ? streamInfo.sqDepth : 0U) + tail - head + cnt >= streamInfo.sqDepth)
           && (tail != head)) { // 判断剩余sqe空间是否足够下发
        // 需要放在while循环进来后第一个执行
        CHK_RET(QuerySqStatusByType(aicpuInfo_.devId, streamInfo.sqId, DRV_SQCQ_PROP_SQ_HEAD, head));

        // 非阻塞下发场景，rtsq队列空间不足时直接返回
        if (isBlockLaunch == false) {
            return HCCL_SUCCESS;
        }

        // 当前流无法下发，把其他流都launch一遍，避免等待的其他流没有launch
        std::unique_lock<std::recursive_mutex> lock(streamMapMutex_);
        std::vector<s32> invalidStreamIds;
        for (auto it = streamMap_.begin(); it != streamMap_.end(); ++it) {
            if (it->second.IsInvalid()) { // 跳过已销毁的stream
                invalidStreamIds.push_back(it->first);
                HCCL_WARNING("[DispatcherAiCpu][LaunchTask] skip invalid stream in streamMap, streamId:%d", it->first);
                continue;
            }
            if (it->first != streamInfo.actualStreamId) {
                CHK_RET(LaunchTask(it->second, false));
            }
        }
        for (auto id : invalidStreamIds) {
            streamMap_.erase(id);
        }
        lock.unlock();
        u64 curUsec = GetCurAicpuTimestamp();
        if (dfxTimeOutConfig_.sqFullWaitTimeOut != 0
            && (curUsec - startUsec > NANOSECOND_TO_SECOND * dfxTimeOutConfig_.sqFullWaitTimeOut)) {
            HCCL_ERROR(
                "Rtsq full, timeout %lus. curhead:%u, sqId:%d", dfxTimeOutConfig_.sqFullWaitTimeOut, head,
                streamInfo.sqId);
            return HCCL_E_AGAIN;
        }

        // 等待下发阶段，每隔30s打印一次状态
        if (curUsec - lastUsec > NANOSECOND_TO_SECOND * dfx::kPrintSqInterval) {
            lastUsec = curUsec;
            HCCL_RUN_INFO(
                "[LaunchTask][WaitLaunchWhileLoop]Current state. sqid:%d, head:%u, tail:%u, cnt:%u", streamInfo.sqId,
                head, tail, cnt);
        }

        // 下发过程中出现cqe异常
        if (checkOpExecStatusCallback_ != nullptr) {
            HcclResult opExecStatus = checkOpExecStatusCallback_();
            CHK_PRT_RET(
                opExecStatus != HCCL_SUCCESS,
                HCCL_ERROR("hccl aicpu stop launch for task exception or stop command, ret:%d", opExecStatus),
                opExecStatus);
        }
    }

    uint32_t left = streamInfo.sqDepth - tail; // sqeAddr 剩余空间
    const auto tailSqeIdx = sqeContextBuffer->tailSqeIdx;
    HCCL_INFO("cpy sqe, left:%u, tailSqeId:%u, cnt:%u, streamId:%u", left, tailSqeIdx, cnt, stream.id());
    if (cnt <= left) { // 剩余buffer放得下新增sqe
        CHK_SAFETY_FUNC_RET(memcpy_s(
            reinterpret_cast<uint8_t*>(streamInfo.sqBaseAddr) + tail * HCCL_SQE_SIZE, left * HCCL_SQE_SIZE,
            sqeContextBuffer->localBuff + (tailSqeIdx - cnt) * HCCL_SQE_SIZE, cnt * HCCL_SQE_SIZE));

        CHK_SAFETY_FUNC_RET(memcpy_s(
            sqeContextBuffer->rtsMirrorBuffer + tail * HCCL_SQE_SIZE, left * HCCL_SQE_SIZE,
            sqeContextBuffer->localBuff + (tailSqeIdx - cnt) * HCCL_SQE_SIZE, cnt * HCCL_SQE_SIZE));

        CHK_SAFETY_FUNC_RET(
            memcpy_s(sqeContextBuffer->rtsqSqeType + tail, left, sqeContextBuffer->sqeType + (tailSqeIdx - cnt), cnt));
        CHK_SAFETY_FUNC_RET(memcpy_s(
            sqeContextBuffer->rtsDfxInfo + tail, left * sizeof(AicpuDfxInfo),
            sqeContextBuffer->dfxInfo + (tailSqeIdx - cnt), cnt * sizeof(AicpuDfxInfo)));
    } else {
        CHK_SAFETY_FUNC_RET(memcpy_s(
            reinterpret_cast<uint8_t*>(streamInfo.sqBaseAddr) + tail * HCCL_SQE_SIZE, left * HCCL_SQE_SIZE,
            sqeContextBuffer->localBuff + (tailSqeIdx - cnt) * HCCL_SQE_SIZE, left * HCCL_SQE_SIZE));

        CHK_SAFETY_FUNC_RET(memcpy_s(
            reinterpret_cast<uint8_t*>(streamInfo.sqBaseAddr), streamInfo.sqDepth * HCCL_SQE_SIZE,
            sqeContextBuffer->localBuff + (tailSqeIdx - cnt + left) * HCCL_SQE_SIZE, (cnt - left) * HCCL_SQE_SIZE));

        CHK_SAFETY_FUNC_RET(memcpy_s(
            sqeContextBuffer->rtsMirrorBuffer + tail * HCCL_SQE_SIZE, left * HCCL_SQE_SIZE,
            sqeContextBuffer->localBuff + (tailSqeIdx - cnt) * HCCL_SQE_SIZE, left * HCCL_SQE_SIZE));

        CHK_SAFETY_FUNC_RET(memcpy_s(
            sqeContextBuffer->rtsMirrorBuffer, streamInfo.sqDepth * HCCL_SQE_SIZE,
            sqeContextBuffer->localBuff + (tailSqeIdx - cnt + left) * HCCL_SQE_SIZE, (cnt - left) * HCCL_SQE_SIZE));

        CHK_SAFETY_FUNC_RET(
            memcpy_s(sqeContextBuffer->rtsqSqeType + tail, left, sqeContextBuffer->sqeType + (tailSqeIdx - cnt), left));
        CHK_SAFETY_FUNC_RET(memcpy_s(
            sqeContextBuffer->rtsqSqeType + 0, streamInfo.sqDepth,
            sqeContextBuffer->sqeType + (tailSqeIdx - cnt + left), (cnt - left)));
        CHK_SAFETY_FUNC_RET(memcpy_s(
            sqeContextBuffer->rtsDfxInfo + tail, left * sizeof(AicpuDfxInfo),
            sqeContextBuffer->dfxInfo + (tailSqeIdx - cnt), left * sizeof(AicpuDfxInfo)));
        CHK_SAFETY_FUNC_RET(memcpy_s(
            sqeContextBuffer->rtsDfxInfo + 0, streamInfo.sqDepth * sizeof(AicpuDfxInfo),
            sqeContextBuffer->dfxInfo + (tailSqeIdx - cnt + left), (cnt - left) * sizeof(AicpuDfxInfo)));
    }
    // 打印算子展开下发的SQE内容for debug
    // 设置HCCL_DEBUG_CONFIG="task", 或者设置ASCEND_GLOBAL_LOG_LEVEL=0
    if ((UNLIKELY(GetExternalInputDebugConfig() & PLF_TASK)) || UNLIKELY(HcclCheckLogLevel(HCCL_LOG_DEBUG))) {
        const int32_t streamId = stream.GetHcclStreamInfo().actualStreamId;
        PLF_CONFIG_DEBUG(
            PLF_TASK, "[DispatcherAicpu][LaunchTask] dump content of %u dispatched SQEs with stream id %u", cnt,
            streamId);

        uint8_t* sqeArray = sqeContextBuffer->localBuff + (tailSqeIdx - cnt) * HCCL_SQE_SIZE;
        uint8_t* sqeTypeArray = sqeContextBuffer->sqeType + (tailSqeIdx - cnt);
        AicpuDfxInfo* sqeDfxInfoArray = sqeContextBuffer->dfxInfo + (tailSqeIdx - cnt);
        for (size_t sqeIdx = 0; sqeIdx < cnt; ++sqeIdx) {
            uint8_t* sqePtr = sqeArray + sqeIdx * HCCL_SQE_SIZE;
            const uint8_t sqeType = sqeTypeArray[sqeIdx];
            if (sqeType == SqeType::FLIP_PLACEHOLDER_SQE) {
                const rtStarsPlaceHolderSqe_t* placeholderSqePtr
                    = reinterpret_cast<const rtStarsPlaceHolderSqe_t*>(sqeArray + sqeIdx * HCCL_SQE_SIZE);
                PLF_CONFIG_DEBUG(
                    PLF_TASK,
                    "[DispatcherAicpu][LaunchTask] %uth dispatched SQE (placeholder) header.type[%u] taskid[%u]",
                    sqeIdx, placeholderSqePtr->header.type, placeholderSqePtr->header.taskId);
            } else {
                PLF_CONFIG_DEBUG(PLF_TASK, "[DispatcherAicpu][LaunchTask] %uth dispatched SQE", sqeIdx);
            }

            CHK_RET(OpUnfoldCache::DumpSqeContent(sqePtr, sqeType));

            const AicpuDfxInfo& dfxinfo = sqeDfxInfoArray[sqeIdx];
            PLF_CONFIG_DEBUG(
                PLF_TASK, "[DispatcherAicpu][LaunchTask] AicpuDfxInfo: remoteRank[%u] opRingBufferIdx[%u] notifyId[%u]",
                dfxinfo.remoteRank, dfxinfo.opRingBufferIdx, dfxinfo.notifyId);
        }
    }

    // 当前算子展开的SQE需要被动态缓存
    if (needAddSqe_) {
        CHK_PTR_NULL(cachePtr_);

        // 查找key对应的cache entry, 如果不存在 (即当前算子第一次LaunchTask), 创建新的cache entry
        OpUnfoldCacheEntry* entryPtr = nullptr;
        CHK_RET(cachePtr_->FindEntry(key_, &entryPtr));
        if (entryPtr == nullptr) {
            CHK_RET(cachePtr_->AddEntry(key_, userInputMemRanges_, userOutputMemRanges_, &entryPtr));
        }
        CHK_PTR_NULL(entryPtr);

        // 准备SQE相关信息的数组基地址
        uint8_t* sqeArray = sqeContextBuffer->localBuff + (tailSqeIdx - cnt) * HCCL_SQE_SIZE;
        uint8_t* sqeTypeArray = sqeContextBuffer->sqeType + (tailSqeIdx - cnt);
        AicpuDfxInfo* sqeDfxInfoArray = sqeContextBuffer->dfxInfo + (tailSqeIdx - cnt);

        // 遍历sqeType找到placeholder的位置
        std::vector<size_t> placeholderIdxes;
        uint8_t* curSqeTypePtr = sqeTypeArray;
        for (size_t sqeTypeIdx = 0; sqeTypeIdx < cnt; ++sqeTypeIdx) {
            if (*curSqeTypePtr == SqeType::FLIP_PLACEHOLDER_SQE) {
                placeholderIdxes.emplace_back(sqeTypeIdx);
            }
            ++curSqeTypePtr;
        }

        // 在动态缓存中分配实际需要的SQE数组
        const size_t cacheableSqeCount = cnt - placeholderIdxes.size();
        const int32_t streamId = stream.GetHcclStreamInfo().actualStreamId;
        size_t arrayIdx = 0;
        CHK_RET(entryPtr->AllocSqeArray(cacheableSqeCount, streamId, arrayIdx));

        // 分段拷贝SQE相关信息到cache entry中
        size_t cacheableSqeStartIdx = 0; // SQE start index (在动态缓存对应SQE数组中的索引)
        size_t bufferSqeStartIdx = 0;    // SQE start index (在SQE ring buffer中的索引)
        for (size_t i = 0; i < placeholderIdxes.size(); ++i) {
            // [bufferSqeStartIdx, curPlaceholderIdx) -> [cacheableSqeStartIdx, cacheableSqeStartIdx + curPlaceholderIdx
            // - bufferSqeStartIdx)
            const size_t curPlaceholderIdx = placeholderIdxes[i];
            HCCL_INFO(
                "[DispatcherAicpu][LaunchTask] %uth placeholder copy dispatchedSqeArray[%u:%u) into "
                "cachedSqeArrays[%u][%u:%u)",
                i, bufferSqeStartIdx, curPlaceholderIdx, arrayIdx, cacheableSqeStartIdx,
                cacheableSqeStartIdx + curPlaceholderIdx - bufferSqeStartIdx);
            if (curPlaceholderIdx <= bufferSqeStartIdx) { // NO non-placeholder dispatched SQE to admit
                // NOTE: NO need to change cacheableSqeStartIdx
                bufferSqeStartIdx = curPlaceholderIdx + 1;
            } else {
                const size_t curSqeCount = curPlaceholderIdx - bufferSqeStartIdx;
                CHK_RET(entryPtr->MemcpySqeArray(
                    arrayIdx, cacheableSqeStartIdx, curSqeCount, sqeArray + bufferSqeStartIdx * HCCL_SQE_SIZE,
                    sqeTypeArray + bufferSqeStartIdx, sqeDfxInfoArray + bufferSqeStartIdx, isAlltoallv_,
                    alltoallvMetadataPtr_));
                cacheableSqeStartIdx += curSqeCount;
                bufferSqeStartIdx = curPlaceholderIdx + 1;
            }
        }

        // 存在剩余SQE, 即最后一个SQE不是placeholder
        if (LIKELY(bufferSqeStartIdx < cnt)) {
            // [bufferSqeStartIdx, cnt - 1] -> [cacheableSqeStartIdx, cacheableSqeStartIdx + cnt - bufferSqeStartIdx)
            const size_t curSqeCount = cnt - bufferSqeStartIdx;
            CHK_RET(entryPtr->MemcpySqeArray(
                arrayIdx, cacheableSqeStartIdx, curSqeCount, sqeArray + bufferSqeStartIdx * HCCL_SQE_SIZE,
                sqeTypeArray + bufferSqeStartIdx, sqeDfxInfoArray + bufferSqeStartIdx, isAlltoallv_,
                alltoallvMetadataPtr_));
        }
    }

    CHK_RET(ConfigSqStatusByType(aicpuInfo_.devId, streamInfo.sqId, DRV_SQCQ_PROP_SQ_TAIL, newTail));
    tail = newTail;
    PLF_CONFIG_INFO(
        PLF_TASK, "%s success, sqid:%d, sqe_num:%u, curHead:%u, curtail:%u", __func__, streamInfo.sqId, cnt, head,
        tail);
    sqeContextBuffer->sqeCnt = 0;
    return HCCL_SUCCESS;
}

HcclResult DispatcherAiCpu::LaunchTasksEx(hccl::Stream& stream, std::vector<Stream>& subStreams)
{
    /* 两阶段模式，主流待正式执行时再下 */
    /* 一阶段第一次，可以先下主流 */
    HcclResult ret = LaunchTask(stream, true);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[DispatcherAiCpu][LaunchTasksEx] "
            "launch task failed, sqid:%u, ret:%u",
            stream.sqId(), ret);
        return ret;
    }

    for (u32 index = 0; index < subStreams.size(); index++) {
        ret = LaunchTask(subStreams[index], true);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[DispatcherAiCpu][LaunchTasksEx] "
                "launch task failed, sqid:%u, ret:%u",
                subStreams[index].sqId(), ret);
            return ret;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult DispatcherAiCpu::LaunchAllTasks()
{
    std::lock_guard<std::recursive_mutex> lock(streamMapMutex_);
    std::vector<s32> invalidStreamIds;
    for (auto it = streamMap_.begin(); it != streamMap_.end(); ++it) {
        if (it->second.IsInvalid()) { // 跳过已销毁的stream
            invalidStreamIds.push_back(it->first);
            HCCL_WARNING("[DispatcherAiCpu][LaunchAllTasks] skip invalid stream in streamMap, streamId:%d", it->first);
            continue;
        }
        HcclResult ret = LaunchTask(it->second, true);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("DispatcherAiCpu][LaunchAllTasks] launch task failed, sqid:%u, ret:%u", it->second.sqId(), ret);
            return ret;
        }
    }
    for (auto id : invalidStreamIds) {
        streamMap_.erase(id);
    }

    return HCCL_SUCCESS;
}

HcclResult DispatcherAiCpu::ReduceAsync(
    const void* src, void* dst, u64 dataCount, const HcclDataType datatype, HcclReduceOp redOp, Stream& stream,
    HcclReduceType reduceType)
{
    return (reduceType == HcclReduceType::HCCL_INLINE_REDUCE) ?
               InlineReduceAsync(src, dataCount, datatype, redOp, stream, dst) :
               TbeReduceAsync(src, dst, dataCount, datatype, redOp, stream, dst);
}

HcclResult DispatcherAiCpu::InlineReduceAsync(
    const void* src, u64 dataCount, const HcclDataType datatype, HcclReduceOp redOp, hccl::Stream& stream, void* dst,
    u32 remoteUserRank, hccl::LinkType inLinkType)
{
    // 参数有效性检查
    CHK_PTR_NULL(stream.ptr());
    if (dataCount == 0) {
        HCCL_INFO("%s src memory size is 0, not need inline reduce.", __func__);
        return HCCL_SUCCESS;
    }
    const HcclComStreamInfo& streamInfo = stream.GetHcclStreamInfo();

    aclDataType runtimeDataType = DT_MAP_TABLE[datatype];
    aclrtReduceKind rtReduceOp = RK_MAP_TABLE[redOp];

    // 将数据按4GB切分循环处理
    uint64_t spiltLoop = 0;
    uint64_t addr_offset = 0;
    uint64_t countSplit = 0;
    uint64_t countSize = dataCount * SIZE_TABLE[datatype];
    uint8_t* sqeBuffer = nullptr;
    uint8_t* sqeTypeAddr = nullptr;
    uint8_t* sqeDfxInfoAddr = nullptr;
    uint16_t taskId = 0U;

    if (countSize > HCCL_SDMA_MAX_COUNT_4GB) {
        spiltLoop = (countSize % HCCL_SDMA_MAX_COUNT_4GB) ? (countSize / HCCL_SDMA_MAX_COUNT_4GB) :
                                                            ((countSize / HCCL_SDMA_MAX_COUNT_4GB) - 1);
        HCCL_INFO(
            "%s InlineReduceAsync SDMA task countSize is bigger than 4GB"
            " and do segmentation splitloop:%llu",
            __func__, spiltLoop);
    }
    uint8_t linkType = static_cast<uint8_t>(inLinkType);
    for (uint64_t index = 0; index <= spiltLoop; index++) {
        addr_offset = index * HCCL_SDMA_MAX_COUNT_4GB;
        countSplit = (index == spiltLoop) ? (countSize - index * HCCL_SDMA_MAX_COUNT_4GB) : (HCCL_SDMA_MAX_COUNT_4GB);
        void* srcSplit = static_cast<void*>(static_cast<char*>(const_cast<void*>(src)) + addr_offset);
        void* dstSplit = static_cast<void*>(static_cast<char*>(dst) + addr_offset);

        CHK_RET(GetStreamSqeBufferAddr(stream, sqeBuffer, sqeTypeAddr, sqeDfxInfoAddr, taskId));
        AicpuDfxInfo* const dfxInfo = (AicpuDfxInfo*)sqeDfxInfoAddr;
        dfxInfo->opRingBufferIdx = opRingBufferIdx_;
        dfxInfo->remoteRank = remoteUserRank;
        dfxInfo->notifyId = INVALID_VALUE_RANKID;
        addOneMemcpySqe_(
            streamInfo.actualStreamId, taskId, srcSplit, countSplit, runtimeDataType, rtReduceOp, dstSplit, 0,
            aicpuInfo_.ssid, aicpuInfo_.devId, aicpuInfo_.overflowAddr, linkType, sqeBuffer, sqeTypeAddr, hcclQos_);

        PLF_CONFIG_INFO(
            PLF_TASK,
            "%s para: linkType[%u] srcSplit[%p] dstSplit[%p] countSplit[%llu] taskId[%u] streamId[%u] remoteRank[%u] "
            "rtDatatType[%d] rtReduceOp[%d]",
            __func__, linkType, srcSplit, dstSplit, countSplit, taskId, streamInfo.actualStreamId, remoteUserRank,
            runtimeDataType, rtReduceOp);
    }

    return HCCL_SUCCESS;
}

HcclResult DispatcherAiCpu::TbeReduceAsync(
    [[maybe_unused]] const void* src1, [[maybe_unused]] const void* src2, [[maybe_unused]] u64 count,
    [[maybe_unused]] const HcclDataType datatype, [[maybe_unused]] HcclReduceOp redOp, [[maybe_unused]] Stream& stream,
    [[maybe_unused]] const void* dst)
{
    HCCL_ERROR("[DispatcherAiCpu][TbeReduceAsync] aicpu do not support the tbe reduce");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult DispatcherAiCpu::RdmaSend(u32 dbindex, u64 dbinfo, hccl::Stream& stream, RdmaTaskInfo& taskInfo)
{
    const HcclComStreamInfo& streamInfo = stream.GetHcclStreamInfo();

    uint8_t* sqeBuffer = nullptr;
    uint8_t* sqeTypeAddr = nullptr;
    uint8_t* sqeDfxInfoAddr = nullptr;
    uint16_t taskId = 0U;

    CHK_RET(GetStreamSqeBufferAddr(stream, sqeBuffer, sqeTypeAddr, sqeDfxInfoAddr, taskId));
    AicpuDfxInfo* const dfxInfo = (AicpuDfxInfo*)sqeDfxInfoAddr;
    dfxInfo->opRingBufferIdx = opRingBufferIdx_;
    dfxInfo->remoteRank = taskInfo.remoteRank;
    dfxInfo->notifyId = INVALID_UINT; // 多个wr只敲一次doorbell的情况下，一般只会有一个notify

    uint32_t wrLen = 0; // 统计wr的总数据量
    for (const WrInformation& wr : taskInfo.wrInfos) {
        wrLen += wr.wrData.memList.len;
        dfxInfo->notifyId = (wr.notifyId != INVALID_UINT) ? wr.notifyId : dfxInfo->notifyId;
    }

    u64 dbAddr = CalcDbAddr(dbindex);
    addOneRdmaDbSendSqe_(
        streamInfo.actualStreamId, taskId, dbinfo, dbAddr, wrLen, static_cast<uint8_t>(taskInfo.rdmaType), sqeBuffer,
        sqeTypeAddr);

    PLF_CONFIG_INFO(
        PLF_TASK, "%s para: streamId[%u] taskId[%u] remoteRank[%u] RdmaType[%d] wrLen[%u] notifyId[%u]", __func__,
        streamInfo.actualStreamId, taskId, taskInfo.remoteRank, taskInfo.rdmaType, wrLen, dfxInfo->notifyId);

    return HCCL_SUCCESS;
}

HcclResult DispatcherAiCpu::RdmaRecord(
    [[maybe_unused]] u32 dbindex, [[maybe_unused]] u64 dbinfo, [[maybe_unused]] const struct SendWr& wr,
    [[maybe_unused]] hccl::Stream& stream, [[maybe_unused]] RdmaType rdmaType, [[maybe_unused]] u32 userRank,
    [[maybe_unused]] u64 offset, [[maybe_unused]] u32 notifyId)
{
    return HCCL_SUCCESS;
}

HcclResult DispatcherAiCpu::GetStreamSqeBufferAddr(
    hccl::Stream& stream, uint8_t*& sqeBufferAddr, uint8_t*& sqeTypeAddr, uint8_t*& sqeDfxInfoAddr, uint16_t& taskId)
{
    SaveStreamInfo(stream);
    HcclSqeContext* sqeContext = stream.GetSqeContextPtr();
    CHK_PTR_NULL(sqeContext);
    if (UNLIKELY(sqeContext->buffer.sqeCnt >= HCCL_PER_LAUNCH_SQE_CNT)) {
        HCCL_INFO("GetStreamSqeBufferAddr tailSqeIdx[%u], try to launchTask", sqeContext->buffer.tailSqeIdx);
        CHK_RET(LaunchTask(stream, true));
    }
    if (UNLIKELY(sqeContext->buffer.tailSqeIdx >= HCCL_SQE_MAX_CNT)) {
        CHK_RET(LaunchTask(stream, true));

        if (callback_ != nullptr) {
            hccl::AiCPUStreamTasks para(stream.id(), reinterpret_cast<void*>(sqeContext));
            hccl::TaskPara taskPara(TaskType::TASK_BATCH_REPORT, para);
            callback_(callBackUserPtr_, (void*)&taskPara, sizeof(struct TaskPara));
        }
    }
    SqeRingBuffer* sqeContextBuffer = &(sqeContext->buffer);
    uint16_t flipNum = sqeContextBuffer->filpNum;
    uint16_t nextTaskId = sqeContextBuffer->tailSqeTaskId;
    // nextTaskId=0的时候下发PlaceHolder
    if (UNLIKELY(nextTaskId == 0 && flipNum != 0)) {
        CHK_RET(AddFlipTask(stream));
    }
    if (UNLIKELY(sqeContext->buffer.tailSqeIdx >= HCCL_SQE_MAX_CNT)) {
        CHK_RET(LaunchTask(stream, true));

        if (callback_ != nullptr) {
            hccl::AiCPUStreamTasks para(stream.id(), reinterpret_cast<void*>(sqeContext));
            hccl::TaskPara taskPara(TaskType::TASK_BATCH_REPORT, para);
            callback_(callBackUserPtr_, (void*)&taskPara, sizeof(struct TaskPara));
        }
    }
    CHK_RET(stream.GetNextSqeBufferAddr(sqeBufferAddr, sqeTypeAddr, sqeDfxInfoAddr, taskId));
    return HCCL_SUCCESS;
}

HcclResult DispatcherAiCpu::WaitRtsq(Stream& stream, const size_t& sqeCount, const bool isBlockLaunch)
{
    // 注意: 目前WaitRtsq不会被递归调用, 所以isBlockLaunch永远为true; 为防止以后LaunchTask递归使用WaitRtsq,
    // 编码时考虑isBlockLaunch为false的情况

    // 检验入参
    const HcclComStreamInfo& streamInfo = stream.GetHcclStreamInfo();
    if (sqeCount == 0) {
        CHK_PRT_CONT(
            isBlockLaunch, HCCL_DEBUG(
                               "[DispatcherAiCpu][WaitRtsq] no sqe, streamId:%d, sqId:%u", streamInfo.actualStreamId,
                               streamInfo.sqId));
        return HCCL_SUCCESS;
    } else if (sqeCount > streamInfo.sqDepth) {
        HCCL_ERROR(
            "[DispatcherAiCpu][WaitRtsq] sqeCount %u should be smaller than sqDepth %u]", sqeCount, streamInfo.sqDepth);
        return HCCL_E_PTR;
    }

    // Get head and tail of RTSQ ring buffer
    HcclSqeContext* sqeContext = stream.GetSqeContextPtr();
    CHK_PTR_NULL(sqeContext);
    SqeRingBuffer* sqeContextBuffer = &(sqeContext->buffer);
    CHK_PTR_NULL(sqeContextBuffer);
    uint32_t& head = sqeContextBuffer->sqHead;
    uint32_t& tail = sqeContextBuffer->sqTail;

    // Dump debug information
    const uint32_t newTail = (tail + sqeCount) % streamInfo.sqDepth;
    // 仅在阻塞下发场景打印，避免非阻塞场景调用时刷屏
    CHK_PRT_CONT(
        isBlockLaunch, HCCL_INFO(
                           "[DispatcherAicpu][WaitRtsq] sqid:%d sqeCount:%u head:%u curtail:%u newTail:%u",
                           streamInfo.sqId, sqeCount, head, tail, newTail));

    // 轮询RTSQ直至获得足够大的剩余空间
    u64 startUsec = GetCurAicpuTimestamp();
    u64 lastUsec = startUsec;
    while (((tail < head ? streamInfo.sqDepth : 0U) + tail - head + sqeCount >= streamInfo.sqDepth)
           && (tail != head)) { // 判断RTSQ中剩余sqe空间是否足够下发
        // 需要放在while循环进来后第一个执行 (获取最新的RTSQ head, 查看RTSQ的消费进度)
        CHK_RET(QuerySqStatusByType(aicpuInfo_.devId, streamInfo.sqId, DRV_SQCQ_PROP_SQ_HEAD, head));

        // 非阻塞下发场景，rtsq队列空间不足时直接返回
        if (isBlockLaunch == false) {
            return HCCL_SUCCESS;
        }

        // 当前流无法下发，把其他流都launch一遍，避免等待的其他流没有launch
        std::unique_lock<std::recursive_mutex> lock(streamMapMutex_);
        std::vector<s32> invalidStreamIds;
        for (auto it = streamMap_.begin(); it != streamMap_.end(); ++it) {
            if (it->second.IsInvalid()) { // 跳过已销毁的stream
                invalidStreamIds.push_back(it->first);
                HCCL_WARNING("[DispatcherAiCpu][WaitRtsq] skip invalid stream in streamMap, streamId:%d", it->first);
                continue;
            }
            if (it->first != streamInfo.actualStreamId) { // 不是当前stream
                CHK_RET(LaunchTask(it->second, false));   // 非阻塞launch
            }
        }
        for (auto id : invalidStreamIds) {
            streamMap_.erase(id);
        }
        lock.unlock();

        // 等待超时
        u64 curUsec = GetCurAicpuTimestamp();
        if (dfxTimeOutConfig_.sqFullWaitTimeOut != 0
            && (curUsec - startUsec > NANOSECOND_TO_SECOND * dfxTimeOutConfig_.sqFullWaitTimeOut)) {
            HCCL_ERROR(
                "[DispatcherAicpu][WaitRtsq] Rtsq full, timeout %lus. curhead:%u, sqId:%d",
                dfxTimeOutConfig_.sqFullWaitTimeOut, head, streamInfo.sqId);
            return HCCL_E_AGAIN;
        }

        // 等待下发阶段，每隔30s打印一次状态
        if (curUsec - lastUsec > NANOSECOND_TO_SECOND * dfx::kPrintSqInterval) {
            lastUsec = curUsec;
            HCCL_RUN_INFO(
                "[DispatcherAicpu][WaitRtsq] Current state. sqid:%d, head:%u, tail:%u, sqeCount:%u", streamInfo.sqId,
                head, tail, sqeCount);
        }

        // 等待下发过程中出现cqe异常, 需要终止当前算子SQE的下发过程
        if (checkOpExecStatusCallback_ != nullptr) {
            HcclResult opExecStatus = checkOpExecStatusCallback_();
            CHK_PRT_RET(
                opExecStatus != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[DispatcherAicpu][WaitRtsq] hccl aicpu stop launch for task exception or stop command, ret:%d",
                    opExecStatus),
                opExecStatus);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult DispatcherAiCpu::MemcpyRtsq(
    Stream& stream, const size_t sqeCount, const uint8_t* sqeArray, const uint8_t* sqeTypeArray,
    const AicpuDfxInfo* sqeDfxInfoArray, const bool profL1Enable, const std::vector<uint64_t>& profTimestamps,
    const size_t profTimestampStartIdx)
{
    // 检验入参
    const HcclComStreamInfo& streamInfo = stream.GetHcclStreamInfo();
    if (sqeCount == 0) {
        HCCL_DEBUG(
            "[DispatcherAiCpu][MemcpyRtsq] no sqe, streamId:%d, sqId:%u", streamInfo.actualStreamId, streamInfo.sqId);
        return HCCL_SUCCESS;
    } else if (sqeCount > streamInfo.sqDepth) {
        HCCL_ERROR(
            "[DispatcherAiCpu][MemcpyRtsq] sqeCount %u should be smaller than sqDepth %u]", sqeCount,
            streamInfo.sqDepth);
        return HCCL_E_PTR;
    }
    CHK_PTR_NULL(sqeArray);
    CHK_PTR_NULL(sqeTypeArray);
    CHK_PTR_NULL(sqeDfxInfoArray);
    if (profL1Enable) {
        // 会访问profTimestamps[profTimestampStartIdx, profTimestampStartIdx + sqeCount - 1]
        CHK_PRT_RET(
            profTimestamps.size() == 0, HCCL_ERROR("[DispatcherAiCpu][MemcpyRtsq] empty profTimestamps"),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(
            profTimestampStartIdx >= profTimestamps.size(),
            HCCL_ERROR(
                "[DispatcherAiCpu][MemcpyRtsq] profTimestampStartIdx[%u] >= profTimestamps.size[%u]",
                profTimestampStartIdx, profTimestamps.size()),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(
            (profTimestampStartIdx + sqeCount - 1) >= profTimestamps.size(),
            HCCL_ERROR(
                "[DispatcherAiCpu][MemcpyRtsq] profTimestampStartIdx[%u] + sqeCount[%u] - 1 >= profTimestamps.size[%u]",
                profTimestampStartIdx, sqeCount, profTimestamps.size()),
            HCCL_E_INTERNAL);
    }

    // 获得RTSQ的head和tail
    HcclSqeContext* sqeContext = stream.GetSqeContextPtr();
    CHK_PTR_NULL(sqeContext);
    SqeRingBuffer* sqeContextBuffer = &(sqeContext->buffer);
    CHK_PTR_NULL(sqeContextBuffer);
    uint32_t& head = sqeContextBuffer->sqHead;
    uint32_t& tail = sqeContextBuffer->sqTail;

    // Dump debug information
    const uint32_t newTail = (tail + sqeCount) % streamInfo.sqDepth;
    HCCL_INFO(
        "[DispatcherAicpu][MemcpyRtsq] before memcpy, sqid:%d sqeCount:%u head:%u curtail:%u newTail:%u",
        streamInfo.sqId, sqeCount, head, tail, newTail);

    // 准备memcpy中目的末端基地址 (RTSQ从tail开始拷贝, [head, tail)为待执行SQE)
    uint8_t* rtsqSqeTailBaseAddr = reinterpret_cast<uint8_t*>(streamInfo.sqBaseAddr) + tail * HCCL_SQE_SIZE;
    uint8_t* mirrorRtsqSqeTailBaseAddr = sqeContextBuffer->rtsMirrorBuffer + tail * HCCL_SQE_SIZE;
    uint8_t* rtsqSqeTypeTailBaseAddr = sqeContextBuffer->rtsqSqeType + tail;
    AicpuDfxInfo* rtsqDfxInfoTailBaseAddr = sqeContextBuffer->rtsDfxInfo + tail;

    uint32_t tailLeft = streamInfo.sqDepth - tail; // RTSQ tail到buffer末端的剩余空间 (不包括buffer前端到head的剩余空间)
    HCCL_INFO(
        "[DispatcherAicpu][MemcpyRtsq] cpy sqe, tailLeft:%u, sqeCount:%u, streamId:%u", tailLeft, sqeCount,
        stream.id());
    if (sqeCount <= tailLeft) { // buffer末端剩余空间放得下新增sqe
        // 向buffer末端拷贝sqeCount个SQE信息

        // 拷贝SQE内容到RTSQ
        CHK_SAFETY_FUNC_RET(
            memcpy_s(rtsqSqeTailBaseAddr, tailLeft * HCCL_SQE_SIZE, sqeArray, sqeCount * HCCL_SQE_SIZE));

        // 拷贝SQE内容到RTSQ mirror
        CHK_SAFETY_FUNC_RET(
            memcpy_s(mirrorRtsqSqeTailBaseAddr, tailLeft * HCCL_SQE_SIZE, sqeArray, sqeCount * HCCL_SQE_SIZE));

        // 拷贝SQE类型
        CHK_SAFETY_FUNC_RET(memcpy_s(rtsqSqeTypeTailBaseAddr, tailLeft, sqeTypeArray, sqeCount));

        // 拷贝SQE DfxInfo
        CHK_SAFETY_FUNC_RET(memcpy_s(
            rtsqDfxInfoTailBaseAddr, tailLeft * sizeof(AicpuDfxInfo), sqeDfxInfoArray,
            sqeCount * sizeof(AicpuDfxInfo)));
    } else { // 需要buffer末端和首端的剩余空间
        // 先向buffer末端拷贝tailLeft个SQE信息, 再向buffer首端拷贝sqeCount-tailLeft个SQE信息

        // 拷贝SQE内容到RTSQ
        CHK_SAFETY_FUNC_RET(
            memcpy_s(rtsqSqeTailBaseAddr, tailLeft * HCCL_SQE_SIZE, sqeArray, tailLeft * HCCL_SQE_SIZE));
        CHK_SAFETY_FUNC_RET(memcpy_s(
            reinterpret_cast<uint8_t*>(streamInfo.sqBaseAddr), streamInfo.sqDepth * HCCL_SQE_SIZE,
            sqeArray + tailLeft * HCCL_SQE_SIZE, (sqeCount - tailLeft) * HCCL_SQE_SIZE));

        // 拷贝SQE内容到RTSQ mirror
        CHK_SAFETY_FUNC_RET(
            memcpy_s(mirrorRtsqSqeTailBaseAddr, tailLeft * HCCL_SQE_SIZE, sqeArray, tailLeft * HCCL_SQE_SIZE));
        CHK_SAFETY_FUNC_RET(memcpy_s(
            sqeContextBuffer->rtsMirrorBuffer, streamInfo.sqDepth * HCCL_SQE_SIZE, sqeArray + tailLeft * HCCL_SQE_SIZE,
            (sqeCount - tailLeft) * HCCL_SQE_SIZE));

        // 拷贝SQE type
        CHK_SAFETY_FUNC_RET(memcpy_s(rtsqSqeTypeTailBaseAddr, tailLeft, sqeTypeArray, tailLeft));
        CHK_SAFETY_FUNC_RET(memcpy_s(
            sqeContextBuffer->rtsqSqeType, streamInfo.sqDepth, sqeTypeArray + tailLeft, (sqeCount - tailLeft)));

        // 拷贝SQE DfxInfo
        CHK_SAFETY_FUNC_RET(memcpy_s(
            rtsqDfxInfoTailBaseAddr, tailLeft * sizeof(AicpuDfxInfo), sqeDfxInfoArray,
            tailLeft * sizeof(AicpuDfxInfo)));
        CHK_SAFETY_FUNC_RET(memcpy_s(
            sqeContextBuffer->rtsDfxInfo, streamInfo.sqDepth * sizeof(AicpuDfxInfo), sqeDfxInfoArray + tailLeft,
            (sqeCount - tailLeft) * sizeof(AicpuDfxInfo)));
    }

    // 更新RTSQ ring buffer的tail
    CHK_RET(ConfigSqStatusByType(aicpuInfo_.devId, streamInfo.sqId, DRV_SQCQ_PROP_SQ_TAIL, newTail));
    tail = newTail;
    PLF_CONFIG_INFO(
        PLF_TASK, "%s success, sqid:%d, sqe_num:%u, curHead:%u, curtail:%u", __func__, streamInfo.sqId, sqeCount, head,
        tail);

    // 上报profiling信息
    if (profL1Enable) {
        // Cache hit下SQE ring buffer中[tail-cnt, tail)为待下发SQE, 其数量一定为0 (否则更新tailSqeIdx后,
        // 会导致待下发SQE未下发, 而已下发SQE重复下发)
        CHK_PRT_RET(
            sqeContextBuffer->sqeCnt != 0,
            HCCL_ERROR(
                "[DispatcherAicpu][MemcpyRtsq] sqeContextBuffer->sqeCnt[%u] should be zero!", sqeContextBuffer->sqeCnt),
            HCCL_E_INTERNAL);

        // 上报flip placeholder的profiling信息
        if ((*sqeTypeArray) == SqeType::FLIP_PLACEHOLDER_SQE) {
            CHK_PRT_RET(
                sqeCount != 1, HCCL_ERROR("[DispatcherAicpu][MemcpyRtsq] sqeCount[%u] should be 1!", sqeCount),
                HCCL_E_INTERNAL);

            // 注意: 参考AddFlipTask, 先上报flip task (提醒profiling翻转taskid), 之后再拷贝到SQE ring buffer
            // (捕捉placeholder SQE相关的profiling信息) 注意:
            // ProfilingManager::TaskProfilingCallBack->ReportFilpTask不会扫描SQE ring buffer,
            // 也不会更新其中的streamToSqeIdxMap_
            if (callback_ != nullptr) {
                const rtStarsPlaceHolderSqe_t* placeholderSqePtr
                    = reinterpret_cast<const rtStarsPlaceHolderSqe_t*>(sqeArray);
                hccl::FlipTaskPara para(
                    stream.id(), placeholderSqePtr->header.taskId, placeholderSqePtr->u.flip_task_info.flipNumReport);
                hccl::TaskPara taskPara(TaskType::TASK_FLIP, para);
                callback_(callBackUserPtr_, (void*)&taskPara, sizeof(struct TaskPara));
            }
        }

        // 循环拷贝cached SQE到SQE ring buffer中
        size_t reportSqeCount = 0;
        while (reportSqeCount < sqeCount) {
            CHK_PRT_RET(
                sqeContextBuffer->tailSqeIdx > HCCL_SQE_MAX_CNT,
                HCCL_ERROR(
                    "[DispatcherAicpu][MemcpyRtsq] tailSqeIdx[%u] > HCCL_SQE_MAX_CNT[%u]", sqeContextBuffer->tailSqeIdx,
                    HCCL_SQE_MAX_CNT),
                HCCL_E_INTERNAL);
            const size_t sqeTailLeft = HCCL_SQE_MAX_CNT - sqeContextBuffer->tailSqeIdx;
            if (sqeTailLeft > 0) {
                // 准备profiling上报的目的末端基地址 (SQE ring buffer从tail开始拷贝)
                uint8_t* sqeLocalBuffTailBaseAddr
                    = sqeContextBuffer->localBuff + sqeContextBuffer->tailSqeIdx * HCCL_SQE_SIZE;
                uint8_t* sqeTypeTailBaseAddr = sqeContextBuffer->sqeType + sqeContextBuffer->tailSqeIdx;
                AicpuDfxInfo* dfxInfoTailBaseAddr = sqeContextBuffer->dfxInfo + sqeContextBuffer->tailSqeIdx;
                uint64_t* profTimestapTailBaseAddr = sqeContextBuffer->profTimestap + sqeContextBuffer->tailSqeIdx;

                // 向SQE ring buffer末端拷贝SQE信息[reportSqeCount, reportSqeCount + tmpSqeCount - 1]
                // (只用于profiling上报, 不会下发)
                const size_t tmpSqeCount = std::min(sqeCount - reportSqeCount, sqeTailLeft);
                HCCL_INFO(
                    "[DispatcherAicpu][MemcpyRtsq] report sqe profiling, sqeCount[%u] reportSqeCount[%u] "
                    "tailSqeIdx[%u] sqeTailLeft[%u] tmpSqeCount[%u]",
                    sqeCount, reportSqeCount, sqeContextBuffer->tailSqeIdx, sqeTailLeft, tmpSqeCount);

                // 拷贝SQE内容
                CHK_SAFETY_FUNC_RET(memcpy_s(
                    sqeLocalBuffTailBaseAddr, sqeTailLeft * HCCL_SQE_SIZE, sqeArray + reportSqeCount * HCCL_SQE_SIZE,
                    tmpSqeCount * HCCL_SQE_SIZE));

                // 拷贝SQE类型
                CHK_SAFETY_FUNC_RET(
                    memcpy_s(sqeTypeTailBaseAddr, sqeTailLeft, sqeTypeArray + reportSqeCount, tmpSqeCount));

                // 拷贝SQE DfxInfo
                CHK_SAFETY_FUNC_RET(memcpy_s(
                    dfxInfoTailBaseAddr, sqeTailLeft * sizeof(AicpuDfxInfo), sqeDfxInfoArray + reportSqeCount,
                    tmpSqeCount * sizeof(AicpuDfxInfo)));

                // 拷贝SQE timestamp
                CHK_SAFETY_FUNC_RET(memcpy_s(
                    profTimestapTailBaseAddr, sqeTailLeft * sizeof(uint64_t),
                    profTimestamps.data() + profTimestampStartIdx + reportSqeCount, tmpSqeCount * sizeof(uint64_t)));

                // 注意: sqeContextBuffer->sqeCnt不更新, 仍然为0 (即拷贝的SQE信息为已下发待上报), 避免SQE重复下发
                sqeContextBuffer->tailSqeIdx += static_cast<uint16_t>(tmpSqeCount);
                reportSqeCount += tmpSqeCount;
            } else {
                // 注意: SQE ring buffer中待下发SQE数量一定为0, 不需要调用LaunchTask将待下发变成已下发待上报,
                // 可以直接上报profiling将已下发待上报变成已上报 注意: 调用后,
                // ProfilingManager::StartReportSqeIdx为HCCL_SQE_MAX_CNT
                if (callback_ != nullptr) {
                    hccl::AiCPUStreamTasks para(stream.id(), reinterpret_cast<void*>(sqeContext));
                    hccl::TaskPara taskPara(TaskType::TASK_BATCH_REPORT, para);
                    callback_(callBackUserPtr_, (void*)&taskPara, sizeof(struct TaskPara));
                }

                // SQE ring buffer中所有SQE均为已上报 -> 清理SQE ring buffer
                HCCL_INFO("[DispatcherAicpu][MemcpyRtsq] Sqe index to %u, need clear", HCCL_SQE_MAX_CNT);
                CHK_PRT_RET(
                    sqeContextBuffer->sqeCnt != 0,
                    HCCL_ERROR(
                        "[DispatcherAicpu][MemcpyRtsq] Sqe index to %u, but sqeCnt[%u] is not 0", HCCL_SQE_MAX_CNT,
                        sqeContextBuffer->sqeCnt),
                    HCCL_E_INTERNAL);
                CHK_RET(stream.ClearLocalBuff()); // 会将stream.sqeContextBuffer中的sqeCnt和tailSqeIdx设置为0
                CHK_PRT_RET(
                    sqeContextBuffer->sqeCnt != 0,
                    HCCL_ERROR(
                        "[DispatcherAicpu][MemcpyRtsq] sqeCnt[%u] should be 0 after clear", sqeContextBuffer->sqeCnt),
                    HCCL_E_INTERNAL);
                CHK_PRT_RET(
                    sqeContextBuffer->tailSqeIdx != 0,
                    HCCL_ERROR(
                        "[DispatcherAicpu][MemcpyRtsq] tailSqeIdx[%u] should be 0 after clear",
                        sqeContextBuffer->tailSqeIdx),
                    HCCL_E_INTERNAL);

                // 参考HcclCommAicpu::ClearLocalBuff, 调用Stream::ClearLocalBuff后,
                // 应该调用ProfilingManager::UpdateStartReportSqeIdx手动将ProfilingManager::StartReportSqeIdx设置为0
                // 注意: 由于platform暂未将UpdateStartReportSqeIdx作为回调函数传入, 无法直接调用此framework函数 ->
                // 通过callback_ (即ProfilingManager::TaskProfilingCallBack)
                // 间接将ProfilingManager::StartReportSqeIdx设置为0 注意:
                // 由于调用前ProfilingManager::StartReportSqeIdx为HCCL_SQE_MAX_CNT, tailSqeIdx为0,
                // 即从startIdx=HCCL_SQE_MAX_CNT到endIdx=0, ProfilingManager不会进入profiling上报代码,
                // 而是只会调用UpdateStartReportSqeIdx设置StartReportSqeIdx为0
                if (callback_ != nullptr) {
                    HCCL_INFO("[DispatcherAicpu][MemcpyRtsq] re-invoke callback_ to reset StartReportSqeIdx as 0 in "
                              "ProfilingManager");

                    hccl::AiCPUStreamTasks para(stream.id(), reinterpret_cast<void*>(sqeContext));
                    hccl::TaskPara taskPara(TaskType::TASK_BATCH_REPORT, para);
                    callback_(callBackUserPtr_, (void*)&taskPara, sizeof(struct TaskPara));
                }
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult DispatcherAiCpu::AddFlipTask(Stream& stream)
{
    HcclSqeContext* sqeContext = stream.GetSqeContextPtr();
    CHK_PTR_NULL(sqeContext);
    SqeRingBuffer* sqeContextBuffer = &(sqeContext->buffer);
    CHK_PTR_NULL(sqeContextBuffer);
    uint16_t flipNum = sqeContextBuffer->filpNum;
    uint16_t taskId = sqeContextBuffer->tailSqeTaskId;

    if (callback_ != nullptr) {
        hccl::FlipTaskPara para(stream.id(), taskId, flipNum);
        hccl::TaskPara taskPara(TaskType::TASK_FLIP, para);
        callback_(callBackUserPtr_, (void*)&taskPara, sizeof(struct TaskPara));
    }

    const HcclComStreamInfo& streamInfo = stream.GetHcclStreamInfo();

    uint8_t* sqeBufferAddr = nullptr;
    uint8_t* sqeTypeAddr = nullptr;
    uint8_t* sqeDfxInfoAddr = nullptr;
    CHK_RET(stream.GetNextSqeBufferAddr(sqeBufferAddr, sqeTypeAddr, sqeDfxInfoAddr, taskId));

    AicpuDfxInfo* const dfxInfo = (AicpuDfxInfo*)sqeDfxInfoAddr;
    dfxInfo->opRingBufferIdx = opRingBufferIdx_;
    dfxInfo->remoteRank = INVALID_VALUE_RANKID;
    dfxInfo->notifyId = INVALID_VALUE_RANKID;
    addOneFlipPlaceHolderSqe_(streamInfo.actualStreamId, flipNum, taskId, sqeBufferAddr, sqeTypeAddr);

    PLF_CONFIG_INFO(
        PLF_TASK, "%s para: taskId[%u] streamId[%u] flipNum[%u]", __func__, taskId, streamInfo.actualStreamId, flipNum);
    return HCCL_SUCCESS;
}

HcclResult DispatcherAiCpu::AddRetryPreamble(Stream& stream) { return AddFlipTask(stream); }

void DispatcherAiCpu::SaveStreamInfo(hccl::Stream& stream)
{
    const HcclComStreamInfo& streamInfo = stream.GetHcclStreamInfo();
    std::lock_guard<std::recursive_mutex> lock(streamMapMutex_);
    auto it = streamMap_.find(streamInfo.actualStreamId);
    if (it == streamMap_.end()) {
        streamMap_.insert({streamInfo.actualStreamId, stream});
        HCCL_INFO("[DispatcherAiCpu][SaveStreamInfo] stream id[%d]", streamInfo.actualStreamId);
    } else if (it->second.IsInvalid()) {
        // stream id复用: 旧stream已销毁(IsInvalid=true), 用新stream覆盖, 否则dispatcher会一直跳过该id
        it->second = stream;
        HCCL_INFO(
            "[DispatcherAiCpu][SaveStreamInfo] refresh invalidated stream id stream id[%d]", streamInfo.actualStreamId);
    }
    return;
}

HcclResult DispatcherAiCpu::StreamSync(Stream& stream)
{
    uint32_t head = 0;
    uint32_t tail = 0;
    const HcclComStreamInfo* streamInfo;
    u64 startUsec = GetCurAicpuTimestamp();
    u64 lastUsec = startUsec;
    CHK_RET(stream.GetStreamInfo(streamInfo));

    CHK_RET(QuerySqStatusByType(aicpuInfo_.devId, streamInfo->sqId, DRV_SQCQ_PROP_SQ_TAIL, tail));
    HCCL_INFO("StreamSync aicpu stream sqid[%d] tail[%u]", streamInfo->sqId, tail);
    do {
        CHK_RET(QuerySqStatusByType(aicpuInfo_.devId, streamInfo->sqId, DRV_SQCQ_PROP_SQ_HEAD, head));
        u64 curUsec = GetCurAicpuTimestamp();
        if (curUsec - startUsec > NANOSECOND_TO_SECOND * dfxTimeOutConfig_.sqeTimeOutTimeOut) {
            HCCL_ERROR(
                "stream sync timeout %lus. curhead:%u, curtail:%u, sqId:%d", dfxTimeOutConfig_.sqeTimeOutTimeOut, head,
                tail, streamInfo->sqId);
            return HCCL_E_TIMEOUT;
        }

        // 等待下发阶段，每隔30s打印一次状态
        if (curUsec - lastUsec > NANOSECOND_TO_SECOND * dfx::kPrintSqInterval) {
            lastUsec = curUsec;
            HCCL_RUN_INFO("[StreamSync]Current state. sqid:%d, head:%u, tail:%u", streamInfo->sqId, head, tail);
        }
    } while (head != tail);

    return HCCL_SUCCESS;
}

u64 DispatcherAiCpu::CalcDbAddr(u32 dbindex)
{
    u64 dbAddr = 0;
    if (aicpuInfo_.devType == DevType::DEV_TYPE_910_93) {
        // 910_93 HCCS_SW 组网
        constexpr u64 roceBaseAddr = 0x202000000000ULL;
        constexpr u64 roceVfDbCfg0Reg = 0x230ULL;
        constexpr u64 chipAddrOffset = 0x20000000000ULL;
        constexpr u64 dieAddrOffset = 0x10000000000ULL;
        constexpr u32 dbDieIdMask = 0x00ff0000;
        constexpr u32 dbDieIdShift = 16; // 16 is dbDieIdShift
        dbAddr = roceBaseAddr + roceVfDbCfg0Reg + chipAddrOffset * aicpuInfo_.chipId
                 + dieAddrOffset * ((dbindex & dbDieIdMask) >> dbDieIdShift);
    } else {
        constexpr u64 roceBaseAddr = 0x2000000000ULL;
        constexpr u64 roceVfDbCfg0Reg = 0x230ULL;
        constexpr u64 chipAddrOffset = 0x80000000000ULL;
        constexpr u64 dieAddrOffset = 0x10000000000ULL;
        constexpr u32 dbDieIdMask = 0x00ff0000;
        constexpr u32 dbDieIdShift = 16; // 16 is dbDieIdShift
        dbAddr = roceBaseAddr + roceVfDbCfg0Reg + chipAddrOffset * aicpuInfo_.chipId
                 + dieAddrOffset * ((dbindex & dbDieIdMask) >> dbDieIdShift);
    }

    HCCL_DEBUG(
        "%s dbindex:%u, devType:%u, chipId:%lld, dbAddr:%llu", __func__, dbindex, aicpuInfo_.devType, aicpuInfo_.chipId,
        dbAddr);
    return dbAddr;
}

void DispatcherAiCpu::InitTimeOutConfig()
{
    dfxTimeOutConfig_.useCredit = false;
    dfxTimeOutConfig_.sqeTimeOutTimeOut = GetMaxNotifyWaitTime();
    dfxTimeOutConfig_.sqeCreditTimeOut = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT;
    dfxTimeOutConfig_.sqeWaitTimeOut = dfx::kKfcTimeOut;
    dfxTimeOutConfig_.sqFullWaitTimeOut = dfx::kSqFullWaitTimeOut;
    HCCL_INFO(
        "[DispatcherAiCpu][InitTimeOutConfig]DFX timeout config init successfully with details: [%s]",
        dfxTimeOutConfig_.ToString().c_str());
}
} // namespace hccl
