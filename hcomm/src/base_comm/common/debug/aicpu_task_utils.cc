/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_task_utils.h"

using Hccl::GetPlfDebugConfigValue;
using Hccl::PLF_TASK;
using Hccl::Rt91095StarsMemcpySqe;
using Hccl::Rt91095StarsNotifySqe;
using Hccl::Rt91095StarsSqeHeader;
using Hccl::Rt91095StarsSqeType;
using Hccl::Rt91095StarsUbdmaDBmodeSqe;
using Hccl::UdmaSqeCommon;
using Hccl::UdmaSqeWrite;
using Hccl::UdmaSqeWriteWithNotify;
using Hccl::UdmaSqOpcode;
#ifdef HCCL_V2 // hccl_v2
using Hccl::HCCL_LOG_DEBUG;
using Hccl::HCCL_LOG_INFO;
using Hccl::HcclCheckLogLevel;
#endif

namespace hcomm {

HcclResult AicpuTaskUtils::DumpSqeContent(const uint8_t* sqePtr)
{
    if ((UNLIKELY(GetPlfDebugConfigValue() & PLF_TASK)) || UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
        CHK_PTR_NULL(sqePtr);
        const Rt91095StarsSqeHeader* sqeHeaderPtr
            = static_cast<const Rt91095StarsSqeHeader*>(static_cast<const void*>(sqePtr));
        const Rt91095StarsSqeType sqeType = static_cast<Rt91095StarsSqeType>(sqeHeaderPtr->type);
        switch (sqeType) {
            case Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA:
                CHK_RET(DumpUbdmaSqe_(sqePtr));
                break;
            case Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD:
            case Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT:
                CHK_RET(DumpNotifySqe_(sqePtr));
                break;
            case Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA:
                CHK_RET(DumpSdmaSqe_(sqePtr));
                break;
            default:
                HCCL_WARNING("[AicpuTaskUtils][DumpSqeContent] sqeType[%u] is unsupported", sqeType);
                break;
        }
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskUtils::DumpUbdmaSqe_(const uint8_t* sqePtr)
{
    const Rt91095StarsUbdmaDBmodeSqe* ubDmaSqe
        = static_cast<const Rt91095StarsUbdmaDBmodeSqe*>(static_cast<const void*>(sqePtr));
    PLF_CONFIG_INFO(
        PLF_TASK,
        "[AicpuTaskUtils][DumpSqeContent] type[%u] "
        "rtStreamId[%u] taskId[%u] mode[%u] doorbellNum[%u] kernelCredit[%u] jettyId1[%u] funcId1[%u] "
        "piValue1[%u] dieId1[%u]",
        ubDmaSqe->header.type, ubDmaSqe->header.rtStreamId, ubDmaSqe->header.taskId, ubDmaSqe->mode,
        ubDmaSqe->doorbellNum, ubDmaSqe->kernelCredit, ubDmaSqe->jettyId1, ubDmaSqe->funcId1, ubDmaSqe->piValue1,
        ubDmaSqe->dieId1);
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskUtils::DumpNotifySqe_(const uint8_t* sqePtr)
{
    const Rt91095StarsNotifySqe* notifySqe
        = static_cast<const Rt91095StarsNotifySqe*>(static_cast<const void*>(sqePtr));
    PLF_CONFIG_INFO(
        PLF_TASK,
        "[AicpuTaskUtils][DumpSqeContent] type[%u] rtStreamId[%u] taskId[%u] wrCqe[%u] "
        "notifyId[%u] subType[%u] kernelCredit[%u] cntFlag[%u] clrFlag[%u] cntValue[%u] waitModeBit[%u] "
        "recordModeBit[%u] bitmap[%u] timeout[%u]",
        notifySqe->header.type, notifySqe->header.rtStreamId, notifySqe->header.taskId, notifySqe->header.wrCqe,
        notifySqe->notifyId, notifySqe->subType, notifySqe->kernelCredit, notifySqe->cntFlag, notifySqe->clrFlag,
        notifySqe->cntValue, notifySqe->waitModeBit, notifySqe->recordModeBit, notifySqe->bitmap, notifySqe->timeout);
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskUtils::DumpSdmaSqe_(const uint8_t* sqePtr)
{
    const Rt91095StarsMemcpySqe* sdmaSqe = static_cast<const Rt91095StarsMemcpySqe*>(static_cast<const void*>(sqePtr));
    constexpr uint64_t UINT32_BIT_WIDTH = 32;
    const uint64_t srcAddr = (static_cast<uint64_t>(sdmaSqe->u.strideMode0.srcAddrHigh) << UINT32_BIT_WIDTH)
                             | sdmaSqe->u.strideMode0.srcAddrLow;
    const uint64_t dstAddr = (static_cast<uint64_t>(sdmaSqe->u.strideMode0.dstAddrHigh) << UINT32_BIT_WIDTH)
                             | sdmaSqe->u.strideMode0.dstAddrLow;
    PLF_CONFIG_INFO(
        PLF_TASK,
        "[AicpuTaskUtils][DumpSqeContent] type[%u] "
        "rtStreamId[%u] taskId[%u] wrCqe[%u] opcode[%u] kernelCredit[%u] sssv[%u] dssv[%u] sns[%u] dns[%u] "
        "mapamPartId[%u] length[%u] srcAddr[0x%016llx] dstAddr[0x%016llx]",
        sdmaSqe->header.type, sdmaSqe->header.rtStreamId, sdmaSqe->header.taskId, sdmaSqe->header.wrCqe,
        sdmaSqe->opcode, sdmaSqe->kernelCredit, sdmaSqe->sssv, sdmaSqe->dssv, sdmaSqe->sns, sdmaSqe->dns,
        sdmaSqe->mapamPartId, sdmaSqe->u.strideMode0.lengthMove, static_cast<unsigned long long>(srcAddr),
        static_cast<unsigned long long>(dstAddr));
    return HCCL_SUCCESS;
}

HcclResult AicpuTaskUtils::DumpWqeContent(const uint8_t* wqePtr)
{
    // 注意: UdmaSqOpcode::UDMA_OPC_READ/UDMA_OPC_WRITE均使用UdmaSqeWrite
    if ((UNLIKELY(GetPlfDebugConfigValue() & PLF_TASK)) || UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
        CHK_PTR_NULL(wqePtr);
        const UdmaSqeCommon* wqeCommonPtr = static_cast<const UdmaSqeCommon*>(static_cast<const void*>(wqePtr));
        const uint8_t wqeCode = static_cast<uint8_t>(wqeCommonPtr->opcode);
        switch (wqeCode) {
            case UdmaSqOpcode::UDMA_OPC_READ:
            case UdmaSqOpcode::UDMA_OPC_WRITE:
                CHK_RET(DumpReadWriteWqe_(wqePtr));
                break;
            case UdmaSqOpcode::UDMA_OPC_WRITE_WITH_IMM:
                CHK_RET(DumpWriteWithNotifyWqe_(wqePtr));
                break;
            default: {
                HCCL_WARNING("[AicpuTaskUtils][DumpWqeContent] wqeCode[%u] is unsupported", wqeCode);
                break;
            }
        }
    }

    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskUtils::DumpReadWriteWqe_(const uint8_t* wqePtr)
{
    const UdmaSqeCommon* wqeCommonPtr = static_cast<const UdmaSqeCommon*>(static_cast<const void*>(wqePtr));
    const UdmaSqeWrite* udmaSqeWrite = static_cast<const UdmaSqeWrite*>(static_cast<const void*>(wqePtr));
    constexpr uint64_t UINT32_BIT_WIDTH = 32;
    const uint64_t rmtAddr
        = (static_cast<uint64_t>(wqeCommonPtr->rmtAddrHigh) << UINT32_BIT_WIDTH) | wqeCommonPtr->rmtAddrLow;
    const char* wqeType = wqeCommonPtr->inlineEn != 0 ?
                              "InlineWrite" :
                              ((wqeCommonPtr->opcode == UdmaSqOpcode::UDMA_OPC_READ) ? "Read" : "Write");
    if (wqeCommonPtr->inlineEn != 0) {
        PLF_CONFIG_INFO(
            PLF_TASK,
            "[AicpuTaskUtils][DumpWqeContent] type[%s] placeOdr[%u] "
            "compOrder[%u] fence[%u] cqe[%u] opcode[%u] "
            "tpn[%u] rmtEid[%u,%u,%u,%u] rmtAddr[0x%016llx] "
            "inlineEn[%u] inlineMsgLen[%u]",
            wqeType, wqeCommonPtr->placeOdr, wqeCommonPtr->compOrder, wqeCommonPtr->fence, wqeCommonPtr->cqe,
            wqeCommonPtr->opcode, wqeCommonPtr->tpn, wqeCommonPtr->rmtEid[0], wqeCommonPtr->rmtEid[1],
            wqeCommonPtr->rmtEid[2], wqeCommonPtr->rmtEid[3], static_cast<unsigned long long>(rmtAddr),
            wqeCommonPtr->inlineEn, wqeCommonPtr->inlineMsgLen);
    } else {
        const uint64_t locAddr = (static_cast<uint64_t>(udmaSqeWrite->u.sge.dataAddrHigh) << UINT32_BIT_WIDTH)
                                 | udmaSqeWrite->u.sge.dataAddrLow;
        PLF_CONFIG_INFO(
            PLF_TASK,
            "[AicpuTaskUtils][DumpWqeContent] type[%s] placeOdr[%u] "
            "compOrder[%u] fence[%u] cqe[%u] opcode[%u] tpn[%u] rmtEid[%u,%u,%u,%u] "
            "rmtAddr[0x%016llx] length[%u] locAddr[0x%016llx] "
            "udfFlag[%u] udfType[%u] reduceType[%u] reduceOp[%u]",
            wqeType, wqeCommonPtr->placeOdr, wqeCommonPtr->compOrder, wqeCommonPtr->fence, wqeCommonPtr->cqe,
            wqeCommonPtr->opcode, wqeCommonPtr->tpn, wqeCommonPtr->rmtEid[0], wqeCommonPtr->rmtEid[1],
            wqeCommonPtr->rmtEid[2], wqeCommonPtr->rmtEid[3], static_cast<unsigned long long>(rmtAddr),
            udmaSqeWrite->u.sge.length, static_cast<unsigned long long>(locAddr), wqeCommonPtr->udfFlag,
            wqeCommonPtr->inlinedata.udfData.udfType, wqeCommonPtr->inlinedata.udfData.reduceType,
            wqeCommonPtr->inlinedata.udfData.reduceOp);
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskUtils::DumpWriteWithNotifyWqe_(const uint8_t* wqePtr)
{
    const UdmaSqeCommon* wqeCommonPtr = static_cast<const UdmaSqeCommon*>(static_cast<const void*>(wqePtr));
    const UdmaSqeWriteWithNotify* udmaSqeWriteWithNotify
        = static_cast<const UdmaSqeWriteWithNotify*>(static_cast<const void*>(wqePtr));
    constexpr uint64_t UINT32_BIT_WIDTH = 32;
    const uint64_t rmtAddr
        = (static_cast<uint64_t>(wqeCommonPtr->rmtAddrHigh) << UINT32_BIT_WIDTH) | wqeCommonPtr->rmtAddrLow;
    const uint64_t locAddr
        = (static_cast<uint64_t>(udmaSqeWriteWithNotify->localU.sge.dataAddrHigh) << UINT32_BIT_WIDTH)
          | udmaSqeWriteWithNotify->localU.sge.dataAddrLow;
    const uint64_t notifyAddr
        = (static_cast<uint64_t>(udmaSqeWriteWithNotify->notify.notifyAddrHigh) << UINT32_BIT_WIDTH)
          | udmaSqeWriteWithNotify->notify.notifyAddrLow;
    const uint64_t notifyData
        = (static_cast<uint64_t>(udmaSqeWriteWithNotify->notify.notifyDataHigh) << UINT32_BIT_WIDTH)
          | udmaSqeWriteWithNotify->notify.notifyDataLow;
    PLF_CONFIG_INFO(
        PLF_TASK,
        "[AicpuTaskUtils][DumpWqeContent] type[WriteWithNotify] placeOdr[%u] "
        "compOrder[%u] fence[%u] cqe[%u] opcode[%u] tpn[%u] rmtEid[%u,%u,%u,%u] "
        "rmtAddr[0x%016llx] length[%u] locAddr[0x%016llx] "
        "notifyAddr[0x%016llx] notifyData[0x%016llx] "
        "udfFlag[%u] udfType[%u] reduceType[%u] reduceOp[%u]",
        wqeCommonPtr->placeOdr, wqeCommonPtr->compOrder, wqeCommonPtr->fence, wqeCommonPtr->cqe, wqeCommonPtr->opcode,
        wqeCommonPtr->tpn, wqeCommonPtr->rmtEid[0], wqeCommonPtr->rmtEid[1], wqeCommonPtr->rmtEid[2],
        wqeCommonPtr->rmtEid[3], static_cast<unsigned long long>(rmtAddr), udmaSqeWriteWithNotify->localU.sge.length,
        static_cast<unsigned long long>(locAddr), static_cast<unsigned long long>(notifyAddr),
        static_cast<unsigned long long>(notifyData), wqeCommonPtr->udfFlag, wqeCommonPtr->inlinedata.udfData.udfType,
        wqeCommonPtr->inlinedata.udfData.reduceType, wqeCommonPtr->inlinedata.udfData.reduceOp);
    return HCCL_SUCCESS;
}

} // namespace hcomm
