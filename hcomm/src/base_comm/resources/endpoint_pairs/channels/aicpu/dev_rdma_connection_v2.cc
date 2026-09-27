/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "./dev_rdma_connection_v2.h"
#include "log.h"
#include "acl/acl_rt.h"
#include "hccp.h"

namespace hcomm {
static constexpr uint32_t DEFAULT_CQN = 0;
// DbVendorSpecified bitfield layout, matching Roce3DbEntry.dw0.bs:
//   cos      at bits 24-26 (3 bits)
//   mtuShift at bits 50-52 (3 bits)
static constexpr uint32_t UB_DB_VENDOR_COS_SHIFT = 24;
static constexpr uint32_t UB_DB_VENDOR_MTUSHIFT_SHIFT = 50;
static constexpr uint32_t UB_DB_VENDOR_FIELD_MASK = 0x7;

DevRdmaConnectionV2::DevRdmaConnectionV2(Hccl::Socket* socket, RdmaHandle rdmaHandle, uint32_t cqAttrFlags)
    : socket_(socket),
      rdmaHandle_(rdmaHandle),
      cqAttrFlags_(cqAttrFlags),
      ndaQpInfo_{}
{}

HcclResult DevRdmaConnectionV2::Init()
{
    if (rdmaConnStatus_ != RdmaConnStatus::CLOSED) {
        HCCL_INFO(
            "[DevRdmaConnectionV2][%s] status[%s] does not need init.", __func__, rdmaConnStatus_.Describe().c_str());
        return HCCL_SUCCESS;
    }

    GetNdaOps();
    CHK_RET(GetDirectFlag());
    CHK_RET(GetDmaMode());

    rdmaConnStatus_ = RdmaConnStatus::INIT;
    return HCCL_SUCCESS;
}

DevRdmaConnectionV2::~DevRdmaConnectionV2()
{
    if (rdmaConnStatus_ == RdmaConnStatus::CLOSED || rdmaConnStatus_ == RdmaConnStatus::INIT) {
        return;
    }
    HcclResult ret = DestroyQp();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s]call DestroyQp failed: hcclRet -> %d", __func__, ret);
    }
}

std::string DevRdmaConnectionV2::Describe() const
{
    return Hccl::StringFormat("DevRdmaConnectionV2[status=%s]", rdmaConnStatus_.Describe().c_str());
}

static void* NdaAlloc(size_t size)
{
    void* ptr = nullptr;
    aclError ret = aclrtMalloc(&ptr, size, static_cast<aclrtMemMallocPolicy>(ACL_MEM_TYPE_HIGH_BAND_WIDTH));
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[NdaAlloc] aclrtMalloc failed, ret[%d], size[%zu]", ret, size);
        return nullptr;
    }
    return ptr;
}

static void NdaFree(void* ptr)
{
    if (ptr == nullptr) {
        return;
    }
    aclError ret = aclrtFree(ptr);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[NdaFree] aclrtFree failed, ret[%d], ptr[%p]", ret, ptr);
    }
}

static void NdaMemset(void* dst, int value, size_t count)
{
    aclError ret = aclrtMemset(dst, count, value, count);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[NdaMemset] aclrtMemset failed, ret[%d], dst[%p], value[%d], count[%zu]", ret, dst, value, count);
    }
}

static int NdaMemcpy(void* dst, size_t dstSize, void* src, size_t srcSize, uint32_t direct)
{
    aclrtMemcpyKind kind = ACL_MEMCPY_DEFAULT;
    switch (direct) {
        case MEMCPY_DIRECT_HOST_TO_HOST: {
            kind = ACL_MEMCPY_HOST_TO_HOST;
            break;
        }
        case MEMCPY_DIRECT_HOST_TO_DEVICE: {
            kind = ACL_MEMCPY_HOST_TO_DEVICE;
            break;
        }
        case MEMCPY_DIRECT_DEVICE_TO_HOST: {
            kind = ACL_MEMCPY_DEVICE_TO_HOST;
            break;
        }
        case MEMCPY_DIRECT_DEVICE_TO_DEVICE: {
            kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
            break;
        }
        default: {
            HCCL_ERROR("[MemcpyKindTranslate]Not support the memory copy type[%d].", direct);
            return -1;
        }
    }
    aclError ret = aclrtMemcpy(dst, dstSize, src, srcSize, kind);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[NdaMemcpy] aclrtMemcpy failed, ret[%d], dst[%p], src[%p], dstSize[%zu], srcSize[%zu]", ret, dst, src,
            dstSize, srcSize);
        return -1;
    }
    return 0;
}

void DevRdmaConnectionV2::GetNdaOps()
{
    ndaOps_ = {.alloc = NdaAlloc, .free = NdaFree, .memset_s = NdaMemset, .memcpy_s = NdaMemcpy};
}

HcclResult DevRdmaConnectionV2::GetDirectFlag()
{
    s32 ret = RaNdaGetDirectFlag(rdmaHandle_, &directFlag_);
    if (ret != 0) {
        HCCL_ERROR(
            "[DevRdmaConnectionV2][GetDirectFlag]errNo[0x%016llx] get directFlag fail. "
            "return[%d], params: rdmaHandle[%p], directFlag[%d]",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), ret, rdmaHandle_, directFlag_);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult DevRdmaConnectionV2::GetDmaMode()
{
    switch (directFlag_) {
        case DIRECT_FLAG_PCIE: {
            dmaMode_ = QBUF_DMA_MODE_DEFAULT;
            break;
        }
        case DIRECT_FLAG_UB: {
            dmaMode_ = QBUF_DMA_MODE_INDEP_UB;
            break;
        }
        default: {
            HCCL_ERROR("[GetDmaMode]Not support the directFlag [%d].", directFlag_);
            dmaMode_ = QBUF_DMA_MODE_MAX;
            return HCCL_E_INTERNAL;
        }
    }
    HCCL_INFO("[GetDmaMode] directFlag[%d], dmaMode[%d]", directFlag_, dmaMode_);
    return HCCL_SUCCESS;
}

HcclResult DevRdmaConnectionV2::CreateQp()
{
    if (socket_->GetStatus() != Hccl::SocketStatus::OK) {
        HCCL_WARNING("[DevRdmaConnectionV2::CreateQp] socket status is not ok, please check socket connection.");
        return HCCL_E_AGAIN;
    }

    // cqAttrFlags不为默认值0时，仅QBUF_DMA_MODE_DEFAULT（云脉网卡）支持配置，NPU网卡（INDEP_UB）不支持
    uint32_t cqAttrFlags = 0;
    if (cqAttrFlags_ != 0) {
        if (dmaMode_ == QBUF_DMA_MODE_INDEP_UB) {
            HCCL_ERROR(
                "[DevRdmaConnectionV2::CreateQp] NPU NIC does not support configuring cqAttrFlags[%u].", cqAttrFlags_);
            return HCCL_E_NOT_SUPPORT;
        }
        cqAttrFlags = cqAttrFlags_;
        HCCL_WARNING("[DevRdmaConnectionV2::CreateQp] CqAttrFlags is configured to [%u].", cqAttrFlags);
    }

    CHK_RET(Hccl::HrtRaNdaCqCreate(rdmaHandle_, &ndaOps_, dmaMode_, cqAttrFlags, &ndaCqInfo_, &cqHandle_));

    CHK_RET(Hccl::HrtRaNdaQpCreate(rdmaHandle_, &ndaOps_, dmaMode_, &ndaCqInfo_, &ndaQpInfo_, &qpHandle_));

    SqPiMem_ = hccl::DeviceMem::alloc(sizeof(void*));
    SqCiMem_ = hccl::DeviceMem::alloc(sizeof(void*));
    CqPiMem_ = hccl::DeviceMem::alloc(sizeof(void*));
    CqCiMem_ = hccl::DeviceMem::alloc(sizeof(void*));
    CHK_PRT_RET(
        !SqPiMem_ || !SqCiMem_ || !CqPiMem_ || !CqCiMem_,
        HCCL_ERROR(
            "%s DeviceMem::alloc for SqPi_ or SqCi_ or CqPi_ or CqCi_ failed, size=%zu", __func__, sizeof(void*)),
        HCCL_E_MEMORY);

    rdmaConnStatus_ = RdmaConnStatus::QP_CREATED;
    return HCCL_SUCCESS;
}

HcclResult DevRdmaConnectionV2::DestroyQp()
{
    if (rdmaConnStatus_ == RdmaConnStatus::CLOSED || rdmaConnStatus_ == RdmaConnStatus::INIT) {
        return HCCL_SUCCESS;
    }

    HcclResult ret = HCCL_SUCCESS;

    if (qpHandle_ != nullptr) {
        Hccl::HrtRaQpDestroy(qpHandle_);
        qpHandle_ = nullptr;
    }

    if (cqHandle_ != nullptr) {
        ret = Hccl::HrtRaNdaCqDestroy(rdmaHandle_, cqHandle_);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[DevRdmaConnectionV2::%s] HrtRaNdaCqDestroy failed, ret[%d]", __func__, ret);
        }
        cqHandle_ = nullptr;
    }

    SqPiMem_ = hccl::DeviceMem();
    SqCiMem_ = hccl::DeviceMem();
    CqPiMem_ = hccl::DeviceMem();
    CqCiMem_ = hccl::DeviceMem();

    rdmaConnStatus_ = RdmaConnStatus::CLOSED;
    return ret;
}

HcclResult DevRdmaConnectionV2::GetExchangeDto(std::unique_ptr<Hccl::Serializable>& serial)
{
    if (rdmaConnStatus_ != RdmaConnStatus::QP_CREATED && rdmaConnStatus_ != RdmaConnStatus::QP_MODIFIED) {
        HCCL_ERROR(
            "[DevRdmaConnectionV2][%s] status[%s] is not expected.", __func__, rdmaConnStatus_.Describe().c_str());
        return HCCL_E_AGAIN;
    }

    struct QpAttr localQpAttr;
    s32 ret = RaGetQpAttr(qpHandle_, &localQpAttr);
    if (ret != 0) {
        HCCL_ERROR("[DevRdmaConnectionV2::%s]RaGetQpAttr failed, ret[%d]", __func__, ret);
        return HCCL_E_ROCE_CONNECT;
    }
    std::unique_ptr<ExchangeRdmaConnDto> dto = nullptr;
    EXCEPTION_CATCH(
        dto = std::make_unique<ExchangeRdmaConnDto>(localQpAttr.qpn, localQpAttr.psn, localQpAttr.gidIdx),
        return HCCL_E_PTR);
    CHK_SAFETY_FUNC_RET(memcpy_s(dto->gid_, HCCP_GID_RAW_LEN, localQpAttr.gid, HCCP_GID_RAW_LEN));
    serial = std::unique_ptr<Hccl::Serializable>(std::move(dto));
    return HCCL_SUCCESS;
}

HcclResult DevRdmaConnectionV2::ParseRmtExchangeDto(const Hccl::Serializable& rmtDto)
{
    auto dto = dynamic_cast<const ExchangeRdmaConnDto&>(rmtDto);
    HCCL_INFO("[DevRdmaConnectionV2][%s] remoteConnDto[%s]", __func__, dto.Describe().c_str());
    rmtQpAttr_.psn = dto.psn_;
    rmtQpAttr_.qpn = dto.qpn_;
    rmtQpAttr_.gid_idx = dto.gid_idx_;
    CHK_SAFETY_FUNC_RET(memcpy_s(rmtQpAttr_.gid, HCCP_GID_RAW_LEN, dto.gid_, HCCP_GID_RAW_LEN));
    return HCCL_SUCCESS;
}

HcclResult DevRdmaConnectionV2::ModifyQp()
{
    if (rdmaConnStatus_ == RdmaConnStatus::QP_MODIFIED) {
        HCCL_WARNING(
            "[DevRdmaConnectionV2][%s] modify qp already, status[%s].", __func__, rdmaConnStatus_.Describe().c_str());
        return HCCL_SUCCESS;
    }
    if (rdmaConnStatus_ != RdmaConnStatus::QP_CREATED) {
        HCCL_ERROR(
            "[DevRdmaConnectionV2][%s] status[%s] is not expected.", __func__, rdmaConnStatus_.Describe().c_str());
        return HCCL_E_AGAIN;
    }

    if (!rmtQpAttr_.IsValid()) {
        HCCL_ERROR("[DevRdmaConnectionV2][%s] remote Qp Attr is empty, exchange qp attr first", __func__);
        return HCCL_E_INTERNAL;
    }

    struct QpAttr localQpAttr;
    s32 ret = RaGetQpAttr(qpHandle_, &localQpAttr);
    if (ret != 0) {
        HCCL_ERROR("[DevRdmaConnectionV2::%s]RaGetQpAttr failed, ret[%d]", __func__, ret);
        return HCCL_E_ROCE_CONNECT;
    }

    struct TypicalQp localQp;
    struct TypicalQp rmtQp;
    localQp.sl = qpInfo_.serviceLevel;
    localQp.tc = qpInfo_.trafficClass;
    localQp.retryCnt = qpInfo_.retryCnt;
    localQp.retryTime = qpInfo_.retryInterval;
    localQp.qpn = localQpAttr.qpn;
    localQp.psn = localQpAttr.psn;
    localQp.gidIdx = localQpAttr.gidIdx;
    (void)memcpy_s(localQp.gid, HCCP_GID_RAW_LEN, localQpAttr.gid, HCCP_GID_RAW_LEN);
    rmtQp.sl = qpInfo_.serviceLevel;
    rmtQp.tc = qpInfo_.trafficClass;
    rmtQp.retryCnt = qpInfo_.retryCnt;
    rmtQp.retryTime = qpInfo_.retryInterval;
    rmtQp.qpn = rmtQpAttr_.qpn;
    rmtQp.psn = rmtQpAttr_.psn;
    rmtQp.gidIdx = rmtQpAttr_.gid_idx;
    (void)memcpy_s(rmtQp.gid, HCCP_GID_RAW_LEN, rmtQpAttr_.gid, HCCP_GID_RAW_LEN);
    ret = RaTypicalQpModify(qpHandle_, &localQp, &rmtQp);
    if (ret != 0) {
        HCCL_ERROR("[modify][ra_qp]modify qp failed, ret(%d)", ret);
        return HCCL_E_ROCE_CONNECT;
    }
    rdmaConnStatus_ = RdmaConnStatus::QP_MODIFIED;
    return HCCL_SUCCESS;
}

HcclResult DevRdmaConnectionV2::BuildSqContext(SqContext* context)
{
    if (context == nullptr) {
        HCCL_ERROR("[GetSqContext] Invalid null pointer for context.");
        return HCCL_E_PTR;
    }

    // 零初始化roceSq, 确保不同dmaMode下未使用字段有确定值
    (void)memset_s(
        &context->contextInfo.roceSq, sizeof(context->contextInfo.roceSq), 0, sizeof(context->contextInfo.roceSq));

    struct QpAttr localQpAttr;
    s32 ret = RaGetQpAttr(qpHandle_, &localQpAttr);
    if (ret != 0) {
        HCCL_ERROR("[DevRdmaConnectionV2::%s]RaGetQpAttr failed, ret[%d]", __func__, ret);
        return HCCL_E_ROCE_CONNECT;
    }

    context->type = SQ_CONTEXT_TYPE_ROCE;
    context->contextInfo.roceSq.qpn = localQpAttr.qpn;
    context->contextInfo.roceSq.sqVa = ndaQpInfo_.sqInfo.qBuf.base;
    context->contextInfo.roceSq.wqeSize = ndaQpInfo_.sqInfo.qBuf.entrySize;
    context->contextInfo.roceSq.depth = ndaQpInfo_.sqInfo.qBuf.entryCnt;
    context->contextInfo.roceSq.headAddr = ReinterpretAs<uint64_t>(SqPiMem_.ptr());
    context->contextInfo.roceSq.tailAddr = ReinterpretAs<uint64_t>(SqCiMem_.ptr());
    context->contextInfo.roceSq.sl = qpInfo_.serviceLevel;
    context->contextInfo.roceSq.dbHwVa = ReinterpretAs<uint64_t>(ndaQpInfo_.sqInfo.dbHwVa.iovBase);

    if (dmaMode_ == QBUF_DMA_MODE_INDEP_UB) {
        context->contextInfo.roceSq.dbSwVa = ReinterpretAs<uint64_t>(ndaQpInfo_.sqInfo.dbrPiVa.iovBase);
        uint8_t mtuShift = static_cast<uint8_t>(localQpAttr.pathMtu - 1);
        uint8_t dbCos = static_cast<uint8_t>(localQpAttr.vendorPrivInfo & 0xFF);
        context->contextInfo.roceSq.dbVendorSpecified
            |= (static_cast<uint64_t>(mtuShift & UB_DB_VENDOR_FIELD_MASK) << UB_DB_VENDOR_MTUSHIFT_SHIFT);
        context->contextInfo.roceSq.dbVendorSpecified
            |= (static_cast<uint64_t>(dbCos & UB_DB_VENDOR_FIELD_MASK) << UB_DB_VENDOR_COS_SHIFT);

        HCCL_INFO("[DevRdmaConnectionV2][%s] mtuShift=%u, dbCos=%u", __func__, mtuShift, dbCos);
    }

    HCCL_INFO(
        "[DevRdmaConnectionV2][%s] type=%u, QPN=%u, SQ_VA=0x%llx, WQE_SIZE=%u, "
        "SQ_DEPTH=%u, SQ_HEAD_ADDR=0x%llx, SQ_TAIL_ADDR=0x%llx, "
        "SL=%u, DB_HW_VA=0x%llx, DB_SW_VA=0x%llx,"
        "DbVendorSpecified=0x%llx",
        __func__, context->type, context->contextInfo.roceSq.qpn, context->contextInfo.roceSq.sqVa,
        context->contextInfo.roceSq.wqeSize, context->contextInfo.roceSq.depth, context->contextInfo.roceSq.headAddr,
        context->contextInfo.roceSq.tailAddr, context->contextInfo.roceSq.sl, context->contextInfo.roceSq.dbHwVa,
        context->contextInfo.roceSq.dbSwVa, context->contextInfo.roceSq.dbVendorSpecified);
    return HCCL_SUCCESS;
}

HcclResult DevRdmaConnectionV2::BuildCqContext(CqContext* context)
{
    if (context == nullptr) {
        HCCL_ERROR("[GetCqContext] Invalid null pointer for context.");
        return HCCL_E_PTR;
    }

    // 零初始化roceCq, 确保不同dmaMode下未使用字段有确定值
    (void)memset_s(
        &context->contextInfo.roceCq, sizeof(context->contextInfo.roceCq), 0, sizeof(context->contextInfo.roceCq));

    context->type = CQ_CONTEXT_TYPE_ROCE;
    context->contextInfo.roceCq.cqVa = ndaCqInfo_.cqInfo.qBuf.base;
    context->contextInfo.roceCq.cqeSize = ndaCqInfo_.cqInfo.qBuf.entrySize;
    context->contextInfo.roceCq.cqDepth = ndaCqInfo_.cqInfo.qBuf.entryCnt;
    context->contextInfo.roceCq.headAddr = ReinterpretAs<uint64_t>(CqPiMem_.ptr());
    context->contextInfo.roceCq.tailAddr = ReinterpretAs<uint64_t>(CqCiMem_.ptr());
    // PCIe模式(DEFAULT): 仅使用硬DB; UBNIC模式(INDEP_UB): 仅使用软DB
    if (dmaMode_ == QBUF_DMA_MODE_DEFAULT) {
        context->contextInfo.roceCq.cqn = ndaCqInfo_.resv[0]; // 云脉网卡NDA直驱，协商ndaCqInfo_.resv[0]字段为CQN
        context->contextInfo.roceCq.dbHwVa = ReinterpretAs<uint64_t>(ndaCqInfo_.cqInfo.dbHwVa.iovBase);
    } else {
        context->contextInfo.roceCq.cqn = DEFAULT_CQN;
        context->contextInfo.roceCq.dbSwVa = ReinterpretAs<uint64_t>(ndaCqInfo_.cqInfo.dbrCiVa.iovBase);
    }

    HCCL_INFO(
        "[DevRdmaConnectionV2][%s] type=%u, CQN=%u, CQ_VA=0x%llx, CQE_SIZE=%u, CQ_DEPTH=%u, "
        "CQ_HEAD_ADDR=0x%llx, CQ_TAIL_ADDR=0x%llx, DB_HW_VA=0x%llx, DB_SW_VA=0x%llx]",
        __func__, context->type, context->contextInfo.roceCq.cqn, context->contextInfo.roceCq.cqVa,
        context->contextInfo.roceCq.cqeSize, context->contextInfo.roceCq.cqDepth, context->contextInfo.roceCq.headAddr,
        context->contextInfo.roceCq.tailAddr, context->contextInfo.roceCq.dbHwVa, context->contextInfo.roceCq.dbSwVa);

    return HCCL_SUCCESS;
}

std::vector<char> DevRdmaConnectionV2::GetSqUniqueId() const
{
    HCCL_DEBUG("start packing sq uniqueId");
    struct QpAttr localQpAttr;
    s32 ret = RaGetQpAttr(qpHandle_, &localQpAttr);
    if (ret != 0) {
        HCCL_ERROR("[DevRdmaConnectionV2::%s]RaGetQpAttr failed, ret[%d]", __func__, ret);
        return {};
    }
    uint8_t mtuShift = static_cast<uint8_t>(localQpAttr.pathMtu - 1);
    uint8_t dbCos = static_cast<uint8_t>(localQpAttr.vendorPrivInfo & 0xFF);
    uint64_t dbVendorSpecified = 0;
    dbVendorSpecified |= (static_cast<uint64_t>(mtuShift & UB_DB_VENDOR_FIELD_MASK) << UB_DB_VENDOR_MTUSHIFT_SHIFT);
    dbVendorSpecified |= (static_cast<uint64_t>(dbCos & UB_DB_VENDOR_FIELD_MASK) << UB_DB_VENDOR_COS_SHIFT);

    Hccl::BinaryStream binaryStream;
    // 打包1825网卡NDA直驱资源
    binaryStream << localQpAttr.qpn;
    binaryStream << ndaQpInfo_.sqInfo.qBuf.base;
    binaryStream << ndaQpInfo_.sqInfo.qBuf.entrySize;
    binaryStream << ndaQpInfo_.sqInfo.qBuf.entryCnt;
    binaryStream << ReinterpretAs<uint64_t>(SqPiMem_.ptr());
    binaryStream << ReinterpretAs<uint64_t>(SqCiMem_.ptr());
    binaryStream << ReinterpretAs<uint64_t>(ndaQpInfo_.sqInfo.dbHwVa.iovBase);
    binaryStream << ReinterpretAs<uint64_t>(ndaQpInfo_.sqInfo.dbrPiVa.iovBase);
    binaryStream << static_cast<uint8_t>(qpInfo_.serviceLevel);
    binaryStream << dbVendorSpecified;

    HCCL_DEBUG(
        "[DevRdmaConnectionV2][%s] mtuShift[%d], dbCos[%u], dbVendorSpecified[0x%llx]", __func__, mtuShift, dbCos,
        dbVendorSpecified);

    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> DevRdmaConnectionV2::GetCqUniqueId() const
{
    HCCL_DEBUG("start packing cq uniqueId");
    Hccl::BinaryStream binaryStream;
    // 打包1825网卡NDA直驱资源
    binaryStream << DEFAULT_CQN;
    binaryStream << ndaCqInfo_.cqInfo.qBuf.base;
    binaryStream << ndaCqInfo_.cqInfo.qBuf.entrySize;
    binaryStream << ndaCqInfo_.cqInfo.qBuf.entryCnt;
    binaryStream << ReinterpretAs<uint64_t>(CqPiMem_.ptr());
    binaryStream << ReinterpretAs<uint64_t>(CqCiMem_.ptr());
    binaryStream << ReinterpretAs<uint64_t>(ndaCqInfo_.cqInfo.dbrCiVa.iovBase);
    binaryStream << static_cast<uint64_t>(0); // CQ DbVendorSpecified placeholder

    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> DevRdmaConnectionV2::GetUniqueId() const
{
    HCCL_DEBUG("start packing conn uniqueId");
    Hccl::BinaryStream binaryStream;
    binaryStream << dmaMode_;
    binaryStream << GetSqUniqueId();
    binaryStream << GetCqUniqueId();

    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

} // namespace hcomm
