/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "transport_urma_mem.h"

namespace Hccl {
TransportUrmaMem::TransportUrmaMem(
    BaseMemTransport* transport, hcomm::RmaBufferMgr<BufferKey<uintptr_t, u64>, shared_ptr<HcclBuf>>& remoteHcclBufMgr)
    : transport_(transport),
      remoteHcclBufMgr_(remoteHcclBufMgr)
{}

TransportUrmaMem::~TransportUrmaMem() { HCCL_INFO("TransportUrmaMem Destroy"); }

HcclResult TransportUrmaMem::FillRmaBufferSlice(
    const RmaOpMem& localMem, const RmaOpMem& remoteMem, RmaBufferSlice& localRmaBufferSlice,
    RmtRmaBufferSlice& remoteRmaBufferSlice)
{
    void* remoteAddr = remoteMem.addr;
    void* localAddr = localMem.addr;
    u64 byteSize = std::min(remoteMem.size, localMem.size);
    auto localKey = BufferKey<uintptr_t, u64>(ReinterpretAs<uintptr_t>(localAddr), byteSize);
    auto remoteKey = BufferKey<uintptr_t, u64>(ReinterpretAs<uintptr_t>(remoteAddr), byteSize);

    auto localBuffer = LocalUbRmaBufferManager::GetInstance()->Find(localKey);
    CHK_PRT_RET(
        !localBuffer.first,
        HCCL_ERROR(
            "[TransportUrmaMem][FillRmaBufferSlice] Can't find localBuffer by key {%p, %llu}", localAddr, byteSize),
        HCCL_E_INTERNAL);

    auto remoteHcclBuf = remoteHcclBufMgr_.Find(remoteKey);
    CHK_PRT_RET(
        !remoteHcclBuf.first,
        HCCL_ERROR(
            "[TransportUrmaMem][FillRmaBufferSlice] Can't find remoteBuffer by key {%p, %llu}", remoteAddr, byteSize),
        HCCL_E_INTERNAL);
    auto remoteBuffer = static_cast<RemoteUbRmaBuffer*>(remoteHcclBuf.second->handle);

    u64 localDataOffSet
        = static_cast<u8*>(localAddr) - static_cast<u8*>((void*)(localBuffer.second->GetBuf()->GetAddr()));
    u64 remoteDataOffSet
        = static_cast<u8*>(remoteAddr) - static_cast<u8*>(ReinterpretAs<void*>(remoteBuffer->GetAddr()));

    localRmaBufferSlice.addr
        = ReinterpretAs<u64>(static_cast<u8*>((void*)(localBuffer.second->GetBuf()->GetAddr())) + localDataOffSet);
    localRmaBufferSlice.size = byteSize;
    localRmaBufferSlice.buf = localBuffer.second.get();

    remoteRmaBufferSlice.addr = ReinterpretAs<u64>(remoteBuffer->GetAddr() + remoteDataOffSet);
    remoteRmaBufferSlice.size = byteSize;
    remoteRmaBufferSlice.buf = remoteBuffer;

    HCCL_INFO(
        "[TransportUrmaMem][FillRmaBufferSlice] Local [%p], buff[%llu], offset[%llu], after mapping is [%llu], "
        "Datasize is [%llu].",
        localAddr, localBuffer.second->GetBuf()->GetAddr(), localDataOffSet, localRmaBufferSlice.addr, byteSize);

    HCCL_INFO(
        "[TransportUrmaMem][FillRmaBufferSlice] rmt [%p], buff[%llu], offset[%llu], after mapping is [%llu], Datasize "
        "is [%llu].",
        remoteAddr, remoteBuffer->GetAddr(), remoteDataOffSet, remoteRmaBufferSlice.addr, byteSize);

    return HCCL_SUCCESS;
}

// 2 is sizeof(float16), 8 is sizeof(float64), 2 is sizeof(bfloat16)..
constexpr u32 SIZE_TABLE[HCCL_DATA_TYPE_RESERVED]
    = {sizeof(s8),
       sizeof(s16),
       sizeof(s32),
       2,
       sizeof(float),
       sizeof(s64),
       sizeof(u64),
       sizeof(u8),
       sizeof(u16),
       sizeof(u32),
       8,
       2,
       16,
       2,
       1,
       1,
       1,
       1};

inline HcclResult SalGetDataTypeSize(HcclDataType dataType, u32& dataTypeSize)
{
    if ((dataType >= HCCL_DATA_TYPE_INT8) && (dataType < HCCL_DATA_TYPE_RESERVED)) {
        dataTypeSize = SIZE_TABLE[dataType];
    } else {
        HCCL_ERROR(
            "[Get][DataTypeSize]errNo[0x%016llx] get data size failed. dataType[%u] is invalid.",
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), dataType);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportUrmaMem::BatchBufferSlice(
    const HcclOneSideOpDesc* oneSideDescs, u32 descNum, RmaBufferSlice* localRmaBufferSlice,
    RmtRmaBufferSlice* remoteRmaBufferSlice)
{
    HCCL_INFO("[TransportUrmaMem][BatchBufferSlice] BatchBufferSlice Start");

    // 参数校验
    CHK_PTR_NULL(oneSideDescs);
    CHK_PTR_NULL(localRmaBufferSlice);
    CHK_PTR_NULL(remoteRmaBufferSlice);

    RmaOpMem remoteMem[MAX_DESC_NUM] = {};
    RmaOpMem localMem[MAX_DESC_NUM] = {};

    if (descNum > MAX_DESC_NUM) {
        THROW<InternalException>(
            StringFormat("[TransportUrmaMem][BatchBufferSlice] Desc item[%u] is out of range.", descNum));
    }

    for (u32 i = 0; i < descNum; i++) {
        if (oneSideDescs[i].count == 0) {
            HCCL_WARNING("[TransportUrmaMem][BatchBufferSlice] Desc item[%u] count is 0.", i);
        }
        u32 unitSize{0};
        HCCL_INFO("[TransportUrmaMem][BatchBufferSlice] SalGetDataTypeSize start");
        if (SalGetDataTypeSize(oneSideDescs[i].dataType, unitSize) != HCCL_SUCCESS) {
            THROW<InternalException>(StringFormat("[TransportUrmaMem][BatchBufferSlice] Get dataType size failed!"));
        }

        u64 byteSize = oneSideDescs[i].count * unitSize;
        remoteMem[i] = {oneSideDescs[i].remoteAddr, byteSize};
        localMem[i] = {oneSideDescs[i].localAddr, byteSize};
        HCCL_INFO("[TransportUrmaMem][BatchBufferSlice] FillRmaBufferSlice start");
        CHK_RET(FillRmaBufferSlice(localMem[i], remoteMem[i], localRmaBufferSlice[i], remoteRmaBufferSlice[i]));
    }
    return HCCL_SUCCESS;
}
} // namespace Hccl
