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
#include "resources/endpoints/endpoint.h"
#include "hccl_common.h"
#include "ub_reged_mem_mgr.h"
#include <algorithm>
#include "log.h"
#include "hccl/hccl_res.h"
#include "hccl_mem_v2.h"
#include "exchange_ub_buffer_dto.h"
#include "local_ub_rma_buffer_manager.h"
#include "remote_rma_buffer.h"
#include "local_ub_rma_buffer.h"

namespace hcomm {

UbRegedMemMgr::UbRegedMemMgr(RdmaHandle rdmaHandle) : rdmaHandle_(rdmaHandle)
{
    localUbRmaBufferMgr_ = std::make_unique<LocalUbRmaBufferMgr>();
}

HcclResult UbRegedMemMgr::RegisterMemory(const HcommMem* mem, const char* memTag, void** memHandle)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(mem);
    CHK_PTR_NULL(memHandle);
    CHK_PTR_NULL(localUbRmaBufferMgr_);
    std::lock_guard<std::mutex> lock(memMtx_);
    return RegisterMemoryImpl(
        *mem, memTag, memHandle, localUbRmaBufferMgr_, allRegisteredBuffers_, &handlesRecords_, "UbRegedMemMgr",
        [&](auto& bufPtr, auto& parent) {
            return std::make_shared<Hccl::LocalUbRmaBuffer>(bufPtr, rdmaHandle_, *parent);
        },
        [&](auto& bufPtr) {
            return std::make_shared<Hccl::LocalUbRmaBuffer>(bufPtr, rdmaHandle_);
        });
}

HcclResult UbRegedMemMgr::UnregisterMemory(void* memHandle)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(memHandle);
    CHK_PTR_NULL(localUbRmaBufferMgr_);
    std::lock_guard<std::mutex> lock(memMtx_);
    return UnregisterMemoryImpl(
        memHandle, localUbRmaBufferMgr_, allRegisteredBuffers_, &handlesRecords_,
        [](auto* b) {
            return b->GetMemRegOutParam();
        },
        [](const void* lhs, const void* rhs) {
            return Hccl::LocalUbRmaBuffer::IsSameMemRegOutParam(lhs, rhs);
        });
}

HcclResult UbRegedMemMgr::GetMemDesc(const EndpointDesc endpointDesc, Hccl::LocalUbRmaBuffer* localUbRmaBuffer) const
{
    auto dto = localUbRmaBuffer->GetExchangeDto();
    Hccl::BinaryStream localUbRmaBufferStream;
    dto->Serialize(localUbRmaBufferStream);
    std::vector<char> tempLocalMemDesc;
    localUbRmaBufferStream.Dump(tempLocalMemDesc);
    HCCL_DEBUG("[UbRegedMemMgr][GetMemDesc] [%s] dump data size [%u]", __func__, tempLocalMemDesc.size());
    // 判断内存描述符是否正确导出
    if (tempLocalMemDesc.empty()) {
        HCCL_ERROR("[UbRegedMemMgr][GetMemDesc] [%s] tempLocalMemDesc export failed.", __func__);
        return HCCL_E_INTERNAL;
    }

    std::vector<char> tempLocalEndpointDesc;
    tempLocalEndpointDesc.resize(sizeof(EndpointDesc));
    if (memcpy_s(tempLocalEndpointDesc.data(), sizeof(EndpointDesc), &endpointDesc, sizeof(EndpointDesc)) != EOK) {
        HCCL_ERROR("[UbRegedMemMgr][GetMemDesc] [%s] endpointDesc memcpy_s failed.", __func__);
        return HCCL_E_INTERNAL;
    }

    tempLocalMemDesc.insert(tempLocalMemDesc.end(), tempLocalEndpointDesc.begin(), tempLocalEndpointDesc.end());

    // 内存描述符拷贝
    localUbRmaBuffer->Desc = std::move(tempLocalMemDesc);
    return HCCL_SUCCESS;
}

HcclResult
UbRegedMemMgr::MemoryExport(const EndpointDesc& endpointDesc, void* memHandle, void** memDesc, uint32_t* memDescLen)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);

    CHK_PTR_NULL(memHandle);
    CHK_PTR_NULL(memDesc);
    CHK_PTR_NULL(memDescLen);
    std::lock_guard<std::mutex> lock(memMtx_);

    Hccl::LocalUbRmaBuffer* localUbRmaBuffer = nullptr;
    CHK_RET(ValidateMemExportHandle(memHandle, allRegisteredBuffers_, localUbRmaBuffer));

    // 获取序列化信息
    CHK_RET(GetMemDesc(endpointDesc, localUbRmaBuffer));

    *memDescLen = static_cast<uint32_t>(localUbRmaBuffer->Desc.size());
    *memDesc = static_cast<void*>(localUbRmaBuffer->Desc.data());

    return HCCL_SUCCESS;
}

HcclResult UbRegedMemMgr::GetParamsFromMemDesc(
    const void* memDesc, uint32_t descLen, EndpointDesc& endpointDesc, Hccl::ExchangeUbBufferDto& dto) const
{
    const char* description = static_cast<const char*>(memDesc);

    CHK_PRT_RET(
        descLen < sizeof(EndpointDesc),
        HCCL_ERROR("[%s] descLen[%u] is too small, expected at least %zu", __func__, descLen, sizeof(EndpointDesc)),
        HCCL_E_PARA);
    // 从memDesc末尾提取EndpointDesc
    if (memcpy_s(
            &endpointDesc, sizeof(EndpointDesc), description + descLen - sizeof(EndpointDesc), sizeof(EndpointDesc))
        != EOK) {
        HCCL_ERROR(
            "[UbRegedMemMgr][GetParamsFromMemDesc] [%s] endpointDesc copy error. aim size:[%llu]", __func__,
            sizeof(EndpointDesc));
        return HCCL_E_INTERNAL;
    }

    // 反序列化
    std::vector<char> tempDesc{};
    tempDesc.resize(TRANSPORT_EMD_ESC_SIZE);
    tempDesc.assign(description, description + descLen - sizeof(EndpointDesc));
    Hccl::BinaryStream remoteUbRmaBufferStream(tempDesc);
    dto.Deserialize(remoteUbRmaBufferStream);
    return HCCL_SUCCESS;
}

HcclResult UbRegedMemMgr::MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(memDesc);
    CHK_PTR_NULL(outMem);
    std::lock_guard<std::mutex> lock(memMtx_);

    EndpointDesc endpointDesc;
    Hccl::ExchangeUbBufferDto dto;
    CHK_RET(GetParamsFromMemDesc(memDesc, descLen, endpointDesc, dto));

    // 构造RemoteUbRmaBuffer
    std::shared_ptr<Hccl::RemoteUbRmaBuffer> remoteUbRmaBuffer;
    EXCEPTION_CATCH(remoteUbRmaBuffer = std::make_shared<Hccl::RemoteUbRmaBuffer>(rdmaHandle_, dto),
                    return HCCL_E_PTR;);
    CHK_SMART_PTR_NULL(remoteUbRmaBuffer);

    // 放到RemoteUbRmaBufferMgr_
    hccl::BufferKey<uintptr_t, u64> tempKey(static_cast<uintptr_t>(dto.addr), dto.size);
    if (remoteUbRmaBufferMgrs_.find(endpointDesc) == remoteUbRmaBufferMgrs_.end()) {
        std::unique_ptr<RemoteUbRmaBufferMgr> remoteUbRmaBufferMgr;
        EXCEPTION_CATCH((remoteUbRmaBufferMgr = std::make_unique<RemoteUbRmaBufferMgr>()), return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(remoteUbRmaBufferMgr);
        remoteUbRmaBufferMgrs_[endpointDesc] = std::move(remoteUbRmaBufferMgr);
        HCCL_INFO("remoteUbRmaBufferMgrs_ add remoteUbRmaBufferMgr successfully!");
    }

    auto resultPair = remoteUbRmaBufferMgrs_[endpointDesc]->Add(tempKey, remoteUbRmaBuffer);
    if (!resultPair.second) {
        HCCL_ERROR("[UbRegedMemMgr][MemoryImport] This memDesc has already been imported!");
        return HCCL_E_AGAIN;
    }

    outMem->addr = ReinterpretAs<void*>(remoteUbRmaBuffer->GetAddr());
    outMem->size = remoteUbRmaBuffer->GetSize();

    return HCCL_SUCCESS;
}

HcclResult UbRegedMemMgr::MemoryUnimport(const void* memDesc, uint32_t descLen)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(memDesc);
    std::lock_guard<std::mutex> lock(memMtx_);

    EndpointDesc endpointDesc;
    Hccl::ExchangeUbBufferDto dto;
    CHK_RET(GetParamsFromMemDesc(memDesc, descLen, endpointDesc, dto));

    if (remoteUbRmaBufferMgrs_.find(endpointDesc) == remoteUbRmaBufferMgrs_.end()) {
        HCCL_ERROR("[UrmaRegedMemMgr][MemoryUnimport] Remote buffer manager Not Found.");
        return HCCL_E_NOT_FOUND;
    }

    // 删除RemoteUbRmaBuffer
    HCCL_INFO("[MemoryUnimport][Ub] MemoryUnimport");
    hccl::BufferKey<uintptr_t, u64> tempKey(static_cast<uintptr_t>(dto.addr), dto.size);

    bool resultPair = false;
    EXCEPTION_CATCH(resultPair = remoteUbRmaBufferMgrs_[endpointDesc]->Del(tempKey), return HCCL_E_NOT_FOUND);
    // 计数器大于1时，返回false，说明框架层有其它设备在使用这段内存，返回HCCL_E_AGAIN
    if (!resultPair) {
        HCCL_INFO("[UrmaRegedMemMgr][[MemoryUnimport] Memory reference count is larger than 0"
                  "(used by other RemoteRank).");
        return HCCL_E_AGAIN;
    }
    if (remoteUbRmaBufferMgrs_[endpointDesc]->size() == 0) {
        remoteUbRmaBufferMgrs_.erase(endpointDesc);
    }
    return HCCL_SUCCESS;
}

HcclResult UbRegedMemMgr::GetAllMemHandles(void** memHandles, uint32_t* memHandleNum)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    std::lock_guard<std::mutex> lock(memMtx_);
    CHK_PTR_NULL(memHandles);
    CHK_PTR_NULL(memHandleNum);
    *memHandleNum = static_cast<uint32_t>(handlesRecords_.size());
    *memHandles = handlesRecords_.empty() ? nullptr : static_cast<void*>(handlesRecords_.data());
    HCCL_INFO("[UbRegedMemMgr][GetAllMemHandles] memHandleNum[%u]", *memHandleNum);
    return HCCL_SUCCESS;
}

} // namespace hcomm
