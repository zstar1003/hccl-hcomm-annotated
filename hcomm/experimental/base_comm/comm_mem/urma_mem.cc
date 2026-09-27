/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "endpoint_mgr.h"
#include "hccl_common.h"
#include "urma_mem.h"
#include <algorithm>
#include "log.h"
#include "hccl/hccl_res.h"
#include "hccl_mem_v2.h"
#include "exchange_ub_buffer_dto.h"
#include "local_ub_rma_buffer_manager.h"
#include "remote_rma_buffer.h"
#include "local_ub_rma_buffer.h"

namespace hcomm_experimental {

UbRegedMemMgr::UbRegedMemMgr() { localUbRmaBufferMgr_ = std::make_unique<LocalUbRmaBufferMgr>(); }

HcclResult UbRegedMemMgr::RegisterMemory(HcommMem mem, const char* memTag, void** memHandle)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(this->localUbRmaBufferMgr_);
    CHK_PTR_NULL(memHandle);
    CHK_PTR_NULL(mem.addr);
    CHK_PRT_RET(mem.size == 0, HCCL_ERROR("[%s] mem size is zero", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        mem.type == COMM_MEM_TYPE_INVALID, HCCL_ERROR("[%s] invalid mem type [%d]", __func__, mem.type), HCCL_E_PARA);

    std::shared_ptr<Hccl::LocalUbRmaBuffer> localUbRmaBuffer = nullptr;

    // LocalUbRmaBuffer构造函数存在注册动作，在调用该构造函数前需检查是否注册过
    Hccl::BufferKey<uintptr_t, u64> tempKey(reinterpret_cast<uintptr_t>(mem.addr), mem.size);
    auto findPair = localUbRmaBufferMgr_->Find(tempKey);
    if (findPair.first) {
        localUbRmaBuffer = findPair.second;
    } else {
        // 构造LocalUbRmaBuffer
        std::shared_ptr<Hccl::Buffer> localBufferPtr = nullptr;
        EXCEPTION_CATCH(
            (localBufferPtr = std::make_shared<Hccl::Buffer>(
                 reinterpret_cast<uintptr_t>(mem.addr), mem.size, static_cast<HcclMemType>(mem.type), memTag)),
            return HCCL_E_PTR);

        if (memTag && (strcmp(memTag, "HcclBuffer") == 0)) {
            EXCEPTION_CATCH(
                (localUbRmaBuffer = std::make_shared<Hccl::LocalUbRmaBuffer>(localBufferPtr)), return HCCL_E_PTR);
        } else {
            EXCEPTION_CATCH(
                (localUbRmaBuffer = std::make_shared<Hccl::LocalUbRmaBuffer>(localBufferPtr, this->rdmaHandle_)),
                return HCCL_E_PTR);
        }
    }

    // 重新构造key确保注册到计数器的key和接口返回的memHandle对应
    Hccl::BufferKey<uintptr_t, u64> actualRegKey(
        localUbRmaBuffer->GetAddr(), static_cast<uint64_t>(localUbRmaBuffer->GetSize()));

    // 注册到LocalUbRmaBuffer计数器
    auto resultPair = localUbRmaBufferMgr_->AddWithoutCheck(actualRegKey, localUbRmaBuffer);

    std::shared_ptr<Hccl::LocalUbRmaBuffer>& localBuffer = resultPair.first->second.buffer;
    CHK_SMART_PTR_NULL(localBuffer);
    *memHandle = static_cast<void*>(localBuffer.get());

    // 已注册：输入key是表中某一最相近key的全集。 返回添加该key的迭代器，及false
    // 未注册：输入key是表中某一最相近key的空集。 返回添加成功的迭代器，及true
    if (resultPair.second) {
        HCCL_INFO("[UbRegedMemMgr][RegisterMemory]Register memory success! Add key {%p, %llu}", mem.addr, mem.size);
    } else {
        HCCL_INFO(
            "[UbRegedMemMgr][RegisterMemory]Memory is already registered, just increase the reference count. Add key "
            "{%p, %llu}",
            mem.addr, mem.size);
        ;
        return HCCL_SUCCESS;
    }

    this->allRegisteredBuffers_.push_back(localBuffer);
    return HCCL_SUCCESS;
}

HcclResult UbRegedMemMgr::UnregisterMemory(void* memHandle)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(this->localUbRmaBufferMgr_);

    Hccl::LocalUbRmaBuffer* buffer = static_cast<Hccl::LocalUbRmaBuffer*>(memHandle);
    CHK_PTR_NULL(buffer);
    auto bufferInfo = buffer->GetBufferInfo();

    // 从LocalRamBuffer计数器删除
    Hccl::BufferKey<uintptr_t, u64> tempKey(bufferInfo.first, bufferInfo.second);
    bool resultPair = false;
    EXCEPTION_CATCH(resultPair = this->localUbRmaBufferMgr_->Del(tempKey), return HCCL_E_NOT_FOUND);
    // 计数器大于1时，返回false，说明框架层有其它设备在使用这段内存，返回HCCL_E_AGAIN
    if (!resultPair) {
        HCCL_INFO("[UbRegedMemMgr][[UnregisterMemory] Memory reference count is larger than 0 "
                  "(used by other RemoteRank), do not deregister memory.");
        return HCCL_SUCCESS;
    }

    // 删除vector中的LocalUbRmaBuffer
    auto it = std::find_if(
        allRegisteredBuffers_.begin(), allRegisteredBuffers_.end(),
        [buffer](const std::shared_ptr<Hccl::LocalUbRmaBuffer>& ptr) {
            return ptr.get() == buffer;
        });

    if (it == allRegisteredBuffers_.end()) {
        HCCL_ERROR("[UbRegedMemMgr][UnregisterMemory] Memory not found in vector!");
        return HCCL_E_NOT_FOUND;
    }

    allRegisteredBuffers_.erase(it);
    return HCCL_SUCCESS;
}

HcclResult UbRegedMemMgr::GetMemDesc(const EndpointDesc endpointDesc, Hccl::LocalUbRmaBuffer* localUbRmaBuffer)
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
UbRegedMemMgr::MemoryExport(const EndpointDesc endpointDesc, void* memHandle, void** memDesc, uint32_t* memDescLen)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);

    // 获取序列化信息
    Hccl::LocalUbRmaBuffer* localUbRmaBuffer = reinterpret_cast<Hccl::LocalUbRmaBuffer*>(memHandle);

    // 获取序列化信息
    CHK_RET(GetMemDesc(endpointDesc, localUbRmaBuffer));

    *memDescLen = static_cast<uint32_t>(localUbRmaBuffer->Desc.size());
    *memDesc = static_cast<void*>(localUbRmaBuffer->Desc.data());

    return HCCL_SUCCESS;
}

HcclResult UbRegedMemMgr::GetParamsFromMemDesc(
    const void* memDesc, uint32_t descLen, EndpointDesc& endpointDesc, Hccl::ExchangeUbBufferDto& dto)
{
    const char* description = static_cast<const char*>(memDesc);

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

    EndpointDesc endpointDesc;
    Hccl::ExchangeUbBufferDto dto;
    CHK_RET(GetParamsFromMemDesc(memDesc, descLen, endpointDesc, dto));

    // 构造RemoteUbRmaBuffer
    std::shared_ptr<Hccl::RemoteUbRmaBuffer> remoteUbRmaBuffer;
    EXCEPTION_CATCH(remoteUbRmaBuffer = std::make_shared<Hccl::RemoteUbRmaBuffer>(this->rdmaHandle_, dto),
                    return HCCL_E_PTR;);
    CHK_SMART_PTR_NULL(remoteUbRmaBuffer);

    // 放到RemoteUbRmaBufferMgr_
    Hccl::BufferKey<uintptr_t, u64> tempKey(static_cast<uintptr_t>(dto.addr), dto.size);
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

    outMem->addr = reinterpret_cast<void*>(remoteUbRmaBuffer->GetAddr());
    outMem->size = remoteUbRmaBuffer->GetSize();

    return HCCL_SUCCESS;
}

HcclResult UbRegedMemMgr::MemoryUnimport(const void* memDesc, uint32_t descLen)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);

    EndpointDesc endpointDesc;
    Hccl::ExchangeUbBufferDto dto;
    CHK_RET(GetParamsFromMemDesc(memDesc, descLen, endpointDesc, dto));

    if (remoteUbRmaBufferMgrs_.find(endpointDesc) == remoteUbRmaBufferMgrs_.end()) {
        HCCL_ERROR("[UrmaRegedMemMgr][MemoryUnimport] Remote buffer manager Not Found.");
        return HCCL_E_NOT_FOUND;
    }

    // 删除RemoteUbRmaBuffer
    HCCL_INFO("[MemoryUnimport][Ub] MemoryUnimport");
    Hccl::BufferKey<uintptr_t, u64> tempKey(static_cast<uintptr_t>(dto.addr), dto.size);

    bool resultPair = false;
    EXCEPTION_CATCH(resultPair = remoteUbRmaBufferMgrs_[endpointDesc]->Del(tempKey), return HCCL_E_NOT_FOUND);
    // 计数器大于1时，返回false，说明框架层有其它设备在使用这段内存，返回HCCL_E_AGAIN
    if (!resultPair) {
        HCCL_INFO("[UrmaRegedMemMgr][[MemoryUnimport] Memory reference count is larger than 0"
                  "(used by other RemoteRank).");
        return HCCL_E_AGAIN;
    }
    return HCCL_SUCCESS;
}

HcclResult UbRegedMemMgr::GetAllMemHandles(void** memHandles, uint32_t* memHandleNum)
{
    HCCL_INFO("[%s] Begin", __FUNCTION__);
    CHK_PTR_NULL(memHandleNum);

    uint32_t bufferCount = static_cast<uint32_t>(allRegisteredBuffers_.size());
    *memHandleNum = bufferCount;

    HCCL_INFO("[UbRegedMemMgr][[GetAllMemHandles] memHandleNum is [%d]", bufferCount);

    *memHandles = (bufferCount == 0) ? nullptr : reinterpret_cast<void*>(allRegisteredBuffers_.data());

    return HCCL_SUCCESS;
}

} // namespace hcomm_experimental
