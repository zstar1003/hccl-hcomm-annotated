/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_mem_v2.h"
#include "log.h"
#include "exchange_ub_buffer_dto.h"
#include "local_ub_rma_buffer_manager.h"
#include "remote_rma_buffer.h"
#include "local_ub_rma_buffer.h"

using namespace Hccl;

HcclResult HcclMemRegV2(HcclNetDev netDev, const HcclMem* mem, HcclBuf* buf)
{
    if (netDev == nullptr || mem == nullptr || buf == nullptr) {
        HCCL_ERROR("[%s] netDev[%p] or mem[%p] or buf[%p] is null", __func__, netDev, mem, buf);
        return HCCL_E_PTR;
    }
    HCCL_INFO("[%s] Begin, addr[%p], size[%llu], type[%d]", __func__, mem->addr, mem->size, mem->type);
    // 仅支持UB类型
    HcclNetDevice* hcclNetDevice = static_cast<HcclNetDevice*>(netDev);
    if (!hcclNetDevice->IsUB()) {
        HCCL_ERROR("[%s] only support UB", __func__);
        return HCCL_E_NOT_SUPPORT;
    }

    // 构造LocalUbRmaBuffer
    auto getBuffFunc = [&]() -> HcclResult {
        std::shared_ptr<Buffer> localBufferPtr
            = make_shared<Buffer>(reinterpret_cast<uintptr_t>(mem->addr), mem->size, mem->type);
        std::shared_ptr<LocalUbRmaBuffer> localUbRmaBuffer
            = make_shared<LocalUbRmaBuffer>(localBufferPtr, hcclNetDevice, false);
        LocalUbRmaBufferMgr* localRmaBufferMgr = LocalUbRmaBufferManager::GetInstance();

        // 注册到LocalUbRmaBuffer计数器
        BufferKey<uintptr_t, u64> tempKey(reinterpret_cast<uintptr_t>(mem->addr), mem->size);
        auto resultPair = localRmaBufferMgr->Add(tempKey, localUbRmaBuffer);
        if (resultPair.first == localRmaBufferMgr->End()) {
            // 若已注册内存有交叉，返回HCCL_E_INTERNAL
            HCCL_ERROR("[%s]The memory overlaps with the memory that has been registered.", __func__);
            return HCCL_E_INTERNAL;
        }
        buf->addr = mem->addr;
        buf->len = mem->size;
        buf->handle = resultPair.first->second.buffer.get();
        return HCCL_SUCCESS;
    };
    TRY_CATCH_RETURN(getBuffFunc());

    HCCL_INFO("[%s]End, addr[%p], size[%llu], handle[%p]", __func__, buf->addr, buf->len, buf->handle);
    return HCCL_SUCCESS;
}

HcclResult HcclMemDeregV2(const HcclBuf* buf)
{
    if (buf == nullptr) {
        HCCL_ERROR("[%s]buf[%p] is null", __func__, buf);
        return HCCL_E_PTR;
    }
    HCCL_INFO("[%s] Begin, addr[%p], size[%llu], handle[%p]", __func__, buf->addr, buf->len, buf->handle);
    // 从LocalRamBuffer计数器删除HcclBuf
    LocalUbRmaBufferMgr* localRmaBufferMgr = LocalUbRmaBufferManager::GetInstance();
    BufferKey<uintptr_t, u64> tempKey(reinterpret_cast<uintptr_t>(buf->addr), buf->len);
    try {
        auto resultPair = localRmaBufferMgr->Del(tempKey);
        // 计数器大于1时，返回false，说明框架层有其它设备在使用这段内存，返回HCCL_E_AGAIN
        if (!resultPair) {
            HCCL_INFO("[HcclOneSidedService][DeregMem]Memory reference count is larger than 0"
                      "(used by other RemoteRank), do not deregister memory.");
            return HCCL_E_AGAIN;
        }
        return HCCL_SUCCESS;
    } catch (const std::out_of_range& e) {
        // 若计数器内未找到buf，返回HCCL_E_NOT_FOUND
        HCCL_ERROR("[%s] %s", __func__, e.what());
        return HCCL_E_NOT_FOUND;
    }
}

HcclResult HcclMemExportV2(HcclBuf* buf, char** outDesc, uint64_t* outDescLen)
{
    if (buf == nullptr || buf->handle == nullptr || outDesc == nullptr || outDescLen == nullptr) {
        HCCL_ERROR(
            "[%s] buf[%p] or buf->handle or outDesc[%p] or outDescLen[%p] is null", __func__, buf, outDesc, outDescLen);
        return HCCL_E_PTR;
    }
    HCCL_INFO("[%s] Begin, addr[%p], size[%llu], handle[%p]", __func__, buf->addr, buf->len, buf->handle);
    // 获取序列化信息
    LocalUbRmaBuffer* localUbRmaBuffer = reinterpret_cast<LocalUbRmaBuffer*>(buf->handle);
    std::unique_ptr<Serializable> dto = localUbRmaBuffer->GetExchangeDto();
    BinaryStream localRdmaRmaBufferStream;
    dto->Serialize(localRdmaRmaBufferStream);
    std::vector<char> tempLocalMemDesc;
    localRdmaRmaBufferStream.Dump(tempLocalMemDesc);
    HCCL_DEBUG("[%s] dump data size [%zu]", __func__, tempLocalMemDesc.size());
    // 判断内存描述符是否正确导出
    if (tempLocalMemDesc.empty()) {
        HCCL_ERROR("[%s] tempLocalMemDesc export failed.", __func__);
        return HCCL_E_INTERNAL;
    }

    // 内存描述符拷贝
    *outDescLen = tempLocalMemDesc.size();
    if (memcpy_s(*outDesc, TRANSPORT_EMD_ESC_SIZE, tempLocalMemDesc.data(), tempLocalMemDesc.size()) != EOK) {
        HCCL_ERROR("[%s] tempLocalMemDesc copy error. aim size:[%zu]", __func__, tempLocalMemDesc.size());
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[%s]End, outDescLen[%llu]", __func__, *outDescLen);
    return HCCL_SUCCESS;
}

HcclResult HcclMemImportV2(const char* description, uint64_t descLen, bool isRemote, HcclBuf* outBuf, HcclNetDev netDev)
{
    if (description == nullptr || outBuf == nullptr || netDev == nullptr) {
        HCCL_ERROR("[%s] description[%p] or outBuf[%p] or netDev[%p] is null", __func__, description, outBuf, netDev);
        return HCCL_E_PTR;
    }
    (void)(isRemote);
    HCCL_INFO("[%s] Begin, descLen[%llu]", __func__, descLen);
    // 仅支持UB类型
    HcclNetDevice* hcclNetDevice = static_cast<HcclNetDevice*>(netDev);
    if (!hcclNetDevice->IsUB()) {
        HCCL_ERROR("[%s] only support UB", __func__);
        return HCCL_E_NOT_SUPPORT;
    }

    // 反序列化
    std::vector<char> tempDesc{};
    tempDesc.resize(TRANSPORT_EMD_ESC_SIZE);
    tempDesc.assign(description, description + descLen);
    ExchangeUbBufferDto dto;
    BinaryStream remoteRdmaRmaBufferStream(tempDesc);
    dto.Deserialize(remoteRdmaRmaBufferStream);

    // 构造RemoteUbRmaBuffer
    RemoteUbRmaBuffer* remoteUbRmaBuffer = new (std::nothrow) RemoteUbRmaBuffer(hcclNetDevice->GetRdmaHandle(), dto);
    if (remoteUbRmaBuffer == nullptr) {
        HCCL_ERROR("[%s] Failed to allocate RemoteUbRmaBuffer", __func__);
        return HCCL_E_PTR;
    }

    // 填充HcclBuf
    outBuf->addr = reinterpret_cast<void*>(remoteUbRmaBuffer->GetAddr());
    outBuf->len = remoteUbRmaBuffer->GetSize();
    outBuf->handle = static_cast<void*>(remoteUbRmaBuffer);
    HCCL_INFO("[%s]End, addr[%p], size[%llu], handle[%p]", __func__, outBuf->addr, outBuf->len, outBuf->handle);
    return HCCL_SUCCESS;
}

HcclResult HcclMemCloseV2(HcclBuf* buf)
{
    if (buf == nullptr || buf->handle == nullptr) {
        HCCL_ERROR("[%s] buf[%p] or buf->handle is null", __func__, buf);
        return HCCL_E_PTR;
    }
    HCCL_INFO("[%s] Begin, addr[%p], size[%llu], handle[%p]", __func__, buf->addr, buf->len, buf->handle);
    // 仅支持UB类型
    RemoteRmaBuffer* remoteRmaBuffer = static_cast<RemoteRmaBuffer*>(buf->handle);
    if (remoteRmaBuffer->GetRmaType() != RmaType::UB) {
        HCCL_ERROR("[%s] only support UB", __func__);
        return HCCL_E_NOT_SUPPORT;
    }

    // 删除RemoteUbRmaBuffer
    HCCL_INFO("[HcclMemCloseV2][Ub] CloseMem");
    RemoteUbRmaBuffer* remoteUbRmaBuffer = static_cast<RemoteUbRmaBuffer*>(remoteRmaBuffer);
    delete remoteUbRmaBuffer;
    return HCCL_SUCCESS;
}
