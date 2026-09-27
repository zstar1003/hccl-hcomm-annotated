/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_mem.h"
#include "hccl_network.h"
#include "remote_ipc_rma_buffer.h"
#include "remote_rdma_rma_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif
HcclResult __attribute__((weak)) HcclMemRegV2(HcclNetDev netDev, const HcclMem* mem, HcclBuf* buf);
HcclResult __attribute__((weak)) HcclMemDeregV2(const HcclBuf* buf);
HcclResult __attribute__((weak)) HcclMemExportV2(HcclBuf* buf, char** outDesc, uint64_t* outDescLen);
HcclResult __attribute__((weak))
HcclMemImportV2(const char* description, uint64_t descLen, bool isRemote, HcclBuf* outBuf, HcclNetDev netDev);
HcclResult __attribute__((weak)) HcclMemCloseV2(HcclBuf* buf);
#ifdef __cplusplus
}
#endif

using namespace hccl;

using LocalIpcRmaBufferMgr = NetDevContext::LocalIpcRmaBufferMgr;
using LocalRdmaRmaBufferMgr = NetDevContext::LocalRdmaRmaBufferMgr;

static HcclResult HcclMemRegIpc(NetDevContext* netDevCtx, const HcclMem* mem, HcclBuf* buf)
{
    std::shared_ptr<LocalIpcRmaBufferMgr> localRmaBufferMgr = netDevCtx->GetlocalIpcRmaBufferMgr();
    if (!localRmaBufferMgr) {
        HCCL_ERROR("[HcclMemRegIpc]Can't get LocalIpcRmaBufferMgr");
        return HCCL_E_INTERNAL;
    }

    RmaMemType memType = static_cast<RmaMemType>(mem->type);
    u64 size = static_cast<u64>(mem->size);
    BufferKey<uintptr_t, u64> tempKey(reinterpret_cast<uintptr_t>(mem->addr), size);
    std::shared_ptr<LocalIpcRmaBuffer> localbufferPtr = nullptr;
    EXCEPTION_CATCH(
        (localbufferPtr = std::make_shared<LocalIpcRmaBuffer>(netDevCtx, mem->addr, size, memType)), return HCCL_E_PTR);
    auto resultPair = localRmaBufferMgr->Add(tempKey, localbufferPtr);
    if (resultPair.first == localRmaBufferMgr->End()) {
        // 输入key是表中某一个最相近key的交集、子集。返回空迭代器
        HCCL_ERROR("[HcclMemRegIpc]The memory that is expected to be"
                   " registered overlaps with the memory that has been registered, please check params");
        return HCCL_E_INTERNAL;
    }
    // 已注册：输入key是表中某一最相近key的全集。 返回添加该key的迭代器，及false
    // 未注册：输入key是表中某一最相近key的空集。 返回添加成功的迭代器，及true
    std::shared_ptr<LocalIpcRmaBuffer> localBuffer = resultPair.first->second.buffer;
    buf->addr = localBuffer->GetAddr();
    buf->len = localBuffer->GetSize();
    auto rmaBufferPtr = dynamic_cast<RmaBuffer*>(localBuffer.get());
    CHK_PTR_NULL(rmaBufferPtr);
    buf->handle = static_cast<void*>(rmaBufferPtr);
    if (resultPair.second) {
        HcclResult ret = localBuffer->Init();
        if (ret != HCCL_SUCCESS) {
            // 此分支中一定删除成功
            localRmaBufferMgr->Del(tempKey);
            HCCL_ERROR("[HcclMemRegRoce]localbuffer init failed %d.", ret);
            return ret;
        }
        HCCL_INFO("[HcclMemRegIpc]Register memory success! Add key {%p, %llu}", mem->addr, size);
        return HCCL_SUCCESS;
    } else { // 内存再次注册时
        HCCL_INFO(
            "[HcclMemRegIpc]Memory is already registered, just increase the reference count. Add key "
            "{%p, %llu}",
            mem->addr, size);
        ;
        return HCCL_E_AGAIN;
    }
}

static HcclResult HcclMemDeregIpc(NetDevContext* netDevCtx, const HcclBuf* buf)
{
    std::shared_ptr<LocalIpcRmaBufferMgr> localRmaBufferMgr = netDevCtx->GetlocalIpcRmaBufferMgr();
    if (!localRmaBufferMgr) {
        HCCL_ERROR("[HcclMemDeregIpc]Can't get LocalIpcRmaBufferMgr");
        return HCCL_E_INTERNAL;
    }

    BufferKey<uintptr_t, u64> tempKey(reinterpret_cast<uintptr_t>(buf->addr), buf->len);
    if (localRmaBufferMgr->Del(tempKey)) {
        // 删除成功：输入key是表中某一最相近key的全集，计数-1后为0，返回true
        HCCL_INFO("[HcclMemDeregIpc]Memory reference count is 0, deregister memory.");
        return HCCL_SUCCESS;
    } else {
        // 删除失败：输入key是表中某一最相近key的全集，计数不为0（存在其他remoteRank使用），返回false
        HCCL_INFO("[HcclMemDeregIpc]Memory reference count is larger than 0 "
                  "(used by other RemoteRank), do not deregister memory.");
        return HCCL_E_AGAIN;
    }
}

static HcclResult HcclMemRegRoce(NetDevContext* netDevCtx, const HcclMem* mem, HcclBuf* buf)
{
    std::shared_ptr<LocalRdmaRmaBufferMgr> localRmaBufferMgr = netDevCtx->GetlocalRdmaRmaBufferMgr();
    if (!localRmaBufferMgr) {
        HCCL_ERROR("[HcclMemRegRoce] can't get LocalRdmaRmaBufferMgr");
        return HCCL_E_INTERNAL;
    }

    RmaMemType memType = static_cast<RmaMemType>(mem->type);
    u64 size = static_cast<u64>(mem->size);
    BufferKey<uintptr_t, u64> tempKey(reinterpret_cast<uintptr_t>(mem->addr), size);
    std::shared_ptr<LocalRdmaRmaBuffer> localbufferPtr = nullptr;
    EXCEPTION_CATCH(
        (localbufferPtr = std::make_shared<LocalRdmaRmaBuffer>(netDevCtx, mem->addr, size, memType)),
        return HCCL_E_PTR);
    auto resultPair = localRmaBufferMgr->Add(tempKey, localbufferPtr);
    if (resultPair.first == localRmaBufferMgr->End()) {
        // 输入key是表中某一个最相近key的交集、子集。返回空迭代器
        HCCL_ERROR("[HcclMemRegRoce]The memory that is expected to be"
                   " registered overlaps with the memory that has been registered, please check params");
        return HCCL_E_INTERNAL;
    }
    // 已注册：输入key是表中某一最相近key的全集。 返回添加该key的迭代器，及false
    // 未注册：输入key是表中某一最相近key的空集。 返回添加成功的迭代器，及true
    std::shared_ptr<LocalRdmaRmaBuffer> localBuffer = resultPair.first->second.buffer;
    buf->addr = localBuffer->GetAddr();
    buf->len = localBuffer->GetSize();
    auto rmaBufferPtr = dynamic_cast<RmaBuffer*>(localBuffer.get());
    CHK_PTR_NULL(rmaBufferPtr);
    buf->handle = static_cast<void*>(rmaBufferPtr);
    if (resultPair.second) {
        HcclResult ret = localBuffer->Init();
        if (ret != HCCL_SUCCESS) {
            // 此分支中一定删除成功
            localRmaBufferMgr->Del(tempKey);
            HCCL_ERROR("[HcclMemRegRoce]localbuffer init failed %d.", ret);
            return ret;
        }
        HCCL_INFO("[HcclMemRegRoce]Register memory success! Add key {%p, %llu}", mem->addr, size);
        return HCCL_SUCCESS;
    } else { // 内存再次注册时
        HCCL_INFO(
            "[HcclMemRegRoce]Memory is already registered, just increase the reference count. Add key "
            "{%p, %llu}",
            mem->addr, size);
        ;
        return HCCL_E_AGAIN;
    }
}

static HcclResult HcclMemDeregRoce(NetDevContext* netDevCtx, const HcclBuf* buf)
{
    std::shared_ptr<LocalRdmaRmaBufferMgr> localRmaBufferMgr = netDevCtx->GetlocalRdmaRmaBufferMgr();
    if (!localRmaBufferMgr) {
        HCCL_ERROR("[HcclMemDeregRoce]Can't get LocalRdmaRmaBufferMgr");
        return HCCL_E_INTERNAL;
    }

    BufferKey<uintptr_t, u64> tempKey(reinterpret_cast<uintptr_t>(buf->addr), buf->len);
    if (localRmaBufferMgr->Del(tempKey)) {
        // 删除成功：输入key是表中某一最相近key的全集，计数-1后为0，返回true
        HCCL_INFO("[HcclMemDeregRoce]Memory reference count is 0, deregister memory.");
        return HCCL_SUCCESS;
    } else {
        // 删除失败：输入key是表中某一最相近key的全集，计数不为0（存在其他remoteRank使用），返回false
        HCCL_INFO("[HcclMemDeregRoce]Memory reference count is larger than 0 "
                  "(used by other RemoteRank), do not deregister memory.");
        return HCCL_E_AGAIN;
    }
}

static HcclResult HcclMemRempRoce(NetDevContext* netDevCtx, const HcclMem* memArray, u64 arraySize)
{
    std::shared_ptr<LocalRdmaRmaBufferMgr> localRmaBufferMgr = netDevCtx->GetlocalRdmaRmaBufferMgr();
    if (!localRmaBufferMgr) {
        HCCL_ERROR("[HcclMemRempRoce]Can't get LocalRdmaRmaBufferMgr");
        return HCCL_E_INTERNAL;
    }
    HCCL_RUN_INFO("[HcclMemRempRoce] arraySize[%u]", arraySize);
    std::unordered_map<void*, bool> remapAddr;
    for (u64 i = 0; i < arraySize; i++) {
        const HcclMem& memInfo = memArray[i];

        // 检查地址和大小是否有效
        if (memInfo.addr == nullptr || memInfo.size <= 0 || memInfo.type != HcclMemType::HCCL_MEM_TYPE_DEVICE) {
            continue;
        }

        // 检查地址是否已经处理过
        if (remapAddr.find(memInfo.addr) != remapAddr.end()) {
            continue;
        }

        // 查找地址是否注册过
        BufferKey<uintptr_t, u64> searchKey(reinterpret_cast<uintptr_t>(memInfo.addr), 1U);
        auto bufferIter = localRmaBufferMgr->Find(searchKey);
        if (!bufferIter.first) {
            HCCL_ERROR(
                "[HcclMemRempRoce]Memory addr[%p] size[%llu] has not been registered.", memInfo.addr, memInfo.size);
            return HCCL_E_PARA;
        }

        // 计算需要注册的内存大小
        u64 size = std::min(static_cast<u64>(memInfo.size), bufferIter.second->GetSize());

        // 注册内存
        HCCL_RUN_INFO("[HcclMemRempRoce]Re-register memory addr[%p] size[%llu].", memInfo.addr, size);
        HcclResult ret = bufferIter.second->Remap(memInfo.addr, size);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[HcclMemRempRoce]remap mem failed,addr[%p], size[%llu]", memInfo.addr, size), ret);

        // 标记地址已处理
        remapAddr.emplace(memInfo.addr, true);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclMemReg(HcclNetDev netDev, const HcclMem* mem, HcclBuf* buf)
{
    CHK_PTR_NULL(netDev);
    CHK_PTR_NULL(mem);
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(mem->addr);
    CHK_PRT_RET(
        (mem->type != HCCL_MEM_TYPE_DEVICE) && (mem->type != HCCL_MEM_TYPE_HOST),
        HCCL_ERROR("[HcclMemReg]memoryType[%d] must be device or host", mem->type), HCCL_E_PARA);
    CHK_PRT_RET(mem->size == 0, HCCL_ERROR("[HcclMemReg]memory size[%lld] is invalid", mem->size), HCCL_E_PARA);

    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        return HcclMemRegV2(netDev, mem, buf);
    }

    NetDevContext* netDevCtx = static_cast<NetDevContext*>(netDev);
    if (netDevCtx->GetNicType() == NicType::VNIC_TYPE) {
        return HcclMemRegIpc(netDevCtx, mem, buf);
    } else {
        return HcclMemRegRoce(netDevCtx, mem, buf);
    }
}

HcclResult HcclMemDereg(const HcclBuf* buf)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(buf->addr);
    CHK_PTR_NULL(buf->handle);
    CHK_PRT_RET(buf->len == 0U, HCCL_ERROR("[HcclMemDereg]buf size[%llu] is invalid", buf->len), HCCL_E_PARA);

    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        return HcclMemDeregV2(buf);
    }

    RmaBuffer* rmaBuffer = static_cast<RmaBuffer*>(buf->handle);
    NetDevContext* netDevCtx = static_cast<NetDevContext*>(const_cast<void*>(rmaBuffer->GetNetDevCtx()));
    if (netDevCtx->GetNicType() == NicType::VNIC_TYPE) {
        return HcclMemDeregIpc(netDevCtx, buf);
    } else {
        return HcclMemDeregRoce(netDevCtx, buf);
    }
}

HcclResult HcclMemRemap(HcclNetDev netDev, const HcclMem* memArray, uint64_t arraySize)
{
    CHK_PTR_NULL(netDev);
    CHK_PTR_NULL(memArray);
    CHK_PRT_RET(arraySize == 0U, HCCL_ERROR("[HcclMemReMap]arraySize[%llu] is invalid", arraySize), HCCL_E_PARA);

    NetDevContext* netDevCtx = static_cast<NetDevContext*>(netDev);
    if (netDevCtx->GetNicType() == NicType::VNIC_TYPE) {
        HCCL_INFO("[HcclMemReMap][ReMapMemIpc] doesn't support ReMapMem");
        return HCCL_SUCCESS;
    } else {
        return HcclMemRempRoce(netDevCtx, memArray, arraySize);
    }
}

HcclResult HcclMemExport(HcclBuf* buf, char** outDesc, uint64_t* outDescLen)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(outDesc);
    CHK_PTR_NULL(outDescLen);
    CHK_PTR_NULL(buf->addr);
    CHK_PTR_NULL(buf->handle);
    CHK_PRT_RET(buf->len == 0U, HCCL_ERROR("[HcclMemExport]buf size[%llu] is invalid", buf->len), HCCL_E_PARA);

    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        return HcclMemExportV2(buf, outDesc, outDescLen);
    }

    RmaBuffer* rmaBuffer = static_cast<RmaBuffer*>(buf->handle);
    if (rmaBuffer->GetRmaType() == RmaType::IPC_RMA) {
        LocalIpcRmaBuffer* localRmaBufer = dynamic_cast<LocalIpcRmaBuffer*>(rmaBuffer);
        CHK_PTR_NULL(localRmaBufer);
        std::string& tempLocalMemDesc = localRmaBufer->Serialize();
        if (tempLocalMemDesc.empty()) {
            HCCL_ERROR("[HcclMemExport][Ipc]tempLocalMemDesc is empty.");
            return HCCL_E_INTERNAL;
        }

        *outDesc = const_cast<char*>(tempLocalMemDesc.c_str());
        *outDescLen = tempLocalMemDesc.length();
    } else {
        LocalRdmaRmaBuffer* localRmaBufer = dynamic_cast<LocalRdmaRmaBuffer*>(rmaBuffer);
        CHK_PTR_NULL(localRmaBufer);
        std::string& tempLocalMemDesc = localRmaBufer->Serialize();
        if (tempLocalMemDesc.empty()) {
            HCCL_ERROR("[HcclMemExport][Roce]tempLocalMemDesc is empty.");
            return HCCL_E_INTERNAL;
        }

        *outDesc = const_cast<char*>(tempLocalMemDesc.c_str());
        *outDescLen = tempLocalMemDesc.length();
    }
    return HCCL_SUCCESS;
}

HcclResult HcclMemGrant(HcclBuf* localBuf, const HcclMemGrantInfo* remoteGrantInfo)
{
    CHK_PTR_NULL(localBuf);
    CHK_PTR_NULL(remoteGrantInfo);
    CHK_PTR_NULL(localBuf->addr);
    CHK_PTR_NULL(localBuf->handle);
    CHK_PRT_RET(localBuf->len == 0U, HCCL_ERROR("[HcclMemGrant]buf size[%llu] is invalid", localBuf->len), HCCL_E_PARA);
    RmaBuffer* rmaBuffer = static_cast<RmaBuffer*>(localBuf->handle);
    if (rmaBuffer->GetRmaType() == RmaType::IPC_RMA) {
        LocalIpcRmaBuffer* localRmaBufer = dynamic_cast<LocalIpcRmaBuffer*>(rmaBuffer);
        CHK_PTR_NULL(localRmaBufer);
        HcclResult ret = localRmaBufer->Grant(remoteGrantInfo->remotePid, remoteGrantInfo->remoteSdid);
        CHK_PRT_RET((ret != HCCL_SUCCESS), HCCL_ERROR("[HcclMemGrant]Grant error"), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult
HcclMemImport(const char* description, uint32_t descLen, bool isRemote, HcclBuf* outBuf, HcclNetDevCtx netDevCtx)
{
    CHK_PTR_NULL(netDevCtx);
    CHK_PTR_NULL(description);
    CHK_PTR_NULL(outBuf);
    CHK_PRT_RET(
        (descLen == 0), HCCL_ERROR("[HcclMemImport]input parameter is invalid descLen[%u] ", descLen), HCCL_E_PARA);
    CHK_PRT_RET(
        descLen > TRANSPORT_EMD_ESC_SIZE,
        HCCL_ERROR("[HcclMemImport]descLen[%u] is larger than limit[%u] ", descLen, TRANSPORT_EMD_ESC_SIZE),
        HCCL_E_PARA);
    if (isRemote == false) {
        HCCL_WARNING("[HcclMemImport]isRemote[%d] is invalid", isRemote);
    }

    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        return HcclMemImportV2(description, descLen, isRemote, outBuf, netDevCtx);
    }

    std::string tempDesc = std::string(description, descLen);
    u8 rmaType = static_cast<unsigned char>(description[0]);
    switch (rmaType) {
        case static_cast<int>(RmaType::IPC_RMA): {
            RemoteIpcRmaBuffer* tempRemoteBufferPtr = new (std::nothrow) RemoteIpcRmaBuffer(netDevCtx);
            CHK_PTR_NULL(tempRemoteBufferPtr);
            HcclResult deRet = tempRemoteBufferPtr->Deserialize(tempDesc);
            HcclResult openRet = tempRemoteBufferPtr->Open();
            if (deRet != HCCL_SUCCESS || openRet != HCCL_SUCCESS) {
                delete tempRemoteBufferPtr;
                CHK_PRT_RET(
                    deRet != HCCL_SUCCESS, HCCL_ERROR("[HcclMemImport]RemoteBuffer Deserialize failed."), deRet);
                CHK_PRT_RET(openRet != HCCL_SUCCESS, HCCL_ERROR("[HcclMemImport]RemoteBuffer Open failed."), openRet);
            }
            outBuf->addr = tempRemoteBufferPtr->GetAddr();
            outBuf->len = tempRemoteBufferPtr->GetSize();
            outBuf->handle = static_cast<void*>(tempRemoteBufferPtr);
            break;
        }
        case static_cast<int>(RmaType::RDMA_RMA): {
            RemoteRdmaRmaBuffer* tempRemoteBufferPtr = new (std::nothrow) RemoteRdmaRmaBuffer();
            CHK_PTR_NULL(tempRemoteBufferPtr);
            HcclResult ret = tempRemoteBufferPtr->Deserialize(tempDesc);
            if (ret != HCCL_SUCCESS) {
                delete tempRemoteBufferPtr;
                HCCL_ERROR("[HcclMemImport]RemoteBuffer Deserialize failed.");
                return ret;
            }
            outBuf->addr = tempRemoteBufferPtr->GetAddr();
            outBuf->len = tempRemoteBufferPtr->GetSize();
            outBuf->handle = static_cast<void*>(tempRemoteBufferPtr);
            break;
        }
        default: {
            HCCL_ERROR("[HcclMemImport]RmaType[%u] is invalid", rmaType);
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclMemClose(HcclBuf* buf)
{
    // remoteIpcRmaBufferMgr_ 和 remoteRdmaRmaBufferMgr_ 要抽到HcclOneSidedConn里
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(buf->handle);
    RmaBuffer* rmaBuffer = static_cast<RmaBuffer*>(buf->handle);

    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        return HcclMemCloseV2(buf);
    }

    if (rmaBuffer->GetRmaType() == RmaType::IPC_RMA) {
        HCCL_INFO("[HcclMemClose][Ipc] CloseMem");
        RemoteIpcRmaBuffer* tempRemoteBufferPtr = static_cast<RemoteIpcRmaBuffer*>(buf->handle);
        HcclResult ret = tempRemoteBufferPtr->Close();
        delete rmaBuffer;
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[HcclMemClose]RemoteBuffer Close failed"), ret);
    } else if (rmaBuffer->GetRmaType() == RmaType::RDMA_RMA) {
        HCCL_INFO("[HcclMemClose][Roce] CloseMem");
        delete rmaBuffer;
    } else {
        HCCL_ERROR("[HcclMemClose]RmaType[%d] is invalid", rmaBuffer->GetRmaType());
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}
