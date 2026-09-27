/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transport_ipc_mem.h"
#include "log.h"
#include "dispatcher_pub.h"
#include "hccl_network.h"

namespace hccl {
using namespace std;
using LocalIpcRmaBufferMgr = NetDevContext::LocalIpcRmaBufferMgr;

TransportIpcMem::TransportIpcMem(
    const std::unique_ptr<NotifyPool>& notifyPool, const HcclNetDevCtx& netDevCtx, const HcclDispatcher& dispatcher,
    AttrInfo& attrInfo, bool aicpuUnfoldMode)
    : TransportMem(notifyPool, netDevCtx, dispatcher, attrInfo, aicpuUnfoldMode),
      sdid_(attrInfo.sdid),
      serverId_(attrInfo.serverId)
{}

TransportIpcMem::~TransportIpcMem() {}

HcclResult
TransportIpcMem::ExchangeMemDesc(const RmaMemDescs& localMemDescs, RmaMemDescs& remoteMemDescs, u32& actualNumOfRemote)
{
    return DoExchangeMemDesc(localMemDescs, remoteMemDescs, actualNumOfRemote);
}

HcclResult TransportIpcMem::EnableMemAccess(const RmaMemDesc& remoteMemDesc, RmaMem& remoteMem)
{
    std::string tempDesc = RmaMemDescCopyToStr(remoteMemDesc);

    // 创建远程缓冲区，并进行反序列化、打开
    std::shared_ptr<RemoteIpcRmaBuffer> tempRemoteBufferPtr = nullptr;
    EXCEPTION_CATCH((tempRemoteBufferPtr = make_shared<RemoteIpcRmaBuffer>(netDevCtx_)), return HCCL_E_PARA);
    HcclResult ret = tempRemoteBufferPtr->Deserialize(tempDesc);
    CHK_PRT_RET(
        (ret != HCCL_SUCCESS), HCCL_ERROR("[TransportIpcMem][EnableMemAccess]RemoteBuffer Deserialize failed."), ret);

    ret = tempRemoteBufferPtr->Open();
    CHK_PRT_RET((ret != HCCL_SUCCESS), HCCL_ERROR("[TransportIpcMem][EnableMemAccess]RemoteBuffer Open failed."), ret);

    BufferKey<uintptr_t, u64> tempKey(
        reinterpret_cast<uintptr_t>(tempRemoteBufferPtr->GetAddr()), tempRemoteBufferPtr->GetSize());
    auto resultPair = remoteIpcRmaBufferMgr_.Add(tempKey, tempRemoteBufferPtr);
    if (resultPair.first == remoteIpcRmaBufferMgr_.End()) {
        // 输入key与已有的内存重叠
        HCCL_ERROR("[TransportIpcMem][EnableMemAccess]The memory that is expected to enable"
                   " overlaps with the memory that has been enabled, please check params");
        return HCCL_E_INTERNAL;
    }

    // 已使能：输入key是表中某一最相近key的全集。 返回添加该key的迭代器，及false
    // 未使能：输入key是表中某一最相近key的空集。 返回添加成功的迭代器，及true
    std::string logInfo = resultPair.second ? "Enable memory access success!" :
                                              "Memory is already enabled, just increase the reference count.";
    HCCL_INFO("[TransportIpcMem][EnableMemAccess]:%s", logInfo.c_str());

    // 填充出参TransportRmaMem信息
    remoteMem.addr = tempRemoteBufferPtr->GetAddr();
    remoteMem.size = tempRemoteBufferPtr->GetSize();
    remoteMem.type = tempRemoteBufferPtr->GetMemType();
    return HCCL_SUCCESS;
}

HcclResult TransportIpcMem::DisableMemAccess(const RmaMemDesc& remoteMemDesc)
{
    // 内存去使能管理
    std::string tempDesc = RmaMemDescCopyToStr(remoteMemDesc);
    std::shared_ptr<RemoteIpcRmaBuffer> tempRemoteBuffer = make_shared<RemoteIpcRmaBuffer>(netDevCtx_);
    HcclResult ret = tempRemoteBuffer->Deserialize(tempDesc);
    CHK_PRT_RET(
        (ret != HCCL_SUCCESS), HCCL_ERROR("[TransportIpcMem][DisableMemAccess]RemoteBuffer Deserialize failed."), ret);

    ret = tempRemoteBuffer->Close();
    CHK_PRT_RET(
        (ret != HCCL_SUCCESS), HCCL_ERROR("[TransportIpcMem][DisableMemAccess]RemoteBuffer Close failed."), ret);

    BufferKey<uintptr_t, u64> tempKey(
        reinterpret_cast<uintptr_t>(tempRemoteBuffer->GetAddr()), tempRemoteBuffer->GetSize());
    try {
        if (remoteIpcRmaBufferMgr_.Del(tempKey)) {
            // 删除成功：输入key是表中某一最相近key的全集，计数-1后为0，返回true
            HCCL_INFO("[TransportIpcMem][DisableMemAccess]Memory reference count is 0, disable memory access.");
        } else {
            // 删除失败：输入key是表中某一最相近key的全集，计数不为0（存在其他remoteRank使用），返回false
            HCCL_INFO("[TransportIpcMem][DisableMemAccess]Memory reference count is larger than 0"
                      "(used by other RemoteRank), do not disable memory.");
        }
        return HCCL_SUCCESS;
    } catch (std::out_of_range& e) {
        HCCL_ERROR("[TransportIpcMem][DisableMemAccess] catch RmaBufferMgr Del exception: %s", e.what());
        return HCCL_E_NOT_FOUND;
    }
}

HcclResult TransportIpcMem::FillRmaBufferSlice(
    const HcclBuf& localMem, const HcclBuf& remoteMem, RmaBufferSlice& localRmaBufferSlice,
    RmaBufferSlice& remoteRmaBufferSlice)
{
    void* remoteAddr = remoteMem.addr;
    void* localAddr = localMem.addr;
    u64 byteSize = std::min(remoteMem.len, localMem.len);
    //  local-handle还在map中获取，remote-handle从外部传入
    auto localKey = BufferKey<uintptr_t, u64>(reinterpret_cast<uintptr_t>(localAddr), byteSize);

    NetDevContext* netDevCtx = static_cast<NetDevContext*>(netDevCtx_);
    std::shared_ptr<LocalIpcRmaBufferMgr> localRmaBufferMgr = netDevCtx->GetlocalIpcRmaBufferMgr();
    if (!localRmaBufferMgr) {
        HCCL_ERROR("[TransportIpcMem] can't get LocalIpcRmaBufferMgr");
        return HCCL_E_INTERNAL;
    }

    auto localBuffer = localRmaBufferMgr->Find(localKey);
    CHK_PRT_RET(
        !localBuffer.first,
        HCCL_ERROR(
            "[TransportIpcMem][FillRmaBufferSlice] Can't find localBuffer by key {%p, %llu}", localAddr, byteSize),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        !localBuffer.second->GetAddr(),
        HCCL_ERROR("[TransportIpcMem][FillRmaBufferSlice] The addr of local Buffer or remote buffer is nullptr."),
        HCCL_E_NOT_FOUND);
    CHK_PRT_RET(
        !localBuffer.second->GetDevAddr(),
        HCCL_ERROR("[TransportIpcMem][FillRmaBufferSlice] The dev addr of local Buffer is nullptr."), HCCL_E_NOT_FOUND);
    CHK_RET(CheckHcclBuffer(localAddr, localBuffer.second.get()));

    RmaBuffer* remoteBuffer = static_cast<RmaBuffer*>(remoteMem.handle);
    CHK_PRT_RET(
        !remoteBuffer->GetDevAddr(),
        HCCL_ERROR("[TransportIpcMem][FillRmaBufferSlice] The dev addr of remote buffer is nullptr."),
        HCCL_E_NOT_FOUND);
    CHK_PRT_RET(
        !remoteBuffer->GetAddr(),
        HCCL_ERROR("[TransportIpcMem][FillRmaBufferSlice] The addr of remote buffer is nullptr."), HCCL_E_NOT_FOUND);
    CHK_RET(CheckHcclBuffer(remoteAddr, remoteBuffer));

    u64 localDataOffSet = static_cast<u8*>(localAddr) - static_cast<u8*>(localBuffer.second->GetAddr());
    u64 remoteDataOffSet = static_cast<u8*>(remoteAddr) - static_cast<u8*>(remoteBuffer->GetAddr());
    localRmaBufferSlice.addr = static_cast<void*>(static_cast<u8*>(localBuffer.second->GetDevAddr()) + localDataOffSet);
    localRmaBufferSlice.len = byteSize;
    localRmaBufferSlice.rmaBuffer = localBuffer.second;
    localRmaBufferSlice.memType = localBuffer.second->GetMemType();
    remoteRmaBufferSlice.addr = static_cast<void*>(static_cast<u8*>(remoteBuffer->GetDevAddr()) + remoteDataOffSet);
    remoteRmaBufferSlice.len = byteSize;
    std::shared_ptr<RmaBuffer> temp(
        remoteBuffer, []([[maybe_unused]] RmaBuffer* p) {}); // 在外部进行删除操作，内部不能用智能指针进行生命周期管理
    remoteRmaBufferSlice.rmaBuffer = temp;
    remoteRmaBufferSlice.memType = remoteBuffer->GetMemType();
    HCCL_INFO(
        "[TransportIpcMem][FillRmaBufferSlice] Local address before mapping is [%p], after mapping is [%p]."
        "Remote address before mapping is [%p], after mapping is [%p]. Datasize is [%llu].",
        localAddr, localRmaBufferSlice.addr, remoteAddr, remoteRmaBufferSlice.addr, byteSize);
    return HCCL_SUCCESS;
}

HcclResult TransportIpcMem::FillRmaBufferSlice(
    const RmaOpMem& localMem, const RmaOpMem& remoteMem, RmaBufferSlice& localRmaBufferSlice,
    RmaBufferSlice& remoteRmaBufferSlice)
{
    void* remoteAddr = remoteMem.addr;
    void* localAddr = localMem.addr;
    u64 byteSize = std::min(remoteMem.size, localMem.size);

    auto localKey = BufferKey<uintptr_t, u64>(reinterpret_cast<uintptr_t>(localAddr), byteSize);
    auto remoteKey = BufferKey<uintptr_t, u64>(reinterpret_cast<uintptr_t>(remoteAddr), byteSize);

    NetDevContext* netDevCtx = static_cast<NetDevContext*>(netDevCtx_);
    std::shared_ptr<LocalIpcRmaBufferMgr> localRmaBufferMgr = netDevCtx->GetlocalIpcRmaBufferMgr();
    if (!localRmaBufferMgr) {
        HCCL_ERROR("[TransportIpcMem] can't get LocalIpcRmaBufferMgr");
        return HCCL_E_INTERNAL;
    }

    auto localBuffer = localRmaBufferMgr->Find(localKey);
    CHK_PRT_RET(
        !localBuffer.first,
        HCCL_ERROR(
            "[TransportIpcMem][FillRmaBufferSlice] Can't find localBuffer by key {%p, %llu}", localAddr, byteSize),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        !localBuffer.second->GetAddr(),
        HCCL_ERROR("[TransportIpcMem][FillRmaBufferSlice] The addr of local Buffer or remote buffer is nullptr."),
        HCCL_E_NOT_FOUND);
    CHK_PRT_RET(
        !localBuffer.second->GetDevAddr(),
        HCCL_ERROR("[TransportIpcMem][FillRmaBufferSlice] The dev addr of local Buffer is nullptr."), HCCL_E_NOT_FOUND);
    CHK_RET(CheckHcclBuffer(localAddr, localBuffer.second.get()));

    auto remoteBuffer = remoteIpcRmaBufferMgr_.Find(remoteKey);
    CHK_PRT_RET(
        !remoteBuffer.first,
        HCCL_ERROR(
            "[TransportIpcMem][FillRmaBufferSlice] Can't find remoteBuffer by key {%p, %llu}", remoteAddr, byteSize),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        !remoteBuffer.second->GetDevAddr(),
        HCCL_ERROR("[TransportIpcMem][FillRmaBufferSlice] The dev addr of remote buffer is nullptr."),
        HCCL_E_NOT_FOUND);
    CHK_PRT_RET(
        !remoteBuffer.second->GetAddr(),
        HCCL_ERROR("[TransportIpcMem][FillRmaBufferSlice] The addr of remote buffer is nullptr."), HCCL_E_NOT_FOUND);
    CHK_RET(CheckHcclBuffer(remoteAddr, remoteBuffer.second.get()));

    u64 localDataOffSet = static_cast<u8*>(localAddr) - static_cast<u8*>(localBuffer.second->GetAddr());
    u64 remoteDataOffSet = static_cast<u8*>(remoteAddr) - static_cast<u8*>(remoteBuffer.second->GetAddr());
    localRmaBufferSlice.addr = static_cast<void*>(static_cast<u8*>(localBuffer.second->GetDevAddr()) + localDataOffSet);
    localRmaBufferSlice.len = byteSize;
    localRmaBufferSlice.rmaBuffer = localBuffer.second;
    localRmaBufferSlice.memType = localBuffer.second->GetMemType();
    remoteRmaBufferSlice.addr
        = static_cast<void*>(static_cast<u8*>(remoteBuffer.second->GetDevAddr()) + remoteDataOffSet);
    remoteRmaBufferSlice.len = byteSize;
    remoteRmaBufferSlice.rmaBuffer = remoteBuffer.second;
    remoteRmaBufferSlice.memType = remoteBuffer.second->GetMemType();

    HCCL_INFO(
        "[TransportIpcMem][FillRmaBufferSlice] Local address before mapping is [%p], after mapping is [%p]."
        "Remote address before mapping is [%p], after mapping is [%p]. Datasize is [%llu].",
        localAddr, localRmaBufferSlice.addr, remoteAddr, remoteRmaBufferSlice.addr, byteSize);
    return HCCL_SUCCESS;
}

HcclResult TransportIpcMem::SetSocket([[maybe_unused]] const std::shared_ptr<HcclSocket>& socket)
{
    HCCL_INFO("TransportIpcMem doesn't need to set socket");
    return HCCL_SUCCESS;
}

HcclResult TransportIpcMem::Connect([[maybe_unused]] s32 timeoutSec)
{
    HCCL_INFO("TransportIpcMem doesn't need to connect socket");
    return HCCL_SUCCESS;
}

HcclResult TransportIpcMem::Write(const HcclBuf& remoteMem, const HcclBuf& localMem, const rtStream_t& stream)
{
    CHK_PRT_RET(
        (localMem.addr == nullptr) || (remoteMem.addr == nullptr),
        HCCL_ERROR("[TransportIpcMem]localMem addr or remoteMem addr is invalid"), HCCL_E_PARA);
    CHK_PRT_RET(
        (localMem.len == 0U) || (remoteMem.len == 0U),
        HCCL_ERROR(
            "[TransportIpcMem]localMem size[%llu] or remoteMem size[%llu] is invalid", localMem.len, remoteMem.len),
        HCCL_E_PARA);
    CHK_PRT_RET(stream == nullptr, HCCL_ERROR("[TransportIpcMem]stream is invalid"), HCCL_E_PARA);

    RmaBufferSlice localRmaBufferSlice{};
    RmaBufferSlice remoteRmaBufferSlice{};
    CHK_RET(FillRmaBufferSlice(localMem, remoteMem, localRmaBufferSlice, remoteRmaBufferSlice));
    return TransportIpc(remoteRmaBufferSlice, localRmaBufferSlice, stream);
}

HcclResult TransportIpcMem::Write(const RmaOpMem& remoteMem, const RmaOpMem& localMem, const rtStream_t& stream)
{
    CHK_PRT_RET(
        (localMem.addr == nullptr) || (remoteMem.addr == nullptr),
        HCCL_ERROR("[TransportIpcMem]localMem addr or remoteMem addr is invalid"), HCCL_E_PARA);
    CHK_PRT_RET(
        (localMem.size == 0U) || (remoteMem.size == 0U),
        HCCL_ERROR(
            "[TransportIpcMem]localMem size[%llu] or remoteMem size[%llu] is invalid", localMem.size, remoteMem.size),
        HCCL_E_PARA);
    CHK_PRT_RET(stream == nullptr, HCCL_ERROR("[TransportIpcMem]stream is invalid"), HCCL_E_PARA);

    RmaBufferSlice localRmaBufferSlice{};
    RmaBufferSlice remoteRmaBufferSlice{};
    CHK_RET(FillRmaBufferSlice(localMem, remoteMem, localRmaBufferSlice, remoteRmaBufferSlice));
    return TransportIpc(remoteRmaBufferSlice, localRmaBufferSlice, stream);
}

HcclResult TransportIpcMem::Read(const HcclBuf& localMem, const HcclBuf& remoteMem, const rtStream_t& stream)
{
    CHK_PRT_RET(
        (localMem.addr == nullptr) || (remoteMem.addr == nullptr),
        HCCL_ERROR("[TransportIpcMem]localMem addr or remoteMem addr is invalid"), HCCL_E_PARA);
    CHK_PRT_RET(
        (localMem.len == 0U) || (remoteMem.len == 0U),
        HCCL_ERROR(
            "[TransportIpcMem]localMem size[%llu] or remoteMem size[%llu] is invalid", localMem.len, remoteMem.len),
        HCCL_E_PARA);
    CHK_PRT_RET(stream == nullptr, HCCL_ERROR("[TransportIpcMem]stream is invalid"), HCCL_E_PARA);

    RmaBufferSlice localRmaBufferSlice{};
    RmaBufferSlice remoteRmaBufferSlice{};
    CHK_RET(FillRmaBufferSlice(localMem, remoteMem, localRmaBufferSlice, remoteRmaBufferSlice));
    return TransportIpc(localRmaBufferSlice, remoteRmaBufferSlice, stream);
}

HcclResult TransportIpcMem::Read(const RmaOpMem& localMem, const RmaOpMem& remoteMem, const rtStream_t& stream)
{
    CHK_PRT_RET(
        (localMem.addr == nullptr) || (remoteMem.addr == nullptr),
        HCCL_ERROR("[TransportIpcMem]localMem addr or remoteMem addr is invalid"), HCCL_E_PARA);
    CHK_PRT_RET(
        (localMem.size == 0U) || (remoteMem.size == 0U),
        HCCL_ERROR(
            "[TransportIpcMem]localMem size[%llu] or remoteMem size[%llu] is invalid", localMem.size, remoteMem.size),
        HCCL_E_PARA);
    CHK_PRT_RET(stream == nullptr, HCCL_ERROR("[TransportIpcMem]stream is invalid"), HCCL_E_PARA);

    RmaBufferSlice localRmaBufferSlice{};
    RmaBufferSlice remoteRmaBufferSlice{};
    CHK_RET(FillRmaBufferSlice(localMem, remoteMem, localRmaBufferSlice, remoteRmaBufferSlice));
    return TransportIpc(localRmaBufferSlice, remoteRmaBufferSlice, stream);
}

HcclResult TransportIpcMem::AddOpFence([[maybe_unused]] const rtStream_t& stream)
{
    HCCL_INFO("TransportIpcMem doesn't need to add op fence");
    return HCCL_SUCCESS;
}

HcclResult TransportIpcMem::GetMemInfo(u32& lkey, u32& rkey, HcclBuf& localMem, HcclBuf& remoteMem)
{
    CHK_PRT_RET(
        (localMem.addr == nullptr) || (remoteMem.addr == nullptr),
        HCCL_ERROR(
            "[TransportIpcMem] localMem addr[%p] or remoteMem addr[%p] is invalid", localMem.addr, remoteMem.addr),
        HCCL_E_PARA);
    CHK_PRT_RET(
        (localMem.len == 0U) || (remoteMem.len == 0U),
        HCCL_ERROR(
            "[TransportIpcMem] localMem size[%llu] or remoteMem size[%llu] is invalid", localMem.len, remoteMem.len),
        HCCL_E_PARA);

    RmaBufferSlice localRmaBufferSlice{};
    RmaBufferSlice remoteRmaBufferSlice{};
    CHK_RET(FillRmaBufferSlice(localMem, remoteMem, localRmaBufferSlice, remoteRmaBufferSlice));
    lkey = 0U;
    localMem.addr = localRmaBufferSlice.addr;
    localMem.len = localRmaBufferSlice.len;

    rkey = 0U;
    remoteMem.addr = remoteRmaBufferSlice.addr;
    remoteMem.len = remoteRmaBufferSlice.len;

    return HCCL_SUCCESS;
}

HcclResult TransportIpcMem::GetTransInfo(
    [[maybe_unused]] HcclQpInfoV2& qpInfo, u32* lkey, u32* rkey, HcclBuf* localMem, HcclBuf* remoteMem, u32 num)
{
    CHK_PTR_NULL(lkey);
    CHK_PTR_NULL(rkey);
    CHK_PTR_NULL(localMem);
    CHK_PTR_NULL(remoteMem);
    CHK_PRT_RET(num == 0, HCCL_ERROR("[GetTransInfo] mem num should not be zero"), HCCL_E_PARA);

    // GetTransInfo为TransportMem对外接口，TransportRoceMem将最后一个localMem用来反OpFence
    // Ipc mem无需Opfence，最后一个rkey/rkey/localMem/remoteMem空着
    for (u32 i = 0; i < num - 1; ++i) {
        HcclResult ret = GetMemInfo(lkey[i], rkey[i], localMem[i], remoteMem[i]);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[GetTransInfo] failed at index[%u], localAddr[%p,%llu], "
                "remoteAddr[%p,%llu]",
                i, localMem[i].addr, localMem[i].len, remoteMem[i].addr, remoteMem[i].len),
            ret);
    }
    return HCCL_SUCCESS;
}

HcclResult TransportIpcMem::WaitOpFence([[maybe_unused]] const rtStream_t& stream)
{
    HCCL_DEBUG("TransportIpcMem doesn't need to wait fence");
    return HCCL_SUCCESS;
}

HcclResult TransportIpcMem::BatchWrite(
    [[maybe_unused]] const std::vector<MemDetails>& remoteMems,
    [[maybe_unused]] const std::vector<MemDetails>& localMems, [[maybe_unused]] Stream& stream)
{
    HCCL_ERROR("TransportIpcMem doesn't support BatchWrite");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportIpcMem::BatchRead(
    [[maybe_unused]] const std::vector<MemDetails>& localMems,
    [[maybe_unused]] const std::vector<MemDetails>& remoteMems, [[maybe_unused]] Stream& stream)
{
    HCCL_ERROR("TransportIpcMem doesn't support BatchRead");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportIpcMem::AddOpFence(
    [[maybe_unused]] const MemDetails& localFenceMem, [[maybe_unused]] const MemDetails& remoteFenceMem,
    [[maybe_unused]] Stream& stream)
{
    HCCL_ERROR("TransportIpcMem doesn't support AICPU AddOpFence");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportIpcMem::TransportIpc(
    const RmaBufferSlice& dstRmaBufferSlice, const RmaBufferSlice& srcRmaBufferSlice, const rtStream_t& stream)
{
    CHK_PTR_NULL(dstRmaBufferSlice.addr);
    CHK_PTR_NULL(srcRmaBufferSlice.addr);
    Stream hcclStream(stream);
    DispatcherPub* dispatcher = reinterpret_cast<DispatcherPub*>(dispatcher_);
    if (dstRmaBufferSlice.memType == RmaMemType::HOST || srcRmaBufferSlice.memType == RmaMemType::HOST) {
        CHK_RET(dispatcher->MemcpyAsyncWithoutCheckKind(
            dstRmaBufferSlice.addr, dstRmaBufferSlice.len, srcRmaBufferSlice.addr, srcRmaBufferSlice.len,
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, hcclStream, remoteRankId_,
            hccl::LinkType::LINK_HCCS));
    } else {
        CHK_RET(dispatcher->MemcpyAsync(
            dstRmaBufferSlice.addr, dstRmaBufferSlice.len, srcRmaBufferSlice.addr, srcRmaBufferSlice.len,
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, hcclStream, remoteRankId_,
            hccl::LinkType::LINK_HCCS));
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
