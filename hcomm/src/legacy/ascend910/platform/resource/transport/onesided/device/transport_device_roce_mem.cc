/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transport_device_roce_mem.h"
#include "log.h"
#include "dispatcher_pub.h"
#include "adapter_verbs.h"

namespace hccl {
std::atomic<u64> TransportDeviceRoceMem::wrIdOffset_;

TransportDeviceRoceMem::TransportDeviceRoceMem(
    const std::unique_ptr<NotifyPool>& notifyPool, const HcclNetDevCtx& netDevCtx, const HcclDispatcher& dispatcher,
    AttrInfo& attrInfo, bool aicpuUnfoldMode, const HcclQpInfoV2& qpInfo)
    : TransportMem(notifyPool, netDevCtx, dispatcher, attrInfo, aicpuUnfoldMode),
      timeout_{std::chrono::microseconds((attrInfo.timeout == INVALID_UINT) ? 0 : attrInfo.timeout)},
      qpInfo_{qpInfo}
{}

TransportDeviceRoceMem::~TransportDeviceRoceMem() {}

HcclResult TransportDeviceRoceMem::ExchangeMemDesc(
    [[maybe_unused]] const RmaMemDescs& localMemDescs, [[maybe_unused]] RmaMemDescs& remoteMemDescs,
    [[maybe_unused]] u32& actualNumOfRemote)
{
    HCCL_ERROR("TransportDeviceRoceMem doesn't support ExchangeMemDesc");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportDeviceRoceMem::EnableMemAccess(
    [[maybe_unused]] const RmaMemDesc& remoteMemDesc, [[maybe_unused]] RmaMem& remoteMem)
{
    HCCL_ERROR("TransportDeviceRoceMem doesn't support EnableMemAccess");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportDeviceRoceMem::DisableMemAccess([[maybe_unused]] const RmaMemDesc& remoteMemDesc)
{
    HCCL_ERROR("TransportDeviceRoceMem doesn't support DisableMemAccess");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportDeviceRoceMem::SetSocket([[maybe_unused]] const std::shared_ptr<HcclSocket>& socket)
{
    HCCL_ERROR("TransportDeviceRoceMem doesn't support SetSocket");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportDeviceRoceMem::Connect([[maybe_unused]] s32 timeoutSec)
{
    HCCL_ERROR("TransportDeviceRoceMem doesn't support Connect");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportDeviceRoceMem::Write(
    [[maybe_unused]] const HcclBuf& remoteMem, [[maybe_unused]] const HcclBuf& localMem,
    [[maybe_unused]] const rtStream_t& stream)
{
    HCCL_ERROR("TransportDeviceRoceMem doesn't support HcclBuf Write");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportDeviceRoceMem::Read(
    [[maybe_unused]] const HcclBuf& localMem, [[maybe_unused]] const HcclBuf& remoteMem,
    [[maybe_unused]] const rtStream_t& stream)
{
    HCCL_ERROR("TransportDeviceRoceMem doesn't support HcclBuf Read");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportDeviceRoceMem::Write(
    [[maybe_unused]] const RmaOpMem& remoteMem, [[maybe_unused]] const RmaOpMem& localMem,
    [[maybe_unused]] const rtStream_t& stream)
{
    HCCL_ERROR("TransportDeviceRoceMem doesn't support RmaOpMem Write");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportDeviceRoceMem::Read(
    [[maybe_unused]] const RmaOpMem& localMem, [[maybe_unused]] const RmaOpMem& remoteMem,
    [[maybe_unused]] const rtStream_t& stream)
{
    HCCL_ERROR("TransportDeviceRoceMem doesn't support RmaOpMem Read");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportDeviceRoceMem::AddOpFence([[maybe_unused]] const rtStream_t& stream)
{
    HCCL_ERROR("TransportDeviceRoceMem doesn't support HOST AddOpFence");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportDeviceRoceMem::GetTransInfo(
    [[maybe_unused]] HcclQpInfoV2& qpInfo, [[maybe_unused]] u32* lkey, [[maybe_unused]] u32* rkey,
    [[maybe_unused]] HcclBuf* localMem, [[maybe_unused]] HcclBuf* remoteMem, [[maybe_unused]] u32 num)
{
    HCCL_ERROR("TransportDeviceRoceMem doesn't support GetTransInfo");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportDeviceRoceMem::WaitOpFence([[maybe_unused]] const rtStream_t& stream)
{
    HCCL_ERROR("TransportDeviceRoceMem doesn't support WaitOpFence");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportDeviceRoceMem::TransportDeviceRoceMem::BatchWrite(
    const std::vector<MemDetails>& remoteMems, const std::vector<MemDetails>& localMems, Stream& stream)
{
    return BatchOp(stream, localMems, remoteMems, false, false);
}

HcclResult TransportDeviceRoceMem::BatchRead(
    const std::vector<MemDetails>& localMems, const std::vector<MemDetails>& remoteMems, Stream& stream)
{
    return BatchOp(stream, localMems, remoteMems, true, false);
}

HcclResult
TransportDeviceRoceMem::AddOpFence(const MemDetails& localFenceMem, const MemDetails& remoteFenceMem, Stream& stream)
{
    std::vector<MemDetails> localMems(1, localFenceMem);
    std::vector<MemDetails> remoteMems(1, remoteFenceMem);
    return BatchOp(stream, localMems, remoteMems, true, true);
}

HcclResult TransportDeviceRoceMem::DoorBellSend(Stream& stream, u64 dbInfo, u32 wrDataLen, bool fence)
{
    HCCL_DEBUG("[DoorBellSend] dbIndex[%#x] dbInfo[%#llx] remoteRankId[%u]", qpInfo_.dbIndex, dbInfo, remoteRankId_);
    RdmaTaskInfo rdmaInfo;
    WrInformation wrInfo;
    wrInfo.notifyId = 0;
    wrInfo.wrData.memList.len = wrDataLen;
    rdmaInfo.wrInfos.emplace_back(wrInfo);
    rdmaInfo.rdmaType = fence ? RdmaType::RDMA_SEND_NOTIFY : RdmaType::RDMA_SEND_PAYLOAD;
    rdmaInfo.remoteRank = remoteRankId_;
    HcclResult ret = HCCL_SUCCESS;
    DispatcherPub* dispatcher = static_cast<DispatcherPub*>(dispatcher_);
    ret = dispatcher->RdmaSend(qpInfo_.dbIndex, dbInfo, stream, rdmaInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[DoorBellSend] RdmaSend failed, ret[%u]. dbIndex[%#x] dbInfo[%#llx] remoteRankId[%u]", ret,
            qpInfo_.dbIndex, dbInfo, rdmaInfo.remoteRank),
        ret);
    std::vector<Stream> subStreams;
    ret = dispatcher->LaunchTasksEx(stream, subStreams);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[DoorBellSend] LaunchTask failed, ret[%u]. dbIndex[%#x] dbInfo[%#llx] remoteRankId[%u]", ret,
            qpInfo_.dbIndex, dbInfo, rdmaInfo.remoteRank),
        ret);
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceRoceMem::FillMemDetails(
    std::vector<MemDetails>& localMemList, std::vector<MemDetails>& remoteMemList, MemDetails& localMem,
    MemDetails& remoteMem)
{
    CHK_PRT_RET(
        localMem.size != remoteMem.size,
        HCCL_ERROR(
            "[TransportDeviceRoceMem][FillMemDetails] "
            "local buffer size[%llu] is not equal to remote buffer size[%llu]",
            localMem.size, remoteMem.size),
        HCCL_E_PARA);
    u64 remainingBytes = localMem.size;
    while (remainingBytes > 0) {
        const u64 chunkBytes = (remainingBytes > MAX_RDMA_WQE_SIZE) ? MAX_RDMA_WQE_SIZE : remainingBytes;
        localMem.size = chunkBytes;
        remoteMem.size = chunkBytes;
        localMemList.emplace_back(localMem);
        remoteMemList.emplace_back(remoteMem);
        localMem.addr += chunkBytes;
        remoteMem.addr += chunkBytes;
        remainingBytes -= chunkBytes;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceRoceMem::BatchOp(
    Stream& stream, const std::vector<MemDetails>& localMems, const std::vector<MemDetails>& remoteMems, bool isRead,
    bool fence)
{
    constexpr u32 MAX_RDMA_WQE_NUM = 64; // related to qp depth
    CHK_PRT_RET(
        localMems.size() != remoteMems.size(),
        HCCL_ERROR(
            "[TransportDeviceRoceMem][BatchOp] "
            "local buffer num[%llu] is not equal to remote buffer num[%llu]",
            localMems.size(), remoteMems.size()),
        HCCL_E_PARA);
    u64 dbInfo = 0;
    u32 wqeCount = 0;
    u64 wrDataLen = 0;
    const u32 memNum = localMems.size();
    for (u32 index = 0; index < memNum; index++) {
        MemDetails localMem = localMems[index];
        MemDetails remoteMem = remoteMems[index];
        wrDataLen += localMem.size;
        std::vector<MemDetails> localMemList;
        std::vector<MemDetails> remoteMemList;
        CHK_RET(FillMemDetails(localMemList, remoteMemList, localMem, remoteMem));
        CHK_RET(BatchPostSend(stream, dbInfo, localMemList, remoteMemList, isRead, fence, wqeCount, wrDataLen));
        if (wqeCount >= MAX_RDMA_WQE_NUM) {
            CHK_RET(DoorBellSend(stream, dbInfo, wrDataLen, fence));
            wqeCount = 0;
            wrDataLen = 0;
        }
    }
    if (wqeCount != 0) {
        CHK_RET(DoorBellSend(stream, dbInfo, wrDataLen, fence));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceRoceMem::BatchPostSend(
    Stream& stream, u64& dbInfo, std::vector<MemDetails>& localMemList, std::vector<MemDetails>& remoteMemList,
    bool isRead, bool fence, u32& wqeCount, u64& wrDataLen)
{
    const u32 wrTotalCount = localMemList.size();
    u32 sendWrCount = 0;
    while (sendWrCount < wrTotalCount) {
        const u32 wrCount = std::min(wrTotalCount - sendWrCount, SEND_WR_LEN);
        CHK_RET(PostSend(
            stream, dbInfo, &(localMemList[sendWrCount]), &(remoteMemList[sendWrCount]), wrCount, isRead, fence,
            wqeCount, wrDataLen));
        sendWrCount += wrCount;
        wqeCount += wrCount;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceRoceMem::PostSend(
    Stream& stream, u64& dbInfo, struct MemDetails* localMems, struct MemDetails* remoteMems, u32 memNum, bool isRead,
    bool fence, u32& wqeCount, u64& wrDataLen)
{
    constexpr u32 RETRY_DELAY_THRESH = 100;
    u32 retryCount = 0;
    auto startTime = std::chrono::steady_clock::now();
    HcclResult ret = HCCL_E_NETWORK;
    while (ret != HCCL_SUCCESS) {
        RdmaOp opCode = isRead ? RdmaOp::OP_READ : RdmaOp::OP_WRITE;
        ret = RdmaPostSend(dbInfo, localMems, remoteMems, memNum, opCode, fence);
        if (ret == HCCL_E_AGAIN) {
            if ((retryCount == 0) && (wqeCount != 0)) {
                HCCL_WARNING(
                    "[PostSend] retry with DoorBellSend, isRead[%u] remoteRankId[%u] wqeCount[%u]", isRead,
                    remoteRankId_, wqeCount);
                CHK_RET(DoorBellSend(stream, dbInfo, wrDataLen, fence));
                wqeCount = 0;
                wrDataLen = 0;
            } else {
                CHK_PRT_RET(
                    timeout_ == std::chrono::microseconds(0),
                    HCCL_ERROR("[PostSend] failed without retry, isRead[%u] remoteRankId[%u]", isRead, remoteRankId_),
                    ret);
                auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - startTime);
                CHK_PRT_RET(
                    elapsedTime >= timeout_,
                    HCCL_ERROR(
                        "[PostSend] failed after timeout, elapsedTime[%lld us] isRead[%u] remoteRankId[%u]",
                        elapsedTime.count(), isRead, remoteRankId_),
                    ret);
                if (retryCount % RETRY_DELAY_THRESH == 0) {
                    HCCL_WARNING(
                        "[PostSend] retryCount[%u] after failed, elapsedTime[%lld us] isRead[%u] "
                        "remoteRankId[%u]",
                        retryCount, elapsedTime.count(), isRead, remoteRankId_);
                }
                SaluSleep(
                    ONE_MILLISECOND_OF_USLEEP * std::min(CeilDiv(retryCount, RETRY_DELAY_THRESH), RETRY_DELAY_THRESH));
            }
            ++retryCount;
            continue; // to retry
        }
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[PostSend] HnsPostSend failed[%u], isRead[%u] remoteRankId[%u]", ret, isRead, remoteRankId_),
            ret);
        if (retryCount != 0) {
            HCCL_INFO("[PostSend] retry success, isRead[%u] remoteRankId[%u]", isRead, remoteRankId_);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceRoceMem::RdmaPostSend(
    u64& dbInfo, MemDetails* localMems, MemDetails* remoteMems, u32 memNum, RdmaOp opCode, bool fence)
{
    CHK_PTR_NULL(localMems);
    CHK_PTR_NULL(remoteMems);

    CHK_PRT_RET(
        memNum > SEND_WR_LEN,
        HCCL_ERROR(
            "[TransportDeviceRoceMem][RdmaPostSend] buffer size is:%u over SEND_WR_LEN: %u", memNum, SEND_WR_LEN),
        HCCL_E_PARA);
    const u32 last = memNum - 1;
    struct ibv_send_wr sendWr[SEND_WR_LEN] = {};
    struct ibv_sge sge[SEND_WR_LEN] = {};
    for (u32 index = 0; index < memNum; index++) {
        // 设置WR的SGE
        sge[index].addr = localMems[index].addr;
        sge[index].length = remoteMems[index].size;
        sge[index].lkey = localMems[index].key;

        // 设置WR属性
        sendWr[index].wr_id = wrIdOffset_.fetch_add(1, std::memory_order_relaxed);
        sendWr[index].num_sge = 1; // 只有一个SGE
        sendWr[index].sg_list = &sge[index];
        sendWr[index].wr.rdma.remote_addr = remoteMems[index].addr;
        sendWr[index].wr.rdma.rkey = remoteMems[index].key;
        sendWr[index].next = (index == last) ? nullptr : &sendWr[index + 1];         // 第一个WR指向第二个WR
        sendWr[index].send_flags = fence ? (IBV_SEND_SIGNALED | IBV_SEND_FENCE) : 0; // addfence时才需要回复CQE
        sendWr[index].opcode = static_cast<enum ibv_wr_opcode>(opCode);
        HCCL_DEBUG(
            "[TransportDeviceRoceMem][RdmaPostSend] Direct ibv_post_send[%llu], opcode=[0x%x], "
            "remote_addr=[0x%llx], size=[%u], fence[%u], send_flags[%d]",
            wrIdOffset_.load(), sendWr[index].opcode, sendWr[index].wr.rdma.remote_addr, sendWr[index].sg_list->length,
            fence, sendWr[index].send_flags);
    }

    struct ibv_send_wr* badWr = nullptr;
    struct WrExpRsp exp_rsp = {};
    struct ibv_qp* qp = reinterpret_cast<struct ibv_qp*>(qpInfo_.qpPtr);
    CHK_PTR_NULL(qp);
    HCCL_DEBUG(
        "[TransportDeviceRoceMem][RdmaPostSend] qp=%p, handle=%u, qp_num=%u, qp_type=%d, qp_stat=%d", qp, qp->handle,
        qp->qp_num, qp->qp_type, qp->state);
    HcclResult ret = HrtHnsIbvExpPostSend(qp, &sendWr[0], &badWr, &exp_rsp);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN,
        HCCL_ERROR(
            "[TransportDeviceRoceMem][RdmaPostSend] failed, qp=%p, handle=%u, qp_num=%u, qp_type=%d, qp_stat=%d", qp,
            qp->handle, qp->qp_num, qp->qp_type, qp->state),
        ret);
    if (ret == HCCL_SUCCESS) {
        dbInfo = exp_rsp.db_info;
    }
    return ret;
}
} // namespace hccl
