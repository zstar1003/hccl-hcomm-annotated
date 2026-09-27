/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "send_recv_executor.h"
#include "device_capacity.h"
#include "dispatcher_pub.h"
#include "adapter_rts.h"
#include "externalinput_pub.h"

namespace hccl {
constexpr u32 HCCL_POLL_CQ_DEPTH = 32;
constexpr s32 HCCL_POLL_CQ_ONETIME = 1;
constexpr u32 HCCL_POLL_CQ_INTERVAL = 100;
constexpr u64 MAX_RDMA_WQE_SIZE = 2ULL * 1024 * 1024 * 1024; // RDMA最大WQE限制, 2G限制是RDMA导致

struct MrInfoT AscendMrInfo2MrInfo(AscendMrInfo* ascendMrInfo)
{
    struct MrInfoT innerMrInfo = {};
    innerMrInfo.addr = reinterpret_cast<void*>(ascendMrInfo->addr);
    innerMrInfo.size = ascendMrInfo->size;
    innerMrInfo.lkey = ascendMrInfo->key;
    return innerMrInfo;
}

SendRecvExecutor::SendRecvExecutor(
    HcclRtStream stream, QpHandle qpHandle, const struct MrInfoT& localWindowMem, const struct MrInfoT& remoteWindowMem,
    const struct MrInfoT& localSyncMemPrepare, const struct MrInfoT& localSyncMemDone,
    const struct MrInfoT& localSyncMemAck, const struct MrInfoT& remoteSyncMemPrepare,
    const struct MrInfoT& remoteSyncMemDone, const struct MrInfoT& remoteSyncMemAck, u32 immData, const u64 chunkNum)
    : stream_(stream),
      qpHandle_(qpHandle),
      localWindowMem_(localWindowMem),
      remoteWindowMem_(remoteWindowMem),
      localSyncMemPrepare_(localSyncMemPrepare),
      localSyncMemDone_(localSyncMemDone),
      localSyncMemAck_(localSyncMemAck),
      remoteSyncMemPrepare_(remoteSyncMemPrepare),
      remoteSyncMemDone_(remoteSyncMemDone),
      remoteSyncMemAck_(remoteSyncMemAck),
      immData_(immData),
      chunkSize_(chunkNum),
      notifyWaitMode_(SyncMode::DEFAULT_TIMEWAITSYNCMODE)
{}

SendRecvExecutor::SendRecvExecutor(HcclRtStream stream, QpHandle qpHandle, AscendSendRecvLinkInfo* linkInfo)
    : stream_(stream),
      qpHandle_(qpHandle),
      localSyncMemPrepare_(AscendMrInfo2MrInfo(linkInfo->localSyncMemPrepare)),
      localSyncMemDone_(AscendMrInfo2MrInfo(linkInfo->localSyncMemDone)),
      localSyncMemAck_(AscendMrInfo2MrInfo(linkInfo->localSyncMemAck)),
      remoteSyncMemPrepare_(AscendMrInfo2MrInfo(linkInfo->remoteSyncMemPrepare)),
      remoteSyncMemDone_(AscendMrInfo2MrInfo(linkInfo->remoteSyncMemDone)),
      remoteSyncMemAck_(AscendMrInfo2MrInfo(linkInfo->remoteSyncMemAck)),
      immData_(linkInfo->immData),
      wqePerDoorBell_(linkInfo->wqePerDoorbell)
{}

SendRecvExecutor::SendRecvExecutor(HcclRtStream stream, QpHandle qpHandle, AscendSendLinkInfo* linkInfo)
    : stream_(stream),
      qpHandle_(qpHandle),
      localSyncMemAck_(AscendMrInfo2MrInfo(linkInfo->localSyncMemAck)),
      wqePerDoorBell_(linkInfo->wqePerDoorbell),
      remoteNotifyValueMem_(AscendMrInfo2MrInfo(linkInfo->remoteNotifyValueMem))
{}

SendRecvExecutor::SendRecvExecutor(
    HcclRtStream stream, QpHandle qpHandle, AscendMrInfo* localSyncMemDone, AscendMrInfo* remoteSyncMemAck)
    : stream_(stream),
      qpHandle_(qpHandle),
      localSyncMemDone_(AscendMrInfo2MrInfo(localSyncMemDone)),
      remoteSyncMemAck_(AscendMrInfo2MrInfo(remoteSyncMemAck))
{}

SendRecvExecutor::~SendRecvExecutor() {}

HcclResult SendRecvExecutor::Init()
{
    CHK_RET(hrtGetNotifySize(notifySize_));
    if (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET) {
        notifyWaitMode_ = SyncMode::CONFIGURABLE_TIMEWAITSYNCMODE;
    }
    CHK_RET(TypicalSyncMem::GetInstance().GetNotifyHandle(
        reinterpret_cast<u64>(localSyncMemPrepare_.addr), prepareNotify_));
    CHK_PTR_NULL(prepareNotify_);

    CHK_RET(TypicalSyncMem::GetInstance().GetNotifyHandle(reinterpret_cast<u64>(localSyncMemAck_.addr), ackNotify_));
    CHK_PTR_NULL(ackNotify_);

    CHK_RET(TypicalSyncMem::GetInstance().GetNotifyHandle(reinterpret_cast<u64>(localSyncMemDone_.addr), doneNotify_));
    CHK_PTR_NULL(doneNotify_);

    CHK_RET(TypicalSyncMem::GetInstance().GetNotifySrcMem(notifySrcMem_));
    CHK_PTR_NULL(notifySrcMem_.addr);

    HCCL_INFO(
        "[SendRecvExecutor][Init] SendRecvExecutor init success! notifySize[%u], notifyWaitMode[%d], "
        "prepareNotify[%p], ackNotify[%p], doneNotify[%p], notifySrcMem addr[%p], localWindowMem addr[%p], "
        "remoteWindowMem addr[%p], remoteSyncMemPrepare addr[%p], remoteSyncMemDone addr[%p], "
        "remoteSyncMemAck addr[%p], immData[%u], wqePerDoorBell[%u]",
        notifySize_, notifyWaitMode_, prepareNotify_, ackNotify_, doneNotify_, notifySrcMem_.addr, localWindowMem_.addr,
        remoteWindowMem_.addr, remoteSyncMemPrepare_.addr, remoteSyncMemDone_.addr, remoteSyncMemAck_.addr, immData_,
        wqePerDoorBell_);
    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::WaitPutInit()
{
    CHK_RET(hrtGetNotifySize(notifySize_));
    if (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET) {
        notifyWaitMode_ = SyncMode::CONFIGURABLE_TIMEWAITSYNCMODE;
    }
    CHK_RET(TypicalSyncMem::GetInstance().GetNotifyHandle(reinterpret_cast<u64>(localSyncMemDone_.addr), doneNotify_));
    CHK_PTR_NULL(doneNotify_);

    CHK_RET(TypicalSyncMem::GetInstance().GetNotifySrcMem(notifySrcMem_));
    CHK_PTR_NULL(notifySrcMem_.addr);

    HCCL_INFO(
        "[SendRecvExecutor][WaitPutInit] SendRecvExecutor init success! notifySize[%u], notifyWaitMode[%d], "
        "prepareNotify[%p], ackNotify[%p], doneNotify[%p], notifySrcMem addr[%p], localWindowMem addr[%p], "
        "remoteWindowMem addr[%p], remoteSyncMemPrepare addr[%p], remoteSyncMemDone addr[%p], "
        "remoteSyncMemAck addr[%p], immData[%u], wqePerDoorBell[%u]",
        notifySize_, notifyWaitMode_, prepareNotify_, ackNotify_, doneNotify_, notifySrcMem_.addr, localWindowMem_.addr,
        remoteWindowMem_.addr, remoteSyncMemPrepare_.addr, remoteSyncMemDone_.addr, remoteSyncMemAck_.addr, immData_,
        wqePerDoorBell_);
    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::IsOverlappedWithWinMem(void* userPtr, u64 userMemSize, bool& isOverlapped)
{
    if (userPtr >= localWindowMem_.addr
        && static_cast<u8*>(userPtr) + userMemSize <= static_cast<u8*>(localWindowMem_.addr) + localWindowMem_.size) {
        isOverlapped = true;
    } else if (
        static_cast<u8*>(userPtr) + userMemSize <= static_cast<u8*>(localWindowMem_.addr)
        || static_cast<u8*>(userPtr) >= static_cast<u8*>(localWindowMem_.addr) + localWindowMem_.size) {
        isOverlapped = false;
    } else {
        HCCL_ERROR(
            "[SendRecvExecutor][IsOverlappedWithWinMem] The user mem addr or size is illegal. "
            "The addr of user mem is %p, user mem size is %llu. The addr of window mem is %p, window mem size is %llu.",
            userPtr, userMemSize, localWindowMem_.addr, localWindowMem_.size);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::Send(void* inputPtr, u64 count, HcclDataType dataType)
{
    HcclResult ret = HCCL_SUCCESS;
    void* windowsMemPtr = localWindowMem_.addr;
    uint32_t unitSize = SIZE_TABLE[dataType];
    uint8_t* curInputPtr = static_cast<uint8_t*>(inputPtr);
    CHK_PTR_NULL(curInputPtr);
    uint64_t inputOffset = 0;
    uint64_t countLeft = count;
    Stream streamObj(stream_);
    // 判断userMem是否是windowMem的一部分
    bool isOverlapped = false;
    CHK_RET(IsOverlappedWithWinMem(inputPtr, count * unitSize, isOverlapped));

    u64 maxCountPerLoop = localWindowMem_.size / unitSize;

    while (countLeft > 0) {
        // 防止数据回绕
        CHK_PRT_RET(countLeft > count, HCCL_ERROR("[SendRecvExecutor][Send] countLeft is underflow."), HCCL_E_PARA);
        curInputPtr += inputOffset;
        HCCL_DEBUG("[SendRecvExecutor]][Send] InputOffset[%llu]", inputOffset);
        u64 curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize;
        HCCL_DEBUG(
            "[SendRecvExecutor][Send] curInputPtr[%p], curCount[%llu], curSize[%llu]", curInputPtr, curCount, curSize);
        DeviceMem inMem(curInputPtr, curSize);
        // 如果userMem是否是windowMem的一部分，跳过D2D拷
        if (isOverlapped) {
            ret = SendRun(inMem);
        } else {
            DeviceMem inWindowMem(windowsMemPtr, curSize);
            CHK_RET(MemcpyAsyncD2D(inWindowMem, inMem, streamObj));
            ret = SendRun(inWindowMem);
        }
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[SendRecvExecutor][Send] errNo[0x%016llx] send error, ptr[%p], count[%llu], dataType[%d]",
                HCCL_ERROR_CODE(ret), windowsMemPtr, curCount, dataType),
            ret);
        CHK_PRT_RET((curCount == 0), HCCL_ERROR("[SendRecvExecutor]][Send]In OP_BASE curCount is zero"), HCCL_E_PARA);
        countLeft -= curCount;
        inputOffset = curSize;
    }
    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::Receive(void* outputPtr, u64 count, HcclDataType dataType)
{
    HcclResult ret = HCCL_SUCCESS;
    void* windowsMemPtr = localWindowMem_.addr;
    uint32_t unitSize = SIZE_TABLE[dataType];
    uint8_t* curOutPutPtr = static_cast<uint8_t*>(outputPtr);
    CHK_PTR_NULL(curOutPutPtr);
    uint64_t outputOffset = 0;
    uint64_t countLeft = count;
    Stream streamObj(stream_);
    u64 maxCountPerLoop = localWindowMem_.size / unitSize;

    while (countLeft > 0) {
        // 防止数据回绕
        CHK_PRT_RET(countLeft > count, HCCL_ERROR("[SendRecvExecutor][Receive] countLeft is underflow."), HCCL_E_PARA);
        curOutPutPtr += outputOffset;
        HCCL_INFO("[SendRecvExecutor][Receive] inputOffset[%llu]", outputOffset);
        u64 curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位 byte
        HCCL_INFO(
            "[SendRecvExecutor][Receive] curOutPutPtr[%p], curCount[%llu], curSize[%llu]", curOutPutPtr, curCount,
            curSize);
        DeviceMem outMem(curOutPutPtr, curSize);
        DeviceMem outWindowMem(windowsMemPtr, curSize);
        if (immData_ != 0) {
            ret = ReceiveRunByPollCq(outWindowMem);
        } else {
            ret = ReceiveRun(outWindowMem);
        }
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[SendRecvExecutor][Receive] errNo[0x%016llx] Receive error, ptr[%p], count[%llu], dataType[%d]",
                HCCL_ERROR_CODE(ret), windowsMemPtr, curCount, dataType),
            ret);
        CHK_RET(MemcpyAsyncD2D(outMem, outWindowMem, streamObj));
        CHK_PRT_RET((curCount == 0), HCCL_ERROR("[SendRecvExecutor][Receive]In OP_BASE curCount is zero"), HCCL_E_PARA);
        countLeft -= curCount;
        outputOffset = curSize;
    }
    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::Put(void* inputPtr, u64 count, HcclDataType dataType)
{
    HcclResult ret = HCCL_SUCCESS;
    void* windowsMemPtr = localWindowMem_.addr;
    uint32_t unitSize = SIZE_TABLE[dataType];
    uint8_t* curInputPtr = static_cast<uint8_t*>(inputPtr);
    CHK_PTR_NULL(curInputPtr);
    uint64_t inputOffset = 0;
    uint64_t countLeft = count;
    Stream streamObj(stream_);
    // 判断userMem是否是windowMem的一部分
    bool isOverlapped = false;
    CHK_RET(IsOverlappedWithWinMem(inputPtr, count * unitSize, isOverlapped));

    u64 maxCountPerLoop = localWindowMem_.size / unitSize;

    while (countLeft > 0) {
        // 防止数据回绕
        CHK_PRT_RET(countLeft > count, HCCL_ERROR("[SendRecvExecutor][Put] countLeft is underflow."), HCCL_E_PARA);
        curInputPtr += inputOffset;
        HCCL_DEBUG("[SendRecvExecutor][Put] InputOffset[%llu]", inputOffset);
        u64 curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize;
        HCCL_DEBUG(
            "[SendRecvExecutor][Put] curInputPtr[%p], curCount[%llu], curSize[%llu]", curInputPtr, curCount, curSize);
        DeviceMem inMem(curInputPtr, curSize);
        // 如果userMem是否是windowMem的一部分，跳过D2D拷贝
        if (isOverlapped) {
            ret = PutRun(inMem);
        } else {
            DeviceMem inWindowMem(windowsMemPtr, curSize);
            CHK_RET(MemcpyAsyncD2D(inWindowMem, inMem, streamObj));
            ret = PutRun(inWindowMem);
        }
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[SendRecvExecutor][Put] errNo[0x%016llx] send error, ptr[%p], count[%llu], dataType[%d]",
                HCCL_ERROR_CODE(ret), windowsMemPtr, curCount, dataType),
            ret);
        CHK_PRT_RET((curCount == 0), HCCL_ERROR("[SendRecvExecutor]][Put]In OP_BASE curCount is zero"), HCCL_E_PARA);
        countLeft -= curCount;
        inputOffset = curSize;
    }
    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::SendRun(DeviceMem& sendBuffer)
{
    HcclResult ret = HCCL_SUCCESS;
    if (!sendBuffer) {
        HCCL_ERROR("[SendRecvExecutor][SendRun] Send buffer ptr is null.");
        return HCCL_E_PTR;
    }
    u64 sizePerRound = 0;
    u64 sizePerSlice = chunkSize_;
    u64 length = sendBuffer.size();
    u64 offset = 0;

    for (u64 sizeResidue = length; sizeResidue > 0; sizeResidue -= sizePerRound) {
        // 防止数据回绕
        CHK_PRT_RET(
            sizeResidue > length, HCCL_ERROR("[SendRecvExecutor][SendRun] countLeft is underflow."), HCCL_E_PARA);
        ret = WaitSignal(prepareNotify_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendRecvExecutor][SendRun] Wait prepare failed"), ret);
        offset += sizePerRound;
        sizePerRound = (sizeResidue > sizePerSlice) ? sizePerSlice : sizeResidue;
        void* localAddr = static_cast<u8*>(localWindowMem_.addr) + offset;
        HCCL_INFO("rx async inputmem's offset[%llu] size[%llu]", offset, sizePerRound);

        ret = PayLoad(localAddr, offset, sizePerRound);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[SendRecvExecutor][SendRun] Send data fail with offset[%llu] "
                "size[%llu] failed",
                offset, sizePerRound),
            ret);

        if (immData_ == 0) {
            ret = RecordNotify(
                remoteSyncMemDone_.addr, remoteSyncMemDone_.lkey, notifySrcMem_.addr, notifySrcMem_.lkey, notifySize_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendRecvExecutor][SendRun] Record done failed"), ret);
        }

        ret = WaitSignal(ackNotify_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendRecvExecutor][SendRun] Wait ack failed"), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::PutRun(DeviceMem& putBuffer)
{
    HcclResult ret = HCCL_SUCCESS;
    if (!putBuffer) {
        HCCL_ERROR("[SendRecvExecutor][PutRun] Send buffer ptr is null.");
        return HCCL_E_PTR;
    }
    u64 sizePerRound = 0;
    u64 sizePerSlice = chunkSize_;
    u64 length = putBuffer.size();
    u64 offset = 0;

    for (u64 sizeResidue = length; sizeResidue > 0; sizeResidue -= sizePerRound) {
        // 防止数据回绕
        CHK_PRT_RET(
            sizeResidue > length, HCCL_ERROR("[SendRecvExecutor][PutRun] countLeft is underflow."), HCCL_E_PARA);
        offset += sizePerRound;
        sizePerRound = (sizeResidue > sizePerSlice) ? sizePerSlice : sizeResidue;
        void* localAddr = static_cast<u8*>(localWindowMem_.addr) + offset;
        HCCL_INFO("rx async inputmem's offset[%llu] size[%llu]", offset, sizePerRound);

        ret = PayLoad(localAddr, offset, sizePerRound);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[SendRecvExecutor][PutRun] Send data fail with offset[%llu] "
                "size[%llu] failed",
                offset, sizePerRound),
            ret);

        if (immData_ == 0) {
            ret = RecordNotify(
                remoteSyncMemDone_.addr, remoteSyncMemDone_.lkey, notifySrcMem_.addr, notifySrcMem_.lkey, notifySize_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendRecvExecutor][PutRun] Record done failed"), ret);
        }

        ret = WaitSignal(ackNotify_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendRecvExecutor][PutRun] Wait ack failed"), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::PollCq()
{
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(static_cast<s32>(GetExternalInputHcclExecTimeOut()));

    struct ibv_wc wc[HCCL_POLL_CQ_DEPTH];
    while ((std::chrono::steady_clock::now() - startTime) < timeout) {
        s32 num = hrtRaPollCq(qpHandle_, false, HCCL_POLL_CQ_ONETIME, wc);
        if (num < 0) {
            HCCL_ERROR("[SendRecvExecutor][PollCq] Poll Cq error, return [%d]", num);
            return HCCL_E_INTERNAL;
        } else if (num < HCCL_POLL_CQ_ONETIME) {
            SaluSleep(HCCL_POLL_CQ_INTERVAL);
            continue;
        }
        for (int i = 0; i < num; i++) {
            if (wc[i].status != 0) {
                HCCL_ERROR("rdma poll tag sq failed, cqe status[%u]", wc[i].status);
                return HCCL_E_INTERNAL;
            }
        }
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[SendRecvExecutor][PollCq] Wait Cqe timeOut[%d] s", GetExternalInputHcclLinkTimeOut());
    return HCCL_E_TIMEOUT;
}

HcclResult SendRecvExecutor::ReceiveRunByPollCq(DeviceMem& receiveBuffer)
{
    HcclResult ret = HCCL_SUCCESS;
    if (!receiveBuffer) {
        HCCL_ERROR("[SendRecvExecutor][ReceiveRunByPollCq] Receive buffer ptr is null.");
        return HCCL_E_PTR;
    }
    u64 sizePerRound = 0;
    u64 sizePerSlice = chunkSize_;
    u64 length = receiveBuffer.size();

    u64 offset = 0;

    for (u64 sizeResidue = length; sizeResidue > 0; sizeResidue -= sizePerRound) {
        // 防止数据回绕
        CHK_PRT_RET(
            sizeResidue > length, HCCL_ERROR("[SendRecvExecutor][ReceiveRunByPollCq] countLeft is underflow."),
            HCCL_E_PARA);
        offset += sizePerRound;
        sizePerRound = (sizeResidue > sizePerSlice) ? sizePerSlice : sizeResidue;
        HCCL_INFO("rx async inputmem's offset[%llu] size[%llu]", offset, sizePerRound);

        void* localAddr = static_cast<u8*>(localWindowMem_.addr) + offset;

        std::vector<struct RecvWrlistData> recvWrVec(1);
        recvWrVec[0].wrId = reinterpret_cast<u64>(localWindowMem_.addr);
        recvWrVec[0].memList.addr = reinterpret_cast<u64>(localAddr);
        recvWrVec[0].memList.len = sizePerRound;
        recvWrVec[0].memList.lkey = localWindowMem_.lkey;

        struct RecvWrlistData* recvWr = recvWrVec.data();
        u32 completeNum = 0;
        ret = hrtRaRecvWrlist(qpHandle_, recvWr, 1, &completeNum);
        if (ret == HCCL_SUCCESS && completeNum == 1) {
            HCCL_INFO("[SendRecvExecutor][ReceiveRunByPollCq] Exec hrtRaRecvWrlist success.");
        } else {
            HCCL_ERROR(
                "[SendRecvExecutor][ReceiveRunByPollCq] In RdmaDataTransport, hrtRaRecvWrlist failed. ret[%d], "
                "completeNum[%d].",
                ret, completeNum);
            return HCCL_E_NETWORK;
        }

        ret = RecordNotify(
            remoteSyncMemPrepare_.addr, remoteSyncMemPrepare_.lkey, notifySrcMem_.addr, notifySrcMem_.lkey,
            notifySize_);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("SendRecvExecutor][ReceiveRunByPollCq] Record prepare failed"), ret);

        CHK_RET(PollCq());

        ret = RecordNotify(
            remoteSyncMemAck_.addr, remoteSyncMemAck_.lkey, notifySrcMem_.addr, notifySrcMem_.lkey, notifySize_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendRecvExecutor][ReceiveRunByPollCq] Record ack failed"), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::ReceiveRun(DeviceMem& receiveBuffer)
{
    HcclResult ret = HCCL_SUCCESS;
    if (!receiveBuffer) {
        HCCL_ERROR("[SendRecvExecutor][ReceiveRun] Receive buffer ptr is null.");
        return HCCL_E_PTR;
    }
    u64 sizePerRound = 0;
    u64 sizePerSlice = chunkSize_;
    u64 length = receiveBuffer.size();
    u64 offset = 0;

    for (u64 sizeResidue = length; sizeResidue > 0; sizeResidue -= sizePerRound) {
        // 防止数据回绕
        CHK_PRT_RET(
            sizeResidue > length, HCCL_ERROR("[SendRecvExecutor][ReceiveRun] countLeft is underflow."), HCCL_E_PARA);
        ret = RecordNotify(
            remoteSyncMemPrepare_.addr, remoteSyncMemPrepare_.lkey, notifySrcMem_.addr, notifySrcMem_.lkey,
            notifySize_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("SendRecvExecutor][ReceiveRun] Record prepare failed"), ret);
        offset += sizePerRound;
        sizePerRound = (sizeResidue > sizePerSlice) ? sizePerSlice : sizeResidue;

        HCCL_INFO("[SendRecvExecutor][ReceiveRun]rx async inputmem's offset[%llu] size[%llu]", offset, sizePerRound);

        ret = WaitSignal(doneNotify_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendRecvExecutor][ReceiveRun] Wait done failed"), ret);

        ret = RecordNotify(
            remoteSyncMemAck_.addr, remoteSyncMemAck_.lkey, notifySrcMem_.addr, notifySrcMem_.lkey, notifySize_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendRecvExecutor][ReceiveRun] Wait ack failed"), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::RecordNotify(
    void* dstMemPtr, u32 rkey, const void* srcMemPtr, u32 lkey, u64 srcMemSize, uint32_t rdmaOp, int sendFlag)
{
    struct SgList list = {};
    struct SendWrV2 wr = {};
    // 构造wr信息
    list.addr = static_cast<u64>(reinterpret_cast<uintptr_t>(srcMemPtr));
    list.len = srcMemSize;
    list.lkey = lkey;

    wr.bufList = &list;
    wr.bufNum = 1; /* 此处list只有一个，设置为1 */
    wr.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(dstMemPtr));
    wr.rkey = rkey;
    wr.op = rdmaOp;
    wr.sendFlag = sendFlag;

    HCCL_INFO(
        "[SendRecvExecutor][RecordNotify] "
        "Notify's dst addr[%p], local addr[%p], data's len[%u], remote mr key[%u], local mr key[%u]",
        wr.dstAddr, wr.bufList->addr, wr.bufList->len, wr.rkey, wr.bufList->lkey);

    // RDMA异步发送
    CHK_RET(RdmaSendAsync(wr));
    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::WaitSignal(HcclRtSignal signal)
{
    if (notifyWaitMode_ == SyncMode::CONFIGURABLE_TIMEWAITSYNCMODE) {
        // GetExternalInputHcclExecTimeOut 返回 double，接口参数为整型秒数
        CHK_RET(hrtNotifyWaitWithTimeOut(
            static_cast<HcclRtNotify>(signal), stream_, static_cast<s32>(GetExternalInputHcclExecTimeOut())));
    } else {
        CHK_RET(hrtNotifyWaitWithTimeOut(static_cast<HcclRtNotify>(signal), stream_, NOTIFY_DEFAULT_WAIT_TIME));
    }
    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::PayLoad(const void* src, u64 dstOffset, u64 len)
{
    HcclResult ret;
    HCCL_DEBUG(
        "[SendRecvExecutor][PayLoad] Local window memory srcPtr[%p] len[%llu] dstOffset[%llu]", src, len, dstOffset);

    u32 txSendDataTimes = (len == 0) ? 1 : (len + RDMA_SEND_MAX_SIZE - 1) / RDMA_SEND_MAX_SIZE;

    for (u32 txSendDataIdx = 0; txSendDataIdx < txSendDataTimes; txSendDataIdx++) {
        u64 txSendDataOffset = txSendDataIdx * RDMA_SEND_MAX_SIZE;
        u64 txSendDataSize = (txSendDataIdx == (txSendDataTimes - 1)) ? len - txSendDataOffset : RDMA_SEND_MAX_SIZE;

        void* txdstMemPtr
            = reinterpret_cast<void*>(reinterpret_cast<u8*>(remoteWindowMem_.addr) + dstOffset + txSendDataOffset);

        const void* txsrcMemPtr = reinterpret_cast<const void*>(reinterpret_cast<const char*>(src) + txSendDataOffset);
        struct SendWrV2 wr {};
        // 构造wr信息
        wr.bufNum = 1; /* 此处list只有一个，设置为1 */
        wr.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(txdstMemPtr));
        wr.rkey = remoteWindowMem_.lkey;
        wr.sendFlag = RA_SEND_SIGNALED;
        if (immData_ != 0) {
            wr.op = RA_WR_RDMA_WRITE_WITH_IMM;
            wr.ext.immData = immData_;
        } else {
            wr.op = RA_WR_RDMA_WRITE;
        }
        struct SgList list = {};
        list.addr = static_cast<u64>(reinterpret_cast<uintptr_t>(txsrcMemPtr));
        list.len = txSendDataSize;
        list.lkey = localWindowMem_.lkey;
        wr.bufList = &list;
        ret = RdmaSendAsync(wr);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[SendRecvExecutor][PayLoad]errNo[0x%016llx] In lbv exp, add wqe list failed."
                "srcMemSize[%llu]",
                HCCL_ERROR_CODE(ret), txSendDataSize),
            ret);
    }

    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::RdmaSendAsync(struct SendWrV2& wr)
{
    HcclResult ret = HCCL_SUCCESS;
    struct SendWrRsp opRsp = {};
    HCCL_DEBUG(
        "[SendRecvExecutor][RdmaSendAsync] dst_addr[%p], src_addr[%p], len[%u]", wr.dstAddr, wr.bufList->addr,
        wr.bufList->len);

    CHK_RET(HrtRaSendWrV2(qpHandle_, &wr, &opRsp, HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));

    u32 dbIndex = static_cast<u32>(opRsp.db.dbIndex);
    u64 dbInfo = static_cast<u64>(opRsp.db.dbInfo);

    if ((dbIndex == INVALID_UINT) && (dbInfo == INVALID_U64)) {
        // zero byte message 不需要下发rdma send task
        HCCL_DEBUG("[SendRecvExecutor][RdmaSendAsync] dbIndex and dbInfo is invalid.");
        return HCCL_SUCCESS;
    }

    ret = hrtRDMADBSend(dbIndex, dbInfo, stream_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[SendRecvExecutor][RdmaSendAsync]errNo[0x%016llx] In lbv exp op base mode, "
            "rdma send failed. dbIndex[%u] dbInfo[%llu]",
            HCCL_ERROR_CODE(ret), dbIndex, dbInfo),
        ret);
    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::MemcpyAsyncD2D(hccl::DeviceMem& dst, const hccl::DeviceMem& src, hccl::Stream& stream)
{
    CHK_PTR_NULL(dst.ptr());
    CHK_PTR_NULL(src.ptr());

    if (stream.ptr() == nullptr) {
        CHK_SAFETY_FUNC_RET(memcpy_s(dst.ptr(), dst.size(), src.ptr(), src.size()));
        return HCCL_E_PARA;
    }

    if (src.size() == 0) {
        HCCL_DEBUG("[SendRecvExecutor][MemcpyAsyncD2D] count is 0, return success.");
        return HCCL_SUCCESS;
    }

    uint64_t spiltLoop = 0;
    uint64_t addrOffset = 0;
    uint64_t contSplit = 0;
    if (src.size() > HCCL_SDMA_MAX_COUNT_4GB) {
        spiltLoop = (src.size() % HCCL_SDMA_MAX_COUNT_4GB != 0) ? (src.size() / HCCL_SDMA_MAX_COUNT_4GB) :
                                                                  ((src.size() / HCCL_SDMA_MAX_COUNT_4GB) - 1);
        HCCL_INFO(
            "[SendRecvExecutor][MemcpyAsyncD2D] MemcpyAsync SDMA task countSize is bigger than 4GB "
            "and do segmentation splitloop[%llu]",
            spiltLoop);
    }
    /* SDMA任务拆分 */
    for (uint64_t index = 0; index <= spiltLoop; index++) {
        addrOffset = index * HCCL_SDMA_MAX_COUNT_4GB;
        contSplit = (index == spiltLoop) ? (src.size() - index * HCCL_SDMA_MAX_COUNT_4GB) : (HCCL_SDMA_MAX_COUNT_4GB);
        void* srcSplit = static_cast<void*>(static_cast<u8*>(const_cast<void*>(src.ptr())) + addrOffset);
        void* dstSplit = static_cast<void*>(static_cast<u8*>(dst.ptr()) + addrOffset);

        CHK_RET(hrtMemAsyncCopy(
            dstSplit, dst.size(), const_cast<const void*>(srcSplit), contSplit,
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream.ptr()));
    }
    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::BatchPutMR(u32 num, AscendMrInfo* putMRList, AscendMrInfo* remoteMRList)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 sendWrNum = 0;
    for (u32 i = 0; i < num; i++) {
        if (i != num - 1) {
            CHK_RET(PayLoadMR(putMRList + i, remoteMRList + i, sendWrNum));
        } else {
            // 最后一组数据特殊处理，发送立即数
            CHK_RET(PayLoadMR(putMRList + i, remoteMRList + i, sendWrNum, true));
        }
    }

    if (immData_ == 0) {
        ret = RecordNotify(
            remoteSyncMemDone_.addr, remoteSyncMemDone_.lkey, notifySrcMem_.addr, notifySrcMem_.lkey, notifySize_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendRecvExecutor][BatchPutMR] Record done failed"), ret);
    }

    ret = WaitSignal(ackNotify_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendRecvExecutor][BatchPutMR] Wait ack failed"), ret);

    return HCCL_SUCCESS;
}

HcclResult
SendRecvExecutor::PayLoadMR(AscendMrInfo* putMRInfo, AscendMrInfo* remoteMRInfo, u32& wrNum, bool isLastMRtoPut)
{
    CHK_PRT_RET(
        putMRInfo->size != remoteMRInfo->size,
        HCCL_ERROR(
            "[SendRecvExecutor][PayLoadMR] The size of localMR"
            "is different from remoteMR. LocalMR size is [%u], remoteMR size is[%u].",
            putMRInfo->size, remoteMRInfo->size),
        HCCL_E_PARA);
    u64 remainingSize = putMRInfo->size;
    u64 offSet = 0;
    u64 byteChunkSize = 0;
    u64 localAddr = 0;
    u64 remoteAddr = 0;
    bool isLastSlice = false;
    while (remainingSize > 0) {
        localAddr = putMRInfo->addr + offSet;
        remoteAddr = remoteMRInfo->addr + offSet;
        byteChunkSize = remainingSize > MAX_RDMA_WQE_SIZE ? MAX_RDMA_WQE_SIZE : remainingSize;
        isLastSlice = remainingSize > MAX_RDMA_WQE_SIZE ? false : true;
        struct SgList list = {};
        list.addr = localAddr;
        list.len = byteChunkSize;
        list.lkey = putMRInfo->key;
        struct SendWrV2 wr = {};
        wr.bufList = &list;
        wr.bufNum = 1; /* 此处list只有一个，设置为1 */
        wr.dstAddr = remoteAddr;
        wr.rkey = remoteMRInfo->key;
        if (isLastMRtoPut && isLastSlice && immData_ != 0) {
            wr.op = RA_WR_RDMA_WRITE_WITH_IMM;
            wr.ext.immData = immData_;
        } else {
            wr.op = RA_WR_RDMA_WRITE;
        }

        CHK_RET(MultiWqeOneDoorBellSend(isLastMRtoPut && isLastSlice, wrNum, wr));
        remainingSize -= byteChunkSize;
        offSet += byteChunkSize;
    }
    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::MultiWqeOneDoorBellSend(bool isLastWr, u32& wrNum, struct SendWrV2& wr)
{
    HcclResult ret = HCCL_SUCCESS;
    wrNum++;
    // 多个wqe生成一个cqe
    if (isLastWr || wrNum == wqePerDoorBell_) {
        wr.sendFlag = RA_SEND_SIGNALED;
    } else {
        wr.sendFlag = 0;
    }
    struct SendWrRsp opRsp = {};
    CHK_RET(HrtRaSendWrV2(qpHandle_, &wr, &opRsp, HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    HCCL_DEBUG(
        "[SendRecvExecutor][MultiWqeOneDoorBellSend] End SendWr, wr op[%d], localAddr[%p], remoteAddr[%p], "
        "len[%llu], local key[%u], remote key[%u].",
        wr.op, wr.bufList->addr, wr.dstAddr, wr.bufList->len, wr.bufList->lkey, wr.rkey);

    if (static_cast<u32>(opRsp.db.dbIndex) == INVALID_UINT && static_cast<u64>(opRsp.db.dbInfo) == INVALID_U64) {
        // zero byte message 不需要下发rdma send task
        HCCL_DEBUG("[SendRecvExecutor][MultiWqeOneDoorBellSend] dbIndex and dbInfo is invalid.");
        return HCCL_SUCCESS;
    }

    // 每下发wqePerDoorBell_个wr敲一次doorbell
    if (isLastWr || wrNum == wqePerDoorBell_) {
        u32 dbIndex = static_cast<u32>(opRsp.db.dbIndex);
        u64 dbInfo = static_cast<u64>(opRsp.db.dbInfo);
        HCCL_DEBUG(
            "[SendRecvExecutor][MultiWqeOneDoorBellSend] Start RDMADBSend, dbIndex[%u], dbInfo[%llu], wrNum[%u], "
            "isLastMR[%d].",
            dbIndex, dbInfo, wrNum, isLastWr);
        ret = hrtRDMADBSend(dbIndex, dbInfo, stream_);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[SendRecvExecutor][MultiWqeOneDoorBellSend]errNo[0x%016llx] In lbv exp op base mode, "
                "rdma send failed. dbIndex[%u] dbInfo[%llu]",
                HCCL_ERROR_CODE(ret), dbIndex, dbInfo),
            ret);
        wrNum = 0;
    }
    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::WaitPutMR()
{
    HcclResult ret = HCCL_SUCCESS;

    ret = WaitSignalUnlimitedTime(doneNotify_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendRecvExecutor][WaitPutMR] Wait done failed"), ret);

    ret = RecordNotify(
        remoteSyncMemAck_.addr, remoteSyncMemAck_.lkey, notifySrcMem_.addr, notifySrcMem_.lkey, notifySize_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendRecvExecutor][WaitPutMR] Record ack failed"), ret);

    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::ProcessRCQ(AscendMrInfo* lastMRInfo)
{
    HcclResult ret = HCCL_SUCCESS;
    u64 remainingSize = lastMRInfo->size;
    u64 localAddr = lastMRInfo->addr;
    u64 offSet = 0;
    while (remainingSize > MAX_RDMA_WQE_SIZE) {
        remainingSize -= MAX_RDMA_WQE_SIZE;
        offSet += MAX_RDMA_WQE_SIZE;
    }
    std::vector<struct RecvWrlistData> recvWrVec(1);
    recvWrVec[0].wrId = localAddr;
    recvWrVec[0].memList.addr = localAddr + offSet;
    recvWrVec[0].memList.len = remainingSize;
    recvWrVec[0].memList.lkey = lastMRInfo->key;

    struct RecvWrlistData* recvWr = recvWrVec.data();
    u32 completeNum = 0;
    ret = hrtRaRecvWrlist(qpHandle_, recvWr, 1, &completeNum);
    if (ret == HCCL_SUCCESS && completeNum == 1) {
        HCCL_INFO("[SendRecvExecutor][TestBatchPutMR] Exec hrtRaRecvWrlist success.");
    } else {
        HCCL_ERROR(
            "[SendRecvExecutor][TestBatchPutMR] In RdmaDataTransport, hrtRaRecvWrlist failed. ret[%d], "
            "completeNum[%d].",
            ret, completeNum);
        return HCCL_E_NETWORK;
    }

    CHK_RET(PollCq());
    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::WaitSignalUnlimitedTime(HcclRtSignal signal)
{
    // 超时时间配置成0，代表永不超时
    CHK_RET(hrtNotifyWaitWithTimeOut(static_cast<HcclRtNotify>(signal), stream_, 0));
    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::WaitPutMROnlyWait()
{
    HcclResult ret = HCCL_SUCCESS;

    ret = WaitSignalUnlimitedTime(doneNotify_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendRecvExecutor][WaitPutMROnlyWait] Wait done failed"), ret);

    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::WaitPutMROnlyRecord()
{
    HcclResult ret = HCCL_SUCCESS;

    ret = RecordNotify(
        remoteSyncMemAck_.addr, remoteSyncMemAck_.lkey, notifySrcMem_.addr, notifySrcMem_.lkey, notifySize_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendRecvExecutor][WaitPutMROnlyWait] Record ack failed"), ret);

    return HCCL_SUCCESS;
}

HcclResult SendRecvExecutor::OneSideBatchPutMR(u32 num, AscendMrInfo* putMRList, AscendMrInfo* remoteMRList)
{
    CHK_RET(hrtGetNotifySize(notifySize_));
    if (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET) {
        notifyWaitMode_ = SyncMode::CONFIGURABLE_TIMEWAITSYNCMODE;
    }
    CHK_RET(TypicalSyncMem::GetInstance().GetNotifyHandle(reinterpret_cast<u64>(localSyncMemAck_.addr), ackNotify_));
    CHK_PTR_NULL(ackNotify_);
    HCCL_INFO(
        "[OneSideBatchPutMR] notifySize[%u], notifyWaitMode[%d], "
        "ackNotify[%p], remoteNotifyValueMem addr[%p], remoteNotifyValueMem len[%llu], remoteNotifyValueMem key[%u], "
        "wqePerDoorBell[%u]",
        notifySize_, notifyWaitMode_, ackNotify_, remoteNotifyValueMem_.addr, remoteNotifyValueMem_.size,
        remoteNotifyValueMem_.lkey, wqePerDoorBell_);

    u32 sendWrNum = 0;
    for (u32 i = 0; i < num; i++) {
        if (i != num - 1) {
            CHK_RET(PayLoadMR(putMRList + i, remoteMRList + i, sendWrNum));
        } else {
            // 最后一组数据特殊处理，发送立即数
            CHK_RET(PayLoadMR(putMRList + i, remoteMRList + i, sendWrNum, true));
        }
    }

    HcclResult ret = RecordNotify(
        remoteNotifyValueMem_.addr, remoteNotifyValueMem_.lkey, localSyncMemAck_.addr, localSyncMemAck_.lkey,
        notifySize_, RA_WR_RDMA_READ, RA_SEND_SIGNALED | RA_SEND_FENCE);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[OneSideBatchPutMR] Record ack failed"), ret);

    ret = WaitSignal(ackNotify_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[OneSideBatchPutMR] Wait ack failed"), ret);

    return ret;
}

} // namespace hccl
