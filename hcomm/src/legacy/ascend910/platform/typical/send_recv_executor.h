/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SEND_RECV_EXECUTOR_H
#define SEND_RECV_EXECUTOR_H
#include "dtype_common.h"
#include "hccl_common.h"
#include "adapter_hccp.h"
#include "stream_pub.h"
#include "dispatcher.h"
#include "typical_qp_manager.h"
#include "typical_window_mem.h"
#include "typical_sync_mem.h"
#include "interface_hccl.h"

namespace hccl {
constexpr u64 HCCL_CHUNK_NUM = 1024 * 1024 * 1024;
class SendRecvExecutor {
public:
    // Send/Recv场景
    SendRecvExecutor(
        HcclRtStream stream, QpHandle qpHandle, const struct MrInfoT& localWindowMem,
        const struct MrInfoT& remoteWindowMem, const struct MrInfoT& localSyncMemPrepare,
        const struct MrInfoT& localSyncMemDone, const struct MrInfoT& localSyncMemAck,
        const struct MrInfoT& remoteSyncMemPrepare, const struct MrInfoT& remoteSyncMemDone,
        const struct MrInfoT& remoteSyncMemAck, u32 immData, const u64 chunkNum = HCCL_CHUNK_NUM);
    // BatchPut场景
    SendRecvExecutor(HcclRtStream stream, QpHandle qpHandle, AscendSendRecvLinkInfo* linkInfo);
    SendRecvExecutor(
        HcclRtStream stream, QpHandle qpHandle, AscendMrInfo* localSyncMemDone, AscendMrInfo* remoteSyncMemAck);
    SendRecvExecutor(HcclRtStream stream, QpHandle qpHandle, AscendSendLinkInfo* linkInfo);
    ~SendRecvExecutor();
    HcclResult Init();
    HcclResult OneSideBatchPutMR(u32 num, AscendMrInfo* putMRList, AscendMrInfo* remoteMRList);
    HcclResult WaitPutInit();
    HcclResult Send(void* inputPtr, u64 count, HcclDataType dataType);
    HcclResult Receive(void* outputPtr, u64 count, HcclDataType dataType);
    HcclResult BatchPutMR(u32 num, AscendMrInfo* putMRList, AscendMrInfo* remoteMRList);
    HcclResult Put(void* inputPtr, u64 count, HcclDataType dataType);
    HcclResult WaitPutMR();
    HcclResult WaitPutMROnlyWait();
    HcclResult WaitPutMROnlyRecord();

private:
    HcclResult PollCq();
    HcclResult IsOverlappedWithWinMem(void* inputPtr, u64 userMemSize, bool& isOverlapped);
    HcclResult SendRun(DeviceMem& sendBuffer);
    HcclResult PutRun(DeviceMem& putBuffer);
    HcclResult ReceiveRun(DeviceMem& receiveBuffer);
    HcclResult ReceiveRunByPollCq(DeviceMem& receiveBuffer);
    HcclResult RecordNotify(
        void* dstMemPtr, u32 rkey, const void* srcMemPtr, u32 lkey, u64 srcMemSize, uint32_t rdmaOp = RA_WR_RDMA_WRITE,
        int sendFlag = RA_SEND_SIGNALED);
    HcclResult WaitSignal(HcclRtSignal signal);
    HcclResult PayLoad(const void* src, u64 dstOffset, u64 len);
    HcclResult RdmaSendAsync(struct SendWrV2& wr);
    HcclResult MemcpyAsyncD2D(hccl::DeviceMem& dst, const hccl::DeviceMem& src, hccl::Stream& stream);
    HcclResult PayLoadMR(AscendMrInfo* putMRInfo, AscendMrInfo* remoteMRInfo, u32& wrNum, bool isLastMRtoPut = false);
    HcclResult ProcessRCQ(AscendMrInfo* lastMRInfo);
    HcclResult MultiWqeOneDoorBellSend(bool isLastWr, u32& wrNum, struct SendWrV2& wr);
    HcclResult WaitSignalUnlimitedTime(HcclRtSignal signal);

private:
    HcclRtStream stream_;
    QpHandle qpHandle_;
    struct MrInfoT localWindowMem_ {};
    struct MrInfoT remoteWindowMem_ {};

    struct MrInfoT localSyncMemPrepare_ {};
    struct MrInfoT localSyncMemDone_ {};
    struct MrInfoT localSyncMemAck_ {};

    struct MrInfoT remoteSyncMemPrepare_ {};
    struct MrInfoT remoteSyncMemDone_ {};
    struct MrInfoT remoteSyncMemAck_ {};
    struct MrInfoT notifySrcMem_ {};

    HcclRtSignal prepareNotify_ = nullptr;
    HcclRtSignal ackNotify_ = nullptr;
    HcclRtSignal doneNotify_ = nullptr;
    u32 notifySize_{0};

    u32 immData_ = 0;
    u64 chunkSize_{0};
    SyncMode notifyWaitMode_{SyncMode::DEFAULT_TIMEWAITSYNCMODE};
    u32 wqePerDoorBell_ = 10;

    struct MrInfoT remoteNotifyValueMem_ {};
};
} // namespace hccl

#endif
