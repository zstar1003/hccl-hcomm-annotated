/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KFC_RPC_SERVERV2_H
#define AICPU_KFC_RPC_SERVERV2_H

#include "hccl_tiling_msg.h"
#include "hccl_msg.h"
#include "common/aicpu_hccl_def.h"
#include "common/aicpu_kfc_def.h"
#include "stream_pub.h"

class AicpuKfcRpcServerV2 {
public:
    AicpuKfcRpcServerV2() = default;
    ~AicpuKfcRpcServerV2() = default;
    HcclResult Init(const HcclMC2WorkSpace& workspaceInfo, const HcclApi::Mc2InitTilingInner* tilingData = nullptr);
    void Reset();
    HcclApi::HcclMsgArea* GetHcclMsgArea(void);
    HcclApi::HcclMsg (*GetMsgWorkSpace())[HcclApi::HCCL_MSG_CNT];
    uint64_t GetFinishAddr(int32_t idx) const;
    uint64_t GetCommitareaAddr(int32_t idx) const;
    uint64_t GetFinishAddrByHandleId(HcclHandle handleId);
    bool GetIsFinalize(u32 queueId = HcclApi::MAX_QUE_NUM);
    void SetIsFinalize(u32 queueId, bool finalize);
    HcclResult AddCcoreNotify(HcclDispatcher dispatcherPtr, u64 recordAddr, uint32_t turnNum, hccl::Stream* stream);
    HcclResult
    AddCcoreWait(HcclDispatcher dispatcherPtr, u64 waitAddr, uint32_t turnNum, hccl::Stream* stream, bool isLast);
    HcclResult AddFlipTask(HcclDispatcher dispatcherPtr, hccl::Stream* stream);
    HcclResult ResetCommitTaskAdd(HcclDispatcher dispatcherPtr, hccl::Stream* stream);
    void WriteFinishWhenAllFinalize();
    void WriteRestartFlag();
    uint32_t GetMsgPos(u32 queueId = 0U) const { return msgPos_[queueId]; }
    void SetMsgPos(u32 queueId, u32 pos) { msgPos_[queueId] = pos; }
    bool IsPrintLog() const;
    void SetMsgRepeatCnt(u8 repeatCnt);
    int32_t GetMsgRepeatCnt(HcclHandle handleId);
    int32_t GetMsgHandlePos(HcclHandle handleId);
    void PrintAllHcclMsgArea(u32 rankSize);
    void PrintAllHcclMsgAreaData();
    void PrintMsg(HcclApi::HcclMsg* hcclMsg, uint32_t msgPos, u32 rankSize);
    void SetMsgPosForKernel(uint32_t msgPos) { msgPosForKernel_ = msgPos; }
    uint32_t GetMsgPosForKernel(void) const { return msgPosForKernel_; }
    void SetMsgHandlePos(uint32_t msgPos, HcclHandle handleId);
    void SetNeedRetryFlag(bool needRetryFlag);
    bool ReadAddrMsg(HcclApi::HcclMsg* hcclMsg, HcclApi::HcclMsg* msgList, u32 queueIdx, u32 msgPos, u32 rankSize);
    bool IsExceedLimit(HcclCMDType commType, u32 rankSize);
    HcclApi::HcclMsgExt* GetHcclMsgExtPtr();
    HcclResult ProcessExpectPrepareMsg(uint8_t seqNum, uint8_t expectId);

public:
    void SetStepSize(u8 stepSize) { curStepSize_ = stepSize; };
    void SetTotalStep(u16 totalStep) { totalStep_ = totalStep; };
    u16 GetStepSize() const { return curStepSize_; }
    u64 GetTurnNumAddr() const { return turnNumAddr_; }
    u32 GetTotalQueueNum() const { return totalQueueNum_; }
    BarrierInfo* GetBarrierInfoByGroupIdx(u32 idx) { return barrierFlags_[idx]; }
    void ClearBarrierStatus(u32 groupIdx, u32 start, u32 cnt)
    {
        (void)memset_s(&(barrierFlags_[groupIdx][start]), cnt * sizeof(BarrierInfo), 0, cnt * sizeof(BarrierInfo));
    }
    u32* GetBarrierFinishCnts() { return barrierFinishCnt_; }
    u64 GetTilingBaseAddr() const { return tilingBaseAddr_; }
    void GetLocalQueueRange(u32& start, u32& end);
    void DumpBarrierInfo(u32 groupIdx, u32 sqId, u32 devId);

private:
    bool ReadValidMsg(HcclApi::HcclMsg* rMsg, HcclApi::HcclMsg* msg, bool needReProcess, uint32_t msgPos, u32 rankSize);
    bool CheckDebugMode(HcclApi::HcclMsg* rMsg);
    bool ReadValidMsgExtArea(int32_t idx, u32 rankSize);

private:
    uint64_t workSpace_ = 0;
    HcclApi::HcclMsgArea* hcclMsgArea_ = nullptr;
    uint32_t repeatCnt_[HcclApi::HCCL_MSG_CNT];
    int8_t handleIdMsgPosition_[HcclApi::HCCL_MAX_HANDLE_ID];
    uint64_t streamId_;
    uint32_t msgPosForKernel_;
    uint32_t msgPos_[HcclApi::MAX_QUE_NUM];
    bool needReProcess_ = false;
    bool isFinalize_[HcclApi::MAX_QUE_NUM];
    std::shared_ptr<HcclApi::HcclMsgExt> msgExt_ = std::make_shared<HcclApi::HcclMsgExt>();
    u64 prepareTime_[HcclApi::MAX_QUE_NUM]; // 记录 Prepare 消息的时间
    u8 eventPrintTurn_;                     // 记录打印event的turn
    bool isPrintLog_ = false;
    u8 curStepSize_ = 0U;
    u16 totalStep_ = 0U;
    u64 turnNumAddr_;
    u32 blockNum_ = 1U;
    u32 totalQueueNum_ = 0U;
    BarrierInfo barrierFlags_[MAX_COMM_CTX_NUM][HcclApi::MAX_QUE_NUM];
    u32 barrierFinishCnt_[MAX_AICPU_NUM_BLOCKS];
    u64 tilingBaseAddr_;
};

#endif // __AICPU_RPC_SERVERV2_H__
