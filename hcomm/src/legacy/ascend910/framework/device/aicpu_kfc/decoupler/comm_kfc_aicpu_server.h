/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMM_KFC_AICPU_SERVER_H
#define HCCL_COMM_KFC_AICPU_SERVER_H

#include <unordered_set>
#include <unordered_map>
#include "hccl_msg.h"
#include <hccl/hccl_types.h>
#include "aicpu_hccl_common.h"
#include "common/aicpu_kfc_def.h"

class CommKfcAicpuServer {
public:
    CommKfcAicpuServer(u32 groupIdx) : groupIdx_(groupIdx) {}
    ~CommKfcAicpuServer() = default;
    HcclApi::HcclMsgArea* GetMsgAreaAddr() const { return msgArea_; }
    u32 GetRankNum() const { return rankNum_; }
    HcclResult AddOpContext(const HcclApi::CommKfcContext* ctx);
    HcclResult Orchestrate(const HcclApi::HcclMsg& msg, HcclApi::HcclMsgExt& extMsg, u32 msgPos);
    HcclResult Finalize(u32 msgPos);
    HcclResult IsAllTaskFinished(u32 msgPos, bool& isFinish);
    HcclResult InterGroupSync(const CommKfcAicpuServer& otherServer, HcclHandle handle);
    HcclResult CheckTimeOut(u32 msgPos);
    HcclResult ErrorDfxProcess(HcclResult errorCode);

private:
#ifdef CCL_LLT
    static constexpr u64 KFC_NSEC_PER_SEC = 1000000UL;
#else
    static constexpr u64 KFC_NSEC_PER_SEC = 1000000000UL;
#endif
    HcclResult GetServerInfoForSync(HcclHandle handle, u32& msgPos, u32& repeat) const;
    void KeepAlive() { lastMsgTimestamp_ = GetCurCpuTimestamp(); }
    bool IsTimeout() const { return GetCurCpuTimestamp() - lastMsgTimestamp_ >= timeout_ * KFC_NSEC_PER_SEC; }
    void SetMsgPosByHandle(HcclHandle handle, u32 msgPos) { handleIdToMsgPos_[handle] = msgPos; }
    void SetRepeatByHandle(HcclHandle handle, u32 repeat) { handleIdToRepeat_[handle] = repeat; }

private:
    std::unordered_map<uintptr_t, void*> ctxToOpHandle_{};
    std::unordered_map<HcclHandle, u32> handleIdToMsgPos_{};
    std::unordered_map<HcclHandle, u32> handleIdToRepeat_{};
    HcclApi::HcclMsgArea* msgArea_{nullptr};
    u64 lastMsgTimestamp_{0UL};
    u64 timeout_{30UL};
    u64 turnNumsAddr_{0UL};
    u32 groupIdx_;
    u32 rankNum_{0U};
};

#endif // HCCL_COMM_KFC_AICPU_SERVER_H
