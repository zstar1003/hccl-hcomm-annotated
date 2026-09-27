/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_DISPATCH_H__
#define __AICPU_DISPATCH_H__

#include "common/aicpu_hccl_def.h"
#include "common/aicpu_kfc_def.h"

class AicpuDispatcher {
public:
    static HcclResult SignalWait(u16 streamId, u16 notifyId, bool innerChip, bool preNotify);
    static HcclResult SignalRecord(u16 streamId, u16 notifyId, bool innerChip, bool preNotify);
    static HcclResult AicpuUnfoldSignalWait(u16 streamId, u16 notifyId, bool innerChip);
    static HcclResult AicpuUnfoldSignalRecord(u16 streamId, u16 notifyId, bool innerChip);
    static HcclResult SignalWaitWithNotify(u16 streamId, u16 notifyId, bool innerChip, AicpuComSignalInfo* notifyInfo);
    static HcclResult
    SignalRecordWithNotify(u16 streamId, u16 notifyId, bool innerChip, AicpuComSignalInfo* notifyInfo);
    static HcclResult
    CopyData(u16 streamId, void* src, void* dst, u32 len, HcclDataType dataType, HcclReduceOp reduceOp, u32 remoteRank);
    static HcclResult CopyData(
        uint16_t streamId, u64 src, u64 dst, uint32_t len, HcclDataType dataType, HcclReduceOp reduceOp,
        u32 remoteRank);
    static HcclResult LaunchTask(uint32_t streamId);
    static HcclResult AddCcoreWait(uint16_t streamId, u64 waitAddr, uint32_t turnNum, bool isLast);
    static HcclResult AddWaitStartTaskOnMainStream(u16 streamId);
    static HcclResult AddCcoreNotify(uint16_t streamId, uint32_t turnNum);
    static HcclResult AddExecEndTaskOnMainStream(u16 streamId);
    static HcclResult AddAllEndTaskOnMainStream(u16 streamId);
    static HcclResult RdmaSend(uint16_t streamId, u64 dbInfo, u64 dbAddr, u32 userRank);
    static const bool NO_IPC = true;
    static const bool IPC = false;
    static const bool PRE_SYNC = true;
    static const bool POST_SYNC = false;
};
#endif
