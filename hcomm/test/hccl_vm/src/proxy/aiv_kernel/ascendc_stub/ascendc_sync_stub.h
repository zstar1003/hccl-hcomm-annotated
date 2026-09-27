/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_C_SYNC_STUB_H
#define ASCEND_C_SYNC_STUB_H

#include "ai_core_stub.h"
#include "aiv_task.h"
#include "ascendc_base_stub.h"
#include "global_tensor_stub.h"
#include "sim_log.h"
#include "local_tensor_stub.h"

namespace AscendC {
inline void pipe_barrier(pipe_t pipe)
{
    auto curBlockIdx = GetBlockIdx();
    auto aiv = AivSim::AivKernelExecutor::GetInstance().GetAivCore(curBlockIdx);
    if (aiv == nullptr) {
        HCCL_VM_ERROR("Get AIV core failed, curBlockIdx={:d}", curBlockIdx);
        return;
    }

    switch (pipe) {
        case pipe_t::PIPE_ALL: {
            auto sTask = std::make_shared<AivSim::AivTaskPipeBarrier>(pipe);
            auto mte2Task = std::make_shared<AivSim::AivTaskPipeBarrier>(pipe);
            auto mte3Task = std::make_shared<AivSim::AivTaskPipeBarrier>(pipe);

            sTask->AddBarrierGroup(mte2Task);
            sTask->AddBarrierGroup(mte3Task);

            mte2Task->AddBarrierGroup(sTask);
            mte2Task->AddBarrierGroup(mte3Task);

            mte3Task->AddBarrierGroup(sTask);
            mte3Task->AddBarrierGroup(mte2Task);

            aiv->AppendScalar(sTask);
            aiv->AppendMTE2(mte2Task);
            aiv->AppendMTE3(mte3Task);
            break;
        }

        case pipe_t::PIPE_S: {
            aiv->AppendScalar(std::make_shared<AivSim::AivTaskPipeBarrier>(pipe));
            break;
        }

        case pipe_t::PIPE_MTE2: {
            aiv->AppendMTE2(std::make_shared<AivSim::AivTaskPipeBarrier>(pipe));
            break;
        }

        case pipe_t::PIPE_MTE3: {
            aiv->AppendMTE3(std::make_shared<AivSim::AivTaskPipeBarrier>(pipe));
            break;
        }

        default:
            HCCL_VM_ERROR("PipeBarrier type[{:d}] not support", static_cast<uint32_t>(pipe));
    }
}

inline void send_flag(uint32_t targetRank, uint64_t commInfoOffset, int32_t curTag)
{
    auto curBlockIdx = GetBlockIdx();
    auto aiv = AivSim::AivKernelExecutor::GetInstance().GetAivCore(curBlockIdx);
    if (aiv == nullptr) {
        HCCL_VM_ERROR("Get AIV core failed, curBlockIdx={:d}", curBlockIdx);
        return;
    }
    aiv->AppendScalar(std::make_shared<AivSim::AivTaskSendFlag>(targetRank, commInfoOffset, curTag));
}

inline void recv_flag(uint32_t targetRank, uint64_t commInfoOffset, int32_t curTag)
{
    auto curBlockIdx = GetBlockIdx();
    auto aiv = AivSim::AivKernelExecutor::GetInstance().GetAivCore(curBlockIdx);
    if (aiv == nullptr) {
        HCCL_VM_ERROR("Get AIV core failed, curBlockIdx={:d}", curBlockIdx);
        return;
    }
    aiv->AppendScalar(std::make_shared<AivSim::AivTaskRecvFlag>(targetRank, commInfoOffset, curTag));
}

template <pipe_t pipe>
__aicore__ void PipeBarrier()
{
    pipe_barrier(pipe);
}

pipe_t inline GetEventSrcPipe(HardEvent event)
{
    switch (event) {
        case HardEvent::MTE3_MTE2:
        case HardEvent::MTE3_S:
            return pipe_t::PIPE_MTE3;

        case HardEvent::S_MTE2:
        case HardEvent::S_MTE3:
            return pipe_t::PIPE_S;

        case HardEvent::MTE2_S:
        case HardEvent::MTE2_MTE3:
            return pipe_t::PIPE_MTE2;

        default:
            HCCL_VM_ERROR("event[{:d}] not support", static_cast<uint32_t>(event));
            return pipe_t::PIPE_ALL;
    }
}

pipe_t inline GetEventDstPipe(HardEvent event)
{
    switch (event) {
        case HardEvent::MTE3_S:
        case HardEvent::MTE2_S:
            return pipe_t::PIPE_S;

        case HardEvent::S_MTE2:
        case HardEvent::MTE3_MTE2:
            return pipe_t::PIPE_MTE2;

        case HardEvent::S_MTE3:
        case HardEvent::MTE2_MTE3:
            return pipe_t::PIPE_MTE3;

        default:
            HCCL_VM_ERROR("event[{:d}] not support", static_cast<uint32_t>(event));
            return pipe_t::PIPE_ALL;
    }
}

template <HardEvent event>
__aicore__ void SetFlag(TEventID eventID)
{
    auto curBlockIdx = GetBlockIdx();
    auto aiv = AivSim::AivKernelExecutor::GetInstance().GetAivCore(curBlockIdx);
    if (aiv == nullptr) {
        HCCL_VM_ERROR("Get AIV core failed, curBlockIdx={:d}", curBlockIdx);
        return;
    }

    pipe_t srcPipe = GetEventSrcPipe(event);
    pipe_t dstPipe = GetEventDstPipe(event);
    auto task = std::make_shared<AivSim::AivTaskSetFlag>(srcPipe, dstPipe, eventID);

    switch (srcPipe) {
        case pipe_t::PIPE_S:
            aiv->AppendScalar(task);
            break;
        case pipe_t::PIPE_MTE2:
            aiv->AppendMTE2(task);
            break;
        case pipe_t::PIPE_MTE3:
            aiv->AppendMTE3(task);
            break;
        default:
            HCCL_VM_ERROR("SetFlag pipe type[{:d}] not support", static_cast<uint32_t>(srcPipe));
    }
}

template <HardEvent event>
__aicore__ void WaitFlag(TEventID eventID)
{
    auto curBlockIdx = GetBlockIdx();
    auto aiv = AivSim::AivKernelExecutor::GetInstance().GetAivCore(curBlockIdx);
    if (aiv == nullptr) {
        HCCL_VM_ERROR("Get AIV core failed, curBlockIdx={:d}", curBlockIdx);
        return;
    }

    pipe_t srcPipe = GetEventSrcPipe(event);
    pipe_t dstPipe = GetEventDstPipe(event);
    auto task = std::make_shared<AivSim::AivTaskWaitFlag>(srcPipe, dstPipe, eventID);

    switch (dstPipe) {
        case pipe_t::PIPE_S:
            aiv->AppendScalar(task);
            break;
        case pipe_t::PIPE_MTE2:
            aiv->AppendMTE2(task);
            break;
        case pipe_t::PIPE_MTE3:
            aiv->AppendMTE3(task);
            break;
        default:
            HCCL_VM_ERROR("WaitFlag pipe type[{:d}] not support", static_cast<uint32_t>(dstPipe));
    }
}

template <bool isAIVOnly = true>
__aicore__ void SyncAll()
{
    auto curBlockIdx = GetBlockIdx();
    auto aiv = AivSim::AivKernelExecutor::GetInstance().GetAivCore(curBlockIdx);
    if (aiv == nullptr) {
        HCCL_VM_ERROR("Get AIV core failed, curBlockIdx={:d}", curBlockIdx);
        return;
    }
    auto curSyncRound = aiv->GetSyncRound();
    aiv->AppendScalar(std::make_shared<AivSim::AivTaskSyncAll>(curSyncRound));
    aiv->AppendMTE2(std::make_shared<AivSim::AivTaskSyncAll>(curSyncRound));
    aiv->AppendMTE3(std::make_shared<AivSim::AivTaskSyncAll>(curSyncRound));
}
} // namespace AscendC

#endif // ASCEND_C_SYNC_STUB_H
