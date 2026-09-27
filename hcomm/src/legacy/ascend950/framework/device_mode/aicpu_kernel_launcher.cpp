/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_kernel_launcher.h"
#include <memory>
#include "internal_exception.h"
#include "coll_service_device_mode.h"
#include "aicpu_ins_preprocessor.h"
#include "coll_service_ai_cpu_impl.h"
#include "exception_util.h"

namespace Hccl {

template <class T, class U>
u16 CalcFieldOffset(T* target, U* base)
{
    return static_cast<u16>(reinterpret_cast<const char*>(target) - reinterpret_cast<const char*>(base));
}

void AicpuKernelLauncher::AicpuKernelLaunch(const Stream& stream, const string& algName) const
{
    HCCL_INFO("[AicpuKernelLauncher::%s] start.", __func__);

    auto op = comm->GetCurrentCollOperator();
    HcclKernelLaunchParam param;

    s32 ret = strcpy_s(param.kernel.algName, sizeof(param.kernel.algName), algName.data());
    if (ret != EOK) {
        THROW<InternalException>(StringFormat("AicpuKernelLaunch, strcpy_s algName failed!"));
    }

    ret = strcpy_s(param.kernel.opTag, sizeof(param.kernel.opTag), op->opTag.data());
    if (ret != EOK) {
        THROW<InternalException>(StringFormat("AicpuKernelLaunch, strcpy_s opTag failed!"));
    }

    HCCL_INFO(
        "AicpuKernelLauncher::AicpuKernelLaunch param.kernel.algName: %s, opTag %s", param.kernel.algName,
        op->opTag.c_str());

    auto aicpuInsPreprocessor = dynamic_cast<CollServiceDeviceMode*>(comm->GetCollService())->GetAicpuInsPreprocessor();
    DevBuffer* mem = aicpuInsPreprocessor->GetAicpuResBuffer(algName);
    param.kernel.binaryResAddr = mem->GetAddr();
    param.kernel.binaryResSize = mem->GetSize();
    aicpuInsPreprocessor->SetAicpuResExisted(algName);

    SetHcclKernelLaunchParam(param);

    AddPostToUserStream(stream);
    const auto funcHandle = comm->GetAicpuKernelFuncHandle(param.kernelName);

    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = comm->GetNotifyTimeoutCfg().GetNotifyTimeout();
    aclrtLaunchKernelCfg cfg;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    constexpr u32 numBlocks = 1;
    if (op->opMode == OpMode::OPBASE) {
        HrtAicpuLaunchKernelWithHostArgs(
            funcHandle, numBlocks, comm->GetAicpuStreamManager().GetFreeStream()->GetPtr(), &cfg, &param.kernel,
            sizeof(HcclKernelParamLite));
        HCCL_INFO(
            "[AicpuKernelLauncher][AicpuKernelLaunch] param.kernel.algName: %s OPBASE mode "
            "HrtAicpuLaunchKernelWithHostArgs end!",
            param.kernel.algName);
    } else if (op->opMode == OpMode::OFFLOAD) {
        HrtAicpuLaunchKernelWithHostArgs(
            funcHandle, numBlocks, stream.GetPtr(), &cfg, &param.kernel, sizeof(HcclKernelParamLite));
        HCCL_INFO(
            "[AicpuKernelLauncher][AicpuKernelLaunch] param.kernel.algName: %s OFFLOAD mode "
            "HrtAicpuLaunchKernelWithHostArgs end!",
            param.kernel.algName);
    }
    AddWaitToUserStream(stream);
    HCCL_INFO("[AicpuKernelLauncher::%s] end.", __func__);
}

void AicpuKernelLauncher::SetOpbaseBufferParam(HcclKernelLaunchParam& param, CollOperator& op) const
{
    HCCL_INFO("[AicpuKernelLauncher::%s] start.", __func__);

    auto buffer = comm->GetCclBuffer();
    param.kernel.comm.opBaseScratch.addr = buffer->GetAddr();
    param.kernel.comm.opBaseScratch.size = buffer->GetSize();
    InitAicpuLocBufLite(param.kernel.comm.opBaseScratch, buffer->GetAddr(), buffer->GetSize(), "opBaseScratch");
    if (op.inputMem != nullptr) {
        InitAicpuLocBufLite(param.kernel.op.input, op.inputMem->GetAddr(), op.inputMem->GetSize(), "inputMem");
    }

    if (op.outputMem != nullptr) {
        InitAicpuLocBufLite(param.kernel.op.output, op.outputMem->GetAddr(), op.outputMem->GetSize(), "outputMem");
    }
    HCCL_INFO(
        "[AicpuKernelLauncher::%s] end, SetOpbaseBufferParam param.kernel.comm.opBaseScratch.addr %llu, "
        "param.kernel.comm.opBaseScratch.size %llu",
        __func__, param.kernel.comm.opBaseScratch.addr, param.kernel.comm.opBaseScratch.size);
}

void AicpuKernelLauncher::SetOffloadBufferParam(HcclKernelLaunchParam& param, CollOperator& op) const
{
    HCCL_INFO("[AicpuKernelLauncher::%s] start.", __func__);

    auto offloadInput = comm->GetDataBufferManager().Get(op.opTag, BufferType::INPUT);
    if (offloadInput != nullptr) {
        InitAicpuLocBufLite(param.kernel.op.input, op.inputMem->GetAddr(), op.inputMem->GetSize(), "inputMem");
    }

    auto offloadOuput = comm->GetDataBufferManager().Get(op.opTag, BufferType::OUTPUT);
    if (offloadOuput != nullptr) {
        InitAicpuLocBufLite(param.kernel.op.output, op.outputMem->GetAddr(), op.outputMem->GetSize(), "outputMem");
    }

    auto offloadScartch = comm->GetDataBufferManager().Get(op.opTag, BufferType::SCRATCH);
    if (offloadScartch != nullptr) {
        InitAicpuLocBufLite(param.kernel.op.scratch, op.scratchMem->GetAddr(), op.scratchMem->GetSize(), "scratchMem");
    }

    HCCL_INFO("[AicpuKernelLauncher::%s] end.", __func__);
}

void AicpuKernelLauncher::SetHcclKernelLaunchParam(HcclKernelLaunchParam& param) const
{
    HCCL_INFO("[AicpuKernelLauncher::%s] start.", __func__);

    CollOperator op = *comm->GetCurrentCollOperator();

    param.kernel.comm.idIndex = comm->GetIdIndex();
    param.kernel.comm.myRank = comm->GetMyRank();
    param.kernel.comm.rankSize = comm->GetRankSize();
    param.kernel.comm.devType = comm->GetDevType();
    param.kernel.comm.devPhyId = comm->GetDevicePhyId();
    auto collService = comm->GetCollService();
    param.kernel.comm.opCounterAddr = static_cast<u64>(collService->GetOpCounterBuf()->GetAddr());

    if (op.opMode == OpMode::OPBASE) {
        SetOpbaseBufferParam(param, op);
    } else {
        SetOffloadBufferParam(param, op);
    }

    param.kernel.op.algOperator.opMode = op.opMode;
    param.kernel.op.algOperator.opType = op.opType;
    param.kernel.op.algOperator.reduceOp = op.reduceOp;
    param.kernel.op.algOperator.dataType = op.dataType;
    param.kernel.op.algOperator.dataCount = op.dataCount;
    param.kernel.op.algOperator.root = op.root;
    if (op.opType == OpType::ALLTOALL) {
        param.kernel.op.algOperator.all2AllDataDes = op.all2AllDataDes;
    } else if (op.opType == OpType::ALLTOALLV) {
        auto aicpuInsPreprocessor
            = dynamic_cast<CollServiceDeviceMode*>(comm->GetCollService())->GetAicpuInsPreprocessor();
        aicpuInsPreprocessor->SetAicpuKernelLaunchParam(param);
    }

    param.kernel.op.sendRecvRemoteRank = op.sendRecvRemoteRank;

    HCCL_INFO("[AicpuKernelLauncher::%s] end.", __func__);
}

// 待解决：当前方案未确定此处的取值
constexpr u32 HOST_DEVICE_SYNC_TIMEOUT = 1000;

void AicpuKernelLauncher::AddPostToUserStream(const Stream& stream) const
{
    auto postNotify = comm->GetHostDeviceSyncNotifyManager().GetDeviceWaitNotify();

    postNotify->Post(stream);
}

void AicpuKernelLauncher::AddWaitToUserStream(const Stream& stream) const
{
    auto waitNotify = comm->GetHostDeviceSyncNotifyManager().GetHostWaitNotify();

    waitNotify->Wait(stream, HOST_DEVICE_SYNC_TIMEOUT);
}

} // namespace Hccl
