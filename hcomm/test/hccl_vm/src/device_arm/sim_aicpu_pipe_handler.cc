/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sim_aicpu_pipe_handler.h"
#include "sim_aicpu_pipe_msg.h"
#include <unistd.h>
#include <string>
#include <dlfcn.h>
#include "sim_log.h"
#include "sim_common_api.h"
#include "db_sim_op_db_ops.h"
#include "hccl_device_pub.h"
#include "store_sim_memory_manager.h"
#include "sim_kernel_lib_mgr.h"
#include "aicpu_args_stub.h"

namespace sim {

extern uint64_t g_currOpDetailId;

int HandlePipeCmdSetDevId(uint8_t* payload, uint16_t payloadLen)
{
    SetDevIdPayload* req = reinterpret_cast<SetDevIdPayload*>(payload);
    uint32_t curRankId = static_cast<uint32_t>(req->rankId);
    uint64_t curDeviceKey = req->deviceKey;
    SetCurRankId(curRankId);
    SetCurDeviceKey(curDeviceKey);
    HCCL_VM_INFO(
        "Process[{}] PIPE_RSP_SET_DEV_ID, set rankId = [{}], deviceKey = [{}]", getpid(), curRankId, curDeviceKey);
    uint64_t donePayload = 0;
    DeviceSendMsg(PIPE_RSP_SET_DEV_ID, &donePayload, sizeof(donePayload));
    return 0;
}

int HandlePipeCmdGetDevPtr(uint8_t* payload, uint16_t payloadLen)
{
    DevMemOpPayload* req = reinterpret_cast<DevMemOpPayload*>(payload);
    RspGetDevPtrPayload donePayload{};
    void* shmptr = sim::MemoryManager::GetInstance().AcquireMemByName(req->memName);
    if (shmptr == nullptr) {
        donePayload.ptr = 0;
        HCCL_VM_ERROR("acquire {} shm failed.", req->memName);
    } else {
        donePayload.ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(shmptr));
    }
    HCCL_VM_INFO("[device] acquire {} shm ptr:{:p}", req->memName, shmptr);
    DeviceSendMsg(PIPE_RSP_GET_DEV_PTR, &donePayload, sizeof(donePayload));
    return 0;
}

static void ExecuteAicpuKernel(uint32_t rankId, ExecKernelPayload* kernelReq)
{
    uint64_t args = kernelReq->args;
    std::string kernelSo = kernelReq->soName;
    std::string kernelName = kernelReq->kernelName;

    HCCL_VM_INFO("rankId[{}] kernel[{}] start run...", rankId, kernelName);
    std::string libDir = InstallPath::ResolveToInstallRoot("lib/" + GetArchStr()) + "/" + kernelSo;
    KernelFn fn = sim::KernelLibManager::GetInstance().GetOrLoadFunc(libDir, kernelName);
    if (!fn) {
        HCCL_VM_ERROR("failed to resolve function");
        return;
    }

    void* ptr = reinterpret_cast<void*>(args);
    if (ptr == nullptr) {
        HCCL_VM_ERROR("[device] rankId[{}] init func handle failed null ptr.", rankId);
        return;
    }

    // CCU退化AICPU场景此处使用的内存先于device进程启动前分配，使用前需转换
    if (kernelName == "RunAicpuCommInit") {
        CommAicpuParam* param = reinterpret_cast<CommAicpuParam*>(ptr);
        param->kfcControlTransferH2DParams.deviceAddr
            = GetDevMapperAddrByDevAddr(param->kfcControlTransferH2DParams.deviceAddr);
        param->kfcControlTransferH2DParams.readCacheAddr
            = GetDevMapperAddrByDevAddr(param->kfcControlTransferH2DParams.readCacheAddr);
        param->kfcStatusTransferD2HParams.deviceAddr
            = GetDevMapperAddrByDevAddr(param->kfcStatusTransferD2HParams.deviceAddr);
    }

    fn(ptr);

    HCCL_VM_INFO("rankId[{}] kernel[{}] finish run...", rankId, kernelName);
}

int HandlePipeCmdExecKernel(uint8_t* payload, uint16_t payloadLen)
{
    if (payloadLen < sizeof(ExecKernelPayload)) {
        HCCL_VM_ERROR("[device] EXEC_KERNEL payload too small: {} < {}", payloadLen, sizeof(ExecKernelPayload));
        return -1;
    }

    ExecKernelPayload* kernelReq = reinterpret_cast<ExecKernelPayload*>(payload);
    uint32_t rankId = 0;
    GetCurRankId(&rankId);
    std::string kernelName(kernelReq->kernelName);
    HCCL_VM_INFO("[device] rankId[{}] executing kernel:{}", rankId, kernelName);

    // device进程查询并使用host进程记录的opDetailId
    uint32_t opDetailId = 0;
    int ret = sim::QueryNewestOpDeatailIdByPid(getppid(), opDetailId);
    if (ret != 0) {
        HCCL_VM_ERROR("[device] QueryNewestOpDeatailIdByPid failed.");
    }

    sim::g_currOpDetailId = opDetailId;

    ExecuteAicpuKernel(rankId, kernelReq);
    RspExecKernelPayload donePayload{};
    donePayload.status = 0;
    DeviceSendMsg(PIPE_RSP_EXEC_KERNEL, &donePayload, sizeof(donePayload));
    return 0;
}

int HandlePipeCmdFreeDevPtr(uint8_t* payload, uint16_t payloadLen)
{
    DevMemOpPayload* req = reinterpret_cast<DevMemOpPayload*>(payload);
    sim::MemoryManager::GetInstance().ReleaseMemByName(req->memName);
    HCCL_VM_INFO("[device] release memName:{} shm", req->memName);
    RspFreeDevPtrPayload donePayload{};
    donePayload.status = 0;
    DeviceSendMsg(PIPE_RSP_FREE_DEV_PTR, &donePayload, sizeof(donePayload));
    return 0;
}

} // namespace sim
