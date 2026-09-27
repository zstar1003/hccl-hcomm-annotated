/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TASK_SERVICE_H
#define TASK_SERVICE_H

#include <functional>
#include <unordered_map>
#include <string>
#include <cstdint>
#include "hccl_types.h"
#include "task_param.h"

namespace Hccl {
using CallbackTemplate = std::function<int32_t(uint64_t, int32_t)>;
using ProfCallbackTemplate = std::function<HcclResult(const TaskParam&, uint64_t)>;
using ReportCallbackTemplate = std::function<HcclResult()>;
/**
 * 1. 使用共享 HBM 内存传递任务信息和数据
 * 2. 内存布局(shmemPtr_)分为两块等长区域：
 *    - NPU -> DPU (npu2dpuShmem)
 * +----------------+---------------------------+------------------+--------------------+----------------+
 * | flag (uint8_t) | taskType (256字节定长空间) | msgId (uint32_t) | timeout (uint32_t) |      data      |
 * +----------------+---------------------------+------------------+--------------------+----------------+
 *    - DPU -> NPU (dpu2npuShmem)
 * +----------------+---------------------------+------------------+--------------------+
 * | flag (uint8_t) | taskType (256字节定长空间) | msgId (uint32_t) | timeout (uint32_t) |
 * +----------------+---------------------------+------------------+--------------------+
 * 3. host侧内存装载npu2dpuShmem 的 data部分
 * +----------------+
 * |      data      |
 * +----------------+
 */
class TaskService {
public:
    TaskService() = default;
    TaskService(
        void* deviceMem, int32_t deviceMemSize, void* hostMem, int32_t hostMemSize, std::string commId, uint32_t devId);
    HcclResult TaskRun();
    HcclResult TaskRegister(std::string taskType, CallbackTemplate callback);
    HcclResult TaskUnRegister(std::string taskType);
    HcclResult TaskProfRegister(ProfCallbackTemplate profCallback);
    HcclResult TaskReportRegister(ReportCallbackTemplate reportCallback);

private:
    HcclResult WriteFlag(uint8_t* flagPtr, uint8_t newFlag) const;
    HcclResult ReadFlag(uint8_t* ctrlHdr, uint64_t hdrLen, uint8_t& flag) const;
    HcclResult ReadTaskType(
        const uint8_t* ctrlHdr, uint64_t hdrLen, const uint8_t* srcTaskTypePtr, std::string& taskTypeStr) const;
    HcclResult ExecuteTask(uint8_t* ctrlHdr, uint64_t hdrLen, uint8_t* srcPtr, std::string taskTypeStr);
    HcclResult SynchronizeControlInfo(uint8_t* ctrlHdr, [[maybe_unused]] uint64_t hdrLen);
    HcclResult ProcessTaskOk(uint8_t* ctrlHdr, uint64_t hdrLen, uint8_t* srcFlagPtr, uint8_t* srcTaskTypePtr);
    HcclResult ExecuteTaskClean() const;
    HcclResult ExecuteTaskexception(int32_t ret);
    HcclResult ExecuteExit(uint8_t* srcFlagPtr) const;

private:
    std::unordered_map<std::string, CallbackTemplate> callbacks_;
    ProfCallbackTemplate profCallback_{nullptr};
    ReportCallbackTemplate reportCallback_{nullptr};
    void* npu2dpuMem_{nullptr};
    void* dpu2npuMem_{nullptr};
    int32_t shmemSize_{0};
    int32_t leftSize_{0}; // npu2dpuMem_中除去控制信息后剩余的可用空间大小
    void* hostMem_{nullptr};
    int32_t hostMemSize_{0};
    std::string commId_{};
    uint32_t devId_{UINT32_MAX};
};
} // namespace Hccl

#endif // TASK_SERVICE_H
