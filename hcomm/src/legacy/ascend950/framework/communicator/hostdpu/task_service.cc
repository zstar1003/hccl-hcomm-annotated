/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_service.h"
#include <algorithm>
#include "profiling/dlprof_function_v2.h"
#include <thread>
#include <chrono>
#include <atomic>
#include "acl/acl_rt.h"
#include "log.h"
#include "dpu_kernel_entrance.h"
#include "env_config/env_config_v2.h"

namespace Hccl {
constexpr uint32_t CTRL_HDR_FLAG_LENGTH = 1;
constexpr uint32_t TASKTYPE_ADDR_LENGTH = 256;
constexpr uint32_t CTRL_HDR_MSG_ID_LEN = 4;
constexpr uint32_t TIMEOUT_SIZE_BYTE = 4;      // timeout字段长度为4字节，表示超时时间，单位为秒
constexpr uint32_t CTRL_HDR_DATA_SIZE_LEN = 8; // size_t 在不同平台上长度不同，取最大值
constexpr uint32_t CTRL_HDR_DEFAULT_DATA_LEN = 512;

constexpr uint8_t TASK_UNSET = 0;
constexpr uint8_t TASK_OK = 1;
constexpr uint8_t TASK_TERMINATE = 2;
constexpr uint8_t MEMORY_DEVIDE = 2;

TaskService::TaskService(
    void* deviceMem, int32_t deviceMemSize, void* hostMem, int32_t hostMemSize, std::string commId, uint32_t devId)
    : npu2dpuMem_(deviceMem),
      shmemSize_(deviceMemSize / MEMORY_DEVIDE),
      hostMem_(hostMem),
      hostMemSize_(hostMemSize),
      commId_(commId),
      devId_(devId)
{
    int32_t controlSize = sizeof(uint8_t) + sizeof(char) * TASKTYPE_ADDR_LENGTH + sizeof(uint32_t) + TIMEOUT_SIZE_BYTE
                          + CTRL_HDR_DATA_SIZE_LEN;
    if (shmemSize_ < controlSize) {
        leftSize_ = 0;
    } else {
        leftSize_ = shmemSize_ - controlSize;
    }
    dpu2npuMem_ = static_cast<uint8_t*>(npu2dpuMem_) + shmemSize_;
}

HcclResult TaskService::TaskRegister(std::string taskType, CallbackTemplate callback)
{
    HCCL_INFO("[TaskService::%s] taskType[%s]", __func__, taskType.c_str());
    callbacks_.insert({taskType, callback});
    return HCCL_SUCCESS;
}

HcclResult TaskService::TaskUnRegister(std::string taskType)
{
    HCCL_INFO("[TaskService::%s] taskType[%s]", __func__, taskType.c_str());
    if (callbacks_.find(taskType) == callbacks_.end()) {
        HCCL_WARNING("[TaskService::%s] TaskType Not Found", __func__);
        return HCCL_E_NOT_FOUND;
    }
    callbacks_.erase(taskType);
    return HCCL_SUCCESS;
}

HcclResult TaskService::TaskProfRegister(ProfCallbackTemplate profCallback)
{
    if (profCallback_ != nullptr) {
        return HCCL_SUCCESS;
    }
    profCallback_ = profCallback;
    return HCCL_SUCCESS;
}

HcclResult TaskService::TaskReportRegister(ReportCallbackTemplate reportCallback)
{
    if (reportCallback_ != nullptr) {
        return HCCL_SUCCESS;
    }
    reportCallback_ = reportCallback;
    return HCCL_SUCCESS;
}

HcclResult TaskService::WriteFlag(uint8_t* flagPtr, uint8_t newFlag) const
{
    errno_t ret = memcpy_s(flagPtr, sizeof(newFlag), &newFlag, sizeof(newFlag));
    if (ret != EOK) {
        HCCL_ERROR("[TaskService::TaskRun] set flag failed: %d", ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult TaskService::ReadFlag(uint8_t* ctrlHdr, uint64_t hdrLen, uint8_t& flag) const
{
    errno_t ret = memcpy_s(ctrlHdr, hdrLen, npu2dpuMem_, hdrLen);
    if (ret != EOK) {
        HCCL_ERROR("[TaskService::%s] memcpy_s failed on flag, return[%d].", __func__, ret);
        return HCCL_E_INTERNAL;
    }
    flag = *ctrlHdr;
    return HCCL_SUCCESS;
}

HcclResult TaskService::ReadTaskType(
    const uint8_t* ctrlHdr, [[maybe_unused]] uint64_t hdrLen, const uint8_t* srcTaskTypePtr,
    std::string& taskTypeStr) const
{
    CHK_PTR_NULL(srcTaskTypePtr);
    // 读 taskType
    char* taskType = new char[TASKTYPE_ADDR_LENGTH];
    CHK_PTR_NULL(taskType);
    int ret = memcpy_s(
        taskType, (sizeof(char) * TASKTYPE_ADDR_LENGTH), ctrlHdr + CTRL_HDR_FLAG_LENGTH,
        (sizeof(char) * TASKTYPE_ADDR_LENGTH));
    if (ret != EOK) {
        HCCL_ERROR("[%s] memcpy failed on taskType, return[%d].", __func__, ret);
        delete[] taskType;
        return HCCL_E_INTERNAL;
    }
    //  查找 \0
    auto it = std::find(taskType, taskType + TASKTYPE_ADDR_LENGTH, '\0');
    if (it == taskType + TASKTYPE_ADDR_LENGTH) {
        HCCL_ERROR("[TaskService::TaskRun] No Null Character Found Within TaskType Max Length");
        delete[] taskType;
        return HCCL_E_PARA;
    }
    taskTypeStr = std::string(taskType);
    delete[] taskType;
    HCCL_INFO("[TaskService::TaskRun] read taskType = %s", taskTypeStr.c_str());

    uint32_t msgId{0};
    ret = memcpy_s(&msgId, sizeof(msgId), ctrlHdr + CTRL_HDR_FLAG_LENGTH + TASKTYPE_ADDR_LENGTH, sizeof(msgId));
    if (ret != EOK) {
        HCCL_ERROR("[%s] memcpy failed on msgId, return[%d].", __func__, ret);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[TaskService::TaskRun] read msgId = %u.", msgId);
    return HCCL_SUCCESS;
}

// 共享内存排布：|stop flag[1]|hcclret[2]|dstret[2]|
// 其中，stop flag为aicpu侧读取是否停止的标志位, hcclret为aicpu背景线程读取是否有错的标志位,
// dstret为host侧taskexception回调读取是否有错的标志位
HcclResult TaskService::ExecuteTaskexception(int32_t ret)
{
    void* taskexpShmem = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_serMapMutex);
        auto outerIt = g_taskExpMemMap.find(commId_);
        if (outerIt == g_taskExpMemMap.end()) {
            HCCL_ERROR("TaskService::ExecuteTaskexception commId not in g_taskExpMemMap, please check");
            return HCCL_E_NOT_FOUND;
        }
        auto innerIt = outerIt->second.find(devId_);
        if (innerIt == outerIt->second.end()) {
            HCCL_ERROR("TaskService::ExecuteTaskexception devId not in g_taskExpMemMap, please check");
            return HCCL_E_NOT_FOUND;
        }
        taskexpShmem = innerIt->second;
    }
    HcclResult hcclRet = static_cast<HcclResult>(ret);
    if (taskexpShmem != nullptr) {
        uint8_t* stopFlagPtr = static_cast<uint8_t*>(taskexpShmem);
        uint8_t* hcclRetPtr = stopFlagPtr + sizeof(uint8_t);
        uint8_t* dstRetPtr = hcclRetPtr + sizeof(uint16_t);
        auto ret = memcpy_s(hcclRetPtr, sizeof(uint16_t), &hcclRet, sizeof(uint16_t)); // aicpu背景线程轮询的标志位
        if (ret != 0) {
            HCCL_ERROR("[TaskService::ExecuteTaskexception] memcpy ret for device failed.");
            return HCCL_E_MEMORY;
        }
        ret = memcpy_s(dstRetPtr, sizeof(uint16_t), &hcclRet, sizeof(uint16_t)); // host侧taskexception回调执读出错误码
        if (ret != 0) {
            HCCL_ERROR("[TaskService::ExecuteTaskexception] memcpy ret for host failed.");
            return HCCL_E_MEMORY;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TaskService::ExecuteTaskClean() const
{
    errno_t ret = memset_s(hostMem_, hostMemSize_, 0, hostMemSize_);
    if (ret != EOK) {
        HCCL_ERROR("memset hostMem failed: %d", ret);
        return HCCL_E_MEMORY;
    }
    ret = memset_s(npu2dpuMem_, shmemSize_, 0, shmemSize_);
    if (ret != EOK) {
        HCCL_ERROR("memset npu2dpuMem_ failed: %d", ret);
        return HCCL_E_MEMORY;
    }
    ret = memset_s(dpu2npuMem_, shmemSize_, 0, shmemSize_);
    if (ret != EOK) {
        HCCL_ERROR("memset dpu2npuMem_ failed: %d", ret);
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}

HcclResult TaskService::ExecuteTask(uint8_t* ctrlHdr, uint64_t hdrLen, uint8_t* srcPtr, std::string taskTypeStr)
{
    uint64_t beginTime = Hccl::DlProfFunction::GetInstance().dlMsprofSysCycleTime();
    auto itFunc = callbacks_.find(taskTypeStr);
    if (itFunc == callbacks_.end()) {
        HCCL_ERROR("[TaskService::TaskRun] Callback of taskType[%s] Not Found", taskTypeStr.c_str());
        std::string taskTypeMsg{"map{"};
        for (const auto& pair : callbacks_) {
            taskTypeMsg += pair.first;
        }
        taskTypeMsg += "}";
        HCCL_ERROR("[TaskService::TaskRun] Callback key : %s", taskTypeMsg.c_str());
        return HCCL_E_NOT_FOUND;
    }

    // copy data
    uint64_t dataLen = *reinterpret_cast<size_t*>(
        ctrlHdr + CTRL_HDR_FLAG_LENGTH + TASKTYPE_ADDR_LENGTH + CTRL_HDR_MSG_ID_LEN + TIMEOUT_SIZE_BYTE);
    if (dataLen > static_cast<uint64_t>(leftSize_) || dataLen > static_cast<uint64_t>(hostMemSize_)) {
        HCCL_ERROR(
            "[TaskService::%s] dataLen[%llu] larger than leftSize[%d] or hostMemSize[%d]", __func__, dataLen, leftSize_,
            hostMemSize_);
        return HCCL_E_PARA;
    }
    uint32_t ctrlHdrLen = CTRL_HDR_FLAG_LENGTH + TASKTYPE_ADDR_LENGTH + CTRL_HDR_MSG_ID_LEN + TIMEOUT_SIZE_BYTE
                          + CTRL_HDR_DATA_SIZE_LEN;
    /* ctrlHdr提前从deviceMem copy一定长度，如果长度够，直接从ctrlHdr copy，减少一次aclmemcpy耗时 */
    uint8_t* dataPtr = nullptr;
    if (hdrLen < ctrlHdrLen + dataLen) {
        dataPtr = srcPtr + ctrlHdrLen;
    } else {
        dataPtr = ctrlHdr + ctrlHdrLen;
    }
    errno_t ret = memcpy_s(hostMem_, leftSize_, dataPtr, dataLen);
    if (ret != EOK) {
        HCCL_ERROR("control data memcpy failed: %d", ret);
        return HCCL_E_MEMORY;
    }
    auto callbackRet = itFunc->second(reinterpret_cast<uint64_t>(hostMem_), dataLen);
    if (callbackRet != 0) {
        // dpu任务出错，清理DPUTAG共享内存内容
        CHK_RET(ExecuteTaskClean());
        // 写DPUTASKEXCEPTION
        CHK_RET(ExecuteTaskexception(callbackRet));
        return HCCL_E_INTERNAL;
    }
    if (profCallback_ != nullptr) {
        TaskParam taskParam{};
        taskParam.beginTime = beginTime;
        taskParam.taskType = Hccl::TaskParamType::TASK_DPU_KERNEL;
        taskParam.endTime = Hccl::DlProfFunction::GetInstance().dlMsprofSysCycleTime();
        taskParam.isMaster = true;
        profCallback_(taskParam, INVALID_U64);
    }
    return HCCL_SUCCESS;
}

HcclResult TaskService::SynchronizeControlInfo([[maybe_unused]] uint8_t* ctrlHdr, [[maybe_unused]] uint64_t hdrLen)
{
    CHK_PTR_NULL(npu2dpuMem_);
    CHK_PTR_NULL(dpu2npuMem_);
    // npu2dpu -> dpu2npu
    int32_t controlDataSize = sizeof(uint8_t) + sizeof(char) * TASKTYPE_ADDR_LENGTH + sizeof(uint32_t);
    HCCL_INFO("[TaskService::TaskRun] Send response: npu2dpu -> dpu2npu memcpy");
    errno_t ret = memcpy_s(dpu2npuMem_, controlDataSize, npu2dpuMem_, controlDataSize);
    if (ret != EOK) {
        HCCL_ERROR("[TaskService::TaskRun] npu2dpu -> dpu2npu memcpy failed: %d", ret);
        return HCCL_E_INTERNAL;
    }

    std::atomic_thread_fence(std::memory_order_seq_cst);

    uint8_t newFlag = 1;
    HCCL_INFO("[TaskService::TaskRun] Send response: Set dpu2npu flag -> 1");
    static_cast<std::atomic<uint8_t>*>(dpu2npuMem_)->store(newFlag, std::memory_order_release);
    return HCCL_SUCCESS;
}

HcclResult TaskService::ProcessTaskOk(uint8_t* ctrlHdr, uint64_t hdrLen, uint8_t* srcFlagPtr, uint8_t* srcTaskTypePtr)
{
    std::string taskTypeStr;
    HCCL_INFO("[TaskService::TaskRun] flag = %u.", TASK_OK);
    HCCL_INFO("[TaskService::TaskRun] Set npu2dpu flag -> %u.", TASK_UNSET);
    CHK_RET(WriteFlag(srcFlagPtr, TASK_UNSET));
    CHK_RET(ReadTaskType(ctrlHdr, hdrLen, srcTaskTypePtr, taskTypeStr));
    CHK_RET(ExecuteTask(ctrlHdr, hdrLen, srcFlagPtr, taskTypeStr));
    if (reportCallback_ != nullptr) {
        CHK_RET(reportCallback_());
    }
    CHK_RET(SynchronizeControlInfo(ctrlHdr, hdrLen));
    return HCCL_SUCCESS;
}

HcclResult TaskService::ExecuteExit(uint8_t* srcFlagPtr) const
{
    CHK_RET(WriteFlag(srcFlagPtr, TASK_TERMINATE_RESPONSE));
    HCCL_INFO("[TaskService::TaskRun] Exiting.");
    return HCCL_SUCCESS;
}

HcclResult TaskService::TaskRun()
{
    CHK_PTR_NULL(hostMem_);
    CHK_PTR_NULL(npu2dpuMem_);
    CHK_PTR_NULL(dpu2npuMem_);
    HCCL_INFO(
        "[TaskService::%s] TaskService{npu2dpuMem:%p; dpu2npuMem:%p; hostMem:%p}", __func__, npu2dpuMem_, dpu2npuMem_,
        hostMem_);
    if (leftSize_ <= 0) {
        HCCL_ERROR("[TaskService::%s] dataSize[%d] illegal", __func__, leftSize_);
        return HCCL_E_INTERNAL;
    }
    if (leftSize_ > hostMemSize_) {
        HCCL_ERROR("[TaskService::%s] hostMemSize[%d] less than dataSize[%d]", __func__, hostMemSize_, leftSize_);
        return HCCL_E_INTERNAL;
    }
    uint8_t flag{0};
    uint8_t* srcFlagPtr = static_cast<uint8_t*>(npu2dpuMem_);
    uint8_t* dstTimeoutPtr
        = static_cast<uint8_t*>(dpu2npuMem_) + sizeof(flag) + sizeof(char) * TASKTYPE_ADDR_LENGTH + CTRL_HDR_MSG_ID_LEN;
    uint8_t* srcTaskTypePtr = srcFlagPtr + sizeof(flag);
    uint64_t hdrLen = CTRL_HDR_FLAG_LENGTH + TASKTYPE_ADDR_LENGTH + CTRL_HDR_MSG_ID_LEN + TIMEOUT_SIZE_BYTE
                      + CTRL_HDR_DATA_SIZE_LEN + CTRL_HDR_DEFAULT_DATA_LEN;
    uint8_t ctrlHdr[hdrLen];
    u32 timeout = Hccl::EnvConfig::GetInstance().GetRtsConfig().GetExecTimeOut();
    timeout = timeout > 1 ? timeout - 1 : timeout; // 执行超时时间减1秒，避免aicpu侧超时
    CHK_SAFETY_FUNC_RET(memcpy_s(dstTimeoutPtr, sizeof(timeout), &timeout, sizeof(timeout)));

    while (true) {
        CHK_RET(ReadFlag(ctrlHdr, hdrLen, flag));
        switch (flag) {
            case TASK_UNSET:
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            case TASK_OK:
                CHK_RET(ProcessTaskOk(ctrlHdr, hdrLen, srcFlagPtr, srcTaskTypePtr));
                continue;
            case TASK_TERMINATE:
                HCCL_INFO("[TaskService::TaskRun] flag = %u.", flag);
                HCCL_INFO("[TaskService::TaskRun] Set npu2dpu flag -> %u. Task Run END.", TASK_TERMINATE_RESPONSE);
                CHK_RET(ExecuteExit(srcFlagPtr));
                return HCCL_SUCCESS;
            case TASK_TERMINATE_RESPONSE:
                HCCL_INFO("[TaskService::TaskRun] flag = %u. Exiting.", flag);
                return HCCL_SUCCESS;
            default:
                HCCL_INFO("[TaskService::TaskRun] flag = %u. Unimplemented flag. Exiting.", flag);
                return HCCL_SUCCESS;
        }
    }
    return HCCL_SUCCESS;
}
} // namespace Hccl
