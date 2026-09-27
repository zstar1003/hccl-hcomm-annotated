/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "aicpu_ts_sync_data_c_adpt.h"

#include <chrono>
#include <thread>
#include <atomic>
#include <cmath>

#include "log.h"
#include "thread.h"
#include "aicpu_ts_thread.h"
#include "externalinput_pub.h"

constexpr size_t MSG_TAG_SIZE_BYTE = 256;
constexpr uint32_t TIMEOUT_SIZE_BYTE = 4;      // timeout字段长度为4字节，表示超时时间，单位为秒
constexpr uint32_t CTRL_HDR_DATA_SIZE_LEN = 8; // size_t 在不同平台上长度不同，取最大值

// 同步等待超时(秒)，复用 HCCL_EXEC_TIMEOUT 环境变量配置的算子级执行超时，向上取整为秒
static uint32_t GetSyncWaitTimeoutSeconds()
{
    return static_cast<uint32_t>(std::ceil(GetExternalInputHcclExecTimeOut()));
}

// Msg 数据格式如下（单位：字节）：
// +----------+--------------+-----------+-------------+---------------+-----------------+
// | flag [1] | msgTag [256] | msgId [4] | timeout [4] | data size [8] | data [sizeByte] |
// +----------+--------------+-----------+-------------+---------------+-----------------+
// ^
// handle

static HcclResult WaitFlagReady(uint8_t* srcFlagPtr)
{
    HCCL_INFO("[%s] Polling flag START.", __func__);
    const auto timeStart = std::chrono::steady_clock::now();
    const uint32_t timeoutVal = GetSyncWaitTimeoutSeconds();
    HCCL_INFO("[%s] Using timeout = %u seconds.", __func__, timeoutVal);
    auto timeoutSec = std::chrono::seconds(timeoutVal);
    uint8_t flagReadValue{0};
    errno_t ret = EOK;
    while (true) {
        ret = memcpy_s(&flagReadValue, sizeof(flagReadValue), srcFlagPtr, sizeof(flagReadValue));
        CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s][memcpy_s] Polling flag ERROR[%d].", __func__, ret), HCCL_E_INTERNAL);
        if (flagReadValue == 1) {
            break;
        }
        const auto elapsed
            = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - timeStart);
        if (timeoutVal != 0 && elapsed > timeoutSec) {
            HCCL_ERROR("[%s] Polling flag TIMEOUT, timeout[%u]s.", __func__, timeoutVal);
            return HCCL_E_TIMEOUT;
        }
    }
    HCCL_INFO("[%s] Polling flag SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

int32_t HcommSendRequest(MsgHandle handle, const char* msgTag, const void* src, size_t sizeByte, uint32_t* msgId)
{
    uint8_t* const dstOnDevShmem = ReinterpretAs<uint8_t*>(handle);
    CHK_PTR_NULL(dstOnDevShmem);
    CHK_PTR_NULL(msgTag);
    CHK_PTR_NULL(src);

    HCCL_INFO(
        "[%s] START. msgHandle[0x%llx], msgTag[%s], src[0x%llx], sizeByte[%zu].", __func__, handle, msgTag, src,
        sizeByte);

    static std::atomic<uint32_t> s_msgId{0};
    *msgId = s_msgId.fetch_add(1, std::memory_order_relaxed); // msgId 达 UINT32_MAX 后回绕到 0，仅保证唯一性

    const uint8_t flagWriteValue{1};
    uint8_t* const dstFlagPtr = dstOnDevShmem;
    uint8_t* const dstMsgTagPtr = dstFlagPtr + sizeof(flagWriteValue);
    uint8_t* const dstMsgIdPtr = dstMsgTagPtr + MSG_TAG_SIZE_BYTE;
    uint8_t* const dstDataPtr = dstMsgIdPtr + sizeof(*msgId) + TIMEOUT_SIZE_BYTE;
    errno_t ret = EOK;

    HCCL_INFO("[%s] Writing %zu bytes data from src to shared mem START.", __func__, sizeByte);
    ret = memcpy_s(dstDataPtr, sizeof(sizeByte), &sizeByte, sizeof(sizeByte));
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s][memcpy_s] Writing data ERROR[%d].", __func__, ret), HCCL_E_INTERNAL);
    ret = memcpy_s(dstDataPtr + CTRL_HDR_DATA_SIZE_LEN, sizeByte, src, sizeByte);
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s][memcpy_s] Writing data ERROR[%d].", __func__, ret), HCCL_E_INTERNAL);
    HCCL_INFO("[%s] Writing %zu bytes data from src to shared mem SUCCESS.", __func__, sizeByte);

    HCCL_INFO("[%s] Writing %zu bytes msgId to shared mem START. msgId = %u.", __func__, sizeof(*msgId), *msgId);
    ret = memcpy_s(dstMsgIdPtr, sizeof(*msgId), msgId, sizeof(*msgId));
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s][memcpy_s] Writing msgId ERROR[%d].", __func__, ret), HCCL_E_INTERNAL);
    HCCL_INFO("[%s] Writing %zu bytes msgId to shared mem SUCCESS. msgId = %u.", __func__, sizeof(*msgId), *msgId);

    HCCL_INFO("[%s] Writing %zu bytes msgTag to shared mem START.", __func__, MSG_TAG_SIZE_BYTE);
    ret = memcpy_s(dstMsgTagPtr, MSG_TAG_SIZE_BYTE, msgTag, MSG_TAG_SIZE_BYTE);
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s][memcpy_s] Writing msgTag ERROR[%d].", __func__, ret), HCCL_E_INTERNAL);
    HCCL_INFO("[%s] Writing %zu bytes msgTag to shared mem SUCCESS.", __func__, MSG_TAG_SIZE_BYTE);

#if defined(__aarch64__) || defined(__arm__)
    asm volatile("dmb sy" ::: "memory"); // 确保之前的内存写入对其他线程可见
#else
    asm volatile("" ::: "memory"); // 非 ARM 架构(x86)仅用编译屏障防止重排
#endif

    HCCL_INFO("[%s] Setting flag = 1 on shared mem START.", __func__);
    ret = memcpy_s(dstFlagPtr, sizeof(flagWriteValue), &flagWriteValue, sizeof(flagWriteValue));
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s][memcpy_s] Setting flag ERROR[%d].", __func__, ret), HCCL_E_INTERNAL);
    HCCL_INFO("[%s] Setting flag = 1 on shared mem SUCCESS.", __func__);

    HCCL_INFO("[%s] SUCCESS. msgId[%u].", __func__, *msgId);
    return HCCL_SUCCESS;
}

int32_t HcommWaitResponse(MsgHandle handle, void* dst, size_t sizeByte, uint32_t* msgId)
{
    uint8_t* const srcOnDevShmem = ReinterpretAs<uint8_t*>(handle);
    CHK_PTR_NULL(srcOnDevShmem);
    if (sizeByte > 0) {
        CHK_PTR_NULL(dst);
    }
    CHK_PTR_NULL(msgId);

    HCCL_INFO("[%s] START. msgHandle[0x%llx], dst[0x%llx], sizeByte[%zu].", __func__, handle, dst, sizeByte);

    constexpr size_t sizeByteMsgId = sizeof(uint32_t);
    uint8_t flagReadValue{0};
    uint8_t* const srcFlagPtr = srcOnDevShmem;
    uint8_t* const srcMsgIdPtr = srcFlagPtr + sizeof(flagReadValue) + MSG_TAG_SIZE_BYTE;
    uint8_t* const srcTimeoutPtr = srcMsgIdPtr + sizeByteMsgId;
    uint8_t* const srcDataPtr = srcTimeoutPtr + TIMEOUT_SIZE_BYTE;
    errno_t ret = EOK;

    CHK_RET(WaitFlagReady(srcFlagPtr));

    if (sizeByte > 0) {
        HCCL_INFO("[%s] Reading %zu bytes data from shared mem START.", __func__, sizeByte);
        ret = memcpy_s(dst, sizeByte, srcDataPtr, sizeByte);
        CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s][memcpy_s] Reading data ERROR[%d]", __func__, ret), HCCL_E_INTERNAL);
        HCCL_INFO("[%s] Reading %zu bytes data from shared mem SUCCESS.", __func__, sizeByte);
    }

    HCCL_INFO("[%s] Reading %zu bytes msgId from shared mem START.", __func__, sizeByteMsgId);
    ret = memcpy_s(msgId, sizeByteMsgId, srcMsgIdPtr, sizeByteMsgId);
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s][memcpy_s] Reading msgId ERROR[%d].", __func__, ret), HCCL_E_INTERNAL);
    HCCL_INFO("[%s] Reading %zu bytes msgId from shared mem SUCCESS. msgId = %u.", __func__, sizeByteMsgId, *msgId);

    HCCL_INFO("[%s] Setting flag = 0 on shared mem START.", __func__);
    ret = memset_s(srcFlagPtr, sizeof(flagReadValue), 0, sizeof(flagReadValue));
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s][memset_s] Resetting flag ERROR[%d]", __func__, ret), HCCL_E_INTERNAL);
    HCCL_INFO("[%s] Setting flag = 0 on shared mem SUCCESS.", __func__);

    HCCL_INFO("[%s] SUCCESS. msgId[%u].", __func__, *msgId);
    return HCCL_SUCCESS;
}

int32_t HcommThreadSynchronize(ThreadHandle thread)
{
    hccl::Thread* threadPtr = ReinterpretAs<hccl::Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    HCCL_INFO("[%s] START. thread[0x%llx].", __func__, thread);

    if (threadPtr->IsDeviceA5()) {
        HCCL_INFO("[%s] Running on A5.", __func__);
        hccl::AicpuTsThread* aicpuTsThreadPtr = dynamic_cast<hccl::AicpuTsThread*>(threadPtr);
        uint32_t sqHead{0};
        uint32_t sqTail{0};
        HCCL_INFO("[%s] Start waiting for RTSQ's head == tail.", __func__);
        do {
            CHK_RET(aicpuTsThreadPtr->GetSqHeadAndTail(sqHead, sqTail));
        } while (sqHead != sqTail);
        HCCL_INFO("[%s] SUCCESS. RTSQ's head == tail.", __func__);
        return HCCL_SUCCESS;
    }

    HCCL_INFO("[%s] NOT Running on A5. No implementation, return SUCCESS.", __func__);
    return HCCL_SUCCESS;
}
#ifdef __cplusplus
}
#endif // __cplusplus
