/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dpu_notify_manager.h"
#include "log.h"
namespace hcomm {

constexpr uint32_t BYTE_SIZE = 8; // 一个字节占8位，用于比较赋值等，避免使用魔法数字

DpuNotifyManager& DpuNotifyManager::GetInstance()
{
    int numNotify = 8192;
    static DpuNotifyManager instance{numNotify};
    return instance;
}

DpuNotifyManager::DpuNotifyManager(int numEntries)
{
    if (numEntries <= 0) {
        numEntries = 1;
    }
    byteSize = (numEntries + BYTE_SIZE - 1) / BYTE_SIZE;
    notifyIdBitMap = std::vector<unsigned char>(byteSize, 0x00);
    freeBit = std::vector<uint32_t>(byteSize, 0);
}

DpuNotifyManager::~DpuNotifyManager() = default;

void DpuNotifyManager::UpdateFreeBit(uint32_t index)
{
    for (uint32_t j = 0; j < BYTE_SIZE; ++j) {
        if (!(notifyIdBitMap[index] & (1 << j))) { // 检查第j位是否为0
            freeBit[index] = j;
            return;
        }
    }
    freeBit[index] = BYTE_SIZE; // 都检查完了，已满
}

int DpuNotifyManager::AllocSingleNotifyId()
{
    for (uint32_t i = 0; i < byteSize; ++i) {
        if (notifyIdBitMap[i] != 0xFF) {            // 当前字节未满
            notifyIdBitMap[i] |= (1 << freeBit[i]); // 设置为1
            size_t tmp = freeBit[i];                // 后续update会覆盖，所以先存起来。
            UpdateFreeBit(i);
            return i * BYTE_SIZE + tmp;
        }
    }
    return -1; // 无可用资源
}

HcclResult DpuNotifyManager::AllocNotifyIds(
    uint32_t notifyNum,
    std::vector<uint32_t>& notifyIds) // std::unique_ptr<uint64_t[]>& handles)
{
    if (notifyNum == 0) {
        HCCL_INFO("[DpuNotifyManager::%s] notifyNum == 0, no need to alloc.", __func__);
        return HCCL_SUCCESS;
    }

    if (notifyNum > byteSize * BYTE_SIZE) {
        HCCL_ERROR(
            "[DpuNotifyManager::%s] notifyNum[%u] is too large. Maximum: %u.", __func__, notifyNum,
            byteSize * BYTE_SIZE);
        return HCCL_E_PARA;
    }

    std::lock_guard<std::mutex> lock(mtxAlloc_);

    notifyIds.resize(notifyNum);

    for (uint32_t i = 0; i < notifyNum; ++i) {
        int notifyId = AllocSingleNotifyId();
        if (notifyId == -1) {
            // 空间不足，释放之前的申请，并报错
            FreeNotifyIds(i, notifyIds);
            HCCL_ERROR("[DpuNotifyManager::%s] no free notify to alloc.", __func__);
            return HCCL_E_MEMORY;
        }
        notifyIds[i] = notifyId; // 存起来
    }
    return HCCL_SUCCESS;
}

void DpuNotifyManager::FreeSingleNotifyId(uint32_t notifyId)
{
    // 校验输入是否溢出
    if (notifyId >= byteSize * BYTE_SIZE) {
        HCCL_INFO("[DpuNotifyManager::%s] no need to free.", __func__);
        return;
    }
    uint32_t byteIndex = notifyId / BYTE_SIZE;
    uint32_t bitIndex = notifyId % BYTE_SIZE;
    notifyIdBitMap[byteIndex] &= ~(1 << bitIndex);
    UpdateFreeBit(byteIndex);
    return;
}

HcclResult DpuNotifyManager::FreeNotifyIds(uint32_t notifyNum, std::vector<uint32_t>& notifyIds)
{
    if (notifyNum == 0) { // 没有需要回收的notify
        HCCL_INFO("[DpuNotifyManager::%s] notifyNum == 0, no need to free.", __func__);
        return HCCL_SUCCESS;
    }

    if (notifyNum > notifyIds.size()) {
        HCCL_ERROR(
            "[DpuNotifyManager::%s] notifyNum[%u] > notifyIds length[%zu].", __func__, notifyNum, notifyIds.size());
        return HCCL_E_PARA;
    }

    std::lock_guard<std::mutex> lock(mtxFree_);

    for (uint32_t i = 0; i < notifyNum; ++i) {
        FreeSingleNotifyId(notifyIds[i]);
    }
    return HCCL_SUCCESS;
}

} // namespace hcomm
