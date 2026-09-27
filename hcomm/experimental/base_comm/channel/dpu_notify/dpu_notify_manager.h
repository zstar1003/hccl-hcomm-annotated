/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_EXPERIMENTAL_DPU_NOTIFY_MANAGER_H
#define HCOMM_EXPERIMENTAL_DPU_NOTIFY_MANAGER_H

#include <memory>
#include <vector>
#include <mutex>
#include "hccl_types.h"

namespace hcomm_experimental {

class DpuNotifyManager {
public:
    static DpuNotifyManager& GetInstance();
    ~DpuNotifyManager();

    HcclResult AllocNotifyIds(uint32_t notifyNum, std::vector<uint32_t>& notifyIds);
    HcclResult FreeNotifyIds(uint32_t notifyNum, std::vector<uint32_t>& notifyIds);

private:
    DpuNotifyManager(int numEntries);

    std::vector<unsigned char> notifyIdBitMap; // 存放notify的位图
    std::vector<uint32_t> freeBit;             // 存放每个字节第一个非0的位
    uint32_t byteSize;                         // 字节为单位，即8192的bit对应1024字节

    void UpdateFreeBit(uint32_t index);

    int AllocSingleNotifyId();                  // 申请单个notify
    void FreeSingleNotifyId(uint32_t notifyId); // 回收单个notify

    std::mutex mtxAlloc_;
    std::mutex mtxFree_;
};
} // namespace hcomm_experimental

#endif // HCOMM_EXPERIMENTAL_DPU_NOTIFY_MANAGER_H
