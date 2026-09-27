/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HETEROG_MR_MANAGER_PUB_H
#define HETEROG_MR_MANAGER_PUB_H

#include <vector>
#include <list>
#include <mutex>
#include <atomic>
#include <hccl/hccl_types.h>

#include "hccl/base.h"
#include "log.h"

namespace hccl {
constexpr u32 MEM_BLOCK_SIZE = 128;
constexpr u32 MEM_BLOCK_CAP = 64 * 1024;
constexpr u32 MEM_BLOCK_NUM = MEM_BLOCK_CAP - 1;
constexpr u32 MEM_BLOCK_CAP_BIGER = 2048 * 1024;
constexpr u32 MEM_BLOCK_NUM_BIGER = MEM_BLOCK_CAP_BIGER - 1;
constexpr u32 MEM_BLOCK_DOUBLE = 2;
constexpr u32 MEM_BLOCK_RECV_WQE_BATCH_NUM = 192;

struct MemBlockQueue {
    std::atomic<u32> headPos{0};
    std::atomic<u32> tailPos{0};
    std::vector<void*> entry;
    u32 blockCap = MEM_BLOCK_CAP;
    MemBlockQueue() : entry{} {};
    ~MemBlockQueue() {};
    void Init(u32 capNum)
    {
        blockCap = capNum;
        entry.resize(blockCap);
    }
    HcclResult Push(void* item)
    {
        entry[headPos] = item;
        headPos = (headPos + 1) & (blockCap - 1);
        return ((headPos == tailPos) ? HCCL_E_INTERNAL : HCCL_SUCCESS);
    };
    HcclResult Pop(void*& item)
    {
        item = entry[tailPos];
        if (headPos == tailPos) {
            return HCCL_E_INTERNAL;
        }
        tailPos = (tailPos + 1) & (blockCap - 1);
        return HCCL_SUCCESS;
    };
    inline u32 Size() const { return (headPos + blockCap - tailPos) & (blockCap - 1); };
};

class HeterogMemBlocksManager {
public:
    explicit HeterogMemBlocksManager();
    virtual ~HeterogMemBlocksManager();

    HcclResult Init(u32 memBlockNum);

    HcclResult Alloc(std::list<void*>& blockList);

    HcclResult Alloc(void** block)
    {
        if (usableBlockQue_.Size() < 1) {
            HCCL_ERROR("[HeterogMemBlocksManager][Alloc]lack of resources");
            return HCCL_E_PARA;
        }
        std::unique_lock<std::mutex> lock(usableBlockQueMutex_);
        CHK_RET(usableBlockQue_.Pop(*block));
        return HCCL_SUCCESS;
    };

    HcclResult Free(void* block)
    {
        std::unique_lock<std::mutex> lock(usableBlockQueMutex_);
        CHK_RET(usableBlockQue_.Push(block));
        return HCCL_SUCCESS;
    };

    inline void* GetMemAddr() const { return beginAddr_; }

    inline u64 GetMemSize() const { return memSize_; }

private:
    bool isinited_;
    MemBlockQueue usableBlockQue_;
    std::mutex usableBlockQueMutex_;
    void* beginAddr_;
    u64 memSize_;
    s8* memStartAddr_;
};
} // namespace hccl
#endif
