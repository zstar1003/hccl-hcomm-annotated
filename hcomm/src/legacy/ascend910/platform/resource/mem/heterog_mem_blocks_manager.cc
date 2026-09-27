/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "heterog_mem_blocks_manager.h"

constexpr u32 SMALL_PAGE_SIZE = 4096;

namespace hccl {
HeterogMemBlocksManager::HeterogMemBlocksManager()
    : isinited_(false),
      beginAddr_(nullptr),
      memSize_(0),
      memStartAddr_(nullptr)
{}

HeterogMemBlocksManager::~HeterogMemBlocksManager()
{
    if (memStartAddr_ != nullptr) {
        delete[] memStartAddr_;
    }
    memStartAddr_ = nullptr;
    beginAddr_ = nullptr;
}

HcclResult HeterogMemBlocksManager::Init(u32 memBlockNum)
{
    if (isinited_) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO(
        "HeterogMemBlocksManager Init memBlockNum(%u) expected links num[%u]", memBlockNum,
        memBlockNum / (MEM_BLOCK_RECV_WQE_BATCH_NUM * MEM_BLOCK_DOUBLE));

    u64 memLen = MEM_BLOCK_SIZE * memBlockNum + SMALL_PAGE_SIZE;
    memStartAddr_ = new (std::nothrow) s8[memLen];
    if (memStartAddr_ == nullptr) {
        HCCL_ERROR("[Create][HeterogMemBlocksManager]memStartAddr_ is nullptr");
        return HCCL_E_PARA;
    }
    u64 pageSizeNum = reinterpret_cast<u64>(memStartAddr_) / SMALL_PAGE_SIZE;
    beginAddr_ = reinterpret_cast<void*>((pageSizeNum + 1) * SMALL_PAGE_SIZE);

    CHK_SAFETY_FUNC_RET(memset_s(beginAddr_, MEM_BLOCK_SIZE * memBlockNum, 0, MEM_BLOCK_SIZE * memBlockNum));

    usableBlockQue_.Init(memBlockNum + 1);
    memSize_ = MEM_BLOCK_SIZE * memBlockNum;
    for (u32 i = 0; i < memBlockNum; i++) {
        CHK_RET(usableBlockQue_.Push(static_cast<void*>(static_cast<char*>(beginAddr_) + i * MEM_BLOCK_SIZE)));
    }
    isinited_ = true;
    return HCCL_SUCCESS;
}

HcclResult HeterogMemBlocksManager::Alloc(std::list<void*>& blockList)
{
    std::unique_lock<std::mutex> lock(usableBlockQueMutex_);
    if (blockList.size() > usableBlockQue_.Size()) {
        HCCL_ERROR(
            "[HeterogMemBlocksManager][Alloc]lack of resources, blockListSize[%u] usableBlockQue_Size[%u]",
            blockList.size(), usableBlockQue_.Size());
        return HCCL_E_PARA;
    }

    for (auto& block : blockList) {
        CHK_RET(usableBlockQue_.Pop(block));
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
