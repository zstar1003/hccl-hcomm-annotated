/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TYPICAL_WINDOW_MEM_H
#define HCCL_TYPICAL_WINDOW_MEM_H

#include <mutex>
#include <map>
#include "hccl_common.h"

namespace hccl {
class TypicalWindowMem {
public:
    static TypicalWindowMem& GetInstance();

    // Alloc and free window memory
    HcclResult AllocWindowMem(void** ptr, size_t size);
    HcclResult FreeWindowMem(void* ptr);

private:
    TypicalWindowMem();
    ~TypicalWindowMem();
    // Delete copy and move constructors and assign operators
    TypicalWindowMem(TypicalWindowMem const&) = delete;            // Copy construct
    TypicalWindowMem(TypicalWindowMem&&) = delete;                 // Move construct
    TypicalWindowMem& operator=(TypicalWindowMem const&) = delete; // Copy assign
    TypicalWindowMem& operator=(TypicalWindowMem&&) = delete;      // Move assign

    HcclResult FreeAllWinowMem();

    std::mutex windowMemMapMutex_;
    std::map<uint64_t, uint64_t> windowMemMap_; // allocated window mem map
};
} // namespace hccl
#endif // HCCL_TYPICAL_WINDOW_MEM_H
