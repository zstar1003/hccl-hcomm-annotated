/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TYPICAL_MR_MANAGER_H
#define HCCL_TYPICAL_MR_MANAGER_H

#include <mutex>
#include <map>
#include "hccl_common.h"
#include "network/hccp.h"

namespace hccl {

constexpr uint32_t DEFAULT_MR_KEY = 0;

class TypicalMrManager {
public:
    static TypicalMrManager& GetInstance();
    HcclResult RegisterMem(struct MrInfoT& mrInfo);   // register MR
    HcclResult DeRegisterMem(struct MrInfoT& mrInfo); // unregister MR

private:
    TypicalMrManager();
    ~TypicalMrManager();
    // Delete copy and move constructors and assign operators
    TypicalMrManager(TypicalMrManager const&) = delete;            // Copy construct
    TypicalMrManager(TypicalMrManager&&) = delete;                 // Move construct
    TypicalMrManager& operator=(TypicalMrManager const&) = delete; // Copy assign
    TypicalMrManager& operator=(TypicalMrManager&&) = delete;      // Move assign

    HcclResult ReleaseMrResource();

    RdmaHandle rdmaHandle_ = nullptr;
    std::mutex mrMapMutex_;
    std::map<uint32_t, std::pair<struct MrInfoT, MrHandle>> regedMrMap_; // registered MR map
};
} // namespace hccl
#endif // HCCL_TYPICAL_MR_MANAGER_H
