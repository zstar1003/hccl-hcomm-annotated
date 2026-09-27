/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLSIM_STORE_PUB_H
#define HCCLSIM_STORE_PUB_H

#include <cstdint>
#include <memory>

#include "sim_common_defs.h"
#include "sim_models.h"

// Virtual Runtime
struct VmPtrReleaser {
    sim::PhyMemBlock phyMem;
    void operator()(void* ptr) const;
};

using VmUniquePtr = std::unique_ptr<void, VmPtrReleaser>;

HcclSim::HcclVmResult GetAddrByOffset(uint64_t offset, VmUniquePtr& addrPtr);

HcclSim::HcclVmResult InsertTaskToCollection(HcclTaskMetaData* task, uint32_t* index);

#endif
