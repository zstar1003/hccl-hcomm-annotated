/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "store_sim_shm_memory_common.h"

#include <cstdint>

#include "sim_log.h"
#include "store_sim_memory_manager.h"
#include "db_sim_runner_ops.h"

namespace sim {
// 在非Host进程中映射可用的设备地址（设备虚拟地址 --> 进程可用的共享内存地址）
void* AcquireDevPtrInNoHostProcess(void* virPtr, sim::PhyMemBlock& phyMem)
{
    uint64_t devPtr = (uint64_t)(uintptr_t)virPtr;

    // 1. 查询虚拟内存信息
    auto virMemRes = RunnerDB::GetOneByPred<sim::VirtualMemBlock>([devPtr](const sim::VirtualMemBlock& virMem) {
        return (
            (virMem.start_ptr <= devPtr) && (devPtr < (virMem.start_ptr + virMem.size))
            && (virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_DEV));
    });
    if (!virMemRes.second) {
        HCCL_VM_ERROR("can not find this buff offset ptr:{:p}", virPtr);
        return nullptr;
    }

    // 2. 计算offset
    auto offset = devPtr - virMemRes.first.start_ptr;

    // 3. 查询物理内存信息
    auto phyMemId = virMemRes.first.phy_mem_id;
    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("can not find phy Mem id:{:d}", phyMemId);
        return nullptr;
    };
    phyMem = *phyMemRes;

    // 4. 获取共享内存基地址
    std::string memName(phyMemRes->name);
    void* hostBasePtr = sim::MemoryManager::GetInstance().AcquireMemByName(memName.c_str());

    // 5. 计算virPtr对应共享内存实际地址
    auto hostPtr = (void*)((uint64_t)hostBasePtr + offset);

    HCCL_VM_DEBUG(
        "devAddr {:#x} mapping share {:#x}, offset {}, mem name {}", devPtr, (uint64_t)hostPtr, offset,
        memName.c_str());
    return hostPtr;
}

void ReleaseInNoHostProcess(const sim::PhyMemBlock& phyMem)
{
    std::string memName(phyMem.name);
    sim::MemoryManager::GetInstance().ReleaseMemByName(memName.c_str());
}
} // namespace sim
