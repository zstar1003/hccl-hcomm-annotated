/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <store/store_sim_store_pub.h>

#include "sim_log.h"
#include "db_sim_op_db_ops.h"
#include "store_sim_shm_memory_common.h"

using namespace HcclSim;

void VmPtrReleaser::operator()(void* ptr) const
{
    if (ptr != nullptr) {
        sim::ReleaseInNoHostProcess(phyMem);
    }
}

// Virtual Runtime
HcclVmResult GetAddrByOffset(uint64_t offset, VmUniquePtr& addrPtr) // todo
{
    // 1. 入参合法性检查（避免空指针和无效输出）
    if (addrPtr) {
        HCCL_VM_ERROR("错误：指针已持有资源，不可重复获取！");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    void* offserPtr = reinterpret_cast<void*>(static_cast<uintptr_t>(offset));
    sim::PhyMemBlock phyMem{};
    void* addr = sim::AcquireDevPtrInNoHostProcess(offserPtr, phyMem);
    if (addr == nullptr) {
        HCCL_VM_ERROR("错误：无法获取设备地址(addr= {})！", offserPtr);
        return HcclVmResult::HCCL_SIM_E_PTR;
    }

    // 5. 日志打印（可选，便于问题排查）
    HCCL_VM_INFO("计算成功, 虚拟地址: {}, 物理地址: {}", offset, addr);

    addrPtr = VmUniquePtr(addr, VmPtrReleaser{phyMem});

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult InsertTaskToCollection(HcclTaskMetaData* task, uint32_t* index)
{
    static std::atomic<uint32_t> g_taskSeq{0};

    // 1. 入参合法性检查（避免空指针访问）
    if (task == nullptr) {
        HCCL_VM_ERROR("错误：输入任务指针 task 不能为空！");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    if (index == nullptr) {
        HCCL_VM_ERROR("错误：输出索引指针 index 不能为空！");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    // 2. 检查 TaskMemoryManager 是否已初始化
    // auto &taskMemMgr = sim::TaskMemoryManager::GetInstance();
    // 3. 写入任务信息到数据库
    sim::OpTaskTab opTaskInfo;
    opTaskInfo.id = 0;         // 数据库自增，无需设置
    opTaskInfo.opDetailId = 0; // 默认关联ID为0，需在上游调用处关联
    opTaskInfo.taskSeq = g_taskSeq.fetch_add(1, std::memory_order_relaxed);

    // 序列化 HcclTaskMetaData 到 blob
    opTaskInfo.optaskMeta.assign(
        reinterpret_cast<const uint8_t*>(task), reinterpret_cast<const uint8_t*>(task) + sizeof(HcclTaskMetaData));

    auto ret = sim::InsertOpTask(opTaskInfo);
    if (ret != 0) {
        HCCL_VM_ERROR("错误：插入任务到数据库失败 - {}", ret);
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}
