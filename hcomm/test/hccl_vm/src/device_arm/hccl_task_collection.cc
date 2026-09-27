/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_task_collection.h"

#include <cstdint>

#include "sim_log.h"
#include "db_sim_op_db_ops.h"

void InsertTaskToCollectionDev(HcclTaskMetaData* task)
{
    static std::atomic<uint32_t> g_taskSeq{0};

    // 1. 入参合法性检查（避免空指针访问）
    if (task == nullptr) {
        HCCL_VM_ERROR("错误：输入任务指针 task 不能为空！");
        return;
    }

    // 3. 写入任务信息到数据库
    sim::OpTaskTab opTaskInfo;
    opTaskInfo.id = 0;         // 数据库自增，无需设置
    opTaskInfo.opDetailId = 0; // 默认关联ID为0，需在上游调用处关联
    opTaskInfo.taskSeq = g_taskSeq.fetch_add(1, std::memory_order_relaxed);

    // 序列化 HcclTaskMetaData 到 blob
    opTaskInfo.optaskMeta.assign(
        reinterpret_cast<const uint8_t*>(task), reinterpret_cast<const uint8_t*>(task) + sizeof(HcclTaskMetaData));

    auto ret = sim::InsertOpTask(opTaskInfo, true);
    if (ret != 0) {
        HCCL_VM_ERROR("错误：插入任务到数据库失败 - {}", ret);
        return;
    }
    return;
}
