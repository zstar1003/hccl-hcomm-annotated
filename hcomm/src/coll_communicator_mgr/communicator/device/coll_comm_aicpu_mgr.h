/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_COMM_AICPU_MGR_H
#define COLL_COMM_AICPU_MGR_H

#include "coll_comm_aicpu.h"
#include <shared_mutex>
#include <unordered_map>
#include <vector>
#include <string>
#include <utility>

class CollCommAicpuMgr {
public:
    static CollCommAicpuMgr& GetInstance();

    // 通信域初始化 (原 AicpuIndOpCommInit)
    HcclResult InitComm(CommAicpuParam* commAicpuParam);

    // 通信域注册表操作
    CollCommAicpu* AcquireCommForUse(const std::string& group); // 获取并标记使用中 (原 AicpuGetCommMgrbyGroup)
    HcclResult
    AcquireAndCreateComm(const std::string& group, CollCommAicpu** outComm); // 创建或获取通信域（不标记使用中）
    void ReleaseComm(const std::string& group);               // 释放使用标记 (原 AicpuReleaseCommMgrbyGroup)
    CollCommAicpu* GetCurrentComm(const std::string& group);  // 获取当前线程通信域 (原 AicpuGetComm)
    CollCommAicpu* FindCommByGroup(const std::string& group); // 从 map 按 group 查找
    HcclResult DestroyComm(const std::string& group);
    HcclResult GetAllComms(std::vector<std::pair<std::string, CollCommAicpu*>>& aicpuCommInfo);
    std::shared_mutex& GetMutex();
    CollCommAicpu* GetCurrentComm() { return currentComm_; }

    // 全局环境初始化 (原 CollCommAicpu::InitIndopEnv / InitBackGroundThread)
    void InitIndopEnv(CommAicpuParam* commAicpuParam);
    void InitBackGroundThread(u32 devId);

private:
    CollCommAicpuMgr() = default;
    ~CollCommAicpuMgr() = default;
    CollCommAicpuMgr(const CollCommAicpuMgr&) = delete;
    CollCommAicpuMgr& operator=(const CollCommAicpuMgr&) = delete;

    struct CommEntry {
        std::unique_ptr<CollCommAicpu> comm;
        bool isUsed{false};
    };

    std::shared_mutex commMapMutex_;
    std::unordered_map<std::string, CommEntry> commMap_;
    static thread_local CollCommAicpu* currentComm_;
};

#endif // COLL_COMM_AICPU_MGR_H
