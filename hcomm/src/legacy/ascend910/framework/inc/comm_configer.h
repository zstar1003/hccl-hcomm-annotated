/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMM_CONFIGER_H
#define HCCL_COMM_CONFIGER_H

#include <unordered_map>
#include <memory>
#include <mutex>

#include "comm_config_pub.h"
#include "hccl/hccl_types.h"
#include "hccl_common.h"

namespace hccl {
// config通信域粒度配置相关参数
class CommConfiger {
public:
    static CommConfiger& GetInstance();
    ~CommConfiger();

    HcclResult SetCommConfig(CommConfig config, const std::string& identifier = HCCL_WORLD_GROUP);
    HcclResult SetCommConfigExecTimeOut(s32 execTimeOut, const std::string& identifier = HCCL_WORLD_GROUP);
    s32 GetCommConfigExecTimeOut(const std::string& identifier = HCCL_WORLD_GROUP);
    bool GetCommConfigExecTimeOutSet(const std::string& identifier = HCCL_WORLD_GROUP);
    std::vector<HcclAlgoType> GetCommConfigAlgoConfig(
        const std::string& identifier = HCCL_WORLD_GROUP, HcclCMDType opType = HcclCMDType::HCCL_CMD_ALL);
    bool GetCommConfigInterServerRetryEnable(const std::string& identifier = HCCL_WORLD_GROUP);
    bool GetCommConfigInterSuperPodRetryEnable(const std::string& identifier = HCCL_WORLD_GROUP);
    u32 GetCommConfigRetryMaxCnt(const std::string& identifier = HCCL_WORLD_GROUP);
    u32 GetCommConfigRetryHoldTime(const std::string& identifier = HCCL_WORLD_GROUP);
    u32 GetCommConfigRetryIntervalTime(const std::string& identifier = HCCL_WORLD_GROUP);
    void UnRegisterToCommConfiger(const std::string& identifier = HCCL_WORLD_GROUP);

private:
    CommConfiger();
    std::mutex lock_;                                           // 锁保证多线程安全
    std::unordered_map<std::string, CommConfig> commConfigMap_; // 通信域名称与config配置映射关系
    bool initialized_ = false;
};
} // namespace hccl
#endif /* HCCL_COMM_CONFIGER_H */
