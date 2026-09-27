/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_LOADER_H
#define SIM_LOADER_H

#include <hccl_types.h>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "sim_op_db_types.h"

namespace loader {
using ProxyId = uint32_t;

using OpStage = std::vector<sim::CompositeOpDetail>; // 某一次算子任务
using OpPipeline = std::map<uint32_t, OpStage>;      // 所有算子任务

class Loader {
public:
    Loader();
    ~Loader();

    // dbPath: 数据库路径
    HcclResult LoadOpTaskFile(const std::string dbPath = "");

    HcclResult GetSyncInfo(std::vector<sim::SyncRecordTab>& syncRecords);

    HcclResult GetSyncRecordsByStatus(uint8_t status, std::vector<sim::SyncRecordTab>& syncRecords);
    // Runner: 每 sync 一次调用一次，outSyncIter: 输出本次加载的 syncIter
    HcclResult LoadRunnerSingleSync(
        const uint32_t& outSyncIter, std::map<uint32_t, std::vector<sim::CompositeOpDetail>>& compositeDataMap);

    // 根据 syncIter 查询复合算子详情
    HcclResult LoadCompositeOpDetailBySyncIter(
        uint32_t syncIter, std::map<uint32_t, std::vector<sim::CompositeOpDetail>>& compositeDataMap);

    HcclResult GetCcuChannelInfo(std::vector<sim::CcuChannelTab>& channels);
    HcclResult GetJettyMapInfo(std::vector<sim::JettyMapTab>& jettyMaps);
    HcclResult GetInstrResInfo(std::vector<sim::CcuInstrResTab>& instrRes);

    // 返回内部已组装好的完整 Pipeline 缓存
    const OpPipeline& GetOpTasks() const;

private:
    // 成员变量
    OpPipeline opTaskCache_;
};
} // namespace loader

#endif // SIM_LOADER_H
