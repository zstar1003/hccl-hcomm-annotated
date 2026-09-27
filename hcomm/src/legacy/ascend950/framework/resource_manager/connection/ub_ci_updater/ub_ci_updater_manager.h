/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_UB_CI_UPDATER_MANAGER_H
#define HCCLV2_UB_CI_UPDATER_MANAGER_H

#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include "ub_ci_updater.h"
#include "rma_conn_manager.h"

namespace Hccl {

class UbCiUpdaterManager {
public:
    explicit UbCiUpdaterManager(const RmaConnManager* rmaConnMgr);
    ~UbCiUpdaterManager();

    void SaveConnsCi(const std::string& opTag);
    void UpdateConnsCi(const std::string& opTag);

private:
    const RmaConnManager* rmaConnMgrPtr;
    std::unordered_map<std::string, std::vector<std::unique_ptr<DevUbConnection::UbCiUpdater>>> ubCiUpdaters;

    std::vector<std::unique_ptr<DevUbConnection::UbCiUpdater>> BatchCreate(const std::string& opTag) const;
    std::vector<DevUbConnection::UbCiUpdater*> Get(const std::string& opTag);
};

} // namespace Hccl

#endif // HCCLV2_UB_CI_UPDATER_MANAGER_H
