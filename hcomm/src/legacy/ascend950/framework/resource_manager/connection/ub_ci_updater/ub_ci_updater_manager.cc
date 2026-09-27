/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_ci_updater_manager.h"
#include "dev_ub_connection.h"

namespace Hccl {

UbCiUpdaterManager::UbCiUpdaterManager(const RmaConnManager* rmaConnMgr) : rmaConnMgrPtr(rmaConnMgr) {}

UbCiUpdaterManager::~UbCiUpdaterManager() { ubCiUpdaters.clear(); }

void UbCiUpdaterManager::SaveConnsCi(const std::string& opTag)
{
    auto ubCiUpdatersPair = ubCiUpdaters.emplace(opTag, BatchCreate(opTag));
    for (auto& ubCiUpdater : ubCiUpdatersPair.first->second) {
        if (ubCiUpdater == nullptr) {
            continue;
        }
        ubCiUpdater->SaveCi();
    }
}

void UbCiUpdaterManager::UpdateConnsCi(const std::string& opTag)
{
    if (ubCiUpdaters.find(opTag) == ubCiUpdaters.end()) {
        return;
    }
    std::vector<DevUbConnection::UbCiUpdater*> tmpUbCiUpdaters = Get(opTag);
    for (auto& ubCiUpdater : tmpUbCiUpdaters) {
        if (ubCiUpdater == nullptr) {
            continue;
        }
        ubCiUpdater->UpdateCi();
    }
}

std::vector<std::unique_ptr<DevUbConnection::UbCiUpdater>>
UbCiUpdaterManager::BatchCreate(const std::string& opTag) const
{
    std::vector<std::unique_ptr<DevUbConnection::UbCiUpdater>> tmpUbCiUpdaters;
    for (auto& rmaConn : rmaConnMgrPtr->GetOpTagConns(opTag)) {
        if (rmaConn->GetRmaConnType() == RmaConnType::UB) {
            if (dynamic_cast<DevUbConnection*>(rmaConn)->GetUbJfcMode() == HrtUbJfcMode::STARS_POLL) {
                tmpUbCiUpdaters.emplace_back(
                    std::make_unique<DevUbConnection::UbCiUpdater>(dynamic_cast<DevUbConnection*>(rmaConn)));
            }
        }
    }
    return tmpUbCiUpdaters;
}

std::vector<DevUbConnection::UbCiUpdater*> UbCiUpdaterManager::Get(const std::string& opTag)
{
    std::vector<DevUbConnection::UbCiUpdater*> tmpUbCiUpdaters;
    auto tagIter = ubCiUpdaters.find(opTag);
    for (auto& ubCiUpdater : tagIter->second) {
        tmpUbCiUpdaters.emplace_back(ubCiUpdater.get());
    }
    return tmpUbCiUpdaters;
}

} // namespace Hccl
