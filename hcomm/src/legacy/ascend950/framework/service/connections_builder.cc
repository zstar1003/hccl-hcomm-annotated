/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "connections_builder.h"
#include "stl_util.h"
#include "exception_util.h"
#include "rma_conn_exception.h"
#include "communicator_impl.h"

#include <list>

namespace Hccl {
vector<LinkData> ConnectionsBuilder::GetAvailableLinksVec() const
{
    return vector<LinkData>(availableLinks.begin(), availableLinks.end());
}

void ConnectionsBuilder::BatchBuild(const std::string& opTag, const vector<LinkData>& links)
{
    vector<LinkData> pendingLinks;
    for (auto& link : links) {
        if (Contain(availableLinks, link)) {
            continue;
        }
        pendingLinks.emplace_back(link);
    }
    HCCL_INFO(
        "[%s]opTag[%s] links size[%u], pendingLinks size[%u]", __func__, opTag.c_str(), links.size(),
        pendingLinks.size());

    if (pendingLinks.empty()) {
        return;
    }

    // rmaConnections创建
    HCCL_INFO("create rma connection start");
    CreateRmaConnections(opTag, pendingLinks);
    HCCL_INFO("create rma connection end");

    availableLinks.insert(pendingLinks.begin(), pendingLinks.end());
}

void ConnectionsBuilder::CreateRmaConnections(const std::string& opTag, const vector<LinkData>& links) const
{
    list<RmaConnection*> connTasks;
    for (auto& link : links) {
        connTasks.emplace_back(connManager->Create(opTag, link));
    }
    CHECK_NULLPTR(comm, "[ConnectionsBuilder::CreateRmaConnections] comm is nullptr!");
    if (!comm->GetOpCcuFeatureFlag()) { // 非CCU展开，则在transport中交换 connection // 直接return，下面删除？
        return;
    }
    // CCU使用的connection，则走到下面来搞，待CCU transport搞定后，再删除下面这段
    while (!connTasks.empty()) {
        for (auto connIter = connTasks.begin(); connIter != connTasks.end();) {
            CHECK_NULLPTR((*connIter), "[ConnectionsBuilder::CreateRmaConnections] (*connIter) is nullptr!");
            auto status = (*connIter)->GetStatus();
            if (status == RmaConnStatus::READY) {
                connIter = connTasks.erase(connIter);
            } else if (status == RmaConnStatus::CONN_INVALID || status == RmaConnStatus::CLOSE) {
                THROW<RmaConnException>(StringFormat(
                    "Invalid status occurs when creating RMA connection %s!", (*connIter)->Describe().c_str()));
            } else {
                ++connIter;
            }
        }
    }
}

ConnectionsBuilder::ConnectionsBuilder(CommunicatorImpl& communicator)
{
    this->comm = &communicator;
    connManager = &(comm->GetRmaConnManager());
}

} // namespace Hccl
