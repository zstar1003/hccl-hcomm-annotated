/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CONNECTIONBUILDER_H
#define HCCLV2_CONNECTIONBUILDER_H

#include <set>
#include <vector>
#include "rma_conn_manager.h"

namespace Hccl {
class CommunicatorImpl;
class ConnectionsBuilder {
public:
    explicit ConnectionsBuilder(CommunicatorImpl& communicator);

    void BatchBuild(const std::string& opTag, const vector<LinkData>& links);

    vector<LinkData> GetAvailableLinksVec() const;

private:
    void CreateRmaConnections(const std::string& opTag, const vector<LinkData>& links) const;

    RmaConnManager* connManager{nullptr};
    CommunicatorImpl* comm{nullptr};
    std::set<LinkData> availableLinks;
};

} // namespace Hccl

#endif // HCCLV2_CONNECTIONBUILDER_H
