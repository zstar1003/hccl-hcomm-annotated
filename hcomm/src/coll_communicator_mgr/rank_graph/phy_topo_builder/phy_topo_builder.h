/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PHY_TOPO_BUILDER_H
#define PHY_TOPO_BUILDER_H

#include <mutex>
#include <memory>
#include <set>
#include <sys/stat.h>
#include "phy_topo.h"
#include "topo_info.h"
#include "edge_info.h"
#include "topo_common_types.h"

namespace Hccl {
constexpr u32 SUPPORT_MAX_TOPOFILE_SIZE = 512 * 1024; // 512k

struct LinkParams {
    std::shared_ptr<PhyTopo::Node> srcNode;
    std::shared_ptr<PhyTopo::Node> dstNode;
    std::set<std::string> localAPorts;
    std::set<std::string> localBPorts;
    AddrPosition position;
    TopoType topoType;
    u32 topoInstanceId;
    std::set<LinkProtocol> protocols;
};

class PhyTopoBuilder {
public:
    static PhyTopoBuilder& GetInstance();
    void Build(const std::string& topoPath);
    std::shared_ptr<TopoInfo> GetTopoInfo() const;
    void RecoverBuild(const TopoInfo& topoInfo);

private:
    std::shared_ptr<TopoInfo> LoadTopoInfo(const std::string& topoPath);
    std::shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>> CreateGraph(const std::vector<EdgeInfo>& edges) const;
    std::shared_ptr<TopoInfo> topoInfo_;
    std::mutex phyTopoMutex;
};
} // namespace Hccl

#endif // PHY_TOPO_BUILDER_H
