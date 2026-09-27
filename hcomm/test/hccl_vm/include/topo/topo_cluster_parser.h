/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_CLUSTER_PARSER_H
#define TOPO_CLUSTER_PARSER_H

#include <string>

#include "topo_cluster_ir.h"

class ClusterTopoParser {
public:
    ClusterTopoParser();
    ~ClusterTopoParser();

    ParseStatus ParseClusterDir(const std::string& clusterDir);

    const Network& GetNetwork() const { return network_; }
    Network& GetNetwork() { return network_; }

private:
    ParseStatus ScanSuperPodDirs(const std::string& clusterDir);
    ParseStatus ScanServerDirs(const std::string& superpodDir, SuperPod& superpod);
    ParseStatus ParseServer(const std::string& serverDir, Server& server);

    ParseStatus ParseTopoJson(const std::string& topoPath, Server& server);
    ParseStatus ParseRootinfoJson(const std::string& rootinfoPath, Server& server);
    ParseStatus ParseServersInfoJson(const std::string& serversInfoPath);

    void MergeTopoAndRootinfo(Server& server);
    int ExtractDieIdFromPortId(const std::string& portId);

    std::string FindFileBySuffix(const std::string& dir, const std::string& suffix);

    Network network_;
};

#endif // TOPO_CLUSTER_PARSER_H
