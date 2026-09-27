/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_CLUSTER_DUMPER_H
#define TOPO_CLUSTER_DUMPER_H

#include <ostream>
#include <string>

#include "topo_cluster_ir.h"

class ClusterTopoDumper {
public:
    static void DumpToFile(const Network& network, const std::string& filePath);
    static void DumpToStream(const Network& network, std::ostream& os);

private:
    static void DumpNetwork(const Network& network, std::ostream& os, int indent);
    static void DumpSuperPod(const SuperPod& superpod, std::ostream& os, int indent);
    static void DumpServer(const Server& server, std::ostream& os, int indent);
    static void DumpDevice(const Device& device, std::ostream& os, int indent);
    static void DumpPort(const Port& port, std::ostream& os, int indent);
    static void DumpLink(const Link& link, std::ostream& os, int indent);
    static void DumpLinkPortRef(const LinkPortRef& ref, std::ostream& os, int indent);
    static std::string Indent(int level);
};

#endif // TOPO_CLUSTER_DUMPER_H
