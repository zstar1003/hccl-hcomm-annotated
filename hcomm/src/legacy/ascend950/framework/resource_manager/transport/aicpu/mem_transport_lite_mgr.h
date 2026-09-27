/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEM_TRANSPORT_LITE_MANAGER_H
#define MEM_TRANSPORT_LITE_MANAGER_H

#include <vector>
#include "types.h"
#include "virtual_topo.h"
#include "mem_transport_lite.h"
#include "mirror_task_manager_lite.h"
#include "mem_transport_callback_lite.h"

namespace Hccl {

class MemTransportLiteMgr {
public:
    explicit MemTransportLiteMgr(MirrorTaskManagerLite* mirrorTaskMgrLite) : mirrorTaskMgrLite_(mirrorTaskMgrLite) {}

    MemTransportLite* GetOpbase(const LinkData& linkData);

    MemTransportLite* GetOffload(const std::string& opTag, const LinkData& linkData);

    void Reset();

    void ParseOpbasePackedData(std::vector<char>& data);
    void ParseOpbaseAllPackedData(BinaryStream& binaryStream);
    void ParseOffloadPackedData(const std::string& opTag, std::vector<char>& data);
    void ParseOffloadAllPackedData(BinaryStream& binaryStream);

    void ParseAllPackedData(std::vector<char>& data);

private:
    MirrorTaskManagerLite* mirrorTaskMgrLite_{nullptr};
    bool IsOpbaseExist(const LinkData& linkData);

    using MemTransportLiteMap = std::unordered_map<LinkData, std::unique_ptr<MemTransportLite>, hash<Hccl::LinkData>>;

    MemTransportLiteMap opBaseTranspMap;

    using OffloadTransportLiteMap = std::unordered_map<
        std::string, std::unordered_map<LinkData, std::unique_ptr<MemTransportLite>, hash<Hccl::LinkData>>>;

    OffloadTransportLiteMap offloadTranspMap;
};
} // namespace Hccl
#endif
