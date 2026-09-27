/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CONNECTED_LINK_MGR_H
#define HCCLV2_CONNECTED_LINK_MGR_H

#include <vector>
#include <unordered_map>
#include "virtual_topo.h"
#include "types.h"

namespace Hccl {
class ConnectedLinkMgr {
public:
    const std::vector<LinkData>& GetLinks(RankId dstRank);

    const std::vector<LinkData>& GetLinks(u32 level, RankId dstRank); // 待修改, 搞成迭代器

    void Reset();

    void ParsePackedData(std::vector<char>& data);

private:
    std::unordered_map<RankId, std::vector<LinkData>> links;

    std::unordered_map<u32, std::unordered_map<RankId, std::vector<LinkData>>> levelRankPairLinkDataMap;
};
} // namespace Hccl
#endif
