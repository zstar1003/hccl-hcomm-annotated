/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RANK_TABLE_CRC_BRIDGE_H
#define RANK_TABLE_CRC_BRIDGE_H

#include <string>
#include <unordered_map>
#include "hccl/base.h"

class RankTableCrcBridge {
public:
    static RankTableCrcBridge& GetInstance();
    ~RankTableCrcBridge();

    void RecordRankTableJsonCrc(s32 deviceLogicId, const std::string& rankTableJson);
    u32 ConsumeRankTableJsonCrc(s32 deviceLogicId);

private:
    std::unordered_map<s32, u32> rankTableJsonCrcMap_;
};

#endif // RANK_TABLE_CRC_BRIDGE_H
