/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_CHANNEL_MGR_H
#define HCCL_CCU_CHANNEL_MGR_H

#include <vector>

#include "hccl/hccl_types.h"
#include "ccu_device_manager.h"

namespace Hccl {

constexpr uint32_t MASK_VTP = 0xFFFF0000;
constexpr uint32_t MASK_VTP_LOW = 0x0000FFFF;
constexpr uint32_t MASK_VTP_HIGH = 0x000000FF;

struct ChannelResInfo {
    bool allocated{false};
    uint32_t feId{0};
    ChannelInfo channelInfo{};
};

void DumpChannelResInfo(const uint32_t feId, const ChannelInfo& info);
bool IsEidEmpty(const uint8_t (&eidRaw)[URMA_EID_LEN]);

class CcuChannelMgr {
public:
    CcuChannelMgr(const int32_t devLogicId, const uint8_t dieId, const uint32_t devPhyId);
    CcuChannelMgr() = default;
    virtual ~CcuChannelMgr() = default;

    virtual HcclResult Alloc(const ChannelPara& channelPara, std::vector<ChannelInfo>& channelInfos) = 0;
    virtual HcclResult Config(const ChannelCfg& channelCfg) = 0;
    virtual HcclResult Release(const uint32_t channelId) = 0;

protected:
    std::mutex innerMutex;
    int32_t devLogicId{0};
    uint8_t dieId{0};
    uint32_t devPhyId{0};

    std::vector<ChannelResInfo> channelResInfos;
    bool CheckIfChannelAllocated(const uint32_t channelId) const;
};

}; // namespace Hccl

#endif
