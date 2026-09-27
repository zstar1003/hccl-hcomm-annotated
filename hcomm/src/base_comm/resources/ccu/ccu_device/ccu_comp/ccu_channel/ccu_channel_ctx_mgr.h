/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_CHANNEL_CTX_MGR_H
#define CCU_CHANNEL_CTX_MGR_H

#include <vector>
#include <mutex>

#include "hccl_types.h"
#include "ccu_dev_mgr_imp.h"

namespace hcomm {

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

class CcuChannelCtxMgr {
public:
    CcuChannelCtxMgr(const int32_t userDevId, const uint8_t dieId, const uint32_t devPhyId)
        : userDevId_(userDevId),
          dieId_(dieId),
          devPhyId_(devPhyId) {};
    CcuChannelCtxMgr() = default;
    virtual ~CcuChannelCtxMgr() = default;

    virtual HcclResult Init() = 0;

    virtual HcclResult Alloc(const ChannelPara& channelPara, std::vector<ChannelInfo>& channelInfos) = 0;
    virtual HcclResult Config(const ChannelCfg& channelCfg) = 0;
    virtual HcclResult Release(const uint32_t channelId) = 0;

protected:
    bool CheckIfChannelAllocated(const uint32_t channelId) const;

protected:
    std::mutex innerMutex_{};
    int32_t userDevId_{0};
    uint8_t dieId_{0};
    uint32_t devPhyId_{0};

    std::vector<ChannelResInfo> channelResInfos_;
};

} // namespace hcomm

#endif // CCU_CHANNEL_CTX_MGR_H
