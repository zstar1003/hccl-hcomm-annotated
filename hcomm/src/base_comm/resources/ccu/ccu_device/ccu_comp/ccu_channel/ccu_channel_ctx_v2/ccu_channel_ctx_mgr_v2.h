/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_CHANNEL_CTX_MGR_V2_H
#define CCU_CHANNEL_CTX_MGR_V2_H

#include <vector>

#include "ccu_channel_ctx_mgr.h"

#include "ccu_jetty_ctx_mgr_v2.h"

namespace hcomm {

#pragma pack(push, 1)
struct ChannelDataV2 {
    uint8_t eidRaw[URMA_EID_LEN] = {0};
    /********16 Bytes**********/

    uint16_t vtpLow{0};
    /********18 Bytes**********/

    uint16_t vtpHigh : 8;
    uint16_t ioDieId : 1;
    uint16_t rsv3Bits : 7;
    /********20 Bytes**********/

    uint16_t rsv[6] = {0};
    /********32 Bytes**********/

    ChannelDataV2() : vtpHigh(0), ioDieId{0}, rsv3Bits{0} {}
};
#pragma pack(pop)

class CcuChannelCtxMgrV2 : public CcuChannelCtxMgr {
public:
    CcuChannelCtxMgrV2(const int32_t userDevId, const uint8_t dieId, const uint32_t devPhyId);

    CcuChannelCtxMgrV2() = default;
    ~CcuChannelCtxMgrV2() final = default;

    HcclResult Init() final;
    HcclResult Alloc(const ChannelPara& channelPara, std::vector<ChannelInfo>& channelInfos) final;
    HcclResult Config(const ChannelCfg& channelCfg) final;
    HcclResult Release(const uint32_t channelId) final;

private:
    CcuChannelJettyMap channelJettyMap_{8, 1}; // 默认配比8:1

    CcuJettyCtxMgrV2 jettyCtxMgr_{};

    void AllocateChannelResources(
        const ChannelPara& channelPara, const std::vector<CcuJettyInfo>& jettyInfos, uint32_t startChannelId,
        std::vector<ChannelInfo>& channelInfos);
};

}; // namespace hcomm

#endif
