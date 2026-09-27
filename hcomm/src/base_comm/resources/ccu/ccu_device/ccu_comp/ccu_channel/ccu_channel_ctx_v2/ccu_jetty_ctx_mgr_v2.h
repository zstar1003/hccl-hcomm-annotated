/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_JETTY_CTX_MGR_V2_H
#define CCU_JETTY_CTX_MGR_V2_H

#include <vector>

#include "ccu_jetty_ctx_mgr.h"

#include "ccu_res_specs.h"

namespace hcomm {

struct JettyCtxGroup {
    uint32_t startJettyCtxId{0};
    uint32_t startTaJettyId{0};
    uint32_t useCnt{0};     // 记录对应channel是否已经全部释放
    bool configured{false}; // 记录是否已经配置jettyCtx表
};

class CcuJettyCtxMgrV2 : public CcuJettyCtxMgr {
public:
    CcuJettyCtxMgrV2(const int32_t userDevId, const uint8_t dieId, const uint32_t devPhyId);

    CcuJettyCtxMgrV2() = default;
    ~CcuJettyCtxMgrV2() final = default;

    HcclResult Init() final;
    HcclResult Alloc(
        const uint32_t feId, const uint32_t jettyNum, const uint32_t sqSize, std::vector<JettyInfo>& jettyInfos) final;

    HcclResult
    Config(const uint32_t feId, const std::vector<JettyInfo>& jettyInfos, const std::vector<JettyCfg>& jettyCfgs) final;

    HcclResult Release(const uint32_t feId, const std::vector<JettyInfo>& jettyInfos) final;

private:
    CcuChannelJettyMap channelJettyMap_{8, 1}; // 默认配比8:1
    std::vector<JettyCtxGroup> ctxGroups_{};
    HcclResult CheckCtxGroupsByFeId(const uint32_t feId);
};

}; // namespace hcomm

#endif
