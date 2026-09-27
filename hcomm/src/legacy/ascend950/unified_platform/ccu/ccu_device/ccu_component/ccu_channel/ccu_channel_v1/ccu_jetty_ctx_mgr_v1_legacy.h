/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_JETTY_CTX_MGR_V1_H
#define HCCL_CCU_JETTY_CTX_MGR_V1_H

#include <memory>
#include <vector>
#include <unordered_map>

#include "ccu_jetty_ctx_mgr_legacy.h"
#include "ccu_res_allocator_legacy.h"

namespace Hccl {

class CcuJettyCtxMgrV1 : public CcuJettyCtxMgr {
public:
    CcuJettyCtxMgrV1(const int32_t devLogicId, const uint8_t dieId, const uint32_t devPhyId);

    CcuJettyCtxMgrV1() = default;
    ~CcuJettyCtxMgrV1() override = default;

    HcclResult Alloc(
        const uint32_t feId, const uint32_t jettyNum, const uint32_t sqSize,
        std::vector<JettyInfo>& jettyInfos) override;
    HcclResult Config(
        const uint32_t feId, const std::vector<JettyInfo>& jettyInfos, const std::vector<JettyCfg>& jettyCfgs) override;
    HcclResult Release(const uint32_t feId, const std::vector<JettyInfo>& jettyInfos) override;

private:
    struct JettyAllocator {
        PfeJettyStrategy strategy;
        std::unique_ptr<CcuResIdAllocator> idAllocator;

        explicit JettyAllocator(PfeJettyStrategy pfeJettyStrategy) : strategy(pfeJettyStrategy)
        {
            idAllocator = std::make_unique<CcuResIdAllocator>(strategy.size);
        }
    };

    std::unique_ptr<JettyAllocator> allocator_;
    HcclResult GetJettyAllocator(uint32_t feId, JettyAllocator*& allocatorHandle);
};

}; // namespace Hccl

#endif
