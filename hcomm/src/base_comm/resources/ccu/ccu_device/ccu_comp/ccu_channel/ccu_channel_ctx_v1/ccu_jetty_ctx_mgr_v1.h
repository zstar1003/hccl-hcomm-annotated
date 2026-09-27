/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_JETTY_CTX_MGR_V1_H
#define CCU_JETTY_CTX_MGR_V1_H

#include <memory>
#include <vector>
#include <unordered_map>

#include "ccu_jetty_ctx_mgr.h"
#include "ccu_res_allocator.h"
#include "ccu_wqebb_mgr_v1.h"

namespace hcomm {

class CcuJettyCtxMgrV1 : public CcuJettyCtxMgr {
public:
    CcuJettyCtxMgrV1(const int32_t userDevId, const uint8_t dieId, const uint32_t devPhyId)
        : CcuJettyCtxMgr(userDevId, dieId, devPhyId) {};

    CcuJettyCtxMgrV1() = default;
    ~CcuJettyCtxMgrV1() final = default;

    HcclResult Init() final;

    HcclResult Alloc(
        const uint32_t feId, const uint32_t jettyNum, const uint32_t sqSize, std::vector<JettyInfo>& jettyInfos) final;

    HcclResult
    Config(const uint32_t feId, const std::vector<JettyInfo>& jettyInfos, const std::vector<JettyCfg>& jettyCfgs) final;

    HcclResult Release(const uint32_t feId, const std::vector<JettyInfo>& jettyInfos) final;

private:
    struct JettyAllocator {
        PfeJettyStrategy strategy{};
        std::unique_ptr<CcuResIdAllocator> idAllocator{nullptr};

        explicit JettyAllocator(PfeJettyStrategy pfeJettyStrategy) : strategy(pfeJettyStrategy)
        {
            // 外部对空指针进行校验
            idAllocator.reset(new (std::nothrow) CcuResIdAllocator(strategy.size));
        }
    };

    HcclResult GetJettyAllocator(uint32_t feId, JettyAllocator*& allocatorHandle);

private:
    std::unique_ptr<JettyAllocator> allocator_; // 所有FE的Jetty统一打平分配
};

}; // namespace hcomm

#endif
