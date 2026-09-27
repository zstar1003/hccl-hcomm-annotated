/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_WQEBB_MGR_V2_H
#define CCU_WQEBB_MGR_V2_H

#include <memory>
#include <vector>

#include "ccu_wqebb_mgr.h"

namespace hcomm {

class CcuWqeBBMgrV2 : public CcuWqeBBMgr {
public:
    CcuWqeBBMgrV2(const int32_t userDevId, const uint8_t dieId) : CcuWqeBBMgr(userDevId, dieId) {}
    CcuWqeBBMgrV2() = default;
    ~CcuWqeBBMgrV2() final = default;

    HcclResult Init() final;
    HcclResult Alloc(const WqeBBReq& wqeBBReq, ResInfo& wqeBBInfo) final;
    HcclResult Release(const ResInfo& wqeBBInfo) final;

private:
    std::vector<bool> allocatedWqeBBBlocks_; // 记录当前所有jetty的分配状态
};

}; // namespace hcomm

#endif // CCU_WQEBB_MGR_V2_H
