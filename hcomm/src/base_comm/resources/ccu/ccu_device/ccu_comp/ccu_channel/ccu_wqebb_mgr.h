/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_WQEBB_MGR_H
#define CCU_WQEBB_MGR_H

#include "ccu_dev_mgr_imp.h"

namespace hcomm {

struct WqeBBReq {
    uint32_t sqSize{0};
    uint32_t jettyCtxStartId{0};
};

class CcuWqeBBMgr {
public:
    CcuWqeBBMgr(const int32_t userDevId, const uint8_t dieId) : userDevId_(userDevId), dieId_(dieId) {};
    CcuWqeBBMgr() = default;
    virtual ~CcuWqeBBMgr() = default;

    virtual HcclResult Init() = 0;
    virtual HcclResult Alloc(const WqeBBReq& allocCfg, ResInfo& wqeBBInfo) = 0;
    virtual HcclResult Release(const ResInfo& wqeBBInfo) = 0;

protected:
    int32_t userDevId_{0};
    uint8_t dieId_{0};
};

} // namespace hcomm

#endif // CCU_WQEBB_MGR_H
