/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCP_TLV_HDC_MGR_H
#define HCCP_TLV_HDC_MGR_H

#include <mutex>

#include "hccl_types.h"

namespace hcomm {

using TlvHandle = void*;

class HccpTlvHdcMgr {
public:
    static HccpTlvHdcMgr& GetInstance(uint32_t devPhyId);
    TlvHandle GetHandle();
    HcclResult Init();

private:
    HccpTlvHdcMgr() = default;
    ~HccpTlvHdcMgr();
    HccpTlvHdcMgr(const HccpTlvHdcMgr& hccpTlvHdcMgr) = delete;
    HccpTlvHdcMgr& operator=(const HccpTlvHdcMgr& hccpTlvHdcMgr) = delete;
    HcclResult Deinit();

private:
    uint32_t devPhyId_{0};
    std::mutex innerMutex_{};
    bool initFlag_{false};
    TlvHandle tlvHandle_{nullptr};
};

} // namespace hcomm

#endif // HCCP_TLV_HDC_MGR_H
