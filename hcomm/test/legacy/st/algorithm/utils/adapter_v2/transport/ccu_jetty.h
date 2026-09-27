/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_JETTY_H
#define HCCL_CCU_JETTY_H

#include <memory>

#include "ip_address.h"
#include "ccu_dev_mgr.h"
#include "orion_adapter_hccp.h"

namespace Hccl {

class CcuJetty final {
public:
    CcuJetty(const IpAddress& ipAddr, const CcuJettyInfo& jettyInfo);
    ~CcuJetty();
    CcuJetty(const CcuJetty& that) = delete;
    CcuJetty& operator=(const CcuJetty& that) = delete;
    CcuJetty(CcuJetty&& that) = delete;
    CcuJetty& operator=(CcuJetty&& that) = delete;

    HcclResult CreateJetty();

    HrtRaUbCreateJettyParam GetCreateJettyParam() const;
    HrtRaUbJettyCreatedOutParam GetJettyedOutParam() const;

private:
    int32_t devLogicId_{0};
    IpAddress ipAddr_{};
    CcuJettyInfo jettyInfo_{};

    RdmaHandle rdmaHandle_{nullptr};
    bool isCreated_{false};
    bool isError_{false};

    RequestHandle reqHandle_{0};
    std::vector<char> reqDataBuffer_;
    void* jettyHandlePtr_{nullptr};

    HrtRaUbCreateJettyParam inParam_{};
    HrtRaUbJettyCreatedOutParam outParam_{};

    HcclResult HandleAsyncRequest();
};

//
HcclResult CcuCreateJetty(const IpAddress& ipAddr, const CcuJettyInfo& jettyInfo, std::unique_ptr<CcuJetty>& ccuJetty);

} // namespace Hccl
#endif // HCCL_CCU_JETTY_H
