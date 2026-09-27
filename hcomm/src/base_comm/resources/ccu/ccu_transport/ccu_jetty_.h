/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_JETTY_H
#define HCOMM_CCU_JETTY_H

#include "ip_address.h"
#include "ccu_device_pub.h"
#include "hcomm_adapter_hccp.h"
#include "hccl_common.h"

namespace hcomm {

class CcuJetty final {
public:
    CcuJetty(const Hccl::IpAddress& ipAddr, const CcuJettyInfo& jettyInfo); // 暂时改为使用Hccl
    ~CcuJetty();

    HcclResult Init();

    HcclResult SetMappedJettyPriority(uint32_t priority);

    HcclResult CreateJetty(u8 errTimeout);

    HrtRaUbCreateJettyParam GetCreateJettyParam() const;
    HrtRaUbJettyCreatedOutParam GetJettyedOutParam() const;
    HcclResult Clean();
    uint16_t GetJettyId() const { return jettyInfo_.taJettyId; }
    JettyHandle GetJettyHandle() const { return reinterpret_cast<JettyHandle>(jettyHandlePtr_); }
    RdmaHandle GetRdmaHandle() const { return rdmaHandle_; }
    CtxHandle GetCtxHandle() const { return ctxHandle_; }
    uint32_t GetTpn() const { return tpn_; }
    void SetTpn(uint32_t tpn) { tpn_ = tpn; }

private:
    CcuJetty(const CcuJetty& that) = delete;
    CcuJetty& operator=(const CcuJetty& that) = delete;
    CcuJetty(CcuJetty&& that) = delete;
    CcuJetty& operator=(CcuJetty&& that) = delete;

private:
    int32_t userDevId_{0};
    Hccl::IpAddress ipAddr_{};
    CcuJettyInfo jettyInfo_{};

    CtxHandle ctxHandle_{nullptr};
    bool isCreated_{false};
    bool isError_{false};

    RequestHandle reqHandle_{0};
    std::vector<char> reqDataBuffer_{};
    void* jettyHandlePtr_{nullptr};

    HrtRaUbCreateJettyParam inParam_{};
    HrtRaUbJettyCreatedOutParam outParam_{};
    bool mappedJettyPrioritySet_{false};
    uint8_t mappedJettyPriority_{0};

    HcclResult HandleAsyncRequest();
    RdmaHandle rdmaHandle_{nullptr};
    uint32_t tpn_{0};
};

HcclResult
CcuCreateJetty(const Hccl::IpAddress& ipAddr, const CcuJettyInfo& jettyInfo, std::unique_ptr<CcuJetty>& ccuJetty);

} // namespace hcomm
#endif // HCOMM_CCU_JETTY_H
