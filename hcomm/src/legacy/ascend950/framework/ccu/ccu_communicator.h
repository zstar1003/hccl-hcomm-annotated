/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_COMMUNICATOR_H
#define CCU_COMMUNICATOR_H

#include "types.h"
#include "ccu_respack_mgr.h"
#include "ccu_jetty_mgr.h"
#include "ccu_transport_manager.h"
#include "ccu_registered_ctx_mgr.h"
#include "ccu_transport_group_manager.h"
#include "orion_adapter_rts.h"

namespace Hccl {

class CcuCommunicator {
public:
    explicit CcuCommunicator(CommunicatorImpl* comm)
        : comm(comm),
          devLogicId(HrtGetDevice()),
          ccuResPackMgr(),
          ccuJettyMgr(devLogicId),
          ccuTransportMgr(*comm, devLogicId),
          ccuTransportGroupMgr(*comm),
          registeredCcuCtxMgr(devLogicId)
    {}

    CcuResPackMgr* GetCcuResPackMgr();
    CcuJettyMgr* GetCcuJettyMgr();
    CcuTransportMgr* GetCcuTransportMgr();
    CcuTransportGroupMgr* GetCcuTransportGrpMgr();
    RegisteredCcuCtxMgr* GetRegisteredCcuCtxMgr();
    int32_t GetDeviceLogicId() const;
    void AcceleratorFallback() const;

private:
    CommunicatorImpl* comm;
    int32_t devLogicId;
    CcuResPackMgr ccuResPackMgr;
    CcuJettyMgr ccuJettyMgr;
    CcuTransportMgr ccuTransportMgr;
    CcuTransportGroupMgr ccuTransportGroupMgr;
    RegisteredCcuCtxMgr registeredCcuCtxMgr;
};

} // namespace Hccl

#endif // CCU_COMMUNICATOR_H
