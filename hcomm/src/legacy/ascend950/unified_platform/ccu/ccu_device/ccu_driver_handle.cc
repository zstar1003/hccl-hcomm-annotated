/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_driver_handle.h"
#include "hccp_tlv_hdc_manager.h"
#include "hccp_tlv.h"

#include "ccu_context_mgr_imp.h"
#include "ccu_res_batch_allocator_legacy.h"
#include "ccu_component.h"
#include "ccu_res_specs_legacy.h"

namespace Hccl {

CcuDriverHandle::CcuDriverHandle(s32 deviceLogicId) : devLogicId(deviceLogicId) {}

CcuDriverHandle::~CcuDriverHandle()
{
    HCCL_RUN_INFO("Start destroying CCU, deviceLogicId: %d", devLogicId);

    HCCL_INFO("Start destroying CCU, deviceLogicId: %d", devLogicId);

    DECTOR_TRY_CATCH("CcuDriverHandle", {
        // 清理ccu平台层持有的资源与缓存信息
        CtxMgrImp::GetInstance(devLogicId).Deinit();
        CcuResBatchAllocator::GetInstance(devLogicId).Deinit();
        CcuComponent::GetInstance(devLogicId).Deinit();
        CcuResSpecifications::GetInstance(devLogicId).Deinit();
        // 关闭ccu驱动通道
        auto tlvHandle = HccpTlvHdcManager::GetInstance().GetTlvHandle(devLogicId);
        HrtRaTlvRequest(tlvHandle, TLV_MODULE_TYPE_CCU, MSG_TYPE_CCU_UNINIT);
    });

    HCCL_INFO("Destroy CCU success, deviceLogicId: %d", devLogicId);
}

HcclResult CcuDriverHandle::Init() const
{
    HCCL_RUN_INFO("Start initiating CCU, deviceLogicId: %d", devLogicId);

    auto tlvHandle = HccpTlvHdcManager::GetInstance().GetTlvHandle(devLogicId);
    if (HrtRaTlvRequest(tlvHandle, TLV_MODULE_TYPE_CCU, MSG_TYPE_CCU_INIT) == HCCL_E_UNAVAIL) {
        return HCCL_E_UNAVAIL;
    }

    CcuResSpecifications::GetInstance(devLogicId).Init();
    CcuComponent::GetInstance(devLogicId).Init();
    CcuResBatchAllocator::GetInstance(devLogicId).Init();
    CtxMgrImp::GetInstance(devLogicId).Init();

    HCCL_INFO("Init CCU success, deviceLogicId: %d", devLogicId);
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
