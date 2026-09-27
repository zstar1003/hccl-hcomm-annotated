/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_drv_handle_mock.h"
#include "../hccl_api_base_test.h"

#define private public
#define protected public

#include "ccu_drv_handle.h"

#include "ccu_log.h"

#include "hccp_tlv.h"
#include "hccp_tlv_hdc_mgr.h"

#include "src/base_comm/resources/ccu/ccu_device/ccu_res_specs.h"
#include "ccu_pfe_cfg_mgr.h"
#include "ccu_comp.h"
#include "src/base_comm/resources/ccu/ccu_device/ccu_res_batch_allocator.h"

#include "ccu_kernel_mgr.h"

#include "adapter_rts.h"

#undef protected
#undef private

namespace hcomm {

static HcclResult HccpRaTlvRequest(const TlvHandle tlvHandle, const u32 tlvModuleType, const u32 tlvCcuMsgType)
{
    CHK_PTR_NULL(tlvHandle);
    struct TlvMsg sendMsg {};
    struct TlvMsg recvMsg {};
    sendMsg.type = tlvCcuMsgType;

    HCCL_INFO("[%s] tlvHandle[%p].", __func__, tlvHandle);
    constexpr u32 RA_TLV_REQUEST_UNAVAIL = 128308;
    int32_t ret = RaTlvRequest(tlvHandle, tlvModuleType, &sendMsg, &recvMsg);
    if (ret == RA_TLV_REQUEST_UNAVAIL || ret == OTHERS_ENOTSUPP) {
        HCCL_RUN_WARNING(
            "[%s] ra tlv request UNAVAIL, tlvHandle[%p], tlvModeulType[%u], tlvCcuMsgType[%u], ret[%d].", __func__,
            tlvHandle, tlvModuleType, tlvCcuMsgType, ret);
        return HCCL_E_AGAIN; // 代表CCU驱动已被拉起，需要等待其他进程退出
    }

    if (ret != 0) {
        HCCL_ERROR(
            "[Request][RaTlv]errNo[0x%016llx] ra tlv request fail. "
            "return: ret[%d], module type[%u], message type[%u]",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), tlvModuleType, tlvCcuMsgType);
        return HcclResult::HCCL_E_NETWORK;
    }

    HCCL_INFO(
        "tlv request success, tlv module type[%u], "
        "message type[%u]",
        tlvModuleType, tlvCcuMsgType);
    return HcclResult::HCCL_SUCCESS;
}

static CcuResult MockCcuDrvHandleInit(hcomm::CcuDrvHandle* This)
{
    HCCL_RUN_INFO("[CcuDrvHandle][%s], userDevId: %d", __func__, This->userDevId_);
    CCU_CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(This->userDevId_), This->devPhyId_));
    auto& tlvHdcMgr = HccpTlvHdcMgr::GetInstance(This->devPhyId_);
    CCU_CHK_RET(tlvHdcMgr.Init());
    This->tlvHandle_ = tlvHdcMgr.GetHandle();
    CCU_CHK_PTR_NULL(This->tlvHandle_);
    // 拉起CCU驱动如果因其他进程占用重复拉起时，返回EAGAIN，日志检查返回值打印warning
    auto ret = HccpRaTlvRequest(This->tlvHandle_, TLV_MODULE_TYPE_CCU, MSG_TYPE_CCU_INIT);
    if (ret == HcclResult::HCCL_E_AGAIN) {
        HCCL_RUN_WARNING(
            "[%s] HccpRaTlvRequest ret[%d], repeat init ccu, userDevId[%d].", __func__, ret, This->userDevId_);
        return CcuResult::CCU_E_DRV_BUSY;
    }
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[%s] failed to init ccu driver, ret[%d] is unexpected.", __func__, ret);
        return CcuResult::CCU_E_DRV_INIT_FAILED;
    }

    CCU_CHK_RET(CcuResSpecifications::GetInstance(This->userDevId_).Init());
    CCU_CHK_RET(CcuPfeCfgMgr::GetInstance(This->userDevId_).Init());
    CCU_CHK_RET(CcuComponent::GetInstance(This->userDevId_).Init());
    CCU_CHK_RET(CcuResBatchAllocator::GetInstance(This->userDevId_).Init());
    CCU_CHK_RET(CcuKernelMgr::GetInstance(This->userDevId_).Init());

    return CcuResult::CCU_SUCCESS;
}

static CcuResult MockCcuDrvHandleDeinit(hcomm::CcuDrvHandle* This)
{
    // 释放流程不打断，不抛异常，尽量尝试释放所有资源
    // 释放有时序要求
    HCCL_RUN_INFO("[CcuDrvHandle] start to deinit ccu driver, userDevId[%d].", This->userDevId_);
    (void)CcuKernelMgr::GetInstance(This->userDevId_).Deinit();
    (void)CcuResBatchAllocator::GetInstance(This->userDevId_).Deinit();
    (void)CcuComponent::GetInstance(This->userDevId_).Deinit();
    (void)CcuPfeCfgMgr::GetInstance(This->userDevId_).Deinit();
    (void)CcuResSpecifications::GetInstance(This->userDevId_).Deinit();

    if (This->tlvHandle_ != 0) {
        (void)HccpRaTlvRequest(This->tlvHandle_, TLV_MODULE_TYPE_CCU, MSG_TYPE_CCU_UNINIT);
        This->tlvHandle_ = 0;
    }

    return CcuResult::CCU_SUCCESS;
}

} // namespace hcomm

void MockCcuDrvHandle()
{
    MOCKER_CPP(&hcomm::CcuDrvHandle::Init).stubs().will(invoke(hcomm::MockCcuDrvHandleInit));
    MOCKER_CPP(&hcomm::CcuDrvHandle::Deinit).stubs().will(invoke(hcomm::MockCcuDrvHandleDeinit));
}
