/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_ccu_res.h"

#include <mutex>

#include "hccl_comm_pub.h"

#include "exception_handler.h"
#include "ccu_instance_mgr.h"
#include "ccu_device_res.h"

static HcclResult CreateCcuInsByFixedResNum(
    const char* funcName, const std::string& commId, hccl::MyRank* myRank, CcuInsHandle& ccuInsHandle)
{
    const auto opExpansionMode = myRank->GetOpExpansionMode();
    auto ccuInsType = hccl::OpExpansionModeToCcuInstanceType(opExpansionMode);
    if (ccuInsType == CcuInstanceType::CCU_UNUSED) {
        HCCL_WARNING(
            "[%s] failed to get ccu instance, commId[%s] op expansion mode[%u].", funcName, commId.c_str(),
            opExpansionMode);
        return HcclResult::HCCL_E_UNAVAIL;
    }

    CcuInsHandle newHandle = 0;
    auto ccuRet = CcuResult::CCU_SUCCESS;
    // CCU_MS 资源不足（CCU_E_UNAVAIL）时降级到 CCU_SCHED 重试一次；CCU_SCHED 不再继续降级
    while (true) {
        ccuRet = HcommCcuInsCreateLegacy(ccuInsType, &newHandle);
        if (ccuRet != CcuResult::CCU_E_UNAVAIL || ccuInsType != CcuInstanceType::CCU_MS) {
            break;
        }
        HCCL_WARNING(
            "[%s] ccu instance resource unavailable for CCU_MS, fallback to CCU_SCHED, commId[%s].", funcName,
            commId.c_str());
        ccuInsType = CcuInstanceType::CCU_SCHED;
    }
    if (ccuRet == CcuResult::CCU_E_UNAVAIL) {
        HCCL_WARNING(
            "[%s] failed to create ccu instance, resources are unavailable, "
            "commId[%s] insType[%d] ret[%d].",
            funcName, commId.c_str(), ccuInsType, ccuRet);
        return static_cast<HcclResult>(ccuRet);
    }
    if (ccuRet != CcuResult::CCU_SUCCESS) {
        HCCL_ERROR(
            "[%s] failed to create ccu instance, commId[%s] insType[%d] ret[%d].", funcName, commId.c_str(), ccuInsType,
            ccuRet);
        return static_cast<HcclResult>(ccuRet);
    }
    myRank->SetCcuInstance(newHandle);
    ccuInsHandle = newHandle;
    return HcclResult::HCCL_SUCCESS;
}

/**
 * @note 职责：集合通信的通信域CCU管理的C接口的C到C++适配
 */
HcclResult HcclCommQueryCcuIns(HcclComm comm, CcuInsHandle* insHandles, uint32_t* insNum)
{
    EXCEPTION_HANDLE_BEGIN

    HcclUs startut = TIME_NOW();

    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(insHandles);
    CHK_PTR_NULL(insNum);
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    const auto& commId = hcclComm->GetIdentifier();
    HCCL_INFO("[%s] CommId[%s] query ccu instance.", __func__, commId.c_str());

    // CCU不支持A5之前代际
    if (!hcclComm->IsCommunicatorV2()) {
        HCCL_RUN_WARNING("[%s] is not supported.", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    auto* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    auto* myRank = collComm->GetMyRank();
    CHK_PTR_NULL(myRank);

    // 查询通信域自有的 ccuInsHandle_；为 0 时按 opExpansionMode 创建
    auto ccuInsHandle = myRank->GetCcuInstance();
    if (ccuInsHandle == 0) {
        auto ret = CreateCcuInsByFixedResNum(__func__, commId, myRank, ccuInsHandle);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_WARNING("[%s] failed to get ccu instance, commId[%s] ret[%d]", __func__, commId.c_str(), ret);
            return ret;
        }
    }

    insHandles[0] = ccuInsHandle;
    *insNum = 1;
    HCCL_INFO("[%s] success, take time [%lld]us.", __func__, DURATION_US(TIME_NOW() - startut).count());

    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclCommQueryAssignedCcuIns(HcclComm comm, CcuInsHandle* insHandles, uint32_t* insNum)
{
    EXCEPTION_HANDLE_BEGIN

    HcclUs startut = TIME_NOW();

    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(insHandles);
    CHK_PTR_NULL(insNum);
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    const auto& commId = hcclComm->GetIdentifier();
    HCCL_INFO("[%s] CommId[%s] query assigned ccu instance.", __func__, commId.c_str());

    // CCU不支持A5之前代际
    if (!hcclComm->IsCommunicatorV2()) {
        HCCL_RUN_WARNING("[%s] is not supported.", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    auto* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    auto* myRank = collComm->GetMyRank();
    CHK_PTR_NULL(myRank);

    // 查询绑定的 assignedCcuInsHandle_；未绑定时返回 UNAVAIL，不创建
    auto assignedCcuInsHandle = myRank->GetAssignedCcuInstance();
    if (assignedCcuInsHandle == 0) {
        HCCL_WARNING("[%s] assigned ccu instance not exist, commId[%s].", __func__, commId.c_str());
        return HcclResult::HCCL_E_UNAVAIL;
    }

    insHandles[0] = assignedCcuInsHandle;
    *insNum = 1;
    HCCL_INFO("[%s] success, take time [%lld]us.", __func__, DURATION_US(TIME_NOW() - startut).count());

    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclCommAssignCcuIns(HcclComm comm, CcuInsHandle insHandle)
{
    EXCEPTION_HANDLE_BEGIN

    HcclUs startut = TIME_NOW();

    CHK_PTR_NULL(comm);
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    const auto& commId = hcclComm->GetIdentifier();
    HCCL_INFO(
        "[%s] CommId[%s] assign ccu instance[%llu].", __func__, commId.c_str(),
        static_cast<unsigned long long>(insHandle));

    if (insHandle == 0) {
        HCCL_ERROR(
            "[%s] failed, commId[%s] insHandle[%llu] is invalid.", __func__, commId.c_str(),
            static_cast<unsigned long long>(insHandle));
        return HcclResult::HCCL_E_PARA;
    }

    if (!hcclComm->IsCommunicatorV2()) {
        HCCL_RUN_WARNING("[%s] is not supported.", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    auto* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    auto* myRank = collComm->GetMyRank();
    CHK_PTR_NULL(myRank);

    {
        // 仅保证多个 Assign 调用之间并发安全，Query 和通信域销毁由调用方保证不与 Assign 并发。
        static std::mutex assignCcuInsMutex;
        std::lock_guard<std::mutex> lock(assignCcuInsMutex);

        auto oldInsHandle = myRank->GetAssignedCcuInstance();
        if (oldInsHandle != 0) {
            HCCL_ERROR(
                "[%s] failed, commId[%s] already has assigned ccu instance[%llu], "
                "new instance[%llu] will not be assigned.",
                __func__, commId.c_str(), static_cast<unsigned long long>(oldInsHandle),
                static_cast<unsigned long long>(insHandle));
            return HcclResult::HCCL_E_PARA;
        }

        const auto devLogicId = collComm->GetDeviceLogicId();
        auto* ccuIns = hcomm::CcuInstanceMgr::GetInstance(devLogicId).Get(insHandle);
        if (ccuIns == nullptr) {
            HCCL_ERROR(
                "[%s] failed, commId[%s] ccu instance[%llu] is not found.", __func__, commId.c_str(),
                static_cast<unsigned long long>(insHandle));
            return HcclResult::HCCL_E_NOT_FOUND;
        }

        myRank->SetAssignedCcuInstance(insHandle);
    }

    HCCL_INFO(
        "[%s] success, commId[%s] ccu instance[%llu], take time [%lld]us.", __func__, commId.c_str(),
        static_cast<unsigned long long>(insHandle), DURATION_US(TIME_NOW() - startut));

    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}
