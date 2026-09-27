/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_instance.h"

#include <algorithm>

#include "log.h"
#include "ccu_log.h"

#include "hcom_common.h"

#include "ccu_res_pack.h"
#include "ccu_kernel_mgr.h"
#include "ccu_var_event_res_mgr.h"
#include "ccu_res_specs.h"
#include "ccu_dev_mgr_imp.h"
#include "config_plf_log_v2.h"

namespace hcomm {

using Hccl::PLF_RES;

CcuInstance::~CcuInstance()
{
    // 主动释放资源保证时序，不得随意调整顺序
    for (auto& kernelHandle : kernelHandles_) {
        if (kernelHandle != 0) {
            (void)CcuKernelMgr::GetInstance(userDevId_).UnRegister(kernelHandle);
            kernelHandle = 0;
        }
    }
    kernelHandles_.clear();

    (void)CcuVarEventResMgr::GetInstance(userDevId_).ReleaseByInstance(insHandle_);

    resPack_ = nullptr; // 释放instance持有的CCU资源
    if (ccuDrvHandle_) {
        ccuDrvHandle_ = nullptr; // 先减少引用计数，再尝试关闭
        (void)CcuDeinitFeature(userDevId_);
        // 尝试关闭CCU功能，最后一个调用时会关闭CCU驱动
    }
}

CcuResult CcuInstance::InitByInsType(const CcuInstanceType insType)
{
    if (insType >= CcuInstanceType::CCU_UNUSED) {
        HCCL_ERROR("[CcuInstance][%s] failed, CcuInstanceType[%d] is invalid.", __func__, insType);
        return CcuResult::CCU_E_PARA;
    }

    userDevId_ = HcclGetThreadDeviceId();

    if (!ccuDrvHandle_) {
        CCU_CHK_RET(CcuInitFeature(userDevId_, ccuDrvHandle_));
    }

    if (!resPack_) {
        resPack_.reset(new (std::nothrow) CcuResPack());
        CCU_CHK_PTR_NULL(resPack_);
        CCU_CHK_RET(resPack_->InitByInsType(insType));
        CCU_CHK_RET(FillTotalResDescs());
    }

    isFixedResNum_ = true;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuInstance::InitByResDescs(const CcuResDesc* descs[], uint32_t descNum)
{
    if (descs == nullptr || descNum == 0 || descNum > hcomm::CCU_MAX_IODIE_NUM) {
        HCCL_ERROR("[CcuInstance][%s] failed, invalid descs[%p] descNum[%u].", __func__, descs, descNum);
        return CcuResult::CCU_E_PARA;
    }

    userDevId_ = HcclGetThreadDeviceId();

    if (!ccuDrvHandle_) {
        CCU_CHK_RET(CcuInitFeature(userDevId_, ccuDrvHandle_));
    }

    if (!resPack_) {
        resPack_.reset(new (std::nothrow) CcuResPack());
        CCU_CHK_PTR_NULL(resPack_);
        CCU_CHK_RET(resPack_->InitByResDescs(descs, descNum));
        CCU_CHK_RET(FillTotalResDescs());
    }

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuInstance::InitByAllRes()
{
    userDevId_ = HcclGetThreadDeviceId();

    if (!ccuDrvHandle_) {
        CCU_CHK_RET(CcuInitFeature(userDevId_, ccuDrvHandle_));
    }

    // 申请当前 Device 上所有已使能 ioDie 的全部资源：
    // 先查询各 die 是否启用，仅对已使能的 die 调用 CcuGetXXXNum 获取资源总量，
    // 未启用的 die 其 resNum 保持 0（resNum 默认初始化为 0）
    std::array<hcomm::CcuResDesc, hcomm::CCU_MAX_IODIE_NUM> descs{};
    std::array<const hcomm::CcuResDesc*, hcomm::CCU_MAX_IODIE_NUM> descPtrs{};
    for (uint8_t i = 0; i < hcomm::CCU_MAX_IODIE_NUM; i++) {
        descs[i].dieId = i;
        descPtrs[i] = &descs[i];
    }

    for (uint8_t dieId = 0; dieId < hcomm::CCU_MAX_IODIE_NUM; dieId++) {
        bool dieEnable = false;
        CCU_CHK_RET(CcuGetDieEnableInfo(userDevId_, dieId, dieEnable));
        if (!dieEnable) {
            continue; // 未启用的 die，resNum 保持 0
        }

        uint32_t num = 0;
        // 各资源类型总量查询（per-die），按块分的资源查的是块大小*块总数
        CCU_CHK_RET(CcuGetLoopEngineNum(userDevId_, dieId, num));
        CCU_CHK_RET(descs[dieId].SetResNum(ResType::LOOP, num));
        CCU_CHK_RET(CcuGetMsNum(userDevId_, dieId, num));
        CCU_CHK_RET(descs[dieId].SetResNum(ResType::MS, num));
        CCU_CHK_RET(CcuGetCkeNum(userDevId_, dieId, num));
        CCU_CHK_RET(descs[dieId].SetResNum(ResType::CKE, num));
        CCU_CHK_RET(CcuGetXnNum(userDevId_, dieId, num));
        CCU_CHK_RET(descs[dieId].SetResNum(ResType::XN, num));
        CCU_CHK_RET(CcuGetGsaNum(userDevId_, dieId, num));
        CCU_CHK_RET(descs[dieId].SetResNum(ResType::GSA, num));
        CCU_CHK_RET(CcuGetInstructionNum(userDevId_, dieId, num));
        CCU_CHK_RET(descs[dieId].SetResNum(ResType::INS, num));
        CCU_CHK_RET(CcuGetMissionNum(userDevId_, dieId, num));
        CCU_CHK_RET(descs[dieId].SetResNum(ResType::MISSION, num));
        CCU_CHK_RET(CcuGetCascCntNum(userDevId_, dieId, num));
        CCU_CHK_RET(descs[dieId].SetResNum(ResType::CASC_CNT, num));
    }

    if (!resPack_) {
        resPack_.reset(new (std::nothrow) CcuResPack());
        CCU_CHK_PTR_NULL(resPack_);
        CCU_CHK_RET(resPack_->InitByResDescs(descPtrs.data(), hcomm::CCU_MAX_IODIE_NUM));
        CCU_CHK_RET(FillTotalResDescs());
    }

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuInstance::Reset()
{
    if (!resPack_) {
        return CcuResult::CCU_SUCCESS;
    }

    untranslatedKernelHandles_.clear();
    CCU_CHK_RET(resPack_->Reset());

    CCU_CHK_RET(CcuVarEventResMgr::GetInstance(userDevId_).ExcludeAllocatedFromRepo(insHandle_));
    return CcuResult::CCU_SUCCESS;
}

CcuResPack* CcuInstance::GetResPack() { return resPack_.get(); }

void CcuInstance::SetHandle(CcuInsHandle insHandle) { insHandle_ = insHandle; }

// 累加某 die 上某资源 vector 中各 ResInfo::num，得到该 die 该资源的占用数量
static uint32_t SumResNum(const std::vector<ResInfo>& resInfos, std::string resName)
{
    uint32_t total = 0;
    for (const auto& info : resInfos) {
        total += info.num;
    }
    PLF_CONFIG_INFO(PLF_RES, "[CcuInstance][FillTotalResDescs] resType: %s, resNum: %u", resName.c_str(), total);
    return total;
}

CcuResult CcuInstance::FillTotalResDescs()
{
    if (resPack_ == nullptr) {
        HCCL_ERROR("[CcuInstance][%s] failed, resPack_ is nullptr.", __func__);
        return CcuResult::CCU_E_INTERNAL;
    }
    const auto& resRepo = resPack_->GetCcuResRepo();

    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        totalResDescs_[dieId].dieId = dieId;
        // 各资源类型占用数量 = block 路径 + 非 block 连续路径之和
        CCU_CHK_RET(totalResDescs_[dieId].SetResNum(
            ResType::LOOP, SumResNum(resRepo.blockLoopEngine[dieId], "blockLoopEngine")
                               + SumResNum(resRepo.loopEngine[dieId], "loopEngine")));
        CCU_CHK_RET(totalResDescs_[dieId].SetResNum(
            ResType::MS, SumResNum(resRepo.blockMs[dieId], "blockMs") + SumResNum(resRepo.ms[dieId], "ms")));
        CCU_CHK_RET(totalResDescs_[dieId].SetResNum(
            ResType::CKE, SumResNum(resRepo.blockCke[dieId], "blockCke") + SumResNum(resRepo.cke[dieId], "cke")));
        CCU_CHK_RET(totalResDescs_[dieId].SetResNum(
            ResType::XN, SumResNum(resRepo.blockXn[dieId], "blockXn") + SumResNum(resRepo.xn[dieId], "xn")));
        CCU_CHK_RET(totalResDescs_[dieId].SetResNum(
            ResType::GSA, SumResNum(resRepo.blockGsa[dieId], "blockGsa") + SumResNum(resRepo.gsa[dieId], "gsa")));
        CCU_CHK_RET(
            totalResDescs_[dieId].SetResNum(ResType::MISSION, SumResNum(resRepo.mission.mission[dieId], "mission")));
        // INS 当前无对应占用资源项，写 0
        CCU_CHK_RET(totalResDescs_[dieId].SetResNum(ResType::INS, 0));
    }
    return CcuResult::CCU_SUCCESS;
}

const CcuResDesc& CcuInstance::GetTotalResDescs(uint8_t dieId) const { return totalResDescs_[dieId]; }

CcuResult CcuInstance::SaveKernel(const CcuKernelHandle kernelHandle)
{
    kernelHandles_.push_back(kernelHandle);
    untranslatedKernelHandles_.push_back(kernelHandle);
    return CcuResult::CCU_SUCCESS;
}

const std::vector<CcuKernelHandle>& CcuInstance::GetUntranslatedKernels() { return untranslatedKernelHandles_; }

CcuResult CcuInstance::BeginRegister()
{
    if (registerState_ == RegisterState::REGISTERING) {
        HCCL_ERROR(
            "[CcuInstance][%s] failed, previous register round is not ended, "
            "HcommCcuKernelRegisterEnd is missing before a new HcommCcuKernelRegisterStart.",
            __func__);
        return CcuResult::CCU_E_INTERNAL;
    }
    registerState_ = RegisterState::REGISTERING;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuInstance::CheckRegistering() const
{
    if (registerState_ != RegisterState::REGISTERING) {
        HCCL_ERROR(
            "[CcuInstance][%s] failed, HcommCcuKernelRegister must be called between "
            "HcommCcuKernelRegisterStart and HcommCcuKernelRegisterEnd.",
            __func__);
        return CcuResult::CCU_E_INTERNAL;
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuInstance::EndRegister()
{
    if (registerState_ == RegisterState::IDLE) {
        HCCL_ERROR(
            "[CcuInstance][%s] failed, HcommCcuKernelRegisterEnd is called without a matching "
            "HcommCcuKernelRegisterStart.",
            __func__);
        return CcuResult::CCU_E_INTERNAL;
    }
    if (registerState_ == RegisterState::REGISTER_ABORTED) {
        HCCL_WARNING(
            "[CcuInstance][%s] previous register round was aborted due to error, "
            "close it to keep Start/End paired, no kernel will be translated.",
            __func__);
    }
    registerState_ = RegisterState::IDLE;
    return CcuResult::CCU_SUCCESS;
}

void CcuInstance::AbortRegister()
{
    for (auto kernelHandle : untranslatedKernelHandles_) {
        if (kernelHandle == 0) {
            continue;
        }
        (void)CcuKernelMgr::GetInstance(userDevId_).UnRegister(kernelHandle);
        auto it = std::find(kernelHandles_.begin(), kernelHandles_.end(), kernelHandle);
        if (it != kernelHandles_.end()) {
            kernelHandles_.erase(it);
        }
    }
    untranslatedKernelHandles_.clear();
    registerState_ = RegisterState::REGISTER_ABORTED;
}

CcuResult CcuInstance::CascCntHandleAlloc(uint8_t dieId, HcommCcuCascCntHandle handle)
{
    if (resPack_ == nullptr) {
        HCCL_ERROR("[CcuInstance][%s] failed, resPack_ is nullptr.", __func__);
        return CcuResult::CCU_E_INTERNAL;
    }

    CCU_CHK_RET(resPack_->AcquireCascCntBlock(dieId, handle));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuInstance::GetCascCntBlock(HcommCcuCascCntHandle handle, CntXnBlock& cascCntBlock)
{
    if (resPack_ == nullptr) {
        HCCL_ERROR("[CcuInstance][%s] failed, resPack_ is nullptr.", __func__);
        return CcuResult::CCU_E_INTERNAL;
    }

    CCU_CHK_RET(resPack_->GetCascCntBlock(handle, cascCntBlock));
    return CcuResult::CCU_SUCCESS;
}
} // namespace hcomm
