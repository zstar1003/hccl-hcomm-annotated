/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_res.h"

#include <array>

#include "ccu_device_res.h"
#include "ccu_device_pub.h"

#include "ccu_log.h"

#include "hcom_common.h"
#include "op_base.h"

#include "ccu_kernel_mgr.h"
#include "ccu_instance_mgr.h"
#include "ccu_res_desc.h"
#include "ccu_res_desc_mgr.h"
#include "ccu_res_type_converter.h"

CcuResult HcommCcuInsResDescCreate(uint32_t dieId, HcommCcuResDescHandle* resDesc)
{
    if (dieId >= hcomm::CCU_MAX_IODIE_NUM) {
        HCCL_ERROR("[%s] dieId[%u] is invalid, dieId should be in [0, %u).", __func__, dieId, hcomm::CCU_MAX_IODIE_NUM);
        return CcuResult::CCU_E_PARA;
    }

    CCU_CHK_PTR_NULL(resDesc);

    int32_t userDevId = INVALID_INT;
    CCU_CHK_RET(HcclDeviceRefresh(userDevId));
    CCU_CHK_RET(hcomm::CcuInstanceMgr::GetInstance(userDevId).GetResDescMgr().Create(dieId, *resDesc));
    HCCL_INFO("[%s] success, handle[0x%llx] dieId[%u]", __func__, *resDesc, dieId);
    return CcuResult::CCU_SUCCESS;
}

CcuResult HcommCcuInsResDescDestroy(HcommCcuResDescHandle resDesc)
{
    int32_t userDevId = INVALID_INT;
    CCU_CHK_RET(HcclDeviceRefresh(userDevId));
    CCU_CHK_RET(hcomm::CcuInstanceMgr::GetInstance(userDevId).GetResDescMgr().Destroy(resDesc));
    HCCL_INFO("[%s] success, handle[0x%llx]", __func__, resDesc);
    return CcuResult::CCU_SUCCESS;
}

CcuResult HcommCcuInsResDescSetNum(HcommCcuResDescHandle resDesc, HcommCcuResType resType, uint32_t resNum)
{
    hcomm::ResType ccuResType{hcomm::ResType::INVALID};
    CCU_CHK_RET(hcomm::ConvertHcommCcuResTypeToHcclResType(resType, ccuResType));

    int32_t userDevId = INVALID_INT;
    CCU_CHK_RET(HcclDeviceRefresh(userDevId));
    CCU_CHK_RET(hcomm::CcuInstanceMgr::GetInstance(userDevId).GetResDescMgr().SetResNum(resDesc, ccuResType, resNum));
    HCCL_INFO("[%s] success, handle[0x%llx] resType[%d] resNum[%u]", __func__, resDesc, resType, resNum);
    return CcuResult::CCU_SUCCESS;
}

CcuResult HcommCcuInsResDescQueryNum(HcommCcuResDescHandle resDesc, HcommCcuResType resType, uint32_t* resNum)
{
    CCU_CHK_PTR_NULL(resNum);
    hcomm::ResType ccuResType{hcomm::ResType::INVALID};
    CCU_CHK_RET(hcomm::ConvertHcommCcuResTypeToHcclResType(resType, ccuResType));

    int32_t userDevId = INVALID_INT;
    CCU_CHK_RET(HcclDeviceRefresh(userDevId));
    CCU_CHK_RET(
        hcomm::CcuInstanceMgr::GetInstance(userDevId).GetResDescMgr().QueryResNum(resDesc, ccuResType, *resNum));
    HCCL_INFO("[%s] success, handle[0x%llx] resType[%d] resNum[%u]", __func__, resDesc, resType, *resNum);
    return CcuResult::CCU_SUCCESS;
}

CcuResult HcommCcuInsResDescQueryDieId(HcommCcuResDescHandle resDesc, uint32_t* dieId)
{
    CCU_CHK_PTR_NULL(dieId);
    int32_t userDevId = INVALID_INT;
    CCU_CHK_RET(HcclDeviceRefresh(userDevId));
    CCU_CHK_RET(hcomm::CcuInstanceMgr::GetInstance(userDevId).GetResDescMgr().QueryDieId(resDesc, *dieId));
    HCCL_INFO("[%s] success, handle[0x%llx] dieId[%u]", __func__, resDesc, *dieId);
    return CcuResult::CCU_SUCCESS;
}

CcuResult HcommCcuQueryRemainResDesc(HcommCcuResDescHandle resDesc)
{
    int32_t userDevId = INVALID_INT;
    CCU_CHK_RET(HcclDeviceRefresh(userDevId));

    // CCU 驱动未拉起时无法查询硬件资源, 提前返回
    if (!hcomm::CcuIsInited(userDevId)) {
        HCCL_WARNING("[%s] failed, CCU feature is not inited, userDevId[%d].", __func__, userDevId);
        return CcuResult::CCU_E_UNAVAIL;
    }

    auto& resDescMgr = hcomm::CcuInstanceMgr::GetInstance(userDevId).GetResDescMgr();

    // die 合法性前置校验
    uint32_t dieId = 0;
    CCU_CHK_RET(resDescMgr.QueryDieId(resDesc, dieId));
    bool enableFlag = false;
    // CcuGetDieEnableInfo 中会做dieId合法性校验以及查询是否使能
    CCU_CHK_RET(hcomm::CcuGetDieEnableInfo(userDevId, static_cast<uint8_t>(dieId), enableFlag));
    if (!enableFlag) {
        HCCL_WARNING("[%s] failed, dieId[%u] is not enabled.", __func__, dieId);
        return CcuResult::CCU_E_UNAVAIL;
    }

    // 委托 CcuResDescMgr 在锁内查询剩余资源，防止 Get→Destroy 的 use-after-free
    CCU_CHK_RET(resDescMgr.QueryRemainRes(resDesc, userDevId));

    HCCL_INFO("[%s] success, handle[0x%llx] dieId[%u]", __func__, resDesc, dieId);
    return CcuResult::CCU_SUCCESS;
}

static CcuResult ConvertCcuResReqToResDesc(
    hcomm::CcuResDescMgr& resDescMgr, HcommCcuResDescHandle resDesc, const hcomm::CcuResReq& resReq,
    uint32_t instrCount, uint32_t selectedDie)
{
    const uint32_t loopNum = resReq.loopEngineReq[selectedDie] + resReq.blockLoopEngineReq[selectedDie];
    const uint32_t msNum = resReq.msReq[selectedDie] + resReq.blockMsReq[selectedDie];
    const uint32_t xnNum = resReq.xnReq[selectedDie] + resReq.blockXnReq[selectedDie];
    const uint32_t gsaNum = resReq.gsaReq[selectedDie] + resReq.blockGsaReq[selectedDie];
    const uint32_t ckeNum = resReq.ckeReq[selectedDie] + resReq.blockCkeReq[selectedDie];
    const uint32_t missionNum = resReq.missionReq.req[selectedDie];

    CCU_CHK_RET(resDescMgr.SetResNum(resDesc, hcomm::ResType::LOOP, loopNum));
    CCU_CHK_RET(resDescMgr.SetResNum(resDesc, hcomm::ResType::MS, msNum));
    CCU_CHK_RET(resDescMgr.SetResNum(resDesc, hcomm::ResType::XN, xnNum));
    CCU_CHK_RET(resDescMgr.SetResNum(resDesc, hcomm::ResType::GSA, gsaNum));
    CCU_CHK_RET(resDescMgr.SetResNum(resDesc, hcomm::ResType::CKE, ckeNum));
    CCU_CHK_RET(resDescMgr.SetResNum(resDesc, hcomm::ResType::MISSION, missionNum));
    CCU_CHK_RET(resDescMgr.SetResNum(resDesc, hcomm::ResType::INS, instrCount));

    HCCL_INFO(
        "[HcommCcuKernelQueryResReq] success, aggregated resource request, not allocated resource, "
        "resDesc[0x%llx], dieId[%u], loop[%u], ms[%u], xn[%u], gsa[%u], cke[%u], mission[%u], ins[%u].",
        static_cast<unsigned long long>(resDesc), selectedDie, loopNum, msNum, xnNum, gsaNum, ckeNum, missionNum,
        instrCount);
    return CcuResult::CCU_SUCCESS;
}

CcuResult HcommCcuKernelQueryResReq(
    const void* kernelFunc, const void** kernelArgs, uint32_t argNum, HcommCcuResDescHandle resDesc)
{
    HCCL_INFO(
        "[%s] begin, argNum[%u], resDesc[0x%llx], kernelFunc[%p].", __func__, argNum,
        static_cast<unsigned long long>(resDesc), kernelFunc);
    CCU_CHK_PTR_NULL(kernelFunc);
    if (resDesc == 0 || argNum > 1) {
        HCCL_ERROR(
            "[%s] failed, resDesc[0x%llx], argNum[%u].", __func__, static_cast<unsigned long long>(resDesc), argNum);
        return CcuResult::CCU_E_PARA;
    }
    if (argNum == 1) {
        CHK_PRT_RET(
            kernelArgs == nullptr, HCCL_ERROR("[%s] failed, kernelArgs is nullptr while argNum[%u].", __func__, argNum),
            CcuResult::CCU_E_PTR);
        CHK_PRT_RET(
            kernelArgs[0] == nullptr,
            HCCL_ERROR("[%s] failed, kernelArgs[0] is nullptr while argNum[%u].", __func__, argNum),
            CcuResult::CCU_E_PTR);
    }

    int32_t userDevId = INVALID_INT;
    CCU_CHK_RET(HcclDeviceRefresh(userDevId));
    auto& resDescMgr = hcomm::CcuInstanceMgr::GetInstance(userDevId).GetResDescMgr();
    uint32_t descriptorDieId = hcomm::CCU_MAX_IODIE_NUM;
    CCU_CHK_RET(resDescMgr.QueryDieId(resDesc, descriptorDieId));
    HCCL_INFO(
        "[%s] descriptor queried, resDesc[0x%llx], descriptorDieId[%u].", __func__,
        static_cast<unsigned long long>(resDesc), descriptorDieId);
    if (descriptorDieId >= hcomm::CCU_MAX_IODIE_NUM) {
        HCCL_ERROR("[%s] failed, descriptor dieId[%u] is invalid.", __func__, descriptorDieId);
        return CcuResult::CCU_E_PARA;
    }

    CCU_EXCEPTION_HANDLE_BEGIN
    hcomm::CcuResReq resReq{};
    uint32_t instrCount = 0;
    auto& kernelMgr = hcomm::CcuKernelMgr::GetInstance(userDevId);
    CCU_CHK_RET(kernelMgr.GetKernelResourceRequest(
        descriptorDieId, "KernelForHcommCcuKernelQueryResReq", kernelFunc, kernelArgs, argNum, resReq, instrCount));
    CCU_CHK_RET(ConvertCcuResReqToResDesc(resDescMgr, resDesc, resReq, instrCount, descriptorDieId));
    CCU_EXCEPTION_HANDLE_END

    return CcuResult::CCU_SUCCESS;
}

// 按 CCU 实例类型创建 ccu 实例（兼容 hccl 旧版本）
CcuResult HcommCcuInsCreateLegacy(const CcuInstanceType insType, CcuInsHandle* ccuInsHandle)
{
    CCU_CHK_PTR_NULL(ccuInsHandle);

    int32_t userDevId = INVALID_INT;
    CCU_CHK_RET(HcclDeviceRefresh(userDevId));
    auto& insMgr = hcomm::CcuInstanceMgr::GetInstance(userDevId);

    CCU_CHK_RET(insMgr.CreateByInsType(insType, *ccuInsHandle));

    return CcuResult::CCU_SUCCESS;
}

CcuResult HcommCcuInsCreate(const HcommCcuResDescHandle* resDescs, uint32_t resDescNum, CcuInsHandle* ccuInsHandle)
{
    CCU_CHK_PTR_NULL(resDescs);
    CCU_CHK_PTR_NULL(ccuInsHandle);
    if (resDescNum == 0 || resDescNum > hcomm::CCU_MAX_IODIE_NUM) {
        HCCL_ERROR(
            "[%s] failed, resDescNum[%u] is invalid, should be in (0, %u].", __func__, resDescNum,
            hcomm::CCU_MAX_IODIE_NUM);
        return CcuResult::CCU_E_PARA;
    }

    int32_t userDevId = INVALID_INT;
    CCU_CHK_RET(HcclDeviceRefresh(userDevId));
    auto& insMgr = hcomm::CcuInstanceMgr::GetInstance(userDevId);

    // 从 CcuResDescMgr 中用 resDesc 数组里的 Handle Get 获得 CcuResDesc，构造指针数组
    std::array<const hcomm::CcuResDesc*, hcomm::CCU_MAX_IODIE_NUM> descPtrs{};
    for (uint32_t i = 0; i < resDescNum; i++) {
        descPtrs[i] = insMgr.GetResDescMgr().Get(resDescs[i]);
        CCU_CHK_PTR_NULL(descPtrs[i]);
    }

    // 如果入参为 2 个 resDesc，里面的 dieId 不能重复
    if (resDescNum == 2 && descPtrs[0]->dieId == descPtrs[1]->dieId) {
        HCCL_ERROR("[%s] failed, dieId[%u] duplicated in resDescs.", __func__, descPtrs[0]->dieId);
        return CcuResult::CCU_E_PARA;
    }

    CCU_CHK_RET(insMgr.CreateByResDescs(descPtrs.data(), resDescNum, *ccuInsHandle));
    return CcuResult::CCU_SUCCESS;
}

CcuResult HcommCcuInsCreateDefault(const uint32_t* dieIds, uint32_t dieNum, CcuInsHandle* ccuInsHandle)
{
    (void)dieIds;
    (void)dieNum;
    // dieIds/dieNum 为保留参数，当前版本申请当前 Device 上所有已使能 ioDie 的全部资源
    CCU_CHK_PTR_NULL(ccuInsHandle);

    int32_t userDevId = INVALID_INT;
    CCU_CHK_RET(HcclDeviceRefresh(userDevId));
    CCU_CHK_RET(hcomm::CcuInstanceMgr::GetInstance(userDevId).CreateByAllRes(*ccuInsHandle));
    return CcuResult::CCU_SUCCESS;
}

CcuResult HcommCcuInsQueryResDesc(CcuInsHandle ccuInsHandle, HcommCcuResDescHandle resDesc)
{
    if (ccuInsHandle == 0 || resDesc == 0) {
        HCCL_ERROR("[%s] failed, invalid ccuInsHandle[%llu] resDesc[%llu].", __func__, ccuInsHandle, resDesc);
        return CcuResult::CCU_E_PARA;
    }

    int32_t userDevId = INVALID_INT;
    CCU_CHK_RET(HcclDeviceRefresh(userDevId));
    auto& insMgr = hcomm::CcuInstanceMgr::GetInstance(userDevId);

    auto& resDescMgr = insMgr.GetResDescMgr();
    uint32_t descriptorDieId = hcomm::CCU_MAX_IODIE_NUM;
    CCU_CHK_RET(resDescMgr.QueryDieId(resDesc, descriptorDieId));
    const uint8_t dieId = static_cast<uint8_t>(descriptorDieId);
    if (dieId >= hcomm::CCU_MAX_IODIE_NUM) {
        HCCL_ERROR("[%s] failed, dieId[%u] is invalid.", __func__, dieId);
        return CcuResult::CCU_E_PARA;
    }

    CCU_CHK_RET(insMgr.QueryInsResDesc(ccuInsHandle, dieId, resDesc));

    return CcuResult::CCU_SUCCESS;
}

CcuResult HcommCcuInsDestroy(CcuInsHandle ccuInsHandle)
{
    int32_t userDevId = INVALID_INT;
    CCU_CHK_RET(HcclDeviceRefresh(userDevId));
    CCU_CHK_RET(hcomm::CcuInstanceMgr::GetInstance(userDevId).Destroy(ccuInsHandle));

    return CcuResult::CCU_SUCCESS;
}

CcuResult HcommCcuCascCntAlloc(CcuInsHandle ccuInsHandle, uint8_t dieId, HcommCcuCascCntHandle* handle)
{
    // 输出指针先校验：接口约定失败时 handle 置 0，须在一切可能失败的分支之前完成清零，
    // 因此其校验先于 insHandle/dieId 的值校验；成功时由 Create 写入有效句柄覆盖。
    CCU_CHK_PTR_NULL(handle);
    *handle = 0;

    if (ccuInsHandle == 0) {
        HCCL_ERROR("[%s] failed, invalid ccuInsHandle[%llu].", __func__, ccuInsHandle);
        return CcuResult::CCU_E_PARA;
    }
    if (dieId >= hcomm::CCU_MAX_IODIE_NUM) {
        HCCL_ERROR("[%s] dieId[%u] is invalid, dieId should in [0, %u).", __func__, dieId, hcomm::CCU_MAX_IODIE_NUM);
        return CcuResult::CCU_E_PARA;
    }

    int32_t devLogicId = INVALID_INT;
    CCU_CHK_RET(HcclDeviceRefresh(devLogicId));
    CCU_CHK_RET(hcomm::CcuInstanceMgr::GetInstance(devLogicId).CascCntHandleAlloc(ccuInsHandle, dieId, *handle));

    return CcuResult::CCU_SUCCESS;
}

CcuResult HcommCcuCascCntGetMem(HcommCcuCascCntHandle handle, CommMem* commMem)
{
    if (handle == 0) {
        HCCL_ERROR("[%s] failed, invalid ccuCascCntHandle[%llu].", __func__, handle);
        return CcuResult::CCU_E_PARA;
    }
    CCU_CHK_PTR_NULL(commMem);

    int32_t devLogicId = INVALID_INT;
    CCU_CHK_RET(HcclDeviceRefresh(devLogicId));
    hcomm::CntXnBlock cascCntBlock;
    CCU_CHK_RET(hcomm::CcuInstanceMgr::GetInstance(devLogicId).GetCascCntBlock(handle, cascCntBlock));

    commMem->addr = cascCntBlock.wishCntXnsMem.first;
    commMem->size = cascCntBlock.wishCntXnsMem.second;
    commMem->type = CommMemType::COMM_MEM_TYPE_CCU;

    return CcuResult::CCU_SUCCESS;
}
