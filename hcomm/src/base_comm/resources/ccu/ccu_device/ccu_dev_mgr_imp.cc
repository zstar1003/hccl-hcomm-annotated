/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_dev_mgr_imp.h"

#include "hccl_common.h"
#include "eid_info_mgr.h"

#include "ccu_comp.h"
#include "ccu_res_specs.h"
#include "ccu_res_batch_allocator.h"

// 支持ccu新老通信域混跑临时添加
#include "log.h"
#include "unified_platform/ccu/ccu_device/ccu_component/ccu_component.h"
#include "unified_platform/ccu/ccu_device/ccu_res_specs_legacy.h"
#include "unified_platform/ccu/ccu_device/ccu_res_batch_allocator_legacy.h"
#include "orion_adpt_utils.h"
#include "exception_handler.h"

/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * #include "ccu_comp.h"
 * #include "ccu_res_specs.h"
 * #include "ccu_res_batch_allocator.h"
 */

// 引入主板类型查询接口，后续应根据ccu驱动提供的信息用于判断
// 当前先简化修改
#include "./ccu_res_specs.h"
#include "adapter_rts.h"

#include "ccu_types.h"
#include "ccu_log.h"
#include "ccu_res_desc.h"

#include "dev_type.h"

namespace hcomm {

static std::unordered_map<int32_t, std::shared_ptr<CcuDrvHandle>> ccuDrvHandleMap;
static std::mutex ccuDrvHandleMutex;
static bool ccuDriverInitAgainFlag = false; // 记录每个进程CCU驱动是否重复拉起
static thread_local Hccl::HcclMainboardId mainBoardType = Hccl::HcclMainboardId::MAINBOARD_OTHERS; // 记录本卡的主板类型

inline bool CheckCcuOpenSourceEnable()
{
    // A6 不支持legacy ccu mc2，可以完全切换至开源流程
    auto devType = DevType::DEV_TYPE_COUNT;
    (void)hrtGetDeviceType(devType);
    return devType == DevType::DEV_TYPE_960;
}

CcuResult CcuInitFeature(const int32_t userDevId, std::shared_ptr<CcuDrvHandle>& ccuDrvHandle)
{
    if (userDevId >= static_cast<int32_t>(MAX_MODULE_DEVICE_NUM)) {
        HCCL_ERROR(
            "[%s] failed, userDevId[%d] is too large, should be less than %u.", __func__, userDevId,
            MAX_MODULE_DEVICE_NUM);
        return CcuResult::CCU_E_PARA;
    }

    std::lock_guard<std::mutex> lock(ccuDrvHandleMutex);
    // ccu驱动已重复拉起失败时，直接返回，在锁保护内返回
    if (ccuDriverInitAgainFlag) {
        return CcuResult::CCU_E_DRV_BUSY;
    }

    auto iter = ccuDrvHandleMap.find(userDevId);
    if (iter != ccuDrvHandleMap.end()) {
        ccuDrvHandle = iter->second;
        HCCL_RUN_INFO("[%s] userDevId[%d] init ccu feature, handle[%p].", __func__, userDevId, ccuDrvHandle.get());
        return CcuResult::CCU_SUCCESS;
    }

    std::shared_ptr<CcuDrvHandle> drvHandle = nullptr;
    drvHandle.reset(new (std::nothrow) CcuDrvHandle(userDevId));
    CCU_CHK_PTR_NULL(drvHandle);

    auto ret = drvHandle->Init();
    if (ret == CcuResult::CCU_E_DRV_BUSY) {
        HCCL_RUN_WARNING(
            "[%s] failed but passed, userDevId[%d] ccu driver has been "
            "inited by another process, this process will not try to init anymore.",
            __func__, userDevId);
        ccuDriverInitAgainFlag = true; // 记录该进程ccu驱动已拉起失败
        drvHandle = nullptr;           // 主动置空触发资源销毁，控制释放时序
        return ret;
    }
    CCU_CHK_RET(ret);

    ccuDrvHandleMap[userDevId] = drvHandle;
    ccuDrvHandle = ccuDrvHandleMap[userDevId];
    HCCL_RUN_INFO("[%s] userDevId[%d] init ccu feature, handle[%p].", __func__, userDevId, ccuDrvHandle.get());
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuDeinitFeature(const int32_t userDevId)
{
    std::lock_guard<std::mutex> lock(ccuDrvHandleMutex);
    auto iter = ccuDrvHandleMap.find(userDevId);
    if (iter == ccuDrvHandleMap.end()) {
        HCCL_INFO("[%s] passed, ccu feature was not inited, userDevId[%d].", __func__, userDevId);
        return CcuResult::CCU_SUCCESS;
    }

    auto& ccuDrvHandle = ccuDrvHandleMap[userDevId];
    if (ccuDrvHandle.use_count() == 1) {
        HCCL_RUN_INFO(
            "[%s] entry, start to deinit ccu feature, "
            "handle[%p] userDevId[%d].",
            __func__, ccuDrvHandle.get(), userDevId);
        ccuDrvHandle = nullptr;
        ccuDrvHandleMap.erase(userDevId);
    }

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuGetDieEnableInfo(int32_t userDevId, uint8_t dieId, bool& enableFlag)
{
    CHK_PRT_RET(
        dieId >= CCU_MAX_IODIE_NUM,
        HCCL_ERROR(
            "[%s] failed, dieId[%u] is invalid, should be in [0-%u), userDevId[%d].", __func__, dieId,
            CCU_MAX_IODIE_NUM, userDevId),
        CcuResult::CCU_E_PARA);

    const auto& dieEnableFlags = CheckCcuOpenSourceEnable() ?
                                     CcuComponent::GetInstance(userDevId).GetDieEnableFlags() :
                                     Hccl::CcuComponent::GetInstance(userDevId).GetDieEnableFlags();

    enableFlag = dieEnableFlags[dieId];
    return CcuResult::CCU_SUCCESS;
}

// 查询指定 die 上各资源类型可分配的总量
CcuResult CcuGetLoopEngineNum(int32_t userDevId, uint8_t dieId, uint32_t& num)
{
    CCU_CHK_RET(CcuDevMgrImp::GetAllocatableMaxLoopEngineNum(userDevId, dieId, num));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuGetMsNum(int32_t userDevId, uint8_t dieId, uint32_t& num)
{
    CCU_CHK_RET(CcuDevMgrImp::GetAllocatableMaxMsNum(userDevId, dieId, num));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuGetCkeNum(int32_t userDevId, uint8_t dieId, uint32_t& num)
{
    CCU_CHK_RET(CcuDevMgrImp::GetAllocatableMaxCkeNum(userDevId, dieId, num));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuGetXnNum(int32_t userDevId, uint8_t dieId, uint32_t& num)
{
    CCU_CHK_RET(CcuDevMgrImp::GetAllocatableMaxXnNum(userDevId, dieId, num));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuGetGsaNum(int32_t userDevId, uint8_t dieId, uint32_t& num)
{
    CCU_CHK_RET(CcuDevMgrImp::GetAllocatableMaxGsaNum(userDevId, dieId, num));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuGetInstructionNum(int32_t userDevId, uint8_t dieId, uint32_t& num)
{
    CCU_CHK_RET(CcuDevMgrImp::GetResSpecsInstructionNum(userDevId, dieId, num));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuGetMissionNum(int32_t userDevId, uint8_t dieId, uint32_t& num)
{
    CCU_CHK_RET(CcuDevMgrImp::GetResSpecsMissionNum(userDevId, dieId, num));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuGetCascCntNum(int32_t userDevId, uint8_t dieId, uint32_t& num)
{
    CCU_CHK_RET(CcuDevMgrImp::GetResSpecsCascCntNum(userDevId, dieId, num));
    return CcuResult::CCU_SUCCESS;
}

HcclResult CcuGetMainboardType(uint32_t userDevId, Hccl::HcclMainboardId& hcclMainboardId)
{
    CHK_RET(CcuGetMainboardId(userDevId, hcclMainboardId));
    return HcclResult::HCCL_SUCCESS;
}

// 单个描述符的资源数量映射到 CcuResReq 的 block 字段
static CcuResult FillResReqByResDesc(CcuResReq& resReq, uint8_t dieId, const CcuResDesc& desc)
{
    uint32_t num = 0;
    CCU_CHK_RET(desc.QueryResNum(ResType::LOOP, num));
    resReq.blockLoopEngineReq[dieId] = num;
    CCU_CHK_RET(desc.QueryResNum(ResType::MS, num));
    resReq.blockMsReq[dieId] = num;
    CCU_CHK_RET(desc.QueryResNum(ResType::CKE, num));
    resReq.blockCkeReq[dieId] = num;
    CCU_CHK_RET(desc.QueryResNum(ResType::XN, num));
    resReq.blockXnReq[dieId] = num;
    CCU_CHK_RET(desc.QueryResNum(ResType::GSA, num));
    resReq.blockGsaReq[dieId] = num;

    resReq.loopEngineReq[dieId] = 0;
    resReq.msReq[dieId] = 0;
    resReq.ckeReq[dieId] = 0;
    resReq.xnReq[dieId] = 0;
    resReq.gsaReq[dieId] = 0;
    return CcuResult::CCU_SUCCESS;
}

// 查询各 die 是否启用；若全部未启用则返回错误
static CcuResult CheckEnabledDies(int32_t userDevId, std::array<bool, CCU_MAX_IODIE_NUM>& dieEnableFlags)
{
    dieEnableFlags = {false, false};
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        CCU_CHK_RET(CcuGetDieEnableInfo(userDevId, dieId, dieEnableFlags[dieId]));
    }

    if (!dieEnableFlags[0] && !dieEnableFlags[1]) {
        HCCL_ERROR("[%s] failed, all ccu dies are disable, userDevId[%d].", __func__, userDevId);
        return CcuResult::CCU_E_INTERNAL;
    }
    return CcuResult::CCU_SUCCESS;
}

// 根据资源描述符数组构造 CcuResReq：
// 跳过未启用 die；block 字段由各 die 的 resDesc 填充；
// missionReq 取所有 die 的最大值，再统一回填到各启用 die
static CcuResult BuildResReqByDescs(
    const CcuResDesc* descs[], uint32_t descNum, const std::array<bool, CCU_MAX_IODIE_NUM>& dieEnableFlags,
    int32_t userDevId, CcuResReq& resReq)
{
    resReq = CcuResReq{};
    resReq.missionReq.reqType = MissionReqType::FUSION_MULTIPLE_DIE;

    uint32_t maxMissionReq = 0;
    for (uint32_t i = 0; i < descNum; i++) {
        if (descs[i] == nullptr) {
            HCCL_ERROR("[%s] failed, descs[%u] is nullptr, userDevId[%d].", __func__, i, userDevId);
            return CcuResult::CCU_E_PARA;
        }
        uint8_t dieId = 0;
        dieId = static_cast<uint8_t>(descs[i]->dieId);
        if (dieId >= CCU_MAX_IODIE_NUM) {
            HCCL_ERROR("[%s] failed, dieId[%u] is invalid, userDevId[%d].", __func__, dieId, userDevId);
            return CcuResult::CCU_E_PARA;
        }

        // 不跳过未启用的die，如果die未启用，但请求资源，在分配资源时检查返回错误
        CCU_CHK_RET(FillResReqByResDesc(resReq, dieId, *descs[i]));
        uint32_t missionNum = 0;
        CCU_CHK_RET(descs[i]->QueryResNum(ResType::MISSION, missionNum));
        if (missionNum > maxMissionReq) {
            maxMissionReq = missionNum;
        }
    }

    if (maxMissionReq > 0) {
        for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
            if (dieEnableFlags[dieId]) {
                resReq.missionReq.req[dieId] = maxMissionReq;
            }
        }
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult
CcuAllocResHandleByResDescs(int32_t userDevId, const CcuResDesc* descs[], uint32_t descNum, CcuResHandle& resHandle)
{
    if (descs == nullptr || descNum == 0 || descNum > hcomm::CCU_MAX_IODIE_NUM) {
        HCCL_ERROR("[%s] failed, invalid descs[%p] descNum[%u], userDevId[%d].", __func__, descs, descNum, userDevId);
        return CcuResult::CCU_E_PARA;
    }

    std::array<bool, CCU_MAX_IODIE_NUM> dieEnableFlags = {false, false};
    CCU_CHK_RET(CheckEnabledDies(userDevId, dieEnableFlags));

    CcuResReq resReq{};
    CCU_CHK_RET(BuildResReqByDescs(descs, descNum, dieEnableFlags, userDevId, resReq));

    if (mainBoardType == Hccl::HcclMainboardId::MAINBOARD_OTHERS) {
        CCU_CHK_RET(CcuGetMainboardId(userDevId, mainBoardType));
    }

    CCU_CHK_RET(CcuDevMgrImp::AllocResHandle(userDevId, resReq, resHandle));

    HCCL_INFO(
        "[%s] succeed, get res handle[%llx], userDevId[%d], descNum[%u]", __func__, resHandle, userDevId, descNum);
    return CcuResult::CCU_SUCCESS;
}

constexpr u32 CCU_MS_DEFAULT_BLOCK_LOOP_ENGINE_REQ = 8 * 8 * 2;
constexpr u32 CCU_MS_DEFAULT_BLOCK_MS_REQ = 64 * 8 * 2;
constexpr u32 CCU_MS_DEFAULT_BLOCK_CKE_REQ = 32 + 8 * 8 * 2;
constexpr u32 CCU_MS_DEFAULT_BLOCK_XN_REQ = 400;
constexpr u32 CCU_MS_DEFAULT_GSA_REQ = 400;
constexpr u32 CCU_MS_DEFAULT_MISSIONREQ_REQ = 2;
inline void ConfigCcuResReqCcuMs(CcuResReq& resReq, uint8_t dieId, CcuVersion version)
{
    resReq.loopEngineReq[dieId] = 0;
    resReq.blockLoopEngineReq[dieId] = CCU_MS_DEFAULT_BLOCK_LOOP_ENGINE_REQ;
    resReq.msReq[dieId] = 0;
    resReq.blockMsReq[dieId] = CCU_MS_DEFAULT_BLOCK_MS_REQ;
    resReq.ckeReq[dieId] = 0;
    resReq.blockCkeReq[dieId] = CCU_MS_DEFAULT_BLOCK_CKE_REQ;
    resReq.xnReq[dieId] = 0;
    resReq.gsaReq[dieId] = 0;
    if (version == CcuVersion::CCU_V2) {
        resReq.blockXnReq[dieId] = CCU_MS_DEFAULT_BLOCK_XN_REQ * 2; // V2场景下申请2倍的Xn数量
        resReq.blockGsaReq[dieId] = 0;
    } else {
        resReq.blockXnReq[dieId] = CCU_MS_DEFAULT_BLOCK_XN_REQ;
        resReq.blockGsaReq[dieId] = CCU_MS_DEFAULT_GSA_REQ;
    }
    resReq.missionReq.reqType = MissionReqType::FUSION_MULTIPLE_DIE;
    resReq.missionReq.req[dieId] = CCU_MS_DEFAULT_MISSIONREQ_REQ;
}

constexpr u32 CCU_SCHED_DEFAULT_BLOCK_LOOP_ENGINE_REQ = 16;
constexpr u32 CCU_SCHED_DEFAULT_BLOCK_MS_REQ = 128;
constexpr u32 CCU_SCHED_DEFAULT_BLOCK_CKE_REQ = 32 + 16;
constexpr u32 CCU_SCHED_DEFAULT_BLOCK_XN_REQ = 400;
constexpr u32 CCU_SCHED_DEFAULT_GSA_REQ = 400;
constexpr u32 CCU_SCHED_DEFAULT_MISSIONREQ_REQ = 2;
inline void ConfigCcuResReqCcuSched(CcuResReq& resReq, uint8_t dieId, CcuVersion version)
{
    resReq.loopEngineReq[dieId] = 0;
    resReq.blockLoopEngineReq[dieId] = CCU_SCHED_DEFAULT_BLOCK_LOOP_ENGINE_REQ;
    resReq.msReq[dieId] = 0;
    resReq.blockMsReq[dieId] = CCU_SCHED_DEFAULT_BLOCK_MS_REQ;
    resReq.ckeReq[dieId] = 0;
    resReq.blockCkeReq[dieId] = CCU_SCHED_DEFAULT_BLOCK_CKE_REQ;
    resReq.xnReq[dieId] = 0;
    resReq.gsaReq[dieId] = 0;
    if (version == CcuVersion::CCU_V2) {
        resReq.blockXnReq[dieId] = CCU_SCHED_DEFAULT_BLOCK_XN_REQ * 2; // V2场景下申请2倍的Xn数量
        resReq.blockGsaReq[dieId] = 0;
    } else {
        resReq.blockXnReq[dieId] = CCU_SCHED_DEFAULT_BLOCK_XN_REQ;
        resReq.blockGsaReq[dieId] = CCU_SCHED_DEFAULT_GSA_REQ;
    }

    resReq.missionReq.reqType = MissionReqType::FUSION_MULTIPLE_DIE;
    resReq.missionReq.req[dieId] = CCU_SCHED_DEFAULT_MISSIONREQ_REQ;
}

// CCU设备管理对集合通信提供的接口
CcuResult CcuAllocResHandleByInsType(int32_t userDevId, CcuInstanceType ccuInsType, CcuResHandle& resHandle)
{
    if (ccuInsType >= CcuInstanceType::CCU_UNUSED) {
        HCCL_ERROR("[%s] failed, error ccu instance type[%d], userDevId[%d].", __func__, ccuInsType, userDevId);
        return CcuResult::CCU_E_PARA;
    }

    std::array<bool, CCU_MAX_IODIE_NUM> dieEnableFlags = {false, false};
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        CCU_CHK_RET(CcuGetDieEnableInfo(userDevId, dieId, dieEnableFlags[dieId]));
    }

    if (!dieEnableFlags[0] && !dieEnableFlags[1]) {
        HCCL_ERROR("[%s] failed, all ccu dies are disable, userDevId[%d].", __func__, userDevId);
        return CcuResult::CCU_E_INTERNAL;
    }

    CcuVersion ccuVersion = CcuVersion::INVALID;
    CCU_CHK_RET(CcuDevMgrImp::GetCcuVersion(userDevId, ccuVersion));
    if (ccuVersion == CcuVersion::INVALID) {
        HCCL_RUN_WARNING(
            "[%s] failed, userDevId[%d] ccu version is invalid, "
            "should fallback to aicpu.",
            __func__, userDevId);
        return CcuResult::CCU_E_UNAVAIL;
    }

    CcuResReq resReq{};
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        if (!dieEnableFlags[dieId]) {
            continue;
        }

        if (ccuInsType == CcuInstanceType::CCU_MS) {
            ConfigCcuResReqCcuMs(resReq, dieId, ccuVersion);
        } else {
            ConfigCcuResReqCcuSched(resReq, dieId, ccuVersion);
        }
    }

    if (mainBoardType == Hccl::HcclMainboardId::MAINBOARD_OTHERS) {
        CCU_CHK_RET(CcuGetMainboardId(userDevId, mainBoardType));
    }

    if (mainBoardType == Hccl::HcclMainboardId::MAINBOARD_PCIE_STD
        && ccuInsType == CcuInstanceType::CCU_MS) { // 标卡环境下配置CCU_MS拦截报错
        HCCL_ERROR(
            "[%s] ccuInstanceType[%d] not support in %s", __func__, ccuInsType, mainBoardType.Describe().c_str());
        return CcuResult::CCU_E_NOT_SUPPORT;
    }

    CCU_CHK_RET(CcuDevMgrImp::AllocResHandle(userDevId, resReq, resHandle));

    HCCL_INFO("[%s] succeed, get res handle[%p], userDevId[%d]", __func__, resHandle, userDevId);
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuCheckResource(const int32_t userDevId, const CcuResHandle resHandle, CcuResRepository& resRepo)
{
    CCU_CHK_RET(CcuDevMgrImp::GetResource(userDevId, resHandle, resRepo));
    return CcuResult::CCU_SUCCESS;
}

HcclResult CcuReleaseResHandle(const int32_t userDevId, const CcuResHandle resHandle)
{
    CHK_RET(CcuDevMgrImp::ReleaseResHandle(userDevId, resHandle));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuAllocChannels(
    const int32_t userDevId, const CcuChannelPara& ccuChannelPara, std::vector<CcuChannelInfo>& ccuChannelInfos)
{
    Hccl::IpAddress ipAddr{};
    CHK_RET(CommAddrToIpAddress(ccuChannelPara.commAddr, ipAddr)); // 为了打印信息暂时添加
    HCCL_INFO(
        "[%s] new allocation request: userDevId[%d], ipAddr[%s], "
        "channelnum[%u], jettyNum[%u], sqSize[%u].",
        __func__, userDevId, ipAddr.Describe().c_str(), ccuChannelPara.channelNum, ccuChannelPara.jettyNum,
        ccuChannelPara.sqSize);

    uint32_t devPhyId{0};
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(userDevId), devPhyId));

    DevEidInfo eidInfo{};
    CHK_RET(EidInfoMgr::GetInstance(devPhyId).GetEidInfoByAddr(ccuChannelPara.commAddr, eidInfo));
    const uint8_t dieId = static_cast<uint8_t>(eidInfo.dieId);
    const uint32_t feId = eidInfo.funcId;
    ChannelPara para{};
    para.feId = feId;
    para.jettyNum = ccuChannelPara.jettyNum;
    para.sqSize = ccuChannelPara.sqSize;

    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ?
              CcuComponent::GetInstance(userDevId).AllocChannels(dieId, para, ccuChannelInfos) :
              Hccl::CcuComponent::GetInstance(userDevId).AllocChannels(dieId, para, ccuChannelInfos);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuReleaseChannel(const int32_t userDevId, const uint8_t dieId, const uint32_t ccuChannelId)
{
    HCCL_INFO(
        "[%s] new release request: userDevId[%d], dieId[%u], "
        "ccuChannelId[%u].",
        __func__, userDevId, dieId, ccuChannelId);

    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ? CcuComponent::GetInstance(userDevId).ReleaseChannel(dieId, ccuChannelId) :
                                       Hccl::CcuComponent::GetInstance(userDevId).ReleaseChannel(dieId, ccuChannelId);
    EXCEPTION_HANDLE_END
    return ret;
}

// 以下为hcomm基础通信内部CCU流程使用的接口
HcclResult CcuDevMgrImp::GetCcuVersion(const int32_t userDevId, CcuVersion& ccuVersion)
{
    EXCEPTION_HANDLE_BEGIN
    ccuVersion = CheckCcuOpenSourceEnable() ? CcuResSpecifications::GetInstance(userDevId).GetCcuVersion() :
                                              Hccl::CcuResSpecifications::GetInstance(userDevId).GetCcuVersion();
    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CcuDevMgrImp::GetCcuResourceSpaceBufInfo(const int32_t userDevId, const uint8_t dieId, uint64_t& addr, uint64_t& size)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ?
              CcuComponent::GetInstance(userDevId).GetCcuResourceSpaceBufInfo(dieId, addr, size) :
              Hccl::CcuComponent::GetInstance(userDevId).GetCcuResourceSpaceBufInfo(dieId, addr, size);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetCcuResourceSpaceTokenInfo(
    const int32_t userDevId, const uint8_t dieId, uint64_t& tokenId, uint64_t& tokenValue)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ?
              CcuComponent::GetInstance(userDevId).GetCcuResourceSpaceTokenInfo(dieId, tokenId, tokenValue) :
              Hccl::CcuComponent::GetInstance(userDevId).GetCcuResourceSpaceTokenInfo(dieId, tokenId, tokenValue);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::ConfigChannel(const int32_t userDevId, const uint8_t dieId, ChannelCfg& cfg)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ? CcuComponent::GetInstance(userDevId).ConfigChannel(dieId, cfg) :
                                       Hccl::CcuComponent::GetInstance(userDevId).ConfigChannel(dieId, cfg);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetLoopChannelId(
    const int32_t userDevId, const uint8_t srcDieId, const uint8_t dstDieId, uint32_t& channIdx)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ?
              CcuComponent::GetInstance(userDevId).GetLoopChannelId(srcDieId, dstDieId, channIdx) :
              Hccl::CcuComponent::GetInstance(userDevId).GetLoopChannelId(srcDieId, dstDieId, channIdx);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult
CcuDevMgrImp::GetResource(const int32_t userDevId, const CcuResHandle resHandle, CcuResRepository& ccuResRepo)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ?
              CcuResBatchAllocator::GetInstance(userDevId).GetResource(resHandle, ccuResRepo) :
              Hccl::CcuResBatchAllocator::GetInstance(userDevId).GetResource(resHandle, ccuResRepo);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::AllocResHandle(const int32_t userDevId, const CcuResReq resReq, CcuResHandle& resHandle)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ?
              CcuResBatchAllocator::GetInstance(userDevId).AllocResHandle(resReq, resHandle) :
              Hccl::CcuResBatchAllocator::GetInstance(userDevId).AllocResHandle(resReq, resHandle);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::ReleaseResHandle(const int32_t userDevId, const CcuResHandle handle)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ? CcuResBatchAllocator::GetInstance(userDevId).ReleaseResHandle(handle) :
                                       Hccl::CcuResBatchAllocator::GetInstance(userDevId).ReleaseResHandle(handle);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::QueryRemainRes(
    const int32_t userDevId, const uint8_t dieId, const ResType& internalType, uint32_t& remainNum)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ?
              CcuResBatchAllocator::GetInstance(userDevId).QueryRemainRes(dieId, internalType, remainNum) :
              Hccl::CcuResBatchAllocator::GetInstance(userDevId).QueryRemainRes(dieId, internalType, remainNum);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::AllocIns(const int32_t userDevId, const uint8_t dieId, const uint32_t num, ResInfo& insInfo)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ? CcuComponent::GetInstance(userDevId).AllocIns(dieId, num, insInfo) :
                                       Hccl::CcuComponent::GetInstance(userDevId).AllocIns(dieId, num, insInfo);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::ReleaseIns(const int32_t userDevId, const uint8_t dieId, const ResInfo& insInfo)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ? CcuComponent::GetInstance(userDevId).ReleaseIns(dieId, insInfo) :
                                       Hccl::CcuComponent::GetInstance(userDevId).ReleaseIns(dieId, insInfo);
    EXCEPTION_HANDLE_END
    return ret;
}

uint32_t CcuDevMgrImp::GetInsConsecutiveRemainSize(const int32_t userDevId, const uint8_t dieId)
{
    return CheckCcuOpenSourceEnable() ? CcuComponent::GetInstance(userDevId).GetInsConsecutiveRemainSize(dieId) :
                                        Hccl::CcuComponent::GetInstance(userDevId).GetInsConsecutiveRemainSize(dieId);
}

uint32_t CcuDevMgrImp::GetCascCntBlockRemainSize(const int32_t userDevId, const uint8_t dieId)
{
    return CheckCcuOpenSourceEnable() ? CcuComponent::GetInstance(userDevId).GetCascCntBlockRemainSize(dieId) : 0;
}

HcclResult
CcuDevMgrImp::AllocCke(const int32_t userDevId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& ckeInfos)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ? CcuComponent::GetInstance(userDevId).AllocCke(dieId, num, ckeInfos) :
                                       Hccl::CcuComponent::GetInstance(userDevId).AllocCke(dieId, num, ckeInfos);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::ReleaseCke(const int32_t userDevId, const uint8_t dieId, const std::vector<ResInfo>& ckeInfos)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ? CcuComponent::GetInstance(userDevId).ReleaseCke(dieId, ckeInfos) :
                                       Hccl::CcuComponent::GetInstance(userDevId).ReleaseCke(dieId, ckeInfos);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult
CcuDevMgrImp::AllocXn(const int32_t userDevId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& xnInfos)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ? CcuComponent::GetInstance(userDevId).AllocXn(dieId, num, xnInfos) :
                                       Hccl::CcuComponent::GetInstance(userDevId).AllocXn(dieId, num, xnInfos);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::ReleaseXn(const int32_t userDevId, const uint8_t dieId, const std::vector<ResInfo>& xnInfos)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ? CcuComponent::GetInstance(userDevId).ReleaseXn(dieId, xnInfos) :
                                       Hccl::CcuComponent::GetInstance(userDevId).ReleaseXn(dieId, xnInfos);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetMissionKey(const int32_t userDevId, const uint8_t dieId, uint32_t& missionKey)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ?
              CcuResSpecifications::GetInstance(userDevId).GetMissionKey(dieId, missionKey) :
              Hccl::CcuResSpecifications::GetInstance(userDevId).GetMissionKey(dieId, missionKey);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetResSpecsInstructionNum(const int32_t userDevId, const uint8_t dieId, uint32_t& instrNum)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ?
              CcuResSpecifications::GetInstance(userDevId).GetInstructionNum(dieId, instrNum) :
              Hccl::CcuResSpecifications::GetInstance(userDevId).GetInstructionNum(dieId, instrNum);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetAllocatableMaxLoopEngineNum(const int32_t userDevId, const uint8_t dieId, uint32_t& loopNum)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ?
              CcuResBatchAllocator::GetInstance(userDevId).GetAllocatableMaxBlockResNum(ResType::LOOP, dieId, loopNum) :
              Hccl::CcuResBatchAllocator::GetInstance(userDevId).GetAllocatableMaxBlockResNum(
                  ResType::LOOP, dieId, loopNum);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetAllocatableMaxMsNum(const int32_t userDevId, const uint8_t dieId, uint32_t& msNum)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ?
              CcuResBatchAllocator::GetInstance(userDevId).GetAllocatableMaxBlockResNum(ResType::MS, dieId, msNum) :
              Hccl::CcuResBatchAllocator::GetInstance(userDevId).GetAllocatableMaxBlockResNum(
                  ResType::MS, dieId, msNum);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetAllocatableMaxCkeNum(const int32_t userDevId, const uint8_t dieId, uint32_t& ckeNum)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ?
              CcuResBatchAllocator::GetInstance(userDevId).GetAllocatableMaxBlockResNum(ResType::CKE, dieId, ckeNum) :
              Hccl::CcuResBatchAllocator::GetInstance(userDevId).GetAllocatableMaxBlockResNum(
                  ResType::CKE, dieId, ckeNum);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetAllocatableMaxXnNum(const int32_t userDevId, const uint8_t dieId, uint32_t& xnNum)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ?
              CcuResBatchAllocator::GetInstance(userDevId).GetAllocatableMaxBlockResNum(ResType::XN, dieId, xnNum) :
              Hccl::CcuResBatchAllocator::GetInstance(userDevId).GetAllocatableMaxBlockResNum(
                  ResType::XN, dieId, xnNum);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetAllocatableMaxGsaNum(const int32_t userDevId, const uint8_t dieId, uint32_t& gsaNum)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ?
              CcuResBatchAllocator::GetInstance(userDevId).GetAllocatableMaxBlockResNum(ResType::GSA, dieId, gsaNum) :
              Hccl::CcuResBatchAllocator::GetInstance(userDevId).GetAllocatableMaxBlockResNum(
                  ResType::GSA, dieId, gsaNum);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetResSpecsMissionNum(const int32_t userDevId, const uint8_t dieId, uint32_t& missionNum)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ?
              CcuResSpecifications::GetInstance(userDevId).GetMissionNum(dieId, missionNum) :
              Hccl::CcuResSpecifications::GetInstance(userDevId).GetMissionNum(dieId, missionNum);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetXnBaseAddr(const int32_t userDevId, const uint8_t dieId, uint64_t& xnBaseAddr)
{
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ?
              CcuResSpecifications::GetInstance(userDevId).GetXnBaseAddr(dieId, xnBaseAddr) :
              Hccl::CcuResSpecifications::GetInstance(userDevId).GetXnBaseAddr(dieId, xnBaseAddr);
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuDevMgrImp::GetResSpecsCascCntNum(const int32_t userDevId, const uint8_t dieId, uint32_t& cascCntNum)
{
    if (!CheckCcuOpenSourceEnable()) {
        // 此处不同于兄弟接口返回 HCCL_E_NOT_SUPPORT：0.5RTT 级联计数器属于可选特性，
        // legacy 版本无该硬件能力，按“数量为 0”成功返回，避免调用方将其当作错误处理。
        HCCL_WARNING(
            "[CcuDevMgrImp][%s] is not supported for legacy interface, cascCntNum is set to 0, "
            "userDevId[%d], dieId[%u].",
            __func__, userDevId, dieId);
        cascCntNum = 0;
        return HcclResult::HCCL_SUCCESS;
    }
    return CcuResSpecifications::GetInstance(userDevId).GetCascCntNum(dieId, cascCntNum);
}

HcclResult CcuDevMgrImp::GetCkeBaseAddr(const int32_t userDevId, const uint8_t dieId, uint64_t& ckeBaseAddr)
{
    if (!CheckCcuOpenSourceEnable()) {
        HCCL_WARNING("[CcuDevMgrImp][%s] is not supported for legacy interface.", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    return CcuResSpecifications::GetInstance(userDevId).GetCkeBaseAddr(dieId, ckeBaseAddr);
}

HcclResult
CcuDevMgrImp::GetXnOffsetCcumAddrById(const int32_t userDevId, const uint8_t dieId, uint16_t id, uint64_t& xnAddr)
{
    if (!CheckCcuOpenSourceEnable()) {
        HCCL_WARNING("[CcuDevMgrImp][%s] is not supported for legacy interface.", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    return CcuResSpecifications::GetInstance(userDevId).GetXnOffsetCcumAddrById(dieId, id, xnAddr);
}

HcclResult
CcuDevMgrImp::GetCkeOffsetCcumAddrById(const int32_t userDevId, const uint8_t dieId, uint16_t id, uint64_t& ckeAddr)
{
    if (!CheckCcuOpenSourceEnable()) {
        HCCL_WARNING("[CcuDevMgrImp][%s] is not supported for legacy interface.", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    return CcuResSpecifications::GetInstance(userDevId).GetCkeOffsetCcumAddrById(dieId, id, ckeAddr);
}

HcclResult CheckDieValid(
    const char* funcName, const int32_t userDevId, const uint8_t dieId,
    const std::array<bool, CCU_MAX_IODIE_NUM>& dieEnableFlags)
{
    CHK_PRT_RET(
        dieId >= CCU_MAX_IODIE_NUM,
        HCCL_ERROR(
            "[%s] failed, dieId[%u] is invalid, should be in [0-%u), userDevId[%d].", funcName, dieId,
            CCU_MAX_IODIE_NUM, userDevId),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET(
        !dieEnableFlags[dieId],
        HCCL_ERROR("[%s] failed, dieId[%u] is disable, userDevId[%d].", funcName, dieId, userDevId),
        HcclResult::HCCL_E_PARA);

    return HcclResult::HCCL_SUCCESS;
}

bool CcuIsInited(const int32_t userDevId)
{
    HCCL_INFO("[CcuIsInited] Input params: userDevId[%d]", userDevId);
    CHK_PRT_RET(
        (userDevId < 0 || static_cast<u32>(userDevId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR("[CcuIsInited] userDevId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", userDevId, MAX_MODULE_DEVICE_NUM),
        false);

    if (!CheckCcuOpenSourceEnable()) {
        return Hccl::CcuComponent::GetInstance(userDevId).IsInited();
    }

    std::lock_guard<std::mutex> lock(ccuDrvHandleMutex);
    // ccu驱动已重复拉起失败时，直接返回，在锁保护内返回
    if (ccuDriverInitAgainFlag) {
        return false;
    }

    auto iter = ccuDrvHandleMap.find(userDevId);
    if (iter == ccuDrvHandleMap.end()) {
        return false;
    }

    return true;
}

HcclResult CcuSetTaskKill(const int32_t userDevId)
{
    HCCL_INFO("[CcuSetTaskKill] Input params: userDevId[%d]", userDevId);
    // 入参校验拦截
    CHK_PRT_RET(
        (userDevId < 0 || static_cast<u32>(userDevId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR("[CcuSetTaskKill] userDevId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", userDevId, MAX_MODULE_DEVICE_NUM),
        HcclResult::HCCL_E_PARA);
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ? CcuComponent::GetInstance(userDevId).SetTaskKill() :
                                       Hccl::CcuComponent::GetInstance(userDevId).SetTaskKill();
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuSetTaskKillDone(const int32_t userDevId)
{
    HCCL_INFO("[CcuSetTaskKillDone] Input params: userDevId[%d]", userDevId);
    // 入参校验拦截
    CHK_PRT_RET(
        (userDevId < 0 || static_cast<u32>(userDevId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR(
            "[CcuSetTaskKillDone] userDevId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", userDevId, MAX_MODULE_DEVICE_NUM),
        HcclResult::HCCL_E_PARA);
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ? CcuComponent::GetInstance(userDevId).SetTaskKillDone() :
                                       Hccl::CcuComponent::GetInstance(userDevId).SetTaskKillDone();
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuCleanTaskKillState(const int32_t userDevId)
{
    HCCL_INFO("[CcuCleanTaskKillState] Input params: userDevId[%d]", userDevId);
    // 入参校验拦截
    CHK_PRT_RET(
        (userDevId < 0 || static_cast<u32>(userDevId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR(
            "[CcuCleanTaskKillState] userDevId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", userDevId, MAX_MODULE_DEVICE_NUM),
        HcclResult::HCCL_E_PARA);
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ? CcuComponent::GetInstance(userDevId).CleanTaskKillState() :
                                       Hccl::CcuComponent::GetInstance(userDevId).CleanTaskKillState();
    EXCEPTION_HANDLE_END
    return ret;
}

HcclResult CcuCleanDieCkes(const int32_t userDevId, const uint8_t dieId)
{
    HCCL_INFO("[CcuCleanDieCkes] Input params: userDevId[%d], dieId[%u]", userDevId, dieId);
    // 入参校验拦截
    CHK_PRT_RET(
        (userDevId < 0 || static_cast<u32>(userDevId) >= MAX_MODULE_DEVICE_NUM),
        HCCL_ERROR(
            "[CcuCleanDieCkes] userDevId[%d] error, MAX_MODULE_DEVICE_NUM[%u]", userDevId, MAX_MODULE_DEVICE_NUM),
        HcclResult::HCCL_E_PARA);
    HcclResult ret;
    EXCEPTION_HANDLE_BEGIN
    ret = CheckCcuOpenSourceEnable() ? CcuComponent::GetInstance(userDevId).CleanDieCkes(dieId) :
                                       Hccl::CcuComponent::GetInstance(userDevId).CleanDieCkes(dieId);
    EXCEPTION_HANDLE_END
    return ret;
}

CcuResult CcuAllocCntXnBlock(const int32_t userDevId, const uint8_t dieId, CntXnBlock& cntXnBlock)
{
    if (!CheckCcuOpenSourceEnable()) {
        HCCL_WARNING("[CcuDevMgrImp][%s] is not supported for legacy interface.", __func__);
        return CcuResult::CCU_E_NOT_SUPPORT;
    }

    HCCL_INFO("[%s] new alloc count xn block request: userDevId[%d], dieId[%u].", __func__, userDevId, dieId);
    HcclResult ret = CcuComponent::GetInstance(userDevId).AllocCntXnBlock(dieId, cntXnBlock);
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        // 资源不足(本 die 配置寄存器耗尽或 cntXn 不足), 保留可回退语义, 供上层区分于内部错误
        HCCL_WARNING("[%s] alloc count xn block unavailable: userDevId[%d], dieId[%u].", __func__, userDevId, dieId);
        return CcuResult::CCU_E_UNAVAIL;
    }
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[%s] alloc count xn block failed: ret[%d].", __func__, ret);
        return CcuResult::CCU_E_INTERNAL;
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuReleaseCntXnBlock(const int32_t userDevId, const uint8_t dieId, const CntXnBlock& cntXnBlock)
{
    if (!CheckCcuOpenSourceEnable()) {
        HCCL_WARNING("[CcuDevMgrImp][%s] is not supported for legacy interface.", __func__);
        return CcuResult::CCU_E_NOT_SUPPORT;
    }

    HCCL_INFO("[%s] release count xn block request: userDevId[%d], dieId[%u].", __func__, userDevId, dieId);
    if (CcuComponent::GetInstance(userDevId).ReleaseCntXnBlock(dieId, cntXnBlock) != HcclResult::HCCL_SUCCESS) {
        return CcuResult::CCU_E_INTERNAL;
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult
CcuQueryTokenInfo(const int32_t userDevId, uint64_t srcVa, uint64_t size, uint64_t& tokenId, uint64_t& tokenValue)
{
    if (!CheckCcuOpenSourceEnable()) {
        HCCL_WARNING("[CcuDevMgrImp][%s] is not supported for legacy interface.", __func__);
        return CcuResult::CCU_E_NOT_SUPPORT;
    }

    HCCL_INFO(
        "[%s] query token info request: userDevId[%d], srcVa[%llu], size[%llu].", __func__, userDevId, srcVa, size);
    HcclResult ret = CcuComponent::GetInstance(userDevId).QueryTokenInfo(srcVa, size, tokenId, tokenValue);
    if (ret == HcclResult::HCCL_E_NOT_FOUND) {
        HCCL_WARNING("[%s] query token info not found, ret[%d].", __func__, ret);
        return CcuResult::CCU_E_NOT_FOUND;
    } else if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[%s] query token info failed, ret[%d].", __func__, ret);
        return CcuResult::CCU_E_INTERNAL;
    }
    return CcuResult::CCU_SUCCESS;
}
}; // namespace hcomm
