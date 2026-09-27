/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_mgr.h"

#include <cstdint>

#include <acl/acl.h>

#include "hccl_common.h"
#include "ccu_types.h"
#include "exception_handler.h"
#include "adapter_rts.h"
#include "ccu_assist_v1.h"
#include "dev_buffer.h"
#include "ccu_ins_generator_v1.h"
#include "ccu_ins_generator_v2.h"
#include "ccu_dev_mgr_imp.h"

#include "ccu_rep_base_v1.h"
#include "ccu_rep_block_v1.h"
#include "ccu_rep_type_v1.h"

#include "hcomm_adapter_hccp.h"

#include "ccu_log.h"
#include "ccu_kernel_func.h"
#include "config_plf_log_v2.h"

namespace hcomm {

using Hccl::PLF_RES;

HcclResult GetHcclVersionForCcuKernelMgr(int& hcclVersion)
{
    char hcclPkgName[] = "hccl";
    aclError aclRet = aclsysGetVersionNum(hcclPkgName, &hcclVersion);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS,
        HCCL_ERROR("[GetHcclVersionForCcuKernelMgr] aclsysGetVersionNum failed, aclRet[%d].", aclRet), HCCL_E_INTERNAL);
    HCCL_RUN_INFO("[GetHcclVersionForCcuKernelMgr] hccl version is %d.", hcclVersion);
    return HCCL_SUCCESS;
}

constexpr int MAX_HCCL_VERSION_USING_CCU_RES_STATIC_ALLOC = 90100000;

static HcclResult ComputeKernelInstrRegionSize(CcuKernel* kernel, const int32_t userDevId, uint32_t& regionSize);

CcuKernelMgr::~CcuKernelMgr()
{
    if (!initializedFlag_) {
        return;
    }

    if (instructionLoadDevMem_) {
        HCCL_RUN_INFO(
            "[CcuKernelMgr][~CcuKernelMgr]: userDevId[%d], free addr[%p]", userDevId_, instructionLoadDevMem_);
        (void)hrtFree(instructionLoadDevMem_);
        instructionLoadDevMem_ = nullptr;
    }

    (void)Deinit();
}

CcuKernelMgr& CcuKernelMgr::GetInstance(const int32_t userDevId)
{
    static CcuKernelMgr kernelManager[MAX_MODULE_DEVICE_NUM + 1];

    int32_t validUserDevId = userDevId;
    if (validUserDevId < 0 || static_cast<uint32_t>(validUserDevId) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[CcuKernelMgr][%s] use the backup device, userDevId[%d] should be "
            "less than %u.",
            __func__, validUserDevId, MAX_MODULE_DEVICE_NUM);
        validUserDevId = MAX_MODULE_DEVICE_NUM; // 使用备份设备
    }

    kernelManager[validUserDevId].userDevId_ = validUserDevId;
    return kernelManager[validUserDevId];
}

HcclResult CcuKernelMgr::Init()
{
    std::unique_lock<std::mutex> lock(kernelMapMutex_);
    if (initializedFlag_) {
        return HcclResult::HCCL_SUCCESS;
    }

    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        bool enableFlag = false;
        CHK_RET(static_cast<HcclResult>(CcuGetDieEnableInfo(userDevId_, dieId, enableFlag)));
        if (!enableFlag) {
            continue;
        }

        CHK_RET(InstantiationTranslator(dieId));
    }

    initializedFlag_ = true;
    kernelMap_.clear();

    CHK_RET(CcuDevMgrImp::GetCcuVersion(userDevId_, ccuVersion_));
    HCCL_INFO("[CcuKernelMgr] Get CcuVersion[%d](0: CcuV1, 1: CcuV2, 2: Invalid)", ccuVersion_);
    if (ccuVersion_ == CcuVersion::INVALID) {
        HCCL_ERROR("[CcuKernelMgr][%s] Invalid chip type, abort Init.", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (ccuVersion_ == CcuVersion::CCU_V2) {
        HCCL_INFO("[CcuKernelMgr] Init CcuInsGeneratorV2");
        insGenePtr = std::make_shared<CcuRep::CcuInsGeneratorV2>();
        return HcclResult::HCCL_SUCCESS;
    }

    HCCL_INFO("[CcuKernelMgr] Init CcuInsGeneratorV1");
    insGenePtr = std::make_shared<CcuRep::CcuInsGeneratorV1>();
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelMgr::Deinit()
{
    // 不需要主动释放CCU指令空间等资源，因为设备管理与kernelMgr都为静态，生命周期一致
    std::unique_lock<std::mutex> lock(kernelMapMutex_);
    translatorResPack.handles.clear();
    initializedFlag_ = false;
    kernelMap_.clear();
    translators.clear();
    referenceMgrs.clear();
    return HcclResult::HCCL_SUCCESS;
}

CcuResult CcuKernelMgr::Register(
    CcuResPack& resPack, const uint32_t dieId, const char* kernelFuncName, const void* kernelFunc,
    const void** kernelArgs, const uint32_t argNum, CcuInstance* ccuIns, CcuKernelHandle& kernelHandle)
{
    // 允许kernelFuncName为空，此时传递默认名称
    CCU_CHK_PTR_NULL(kernelFunc);

    // 当前argNum仅允许 0 或 1
    if (argNum > 1) {
        HCCL_ERROR("[%s] failed, argNum[%u] now only support 0 or 1.", __func__, argNum);
        return CcuResult::CCU_E_PARA;
    }

    // 注意处理时序，需要先重置后处理rep
    std::unique_lock<std::mutex> lock(kernelMapMutex_);
    CCU_CHK_RET(BuildKernel(dieId, kernelFuncName, kernelFunc, kernelArgs, argNum, ccuIns));

    CcuResult ret = AllocRes(resPack);
    if (ret != CcuResult::CCU_SUCCESS) {
        HCCL_WARNING("[%s] AllocRes failed, maybe resource not enough, please check ret[%d]", __func__, ret);
        return ret;
    }

    kernelId_++;
    kernelMap_[kernelId_] = std::move(currKernel_);

    kernelHandle = kernelId_;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernelMgr::BuildKernel(
    uint32_t dieId, const char* kernelFuncName, const void* kernelFunc, const void** kernelArgs, uint32_t argNum,
    CcuInstance* ccuIns)
{
    currKernel_ = std::make_unique<CcuKernel>(); // 重置待构建kernel
    // 执行算法流程时将资源占用临时记录在 die 0，后续确定实际 die 并迁移资源
    currKernel_->SetDieId(0);
    CCU_CHK_RET(currKernel_->SetupProfilingInfo(kernelFuncName));

    // 初始化翻译器（需在执行kernel func前设置，因为func执行时会创建rep对象）
    currKernel_->SetInsGenerater(insGenePtr.get());
    currKernel_->SetCcuVersion(ccuVersion_);

    if (argNum == 0) {
        auto ccuKernelFunc = reinterpret_cast<CcuKernelFuncNoArg>(kernelFunc);
        CCU_CHK_RET(ccuKernelFunc()); // 执行算法流程，生成rep和计算资源占用
    } else {
        CCU_CHK_PTR_NULL(kernelArgs);
        const void* kernelArg = kernelArgs[0];
        CCU_CHK_PTR_NULL(kernelArg);
        const auto ccuKernelArg = const_cast<CcuKernelArg>(kernelArg);
        auto ccuKernelFunc = reinterpret_cast<CcuKernelFuncOneArg>(kernelFunc);
        CCU_CHK_RET(ccuKernelFunc(ccuKernelArg)); // 执行算法流程，生成rep和计算资源占用
    }

    currKernel_->FlushClosablePendingIfs(); // 处理未闭合的if
    if (ccuIns != nullptr && ccuIns->IsFixedResNum()) {
        // 按固定资源数量创建的 ccu instance，外部 dieId 始终为 0，从 channel 获取实际 dieId
        CCU_CHK_RET(currKernel_->ApplyDieFromChannels());
    } else {
        // 按需创建的 ccu instance，校验所有 channel 使用相同的 die，然后将资源占用从 die 0 迁移到指定 die
        CCU_CHK_RET(currKernel_->ValidateAndApplyDie(dieId));
    }
    CCU_CHK_RET(PrepareConstValueResources()); // 记录翻译过程所需常量并申请对应资源
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernelMgr::GetKernelResourceRequest(
    const uint32_t dieId, const char* kernelFuncName, const void* kernelFunc, const void** kernelArgs,
    const uint32_t argNum, CcuResReq& resReq, uint32_t& instrCount)
{
    CCU_CHK_PTR_NULL(kernelFunc);
    if (argNum > 1) {
        HCCL_ERROR("[%s] failed, argNum[%u] now only support 0 or 1.", __func__, argNum);
        return CcuResult::CCU_E_PARA;
    }
    if (argNum == 1) {
        CCU_CHK_PTR_NULL(kernelArgs);
        CCU_CHK_PTR_NULL(kernelArgs[0]);
    }

    std::unique_lock<std::mutex> lock(kernelMapMutex_);
    currKernel_.reset();
    struct CurrentKernelGuard {
        explicit CurrentKernelGuard(std::unique_ptr<CcuKernel>& kernel) : kernel_(kernel) {}
        ~CurrentKernelGuard() { kernel_.reset(); }
        std::unique_ptr<CcuKernel>& kernel_;
    } guard(currKernel_);

    CCU_CHK_RET(BuildKernel(dieId, kernelFuncName, kernelFunc, kernelArgs, argNum, nullptr));
    resReq = currKernel_->GetResourceRequest();
    const uint32_t kernelInstrCount = currKernel_->GetInstrCount();
    const uint32_t translatorInstrCount = CcuRepTranslator::GetInstrNum(userDevId_);
    const uint32_t constInstrCount = static_cast<uint32_t>(currKernel_->GetConstValue2VarMap().size());
    // reserve * latency 用 uint64_t 承载, 避免分项乘法在 uint32_t 空间内溢出使日志值失真;
    // 分项仅用于观测, 溢出的最终防护在 ComputeKernelInstrRegionSize 内 (饱和 + ERROR).
    const uint64_t ckeReserveInstrCount
        = static_cast<uint64_t>(currKernel_->GetRepNeedToAddLatency()) * CcuRep::CCU_CKE_RAW_LATENCY;
    const uint64_t xnReserveInstrCount
        = static_cast<uint64_t>(currKernel_->GetLsxRepReserveCount()) * CcuRep::CCU_XN_RAW_LATENCY;
    // 总数统一走 ComputeKernelInstrRegionSize, 与申请/释放口径保持结构一致; 分项仅用于日志观测.
    // 溢出时该函数返回错误, 经 CCU_CHK_RET 中断查询, 避免把非法 instrCount 回给上层.
    CCU_CHK_RET(ComputeKernelInstrRegionSize(currKernel_.get(), userDevId_, instrCount));
    HCCL_INFO(
        "[HcommCcuKernelQueryResReq][%s] resource request instruction count, kernelInstrCount[%u], "
        "translatorInstrCount[%u], constInstrCount[%u], ckeReserveInstrCount[%llu], xnReserveInstrCount[%llu], "
        "totalInstrCount[%u].",
        __func__, kernelInstrCount, translatorInstrCount, constInstrCount,
        static_cast<unsigned long long>(ckeReserveInstrCount), static_cast<unsigned long long>(xnReserveInstrCount),
        instrCount);
    return CcuResult::CCU_SUCCESS;
}

static void DumpResReqInfo(const CcuResReq& totalRes)
{
    for (uint32_t i = 0; i < CCU_MAX_IODIE_NUM; i++) {
        if (totalRes.msReq[i] != 0 || totalRes.blockMsReq[i] != 0 || totalRes.ckeReq[i] != 0
            || totalRes.blockCkeReq[i] != 0 || totalRes.loopEngineReq[i] != 0 || totalRes.blockLoopEngineReq[i] != 0
            || totalRes.gsaReq[i] != 0 || totalRes.blockGsaReq[i] != 0 || totalRes.xnReq[i] != 0
            || totalRes.blockXnReq[i] != 0 || totalRes.missionReq.req[i] != 0) {
            HCCL_INFO(
                "DumpResReqInfo: dieId[%u], msReq[%u], blockMsReq[%u], ckeReq[%u], blockCkeReq[%u], "
                "loopEngineReq[%u], blockLoopEngineReq[%u], gsaReq[%u], blockGsaReq[%u], xnReq[%u], blockXnReq[%u], "
                "missionReq[%u]",
                i, totalRes.msReq[i], totalRes.blockMsReq[i], totalRes.ckeReq[i], totalRes.blockCkeReq[i],
                totalRes.loopEngineReq[i], totalRes.blockLoopEngineReq[i], totalRes.gsaReq[i], totalRes.blockGsaReq[i],
                totalRes.xnReq[i], totalRes.blockXnReq[i], totalRes.missionReq.req[i]);
        }
    }
}

inline int32_t GetResTotalNum(const std::vector<ResInfo>& resInfos)
{
    int32_t resNum = 0;
    for (ResInfo resInfo : resInfos) {
        resNum += static_cast<int32_t>(resInfo.num);
    }
    return resNum;
}

static void GetResNumFromResPack(CcuResPack& resPack, CcuResReq& totalRes)
{
    // 获取通信域当前所持有的资源
    const auto& tmpResRepository = resPack.GetCcuResRepo();

    // 合并获取的所持有的资源信息, 按照类型合并资源总和到totalRes的第0个vector中
    for (u32 i = 0; i < CCU_MAX_IODIE_NUM; i++) {
        totalRes.msReq[i] += GetResTotalNum(tmpResRepository.ms[i]);
        totalRes.blockMsReq[i] += GetResTotalNum(tmpResRepository.blockMs[i]);
        totalRes.ckeReq[i] += GetResTotalNum(tmpResRepository.cke[i]);
        totalRes.blockCkeReq[i] += GetResTotalNum(tmpResRepository.blockCke[i]);
        totalRes.loopEngineReq[i] += GetResTotalNum(tmpResRepository.loopEngine[i]);
        totalRes.blockLoopEngineReq[i] += GetResTotalNum(tmpResRepository.blockLoopEngine[i]);
        totalRes.gsaReq[i] += GetResTotalNum(tmpResRepository.gsa[i]);
        totalRes.blockGsaReq[i] += GetResTotalNum(tmpResRepository.blockGsa[i]);
        totalRes.xnReq[i] += GetResTotalNum(tmpResRepository.xn[i]);
        totalRes.blockXnReq[i] += GetResTotalNum(tmpResRepository.blockXn[i]);
        totalRes.missionReq.req[i] += GetResTotalNum(tmpResRepository.mission.mission[i]);
    }

    DumpResReqInfo(totalRes);
    HCCL_INFO("GetResPackTotalResNum:dumpInfos success.");
}

inline uint32_t GetReqResNum(const uint32_t reqRes, const uint32_t totalRes)
{
    return ((reqRes > totalRes) ? (reqRes - totalRes) : 0);
}

static bool CheckResIfAvailable(const CcuResReq& totalRes, const CcuResReq& resReq)
{
    DumpResReqInfo(resReq);

    CcuResReq needResReq{};
    // todo: 优化为遍历数组
    for (u32 i = 0; i < CCU_MAX_IODIE_NUM; i++) {
        needResReq.msReq[i] = GetReqResNum(resReq.msReq[i], totalRes.msReq[i]);
        needResReq.blockMsReq[i] = GetReqResNum(resReq.blockMsReq[i], totalRes.blockMsReq[i]);
        needResReq.ckeReq[i] = GetReqResNum(resReq.ckeReq[i], totalRes.ckeReq[i]);
        needResReq.blockCkeReq[i] = GetReqResNum(resReq.blockCkeReq[i], totalRes.blockCkeReq[i]);
        needResReq.loopEngineReq[i] = GetReqResNum(resReq.loopEngineReq[i], totalRes.loopEngineReq[i]);
        needResReq.blockLoopEngineReq[i] = GetReqResNum(resReq.blockLoopEngineReq[i], totalRes.blockLoopEngineReq[i]);
        needResReq.gsaReq[i] = GetReqResNum(resReq.gsaReq[i], totalRes.gsaReq[i]);
        needResReq.blockGsaReq[i] = GetReqResNum(resReq.blockGsaReq[i], totalRes.blockGsaReq[i]);
        needResReq.xnReq[i] = GetReqResNum(resReq.xnReq[i], totalRes.xnReq[i]);
        needResReq.blockXnReq[i] = GetReqResNum(resReq.blockXnReq[i], totalRes.blockXnReq[i]);
        needResReq.missionReq.req[i] = GetReqResNum(resReq.missionReq.req[i], totalRes.missionReq.req[i]);

        if (needResReq.missionReq.req[i] > 0) {
            needResReq.missionReq.reqType = resReq.missionReq.reqType;
        }

        if (needResReq.msReq[i] != 0 || needResReq.blockMsReq[i] != 0 || needResReq.ckeReq[i] != 0
            || needResReq.blockCkeReq[i] != 0 || needResReq.loopEngineReq[i] != 0
            || needResReq.blockLoopEngineReq[i] != 0 || needResReq.gsaReq[i] != 0 || needResReq.blockGsaReq[i] != 0
            || needResReq.xnReq[i] != 0 || needResReq.blockXnReq[i] != 0 || needResReq.missionReq.req[i] != 0) {
            HCCL_WARNING(
                "[CcuKernelMgr][%s] dieId[%u] not enough, msReq[%u] blockMsReq[%u] ckeReq[%u]"
                "blockCkeReq[%u] loopEngineReq[%u] blockLoopEngineReq[%u] gsaReq[%u] blockGsaReq[%u] xnReq[%u]"
                "blockXnReq[%u] missionReq[%u].",
                __func__, i, needResReq.msReq[i], needResReq.blockMsReq[i], needResReq.ckeReq[i],
                needResReq.blockCkeReq[i], needResReq.loopEngineReq[i], needResReq.blockLoopEngineReq[i],
                needResReq.gsaReq[i], needResReq.blockGsaReq[i], needResReq.xnReq[i], needResReq.blockXnReq[i],
                needResReq.missionReq.req[i]);
            return false;
        }
    }

    return true;
}

static void MoveResInfo(std::vector<ResInfo>& dest, std::vector<ResInfo>& source, const uint32_t resNum)
{
    // Register 前序流程已检查资源不足场景
    if (resNum == 0) {
        return;
    }

    dest.clear();
    auto iter = source.begin();
    uint32_t remain = resNum;
    while (remain > 0 && iter != source.end()) {
        auto& srcBlock = *iter;
        const uint32_t take = std::min(remain, srcBlock.num);
        dest.emplace_back(srcBlock.startId, take);

        if (take == srcBlock.num) {
            // 完全用掉这个资源，source中移除
            iter = source.erase(iter);
        } else {
            // 只用了部分，更新source中的资源
            srcBlock.startId += take;
            srcBlock.num -= take;
        }

        remain -= take; // 更新剩余需要的资源数量
    }
}

static void LoadRes(std::unique_ptr<CcuKernel>& kernel, CcuResPack& resPack)
{
    const CcuResReq& resReq = kernel->GetResourceRequest();
    CcuResRepository& totalResRepo = resPack.GetCcuResRepo();
    CcuResRepository kernelResRepo{};

    for (uint8_t i = 0; i < CCU_MAX_IODIE_NUM; i++) { // todo: 建议改成dieId
        MoveResInfo(kernelResRepo.loopEngine[i], totalResRepo.loopEngine[i], resReq.loopEngineReq[i]);
        MoveResInfo(kernelResRepo.blockLoopEngine[i], totalResRepo.blockLoopEngine[i], resReq.blockLoopEngineReq[i]);
        MoveResInfo(kernelResRepo.ms[i], totalResRepo.ms[i], resReq.msReq[i]);
        MoveResInfo(kernelResRepo.blockMs[i], totalResRepo.blockMs[i], resReq.blockMsReq[i]);
        MoveResInfo(kernelResRepo.cke[i], totalResRepo.cke[i], resReq.ckeReq[i]);
        MoveResInfo(kernelResRepo.blockCke[i], totalResRepo.blockCke[i], resReq.blockCkeReq[i]);
        MoveResInfo(kernelResRepo.blockXn[i], totalResRepo.blockXn[i], resReq.blockXnReq[i]);
        MoveResInfo(kernelResRepo.xn[i], totalResRepo.xn[i], resReq.xnReq[i]);
        MoveResInfo(kernelResRepo.gsa[i], totalResRepo.gsa[i], resReq.gsaReq[i]);
        MoveResInfo(kernelResRepo.blockGsa[i], totalResRepo.blockGsa[i], resReq.blockGsaReq[i]);
        MoveResInfo(kernelResRepo.mission.mission[i], totalResRepo.mission.mission[i], resReq.missionReq.req[i]);
    }

    kernel->SetResRepository(kernelResRepo);
    kernel->SetCascCntBlock(resPack.GetCascCntBlocks());
}

// 指令空间区域大小的唯一计算入口:
//   裸指令数 (rep InstrCount 累加) + 翻译器结构指令 (GetInstrNum) + 常量赋值指令
//   + 每个会翻译出 waitCKEId!=0 && clearType=1 的 set/clearCKE 的 rep 预留 CCU_CKE_RAW_LATENCY 条
//     NOP 空间 (计入集合见 CcuKernel::GetRepNeedToAddLatency / IsCkeWaitRep: 三种 wait 类 +
//     LOAD/LOAD_VAR/STORE/STORE_VAR/RECORD_SHARED_NOTIFY).
//   + 每个会翻译出 LoadX/StoreX/ClearX (half-rtt 特殊指令) 的 rep 预留 2 * CCU_XN_RAW_LATENCY 条 NOP
//     空间 (计入集合见 CcuKernel::GetLsxRepReserveCount / IsLdStXRep: LOAD_ADD_IMM / STORE_ADD_IMM /
//     CASC_CNT_CLEAR). 之所以是 2 倍: 后端优化对 lsx/clearx 存在两个方向的写后读补 NOP —
//     (1) lsx 写 xn/array -> 后续任意指令读 (该 rep 作为写者, 其后读者补 NOP);
//     (2) 任意前序指令写 xn/array -> 本 lsx/clearx 读该 xn/array (硬件对 lsx/clearx 读操作数
//         interlock 失效, 该 rep 作为读者, 在其之前补 NOP).
//     单个 lsx/clearx rep 前后两侧最坏各补 (CCU_XN_RAW_LATENCY - 1) 条, 故按 2 * CCU_XN_RAW_LATENCY
//     预留 (含 1 条余量) 即可从构造上保证优化后指令数不越界.
// 后端优化 cke-only 档只会为 CKE / XN 写后读补 NOP: CKE 每个此类 rep 最多补 (latency-1) 条;
// lsx/clearx 每个此类 rep 因双向写后读最坏补 2*(latency-1) 条, 故 XN 侧按 2*latency 预留.
// 此预留可从构造上保证优化后指令数不超过申请区. 申请 / 查询 / 释放三处必须走本函数,
// 保证口径一致 (尤其申请与释放必须完全相等).
static HcclResult ComputeKernelInstrRegionSize(CcuKernel* kernel, const int32_t userDevId, uint32_t& regionSize)
{
    // 各分项均为 uint32_t, 乘法 (reserve * latency) 与累加在 uint32_t 空间内理论上可能溢出.
    // 此处全程用 uint64_t 承载中间结果; 一旦超过 UINT32_MAX 说明指令区规模异常 (不可恢复的内部
    // 错误), 打 ERROR 并返回 HCCL_E_INTERNAL 由调用方 CHK_RET 中断执行, 避免截断/饱和后继续导致
    // 后续指令区申请与越界写不一致. 申请/查询/释放/校验四处共用本函数, 口径一致.
    const uint64_t total
        = static_cast<uint64_t>(kernel->GetInstrCount())
          + static_cast<uint64_t>(CcuRep::CcuRepTranslator::GetInstrNum(userDevId))
          + static_cast<uint64_t>(kernel->GetConstValue2VarMap().size())
          + static_cast<uint64_t>(kernel->GetRepNeedToAddLatency()) * CcuRep::CCU_CKE_RAW_LATENCY
          + static_cast<uint64_t>(kernel->GetLsxRepReserveCount()) * (2ULL * CcuRep::CCU_XN_RAW_LATENCY);
    if (total > static_cast<uint64_t>(UINT32_MAX)) {
        HCCL_ERROR(
            "[CcuKernelMgr][%s] kernel instr region size[%llu] overflow uint32. "
            "instrCount[%u], translatorInstrCount[%u], constInstrCount[%zu], repNeedToAddLatency[%u], "
            "lsxRepReserveCount[%u].",
            __func__, static_cast<unsigned long long>(total), kernel->GetInstrCount(),
            CcuRep::CcuRepTranslator::GetInstrNum(userDevId), kernel->GetConstValue2VarMap().size(),
            kernel->GetRepNeedToAddLatency(), kernel->GetLsxRepReserveCount());
        return HcclResult::HCCL_E_INTERNAL;
    }
    regionSize = static_cast<uint32_t>(total);
    return HcclResult::HCCL_SUCCESS;
}

static CcuResult AllocInstrRes(std::unique_ptr<CcuKernel>& kernel, const int32_t userDevId)
{
    uint32_t instrCount = 0;
    CCU_CHK_RET(ComputeKernelInstrRegionSize(kernel.get(), userDevId, instrCount));
    const uint32_t dieId = kernel->GetDieId();
    ResInfo insInfo(0, 0);
    CCU_CHK_RET(CcuDevMgrImp::AllocIns(userDevId, dieId, instrCount, insInfo));
    PLF_CONFIG_INFO(
        PLF_RES, "[CcuKernelMgr]: userDevId[%d], dieId[%u], startId[%u], count[%u]", userDevId, dieId, insInfo.startId,
        insInfo.num);
    kernel->SetInstrId(insInfo.startId);

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernelMgr::PrepareConstValueResources()
{
    // insGenerator统计rep中常量，并填写当前kernel的常量表，当前只有A6有对应处理，A5没有常量处理需求
    CCU_CHK_PTR_NULL(currKernel_);
    const auto& repVec = currKernel_->GetRepSequence();

    const auto& translator = translators[currKernel_->GetDieId()][0];
    CCU_CHK_PTR_NULL(translator);
    const auto& transDep = translator->GetTransDep(); // 此时未分配missionid，取0对应的transDep读取常量
    CCU_CHK_PTR_NULL(insGenePtr);
    for (uint32_t index = 0; index < repVec.size(); index++) {
        const auto& curRepType = repVec[index]->Type();
        CcuRep::CcuRepBase* curRepPtr = repVec[index].get();
        CCU_CHK_PTR_NULL(curRepPtr);
        HCCL_DEBUG("Current rep[%d] ptr[%p] repType[%d]", index, curRepPtr, curRepType);

        // 遍历每个rep，包括repBlock中的每个rep，将常量资源需求记录在currkernel中
        CCU_CHK_RET(insGenePtr->PrepareConstValue(curRepPtr, transDep, currKernel_.get()));
        if (curRepType == CcuRep::CcuRepType::BLOCK || curRepType == CcuRep::CcuRepType::FUNC_BLOCK
            || curRepType == CcuRep::CcuRepType::LOOP_BLOCK) {
            CcuRep::CcuRepBlock* curRepBlockPtr = static_cast<CcuRep::CcuRepBlock*>(curRepPtr);
            CCU_CHK_PTR_NULL(curRepBlockPtr);
            for (const auto& repInBlock : curRepBlockPtr->GetReps()) {
                CCU_CHK_RET(insGenePtr->PrepareConstValue(repInBlock.get(), transDep, currKernel_.get()));
            }
        }
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuKernelMgr::AllocRes(CcuResPack& resPack)
{
    CcuResReq leftRes{};
    GetResNumFromResPack(resPack, leftRes);

    const CcuResReq& resReq = currKernel_->GetResourceRequest();
    // todo： 需要整改，传递资源不足的信息
    if (!CheckResIfAvailable(leftRes, resReq)) {
        HCCL_WARNING("[CcuKernelMgr][%s] resource is not enough.", __func__);
        return CcuResult::CCU_E_UNAVAIL;
    }

    // 申请指令空间资源
    CCU_CHK_RET(AllocInstrRes(currKernel_, userDevId_));

    // 资源从respack转移至kernel
    LoadRes(currKernel_, resPack);

    return CcuResult::CCU_SUCCESS;
}

template <typename T1, typename T2>
HcclResult
ResetRepResourceTemplate(std::vector<T1>& resource, const std::vector<T2>& repository, const uint32_t startIndex = 0)
{
    if (resource.size() > repository.size() - startIndex) {
        HCCL_ERROR(
            "[CcuKernelMgr][ResetRepResourceTemplate]resource size[%u] bigger "
            "repository size[%u] typeid[%s]",
            resource.size(), repository.size(), typeid(T1).name());
        return HcclResult::HCCL_E_INTERNAL;
    }

    for (uint32_t j = 0; j < resource.size(); j++) {
        resource[j].Reset(repository[j + startIndex].startId);
    }

    return HcclResult::HCCL_SUCCESS;
}

static HcclResult
ResetRepResourceToResRepository(CcuRepResource& totalRepRes, const CcuResRepository& totalResRepository)
{
    // 遍历translatorRepRes, 将每个rep的虚拟资源翻译到实际物理资源上
    for (u32 i = 0; i < CCU_MAX_IODIE_NUM; i++) {
        CHK_RET(ResetRepResourceTemplate(totalRepRes.ccubufs[i], totalResRepository.ms[i]));
        CHK_RET(ResetRepResourceTemplate(totalRepRes.blockCcubufs[i], totalResRepository.blockMs[i]));
        CHK_RET(ResetRepResourceTemplate(totalRepRes.executor[i], totalResRepository.loopEngine[i]));
        CHK_RET(ResetRepResourceTemplate(totalRepRes.blockExecutor[i], totalResRepository.blockLoopEngine[i]));
        CHK_RET(ResetRepResourceTemplate(totalRepRes.completedEvent[i], totalResRepository.cke[i]));
        CHK_RET(ResetRepResourceTemplate(totalRepRes.blockCompletedEvent[i], totalResRepository.blockCke[i]));
        CHK_RET(ResetRepResourceTemplate(
            totalRepRes.localNotify[i], totalResRepository.blockCke[i],
            totalRepRes.blockCompletedEvent[i].size())); // 两类资源都使用cke，需要调整起始分配位置
        CHK_RET(ResetRepResourceTemplate(totalRepRes.address[i], totalResRepository.gsa[i]));
        CHK_RET(ResetRepResourceTemplate(totalRepRes.blockAddress[i], totalResRepository.blockGsa[i]));
        CHK_RET(ResetRepResourceTemplate(totalRepRes.variable[i], totalResRepository.xn[i]));
        CHK_RET(ResetRepResourceTemplate(totalRepRes.continuousVariable[i], totalResRepository.blockXn[i]));
    }
    return HcclResult::HCCL_SUCCESS;
}

using DieResInfos = std::array<std::vector<ResInfo>, CCU_MAX_IODIE_NUM>;
static HcclResult SaveKernelMissionInfo(CcuKernel* kernel, const DieResInfos& missionId, const int32_t userDevId)
{
    const uint32_t dieId = kernel->GetDieId();
    uint32_t missionKey{0};
    CHK_RET(CcuDevMgrImp::GetMissionKey(userDevId, dieId, missionKey));

    HCCL_INFO("[CcuKernelMgr][%s] userDevId[%d] dieId[%u]", __func__, userDevId, dieId);

    kernel->SetMissionKey(missionKey);
    // 从missionId中获取一个元素并从missionId中删除，当前应只有一个元素，且无需删除
    if (missionId[dieId].empty()) {
        HCCL_ERROR("[%s] failed, userDevId[%d] dieId[%u] do not have missions.", __func__, userDevId, dieId);
        return HcclResult::HCCL_E_INTERNAL;
    }

    kernel->SetMissionId(missionId[dieId].back().startId);
    return HcclResult::HCCL_SUCCESS;
}

static void DumpResRepositoryInfo(const CcuResRepository& resRepo)
{
    for (uint32_t i = 0; i < CCU_MAX_IODIE_NUM; i++) {
        if (resRepo.ms[i].size() != 0 || resRepo.blockMs[i].size() != 0 || resRepo.cke[i].size() != 0
            || resRepo.blockCke[i].size() != 0 || resRepo.loopEngine[i].size() != 0
            || resRepo.blockLoopEngine[i].size() != 0 || resRepo.gsa[i].size() != 0 || resRepo.blockGsa[i].size() != 0
            || resRepo.xn[i].size() != 0 || resRepo.blockXn[i].size() != 0 || resRepo.mission.mission[i].size() != 0) {
            HCCL_INFO(
                "DumpResRepository: dieId[%u], ms size[%u], blockMs size[%u], cke size[%u], blockCke size[%u], "
                "loopEngine size[%u], blockLoopEngine size[%u], gsa size[%u], blockGsa size[%u], xn size[%u], "
                "block xn size[%u], mission size[%u]",
                i, resRepo.ms[i].size(), resRepo.blockMs[i].size(), resRepo.cke[i].size(), resRepo.blockCke[i].size(),
                resRepo.loopEngine[i].size(), resRepo.blockLoopEngine[i].size(), resRepo.gsa[i].size(),
                resRepo.blockGsa[i].size(), resRepo.xn[i].size(), resRepo.blockXn[i].size(),
                resRepo.mission.mission[i].size());
        }
    }
}

inline void ExpandResInfo(std::vector<ResInfo>& expendResInfos, const std::vector<ResInfo>& resInfos)
{
    // 将resInfo中的资源信息还原为单个资源粒度
    for (auto& resInfo : resInfos) {
        for (uint32_t id = 0; id < resInfo.num; id++) {
            expendResInfos.push_back({(resInfo.startId + id), {1}});
        }
    }
}

static CcuResult ExpandResRepo(CcuResRepository& totalRes, const CcuResRepository& tmpResRepository)
{
    // 合并获取的所持有的资源信息, 按照类型合并资源总和到totalRes中
    for (u32 i = 0; i < CCU_MAX_IODIE_NUM; i++) {
        ExpandResInfo(totalRes.ms[i], tmpResRepository.ms[i]);
        ExpandResInfo(totalRes.blockMs[i], tmpResRepository.blockMs[i]);
        ExpandResInfo(totalRes.loopEngine[i], tmpResRepository.loopEngine[i]);
        ExpandResInfo(totalRes.blockLoopEngine[i], tmpResRepository.blockLoopEngine[i]);
        ExpandResInfo(totalRes.cke[i], tmpResRepository.cke[i]);
        ExpandResInfo(totalRes.blockCke[i], tmpResRepository.blockCke[i]);
        ExpandResInfo(totalRes.gsa[i], tmpResRepository.gsa[i]);
        ExpandResInfo(totalRes.blockGsa[i], tmpResRepository.blockGsa[i]);
        ExpandResInfo(totalRes.xn[i], tmpResRepository.xn[i]);
        ExpandResInfo(totalRes.blockXn[i], tmpResRepository.blockXn[i]);
        ExpandResInfo(totalRes.mission.mission[i], tmpResRepository.mission.mission[i]);
    }
    DumpResRepositoryInfo(totalRes);
    return CcuResult::CCU_SUCCESS;
}

template <typename T>
static HcclResult MergeExportedResources(
    const std::unordered_map<std::string, T>& inputRes, std::unordered_map<std::string, T>& outputRes)
{
    for (const auto& item : inputRes) {
        const auto& resTag = item.first;
        if (outputRes.find(resTag) != outputRes.end()) {
            HCCL_ERROR(
                "[CcuKernelMgr][%s] failed, exported resource tag[%s] is already existed, "
                "please check.",
                __func__, resTag);
            return HcclResult::HCCL_E_PARA;
        }

        outputRes.insert(item);
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename T>
static HcclResult ResetImportedResources(
    std::unordered_map<std::string, T>& importedRes, const std::unordered_map<std::string, T>& exportedRes)
{
    for (auto& item : importedRes) {
        const auto& resTag = item.first;
        const auto& iter = exportedRes.find(resTag);
        if (iter == exportedRes.end()) {
            HCCL_ERROR("[CcuKernelMgr][%s] failed to find exported resources by tag[%s].", __func__, resTag.c_str());
            return HcclResult::HCCL_E_NOT_FOUND;
        }

        item.second.Reset(iter->second.Id(), iter->second.DieId());
    }

    return HcclResult::HCCL_SUCCESS;
}

static HcclResult ProcessInterCtxRes(const std::vector<CcuKernel*>& kernels)
{
    std::unordered_map<std::string, CcuRep::LocalNotify> totalExportedNotifies;

    for (const auto kernel : kernels) {
        const auto& exportedRes = kernel->GetExportedRes();
        CHK_RET(MergeExportedResources(exportedRes.sharedNotifies, totalExportedNotifies));
    }

    for (auto kernel : kernels) {
        auto& importedRes = kernel->GetImportedRes();
        CHK_RET(ResetImportedResources(importedRes.sharedNotifies, totalExportedNotifies));
    }

    return HcclResult::HCCL_SUCCESS;
}

static HcclResult TransRepResToPhyRes(const std::vector<CcuKernel*>& kernels, const int32_t userDevId)
{
    for (auto kernel : kernels) {
        const auto& totalResRepository = kernel->GetResRepository();
        auto& totalRepRes = kernel->GetResource();

        // 将ccu kernel持有的物理资源赋给资源对象
        CcuResRepository expandedResRepo{};
        ExpandResRepo(expandedResRepo, totalResRepository);
        CHK_RET(ResetRepResourceToResRepository(totalRepRes, expandedResRepo));

        CHK_RET(SaveKernelMissionInfo(kernel, totalResRepository.mission.mission, userDevId));
    }

    CHK_RET(ProcessInterCtxRes(kernels));

    return HcclResult::HCCL_SUCCESS;
}

CcuResult CcuKernelMgr::Translate(const std::vector<CcuKernelHandle>& kernelHandles)
{
    if (kernelHandles.empty()) {
        HCCL_INFO("[CcuKernelMgr][%s] passed, kernelHandles are empty.", __func__);
        return CcuResult::CCU_SUCCESS;
    }

    std::vector<CcuKernel*> kernels{};
    std::unique_lock<std::mutex> mapLock(kernelMapMutex_);
    for (const auto kernelHandle : kernelHandles) {
        const auto& iter = kernelMap_.find(kernelHandle);
        if (iter == kernelMap_.end()) {
            HCCL_ERROR(
                "[CcuKernelMgr][%s] failed to find kernel by ccu kernel handle[0x%llx].", __func__, kernelHandle);
            return CcuResult::CCU_E_NOT_FOUND;
        }

        kernels.push_back(iter->second.get());
    }
    mapLock.unlock();

    constexpr bool isFuncBlock = false; // 当前不支持MC2

    std::unique_lock<std::mutex> translateLock(translateMutex_);
    CCU_CHK_RET(TransRepResToPhyRes(kernels, userDevId_));
    CCU_CHK_RET(TransRepSequenceToMicrocode(kernels, isFuncBlock));

    for (auto& referenceMgrMap : referenceMgrs) {
        for (auto& referenceMgr : referenceMgrMap.second) {
            referenceMgr.second->ClearRepReference();
        }
    }
    return CcuResult::CCU_SUCCESS;
}

static HcclResult ReleaseInstrRes(CcuKernel* kernel, const int32_t userDevId)
{
    uint32_t instrCount = 0;
    CHK_RET(ComputeKernelInstrRegionSize(kernel, userDevId, instrCount));
    const ResInfo insInfo{kernel->GetInstrId(), instrCount};
    const uint8_t dieId = static_cast<uint8_t>(kernel->GetDieId());
    HCCL_INFO(
        "[CcuKernelMgr][%s] userDevId[%d], dieId[%u], startId[%u], count[%u]", __func__, userDevId, dieId,
        insInfo.startId, insInfo.num);
    CHK_RET(CcuDevMgrImp::ReleaseIns(userDevId, dieId, insInfo));

    return HcclResult::HCCL_SUCCESS;
}

CcuResult CcuKernelMgr::UnRegister(CcuKernelHandle kernelHandle)
{
    std::unique_lock<std::mutex> lock(kernelMapMutex_);

    // 校验kernelMap_中是否存在executorId对应的kernel
    auto it = kernelMap_.find(kernelHandle);
    CHK_PRT_RET(
        it == kernelMap_.end(),
        HCCL_ERROR("[CcuKernelMgr][%s] kernelHandle [%llu] does not exist", __func__, kernelHandle),
        CcuResult::CCU_E_NOT_FOUND);

    auto kernel = it->second.get();
    CCU_CHK_RET(ReleaseInstrRes(kernel, userDevId_));
    kernelMap_.erase(kernelHandle);
    return CcuResult::CCU_SUCCESS;
}

HcclResult CcuKernelMgr::GetResPackTotalResRepository(
    const CcuKernelMgr::CcuTranslatResPack& resPack, CcuResRepository& totalRes) const
{
    CcuResRepository tmpResRepository{};
    // 获取通信域当前所持有的资源
    for (CcuResHandle resHandle : resPack.handles) {
        CHK_RET(CcuDevMgrImp::GetResource(userDevId_, resHandle, tmpResRepository));
        ExpandResRepo(totalRes, tmpResRepository);
        HCCL_INFO("[%s] succeed, userDevId[%d] resHandle[%p].", __func__, userDevId_, resHandle);
    }
    return HcclResult::HCCL_SUCCESS;
}

static void MergeCcuResReq(CcuResReq& resReqA, const CcuResReq& resReqB)
{
    // 合并获取的所持有的资源信息, 按照类型合并资源总和到totalRes的第0个vector中
    for (uint32_t i = 0; i < CCU_MAX_IODIE_NUM; i++) {
        resReqA.msReq[i] += resReqB.msReq[i];
        resReqA.blockMsReq[i] += resReqB.blockMsReq[i];
        resReqA.ckeReq[i] += resReqB.ckeReq[i];
        resReqA.blockCkeReq[i] += resReqB.blockCkeReq[i];
        resReqA.loopEngineReq[i] += resReqB.loopEngineReq[i];
        resReqA.blockLoopEngineReq[i] += resReqB.blockLoopEngineReq[i];
        resReqA.gsaReq[i] += resReqB.gsaReq[i];
        resReqA.blockGsaReq[i] += resReqB.blockGsaReq[i];
        resReqA.xnReq[i] += resReqB.xnReq[i];
        resReqA.blockXnReq[i] += resReqB.blockXnReq[i];
        resReqA.missionReq.req[i] += resReqB.missionReq.req[i];

        if (resReqB.missionReq.req[i] > 0) {
            resReqA.missionReq.reqType = resReqB.missionReq.reqType;
        }
    }
}

HcclResult CcuKernelMgr::InstantiationTranslator(const uint16_t dieId)
{
    if (translators.find(dieId) != translators.end()) {
        return HcclResult::HCCL_SUCCESS;
    }

    std::array<uint16_t, CCU_MAX_IODIE_NUM> tmpChannelId{};
    uint32_t channelId = 0;
    // 获取innerDieChannelId
    auto ret = CcuDevMgrImp::GetLoopChannelId(userDevId_, dieId, dieId, channelId);
    CHK_RET(ret);

    tmpChannelId[0] = channelId;
    // 获取interDieChannelId
    uint8_t dstDieId = ((dieId == 0) ? 1 : 0);
    ret = CcuDevMgrImp::GetLoopChannelId(userDevId_, dieId, dstDieId, channelId);
    CHK_RET(ret);
    tmpChannelId[1] = channelId;

    uint64_t tokenId = 0;
    uint64_t tokenValue = 0;
    ret = CcuDevMgrImp::GetCcuResourceSpaceTokenInfo(userDevId_, dieId, tokenId, tokenValue);
    CHK_RET(ret);

    std::pair<uint64_t, uint64_t> ccuTokenInfo(tokenId, tokenValue);
    Hccl::DevBuffer tmpDevMem{1}; // 临时申请device hbm内存用于查询token信息
    auto hbmTokenInfo = hcomm::CcuRep::GetTokenInfo(tmpDevMem.GetAddr(), 1);

    CcuResReq totalResReq{};
    // 实例化CcuRepReferenceManager和CcuRepTranslator，并为CcuRepReferenceManager绑定物理资源
    for (uint32_t i = 0; i < 16; i++) { // mgr有16个
        referenceMgrs[dieId][i] = std::make_shared<hcomm::CcuRep::CcuRepReferenceManager>(dieId);
        translators[dieId][i] = std::make_shared<hcomm::CcuRep::CcuRepTranslator>(
            userDevId_, dieId, referenceMgrs[dieId][i], tmpChannelId, ccuTokenInfo, hbmTokenInfo);

        // 统计&合并refManager和translator所有资源REQ
        auto refMangerResReq = CcuRep::CcuRepReferenceManager::GetResReq(dieId);
        auto transLatorResReq = CcuRep::CcuRepTranslator::GetResReq(userDevId_, dieId);
        MergeCcuResReq(totalResReq, refMangerResReq);
        MergeCcuResReq(totalResReq, transLatorResReq);
    }
    DumpResReqInfo(totalResReq);

    // 为refManager和translator申请物理资源
    CcuResHandle handle;
    CHK_RET(CcuDevMgrImp::AllocResHandle(userDevId_, totalResReq, handle));
    translatorResPack.handles.push_back(handle);

    CcuRepResource translatorRepRes;
    for (uint32_t i = 0; i < 16; i++) { // mgr有16个
        referenceMgrs[dieId][i]->GetRes(translatorRepRes);
        translators[dieId][i]->GetRes(translatorRepRes);
    }

    CcuResRepository totalResRepository;
    CHK_RET(GetResPackTotalResRepository(translatorResPack, totalResRepository));
    // 将kernel中的rep虚拟资源按类型进行和CCU物理资源映射
    CHK_RET(ResetRepResourceToResRepository(translatorRepRes, totalResRepository));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelMgr::LoadInstruction(const CcuRep::CcuInstrInfo& instrInfo, const uint32_t dieId)
{
    const uint64_t instrInfoSize = instrInfo.instrVec.size() * sizeof(hcomm::CcuRep::CcuInstr);

    if (!instructionLoadDevMem_) {
        uint32_t instrNum = 0;
        CHK_RET(CcuDevMgrImp::GetResSpecsInstructionNum(userDevId_, 0, instrNum));
        HCCL_INFO("[CcuKernelMgr]LoadInstruction: userDevId[%d], instrNum[%u]", userDevId_, instrNum);
        CHK_RET(hrtMalloc(&instructionLoadDevMem_, instrNum * sizeof(hcomm::CcuRep::CcuInstr)));
    }

    CHK_RET(hrtMemcpy(
        instructionLoadDevMem_, instrInfoSize, instrInfo.instrVec.data(), instrInfoSize,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    uint32_t devPhyId = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(userDevId_), devPhyId));

    CustomChannelInfoIn inBuff{};
    CustomChannelInfoOut outBuff{};

    // 设置操作码和通道数据
    inBuff.op = CcuOpcodeType::CCU_U_OP_SET_INSTRUCTION;
    inBuff.offsetStartIdx = instrInfo.startInstrId;
    inBuff.data.dataInfo.udieIdx = dieId;
    inBuff.data.dataInfo.dataArraySize = 1;
    inBuff.data.dataInfo.dataLen = instrInfoSize;

    CcuDataTypeUnion tmp{};
    tmp.insinfo.resourceAddr = reinterpret_cast<uint64_t>(instructionLoadDevMem_);
    (void)memcpy_s(inBuff.data.dataInfo.dataArray, sizeof(CcuDataTypeUnion), &tmp, sizeof(CcuDataTypeUnion));

    auto ret = HccpRaTlvCcuCustomChannel(userDevId_, static_cast<void*>(&inBuff), static_cast<void*>(&outBuff));
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CcuResSpecifications][%s] failed to call ccu driver, "
            "userDevId[%d] devPhyId[%u] dieId[%d] op[%s] ret[%d].",
            __func__, userDevId_, devPhyId, dieId, "SET_INSTRUCTION", ret);
        return ret;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelMgr::TransRepSequenceToMicrocode(const std::vector<CcuKernel*>& kernels, bool isFuncBlock)
{
    for (auto kernel : kernels) {
        const uint32_t dieId = kernel->GetDieId();
        const uint32_t missionId = kernel->GetMissionId();

        EXCEPTION_HANDLE_BEGIN
        const auto& instrInfo = translators[dieId][missionId]->Translate(
            kernel, kernel->GetRepSequence(), kernel->GetInstrId(), isFuncBlock);

        // 后端优化会插 NOP 改变指令数; 按与申请同一口径校验不越界, 把静默越界变成快速失败.
        uint32_t regionSize = 0;
        CHK_RET(ComputeKernelInstrRegionSize(kernel, userDevId_, regionSize));
        CHK_PRT_RET(
            instrInfo.instrVec.size() > regionSize,
            HCCL_ERROR(
                "[CcuKernelMgr][%s] optimized instr count[%zu] exceeds reserved region size[%u], "
                "dieId[%u] startId[%u]. Check cke reservation / backend optimizer NOP insertion.",
                __func__, instrInfo.instrVec.size(), regionSize, dieId, kernel->GetInstrId()),
            HcclResult::HCCL_E_INTERNAL);

        CHK_RET(LoadInstruction(instrInfo, dieId));

        kernel->SetCcuInstrInfo(instrInfo); // 指令下发成功后可以对kernel进行launch
        EXCEPTION_HANDLE_END
    }

    return HcclResult::HCCL_SUCCESS;
}

CcuKernel* CcuKernelMgr::GetKernel(CcuKernelHandle kernelHandle)
{
    std::unique_lock<std::mutex> lock(kernelMapMutex_);
    auto it = kernelMap_.find(kernelHandle);
    if (it == kernelMap_.end()) {
        HCCL_ERROR("[CcuKernelMgr][%s] handle[%llx] is not existed.", __func__, kernelHandle);
        return nullptr;
    }

    return it->second.get();
}

CcuResult CcuKernelMgr::GetCcuKernelInfo(CcuKernelHandle kernelHandle, CcuKernelInfo& info)
{
    std::unique_lock<std::mutex> lock(kernelMapMutex_);
    auto it = kernelMap_.find(kernelHandle);
    if (it == kernelMap_.end()) {
        HCCL_ERROR("[CcuKernelMgr][%s] handle[%llx] is not existed.", __func__, kernelHandle);
        return CcuResult::CCU_E_NOT_FOUND;
    }
    // 在锁内填充 info，避免裸指针逃逸锁后 kernel 被 UnRegister 导致 use-after-free
    CCU_CHK_RET(it->second->GetCcuKernelInfo(info));
    return CcuResult::CCU_SUCCESS;
}

CcuKernel* CcuKernelMgr::GetCurrentKernel() { return currKernel_.get(); }

} // namespace hcomm
