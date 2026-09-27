/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_base.h"
#include <algorithm>
#include <future>
#include <map>
#include <new>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <hccl/hccl_types.h>

#include "hccl/base.h"
#include "param_check_pub.h"
#include "rank_consistentcy_checker.h"
#include "externalinput_pub.h"
#include "env_config.h"
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
#include "coll_comm.h"
#endif
#include "../common/src/topo/topoinfo_detect.h"
#include "../common/src/topo/topoinfo_detect_scalable.h"
#include "../common/src/topo/topoinfo_ranktable_partition.h"
#include "../common/src/state_guard.h"
#include "sal_pub.h"
#include "profiling_manager_pub.h"
#include "adapter_prof.h"
#include "adapter_rts_common.h"
#include "device_capacity.h"
#include "mem_host_pub.h"
#include "hcom_common.h"
#include "comm_config_pub.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mmpa_api.h"
#include "aicpu_operator_pub.h"
#include "../nslbdp/hccl_nslbdp.h"
#include "hccl_group.h"
#include "hostdpu/dpu_kernel_entrance.h"
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
#include "coll_comm_mgr.h"
#include "env_ub_config.h"
#endif

#define DOUBLE_SIZE 2

using namespace std;
using namespace hccl;

typedef int32_t Callback(uint64_t, int32_t);

const std::string HCCL_ALLTOALL = "ALLTOALL";
const std::string HCCL_ALLTOALLV = "ALLTOALLV";
const std::string HCCL_ALLTOALLVC = "ALLTOALLVC";

HcclResult CallMsprofReportHostApi(
    hccl::hcclComm* hcclComm, HcclCMDType cmdType, uint64_t beginTime, u64 count, HcclDataType dataType,
    const std::string& tag)
{
    if (GetIfProfile()) {
        AlgType algType;
        if (cmdType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
            algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_PAIRWISE;
            algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RESERVED;
        } else if (cmdType == HcclCMDType::HCCL_CMD_SEND || cmdType == HcclCMDType::HCCL_CMD_RECEIVE) {
            algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
            algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RESERVED;
        } else {
            CHK_RET(hcclComm->GetAlgType(algType, cmdType));
        }

        u32 numBlocks = 0;
        hcclComm->GetNumBlocks(numBlocks);
        uint64_t groupName = hrtMsprofGetHashId(hcclComm->GetIdentifier().c_str(), hcclComm->GetIdentifier().length());
        HCCL_INFO("[%s] groupName[%llu], groupNameStr[%s]", __func__, groupName, hcclComm->GetIdentifier().c_str());
        CHK_RET_AND_PRINT_IDE(
            ProfilingManagerPub::CallMsprofReportHostApi(
                cmdType, beginTime, count, dataType, algType, groupName, numBlocks),
            tag.c_str());
    }
    hcclComm->SetAivCoreLimit(0);
    return HCCL_SUCCESS;
}

thread_local s32 g_hcclDeviceId = INVALID_INT;
std::mutex g_opHcomOneSideMutex{};

HcclResult HcclGetDeviceId(void)
{
    if (g_hcclDeviceId == INVALID_INT) {
        CHK_PRT_RET(
            hrtGetDevice(&g_hcclDeviceId) != HCCL_SUCCESS,
            HCCL_WARNING("[HcclGetDeviceId] get fail deviceLogicId[%d]", g_hcclDeviceId), HCCL_E_INTERNAL);
    }
    u32 maxDeviceNum;
    CHK_RET(GetMaxDevNum(maxDeviceNum));
    CHK_PRT_RET(
        static_cast<u32>(g_hcclDeviceId) >= maxDeviceNum,
        HCCL_WARNING(
            "[HcclGetDeviceId]deviceLogicId[%d] is bigger than maxDeviceNum:[%u]", g_hcclDeviceId, maxDeviceNum),
        HCCL_E_INTERNAL);
    HCCL_INFO("[HcclGetDeviceId] deviceLogicId[%d] ", g_hcclDeviceId);
    return HCCL_SUCCESS;
}

s32 HcclGetThreadDeviceId()
{
    CHK_PRT_RET(
        HcclGetDeviceId() != HCCL_SUCCESS,
        HCCL_WARNING("[HcclGetThreadDeviceId] get fail deviceLogicId[%d]", g_hcclDeviceId), INVALID_INT);
    return g_hcclDeviceId;
}

HcclResult
GetDeviceComm(uint32_t ndev, const HcclRootInfo& rootHandle, const s32 rank, const s32 logicDeviceId, HcclComm& comm)
{
    // 给当前线程添加名字
    SetThreadName("Hccl_GetDevComm");

    CHK_PRT_RET(
        hrtSetDevice(logicDeviceId) != HCCL_SUCCESS,
        HCCL_ERROR("[GetDeviceComm] set fail logicDeviceId[%d]", logicDeviceId), HCCL_E_INTERNAL);
    HcclResult ret = HcclCommInitRootInfo(ndev, &rootHandle, rank, &comm);
    if (ret != HCCL_SUCCESS || comm == nullptr) {
        comm = nullptr;
        HCCL_ERROR("[GetDeviceComm] rank[%d] Get device comm failed!", rank);
        CHK_PRT_RET(
            hrtResetDevice(logicDeviceId) != HCCL_SUCCESS,
            HCCL_ERROR("[GetDeviceComm] reset fail logicDeviceId[%d]", logicDeviceId), HCCL_E_INTERNAL);
        return ret;
    }
    hcclComm* pComm = static_cast<hcclComm*>(comm);
    pComm->ResetDeviceEnable();
    return HCCL_SUCCESS;
}

HcclResult HcclGetCommAll(uint32_t ndev, int32_t* devices, HcclComm* comms)
{
    // 入参校验
    CHK_PRT_RET(ndev == 0, HCCL_ERROR("[HcclGetCommAll] ndev is invalid, ndev[%u]", ndev), HCCL_E_PARA);
    CHK_PTR_NULL(comms);
    CHK_PTR_NULL(devices);

    // 给当前线程添加名字
    SetThreadName("Hccl_GetCommAll");

    CHK_PRT_RET(
        hrtSetDevice(devices[0]) != HCCL_SUCCESS, HCCL_ERROR("[HcclGetCommAll] set fail devices[0][%d]", devices[0]),
        HCCL_E_INTERNAL);

    // 获取通信域之前, 先把所有通信域设置为空
    for (uint32_t i = 0; i < ndev; i++) {
        comms[i] = nullptr;
    }

    HcclRootInfo rootHandle;
    CHK_RET(HcclGetRootInfo(&rootHandle));

    std::vector<std::unique_ptr<std::thread>> threads(ndev);
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        threads[rankId].reset(new (std::nothrow) std::thread(
            &GetDeviceComm, ndev, std::ref(rootHandle), rankId, devices[rankId], std::ref(comms[rankId])));
        CHK_PRT_RET(!threads[rankId], HCCL_ERROR("[HcclGetCommAll]threads[%u] reset failed ", rankId), HCCL_E_INTERNAL);
    }
    for (uint32_t i = 0; i < ndev; i++) {
        threads[i]->join();
    }

    // 如果任何一个通信域初始化失败，将所有已经成功创建的通信域销毁
    bool isFailed = false;
    for (uint32_t i = 0; i < ndev; ++i) {
        if (comms[i] == nullptr) {
            HCCL_ERROR("[HcclGetCommAll] rank[%u] get comm failed!", i);
            isFailed = true;
            break;
        }
    }
    if (isFailed) {
        for (uint32_t i = 0; i < ndev; ++i) {
            if (comms[i] != nullptr) {
                (void)HcclCommDestroy(comms[i]);
            }
        }
        return HCCL_E_INTERNAL;
    }

    CHK_PRT_RET(
        hrtResetDevice(devices[0]) != HCCL_SUCCESS,
        HCCL_ERROR("[HcclGetCommAll] reset fail devices[0][%d]", devices[0]), HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

HcclResult GetDeviceCollComm(
    [[maybe_unused]] const s32 rank, [[maybe_unused]] HcclCommConfig* config, [[maybe_unused]] const s32 logicDeviceId,
    [[maybe_unused]] HcclComm& comm)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    // 给当前线程添加名字
    SetThreadName("Hccl_GetDeviceCollComm");

    CHK_PRT_RET(
        hrtSetDevice(logicDeviceId) != HCCL_SUCCESS,
        HCCL_ERROR("[GetDeviceCollComm] set fail logicDeviceId[%d]", logicDeviceId), HCCL_E_INTERNAL);

    HcclComm newComm;
    HcclResult ret = HcclCommInitCollComm(rank, &comm, config, &newComm);
    if (ret != HCCL_SUCCESS || newComm == nullptr) {
        newComm = nullptr;
        HcclCommDestroyV2(comm);
        comm = nullptr;
        HCCL_ERROR("[GetDeviceCollComm] rank[%d] Get device coll comm failed!", rank);
        CHK_PRT_RET(
            hrtResetDevice(logicDeviceId) != HCCL_SUCCESS,
            HCCL_ERROR("[GetDeviceCollComm] reset fail logicDeviceId[%d]", logicDeviceId), HCCL_E_INTERNAL);
        return ret;
    }
    comm = newComm;
#endif
    return HCCL_SUCCESS;
}

HcclResult HcclGetCollCommAll(uint32_t ndev, int32_t* devices, HcclComm* comms)
{
    // 入参校验
    CHK_PRT_RET(ndev == 0, HCCL_ERROR("[HcclGetCollCommAll] ndev is invalid, ndev[%u]", ndev), HCCL_E_PARA);
    CHK_PTR_NULL(comms);
    CHK_PTR_NULL(devices);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    // 给当前线程添加名字
    SetThreadName("Hccl_GetCollCommAll");

    CHK_PRT_RET(
        hrtSetDevice(devices[0]) != HCCL_SUCCESS,
        HCCL_ERROR("[HcclGetCollCommAll] set fail devices[0][%d]", devices[0]), HCCL_E_INTERNAL);

    constexpr HcclCommConfig* config = nullptr;
    std::vector<std::unique_ptr<std::thread>> threads(ndev);
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        threads[rankId].reset(new (std::nothrow) std::thread(
            &GetDeviceCollComm, rankId, config, devices[rankId], std::ref(comms[rankId])));
        CHK_PRT_RET(
            !threads[rankId], HCCL_ERROR("[HcclGetCollCommAll]threads[%u] reset failed ", rankId), HCCL_E_INTERNAL);
    }
    for (uint32_t i = 0; i < ndev; i++) {
        threads[i]->join();
    }

    // 如果任何一个通信域转换失败，将所有已经成功创建的通信域销毁
    bool isFailed = false;
    for (uint32_t i = 0; i < ndev; ++i) {
        if (comms[i] == nullptr) {
            HCCL_ERROR("[HcclGetCollCommAll] rank[%u] get comm failed!", i);
            isFailed = true;
            break;
        }
    }
    if (isFailed) {
        for (uint32_t i = 0; i < ndev; ++i) {
            if (comms[i] != nullptr) {
                (void)HcclCommDestroy(comms[i]);
            }
        }
        return HCCL_E_INTERNAL;
    }

    CHK_PRT_RET(
        hrtResetDevice(devices[0]) != HCCL_SUCCESS,
        HCCL_ERROR("[HcclGetCollCommAll] reset fail devices[0][%d]", devices[0]), HCCL_E_INTERNAL);
#endif
    return HCCL_SUCCESS;
}

HcclResult
HcclCollCommInitAll([[maybe_unused]] uint32_t ndev, [[maybe_unused]] int32_t* devices, [[maybe_unused]] HcclComm* comms)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HcclUs startut = TIME_NOW();
    std::string devicesStr;
    for (size_t i = 0; i < ndev; ++i) {
        std::string deviceStr = std::to_string(devices[i]);
        devicesStr += deviceStr;
        if (i != ndev - 1) {
            devicesStr += " ";
        }
    }
    HCCL_RUN_INFO("Entry-HcclCollCommInitAll, ndev:[%u], devices:[%s].", ndev, devicesStr.c_str());

    std::future<HcclResult> threadResult;
    std::unique_ptr<std::thread> getCommThread;
    getCommThread.reset(new (std::nothrow) std::thread([=, &threadResult]() {
        threadResult = std::async(std::launch::async, HcclGetCollCommAll, ndev, devices, comms);
    }));
    CHK_PRT_RET(!getCommThread, HCCL_ERROR("[HcclCollCommInitAll]thread reset failed "), HCCL_E_INTERNAL);
    getCommThread->join();

    HcclResult ret = threadResult.get();
    if (ret != HCCL_SUCCESS) {
        for (uint32_t i = 0; i < ndev; ++i) {
            if (comms[i] != nullptr) {
                (void)HcclCommDestroy(comms[i]);
                comms[i] = nullptr;
            }
        }
        HCCL_ERROR("HcclCollCommInitAll failed! threadResult[%d]", ret);
        return ret;
    }
    s32 deviceLogicId = HcclGetThreadDeviceId();
    HCCL_RUN_INFO(
        "HcclCollCommInitAll success, take time [%lld]us, deviceLogicId[%d].", DURATION_US(TIME_NOW() - startut),
        deviceLogicId);
#endif
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitAll(uint32_t ndev, int32_t* devices, HcclComm* comms)
{
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));

    // 入参校验
    CHK_PRT_RET(ndev <= 0, HCCL_ERROR("[HcclCommInitAll] ndev is invalid, ndev[%u]", ndev), HCCL_E_PARA);
    CHK_PTR_NULL(comms);
    CHK_PTR_NULL(devices);

    // 判断设备List中是否有重复id,报错退出
    set<int32_t> devSet(devices, devices + ndev);
    uint32_t devSetSize = devSet.size();
    CHK_PRT_RET(
        (devSetSize != ndev),
        HCCL_ERROR(
            "[HcclCommInitAll] Duplicate device id exist in the device list. devSetSize:[%u], ndev:[%u]", devSetSize,
            ndev),
        HCCL_E_PARA);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        CHK_RET(HcclCommInitAllV2(ndev, devices, comms));
        CHK_RET(HcclCollCommInitAll(ndev, devices, comms));
        return HCCL_SUCCESS;
    }());
#endif
    std::future<HcclResult> threadResult;
    std::unique_ptr<std::thread> getCommThread;
    getCommThread.reset(new (std::nothrow) std::thread([=, &threadResult]() {
        threadResult = std::async(std::launch::async, HcclGetCommAll, ndev, devices, comms);
    }));
    CHK_PRT_RET(!getCommThread, HCCL_ERROR("[HcclCommInitAll]thread reset failed "), HCCL_E_INTERNAL);
    getCommThread->join();

    HcclResult ret = threadResult.get();
    if (ret != HCCL_SUCCESS) {
        for (uint32_t i = 0; i < ndev; ++i) {
            if (comms[i] != nullptr) {
                (void)HcclCommDestroy(comms[i]);
                comms[i] = nullptr;
            }
        }
        HCCL_ERROR("HcclCommInitAll failed! threadResult[%d]", ret);
        return ret;
    }
    HCCL_RUN_INFO(
        "HcclCommInitAll success, take time [%lld]us, deviceLogicId[%d]", DURATION_US(TIME_NOW() - startut),
        deviceLogicId);
    return HCCL_SUCCESS;
}

std::unordered_map<s32, std::unordered_map<std::string, std::shared_ptr<HcclOpInfoCtx>>> g_oneSidedCommHcomInfos;
std::set<HcclComm> g_oneSidedCommSet;

/* 仅提供判断功能, 调用前需校验参数有效性*/
bool IsOneSidedComm(HcclComm comm) { return g_oneSidedCommSet.find(comm) != g_oneSidedCommSet.end(); }

/* 仅提供判断功能, 调用前需校验参数有效性*/
bool IsCommNameExistInOneSidedComms(s32 deviceLogicId, const std::string& commName)
{
    bool exist = g_oneSidedCommHcomInfos.count(deviceLogicId) != 0
                 && g_oneSidedCommHcomInfos[deviceLogicId].count(commName) != 0;
    if (exist && g_oneSidedCommHcomInfos[deviceLogicId][commName]->isUsed) {
        return true;
    }
    return false;
}

HcclResult DeInitOneSidedHcomInfo(s32 deviceLogicId, const std::string& commName)
{
    CHK_PRT_RET(
        deviceLogicId == INVALID_INT, HCCL_ERROR("[HcclCommDestroy][DeInitOneSidedHcomInfo] deviceLogicId is error."),
        HCCL_E_PARA);
    CHK_PRT_RET(
        commName.empty(), HCCL_ERROR("[HcclCommDestroy][DeInitOneSidedHcomInfo] commName is error."), HCCL_E_PARA);
    g_oneSidedCommHcomInfos[deviceLogicId].erase(commName);
    return HCCL_SUCCESS;
}

/*
 * g_oneSidedCommHcomInfos 初始化
 * s32 deviceLogicId
 * const string &commName : 通信域名，用户确保全局唯一
 */
HcclResult InitOneSidedHcomInfo(s32 deviceLogicId, const std::string& commName)
{
    CHK_PRT_RET(
        deviceLogicId == INVALID_INT, HCCL_ERROR("[InitOneSidedHcomInfo] deviceLogicId is error."), HCCL_E_PARA);
    CHK_PRT_RET(commName.empty(), HCCL_ERROR("[InitOneSidedHcomInfo] commName is error."), HCCL_E_PARA);
    // comm name exit && isUsed = true
    bool isCommNameExist = IsCommNameExistInOneSidedComms(deviceLogicId, commName);
    CHK_PRT_RET(isCommNameExist, HCCL_ERROR("[Init][InitOneSidedHcomInfo] comm Name exist."), HCCL_E_PARA);
    // 确保 deviceLogicId 和 commName 的 map 已经被初始化
    if (g_oneSidedCommHcomInfos.find(deviceLogicId) == g_oneSidedCommHcomInfos.end()) {
        g_oneSidedCommHcomInfos[deviceLogicId] = {};
    }
    // comm name not exit
    if (g_oneSidedCommHcomInfos[deviceLogicId].count(commName) == 0) {
        std::shared_ptr<HcclOpInfoCtx> opBaseHcomPtr;
        EXCEPTION_CATCH((opBaseHcomPtr = std::make_shared<HcclOpInfoCtx>()), return HCCL_E_PARA);
        g_oneSidedCommHcomInfos[deviceLogicId][commName] = opBaseHcomPtr;
    }
    // comm name exit && isUsed = False
    g_oneSidedCommHcomInfos[deviceLogicId][commName]->isUsed = true;
    return HCCL_SUCCESS;
}

HcclOpInfoCtx& GetOneSidedOpInfoCtx(s32 deviceLogicId, const std::string& commName)
{
    std::shared_ptr<HcclOpInfoCtx> oneSidedHComPtr = g_oneSidedCommHcomInfos[deviceLogicId][commName];
    return *oneSidedHComPtr;
}

HcclResult CheckOpBasedHcom(HcclOpInfoCtx& opBaseHcom, const uint32_t rank, const CommConfig& commConfig)
{
    /* 防止重复调用初始化 */
    CHK_PRT_RET(
        (opBaseHcom.pComm != nullptr),
        HCCL_ERROR(
            "[Init][CheckOpBasedHcom]errNo[0x%016llx] rank[%u] "
            "op_base hccl multiple initialization",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL), rank),
        HCCL_E_UNAVAIL);
    const std::string commIdentifier = commConfig.GetConfigCommName();
    auto iter = opBaseHcom.opGroup2CommMap.find(commIdentifier);
    CHK_PRT_RET(
        iter != opBaseHcom.opGroup2CommMap.end(),
        HCCL_ERROR(
            "[Init][CheckOpBasedHcom]errNo[0x%016llx] The comm name[%s] already exists in Group2Comm map.",
            HCCL_ERROR_CODE(HCCL_E_PARA), commIdentifier.c_str()),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitCollComm(
    [[maybe_unused]] uint32_t rank, void** commV2, [[maybe_unused]] const HcclCommConfig* config, HcclComm* comm)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(*commV2);
    *comm = nullptr;
    HCCL_INFO("[HcclCommInitCollComm] CollComm init start.");
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HcclUs startut = TIME_NOW();

    // 图模式
    u32 rankNum = 0;
    CHK_RET(HcclGetRankSizeV2(*commV2, &rankNum));
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    CHK_RET(HcclGetCommNameV2(*commV2, commName));
    // 获取cclbuffer
    uintptr_t cclBufferAddr{0};
    std::size_t cclBufferSize{0};
    HcclMemType cclBufferMemType{HcclMemType::HCCL_MEM_TYPE_DEVICE};
    CHK_RET(HcclGetCclBuffer(*commV2, cclBufferAddr, cclBufferSize, cclBufferMemType));
    HcclMem cclBuffer;
    cclBuffer.size = static_cast<uint64_t>(cclBufferSize);
    cclBuffer.type = cclBufferMemType;
    cclBuffer.addr = reinterpret_cast<void*>(cclBufferAddr);
    HcclCommPtr hcclCommPtr = nullptr;
    EXCEPTION_CATCH(
        hcclCommPtr = make_shared<hccl::hcclComm>(cclBufferSize, cclBufferSize, commName), return HCCL_E_PTR);
    CommConfig commConfig(commName);
    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    CHK_RET(CheckOpBasedHcom(opBaseHcom, rank, commConfig));

    void* rankGraph = nullptr;
    CHK_RET(HcclGetRankGraphV2(commV2, &rankGraph));

    // Collcomm初始化
    CHK_RET(hcclCommPtr->InitCollComm(*commV2, rankGraph, rank, cclBuffer, commName, config));

    std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
    opBaseHcom.opGroup2CommMap[hcclCommPtr->GetIdentifier()] = hcclCommPtr;
    lock.unlock();

    HcclResult ret = HcomSetGroupTopoInfo(commName, rankNum);
    if (ret != HCCL_SUCCESS) {
        HcomUnSetGroupTopoInfo(commName);
        std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
        opBaseHcom.opGroup2CommMap.erase(commName);
        lock.unlock();
        CHK_RET(ret);
    }
    *comm = static_cast<HcclComm>(hcclCommPtr.get());
    HCCL_RUN_INFO("[%s] success, take time [%lld]us.", __func__, DURATION_US(TIME_NOW() - startut));
#endif
    return HCCL_SUCCESS;
}

HcclResult InitCommClusterInfo(
    std::string& rankTableM, const uint32_t rank, const CommConfig& commConfig, HcclOpInfoCtx& opBaseHcom,
    HcclComm* comm)
{
    u32 rankTableSize = 0;
    HcclResult ret = HcomCheckRankTable(rankTableM.c_str(), rankTableSize);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][CommClusterInfo]check rankTable string error, rankTableSize [%u].", rankTableSize),
        HCCL_E_PARA);

    const std::string commIdentifier = commConfig.GetConfigCommName();
    opBaseHcom.pComm.reset(new (std::nothrow) hccl::hcclComm(
        commConfig.GetConfigBufferSize(), commConfig.GetConfigBufferSize(), commIdentifier,
        commConfig.GetConfigBufferName()));
    CHK_PTR_NULL(opBaseHcom.pComm);

    /* --------------初始化------------------------- */
    bool errorFlag = false;
    do {
        RankConsistentcyChecker::GetInstance().SetCheckCannVersionSwitch(true); // 打开CANN软件版本校验开关
        ret = InitOtherInfo(opBaseHcom.params, rankTableM.c_str());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] init other Info.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        HCCL_INFO("rootInfo[%s]", opBaseHcom.params.id.internal);

        ret = InitWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] init work flow mode error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        ret = CfgGetClusterInfo(
            rankTableM, to_string(rank), opBaseHcom.params, opBaseHcom.rankTable,
            commConfig.GetConfigInterSuperPodRetryEnable());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Init][CommClusterInfo]errNo[0x%016llx]"
                "info error:rank[%u]",
                HCCL_ERROR_CODE(ret), rank),
            errorFlag = true);

        ret = opBaseHcom.pComm->init(opBaseHcom.params, commConfig, opBaseHcom.rankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] hcclComm init error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置确定性计算配置 */
        ret = opBaseHcom.pComm->SetDeterministicConfig(commConfig.GetConfigDeterministic());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] set deterministic error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        // 设置TC/SL配置
        ret = opBaseHcom.pComm->SetQpQosAttr(commConfig.GetConfigTrafficClass(), commConfig.GetConfigServiceLevel());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Init][CommClusterInfo]errNo[0x%016llx] set TC and SL error or Invalid configuration parameter.",
                HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        if (commConfig.GetConfigJobID() != 0) {
            HCCL_RUN_INFO(
                "[NSLBDP]GetConfigJobID = %llu,GetConfigWorldRankID = %u.", commConfig.GetConfigJobID(),
                commConfig.GetConfigWorldRankID());
            hcclNslbDp::GetInstance().SetGlobalCommTaskId(commConfig.GetConfigJobID());
            hcclNslbDp::GetInstance().SetGlobalCommNodeId(commConfig.GetConfigWorldRankID());
        }

        /* 设置AIV模式 */
        ret = opBaseHcom.pComm->SetAivModeConfig(commConfig.GetConfigAivMode());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] set aivMode error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置only AIV模式 */
        ret = opBaseHcom.pComm->SetOnlyAivModeConfig(commConfig.GetConfigIsOnlyAivMode());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] set only aivMode error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置AICPU */
        ret = opBaseHcom.pComm->SetAicpuUnfoldConfig(commConfig.GetConfigAicpuUnfold());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] set aicpu error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置HcclExecTimeOut */
        ret = opBaseHcom.pComm->SetExecTimeOutConfig(commConfig.GetConfigExecTimeOut());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] set execTimeOut error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置HcclAlgo */
        ret = opBaseHcom.pComm->SetAlgoConfig(commConfig.GetConfigHcclAlgoMap());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] set hcclAlgo error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        ret = ShowRanktableConfigInfo(opBaseHcom.cloudFlag, opBaseHcom.params, opBaseHcom.rankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] put ranktable info error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        /* 设置独立算子参数 */
        ret = opBaseHcom.pComm->SetIndependentOpConfig(commConfig, opBaseHcom.rankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Init][CommClusterInfo]errNo[0x%016llx] set SetIndependentOpConfig error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        // 初始化完成的comm指针赋给出参
        *comm = opBaseHcom.pComm.get();
        std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
        opBaseHcom.opGroup2CommMap[opBaseHcom.pComm->GetIdentifier()] = opBaseHcom.pComm;
        lock.unlock();

        // 特殊场景，当comm name被手动配置为HCCL_WORLD_GROUP时，需要将pComm赋值到hcomInfo.pComm
        if (opBaseHcom.pComm->GetIdentifier() == HCCL_WORLD_GROUP) {
            HcomGetCtxHomInfo().pComm = opBaseHcom.pComm;
        }
        ret = HcomSetGroupTopoInfo(opBaseHcom.pComm->GetIdentifier().c_str(), opBaseHcom.rankTable.rankNum);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] set group topo info error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        ret = opBaseHcom.pComm->InitHccpChannel();
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("InitHccp channel unsuccessful ret:[%u].", ret);
        }
    } while (0);

    if (errorFlag) {
        HCCL_ERROR(
            "[Init][CommClusterInfo]HcclCommInitClusterInfo failed, rankNum[%u], rank[%u], server[%s],"
            "device[%d], return[0x%016llx]",
            opBaseHcom.rankTable.rankNum, rank, opBaseHcom.params.serverId.c_str(), opBaseHcom.params.logicDevId,
            HCCL_ERROR_CODE(ret));
        (void)HcclCommDestroy(opBaseHcom.pComm.get());
        opBaseHcom.pComm = nullptr;
        *comm = nullptr;
        return ret;
    }
    if (hcclNslbDp::GetInstance().GetGlobalCommTaskId() != 0 && hcclNslbDp::GetInstance().InitNetCo() == HCCL_SUCCESS) {
        HCCL_INFO("HCCL try to entry SetGlobalRank_RankTableExit.");
        /* NSLB 填充 表1 表4 */
        CHK_RET(hcclNslbDp::GetInstance().SetCommInfo_RankTableExit(opBaseHcom.rankTable));
        hcclNslbDp::GetInstance().SetGlobalRank_RankTableExit(opBaseHcom.rankTable);
        hcclNslbDp::GetInstance().SendGlobalRankTable(rank);
    }
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    CHK_RET(opBaseHcom.pComm->InitCollCommInner(rank));
#endif
    /* 关键状态记录 */
    HCCL_INFO(
        "%s success, rankNum[%u], rank[%u], server[%s], device[%d].", __func__, opBaseHcom.rankTable.rankNum, rank,
        opBaseHcom.params.serverId.c_str(), opBaseHcom.params.logicDevId);
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitClusterInfoWrapper([[maybe_unused]] struct hcclAsyncJob* job_)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    struct hcclCommInitRankTableAsyncJob* job = static_cast<hcclCommInitRankTableAsyncJob*>(job_);
    uint32_t rank = job->rank;
    HcclComm* comm = job->initComm;
    const char* clusterInfo = job->clusterInfo;
    s32 devId = job->devId;
    HCCL_DEBUG("[HcclCommInitClusterInfoWrapper] Set device devId: %d", devId);
    CHK_PRT_RET(
        hrtSetDevice(devId) != HCCL_SUCCESS, HCCL_ERROR("[HcclCommInitClusterInfoWrapper] set fail device[%d]", devId),
        HCCL_E_INTERNAL);
    HCCL_DEBUG("[HcclCommInitClusterInfoWrapper] Done Set device devId: %d", devId);

    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));
    HCCL_RUN_INFO(
        "Entry-%s: clusterInfo[%s], rank[%u], deviceLogicId[%d].", __func__, clusterInfo, rank, deviceLogicId);
    // 入参合法性校验
    CHK_PTR_NULL(clusterInfo);
    CHK_PTR_NULL(comm);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        void* commV2 = nullptr;
        CHK_RET(HcclCommInitClusterInfoV2(clusterInfo, rank, &commV2));
        constexpr HcclCommConfig* config = nullptr; // 未配置为默认加速模式
        HcclResult ret = HcclCommInitCollComm(rank, &commV2, config, comm);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HcclCommInitCollComm]HcclCommInitCollComm failed.Destroy comv2");
            CHK_RET(HcclCommDestroyV2(commV2));
            commV2 = nullptr;
            *comm = nullptr;
            return ret;
        }
        return HCCL_SUCCESS;
    }());
#endif
    HcclResult ret = InitExternalInput();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init external input error.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);
    ret = InitEnvConfig();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init environment config error.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    std::string identifier = HCCL_WORLD_GROUP;
    CommConfig commConfig(identifier);
    std::string rankTableM;
    std::string realFilePath;
    ret = HcomLoadRanktableFile(clusterInfo, rankTableM, realFilePath);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][HcclCommInitClusterInfoWrapper]errNo[0x%016llx], clusterInfo[%s], rank[%u], "
            "load rankTable error.",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL), clusterInfo, rank),
        HCCL_E_INTERNAL);

    HCCL_INFO("%s success, clusterInfoRealPath[%s].", __func__, realFilePath.c_str());

    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    CHK_RET(CheckOpBasedHcom(opBaseHcom, rank, commConfig));

    CHK_RET(InitCommClusterInfo(rankTableM, rank, commConfig, opBaseHcom, comm));

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "[HCCL_TRACE]%s success, take time [%lld]us, clusterInfo[%s], rank[%u], deviceLogicId[%d].", __func__,
        DURATION_US(TIME_NOW() - startut), clusterInfo, rank, deviceLogicId);
#endif
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitClusterInfo(
    [[maybe_unused]] const char* clusterInfo, [[maybe_unused]] uint32_t rank, [[maybe_unused]] HcclComm* comm)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    if (hcclGroupDepth > 0) {
        HcclResult ret = HCCL_SUCCESS;
        std::shared_ptr<struct hcclCommInitRankTableAsyncJob> job;
        EXCEPTION_CATCH((job = std::make_shared<struct hcclCommInitRankTableAsyncJob>()), return HCCL_E_PARA);
        job->clusterInfo = clusterInfo;
        job->rank = rank;
        job->initComm = comm;
        s32 devId = 0;
        CHK_RET(HcclDeviceRefresh(devId));
        job->devId = devId;
        ret = commInitTaskAppend(job, HcclCommInitClusterInfoWrapper, comm);
        return ret;
    }
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));
    HCCL_RUN_INFO(
        "Entry-%s: clusterInfo[%s], rank[%u], deviceLogicId[%d].", __func__, clusterInfo, rank, deviceLogicId);
    // 入参合法性校验
    CHK_PTR_NULL(clusterInfo);
    CHK_PTR_NULL(comm);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        void* commV2 = nullptr;
        CHK_RET(HcclCommInitClusterInfoV2(clusterInfo, rank, &commV2));
        constexpr HcclCommConfig* config = nullptr; // 未配置为默认加速模式
        HcclResult ret = HcclCommInitCollComm(rank, &commV2, config, comm);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HcclCommInitCollComm]HcclCommInitCollComm failed.Destroy comv2");
            CHK_RET(HcclCommDestroyV2(commV2));
            commV2 = nullptr;
            *comm = nullptr;
            return ret;
        }
        return HCCL_SUCCESS;
    }());
#endif
    HcclResult ret = InitExternalInput();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init external input error.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);
    ret = InitEnvConfig();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init environment config error.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    std::string identifier = HCCL_WORLD_GROUP;
    CommConfig commConfig(identifier);
    std::string rankTableM;
    std::string realFilePath;
    ret = HcomLoadRanktableFile(clusterInfo, rankTableM, realFilePath);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][HcclCommInitClusterInfo]errNo[0x%016llx], clusterInfo[%s], rank[%u], "
            "load rankTable error.",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL), clusterInfo, rank),
        HCCL_E_INTERNAL);

    HCCL_INFO("%s success, clusterInfoRealPath[%s].", __func__, realFilePath.c_str());

    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    CHK_RET(CheckOpBasedHcom(opBaseHcom, rank, commConfig));

    CHK_RET(InitCommClusterInfo(rankTableM, rank, commConfig, opBaseHcom, comm));

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "[HCCL_TRACE]%s success, take time [%lld]us, clusterInfo[%s], rank[%u], deviceLogicId[%d].", __func__,
        DURATION_US(TIME_NOW() - startut), clusterInfo, rank, deviceLogicId);
#endif
    return HCCL_SUCCESS;
}

HcclResult
HcclCommInitClusterInfoMemConfig(const char* rankTableString, uint32_t rank, HcclCommConfig* config, HcclComm* comm)
{
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));

    // 入参合法性校验
    CHK_PTR_NULL(rankTableString);
    CHK_PTR_NULL(config);
    CHK_PTR_NULL(config->hcclCommName);
    CHK_PTR_NULL(comm);

    HCCL_RUN_INFO(
        "Entry-%s: rankTableString[%s], rank[%u], deviceLogicId[%d].", __func__, rankTableString, rank, deviceLogicId);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        void* commV2 = nullptr;
        CHK_RET(HcclCommInitClusterInfoMemConfigV2(rankTableString, rank, config, &commV2));
        HcclResult ret = HcclCommInitCollComm(rank, &commV2, config, comm);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HcclCommInitCollComm]HcclCommInitCollComm failed.Destroy comv2");
            CHK_RET(HcclCommDestroyV2(commV2));
            commV2 = nullptr;
            *comm = nullptr;
            return ret;
        }
        return HCCL_SUCCESS;
    }());
#endif
    HcclResult ret = InitExternalInput();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init external input error", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);
    ret = InitEnvConfig();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init environment config error.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    CHK_PRT_RET(
        strlen(config->hcclCommName) == 0,
        HCCL_ERROR("[Init][HcclCommInitClusterInfoMemConfig] hcclCommName is error."), HCCL_E_PARA);

    std::string rankTableM(rankTableString);
    std::string identifier = config->hcclCommName;
    CommConfig commConfig(identifier);
    HCCL_RUN_INFO("Entry-%s: %s", "hcclCommName", identifier.c_str());

    /* 读取用户配置 */
    ret = commConfig.Load(config);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][HcclCommInitClusterInfoMemConfig]errNo[0x%016llx] load comm config failed.", HCCL_ERROR_CODE(ret)),
        HCCL_E_PARA);

    CHK_PRT_RET(
        deviceLogicId == INVALID_INT, HCCL_ERROR("[Init][HcclCommInitClusterInfoMemConfig] deviceLogicId is error."),
        HCCL_E_PARA);

    bool isCommNameExist = IsCommNameExistInOneSidedComms(deviceLogicId, identifier);
    CHK_PRT_RET(isCommNameExist, HCCL_ERROR("[Init][HcclCommInitClusterInfoMemConfig] comm Name exist."), HCCL_E_PARA);

    CHK_RET(InitOneSidedHcomInfo(deviceLogicId, identifier));

    const std::string commIdentifier = commConfig.GetConfigCommName();
    HcclOpInfoCtx& oneSidedHCom = GetOneSidedOpInfoCtx(deviceLogicId, commIdentifier);

    AddOneSidedIdentifier(identifier);

    ret = InitCommClusterInfo(rankTableM, rank, commConfig, oneSidedHCom, comm);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[Init][HcclCommInitClusterInfoMemConfig] InitCommClusterInfo failed");
        DeleteOneSidedIdentifier(identifier);
        DeInitOneSidedHcomInfo(deviceLogicId, identifier);
        return ret;
    }

    g_oneSidedCommSet.insert(*comm);

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "[HCCL_TRACE]%s success, take time [%lld]us, rankTableString[%s], rank[%u], deviceLogicId[%d].", __func__,
        DURATION_US(TIME_NOW() - startut), rankTableString, rank, deviceLogicId);
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitClusterInfoConfigWrapper([[maybe_unused]] struct hcclAsyncJob* job_)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    struct hcclCommInitRankTableConfigAsyncJob* job = static_cast<hcclCommInitRankTableConfigAsyncJob*>(job_);
    uint32_t rank = job->rank;
    HcclComm* comm = job->initComm;
    const char* clusterInfo = job->clusterInfo;
    HcclCommConfig* config = job->config;
    s32 devId = job->devId;
    HCCL_DEBUG("[HcclCommInitClusterInfoConfigWrapper] Set device devId: %d", devId);
    CHK_PRT_RET(
        hrtSetDevice(devId) != HCCL_SUCCESS,
        HCCL_ERROR("[HcclCommInitClusterInfoConfigWrapper] set fail device[%d]", devId), HCCL_E_INTERNAL);
    HCCL_DEBUG("[HcclCommInitClusterInfoConfigWrapper] Done Set device devId: %d", devId);

    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));
    HCCL_RUN_INFO(
        "Entry-%s: clusterInfo[%s], rank[%u], deviceLogicId[%d].", __func__, clusterInfo, rank, deviceLogicId);
    // 入参合法性校验
    CHK_PTR_NULL(clusterInfo);
    CHK_PTR_NULL(comm);

    // 检查配置参数是否为空
    RPT_INPUT_ERR(
        config == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommInitClusterInfoConfigWrapper", "nullptr", "config", "non-null pointer"}));
    CHK_SMART_PTR_NULL(config);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    const char* socNamePtr = aclrtGetSocName();
    CHK_PTR_NULL(socNamePtr);
    HCCLV2_FUNC_RUN(
        [&]() -> HcclResult {
            void* commV2 = nullptr;
            CHK_RET(HcclCommInitClusterInfoConfigV2(clusterInfo, rank, config, &commV2));
            HcclResult ret = HcclCommInitCollComm(rank, &commV2, config, comm);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[HcclCommInitCollComm]HcclCommInitCollComm failed.Destroy comv2");
                CHK_RET(HcclCommDestroyV2(commV2));
                commV2 = nullptr;
                *comm = nullptr;
                return ret;
            }
            return HCCL_SUCCESS;
        }(),
        socNamePtr);
#endif
    HcclResult ret = InitExternalInput();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init external input error.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);
    ret = InitEnvConfig();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init environment config error.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    std::string identifier = HCCL_WORLD_GROUP;
    CommConfig commConfig(identifier);
    ret = commConfig.Load(config);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] load comm config failed.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    std::string rankTableM;
    std::string realFilePath;
    ret = HcomLoadRanktableFile(clusterInfo, rankTableM, realFilePath);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][HcclCommInitClusterInfoConfigWrapper]errNo[0x%016llx] clusterInfo[%s] rank[%u] "
            "load rankTable error.",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL), clusterInfo, rank),
        HCCL_E_INTERNAL);

    HCCL_INFO("%s success, clusterInfoRealPath[%s].", __func__, realFilePath.c_str());

    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    CHK_RET(CheckOpBasedHcom(opBaseHcom, rank, commConfig));

    CHK_RET(InitCommClusterInfo(rankTableM, rank, commConfig, opBaseHcom, comm));

    // 记录groupName和UDI的映射
    HCCL_PROFILER_ADD_GROUP_UDI(commConfig.GetConfigCommName(), commConfig.GetConfigUdi());

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "[HCCL_TRACE]%s success, take time [%lld]us, clusterInfo[%s], rank[%u], deviceLogicId[%d].", __func__,
        DURATION_US(TIME_NOW() - startut), clusterInfo, rank, deviceLogicId);
#endif
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitClusterInfoConfig(
    [[maybe_unused]] const char* clusterInfo, [[maybe_unused]] uint32_t rank, [[maybe_unused]] HcclCommConfig* config,
    [[maybe_unused]] HcclComm* comm)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    if (hcclGroupDepth > 0) {
        HcclResult ret = HCCL_SUCCESS;
        std::shared_ptr<struct hcclCommInitRankTableConfigAsyncJob> job;
        EXCEPTION_CATCH((job = std::make_shared<struct hcclCommInitRankTableConfigAsyncJob>()), return HCCL_E_PARA);
        job->clusterInfo = clusterInfo;
        job->rank = rank;
        job->config = config;
        job->initComm = comm;
        s32 devId = 0;
        CHK_RET(HcclDeviceRefresh(devId));
        job->devId = devId;
        ret = commInitTaskAppend(job, HcclCommInitClusterInfoConfigWrapper, comm);
        return ret;
    }
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));
    HCCL_RUN_INFO(
        "Entry-%s: clusterInfo[%s], rank[%u], deviceLogicId[%d].", __func__, clusterInfo, rank, deviceLogicId);
    // 入参合法性校验
    CHK_PTR_NULL(clusterInfo);
    CHK_PTR_NULL(comm);

    // 检查配置参数是否为空
    RPT_INPUT_ERR(
        config == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommInitClusterInfoConfig", "nullptr", "config", "non-null pointer"}));
    CHK_SMART_PTR_NULL(config);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        void* commV2 = nullptr;
        CHK_RET(HcclCommInitClusterInfoConfigV2(clusterInfo, rank, config, &commV2));
        HcclResult ret = HcclCommInitCollComm(rank, &commV2, config, comm);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HcclCommInitCollComm]HcclCommInitCollComm failed.Destroy comv2");
            CHK_RET(HcclCommDestroyV2(commV2));
            commV2 = nullptr;
            *comm = nullptr;
            return ret;
        }
        return HCCL_SUCCESS;
    }());
#endif
    HcclResult ret = InitExternalInput();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init external input error.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);
    ret = InitEnvConfig();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init environment config error.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    std::string identifier = HCCL_WORLD_GROUP;
    CommConfig commConfig(identifier);
    ret = commConfig.Load(config);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] load comm config failed.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    std::string rankTableM;
    std::string realFilePath;
    ret = HcomLoadRanktableFile(clusterInfo, rankTableM, realFilePath);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][HcclCommInitClusterInfoConfig]errNo[0x%016llx] clusterInfo[%s] rank[%u] "
            "load rankTable error.",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL), clusterInfo, rank),
        HCCL_E_INTERNAL);

    HCCL_INFO("%s success, clusterInfoRealPath[%s].", __func__, realFilePath.c_str());

    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    CHK_RET(CheckOpBasedHcom(opBaseHcom, rank, commConfig));

    CHK_RET(InitCommClusterInfo(rankTableM, rank, commConfig, opBaseHcom, comm));

    // 记录groupName和UDI的映射
    HCCL_PROFILER_ADD_GROUP_UDI(commConfig.GetConfigCommName(), commConfig.GetConfigUdi());

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "[HCCL_TRACE]%s success, take time [%lld]us, clusterInfo[%s], rank[%u], deviceLogicId[%d].", __func__,
        DURATION_US(TIME_NOW() - startut), clusterInfo, rank, deviceLogicId);
#endif
    return HCCL_SUCCESS;
}

HcclResult HcclCreateSubCommConfigInner(
    [[maybe_unused]] hccl::hcclComm* globalComm, [[maybe_unused]] uint32_t rankNum, [[maybe_unused]] uint32_t* rankIds,
    [[maybe_unused]] uint32_t subCommRankId, [[maybe_unused]] CommConfig& commConfig,
    [[maybe_unused]] HcclComm* subComm)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams globalParams{};
    RankTable_t globalRankTable{};
    CHK_RET(globalComm->GetCommParams(globalParams));
    CHK_RET(globalComm->GetCommRankTable(globalRankTable));

    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);

    const std::string commIdentifier = commConfig.GetConfigCommName();
    auto iter = opBaseHcom.opGroup2CommMap.find(commIdentifier);
    CHK_PRT_RET(
        iter != opBaseHcom.opGroup2CommMap.end(),
        HCCL_ERROR(
            "[%s]errNo[0x%016llx]The comm name[%s] already exists in Group2Comm map.", __func__,
            HCCL_ERROR_CODE(HCCL_E_PARA), commIdentifier.c_str()),
        HCCL_E_PARA);

    std::shared_ptr<hccl::hcclComm> pComm;
    pComm.reset(new (std::nothrow) hccl::hcclComm(
        commConfig.GetConfigBufferSize(), commConfig.GetConfigBufferSize(), commIdentifier,
        commConfig.GetConfigBufferName()));
    CHK_PTR_NULL(pComm);

    bool errorFlag = false;
    hccl::HcclCommParams subParams{};
    hccl::RankTable_t subRankTable{};
    do {
        RankConsistentcyChecker::GetInstance().SetCheckCannVersionSwitch(true); // 打开CANN软件版本校验开关

        std::unique_ptr<TopoinfoRanktablePartition> pTopoPartition;
        pTopoPartition.reset(new (std::nothrow) hccl::TopoinfoRanktablePartition(globalParams, globalRankTable));
        CHK_SMART_PTR_NULL(pTopoPartition);
        CHK_RET(pTopoPartition->GenerateSubRankTable(rankNum, rankIds, subRankTable));
        CHK_RET(pTopoPartition->GenerateSubParams(subRankTable, subCommRankId, subParams));

        std::string rankTableM = "";
        CHK_RET(pTopoPartition->GetRankTableStr(subRankTable, rankTableM));

        ret = InitOtherInfo(subParams, rankTableM.c_str());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s]errNo[0x%016llx] init other Info.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        ret = pComm->init(subParams, commConfig, subRankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s]errNo[0x%016llx] hcclComm init error.", __func__, HCCL_ERROR_CODE(ret)), errorFlag = true);
        HCCL_INFO("[HcclCreateSubCommConfigInner]comm id[%s]", subParams.id.internal);

        /* 设置确定性计算配置 */
        ret = pComm->SetDeterministicConfig(commConfig.GetConfigDeterministic());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s]errNo[0x%016llx] set deterministic error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        // 设置TC/SL配置
        ret = pComm->SetQpQosAttr(commConfig.GetConfigTrafficClass(), commConfig.GetConfigServiceLevel());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s]errNo[0x%016llx] set TC and SL error", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        if (commConfig.GetConfigJobID() != 0) {
            HCCL_RUN_INFO(
                "[NSLBDP]GetConfigJobID = %llu,GetConfigWorldRankID = %u.", commConfig.GetConfigJobID(),
                commConfig.GetConfigWorldRankID());
            hcclNslbDp::GetInstance().SetGlobalCommTaskId(commConfig.GetConfigJobID());
            hcclNslbDp::GetInstance().SetGlobalCommNodeId(commConfig.GetConfigWorldRankID());
        }
        /* 设置AIV模式 */
        ret = pComm->SetAivModeConfig(commConfig.GetConfigAivMode());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s]errNo[0x%016llx] set aivMode error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置only AIV模式 */
        ret = pComm->SetOnlyAivModeConfig(commConfig.GetConfigIsOnlyAivMode());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s]errNo[0x%016llx] set only aivMode error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置AICPU */
        ret = pComm->SetAicpuUnfoldConfig(commConfig.GetConfigAicpuUnfold());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s]errNo[0x%016llx] set aicpu error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置HcclExecTimeOut */
        ret = pComm->SetExecTimeOutConfig(commConfig.GetConfigExecTimeOut());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] set execTimeOut error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置HcclAlgo */
        ret = pComm->SetAlgoConfig(commConfig.GetConfigHcclAlgoMap());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] set hcclAlgo error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        ret = InitWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s]errNo[0x%016llx] init workflow mode error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        ret = DisplayRanktableInfo(subRankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s]errNo[0x%016llx] print ranktable info error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置独立算子参数 */
        ret = pComm->SetIndependentOpConfig(commConfig, subRankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s] errNo[0x%016llx] set SetIndependentOpConfig error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        // 初始化完成的comm指针赋给出参
        *subComm = pComm.get();
        std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
        opBaseHcom.opGroup2CommMap[pComm->GetIdentifier()] = pComm;
        lock.unlock();

        ret = HcomSetGroupTopoInfo(pComm->GetIdentifier().c_str(), rankNum);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s]errNo[0x%016llx] set group topo info error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        ret = pComm->InitHccpChannel();
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("InitHccp channel unsuccessful ret:[%u].", ret);
        }
    } while (0);

    if (errorFlag) {
        HCCL_ERROR(
            "[%s]Create sub communication failed, return[0x%016llx], "
            "rankNum[%u], subCommRankId[%u], sub comm identifier[%s], server[%s], logicDevId[%d]",
            __func__, HCCL_ERROR_CODE(ret), rankNum, subCommRankId, commIdentifier.c_str(),
            GetLocalServerId(subParams.serverId).c_str(), subParams.logicDevId);
        (void)HcclCommDestroy(pComm.get());
        return ret;
    }
    std::string identifier = pComm->GetIdentifier();

    /* NSLB 填充 表1 */
    CHK_RET(hcclNslbDp::GetInstance().SetCommInfo_NoRankTable(subRankTable, identifier, subCommRankId));

    HCCL_RUN_INFO(
        "%s success, sub comm identifier[%s], rankNum[%u], rank[%u], server[%s], device[%d].", __func__,
        commIdentifier.c_str(), subRankTable.rankNum, subCommRankId, subParams.serverId.c_str(), subParams.logicDevId);
#endif
    return HCCL_SUCCESS;
}

HcclResult SubCommIsOneSidedComm(const HcclComm* comm)
{
    if (IsOneSidedComm(*comm)) {
        HCCL_ERROR(
            "[%s]errNo[0x%016llx] oneSidedComm does not support create sub comm.", __func__,
            HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT));
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCreateSubCommConfig(
    HcclComm* comm, uint32_t rankNum, uint32_t* rankIds, uint64_t subCommId, uint32_t subCommRankId,
    HcclCommConfig* config, HcclComm* subComm)
{
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));
    HCCL_RUN_INFO(
        "Entry-%s: rankNum[%u], rank[%u], deviceLogicId[%d]", __func__, rankNum, subCommRankId, deviceLogicId);
    CHK_SMART_PTR_NULL(subComm);
    if (*subComm != nullptr) {
        HCCL_WARNING(
            "[%s]The value pointed by output param subComm is not nullptr. "
            "Please be ware of possible memory leak.",
            __func__);
    }
    CHK_PRT_RET(
        rankIds == nullptr && subCommId == INVALID_SUBCOMM_ID,
        HCCL_RUN_INFO(
            "[HCCL_TRACE]HcclCreateSubCommConfig return, rankIds is nullptr and subCommId is 0xFFFFFFFF, "
            "this device is not in the sub comm, deviceLogicId[%u].",
            deviceLogicId),
        HCCL_SUCCESS);
    CHK_PRT_RET(
        rankIds == nullptr || subCommId == INVALID_SUBCOMM_ID,
        HCCL_ERROR(
            "[%s]errNo[0x%016llx] "
            "rankIds[%p] is nullptr xor subCommId[%llu] is invalid. "
            "The two parameters should only be both valid or both invalid.",
            __func__, HCCL_ERROR_CODE(HCCL_E_PARA), rankIds, subCommId),
        HCCL_E_PARA);

    HcclResult ret = HCCL_SUCCESS;
    // 入参合法性校验
    CHK_PRT_RET(
        (rankNum == 0),
        HCCL_ERROR("[%s]errNo[0x%016llx] Rank num cannot be zero.", __func__, HCCL_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);
    CHK_PRT_RET(
        (subCommRankId >= rankNum),
        HCCL_ERROR(
            "[%s]errNo[0x%016llx] subCommRankId[%u] should be less "
            "than rankNum[%u].",
            __func__, HCCL_ERROR_CODE(HCCL_E_PARA), subCommRankId, rankNum),
        HCCL_E_PARA);

    RPT_INPUT_ERR(
        config == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCreateSubCommConfig", "nullptr", "config", "non-null pointer"}));
    CHK_SMART_PTR_NULL(config);
    CHK_SMART_PTR_NULL(comm);
    CHK_RET(SubCommIsOneSidedComm(comm));
    hccl::hcclComm* globalComm = static_cast<hccl::hcclComm*>(*comm);
    CHK_PTR_NULL(globalComm);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        HcclComm commV2 = globalComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        HcclComm subCommV2 = nullptr;
        CHK_RET(HcclCreateSubCommConfigV2(&commV2, rankNum, rankIds, subCommId, subCommRankId, config, &subCommV2));
        HcclResult ret = HcclCommInitCollComm(subCommRankId, &subCommV2, config, subComm);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HcclCommInitCollComm]HcclCommInitCollComm failed. Destroy subCommV2");
            CHK_RET(HcclCommDestroyV2(subCommV2));
            subCommV2 = nullptr;
            *subComm = nullptr;
            return ret;
        }
        return HCCL_SUCCESS;
    }());
#endif
    ret = InitExternalInput();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init external input error", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);
    ret = InitEnvConfig();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init environment config error.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    std::string identifier = globalComm->GetIdentifier() + "_sub_" + to_string(subCommId);
    CommConfig commConfig(identifier);
    ret = commConfig.Load(config);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] load comm config failed.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    CHK_RET(HcclCreateSubCommConfigInner(globalComm, rankNum, rankIds, subCommRankId, commConfig, subComm));

    // 记录groupName和UDI的映射
    HCCL_PROFILER_ADD_GROUP_UDI(commConfig.GetConfigCommName(), commConfig.GetConfigUdi());

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "[HCCL_TRACE]%s success, take time [%lld]us, "
        "sub comm identifier[%s], rankNum[%u], rank[%u], deviceLogicId[%d]",
        __func__, DURATION_US(TIME_NOW() - startut), commConfig.GetConfigCommName().c_str(), rankNum, subCommRankId,
        deviceLogicId);
    return HCCL_SUCCESS;
}

// 校验 root handle 可装入 HCCL_ROOT_INFO_BYTES（Get 与 Init 侧共用，防止溢出/越界读取）
HcclResult CheckRootHandleSize(const char* tag, size_t handleSize)
{
    if (handleSize > HCCL_ROOT_INFO_BYTES) {
        HCCL_ERROR("[%s]hccl root info overflow. max length: %u, actual:%zu", tag, HCCL_ROOT_INFO_BYTES, handleSize);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

// 单root root 侧建链：生成 HcclRootHandle 并登记 server，供各 rank 通过 HcclCommInitRootInfo 建链
HcclResult HcclGetRootInfo([[maybe_unused]] HcclRootInfo* rootInfo)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));

    // input check
    CHK_PTR_NULL(rootInfo);
    HCCL_RUN_INFO("Entry-HcclGetRootInfo:rootInfo[%p], deviceLogicId[%d] ", rootInfo, deviceLogicId);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN(HcclGetRootInfoV2(rootInfo));
#endif
    // get commId from env
    CHK_RET(InitExternalInput());
    CHK_RET(InitEnvConfig());

    HcclRootHandle rootHandle;
    std::shared_ptr<TopoInfoDetect> topoDetectServer;
    EXCEPTION_CATCH((topoDetectServer = std::make_shared<TopoInfoDetect>()), return HCCL_E_MEMORY);
    HcclResult ret = topoDetectServer->SetupServer(rootHandle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]%s failed, ret[%u]", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(),
            __func__, ret),
        ret);

    if (sizeof(HcclRootHandle) > HCCL_ROOT_INFO_BYTES) {
        HCCL_ERROR(
            "[Get][RootInfo]hccl root info overflow. max length: %u, actual:%zu, identifier[%s]", HCCL_ROOT_INFO_BYTES,
            sizeof(HcclRootHandle), rootHandle.identifier);
        return HCCL_E_INTERNAL;
    } else {
        s32 sRet = memcpy_s(rootInfo->internal, HCCL_ROOT_INFO_BYTES, &rootHandle, sizeof(HcclRootHandle));
        CHK_PRT_RET(
            sRet != EOK,
            HCCL_ERROR(
                "[Get][RootInfo]memcpy root info fail. errorno[%d] "
                "params:destMaxSize[%u], count[%u]",
                sRet, HCCL_ROOT_INFO_BYTES, sizeof(HcclRootHandle)),
            HCCL_E_MEMORY);
    }

    HcclOpInfoCtx& opBaseInfo = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    EXCEPTION_CATCH(
        opBaseInfo.hcclCommTopoInfoDetectServer.insert({rootHandle.identifier, topoDetectServer}),
        return HCCL_E_MEMORY);
    /* 首节点诊断信息记录 */
    HCCL_RUN_INFO(
        "[HCCL_TRACE]HcclGetRootInfo success, take time [%lld]us, identifier[%s]", DURATION_US(TIME_NOW() - startut),
        rootHandle.identifier);
#endif
    return HCCL_SUCCESS;
}

// 多root(scalable) root 侧建链：
// 建立 agent 服务监听 + root间 mesh 全互联监听，产出 HcclScalableRootHandle 并登记 server。
HcclResult HcclGetRootInfoScalable([[maybe_unused]] HcclRootInfo* rootInfo)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));

    // input check
    CHK_PTR_NULL(rootInfo);
    HCCL_RUN_INFO("Entry-HcclGetRootInfoScalable:rootInfo[%p], deviceLogicId[%d] ", rootInfo, deviceLogicId);

    // get commId from env
    CHK_RET(InitExternalInput());
    CHK_RET(InitEnvConfig());

    HcclScalableRootHandle scalableRootHandle{};
    std::shared_ptr<TopoInfoDetectScalable> scalableServer;
    EXCEPTION_CATCH((scalableServer = std::make_shared<TopoInfoDetectScalable>()), return HCCL_E_MEMORY);
    HcclResult ret = scalableServer->SetupScalableRoot(scalableRootHandle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]%s failed, ret[%u]", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(),
            __func__, ret),
        ret);

    CHK_RET(CheckRootHandleSize("[Get][RootInfoScalable]", sizeof(HcclScalableRootHandle)));
    s32 sRet = memcpy_s(rootInfo->internal, HCCL_ROOT_INFO_BYTES, &scalableRootHandle, sizeof(HcclScalableRootHandle));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[Get][RootInfoScalable]memcpy scalable root info fail. errorno[%d] "
            "params:destMaxSize[%u], count[%zu]",
            sRet, HCCL_ROOT_INFO_BYTES, sizeof(HcclScalableRootHandle)),
        HCCL_E_MEMORY);

    HcclOpInfoCtx& opBaseInfo = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    EXCEPTION_CATCH(
        opBaseInfo.hcclCommTopoInfoDetectServer.insert({scalableRootHandle.rootHandle.identifier, scalableServer}),
        return HCCL_E_MEMORY);
    /* 首节点诊断信息记录 */
    HCCL_RUN_INFO(
        "[HCCL_TRACE]HcclGetRootInfoScalable success, take time [%lld]us, identifier[%s], meshPort[%u]",
        DURATION_US(TIME_NOW() - startut), scalableRootHandle.rootHandle.identifier, scalableRootHandle.meshPort);
#endif
    return HCCL_SUCCESS;
}

HcclResult GetSelfClusterInfo(const HcclBasicRankInfo& rankInfo, HcclCommParams& params)
{
    params.deviceType = rankInfo.deviceType;
    params.rank = rankInfo.rank;
    params.userRank = rankInfo.rank;
    params.logicDevId = rankInfo.deviceLogicID;
    params.totalRanks = rankInfo.rankSize;
    params.serverId = rankInfo.hostIP.GetReadableAddress();

    return HCCL_SUCCESS;
}

HcclResult HcclGetCommName(HcclComm commHandle, char* commName)
{
    CHK_PTR_NULL(commHandle);
    CHK_PTR_NULL(commName);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(commHandle);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclGetCommNameV2(commV2, commName));
        return HCCL_SUCCESS;
    }());
#endif
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(commHandle);
    s32 ret = strncpy_s(
        commName, ROOTINFO_INDENTIFIER_MAX_LENGTH, hcclComm->GetIdentifier().c_str(), hcclComm->GetIdentifier().size());
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("HcclGetCommName str copy fail. return[%d]", ret), HCCL_E_INTERNAL);
    HCCL_INFO("HcclGetCommName input handle=%p commName=%s", commHandle, commName);
    return HCCL_SUCCESS;
}

HcclResult
HcclGetCommHandle([[maybe_unused]] const char* commName, [[maybe_unused]] std::shared_ptr<hccl::hcclComm>& comm)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    CHK_PTR_NULL(commName);
    std::string group(commName);

    s32 deviceLogicId = 0;
    HcclResult ret = HCCL_SUCCESS;
    ret = hrtGetDevice(&deviceLogicId);
    if (ret == HCCL_SUCCESS && IsCommNameExistInOneSidedComms(deviceLogicId, commName)) {
        HcclOpInfoCtx& oneSidedHcom = GetOneSidedOpInfoCtx(deviceLogicId, commName);
        comm = oneSidedHcom.pComm;
        return HCCL_SUCCESS;
    }

    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
    auto iter = opBaseHcom.opGroup2CommMap.find(group);
    if (iter == opBaseHcom.opGroup2CommMap.end()) {
        HCCL_WARNING("please check the group name is correct, group=%s", commName);
        return HCCL_E_PARA;
    } else {
        comm = iter->second;
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult HcclCommGetHandleWithName([[maybe_unused]] const char* commName, [[maybe_unused]] HcclComm* comm)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    CHK_PTR_NULL(commName);
    CHK_PTR_NULL(comm);
    std::string group(commName);

    s32 deviceLogicId = 0;
    HcclResult ret = HCCL_SUCCESS;
    ret = hrtGetDevice(&deviceLogicId);
    if (ret == HCCL_SUCCESS && IsCommNameExistInOneSidedComms(deviceLogicId, commName)) {
        HcclOpInfoCtx& oneSidedHcom = GetOneSidedOpInfoCtx(deviceLogicId, commName);
        *comm = static_cast<HcclComm>(oneSidedHcom.pComm.get());
        return HCCL_SUCCESS;
    }

    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
    auto iter = opBaseHcom.opGroup2CommMap.find(group);
    if (iter == opBaseHcom.opGroup2CommMap.end()) {
        HCCL_ERROR("please check the group name is correct, group=%s", commName);
        return HCCL_E_PARA;
    } else {
        *comm = static_cast<HcclComm>(iter->second.get());
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult HcclGetCommConnections(
    [[maybe_unused]] const HcclRootHandle& rootHandle, [[maybe_unused]] const std::string& identifier,
    [[maybe_unused]] HcclCommConnections& commConnections)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HcclOpInfoCtx& opBaseInfo = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    auto iterServer = opBaseInfo.hcclCommTopoInfoDetectServer.find(rootHandle.identifier);
    if (iterServer == opBaseInfo.hcclCommTopoInfoDetectServer.end()) {
        commConnections.isRoot = false;
    } else {
        commConnections.isRoot = true;
        CHK_RET(iterServer->second->GetServerConnections(commConnections.serverConnections));
    }

    auto iterAgent = opBaseInfo.hcclCommTopoInfoDetectAgent.find(identifier);
    if (iterAgent == opBaseInfo.hcclCommTopoInfoDetectAgent.end()) {
        HCCL_ERROR("hccl get agent connections failed, identifier=%s", identifier.c_str());
        return HCCL_E_PARA;
    } else {
        CHK_RET(iterAgent->second->GetAgentConnection(commConnections.agentConnection));
    }
#endif
    return HCCL_SUCCESS;
}

void HcclCloseCommConnections([[maybe_unused]] const std::string& identifier)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HcclOpInfoCtx& opBaseInfo = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    EXCEPTION_CATCH(opBaseInfo.hcclCommTopoInfoDetectServer.erase(identifier), return);
    EXCEPTION_CATCH(opBaseInfo.hcclCommTopoInfoDetectAgent.erase(identifier), return);
#endif
}

HcclResult SetupHierarchical(
    const u32 nRanks, const u32 rank, const HcclRootHandle& rootHandle,
    std::shared_ptr<TopoInfoDetect>& topoDetectAgent, std::shared_ptr<TopoInfoDetect>& topoDetectMember,
    HcclRankHandle& groupLeader)
{
    HcclResult ret;
    HcclRankHandle rankHandle;
    std::vector<HcclIpAddress> whitelist;
    std::shared_ptr<TopoInfoDetect> topoDetectGroupLeader;
    EXCEPTION_CATCH((topoDetectGroupLeader = std::make_shared<TopoInfoDetect>()), return HCCL_E_MEMORY);

    CHK_PTR_NULL(topoDetectAgent);

    ret = topoDetectAgent->PrepareHandle(rankHandle, whitelist);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][CommRootInfo][SetupHierarchical]errNo[0x%016llx] "
            "prepare rank handle error",
            HCCL_ERROR_CODE(ret)),
        ret);

    CommConfig commConfig;
    ret = topoDetectAgent->SetupAgent(nRanks, rank, rootHandle, rankHandle, commConfig); // member connect to root
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][CommRootInfo][SetupHierarchical]errNo[0x%016llx] "
            "setup topo detect error",
            HCCL_ERROR_CODE(ret)),
        ret);

    ret = topoDetectAgent->GetGroupLeader(groupLeader); // get group leader  此时group leader未监听
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][CommRootInfo][SetupHierarchical]errNo[0x%016llx] "
            "get group leader info error",
            HCCL_ERROR_CODE(ret)),
        ret);

    std::shared_ptr<HcclSocket> agentConnRoot;
    ret = topoDetectAgent->GetAgentConnection(agentConnRoot);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][CommRootInfo][SetupHierarchical]errNo[0x%016llx] "
            "get agent connection ptr error",
            HCCL_ERROR_CODE(ret)),
        ret);
    // 判读当前rank是否被选为GroupLeader
    if (std::strcmp(groupLeader.ip, rankHandle.ip) == 0 && (groupLeader.rankId == rank)) {
        HCCL_RUN_INFO("[Init][CommRootInfo][SetupHierarchical]rank[%u] is group leader", rank);
        CHK_PTR_NULL(topoDetectGroupLeader);
        std::shared_ptr<HcclSocket> groupLeaderConnRoot;

        // 保留下Agent->Server的Socket
        ret = topoDetectAgent->GetAgentConnection(groupLeaderConnRoot);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Init][CommRootInfo][SetupHierarchical]errNo[0x%016llx] "
                "get group leader connection ptr error",
                HCCL_ERROR_CODE(ret)),
            ret);

        // 开启GroupLeader监听
        ret = topoDetectGroupLeader->GroupLeaderListen(rankHandle, whitelist); // rank bind one local port
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Init][CommRootInfo][SetupHierarchical]errNo[0x%016llx] "
                "rank bind port error",
                HCCL_ERROR_CODE(ret)),
            ret);

        // 传给root 监听的端口
        ret = topoDetectAgent->SendGroupLeaderPort(groupLeaderConnRoot, rankHandle);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Init][CommRootInfo][SetupHierarchical]errNo[0x%016llx] "
                "setup group leader error",
                HCCL_ERROR_CODE(ret)),
            ret);

        HCCL_RUN_INFO("rankHandle.port[%u]", rankHandle.port);

        ret = topoDetectGroupLeader->GroupLeaderAccept(rankHandle, whitelist, groupLeaderConnRoot);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Init][CommRootInfo][SetupHierarchical]errNo[0x%016llx] "
                "setup group leader error",
                HCCL_ERROR_CODE(ret)),
            ret);
    }
    ret = topoDetectAgent->SetupRank(agentConnRoot);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][CommRootInfo][SetupHierarchical]errNo[0x%016llx] "
            "set up rank to get newest grpleader error",
            HCCL_ERROR_CODE(ret)),
        ret);
    ret = topoDetectAgent->GetGroupLeader(
        groupLeader); // get group leader 此时group leader已监听 更新groupLeader 准备连接groupLeader
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][CommRootInfo][SetupHierarchical]errNo[0x%016llx] "
            "get group leader info error",
            HCCL_ERROR_CODE(ret)),
        ret);
    CHK_PTR_NULL(topoDetectMember);
    ret = topoDetectMember->SetupGroupMember(nRanks, rank, groupLeader); // group member connect to group leader
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][CommRootInfo][SetupHierarchical]errNo[0x%016llx] "
            "setup group member error",
            HCCL_ERROR_CODE(ret)),
        ret);

    return HCCL_SUCCESS;
}

HcclResult GetTopoDetectInfo(
    hccl::HcclCommParams& params, RankTable_t& rankTable, HcclBasicRankInfo& localRankInfo,
    const HcclRootHandle& rootHandle, std::shared_ptr<TopoInfoDetect>& topoDetectAgent,
    std::shared_ptr<TopoInfoDetect>& topoDetectMember)
{
    HcclResult ret = HCCL_SUCCESS;
    ret = topoDetectMember->GetClusterInfo(rankTable);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][CommRootInfo][GetTopoDetectInfo]errNo[0x%016llx] "
            "GetClusterInfo error",
            HCCL_ERROR_CODE(ret)),
        ret);

    ret = topoDetectMember->GetLocalRankInfo(localRankInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][CommRootInfo][GetTopoDetectInfo]errNo[0x%016llx] "
            "GetLocalRankInfo error.",
            HCCL_ERROR_CODE(ret)),
        ret);

    ret = GetSelfClusterInfo(localRankInfo, params);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][CommRootInfo][GetTopoDetectInfo]errNo[0x%016llx] "
            "GetRankInfo error.",
            HCCL_ERROR_CODE(ret)),
        ret);

    ret = topoDetectMember->WaitComplete(rootHandle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][CommRootInfo][GetTopoDetectInfo]errNo[0x%016llx] "
            "wait complete topo detect error",
            HCCL_ERROR_CODE(ret)),
        ret);

    ret = topoDetectAgent->GetAgentListenSocket(params.commPortConfig);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[Init][CommRootInfo][GetTopoDetectInfo]HcclGetCommListenSockets failed."),
        ret);

    return HCCL_SUCCESS;
}

HcclResult InitCommRootInfo(
    [[maybe_unused]] const u32 nRanks, [[maybe_unused]] const u32 rank,
    [[maybe_unused]] const HcclRootHandle& rootHandle, [[maybe_unused]] const CommConfig& commConfig,
    [[maybe_unused]] HcclComm* comm, [[maybe_unused]] bool isScalable = false)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HcclResult ret = HCCL_SUCCESS;
    bool errorFlag = false;
    std::shared_ptr<hccl::hcclComm> pComm;
    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    const std::string commIdentifier = commConfig.GetConfigCommName();
    auto iter = opBaseHcom.opGroup2CommMap.find(commIdentifier);
    CHK_PRT_RET(
        iter != opBaseHcom.opGroup2CommMap.end(),
        HCCL_ERROR(
            "[Init][InitCommRootInfo]errNo[0x%016llx] The comm name[%s] already exists in Group2Comm map.",
            HCCL_ERROR_CODE(HCCL_E_PARA), commIdentifier.c_str()),
        HCCL_E_PARA);
    hccl::HcclCommParams params;
    RankTable_t rankTable;
    HcclBasicRankInfo localRankInfo;

    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    bool retryEnable
        = devType == DevType::DEV_TYPE_910_93 && !commConfig.GetConfigAivMode()
          && (commConfig.GetConfigInterServerRetryEnable() || commConfig.GetConfigInterSuperPodRetryEnable());
    HCCL_INFO("[InitCommRootInfo] retryEnable is [%d]", retryEnable);

    do {
        RankConsistentcyChecker::GetInstance().SetCheckCannVersionSwitch(true); // 打开CANN软件版本校验开关
        pComm.reset(new hccl::hcclComm(
            commConfig.GetConfigBufferSize(), commConfig.GetConfigBufferSize(), commIdentifier,
            commConfig.GetConfigBufferName()));
        CHK_SMART_PTR_NULL(pComm);

        std::shared_ptr<TopoInfoDetect> topoDetectAgent;
        EXCEPTION_CATCH((topoDetectAgent = std::make_shared<TopoInfoDetect>()), return HCCL_E_MEMORY);
        topoDetectAgent->SetIsInterSuperPodRetryEnable(commConfig.GetConfigInterSuperPodRetryEnable());
        // 32k 作为agent开启阈值；scalable 建链按 root 分组后组内规模必然小于 nRanks，始终走 flat 路径
        if (nRanks > TOPO_HIERARCHICAL_ENABLE_THRESHOLD && !isScalable) {
            HCCL_RUN_INFO("[Init][CommRootInfo][Hierarchical]nRanks[%u] entry hierarchical topo detect.", nRanks);

            std::shared_ptr<TopoInfoDetect> topoDetectMember;
            EXCEPTION_CATCH((topoDetectMember = std::make_shared<TopoInfoDetect>()), return HCCL_E_MEMORY);
            topoDetectMember->SetIsInterSuperPodRetryEnable(commConfig.GetConfigInterSuperPodRetryEnable());

            HcclRankHandle groupLeader;
            ret = SetupHierarchical(nRanks, rank, rootHandle, topoDetectAgent, topoDetectMember, groupLeader);
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Init][CommRootInfo]errNo[0x%016llx] setup "
                    "hierarchical error",
                    HCCL_ERROR_CODE(ret)),
                errorFlag = true);

            ret = GetTopoDetectInfo(params, rankTable, localRankInfo, groupLeader, topoDetectAgent, topoDetectMember);
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Init][CommRootInfo][Hierarchical]errNo[0x%016llx] setup "
                    "GetTopoDetectInfo error",
                    HCCL_ERROR_CODE(ret)),
                errorFlag = true);
        } else {
            HCCL_RUN_INFO("[Init][CommRootInfo][Flat]nRanks[%u] entry flat topo detect.", nRanks);

            ret = topoDetectAgent->SetupAgent(nRanks, rank, rootHandle, rootHandle, commConfig, isScalable);
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Init][CommRootInfo][Flat]errNo[0x%016llx] "
                    "setup flat topo detect error",
                    HCCL_ERROR_CODE(ret)),
                errorFlag = true);
            ret = GetTopoDetectInfo(params, rankTable, localRankInfo, rootHandle, topoDetectAgent, topoDetectAgent);
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Init][CommRootInfo][Flat]errNo[0x%016llx] setup "
                    "GetTopoDetectInfo error",
                    HCCL_ERROR_CODE(ret)),
                errorFlag = true);
        }

        /* 初始化hccl comm */

        CHK_RET(DisplayRanktableInfo(rankTable));

        if (retryEnable) {
            EXCEPTION_CATCH(
                opBaseHcom.hcclCommTopoInfoDetectAgent.insert({commIdentifier, topoDetectAgent}), return HCCL_E_MEMORY);
            ret = HcclGetCommConnections(rootHandle, commIdentifier, params.commConnections);
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS, HCCL_ERROR("[Init][RootInfo]HcclGetCommConnections failed."), errorFlag = true);
        } else {
            ret = topoDetectAgent->Teardown();
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Init][RootInfo]errNo[0x%016llx] Teardown topo detect error", HCCL_ERROR_CODE(ret)),
                errorFlag = true);
        }

        ret = InitWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] init work flow mode error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        ret = InitOtherInfo(params, nullptr);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] init other Info", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        HCCL_INFO("rootInfo[%s], params.logiceDevice[%d]", params.id.internal, params.logicDevId);
        ret = pComm->init(params, commConfig, rankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] hcclComm init error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置确定性计算配置 */
        ret = pComm->SetDeterministicConfig(commConfig.GetConfigDeterministic());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] set deterministic error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        // 设置TC/SL配置
        ret = pComm->SetQpQosAttr(commConfig.GetConfigTrafficClass(), commConfig.GetConfigServiceLevel());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[InitCommRootInfo]errNo[0x%016llx] set TC and SL error or Invalid configuration parameter.",
                HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        if (commConfig.GetConfigJobID() != 0) {
            HCCL_RUN_INFO(
                "[NSLBDP]GetConfigJobID = %llu,GetConfigWorldRankID = %u.", commConfig.GetConfigJobID(),
                commConfig.GetConfigWorldRankID());
            hcclNslbDp::GetInstance().SetGlobalCommTaskId(commConfig.GetConfigJobID());
            hcclNslbDp::GetInstance().SetGlobalCommNodeId(commConfig.GetConfigWorldRankID());
        }

        // 设置HCCL QOS配置
        ret = pComm->SetHcclQos(commConfig.GetConfigHcclQos());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s]errNo[0x%016llx] set hccl qos error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置AIV模式 */
        ret = pComm->SetAivModeConfig(commConfig.GetConfigAivMode());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] set aivMode error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置only AIV模式 */
        ret = pComm->SetOnlyAivModeConfig(commConfig.GetConfigIsOnlyAivMode());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] set only aivMode error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置AICPU */
        ret = pComm->SetAicpuUnfoldConfig(commConfig.GetConfigAicpuUnfold());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] set aicpu error.", HCCL_ERROR_CODE(ret)), errorFlag = true);

        /* 设置独立算子参数 */
        ret = pComm->SetIndependentOpConfig(commConfig, rankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] set SetIndependentOpConfig error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置HcclExecTimeOut */
        ret = pComm->SetExecTimeOutConfig(commConfig.GetConfigExecTimeOut());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] set execTimeOut error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置HcclAlgo */
        ret = pComm->SetAlgoConfig(commConfig.GetConfigHcclAlgoMap());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] set hcclAlgo error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        // 初始化完成的comm指针赋给出参
        *comm = pComm.get();
        std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
        opBaseHcom.opGroup2CommMap[pComm->GetIdentifier()] = pComm;
        lock.unlock();

        // 特殊场景，当comm name被手动配置为HCCL_WORLD_GROUP时，需要将pComm赋值到hcomInfo.pComm
        if (pComm->GetIdentifier() == HCCL_WORLD_GROUP) {
            HcomGetCtxHomInfo().pComm = pComm;
        }

        ret = HcomSetGroupTopoInfo(pComm->GetIdentifier().c_str(), nRanks);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] setGroupTopoInfo error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        ret = pComm->InitHccpChannel();
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("InitHccp channel unsuccessful ret:[%u].", ret);
        }
        if (hcclNslbDp::GetInstance().GetGlobalCommTaskId() != 0) {
            DevType nslb_devType;
            CHK_RET(hrtGetDeviceType(nslb_devType));
            if (nslb_devType == DevType::DEV_TYPE_910_93) {
                hcclNslbDp::GetInstance().SetDeviceType();
            }
            if (hcclNslbDp::GetInstance().InitNetCo() == HCCL_SUCCESS) {
                std::string identifier_nslb = commIdentifier;
                hcclNslbDp::GetInstance().InitCmmDesc(identifier_nslb);
                HCCL_INFO(
                    "nslb_InitCommRootInfo rankTable.rankList.size:[%zu], identifier_nslb[%s].",
                    rankTable.rankList.size(), identifier_nslb.c_str());
                hcclNslbDp::GetInstance().SetGlobalCommRankTable_RootInfo(
                    rankTable, localRankInfo, pComm->GetRankLists(), identifier_nslb, nRanks, rank);
                hcclNslbDp::GetInstance().SetGlobalDisRankTable(localRankInfo);
            } else {
                HCCL_WARNING("nslbdp try to init hccp failed.");
            }
        }
    } while (0);

    std::string defaultIdentifier = rootHandle.identifier;
    bool serverExist = opBaseHcom.hcclCommTopoInfoDetectServer.find(defaultIdentifier)
                       != opBaseHcom.hcclCommTopoInfoDetectServer.end();
    if (defaultIdentifier.compare(commIdentifier) != 0 && retryEnable && serverExist) {
        EXCEPTION_CATCH(
            opBaseHcom.hcclCommTopoInfoDetectServer.insert(
                {commIdentifier, opBaseHcom.hcclCommTopoInfoDetectServer[defaultIdentifier]}),
            return HCCL_E_MEMORY);
        EXCEPTION_CATCH(opBaseHcom.hcclCommTopoInfoDetectServer.erase(defaultIdentifier), return HCCL_E_MEMORY);
        HCCL_INFO(
            "[InitCommRootInfo] replace key of topoDetectServer from [%s] to [%s]", defaultIdentifier.c_str(),
            commIdentifier.c_str());
    } else if (!retryEnable && serverExist) {
        EXCEPTION_CATCH(opBaseHcom.hcclCommTopoInfoDetectServer.erase(defaultIdentifier), return HCCL_E_MEMORY);
        HCCL_INFO("[InitCommRootInfo] close topoDetectServer identifier[%s]", commIdentifier.c_str());
    }

    if (errorFlag) {
        HCCL_ERROR(
            "[InitCommRootInfo]Init failed, return[0x%016llx], rankNum[%u], rank[%u], "
            "rootInfo identifier[%s], server[%s], logicDevId[%d]",
            HCCL_ERROR_CODE(ret), nRanks, rank, commIdentifier.c_str(), GetLocalServerId(params.serverId).c_str(),
            params.logicDevId);
        (void)HcclCommDestroy(pComm.get());
        return ret;
    }

    HCCL_INFO(
        "[InitCommRootInfo]Init success, rankNum[%u], rank[%u], rootInfo identifier[%s], server[%s], "
        "logicDevId[%d]",
        nRanks, rank, commIdentifier.c_str(), params.serverId.c_str(), params.logicDevId);

#endif
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitRootInfoInner(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm)
{
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));

    CHK_PRT_RET(
        (nRanks == 0),
        HCCL_ERROR(
            "[Init][CommRootInfoInner]errNo[0x%016llx] nRanks[%u] should "
            "be greater than 0.",
            HCCL_ERROR_CODE(HCCL_E_PARA), nRanks),
        HCCL_E_PARA);

    CHK_PRT_RET(
        (rank >= nRanks),
        HCCL_ERROR(
            "[Init][CommRootInfoInner]errNo[0x%016llx] rank[%u] should "
            "be less than nRanks[%u].",
            HCCL_ERROR_CODE(HCCL_E_PARA), rank, nRanks),
        HCCL_E_PARA);

    CHK_SMART_PTR_NULL(comm);
    CHK_SMART_PTR_NULL(rootInfo);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        void* commV2 = nullptr;
        std::string fooidentifier;
        CHK_RET(HcclCommInitRootInfoV2(nRanks, rootInfo, rank, &commV2, fooidentifier));
        constexpr HcclCommConfig* config = nullptr; // 未配置为默认加速模式
        HcclResult ret = HcclCommInitCollComm(rank, &commV2, config, comm);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HcclCommInitCollComm]HcclCommInitCollComm failed.Destroy comv2");
            CHK_RET(HcclCommDestroyV2(commV2));
            commV2 = nullptr;
            *comm = nullptr;
            return ret;
        }
        return HCCL_SUCCESS;
    }());
#endif

    HcclResult ret = InitExternalInput();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][CommRootInfoInner]errNo[0x%016llx] init "
            "external input error",
            HCCL_ERROR_CODE(ret)),
        HCCL_E_PARA);
    ret = InitEnvConfig();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][CommRootInfoInner]errNo[0x%016llx] init environment config error.", HCCL_ERROR_CODE(ret)),
        HCCL_E_PARA);

    HcclRootHandle rootHandle;
    s32 sRet = memcpy_s(&rootHandle, sizeof(HcclRootHandle), rootInfo->internal, sizeof(HcclRootHandle));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[Init][RootInfoInner]memcpy root info fail. errorno[%d] "
            "params:destMaxSize[%u], count[%u]",
            sRet, sizeof(HcclRootHandle), sizeof(HcclRootHandle)),
        HCCL_E_MEMORY);
    rootHandle.identifier[ROOTINFO_INDENTIFIER_MAX_LENGTH - 1] = '\0';

    /* 接口交互信息日志 */
    HCCL_RUN_INFO(
        "Entry-HcclCommInitRootInfoInner:ranks[%u], rank[%u], rootinfo: host ip[%s] port[%u] "
        "nicDeploy[%d] identifier[%s], deviceLogicId[%d]",
        nRanks, rank, rootHandle.ip, rootHandle.port, rootHandle.nicDeploy, rootHandle.identifier, deviceLogicId);

    CommConfig commConfig(rootHandle.identifier);

    /* --------------初始化------------------------- */
    HCCL_INFO("HCCL nslbdp entry InitCommRootInfo.");
    ret = InitCommRootInfo(nRanks, rank, rootHandle, commConfig, comm);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][CommRootInfoConfig]errNo[0x%016llx]HcclCommInitRootInfo failed.", HCCL_ERROR_CODE(ret)),
        ret);

    // 记录groupName和UDI的映射
    HCCL_PROFILER_ADD_GROUP_UDI(commConfig.GetConfigCommName(), commConfig.GetConfigUdi());
    if (hcclNslbDp::GetInstance().GetGlobalCommTaskId() != 0) {
        /* NSLB 发送 */
        HCCL_INFO("hcclNslbDp entry Table FIVE rank[%u]", rank);
        hcclNslbDp::GetInstance().SendGlobalDisRankTable();
    }

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "[HCCL_TRACE]HcclCommInitRootInfoInner success, take time [%lld]us, rankNum[%u], rank[%u]",
        DURATION_US(TIME_NOW() - startut), nRanks, rank);
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitRootInfoInnerWrapper(struct hcclAsyncJob* job_)
{
    struct hcclCommInitAsyncJob* job = static_cast<hcclCommInitAsyncJob*>(job_);
    uint32_t nRanks = job->nRanks;
    const HcclRootInfo* rootInfo = job->rootInfo;
    uint32_t rank = job->rank;
    HcclComm* comm = job->initComm;
    s32 devId = job->devId;
    HCCL_DEBUG("[HcclCommInitRootInfoInnerWrapper] Set device devId: %d", devId);
    CHK_PRT_RET(
        hrtSetDevice(devId) != HCCL_SUCCESS, HCCL_ERROR("[HcclCommInitRootInfo] set fail device[%d]", devId),
        HCCL_E_INTERNAL);

    HcclResult ret = HCCL_SUCCESS;
    ret = HcclCommInitRootInfoInner(nRanks, rootInfo, rank, comm);
    return ret;
}

HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm)
{
    HCCL_INFO("hcclGroupDepth=[%d]", hcclGroupDepth);
    HcclResult ret = HCCL_SUCCESS;
    if (hcclGroupDepth > 0) {
        std::shared_ptr<struct hcclCommInitAsyncJob> job;
        EXCEPTION_CATCH((job = std::make_shared<struct hcclCommInitAsyncJob>()), return HCCL_E_PARA);
        job->nRanks = nRanks;
        job->rootInfo = rootInfo;
        job->rank = rank;
        job->initComm = comm;
        s32 devId = 0;
        CHK_RET(HcclDeviceRefresh(devId));
        job->devId = devId;
        ret = commInitTaskAppend(job, HcclCommInitRootInfoInnerWrapper, comm);
        return ret;
    }
    ret = HcclCommInitRootInfoInner(nRanks, rootInfo, rank, comm);
    return ret;
}

HcclResult HcclCommInitRootInfoConfigInner(
    uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, const HcclCommConfig* config, HcclComm* comm)
{
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));

    // 检查配置参数是否为空
    RPT_INPUT_ERR(
        config == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommInitRootInfoConfigInner", "nullptr", "config", "non-null pointer"}));

    CHK_PRT_RET(
        (nRanks == 0),
        HCCL_ERROR(
            "[Init][CommRootInfoConfigInner]errNo[0x%016llx] nRanks[%u] should be greater than 0.",
            HCCL_ERROR_CODE(HCCL_E_PARA), nRanks),
        HCCL_E_PARA);

    CHK_PRT_RET(
        (rank >= nRanks),
        HCCL_ERROR(
            "[Init][CommRootInfoConfigInner]errNo[0x%016llx] rank[%u] should be less than nRanks[%u].",
            HCCL_ERROR_CODE(HCCL_E_PARA), rank, nRanks),
        HCCL_E_PARA);

    CHK_PTR_NULL(comm);
    CHK_SMART_PTR_NULL(rootInfo);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        void* commV2 = nullptr;
        CHK_RET(HcclCommInitRootInfoConfigV2(nRanks, rootInfo, rank, config, &commV2));
        HcclResult ret = HcclCommInitCollComm(rank, &commV2, config, comm);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HcclCommInitCollComm]HcclCommInitCollComm failed.Destroy comv2");
            CHK_RET(HcclCommDestroyV2(commV2));
            commV2 = nullptr;
            *comm = nullptr;
            return ret;
        }
        return HCCL_SUCCESS;
    }());
#endif

    HcclResult ret = InitExternalInput();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][CommRootInfoConfigInner]errNo[0x%016llx] init "
            "external input error",
            HCCL_ERROR_CODE(ret)),
        HCCL_E_PARA);
    ret = InitEnvConfig();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][CommRootInfoConfigInner]errNo[0x%016llx] init "
            "environment config error",
            HCCL_ERROR_CODE(ret)),
        HCCL_E_PARA);

    HcclRootHandle rootHandle;
    s32 sRet = memcpy_s(&rootHandle, sizeof(HcclRootHandle), rootInfo->internal, sizeof(HcclRootHandle));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[Init][RootInfo]memcpy root info fail. errorno[%d] "
            "params:destMaxSize[%u], count[%u]",
            sRet, sizeof(HcclRootHandle), sizeof(HcclRootHandle)),
        HCCL_E_MEMORY);
    rootHandle.identifier[ROOTINFO_INDENTIFIER_MAX_LENGTH - 1] = '\0';

    /* 读取用户配置 */
    CommConfig commConfig(rootHandle.identifier);
    ret = commConfig.Load(config);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][CommRootInfoConfigInner]errNo[0x%016llx] load comm config failed.", HCCL_ERROR_CODE(ret)),
        HCCL_E_PARA);

    /* 接口交互信息日志 */
    HCCL_RUN_INFO(
        "Entry-HcclCommInitRootInfoConfigInner:ranks[%u], rank[%u], rootinfo: host ip[%s] "
        "port[%u] nicDeploy[%d] identifier[%s], deviceLogicId[%d]",
        nRanks, rank, rootHandle.ip, rootHandle.port, rootHandle.nicDeploy, commConfig.GetConfigCommName().c_str(),
        deviceLogicId);

    /* --------------初始化------------------------- */
    ret = InitCommRootInfo(nRanks, rank, rootHandle, commConfig, comm);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][CommRootInfoConfigInner]errNo[0x%016llx]HcclCommInitRootInfoConfigInner failed.",
            HCCL_ERROR_CODE(ret)),
        ret);

    if (hcclNslbDp::GetInstance().GetGlobalCommTaskId() != 0) {
        /* NSLB 发送 */
        HCCL_INFO("hcclNslbDp-sendTable 5 rank[%u]", rank);
        hcclNslbDp::GetInstance().SendGlobalDisRankTable();
    }

    // 记录groupName和UDI的映射
    HCCL_PROFILER_ADD_GROUP_UDI(commConfig.GetConfigCommName(), commConfig.GetConfigUdi());

    HCCL_RUN_INFO(
        "[HCCL_TRACE]HcclCommInitRootInfoConfigInner success, take time [%lld]us, "
        "rankNum[%u], rank[%u]",
        DURATION_US(TIME_NOW() - startut), nRanks, rank);

    return HCCL_SUCCESS;
}

HcclResult HcclCommInitRootInfoConfigInnerWrapper(struct hcclAsyncJob* job_)
{
    struct hcclCommInitConfigAsyncJob* job = static_cast<hcclCommInitConfigAsyncJob*>(job_);
    uint32_t nRanks = job->nRanks;
    const HcclRootInfo* rootInfo = job->rootInfo;
    uint32_t rank = job->rank;
    HcclComm* comm = job->initComm;
    const HcclCommConfig* config = job->config;
    s32 devId = job->devId;
    HCCL_DEBUG("[HcclCommInitRootInfoConfigInnerWrapper] Set device devId: %d", devId);
    CHK_PRT_RET(
        hrtSetDevice(devId) != HCCL_SUCCESS,
        HCCL_ERROR("[HcclCommInitRootInfoConfigInnerWrapper] set fail device[%d]", devId), HCCL_E_INTERNAL);

    HcclResult ret = HCCL_SUCCESS;
    ret = HcclCommInitRootInfoConfigInner(nRanks, rootInfo, rank, config, comm);
    return ret;
}

HcclResult HcclCommInitRootInfoConfig(
    uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, const HcclCommConfig* config, HcclComm* comm)
{
    HcclResult ret = HCCL_SUCCESS;
    HCCL_INFO("hcclGroupDepth=[%d]", hcclGroupDepth);
    if (hcclGroupDepth > 0) {
        std::shared_ptr<struct hcclCommInitConfigAsyncJob> job;
        EXCEPTION_CATCH((job = std::make_shared<struct hcclCommInitConfigAsyncJob>()), return HCCL_E_PARA);
        job->nRanks = nRanks;
        job->rootInfo = rootInfo;
        job->rank = rank;
        job->initComm = comm;
        job->config = config;
        s32 devId = 0;
        CHK_RET(HcclDeviceRefresh(devId));
        job->devId = devId;
        ret = commInitTaskAppend(job, HcclCommInitRootInfoConfigInnerWrapper, comm);
        return ret;
    }
    ret = HcclCommInitRootInfoConfigInner(nRanks, rootInfo, rank, config, comm);
    return ret;
}

// 参数校验 + 环境初始化
HcclResult ValidateAndInitRootInfoScalable(
    uint32_t nRanks, uint32_t nRoot, uint32_t rank, uint32_t nExtRoot, const HcclCommConfig* config, HcclComm* comm,
    const HcclRootInfo* rootInfoList)
{
    CHK_PRT_RET(
        (nRanks == 0),
        HCCL_ERROR(
            "[Init][CommRootInfoScalableInner]errNo[0x%016llx] nRanks[%u] should be greater than 0.",
            HCCL_ERROR_CODE(HCCL_E_PARA), nRanks),
        HCCL_E_PARA);
    CHK_PRT_RET(
        (nRoot == 0 || nRoot > nRanks),
        HCCL_ERROR(
            "[Init][CommRootInfoScalableInner]errNo[0x%016llx] nRoot[%u] should be in range (0, nRanks[%u]].",
            HCCL_ERROR_CODE(HCCL_E_PARA), nRoot, nRanks),
        HCCL_E_PARA);
    CHK_PRT_RET(
        (rank >= nRanks),
        HCCL_ERROR(
            "[Init][CommRootInfoScalableInner]errNo[0x%016llx] rank[%u] should be less than nRanks[%u].",
            HCCL_ERROR_CODE(HCCL_E_PARA), rank, nRanks),
        HCCL_E_PARA);
    CHK_PRT_RET(
        (nExtRoot != 0),
        HCCL_ERROR(
            "[Init][CommRootInfoScalableInner]errNo[0x%016llx] nExtRoot[%u] only supports 0 currently.",
            HCCL_ERROR_CODE(HCCL_E_PARA), nExtRoot),
        HCCL_E_PARA);
    CHK_PTR_NULL(comm);
    CHK_SMART_PTR_NULL(rootInfoList);
    CHK_PTR_NULL(config);

    HcclResult ret = InitExternalInput();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][CommRootInfoScalableInner]errNo[0x%016llx] init external input error",
            HCCL_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);
    ret = InitEnvConfig();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][CommRootInfoScalableInner]errNo[0x%016llx] init environment config error",
            HCCL_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

// scalable 建链为 host 侧特性：设备侧（HCCD/CCL_KERNEL_AICPU）不支持，不编译该部分
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
// 解析 nRoot 个 HcclScalableRootHandle（rootInfoList[i].internal 由 HcclGetRootInfoScalable 填充）
HcclResult
ParseScalableRootHandles(const HcclRootInfo* rootInfoList, u32 nRoot, std::vector<HcclScalableRootHandle>& rootHandles)
{
    CHK_RET(CheckRootHandleSize("[Init][RootInfoScalable]", sizeof(HcclScalableRootHandle)));
    for (u32 i = 0; i < nRoot; ++i) {
        s32 sRet = memcpy_s(
            &rootHandles[i], sizeof(HcclScalableRootHandle), rootInfoList[i].internal, sizeof(HcclScalableRootHandle));
        CHK_PRT_RET(
            sRet != EOK, HCCL_ERROR("[Init][RootInfoScalable]memcpy scalable root info[%u] fail. errorno[%d]", i, sRet),
            HCCL_E_MEMORY);
        rootHandles[i].rootHandle.identifier[ROOTINFO_INDENTIFIER_MAX_LENGTH - 1] = '\0';
    }
    return HCCL_SUCCESS;
}

// 组装 root 间 mesh 全互联信息：所有 root 的 ip + meshPort，下标 = rootIndex
HcclResult BuildScalableMeshInfo(
    u32 nRoot, u32 rootIndex, u32 groupSize, const std::vector<HcclScalableRootHandle>& rootHandles,
    ScalableServerInfo& scalableInfo)
{
    scalableInfo.nRoot = nRoot;
    scalableInfo.rootIndex = rootIndex;
    scalableInfo.groupSize = groupSize;
    scalableInfo.meshInfos.resize(nRoot);
    for (u32 i = 0; i < nRoot; ++i) {
        s32 sRet = memcpy_s(
            scalableInfo.meshInfos[i].ip, sizeof(scalableInfo.meshInfos[i].ip), rootHandles[i].rootHandle.ip,
            sizeof(scalableInfo.meshInfos[i].ip));
        CHK_PRT_RET(
            sRet != EOK, HCCL_ERROR("[Init][RootInfoScalable]memcpy mesh ip[%u] fail. errorno[%d]", i, sRet),
            HCCL_E_MEMORY);
        scalableInfo.meshInfos[i].meshPort = rootHandles[i].meshPort;
    }
    return HCCL_SUCCESS;
}

// root 侧：先注入 root 间 mesh 全互联信息，server 线程才能继续 accept 并做 root 间合并
HcclResult InjectScalableRootMeshInfo(
    const HcclRootHandle& groupRootHandle, bool isRootRank, const ScalableServerInfo& scalableInfo)
{
    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    auto iter = opBaseHcom.hcclCommTopoInfoDetectServer.find(groupRootHandle.identifier);
    CHK_PRT_RET(
        iter == opBaseHcom.hcclCommTopoInfoDetectServer.end(),
        HCCL_ERROR(
            "[Init][CommRootInfoScalableInner]scalable root server[%s] not found. isRootRank[%d]",
            groupRootHandle.identifier, isRootRank),
        HCCL_E_INTERNAL);
    auto topoDetectServer = std::dynamic_pointer_cast<TopoInfoDetectScalable>(iter->second);
    CHK_PRT_RET(
        topoDetectServer == nullptr,
        HCCL_ERROR(
            "[Init][CommRootInfoScalableInner]scalable root server[%s] is not TopoInfoDetectScalable.",
            groupRootHandle.identifier),
        HCCL_E_INTERNAL);
    // 拿到全部 root 的 mesh 信息后，按与 connect/accept 一致的 tag 下发 root 间 mesh 白名单
    CHK_RET(topoDetectServer->AddMeshSocketWhiteList(scalableInfo));
    CHK_RET(topoDetectServer->SetScalableServerInfo(scalableInfo));
    return HCCL_SUCCESS;
}

// 基于组root信息初始化comm，并处理NSLB全局通信表下发
HcclResult InitScalableComm(
    u32 nRanks, u32 rank, const HcclRootHandle& groupRootHandle, const CommConfig& commConfig, HcclComm* comm)
{
    HcclResult ret = InitCommRootInfo(nRanks, rank, groupRootHandle, commConfig, comm, true);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][CommRootInfoScalableInner]errNo[0x%016llx] HcclCommInitRootInfoScalable failed.",
            HCCL_ERROR_CODE(ret)),
        ret);

    if (hcclNslbDp::GetInstance().GetGlobalCommTaskId() != 0) {
        /* NSLB 发送 */
        HCCL_INFO("hcclNslbDp-sendTable 5 rank[%u]", rank);
        hcclNslbDp::GetInstance().SendGlobalDisRankTable();
    }
    return HCCL_SUCCESS;
}
#endif

HcclResult HcclCommInitRootInfoScalableInner(
    uint32_t nRanks, uint32_t nRoot, const HcclRootInfo* rootInfoList, uint32_t rank, uint32_t nExtRoot,
    const HcclCommConfig* config, HcclComm* comm)
{
    [[maybe_unused]] HcclUs startut = TIME_NOW();
    [[maybe_unused]] s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));

    // 参数校验 + 环境初始化
    CHK_RET(ValidateAndInitRootInfoScalable(nRanks, nRoot, rank, nExtRoot, config, comm, rootInfoList));

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    // 解析 nRoot 个 HcclScalableRootHandle（rootInfoList[i].internal 由 HcclGetRootInfoScalable 填充）
    std::vector<HcclScalableRootHandle> rootHandles(nRoot);
    CHK_RET(ParseScalableRootHandles(rootInfoList, nRoot, rootHandles));

    // 分组信息：rank 落入哪个 root 分组、组内规模、以及当前 rank 是否为本组 root
    const u32 rootIndex = TopoInfoDetectScalable::RootIdFromRank(nRanks, nRoot, rank);
    const u32 groupSize = TopoInfoDetectScalable::NRankFromRoot(rootIndex, nRanks, nRoot);
    const bool isRootRank = TopoInfoDetectScalable::RankHasRoot(rank, nRanks, nRoot);
    const HcclRootHandle& groupRootHandle = rootHandles[rootIndex].rootHandle;

    // 组装 root 间 mesh 全互联信息：所有 root 的 ip + meshPort，下标 = rootIndex
    ScalableServerInfo scalableInfo;
    CHK_RET(BuildScalableMeshInfo(nRoot, rootIndex, groupSize, rootHandles, scalableInfo));

    // 统一以 root[0] 的 identifier 作为整个 scalable comm 的标识，保证跨 root 组数据面 tag 一致
    CommConfig commConfig(rootHandles[0].rootHandle.identifier);
    HcclResult ret = commConfig.Load(config);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][CommRootInfoScalableInner]errNo[0x%016llx] load comm config failed.", HCCL_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);

    // root 侧：先注入 root 间 mesh 全互联信息，server 线程才能继续 accept 并做 root 间合并
    if (isRootRank) {
        CHK_RET(InjectScalableRootMeshInfo(groupRootHandle, isRootRank, scalableInfo));
    }

    /* 接口交互信息日志 */
    HCCL_RUN_INFO(
        "Entry-HcclCommInitRootInfoScalableInner:nRanks[%u], nRoot[%u], rank[%u], rootIndex[%u], groupSize[%u], "
        "isRootRank[%d], groupRoot: host ip[%s] port[%u] identifier[%s], deviceLogicId[%d]",
        nRanks, nRoot, rank, rootIndex, groupSize, isRootRank, groupRootHandle.ip, groupRootHandle.port,
        groupRootHandle.identifier, deviceLogicId);

    /* --------------初始化------------------------- */
    CHK_RET(InitScalableComm(nRanks, rank, groupRootHandle, commConfig, comm));

    // 记录groupName和UDI的映射
    HCCL_PROFILER_ADD_GROUP_UDI(commConfig.GetConfigCommName(), commConfig.GetConfigUdi());

    HCCL_RUN_INFO(
        "[HCCL_TRACE]HcclCommInitRootInfoScalableInner success, take time [%lld]us, rankNum[%u], nRoot[%u], rank[%u]",
        DURATION_US(TIME_NOW() - startut), nRanks, nRoot, rank);
#endif

    return HCCL_SUCCESS;
}

// Wrapper：异步job线程中设置设备上下文并进入scalable建链Inner
HcclResult HcclCommInitRootInfoScalableInnerWrapper(struct hcclAsyncJob* job_)
{
    struct hcclCommInitScalableAsyncJob* job = static_cast<hcclCommInitScalableAsyncJob*>(job_);
    s32 devId = job->devId;
    HCCL_DEBUG("[HcclCommInitRootInfoScalableInnerWrapper] Set device devId: %d", devId);
    CHK_PRT_RET(
        hrtSetDevice(devId) != HCCL_SUCCESS,
        HCCL_ERROR("[HcclCommInitRootInfoScalableInnerWrapper] set fail device[%d]", devId), HCCL_E_INTERNAL);

    HcclResult ret = HCCL_SUCCESS;
    ret = HcclCommInitRootInfoScalableInner(
        job->nRanks, job->nRoot, job->rootInfoList, job->rank, job->nExtRoot, job->config, job->initComm);
    return ret;
}

HcclResult HcclCommInitRootInfoScalable(
    uint32_t nRanks, uint32_t nRoot, const HcclRootInfo* rootInfoList, uint32_t rank, uint32_t nExtRoot,
    const HcclCommConfig* config, HcclComm* comm)
{
    HCCL_INFO("hcclGroupDepth=[%d]", hcclGroupDepth);
    HcclResult ret = HCCL_SUCCESS;
    if (hcclGroupDepth > 0) {
        std::shared_ptr<struct hcclCommInitScalableAsyncJob> job;
        EXCEPTION_CATCH((job = std::make_shared<struct hcclCommInitScalableAsyncJob>()), return HCCL_E_PARA);
        job->nRanks = nRanks;
        job->nRoot = nRoot;
        job->rootInfoList = rootInfoList;
        job->rank = rank;
        job->nExtRoot = nExtRoot;
        job->initComm = comm;
        job->config = config;
        s32 devId = 0;
        CHK_RET(HcclDeviceRefresh(devId));
        job->devId = devId;
        ret = commInitTaskAppend(job, HcclCommInitRootInfoScalableInnerWrapper, comm);
        return ret;
    }
    ret = HcclCommInitRootInfoScalableInner(nRanks, nRoot, rootInfoList, rank, nExtRoot, config, comm);
    return ret;
}

HcclResult HcclSetConfig([[maybe_unused]] HcclConfig config, [[maybe_unused]] HcclConfigValue configValue)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    if (config == HCCL_DETERMINISTIC) {
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
        HCCLV2_FUNC_RUN(HcclSetConfigV2(config, configValue));
#endif
        char* mmSysGetEnvValue = nullptr;
        MM_SYS_GET_ENV(MM_ENV_HCCL_DETERMINISTIC, mmSysGetEnvValue);
        std::string hcclDeterministicEnv = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
        if (hcclDeterministicEnv == "EmptyString") {
            if (configValue.value != DETERMINISTIC_STRICT && configValue.value != DETERMINISTIC_ENABLE
                && configValue.value != DETERMINISTIC_DISABLE) {
                HCCL_ERROR("[HcclSetConfig] HCCL_DETERMINISTIC is only support 0, 1 or 2");
                return HCCL_E_PARA;
            } else {
                DevType devType;
                CHK_RET(hrtGetDeviceType(devType));
                if (configValue.value == DETERMINISTIC_STRICT && devType != DevType::DEV_TYPE_910B
                    && devType != DevType::DEV_TYPE_910_93) {
                    HCCL_ERROR(
                        "[HcclSetConfig] configValue[%d], reduce order preservation is not supported for"
                        " devType[%d]",
                        configValue.value, devType);
                    return HCCL_E_NOT_SUPPORT;
                }
                CHK_RET(SetDeterministic(configValue.value));
                HCCL_INFO("[HcclSetConfig] Set HCCL_DETERMINISTIC to %u", configValue.value);
            }
        } else {
            HCCL_WARNING("[HcclSetConfig] HCCL_DETERMINISTIC has been set by Env, so will not be reset again");
            return HCCL_SUCCESS;
        }
        HcclOpInfoCtx& opBaseInfo = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
        // 遍历所有的通信域设置其确定性计算配置参数
        for (auto it = opBaseInfo.opGroup2CommMap.begin(); it != opBaseInfo.opGroup2CommMap.end(); it++) {
            CHK_RET(it->second->SetDeterministicConfig(configValue.value));
        }
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult HcclGetConfig(HcclConfig config, HcclConfigValue* configValue)
{
    CHK_PTR_NULL(configValue);
    if (config == HCCL_DETERMINISTIC) {
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
        HCCLV2_FUNC_RUN(HcclGetConfigV2(config, configValue));
#endif
        configValue->value = static_cast<int32_t>(GetExternalInputHcclDeterministicV2());
        HCCL_INFO("[HcclGetConfig] HCCL_DETERMINISTIC is [%d]", configValue->value);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclSetIfProfile()
{
    bool ifOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    bool state = ProfilingManagerPub::GetAllState();
    SetIfProfile((!ifOpbase) || (!state));
    return HCCL_SUCCESS;
}

void HcclResetIfProfile() { SetIfProfile(true); }

HcclResult
HcclBroadcastInner(void* buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    // 入参合法性校验
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return broadcast success"), HCCL_SUCCESS);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclBroadcastInner", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclBroadcastInner", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(
        buf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclBroadcastInner", "nullptr", "buf", "non-null pointer"}));
    CHK_PTR_NULL(buf);

    // Group特性
    if (hcclGroupDepth > 0) {
        struct hcclOpInfo info;
        info.coll = HcclCMDType::HCCL_CMD_BROADCAST;
        info.sendbuff = buf;
        info.sendCount = count;
        info.sendType = dataType;
        info.recvType = dataType;
        info.op = HCCL_REDUCE_SUM;
        info.root = root;
        info.comm = comm;
        info.stream = stream;
        CHK_RET(taskAppend(comm, info));
        HCCL_INFO(
            "[HcclBroadcast] Finish taskAppend, count [%d] dataType [%s] root [%u]", count,
            GetDataTypeEnumStr(dataType).c_str(), root);
        return HCCL_SUCCESS;
    }

    HcclUs startut = TIME_NOW();
    bool isCapture;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    uint64_t modelId = 0xFFFFFFFF;
    CHK_PRT(GetCaptureInfo(stream, captureStatus, modelId, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    uint64_t beginTime = hrtMsprofSysCycleTime();

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclBroadcastV2(buf, count, dataType, root, commV2, stream));
        return HCCL_SUCCESS;
    }());
#endif
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    s32 threadID = SalGetTid();
    ProfilingManagerPub::SetThreadCaptureStatus(threadID, isCapture);
    // 同通信域同算子复用tag
    const string tag = "Broadcast_" + hcclComm->GetIdentifier();

    CHK_RET(HcomCheckOpParam(tag.c_str(), count, dataType, stream));

    HcomCollOpInfo opInfo = {"", buf, buf, count, dataType, root, HCCL_REDUCE_RESERVED, 0};

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, root), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        CHK_RET(HcclDeviceRefresh(deviceLogicId));

        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(localRank), tag.c_str());

        s32 streamId = 0;
        CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], buf[%p], count[%llu], dataType[%s], root[%u], localRank[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), buf, count, GetDataTypeEnumStr(dataType).c_str(), root, localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclBroadcastInner:" + std::string(stackLogBuffer) + ", capture status["
                              + to_string(captureStatus) + "], model id[" + to_string(modelId) + "].";
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    }

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->CreateOpBasedResources(HcclCMDType::HCCL_CMD_BROADCAST, tag, opInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(buf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->BroadcastOutPlace(tag, buf, count, dataType, root, stream), tag.c_str());

    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_BROADCAST, beginTime, count, dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    ProfilingManagerPub::DeleteThreadCaptureStatus(threadID);

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo
            = "HcclBroadcastInner:success,take time: " + std::to_string(DURATION_US(endut - startut).count()) + " us,"
              + std::string(stackLogBuffer);
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    }
    std::string identifier = hcclComm->GetIdentifier();
    AlgType nslbAlgType;
    CHK_RET(hcclComm->GetAlgType(nslbAlgType, HcclCMDType::HCCL_CMD_BROADCAST));

    if (hcclNslbDp::GetInstance().GetGlobalCommTaskId() != 0) {
        AlgTypeLevel1 algValue = nslbAlgType.algoLevel1;
        uint8_t nslbAlg = hcclNslbDp::GetInstance().GetNslbLevel1AlgType(algValue);
        DevType devType;
        CHK_RET(hrtGetDeviceType(devType));
        if (devType == DevType::DEV_TYPE_910_93) {
            AlgTypeLevel2 algValue2 = nslbAlgType.algoLevel2;
            nslbAlg = hcclNslbDp::GetInstance().GetNslbLevel2AlgType(algValue2);
        }
        // NSLB 填充 表6
        hcclNslbDp::GetInstance().SetNslbDpRootRank(HcclCMDType::HCCL_CMD_BROADCAST, root, identifier, nslbAlg);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclReduceScatterInner(
    void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream)
{
    // 入参合法性校验
    CHK_PRT_RET(recvCount == 0, HCCL_WARNING("input recvCount is 0, return ReduceScatter success"), HCCL_SUCCESS);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclReduceScatterInner", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclReduceScatterInner", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(
        sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclReduceScatterInner", "nullptr", "sendBuf", "non-null pointer"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(
        recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclReduceScatterInner", "nullptr", "recvBuf", "non-null pointer"}));
    CHK_PTR_NULL(recvBuf);

    // Group特性
    if (hcclGroupDepth > 0) {
        struct hcclOpInfo info;
        info.coll = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
        info.sendbuff = sendBuf;
        info.recvbuff = recvBuf;
        info.recvCount = recvCount;
        info.sendType = dataType;
        info.recvType = dataType;
        info.op = op;
        info.comm = comm;
        info.stream = stream;
        CHK_RET(taskAppend(comm, info));
        HCCL_INFO(
            "[HcclReduceScatter] Finish taskAppend, count [%d] dataType [%s]", recvCount,
            GetDataTypeEnumStr(dataType).c_str());
        return HCCL_SUCCESS;
    }

    HcclUs startut = TIME_NOW();
    bool isCapture;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    uint64_t modelId = 0xFFFFFFFF;
    CHK_PRT(GetCaptureInfo(stream, captureStatus, modelId, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    uint64_t beginTime = hrtMsprofSysCycleTime();

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclReduceScatterV2(sendBuf, recvBuf, recvCount, dataType, op, commV2, stream));
        return HCCL_SUCCESS;
    }());
#endif
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    s32 threadID = SalGetTid();
    ProfilingManagerPub::SetThreadCaptureStatus(threadID, isCapture);
    // 同通信域同算子复用tag
    const string tag = "ReduceScatter_" + hcclComm->GetIdentifier();

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), recvCount, dataType, stream), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReductionOp("HcclReduceScatterInner", op), tag.c_str());
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    CHK_RET_AND_PRINT_IDE(HcomCheckReduceDataType(dataType, op, devType), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        CHK_RET(HcclDeviceRefresh(deviceLogicId));

        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(localRank), tag.c_str());

        s32 streamId = 0;
        CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], recvCount[%llu], dataType[%s], op[%s], "
            "localRank[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, recvCount, GetDataTypeEnumStr(dataType).c_str(),
            GetReduceOpEnumStr(op).c_str(), localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclReduceScatterInner:" + std::string(stackLogBuffer) + ", capture status["
                              + to_string(captureStatus) + "], model id[" + to_string(modelId) + "].";
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    }

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetOverFlowAddr(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(
        hcclComm->ReduceScatterOutPlace(tag, sendBuf, recvBuf, recvCount, dataType, op, stream), tag.c_str());
    CHK_RET(
        CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, beginTime, recvCount, dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    ProfilingManagerPub::DeleteThreadCaptureStatus(threadID);

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo
            = "HcclReduceScatterInner:success,take time: " + std::to_string(DURATION_US(endut - startut).count())
              + " us," + std::string(stackLogBuffer);
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult HcclReduceScatterVInner(
    void* sendBuf, const void* sendCounts, const void* sendDispls, void* recvBuf, uint64_t recvCount,
    HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    // 入参合法性校验
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclReduceScatterVInner", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    RPT_INPUT_ERR(
        sendCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclReduceScatterVInner", "nullptr", "sendCounts", "non-null pointer"}));
    CHK_PTR_NULL(sendCounts);
    RPT_INPUT_ERR(
        sendDispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclReduceScatterVInner", "nullptr", "sendDispls", "non-null pointer"}));
    CHK_PTR_NULL(sendDispls);
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclReduceScatterVInner", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    if (UNLIKELY(recvCount > 0 && recvBuf == nullptr)) {
        RPT_INPUT_ERR(
            true, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
            std::vector<std::string>({"HcclReduceScatterVInner", "nullptr", "recvBuf", "non-null pointer"}));
        CHK_PTR_NULL(recvBuf);
    }

    // Group特性
    if (hcclGroupDepth > 0) {
        struct hcclOpInfo info;
        info.coll = HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V;
        info.sendbuff = sendBuf;
        info.recvbuff = recvBuf;
        info.sendCounts = sendCounts;
        info.sdispls = sendDispls;
        info.recvCount = recvCount;
        info.sendType = dataType;
        info.recvType = dataType;
        info.op = op;
        info.comm = comm;
        info.stream = stream;
        CHK_RET(taskAppend(comm, info));
        HCCL_INFO(
            "[HcclReduceScatterV] Finish taskAppend, recvCount [%d] dataType [%s]", recvCount,
            GetDataTypeEnumStr(dataType).c_str());
        return HCCL_SUCCESS;
    }

    HcclUs startut = TIME_NOW();
    bool isCapture;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    uint64_t modelId = 0xFFFFFFFF;
    CHK_PRT(GetCaptureInfo(stream, captureStatus, modelId, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    s32 threadID = SalGetTid();
    ProfilingManagerPub::SetThreadCaptureStatus(threadID, isCapture);
    uint64_t beginTime = hrtMsprofSysCycleTime();

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclReduceScatterVV2(
            sendBuf, const_cast<void*>(sendCounts), const_cast<void*>(sendDispls), recvBuf, recvCount, dataType, op,
            commV2, stream));
        return HCCL_SUCCESS;
    }());
#endif
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);
    // 同通信域同算子复用tag
    const string tag = "ReduceScatterV_" + hcclComm->GetIdentifier();

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), recvCount, dataType, stream), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReductionOp("HcclReduceScatterVInner", op), tag.c_str());
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    CHK_RET_AND_PRINT_IDE(HcomCheckReduceDataType(dataType, op, devType), tag.c_str());

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(userRank), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), tag.c_str());

    u64 maxCount = 0;
    u64 inputCount = 0;
    u64* counts = static_cast<u64*>(const_cast<void*>(sendCounts));
    for (u32 i = 0; i < rankSize; i++) {
        CHK_PRT_RET(
            counts[i] > SYS_MAX_COUNT,
            HCCL_ERROR(
                "HcclReduceScatterVInner sendCounts[%u][%llu] is invalid.(bigger than MAX count[%llu])", i, counts[i],
                SYS_MAX_COUNT),
            HCCL_E_PARA);
        inputCount += counts[i];
        maxCount = std::max(maxCount, counts[i]);
    }
    CHK_PRT_RET(
        inputCount == 0,
        HCCL_WARNING("The inputCount is 0, this ReduceScatter v has no task to execute, "
                     "returning success."),
        HCCL_SUCCESS);
    RPT_INPUT_ERR(
        sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclReduceScatterVInner", "nullptr", "sendBuf", "non-null pointer"}));
    CHK_PTR_NULL(sendBuf);

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        CHK_RET(HcclDeviceRefresh(deviceLogicId));

        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(localRank), tag.c_str());

        s32 streamId = 0;
        CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], sendCounts[%p], sendDispls[%p], recvCount[%llu], dataType[%s], op[%s], "
            "localRank[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, sendCounts, sendDispls, recvCount, GetDataTypeEnumStr(dataType).c_str(),
            GetReduceOpEnumStr(op).c_str(), localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclReduceScatterVInner:" + std::string(stackLogBuffer) + ", capture status["
                              + to_string(captureStatus) + "], model id[" + to_string(modelId) + "].";
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    }

    PrintCountsAndDispls(rankSize, sendCounts, sendDispls, tag.c_str());

    CheckCountsAndDispls(rankSize, sendCounts, sendDispls, tag.c_str());

    const u64 countOfThisRank = static_cast<const u64*>(sendCounts)[userRank];
    CHK_PRT_RET(
        recvCount != countOfThisRank,
        HCCL_ERROR(
            "[HcclReduceScatterVInner] input recvCount[%llu] is not equal to sendCounts[%u][%llu]", recvCount, userRank,
            countOfThisRank),
        HCCL_E_PARA);

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());

    if (recvBuf != nullptr) {
        CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());
    }

    CHK_RET_AND_PRINT_IDE(SetOverFlowAddr(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(
        hcclComm->ReduceScatterVOutPlace(
            tag, sendBuf, recvBuf, sendCounts, sendDispls, recvCount, dataType, op, stream),
        tag.c_str());

    CHK_RET(
        CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, beginTime, maxCount, dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    ProfilingManagerPub::DeleteThreadCaptureStatus(threadID);

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo
            = "HcclReduceScatterVInner:success,take time: " + std::to_string(DURATION_US(endut - startut).count())
              + " us," + std::string(stackLogBuffer);
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult CheckScatterInputPara(const HcclComm comm, const void* recvBuf)
{
    // 入参合法性校验
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclScatterInner", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(
        recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclScatterInner", "nullptr", "recvBuf", "non-null pointer"}));
    CHK_PTR_NULL(recvBuf);

    return HCCL_SUCCESS;
}

HcclResult HcclScatterInner(
    void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root, HcclComm comm,
    aclrtStream stream)
{
    // 入参合法性校验
    CHK_PRT_RET(recvCount == 0, HCCL_WARNING("input recvCount is 0, return scatter success"), HCCL_SUCCESS);
    CHK_RET(CheckScatterInputPara(comm, recvBuf));
    CHK_PTR_NULL(stream);

    // Group特性
    if (hcclGroupDepth > 0) {
        struct hcclOpInfo info;
        info.coll = HcclCMDType::HCCL_CMD_SCATTER;
        info.sendbuff = sendBuf;
        info.recvbuff = recvBuf;
        info.recvCount = recvCount;
        info.sendType = dataType;
        info.recvType = dataType;
        info.root = root;
        info.comm = comm;
        info.stream = stream;
        CHK_RET(taskAppend(comm, info));
        HCCL_INFO(
            "[HcclScatter] Finish taskAppend, recvCount [%d] dataType [%s]", recvCount,
            GetDataTypeEnumStr(dataType).c_str());
        return HCCL_SUCCESS;
    }

    HcclUs startut = TIME_NOW();
    bool isCapture;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    uint64_t modelId = 0xFFFFFFFF;
    CHK_PRT(GetCaptureInfo(stream, captureStatus, modelId, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    uint64_t beginTime = hrtMsprofSysCycleTime();

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclScatterV2(sendBuf, recvBuf, recvCount, dataType, root, commV2, stream));
        return HCCL_SUCCESS;
    }());
#endif
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    s32 threadID = SalGetTid();
    ProfilingManagerPub::SetThreadCaptureStatus(threadID, isCapture);
    u32 commRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetUserRank(commRank));
    if (commRank == root) { // 本rank为root节点，send_buff不为空
        RPT_INPUT_ERR(
            sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
            std::vector<std::string>({"HcclScatterInner", "nullptr", "sendBuf", "non-null pointer"}));
        CHK_PTR_NULL(sendBuf);
    }

    // 同通信域同算子复用tag
    const string tag = "Scatter_" + hcclComm->GetIdentifier();

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), recvCount, dataType, stream), tag.c_str());

    HcomCollOpInfo opInfo = {"", sendBuf, recvBuf, recvCount, dataType, root, HCCL_REDUCE_RESERVED, 0};

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, root), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        CHK_RET(HcclDeviceRefresh(deviceLogicId));

        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(localRank), tag.c_str());

        s32 streamId = 0;
        CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], recvCount[%llu], dataType[%s], root[%u], localRank[%u], streamId[%d], "
            "deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, recvCount, GetDataTypeEnumStr(dataType).c_str(), root, localRank, streamId,
            deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclScatterInner:" + std::string(stackLogBuffer) + ", capture status["
                              + to_string(captureStatus) + "], model id[" + to_string(modelId) + "].";
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    }

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->CreateOpBasedResources(HcclCMDType::HCCL_CMD_SCATTER, tag, opInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(
        hcclComm->ScatterOutPlace(tag, sendBuf, recvBuf, recvCount, dataType, root, stream), tag.c_str());

    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_SCATTER, beginTime, recvCount, dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    ProfilingManagerPub::DeleteThreadCaptureStatus(threadID);

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo = "HcclScatterInner:success,take time: "
                              + std::to_string(DURATION_US(endut - startut).count()) + " us, tag: " + tag;
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    }
    std::string identifier = hcclComm->GetIdentifier();
    AlgType algType;
    CHK_RET(hcclComm->GetAlgType(algType, HcclCMDType::HCCL_CMD_SCATTER));

    if (hcclNslbDp::GetInstance().GetGlobalCommTaskId() != 0) {
        DevType nslb_devType;
        CHK_RET(hrtGetDeviceType(nslb_devType));
        AlgTypeLevel1 algValue = algType.algoLevel1;
        uint8_t nslbAlg = hcclNslbDp::GetInstance().GetNslbLevel1AlgType(algValue);
        if (nslb_devType == DevType::DEV_TYPE_910_93) {
            AlgTypeLevel2 algValue2 = algType.algoLevel2;
            nslbAlg = hcclNslbDp::GetInstance().GetNslbLevel2AlgType(algValue2);
        }
        // NSLB 填充 表6
        hcclNslbDp::GetInstance().SetNslbDpRootRank(HcclCMDType::HCCL_CMD_SCATTER, root, identifier, nslbAlg);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclAllGatherInner(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    // 入参合法性校验
    CHK_PRT_RET(sendCount == 0, HCCL_WARNING("input sendCount is 0, return HcclAllGatherInner success"), HCCL_SUCCESS);
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclAllGatherInner", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(
        sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclAllGatherInner", "nullptr", "sendBuf", "non-null pointer"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(
        recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclAllGatherInner", "nullptr", "recvBuf", "non-null pointer"}));
    CHK_PTR_NULL(recvBuf);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclAllGatherInner", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);

    // Group特性
    if (hcclGroupDepth > 0) {
        struct hcclOpInfo info;
        info.coll = HcclCMDType::HCCL_CMD_ALLGATHER;
        info.sendbuff = sendBuf;
        info.recvbuff = recvBuf;
        info.sendCount = sendCount;
        info.sendType = dataType;
        info.recvType = dataType;
        info.comm = comm;
        info.stream = stream;
        CHK_RET(taskAppend(comm, info));
        HCCL_INFO(
            "[HcclAllGather] Finish taskAppend, sendCount [%d] dataType [%s]", sendCount,
            GetDataTypeEnumStr(dataType).c_str());
        return HCCL_SUCCESS;
    }

    HcclUs startut = TIME_NOW();
    bool isCapture;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    uint64_t modelId = 0xFFFFFFFF;
    CHK_PRT(GetCaptureInfo(stream, captureStatus, modelId, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    uint64_t beginTime = hrtMsprofSysCycleTime();

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclAllGatherV2(sendBuf, recvBuf, sendCount, dataType, commV2, stream));
        return HCCL_SUCCESS;
    }());
#endif
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    s32 threadID = SalGetTid();
    ProfilingManagerPub::SetThreadCaptureStatus(threadID, isCapture);
    // 同通信域同算子复用tag
    const std::string tag = "AllGather_" + hcclComm->GetIdentifier();
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), sendCount, dataType, stream), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        CHK_RET(HcclDeviceRefresh(deviceLogicId));

        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(localRank), tag.c_str());

        s32 streamId = 0;
        CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], sendCount[%llu], dataType[%s], localRank[%u], streamId[%d], "
            "deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, sendCount, GetDataTypeEnumStr(dataType).c_str(), localRank, streamId,
            deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclAllGatherInner:" + std::string(stackLogBuffer) + ", capture status["
                              + to_string(captureStatus) + "], model id[" + to_string(modelId) + "].";
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    }

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->AllGatherOutPlace(tag, sendBuf, recvBuf, sendCount, dataType, stream), tag.c_str());
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLGATHER, beginTime, sendCount, dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    ProfilingManagerPub::DeleteThreadCaptureStatus(threadID);

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo
            = "HcclAllGatherInner:success,take time: " + std::to_string(DURATION_US(endut - startut).count()) + " us,"
              + std::string(stackLogBuffer);
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult HcclAllGatherVInner(
    void* sendBuf, uint64_t sendCount, void* recvBuf, const void* recvCounts, const void* recvDispls,
    HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    // 入参合法性校验
    RPT_INPUT_ERR(
        recvCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclAllGatherVInner", "nullptr", "recvCounts", "non-null pointer"}));
    CHK_PTR_NULL(recvCounts);
    RPT_INPUT_ERR(
        recvDispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclAllGatherVInner", "nullptr", "recvDispls", "non-null pointer"}));
    CHK_PTR_NULL(recvDispls);
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclAllGatherVInner", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclAllGatherVInner", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    if (UNLIKELY(sendCount > 0 && sendBuf == nullptr)) {
        RPT_INPUT_ERR(
            true, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
            std::vector<std::string>({"HcclAllGatherVInner", "nullptr", "sendBuf", "non-null pointer"}));
        CHK_PTR_NULL(sendBuf);
    }

    // Group特性
    if (hcclGroupDepth > 0) {
        struct hcclOpInfo info;
        info.coll = HcclCMDType::HCCL_CMD_ALLGATHER_V;
        info.sendbuff = sendBuf;
        info.recvbuff = recvBuf;
        info.sendCount = sendCount;
        info.recvCounts = recvCounts;
        info.rdispls = recvDispls;
        info.sendType = dataType;
        info.recvType = dataType;
        info.comm = comm;
        info.stream = stream;
        CHK_RET(taskAppend(comm, info));
        HCCL_INFO(
            "[HcclAllGatherV] Finish taskAppend, sendCount [%d] dataType [%s]", sendCount,
            GetDataTypeEnumStr(dataType).c_str());
        return HCCL_SUCCESS;
    }

    HcclUs startut = TIME_NOW();
    bool isCapture;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    uint64_t modelId = 0xFFFFFFFF;
    CHK_PRT(GetCaptureInfo(stream, captureStatus, modelId, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    s32 threadID = SalGetTid();
    ProfilingManagerPub::SetThreadCaptureStatus(threadID, isCapture);
    uint64_t beginTime = hrtMsprofSysCycleTime();

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclAllGatherVV2(
            sendBuf, sendCount, recvBuf, const_cast<void*>(recvCounts), const_cast<void*>(recvDispls), dataType, commV2,
            stream));
        return HCCL_SUCCESS;
    }());
#endif
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);
    // 同通信域同算子复用tag
    const std::string tag = "AllGatherV_" + hcclComm->GetIdentifier();
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), sendCount, dataType, stream), tag.c_str());

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(userRank), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), tag.c_str());

    u64 maxCount = 0;
    u64 outputCount = 0;
    u64* counts = static_cast<u64*>(const_cast<void*>(recvCounts));
    for (u32 i = 0; i < rankSize; i++) {
        CHK_PRT_RET(
            counts[i] > SYS_MAX_COUNT,
            HCCL_ERROR(
                "HcclAllGatherVInner recvCounts[%u][%llu] is invalid.(bigger than MAX count[%llu])", i, counts[i],
                SYS_MAX_COUNT),
            HCCL_E_PARA);
        outputCount += counts[i];
        maxCount = std::max(maxCount, counts[i]);
    }
    CHK_PRT_RET(
        outputCount == 0,
        HCCL_WARNING("The outputCount is 0, this AllGatherV has no task to execute, "
                     "returning success."),
        HCCL_SUCCESS);
    RPT_INPUT_ERR(
        recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclAllGatherVInner", "nullptr", "recvBuf", "non-null pointer"}));
    CHK_PTR_NULL(recvBuf);

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (GetExternalInputHcclEnableEntryLog()) {
        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(localRank), tag.c_str());

        s32 streamId = 0;
        CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

        s32 deviceLogicId = 0;
        CHK_RET(HcclDeviceRefresh(deviceLogicId));

        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], sendCount[%llu], recvCounts[%u], recvDispls[%u], "
            "dataType[%s], localRank[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, sendCount, recvCounts, recvDispls, GetDataTypeEnumStr(dataType).c_str(),
            localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclAllGatherVInner:" + std::string(stackLogBuffer) + ", capture status["
                              + to_string(captureStatus) + "], model id[" + to_string(modelId) + "].";
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    }

    PrintCountsAndDispls(rankSize, recvCounts, recvDispls, tag.c_str());

    CheckCountsAndDispls(rankSize, recvCounts, recvDispls, tag.c_str());

    const u64 countOfThisRank = static_cast<const u64*>(recvCounts)[userRank];
    CHK_PRT_RET(
        sendCount != countOfThisRank,
        HCCL_ERROR(
            "[HcclAllGatherVInner] input sendCount[%llu] is not equal to recvCounts[%u][%llu]", sendCount, userRank,
            countOfThisRank),
        HCCL_E_PARA);

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    if (sendBuf != nullptr) {
        CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());
    }

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(
        hcclComm->AllGatherVOutPlace(tag, sendBuf, recvBuf, sendCount, recvCounts, recvDispls, dataType, stream),
        tag.c_str());

    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLGATHER_V, beginTime, maxCount, dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    ProfilingManagerPub::DeleteThreadCaptureStatus(threadID);

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo
            = "HcclAllGatherVInner:success,take time: " + std::to_string(DURATION_US(endut - startut).count()) + " us,"
              + std::string(stackLogBuffer);
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult HcclSendInner(
    void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm, aclrtStream stream)
{
    // 入参合法性校验
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return HcclSendInner success."), HCCL_SUCCESS);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(stream);
    CHK_RET(HcomCheckCount(count));

    // Group特性
    if (hcclGroupDepth > 0) {
        HCCL_INFO("[HcclSendInner] groupDepth[%d]", hcclGroupDepth);
        struct hcclOpInfo info;
        info.coll = HcclCMDType::HCCL_CMD_SEND;
        info.sendbuff = sendBuf;
        info.sendCount = count;
        info.sendType = dataType;
        info.root = destRank;
        info.comm = comm;
        info.stream = stream;
        CHK_RET(taskAppend(comm, info));
        HCCL_INFO(
            "[HcclSend] Finish taskAppend, count [%lld] dataType [%s] destRank [%u]", count,
            GetDataTypeEnumStr(dataType).c_str(), destRank);
        return HCCL_SUCCESS;
    }

    HcclUs startut = TIME_NOW();
    bool isCapture;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    uint64_t modelId = 0xFFFFFFFF;
    CHK_PRT(GetCaptureInfo(stream, captureStatus, modelId, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    uint64_t beginTime = hrtMsprofSysCycleTime();

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclSendV2(sendBuf, count, dataType, destRank, commV2, stream));
        return HCCL_SUCCESS;
    }());
#endif
    CHK_RET(HcomCheckDataType(dataType));
    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    s32 threadID = SalGetTid();
    ProfilingManagerPub::SetThreadCaptureStatus(threadID, isCapture);
    // 同算子复用tag，为实现通信域复用，根据srRank和dstRank构造Tag
    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetGroupRank(localRank));

    const string tag = "worldCommSendRecv_" + std::to_string(localRank) + "_" + std::to_string(destRank) + "_"
                       + hcclComm->GetIdentifier();

    HcomCollOpInfo opInfo = {"", sendBuf, sendBuf, count, dataType, 0, HCCL_REDUCE_RESERVED, 0};
    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        CHK_RET(HcclDeviceRefresh(deviceLogicId));

        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], count[%llu], dataType[%s], localRank[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, count, GetDataTypeEnumStr(dataType).c_str(), localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclSendInner:" + std::string(stackLogBuffer) + ", capture status["
                              + to_string(captureStatus) + "], model id[" + to_string(modelId) + "].";
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    }

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->CreateOpBasedResources(HcclCMDType::HCCL_CMD_SEND, tag, opInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->SendOutPlace(tag, sendBuf, count, dataType, destRank, stream), tag.c_str());
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_SEND, beginTime, count, dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    ProfilingManagerPub::DeleteThreadCaptureStatus(threadID);

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo = "HcclSendInner:success,take time: " + std::to_string(DURATION_US(endut - startut).count())
                              + " us," + std::string(stackLogBuffer);
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult
HcclRecvInner(void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm, aclrtStream stream)
{
    // 入参合法性校验
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return HcclRecvInner success."), HCCL_SUCCESS);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(stream);
    CHK_RET(HcomCheckCount(count));

    // Group特性
    if (hcclGroupDepth > 0) {
        HCCL_INFO("[HcclRecvInner] groupDepth[%d]", hcclGroupDepth);
        struct hcclOpInfo info;
        info.coll = HcclCMDType::HCCL_CMD_RECEIVE;
        info.recvbuff = recvBuf;
        info.recvCount = count;
        info.recvType = dataType;
        info.root = srcRank;
        info.comm = comm;
        info.stream = stream;
        CHK_RET(taskAppend(comm, info));
        HCCL_INFO(
            "[HcclRecv] Finish taskAppend, count [%d] dataType [%s] srcRank [%u]", count,
            GetDataTypeEnumStr(dataType).c_str(), srcRank);
        return HCCL_SUCCESS;
    }

    HcclUs startut = TIME_NOW();
    bool isCapture;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    uint64_t modelId = 0xFFFFFFFF;
    CHK_PRT(GetCaptureInfo(stream, captureStatus, modelId, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    uint64_t beginTime = hrtMsprofSysCycleTime();

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclRecvV2(recvBuf, count, dataType, srcRank, commV2, stream));
        return HCCL_SUCCESS;
    }());
#endif
    CHK_RET(HcomCheckDataType(dataType));
    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    s32 threadID = SalGetTid();
    ProfilingManagerPub::SetThreadCaptureStatus(threadID, isCapture);
    // 同算子复用tag，为实现通信域复用，根据srRank和dstRank构造Tag
    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetGroupRank(localRank));

    const string tag = "worldCommSendRecv_" + std::to_string(srcRank) + "_" + std::to_string(localRank) + "_"
                       + hcclComm->GetIdentifier();

    HcomCollOpInfo opInfo = {"", recvBuf, recvBuf, count, dataType, 0, HCCL_REDUCE_RESERVED, 0};
    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        CHK_RET(HcclDeviceRefresh(deviceLogicId));

        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], recvBuf[%p], count[%llu], dataType[%s], localRank[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), recvBuf, count, GetDataTypeEnumStr(dataType).c_str(), localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclRecvInner:" + std::string(stackLogBuffer) + ", capture status["
                              + to_string(captureStatus) + "], model id[" + to_string(modelId) + "].";
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    }

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->CreateOpBasedResources(HcclCMDType::HCCL_CMD_RECEIVE, tag, opInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->ReceiveOutPlace(tag, recvBuf, count, dataType, srcRank, stream), tag.c_str());
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_RECEIVE, beginTime, count, dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    ProfilingManagerPub::DeleteThreadCaptureStatus(threadID);

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo = "HcclRecvInner:success,take time: " + std::to_string(DURATION_US(endut - startut).count())
                              + " us," + std::string(stackLogBuffer);
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedCommDestroy(HcclComm comm, s32 deviceLogicId, HcclUs startut)
{
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string group = hcclComm->GetIdentifier();
    CHK_PRT_RET(
        group.empty(), HCCL_ERROR("[HcclCommDestroy][HcclOneSidedCommDestroy] commName is error."), HCCL_E_PARA);
    HCCL_RUN_INFO("Entry-%s: deviceLogicId[%d], commName[%s]", __func__, deviceLogicId, group.c_str());

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HcclResult ret = hcclComm->DeinitOneSidedService();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcclCommDestroy][HcclOneSidedCommDestroy] Deinit one sided service"
            " failed, commName[%s].",
            group.c_str()),
        ret);
#endif

    HcclOpInfoCtx& opBaseHcom = GetOneSidedOpInfoCtx(deviceLogicId, hcclComm->GetIdentifier());

    g_oneSidedCommSet.erase(comm);
    DeleteOneSidedIdentifier(group);

    std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
    auto iter = opBaseHcom.opGroup2CommMap.find(group);
    if (iter != opBaseHcom.opGroup2CommMap.end()) {
        EXCEPTION_CATCH(opBaseHcom.opGroup2CommMap.erase(group), return HCCL_E_MEMORY);
        HcclCloseCommConnections(group);
    } else {
        HCCL_ERROR(
            "[HcclCommDestroy] comm is not exist, comm=%p, group=%s, deviceLogicId=%d", comm, group.c_str(),
            deviceLogicId);
        return HCCL_E_PARA;
    }

    opBaseHcom.isUsed = false;
    CHK_RET(DeInitOneSidedHcomInfo(deviceLogicId, group));

    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    HCCL_USER_CRITICAL_LOG(
        "op_base comm destroy complete, take time [%lld]us, group[%s], deviceLogicId[%d]", DURATION_US(endut - startut),
        group.c_str(), deviceLogicId);

    return HCCL_SUCCESS;
}
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
static HcclResult ResetDevice(hccl::hcclComm* hcclComm)
{
    s32 logicDeviceId = 0;
    CHK_RET(hcclComm->GetDeviceId(logicDeviceId));
    g_hcclDeviceId = logicDeviceId;
    if (hcclComm->IsNeedResetDevice()) {
        HCCL_RUN_INFO("op_base com destroy, com is not global com");
        HCCL_RUN_INFO("[HcclCommDestroy] reset logicDeviceId[%d]", logicDeviceId);
        CHK_PRT_RET(
            hrtResetDevice(logicDeviceId) != HCCL_SUCCESS,
            HCCL_ERROR("[HcclCommDestroy] reset fail logicDeviceId[%d]", logicDeviceId), HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}
#endif

HcclResult HcclCommDestroyWrapper([[maybe_unused]] struct hcclAsyncJob* job_)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    struct hcclCommDestroyAsyncJob* job = static_cast<hcclCommDestroyAsyncJob*>(job_);
    HcclComm comm = job->initComm;
    s32 devId = job->devId;
    HCCL_DEBUG("[HcclCommDestroyWrapper] Set device devId: %d", devId);
    CHK_PRT_RET(hrtSetDevice(devId) != HCCL_SUCCESS, HCCL_ERROR("[HcclCommDestroyWrapper] set fail"), HCCL_E_INTERNAL);
    HCCL_DEBUG("[HcclCommDestroyWrapper] Done Set device devId: %d", devId);

    HCCL_RUN_INFO("Entry-%s: op_base comm destroy begin", __func__);

    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    HcclResult ret = HcclDeviceRefresh(deviceLogicId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[HcclCommDestroy] Get device fail, comm=%p", comm), ret);
    CHK_PRT_RET(comm == nullptr, HCCL_WARNING("[Destroy][HcclComm]An empty comm given, skip destroy."), HCCL_SUCCESS);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclCommDestroyV2(
            commV2)); // 临时处理，dpustream的销毁要在其他资源销毁前完成。待新方案CpuThread上库后，原dpuStream删除可以恢复顺序
        string group = hcclComm->GetIdentifier();
        HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
        std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
        auto iter = opBaseHcom.opGroup2CommMap.find(group);
        if (iter != opBaseHcom.opGroup2CommMap.end()) {
            EXCEPTION_CATCH(opBaseHcom.opGroup2CommMap.erase(group), return HCCL_E_MEMORY);
        } else {
            HCCL_ERROR(
                "[HcclCommDestroy] comm is not exist, comm=%p, group=%s, deviceLogicId=%d", comm, group.c_str(),
                deviceLogicId);
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }());
#endif
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclCommState state = hcclComm->GetState();
    if (state == HcclCommState::INUSE) {
        HCCL_WARNING("[HcclCommDestroy] comm is in use, please try again later");
        return HCCL_E_AGAIN;
    }
    hcclComm->DeinitZeroCopyMemoryAgent();
    HCCL_RUN_INFO("[HcclCommDestroy] comm state is %s", HcclCommStateToString(state));

    CHK_RET(hcclComm->SetStopFlag(true));
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    CHK_RET(ResetDevice(hcclComm));

    if (IsOneSidedComm(comm)) {
        return HcclOneSidedCommDestroy(comm, deviceLogicId, startut);
    }

    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    string group;
    if (comm == opBaseHcom.pComm.get()) {
        group = opBaseHcom.pComm->GetIdentifier();
        opBaseHcom.pComm = nullptr;
        HcclCloseCommConnections(group);
    } else {
        HCCL_RUN_INFO("com is not global com");
        group = hcclComm->GetIdentifier();
    }

    // 特殊场景，当comm name被手动配置为HCCL_WORLD_GROUP时，需要将hcomInfo.pComm设为nullptr
    if (hcclComm->GetIdentifier() == HCCL_WORLD_GROUP) {
        HcomGetCtxHomInfo().pComm = nullptr;
    }

    HcomUnSetGroupTopoInfo(group.c_str());

    std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
    auto iter = opBaseHcom.opGroup2CommMap.find(group);
    if (iter != opBaseHcom.opGroup2CommMap.end()) {
        EXCEPTION_CATCH(opBaseHcom.opGroup2CommMap.erase(group), return HCCL_E_MEMORY);
        HcclCloseCommConnections(group);
    } else {
        HCCL_ERROR(
            "[HcclCommDestroy] comm is not exist, comm=%p, group=%s, deviceLogicId=%d", comm, group.c_str(),
            deviceLogicId);
        return HCCL_E_PARA;
    }

    if (ProfilingManagerPub::GetAllState()) {
        ProfilingManagerPub::ClearStoragedProfilingInfo();
    }

    HcclUs endut = TIME_NOW();

    // 删除groupName和UDI的映射
    HCCL_PROFILER_DEL_GROUP_UDI(group);

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "op_base comm destroy complete, take time [%lld]us, group[%s], deviceLogicId[%d].",
        DURATION_US(endut - startut), group.c_str(), deviceLogicId);
#endif
    return HCCL_SUCCESS;
}

HcclResult HcclCommDestroy([[maybe_unused]] HcclComm comm)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    if (hcclGroupDepth > 0) {
        std::shared_ptr<struct hcclCommDestroyAsyncJob> job;
        EXCEPTION_CATCH((job = std::make_shared<struct hcclCommDestroyAsyncJob>()), return HCCL_E_PARA);
        job->initComm = comm;
        s32 devId = 0;
        HcclResult ret = HcclDeviceRefresh(devId);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Group][HcclCommDestroy] Get device fail, comm=%p", comm), ret);
        job->devId = devId;
        ret = commInitTaskAppend(job, HcclCommDestroyWrapper, &comm);
        return ret;
    }
    HCCL_RUN_INFO("Entry-%s: op_base comm destroy begin", __func__);

    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    HcclResult ret = HcclDeviceRefresh(deviceLogicId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[HcclCommDestroy] Get device fail, comm=%p", comm), ret);
    CHK_PRT_RET(comm == nullptr, HCCL_WARNING("[Destroy][HcclComm]An empty comm given, skip destroy."), HCCL_SUCCESS);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        CHK_RET(HcclCommStateNotify(comm, HcclCommStatePhase::HCCL_COMM_STATE_PHASE_DESTROY_PRE));
        // 先拷贝orion通信域地址，避免coll comm销毁后无法获取
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_RET(HcclCommDestroyV2(
            commV2)); // 临时处理，dpustream的销毁要在其他资源销毁前完成。待新方案CpuThread上库后，原dpuStream删除可以恢复顺序
        string group = hcclComm->GetIdentifier();
        HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
        // opGroupMapMutex 仅覆盖对 opGroup2CommMap 的 find/erase 避免回调里销毁子通信域时同线程递归加锁死锁
        {
            std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
            auto iter = opBaseHcom.opGroup2CommMap.find(group);
            if (iter != opBaseHcom.opGroup2CommMap.end()) {
                EXCEPTION_CATCH(opBaseHcom.opGroup2CommMap.erase(group), return HCCL_E_MEMORY);
            } else {
                HCCL_ERROR(
                    "[HcclCommDestroy] comm is not exist, comm=%p, group=%s, deviceLogicId=%d", comm, group.c_str(),
                    deviceLogicId);
                return HCCL_E_PARA;
            }
        }
        CHK_RET(HcclCommStateNotify(comm, HcclCommStatePhase::HCCL_COMM_STATE_PHASE_DESTROY_POST));
        HCCL_RUN_INFO(
            "Entry-HcclCommDestroy V2 group[%s] destroy success, deviceLogicId[%d], comm[%p]", group.c_str(),
            deviceLogicId, comm);
        return HCCL_SUCCESS;
    }());
#endif
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclCommState state = hcclComm->GetState();
    if (state == HcclCommState::INUSE) {
        HCCL_WARNING("[HcclCommDestroy] comm is in use, please try again later");
        return HCCL_E_AGAIN;
    }
    hcclComm->DeinitZeroCopyMemoryAgent();
    HCCL_RUN_INFO("[HcclCommDestroy] comm state is %s", HcclCommStateToString(state));
    CHK_RET(hcclComm->RealeaseShareCCLbuffer());
    CHK_RET(hcclComm->SetStopFlag(true));
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    CHK_RET(ResetDevice(hcclComm));

    std::unique_lock<std::mutex> oneSideLock(g_opHcomOneSideMutex);
    if (IsOneSidedComm(comm)) {
        return HcclOneSidedCommDestroy(comm, deviceLogicId, startut);
    }
    oneSideLock.unlock();

    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
    string group;
    if (comm == opBaseHcom.pComm.get()) {
        group = opBaseHcom.pComm->GetIdentifier();
        opBaseHcom.pComm = nullptr;
        HcclCloseCommConnections(group);
    } else {
        HCCL_RUN_INFO("com is not global com");
        group = hcclComm->GetIdentifier();
    }

    // 特殊场景，当comm name被手动配置为HCCL_WORLD_GROUP时，需要将hcomInfo.pComm设为nullptr
    if (hcclComm->GetIdentifier() == HCCL_WORLD_GROUP) {
        HcomGetCtxHomInfo().pComm = nullptr;
    }

    HcomUnSetGroupTopoInfo(group.c_str());

    std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
    auto iter = opBaseHcom.opGroup2CommMap.find(group);
    if (iter != opBaseHcom.opGroup2CommMap.end()) {
        EXCEPTION_CATCH(opBaseHcom.opGroup2CommMap.erase(group), return HCCL_E_MEMORY);
        HcclCloseCommConnections(group);
    } else {
        HCCL_ERROR(
            "[HcclCommDestroy] comm is not exist, comm=%p, group=%s, deviceLogicId=%d", comm, group.c_str(),
            deviceLogicId);
        return HCCL_E_PARA;
    }

    if (ProfilingManagerPub::GetAllState()) {
        ProfilingManagerPub::ClearStoragedProfilingInfo();
    }

    HcclUs endut = TIME_NOW();

    // 删除groupName和UDI的映射
    HCCL_PROFILER_DEL_GROUP_UDI(group);

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "Entry-HcclCommDestroy op_base comm destroy complete, take time [%lld]us, group[%s], deviceLogicId[%d].",
        DURATION_US(endut - startut), group.c_str(), deviceLogicId);
#endif
    return HCCL_SUCCESS;
}

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
static HcclConfigTypeOpExpansionMode OpExpansionModeValueToModeEnum(const uint32_t value)
{
    constexpr uint32_t defaultMode = 0;
    constexpr uint32_t hostTsMode = 1;
    constexpr uint32_t aicpuTsMode = 2;
    constexpr uint32_t aivMode = 3;
    constexpr uint32_t aivOnlyMode = 4;
    constexpr uint32_t ccuMsMode = 5;
    constexpr uint32_t ccuSchedMode = 6;
    constexpr uint32_t aicpuMode = 7;

    switch (value) {
        case defaultMode:
            return HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_CCU_SCHED;
        case hostTsMode:
            return HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_HOST_TS;
        case aicpuTsMode:
        case aicpuMode:
            return HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AI_CPU;
        case ccuMsMode:
            return HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_CCU_MS;
        case ccuSchedMode:
            return HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_CCU_SCHED;
        case aivMode:
            return HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AIV;
        case aivOnlyMode:
            return HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY;
        default:
            break;
    }

    return HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_INVALID;
}
#endif

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
static HcclResult GetOpExpansionModeConfig(hccl::CollComm* collComm, uint32_t infoLen, void* info)
{
    constexpr size_t infoExpectedLen = sizeof(HcclConfigTypeOpExpansionMode);
    if (static_cast<size_t>(infoLen) != infoExpectedLen) {
        HCCL_ERROR("[%s] infoLen[%u] less than expected[%zu].", __func__, infoLen, infoExpectedLen);
        return HcclResult::HCCL_E_PARA;
    }
    auto* myRank = collComm->GetMyRank();
    CHK_PTR_NULL(myRank);
    const uint32_t opExpansionModeValue = myRank->GetOpExpansionMode();
    const auto opExpansionMode = OpExpansionModeValueToModeEnum(opExpansionModeValue);
    if (opExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_INVALID) {
        HCCL_ERROR("[%s] unknown expansion mode[%d].", __func__, opExpansionMode);
        return HcclResult::HCCL_E_INTERNAL;
    }
    *static_cast<HcclConfigTypeOpExpansionMode*>(info) = opExpansionMode;
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult GetHcclAlgoConfig(hccl::CollComm* collComm, uint32_t infoLen, void* info)
{
    constexpr size_t infoExpectedLen = static_cast<size_t>(HCCL_COMM_ALGO_MAX_LENGTH);
    if (static_cast<size_t>(infoLen) < infoExpectedLen) {
        HCCL_ERROR("[%s] infoLen[%u] less than expected[%zu].", __func__, infoLen, infoExpectedLen);
        return HcclResult::HCCL_E_PARA;
    }
    const std::string& hcclAlgo = collComm->GetCommConfig().GetConfigHcclAlgoStr();
    auto* algoInfo = static_cast<char*>(info);
    int32_t ret = memset_s(algoInfo, infoLen, 0, infoLen);
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s] memset error, ret[%d].", __func__, ret), HCCL_E_INTERNAL);
    if (!hcclAlgo.empty() && hcclAlgo.size() < infoLen) {
        ret = memcpy_s(algoInfo, infoLen, hcclAlgo.c_str(), hcclAlgo.size() + 1);
    } else if (!hcclAlgo.empty()) {
        ret = memcpy_s(algoInfo, infoLen, hcclAlgo.c_str(), infoLen);
    }
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s] memcpy error, ret[%d].", __func__, ret), HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

// UB多channel数量为进程级全局配置（解析自环境变量、进程内唯一），不依赖通信域，
// 故不同于 GetOpExpansionModeConfig/GetHcclAlgoConfig，无需 collComm 入参。
static HcclResult GetUbMultiChannelNumConfig(uint32_t infoLen, void* info)
{
    constexpr size_t infoExpectedLen = sizeof(uint32_t);
    if (static_cast<size_t>(infoLen) != infoExpectedLen) {
        HCCL_ERROR("[%s] infoLen[%u] not expected[%zu].", __func__, infoLen, infoExpectedLen);
        return HcclResult::HCCL_E_PARA;
    }
    const uint32_t num = hccl::GetEnvUbConfig().GetUbMultiChannelNum();
    *static_cast<uint32_t*>(info) = num;
    return HcclResult::HCCL_SUCCESS;
}
#endif

HcclResult HcclConfigGetInfo(
    [[maybe_unused]] HcclComm comm, [[maybe_unused]] HcclConfigType cfgType, [[maybe_unused]] uint32_t infoLen,
    [[maybe_unused]] void* info)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(info);
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    auto* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    switch (cfgType) {
        case HcclConfigType::HCCL_CONFIG_TYPE_OP_EXPANSION_MODE:
            return GetOpExpansionModeConfig(collComm, infoLen, info);
        case HcclConfigType::HCCL_CONFIG_TYPE_HCCL_ALGO:
            return GetHcclAlgoConfig(collComm, infoLen, info);
        case HcclConfigType::HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM:
            return GetUbMultiChannelNumConfig(infoLen, info);
        default:
            HCCL_ERROR("[%s] cfgType[%d] is invalid.", __func__, cfgType);
            return HcclResult::HCCL_E_PARA;
    }
#endif
    HCCL_ERROR("[%s] is not support for hccd or kernel.", __func__);
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult HcclGenerateCommId(hccl::HcclCommParams& params)
{
    s32 sRet = memset_s(params.id.internal, HCCL_ROOT_INFO_BYTES, 0, sizeof(params.id.internal));
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[GenerateCommId]memory set error. return[%d].", sRet), HCCL_E_PARA);

    HcclRootInfo uniqueId;
    std::string group;
    CHK_RET(hcclComm::GetUniqueId(&uniqueId));

    if (!params.isHeterogComm) {
        group = "hccl_world_group";
    } else {
        group = "hccl_heterog_group";
    }

    sRet = snprintf_s(
        params.id.internal, HCCL_ROOT_INFO_BYTES, HCCL_ROOT_INFO_BYTES - 1, "%s%s%s", uniqueId.internal, "-",
        group.c_str());
    CHK_PRT_RET(
        sRet == -1,
        HCCL_ERROR("[GenerateCommId]errNo[0x%016llx] sal snprintf_s error", HCCL_ERROR_CODE(HCCL_E_INTERNAL)),
        HCCL_E_INTERNAL);
    HCCL_INFO("params.id.internal [%s]", params.id.internal);
    return HCCL_SUCCESS;
}

HcclResult InitOtherInfo(hccl::HcclCommParams& params, const char* rankTable)
{
    // 记录版本信息
    std::string curVersion = GetExternalInputCannVersion();
    CHK_RET(RankConsistentcyChecker::GetInstance().RecordVerInfo(curVersion));

    // ranktableCRC计算
    if (rankTable == nullptr) {
        HCCL_INFO("rank table is null, rankTableCrc is 0.");
    } else {
        HcclResult ret = HcomCalcCRC(params, rankTable);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][OtherInfo]errNo[0x%016llx] calc ranktable crc error", HCCL_ERROR_CODE(HCCL_E_INTERNAL)),
            HCCL_E_INTERNAL);
    }

    // 生成通信域标识符
    HcclResult ret = HcclGenerateCommId(params);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][OtherInfo]errNo[0x%016llx] generate CommId error, params: dest[%p]",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), params.id.internal),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterLoop(
    const std::string& tag, void* inputPtr, void* outputPtr, const u64& count, HcclDataType dataType, HcclReduceOp op,
    hccl::hcclComm* hcclComm, rtStream_t stream)
{
    HcclResult ret;
    void* commInputPtr = nullptr;
    void* commOutputPtr = nullptr;
    u64 commInputSize, commOutputSize;

    CHK_RET(hcclComm->GetInCCLbuffer(commInputPtr, commInputSize));

    CHK_RET(hcclComm->GetOutCCLbuffer(commOutputPtr, commOutputSize));

    u32 unitSize;
    CHK_RET(SalGetDataTypeSize(dataType, unitSize));

    char* curInputPtr = static_cast<char*>(inputPtr);
    char* curOutputPtr = static_cast<char*>(outputPtr);
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(hcclComm->GetRankSize(rankSize));

    CHK_PRT_RET(rankSize * unitSize == 0, HCCL_ERROR("The result of rankSize * unitSize is 0"), HCCL_E_PARA);
    u64 maxCountPerLoop = commInputSize / (rankSize * unitSize); // 中转内存单次最多能够接受的output count
    u64 curCount = 0;

    for (u64 countLeft = count, inputOffset = 0, outputOffset = 0; countLeft > 0; countLeft -= curCount) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        HCCL_INFO("-OP_BASE-ReduceScatterLoop:inputOffset[%llu], outputOffset[%llu]", inputOffset, outputOffset);
        // 判断剩余数据量对应的input size是否大于中转input size
        curCount = ((countLeft * unitSize * rankSize) > commInputSize) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节

        for (u32 i = 0; i < rankSize; i++) {
            // 拷贝input上每个slice的数据到中转内存，源端每个slice的size固定为output的size
            ret = hrtMemAsyncCopy(
                static_cast<char*>(commInputPtr) + curSize * i, curSize, curInputPtr + count * unitSize * i, curSize,
                HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Loop][ReduceScatter]In OP_BASE inputbuffer transit,[%u]slice memcopy "
                    "failed",
                    i),
                HCCL_E_MEMORY);
        }

        ret = hcclComm->ReduceScatter(tag, commInputPtr, commOutputPtr, curCount, dataType, op, stream);

        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Loop][ReduceScatter]errNo[0x%016llx] op_base hcclComm ReduceScatter error, "
                "tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s]",
                HCCL_ERROR_CODE(ret), tag.c_str(), commInputPtr, commOutputPtr, curCount,
                GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str()),
            ret);

        CHK_RET(hrtMemAsyncCopy(
            curOutputPtr, curSize, commOutputPtr, curSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE,
            stream));

        CHK_PRT_RET((curCount == 0), HCCL_ERROR("[Loop][ReduceScatter]In OP_BASE curCount is zero"), HCCL_E_PARA);
        inputOffset = curSize;
        outputOffset = curSize;
    }

    return HCCL_SUCCESS;
}

// 获取算子所需workspace memory大小[byte]
HcclResult HcclGetOpBasedMemSize(const HcclCMDType& opType, u64& size, const HcomCollOpInfo& opInfo)
{
    u64 opMemSize = 0;

    if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        // ReduceScatter 算子所需memory大小为 GetExternalInputCCLBuffSize()
        DevType devType;
        CHK_RET(hrtGetDeviceType(devType));
        if (IsSupportSDMAReduce(opInfo.inputAddr, opInfo.outputAddr, opInfo.dataType, opInfo.reduceOp)
            && IsSupportRDMAReduce(opInfo.dataType, opInfo.reduceOp) && devType == DevType::DEV_TYPE_910B) {
            opMemSize = 0;
        } else {
            opMemSize = GetExternalInputCCLBuffSize();
        }
    } else {
        opMemSize = 0;
    }
    size = HCCL_WORKSPACE_MEM_32_KB + opMemSize;
    HCCL_INFO("workspace memory size: op[%d], memory size[%llu]", opType, size);
    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAllInner(
    const void* sendBuf, uint64_t sendCount, HcclDataType sendType, const void* recvBuf, uint64_t recvCount,
    HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    // 入参校验
    CHK_PRT_RET(
        sendCount == 0 && recvCount == 0, HCCL_WARNING("sendCount and recvCount are both 0, return AllToAll success"),
        HCCL_SUCCESS);
    RPT_INPUT_ERR(
        sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclAlltoAllInner", "nullptr", "sendBuf", "non-null pointer"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(
        recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclAlltoAllInner", "nullptr", "recvBuf", "non-null pointer"}));
    CHK_PTR_NULL(recvBuf);
    CHK_PRT_RET(
        sendCount != recvCount,
        HCCL_ERROR("sendCount[%lu] and recvCount[%lu] are not equal, please check params", sendCount, recvCount),
        HCCL_E_PARA);
    CHK_PRT_RET(
        sendType != recvType,
        HCCL_ERROR(
            "sendType[%s] and recvType[%s] are not equal, please check params", GetDataTypeEnumStr(sendType).c_str(),
            GetDataTypeEnumStr(recvType).c_str()),
        HCCL_E_PARA);

    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclAlltoAllInner", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclAlltoAllInner", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    CHK_PRT_RET(sendBuf == recvBuf, HCCL_ERROR("[HcclAlltoAllInner] sendBuf and recvBuf cannot be same."), HCCL_E_PARA);

    // Group特性
    if (hcclGroupDepth > 0) {
        struct hcclOpInfo info;
        info.coll = HcclCMDType::HCCL_CMD_ALLTOALL;
        info.sendbuff = const_cast<void*>(sendBuf);
        info.recvbuff = const_cast<void*>(recvBuf);
        info.sendCount = sendCount;
        info.recvCount = recvCount;
        info.sendType = sendType;
        info.recvType = recvType;
        info.comm = comm;
        info.stream = stream;
        CHK_RET(taskAppend(comm, info));
        HCCL_INFO(
            "[HcclAlltoAll] Finish taskAppend, sendCount [%d] sendType [%s] recvCount [%d] recvType [%s]", sendCount,
            GetDataTypeEnumStr(sendType).c_str(), recvCount, GetDataTypeEnumStr(recvType).c_str());
        return HCCL_SUCCESS;
    }

    HcclUs startut = TIME_NOW();
    bool isCapture;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    uint64_t modelId = 0xFFFFFFFF;
    CHK_PRT(GetCaptureInfo(stream, captureStatus, modelId, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }

    uint64_t beginTime = hrtMsprofSysCycleTime();

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclAlltoAllV2(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, commV2, stream));
        return HCCL_SUCCESS;
    }());
#endif
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    s32 threadID = SalGetTid();
    ProfilingManagerPub::SetThreadCaptureStatus(threadID, isCapture);
    const std::string tag = HCCL_ALLTOALL + "_" + hcclComm->GetIdentifier();
    CHK_RET(HcomCheckOpParam(tag.c_str(), 0, sendType, stream));
    CHK_RET(HcomCheckDataType(recvType));
    // 接口交互信息日志
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        CHK_RET(HcclDeviceRefresh(deviceLogicId));

        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(localRank), tag.c_str());

        s32 streamId = 0;
        CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendCount[%llu], recvCount[%llu], sendType[%s], recvType[%s], localRank[%u], streamId[%d], "
            "deviceLogicId[%d]",
            tag.c_str(), sendCount, recvCount, GetDataTypeEnumStr(sendType).c_str(),
            GetDataTypeEnumStr(recvType).c_str(), localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclAlltoAllInner:" + std::string(stackLogBuffer) + ", capture status["
                              + to_string(captureStatus) + "], model id[" + to_string(modelId) + "].";
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    }

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());
    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(
        hcclComm->AlltoAll(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, stream, tag), tag.c_str());
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLTOALL, beginTime, sendCount, sendType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    ProfilingManagerPub::DeleteThreadCaptureStatus(threadID);

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        std::string endInfo
            = "HcclAlltoAllInner:success,take time: " + std::to_string(DURATION_US(endut - startut).count()) + " us,"
              + std::string(stackLogBuffer);
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    }

    return HCCL_SUCCESS;
}

// sendBuf & recvBuf为device mem, 其它为host mem
HcclResult HcclAlltoAllVInner(
    const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
    const void* recvCounts, const void* rdispls, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    // 入参校验
    RPT_INPUT_ERR(
        sendCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclAlltoAllVInner", "nullptr", "sendCounts", "non-null pointer"}));
    CHK_PTR_NULL(sendCounts);
    RPT_INPUT_ERR(
        sdispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclAlltoAllVInner", "nullptr", "sdispls", "non-null pointer"}));
    CHK_PTR_NULL(sdispls);
    RPT_INPUT_ERR(
        recvCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclAlltoAllVInner", "nullptr", "recvCounts", "non-null pointer"}));
    CHK_PTR_NULL(recvCounts);
    RPT_INPUT_ERR(
        rdispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclAlltoAllVInner", "nullptr", "rdispls", "non-null pointer"}));
    CHK_PTR_NULL(rdispls);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclAlltoAllVInner", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclAlltoAllVInner", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);

    // Group特性
    if (hcclGroupDepth > 0) {
        struct hcclOpInfo info;
        info.coll = HcclCMDType::HCCL_CMD_ALLTOALLV;
        info.sendbuff = const_cast<void*>(sendBuf);
        info.recvbuff = const_cast<void*>(recvBuf);
        info.sendCounts = sendCounts;
        info.recvCounts = recvCounts;
        info.sdispls = sdispls;
        info.rdispls = rdispls;
        info.sendType = sendType;
        info.recvType = recvType;
        info.comm = comm;
        info.stream = stream;
        CHK_RET(taskAppend(comm, info));
        HCCL_INFO(
            "[HcclAlltoAllV] Finish taskAppend, sendType [%s] recvType [%s]", GetDataTypeEnumStr(sendType).c_str(),
            GetDataTypeEnumStr(recvType).c_str());
        return HCCL_SUCCESS;
    }

    HcclUs startut = TIME_NOW();
    bool isCapture;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    uint64_t modelId = 0xFFFFFFFF;
    CHK_PRT(GetCaptureInfo(stream, captureStatus, modelId, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    uint64_t beginTime = hrtMsprofSysCycleTime();

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclAlltoAllVV2(
            sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, commV2, stream));
        return HCCL_SUCCESS;
    }());
#endif
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    s32 threadID = SalGetTid();
    ProfilingManagerPub::SetThreadCaptureStatus(threadID, isCapture);
    u32 rankSize = 0;
    CHK_RET(hcclComm->GetRankSize(rankSize));
    CHK_RET(HcomCheckAlltoAllVExternalMem(sendBuf, sendCounts, recvBuf, recvCounts, rankSize));

    const std::string tag = HCCL_ALLTOALLV + "_" + hcclComm->GetIdentifier();
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), 0, sendType, stream), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckDataType(recvType), tag.c_str());
    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        CHK_RET(HcclDeviceRefresh(deviceLogicId));

        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(localRank), tag.c_str());

        s32 streamId = 0;
        CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], sendCounts[%p], recvCounts[%p], sendType[%s], "
            "recvType[%s], localRank[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, sendCounts, recvCounts, GetDataTypeEnumStr(sendType).c_str(),
            GetDataTypeEnumStr(recvType).c_str(), localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclAlltoAllVInner:" + std::string(stackLogBuffer) + ", capture status["
                              + to_string(captureStatus) + "], model id[" + to_string(modelId) + "].";
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    }

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    if (sendBuf != nullptr) {
        CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());
    }
    if (recvBuf != nullptr) {
        CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());
    }

    if (!GetExternalInputHcclEnableFfts()) {
        CHK_RET_AND_PRINT_IDE(
            hcclComm->AlltoAllV(
                sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, stream, tag),
            tag.c_str());
    } else {
        CHK_RET_AND_PRINT_IDE(
            hcclComm->AlltoAllVOutPlace(
                sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, stream, tag),
            tag.c_str());
    }

    u64 sendCount = 0;
    for (u32 i = 0; i < rankSize; i++) {
        sendCount += *(static_cast<const u64*>(sendCounts) + i);
    }
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLTOALLV, beginTime, sendCount, sendType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    ProfilingManagerPub::DeleteThreadCaptureStatus(threadID);

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo
            = "HcclAlltoAllVInner:success,take time: " + std::to_string(DURATION_US(endut - startut).count()) + " us,"
              + std::string(stackLogBuffer);
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAllVCInner(
    const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType,
    HcclComm comm, rtStream_t stream)
{
    // 参数校验
    RPT_INPUT_ERR(
        sendCountMatrix == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclAlltoAllVCInner", "nullptr", "sendCountMatrix", "non-null pointer"}));
    CHK_PTR_NULL(sendCountMatrix);
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclAlltoAllVCInner", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclAlltoAllVCInner", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);

    // Group特性
    if (hcclGroupDepth > 0) {
        struct hcclOpInfo info;
        info.coll = HcclCMDType::HCCL_CMD_ALLTOALLVC;
        info.sendbuff = const_cast<void*>(sendBuf);
        info.recvbuff = const_cast<void*>(recvBuf);
        info.sendCounts = sendCountMatrix;
        info.sendType = sendType;
        info.recvType = recvType;
        info.comm = comm;
        info.stream = stream;
        CHK_RET(taskAppend(comm, info));
        HCCL_INFO(
            "[HcclAlltoAllVC] Finish taskAppend, sendType [%s] recvType [%s]", GetDataTypeEnumStr(sendType).c_str(),
            GetDataTypeEnumStr(recvType).c_str());
        return HCCL_SUCCESS;
    }

    HcclUs startut = TIME_NOW();
    bool isCapture;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    uint64_t modelId = 0xFFFFFFFF;
    CHK_PRT(GetCaptureInfo(stream, captureStatus, modelId, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    s32 threadID = SalGetTid();
    ProfilingManagerPub::SetThreadCaptureStatus(threadID, isCapture);
    uint64_t beginTime = hrtMsprofSysCycleTime();

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclAlltoAllVCV2(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, commV2, stream));
        return HCCL_SUCCESS;
    }());
#endif
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);

    u32 rankSize = 0;
    CHK_RET(hcclComm->GetRankSize(rankSize));
    u32 rank = 0;
    hcclComm->GetUserRank(rank);
    u32 userRank = 0;
    hcclComm->GetGroupRank(userRank);

    CHK_RET(HcomCheckAlltoAllVCExternalMem(sendBuf, sendCountMatrix, recvBuf, rankSize, rank));
    const std::string tag = HCCL_ALLTOALLVC + "_" + hcclComm->GetIdentifier();
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), 0, sendType, stream), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckDataType(recvType), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (GetExternalInputHcclEnableEntryLog()) {
        u64 sendCountMatrixHash;
        HcomGetHashFromSendCountMatrix(sendCountMatrixHash, sendCountMatrix, rankSize, tag);

        s32 deviceLogicId = 0;
        CHK_RET(HcclDeviceRefresh(deviceLogicId));

        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(localRank), tag.c_str());

        s32 streamId = 0;
        CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], sendCountMatrixHash[%llu], sendType[%s], recvBuf[%p], "
            "recvType[%s], localRank[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, sendCountMatrixHash, GetDataTypeEnumStr(sendType).c_str(), recvBuf,
            GetDataTypeEnumStr(recvType).c_str(), localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclAlltoAllVCInner:" + std::string(stackLogBuffer) + ", capture status["
                              + to_string(captureStatus) + "], model id[" + to_string(modelId) + "].";
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    }

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    if (sendBuf != nullptr) {
        CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());
    }
    if (recvBuf != nullptr) {
        CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());
    }

    if (!GetExternalInputHcclEnableFfts()) {
        CHK_RET_AND_PRINT_IDE(
            hcclComm->AlltoAllVC(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, stream, tag), tag.c_str());
    } else {
        CHK_RET_AND_PRINT_IDE(
            hcclComm->AlltoAllVCOutPlace(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, stream, tag),
            tag.c_str());
    }
    u64 sendCount = 0;
    for (u32 i = 0; i < rankSize; i++) {
        sendCount += *(static_cast<const u64*>(sendCountMatrix) + userRank * rankSize + i);
    }
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLTOALLVC, beginTime, sendCount, sendType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    ProfilingManagerPub::DeleteThreadCaptureStatus(threadID);

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo
            = "HcclAlltoAllVCInner:success,take time: " + std::to_string(DURATION_US(endut - startut).count()) + " us,"
              + std::string(stackLogBuffer);
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult HcclReduceInner(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint32_t root, HcclComm comm,
    aclrtStream stream)
{
    // 入参合法性校验
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return reduce success"), HCCL_SUCCESS);
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclReduceInner", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(
        sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclReduceInner", "nullptr", "sendBuf", "non-null pointer"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(
        recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclReduceInner", "nullptr", "recvBuf", "non-null pointer"}));
    CHK_PTR_NULL(recvBuf);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclReduceInner", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);

    // Group特性
    if (hcclGroupDepth > 0) {
        struct hcclOpInfo info;
        info.coll = HcclCMDType::HCCL_CMD_REDUCE;
        info.sendbuff = sendBuf;
        info.recvbuff = recvBuf;
        info.sendCount = count;
        info.sendType = dataType;
        info.recvType = dataType;
        info.comm = comm;
        info.stream = stream;
        CHK_RET(taskAppend(comm, info));
        HCCL_INFO(
            "[HcclReduce] Finish taskAppend, count [%d] dataType [%s]", count, GetDataTypeEnumStr(dataType).c_str());
        return HCCL_SUCCESS;
    }

    HcclUs startut = TIME_NOW();
    bool isCapture;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    uint64_t modelId = 0xFFFFFFFF;
    CHK_PRT(GetCaptureInfo(stream, captureStatus, modelId, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    uint64_t beginTime = hrtMsprofSysCycleTime();

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclReduceV2(sendBuf, recvBuf, count, dataType, op, root, commV2, stream));
        return HCCL_SUCCESS;
    }());
#endif
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    s32 threadID = SalGetTid();
    ProfilingManagerPub::SetThreadCaptureStatus(threadID, isCapture);
    // 同通信域同算子复用tag
    const string tag = "Reduce_" + hcclComm->GetIdentifier();
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), count, dataType, stream), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReductionOp("HcclReduceInner", op), tag.c_str());
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    CHK_RET_AND_PRINT_IDE(HcomCheckReduceDataType(dataType, op, devType), tag.c_str());

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, root), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        CHK_RET(HcclDeviceRefresh(deviceLogicId));

        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(localRank), tag.c_str());

        s32 streamId = 0;
        CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], count[%llu], dataType[%s], op[%s], root[%u], "
            "localRank[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),
            root, localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclReduceInner:" + std::string(stackLogBuffer) + ", capture status["
                              + to_string(captureStatus) + "], model id[" + to_string(modelId) + "].";
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    }

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetOverFlowAddr(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(
        hcclComm->ReduceOutPlace(tag, sendBuf, recvBuf, count, dataType, op, root, stream), tag.c_str());

    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_REDUCE, beginTime, count, dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    ProfilingManagerPub::DeleteThreadCaptureStatus(threadID);

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo
            = "HcclReduceInner:success,take time: " + std::to_string(DURATION_US(endut - startut).count()) + " us,"
              + std::string(stackLogBuffer);
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    }
    std::string identifier = hcclComm->GetIdentifier();
    AlgType algType;
    CHK_RET(hcclComm->GetAlgType(algType, HcclCMDType::HCCL_CMD_REDUCE));

    if (hcclNslbDp::GetInstance().GetGlobalCommTaskId() != 0) {
        AlgTypeLevel1 algValue = algType.algoLevel1;
        uint8_t nslbAlg = hcclNslbDp::GetInstance().GetNslbLevel1AlgType(algValue);
        if (devType == DevType::DEV_TYPE_910_93) {
            AlgTypeLevel2 algValue2 = algType.algoLevel2;
            nslbAlg = hcclNslbDp::GetInstance().GetNslbLevel2AlgType(algValue2);
        }
        // NSLB 填充 表6
        hcclNslbDp::GetInstance().SetNslbDpRootRank(HcclCMDType::HCCL_CMD_REDUCE, root, identifier, nslbAlg);
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceLoop(
    const std::string& tag, void* inputPtr, void* outputPtr, const u64 count, HcclDataType dataType, HcclReduceOp op,
    const u32 root, hccl::hcclComm* hcclComm, rtStream_t stream)
{
    HcclSetIfProfile();

    void* commInputPtr = nullptr;
    void* commOutputPtr = nullptr;
    u64 commInputSize, commOutputSize;

    HcclResult ret;
    CHK_RET(hcclComm->GetInCCLbuffer(commInputPtr, commInputSize));
    CHK_PTR_NULL(commInputPtr);

    CHK_RET(hcclComm->GetOutCCLbuffer(commOutputPtr, commOutputSize));
    CHK_PTR_NULL(commOutputPtr);

    u32 unitSize;
    CHK_RET(SalGetDataTypeSize(dataType, unitSize));

    char* curInputPtr = static_cast<char*>(inputPtr);
    char* curOutputPtr = static_cast<char*>(outputPtr);
    u64 inputOffset = 0;
    u64 outputOffset = 0;
    u64 countLeft = count;

    while (countLeft > 0) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        HCCL_DEBUG("-OP_BASE-ReduceLoop:inputOffset[%llu], outputOffset[%llu]", inputOffset, outputOffset);
        u64 curCount
            = ((countLeft * unitSize) > commInputSize) ? (commInputSize / unitSize) : countLeft; // 单次执行操作的数据量
        u64 curSize = curCount * unitSize;                                                       // 单位 byte

        HCCL_DEBUG(
            "-OP_BASE-ReduceLoop:curInputPtr[%p], curOutputPtr[%p], curCount[%llu], curSize[%llu]", curInputPtr,
            curOutputPtr, curCount, curSize);

        u32 commRank = INVALID_VALUE_RANKID;
        CHK_RET(hcclComm->GetUserRank(commRank));

        CHK_RET(hrtMemAsyncCopy(
            commInputPtr, curSize, curInputPtr, curSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE,
            stream));

        /* 入参的正确性由HCCL确保 */
        ret = hcclComm->Reduce(tag, commInputPtr, commOutputPtr, curCount, dataType, op, root, stream);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Loop][Reduce]errNo[0x%016llx] op_base hcclComm reduce error, tag[%s], "
                "input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s], root[%u]",
                HCCL_ERROR_CODE(ret), tag.c_str(), commInputPtr, commOutputPtr, curCount,
                GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(), root),
            ret);

        if (commRank == root) { // 只root rank需要把数据从中转内存拷贝出去
            CHK_RET(hrtMemAsyncCopy(
                curOutputPtr, curSize, commOutputPtr, curSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE,
                stream));
        }

        countLeft -= curCount;
        inputOffset = curSize;
        outputOffset = curSize;
    }

    return HCCL_SUCCESS;
}

/*
 * **********************************************************************
 * 单算子GatherAllToAllV的函数接口，目前不对外开放，仅图模式动态shape使用
 * **********************************************************************
 */
HcclResult HcclGatherAlltoAllV(HcomGatherAllToAllVParams params, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(params.addrInfoCountPerRank);
    CHK_PTR_NULL(params.recvcounts);
    CHK_PTR_NULL(params.gatheredbuf);
    CHK_PTR_NULL(params.rdispls);

    const u32 NUM_TWO = 2;
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    u32 rankSize = 0;
    CHK_RET(hcclComm->GetRankSize(rankSize));

    // 同通信域同算子复用tag
    const string tag = "Reduce_" + hcclComm->GetIdentifier();

    std::vector<u64> addrInfoCountPerRank(rankSize, 0);
    CHK_RET_AND_PRINT_IDE(
        hrtMemSyncCopy(
            addrInfoCountPerRank.data(), rankSize * sizeof(u64), params.addrInfoCountPerRank, rankSize * sizeof(u64),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST),
        tag.c_str());
    u64 blockNum = 0;
    for (u32 index = 0; index < rankSize; index++) {
        blockNum += addrInfoCountPerRank[index];
    }
    if (blockNum != 0) {
        CHK_PTR_NULL(params.addrInfo);
    }
    std::vector<u64> addrInfo(blockNum * NUM_TWO, 0);
    CHK_RET_AND_PRINT_IDE(
        hrtMemSyncCopy(
            addrInfo.data(), addrInfo.size() * sizeof(u64), params.addrInfo, addrInfo.size() * sizeof(u64),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST),
        tag.c_str());

    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U, "tag[%s]", tag.c_str());

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclGatherAlltoAllV:" + std::string(stackLogBuffer);
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    }

    // 执行gather
    u64 sendCounts = static_cast<u64>(rankSize);
    u64 sdispls = static_cast<u64>(rankSize);

    // step1 gather
    GatherPara gatherPara;
    gatherPara.addrInfo = addrInfo;
    gatherPara.rankSize = rankSize;
    gatherPara.addrInfoCountPerRank = addrInfoCountPerRank;
    gatherPara.addrLength = params.addrLength;
    CHK_RET_AND_PRINT_IDE(RunGather(&sendCounts, &sdispls, params.gatheredbuf, gatherPara), tag.c_str());

    // step2 alltoallv
    CHK_RET_AND_PRINT_IDE(
        HcclAlltoAllVInner(
            params.gatheredbuf, &sendCounts, &sdispls, params.recvtype, params.recvbuf, params.recvcounts,
            params.rdispls, params.recvtype, comm, stream),
        tag.c_str());

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        std::string endInfo
            = "HcclGatherAlltoAllV:success,take time: " + std::to_string(DURATION_US(endut - startut).count()) + " us,"
              + std::string(stackLogBuffer);
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    }

    return HCCL_SUCCESS;
}

/*
 * **********************************************************************
 * 单算子GatherAllToAllV step1 执行gather，出参作为step2的入参
 * **********************************************************************
 */
HcclResult RunGather(u64* sendCounts, u64* sdispls, void* sendDevBuf, GatherPara& gatherPara)
{
    // 所有 rank 计数为 0 的退化输入下 addrInfo 为空数组，无数据可 gather，直接返回成功避免越界访问
    if (gatherPara.addrInfo.empty()) {
        CHK_SAFETY_FUNC_RET(
            memset_s(sendCounts, gatherPara.rankSize * sizeof(u64), 0, gatherPara.rankSize * sizeof(u64)));
        CHK_SAFETY_FUNC_RET(memset_s(sdispls, gatherPara.rankSize * sizeof(u64), 0, gatherPara.rankSize * sizeof(u64)));
        return HCCL_SUCCESS;
    }
    u64 memSize = 0;
    const u32 GATHER_THREAD_NUM = 16;
    const u32 NUM_TWO = 2;
    u64 perThreadCount = gatherPara.addrInfo.size() / NUM_TWO / GATHER_THREAD_NUM;
    std::vector<u64> perThreadCounts(GATHER_THREAD_NUM, perThreadCount);
    perThreadCounts[GATHER_THREAD_NUM - 1]
        = gatherPara.addrInfo.size() / NUM_TWO - perThreadCount * (GATHER_THREAD_NUM - 1);
    std::vector<u64> offset(GATHER_THREAD_NUM, 0);
    if (gatherPara.addrLength == -1) { // 数据包长度不一样的情况
        u32 offsetIndex = 0;
        for (u32 index = 1; index < gatherPara.addrInfo.size();
             index += NUM_TWO) { // 由于是二元组，单数为数据包的长度，每个循环+2
            /* 如果数据包数量小于线程数量则offset全置为0 */
            if (perThreadCount != 0 && index / NUM_TWO % perThreadCount == 0 && offsetIndex < GATHER_THREAD_NUM) {
                /* 条件1：当累加的数量达到perThreadCount时往offset中填入累加值，即可计算出前面thread产生的offset值 */
                /* 条件2：由于第0个thread的offset为0，后面的线程的offset为前面线程处理数据量的累加，因此对最后一个值弃之不用
                 */
                offset[offsetIndex] = memSize;
                offsetIndex++;
            }
            memSize += gatherPara.addrInfo[index];
        }
    } else {
        memSize = gatherPara.addrInfo.size() / NUM_TWO * gatherPara.addrInfo[1];
        for (u32 index = 0; index < GATHER_THREAD_NUM; index++) {
            offset[index] = index * perThreadCount * gatherPara.addrInfo[1];
        }
    }

    // 多线程拷贝
    HostMem tmpHostMem = HostMem::alloc(memSize);
    CHK_PTR_NULL(tmpHostMem.ptr());
    std::vector<std::unique_ptr<std::thread>> threads(GATHER_THREAD_NUM);
    for (u32 num = 0; num < GATHER_THREAD_NUM; num++) {
        OpBaseMemPara memPara;
        memPara.beginIndex = num * perThreadCount * NUM_TWO;
        memPara.count = perThreadCounts[num];
        memPara.tmpMemSize = memSize;
        threads[num].reset(new (std::nothrow) std::thread(
            &GatherMemCopyThread, tmpHostMem.ptr(), offset[num], std::ref(gatherPara.addrInfo), memPara));
        CHK_PRT_RET(
            !threads[num],
            HCCL_ERROR(
                "[Exec][EnqueueGatherAlltoAllV]threads[%u] reset "
                "failed ",
                num),
            HCCL_E_INTERNAL);
    }

    // 构造入参
    auto ret = memset_s(sendCounts, gatherPara.rankSize * sizeof(u64), 0, gatherPara.rankSize * sizeof(u64));
    CHK_PRT_RET(
        ret != EOK,
        HCCL_ERROR("[Exec][EnqueueGatherAlltoAllV] mem set failed, count[%lld]", gatherPara.rankSize * sizeof(u64)),
        HCCL_E_SYSCALL);
    u64 prevNum = 0;
    u64 nextNum = 0;
    for (u32 index = 0; index < gatherPara.addrInfoCountPerRank.size(); index++) {
        nextNum += gatherPara.addrInfoCountPerRank[index];
        for (u64 i = NUM_TWO * prevNum; i < NUM_TWO * nextNum; i += NUM_TWO) {
            *(sendCounts + index) += gatherPara.addrInfo[i + 1];
        }
        prevNum = nextNum;
    }

    ret = memset_s(sdispls, gatherPara.rankSize * sizeof(u64), 0, gatherPara.rankSize * sizeof(u64));
    CHK_PRT_RET(
        ret != EOK,
        HCCL_ERROR("[Exec][EnqueueGatherAlltoAllV] mem set failed, count[%lld]", gatherPara.rankSize * sizeof(u64)),
        HCCL_E_SYSCALL);
    u64 displ = 0;
    for (u32 i = 0; i < gatherPara.rankSize; i++) {
        *(sdispls + i) = displ;
        displ += *(sendCounts + i);
    }

    // 等待线程执行完毕
    for (u32 num = 0; num < GATHER_THREAD_NUM; num++) {
        threads[num]->join();
    }

    CHK_RET(hrtMemSyncCopy(
        sendDevBuf, memSize, tmpHostMem.ptr(), memSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

/*
 * **********************************************************************
 * 单算子GatherAllToAllV gather多线程拷贝
 * **********************************************************************
 */
void GatherMemCopyThread(void* baseAddr, u64 offset, std::vector<u64>& addrInfo, OpBaseMemPara memCpyPara)
{
    // 给当前线程添加名字
    SetThreadName("Hccl_GatherCopy");

    void* addr = nullptr;
    const u32 NUM_TWO = 2;
    u64 length = 0;
    auto destMax = [&]() -> u64 {
        return memCpyPara.tmpMemSize < offset ? 0 : memCpyPara.tmpMemSize - offset;
    };

    for (u32 index = 0; index < memCpyPara.count; index++) {
        addr = reinterpret_cast<void*>(addrInfo[memCpyPara.beginIndex + NUM_TWO * index]);
        length = addrInfo[memCpyPara.beginIndex + index * NUM_TWO + 1];
        if (memcpy_s(static_cast<s8*>(baseAddr) + offset, destMax(), addr, length) != EOK) {
            HCCL_ERROR(
                "[MemCopy][GatherAlltoAllV] mem copy failed, destMax[%llu], count[%llu]",
                memCpyPara.tmpMemSize - offset, length);
            return;
        }
        offset += length;
    }
}

/*
 * **********************************************************************
 * 获取HCCL错误
 * **********************************************************************
 */
HcclResult HcclGetCommAsyncError(HcclComm comm, HcclResult* asyncError)
{
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommGetAsyncError", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(asyncError);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN(HcclGetCommAsyncErrorV2());
#endif
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_RET(hcclComm->CommCheckErrorCqe(*asyncError));
    if (*asyncError == HCCL_SUCCESS) {
        CHK_RET(hcclComm->CommCheckOpInconsistentError(*asyncError));
    }
    return HCCL_SUCCESS;
}

/*
 * **********************************************************************
 * HCCL提供错误码到字符串的转换
 * **********************************************************************
 */
const char* HcclGetErrorString(HcclResult code)
{
    if (code < HcclResult::HCCL_SUCCESS || code >= HcclResult::HCCL_E_RESERVED) {
        return "unknown error";
    }
    static const std::map<HcclResult, std::string> errorMap
        = {{HCCL_SUCCESS, "no error"},
           {HCCL_E_PARA, "parameter error"},
           {HCCL_E_PTR, "empty pointer"},
           {HCCL_E_MEMORY, "memory error"},
           {HCCL_E_INTERNAL, "internal error"},
           {HCCL_E_NOT_SUPPORT, "not support feature"},
           {HCCL_E_NOT_FOUND, "not found specific resource"},
           {HCCL_E_UNAVAIL, "resource unavailable"},
           {HCCL_E_SYSCALL, "call system interface error"},
           {HCCL_E_TIMEOUT, "timeout"},
           {HCCL_E_OPEN_FILE_FAILURE, "open file fail"},
           {HCCL_E_TCP_CONNECT, "tcp connect fail"},
           {HCCL_E_ROCE_CONNECT, "roce connect fail"},
           {HCCL_E_TCP_TRANSFER, "tcp transfer fail"},
           {HCCL_E_ROCE_TRANSFER, "roce transfer fail"},
           {HCCL_E_RUNTIME, "call runtime api fail"},
           {HCCL_E_DRV, "call driver api fail"},
           {HCCL_E_PROFILING, "call profiling api fail"},
           {HCCL_E_CCE, "call cce api fail"},
           {HCCL_E_NETWORK, "call network api fail"},
           {HCCL_E_AGAIN, "try again"},
           {HCCL_E_REMOTE, "error cqe"},
           {HCCL_E_SUSPENDING, "error communicator suspending"},
           {HCCL_E_OPRETRY_FAIL, "retry constraint"},
           {HCCL_E_OOM, "out of memory"}};

    auto it = errorMap.find(code);
    if (it != errorMap.end()) {
        return it->second.c_str();
    } else {
        return "unknown err";
    }
}

/*
 * 配置溢出检测地址
 */
HcclResult SetOverFlowAddr(hccl::hcclComm* hcclComm)
{
    std::vector<void*> globalWorkSpaceAddr;
    CHK_RET(hcclComm->SetGlobalWorkSpace(globalWorkSpaceAddr));
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
HcclResult HcclCreateComResource(const char* commName, u32 streamMode, void** commContext)
{
    RPT_INPUT_ERR(
        commName == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCreateComResource", "nullptr", "commName", "non-null pointer"}));

    RPT_INPUT_ERR(
        commContext == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCreateComResource", "nullptr", "commContext", "non-null pointer"}));
    // 切换线程后获取不到hcom上下文，需重新刷新一次线程操作的deviceid
    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(commName, hcclComm));
    HcclComm comm = hcclComm.get();
    CHK_RET(HcclCreateComResourceByComm(comm, streamMode, true, commContext));
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus

HcclResult HcclCreateComResourceByComm(
    HcclComm comm, u32 streamMode, bool isOpbaseMode, void** commContext, bool isMC2, void* mc2Tiling)
{
    HcclUs startut = TIME_NOW();
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCreateComResource", "nullptr", "comm", "non-null pointer"}));

    RPT_INPUT_ERR(
        commContext == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCreateComResource", "nullptr", "commContext", "non-null pointer"}));

    // 同通信域同算子复用tag
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    u32 moduleNum = hcclComm->GetModuleNum();
    // mc2算子更改tag
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    string commIdentifier = hcclComm->GetIdentifier();
    string tag = "CreatecomResource_" + commIdentifier;
    std::string algConfigMc2 = "";
    if (isMC2 && devType == DevType::DEV_TYPE_910B && (moduleNum > HCCL_DEVICE_NUM_ONE)) {
        tag += HCCL_MC2_MULTISERVER_SUFFIX;
        const void* tilingList[MAX_HCOM_NUM];
        uint32_t tilingNum;
        CHK_RET(HcclGetInitTilingList(mc2Tiling, tilingList, tilingNum));
        const Mc2HcommCfg* tiling = static_cast<const Mc2HcommCfg*>(tilingList[0]);
        if (tiling != nullptr && string(tiling->groupName) == commIdentifier) {
            algConfigMc2 = string(tiling->algConfig);
        }
    }

    // A2 MC2引擎默认为AICPU
    if (isMC2) {
        CHK_RET(hcclComm->SetAicpuCommEngine(true));
    }

    if (LIKELY(hcclComm->GetCommResource(tag, commContext))) {
        return HCCL_SUCCESS;
    }

    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U, "tag[%s], commContext[%p]", tag.c_str(),
            commContext);
        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    }
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);

    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetUserRank(localRank));

    if (GetExternalInputHcclEnableEntryLog()) {
        /* 接口交互信息日志 */
        std::string logInfo
            = "Entry-HcclCreateComResource:localRank[" + std::to_string(localRank) + "]" + std::string(stackLogBuffer);
        CHK_RET(hcclComm->SaveTraceInfo(logInfo));
    }

    // SetWorkflowMode性能开销hrtGetDevice，0.11us
    HcclUs middleut0 = TIME_NOW();
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    HcclUs middleut1 = TIME_NOW();
    rtStream_t stream;
    CHK_RET(hcclComm->Mc2AiCpuStreamAllocAndGet(streamMode, stream));
    CHK_RET(hcclComm->CreateCommResource(tag, stream, isOpbaseMode, commContext, algConfigMc2));

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo
            = "HcclCreateComResource success, HcclCreateComResource take time ["
              + std::to_string(DURATION_US(endut - startut).count()) + "]us, CreateComResource take time ["
              + std::to_string(DURATION_US(endut - middleut1).count()) + "]us, SetWorkflowMode take time ["
              + std::to_string(DURATION_US(middleut1 - middleut0).count()) + "]us, localRank["
              + std::to_string(localRank) + "] " + std::string(stackLogBuffer);
        CHK_RET(hcclComm->SaveTraceInfo(endInfo));
    }

    return HCCL_SUCCESS;
}

void PrintCountsAndDispls(const u32 length, const void* counts, const void* displs, const std::string& tag)
{
    // 打印counts和displs
    const u64* countsPtr = static_cast<const u64*>(counts);
    const u64* displsPtr = static_cast<const u64*>(displs);
    if (HcclCheckLogLevel(DLOG_DEBUG)) {
        std::ostringstream countsStream;
        std::ostringstream displsStream;
        countsStream << "[ ";
        displsStream << "[ ";
        for (u32 i = 0; i < length; ++i) {
            countsStream << countsPtr[i] << " ";
            displsStream << displsPtr[i] << " ";
        }
        countsStream << "]";
        displsStream << "]";
        HCCL_DEBUG("[PrintCountsAndDispls]tag[%s], counts%s", tag.c_str(), countsStream.str().c_str());
        HCCL_DEBUG("[PrintCountsAndDispls]tag[%s], displs%s", tag.c_str(), displsStream.str().c_str());
    }
}

void CheckCountsAndDispls(const u32 length, const void* counts, const void* displs, const std::string& tag)
{
    // 校验counts和displs是否匹配
    const u64* countsPtr = static_cast<const u64*>(counts);
    const u64* displsPtr = static_cast<const u64*>(displs);
    u64 displsCal = 0;

    for (u32 i = 0; i < length; i++) {
        if (displsCal != displsPtr[i]) {
            HCCL_WARNING(
                "[CheckCountsAndDispls]tag[%s], displs[%u]: [%llu] memory is discontinuous.", tag.c_str(), i,
                displsPtr[i]);
        }

        displsCal = displsCal + countsPtr[i];
    }
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
HcclResult HcclGetAicpuOpStreamNotify(const char* commName, rtStream_t* opstream, void** aicpuNotify)
{
    RPT_INPUT_ERR(
        commName == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclGetAicpuOpStream", "nullptr", "commName", "non-null pointer"}));

    RPT_INPUT_ERR(
        opstream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclGetAicpuOpStream", "nullptr", "opstream", "non-null pointer"}));

    RPT_INPUT_ERR(
        aicpuNotify == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclGetAicpuOpStream", "nullptr", "aicpuNotify", "non-null pointer"}));
    // 切换线程后获取不到hcom上下文，需重新刷新一次线程操作的deviceid
    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(commName, hcclComm));

    CHK_RET(hcclComm->GetAicpuOpStreamNotify(opstream, 1, aicpuNotify));
    return HCCL_SUCCESS;
}

HcclResult HcclGetAicpuOpStreamAndNotify(HcclComm comm, rtStream_t* opstream, u8 aicpuNotifyNum, void** aicpuNotify)
{
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclGetAicpuOpStream", "nullptr", "comm", "non-null pointer"}));

    RPT_INPUT_ERR(
        opstream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclGetAicpuOpStream", "nullptr", "opstream", "non-null pointer"}));

    RPT_INPUT_ERR(
        aicpuNotify == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclGetAicpuOpStream", "nullptr", "aicpuNotify", "non-null pointer"}));
    // 切换线程后获取不到hcom上下文，需重新刷新一次线程操作的deviceid
    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclGetAicpuOpStreamAndNotifyV2(commV2, opstream, aicpuNotifyNum, aicpuNotify));
        return HCCL_SUCCESS;
    }());
#endif
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);

    CHK_RET(hcclComm->GetAicpuOpStreamNotify(opstream, aicpuNotifyNum, aicpuNotify));
    return HCCL_SUCCESS;
}
#ifdef __cplusplus
}
#endif // __cplusplus

HcclResult HcclBatchSendRecvGroup(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    bool isCapture;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    uint64_t modelId = 0xFFFFFFFF;
    CHK_PRT(GetCaptureInfo(stream, captureStatus, modelId, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    s32 threadID = SalGetTid();
    ProfilingManagerPub::SetThreadCaptureStatus(threadID, isCapture);
    uint64_t beginTime = hrtMsprofSysCycleTime();

    // 入参校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    CHK_PTR_NULL(sendRecvInfo);
    CHK_PRT_RET((itemNum == 0), HCCL_WARNING("[BatchSendRecvGroup] taskList itemNum is zero."), HCCL_SUCCESS);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    // 若任务不同，也复用tag
    const string tag = "worldBatchSendRecvGroup_" + hcclComm->GetIdentifier();
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());
    u32 rankId = INVALID_VALUE_RANKID;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetGroupRank(rankId), tag.c_str());

    /* 记录接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        CHK_RET(HcclDeviceRefresh(deviceLogicId));

        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(localRank), tag.c_str());

        s32 streamId = 0;
        CHK_RET(hrtGetStreamId(stream, streamId));

        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], itemNum[%u], localRank[%u], streamId[%d], deviceLogicId[%d]", tag.c_str(), itemNum, localRank,
            streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclBatchSendRecvGroup:" + std::string(stackLogBuffer) + ", capture status["
                              + to_string(captureStatus) + "], model id[" + to_string(modelId) + "].";
        CHK_RET(hcclComm->SaveTraceInfo(logInfo));
    }

    for (u32 i = 0; i < itemNum; i++) {
        CHK_PTR_NULL((sendRecvInfo + i)->buf);
        CHK_RET(HcomCheckDataType((sendRecvInfo + i)->dataType));
        CHK_RET(HcomCheckCount((sendRecvInfo + i)->count));
        CHK_RET(HcomCheckUserRank(rankSize, (sendRecvInfo + i)->remoteRank));
        if (GetExternalInputHcclEnableEntryLog()) {
            char stackLogBuffer[LOG_TMPBUF_SIZE];
            s32 ret = snprintf_s(
                stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
                "SendRecvItem : SendRecvType[%d], remoteRank[%u], count[%llu], dataType[%d], buf[%p].",
                (sendRecvInfo + i)->sendRecvType, (sendRecvInfo + i)->remoteRank, (sendRecvInfo + i)->count,
                (sendRecvInfo + i)->dataType, (sendRecvInfo + i)->buf);
            CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
            std::string logInfo = "[HcclBatchSendRecvGroup]" + std::string(stackLogBuffer);
            CHK_RET(hcclComm->SaveTraceInfo(logInfo));
        }
    }

    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    HCCL_INFO("About to enter BatchSendRecv, itemNum[%u]", itemNum);
    CHK_RET_AND_PRINT_IDE(hcclComm->BatchSendRecv(tag, sendRecvInfo, itemNum, stream), tag.c_str());
    CHK_RET(CallMsprofReportHostApi(
        hcclComm, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, beginTime, sendRecvInfo->count, sendRecvInfo->dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    ProfilingManagerPub::DeleteThreadCaptureStatus(threadID);

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo = "HcclBatchSendRecvGroup:success,take time: "
                              + std::to_string(DURATION_US(endut - startut).count()) + " us, tag: " + tag;
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult HcclBatchSendRecvInner(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        CHK_PTR_NULL(comm);
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclBatchSendRecvV2(sendRecvInfo, itemNum, commV2, stream));
        return HCCL_SUCCESS;
    }());
#endif
    HcclUs startut = TIME_NOW();
    bool isCapture;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    uint64_t modelId = 0xFFFFFFFF;
    CHK_PRT(GetCaptureInfo(stream, captureStatus, modelId, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    s32 threadID = SalGetTid();
    ProfilingManagerPub::SetThreadCaptureStatus(threadID, isCapture);
    uint64_t beginTime = hrtMsprofSysCycleTime();

    // 入参校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    CHK_PTR_NULL(sendRecvInfo);
    CHK_PRT_RET((itemNum == 0), HCCL_WARNING("[BatchSendRecv] taskList itemNum is zero."), HCCL_SUCCESS);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    // 若任务不同，也复用tag
    const string tag = "worldBatchSendRecv_" + hcclComm->GetIdentifier();
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());
    u32 rankId = INVALID_VALUE_RANKID;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetGroupRank(rankId), tag.c_str());

    /* 记录接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        CHK_RET(HcclDeviceRefresh(deviceLogicId));

        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(localRank), tag.c_str());

        s32 streamId = 0;
        CHK_RET(hrtGetStreamId(stream, streamId));

        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], itemNum[%u], localRank[%u], streamId[%d], deviceLogicId[%d]", tag.c_str(), itemNum, localRank,
            streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclBatchSendRecvInner:" + std::string(stackLogBuffer) + ", capture status["
                              + to_string(captureStatus) + "], model id[" + to_string(modelId) + "].";
        CHK_RET(hcclComm->SaveTraceInfo(logInfo));
    }

    for (u32 i = 0; i < itemNum; i++) {
        // 支持数据量为0的场景，buf为空的item跳过
        if ((sendRecvInfo + i)->buf == nullptr) {
            continue;
        }
        CHK_RET(HcomCheckDataType((sendRecvInfo + i)->dataType));
        CHK_RET(HcomCheckCount((sendRecvInfo + i)->count));
        CHK_RET(HcomCheckUserRank(rankSize, (sendRecvInfo + i)->remoteRank));
        if (GetExternalInputHcclEnableEntryLog()) {
            char stackLogBuffer[LOG_TMPBUF_SIZE];
            s32 ret = snprintf_s(
                stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
                "SendRecvItem : SendRecvType[%d], remoteRank[%u], count[%llu], dataType[%d], buf[%p].",
                (sendRecvInfo + i)->sendRecvType, (sendRecvInfo + i)->remoteRank, (sendRecvInfo + i)->count,
                (sendRecvInfo + i)->dataType, (sendRecvInfo + i)->buf);
            CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
            std::string logInfo = "[HcclBatchSendRecvInner]" + std::string(stackLogBuffer);
            CHK_RET(hcclComm->SaveTraceInfo(logInfo));
        }
    }

    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    CHK_RET_AND_PRINT_IDE(hcclComm->BatchSendRecv(tag, sendRecvInfo, itemNum, stream), tag.c_str());
    CHK_RET(CallMsprofReportHostApi(
        hcclComm, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, beginTime, sendRecvInfo->count, sendRecvInfo->dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    ProfilingManagerPub::DeleteThreadCaptureStatus(threadID);

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo = "HcclBatchSendRecvInner:success,take time: "
                              + std::to_string(DURATION_US(endut - startut).count()) + " us, tag: " + tag;
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult HcclDeviceRefresh(s32& deviceLogicId)
{
    HcclResult ret = hrtGetDeviceRefresh(&g_hcclDeviceId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Get][DeviceRefresh]errNo[0x%016llx] g_hcclDeviceId[%d]"
            "get device refresh error.",
            ret, g_hcclDeviceId),
        ret);
    deviceLogicId = g_hcclDeviceId;
    return HCCL_SUCCESS;
}

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
static HcclResult RegisterTaskReportCallback(hccl::CollComm* collComm, HcclComm commV2)
{
    auto hcclCommDfx = collComm->GetHcclCommDfx();
    CHK_PTR_NULL(hcclCommDfx);
    auto reportCallback = [hcclCommDfx]() -> HcclResult {
        bool isOpBase = false;
        bool isCached = false;
        CHK_RET(hcclCommDfx->GetOpModeFlags(isOpBase, isCached));
        return hcclCommDfx->ReportAllTasks(isCached);
    };
    HcclResult ret = HcclTaskReportRegisterV2(commV2, reportCallback);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclTaskRegister] TaskReportRegister failed, ret[0x%016llx]", HCCL_ERROR_CODE(ret)), ret);
    return ret;
}
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
int32_t
HcclTaskRegister([[maybe_unused]] HcclComm comm, [[maybe_unused]] const char* msgTag, [[maybe_unused]] Callback cb)
{
    HCCL_INFO("[HcclTaskRegister] start to register task");
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    CHK_PTR_NULL(comm);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclComm commV2 = hcclComm->GetCommunicatorV2();
    HcclResult ret;
    if (commV2 != nullptr) {
        ret = HcclTaskRegisterV2(commV2, msgTag, cb);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[HcclTaskRegister] TaskRegisterV2 failed, ret[0x%016llx]", HCCL_ERROR_CODE(ret)), ret);
    } else {
        std::string commId = hcclComm->GetIdentifier();
        std::lock_guard<std::mutex> lock(g_serMapMutex);
        CHK_RET(HcclCheckTaskServiceExist(commId, g_hcclDeviceId));
        ret = g_taskServiceMap[commId][g_hcclDeviceId]->TaskRegister(msgTag, cb);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[HcclTaskRegister] TaskRegister failed, ret[0x%016llx]", HCCL_ERROR_CODE(ret)), ret);
        return HCCL_SUCCESS;
    }
    uint32_t dpuStreamId;
    CHK_PTR_NULL(commV2);
    ret = HcclGetDpuSteamIdV2(commV2, dpuStreamId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclTaskRegister] GetDpuSteamIdV2 failed, ret[0x%016llx]", HCCL_ERROR_CODE(ret)), ret);
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    collComm->GetHcclCommDfx()->SetDpuStreamId(dpuStreamId);
    auto profCallback = collComm->GetHcclCommDfx()->GetDpuCallback();
    CHK_PTR_NULL(profCallback);
    ret = HcclTaskRegisterProfV2(commV2, profCallback);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclTaskRegister] TaskProfRegister failed, ret[0x%016llx]", HCCL_ERROR_CODE(ret)), ret);

    return RegisterTaskReportCallback(collComm, commV2);

#endif
    return HCCL_E_NOT_SUPPORT;
}

int32_t HcclTaskUnRegister([[maybe_unused]] HcclComm comm, [[maybe_unused]] const char* msgTag)
{
    HCCL_INFO("[HcclTaskUnRegister] start to register task");
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    CHK_PTR_NULL(comm);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclComm commV2 = hcclComm->GetCommunicatorV2();
    if (commV2 != nullptr) {
        return HcclTaskUnRegisterV2(commV2, msgTag);
    } else {
        std::string commId = hcclComm->GetIdentifier();
        std::lock_guard<std::mutex> lock(g_serMapMutex);
        CHK_RET(HcclCheckTaskServiceExist(commId, g_hcclDeviceId));
        return g_taskServiceMap[commId][g_hcclDeviceId]->TaskUnRegister(msgTag);
    }
#endif
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclGetTopoDesc(HcclComm comm, HcclTopoDescs* topoDescs, uint32_t topoSize)
{
    // 入参合法性校验
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclGetTopoDesc", "nullptr", "comm", "non-null pointer"}));

    RPT_INPUT_ERR(
        topoDescs == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclGetTopoDesc", "nullptr", "topoDescs", "non-null pointer"}));

    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN(HcclGetTopoDescV2());
#endif
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_RET(hcclComm->GetTopoDesc(topoDescs, topoSize));

    return HCCL_SUCCESS;
}

HcclResult HcclCommSuspend(HcclComm comm)
{
    // 入参校验
    CHK_PTR_NULL(comm);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclCommSuspendV2(commV2));
        return HCCL_SUCCESS;
    }());
#endif
    HcclUs startut = TIME_NOW();
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_RET(hcclComm->Suspend());
    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO(
        "HcclCommSuspend:success, take time:[%lld]us, comm[%s]", DURATION_US(endut - startut).count(),
        hcclComm->GetIdentifier().c_str());
    return HCCL_SUCCESS;
}

static std::unordered_map<std::string, pair<HcclCommStateCallback, void*>> g_commStateCallback;
static std::shared_mutex g_callBackMtx; // 保护 g_commStateCallback

// args参数为调用者注册的函数使用，hcomm只负责透传，由注册的回调函数自行校验，
// 回调函数不需要额外参数时, args可传nullptr
// 复写同名回调前，注册方需保证旧回调不在执行且旧 args 不会被释放。
HcclResult HcclCommRegCommStateCallback(const char* regName, HcclCommStateCallback cb, void* args)
{
    CHK_PTR_NULL(regName);
    CHK_PTR_NULL(cb);

    constexpr uint32_t MAX_REG_NAME_LEN = 160;
    const uint32_t nameLen = strlen(regName);
    CHK_PRT_RET(
        (nameLen == 0 || nameLen >= MAX_REG_NAME_LEN),
        HCCL_ERROR("[%s]Invalid regName, valid length is (0, %u)", __func__, MAX_REG_NAME_LEN), HCCL_E_PARA);
    {
        std::unique_lock<std::shared_mutex> lock(g_callBackMtx); // 写锁
        g_commStateCallback[regName] = std::make_pair(cb, args);
    }
    HCCL_RUN_INFO("[%s]Register commStateCallBack success, regName[%s], args[%p].", __func__, regName, args);
    return HCCL_SUCCESS;
}

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
HcclResult HcclCommStateNotify(HcclComm comm, HcclCommStatePhase state)
{
    // 锁内拷贝回调列表快照，释放锁后再遍历调用回调
    std::vector<std::tuple<std::string, HcclCommStateCallback, void*>> snapshot;
    {
        std::shared_lock<std::shared_mutex> lock(g_callBackMtx); // 读锁
        snapshot.reserve(g_commStateCallback.size());
        for (const auto& [regName, cbPair] : g_commStateCallback) {
            snapshot.emplace_back(regName, cbPair.first, cbPair.second);
        }
    }

    for (const auto& [regName, cb, args] : snapshot) {
        HCCL_RUN_INFO(
            "[%s]comm[%p], state[%d], regName[%s], args[%p] callback begin.", __func__, comm, state, regName.c_str(),
            args);
        CHK_RET(cb(comm, state, args));
        HCCL_RUN_INFO(
            "[%s]comm[%p], state[%d], regName[%s], args[%p] callback success.", __func__, comm, state, regName.c_str(),
            args);
    }
    return HCCL_SUCCESS;
}
#endif

HcclResult HcclCommResume(HcclComm comm)
{
    // 入参校验
    CHK_PTR_NULL(comm);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclCommResumeV2(commV2));

        CHK_RET(hcclComm->Resume());
        CHK_RET(HcclCommResumePostCallback(comm));

        return HCCL_SUCCESS;
    }());
#endif
    HcclUs startut = TIME_NOW();
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_RET(hcclComm->Resume());
    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO(
        "HcclCommResume:success, take time:[%lld]us, comm[%s]", DURATION_US(endut - startut).count(),
        hcclComm->GetIdentifier().c_str());
    return HCCL_SUCCESS;
}

uint32_t HcclGetCommConfigCapability()
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN(HcclGetCommConfigCapabilityV2());
#endif
    // RESERVED在枚举中是最后一个，返回RESERVED说明它前面所有的配置项都支持
    return static_cast<uint32_t>(HCCL_COMM_CONFIG_RESERVED);
}

HcclResult HcclCommSetMemoryRange(HcclComm comm, void* baseVirPtr, size_t size, size_t alignment, uint64_t flags)
{
    // 入参校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(baseVirPtr);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN(HcclCommSetMemoryRangeV2(comm, baseVirPtr, size, alignment, flags));

#endif

    HcclUs startut = TIME_NOW();
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_RET(hcclComm->SetMemoryRange(baseVirPtr, size, alignment, flags));
    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO(
        "HcclCommSetMemoryRange:success, take time:[%lld]us, comm[%s] basePtr[%p] size[%lu] alignment[%lu] flags[%lu]",
        DURATION_US(endut - startut).count(), hcclComm->GetIdentifier().c_str(), baseVirPtr, size, alignment, flags);
    return HCCL_SUCCESS;
}

HcclResult HcclCommUnsetMemoryRange(HcclComm comm, void* baseVirPtr)
{
    // 入参校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(baseVirPtr);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclCommUnsetMemoryRangeV2(commV2, baseVirPtr));
        return HCCL_SUCCESS;
    }());
#endif

    HcclUs startut = TIME_NOW();
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_RET(hcclComm->UnsetMemoryRange(baseVirPtr));
    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO(
        "HcclCommUnsetMemoryRange:success, take time:[%lld]us, comm[%s] basePtr[%p]",
        DURATION_US(endut - startut).count(), hcclComm->GetIdentifier().c_str(), baseVirPtr);
    return HCCL_SUCCESS;
}

HcclResult
HcclCommActivateCommMemory(HcclComm comm, void* virPtr, size_t size, size_t offset, void* handle, uint64_t flags)
{
    // 入参校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(virPtr);
    CHK_PTR_NULL(handle);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclCommActivateCommMemoryV2(commV2, virPtr, size, offset, handle, flags));
        return HCCL_SUCCESS;
    }());
#endif

    HcclUs startut = TIME_NOW();
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_RET(hcclComm->ActivateCommMemory(virPtr, size, offset, handle, flags));
    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO(
        "HcclCommActivateCommMemory:success, take time:[%lld]us, comm[%s] virPtr[%p] size[%lu] offset[%lu] "
        "handle[%p] flags[%lu]",
        DURATION_US(endut - startut).count(), hcclComm->GetIdentifier().c_str(), virPtr, size, offset, handle, flags);
    return HCCL_SUCCESS;
}

HcclResult HcclCommDeactivateCommMemory(HcclComm comm, void* virPtr)
{
    // 入参校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(virPtr);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclCommDeactivateCommMemoryV2(commV2, virPtr));
        return HCCL_SUCCESS;
    }());
#endif

    HcclUs startut = TIME_NOW();
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_RET(hcclComm->DeactivateCommMemory(virPtr));
    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO(
        "HcclCommDeactivateCommMemory:success, take time:[%lld]us, comm[%s] virPtr[%p]",
        DURATION_US(endut - startut).count(), hcclComm->GetIdentifier().c_str(), virPtr);
    return HCCL_SUCCESS;
}

HcclResult HcclCommWorkingDevNicSet(HcclComm comm, uint32_t* ranks, bool* useBackup, uint32_t nRanks)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclCommWorkingDevNicSetV2(commV2, ranks, useBackup, nRanks));
        return HCCL_SUCCESS;
    }());
#endif
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommWorkingDevNicSet", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    u32 localRank = INVALID_VALUE_RANKID;
    (void)hcclComm->GetUserRank(localRank);
    HCCL_RUN_INFO(
        "Entry-HcclCommWorkingDevNicSet, comm[%s], rank[%u], nRanks[%u] need to switch nic",
        hcclComm->GetIdentifier().c_str(), localRank, nRanks);

    RPT_INPUT_ERR(
        ranks == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommWorkingDevNicSet", "nullptr", "ranks", "non-null pointer"}));
    CHK_PTR_NULL(ranks);
    RPT_INPUT_ERR(
        useBackup == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommWorkingDevNicSet", "nullptr", "useBackup", "non-null pointer"}));
    CHK_PTR_NULL(useBackup);
    HcclResult ret = hcclComm->SwitchNic(nRanks, ranks, useBackup);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "HcclCommWorkingDevNicSet fail, comm[%s], rank[%u], ret[%u]", hcclComm->GetIdentifier().c_str(), localRank,
            ret),
        ret);

    HCCL_RUN_INFO(
        "HcclCommWorkingDevNicSet success, comm[%s], rank[%u], nRanks[%u] switch nic success.",
        hcclComm->GetIdentifier().c_str(), localRank, nRanks);
    return HCCL_SUCCESS;
}

HcclResult HcclCommRegister(HcclComm comm, void* addr, uint64_t size, void** handle, [[maybe_unused]] uint32_t flag)
{
    // 入参校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(addr);
    CHK_PTR_NULL(handle);
    CHK_PRT_RET(size == 0, HCCL_ERROR("[%s] size is 0, please check size value", __func__), HCCL_E_PARA);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_RET(hcclComm->RegisterCommUserMem(addr, size, handle));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(hcclComm->GetRankSize(rankSize));
    CHK_RET(HcomSetGroupTopoInfo(hcclComm->GetIdentifier().c_str(), rankSize));
    HCCL_RUN_INFO(
        "[%s]Register mem success, group[%s], handle ptr[%p], size[%llu]", __func__, hcclComm->GetIdentifier().c_str(),
        *handle, size);
    return HCCL_SUCCESS;
}

HcclResult HcclCommDeregister(HcclComm comm, void* handle)
{
    // 入参校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(handle);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_RET(hcclComm->DeregisterCommUserMem(handle));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(hcclComm->GetRankSize(rankSize));
    CHK_RET(HcomSetGroupTopoInfo(hcclComm->GetIdentifier().c_str(), rankSize));
    HCCL_RUN_INFO(
        "[%s]Deregister mem success, group[%s], handle ptr[%p]", __func__, hcclComm->GetIdentifier().c_str(), handle);
    return HCCL_SUCCESS;
}

HcclResult HcclCommExchangeMem(HcclComm comm, void* handle, uint32_t* peerRanks, uint32_t peerRankNum)
{
    HcclUs startut = TIME_NOW();
    // 入参校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(handle);
    CHK_PTR_NULL(peerRanks);
    CHK_PRT_RET(
        (peerRankNum == 0 || peerRankNum > MAX_RANK_NUM_A3),
        HCCL_ERROR(
            "[%s]Invalid peerRankNum, valid range is (0, %u], peerRankNum[%u]", __func__, MAX_RANK_NUM_A3, peerRankNum),
        HCCL_E_PARA);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::vector<u32> peerRanksVec(peerRanks, peerRanks + peerRankNum);
    CHK_RET(hcclComm->ExchangeCommUserMem(handle, peerRanksVec));
    HCCL_RUN_INFO(
        "[%s] success, take time [%lld]us, group[%s]", __func__, DURATION_US(TIME_NOW() - startut),
        hcclComm->GetIdentifier().c_str());
    return HCCL_SUCCESS;
}

HcclResult CommGetLocalCCLBuf(HcclComm comm, void** addr, uint64_t* size)
{
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"CommGetLocalCCLBuf", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(
        addr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"CommGetLocalCCLBuf", "nullptr", "addr", "non-null pointer"}));
    CHK_PTR_NULL(addr);
    RPT_INPUT_ERR(
        size == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"CommGetLocalCCLBuf", "nullptr", "size", "non-null pointer"}));
    CHK_PTR_NULL(size);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclResult ret = hcclComm->GetLocalCCLBuf(addr, size);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("CommGetLocalCCLBuf fail, comm[%s]", hcclComm->GetIdentifier().c_str()), ret);

    HCCL_RUN_INFO("CommGetLocalCCLBuf success, comm[%s]", hcclComm->GetIdentifier().c_str());
    return HCCL_SUCCESS;
}

HcclResult CommGetRemoteCCLBuf(HcclComm comm, uint32_t remoteRank, void** addr, uint64_t* size)
{
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"CommGetRemoteCCLBuf", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(
        addr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"CommGetRemoteCCLBuf", "nullptr", "addr", "non-null pointer"}));
    CHK_PTR_NULL(addr);
    RPT_INPUT_ERR(
        size == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"CommGetRemoteCCLBuf", "nullptr", "size", "non-null pointer"}));
    CHK_PTR_NULL(size);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclResult ret = hcclComm->GetRemoteCCLBuf(remoteRank, addr, size);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("CommGetRemoteCCLBuf fail, comm[%s], remoteRank[%u]", hcclComm->GetIdentifier().c_str(), remoteRank),
        ret);

    HCCL_RUN_INFO(
        "CommGetRemoteCCLBuf success, comm[%s], remoteRank[%u]", hcclComm->GetIdentifier().c_str(), remoteRank);
    return HCCL_SUCCESS;
}
HcclResult CommGetKFCWorkSpace(HcclComm comm, void** addr, uint64_t* size)
{
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"CommGetKFCWorkSpace", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(
        addr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"CommGetKFCWorkSpace", "nullptr", "addr", "non-null pointer"}));
    CHK_PTR_NULL(addr);
    RPT_INPUT_ERR(
        size == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"CommGetKFCWorkSpace", "nullptr", "size", "non-null pointer"}));
    CHK_PTR_NULL(size);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclResult ret = hcclComm->GetKFCWorkSpace(addr, size);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("CommGetKFCWorkSpace fail, comm[%s]", hcclComm->GetIdentifier().c_str()), ret);

    HCCL_RUN_INFO("CommGetKFCWorkSpace success, comm[%s]", hcclComm->GetIdentifier().c_str());
    return HCCL_SUCCESS;
}
HcclResult CommGetCCLBufSizeCfg(HcclComm comm, uint64_t* cclBufSize)
{
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"CommGetCCLBufSizeCfg", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(
        cclBufSize == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"CommGetCCLBufSizeCfg", "nullptr", "cclBufSize", "non-null pointer"}));
    CHK_PTR_NULL(cclBufSize);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    const char* socNamePtr = aclrtGetSocName();
    CHK_PTR_NULL(socNamePtr);
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(CommGetCCLBufSizeCfgV2(commV2, cclBufSize));
        return HCCL_SUCCESS;
    }());
#endif
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    uint64_t buffSize = 0;
    if (0 == hcclComm->GetConfigInCCLbufferSize()) {
        buffSize = GetExternalInputCCLBuffSize();
    } else {
        buffSize = hcclComm->GetConfigInCCLbufferSize();
    }
    *cclBufSize = buffSize;
    HCCL_RUN_INFO("CommGetCCLBufSizeCfg success, comm[%s], size[%u]", hcclComm->GetIdentifier().c_str(), buffSize);
    return HCCL_SUCCESS;
}

enum HcclCommSymWindowInnerFlag { HCCL_COMM_SYM_WINDOW_FLAG_DEFAULT = 0, HCCL_COMM_SYM_WINDOW_FLAG_COLL_SYMMETRIC = 1 };

std::unordered_map<HcclCommSymWindow, HcclComm> winHandle2comm;
std::mutex g_winHandleMtx; // 保护所有对称内存Window的winHandle到comm映射

HcclResult HcclCommSymWinRegister(HcclComm comm, void* addr, uint64_t size, HcclCommSymWindow* winHandle, uint32_t flag)
{
    // 入参校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(addr);
    CHK_PTR_NULL(winHandle);
    CHK_PRT_RET(size == 0, HCCL_ERROR("[%s] size is 0, please check size value", __func__), HCCL_E_PARA);
    if (flag == HCCL_COMM_SYM_WINDOW_FLAG_COLL_SYMMETRIC) {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
        if (hcclComm->IsCommunicatorV2()) {
            hccl::CollComm* collComm = hcclComm->GetCollComm();
            CHK_PTR_NULL(collComm);
            CHK_RET(collComm->RegisterWindow(comm, addr, size, winHandle));
        } else {
#endif
            CHK_RET(hcclComm->RegisterWindow(addr, size, winHandle));
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
        }
#endif
        {
            std::lock_guard<std::mutex> lock(g_winHandleMtx);
            winHandle2comm[*winHandle] = comm;
        }
        HCCL_RUN_INFO(
            "[%s]WindowRegister mem success, group[%s], handle ptr[%p], size[%llu]", __func__,
            hcclComm->GetIdentifier().c_str(), *winHandle, size);
    } else if (flag == HCCL_COMM_SYM_WINDOW_FLAG_DEFAULT) {
        HCCL_ERROR("[HcclCommSymWinRegister]flag: 0 is not supported yet.");
        return HCCL_E_PARA;
    } else {
        HCCL_ERROR("[HcclCommSymWinRegister]Invalid flag[%u], must be 0 or 1", flag);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommSymWinDeregister(HcclCommSymWindow winHandle)
{
    // 入参校验
    CHK_PTR_NULL(winHandle);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HcclComm hcommWindowComm = nullptr;
    HcclResult getOwnerRet = hccl::GetHcommWindowComm(winHandle, hcommWindowComm);
    if (getOwnerRet == HCCL_SUCCESS) {
        CHK_PTR_NULL(hcommWindowComm);
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(hcommWindowComm);
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CHK_RET(collComm->DeregisterWindow(winHandle));
        hccl::EraseHcommWindowOwner(winHandle);
        {
            std::lock_guard<std::mutex> lock(g_winHandleMtx);
            winHandle2comm.erase(winHandle);
        }
        HCCL_RUN_INFO("[%s]WindowDeregister mem success, group[%s]", __func__, hcclComm->GetIdentifier().c_str());
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        getOwnerRet != HCCL_E_NOT_FOUND,
        HCCL_ERROR("[%s] query HcommWindow[%p] owner failed, ret[%d]", __func__, winHandle, getOwnerRet), getOwnerRet);
#endif

    // 所有对称内存Window均登记在winHandle2comm中；A5内部索引未命中时也可通过该表查询通信域。
    std::lock_guard<std::mutex> lock(g_winHandleMtx);
    auto iter = winHandle2comm.find(winHandle);
    CHK_PRT_RET(
        iter == winHandle2comm.end(), HCCL_ERROR("[%s] window handle[%p] is not registered", __func__, winHandle),
        HCCL_E_PARA);
    HcclComm comm = iter->second;
    CHK_PTR_NULL(comm);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CHK_RET(collComm->DeregisterWindow(winHandle));
    } else {
#endif
        CHK_RET(hcclComm->DeregisterWindow(winHandle));
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    }
#endif
    winHandle2comm.erase(iter);
    HCCL_RUN_INFO("[%s]WindowDeregister mem success, group[%s]", __func__, hcclComm->GetIdentifier().c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommSymWinGet(HcclComm comm, void* ptr, size_t size, HcclCommSymWindow* winHandle, size_t* offset)
{
    // 入参校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(ptr);
    CHK_PTR_NULL(winHandle);
    CHK_PTR_NULL(offset);
    CHK_PRT_RET(size == 0, HCCL_ERROR("[%s] size is 0, please check size value", __func__), HCCL_E_PARA);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CHK_RET(collComm->GetCommSymWin(ptr, size, winHandle, offset));
    } else {
#endif
        CHK_RET(hcclComm->GetCommSymWin(ptr, size, winHandle, offset));
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    }
#endif
    HCCL_DEBUG(
        "[%s]GetCommSymWin success, group[%s], handle ptr[%p], offset[%llu], size[%llu]", __func__,
        hcclComm->GetIdentifier().c_str(), *winHandle, *offset, size);
    return HCCL_SUCCESS;
}

HcclResult HcclGetCcuTaskInfo(HcclComm comm, void* tilingData, void* ccuTaskGroup)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(tilingData);
    CHK_PTR_NULL(ccuTaskGroup);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    comm = hcclComm->GetCommunicatorV2();
    CHK_PTR_NULL(comm);
    CHK_RET(HcclGetCcuTaskInfoLegacy(comm, tilingData, ccuTaskGroup));
#endif
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus
