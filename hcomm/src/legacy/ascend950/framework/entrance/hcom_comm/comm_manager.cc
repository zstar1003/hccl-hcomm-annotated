/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_manager.h"

#include <list>
#include <mutex>
#include <vector>
#include <string>
#include <fstream>
#include <sys/stat.h>
#include <algorithm>
#include <securec.h>
#include <linux/limits.h>
#include <sstream>

#include "log.h"
#include "hccl/base.h"
#include "hccl_common_v2.h"
#include "hccl/hccl_types.h"
#include "orion_adapter_rts.h"

#include "tp_manager.h"
#include "inner_net_dev_manager.h"
#include "hccp_hdc_manager.h"
#include "hccp_peer_manager.h"
#include "hccp_tlv_hdc_manager.h"
#include "ccu_driver_handle.h"
#include "rdma_handle_manager.h"
#include "socket_handle_manager.h"
#include "host_socket_handle_manager.h"

#include "ccu_context_mgr_imp.h"
#include "ccu_res_batch_allocator_legacy.h"
#include "ccu_component.h"
#include "communicator_callback.h"
#include "types.h"

using namespace std;
using namespace Hccl;

std::mutex g_commInfoV2CtxMutex;

u64 GetFileSize(const std::string& path)
{
    struct stat fileStat;
    if (stat(path.c_str(), &fileStat) != 0) {
        HCCL_ERROR("[GetFileSize] Get file stat failed , file path:%s", path.c_str());
        return 0;
    }
    return static_cast<u64>(fileStat.st_size);
}

HcclResult CcuResAllocAndCtxMgrInit(s32 deviceLogicId)
{
    try {
        CcuComponent::GetInstance(deviceLogicId);
        CcuResBatchAllocator::GetInstance(deviceLogicId);
        CtxMgrImp::GetInstance(deviceLogicId);
    } catch (HcclException& e) {
        HCCL_ERROR(e.what());
        return e.GetErrorCode();
    } catch (exception& e) {
        HCCL_ERROR(e.what());
        return HcclResult::HCCL_E_INTERNAL;
    } catch (...) {
        HCCL_ERROR("Unknown error occurs!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

// 规避默认析构顺序导致单例调用接口时序错误，架构优化后必须删除
// 析构含时序要求接口的单例应在此声明
// 声明顺序与期望析构顺序相反
HcclResult CallSingletons()
{
    s32 deviceLogicId = 0;
    try {
        deviceLogicId = HrtGetDevice();
        // 避免设备粒度单例访问错误设备
        if (deviceLogicId < 0 || static_cast<uint32_t>(deviceLogicId) >= ::MAX_MODULE_DEVICE_NUM) {
            HCCL_WARNING("[CallSingletons] deviceLogicId[%d] may not have device, passed.", deviceLogicId);
            return HCCL_E_RUNTIME;
        }

        // 不同通信域初始化方式时序不同，hdc manager 重复 init 内部会跳过
        HccpHdcManager::GetInstance();
        HccpPeerManager::GetInstance(); // host网卡需要拉起peer模式hccp
        HccpTlvHdcManager::GetInstance();
        RdmaHandleManager::GetInstance();
        InnerNetDevManager::GetInstance();
        SocketHandleManager::GetInstance();
        HostSocketHandleManager::GetInstance(); // host网卡需要
        TpManager::GetInstance(deviceLogicId);
    } catch (HcclException& e) {
        HCCL_ERROR(e.what());
        return e.GetErrorCode();
    } catch (exception& e) {
        HCCL_ERROR(e.what());
        return HcclResult::HCCL_E_INTERNAL;
    } catch (...) {
        HCCL_ERROR("Unknown error occurs!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (CcuResAllocAndCtxMgrInit(deviceLogicId) != HCCL_SUCCESS) {
        // 遗留问题,处理ccu资源申请失败,走aicpu流程
        HCCL_ERROR("Ccu res batch allocator or ctx mgr init failed.");
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

CommManager& CommManager::GetInstance(s32 deviceLogicId)
{
    // 预留额外一个作为兜底通信域
    static CommManager commManager[::MAX_MODULE_DEVICE_NUM + 1]; // 使用全局命名空间变量

    if (deviceLogicId < 0 || static_cast<uint32_t>(deviceLogicId) > ::MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING("[GetInstance] deviceLogicId[%d] is invalid, use backup comm instead.", deviceLogicId);
        deviceLogicId = ::MAX_MODULE_DEVICE_NUM;
    }
    commManager[deviceLogicId].deviceLogicId = deviceLogicId;
    return commManager[deviceLogicId];
};

HcclCommInfoV2& CommManager::GetCommInfoV2() { return commInfoV2; }

void CommManager::PrintChannelInfo()
{
    std::lock_guard<std::mutex> lock(commInfoV2.groupParamsLock);
    u32 channelNum = 0;
    s32 logicDevId = HrtGetDevice();
    HCCL_INFO("[CommManager][PrintChannelInfo]devId[%d].", logicDevId);
    for (u32 dieId = 0; dieId < MAX_CCU_IODIE_NUM; dieId++) {
        auto ret = CcuGetChannelSpecNum(logicDevId, dieId, channelNum);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[CommManager][PrintChannelInfo]Get channel num failed, devId[%d], dieId[%u]", logicDevId, dieId);
            return;
        }
        HCCL_RUN_INFO(
            "[CommManager][PrintChannelInfo]devId[%d], dieId[%u], Channel num[%u].", logicDevId, dieId, channelNum);
    }

    for (const auto& group : commInfoV2.hcclGroupMap) {
        for (u32 dieId = 0; dieId < MAX_CCU_IODIE_NUM; dieId++) {
            u32 channelCount = group.second.pComm->GetUsedChannelCount(dieId);
            if (channelCount != 0) {
                HCCL_RUN_INFO(
                    "[CommManager][PrintChannelInfo]group[%s], dieId[%u], used channel count[%u].", group.first.c_str(),
                    dieId, channelCount);
            }
        }
    }
}

std::function<void()> CommManager::GetPrintChannelInfoCallback()
{
    auto callBack = [this]() {
        PrintChannelInfo();
    };
    return callBack;
}

HcclCommInfoV2& GetCommInfoV2(void)
{
    std::lock_guard<std::mutex> lock(g_commInfoV2CtxMutex);
    s32 logicDevId = 0;
    aclError ret = aclrtGetDevice(&logicDevId);
    if (ret == ACL_SUCCESS && (static_cast<u32>(logicDevId) < ::MAX_MODULE_DEVICE_NUM)) {
        /* 当前线程获取到deviceId, 如果是首次使用该deviceId的HcomInfo, 先判断之前是否已经配置过 */
        HcclCommInfoV2& commInfoV2 = CommManager::GetInstance(logicDevId).GetCommInfoV2();
        if (!commInfoV2.isUsed) {
            HCCL_WARNING("[GetCommInfoV2] logicDevId[%d] is not Used.", logicDevId);

            HcclCommInfoV2& backupCommInfoV2 = CommManager::GetInstance(::MAX_MODULE_DEVICE_NUM).GetCommInfoV2();
            if (backupCommInfoV2.isUsed) {
                return backupCommInfoV2;
            }
        }
        commInfoV2.isUsed = true;
        return commInfoV2;
    }

    /* 当前线程没有获取到deviceId, 查找是否有使用过的Ctx */
    for (u32 i = 0; i <= ::MAX_MODULE_DEVICE_NUM; i++) {
        HcclCommInfoV2& commInfoV2 = CommManager::GetInstance(i).GetCommInfoV2();
        if (commInfoV2.isUsed) {
            HCCL_WARNING("[GetCommInfoV2] no set device Used logicDevId[%u].", i);
            return commInfoV2;
        }
    }

    HCCL_WARNING("[GetCommInfoV2] HrtGetDevice fail.");
    /* 当前线程没有获取到deviceId, 使用兜底Ctx */
    HcclCommInfoV2& backupCommInfoV2 = CommManager::GetInstance(::MAX_MODULE_DEVICE_NUM).GetCommInfoV2();
    backupCommInfoV2.isUsed = true;
    return backupCommInfoV2;
}

static HcclResult ParseRankIdsV2(u32 rankNum, const u32* rankIds, std::vector<u32>& groupRanks)
{
    unordered_set<uint32_t> rankIdSet;
    std::ostringstream printRankIds;
    printRankIds << "input rankIds: ";
    for (u32 i = 0; i < rankNum; i++) {
        CHK_PTR_NULL(rankIds + i);
        printRankIds << "rank[";
        printRankIds << i;
        printRankIds << "] = ";
        printRankIds << rankIds[i];
        if (i < rankNum - 1) {
            printRankIds << ", ";
        }
        CHK_PRT_RET(
            rankIdSet.find(rankIds[i]) != rankIdSet.end(),
            HCCL_ERROR(
                "[ParseRankIdsV2]errNo[0x%016llx], "
                "duplicated rankId[%u] in rankIds.",
                HCCL_ERROR_CODE(HCCL_E_PARA), rankIds[i]),
            HCCL_E_PARA);
        rankIdSet.insert(rankIds[i]);
        groupRanks.push_back(rankIds[i]);
    }
    HCCL_RUN_INFO("Entry-%s: %s", __func__, printRankIds.str().c_str());
    return HCCL_SUCCESS;
}

HcclResult GetHcomRankListV2(u32 rankNum, const u32* rankIds, HcclGroupParamsV2& params, HcclComm globalComm)
{
    HcclCommInfoV2& hcomCommInfoV2 = GetCommInfoV2();

    u32 worldRank = hcomCommInfoV2.commParams.myRank;
    u32 worldRankSize = hcomCommInfoV2.commParams.rankSize;
    if (globalComm != nullptr) {
        Hccl::HcclCommunicator* globalCommunicator = static_cast<Hccl::HcclCommunicator*>(globalComm);
        CHK_RET(globalCommunicator->GetRankId(worldRank));
        CHK_RET(globalCommunicator->GetRankSize(&worldRankSize));
    }

    params.totalRanks = rankNum;
    params.worldRank = worldRank;
    params.groupRank = INVALID_VALUE_RANKID;

    CHK_RET(ParseRankIdsV2(rankNum, rankIds, params.groupRanks));

    std::vector<u32> rankListCopy = params.groupRanks;
    auto maxIt = std::max_element(rankListCopy.begin(), rankListCopy.end());
    if (*maxIt >= worldRankSize) {
        HCCL_ERROR(
            "[get][RankList]errNo[0x%016llx] maxRank[%u] is invalid, worldRankSize[%u]", HCOM_ERROR_CODE(HCCL_E_PARA),
            *maxIt, worldRankSize);
        return HCCL_E_PARA;
    }

    for (u32 i = 0; i < rankNum; i++) {
        if (params.groupRanks[i] == params.worldRank) {
            params.groupRank = i;
            break;
        }
    }

    u32 serverNum = 1; // serverNum初始值应为1，代表groupId为0的serverId;
    params.serverNum = serverNum;

    return HCCL_SUCCESS;
}

// 图模式 创建子通信域 V2
HcclResult HcomCreateGroupImplV2(const std::string& group, u32 rankNum, const std::vector<u32>& rankIds)
{
    HcclUs startut = TIME_NOW();
    /* 接口交互信息日志 */
    rankNum = rankIds.size();
    std::string rankId = "";
    for (u32 i = 0; i < rankNum; i++) {
        rankId += std::to_string(rankIds[i]);
        if (i < rankNum - 1) {
            rankId += ',';
        }
    }
    HCCL_RUN_INFO("Entry-HcomCreateGroup:group[%s], rankNum[%u], rankIds[%s]", group.c_str(), rankNum, rankId.c_str());

    HcclCommInfoV2& hcomCommInfoV2 = GetCommInfoV2();
    CHK_PRT_RET(
        hcomCommInfoV2.pComm == nullptr,
        HCCL_ERROR("[Create][Group]hcomCommInfoV2.pComm is null, please check if the initialize process is called."),
        HCCL_E_PTR);

    /* 已经存在的group不允许再次创建 */
    std::unique_lock<std::mutex> groupParaLock(hcomCommInfoV2.groupParamsLock);
    if (hcomCommInfoV2.hcclGroupMap.find(group) != hcomCommInfoV2.hcclGroupMap.end()) {
        HCCL_ERROR(
            "[Create][Group]errNo[0x%016llx] group[%s] already exists", HCOM_ERROR_CODE(HCCL_E_PARA), group.c_str());
        return HCCL_E_PARA;
    }
    groupParaLock.unlock();

    /* 创建groupParamsV2Tem */
    HcclGroupParamsV2 groupParamsV2Tem;
    CHK_RET(GetHcomRankListV2(rankNum, rankIds.data(), groupParamsV2Tem, nullptr));

    /* 如果是groupRank = INVALID_VALUE_RANKID，即本rank不参与create group */
    if (groupParamsV2Tem.groupRank == INVALID_VALUE_RANKID) {
        HCCL_ERROR(
            "[Create][Group]errNo[0x%016llx] confirm groupRank from worldRank[%d] error",
            HCOM_ERROR_CODE(HCCL_E_NOT_FOUND), hcomCommInfoV2.commParams.myRank);
        return HCCL_E_NOT_FOUND;
    }

    /* 创建子通信域 */
    Hccl::CommParams subCommParams{
        group, static_cast<Hccl::RankId>(groupParamsV2Tem.groupRank), rankNum,
        static_cast<Hccl::RankId>(groupParamsV2Tem.worldRank), hcomCommInfoV2.commParams.devType};
    subCommParams.commDepth = 1;
    auto ret = hcomCommInfoV2.pComm->CreateSubComm(subCommParams, groupParamsV2Tem.groupRanks, groupParamsV2Tem.pComm);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[Create][Group]errNo[0x%016llx] create group failed.", HCOM_ERROR_CODE(ret)),
        ret);

    CHK_SMART_PTR_NULL(groupParamsV2Tem.pComm);
    groupParamsV2Tem.pComm->RegisterAcceStateCallBack(CommunicatorCallback());
    s32 logicDevId = HrtGetDevice();
    CHK_RET(CommManager::GetInstance(logicDevId)
                .SetCommAcceleratorV2(groupParamsV2Tem.pComm.get(), 0)); // 子通信域创建，设置默认accelerator

    groupParaLock.lock();
    hcomCommInfoV2.hcclGroupMap.insert(std::make_pair(group, groupParamsV2Tem));
    groupParaLock.unlock();

    groupParamsV2Tem.pComm->RegisterPrintChannelInfoCallback(
        CommManager::GetInstance(logicDevId).GetPrintChannelInfoCallback());
    HCCL_RUN_INFO(
        "hcom create group[%s] success, take time [%lld]us", group.c_str(), DURATION_US(TIME_NOW() - startut));

    return HCCL_SUCCESS;
}

HcclResult HcomDestroyGroupImplV2(const std::string& group)
{
    HcclCommInfoV2& hcomCommInfoV2 = GetCommInfoV2();

    /* 接口交互信息日志 */
    HCCL_RUN_INFO("Entry-HcomDestroyGroup:group[%s]", group.c_str());

    std::unique_lock<std::mutex> groupParaLock(hcomCommInfoV2.groupParamsLock);
    auto iter = hcomCommInfoV2.hcclGroupMap.find(group);
    if (iter == hcomCommInfoV2.hcclGroupMap.end()) {
        HCCL_ERROR(
            "[Destroy][Group]errNo[0x%016llx] group[%s] does not exist", HCOM_ERROR_CODE(HCCL_E_PARA), group.c_str());
        return HCCL_E_PARA;
    }
    hcomCommInfoV2.hcclGroupMap.erase(group);
    // 通信域销毁，更新ccu使用情况
    hcomCommInfoV2.ccuStatus.RemoveCommId(group);

    groupParaLock.unlock();

    HCCL_RUN_INFO("hcom destroy group[%s] success.", group.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcomGetWorldRankFromGroupRankV2(const char* group, u32 groupRank, u32* worldRank)
{
    HcclCommInfoV2& hcomCommInfoV2 = GetCommInfoV2();
    // 校验通信域非空
    CHK_PRT_RET(
        hcomCommInfoV2.pComm == nullptr,
        HCCL_ERROR("[Get][WorldRank]hcomCommInfoV2.pComm is null, "
                   "please check if the initialize process is called."),
        HCCL_E_PTR);
    // 获取group
    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    if (strGroup == HCCL_WORLD_GROUP) {
        *worldRank = hcomCommInfoV2.commParams.myRank;
        HCCL_INFO(
            "hcom get world rank success, group[%s], groupRank[%u], worldRank[%u]", strGroup.c_str(), groupRank,
            *worldRank);
        return HCCL_SUCCESS;
    }
    std::unique_lock<std::mutex> groupParaLock(hcomCommInfoV2.groupParamsLock);
    auto iter = hcomCommInfoV2.hcclGroupMap.find(strGroup);
    if (iter == hcomCommInfoV2.hcclGroupMap.end()) {
        HCCL_ERROR(
            "[Get][WorldRank]errNo[0x%016llx] group[%s] is not exist", HCOM_ERROR_CODE(HCCL_E_PARA), strGroup.c_str());
        return HCCL_E_PARA;
    }

    // groupRanks判空
    CHK_PRT_RET(
        (iter->second).groupRanks.empty(),
        HCCL_ERROR(
            "[Get][WorldRank]errNo[0x%016llx] group[%s] ranks is empty", HCOM_ERROR_CODE(HCCL_E_INTERNAL),
            strGroup.c_str()),
        HCCL_E_INTERNAL);

    // 校验groupRank合法性
    if (groupRank >= (iter->second).totalRanks) {
        HCCL_ERROR(
            "[Get][WorldRank]errNo[0x%016llx] group[%s] groupRank[%u] is invalid", HCOM_ERROR_CODE(HCCL_E_PARA),
            strGroup.c_str(), groupRank);
        return HCCL_E_PARA;
    }
    *worldRank = (iter->second).groupRanks[groupRank];

    HCCL_INFO(
        "hcom get world rank success, group[%s], groupRank[%u], worldRank[%u]", strGroup.c_str(), groupRank,
        *worldRank);
    return HCCL_SUCCESS;
}

HcclResult HcomGetGroupRankFromWorldRankV2(u32 worldRank, const char* group, u32* groupRank)
{
    HcclCommInfoV2& hcomCommInfoV2 = GetCommInfoV2();
    // 校验通信域非空
    CHK_PRT_RET(
        hcomCommInfoV2.pComm == nullptr,
        HCCL_ERROR("[Get][GroupRank]hcomCommInfoV2.pComm is null, "
                   "please check if the initialize process is called."),
        HCCL_E_PTR);
    // 校验worldRank合法性
    if (worldRank >= hcomCommInfoV2.commParams.rankSize) {
        HCCL_ERROR(
            "[Get][GroupRank]errNo[0x%016llx] world[%u] rank is invalid", HCOM_ERROR_CODE(HCCL_E_PARA), worldRank);
        return HCCL_E_PARA;
    }
    // 获取group
    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    if (strGroup == HCCL_WORLD_GROUP) {
        *groupRank = hcomCommInfoV2.commParams.myRank;
        HCCL_INFO(
            "hcom get group rank success, group[%s], worldRank[%u], groupRank[%u]", strGroup.c_str(), worldRank,
            *groupRank);
        return HCCL_SUCCESS;
    }
    std::unique_lock<std::mutex> groupParaLock(hcomCommInfoV2.groupParamsLock);
    auto iter = hcomCommInfoV2.hcclGroupMap.find(strGroup);
    if (iter == hcomCommInfoV2.hcclGroupMap.end()) {
        HCCL_ERROR(
            "[Get][GroupRank]errNo[0x%016llx] group[%s] is not exist", HCOM_ERROR_CODE(HCCL_E_PARA), strGroup.c_str());
        return HCCL_E_PARA;
    }

    // groupRanks判空
    CHK_PRT_RET(
        (iter->second).groupRanks.empty(),
        HCCL_ERROR(
            "[Get][GroupRank]errNo[0x%016llx] group[%s] ranks is empty", HCOM_ERROR_CODE(HCCL_E_INTERNAL),
            strGroup.c_str()),
        HCCL_E_INTERNAL);

    // 获取groupRank
    for (u32 rank = 0; rank < (iter->second).totalRanks; rank++) {
        if (worldRank == (iter->second).groupRanks[rank]) {
            *groupRank = rank;

            HCCL_INFO(
                "hcom get group rank success, group[%s], worldRank[%u], groupRank[%u]", strGroup.c_str(), worldRank,
                *groupRank);
            return HCCL_SUCCESS;
        }
    }
    return HCCL_E_PARA;
}

HcclResult HcomGetRankSizeV2(const char* group, u32* rankSize)
{
    HcclCommInfoV2& hcomCommInfoV2 = GetCommInfoV2();

    // 获取group
    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    if (strGroup == HCCL_WORLD_GROUP) {
        *rankSize = hcomCommInfoV2.commParams.rankSize;
        HCCL_INFO("hcom get world rank size success, rankSize[%u]", *rankSize);
        return HCCL_SUCCESS;
    }
    std::unique_lock<std::mutex> groupParaLock(hcomCommInfoV2.groupParamsLock);
    auto iter = hcomCommInfoV2.hcclGroupMap.find(strGroup);
    if (iter == hcomCommInfoV2.hcclGroupMap.end()) {
        HCCL_ERROR(
            "[Get][RankSize]errNo[0x%016llx] group[%s] is not exist", HCOM_ERROR_CODE(HCCL_E_PARA), strGroup.c_str());
        return HCCL_E_PARA;
    }
    CHK_SMART_PTR_NULL((iter->second).pComm);
    HcclResult ret = (iter->second).pComm->GetRankSize(rankSize);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Get][RankSize]GetRankSize failed."), HCCL_E_PTR);
    groupParaLock.unlock();

    HCCL_INFO("hcom get rank size success, group[%s], rankSize[%u]", strGroup.c_str(), *rankSize);
    return HCCL_SUCCESS;
}

HcclResult HcomGetCommV2(void** commV2)
{
    CHK_PTR_NULL(commV2);
    HcclCommInfoV2& hcomCommInfoV2 = GetCommInfoV2();
    CHK_PTR_NULL(hcomCommInfoV2.pComm);
    *commV2 = static_cast<void*>(hcomCommInfoV2.pComm.get());
    HCCL_INFO("[HcomGetCommV2] success.");
    return HCCL_SUCCESS;
}

HcclResult HcomGetGroupParamsV2(const char* group, void* groupParams, void** commV2)
{
    HcclCommInfoV2& hcomCommInfoV2 = GetCommInfoV2();
    auto iter = hcomCommInfoV2.hcclGroupMap.find(group);
    if (iter == hcomCommInfoV2.hcclGroupMap.end()) {
        HCCL_ERROR("[HcomGetGroupParamsV2] group[%s] not found", group);
        return HCCL_E_PARA;
    }
    HcclGroupParamsV2& groupParamsV2 = iter->second;
    HcclGroupParamsV2* groupParamsTem = static_cast<HcclGroupParamsV2*>(groupParams);
    *groupParamsTem = groupParamsV2;
    CHK_PTR_NULL(groupParamsV2.pComm);
    *commV2 = static_cast<Hccl::HcclCommunicator*>(groupParamsV2.pComm.get());
    HCCL_INFO("[HcomGetGroupParamsV2] success. group[%s]", group);
    return HCCL_SUCCESS;
}

HcclResult HcomDestroyV2(void)
{
    HcclCommInfoV2& hcomCommInfoV2 = GetCommInfoV2();
    if (hcomCommInfoV2.pComm != nullptr) {
        bool isSetDeviceByHcomm = false;
        if (hcomCommInfoV2.devId != HOST_DEVICE_ID) {
            s32 logicDevId = 0;
            aclError ret = aclrtGetDevice(&logicDevId);
            if (ret == ACL_ERROR_RT_CONTEXT_NULL) {
                // 若当前线程获取不到context，则由hccl进行setDevice，并在通信域析构完成后resetDevice
                HrtSetDevice(hcomCommInfoV2.devId);
                isSetDeviceByHcomm = true;
            } else if (ret != ACL_SUCCESS) {
                HCCL_ERROR("[HcomDestroyV2] get device failed, ret[%d]", ret);
            }
        }
        // 通信域销毁，更新ccu使用情况
        hcomCommInfoV2.ccuStatus.RemoveCommId(hcomCommInfoV2.pComm->GetId());
        hcomCommInfoV2.pComm = nullptr;
        std::unique_lock<std::mutex> groupParaLock(hcomCommInfoV2.groupParamsLock);

        // 通信域销毁，更新子通信域ccu使用情况
        for (auto iterGroup : hcomCommInfoV2.hcclGroupMap) {
            hcomCommInfoV2.ccuStatus.RemoveCommId(iterGroup.first);
        }
        hcomCommInfoV2.hcclGroupMap.clear();
        if (isSetDeviceByHcomm) {
            HrtResetDevice(hcomCommInfoV2.devId);
        }
    }
    return HCCL_SUCCESS;
}

static HcclResult GetRankTableInfo(const char* rankTablePath, std::string& ranktableInfo)
{
    // 校验文件是否存在
    char resolvedPath[PATH_MAX] = {0};
    if (realpath(rankTablePath, resolvedPath) == nullptr) {
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] RanktableRealPath: %s is not a valid real path", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CONFIG.c_str(), HCOM_ERROR_CODE(HCCL_E_INTERNAL), rankTablePath);
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("waiting for json file load complete");
    u64 ranktableFileSize = GetFileSize(resolvedPath);
    if (ranktableFileSize > RANKTABLE_FILE_MAX_SIZE || ranktableFileSize <= 0) {
        HCCL_ERROR(
            "[GetRankTableInfo] ranktablefile size: %u, ranktable must be greater than 0 and less than %u",
            ranktableFileSize, RANKTABLE_FILE_MAX_SIZE);
        return HCCL_E_OPEN_FILE_FAILURE;
    }

    std::ifstream infoFile(resolvedPath, std::ifstream::in);
    if (!infoFile) {
        HCCL_ERROR("open file %s failed", resolvedPath);
        return HCCL_E_INTERNAL;
    }

    std::stringstream rankTableStr;
    rankTableStr << infoFile.rdbuf();
    ranktableInfo = rankTableStr.str();

    return HCCL_SUCCESS;
}

// 图模式 创建全局通信域 V2
HcclResult HcomInitByFileV2(const char* rankTablePath, const char* identify)
{
    // 待解决：目前主要为了芯片验证，非最终版本
    HCCL_RUN_INFO("Entry-HcomInitByFile V950, ranktable[%s], identify[%s]", rankTablePath, identify);

    // 解析myRank
    s32 myRank;
    try {
        myRank = std::atoi(identify);
    } catch (...) {
        HCCL_ERROR("atoi(identify) failed!");
        return HCCL_E_INTERNAL;
    }

    CallSingletons(); // 临时规避，在初始化通信域前声明单例保证时序

    // 防止重复调用初始化
    string commId(HCCL_WORLD_GROUP);
    HcclCommInfoV2& hcomCommInfoV2 = GetCommInfoV2();
    CHK_PRT_RET(
        hcomCommInfoV2.hcclGroupMap.find(commId) != hcomCommInfoV2.hcclGroupMap.end(),
        HCCL_ERROR(
            "[Init][CheckOpBasedHcom]errNo[0x%016llx] The comm name[%s] already exists in Group2Comm map.",
            HCCL_ERROR_CODE(HCCL_E_PARA), commId.c_str()),
        HCCL_E_PARA);

    // 解析ranktable
    std::string ranktableInfo;
    HcclResult ret = GetRankTableInfo(rankTablePath, ranktableInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[HcomInitByFile] get ranktable info failed"), ret);

    bool devUsed = false;
    bool isWorldGroup = true;
    // 临时修改这个为4，后边要改掉这个，在初始流程中，解析完虚拟拓扑后，添加ranksize
    Hccl::CommParams commParams{commId,
                                static_cast<Hccl::RankId>(myRank),
                                0,
                                static_cast<Hccl::RankId>(myRank),
                                Hccl::HrtGetDeviceType(),
                                devUsed,
                                isWorldGroup};
    hcomCommInfoV2.pComm.reset(new (std::nothrow) Hccl::HcclCommunicator(commParams));
    CHK_PTR_NULL(hcomCommInfoV2.pComm);
    auto res = hcomCommInfoV2.pComm->Init(ranktableInfo);
    CHK_PRT_RET(
        res != HcclResult::HCCL_SUCCESS, HCCL_ERROR("[HcomInitByFile] Hccl::Communicator Init failed, res %d", res),
        HCCL_E_INTERNAL);

    hcomCommInfoV2.pComm->RegisterAcceStateCallBack(CommunicatorCallback());
    s32 logicDevId = HrtGetDevice();
    CHK_RET(CommManager::GetInstance(logicDevId)
                .SetCommAcceleratorV2(hcomCommInfoV2.pComm.get(), 0)); // 全局通信域创建，设置默认accelerator

    res = hcomCommInfoV2.pComm->GetRankSize(&commParams.rankSize);
    CHK_PRT_RET(
        res != HCCL_SUCCESS,
        HCCL_ERROR("[HcomInitByFile] Hccl::Communicator GetRankSize failed, rankSize = %u", commParams.rankSize), res);
    hcomCommInfoV2.commParams = commParams;

    HcclGroupParamsV2 params;
    params.pComm = hcomCommInfoV2.pComm;
    std::unique_lock<std::mutex> groupParaLock(hcomCommInfoV2.groupParamsLock);
    hcomCommInfoV2.hcclGroupMap[commId] = params;
    groupParaLock.unlock();

    hcomCommInfoV2.pComm->RegisterPrintChannelInfoCallback(
        CommManager::GetInstance(logicDevId).GetPrintChannelInfoCallback());

    HCCL_INFO(
        "[HcomInitByFile] HcomInitByFile success! logicDevId[%d], commId[%s]", logicDevId, commParams.commId.c_str());

    return HCCL_SUCCESS;
}

HcclResult HcomInitByStringV2(const char* rankTableM, const char* identify)
{
    // 待解决：目前主要为了芯片验证，非最终版本
    HCCL_RUN_INFO("Entry-HcomInitByString V950, rankTableM[%s], identify[%s]", rankTableM, identify);

    // 解析myRank
    s32 myRank;
    try {
        myRank = std::atoi(identify);
    } catch (...) {
        HCCL_ERROR("atoi(identify) failed!");
        return HCCL_E_INTERNAL;
    }

    CallSingletons(); // 临时规避，在初始化通信域前声明单例保证时序

    // 防止重复调用初始化
    string commId(HCCL_WORLD_GROUP);
    HcclCommInfoV2& hcomCommInfoV2 = GetCommInfoV2();
    CHK_PRT_RET(
        hcomCommInfoV2.hcclGroupMap.find(commId) != hcomCommInfoV2.hcclGroupMap.end(),
        HCCL_ERROR(
            "[Init][CheckOpBasedHcom]errNo[0x%016llx] The comm name[%s] already exists in Group2Comm map.",
            HCCL_ERROR_CODE(HCCL_E_PARA), commId.c_str()),
        HCCL_E_PARA);

    bool devUsed = false;
    bool isWorldGroup = true;
    // 临时修改这个为4，后边要改掉这个，在初始流程中，解析完虚拟拓扑后，添加ranksize
    Hccl::CommParams commParams{commId,
                                static_cast<Hccl::RankId>(myRank),
                                0,
                                static_cast<Hccl::RankId>(myRank),
                                Hccl::HrtGetDeviceType(),
                                devUsed,
                                isWorldGroup};
    hcomCommInfoV2.pComm.reset(new (std::nothrow) Hccl::HcclCommunicator(commParams));
    CHK_PTR_NULL(hcomCommInfoV2.pComm);
    auto res = hcomCommInfoV2.pComm->Init(rankTableM);
    CHK_PRT_RET(
        res != HcclResult::HCCL_SUCCESS, HCCL_ERROR("[HcomInitByString] Hccl::Communicator Init failed, res %d", res),
        HCCL_E_INTERNAL);

    hcomCommInfoV2.pComm->RegisterAcceStateCallBack(CommunicatorCallback());
    s32 logicDevId = HrtGetDevice();
    hcomCommInfoV2.devId = logicDevId;
    CHK_RET(CommManager::GetInstance(logicDevId)
                .SetCommAcceleratorV2(hcomCommInfoV2.pComm.get(), 0)); // 全局通信域创建，设置默认accelerator

    res = hcomCommInfoV2.pComm->GetRankSize(&commParams.rankSize);
    CHK_PRT_RET(
        res != HCCL_SUCCESS,
        HCCL_ERROR("[HcomInitByString] Hccl::Communicator GetRankSize failed, rankSize = %u", commParams.rankSize),
        res);
    hcomCommInfoV2.commParams = commParams;

    HcclGroupParamsV2 params;
    params.pComm = hcomCommInfoV2.pComm;
    std::unique_lock<std::mutex> groupParaLock(hcomCommInfoV2.groupParamsLock);
    hcomCommInfoV2.hcclGroupMap[commId] = params;
    groupParaLock.unlock();

    hcomCommInfoV2.pComm->RegisterPrintChannelInfoCallback(
        CommManager::GetInstance(logicDevId).GetPrintChannelInfoCallback());

    HCCL_INFO(
        "[HcomInitByString] HcomInitByString success! logicDevId[%d], commId[%s]", logicDevId,
        commParams.commId.c_str());

    return HCCL_SUCCESS;
}

void CcuStatus::RemoveCommId(const std::string& commId)
{
    auto itMs = std::find(useMsCommIds.begin(), useMsCommIds.end(), commId);
    if (itMs != useMsCommIds.end()) {
        HCCL_DEBUG("[CcuStatus][%s] commId[%s] used ccu ms, removed", __func__, commId.c_str());
        useMsCommIds.erase(itMs);
    }

    auto itSched = std::find(useSchedCommIds.begin(), useSchedCommIds.end(), commId);
    if (itSched != useSchedCommIds.end()) {
        HCCL_DEBUG("[CcuStatus][%s] commId[%s] used ccu sched, removed", __func__, commId.c_str());
        useSchedCommIds.erase(itSched);
    }
}

bool CcuStatus::IsMsAvailable(const std::string& commId) const
{
    auto itMs = std::find(useMsCommIds.begin(), useMsCommIds.end(), commId);
    // ms没有通信域使用，或者就是传入通信域在使用，则可用
    return (useMsCommIds.size() < MAX_NUM_COMM_USING_MS) || (itMs != useMsCommIds.end());
}

HcclResult CcuStatus::InsertCommId(const std::string& commId, bool isUsingCcuMs, bool isUsingCcuSched)
{
    // 先删再加，避免重复添加到两种模式
    RemoveCommId(commId);
    // ccu ms 没有被使用过，则将ccu ms 标记为已使用
    if (isUsingCcuMs) {
        CHK_RET(InsertMsCommId(commId));
    } else if (isUsingCcuSched) {
        InsertSchedCommId(commId);
    } else {
        HCCL_DEBUG("NotUsingCcu comm [%s]", commId.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult CcuStatus::InsertMsCommId(const std::string& commId)
{
    if (!IsMsAvailable(commId)) {
        HCCL_WARNING(
            "[%s] ccu ms has been used by comm [%s], no more than 2 comms can use ccu ms at the same time.", __func__,
            (*(useMsCommIds.begin())).c_str());
        return HCCL_E_INTERNAL;
    }
    HCCL_DEBUG("[%s] UsingCcuMs comm [%s]", __func__, commId.c_str());
    useMsCommIds.push_back(commId);
    return HCCL_SUCCESS;
}

void CcuStatus::InsertSchedCommId(const std::string& commId)
{
    HCCL_DEBUG("[%s] UsingCcuSched comm [%s]", __func__, commId.c_str());
    useSchedCommIds.push_back(commId);
}

HcclResult CommManager::SetCommAcceleratorV2(Hccl::HcclCommunicator* communicator, int32_t accelerator)
{
    CHK_PTR_NULL(communicator);
    if (accelerator < static_cast<int32_t>(HcclAccelerator::DEFAULT)
        || accelerator > static_cast<int32_t>(HcclAccelerator::AICPU)) {
        HCCL_ERROR("[SetCommAcceleratorV2] Invalid accelerator value [%d], valid range is [0,7]", accelerator);
        return HCCL_E_NOT_SUPPORT;
    }
    HcclAccelerator hcclAccelerator = static_cast<HcclAccelerator::Value>(accelerator);

    HcclCommInfoV2& opbasedCommInfoV2 = GetCommInfoV2();
    // 通过进程锁看护，避免多个通信域同时占用CCU_MS
    std::unique_lock<std::mutex> lock(opbasedCommInfoV2.groupParamsLock);
    if ((hcclAccelerator == HcclAccelerator::CCU_MS || hcclAccelerator == HcclAccelerator::CCU_SCHED)
        && !isCcuAvailable) {
        HCCL_WARNING("CCU not support reuse in single device multi-process services, accelerator fallback AICPU_TS");
        hcclAccelerator = HcclAccelerator::AICPU_TS;
    }
    bool isMsAvailable = opbasedCommInfoV2.ccuStatus.IsMsAvailable(communicator->GetId());
    HCCL_INFO(
        "[CommManager][%s] hcclAccelerator is [%s], isMsAvailable is [%d]", __func__,
        hcclAccelerator.Describe().c_str(), isMsAvailable);
    CHK_RET(communicator->SetAccelerator(hcclAccelerator, isMsAvailable));
    return HCCL_SUCCESS;
}

std::shared_ptr<Hccl::CcuDriverHandle> CommManager::GetCcuDriver()
{
    if (isCcuAvailable == true && ccuDriverHandle == nullptr) {
        ccuDriverHandle = std::make_shared<Hccl::CcuDriverHandle>(deviceLogicId);
        if (ccuDriverHandle->Init() == HCCL_E_UNAVAIL) {
            isCcuAvailable = false;
            ccuDriverHandle = nullptr;
            HCCL_WARNING("[CommManager::GetCcuDriver]Tlv already open, isCcuAvailable updated to false");
        }
    }
    return ccuDriverHandle;
}

void CommManager::DeinitCcuDriver()
{
    if (ccuDriverHandle.use_count() == 1) {
        ccuDriverHandle = nullptr;
    }
}

HcclResult HcomGetCcuTaskInfo(const std::string& group, void* tilingData, void* ccuTaskGroup)
{
    CHK_PTR_NULL(tilingData);
    CHK_PTR_NULL(ccuTaskGroup);
    CHK_PRT_RET(group.empty(), HCCL_ERROR("[HcomGetCcuTaskInfo] group is null"), HCCL_E_PARA);

    /* 接口交互信息日志 */
    HCCL_RUN_INFO("HcomGetCcuTaskInfo:group[%s]", group.c_str());

    HcclCommInfoV2& hcomCommInfoV2 = GetCommInfoV2();

    std::unique_lock<std::mutex> groupParaLock(hcomCommInfoV2.groupParamsLock);
    auto iter = hcomCommInfoV2.hcclGroupMap.find(group);
    if (iter == hcomCommInfoV2.hcclGroupMap.end()) {
        HCCL_ERROR(
            "[HcomGetCcuTaskInfo]errNo[0x%016llx] group[%s] is not exist", HCOM_ERROR_CODE(HCCL_E_PARA), group.c_str());
        return HCCL_E_PARA;
    }
    HcclGroupParamsV2& groupParam = iter->second;
    Hccl::HcclCommunicator* comm = static_cast<Hccl::HcclCommunicator*>(groupParam.pComm.get());
    CHK_PTR_NULL(comm);
    auto ret = comm->GetCcuTaskInfo(tilingData, ccuTaskGroup);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomGetCcuTaskInfo] GetCcuTaskInfo failed.");
        return HCCL_E_INTERNAL;
    }

    HCCL_RUN_INFO("HcomGetCcuTaskInfo success group[%s]", group.c_str());
    return HCCL_SUCCESS;
}
