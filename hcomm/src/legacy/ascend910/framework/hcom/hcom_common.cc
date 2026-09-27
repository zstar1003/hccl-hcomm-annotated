/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <list>
#include <vector>
#include <string>
#include <securec.h>
#include <hccl/hccl_types.h>
// ltm指定config路径
#include "common/src/config.h"
#include "hccl/base.h"
#include "param_check_pub.h"
#include "remote_access.h"
#include "../op_base/src/op_base.h"
#include "hccl/hcom.h"
#include "rank_consistentcy_checker.h"
#include "profiling_manager_pub.h"
#include "topoinfo_ranktableParser_pub.h"
#include "stream_pub.h"
#include "hcom_common.h"
#include "comm_configer.h"
#include "hcom_private_v2.h"
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
#include "coll_comm_mgr.h"
#endif

#include "hcom_pub.h"

using namespace std;
using namespace hccl;

// DEV_TYPE_V80 对应 DevType::DEV_TYPE_V80
// DEV_TYPE_V51_310_P3 对应 DevType::DEV_TYPE_310P3
// DEV_TYPE_V71 对应 DevType::DEV_TYPE_V71
// DEV_TYPE_V51_310_P1 对应 DevType::DEV_TYPE_310P1
// DEV_TYPE_V81 对应 DevType::DEV_TYPE_V81
// DEV_TYPE_950 对应 DevType::DEV_TYPE_950
// DEV_TYPE_960 对应 DevType::DEV_TYPE_960
// DEV_TYPE_NOSOC 对应 DevType::DEV_TYPE_NOSOC

DevType MakeEnumToDevType(int makeEnum)
{
    // 正向映射：MAKE_ENUM到DevType
    static std::map<int, DevType> makeEnumToDevType
        = {{0, DevType::DEV_TYPE_910},   {1, DevType::DEV_TYPE_310P3},  {2, DevType::DEV_TYPE_910B},
           {3, DevType::DEV_TYPE_310P1}, {4, DevType::DEV_TYPE_910_93}, {5, DevType::DEV_TYPE_950},
           {6, DevType::DEV_TYPE_960},   {7, DevType::DEV_TYPE_NOSOC}};

    auto it = makeEnumToDevType.find(makeEnum);
    if (it != makeEnumToDevType.end()) {
        return it->second;
    } else {
        HCCL_WARNING("Invalid MAKE_ENUM value");
    }
    return DevType::DEV_TYPE_NOSOC;
}

using HcomCreateGroupCallback = HcclResult (*)(const std::string&, const std::vector<u32>&);
using HcomCallBackGroupIsInit = bool (*)(HcomInfo&);
using HcomDestroyGroupCallback = HcclResult (*)(const std::string&);
using HcomDestroyCallback = HcclResult (*)(HcomInfo&);
HcomCreateGroupCallback g_hcomCreateGroupCallback = nullptr;
HcomCallBackGroupIsInit g_hcomCallBackGroupIsInit = nullptr;
HcomDestroyGroupCallback g_hcomDestroyGroupCallback = nullptr;
HcomDestroyCallback g_hcomDestroyCallback = nullptr;

using HcomSetGroupTopoInfoPtr = HcclResult (*)(const char*, uint32_t);
using HcomUnsetGroupTopoInfoPtr = void (*)(const char*);
HcomSetGroupTopoInfoPtr g_hcomSetGroupTopoInfo = nullptr;
HcomUnsetGroupTopoInfoPtr g_hcomUnsetGroupTopoInfo = nullptr;

using HcomInfoCtx = struct HcomInfoCtxTag {
    HcomInfo hcomInfo;
    shared_ptr<RemoteAccess> remoteAccess;
    vector<MemRegisterAddr> remoteAddrInfos;
    HcomOpTagInfo opTagInfo;
    bool isUsed;

    HcomInfoCtxTag() : remoteAccess(nullptr), remoteAddrInfos(0), isUsed(false) {}
};

// 梯度切分相关的全局变量
namespace hccl {
std::map<std::string, std::vector<u32>> g_segmentIdxMap;
std::map<std::string, std::vector<float>> g_segmentSizeMap;
std::mutex g_segmentIdxMapLock;
std::mutex g_segmentSizeMapLock;
std::mutex g_setTaskNumCalModeLock;
} // namespace hccl

std::mutex g_hcomInfoCtxMutex;
HcomInfoCtx g_hcomInfoCtx[MAX_MODULE_DEVICE_NUM + 1];

std::mutex g_backloggedGroupLock;
std::map<std::string, std::vector<u32>> g_backloggedGroup; // 待创建的group

std::mutex g_destroyDeviceLock;
static std::mutex g_taskNumCalModeMutex;

static bool g_isAutoTuneModeOpen = false;
static bool g_notSupportSecAddrCopyWithOffset = false;

HcomInfoCtx& HcomGetCurHcomCtx(void)
{
    std::lock_guard<std::mutex> lock(g_hcomInfoCtxMutex);
    s32 deviceLogicId = INVALID_INT;
    if (hrtGetDevice(&deviceLogicId) == HCCL_SUCCESS && (static_cast<u32>(deviceLogicId) < MAX_MODULE_DEVICE_NUM)) {
        HCCL_INFO("[HcomGetCurHcomCtx] hrtGetDevice deviceLogicId[%d] ", deviceLogicId);
        /* 当前线程获取到deviceId, 如果是首次使用该deviceId的Ctx, 先判断之前是否已经配置过Ctx */
        if (!g_hcomInfoCtx[deviceLogicId].isUsed) {
            HCCL_INFO("[HcomGetCurHcomCtx] is no Used deviceLogicId[%d] ", deviceLogicId);
            if (g_hcomInfoCtx[MAX_MODULE_DEVICE_NUM].isUsed) {
                return g_hcomInfoCtx[MAX_MODULE_DEVICE_NUM];
            }
        }
        g_hcomInfoCtx[deviceLogicId].isUsed = true;
        return g_hcomInfoCtx[deviceLogicId];
    }

    /* 当前线程没有获取到deviceId, 查找是否有使用过的Ctx */
    for (u32 i = 0; i <= MAX_MODULE_DEVICE_NUM; i++) {
        if (g_hcomInfoCtx[i].isUsed) {
            HCCL_INFO("[HcomGetCurHcomCtx] no set device Used deviceLogicId[%u] ", i);
            return g_hcomInfoCtx[i];
        }
    }

    /* 当前线程没有获取到deviceId, 使用兜底Ctx */
    HCCL_INFO("[HcomGetCurHcomCtx] use cover bottom hcomInfoCtx");
    g_hcomInfoCtx[MAX_MODULE_DEVICE_NUM].isUsed = true;
    return g_hcomInfoCtx[MAX_MODULE_DEVICE_NUM];
}

HcomInfo& HcomGetCtxHomInfoById(u32 idx) { return g_hcomInfoCtx[idx].hcomInfo; }

HcomInfo& HcomGetCtxHomInfo(void)
{
    HcomInfoCtx& curHcomCtx = HcomGetCurHcomCtx();
    return curHcomCtx.hcomInfo;
}

HcomOpTagInfo& HcomGetCtxOpTagInfo(void)
{
    HcomInfoCtx& curHcomCtx = HcomGetCurHcomCtx();
    return curHcomCtx.opTagInfo;
}

bool& HcomGetCtxAutoTuneMode(void) { return g_isAutoTuneModeOpen; }

HcclResult HcomSetGroupTopoInfo(const char* group, uint32_t rankSize)
{
    if (group == nullptr) {
        HCCL_ERROR("[Hcom][HcomSetGroupTopoInfo] group is null, please check");
        return HCCL_E_PTR;
    }
    if (g_hcomSetGroupTopoInfo == nullptr) {
        HCCL_INFO("[Hcom][HcomSetGroupTopoInfo] g_hcomSetGroupTopoInfo is null");
        HcomInfo& hcomInfo = HcomGetCtxHomInfo();
        std::lock_guard<std::mutex> lock(hcomInfo.groupRankNumMapLock);
        // 1. 单算子流程下只记录，后续不做处理。
        // 2. 图模式流程下，后续会通过HcomTopoInfoFuncInstall函数调用回调函数，进行GroupTopoInfo的设置。
        hcomInfo.groupRankNumMap[std::string(group)] = rankSize;
        HCCL_RUN_INFO("[Hcom][HcomSetGroupTopoInfo] store groupRankNumMap, group:%s, rankNum:%u", group, rankSize);
        return HCCL_SUCCESS;
    }
    return g_hcomSetGroupTopoInfo(group, rankSize);
}

HcclResult HcomInitCollComm([[maybe_unused]] uint32_t rank, void** commV2, [[maybe_unused]] HcclCommPtr& comm)
{
    CHK_PTR_NULL(commV2);
    HCCL_INFO("[HcomInitCollComm] CollComm init start.");
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HcclUs startut = TIME_NOW();
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
    EXCEPTION_CATCH(comm = make_shared<hccl::hcclComm>(cclBufferSize, cclBufferSize, commName), return HCCL_E_PTR);
    void* rankGraph = nullptr;
    CHK_RET(HcclGetRankGraphV2(commV2, &rankGraph));
    constexpr HcclCommConfig* config = nullptr;
    CHK_RET(comm->InitCollComm(*commV2, rankGraph, rank, cclBuffer, commName, config));
    HCCL_RUN_INFO("[%s] success, take time [%lld]us.", __func__, DURATION_US(TIME_NOW() - startut));
#endif
    return HCCL_SUCCESS;
}

void HcomUnSetGroupTopoInfo(const char* group)
{
    if (g_hcomUnsetGroupTopoInfo == nullptr) {
        HCCL_INFO("[Hcom][HcomUnSetGroupTopoInfo] g_hcomUnsetGroupTopoInfo is null, can not unset");
        return;
    }
    if (group == nullptr) {
        HCCL_INFO("[Hcom][HcomUnSetGroupTopoInfo] group is null, can not unset");
        return;
    }
    g_hcomUnsetGroupTopoInfo(group);
    return;
}

HcclResult HcomGetCommHandleByGroup([[maybe_unused]] const char* group, [[maybe_unused]] HcclComm* commHandle)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    CHK_PTR_NULL(commHandle);
    CHK_PTR_NULL(group);

    std::shared_ptr<hcclComm> hcclComm;
    s32 deviceLogicId = 0;
    HcclResult ret = HcclDeviceRefresh(deviceLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcomGetCommHandleByGroup]HcclDeviceRefresh failed, group[%s], deviceLogicId[%d], errNo[0x%016llx]", group,
            deviceLogicId, HCOM_ERROR_CODE(ret)),
        ret);

    // MC2单算子和动态图下发性能优化，优先查询返回
    s32 devId = HcclGetThreadDeviceId();
    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclExistDeviceOpInfoCtx(devId);
    std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
    auto iter = opBaseHcom.opGroup2CommMap.find(std::string(group));
    if (iter != opBaseHcom.opGroup2CommMap.end()) {
        hcclComm = iter->second;
        CHK_PRT_RET(
            hcclComm == nullptr,
            HCCL_ERROR(
                "[HcomGetCommHandleByGroup]opBaseHcom.comm is null, group[%s], deviceLogicId[%d]", group,
                deviceLogicId),
            HCCL_E_PTR);
        *commHandle = static_cast<HcclComm>(hcclComm.get());
        return HCCL_SUCCESS;
    }
    lock.unlock();

    ret = HcomGetCommByGroup(group, hcclComm);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcomGetCommHandleByGroup]HcomGetCommByGroup failed, group[%s], deviceLogicId[%d], errNo[0x%016llx]",
            group, deviceLogicId, HCOM_ERROR_CODE(ret)),
        ret);
    *commHandle = static_cast<HcclComm>(hcclComm.get());
#endif
    return HCCL_SUCCESS;
}

HcclResult HcomGetCommByGroup(const char* group, std::shared_ptr<hccl::hcclComm>& hcclComm)
{
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    HcclResult ret = HcomCheckGroupName(group);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Get][CommByGroup]HcomCheckGroupName failed, group[%s], errNo[0x%016llx]", group, HCOM_ERROR_CODE(ret)),
        ret);
    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    if (strGroup == HCCL_WORLD_GROUP) {
        CHK_PRT_RET(hcomInfo.pComm == nullptr, HCCL_WARNING("[Get][CommByGroup]hcomInfo.pComm is null"), HCCL_E_PTR);
        hcclComm = hcomInfo.pComm;
    } else {
        std::unique_lock<std::mutex> groupParaLock(hcomInfo.groupParamsLock);
        auto iter = hcomInfo.hcomGroupMap.find(strGroup);
        if (iter == hcomInfo.hcomGroupMap.end()) {
            ret = HcclGetCommHandle(strGroup.c_str(), hcclComm);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_WARNING(
                    "[Get][CommByGroup]errNo[0x%016llx] group[%s] "
                    "does not exist",
                    HCOM_ERROR_CODE(HCCL_E_NOT_FOUND), strGroup.c_str()),
                HCCL_E_NOT_FOUND);
        } else {
            hcclComm = (iter->second).pSubComm;
            CHK_PRT_RET(
                hcclComm == nullptr, HCCL_ERROR("[Get][CommByGroup] Get Comm is null, group[%s]", strGroup.c_str()),
                HCCL_E_PTR);
        }
        groupParaLock.unlock();
    }

    return HCCL_SUCCESS;
}

void HcomTopoInfoRegCallback(HcclResult (*p1)(const char*, uint32_t), void (*p2)(const char*))
{
    g_hcomSetGroupTopoInfo = p1;
    g_hcomUnsetGroupTopoInfo = p2;

    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    std::unique_lock<std::mutex> lock(hcomInfo.groupRankNumMapLock);
    HCCL_RUN_INFO("[HcomTopoInfoRegCallback] size of groupRankNumMap is %zu", hcomInfo.groupRankNumMap.size());
    for (auto& item : hcomInfo.groupRankNumMap) {
        HCCL_RUN_INFO(
            "[HcomTopoInfoRegCallback] try to set topo info, group:%s, rankNum:%u", item.first.c_str(), item.second);
        HcclResult ret = HcomSetGroupTopoInfo(item.first.c_str(), item.second);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[HcomTopoInfoRegCallback][HcomSetGroupTopoInfo]Set Info failed. errNo[0x%016llx].",
                HCOM_ERROR_CODE(ret));
        }
    }
}

HcclResult HcomStoreBackloggedGroup(const std::string& group, const std::vector<u32>& groupRanks);
HcclResult HcomQueryGroupRef(const char* group, u32& groupRef);
HcclResult HcomDestroyBackloggedGroup(const std::string& group);
HcclResult GetGroupRankInfo(const char* group, RankInfoType rankType, u32 inPara, u32* outPara);

HcclResult GetRankList(u32 rankNum, const u32* rankIds, HcclGroupParams& params)
{
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    CHK_PTR_NULL(hcomInfo.pComm);
    std::vector<RankInfo_t> rankList;
    params.totalRanks = rankNum;
    params.worldRank = hcomInfo.params.rank;
    params.groupRank = INVALID_VALUE_RANKID;
    for (u32 i = 0; i < rankNum; i++) {
        params.groupRanks.push_back(rankIds[i]);
    }

    std::sort(params.groupRanks.begin(), params.groupRanks.end());
    if (params.groupRanks[rankNum - 1] >= hcomInfo.rankTable.rankNum) {
        HCCL_ERROR(
            "[get][RankList]errNo[0x%016llx] groupRanks[%u]:%u is invalid", HCOM_ERROR_CODE(HCCL_E_PARA), rankNum - 1,
            params.groupRanks[rankNum - 1]);
        return HCCL_E_PARA;
    }
    if (hcomInfo.rankTable.rankList.size() <= params.groupRanks[0]) {
        HCCL_ERROR(
            "[get][RankList]errNo[0x%016llx] groupRanks[0] is invalid:[%u]", HCOM_ERROR_CODE(HCCL_E_PARA),
            params.groupRanks[0]);
        return HCCL_E_PARA;
    }
    // groupRanks 个数已经校验非0
    std::string serverId = hcomInfo.rankTable.rankList[params.groupRanks[0]].serverId;
    u32 serverNum = 1; // serverNum初始值应为1，代表groupId为0的serverId;
    RankInfo_t rankInfo;
    for (u32 i = 0; i < rankNum; i++) {
        rankInfo = hcomInfo.rankTable.rankList[params.groupRanks[i]];
        // 校验worldRankID
        if (rankInfo.rankId != params.groupRanks[i]) {
            HCCL_ERROR(
                "[get][RankList]errNo[0x%016llx] in rankList, worldRanks[%u] is invalid", HCOM_ERROR_CODE(HCCL_E_PARA),
                rankInfo.rankId);
            return HCCL_E_PARA;
        }
        if (params.groupRanks[i] == params.worldRank) {
            params.groupRank = i;
        }
        if (rankInfo.serverId != serverId) {
            serverNum++;
            serverId = rankInfo.serverId;
        }
        rankInfo.rankId = i;          // 放入groupRankid
        rankList.push_back(rankInfo); // ranktable中的ranklist是以rankid的顺序排列的
        rankInfo.serverId = "";       // 释放前先指空字符串
    }
    params.serverNum = serverNum;
    bool isStandardCard = false;
    CHK_RET(hcomInfo.pComm->IsStandardCard(isStandardCard));

    if (!isStandardCard && hcomInfo.params.deviceType != DevType::DEV_TYPE_910B
        && hcomInfo.params.deviceType != DevType::DEV_TYPE_910_93) {
        CHK_RET(CheckRankTableConfigInfo(rankList, rankNum, serverNum));
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCreateGroupImpl(const std::string& group, const std::vector<u32>& rankIds)
{
    HcclUs startut = TIME_NOW();
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    std::string rankId;
    for (u32 i = 0; i < rankIds.size(); i++) {
        if (i < rankIds.size() - 1) {
            rankId += to_string(rankIds[i]) + ',';
        } else if (i == rankIds.size() - 1) {
            rankId += to_string(rankIds[i]);
        }
    }
    /* 接口交互信息日志 */
    HCCL_RUN_INFO(
        "Entry-HcomCreateGroup:group[%s], rankNum[%u], rankIds[%s]", group.c_str(), rankIds.size(), rankId.c_str());

    if (hcomInfo.params.commWorkMode == HCCL_MODE_NORMAL) {
        CHK_PRT_RET(
            hcomInfo.pComm == nullptr,
            HCCL_ERROR("[Create][Group]hcomInfo.pComm is null, please check if the initialize process is called."),
            HCCL_E_PTR);
    } else {
        if (g_hcomCreateGroupCallback != nullptr) {
            return g_hcomCreateGroupCallback(group, rankIds);
        }
    }

    CHK_PRT_RET(
        hcomInfo.rankTable.rankList.empty(), HCCL_ERROR("[Create][Group]group[%s] rankList is empty", group.c_str()),
        HCCL_E_INTERNAL);

    /* 已经存在的group不允许再次创建 */
    std::unique_lock<std::mutex> groupParaLock(hcomInfo.groupParamsLock);
    if (hcomInfo.hcomGroupMap.find(group) != hcomInfo.hcomGroupMap.end()) {
        HCCL_ERROR(
            "[Create][Group]errNo[0x%016llx] group[%s] is already exist", HCOM_ERROR_CODE(HCCL_E_PARA), group.c_str());
        return HCCL_E_PARA;
    }
    groupParaLock.unlock();

    HcclGroupParams groupParamsTem;
    CHK_RET(GetRankList(rankIds.size(), rankIds.data(), groupParamsTem));

    // 如果是groupRank = INVALID_VALUE_RANKID，即本rank不参与create group
    if (groupParamsTem.groupRank == INVALID_VALUE_RANKID) {
        HCCL_ERROR(
            "[Create][Group]errNo[0x%016llx] confirm groupRank from worldRank[%u] error",
            HCOM_ERROR_CODE(HCCL_E_NOT_FOUND), hcomInfo.params.rank);
        return HCCL_E_NOT_FOUND;
    }

    /* 入参的正确性由HCCL确保 */
    CHK_RET(hcomInfo.pComm->CreateGroup(
        group, groupParamsTem.groupRank, hcomInfo.params.rank, groupParamsTem.groupRanks, groupParamsTem.pSubComm));
    CHK_SMART_PTR_NULL(groupParamsTem.pSubComm);

    groupParaLock.lock();
    hcomInfo.hcomGroupMap.insert(std::make_pair(group, groupParamsTem));
    groupParaLock.unlock();

    HCCL_RUN_INFO(
        "hcom create group[%s] success, take time [%lld]us", group.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult HcomCreateGroup(const char* group, u32 rankNum, u32* rankIds)
{
    /* 调优模式直接返回success */
    bool& isAutoTuneModeOpen = HcomGetCtxAutoTuneMode();
    if (isAutoTuneModeOpen) {
        return HCCL_SUCCESS;
    }
    RPT_INPUT_ERR(
        group == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomCreateGroup", "nullptr", "group", "non-null pointer"}));
    CHK_PTR_NULL(group);
    HcclResult ret = HcomCheckGroupName(group);
    RPT_INPUT_ERR(
        ret != HCCL_SUCCESS, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>(
            {"HcomCreateGroup",
             {group, strnlen(group, GROUP_NAME_MAX_LEN + 1)},
             "group",
             "a non-empty string of length 1 to " + std::to_string(GROUP_NAME_MAX_LEN)
                 + ", containing only alphanumeric characters and underscores"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] group name is invalid", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCOM_ERROR_CODE(ret)),
        ret);
    CHK_PRT_RET(
        (strncmp(group, HCCL_WORLD_GROUP, sizeof(HCCL_WORLD_GROUP)) == 0),
        HCCL_ERROR(
            "[%s][%s]create group isn't support world group", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_INVALID_ARGUMENT.c_str()),
        HCCL_E_PARA);

    RPT_INPUT_ERR(
        rankIds == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomCreateGroup", "nullptr", "rankIds", "non-null pointer"}));
    CHK_PTR_NULL(rankIds);

    if (rankNum == 0) {
        RPT_INPUT_ERR(
            true, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
            std::vector<std::string>(
                {"HcomCreateGroup", std::to_string(rankNum), "rankNum",
                 "must be a positive integer (greater than 0)"}));
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] group[%s] rankNum[%u] is invalid", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCOM_ERROR_CODE(HCCL_E_PARA), group, rankNum);
        return HCCL_E_PARA;
    }
    // 入参合法性校验 END
    std::vector<u32> ranks(rankIds, rankIds + rankNum);
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        CHK_RET(HcomCreateGroupImplV2(group, rankNum, ranks));
        HcclGroupParams groupParams{};
        void* commV2 = nullptr;
        CHK_RET(HcomGetGroupParamsV2(group, static_cast<void*>(&groupParams), &commV2));
        Hccl::RankId rank = static_cast<Hccl::RankId>(groupParams.groupRank);
        CHK_RET(HcomInitCollComm(rank, &commV2, groupParams.pSubComm));
        CHK_PTR_NULL(groupParams.pSubComm);
        std::unique_lock<std::mutex> groupParaLock(hcomInfo.groupParamsLock);
        hcomInfo.hcomGroupMap.insert(std::make_pair(group, groupParams));
        groupParaLock.unlock();
        CHK_RET(HcomSetGroupTopoInfo(group, rankNum));
        HCCL_INFO("[HcomCreateGroup] create group[%s] success.", group);
        return HCCL_SUCCESS;
    }());
#endif
    if (hcomInfo.pComm == nullptr
        && ((g_hcomCallBackGroupIsInit != nullptr) && (!(g_hcomCallBackGroupIsInit(hcomInfo))))) {
        CHK_RET(HcomStoreBackloggedGroup(group, ranks));
    } else {
        CHK_RET(HcomCreateGroupImpl(group, ranks));
        CHK_RET(HcomSetGroupTopoInfo(group, rankNum));
    }
    return HCCL_SUCCESS;
}

HcclResult DestroyFlag(const char* group, bool flag)
{
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();

    if (group == nullptr) {
        return HCCL_SUCCESS; // 全局通信域不需要查询flag
    }
    std::string strGroup = group;
    std::unique_lock<std::mutex> groupParaLock(hcomInfo.groupParamsLock);
    auto iter = hcomInfo.hcomGroupMap.find(strGroup);
    if (iter == hcomInfo.hcomGroupMap.end()) {
        HCCL_ERROR(
            "[Get][CommByGroup]errNo[0x%016llx] group[%s] does not exist", HCOM_ERROR_CODE(HCCL_E_NOT_FOUND),
            strGroup.c_str());
        return HCCL_E_NOT_FOUND; // 不存在该服务器内相关dev的对应信息
    }
    iter->second.destroyFlag = flag;
    return HCCL_SUCCESS;
}

HcclResult QueryDestroyFlag(const char* group)
{
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    if (group == nullptr) {
        return HCCL_SUCCESS; // 全局通信域不需要查询flag
    }
    std::string strGroup = group;
    std::unique_lock<std::mutex> groupParaLock(hcomInfo.groupParamsLock);
    auto iter = hcomInfo.hcomGroupMap.find(strGroup);
    if (iter == hcomInfo.hcomGroupMap.end()) {
        HCCL_WARNING(
            "[Get][CommByGroup]errNo[0x%016llx] group[%s] does not exist", HCOM_ERROR_CODE(HCCL_E_AGAIN),
            strGroup.c_str());
        return HCCL_E_AGAIN; // 不存在该服务器内相关dev的对应信息
    }
    if (iter->second.destroyFlag) {
        return HCCL_E_AGAIN;
    }
    return HCCL_SUCCESS;
}

HcclResult HcomDestroyGroupImpl(const std::string& group)
{
    /* 调优模式直接返回success */
    bool& isAutoTuneModeOpen = HcomGetCtxAutoTuneMode();
    if (isAutoTuneModeOpen) {
        return HCCL_SUCCESS;
    }
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    CHK_PRT_RET(
        hcomInfo.pComm == nullptr,
        HCCL_ERROR("[Destroy][Group]hcomInfo.pComm is null, "
                   "please check if the initialize process is called."),
        HCCL_E_PTR);
    if (hcomInfo.params.commWorkMode != HCCL_MODE_NORMAL) {
        if (g_hcomDestroyGroupCallback != nullptr) {
            return g_hcomDestroyGroupCallback(group);
        }
    }

    /* 接口交互信息日志 */
    HCCL_RUN_INFO("Entry-HcomDestroyGroup:group[%s]", group.c_str());
    CHK_RET(DestroyFlag(group.c_str(), true));
    u32 ref = 0;
    CHK_RET(HcomQueryGroupRef(group.c_str(), ref));
    while (ref != 0) {
        std::shared_ptr<hccl::hcclComm> hcclComm = nullptr;
        CHK_RET(HcomGetCommByGroup(group.c_str(), hcclComm));
        SaluSleep(ONE_HUNDRED_MICROSECOND_OF_USLEEP);
        CHK_RET(HcomQueryGroupRef(group.c_str(), ref));
    }

    std::unique_lock<std::mutex> groupParaLock(hcomInfo.groupParamsLock);
    auto iter = hcomInfo.hcomGroupMap.find(group);
    if (iter == hcomInfo.hcomGroupMap.end()) {
        HCCL_ERROR(
            "[Destroy][Group]errNo[0x%016llx] group[%s] does not exist", HCOM_ERROR_CODE(HCCL_E_PARA), group.c_str());
        return HCCL_E_PARA;
    }

    CHK_RET(hcomInfo.pComm->DestroyGroup(group));

    (iter->second).groupRanks.clear(); // 清除该服务器内相关group的对应信息

    hcomInfo.hcomGroupMap.erase(group);
    groupParaLock.unlock();

    HCCL_RUN_INFO("hcom destroy group[%s] success.", group.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcomDestroyGroup(const char* group)
{
    /* 调优模式直接返回success */
    bool& isAutoTuneModeOpen = HcomGetCtxAutoTuneMode();
    if (isAutoTuneModeOpen) {
        return HCCL_SUCCESS;
    }

    RPT_INPUT_ERR(
        group == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomDestroyGroup", "nullptr", "group", "non-null pointer"}));
    CHK_PTR_NULL(group);
    CHK_RET(HcomCheckGroupName(group));

    if (strncmp(group, HCCL_WORLD_GROUP, sizeof(HCCL_WORLD_GROUP)) == 0) {
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] destroy group is world group", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        std::unique_lock<std::mutex> groupParaLock(hcomInfo.groupParamsLock);
        CHK_RET(HcomDestroyGroupImplV2(group));
        auto iter = hcomInfo.hcomGroupMap.find(group);
        if (iter == hcomInfo.hcomGroupMap.end()) {
            HCCL_ERROR(
                "[Destroy][Group]errNo[0x%016llx] group[%s] does not exist", HCOM_ERROR_CODE(HCCL_E_PARA), group);
            return HCCL_E_PARA;
        }
        hcomInfo.hcomGroupMap.erase(group);
        groupParaLock.unlock();
        return HCCL_SUCCESS;
    }());
#endif

    if (hcomInfo.pComm == nullptr
        && ((g_hcomCallBackGroupIsInit != nullptr) && (!(g_hcomCallBackGroupIsInit(hcomInfo))))) {
        CHK_RET(HcomDestroyBackloggedGroup(group));
    } else {
        CHK_RET(HcomDestroyGroupImpl(group));
    }

    HcomUnSetGroupTopoInfo(group);

    std::unique_lock<std::mutex> lock(g_backloggedGroupLock);
    if (g_backloggedGroup.find(group) != hcomInfo.backloggedGroup.end()) {
        g_backloggedGroup.erase(group);
        HCCL_INFO("hcom delete g_backlogged group[%s] success.", group);
    }
    return HCCL_SUCCESS;
}

HcclResult HcomFlushBackloggedGroups()
{
    HCCL_INFO("HcomFlushBackloggedGroups");
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    std::unique_lock<std::mutex> backGroupParaLock(g_backloggedGroupLock);
    using ITER = map<string, std::vector<u32>>::iterator;
    for (ITER iter = g_backloggedGroup.begin(); iter != g_backloggedGroup.end();) {
        HCCL_INFO("HcomFlushBackloggedGroups[%s], rank[%u]", iter->first.c_str(), hcomInfo.params.rank);
        if (std::count(iter->second.begin(), iter->second.end(), hcomInfo.params.rank) > 0) {
            HCCL_INFO("HcomFlushBackloggedGroups[%s], rank[%u] success", iter->first.c_str(), hcomInfo.params.rank);
            hcomInfo.backloggedGroup.insert({iter->first, iter->second});
        }
        iter++;
    }
    backGroupParaLock.unlock();

    std::unique_lock<std::mutex> lock(hcomInfo.backloggedGroupLock);
    for (ITER iter = hcomInfo.backloggedGroup.begin(); iter != hcomInfo.backloggedGroup.end();) {
        CHK_RET(HcomCreateGroupImpl(iter->first, iter->second));
        hcomInfo.backloggedGroup.erase(iter++);
    }
    HCCL_INFO("HcomFlushBackloggedGroups success.");
    return HCCL_SUCCESS;
}

HcclResult HcomStoreBackloggedGroup(const std::string& group, const std::vector<u32>& groupRanks)
{
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    // 该线程最开始是否未设置deviceid并获取了ctx
    bool hcomUseDefaultCtx = (&hcomInfo == &(g_hcomInfoCtx[MAX_MODULE_DEVICE_NUM].hcomInfo));
    s32 deviceLogicId = INVALID_INT;
    if (hrtGetDevice(&deviceLogicId) != HCCL_SUCCESS || (static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM)
        || hcomUseDefaultCtx) {
        HCCL_INFO("[device not set]hcom store group[%s]", group.c_str());
        std::unique_lock<std::mutex> groupParaLock(g_backloggedGroupLock);
        if (g_backloggedGroup.find(group) != g_backloggedGroup.end()) {
            HCCL_INFO("[Store][BackloggedGroup]group[%s] already exists", group.c_str());
            if (g_backloggedGroup[group] == groupRanks) {
                HCCL_ERROR("[Store][BackloggedGroup]group[%s] has been created", group.c_str());
                return HCCL_E_PARA;
            }
            g_backloggedGroup[group] = groupRanks;
            HCCL_INFO("[Store][BackloggedGroup]group[%s] updated", group.c_str());
            return HCCL_SUCCESS;
        }

        g_backloggedGroup.insert({group, groupRanks});
        HCCL_INFO("[device not set]hcom store group[%s] success", group.c_str());
        return HCCL_SUCCESS;
    }

    std::unique_lock<std::mutex> groupParaLock(hcomInfo.groupParamsLock);
    if (hcomInfo.hcomGroupMap.find(group) != hcomInfo.hcomGroupMap.end()) {
        HCCL_ERROR("[Store][BackloggedGroup]group[%s] has been created", group.c_str());
        return HCCL_E_PARA;
    }
    groupParaLock.unlock();

    std::unique_lock<std::mutex> lock(hcomInfo.backloggedGroupLock);
    if (hcomInfo.backloggedGroup.find(group) != hcomInfo.backloggedGroup.end()) {
        HCCL_ERROR("[Store][BackloggedGroup]group[%s] already exists", group.c_str());
        return HCCL_E_PARA;
    } else {
        hcomInfo.backloggedGroup.insert({group, groupRanks});
    }
    HCCL_INFO("hcom store group[%s] success.", group.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcomDestroyBackloggedGroup(const std::string& group)
{
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    std::unique_lock<std::mutex> lock(hcomInfo.backloggedGroupLock);
    if (hcomInfo.backloggedGroup.find(group) == hcomInfo.backloggedGroup.end()) {
        if (!hcomInfo.isHcomInit) {
            HCCL_WARNING(
                "[Destroy][BackloggedGroup]group[%s] does not exist, and hcom has not been inited yet", group.c_str());
            return HCCL_SUCCESS;
        } else {
            HCCL_ERROR("[Destroy][BackloggedGroup]group[%s] does not exist", group.c_str());
            return HCCL_E_PARA;
        }
    } else {
        hcomInfo.backloggedGroup.erase(group);
    }
    HCCL_INFO("hcom delete backlogged group[%s] success.", group.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcomGetbackloggedByGroup(const char* group, std::vector<u32>& groupRanks, s32& groupSize)
{
    CHK_RET(HcomCheckGroupName(group));
    std::string groupName = group;

    std::unique_lock<std::mutex> groupLock(g_backloggedGroupLock);
    auto it = g_backloggedGroup.find(groupName);
    if (it != g_backloggedGroup.end()) {
        groupRanks = it->second;
        groupSize = (it->second).size();
        HCCL_INFO("[device not set]get back logged group[%s], groupSize[%d]", groupName.c_str(), groupSize);
        return HCCL_SUCCESS;
    }
    groupLock.unlock();
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    std::unique_lock<std::mutex> groupParaLock(hcomInfo.backloggedGroupLock);
    auto iter = hcomInfo.backloggedGroup.find(groupName);
    if (iter == hcomInfo.backloggedGroup.end()) {
        groupSize = 0;
        HCCL_DEBUG(
            "[Get][CommByGroup]errNo[0x%016llx] group[%s] does not exist", HCOM_ERROR_CODE(HCCL_E_NOT_FOUND), group);
        return HCCL_SUCCESS; // 不存在该服务器内相关dev的对应信息
    }
    groupRanks = iter->second;
    groupSize = (iter->second).size();
    HCCL_INFO("[device set]get back logged group[%s], groupSize[%d]", groupName.c_str(), groupSize);
    return HCCL_SUCCESS;
}

HcclResult HcomQueryGroupRef(const char* group, u32& groupRef)
{
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    if (group == nullptr) {
        return HCCL_SUCCESS; // 全局通信域不需要查询flag
    }
    std::string strGroup = group;
    std::unique_lock<std::mutex> groupParaLock(hcomInfo.groupParamsLock);
    auto iter = hcomInfo.hcomGroupMap.find(strGroup);
    if (iter == hcomInfo.hcomGroupMap.end()) {
        HCCL_WARNING(
            "[Get][CommByGroup]errNo[0x%016llx] group[%s] does not exist", HCOM_ERROR_CODE(HCCL_E_AGAIN),
            strGroup.c_str());
        return HCCL_E_AGAIN; // 不存在该服务器内相关dev的对应信息
    }
    groupRef = iter->second.refCounter;
    return HCCL_SUCCESS;
}

HcclResult HcomGetWorldRankFromGroupRank(const char* group, u32 groupRank, u32* worldRank)
{
    RPT_INPUT_ERR(
        worldRank == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomGetWorldRankFromGroupRank", "nullptr", "worldRank", "non-null pointer"}));
    CHK_PTR_NULL(worldRank);
    bool& isAutoTuneModeOpen = HcomGetCtxAutoTuneMode();
    if (isAutoTuneModeOpen) {
        *worldRank = 0;
        return HCCL_SUCCESS;
    }
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();

    HcclResult ret = HcomCheckGroupName(group);
    RPT_INPUT_ERR(
        ret != HCCL_SUCCESS, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>(
            {"HcomGetWorldRankFromGroupRank",
             {group, strnlen(group, GROUP_NAME_MAX_LEN + 1)},
             "group",
             "a non-empty string of length 1 to " + std::to_string(GROUP_NAME_MAX_LEN)
                 + ", containing only alphanumeric characters and underscores"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] group name is invalid", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCOM_ERROR_CODE(ret)),
        ret);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN(HcomGetWorldRankFromGroupRankV2(group, groupRank, worldRank));
#endif
    if (groupRank >= hcomInfo.params.totalRanks) {
        HCCL_ERROR(
            "[Get][WorldRank]errNo[0x%016llx] groupRank[%u] is out of range[0-%u]", HCOM_ERROR_CODE(HCCL_E_PARA),
            groupRank, hcomInfo.params.totalRanks);
        return HCCL_E_PARA;
    }
    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    CHK_RET(GetGroupRankInfo(strGroup.c_str(), RankInfoType::WORLD_RANK_ID_BY_GROUP, groupRank, worldRank));
    HCCL_INFO(
        "hcom get world rank success, group[%s], groupRank[%u], worldRank[%p]", strGroup.c_str(), groupRank, worldRank);
    return HCCL_SUCCESS;
}

HcclResult HcomGetGroupRankFromWorldRank(u32 worldRank, const char* group, u32* groupRank)
{
    RPT_INPUT_ERR(
        groupRank == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomGetGroupRankFromWorldRank", "nullptr", "groupRank", "non-null pointer"}));
    CHK_PTR_NULL(groupRank);
    bool& isAutoTuneModeOpen = HcomGetCtxAutoTuneMode();
    if (isAutoTuneModeOpen) {
        *groupRank = 0;
        return HCCL_SUCCESS;
    }
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();

    HcclResult ret = HcomCheckGroupName(group);
    RPT_INPUT_ERR(
        ret != HCCL_SUCCESS, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>(
            {"HcomGetGroupRankFromWorldRank",
             {group, strnlen(group, GROUP_NAME_MAX_LEN + 1)},
             "group",
             "a non-empty string of length 1 to " + std::to_string(GROUP_NAME_MAX_LEN)
                 + ", containing only alphanumeric characters and underscores"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] group name is invalid", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCOM_ERROR_CODE(ret)),
        ret);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN(HcomGetGroupRankFromWorldRankV2(worldRank, group, groupRank));
#endif
    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    if (worldRank >= hcomInfo.params.totalRanks) {
        HCCL_ERROR(
            "[Get][GroupRank]errNo[0x%016llx] world[%u] rank is invalid", HCOM_ERROR_CODE(HCCL_E_PARA), worldRank);
        return HCCL_E_PARA;
    }
    CHK_RET(GetGroupRankInfo(strGroup.c_str(), RankInfoType::GROUP_RANK_ID_BY_WORLD, worldRank, groupRank));
    HCCL_INFO(
        "hcom get group rank success, group[%s], worldRank[%u], groupRank[%p]", strGroup.c_str(), worldRank, groupRank);
    return HCCL_SUCCESS;
}

bool HcomFindGroup(const char* group)
{
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    /* 已经存在的group不允许再次创建 */
    std::unique_lock<std::mutex> groupParaLock(hcomInfo.groupParamsLock);
    bool exists = !(hcomInfo.hcomGroupMap.find(group) == hcomInfo.hcomGroupMap.end());
    HCCL_INFO("[Find][Group] group[%s] is exist[%d]", group, exists);
    groupParaLock.unlock();
    return exists;
}

HcclResult GetWorldGroupRankInfo(RankInfoType rankType, u32 inPara, u32* outPara)
{
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    CHK_PTR_NULL(outPara);
    switch (rankType) {
        case RankInfoType::RANK_SIZE_IN_GROUP:
            *outPara = hcomInfo.params.totalRanks;
            break;

        case RankInfoType::RANK_ID_IN_GROUP:
            *outPara = hcomInfo.params.rank;
            break;

        case RankInfoType::WORLD_RANK_ID_BY_GROUP:
        case RankInfoType::GROUP_RANK_ID_BY_WORLD:
            *outPara = inPara;
            break;
        case RankInfoType::SERVER_NUM_IN_GROUP:
            *outPara = hcomInfo.rankTable.serverNum;
            break;
        default:
            HCCL_ERROR(
                "[Get][WorldGroupRankInfo]errNo[0x%016llx] invalid rankInfo type[%d]", HCOM_ERROR_CODE(HCCL_E_PARA),
                rankType);
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult GetGroupRankInfo(const char* group, RankInfoType rankType, u32 inPara, u32* outPara)
{
    CHK_PTR_NULL(outPara);
    // std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    if ((group == nullptr) || (strcmp(group, HCCL_WORLD_GROUP) == 0)) {
        CHK_RET(GetWorldGroupRankInfo(rankType, inPara, outPara));
        return HCCL_SUCCESS;
    }
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();

    std::unique_lock<std::mutex> groupParaLock(hcomInfo.groupParamsLock);
    auto iter = hcomInfo.hcomGroupMap.find(group);
    if (iter == hcomInfo.hcomGroupMap.end()) {
        HCCL_ERROR(
            "[Get][GroupRankInfo]errNo[0x%016llx] group[%s] is not exist", HCOM_ERROR_CODE(HCCL_E_NOT_FOUND), group);
        return HCCL_E_NOT_FOUND; // 不存在该服务器内相关dev的对应信息
    }
    // group ranks判空
    CHK_PRT_RET(
        (iter->second).groupRanks.empty(),
        HCCL_ERROR(
            "[Get][GroupRankInfo]errNo[0x%016llx] group[%s] "
            "ranks is empty",
            HCOM_ERROR_CODE(HCCL_E_INTERNAL), group),
        HCCL_E_INTERNAL);

    switch (rankType) {
        case RankInfoType::RANK_SIZE_IN_GROUP:
            *outPara = (iter->second).totalRanks;
            return HCCL_SUCCESS;

        case RankInfoType::RANK_ID_IN_GROUP:
            *outPara = (iter->second).groupRank;
            return HCCL_SUCCESS;

        case RankInfoType::WORLD_RANK_ID_BY_GROUP:
            if (inPara >= (iter->second).totalRanks) {
                HCCL_ERROR(
                    "[Get][GroupRankInfo]errNo[0x%016llx] group[%s] groupRank[%u] is invalid",
                    HCOM_ERROR_CODE(HCCL_E_PARA), group, inPara);
                return HCCL_E_PARA;
            }
            *outPara = (iter->second).groupRanks[inPara];
            return HCCL_SUCCESS;

        case RankInfoType::GROUP_RANK_ID_BY_WORLD:
            for (u32 rank = 0; rank < (iter->second).totalRanks; rank++) {
                if (inPara == (iter->second).groupRanks[rank]) {
                    *outPara = rank;
                    return HCCL_SUCCESS;
                }
            }
            HCCL_ERROR(
                "[Get][GroupRankInfo]errNo[0x%016llx] invalid rankInfo type[%d]", HCOM_ERROR_CODE(HCCL_E_PARA),
                rankType);
            return HCCL_E_PARA;
        case RankInfoType::SERVER_NUM_IN_GROUP:
            *outPara = (iter->second).serverNum;
            return HCCL_SUCCESS;
        default:
            HCCL_ERROR(
                "[Get][GroupRankInfo]errNo[0x%016llx] invalid rankInfo type[%d]", HCOM_ERROR_CODE(HCCL_E_PARA),
                rankType);
            return HCCL_E_PARA;
    }
}

HcclResult HcomGetRankSize(const char* group, u32* rankSize)
{
    RPT_INPUT_ERR(
        rankSize == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomGetRankSize", "nullptr", "rankSize", "non-null pointer"}));
    CHK_PTR_NULL(rankSize);
    bool& isAutoTuneModeOpen = HcomGetCtxAutoTuneMode();
    if (isAutoTuneModeOpen) {
        *rankSize = 1;
        return HCCL_SUCCESS;
    }

    HcclResult ret = HcomCheckGroupName(group);
    RPT_INPUT_ERR(
        ret != HCCL_SUCCESS, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>(
            {"HcomGetRankSize",
             {group, strnlen(group, GROUP_NAME_MAX_LEN + 1)},
             "group",
             "a non-empty string of length 1 to " + std::to_string(GROUP_NAME_MAX_LEN)
                 + ", containing only alphanumeric characters and underscores"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] group name is invalid", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCOM_ERROR_CODE(ret)),
        ret);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN(HcomGetRankSizeV2(group, rankSize));
#endif
    std::shared_ptr<hccl::hcclComm> hcclComm;
    if (group != nullptr && HcclGetCommHandle(group, hcclComm) == HCCL_SUCCESS) {
        CHK_RET(hcclComm->GetRankSize(*rankSize));
    } else {
        std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
        ret = GetGroupRankInfo(strGroup.c_str(), RankInfoType::RANK_SIZE_IN_GROUP, 0, rankSize);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Get][RankSize]errNo[0x%016llx] get group[%s] rank info error", HCOM_ERROR_CODE(ret),
                strGroup.c_str()),
            ret);
        HCCL_INFO("hcom get rank size success, group[%s]", strGroup.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult HcomDestroyOneDevice(HcomInfo& hcomInfo)
{
    HcclUs startut = TIME_NOW();

    /* 接口交互信息日志 */
    HCCL_RUN_INFO("Entry-HcomDestroy:void");

    // 模型运行结束后hcom destroy时，将CheckInfo信息清空
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();

    // group资源在word group资源销毁之前进行销毁
    hcomInfo.params.commConnections.agentConnection = nullptr;
    hcomInfo.params.commConnections.serverConnections.clear();
    hcomInfo.hcomGroupMap.clear();
    std::unique_lock<std::mutex> backloggedGroupLock(hcomInfo.backloggedGroupLock);
    hcomInfo.backloggedGroup.clear();
    backloggedGroupLock.unlock();

    hcomInfo.rankTable.nicNames.clear();
    hcomInfo.rankTable.rankList.clear();
    g_segmentIdxMap.clear();
    g_segmentSizeMap.clear();
    hcomInfo.params.profilingMode = HcomProfilingMode::PROFILING_CLOSE;
    hcomInfo.params.profilingOption = "";
    hcomInfo.isHcomInit = false;

    if (hcomInfo.params.deviceType != DevType::DEV_TYPE_NOSOC) {
        ProfilingManagerPub::ClearStoragedProfilingInfo();
    }

    /* 关键状态记录 */
    HCCL_USER_CRITICAL_LOG(
        "hcom destroy complete,take time [%lld]us, group[%s], rankNum[%u], rank[%u]", DURATION_US(TIME_NOW() - startut),
        hcomInfo.params.identifier.c_str(), hcomInfo.rankTable.rankNum, hcomInfo.params.rank);

    return HCCL_SUCCESS;
}

HcclResult HcomDestroy(void)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        std::unique_lock<std::mutex> lock(g_destroyDeviceLock);
        for (u32 i = 0; i <= MAX_MODULE_DEVICE_NUM; i++) {
            HcomInfo& hcomInfo = HcomGetCtxHomInfoById(i);
            if (hcomInfo.pComm != nullptr) {
                (void)HcclCommStateNotify(
                    static_cast<HcclComm>(hcomInfo.pComm.get()), HcclCommStatePhase::HCCL_COMM_STATE_PHASE_DESTROY_PRE);
            }
        }
        CHK_RET(HcomDestroyV2());
        for (u32 i = 0; i <= MAX_MODULE_DEVICE_NUM; i++) {
            HcomInfo& hcomInfo = HcomGetCtxHomInfoById(i);
            hcomInfo.pComm = nullptr;
            hcomInfo.hcomGroupMap.clear();
        }
        return HCCL_SUCCESS;
    }());
#endif
    std::unique_lock<std::mutex> lock(g_destroyDeviceLock);
    for (u32 i = 0; i <= MAX_MODULE_DEVICE_NUM; i++) {
        HcomInfo& hcomInfo = HcomGetCtxHomInfoById(i);

        if (!hcomInfo.isHcomInit) {
            if (hcomInfo.pComm != nullptr) {
                hcomInfo.pComm = nullptr;
            }
            HCCL_INFO("[Destroy][Result]hcomInfo[%u].isHcomInit is false.", i);

            /* 接口交互信息日志 */
            HCCL_INFO("Entry-HcomDestroy:void skip");
            if (g_hcomDestroyCallback != nullptr) {
                (void)g_hcomDestroyCallback(hcomInfo);
            }
            continue;
        } else {
            if (hcomInfo.pComm != nullptr) {
                HcomUnSetGroupTopoInfo(hcomInfo.pComm->GetIdentifier().c_str());
            }
        }

        if (hcomInfo.pComm == nullptr
            && ((g_hcomCallBackGroupIsInit != nullptr) && (!(g_hcomCallBackGroupIsInit(hcomInfo))))) {
            HCCL_INFO("[Destroy][Result]hcomInfo[%u].pComm or pCommBase is nullptr.", i);
            continue;
        }

        if (hcomInfo.params.logicDevId != HOST_DEVICE_ID) {
            u32 logicId = hcomInfo.params.logicDevId;
            if (hcomInfo.rankTable.version.compare(HETEROG_CLUSTER_VERSION) == 0) {
                CHK_RET(hrtGetDeviceIndexByPhyId(hcomInfo.params.logicDevId, logicId));
            }
            s32 deviceId = 0;
            if (hrtGetDevice(&deviceId) != HCCL_SUCCESS) {
                CHK_RET(hrtSetDevice(logicId));
                HCCL_INFO("[HcomDestroy][SetDeviceId]logicDevId[%u]", logicId);
            }
        }

        HCCL_INFO("[Destroy][Result]hcomInfo[%u].pComm destroy.", i);
        HcclResult ret = HcomDestroyOneDevice(hcomInfo);
        if (ret == HCCL_SUCCESS) {
            HCCL_INFO("[Destroy][Result]hcomInfo[%u].pComm HcomDestroyOneDevice success.", i);
        } else {
            HCCL_INFO("[Destroy][Result]hcomInfo[%u].pComm HcomDestroyOneDevice fail.", i);
            return ret;
        }

        if (g_hcomDestroyCallback != nullptr) {
            (void)g_hcomDestroyCallback(hcomInfo);
        }

        hcomInfo.pComm = nullptr;
        hcomInfo.hcclCommTopoInfoDetectServer.clear();
        hcomInfo.hcclCommTopoInfoDetectAgent.clear();
    }
    return HCCL_SUCCESS;
}

void HcomGroupCallbackFuncInstall(
    HcclResult (*p1)(const std::string&, const std::vector<u32>&), bool (*p2)(HcomInfo&),
    HcclResult (*p3)(const std::string&), HcclResult (*p4)(HcomInfo&))
{
    g_hcomCreateGroupCallback = p1;
    g_hcomCallBackGroupIsInit = p2;
    g_hcomDestroyGroupCallback = p3;
    g_hcomDestroyCallback = p4;
}

HcclResult HcomSetGradFusionByIndex(const char* group, u32 segmentNum, const u32* IdxList)
{
    bool& isAutoTuneModeOpen = HcomGetCtxAutoTuneMode();
    if (isAutoTuneModeOpen) {
        return HCCL_SUCCESS;
    }

    RPT_INPUT_ERR(
        IdxList == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomSetGradFusionByIndex", "nullptr", "IdxList", "non-null pointer"}));
    CHK_PTR_NULL(IdxList);
    bool bRet = segmentNum == 0;
    RPT_INPUT_ERR(
        bRet, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>(
            {"HcomSetGradFusionByIndex", std::to_string(0), "segmentNum",
             "must be a positive integer (greater than 0)"}));
    CHK_PRT_RET(
        bRet,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] set split IdxList length is zero", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCOM_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);
    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    string idxList;
    for (u32 i = 0; i < segmentNum; i++) {
        if (i < segmentNum - 1) {
            idxList += to_string(IdxList[i]) + ',';
        } else if (i == segmentNum - 1) {
            idxList += to_string(IdxList[i]);
        }
    }
    /* 接口交互信息日志 */
    HCCL_RUN_INFO(
        "Entry-HcomSetGradFusionByIndex:group[%s], segmentNum[%u], IdxList[%s]", strGroup.c_str(), segmentNum,
        idxList.c_str());

    CHK_RET(HcomCheckGroupName(strGroup.c_str()));

    std::vector<u32> tempList;

    for (u32 segidx = 0; segidx < segmentNum; segidx++) {
        tempList.push_back(IdxList[segidx]);
    }

    for (u32 i = 0; i < tempList.size() - 1; i++) {
        if (tempList[i] >= tempList[i + 1]) {
            HCCL_ERROR(
                "[Set][GradFusionByIndex]errNo[0x%016llx] index list is not ascending", HCOM_ERROR_CODE(HCCL_E_PARA));
            return HCCL_E_PARA;
        }
    }
    std::unique_lock<std::mutex> segmentIdxMapLock(g_segmentIdxMapLock);
    g_segmentIdxMap.insert(std::pair<std::string, std::vector<u32>>(strGroup, tempList));
    segmentIdxMapLock.unlock();
    return HCCL_SUCCESS;
}

HcclResult HcomSetGradFusionBySize(const char* group, u32 segmentNum, const float* sizeList)
{
    bool& isAutoTuneModeOpen = HcomGetCtxAutoTuneMode();
    if (isAutoTuneModeOpen) {
        return HCCL_SUCCESS;
    }

    RPT_INPUT_ERR(
        sizeList == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomSetGradFusionBySize", "nullptr", "sizeList", "non-null pointer"}));
    CHK_PTR_NULL(sizeList);
    bool bRet = segmentNum == 0;
    RPT_INPUT_ERR(
        bRet, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>(
            {"HcomSetGradFusionBySize", std::to_string(0), "segmentNum",
             "must be a positive integer (greater than 0)"}));
    CHK_PRT_RET(
        bRet,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] set split sizeList length is zero", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCOM_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);
    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    string strSizeList;
    for (u32 i = 0; i < segmentNum; i++) {
        if (i < segmentNum - 1) {
            strSizeList += to_string(sizeList[i]) + ',';
        } else if (i == segmentNum - 1) {
            strSizeList += to_string(sizeList[i]);
        }
    }
    /* 接口交互信息日志 */
    HCCL_RUN_INFO(
        "Entry-HcomSetGradFusionBySize:group[%s], segmentNum[%u], sizeList[%s]", strGroup.c_str(), segmentNum,
        strSizeList.c_str());

    CHK_RET(HcomCheckGroupName(strGroup.c_str()));
    std::vector<float> tempList;
    float sizeTotal = 0;

    for (u32 sizeIdx = 0; sizeIdx < segmentNum; sizeIdx++) {
        bRet = sizeList[sizeIdx] < 0;
        CHK_PRT_RET(
            bRet,
            HCCL_ERROR(
                "[Set][GradFusionBySize]errNo[0x%016llx] sizeList[%u] less than zero", HCOM_ERROR_CODE(HCCL_E_PARA),
                sizeIdx),
            HCCL_E_PARA);
        tempList.push_back(sizeList[sizeIdx]);
        sizeTotal += sizeList[sizeIdx];
    }

    if (std::fabs(sizeTotal - 100) > 1e-6) { // 判断用户设置总百分比是否为100%
        HCCL_ERROR("[Set][GradFusionBySize]errNo[0x%016llx] size list sum is not 100%%", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    } else {
        std::unique_lock<std::mutex> segmentSizeMapLock(g_segmentSizeMapLock);
        g_segmentSizeMap.insert(std::pair<std::string, std::vector<float>>(strGroup, tempList));
        segmentSizeMapLock.unlock();
        return HCCL_SUCCESS;
    }
}

HcclResult HcomGenerateCommId(hccl::HcclCommParams& params)
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

HcclResult InitHcomMiscInfo(hccl::HcclCommParams& params, const char* rankTable)
{
    CHK_PTR_NULL(rankTable);

    RankConsistentcyChecker::GetInstance().SetCheckCannVersionSwitch(true); // 打开CANN软件版本校验开关

    // 记录版本信息
    std::string curVersion = GetExternalInputCannVersion();
    CHK_RET(RankConsistentcyChecker::GetInstance().RecordVerInfo(curVersion));
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    // 计算rankTable的crc值并保存
    HcclResult ret = HcomCalcCRC(params, rankTable);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][OtherInfo]errNo[0x%016llx] calc ranktable crc error", HCCL_ERROR_CODE(HCCL_E_INTERNAL)),
        HCCL_E_INTERNAL);
    // 生成通信域标识符
    ret = HcomGenerateCommId(hcomInfo.params);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][OtherInfo]errNo[0x%016llx] generate CommId error", HCCL_ERROR_CODE(HCCL_E_INTERNAL)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

bool HcomCheckrtMemcpyAddrAsync(const std::string& group)
{
    float counterVaule = 1.0f;

    // 偏移拷贝的二级指针
    void* deviceMemSrcLevel2 = nullptr;
    void* deviceMemDstLevel2 = nullptr;
    // 偏移拷贝的一级指针
    void* deviceMemSrc = nullptr;
    void* deviceMemDst = nullptr;

    auto deleter = [&deviceMemSrcLevel2, &deviceMemDstLevel2, &deviceMemSrc, &deviceMemDst](void* dst) {
        if (dst != nullptr) {
            CHK_PRT(hrtFree(dst));
            if (dst == deviceMemSrcLevel2) {
                deviceMemSrcLevel2 = nullptr;
            } else if (dst == deviceMemDstLevel2) {
                deviceMemDstLevel2 = nullptr;
            } else if (dst == deviceMemSrc) {
                deviceMemSrc = nullptr;
            } else if (dst == deviceMemDst) {
                deviceMemDst = nullptr;
            }
        }
    };

    CHK_RET(hrtMalloc(&deviceMemSrcLevel2, sizeof(void*)));
    unique_ptr<void, decltype(deleter)> deviceMemSrcLevel2Unique(deviceMemSrcLevel2, deleter);
    CHK_RET(hrtMalloc(&deviceMemDstLevel2, sizeof(void*)));
    unique_ptr<void, decltype(deleter)> deviceMemDstLevel2Unique(deviceMemDstLevel2, deleter);
    CHK_RET(hrtMalloc(&deviceMemSrc, sizeof(float)));
    unique_ptr<void, decltype(deleter)> deviceMemSrcUnique(deviceMemSrc, deleter);
    CHK_RET(hrtMalloc(&deviceMemDst, sizeof(float)));
    unique_ptr<void, decltype(deleter)> deviceMemDstUnique(deviceMemDst, deleter);

    CHK_RET(hrtMemSyncCopy(
        deviceMemDst, sizeof(float), &counterVaule, sizeof(float),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    CHK_RET(hrtMemSyncCopy(
        deviceMemSrc, sizeof(float), &counterVaule, sizeof(float),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    CHK_RET(hrtMemSyncCopy(
        deviceMemDstLevel2, sizeof(void*), &deviceMemDst, sizeof(void*),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    CHK_RET(hrtMemSyncCopy(
        deviceMemSrcLevel2, sizeof(void*), &deviceMemSrc, sizeof(void*),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    u64 destMax = sizeof(s32);
    u64 offset = 0;

    Stream stream(StreamType::STREAM_TYPE_ONLINE);
    bool notSupportSecAddrCopyWithOffset = false;

    HcclResult ret
        = hrtMemcpyAddrAsync(deviceMemDstLevel2, destMax, offset, deviceMemSrcLevel2, destMax, offset, stream.ptr());
    if (ret == HCCL_E_NOT_SUPPORT) {
        notSupportSecAddrCopyWithOffset = true;
    } else {
        CHK_RET(hcclStreamSynchronize(stream.ptr(), CommConfiger::GetInstance().GetCommConfigExecTimeOut(group)));
    }

    g_notSupportSecAddrCopyWithOffset = notSupportSecAddrCopyWithOffset;

    return notSupportSecAddrCopyWithOffset;
}

bool HcomGetSecAddrCopyFlag(const char* socVersion)
{
    HCCL_INFO("[Hcom][HcomGetSecAddrCopyFlag] SecAddrCopyWithOffset flag is %d", g_notSupportSecAddrCopyWithOffset);
    DevType devType;
    std::string socVersionStr(socVersion);
    CHK_RET(hrtGetDeviceTypeBySocVersion(socVersionStr, devType));

    return !g_notSupportSecAddrCopyWithOffset
           && (devType == DevType::DEV_TYPE_910B || devType == DevType::DEV_TYPE_910);
}

HcclResult HcomNormalInit(const char* rankTableM, const char* identify)
{
    HcclResult ret = HCCL_SUCCESS;
    bool& isAutoTuneModeOpen = HcomGetCtxAutoTuneMode();
    isAutoTuneModeOpen = false;
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();

    /*--------------入参合法性检测---------------------*/
    CHK_PTR_NULL(rankTableM);
    CHK_PTR_NULL(identify);

    /* 防止重复调用初始化 */
    CHK_PRT_RET(
        (hcomInfo.pComm != nullptr),
        HCCL_ERROR(
            "[Init][Result]errNo[0x%016llx] identify[%s], "
            "multiple initialization is not supported",
            HCOM_ERROR_CODE(HCCL_E_UNAVAIL), identify),
        HCCL_E_UNAVAIL);

    /* --------------初始化------------------------- */
    bool errorFlag = false;
    s32 logicDevId = 0;
    hcomInfo.params.commWorkMode = WorkMode::HCCL_MODE_NORMAL;
    do {
        ret = InitHcomMiscInfo(hcomInfo.params, rankTableM);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[Init][Result]errNo[0x%016llx] init other Info.", HCOM_ERROR_CODE(ret)),
            errorFlag = true);

        DevType deviceType;
        CHK_PRT_BREAK(hrtGetDevice(&logicDevId) != HCCL_SUCCESS, , errorFlag = true);
        CHK_RET(hrtGetDeviceType(deviceType));
        // 为适配12包，做此修改
        g_notSupportSecAddrCopyWithOffset = HcomCheckrtMemcpyAddrAsync(identify);

        ret = CfgGetClusterInfo(
            rankTableM, identify, hcomInfo.params, hcomInfo.rankTable, GetExternalInputInterSuperPodRetryEnable(),
            deviceType);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Init][Result]errNo[0x%016llx] cfg get ranktable[%p] info "
                "error: identify[%s]",
                HCOM_ERROR_CODE(ret), rankTableM, identify),
            errorFlag = true);

        if (hcomInfo.rankTable.serverNum != SINGLE_SERVER_NUM
            && (deviceType == DevType::DEV_TYPE_310P3 || deviceType == DevType::DEV_TYPE_310P1)) {
            CHK_RET(InitExternalInputHeterog());
        }

        hcomInfo.pComm.reset(new (std::nothrow) hccl::hcclComm(0, 0, HCCL_WORLD_GROUP));

        CHK_PRT_RET(
            hcomInfo.pComm == nullptr,
            HCCL_ERROR("[Init][Result]hcomInfo.pComm is null, "
                       "create failed"),
            HCCL_E_PTR);
        CommConfig commConfig(identify);
        ret = hcomInfo.pComm->init(hcomInfo.params, commConfig, hcomInfo.rankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[Init][Result]errNo[0x%016llx] hcclComm init error", HCOM_ERROR_CODE(ret)),
            errorFlag = true);

        ret = ShowRanktableConfigInfo(hcomInfo.cloudFlag, hcomInfo.params, hcomInfo.rankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][Result]errNo[0x%016llx] put ranktable info error", HCOM_ERROR_CODE(ret)),
            errorFlag = true);
        ret = InitWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][Result]errNo[0x%016llx] init work flow mode error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        ret = HcomFlushBackloggedGroups();
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][Result]errNo[0x%016llx] create backlogged group failed", HCOM_ERROR_CODE(ret)),
            errorFlag = true);

        ret = HcomSetGroupTopoInfo(hcomInfo.pComm->GetIdentifier().c_str(), hcomInfo.rankTable.rankNum);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Init][Result]errNo[0x%016llx] SetGroupTopoInfo error, "
                "group[%s]",
                HCOM_ERROR_CODE(ret), (hcomInfo.pComm->GetIdentifier().c_str())),
            errorFlag = true);
    } while (0);

    if (errorFlag) {
        HCCL_ERROR(
            "[Init][Result]hcom init failed, rankNum[%u], rank[%u], server[%s], device[%d], return[0x%016llx]",
            hcomInfo.rankTable.rankNum, hcomInfo.params.rank, hcomInfo.params.serverId.c_str(), logicDevId,
            HCOM_ERROR_CODE(ret));
        (void)HcomDestroy();
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckInitClusterInfo(const char* rankTableM, const char* identify)
{
    HcclResult ret = HCCL_SUCCESS;
    // rankTable合法性检测
    u32 rankTableSize = 0;
    ret = HcomCheckRankTable(rankTableM, rankTableSize);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] input rankTable error", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCOM_ERROR_CODE(ret)),
        ret);

    // identify合法性检测
    ret = HcomCheckIdentify(identify);
    RPT_INPUT_ERR(
        ret != HCCL_SUCCESS, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>(
            {"HcomInit", {identify, strnlen(identify, IDENTIFY_MAX_LEN + 1)}, "identify", "a valid node identifier"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] identify parameter error", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCOM_ERROR_CODE(ret)),
        ret);
    return ret;
}

HcclResult HcomInitByFile(const char* rankTablePath, const char* identify)
{
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();

    CHK_PTR_NULL(rankTablePath);
    CHK_PTR_NULL(identify);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        CHK_RET(HcomInitByFileV2(rankTablePath, identify));
        u32 rankNum = 0;
        CHK_RET(HcomGetRankSize(HCCL_WORLD_GROUP, &rankNum));
        s32 myRank = std::atoi(identify);
        Hccl::RankId rank = static_cast<Hccl::RankId>(myRank);
        void* commV2 = nullptr;
        CHK_RET(HcomGetCommV2(&commV2));
        CHK_RET(HcomInitCollComm(rank, &commV2, hcomInfo.pComm));
        CHK_RET(HcomSetGroupTopoInfo(HCCL_WORLD_GROUP, rankNum));
        return HCCL_SUCCESS;
    }());
#endif

    HcclUs startut = TIME_NOW();
    HcclResult ret = HCCL_SUCCESS;

    // 读取rankTable文件到内存
    std::string rankTableM;
    std::string realFilePath;
    ret = HcomLoadRanktableFile(rankTablePath, rankTableM, realFilePath);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] rankTablePath[%s] identify[%s] load rankTable error.",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CONFIG.c_str(), HCCL_ERROR_CODE(HCCL_E_INTERNAL),
            rankTablePath, identify),
        HCCL_E_INTERNAL);
    CHK_RET(HcomCheckInitClusterInfo(rankTableM.c_str(), identify));
    HCCL_RUN_INFO("Entry-HcomInitByFile:rankTablePath[%s], identify[%s]", realFilePath.c_str(), identify);

    CHK_RET(InitExternalInput());
    CHK_RET(InitEnvConfig());

    // 调用初始化接口
    ret = HcomNormalInit(rankTableM.c_str(), identify);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcomInitByFile]errNo[0x%016llx] rankTablePath[%s] identify[%s] "
            "hcom init failed.",
            HCCL_ERROR_CODE(ret), realFilePath.c_str(), identify),
        ret);
    hcomInfo.isHcomInit = true;
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "[HCCL_TRACE]hcom init by file success,take time [%lld]us, rankTablePath[%s], rankNum[%u], rank[%u],"
        "server[%s], device[%d]",
        DURATION_US(TIME_NOW() - startut), realFilePath.c_str(), hcomInfo.rankTable.rankNum, hcomInfo.params.rank,
        hcomInfo.params.serverId.c_str(), hcomInfo.params.logicDevId);
    return HCCL_SUCCESS;
}

DevType HcomGetDeviceType()
{
    DevType devType;
    hrtGetDeviceType(devType);
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        HcomGetDevTypeV2(devType);
        HCCL_INFO("LaunchHcomKernel: devType is DEV_TYPE_950 or DEV_TYPE_960");
        return MakeEnumToDevType(static_cast<int>(devType));
    }

    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    return hcomInfo.params.deviceType;
}

HcclResult HcomCreateCommCCLbuffer(const char* group)
{
    RPT_INPUT_ERR(
        group == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomGetDevType", "nullptr", "group", "non-null pointer"}));
    CHK_PTR_NULL(group);

    HcclResult ret = HcomCheckGroupName(group);
    RPT_INPUT_ERR(
        ret != HCCL_SUCCESS, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>(
            {"HcomGetDevType",
             {group, strnlen(group, GROUP_NAME_MAX_LEN + 1)},
             "group",
             "a non-empty string of length 1 to " + std::to_string(GROUP_NAME_MAX_LEN)
                 + ", containing only alphanumeric characters and underscores"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Get][HcomGetDevType]errNo[0x%016llx] group name is invalid", HCOM_ERROR_CODE(ret)), ret);

    DevType devType = HcomGetDeviceType();
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
        HCCL_INFO("HcomCreateCommCclBufV2 start.");
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
        HCCLV2_FUNC_RUN(HcomCreateCommCclBufV2(group));
#endif
        return HCCL_SUCCESS;
    }

    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(group, hcclComm));
    CHK_RET(hcclComm->CreateCommCCLbuffer());
    return HCCL_SUCCESS;
}

HcclResult HcomGetInCCLbuffer(const char* group, void** buffer, u64* size)
{
    RPT_INPUT_ERR(
        group == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomGetInCCLbuffer", "nullptr", "group", "non-null pointer"}));
    CHK_PTR_NULL(group);
    CHK_PTR_NULL(buffer);
    CHK_PTR_NULL(size);

    HcclResult ret = HcomCheckGroupName(group);
    RPT_INPUT_ERR(
        ret != HCCL_SUCCESS, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>(
            {"HcomGetInCCLbuffer",
             {group, strnlen(group, GROUP_NAME_MAX_LEN + 1)},
             "group",
             "a non-empty string of length 1 to " + std::to_string(GROUP_NAME_MAX_LEN)
                 + ", containing only alphanumeric characters and underscores"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Get][HcomGetInCCLbuffer]errNo[0x%016llx] group name is invalid", HCOM_ERROR_CODE(ret)), ret);

    DevType devType = HcomGetDeviceType();
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
        HCCLV2_FUNC_RUN(HcomGetInCclBufV2(group, *buffer, *size));
#endif
        return HCCL_SUCCESS;
    }

    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(group, hcclComm));
    CHK_RET(hcclComm->GetInCCLbuffer(*buffer, *size));
    return HCCL_SUCCESS;
}

HcclResult HcomGetOutCCLbuffer(const char* group, void** buffer, u64* size)
{
    RPT_INPUT_ERR(
        group == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomGetOutCCLbuffer", "nullptr", "group", "non-null pointer"}));
    CHK_PTR_NULL(group);
    CHK_PTR_NULL(buffer);
    CHK_PTR_NULL(size);

    HcclResult ret = HcomCheckGroupName(group);
    RPT_INPUT_ERR(
        ret != HCCL_SUCCESS, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>(
            {"HcomGetOutCCLbuffer",
             {group, strnlen(group, GROUP_NAME_MAX_LEN + 1)},
             "group",
             "a non-empty string of length 1 to " + std::to_string(GROUP_NAME_MAX_LEN)
                 + ", containing only alphanumeric characters and underscores"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Get][HcomGetOutCCLbuffer]errNo[0x%016llx] group name is invalid", HCOM_ERROR_CODE(ret)), ret);

    DevType devType = HcomGetDeviceType();
    if (devType == DevType::DEV_TYPE_950 || devType == DevType::DEV_TYPE_960) {
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
        HCCLV2_FUNC_RUN(HcomGetOutCclBufV2(group, *buffer, *size));
#endif
        return HCCL_SUCCESS;
    }

    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(group, hcclComm));
    CHK_RET(hcclComm->GetOutCCLbuffer(*buffer, *size));
    return HCCL_SUCCESS;
}

HcclResult HcomGetAicpuOpStreamNotify(const char* group, HcclRtStream* opStream, u8 aicpuNotifyNum, void** aicpuNotify)
{
    RPT_INPUT_ERR(
        group == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomGetDevType", "nullptr", "group", "non-null pointer"}));
    CHK_PTR_NULL(group);
    CHK_PTR_NULL(opStream);
    CHK_PTR_NULL(aicpuNotify);

    HcclResult ret = HcomCheckGroupName(group);
    RPT_INPUT_ERR(
        ret != HCCL_SUCCESS, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>(
            {"HcomGetDevType",
             {group, strnlen(group, GROUP_NAME_MAX_LEN + 1)},
             "group",
             "a non-empty string of length 1 to " + std::to_string(GROUP_NAME_MAX_LEN)
                 + ", containing only alphanumeric characters and underscores"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Get][HcomGetDevType]errNo[0x%016llx] group name is invalid", HCOM_ERROR_CODE(ret)), ret);

    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(group, hcclComm));
    CHK_RET(hcclComm->GetAicpuOpStreamNotify(opStream, aicpuNotifyNum, aicpuNotify));
    return HCCL_SUCCESS;
}

HcclResult HcomMc2AiCpuStreamAllocAndGet(const char* group, u32 streamMode, rtStream_t* aiCpuStream)
{
    RPT_INPUT_ERR(
        group == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcomGetDevType", "nullptr", "group", "non-null pointer"}));
    CHK_PTR_NULL(group);
    CHK_PTR_NULL(aiCpuStream);

    HcclResult ret = HcomCheckGroupName(group);
    RPT_INPUT_ERR(
        ret != HCCL_SUCCESS, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>(
            {"HcomGetDevType",
             {group, strnlen(group, GROUP_NAME_MAX_LEN + 1)},
             "group",
             "a non-empty string of length 1 to " + std::to_string(GROUP_NAME_MAX_LEN)
                 + ", containing only alphanumeric characters and underscores"}));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Get][HcomGetDevType]errNo[0x%016llx] group name is invalid", HCOM_ERROR_CODE(ret)), ret);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN(HcomMc2AiCpuStreamAllocAndGetV2(group, streamMode, aiCpuStream));
#endif
    std::shared_ptr<hccl::hcclComm> hcclComm;
    ret = HcomGetCommByGroup(group, hcclComm);
    // 兼容V2，获取通信域失败由外层判断，此处不报ERROR
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_WARNING("[%s] HcomGetCommByGroup fail", __func__), ret);
    CHK_RET(hcclComm->Mc2AiCpuStreamAllocAndGet(streamMode, *aiCpuStream));
    return HCCL_SUCCESS;
}
