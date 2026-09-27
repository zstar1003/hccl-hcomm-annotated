/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_base_v2.h"
#include <algorithm>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <fstream>
#include <string>
#include <hccl/hccl_types.h>
#include <adapter_error_manager_pub.h>

#include "hccl/base.h"
#include "mc2_type.h"
#include "param_check_v2.h"
#include "orion_adapter_rts.h"
#include "hccl_communicator.h"
#include "hccl_common_v2.h"
#include "log.h"
#include "sal.h"
#include "communicator_callback.h"
#include "root_handle_v2.h"
#include "root_info_detect_bridge.h"
#include "stream.h"
#include "env_config_v2.h"
#include "stream_utils.h"

#include "hostdpu/dpu_kernel_entrance.h"
#include "hostdpu/flush_manager.h"

using namespace std;
using namespace Hccl;
std::map<std::string, Hccl::HcclCommunicator*> g_hcclCommunicators[MAX_MODULE_DEVICE_NUM + 1];

static OpType GetOpTypeV2(std::string opTypeName)
{
    HCCL_INFO("GetOpTypeV2 start, opTypeName %s", opTypeName.c_str());
    std::transform(opTypeName.begin(), opTypeName.end(), opTypeName.begin(), ::toupper);
    auto iter = HCOM_OP_TYPE_STR_MAP_V2.find(opTypeName);
    if (iter != HCOM_OP_TYPE_STR_MAP_V2.end()) {
        return iter->second;
    }
    return OpType::OPTYPEINVALID;
}

static void LoadConfigCommName(string& commId, const HcclCommConfig& config)
{
    if (config.hcclCommName[0] == '\0') {
        HCCL_WARNING("[LoadConfigCommName] config.hcclCommName is empty, use default commId[%s]", commId.c_str());
        return;
    }

    auto commNameLength = strlen(config.hcclCommName);
    commNameLength = commNameLength < COMM_NAME_MAX_LENGTH ? commNameLength : COMM_NAME_MAX_LENGTH;
    commId = std::string(config.hcclCommName, commNameLength);
    HCCL_RUN_INFO("Entry-%s: set commName[%s]", __func__, commId.c_str());
}

thread_local s32 g_hcclDeviceId = INVALID_INT;
static HcclResult HcclGetDeviceId(void)
{
    if (g_hcclDeviceId == INVALID_INT) {
        aclError ret = aclrtGetDevice(&g_hcclDeviceId);
        CHK_PRT_RET(
            ret != ACL_SUCCESS, HCCL_WARNING("[HcclGetDeviceId]aclrtGetDevice failed, ret[%d]", ret), HCCL_E_INTERNAL);
    }
    CHK_PRT_RET(
        static_cast<u32>(g_hcclDeviceId) >= MAX_MODULE_DEVICE_NUM,
        HCCL_WARNING(
            "[HcclGetDeviceId]userDevId[%d] is bigger than HCCL_AISERVER_DEVICE_NUM_MAX:[%u]", g_hcclDeviceId,
            MAX_MODULE_DEVICE_NUM),
        HCCL_E_INTERNAL);
    HCCL_INFO("[HcclGetDeviceId] userDevId[%d] ", g_hcclDeviceId);
    return HCCL_SUCCESS;
}

static s32 HcclGetThreadDeviceId()
{
    CHK_PRT_RET(
        HcclGetDeviceId() != HCCL_SUCCESS,
        HCCL_WARNING("[HcclGetThreadDeviceId] get fail userDevId[%d]", g_hcclDeviceId), INVALID_INT);
    return g_hcclDeviceId;
}

template <typename Func>
HcclResult HcclCommOperationImplV2(HcclComm comm, const std::string func_name, Func operate)
{
    s32 userDevId = HcclGetThreadDeviceId();
    s32 devPhyId = HrtGetDevicePhyIdByUserDevId(userDevId);
    HcclUs startut = TIME_NOW();
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    auto ret = operate(*communicator);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] errNo[0x%016llx] userDevId[%d], comm[%s]", func_name.c_str(), ret, userDevId,
            communicator->GetId().c_str()),
        HCCL_E_INTERNAL);
    HCCL_RUN_INFO(
        "%s success, take time [%lld]us, userDevId[%d], devPhyId[%d], comm[%s]", func_name.c_str(),
        DURATION_US(TIME_NOW() - startut), userDevId, devPhyId, communicator->GetId().c_str());
    return HCCL_SUCCESS;
}

namespace {
std::map<HcclReduceOp, ReduceOp> HCCL_OP_REDUCE_MAP
    = {{HCCL_REDUCE_SUM, ReduceOp::SUM},
       {HCCL_REDUCE_PROD, ReduceOp::PROD},
       {HCCL_REDUCE_MAX, ReduceOp::MAX},
       {HCCL_REDUCE_MIN, ReduceOp::MIN}};
}

static void CheckHcclDeterministic(uint32_t hcclDeterministic)
{
    if (hcclDeterministic == 0) {
        HCCL_WARNING("[HcclCommInitClusterInfoConfig] hcclDeterministic[%u] is not support.", hcclDeterministic);
    } else if (hcclDeterministic != HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET && hcclDeterministic != 1) {
        HCCL_WARNING("[HcclCommInitClusterInfoConfig] hcclDeterministic[%u] is invalid.", hcclDeterministic);
    }
}

HcclResult CreateCommConfig(uint32_t rank, HcclCommConfig* config, HcclComm* comm, std::string& ranktableM)
{
    string commId(HCCL_WORLD_GROUP);
    LoadConfigCommName(commId, *config);

    HcclCommInfoV2& opbasedCommInfoV2 = GetCommInfoV2();
    CHK_PRT_RET(
        opbasedCommInfoV2.hcclGroupMap.find(commId) != opbasedCommInfoV2.hcclGroupMap.end(),
        HCCL_ERROR(
            "[CreateCommConfig] errNo[0x%016llx] The comm name[%s] already exists in Group2Comm map.",
            HCCL_ERROR_CODE(HCCL_E_PARA), commId.c_str()),
        HCCL_E_PARA);

    bool devUsed = false;
    bool isWorldGroup = true;
    Hccl::CommParams commParams{commId,
                                static_cast<Hccl::RankId>(rank),
                                0,
                                static_cast<Hccl::RankId>(rank),
                                Hccl::HrtGetDeviceType(),
                                devUsed,
                                isWorldGroup};

    shared_ptr<HcclCommConfig> hcclConf;
    EXCEPTION_CATCH((hcclConf = make_shared<HcclCommConfig>()), return HCCL_E_PTR);
    CHK_SAFETY_FUNC_RET(
        memcpy_s(hcclConf->reserved, sizeof(hcclConf->reserved), config->reserved, sizeof(config->reserved)));

    hcclConf->hcclBufferSize = config->hcclBufferSize;
    if (config->hcclBufferSize == HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET) {
        hcclConf->hcclBufferSize = 0;
        HCCL_INFO("[HcclCommInitClusterInfoConfig] set default HCCL BUFFER");
    }
    CheckHcclDeterministic(config->hcclDeterministic);
    // 默认全都开启确定性计算
    hcclConf->hcclDeterministic = 1;

    CHK_SAFETY_FUNC_RET(memcpy_s(
        hcclConf->hcclCommName, sizeof(hcclConf->hcclCommName), config->hcclCommName, sizeof(config->hcclCommName)));
    CHK_SAFETY_FUNC_RET(
        memcpy_s(hcclConf->hcclUdi, sizeof(hcclConf->hcclUdi), config->hcclUdi, sizeof(config->hcclUdi)));

    opbasedCommInfoV2.pComm.reset(new (std::nothrow) Hccl::HcclCommunicator(commParams, hcclConf.get()));
    CHK_SMART_PTR_NULL(opbasedCommInfoV2.pComm);

    /* --------------初始化------------------------- */
    HcclResult ret = HCCL_SUCCESS;
    bool errorFlag = false;
    do {
        ret = opbasedCommInfoV2.pComm->Init(ranktableM);
        CHK_PRT_BREAK(
            ret != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR(
                "[CreateCommConfig]opbasedCommInfoV2.pComm->Init failed, errNo[0x%016llx]", HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        opbasedCommInfoV2.pComm->RegisterAcceStateCallBack(CommunicatorCallback());
        s32 userDevId = HrtGetDevice();
        ret = CommManager::GetInstance(userDevId).SetCommAcceleratorV2(
            opbasedCommInfoV2.pComm.get(), config->hcclOpExpansionMode); // 通信域创建，设置默认accelerator
        CHK_PRT_BREAK(
            ret != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CreateCommConfig]SetCommAcceleratorV2 failed, errNo[0x%016llx]", HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        *comm = static_cast<HcclComm>(opbasedCommInfoV2.pComm.get());
        HcclGetRankSizeV2(*comm, &commParams.rankSize);
        opbasedCommInfoV2.commParams = commParams;

        HcclGroupParamsV2 params{};
        params.pComm = opbasedCommInfoV2.pComm;
        params.groupRank = static_cast<Hccl::RankId>(rank);
        std::unique_lock<std::mutex> lock(opbasedCommInfoV2.groupParamsLock);
        opbasedCommInfoV2.hcclGroupMap[commId] = params;

        opbasedCommInfoV2.pComm->RegisterPrintChannelInfoCallback(
            CommManager::GetInstance(userDevId).GetPrintChannelInfoCallback());
    } while (0);

    if (errorFlag) {
        HCCL_ERROR(
            "[Init][CreateCommConfig]CreateCommConfig failed, rank[%u], commId[%s],"
            "return[0x%016llx]",
            rank, commId.c_str(), HCCL_ERROR_CODE(ret));
        (void)HcclCommDestroyV2(opbasedCommInfoV2.pComm.get());
        *comm = nullptr;
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult CreateCommConfigRootInfo(
    uint32_t rank, const HcclCommConfig* config, const std::string& identifier, const RankTableInfo& ranktable,
    HcclComm* comm)
{
    // check
    HcclCommInfoV2& opbasedCommInfoV2 = GetCommInfoV2();
    bool commExists = opbasedCommInfoV2.hcclGroupMap.find(identifier) != opbasedCommInfoV2.hcclGroupMap.end();
    RPT_INPUT_ERR(
        commExists, "EI0003", vector<string>({"ccl_op", "value", "parameter", "expect"}),
        vector<string>({"HcclCommInitRootInfo", identifier, "identifier", "an identifier that is not in use"}));
    CHK_PRT_RET(
        commExists,
        HCCL_ERROR(
            "[HcclCommInitRootInfoConfigV2]errNo[0x%016llx] The comm name[%s] already exists in Group2Comm map.",
            HCCL_ERROR_CODE(HCCL_E_PARA), identifier.c_str()),
        HCCL_E_PARA);

    bool devUsed = false;
    bool isWorldGroup = true;
    Hccl::CommParams commParams{identifier,
                                static_cast<Hccl::RankId>(rank),
                                0,
                                static_cast<Hccl::RankId>(rank),
                                Hccl::HrtGetDeviceType(),
                                devUsed,
                                isWorldGroup};

    shared_ptr<HcclCommConfig> hcclConf;
    EXCEPTION_CATCH((hcclConf = make_shared<HcclCommConfig>()), return HCCL_E_PTR);
    CHK_SAFETY_FUNC_RET(
        memcpy_s(hcclConf->reserved, sizeof(hcclConf->reserved), config->reserved, sizeof(config->reserved)));

    hcclConf->hcclBufferSize = config->hcclBufferSize;
    if (config->hcclBufferSize == HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET) {
        hcclConf->hcclBufferSize = 0;
        HCCL_INFO("[HcclCommInitRootInfoConfigV2] set default HCCL BUFFER");
    }
    CheckHcclDeterministic(config->hcclDeterministic);
    // 默认全都开启确定性计算
    hcclConf->hcclDeterministic = 1;

    CHK_SAFETY_FUNC_RET(memcpy_s(
        hcclConf->hcclCommName, sizeof(hcclConf->hcclCommName), config->hcclCommName, sizeof(config->hcclCommName)));
    CHK_SAFETY_FUNC_RET(
        memcpy_s(hcclConf->hcclUdi, sizeof(hcclConf->hcclUdi), config->hcclUdi, sizeof(config->hcclUdi)));

    opbasedCommInfoV2.pComm.reset(new (std::nothrow) Hccl::HcclCommunicator(commParams, hcclConf.get()));
    CHK_SMART_PTR_NULL(opbasedCommInfoV2.pComm);
    HcclResult ret = HCCL_SUCCESS;
    bool errorFlag = false;
    do {
        ret = opbasedCommInfoV2.pComm->Init(ranktable);
        CHK_PRT_BREAK(
            ret != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[%s]opbasedCommInfoV2.pComm->Init failed, errNo[0x%016llx]", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        opbasedCommInfoV2.pComm->RegisterAcceStateCallBack(CommunicatorCallback());
        s32 userDevId = HrtGetDevice();
        ret = CommManager::GetInstance(userDevId).SetCommAcceleratorV2(
            opbasedCommInfoV2.pComm.get(), config->hcclOpExpansionMode); // 通信域创建，设置默认accelerator
        CHK_PRT_BREAK(
            ret != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[%s]SetCommAcceleratorV2 failed, errNo[0x%016llx]", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        *comm = static_cast<HcclComm>(opbasedCommInfoV2.pComm.get());

        HcclGetRankSizeV2(*comm, &commParams.rankSize);
        opbasedCommInfoV2.commParams = commParams;

        HcclGroupParamsV2 params{};
        params.pComm = opbasedCommInfoV2.pComm;
        params.groupRank = static_cast<Hccl::RankId>(rank);
        std::unique_lock<std::mutex> lock(opbasedCommInfoV2.groupParamsLock);
        opbasedCommInfoV2.hcclGroupMap[identifier] = params;

        opbasedCommInfoV2.pComm->RegisterPrintChannelInfoCallback(
            CommManager::GetInstance(userDevId).GetPrintChannelInfoCallback());
    } while (0);

    if (errorFlag) {
        HCCL_ERROR(
            "[Init][%s]CreateCommConfigRootInfo failed return[0x%016llx] rank[%u] identifier[%s] commName[%s] udi[%s]",
            __func__, HCCL_ERROR_CODE(ret), rank, identifier.c_str(), hcclConf->hcclCommName, hcclConf->hcclUdi);
        (void)HcclCommDestroyV2(opbasedCommInfoV2.pComm.get());
        *comm = nullptr;
        return ret;
    }

    return HCCL_SUCCESS;
}

// 单算子 创建全局通信域 Json V2
static HcclResult ParseJsonAndCreateComm(
    nlohmann::json& data, uint32_t rank, HcclCommConfig* config, HcclComm* comm, std::string& ranktableM)
{
    CHK_PRT_RET(
        data.find("rank_count") == data.end(),
        HCCL_ERROR("[HcclCommInitClusterInfo] json object has no property called 'rank_count'"), HCCL_E_PARA);

    HcclResult ret = CreateCommConfig(rank, config, comm, ranktableM);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ParseJsonAndCreateComm]CreateCommConfig fail, errNo[%d]", ret), ret);

    return HCCL_SUCCESS;
}

// 单算子 创建全局通信域 RankTable V2
HcclResult HcclCommInitClusterInfoV2(const char* clusterInfo, uint32_t rank, HcclComm* comm)
{
    HcclUs startut = TIME_NOW();
    s32 userDevId = HcclGetThreadDeviceId();
    s32 devPhyId = HrtGetDevicePhyIdByUserDevId(userDevId);
    CHK_PTR_NULL(clusterInfo);
    CHK_PTR_NULL(comm);

    // 读取ranktable文件
    std::string ranktableM;
    CHK_RET(HcomLoadRankTableFileV2(clusterInfo, ranktableM));

    nlohmann::json data;
    try {
        data = nlohmann::json::parse(ranktableM);
    } catch (const nlohmann::json::parse_error& e) {
        HCCL_ERROR("JSON parse error: %s at byte %d", e.what(), e.byte);
        return HCCL_E_PARA;
    } catch (const nlohmann::json::exception& e) {
        HCCL_ERROR("JSON parse error: %s", e.what());
        return HCCL_E_PARA;
    } catch (...) {
        HCCL_ERROR("load allocated resource to json fail, please check json input");
        return HCCL_E_INTERNAL;
    };

    CHK_RET(CallSingletons()); // 临时规避，在初始化通信域前声明单例保证时序
    // check
    string commId(HCCL_WORLD_GROUP);
    HcclCommInfoV2& opbasedCommInfoV2 = GetCommInfoV2();
    CHK_PRT_RET(
        opbasedCommInfoV2.hcclGroupMap.find(commId) != opbasedCommInfoV2.hcclGroupMap.end(),
        HCCL_ERROR(
            "[HcclCommInitClusterInfoV2]errNo[0x%016llx] The comm name[%s] already exists in Group2Comm map.",
            HCCL_ERROR_CODE(HCCL_E_PARA), commId.c_str()),
        HCCL_E_PARA);

    bool devUsed = false;
    bool isWorldGroup = true;
    Hccl::CommParams commParams{commId,
                                static_cast<Hccl::RankId>(rank),
                                0,
                                static_cast<Hccl::RankId>(rank),
                                Hccl::HrtGetDeviceType(),
                                devUsed,
                                isWorldGroup};
    opbasedCommInfoV2.pComm.reset(new (std::nothrow) Hccl::HcclCommunicator(commParams));
    CHK_PTR_NULL(opbasedCommInfoV2.pComm);
    /* --------------初始化------------------------- */
    HcclResult ret = HCCL_SUCCESS;
    bool errorFlag = false;
    do {
        ret = opbasedCommInfoV2.pComm->Init(ranktableM);
        CHK_PRT_BREAK(
            ret != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[%s]opbasedCommInfoV2.pComm->Init failed, errNo[0x%016llx]", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        opbasedCommInfoV2.pComm->RegisterAcceStateCallBack(CommunicatorCallback());
        s32 userDevId = HrtGetDevice();
        ret = CommManager::GetInstance(userDevId).SetCommAcceleratorV2(
            opbasedCommInfoV2.pComm.get(), 0); // 通信域创建，设置默认accelerator
        CHK_PRT_BREAK(
            ret != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[%s]SetCommAcceleratorV2 failed, errNo[0x%016llx]", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        *comm = static_cast<HcclComm>(opbasedCommInfoV2.pComm.get());

        HcclGetRankSizeV2(*comm, &commParams.rankSize);
        opbasedCommInfoV2.commParams = commParams;

        HcclGroupParamsV2 params{};
        params.pComm = opbasedCommInfoV2.pComm;
        params.groupRank = static_cast<Hccl::RankId>(rank);
        std::unique_lock<std::mutex> lock(opbasedCommInfoV2.groupParamsLock);
        opbasedCommInfoV2.hcclGroupMap[commId] = params;

        opbasedCommInfoV2.pComm->RegisterPrintChannelInfoCallback(
            CommManager::GetInstance(userDevId).GetPrintChannelInfoCallback());
    } while (0);

    if (errorFlag) {
        HCCL_ERROR(
            "[Init][%s]HcclCommInitClusterInfoV2 failed, clusterInfo[%s], rank[%u], userDevId[%d], devPhyId[%d], "
            "commId[%s]"
            "return[0x%016llx]",
            __func__, clusterInfo, rank, userDevId, devPhyId, commId.c_str(), HCCL_ERROR_CODE(ret));
        (void)HcclCommDestroyV2(opbasedCommInfoV2.pComm.get());
        *comm = nullptr;
        return ret;
    }
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "[HCCL_TRACE]%s success, take time [%lld]us, clusterInfo[%s], rank[%u], userDevId[%d], devPhyId[%d], "
        "commId[%s].",
        __func__, DURATION_US(TIME_NOW() - startut), clusterInfo, rank, userDevId, devPhyId, commId.c_str());
    return HCCL_SUCCESS;
}

HcclResult
HcclCommInitClusterInfoMemConfigV2(const char* rankTableString, uint32_t rank, HcclCommConfig* config, HcclComm* comm)
{
    CHK_RET(CallSingletons()); // 临时规避，在初始化通信域前声明单例保证时序

    HCCL_INFO("HcclCommInitClusterInfoMemConfigV2 Begin, commName[%s]", config->hcclCommName);
    std::string rankTableM(rankTableString);
    nlohmann::json data;
    try {
        data = nlohmann::json::parse(rankTableM);
    } catch (const nlohmann::json::parse_error& e) {
        HCCL_ERROR("[RankTable]JSON parse error: %s at byte %d", e.what(), e.byte);
        return HCCL_E_INTERNAL;
    } catch (const nlohmann::json::exception& e) {
        HCCL_ERROR("[RankTable]JSON parse error: %s", e.what());
        return HCCL_E_INTERNAL;
    } catch (...) {
        HCCL_ERROR("[RankTable]load allocated resource to json fail, please check json input");
        return HCCL_E_INTERNAL;
    };
    HcclResult ret = ParseJsonAndCreateComm(data, rank, config, comm, rankTableM);
    CHK_PRT_RET(
        ret,
        HCCL_ERROR(
            "[Parse][Json]errNo[0x%016llx] and create comm failed, commName[%s].", HCCL_ERROR_CODE(ret),
            config->hcclCommName),
        static_cast<HcclResult>(ret));
    HCCL_INFO("HcclCommInitClusterInfoMemConfigV2 End, commName[%s]", config->hcclCommName);
    return HCCL_SUCCESS;
}

HcclResult
HcclCommInitClusterInfoConfigV2(const char* clusterInfo, uint32_t rank, HcclCommConfig* config, HcclComm* comm)
{
    HcclUs startut = TIME_NOW();
    s32 userDevId = HcclGetThreadDeviceId();
    s32 devPhyId = HrtGetDevicePhyIdByUserDevId(userDevId);
    HCCL_RUN_INFO(
        "Entry-HcclCommInitClusterInfoConfig V950, commEngine[%u], commId[%s]", config->hcclOpExpansionMode,
        config->hcclCommName);

    CHK_RET(CallSingletons()); // 临时规避，在初始化通信域前声明单例保证时序

    HcclCommInfoV2& opbasedCommInfoV2 = GetCommInfoV2();
    if (opbasedCommInfoV2.status == DeviceStatus::DEVICE_RECOVERED) {
        if (opbasedCommInfoV2.pComm->GetId() != config->hcclCommName) {
            HCCL_WARNING("[HcclCommInitClusterInfoConfig] Device was recovered, but communicator is search miss.");
        } else {
            *comm = static_cast<HcclComm>(opbasedCommInfoV2.pComm.get());
            return HCCL_SUCCESS;
        }
    }

    HCCL_INFO(
        "HcclCommInitClusterInfoConfig ranktable[%s], config->hcclBufferSize[%u] MB", clusterInfo,
        config->hcclBufferSize);
    if (UNLIKELY(config->hcclBufferSize == 0)) {
        HCCL_ERROR("HcclCommInitClusterInfoConfigV2 config: hcclBufferSize is 0 MB, invalid para");
        return HCCL_E_PARA;
    }
    // 读取ranktable文件
    std::string ranktableM;
    CHK_RET(HcomLoadRankTableFileV2(clusterInfo, ranktableM));

    nlohmann::json data;
    try {
        data = nlohmann::json::parse(ranktableM);
    } catch (const nlohmann::json::parse_error& e) {
        HCCL_ERROR("[RankTable]JSON parse error: %s at byte %d", e.what(), e.byte);
        return HCCL_E_INTERNAL;
    } catch (const nlohmann::json::exception& e) {
        HCCL_ERROR("[RankTable]JSON parse error: %s", e.what());
        return HCCL_E_INTERNAL;
    } catch (...) {
        HCCL_ERROR("[RankTable]load allocated resource to json fail, please check json input");
        return HCCL_E_INTERNAL;
    };

    HcclResult ret = ParseJsonAndCreateComm(data, rank, config, comm, ranktableM);

    CHK_PRT_RET(
        ret, HCCL_ERROR("[Parse][Json]errNo[0x%016llx] and create comm failed.", HCCL_ERROR_CODE(ret)),
        static_cast<HcclResult>(ret));

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "[HCCL_TRACE]%s success, take time [%lld]us, clusterInfo[%s], rank[%u], userDevId[%d], devPhyId[%d], "
        "commId[%s].",
        __func__, DURATION_US(TIME_NOW() - startut), clusterInfo, rank, userDevId, devPhyId, config->hcclCommName);

    return HCCL_SUCCESS;
}

HcclResult HcclCheckTaskServiceExist(const std::string& commId, s32 deviceId)
{
    auto outerIt = g_taskServiceMap.find(commId);
    if (outerIt == g_taskServiceMap.end()) {
        HCCL_ERROR(
            "[CheckTaskServiceExist] TaskService of CommId[%s] deviceId[%d],CommId Not Found", commId.c_str(),
            deviceId);
        return HCCL_E_NOT_FOUND;
    }
    auto innerIt = outerIt->second.find(deviceId);
    if (innerIt == outerIt->second.end()) {
        HCCL_ERROR(
            "[CheckTaskServiceExist] TaskService of CommId[%s] deviceId[%d],deviceId Not Found", commId.c_str(),
            deviceId);
        return HCCL_E_NOT_FOUND;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclTaskRegisterV2(HcclComm comm, const char* msgTag, Callback cb)
{
    HCCL_RUN_INFO("[HcclTaskRegisterV2] start to register task");
    CHK_PTR_NULL(comm);
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    std::string commId = communicator->GetId();
    s32 deviceId = communicator->GetDeviceLogicId();
    std::lock_guard<std::mutex> lock(g_serMapMutex);
    CHK_RET(HcclCheckTaskServiceExist(commId, deviceId));

    return g_taskServiceMap[commId][deviceId]->TaskRegister(msgTag, cb);
}

HcclResult HcclTaskRegisterProfV2(HcclComm comm, ProfCallbackTemplate profCallback)
{
    CHK_PTR_NULL(comm);
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    std::string commId = communicator->GetId();
    HCCL_INFO("[HcclTaskRegisterProfV2] commId[%s]", commId.c_str());
    s32 deviceId = communicator->GetDeviceLogicId();
    std::lock_guard<std::mutex> lock(g_serMapMutex);
    CHK_RET(HcclCheckTaskServiceExist(commId, deviceId));
    return g_taskServiceMap[commId][deviceId]->TaskProfRegister(profCallback);
}

HcclResult HcclTaskReportRegisterV2(HcclComm comm, ReportCallbackTemplate reportCallback)
{
    CHK_PTR_NULL(comm);
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    std::string commId = communicator->GetId();
    HCCL_INFO("[HcclTaskReportRegisterV2] commId[%s]", commId.c_str());
    s32 deviceId = communicator->GetDeviceLogicId();
    std::lock_guard<std::mutex> lock(g_serMapMutex);
    CHK_RET(HcclCheckTaskServiceExist(commId, deviceId));
    return g_taskServiceMap[commId][deviceId]->TaskReportRegister(reportCallback);
}

HcclResult HcclGetDpuSteamIdV2(HcclComm comm, u32& dpuStreamId)
{
    HCCL_RUN_INFO("[HcclTaskRegisterV2] start to Get DpuStreamId");
    CHK_PTR_NULL(comm);
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    auto ret = communicator->GetStreamId(dpuStreamId);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[HcclGetDpuSteamIdV2] GetStreamId failed, ret[0x%016llx]", HCCL_ERROR_CODE(ret));
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclTaskUnRegisterV2(HcclComm comm, const char* msgTag)
{
    CHK_PTR_NULL(comm);
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    std::string commId = communicator->GetId();
    s32 deviceId = communicator->GetDeviceLogicId();
    std::lock_guard<std::mutex> lock(g_serMapMutex);
    HCCL_RUN_INFO(
        "[HcclTaskUnRegisterV2] start to unregister task, g_taskServiceMap.size()==%zu", g_taskServiceMap.size());
    CHK_RET(HcclCheckTaskServiceExist(commId, deviceId));
    return g_taskServiceMap[commId][deviceId]->TaskUnRegister(msgTag);
}

HcclResult HcclGetRootInfoV2(HcclRootInfo* rootInfo)
{
    const RootInfoDetectBridge* bridge = GetRootInfoDetectBridge();
    CHK_PRT_RET(
        bridge == nullptr || bridge->getRootInfo == nullptr,
        HCCL_ERROR("[%s] RootInfoDetect bridge is not registered.", __func__), HCCL_E_INTERNAL);
    return bridge->getRootInfo(rootInfo);
}

HcclResult
GetDeviceCommV2(uint32_t ndev, const HcclRootInfo& rootHandle, const s32 rank, const s32 logicDeviceId, HcclComm& comm)
{
    // 给当前线程添加名字
    SetThreadName("Hccl_GetDevComm");

    TRY_CATCH_RETURN(HrtSetDevice(logicDeviceId));
    std::string identifier;
    HcclResult ret = HcclCommInitRootInfoV2(ndev, &rootHandle, rank, &comm, identifier);
    if (ret != HCCL_SUCCESS || comm == nullptr) {
        comm = nullptr;
        HCCL_ERROR("[GetDeviceComm] rank[%d] Get device comm failed, ret[%d]!", rank, ret);
        TRY_CATCH_RETURN(HrtSetDevice(logicDeviceId));
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclGetCommAllV2(uint32_t ndev, int32_t* devices, HcclComm* comms)
{
    // 入参校验
    CHK_PRT_RET(ndev == 0, HCCL_ERROR("[HcclGetCommAll] ndev is invalid ndev[%u]", ndev), HCCL_E_PARA);
    CHK_PTR_NULL(comms);
    CHK_PTR_NULL(devices);

    // 给当前线程添加名字
    SetThreadName("Hccl_GetCommAllV2");

    TRY_CATCH_RETURN(HrtSetDevice(devices[0]));

    // 获取通信域之前, 先把所有通信域设置为空
    for (uint32_t i = 0; i < ndev; i++) {
        comms[i] = nullptr;
    }

    HcclRootInfo rootHandle;
    CHK_RET(HcclGetRootInfoV2(&rootHandle));

    std::vector<std::unique_ptr<std::thread>> threads(ndev);
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        threads[rankId].reset(new (std::nothrow) std::thread(
            &GetDeviceCommV2, ndev, std::ref(rootHandle), rankId, devices[rankId], std::ref(comms[rankId])));
        CHK_PRT_RET(
            !threads[rankId], HCCL_ERROR("[HcclGetCommAllV2]threads[%u] reset failed ", rankId), HCCL_E_INTERNAL);
    }
    for (uint32_t i = 0; i < ndev; i++) {
        threads[i]->join();
    }

    // 如果任何一个通信域初始化失败，将所有已经成功创建的通信域销毁
    bool isFailed = false;
    for (uint32_t i = 0; i < ndev; ++i) {
        if (comms[i] == nullptr) {
            HCCL_ERROR("[HcclGetCommAllV2] rank[%u] get comm failed!", i);
            isFailed = true;
            break;
        }
    }
    if (isFailed) {
        for (uint32_t i = 0; i < ndev; ++i) {
            if (comms[i] != nullptr) {
                (void)HcclCommDestroyV2(comms[i]);
            }
        }
        return HCCL_E_INTERNAL;
    }

    TRY_CATCH_RETURN(HrtSetDevice(devices[0]));

    return HCCL_SUCCESS;
}

HcclResult HcclCommInitAllV2(uint32_t ndev, int32_t* devices, HcclComm* comms)
{
    HcclUs startut = TIME_NOW();
    std::string devicesStr;
    for (size_t i = 0; i < ndev; ++i) {
        std::string deviceStr = std::to_string(devices[i]);
        devicesStr += deviceStr;
        if (i != ndev - 1) {
            devicesStr += " ";
        }
    }
    HCCL_RUN_INFO("Entry-HcclCommInitAll V950, ndev:[%u], devices:[%s].", ndev, devicesStr.c_str());

    std::future<HcclResult> threadResult;
    std::unique_ptr<std::thread> getCommThread;
    getCommThread.reset(new (std::nothrow) std::thread([=, &threadResult]() {
        threadResult = std::async(std::launch::async, HcclGetCommAllV2, ndev, devices, comms);
    }));
    CHK_PRT_RET(!getCommThread, HCCL_ERROR("[HcclCommInitAll]thread reset failed "), HCCL_E_INTERNAL);
    getCommThread->join();

    HcclResult ret = threadResult.get();
    if (ret != HCCL_SUCCESS) {
        for (uint32_t i = 0; i < ndev; ++i) {
            if (comms[i] != nullptr) {
                (void)HcclCommDestroyV2(comms[i]);
                comms[i] = nullptr;
            }
        }
        HCCL_ERROR("HcclCommInitAll failed! threadResult[%d]", ret);
        return ret;
    }
    s32 userDevId = HcclGetThreadDeviceId();
    s32 devPhyId = HrtGetDevicePhyIdByUserDevId(userDevId);
    HCCL_RUN_INFO(
        "HcclCommInitAll success, take time [%lld]us, userDevId[%d], devPhyId[%d].", DURATION_US(TIME_NOW() - startut),
        userDevId, devPhyId);
    return HCCL_SUCCESS;
}

HcclResult HcclCommDestroyV2(HcclComm comm)
{
    HcclUs startut = TIME_NOW();
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    CHK_PTR_NULL(communicator);
    string commId = communicator->GetId();
    HcclCommInfoV2& opbasedCommInfoV2 = GetCommInfoV2();
    HCCL_RUN_INFO("Entry-HcclCommDestroy V950 comm[%s]", commId.c_str());

    if (communicator->GetCommStatus() == CommStatus::COMM_INUSE) {
        HCCL_WARNING("[HcclCommDestroy] comm is in use, please try again later");
        return HCCL_E_AGAIN;
    }
    std::unique_lock<std::mutex> lock(opbasedCommInfoV2.groupParamsLock);
    if (commId == opbasedCommInfoV2.commParams.commId && opbasedCommInfoV2.pComm != nullptr) {
        // 通信域销毁，更新子通信域ccu使用情况
        opbasedCommInfoV2.ccuStatus.RemoveCommId(opbasedCommInfoV2.pComm->GetId());
        for (auto iterGroup : opbasedCommInfoV2.hcclGroupMap) {
            opbasedCommInfoV2.ccuStatus.RemoveCommId(iterGroup.first);
        }
        opbasedCommInfoV2.pComm = nullptr;
        opbasedCommInfoV2.status = DeviceStatus::DEVICE_IDLE;
    }
    // 通信域销毁，更新ccu使用情况
    opbasedCommInfoV2.ccuStatus.RemoveCommId(commId);
    auto iter = opbasedCommInfoV2.hcclGroupMap.find(commId);
    if (iter != opbasedCommInfoV2.hcclGroupMap.end()) {
        opbasedCommInfoV2.hcclGroupMap.erase(
            commId); // 删除通信域实例，在communicatorImpl类的析构函数中有dpu与npu之间共享内存的销毁
        // 通信域销毁，更新ccu使用情况
        opbasedCommInfoV2.ccuStatus.RemoveCommId(commId);
    } else {
        s32 userDevId = HcclGetThreadDeviceId();
        HCCL_ERROR(
            "[HcclCommDestroyV2] comm is not exist, comm=%p, group=%s, userDevId=%d", comm, commId.c_str(), userDevId);
        return HCCL_E_PARA;
    }
    lock.unlock();

    s32 userDevId = HcclGetThreadDeviceId();
    s32 devPhyId = HrtGetDevicePhyIdByUserDevId(userDevId);
    HCCL_RUN_INFO(
        "Entry-HcclCommDestroy V950 comm[%s] success, take time [%lld]us, userDevId[%d], devPhyId[%d].", commId.c_str(),
        DURATION_US(TIME_NOW() - startut), userDevId, devPhyId);
    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAllV2(
    const void* sendBuf, uint64_t sendCount, HcclDataType sendType, const void* recvBuf, uint64_t recvCount,
    HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    bool isCapture;
    rtModel_t rtModel = nullptr;
    u32 modelId = 0;

    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    const std::string tag = "ALLTOALL_" + communicator->GetId();

    CHK_RET(HcomCheckOpParamV2(tag.c_str(), 0, sendType, stream));
    CHK_RET(HcomCheckDataTypeV2(recvType));
    CHK_RET(GetStreamCaptureInfo(stream, rtModel, isCapture));

    // 接口交互信息日志
    char stackLogBufferV2[LOG_TMPBUF_SIZE];
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        s32 streamId = HrtGetStreamId(stream);
        s32 deviceLogicId = HrtGetDevice();
        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(communicator->GetRankId(localRank), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBufferV2, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], sendCount[%llu], recvCount[%llu], sendType[%s],"
            "recvType[%s], localRank[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, sendCount, recvCount, GetDataTypeEnumStrV2(sendType).c_str(),
            GetDataTypeEnumStrV2(recvType).c_str(), localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclAlltoAllV2:" + std::string(stackLogBufferV2);
        if (isCapture) {
            CHK_PTR_NULL(rtModel);
            // 获取不到modelId会报错
            CHK_RET(GetModelId(rtModel, modelId));
            logInfo += ", model id[" + to_string(modelId) + "].";
        }
        communicator->GetTrace().Save(logInfo);
    }

    static thread_local Hccl::CollOpParams opParams;
    opParams.opType = Hccl::OpType::ALLTOALL;
    opParams.sendBuf = const_cast<void*>(sendBuf);
    opParams.recvBuf = const_cast<void*>(recvBuf);
    opParams.all2AllDataDes.sendCount = sendCount;
    opParams.all2AllDataDes.recvCount = recvCount;
    opParams.all2AllDataDes.sendType = HcclDataTypeToDataType(sendType);
    opParams.all2AllDataDes.recvType = HcclDataTypeToDataType(recvType);
    opParams.dataType = HcclDataTypeToDataType(sendType);
    opParams.opTag = tag;

    CHK_RET_AND_PRINT_IDE(communicator->LoadOpbasedCollOp(opParams, static_cast<void*>(stream)), tag.c_str());

    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo
            = "HcclAlltoAllV2:success,take time: " + std::to_string(DURATION_US(endut - startut).count()) + " us,"
              + std::string(stackLogBufferV2);
        communicator->GetTrace().Save(endInfo);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAllVV2(
    const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
    const void* recvCounts, const void* rdispls, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    bool isCapture;
    rtModel_t rtModel = nullptr;
    u32 modelId = 0;

    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    const std::string tag = "HCCL_ALLTOALLV_" + communicator->GetId();

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), 0, sendType, stream), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckDataTypeV2(recvType), tag.c_str());
    CHK_RET(HcomCheckDataTypeV2(sendType));
    CHK_RET(HcomCheckDataTypeV2(recvType));
    CHK_RET(GetStreamCaptureInfo(stream, rtModel, isCapture));
    /* 根据ranksize校验相关入参 */
    u32 rankSize = 0;
    CHK_RET(communicator->GetRankSize(&rankSize));
    CHK_RET(HcomCheckAlltoAllVExternalMemV2(sendBuf, sendCounts, recvBuf, recvCounts, rankSize));

    // 接口交互信息日志
    char stackLogBufferV2[LOG_TMPBUF_SIZE];
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        s32 streamId = HrtGetStreamId(stream);
        s32 deviceLogicId = HrtGetDevice();
        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(communicator->GetRankId(localRank), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBufferV2, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], sendCounts[%p], recvCounts[%p], sendType[%s],"
            "recvType[%s], localRank[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, sendCounts, recvCounts, GetDataTypeEnumStrV2(sendType).c_str(),
            GetDataTypeEnumStrV2(recvType).c_str(), localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclAlltoAllVV2:" + std::string(stackLogBufferV2);
        if (isCapture) {
            CHK_PTR_NULL(rtModel);
            // 获取不到modelId会报错
            CHK_RET(GetModelId(rtModel, modelId));
            logInfo += ", model id[" + to_string(modelId) + "].";
        }
        communicator->GetTrace().Save(logInfo);
    }

    static thread_local Hccl::CollOpParams opParams;
    opParams.opType = Hccl::OpType::ALLTOALLV;
    opParams.sendBuf = const_cast<void*>(sendBuf);
    opParams.recvBuf = const_cast<void*>(recvBuf);
    opParams.all2AllVDataDes.sendCounts = const_cast<void*>(sendCounts);
    opParams.all2AllVDataDes.recvCounts = const_cast<void*>(recvCounts);
    opParams.all2AllVDataDes.sdispls = const_cast<void*>(sdispls);
    opParams.all2AllVDataDes.rdispls = const_cast<void*>(rdispls);
    opParams.all2AllVDataDes.sendType = HcclDataTypeToDataType(sendType);
    opParams.all2AllVDataDes.recvType = HcclDataTypeToDataType(recvType);
    opParams.dataType = HcclDataTypeToDataType(sendType);
    opParams.opTag = tag;

    CHK_RET_AND_PRINT_IDE(communicator->LoadOpbasedCollOp(opParams, static_cast<void*>(stream)), tag.c_str());

    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo
            = "HcclAlltoAllVV2:success,take time: " + std::to_string(DURATION_US(endut - startut).count()) + " us,"
              + std::string(stackLogBufferV2);
        communicator->GetTrace().Save(endInfo);
    }

    return HCCL_SUCCESS;
}

// 单算子 创建子通信域 V2
HcclResult HcclCreateSubCommConfigV2(
    const HcclComm* comm, uint32_t rankNum, uint32_t* rankIds, uint64_t subCommId, uint32_t subCommRankId,
    HcclCommConfig* config, HcclComm* subComm)
{
    HcclUs startut = TIME_NOW();

    std::ostringstream printRankIds;
    unordered_set<uint32_t> rankIdSet;
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
                "[GetHcomRankListV2]errNo[0x%016llx], "
                "duplicated rankId[%u] in rankIds.",
                HCCL_ERROR_CODE(HCCL_E_PARA), rankIds[i]),
            HCCL_E_PARA);
        rankIdSet.insert(rankIds[i]);
    }

    HCCL_RUN_INFO(
        "Entry-HcclCreateSubCommConfig V950 rankIds[%s], subCommRankId[%u], commEngine[%u], hcclBufferSize[%u] MB",
        printRankIds.str().c_str(), subCommRankId, config->hcclOpExpansionMode, config->hcclBufferSize);

    HcclCommInfoV2& opbasedCommInfoV2 = GetCommInfoV2();
    if (opbasedCommInfoV2.status == DeviceStatus::DEVICE_RECOVERED) {
        if (opbasedCommInfoV2.hcclGroupMap.find(config->hcclCommName) == opbasedCommInfoV2.hcclGroupMap.end()) {
            HCCL_WARNING("[HcclCommInitClusterInfoConfig] Device was recovered, but communicator is search miss.");
        } else {
            *subComm = static_cast<HcclComm>(opbasedCommInfoV2.hcclGroupMap[config->hcclCommName].pComm.get());
            return HCCL_SUCCESS;
        }
    }
    HCCL_RUN_INFO("Entry-HcclCreateSubCommConfig V950 config->hcclBufferSize[%u] MB", config->hcclBufferSize);

    // 用临时副本承接 config，默认值修正施加在副本上，避免污染调用方入参（重复调用时入参被改写导致报错）
    shared_ptr<HcclCommConfig> hcclConf;
    EXCEPTION_CATCH((hcclConf = make_shared<HcclCommConfig>()), return HCCL_E_PTR);
    CHK_SAFETY_FUNC_RET(memcpy_s(hcclConf.get(), sizeof(HcclCommConfig), config, sizeof(HcclCommConfig)));

    CHK_PRT_RET(
        UNLIKELY(hcclConf->hcclBufferSize == 0),
        HCCL_ERROR("HcclCreateSubCommConfigV2 config: hcclBufferSize is 0 MB, invalid para"), HCCL_E_PARA);
    if (hcclConf->hcclBufferSize == HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET) {
        hcclConf->hcclBufferSize = 0;
        HCCL_INFO("[HcclCreateSubCommConfigV2] set default HCCL BUFFER");
    }
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(*comm);

    string commId = communicator->GetId();
    u8 parentDepth = communicator->GetCommDepth();
    CHK_PRT_RET(
        parentDepth >= MAX_SUB_COMM_DEPTH,
        HCCL_ERROR(
            "[HcclCreateSubCommConfigV2] commId [%s] depth [%u] exceeds max depth [%u]", commId.c_str(),
            static_cast<uint32_t>(parentDepth), static_cast<uint32_t>(MAX_SUB_COMM_DEPTH)),
        HCCL_E_NOT_SUPPORT);

    CHK_PRT_RET(
        opbasedCommInfoV2.pComm == nullptr,
        HCCL_ERROR("[Create][Group]HcclCommInfoV2.pComm is null, please check if the initialize process is called."),
        HCCL_E_PTR);

    std::unique_lock<std::mutex> groupParaLock(opbasedCommInfoV2.groupParamsLock);

    string subCommIdStr(commId + "_sub_" + to_string(subCommId));
    CHK_PRT_RET(
        subCommIdStr.size() >= COMM_NAME_MAX_LENGTH,
        HCCL_ERROR(
            "[HcclCreateSubCommConfigV2] sub comm name[%s] length[%zu] exceeds max[%u]", subCommIdStr.c_str(),
            subCommIdStr.size(), COMM_NAME_MAX_LENGTH - 1),
        HCCL_E_PARA);
    LoadConfigCommName(subCommIdStr, *config);
    if (subCommIdStr == commId) {
        HCCL_ERROR(
            "[HcclCreateSubCommConfigV2] sub comm name[%s] should not be same as the world comm name[%s]",
            subCommIdStr.c_str(), commId.c_str());
        return HCCL_E_INTERNAL;
    }

    /* 已经存在的group不允许再次创建 */
    if (opbasedCommInfoV2.hcclGroupMap.find(subCommIdStr) != opbasedCommInfoV2.hcclGroupMap.end()) {
        HCCL_ERROR(
            "[Create][Group]errNo[0x%016llx] group[%s] is already exist", HCCL_ERROR_CODE(HCCL_E_PARA),
            subCommIdStr.c_str());
        return HCCL_E_PARA;
    }

    /* 创建groupParamsV2Tem */
    HcclGroupParamsV2 groupParamsV2Tem;
    CHK_RET(GetHcomRankListV2(rankNum, rankIds, groupParamsV2Tem, *comm));

    /* 如果是groupRank = INVALID_VALUE_RANKID，即本rank不参与create group */
    if (groupParamsV2Tem.groupRank == INVALID_VALUE_RANKID) {
        HCCL_ERROR(
            "[Create][Group]errNo[0x%016llx] confirm groupRank from worldRank[%d] error",
            HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), opbasedCommInfoV2.commParams.myRank);
        return HCCL_E_NOT_FOUND;
    }

    /* 创建子通信域 */
    uint32_t parentRank = 0;
    CHK_RET(communicator->GetRankId(parentRank));
    Hccl::CommParams commParams{
        subCommIdStr, static_cast<Hccl::RankId>(subCommRankId), rankNum, static_cast<Hccl::RankId>(parentRank),
        opbasedCommInfoV2.commParams.devType};
    commParams.commDepth = parentDepth + 1;
    CheckHcclDeterministic(hcclConf->hcclDeterministic);
    // 默认全都开启确定性计算
    hcclConf->hcclDeterministic = 1;
    std::shared_ptr<Hccl::HcclCommunicator> subCommunicator
        = make_shared<Hccl::HcclCommunicator>(commParams, hcclConf.get());

    std::vector<u32> rankIdsVec(rankNum);
    for (uint32_t i = 0; i < rankNum; ++i) {
        rankIdsVec[i] = rankIds[i];
    }
    /* --------------初始化------------------------- */
    HcclResult ret = HCCL_SUCCESS;
    bool errorFlag = false;
    s32 userDevId = HrtGetDevice();
    s32 devPhyId = HrtGetDevicePhyIdByUserDevId(userDevId);
    do {
        ret = communicator->CreateSubComm(commParams, rankIdsVec, subCommunicator, *hcclConf);
        CHK_PRT_BREAK(
            ret != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[%s]communicator->CreateSubComm failed, errNo[0x%016llx]", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        CHK_SMART_PTR_NULL(subCommunicator);

        subCommunicator->RegisterAcceStateCallBack(CommunicatorCallback());

        groupParamsV2Tem.pComm = subCommunicator;

        opbasedCommInfoV2.hcclGroupMap.insert(std::make_pair(subCommIdStr, groupParamsV2Tem));
        groupParaLock.unlock();

        ret = CommManager::GetInstance(userDevId).SetCommAcceleratorV2(
            subCommunicator.get(), config->hcclOpExpansionMode); // 通信域创建，设置默认accelerator
        CHK_PRT_BREAK(
            ret != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[%s]SetCommAcceleratorV2 failed, errNo[0x%016llx]", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        *subComm = subCommunicator.get();
    } while (0);

    if (errorFlag) {
        groupParaLock.unlock();
        HCCL_ERROR(
            "[Init][%s]HcclCreateSubCommConfigV2 failed, userDevId[%d], devPhyId[%d], sub comm[%s], world comm[%s]"
            "return[0x%016llx]",
            __func__, userDevId, devPhyId, subCommIdStr.c_str(), commId.c_str(), HCCL_ERROR_CODE(ret));
        (void)HcclCommDestroyV2(subCommunicator.get());
        *subComm = nullptr;
        return ret;
    }
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "[Create][Group]create group[%s] success, userDevId[%d], devPhyId[%d], take time [%lld]us",
        subCommIdStr.c_str(), userDevId, devPhyId, DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankIdV2(HcclComm comm, uint32_t* rank)
{
    HCCL_RUN_INFO("Entry-HcclGetRankId V950");
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    auto ret = communicator->GetRankId(*rank);
    if (ret != HcclResult::HCCL_SUCCESS) {
        return HCCL_E_INTERNAL;
    }
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "Entry-HcclGetRankId V950 success, comm[%s], rankIdPtr[%p], rankId[%u]", communicator->GetId().c_str(), rank,
        *rank);
    return HCCL_SUCCESS;
}

HcclResult HcclGetCommNameV2(HcclComm commHandle, char* commName)
{
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(commHandle);
    if (communicator == nullptr) {
        HCCL_ERROR("HcclGetCommNameV2 communicator is nullptr");
        return HCCL_E_PTR;
    }
    HCCL_INFO(
        "HcclGetCommNameV2 commId[%s], commId size[%zu], input commName ptr=[%p]", communicator->GetId().c_str(),
        communicator->GetId().size(), commName);
    s32 ret = strncpy_s(
        commName, ROOTINFO_INDENTIFIER_MAX_LENGTH, communicator->GetId().c_str(), communicator->GetId().size() + 1);
    CHK_PRT_RET(
        ret != EOK,
        HCCL_ERROR("HcclGetCommName str copy fail. return[%d], commId[%s]", ret, communicator->GetId().c_str()),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankSizeV2(HcclComm comm, uint32_t* rankSize)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(rankSize);
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    s32 userDevId = HcclGetThreadDeviceId();
    s32 devPhyId = HrtGetDevicePhyIdByUserDevId(userDevId);
    HCCL_RUN_INFO(
        "Entry-HcclGetRankSize V950, commId[%s], userDevId[%d], devPhyId[%d]", communicator->GetId().c_str(), userDevId,
        devPhyId);
    auto ret = communicator->GetRankSize(rankSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HcclGetRankSizeV2 failed, rankSize[%u], commId[%s]", *rankSize, communicator->GetId().c_str());
        return HCCL_E_INTERNAL;
    }
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "Entry-HcclGetRankSize V950 success, comm[%s], rankSizePtr[%p], rankSize[%u]", communicator->GetId().c_str(),
        rankSize, *rankSize);
    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAllVCV2(
    const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType,
    HcclComm comm, rtStream_t stream)
{
    HcclUs startut = TIME_NOW();
    bool isCapture;
    rtModel_t rtModel = nullptr;
    u32 modelId = 0;
    CHK_RET(GetStreamCaptureInfo(stream, rtModel, isCapture));

    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    const std::string tag = "HCCL_ALLTOALLVC_" + communicator->GetId();

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), 0, sendType, stream), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckDataTypeV2(recvType), tag.c_str());
    u32 rankSize = 0;
    CHK_RET(communicator->GetRankSize(&rankSize));
    u32 myRank = INVALID_VALUE_RANKID;
    CHK_RET(communicator->GetRankId(myRank));
    bool isEmpty = false;
    CHK_RET(HcomCheckAlltoAllVCEmptyV2(sendBuf, sendCountMatrix, recvBuf, rankSize, isEmpty));
    if (isEmpty) {
        HCCL_INFO("[HcclAlltoAllVCV2] sendCountMatrix is Empty");
        return HCCL_SUCCESS;
    }
    CHK_RET(HcomCheckAlltoAllVCExternalMemV2(sendBuf, sendCountMatrix, recvBuf, rankSize, myRank));

    u64 sendCountMatrixHash;
    HcomGetHashFromSendCountMatrixV2(sendCountMatrixHash, sendCountMatrix, rankSize, tag);
    /* 接口交互信息日志 */
    char stackLogBufferV2[LOG_TMPBUF_SIZE];
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        s32 streamId = HrtGetStreamId(stream);
        s32 deviceLogicId = HrtGetDevice();
        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(communicator->GetRankId(localRank), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBufferV2, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], sendCountMatrixHash[%llu], sendType[%s], recvBuf[%p],"
            "recvType[%s], localRank[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, sendCountMatrixHash, GetDataTypeEnumStrV2(sendType).c_str(), recvBuf,
            GetDataTypeEnumStrV2(recvType).c_str(), localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclAlltoAllVCV2:" + std::string(stackLogBufferV2);
        if (isCapture) {
            CHK_PTR_NULL(rtModel);
            // 获取不到modelId会报错
            CHK_RET(GetModelId(rtModel, modelId));
            logInfo += ", model id[" + to_string(modelId) + "].";
        }
        communicator->GetTrace().Save(logInfo);
    }

    static thread_local Hccl::CollOpParams opParams;
    opParams.opType = Hccl::OpType::ALLTOALLVC;
    opParams.sendBuf = const_cast<void*>(sendBuf);
    opParams.recvBuf = const_cast<void*>(recvBuf);
    opParams.all2AllVCDataDes.sendCountMatrix = const_cast<void*>(sendCountMatrix);
    opParams.all2AllVCDataDes.sendType = HcclDataTypeToDataType(sendType);
    opParams.all2AllVCDataDes.recvType = HcclDataTypeToDataType(recvType);
    opParams.dataType = HcclDataTypeToDataType(sendType);
    CHK_RET_AND_PRINT_IDE(communicator->LoadOpbasedCollOp(opParams, static_cast<void*>(stream)), tag.c_str());

    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo
            = "HcclAlltoAllVCV2:success,take time: " + std::to_string(DURATION_US(endut - startut).count()) + " us,"
              + std::string(stackLogBufferV2);
        communicator->GetTrace().Save(endInfo);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclReduceV2(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint32_t root, HcclComm comm,
    aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    bool isCapture;
    rtModel_t rtModel = nullptr;
    u32 modelId = 0;

    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    const std::string tag = "Reduce_" + communicator->GetId();

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), count, dataType, stream), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckReductionOpV2(op), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckReduceDataTypeV2(dataType, op), tag.c_str());
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(communicator->GetRankSize(&rankSize), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRankV2(rankSize, root), tag.c_str());
    CHK_RET(GetStreamCaptureInfo(stream, rtModel, isCapture));

    /* 接口交互信息日志 */
    char stackLogBufferV2[LOG_TMPBUF_SIZE];
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        s32 streamId = HrtGetStreamId(stream);
        s32 deviceLogicId = HrtGetDevice();
        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(communicator->GetRankId(localRank), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBufferV2, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], count[%llu], dataType[%s], op[%s], root[%u],"
            "localRank[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, count, GetDataTypeEnumStrV2(dataType).c_str(),
            GetReduceOpEnumStrV2(op).c_str(), root, localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclReduceV2:" + std::string(stackLogBufferV2);
        if (isCapture) {
            CHK_PTR_NULL(rtModel);
            // 获取不到modelId会报错
            CHK_RET(GetModelId(rtModel, modelId));
            logInfo += ", model id[" + to_string(modelId) + "].";
        }
        communicator->GetTrace().Save(logInfo);
    }

    static thread_local Hccl::CollOpParams opParams;
    opParams.opType = Hccl::OpType::REDUCE;
    opParams.dataType = HcclDataTypeToDataType(dataType);
    opParams.reduceOp = HCCL_OP_REDUCE_MAP[op];
    opParams.sendBuf = sendBuf;
    opParams.recvBuf = recvBuf;
    opParams.count = count;
    opParams.root = root;
    opParams.opTag = tag;
    CHK_RET_AND_PRINT_IDE(communicator->LoadOpbasedCollOp(opParams, static_cast<void*>(stream)), tag.c_str());

    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo = "HcclReduceV2:success,take time: " + std::to_string(DURATION_US(endut - startut).count())
                              + " us," + std::string(stackLogBufferV2);
        communicator->GetTrace().Save(endInfo);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclAllReduceV2(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    bool isCapture;
    rtModel_t rtModel = nullptr;
    u32 modelId = 0;

    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    const std::string tag = "AllReduce_" + communicator->GetId();

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), count, dataType, stream), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckReductionOpV2(op), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckReduceDataTypeV2(dataType, op), tag.c_str());
    CHK_RET(GetStreamCaptureInfo(stream, rtModel, isCapture));

    /* 接口交互信息日志 */
    char stackLogBufferV2[LOG_TMPBUF_SIZE];
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        s32 streamId = HrtGetStreamId(stream);
        s32 deviceLogicId = HrtGetDevice();
        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(communicator->GetRankId(localRank), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBufferV2, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], count[%llu], dataType[%s], op[%s], localRank[%u], streamId[%d],"
            "comm[%p], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, count, GetDataTypeEnumStrV2(dataType).c_str(),
            GetReduceOpEnumStrV2(op).c_str(), localRank, streamId, comm, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));

        std::string logInfo = "Entry-HcclAllReduceV2: " + std::string(stackLogBufferV2);
        if (isCapture) {
            CHK_PTR_NULL(rtModel);
            // 获取不到modelId会报错
            CHK_RET(GetModelId(rtModel, modelId));
            logInfo += ", model id[" + to_string(modelId) + "].";
        }
        communicator->GetTrace().Save(logInfo);
    }

    static thread_local Hccl::CollOpParams opParams;
    opParams.opType = Hccl::OpType::ALLREDUCE;
    opParams.dataType = HcclDataTypeToDataType(dataType);
    opParams.reduceOp = HCCL_OP_REDUCE_MAP[op];
    opParams.sendBuf = sendBuf;
    opParams.recvBuf = recvBuf;
    opParams.count = count;
    opParams.opTag = tag;
    CHK_RET_AND_PRINT_IDE(communicator->LoadOpbasedCollOp(opParams, static_cast<void*>(stream)), tag.c_str());

    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo
            = "HcclAllReduceV2:success,take time: " + std::to_string(DURATION_US(endut - startut).count()) + " us,"
              + std::string(stackLogBufferV2);
        communicator->GetTrace().Save(endInfo);
    }

    return HCCL_SUCCESS;
}

HcclResult
HcclBroadcastV2(void* buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    bool isCapture;
    rtModel_t rtModel = nullptr;
    u32 modelId = 0;

    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    const std::string tag = "Broadcast_" + communicator->GetId();

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), count, dataType, stream), tag.c_str());
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(communicator->GetRankSize(&rankSize), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRankV2(rankSize, root), tag.c_str());
    CHK_RET(GetStreamCaptureInfo(stream, rtModel, isCapture));

    /* 接口交互信息日志 */
    char stackLogBufferV2[LOG_TMPBUF_SIZE];
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        s32 streamId = HrtGetStreamId(stream);
        s32 deviceLogicId = HrtGetDevice();
        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(communicator->GetRankId(localRank), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBufferV2, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], buf[%p], count[%llu], dataType[%s], root[%u], localRank[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), buf, count, GetDataTypeEnumStrV2(dataType).c_str(), root, localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclBroadcastV2:" + std::string(stackLogBufferV2);
        if (isCapture) {
            CHK_PTR_NULL(rtModel);
            // 获取不到modelId会报错
            CHK_RET(GetModelId(rtModel, modelId));
            logInfo += ", model id[" + to_string(modelId) + "].";
        }
        communicator->GetTrace().Save(logInfo);
    }

    static thread_local Hccl::CollOpParams opParams;
    opParams.opType = Hccl::OpType::BROADCAST;
    opParams.dataType = HcclDataTypeToDataType(dataType);
    opParams.sendBuf = buf;
    opParams.recvBuf = buf;
    opParams.count = count;
    opParams.root = root;
    opParams.opTag = tag;
    CHK_RET_AND_PRINT_IDE(communicator->LoadOpbasedCollOp(opParams, static_cast<void*>(stream)), tag.c_str());
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo
            = "HcclBroadcastV2:success,take time: " + std::to_string(DURATION_US(endut - startut).count()) + " us,"
              + std::string(stackLogBufferV2);
        communicator->GetTrace().Save(endInfo);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclBarrierV2(HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    HCCL_INFO("HcclBarrierV2 V82");
    // AllReduce入参定义
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    const uint64_t count = 8;
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    s32 userDevId = HcclGetThreadDeviceId();
    s32 devPhyId = HrtGetDevicePhyIdByUserDevId(userDevId);
    HCCL_RUN_INFO(
        "Entry-HcclBarrier V950, commId[%s], userDevId[%d], devPhyId[%d]", communicator->GetId().c_str(), userDevId,
        devPhyId);
    // 申请Device内存
    auto ret = communicator->CreateBarrierMemory(sendBuf, recvBuf, count);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    // 同通信域同算子复用tag
    const string tag = "AllReduce_" + communicator->GetId();

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), count, dataType, stream), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReductionOpV2(op), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReduceDataTypeV2(dataType, op), tag.c_str());

    static thread_local Hccl::CollOpParams opParams;
    opParams.opType = Hccl::OpType::ALLREDUCE;
    opParams.dataType = HcclDataTypeToDataType(dataType);
    opParams.reduceOp = Hccl::ReduceOp::SUM;

    opParams.sendBuf = sendBuf;
    opParams.recvBuf = recvBuf;
    opParams.count = count;
    opParams.opTag = tag;
    ret = communicator->LoadOpbasedCollOp(opParams, static_cast<void*>(stream));
    HCCL_RUN_INFO(
        "Entry-HcclBarrier V950 success, take time [%lld]us, commId[%s], userDevId[%d]",
        DURATION_US(TIME_NOW() - startut), communicator->GetId().c_str(), userDevId);
    return ret;
}

HcclResult HcclGetHeterogModeV2(HcclComm comm, HcclHeterogMode* mode)
{
    (void)comm;
    *mode = HCCL_HETEROG_MODE_HOMOGENEOUS;
    HCCL_INFO("[HcclGetHeterogModeV2] 950 only support homogeneous chip mode");
    return HCCL_SUCCESS;
}

HcclResult HcclCommSuspendV2(HcclComm comm)
{
    CHK_PTR_NULL(comm);
    HCCL_ERROR("HcclCommSuspend V950 not support suspend");

    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclAllocComResourceByTilingV2(HcclComm comm, const void* stream, void* mc2Tiling, void** commContext)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(mc2Tiling);
    CHK_PTR_NULL(commContext);
    HcclUs startut = TIME_NOW();
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);

    uint32_t* pVersion = reinterpret_cast<uint32_t*>(mc2Tiling);
    u32 localRank = INVALID_VALUE_RANKID;
    char stackLogBufferV2[LOG_TMPBUF_SIZE];
    string commIdentifier = communicator->GetId();
    const std::string tag = "HcclAllocComResourceByTilingV2_" + communicator->GetId();
    CHK_RET_AND_PRINT_IDE(communicator->GetRankId(localRank), tag.c_str());
    /* 接口交互信息日志 */
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        s32 ret = snprintf_s(
            stackLogBufferV2, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U, "commIdentifier[%s], version[%u]",
            commIdentifier.c_str(), *pVersion);
        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, commIdentifier[%s].", commIdentifier.c_str()));

        std::string logInfo = "MC2 create resource by tiling: localRank[" + std::to_string(localRank) + "]"
                              + std::string(stackLogBufferV2);
        communicator->GetTrace().Save(logInfo);
    }

    HcclResult ret = communicator->AllocCommResource(mc2Tiling, commContext);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcclAllocComResourceByTilingV2]AllocCommResource fail, errNo[%d], commIdentifier[%s]", ret,
            commIdentifier.c_str()),
        ret);

    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo = "MC2 create resource take time [" + std::to_string(DURATION_US(endut - startut).count())
                              + "]us, localRank[" + std::to_string(localRank) + "] " + std::string(stackLogBufferV2);
        communicator->GetTrace().Save(endInfo);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclGetOpArgsV2(void** opArgs)
{
    CHK_PTR_NULL(opArgs);
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HCCL_RUN_INFO("Entry-HcclGetOpArgs V950, start malloc opArgs in %p", opArgs);
    }
    HcclOpArgs* opArgsMem = (HcclOpArgs*)malloc(sizeof(HcclOpArgs));
    if (opArgsMem == nullptr) {
        HCCL_ERROR("[HcclGetOpArgs] malloc HcclOpArgs mem fail, please check.");
        return HCCL_E_INTERNAL;
    }
    opArgsMem->Init();
    *opArgs = opArgsMem;
    HCCL_RUN_INFO("HcclGetOpArgs malloc HcclOpArgs success, please fill mem[%p->%p] in it.", opArgs, *opArgs);
    return HCCL_SUCCESS;
}

HcclResult HcclFreeOpArgsV2(void* opArgs)
{
    CHK_PTR_NULL(opArgs);
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HCCL_RUN_INFO("Entry-HcclFreeOpArgs V950, free opArgs[%p]", opArgs);
    }
    free(opArgs);
    opArgs = nullptr;
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpSrcDataTypeV2(void* opArgs, uint8_t srcDataType)
{
    CHK_PTR_NULL(opArgs);
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HCCL_RUN_INFO("Entry-HcclSetOpSrcDataType V950, opArgs[%p] set srcDataType[%u]", opArgs, srcDataType);
    }
    HcclOpArgs* opArgsPtr = static_cast<HcclOpArgs*>(opArgs);
    if (srcDataType >= (sizeof(MC2_DATA_TYPE) / sizeof(MC2_DATA_TYPE[0]))) {
        HCCL_ERROR("HcclSetOpSrcDataType set srcDataType[%u] error, it's invalid.", srcDataType);
        return HCCL_E_PARA;
    }
    opArgsPtr->srcDataType = MC2_DATA_TYPE[srcDataType];
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpDstDataTypeV2(void* opArgs, uint8_t dstDataType)
{
    CHK_PTR_NULL(opArgs);
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HCCL_RUN_INFO("Entry-HcclSetOpDstDataType V950, opArgs[%p] set dstDataType[%u]", opArgs, dstDataType);
    }
    HcclOpArgs* opArgsPtr = static_cast<HcclOpArgs*>(opArgs);
    if (dstDataType >= (sizeof(MC2_DATA_TYPE) / sizeof(MC2_DATA_TYPE[0]))) {
        HCCL_ERROR("HcclSetOpDstDataType set dstDataType[%u] error, it's invalid.", dstDataType);
        return HCCL_E_PARA;
    }
    opArgsPtr->dstDataType = MC2_DATA_TYPE[dstDataType];
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpReduceTypeV2(void* opArgs, uint32_t reduceType)
{
    CHK_PTR_NULL(opArgs);
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HCCL_RUN_INFO("Entry-HcclSetOpReduceType V950, opArgs[%p] set reduceType[%u]", opArgs, reduceType);
    }
    HcclOpArgs* opArgsPtr = static_cast<HcclOpArgs*>(opArgs);
    if (reduceType >= (sizeof(MC2_REDUCE_TYPE) / sizeof(MC2_REDUCE_TYPE[0]))) {
        HCCL_ERROR("HcclSetOpReduceType set reduceType[%u] error, it's invalid.", reduceType);
        return HCCL_E_PARA;
    }
    opArgsPtr->reduceType = MC2_REDUCE_TYPE[reduceType];
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpCountV2(void* opArgs, uint64_t count)
{
    CHK_PTR_NULL(opArgs);
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HCCL_RUN_INFO("Entry-HcclSetOpCount V950, opArgs[%p] set count[%llu]", opArgs, count);
    }
    HcclOpArgs* opArgsPtr = static_cast<HcclOpArgs*>(opArgs);
    CHK_RET(HcomCheckCountV2(count));
    opArgsPtr->count = count;
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpAlgConfigV2(void* opArgs, char* algConfig)
{
    CHK_PTR_NULL(opArgs);
    CHK_PTR_NULL(algConfig);
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HCCL_RUN_INFO("Entry-HcclSetOpAlgConfig V950, opArgs[%p]", opArgs);
    }
    HcclOpArgs* opArgsPtr = static_cast<HcclOpArgs*>(opArgs);
    s32 ret = strcpy_s(opArgsPtr->algConfig, ALG_CONFIG_SIZE, algConfig);
    if (ret != EOK) {
        HCCL_ERROR(
            "[HcclSetOpAlgConfig]strcpy_s algConfig failed! result %d, the algConfig len must be less than %u", ret,
            ALG_CONFIG_SIZE);
        return HCCL_E_PARA;
    }
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HCCL_RUN_INFO(
            "HcclSetOpAlgConfig set opArgs[%p] algConfig success, algConfig is [%s]", opArgs, opArgsPtr->algConfig);
    }
    return HCCL_SUCCESS;
};

HcclResult HcclSetOpCommEngineV2(void* opArgs, uint8_t commEngine)
{
    CHK_PTR_NULL(opArgs);
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HCCL_RUN_INFO("Entry-HcclSetOpCommEngine V950, commEngine[%u]", commEngine);
    }
    HcclOpArgs* opArgsPtr = static_cast<HcclOpArgs*>(opArgs);
    opArgsPtr->commEngine = HcclAccelerator(static_cast<HcclAccelerator::Value>(commEngine));
    return HCCL_SUCCESS;
}

HcclResult HcclCommResPrepareWithOpMode(
    Hccl::HcclCommunicator* communicator, const std::string& opName, HcclOpArgs* opArgs, void** addr)
{
    CHK_PTR_NULL(communicator);
    CHK_PTR_NULL(opArgs);
    CHK_PTR_NULL(addr);
    OpType opType = GetOpTypeV2(opName);
    if (opType == OpType::OPTYPEINVALID) {
        HCCL_ERROR(
            "[HcclCommResPrepareWithOpMode] The opName %s match opType is %s", opName.c_str(),
            opType.Describe().c_str());
        return HCCL_E_PARA;
    }

    std::string opTag = opName + communicator->GetId() + "_mc2";
    static thread_local Hccl::CollOpParams opParams;
    opParams.opType = opType;
    opParams.reduceOp = opArgs->reduceType;
    opParams.dataType = opArgs->srcDataType;
    opParams.outputDataType = opArgs->dstDataType;
    opParams.count = opArgs->count;
    opParams.opTag = opTag;
    opParams.algConfig = std::string(opArgs->algConfig);
    opParams.isMc2 = true;
    opParams.commEngine = opArgs->commEngine;
    return communicator->AllocCollOpResource(opParams, addr);
}

HcclResult HcclCommResPrepareV2(HcclComm comm, char* opName, void* opArgs, void** addr)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(opName);
    CHK_PTR_NULL(opArgs);
    CHK_PTR_NULL(addr);
    std::string opNameStr(opName);
    opNameStr = opNameStr.substr(0, MAX_OP_NAME_SIZE);
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HCCL_RUN_INFO(
            "Entry-HcclCommResPrepare V950, opName[%s], opArgs addr[%p], commId[%s]", opNameStr.c_str(), opArgs,
            communicator->GetId().c_str());
    }
    return HcclCommResPrepareWithOpMode(communicator, opNameStr, static_cast<HcclOpArgs*>(opArgs), addr);
}

HcclResult HcclDevMemAcquireV2(HcclComm comm, const char* memTag, uint64_t* size, void** addr, bool* newCreated)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(memTag);
    CHK_PTR_NULL(size);
    CHK_PTR_NULL(addr);
    CHK_PTR_NULL(newCreated);
    std::string memTagStr = "";
    if (memTag != nullptr) {
        char tmpMemTag[MAX_MEM_TAG_SIZE];
        s32 ret = strcpy_s(tmpMemTag, MAX_MEM_TAG_SIZE, memTag);
        if (ret != EOK) {
            HCCL_ERROR(
                "[HcclDevMemAcquire] strcpy_s memTag failed! result %d, the memTag len must be less than %u", ret,
                MAX_MEM_TAG_SIZE);
            return HCCL_E_PARA;
        }
        memTagStr = std::string(tmpMemTag);
    }
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    CHK_RET(communicator->GetDevMemWorkSpace(memTagStr, size, addr, newCreated));
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HCCL_RUN_INFO(
            "Entry-HcclDevMemAcquire V950, memTag[%s], addr[%p], size[%llu], commId[%s]", memTagStr.c_str(), *addr,
            *size, communicator->GetId().c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetHcclBufferV2(HcclComm comm, void** addr, uint64_t* size)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(addr);
    CHK_PTR_NULL(size);
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HCCL_RUN_INFO("Entry-HcclGetHcclBuffer V950, commId[%s]", communicator->GetId().c_str());
    }
    CHK_RET(communicator->GetLocalCclBuffer(addr, size));
    /* 关键状态记录 */
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HCCL_RUN_INFO(
            "HcclGetHcclBuffer success, addr[%p], size[%llu], commId[%s]", addr, *size, communicator->GetId().c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetRemoteIpcHcclBufV2(HcclComm comm, uint64_t remoteRank, void** addr, uint64_t* size)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(addr);
    CHK_PTR_NULL(size);
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HCCL_RUN_INFO(
            "Entry-HcclGetRemoteIpcHcclBuf V950 start, remoteRank[%llu], addr[%p], size[%llu], commId[%s]", remoteRank,
            *addr, *size, communicator->GetId().c_str());
    }
    HCCL_ERROR("Entry-HcclGetRemoteIpcHcclBuf V950 not support, commId[%s]", communicator->GetId().c_str());
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclGetAicpuOpStreamAndNotifyV2(HcclComm comm, rtStream_t* opstream, u8 aicpuNotifyNum, void** aicpuNotify)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(opstream);
    CHK_PTR_NULL(aicpuNotify);
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HCCL_RUN_INFO(
            "Entry-HcclGetAicpuOpStreamAndNotify V950, aicpuNotifyNum[%u], commId[%s]", aicpuNotifyNum,
            communicator->GetId().c_str());
    }
    CHK_RET(communicator->GetAicpuOpStreamNotify(opstream, aicpuNotifyNum, aicpuNotify));
    return HCCL_SUCCESS;
}

HcclResult HcclCommResumeV2(HcclComm comm)
{
    CHK_PTR_NULL(comm);
    HcclUs startut = TIME_NOW();
    HCCL_RUN_INFO("Entry-HcclCommResume V950");
    s32 userDevId = HcclGetThreadDeviceId();
    s32 devPhyId = HrtGetDevicePhyIdByUserDevId(userDevId);
    CHK_RET(static_cast<HcclResult>(Hccl::HcclCcuResumePfeTableProcess(userDevId)));
    CHK_RET(HcclCommResumeImplV2(comm));
    HCCL_RUN_INFO(
        "Entry-HcclCommResume V950 success, userDevId[%d], devPhyId[%d], take time [%lld]us", userDevId, devPhyId,
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult HcclCommResumeImplV2(HcclComm comm)
{
    CHK_PTR_NULL(comm);
    return HcclCommOperationImplV2(comm, "HcclCommResumeV2", [](Hccl::HcclCommunicator& communicator) {
        HCCL_RUN_INFO("Entry-HcclCommResume commId[%s]", communicator.GetId().c_str());
        return static_cast<HcclResult>(communicator.Resume());
    });
}

HcclResult RootInfoDetect(
    const u32 nRanks, u32 rank, const HcclRootHandleV2& rootHandle, RankTableInfo& rankTable,
    RootInfoDetectBridge::DetectContext& detectContext)
{
    const RootInfoDetectBridge* bridge = GetRootInfoDetectBridge();
    CHK_PRT_RET(
        bridge == nullptr || bridge->detectRankTable == nullptr,
        HCCL_ERROR("[%s] RootInfoDetect bridge is not registered.", __func__), HCCL_E_INTERNAL);
    return bridge->detectRankTable(nRanks, rank, rootHandle, rankTable, detectContext);
}

HcclResult
CommInitRootInfo(u32 nRanks, u32 rank, const HcclRootHandleV2& rootHandle, const string& identifier, HcclComm* comm)
{
    // 临时规避，在初始化通信域前声明单例保证时序
    CHK_RET(CallSingletons());
    // check
    HcclCommInfoV2& opbasedCommInfoV2 = GetCommInfoV2();
    bool commExists = opbasedCommInfoV2.hcclGroupMap.find(identifier) != opbasedCommInfoV2.hcclGroupMap.end();
    RPT_INPUT_ERR(
        commExists, "EI0003", vector<string>({"ccl_op", "value", "parameter", "expect"}),
        vector<string>({"HcclCommInitRootInfo", identifier, "identifier", "an identifier that is not in use"}));
    CHK_PRT_RET(
        commExists,
        HCCL_ERROR(
            "[CreateCommConfig] errNo[0x%016llx] The rootHandle[%s] already exists in Group2Comm map.",
            HCCL_ERROR_CODE(HCCL_E_PARA), identifier.c_str()),
        HCCL_E_PARA);

    // rootInfo获取rankTable, 基于rankTable创建通信域
    RootInfoDetectBridge::DetectContext rootInfoDetectContext;
    RankTableInfo rankTable{};
    HcclResult ret = RootInfoDetect(nRanks, rank, rootHandle, rankTable, rootInfoDetectContext);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] errNo[0x%016llx] RootInfoDetect failed, rootHandle[%s].", __func__, HCCL_ERROR_CODE(ret),
            identifier.c_str());
        rankTable.Dump();
        return ret;
    }

    // 打印ranktable
    rankTable.Dump();

    // 创建通信域
    bool devUsed = false;
    bool isWorldGroup = true;
    Hccl::CommParams commParams{
        identifier,
        static_cast<Hccl::RankId>(rank),
        nRanks,
        static_cast<Hccl::RankId>(rank),
        Hccl::HrtGetDeviceType(),
        devUsed,
        isWorldGroup};
    opbasedCommInfoV2.pComm.reset(new (std::nothrow) Hccl::HcclCommunicator(commParams));
    opbasedCommInfoV2.commParams = commParams;

    // 通信域初始化
    CHK_PTR_NULL(opbasedCommInfoV2.pComm);
    /* --------------初始化------------------------- */
    bool errorFlag = false;
    s32 userDevId = HrtGetDevice();
    do {
        ret = opbasedCommInfoV2.pComm->Init(rankTable);
        CHK_PRT_BREAK(
            ret != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[%s]opbasedCommInfoV2.pComm->Init failed, errNo[0x%016llx]", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        // 配置默认加速模式
        opbasedCommInfoV2.pComm->RegisterAcceStateCallBack(CommunicatorCallback());
        ret = CommManager::GetInstance(userDevId).SetCommAcceleratorV2(
            opbasedCommInfoV2.pComm.get(), 0); // 通信域创建，设置默认accelerator
        CHK_PRT_BREAK(
            ret != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[%s]SetCommAcceleratorV2 failed, errNo[0x%016llx]", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        // 保存通信域
        HcclGroupParamsV2 params{};
        params.pComm = opbasedCommInfoV2.pComm;
        params.groupRank = static_cast<Hccl::RankId>(rank);
        std::unique_lock<std::mutex> lock(opbasedCommInfoV2.groupParamsLock);
        opbasedCommInfoV2.hcclGroupMap[identifier] = params;

        opbasedCommInfoV2.pComm->RegisterPrintChannelInfoCallback(
            CommManager::GetInstance(userDevId).GetPrintChannelInfoCallback());

        *comm = static_cast<HcclComm>(opbasedCommInfoV2.pComm.get());
    } while (0);

    if (errorFlag) {
        HCCL_ERROR(
            "[Init][%s]HcclCommInitClusterInfoV2 failed, rankNum[%u], rank[%u], userDevId[%d], rootInfo "
            "identifier[%s],"
            "return[0x%016llx]",
            __func__, nRanks, rank, userDevId, identifier.c_str(), HCCL_ERROR_CODE(ret));
        (void)HcclCommDestroyV2(opbasedCommInfoV2.pComm.get());
        *comm = nullptr;
        return ret;
    }
    HCCL_INFO(
        "[%s] Init success, rankNum[%u], rank[%u], rootInfo identifier[%s], userDevId[%d]", __func__, nRanks, rank,
        identifier.c_str(), userDevId);

    return HCCL_SUCCESS;
}

HcclResult HcclCommInitRootInfoV2(
    uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm, std::string& identifier)
{
    HcclUs startut = TIME_NOW();
    CHK_PTR_NULL(rootInfo);
    HCCL_RUN_INFO("Entry-HcclCommInitRootInfo V950, rankId[%u], rankNum[%u].", rank, nRanks);

    // 获取rootHandle
    HcclRootHandleV2 rootHandle{};
    s32 sRet = memcpy_s(&rootHandle, sizeof(HcclRootHandleV2), rootInfo->internal, sizeof(HcclRootHandleV2));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR("[%s] memcpy root info fail. errorno[%d] length[%u]", __func__, sRet, sizeof(rootHandle)),
        HCCL_E_MEMORY);

    // 获取通信域name
    rootHandle.identifier[ROOTINFO_INDENTIFIER_MAX_LENGTH - 1] = '\0';
    identifier = rootHandle.identifier;

    /* 接口交互信息日志 */
    s32 userDevId = HcclGetThreadDeviceId();
    s32 devPhyId = HrtGetDevicePhyIdByUserDevId(userDevId);
    HCCL_RUN_INFO(
        "Entry-HcclCommInitRootInfo V950, ranks[%u], rank[%u], rootinfo: host ip[%s] port[%u] "
        "netMode[%s] identifier[%s], userDevId[%d], devPhyId[%d]",
        nRanks, rank, rootHandle.ip, rootHandle.listenPort, rootHandle.netMode.Describe().c_str(), identifier.c_str(),
        userDevId, devPhyId);

    // rootInfo获取rankTable, 基于rankTable创建通信域
    HcclResult ret = CommInitRootInfo(nRanks, rank, rootHandle, identifier, comm);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] errNo[0x%016llx] CommInitRootInfo failed, identifier[%s].", __func__, HCCL_ERROR_CODE(ret),
            identifier.c_str()),
        ret);

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "HcclCommInitRootInfoV2 success, take time [%lld]us, rankNum[%u], rank[%u], identifier[%s]",
        DURATION_US(TIME_NOW() - startut), nRanks, rank, identifier.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitRootInfoConfigV2(
    uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, const HcclCommConfig* config, HcclComm* comm)
{
    HcclUs startut = TIME_NOW();
    CHK_PTR_NULL(rootInfo);
    CHK_PTR_NULL(config);
    HCCL_RUN_INFO(
        "Entry-HcclCommInitRootInfoConfig V950: nRanks[%u], rank[%u], commEngine[%s]", nRanks, rank,
        HcclAccelerator(static_cast<HcclAccelerator::Value>(config->hcclOpExpansionMode)).Describe().c_str());
    // 获取rootHandle
    HcclRootHandleV2 rootHandle{};
    s32 sRet = memcpy_s(&rootHandle, sizeof(rootHandle), rootInfo->internal, sizeof(rootHandle));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR("[%s]memcpy root info fail. errorno[%d] count[%u]", __func__, sRet, sizeof(HcclRootHandleV2)),
        HCCL_E_MEMORY);

    // 获取通信域name
    rootHandle.identifier[ROOTINFO_INDENTIFIER_MAX_LENGTH - 1] = '\0';
    string identifier = strlen(config->hcclCommName) != 0 ? config->hcclCommName : rootHandle.identifier;

    /* 接口交互信息日志 */
    HCCL_RUN_INFO(
        "Entry-HcclCommInitRootInfoConfigV2:ranks[%u], rank[%u], rootinfo: host ip[%s] port[%u] "
        "netMode[%s] rootHandle.identifier[%s], identifier[%s]",
        nRanks, rank, rootHandle.ip, rootHandle.listenPort, rootHandle.netMode.Describe().c_str(),
        rootHandle.identifier, identifier.c_str());

    // 临时规避，在初始化通信域前声明单例保证时序
    CHK_RET(CallSingletons());
    HcclCommInfoV2& opbasedCommInfoV2 = GetCommInfoV2();
    bool commExists = opbasedCommInfoV2.hcclGroupMap.find(identifier) != opbasedCommInfoV2.hcclGroupMap.end();
    RPT_INPUT_ERR(
        commExists, "EI0003", vector<string>({"ccl_op", "value", "parameter", "expect"}),
        vector<string>({"HcclCommInitRootInfo", identifier, "identifier", "an identifier that is not in use"}));
    CHK_PRT_RET(
        commExists,
        HCCL_ERROR(
            "[HcclCommInitRootInfoConfigV2]errNo[0x%016llx] The comm name[%s] already exists in Group2Comm map.",
            HCCL_ERROR_CODE(HCCL_E_PARA), identifier.c_str()),
        HCCL_E_PARA);

    HCCL_INFO("HcclCommInitRootInfoConfigV2 config->hcclBufferSize[%u] MB", config->hcclBufferSize);
    if (UNLIKELY(config->hcclBufferSize == 0)) {
        HCCL_ERROR("HcclCommInitRootInfoConfigV2 config: hcclBufferSize is 0 MB, invalid para");
        return HCCL_E_PARA;
    }

    RankTableInfo rankTable{};
    RootInfoDetectBridge::DetectContext rootInfoDetectContext;
    HcclResult ret = RootInfoDetect(nRanks, rank, rootHandle, rankTable, rootInfoDetectContext);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] errNo[0x%016llx] RootInfoDetect failed, identifier[%s].", __func__, HCCL_ERROR_CODE(ret),
            identifier.c_str());
        rankTable.Dump();
        return ret;
    }

    // 打印ranktable
    rankTable.Dump();

    // 创建通信域
    ret = CreateCommConfigRootInfo(rank, config, identifier, rankTable, comm);
    CHK_PRT_RET(
        ret,
        HCCL_ERROR(
            "[%s]errNo[0x%016llx] and create comm failed, identifier[%s].", __func__, HCCL_ERROR_CODE(ret),
            identifier.c_str()),
        static_cast<HcclResult>(ret));

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "HcclCommInitRootInfoConfigV2 success, take time [%lld]us, rankNum[%u], rank[%u], identifier[%s]",
        DURATION_US(TIME_NOW() - startut), nRanks, rank, identifier.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclScatterV2(
    void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root, HcclComm comm,
    aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    bool isCapture;
    rtModel_t rtModel = nullptr;
    u32 modelId = 0;

    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    const std::string tag = "Scatter_" + communicator->GetId();

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), recvCount, dataType, stream), tag.c_str());
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(communicator->GetRankSize(&rankSize), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRankV2(rankSize, root), tag.c_str());
    CHK_RET(GetStreamCaptureInfo(stream, rtModel, isCapture));

    u32 rankId = INVALID_VALUE_RANKID;
    CHK_RET(communicator->GetRankId(rankId));
    if (rankId == root) { // 本rank为root节点，send_buff不为空
        CHK_PTR_NULL(sendBuf);
    }

    /* 接口交互信息日志 */
    char stackLogBufferV2[LOG_TMPBUF_SIZE];
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        s32 streamId = HrtGetStreamId(stream);
        s32 deviceLogicId = HrtGetDevice();
        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(communicator->GetRankId(localRank), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBufferV2, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], recvCount[%llu], dataType[%s], root[%u], localRank[%u], streamId[%d], "
            "deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, recvCount, GetDataTypeEnumStrV2(dataType).c_str(), root, localRank, streamId,
            deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclScatterV2:" + std::string(stackLogBufferV2);
        if (isCapture) {
            CHK_PTR_NULL(rtModel);
            // 获取不到modelId会报错
            CHK_RET(GetModelId(rtModel, modelId));
            logInfo += ", model id[" + to_string(modelId) + "].";
        }
        communicator->GetTrace().Save(logInfo);
    }

    static thread_local Hccl::CollOpParams opParams;
    opParams.opType = Hccl::OpType::SCATTER;
    opParams.dataType = HcclDataTypeToDataType(dataType);
    opParams.sendBuf = sendBuf;
    opParams.recvBuf = recvBuf;
    opParams.count = recvCount;
    opParams.root = root;

    CHK_RET_AND_PRINT_IDE(communicator->LoadOpbasedCollOp(opParams, static_cast<void*>(stream)), tag.c_str());

    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo = "HcclScatterV2:success,take time: " + std::to_string(DURATION_US(endut - startut).count())
                              + " us, tag: " + tag + std::string(stackLogBufferV2);
        communicator->GetTrace().Save(endInfo);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclAllGatherV2(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    bool isCapture;
    rtModel_t rtModel = nullptr;
    u32 modelId = 0;

    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    const std::string tag = "AllGather_" + communicator->GetId();

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), sendCount, dataType, stream), tag.c_str());
    CHK_RET(GetStreamCaptureInfo(stream, rtModel, isCapture));

    /* 接口交互信息日志 */
    char stackLogBufferV2[LOG_TMPBUF_SIZE];
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        s32 streamId = HrtGetStreamId(stream);
        s32 deviceLogicId = HrtGetDevice();
        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(communicator->GetRankId(localRank), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBufferV2, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], sendCount[%llu], dataType[%s], localRank[%u], streamId[%d],"
            "deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, sendCount, GetDataTypeEnumStrV2(dataType).c_str(), localRank, streamId,
            deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclAllGatherV2:" + std::string(stackLogBufferV2);
        if (isCapture) {
            CHK_PTR_NULL(rtModel);
            // 获取不到modelId会报错
            CHK_RET(GetModelId(rtModel, modelId));
            logInfo += ", model id[" + to_string(modelId) + "].";
        }
        communicator->GetTrace().Save(logInfo);
    }

    static thread_local Hccl::CollOpParams opParams;
    opParams.opType = Hccl::OpType::ALLGATHER;
    opParams.dataType = HcclDataTypeToDataType(dataType);
    opParams.sendBuf = sendBuf;
    opParams.recvBuf = recvBuf;
    opParams.count = sendCount;
    opParams.opTag = tag;

    CHK_RET_AND_PRINT_IDE(communicator->LoadOpbasedCollOp(opParams, static_cast<void*>(stream)), tag.c_str());

    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo
            = "HcclAllGatherV2:success,take time: " + std::to_string(DURATION_US(endut - startut).count())
              + " us, tag: " + tag + std::string(stackLogBufferV2);
        communicator->GetTrace().Save(endInfo);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclAllGatherVV2(
    void* sendBuf, uint64_t sendCount, void* recvBuf, void* recvCounts, void* recvDispls, HcclDataType dataType,
    HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    bool isCapture;
    rtModel_t rtModel = nullptr;
    u32 modelId = 0;

    // 获取通信域
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    const std::string tag = "AllGatherV_" + communicator->GetId();

    // 获取rank信息
    uint32_t rankId;
    CHK_RET(communicator->GetRankId(rankId));
    uint32_t rankSize;
    CHK_RET(communicator->GetRankSize(&rankSize));
    // 参数合法性校验
    if (rankSize == 1) {
        // rankSize为1时，退化为AllGather
        // 检查异常回退AGV的情况
        if (sendCount == 0) {
            HCCL_WARNING("[AllGatherV] sendCount is 0 when single rank");
            return HCCL_SUCCESS;
        } else {
            CHK_PRT_RET(sendBuf == nullptr, HCCL_ERROR("[AllGatherV] sendBuf is null when single rank"), HCCL_E_PTR);
            CHK_PRT_RET(recvBuf == nullptr, HCCL_ERROR("[AllGatherV] recvBuf is null when single rank"), HCCL_E_PTR);
        }
        return HcclAllGatherV2(sendBuf, recvBuf, sendCount, dataType, comm, stream);
    }
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), sendCount, dataType, stream), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckVOpParamV2(rankId, rankSize, sendCount, recvCounts), tag.c_str());
    CHK_RET(GetStreamCaptureInfo(stream, rtModel, isCapture));

    /* 接口交互信息日志 */
    char stackLogBufferV2[LOG_TMPBUF_SIZE];
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        s32 streamId = HrtGetStreamId(stream);
        s32 deviceLogicId = HrtGetDevice();
        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(communicator->GetRankId(localRank), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBufferV2, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], sendCount[%llu], recvCounts[%p], recvDispls[%p], "
            "dataType[%s], localRank[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, sendCount, recvCounts, recvDispls, GetDataTypeEnumStrV2(dataType).c_str(),
            localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclAllGatherVV2:" + std::string(stackLogBufferV2);
        if (isCapture) {
            CHK_PTR_NULL(rtModel);
            // 获取不到modelId会报错
            CHK_RET(GetModelId(rtModel, modelId));
            logInfo += ", model id[" + to_string(modelId) + "].";
        }
        communicator->GetTrace().Save(logInfo);
    }

    u64* counts = static_cast<u64*>(recvCounts);
    u64 output = 0;
    for (size_t index = 0; index < rankSize; index++) {
        output += counts[index];
    }
    if (output == 0) {
        HCCL_INFO("[%s] output[%llu] is equal to zero", __func__, output);
        return HCCL_SUCCESS;
    }
    RPT_INPUT_ERR(
        recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclAllGatherVV2", "nullptr", "recvBuf", "not nullptr"}));
    if (recvBuf == nullptr) {
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] recvBuf is nullptr.", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCCL_ERROR_CODE(HCCL_E_PTR));
        return HCCL_E_PTR;
    }
    // opParams组装
    Hccl::CollOpParams opParams;
    opParams.opType = Hccl::OpType::ALLGATHERV;
    opParams.dataType = HcclDataTypeToDataType(dataType);
    opParams.dstRank = rankId;
    opParams.sendBuf = sendBuf;
    opParams.recvBuf = recvBuf;
    opParams.count = sendCount;
    opParams.vDataDes.counts = recvCounts;
    opParams.vDataDes.displs = recvDispls;
    opParams.vDataDes.dataType = HcclDataTypeToDataType(dataType);

    CHK_RET_AND_PRINT_IDE(communicator->LoadOpbasedCollOp(opParams, static_cast<void*>(stream)), tag.c_str());

    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo
            = "HcclAllGatherVV2:success,take time: " + std::to_string(DURATION_US(endut - startut).count())
              + " us, tag: " + tag + std::string(stackLogBufferV2);
        communicator->GetTrace().Save(endInfo);
    }

    return HCCL_SUCCESS;
}

HcclResult ValidateRank(uint32_t rank, Hccl::HcclCommunicator* communicator)
{
    u32 rankSize{};
    CHK_RET(communicator->GetRankSize(&rankSize));
    u32 rankId{INVALID_VALUE_RANKID};
    CHK_RET(communicator->GetRankId(rankId));
    CHK_RET(HcomCheckUserRankV2(rankSize, rank));
    CHK_PRT_RET(rankId == rank, HCCL_ERROR("same rank[%u] is not allowed", rank), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult
HcclSendV2(void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    bool isCapture;
    rtModel_t rtModel = nullptr;
    u32 modelId = 0;

    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    const std::string tag = "SendRecv_" + communicator->GetId();

    CHK_RET(HcomCheckDataTypeV2(dataType));
    CHK_RET(ValidateRank(destRank, communicator));
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), count, dataType, stream), tag.c_str());
    CHK_RET(GetStreamCaptureInfo(stream, rtModel, isCapture));

    /* 接口交互信息日志 */
    char hcclSendStackLogBufferV2[LOG_TMPBUF_SIZE];
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        s32 streamId = HrtGetStreamId(stream);
        s32 deviceLogicId = HrtGetDevice();
        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(communicator->GetRankId(localRank), tag.c_str());

        s32 ret = snprintf_s(
            hcclSendStackLogBufferV2, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], count[%llu], dataType[%s], localRank[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, count, GetDataTypeEnumStrV2(dataType).c_str(), localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclSendV2:" + std::string(hcclSendStackLogBufferV2);
        if (isCapture) {
            CHK_PTR_NULL(rtModel);
            // 获取不到modelId会报错
            CHK_RET(GetModelId(rtModel, modelId));
            logInfo += ", model id[" + to_string(modelId) + "].";
        }
        communicator->GetTrace().Save(logInfo);
    }

    static thread_local Hccl::CollOpParams opParams{};
    opParams.opType = Hccl::OpType::SEND;
    opParams.dataType = HcclDataTypeToDataType(dataType);
    opParams.sendBuf = sendBuf;
    opParams.recvBuf = nullptr;
    opParams.count = count;
    opParams.dstRank = destRank;
    opParams.opTag = tag;

    CHK_RET_AND_PRINT_IDE(communicator->LoadOpbasedCollOp(opParams, static_cast<void*>(stream)), tag.c_str());

    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo = "HcclSendV2:success,take time: " + std::to_string(DURATION_US(endut - startut).count())
                              + " us, tag: " + tag + std::string(hcclSendStackLogBufferV2);
        communicator->GetTrace().Save(endInfo);
    }

    return HCCL_SUCCESS;
}

HcclResult
HcclRecvV2(void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    bool isCapture;
    rtModel_t rtModel = nullptr;
    u32 modelId = 0;

    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    const std::string tag = "SendRecv_" + communicator->GetId();

    CHK_RET(HcomCheckDataTypeV2(dataType));
    CHK_RET(ValidateRank(srcRank, communicator));
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), count, dataType, stream), tag.c_str());
    CHK_RET(GetStreamCaptureInfo(stream, rtModel, isCapture));

    /* 接口交互信息日志 */
    char hcclRecvStackLogBufferV2[LOG_TMPBUF_SIZE];
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        s32 streamId = HrtGetStreamId(stream);
        s32 deviceLogicId = HrtGetDevice();
        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(communicator->GetRankId(localRank), tag.c_str());

        s32 ret = snprintf_s(
            hcclRecvStackLogBufferV2, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], recvBuf[%p], count[%llu], dataType[%s], localRank[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), recvBuf, count, GetDataTypeEnumStrV2(dataType).c_str(), localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclRecvV2:" + std::string(hcclRecvStackLogBufferV2);
        if (isCapture) {
            CHK_PTR_NULL(rtModel);
            // 获取不到modelId会报错
            CHK_RET(GetModelId(rtModel, modelId));
            logInfo += ", model id[" + to_string(modelId) + "].";
        }
        communicator->GetTrace().Save(logInfo);
    }

    static thread_local Hccl::CollOpParams opParams{};
    opParams.opType = Hccl::OpType::RECV;
    opParams.dataType = HcclDataTypeToDataType(dataType);
    opParams.sendBuf = nullptr;
    opParams.recvBuf = recvBuf;
    opParams.count = count;
    opParams.dstRank = srcRank;
    opParams.opTag = tag;

    CHK_RET_AND_PRINT_IDE(communicator->LoadOpbasedCollOp(opParams, static_cast<void*>(stream)), tag.c_str());

    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo = "HcclRecvV2:success,take time: " + std::to_string(DURATION_US(endut - startut).count())
                              + " us, tag: " + tag + std::string(hcclRecvStackLogBufferV2);
        communicator->GetTrace().Save(endInfo);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclReduceScatterV2(
    void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    bool isCapture;
    rtModel_t rtModel = nullptr;
    u32 modelId = 0;

    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    const std::string tag = "ReduceScatter_" + communicator->GetId();

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), recvCount, dataType, stream), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckReductionOpV2(op), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckReduceDataTypeV2(dataType, op), tag.c_str());
    CHK_RET(GetStreamCaptureInfo(stream, rtModel, isCapture));

    /* 接口交互信息日志 */
    char stackLogBufferV2[LOG_TMPBUF_SIZE];
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        s32 streamId = HrtGetStreamId(stream);
        s32 deviceLogicId = HrtGetDevice();
        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(communicator->GetRankId(localRank), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBufferV2, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], recvCount[%llu], dataType[%s], op[%s],"
            "localRank[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, recvCount, GetDataTypeEnumStrV2(dataType).c_str(),
            GetReduceOpEnumStrV2(op).c_str(), localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclReduceScatterV2:" + std::string(stackLogBufferV2);
        if (isCapture) {
            CHK_PTR_NULL(rtModel);
            // 获取不到modelId会报错
            CHK_RET(GetModelId(rtModel, modelId));
            logInfo += ", model id[" + to_string(modelId) + "].";
        }
        communicator->GetTrace().Save(logInfo);
    }

    static thread_local Hccl::CollOpParams opParams;
    opParams.opType = Hccl::OpType::REDUCESCATTER;
    opParams.dataType = HcclDataTypeToDataType(dataType);
    opParams.reduceOp = HCCL_OP_REDUCE_MAP[op];
    opParams.sendBuf = sendBuf;
    opParams.recvBuf = recvBuf;
    opParams.count = recvCount;
    opParams.opTag = tag;

    CHK_RET_AND_PRINT_IDE(communicator->LoadOpbasedCollOp(opParams, static_cast<void*>(stream)), tag.c_str());

    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo
            = "HcclReduceScatterV2:success,take time: " + std::to_string(DURATION_US(endut - startut).count())
              + " us, tag: " + tag + std::string(stackLogBufferV2);
        communicator->GetTrace().Save(endInfo);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclReduceScatterVV2(
    void* sendBuf, void* sendCounts, void* sendDispls, void* recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    bool isCapture;
    rtModel_t rtModel = nullptr;
    u32 modelId = 0;

    // 获取通信域
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    const std::string tag = "ReduceScatterV_" + communicator->GetId();

    // 获取rank信息
    uint32_t rankId;
    CHK_RET(communicator->GetRankId(rankId));
    uint32_t rankSize;
    CHK_RET(communicator->GetRankSize(&rankSize));
    // 参数合法性校验
    if (rankSize == 1) {
        // rankSize为1时，退化为ReduceScatter
        // 检查异常回退RSV的情况
        if (recvCount == 0) {
            HCCL_WARNING("[ReduceScatterV] recvCount is 0 when single rank");
            return HCCL_SUCCESS;
        } else {
            CHK_PRT_RET(
                sendBuf == nullptr, HCCL_ERROR("[ReduceScatterV] sendBuf is null when single rank"), HCCL_E_PTR);
            CHK_PRT_RET(
                recvBuf == nullptr, HCCL_ERROR("[ReduceScatterV] recvBuf is null when single rank"), HCCL_E_PTR);
        }
        return HcclReduceScatterV2(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
    }
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag.c_str(), recvCount, dataType, stream), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckReductionOpV2(op), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckReduceDataTypeV2(dataType, op), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckVOpParamV2(rankId, rankSize, recvCount, sendCounts), tag.c_str());
    CHK_RET(GetStreamCaptureInfo(stream, rtModel, isCapture));

    /* 接口交互信息日志 */
    char stackLogBufferV2[LOG_TMPBUF_SIZE];
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        s32 streamId = HrtGetStreamId(stream);
        s32 deviceLogicId = HrtGetDevice();
        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(communicator->GetRankId(localRank), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBufferV2, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], sendCounts[%p], sendDispls[%p], recvCount[%llu], dataType[%s], op[%s],"
            "localRank[%u], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, sendCounts, sendDispls, recvCount, GetDataTypeEnumStrV2(dataType).c_str(),
            GetReduceOpEnumStrV2(op).c_str(), localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclReduceScatterVV2:" + std::string(stackLogBufferV2);
        if (isCapture) {
            CHK_PTR_NULL(rtModel);
            // 获取不到modelId会报错
            CHK_RET(GetModelId(rtModel, modelId));
            logInfo += ", model id[" + to_string(modelId) + "].";
        }
        communicator->GetTrace().Save(logInfo);
    }

    u64* counts = static_cast<u64*>(sendCounts);
    u64 inputCount = 0;
    for (size_t index = 0; index < rankSize; index++) {
        inputCount += counts[index];
    }
    if (inputCount == 0) {
        HCCL_INFO("[%s] inputCount[%llu] is equal to zero", __func__, inputCount);
        return HCCL_SUCCESS;
    }
    RPT_INPUT_ERR(
        sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclReduceScatterVV2", "nullptr", "sendBuf", "not nullptr"}));
    if (sendBuf == nullptr) {
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] sendBuf is nullptr.", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), HCCL_ERROR_CODE(HCCL_E_PTR));
        return HCCL_E_PTR;
    }
    if (op == HCCL_REDUCE_PROD) {
        HCCL_ERROR("[Check][ReductionOp] Op:[HCCL_REDUCE_PROD] not supported, tag[%s]", tag.c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    // opParams组装
    Hccl::CollOpParams opParams;
    opParams.opType = Hccl::OpType::REDUCESCATTERV;
    opParams.dataType = HcclDataTypeToDataType(dataType);
    opParams.reduceOp = HcclReduceOpToReduceOp(op);
    opParams.dstRank = rankId;
    opParams.sendBuf = sendBuf;
    opParams.recvBuf = recvBuf;
    opParams.count = recvCount;
    opParams.vDataDes.counts = sendCounts;
    opParams.vDataDes.displs = sendDispls;
    opParams.vDataDes.dataType = HcclDataTypeToDataType(dataType);

    CHK_RET_AND_PRINT_IDE(communicator->LoadOpbasedCollOp(opParams, static_cast<void*>(stream)), tag.c_str());

    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo
            = "HcclReduceScatterVV2:success,take time: " + std::to_string(DURATION_US(endut - startut).count())
              + " us, tag: " + tag + std::string(stackLogBufferV2);
        communicator->GetTrace().Save(endInfo);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclBatchSendRecvV2(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    bool isCapture;
    rtModel_t rtModel = nullptr;
    u32 modelId = 0;

    CHK_PTR_NULL(comm);
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    const std::string tag = "HcclBatchSendRecvV2_" + communicator->GetId();
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(sendRecvInfo);
    CHK_PRT_RET(itemNum == 0, HCCL_WARNING("[BatchSendRecv] taskList itemNum is zero."), HCCL_SUCCESS);
    CHK_RET(GetStreamCaptureInfo(stream, rtModel, isCapture));
    u32 rankSize = 0;
    CHK_RET(communicator->GetRankSize(&rankSize)); // 获取rankSize, 后续需要校验userrank是否在0到rankSize-1之间
    for (uint32_t i = 0; i < itemNum; i++) {
        if ((sendRecvInfo + i)->buf == nullptr) {
            continue; // 支持数据量为0的场景，buf为空的跳过
        }
        CHK_RET_AND_PRINT_IDE(
            HcomCheckOpParamV2(tag.c_str(), (sendRecvInfo + i)->count, (sendRecvInfo + i)->dataType, stream),
            tag.c_str());
        CHK_RET(HcomCheckUserRankV2(rankSize, (sendRecvInfo + i)->remoteRank));
    }

    /* 记录接口交互信息日志 */
    char stackLogBufferV2[LOG_TMPBUF_SIZE];
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        s32 streamId = HrtGetStreamId(stream);
        s32 deviceLogicId = HrtGetDevice();
        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(communicator->GetRankId(localRank), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBufferV2, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], itemNum[%u], localRank[%u], streamId[%d], deviceLogicId[%d]", tag.c_str(), itemNum, localRank,
            streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclBatchSendRecvV2:" + std::string(stackLogBufferV2);
        if (isCapture) {
            CHK_PTR_NULL(rtModel);
            CHK_RET(GetModelId(rtModel, modelId));
            logInfo += ", model id[" + to_string(modelId) + "].";
        }
        communicator->GetTrace().Save(logInfo);
    }

    static thread_local Hccl::CollOpParams opParams;
    opParams.opType = Hccl::OpType::BATCHSENDRECV;
    opParams.batchSendRecvDataDes.sendRecvItemsPtr = static_cast<void*>(sendRecvInfo);
    opParams.batchSendRecvDataDes.itemNum = itemNum;
    opParams.dataType = HcclDataTypeToDataType(sendRecvInfo->dataType);

    CHK_RET_AND_PRINT_IDE(communicator->LoadOpbasedCollOp(opParams, static_cast<void*>(stream)), tag.c_str());
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HcclUs endut = TIME_NOW();
        std::string endInfo
            = "HcclBatchSendRecvV2:success,take time: " + std::to_string(DURATION_US(endut - startut).count())
              + " us, tag: " + tag + std::string(stackLogBufferV2);
        communicator->GetTrace().Save(endInfo);
    }
    return HCCL_SUCCESS;
}

// 功能说明：推动式建链，超时退出
HcclResult WaitAllCommReady(s32 deviceLogicId)
{
    try {
        HrtSetDevice(deviceLogicId);
        HcclCommInfoV2& opbasedCommInfoV2 = GetCommInfoV2();
        // 定义最大等待10秒
        constexpr u32 waitTransportReadyTimeoutMs = 10 * 1000; // 待解决
        auto timeout = std::chrono::milliseconds(waitTransportReadyTimeoutMs);
        HcclUs startTime = std::chrono::steady_clock::now();

        // 创建锁，防止在建链过程中，依旧还在恢复通信域
        std::unique_lock<std::mutex> groupParaLock(opbasedCommInfoV2.groupParamsLock);
        HCCL_INFO("[%s] deviceLogicId[%d] start wait all comm ready", __func__, deviceLogicId);
        // 轮巡调度，推动式建链
        while (true) {
            bool isAllCommReady = true;
            Hccl::HcclCommunicator* communicator;
            // 枚举所有通信域进行推动式建链
            for (auto iter = opbasedCommInfoV2.hcclGroupMap.begin(); iter != opbasedCommInfoV2.hcclGroupMap.end();
                 iter++) {
                CHK_PTR_NULL(iter->second.pComm);
                communicator = static_cast<Hccl::HcclCommunicator*>(iter->second.pComm.get());
                if (!communicator->IsCommReady()) {
                    // 只要任意Comm一个没有ready，整体建链结果为 false
                    isAllCommReady = false;
                }
            }
            if (isAllCommReady) {
                break;
            }

            // 超时判断
            if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
                HCCL_ERROR("[%s]WaitAllCommReady timeout, deviceLogicId[%d].", __func__, deviceLogicId);
                return HCCL_E_TIMEOUT;
            }
        }
        HrtResetDevice(deviceLogicId);
    } catch (HcclException& e) {
        HCCL_ERROR(e.what());
        return e.GetErrorCode();
    } catch (std::exception& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_INTERNAL;
    } catch (...) {
        HCCL_ERROR("Unknown error occurs!");
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[%s] all comm ready.", __func__);
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

HcclResult
HcclGetCclBuffer(HcclComm comm, uintptr_t& cclBufferAddr, size_t& cclBufferSize, HcclMemType& cclBufferMemType)
{
    Hccl::HcclCommunicator* communicatorV2 = (static_cast<Hccl::HcclCommunicator*>(comm));
    CHK_PTR_NULL(communicatorV2);
    CHK_RET(communicatorV2->HcclGetCclBuffer(cclBufferAddr, cclBufferSize, cclBufferMemType));
    return HCCL_SUCCESS;
}

HcclResult HcclGetRawCommHandle(const char* commName, HcclComm* commHandle)
{
    CHK_PTR_NULL(commName);
    CHK_PTR_NULL(commHandle);

    HcclCommInfoV2& opbasedCommInfoV2 = GetCommInfoV2();
    std::unique_lock<std::mutex> lock(opbasedCommInfoV2.groupParamsLock);
    HCCL_INFO("[HcclGetRawCommHandle] group:[%s]", commName);
    auto iter = opbasedCommInfoV2.hcclGroupMap.find(commName);
    if (iter == opbasedCommInfoV2.hcclGroupMap.end()) {
        HCCL_ERROR("[HcclGetRawCommHandle] commName [%s] not found, please check.", commName);
        return HCCL_E_PARA;
    }
    *commHandle = static_cast<HcclComm>(opbasedCommInfoV2.hcclGroupMap[commName].pComm.get());
    return HCCL_SUCCESS;
}

HcclResult HcclGetCcuTaskInfoLegacy(HcclComm comm, void* tilingData, void* ccuTaskGroup)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(tilingData);
    CHK_PTR_NULL(ccuTaskGroup);

    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    if (EnvConfig::GetInstance().GetLogConfig().GetEntryLogEnable()) {
        HCCL_RUN_INFO(
            "Entry-HcclGetCcuTaskInfo V950, commId[%s], tilingData[%p], ccuTaskGroup[%p]",
            communicator->GetId().c_str(), tilingData, ccuTaskGroup);
    }
    auto ret = communicator->GetCcuTaskInfo(tilingData, ccuTaskGroup);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HcclGetCcuTaskInfo ret[%d] commId[%s]", ret, communicator->GetId().c_str());
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclSnapshotSave(void* snapshotBuf, uint32_t size, uint32_t step)
{
    // 快照保存只支持ranktable场景，不支持topo探测场景
    HCCL_INFO("[%s] snapshot save start, size[%u], step[%u]", __func__, size, step);
    CHK_PTR_NULL(snapshotBuf);
    // 获取公共信息
    HcclCommInfoV2& opbasedCommInfoV2 = GetCommInfoV2();
    // 获取公共信息里的step信息
    u64 savedStep = opbasedCommInfoV2.step;
    // 如果当前step不等于用户传入的step，直接返回报错
    if (savedStep != step) {
        HCCL_ERROR("[%s] step is not match, savedStep[%u], userInputStep[%u]", __func__, savedStep, step);
        return HCCL_E_PARA;
    }
    // 获取保存的快照
    Hccl::BinaryStream& savedSnapshotBuf = Hccl::SnapShotParser::GetInstance().GetSnapShotBuf();
    uint32_t dataLen = savedSnapshotBuf.GetSize();
    HCCL_INFO("[%s] savedSnapshotBuf data len[%u]", __func__, dataLen);

    // 获取保存的size
    if ((dataLen + sizeof(dataLen) + sizeof(uint32_t)) != size) {
        // 如果当前size不等于用户传入的size，直接返回报错
        HCCL_ERROR("[%s] size is not match, userInputSize[%u] dataLen[%u]", __func__, size, dataLen);
        return HCCL_E_PARA;
    }
    std::vector<char> data;
    savedSnapshotBuf.DumpWithRevert(data);
    HCCL_INFO("[%s] dump data size [%u]", __func__, data.size());
    if (data.empty()) {
        HCCL_INFO("[%s] dump data empty", __func__);
        return HCCL_E_INTERNAL;
    }

    // 计算快照crc值
    uint32_t crcValue{0};
    CHK_RET(Hccl::SnapShotParser::GetInstance().CalcBufCrc32(savedSnapshotBuf, crcValue));

    // 将快照大小拷贝到用户传入的内存中
    s32 sRet = memcpy_s(snapshotBuf, size, static_cast<void*>(&dataLen), sizeof(dataLen));
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[%s] memcpy dataLen failed, return[%d]", __func__, sRet), HCCL_E_MEMORY);
    // 将快照crc值拷贝到用户传入的内存中
    sRet = memcpy_s(
        static_cast<char*>(snapshotBuf) + sizeof(dataLen), size - sizeof(dataLen), static_cast<void*>(&crcValue),
        sizeof(crcValue));
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[%s] memcpy crcValue failed, return[%d]", __func__, sRet), HCCL_E_MEMORY);
    // 将快照拷贝到用户传入的内存中
    sRet = memcpy_s(static_cast<char*>(snapshotBuf) + sizeof(dataLen) + sizeof(crcValue), dataLen, &data[0], dataLen);
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[%s] memcpy data failed, return[%d]", __func__, sRet), HCCL_E_MEMORY);
    return HCCL_SUCCESS;
}

void RecoverSnapshotCcuStatus(const std::shared_ptr<Hccl::SnapShotBuf>& savedSnapshotBuf)
{
    HcclCommInfoV2& opbasedCommInfoV2 = GetCommInfoV2();
    // 通过进程锁看护，避免多个通信域同时占用CCU_MS
    std::unique_lock<std::mutex> lock(opbasedCommInfoV2.groupParamsLock);

    for (auto useMsCommId : savedSnapshotBuf->ccuStatusSnapshot.useMsCommIds) {
        opbasedCommInfoV2.ccuStatus.useMsCommIds.push_back(useMsCommId.data());
    }

    for (auto useSchedCommId : savedSnapshotBuf->ccuStatusSnapshot.useSchedCommIds) {
        opbasedCommInfoV2.ccuStatus.useSchedCommIds.push_back(useSchedCommId.data());
    }
    // 结果打印
    HCCL_INFO(
        "RecoverSnapshotCcuStatus ccuMsCommIds size[%lu] ccuSchedCommIds size[%lu]",
        opbasedCommInfoV2.ccuStatus.useMsCommIds.size(), opbasedCommInfoV2.ccuStatus.useSchedCommIds.size());
    for (auto useMsCommId : opbasedCommInfoV2.ccuStatus.useMsCommIds) {
        HCCL_DEBUG("RecoverSnapshotCcuStatus useMsCommId[%s]", useMsCommId.c_str());
    }
    for (auto useSchedCommId : opbasedCommInfoV2.ccuStatus.useSchedCommIds) {
        HCCL_DEBUG("RecoverSnapshotCcuStatus useSchedCommId[%s]", useSchedCommId.c_str());
    }
}

HcclResult HcclSnapshotRecoverAllComms(
    const char* clusterInfo, const char* changedInfo, void* snapshotBuf, uint32_t snapshotBufSize)
{
    (void)clusterInfo;
    // 入参校验，clusterInfo和changedInfo暂时用不到，先不判空
    HCCL_INFO("[%s] snapshot recover start.", __func__);
    CHK_PTR_NULL(snapshotBuf);

    // 当前不支持在不下发算子的情况下重复调用recover接口
    if (Hccl::SnapShotParser::GetInstance().GetIsNeedLoadOp()) {
        HCCL_ERROR("[%s] snapshot recover failed, it is necessary to use it after load op, please check!", __func__);
        return HCCL_E_INTERNAL;
    }

    s32 deviceLogicId = HcclGetThreadDeviceId();

    CHK_RET(CallSingletons()); // 临时规避，在初始化通信域前声明单例保证时序

    // 获取公共信息
    HcclCommInfoV2& opbasedCommInfoV2 = GetCommInfoV2();
    // 调用恢复快照公共信息函数，让字节流的struct数据恢复到savedSnapshotBuf
    std::shared_ptr<Hccl::SnapShotBuf> savedSnapshotBuf = std::make_shared<Hccl::SnapShotBuf>();
    CHK_RET(
        Hccl::SnapShotParser::GetInstance().ParseSnapshotToLocalBuff(snapshotBuf, snapshotBufSize, *savedSnapshotBuf));

    opbasedCommInfoV2.step = savedSnapshotBuf->snapShotPub.step;
    // 将状态设置为RECOVERED
    opbasedCommInfoV2.status = DeviceStatus::DEVICE_RECOVERED;

    // 创建通信域
    opbasedCommInfoV2.pComm.reset(new (std::nothrow) Hccl::HcclCommunicator(
        savedSnapshotBuf->snapshot.snapShotComm.commParams, &savedSnapshotBuf->snapshot.snapShotComm.config));
    CHK_PTR_NULL(opbasedCommInfoV2.pComm);
    opbasedCommInfoV2.pComm->RegisterAcceStateCallBack(CommunicatorCallback());
    // 恢复全局通讯域
    HCCL_INFO("[%s] recover global group.", __func__);
    CHK_RET(opbasedCommInfoV2.pComm->RecoverComm(
        static_cast<void*>(&savedSnapshotBuf->snapshot.snapShotComm), savedSnapshotBuf->snapShotPub.step, changedInfo));
    HCCL_INFO("[%s] global group recover success.", __func__);
    opbasedCommInfoV2.commParams = savedSnapshotBuf->snapshot.snapShotComm.commParams;

    HcclGroupParamsV2 params{};
    params.pComm = opbasedCommInfoV2.pComm;
    opbasedCommInfoV2.hcclGroupMap[savedSnapshotBuf->snapshot.groupName] = params;

    // 这里变成多线程后，封装成小函数
    for (uint32_t index = 0; index < savedSnapshotBuf->groupNum; index++) {
        std::string groupName(savedSnapshotBuf->subSnapshot[index].groupName);
        // 别重复恢复全局通讯域
        if (groupName == savedSnapshotBuf->snapshot.groupName) {
            continue;
        }
        const Hccl::SnapShotSubComm* snapShotSubComm = &savedSnapshotBuf->subSnapshot[index].snapShotSubComm;
        // 创建一个指针
        std::shared_ptr<Hccl::HcclCommunicator> commImp
            = make_shared<Hccl::HcclCommunicator>(snapShotSubComm->commParams, &snapShotSubComm->config);
        // 循环创建恢复子通讯域
        HCCL_INFO("[%s] recover sub group[%s].", __func__, groupName.c_str());
        CHK_RET(opbasedCommInfoV2.pComm->RecoverSubComm(
            static_cast<const void*>(snapShotSubComm), commImp, savedSnapshotBuf->snapShotPub.step));
        HCCL_INFO("[%s] sub group[%s] recover success.", __func__, groupName.c_str());
        HcclGroupParamsV2 params{};
        params.pComm = commImp;
        // 这里进行多线程后，加锁
        opbasedCommInfoV2.hcclGroupMap[groupName] = params;
    }
    RecoverSnapshotCcuStatus(savedSnapshotBuf);
    HCCL_INFO("[%s] snapshot recover success.", __func__);
    std::unique_ptr<std::thread> waitReadyThread;
    waitReadyThread.reset(new (std::nothrow) std::thread(&WaitAllCommReady, deviceLogicId));
    if (waitReadyThread == nullptr) {
        HCCL_ERROR("[%s] new waitReadyThread failed.", __func__);
        return HCCL_E_INTERNAL;
    }
    // 异步方案目前未分析清楚，采用临时方案规避
    waitReadyThread->join();
    Hccl::SnapShotParser::GetInstance().SetIsNeedLoadOp(true);
    HCCL_INFO("[%s] all comm ready.", __func__);
    return HCCL_SUCCESS;
}
static HcclResult GetAllSnapShotStaticBuf(
    const std::shared_ptr<Hccl::HcclCommunicator>& pComm,
    const std::map<std::string, std::shared_ptr<Hccl::HcclCommunicator>>& hcclGroupMap, uint32_t step,
    Hccl::BinaryStream& buf)
{
    HCCL_INFO("[%s] start", __func__);
    if (!pComm->IsWorldGroup()) {
        HCCL_ERROR("[%s] input comm is not hccl_world_group, please check!", __func__);
        return HCCL_E_INTERNAL;
    }
    // 生成 静态流公共数据,直接往buf写
    Hccl::SnapShotParser::GetInstance().SerializeCommVersionInfo(buf);
    // 存储 全局通信域名称
    buf << pComm->GetId();
    // 存储 全局通信域 静态buf
    Hccl::BinaryStream& pCommBuf = *(static_cast<Hccl::BinaryStream*>(pComm->GetStaticBinaryInfo()));
    std::vector<char> pCommChars{};
    pCommBuf.Dump(pCommChars);
    for (auto c : pCommChars) {
        buf << c;
    }

    size_t pSubCommSize = hcclGroupMap.size() - 1;
    HCCL_INFO("[%s] pSubCommSize[%u]", __func__, pSubCommSize);
    buf << pSubCommSize; // 子通信域 静态buf 数量

    auto iter = hcclGroupMap.begin();
    for (; iter != hcclGroupMap.end(); iter++) {
        auto pSubCommName = iter->first;
        if (pComm->GetId() == pSubCommName) {
            continue;
        }
        buf << pSubCommName; // 子通信域 名字
        // 获取子通信域 静态buf
        auto groupComm = iter->second;
        Hccl::BinaryStream& pSubCommBuf = *(static_cast<Hccl::BinaryStream*>(groupComm->GetStaticBinaryInfo()));
        std::vector<char> pSubCommChars{};
        pSubCommBuf.Dump(pSubCommChars);
        for (auto c : pSubCommChars) {
            buf << c;
        }
    }
    buf << step;
    HCCL_INFO("[%s] end", __func__);

    return HCCL_SUCCESS;
}

static HcclResult GetAllSnapShotDynamicBuf(
    const std::shared_ptr<Hccl::HcclCommunicator>& pComm,
    const std::map<std::string, std::shared_ptr<Hccl::HcclCommunicator>>& hcclGroupMap, Hccl::BinaryStream& buf)
{
    HCCL_INFO("[%s] start", __func__);
    CHK_PTR_NULL(pComm);
    // 全局通信域 动态buf信息
    CHK_RET(pComm->GetSnapShotDynamicBuf(static_cast<void*>(&buf)));

    size_t pSubCommSize = hcclGroupMap.size() - 1;
    buf << pSubCommSize; // 子通信域 动态buf信息 数量
    HCCL_INFO("[%s] hcclGroupMap size[%u]", __func__, hcclGroupMap.size());
    auto iter = hcclGroupMap.begin();
    for (; iter != hcclGroupMap.end(); iter++) {
        auto pSubCommName = iter->first;
        if (pComm->GetId() == pSubCommName) {
            continue;
        }
        // 获取子通信域 建链邻居信息buf
        CHK_RET(iter->second->GetSnapShotDynamicBuf(static_cast<void*>(&buf)));
    }
    HCCL_INFO("[%s] end", __func__);
    return HCCL_SUCCESS;
}

void GetSnapShotCcuStatusBuf(Hccl::BinaryStream& buf)
{
    HcclCommInfoV2& opbasedCommInfoV2 = GetCommInfoV2();
    // 通过进程锁看护，避免多个通信域同时占用CCU_MS
    std::unique_lock<std::mutex> lock(opbasedCommInfoV2.groupParamsLock);

    auto ccuStatus = opbasedCommInfoV2.ccuStatus;
    buf << ccuStatus.useMsCommIds.size();
    HCCL_INFO("useMsCommIds size is %u", ccuStatus.useMsCommIds.size());
    for (auto useMsCommId : ccuStatus.useMsCommIds) {
        buf << useMsCommId;
        HCCL_INFO("useMsCommId is %s", useMsCommId.c_str());
    }

    buf << ccuStatus.useSchedCommIds.size();
    HCCL_INFO("useSchedCommIds size is %u", ccuStatus.useSchedCommIds.size());
    for (auto useSchedCommId : ccuStatus.useSchedCommIds) {
        buf << useSchedCommId;
        HCCL_INFO("useSchedCommId is %s", useSchedCommId.c_str());
    }
}

// 获取快照占用buffer的大小 --给op_base API getSize调，内部生成完整长流。save后释放
HcclResult SnapshotGenerate(
    const std::shared_ptr<Hccl::HcclCommunicator>& pComm,
    const std::map<std::string, std::shared_ptr<Hccl::HcclCommunicator>>& hcclGroupMap, uint32_t step, uint32_t* size)
{
    CHK_PTR_NULL(size);
    CHK_PTR_NULL(pComm);
    HCCL_INFO("[%s] start", __func__);
    Hccl::SnapShotParser::GetInstance().GetSnapShotBuf().Clear();
    CHK_RET(GetAllSnapShotStaticBuf(pComm, hcclGroupMap, step, Hccl::SnapShotParser::GetInstance().GetSnapShotBuf()));
    CHK_RET(GetAllSnapShotDynamicBuf(pComm, hcclGroupMap, Hccl::SnapShotParser::GetInstance().GetSnapShotBuf()));
    GetSnapShotCcuStatusBuf(Hccl::SnapShotParser::GetInstance().GetSnapShotBuf());
    // size = 流长度 + crc（u32）长度 + 存储流长度所需长度
    uint32_t dataLen = static_cast<uint32_t>(Hccl::SnapShotParser::GetInstance().GetSnapShotBuf().GetSize());
    *size = dataLen + sizeof(dataLen) + sizeof(uint32_t); // 快照头上保存一个总长度和crc长度
    HCCL_INFO("[%s] end, size[%u]", __func__, *size);
    return HCCL_SUCCESS;
}

HcclResult HcclSnapshotGetBufSize(uint32_t step, uint32_t* size)
{
    // 校验DevType
    HCCL_INFO("[%s] start", __func__);
    Hccl::DevType devType = HrtGetDeviceType();
    if (devType != DevType::DEV_TYPE_950 && devType != DevType::DEV_TYPE_960) {
        HCCL_INFO("[%s] Get buffer size not support in this device type[%d]", __func__, devType);
        return HCCL_E_NOT_SUPPORT;
    }

    // step保存在g_opbasedCommInfoV2，save的时候校验，防止GetBufSize和Save的step不一致
    HcclCommInfoV2& opbasedCommInfoV2 = GetCommInfoV2();
    opbasedCommInfoV2.step = step;

    std::map<std::string, std::shared_ptr<Hccl::HcclCommunicator>> subCommMap;
    for (auto hcclGroupMap : opbasedCommInfoV2.hcclGroupMap) {
        subCommMap.insert(std::make_pair(hcclGroupMap.first, hcclGroupMap.second.pComm));
    }
    return SnapshotGenerate(opbasedCommInfoV2.pComm, subCommMap, step, size);
}

HcclResult HcclGetTopoDescV2()
{
    HCCL_ERROR("Current chip type does not support GetTopoDesc.");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclGetCommAsyncErrorV2() { return HCCL_SUCCESS; }

HcclResult HcclSetConfigV2(HcclConfig config, HcclConfigValue configValue)
{
    (void)(config);
    (void)(configValue);
    HCCL_WARNING("DETERMINISTIC_ENABLE is default option in 950! Can not set.");
    return HCCL_SUCCESS;
}
HcclResult HcclGetConfigV2(HcclConfig config, HcclConfigValue* configValue)
{
    (void)(config);
    constexpr int32_t DETERMINISTIC_ENABLE = 1; // A5支持确定性，不需要配置
    (*configValue).value = DETERMINISTIC_ENABLE;
    HCCL_WARNING("DETERMINISTIC_ENABLE is default option in 950!");
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankGraphV2(const HcclComm* comm, void** rankGraph)
{
    Hccl::HcclCommunicator* communicatorV2 = (static_cast<Hccl::HcclCommunicator*>(*comm));
    communicatorV2->GetRankGraphV2(*rankGraph);
    return HCCL_SUCCESS;
}

HcclResult HcommFlushV2()
{
    HCCL_INFO("[HcommFlushV2]");
    return FlushManager::GetInstance().Flush();
}

HcclResult CommGetCCLBufSizeCfgV2(HcclComm comm, uint64_t* cclBufSize)
{
    HCCL_RUN_INFO("Entry-CommGetCCLBufSizeCfg V950");
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    CHK_RET(communicator->GetConfigInCCLbufferSize(cclBufSize));
    return HCCL_SUCCESS;
}

HcclResult HcclGetNetLayersV2(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum)
{
    HCCL_RUN_INFO("Entry-HcclGetNetLayersV2 V950");
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    auto ret = communicator->GetNetLayers(netLayers, netLayerNum);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "HcclGetNetLayersV2 get netLayers from communicator failed, commId[%s], ret[%d]",
            communicator->GetId().c_str(), ret);
        return HCCL_E_NOT_FOUND; // 查询失败返回
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetInstSizeByNetLayerV2(HcclComm comm, uint32_t netLayer, uint32_t* rankNum)
{
    HCCL_RUN_INFO("Entry-HcclGetInstSizeByNetLayerV2 V950");
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    auto ret = communicator->GetInstSizeByNetLayer(netLayer, rankNum);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "HcclGetInstSizeByNetLayerV2 get InstSize from communicator failed at netLayer[%u], commId[%s], ret[%d]",
            netLayer, communicator->GetId().c_str(), ret);
        return HCCL_E_NOT_FOUND; // 查询失败返回
    }
    /* 关键状态记录 */
    HCCL_INFO(
        "HcclGetInstSizeByNetLayerV2 success, netLayer[%u], rankNum[%u], commId[%s]", netLayer, *rankNum,
        communicator->GetId().c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclGetInstRanksByNetLayerV2(HcclComm comm, uint32_t netLayer, uint32_t** ranks, uint32_t* rankNum)
{
    HCCL_RUN_INFO("Entry-HcclGetInstRanksByNetLayerV2 V950");
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    auto ret = communicator->GetInstRanksByNetLayer(netLayer, ranks, rankNum);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "HcclGetInstRanksByNetLayerV2 get ranks from communicator failed at netLayer[%u], commId[%s], ret[%d]",
            netLayer, communicator->GetId().c_str(), ret);
        return HCCL_E_NOT_FOUND; // 查询失败返回
    }
    /* 关键状态记录 */
    HCCL_INFO(
        "HcclGetInstRanksByNetLayerV2 success, netLayer[%u], rankNum[%u], commId[%s]", netLayer, *rankNum,
        communicator->GetId().c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclGetInstTopoTypeByNetLayerV2(HcclComm comm, uint32_t netLayer, uint32_t* topoType)
{
    HCCL_RUN_INFO("Entry-HcclGetInstTopoTypeByNetLayer V950");
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    auto ret = communicator->GetInstTopoTypeByNetLayer(netLayer, topoType);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "HcclGetInstTopoTypeByNetLayerV2 get topoType from communicator failed at netLayer[%u], commId[%s], "
            "ret[%d]",
            netLayer, communicator->GetId().c_str(), ret);
        return HCCL_E_NOT_FOUND;
    }
    /* 关键状态记录 */
    HCCL_INFO(
        "HcclGetInstTopoTypeByNetLayer success, netLayer[%u] topoType[%u] commId[%s]", netLayer, *topoType,
        communicator->GetId().c_str());
    return HCCL_SUCCESS;
}

HcclResult
HcclGetInstSizeListByNetLayerV2(HcclComm comm, uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize)
{
    HCCL_RUN_INFO("Entry-HcclGetInstSizeListByNetLayer V950");
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    auto ret = communicator->GetInstSizeListByNetLayer(netLayer, instSizeList, listSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "HcclGetInstSizeListByNetLayerV2 get netInstance size from communicator failed at netLayer[%u], "
            "commId[%s], ret[%d]",
            netLayer, communicator->GetId().c_str(), ret);
        return HCCL_E_NOT_FOUND;
    }
    /* 关键状态记录 */
    HCCL_INFO(
        "HcclGetInstSizeListByNetLayer success, netLayer[%u] listSize[%u] commId[%s]", netLayer, *listSize,
        communicator->GetId().c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclGetLinksV2(
    HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize)
{
    HCCL_RUN_INFO("Entry-HcclGetInstSizeListByNetLayer V950");
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    auto ret = communicator->GetLinks(netLayer, srcRank, dstRank, linkList, listSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "HcclGetLinksV2 get links from communicator failed at netLayer[%u], commId[%s], ret[%d]", netLayer,
            communicator->GetId().c_str(), ret);
        return HCCL_E_NOT_FOUND;
    }
    /* 关键状态记录 */
    HCCL_INFO(
        "HcclGetLinks success, netLayer[%u], srcRank[%u], dstRank[%u], listSize[%u], commId[%s]", netLayer, srcRank,
        dstRank, *listSize, communicator->GetId().c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclGetTopoInstsByLayerV2(HcclComm comm, uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum)
{
    HCCL_RUN_INFO("Entry-HcclGetTopoInstsByLayer V950");
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    auto ret = communicator->GetTopoInstsByLayer(netLayer, topoInsts, topoInstNum);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "HcclGetTopoInstsByLayer get topoInsts from communicator failed at netLayer[%u], commId[%s], ret[%d]",
            netLayer, communicator->GetId().c_str(), ret);
        return HCCL_E_NOT_FOUND;
    }
    /* 关键状态记录 */
    HCCL_INFO(
        "HcclGetTopoInstsByLayer success, netLayer[%u], topoInstNum[%u], commId[%s]", netLayer, *topoInstNum,
        communicator->GetId().c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclGetTopoTypeV2(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType)
{
    HCCL_RUN_INFO("Entry-HcclGetInstSizeListByNetLayer V950");
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    auto ret = communicator->GetTopoType(netLayer, topoInstId, topoType);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "HcclGetTopoType get topoType from communicator failed at netLayer[%u], commId[%s], ret[%d]", netLayer,
            communicator->GetId().c_str(), ret);
        return HCCL_E_NOT_FOUND;
    }
    /* 关键状态记录 */
    HCCL_INFO(
        "HcclGetTopoType success, netLayer[%u] topoType[%u] commId[%s]", netLayer, *topoType,
        communicator->GetId().c_str());
    return HCCL_SUCCESS;
}

HcclResult
HcclGetRanksByTopoInstV2(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum)
{
    HCCL_RUN_INFO("Entry-HcclGetRanksByTopoInst V950");
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    auto ret = communicator->GetRanksByTopoInst(netLayer, topoInstId, ranks, rankNum);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "HcclGetRanksByTopoInst get ranks from communicator failed at netLayer[%u], commId[%s], ret[%d]", netLayer,
            communicator->GetId().c_str(), ret);
        return HCCL_E_NOT_FOUND;
    }
    /* 关键状态记录 */
    HCCL_INFO("HcclGetRanksByTopoInst success, netLayer[%u] rankNum[%u]", netLayer, *rankNum);
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetEndpointNumV2(HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t* num)
{
    HCCL_RUN_INFO("Entry-HcclRankGraphGetEndpointNum V950");
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    auto ret = communicator->GetEndpointNum(layer, topoInstId, num);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "HcclRankGraphGetEndpointNum get endpoint num from communicator failed at netLayer[%u] with "
            "topoInstId[%u], commId[%s], ret[%d]",
            layer, topoInstId, communicator->GetId().c_str(), ret);
        return HCCL_E_NOT_FOUND;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetEndpointDescV2(
    HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc)
{
    HCCL_RUN_INFO("Entry-HcclRankGraphGetEndpointDesc V950");
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    auto ret = communicator->GetEndpointDesc(layer, topoInstId, descNum, endpointDesc);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "HcclRankGraphGetEndpointDesc get endpoint desc from communicator failed at netLayer[%u], commId[%s], "
            "ret[%d]",
            layer, communicator->GetId().c_str(), ret);
        return HCCL_E_NOT_FOUND;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetEndpointInfoV2(
    HcclComm comm, uint32_t rankId, const EndpointDesc* endpointDesc, EndpointAttr endpointAttr, uint32_t infoLen,
    void* info)
{
    HCCL_RUN_INFO("Entry-HcclRankGraphGetEndpointInfo V950");
    Hccl::HcclCommunicator* communicator = static_cast<Hccl::HcclCommunicator*>(comm);
    auto ret = communicator->GetEndpointInfo(rankId, endpointDesc, endpointAttr, infoLen, info);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "HcclRankGraphGetEndpointInfo get info from communicator failed with endpointAttr [%d], commId[%s], "
            "ret[%d]",
            static_cast<s32>(endpointAttr), communicator->GetId().c_str(), ret);
        return HCCL_E_NOT_FOUND;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommWorkingDevNicSetV2(const HcclComm comm, uint32_t* ranks, bool* useBackup, uint32_t nRanks)
{
    (void)comm;
    (void)ranks;
    (void)useBackup;
    (void)nRanks;
    HCCL_ERROR("HcclCommWorkingDevNicSetV2 not support V950.");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
HcclCommSetMemoryRangeV2(const HcclComm comm, void* baseVirPtr, size_t size, size_t alignment, uint64_t flags)
{
    (void)comm;
    (void)baseVirPtr;
    (void)size;
    (void)alignment;
    (void)flags;
    HCCL_ERROR("HcclCommSetMemoryRangeV2 not support V950.");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclCommUnsetMemoryRangeV2(const HcclComm comm, void* baseVirPtr)
{
    (void)comm;
    (void)baseVirPtr;
    HCCL_ERROR("HcclCommUnsetMemoryRangeV2 not support V950.");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclCommActivateCommMemoryV2(
    const HcclComm comm, void* virPtr, size_t size, size_t offset, void* handle, uint64_t flags)
{
    (void)comm;
    (void)virPtr;
    (void)size;
    (void)offset;
    (void)handle;
    (void)flags;
    HCCL_ERROR("HcclCommActivateCommMemoryV2 not support V950.");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclCommDeactivateCommMemoryV2(const HcclComm comm, void* virPtr)
{
    (void)comm;
    (void)virPtr;
    HCCL_ERROR("HcclCommDeactivateCommMemoryV2 not support V950.");
    return HCCL_E_NOT_SUPPORT;
}

uint32_t HcclGetCommConfigCapabilityV2() { return static_cast<uint32_t>(HCCL_COMM_CONFIG_RETRY); }

#ifdef __cplusplus
}
#endif // __cplusplus
