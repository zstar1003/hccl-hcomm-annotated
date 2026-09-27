/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mutex>
#include <sstream>
#include <string>
#include <cmath>
#include "alg_env_config.h"

using namespace hccl;

static std::mutex g_algEnvConfigMutex;

HcclResult ResetAlgEnvConfigInitState()
{
    std::lock_guard<std::mutex> lock(g_algEnvConfigMutex);
    g_algEnvConfig.SetDefaultParams();
    return HCCL_SUCCESS;
}

const std::vector<HcclAlgoType> GetExternalInputHcclAlgoConfig(HcclCMDType opType)
{
    std::lock_guard<std::mutex> lock(g_algEnvConfigMutex);
    return g_algEnvConfig.hcclAlgoConfig[opType];
}

HcclResult SetCommonAlgType(std::vector<std::string>& algos)
{
    std::lock_guard<std::mutex> lock(g_algEnvConfigMutex);
    std::vector<HcclAlgoType> algType;
    CHK_RET(ParseAlgoString("all op type", algos[0], algType));
    for (u32 opType = 0; opType < static_cast<u32>(HcclCMDType::HCCL_CMD_MAX); opType++) {
        g_algEnvConfig.hcclAlgoConfig[static_cast<HcclCMDType>(opType)] = algType;
    }
    return HCCL_SUCCESS;
}

HcclResult SetSpecificAlgType(std::vector<std::string>& algos)
{
    std::lock_guard<std::mutex> lock(g_algEnvConfigMutex);
    for (std::string& algConfig : algos) {
        std::size_t found = algConfig.find("=");
        std::string opStringName = algConfig.substr(0, found);
        if (opStringName == "others") {
            std::vector<HcclAlgoType> algType;
            std::string remainAlgoConfig = algConfig.substr(found + 1);
            CHK_RET(ParseAlgoString("others op type", remainAlgoConfig, algType));
            for (u32 opType = 0; opType < static_cast<u32>(HcclCMDType::HCCL_CMD_MAX); opType++) {
                g_algEnvConfig.hcclAlgoConfig[static_cast<HcclCMDType>(opType)] = algType;
            }
        }
    }
    std::map<std::string, HcclCMDType> hcclOpTypeMap = {
        {"broadcast", HcclCMDType::HCCL_CMD_BROADCAST},
        {"allreduce", HcclCMDType::HCCL_CMD_ALLREDUCE},
        {"reduce", HcclCMDType::HCCL_CMD_REDUCE},
        {"send", HcclCMDType::HCCL_CMD_SEND},
        {"receive", HcclCMDType::HCCL_CMD_RECEIVE},
        {"allgather", HcclCMDType::HCCL_CMD_ALLGATHER},
        {"reducescatter", HcclCMDType::HCCL_CMD_REDUCE_SCATTER},
        {"alltoall", HcclCMDType::HCCL_CMD_ALLTOALL},
        {"gather", HcclCMDType::HCCL_CMD_GATHER},
        {"scatter", HcclCMDType::HCCL_CMD_SCATTER},
        {"sendrecv", HcclCMDType::HCCL_CMD_BATCH_SEND_RECV},
    };
    for (std::string& algConfig : algos) {
        std::size_t found = algConfig.find("=");
        std::string opStringName = algConfig.substr(0, found);
        if (hcclOpTypeMap.find(opStringName) != hcclOpTypeMap.end()) {
            HcclCMDType optype = hcclOpTypeMap[opStringName];
            std::string remainAlgoConfig = algConfig.substr(found + 1);
            std::vector<HcclAlgoType> algType;
            CHK_RET(ParseAlgoString(opStringName, remainAlgoConfig, algType));
            if (algType[0] == HcclAlgoType::HCCL_ALGO_TYPE_NULL) {
                HCCL_ERROR("[SetSpecificAlgType] specific config level0 not support null type.");
                return HCCL_E_PARA;
            }
            g_algEnvConfig.hcclAlgoConfig[optype] = algType;
        } else {
            HCCL_ERROR(
                "[SetSpecificAlgType] specific config optype[%s] is invalid, please check", opStringName.c_str());
            return HCCL_E_PARA;
        }
    }
    g_algEnvConfig.hcclAlgoConfig[HcclCMDType::HCCL_CMD_ALLTOALLV]
        = g_algEnvConfig.hcclAlgoConfig[HcclCMDType::HCCL_CMD_ALLTOALL];
    g_algEnvConfig.hcclAlgoConfig[HcclCMDType::HCCL_CMD_ALLTOALLVC]
        = g_algEnvConfig.hcclAlgoConfig[HcclCMDType::HCCL_CMD_ALLTOALL];
    return HCCL_SUCCESS;
}

HcclResult ParserHcclAlgoLevel(const std::string& algoLevel, u32& level, HcclAlgoType& algoType)
{
    std::size_t found = algoLevel.find(":");
    if ((found == 0) || (found == (algoLevel.length() - 1))) {
        HCCL_ERROR("[Parser][HcclAlgoLevel] algo config is invalid.");
        return HCCL_E_PARA;
    }

    std::string orginalLevel = algoLevel.substr(0, found);
    std::string orginalAlgo = algoLevel.substr(found + 1);

    const std::map<std::string, u32> hcclAlgoLevelMap
        = {{"level0", HCCL_ALGO_LEVEL_0},
           {"level1", HCCL_ALGO_LEVEL_1},
           {"level2", HCCL_ALGO_LEVEL_2},
           {"level3", HCCL_ALGO_LEVEL_3}};

    const std::map<std::string, HcclAlgoType> hcclAlgoTypeMap = {
        {"null", HcclAlgoType::HCCL_ALGO_TYPE_NULL},
        {"ring", HcclAlgoType::HCCL_ALGO_TYPE_RING},
        {"pipeline", HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE},
        {"fullmesh", HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH},
        {"H-D_R", HcclAlgoType::HCCL_ALGO_TYPE_HDR},
        {"pairwise", HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE},
        {"NHR", HcclAlgoType::HCCL_ALGO_TYPE_NHR},
        {"NHR_V1", HcclAlgoType::HCCL_ALGO_TYPE_NHR_V1},
        {"AHC", HcclAlgoType::HCCL_ALGO_TYPE_AHC},
        {"AHC_BROKE", HcclAlgoType::HCCL_ALGO_TYPE_AHC_BROKE},
        {"NB", HcclAlgoType::HCCL_ALGO_TYPE_NB},
        {"NA", HcclAlgoType::HCCL_ALGO_TYPE_NA},
    };

    auto iterAlgoLevel = hcclAlgoLevelMap.find(orginalLevel);
    if (iterAlgoLevel == hcclAlgoLevelMap.end()) {
        HCCL_ERROR("[Parser][HcclAlgoLevel] algo config is invalid, level %s is not supported.", orginalLevel.c_str());
        return HCCL_E_PARA;
    }

    auto iterAlgoType = hcclAlgoTypeMap.find(orginalAlgo);
    if (iterAlgoType == hcclAlgoTypeMap.end()) {
        HCCL_ERROR("[Parser][HcclAlgoLevel] algo config is invalid, algo %s is not supported.", orginalAlgo.c_str());
        return HCCL_E_PARA;
    }

    level = iterAlgoLevel->second;
    algoType = iterAlgoType->second;

    return HCCL_SUCCESS;
}

HcclResult ParseAlgoString(std::string opName, std::string& algoString, std::vector<HcclAlgoType>& algType)
{
    algType = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    std::vector<std::string> algoLevels;
    HcclResult ret = SplitHcclAlgoLevel(algoString, algoLevels);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Set][HcclAlgoConfig]hccl algo config[%s] is invalid. "
            "expect: level0:NA;level1:<algo> or <op0>=level0:NA;level1:<algo0>/<op1>=level0:NA;level1:<algo1>",
            algoString.c_str()),
        ret);
    for (auto algoLevel : algoLevels) {
        u32 level = 0;
        HcclAlgoType algo = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
        ret = ParserHcclAlgoLevel(algoLevel, level, algo);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Set][HcclAlgoConfig]hccl algo config[%s] is invalid. "
                "expect: level0:NA;level1:<algo> or <op0>=level0:NA;level1:<algo0>/<op1>=level0:NA;level1:<algo1>",
                algoString.c_str()),
            ret);
        // 检查是否存在重复配置level
        if (algType[level] != HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT) {
            HCCL_ERROR(
                "[Set][HcclAlgoConfig]hccl algo config[%s] is invalid. "
                "expect: level0:NA;level1:<algo> or <op0>=level0:NA;level1:<algo0>/<op1>=level0:NA;level1:<algo1>",
                algoString.c_str());
            return HCCL_E_PARA;
        }
        algType[level] = algo;
    }
    auto level0Iter = HcclAlgoTypeMap.find(algType[HCCL_ALGO_LEVEL_0]);
    auto level1Iter = HcclAlgoTypeMap.find(algType[HCCL_ALGO_LEVEL_1]);
    auto level2Iter = HcclAlgoTypeMap.find(algType[HCCL_ALGO_LEVEL_2]);
    auto level3Iter = HcclAlgoTypeMap.find(algType[HCCL_ALGO_LEVEL_3]);
    HCCL_RUN_INFO(
        "hccl algo op %s config: config level0:%s, level1:%s, level2:%s, level3:%s", opName.c_str(),
        level0Iter->second.c_str(), level1Iter->second.c_str(), level2Iter->second.c_str(), level3Iter->second.c_str());
    return HCCL_SUCCESS;
}

HcclResult SplitHcclOpType(const std::string& algoConfig, std::vector<std::string>& algos)
{
    std::string remainAlgoConfig;
    std::size_t found = algoConfig.find("/");
    if ((found == 0) || (found == (algoConfig.length() - 1))) {
        HCCL_ERROR("[Split][SplitHcclOpType] algo config is invalid.");
        return HCCL_E_PARA;
    } else if (found != std::string::npos) {
        remainAlgoConfig = algoConfig.substr(found + 1);
    }
    algos.push_back(algoConfig.substr(0, found));
    if (!remainAlgoConfig.empty()) {
        CHK_RET(SplitHcclOpType(remainAlgoConfig, algos));
    }
    return HCCL_SUCCESS;
}

// 新的逐算法的配置和原有的统一配置只可使用一种，发现同时存在时报错
HcclResult CheckAlgoConfigValid(std::vector<std::string>& algos, bool& anyCommonConfig, bool& anySpecificConfig)
{
    for (std::string& algConfig : algos) {
        std::size_t found = algConfig.find("=");
        if ((found == 0) || (found == (algConfig.length() - 1))) {
            HCCL_ERROR("[Split][CheckAlgoConfigValid] algo config is invalid.");
            return HCCL_E_PARA;
        } else if (found != std::string::npos) {
            anySpecificConfig = true;
        } else {
            anyCommonConfig = true;
        }
    }
    if (anyCommonConfig && anySpecificConfig) {
        HCCL_ERROR("[CheckAlgoConfigValid]should not set both algo config way");
        return HCCL_E_PARA;
    }
    if (anyCommonConfig && algos.size() > 1) {
        HCCL_ERROR("[CheckAlgoConfigValid]should only set one common config");
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult SplitHcclAlgoLevel(const std::string& algoConfig, std::vector<std::string>& algos)
{
    std::string remainAlgoConfig;
    std::size_t found = algoConfig.find(";");
    if ((found == 0) || (found == (algoConfig.length() - 1))) {
        HCCL_ERROR("[Split][HcclAlgoLevel] algo config is invalid.");
        return HCCL_E_PARA;
    } else if (found != std::string::npos) {
        remainAlgoConfig = algoConfig.substr(found + 1);
    } else {
        // 最后一组配置,剩余的字符串为空
    }
    algos.push_back(algoConfig.substr(0, found));

    if (algos.size() > HCCL_ALGO_LEVEL_NUM) {
        HCCL_ERROR("[Split][HcclAlgoLevel] algo config is invalid. algo level is more than %u.", HCCL_ALGO_LEVEL_NUM);
        return HCCL_E_PARA;
    }
    if (!remainAlgoConfig.empty()) {
        CHK_RET(SplitHcclAlgoLevel(remainAlgoConfig, algos));
    }

    return HCCL_SUCCESS;
}

s32 GetInternalExecTimeOut()
{
    double timeout = GetExternalInputHcclExecTimeOut();
    // 向上取整获取s32秒级超时时间
    return static_cast<s32>(std::ceil(timeout));
}
