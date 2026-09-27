/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ALG_ENV_CONFIG_H
#define HCCL_ALG_ENV_CONFIG_H

#include <vector>
#include <map>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "common.h"

/*************** For Internal Use ***************/
struct AlgEnvConfig {
    // 初始化标识
    bool initialized;

    std::map<HcclCMDType, std::vector<HcclAlgoType>> hcclAlgoConfig;

    AlgEnvConfig() { SetDefaultParams(); }
    void SetDefaultParams()
    {
        initialized = false;
        // 环境变量参数
        for (u32 opType = 0; opType < static_cast<u32>(HcclCMDType::HCCL_CMD_MAX); opType++) {
            hcclAlgoConfig[static_cast<HcclCMDType>(opType)]
                = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
        }
    }
};

static AlgEnvConfig g_algEnvConfig;

const std::map<HcclAlgoType, std::string> HcclAlgoTypeMap = {
    {HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT, "default"},
    {HcclAlgoType::HCCL_ALGO_TYPE_RING, "ring"},
    {HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE, "pipeline"},
    {HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH, "fullmesh"},
    {HcclAlgoType::HCCL_ALGO_TYPE_HDR, "H-D_R"},
    {HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE, "pairwise"},
    {HcclAlgoType::HCCL_ALGO_TYPE_NHR, "NHR"},
    {HcclAlgoType::HCCL_ALGO_TYPE_NHR_V1, "NHR_V1"},
    {HcclAlgoType::HCCL_ALGO_TYPE_AHC, "AHC"},
    {HcclAlgoType::HCCL_ALGO_TYPE_AHC_BROKE, "AHC_BROKE"},
    {HcclAlgoType::HCCL_ALGO_TYPE_NB, "NB"},
    {HcclAlgoType::HCCL_ALGO_TYPE_NULL, "null"},
    {HcclAlgoType::HCCL_ALGO_TYPE_NA, "NA"},
};

HcclResult ResetAlgEnvConfigInitState();

const std::vector<HcclAlgoType> GetExternalInputHcclAlgoConfig(HcclCMDType opType = HcclCMDType::HCCL_CMD_ALL);

HcclResult SetCommonAlgType(std::vector<std::string>& algos);

HcclResult SetSpecificAlgType(std::vector<std::string>& algos);

HcclResult ParserHcclAlgoLevel(const std::string& algoLevel, u32& level, HcclAlgoType& algoType);

HcclResult ParseAlgoString(std::string opName, std::string& algoString, std::vector<HcclAlgoType>& algType);

HcclResult SplitHcclOpType(const std::string& algoConfig, std::vector<std::string>& algos);

HcclResult CheckAlgoConfigValid(std::vector<std::string>& algos, bool& anyCommonConfig, bool& anySpecificConfig);

HcclResult SplitHcclAlgoLevel(const std::string& algoConfig, std::vector<std::string>& algos);

s32 GetInternalExecTimeOut();
#endif // HCCL_ALG_ENV_CONFIG_H
