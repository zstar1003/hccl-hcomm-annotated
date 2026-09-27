/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_configurator.h"
#include "log.h"
#include "hccl/base.h"
#include "coll_alg_utils.h"

namespace hccl {

constexpr u32 DEVICE_EIGHT = 8;
constexpr u32 DEVICE_FOUR = 4;
constexpr u32 DEVICE_TWO = 2;
constexpr u32 DEVICE_ONE = 1;

AlgConfigurator::AlgConfigurator(HcclAlgoAttr& algoAttr, HcclTopoAttr& topoAttr)
    : algoAttr_(algoAttr),
      topoAttr_(topoAttr),
      deterministic_(GetExternalInputHcclDeterministicV2())
{}

AlgConfigurator::~AlgConfigurator() {}

HcclResult AlgConfigurator::Init(bool isHeterogComm)
{
    if (!isHeterogComm) {
        // 获取算法类型
        CHK_RET(SelectAlgType(topoAttr_.moduleNum, topoAttr_.deviceType, algType_));
        // 获取拓扑类型，根据算法类型转化
        CHK_RET(GetTopoTypeByAlgType(algType_[HcclCMDType::HCCL_CMD_ALL], topoAttr_.deviceType, topoType_));
    } else {
        topoType_ = TopoType::TOPO_TYPE_HETEROG;
    }

    HCCL_INFO("alg configurator init success.");
    return HCCL_SUCCESS;
}

HcclResult
AlgConfigurator::SelectAlgType(u32 moduleNum, const DevType deviceType, std::map<HcclCMDType, AlgType>& algType)
{
    for (u32 opType = 0; opType < static_cast<u32>(HcclCMDType::HCCL_CMD_MAX); opType++) {
        CHK_RET(SelectCurrOpAlgType(moduleNum, deviceType, static_cast<HcclCMDType>(opType), algType));
    }
    return HCCL_SUCCESS;
}

HcclResult AlgConfigurator::SelectCurrOpAlgType(
    u32 moduleNum, const DevType deviceType, HcclCMDType opType, std::map<HcclCMDType, AlgType>& algType)
{
    AlgTypeLevel0 algType0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    AlgTypeLevel1 algType1 = AlgTypeLevel1::ALG_LEVEL1_RESERVED;
    AlgTypeLevel2 algType2
        = AlgTypeLevel2::ALG_LEVEL2_RESERVED; // 第2层拓扑算法, 待梳理后考虑是否和第0层、第1层算法归一

    std::vector<HcclAlgoType> algoCommConfig = algoAttr_.commAlgoConfig[opType];
    HcclAlgoType algoConfigLevel0 = algoCommConfig[HCCL_ALGO_LEVEL_0];
    HcclAlgoType algoConfigLevel1 = algoCommConfig[HCCL_ALGO_LEVEL_1];
    HcclAlgoType algoConfigLevel2 = algoCommConfig[HCCL_ALGO_LEVEL_2];

    bool isConfigAHC
        = (algoConfigLevel1 == HcclAlgoType::HCCL_ALGO_TYPE_AHC
           || algoConfigLevel1 == HcclAlgoType::HCCL_ALGO_TYPE_AHC_BROKE);

    bool isConfigNULL = algoConfigLevel0 == HcclAlgoType::HCCL_ALGO_TYPE_NULL;

    HCCL_INFO(
        "[AlgConfigurator][SelectCurrOpAlgType] isConfigAHC[%u] isConfigNULL[%u] multiModuleDiffDeviceNumMode[%u] "
        "multiSuperPodDiffServerNumMode[%u] multiSuperPodDiffDeviceNumMode[%u]",
        isConfigAHC, isConfigNULL, topoAttr_.multiModuleDiffDeviceNumMode, topoAttr_.multiSuperPodDiffServerNumMode,
        topoAttr_.multiSuperPodDiffDeviceNumMode);

    bool isSupportCmdARS
        = (opType == HcclCMDType::HCCL_CMD_ALLGATHER || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER
           || opType == HcclCMDType::HCCL_CMD_ALLREDUCE);

    bool isSupportCmdAHC
        = (opType == HcclCMDType::HCCL_CMD_ALLGATHER || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER
           || opType == HcclCMDType::HCCL_CMD_ALLREDUCE || opType == HcclCMDType::HCCL_CMD_ALL);

    // server内卡数不对称，且不是ARS的情况
    bool isNoARS
        = (topoAttr_.multiModuleDiffDeviceNumMode
           && !(
               isSupportCmdARS && deviceType == DevType::DEV_TYPE_910_93 && !topoAttr_.multiSuperPodDiffDeviceNumMode));

    // server内卡数对称, server内卡数不对称情况isNoARS已经讨论完毕；
    // 超节点server非对称，且不是AHC的情况
    bool isNoAHC
        = (!topoAttr_.multiModuleDiffDeviceNumMode && topoAttr_.multiSuperPodDiffServerNumMode
           && !(isSupportCmdAHC && isConfigAHC));

    if (Is310P3Common(algoAttr_.isHaveCpuRank, topoAttr_.deviceType)) {
        algType[opType].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING;
        algType[opType].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING;
    } else if (
        !topoAttr_.multiModuleDiffDeviceNumMode && topoAttr_.multiSuperPodDiffServerNumMode
        && (opType == HcclCMDType::HCCL_CMD_ALLGATHER || opType == HcclCMDType::HCCL_CMD_ALLREDUCE
            || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER || opType == HcclCMDType::HCCL_CMD_ALL)) {
        CHK_RET(SetAlgoLevel0(algoConfigLevel0, algType0));
        CHK_RET(SetAlgoLevel1(HcclAlgoType::HCCL_ALGO_TYPE_AHC, moduleNum, algType1, opType));
        CHK_RET(SetAlgoLevel2(algoConfigLevel2, algType2));
        algType[opType].algoLevel0 = algType0;
        algType[opType].algoLevel1 = algType1;
        if (!topoAttr_.isStandardCard && deviceType != DevType::DEV_TYPE_910B && !topoAttr_.isDiffDeviceType) {
            if (topoAttr_.nicList.size() != DEVICE_EIGHT && topoAttr_.deviceNumPerAggregation == DEVICE_EIGHT
                && algType0 != AlgTypeLevel0::ALG_LEVEL0_8P_RING) {
                HCCL_ERROR("[Set][AlgType]nicSize[%zu] error, algType is not 8P ring.", topoAttr_.nicList.size());
                return HCCL_E_PARA;
            }
        }
        HCCL_INFO("[AlgConfigurator][SelectCurrOpAlgType] multiModuleDiffDeviceNumMode is true, set ahc.");
    } else if (
        !topoAttr_.multiModuleDiffDeviceNumMode && topoAttr_.multiSuperPodDiffServerNumMode
        && deviceType == DevType::DEV_TYPE_910_93
        && (opType == HcclCMDType::HCCL_CMD_ALLTOALL || opType == HcclCMDType::HCCL_CMD_ALLTOALLV
            || opType == HcclCMDType::HCCL_CMD_ALLTOALLVC)) {
        // A3 ALLTOALL非对称拓扑
        CHK_RET(SetAlgoLevel0(algoConfigLevel0, algType0));
        CHK_RET(SetAlgoLevel1(algoConfigLevel1, moduleNum, algType1, opType));
        CHK_RET(SetAlgoLevel2(algoConfigLevel2, algType2));
        algType[opType].algoLevel0 = algType0;
        algType[opType].algoLevel1 = algType1;
        algType[opType].algoLevel2 = algType2;
        if (algoConfigLevel0 == HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT
            && algoConfigLevel1 == HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT) {
            algType[opType].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING;
            algType[opType].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING;
            isAlgoLevel1Default_[opType] = false;
        }
    } else if ((isNoARS || isNoAHC) && !isConfigNULL) { // 多server不同卡模式，设置为单层拓扑类型
        algType[opType].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING;
        algType[opType].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING;
        isAlgoLevel1Default_[opType] = false;
        if (algoConfigLevel0 != HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT
            || algoConfigLevel1 != HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT) {
            HCCL_WARNING(
                "multiModuleDiffDeviceNumMode[%d], multiSuperPodDiffServerNumMode_[%d], algorithm type [%d] is "
                "selected by force.",
                topoAttr_.multiModuleDiffDeviceNumMode, topoAttr_.multiSuperPodDiffServerNumMode,
                algType[opType].algoLevel0);
        }
        HCCL_INFO("[AlgConfigurator][SelectCurrOpAlgType] multiModuleDiffDeviceNumMode is true, set default ring.");
    } else if (algoAttr_.isHaveCpuRank) {
        algType[opType].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_STAR;
        algType[opType].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_STAR;
    } else {
        CHK_RET(SetAlgoLevel0(algoConfigLevel0, algType0));
        CHK_RET(SetAlgoLevel1(algoConfigLevel1, moduleNum, algType1, opType));
        CHK_RET(SetAlgoLevel2(algoConfigLevel2, algType2));
        algType[opType].algoLevel0 = algType0;
        algType[opType].algoLevel1 = algType1;
        algType[opType].algoLevel2 = algType2;
        if (!topoAttr_.isStandardCard && deviceType != DevType::DEV_TYPE_910B && !topoAttr_.isDiffDeviceType) {
            if (topoAttr_.nicList.size() != DEVICE_EIGHT && topoAttr_.deviceNumPerAggregation == DEVICE_EIGHT
                && algType0 != AlgTypeLevel0::ALG_LEVEL0_8P_RING) {
                HCCL_ERROR("[Set][AlgType]nicSize[%zu] error, algType is not 8P ring.", topoAttr_.nicList.size());
                return HCCL_E_PARA;
            }
        }
    }

    auto level0Iter = HCCL_ALGO_LEVEL0_NAME_MAP.find(algType0);
    CHK_PRT_RET(
        level0Iter == HCCL_ALGO_LEVEL0_NAME_MAP.end(), HCCL_ERROR("level0: algType0[%u] is invalid.", algType0),
        HCCL_E_INTERNAL);
    auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
    CHK_PRT_RET(
        level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), HCCL_ERROR("level1: algType1[%u] is invalid.", algType1),
        HCCL_E_INTERNAL);
    auto level2Iter = HCCL_ALGO_LEVEL2_NAME_MAP.find(algType2);
    CHK_PRT_RET(
        level2Iter == HCCL_ALGO_LEVEL2_NAME_MAP.end(), HCCL_ERROR("level2: algType2[%u] is invalid.", algType2),
        HCCL_E_INTERNAL);
    HCCL_RUN_INFO(
        "Device Type[%u], average device count[%u], HccsNum[%u], SIONum[%u], HCCS_SW_NUM[%u], optype[%u]", deviceType,
        topoAttr_.deviceNumPerAggregation, topoAttr_.pairLinkInfo[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size(),
        topoAttr_.pairLinkInfo[static_cast<u32>(LinkTypeInServer::SIO_TYPE)].size(),
        topoAttr_.pairLinkInfo[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)].size(), opType);
    return HCCL_SUCCESS;
}

HcclResult AlgConfigurator::SetAlgoLevel0(HcclAlgoType algoConfig, AlgTypeLevel0& algType)
{
    if (topoAttr_.isStandardCard) {
        CHK_RET(SetAlgoLevel0StandardCard(algoConfig, algType));
    } else {
        CHK_RET(SetAlgoLevel0Module(algoConfig, algType));
    }
    return HCCL_SUCCESS;
}

HcclResult
AlgConfigurator::SetAlgoLevel1(HcclAlgoType algoConfig, u32 moduleNum, AlgTypeLevel1& algType, HcclCMDType opType)
{
    HcclAlgoType algoConfigShadow = algoConfig;
    switch (algoConfig) {
        case HcclAlgoType::HCCL_ALGO_TYPE_HDR:
            algType = AlgTypeLevel1::ALG_LEVEL1_HD;
            HCCL_INFO("server num[%u]: level1:hdr algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_RING:
            algType = AlgTypeLevel1::ALG_LEVEL1_RING;
            HCCL_INFO("server num[%u]: level1:ring algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NHR:
            algType = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_INFO("server num[%u]: level1:nhr algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NHR_V1:
            algType = AlgTypeLevel1::ALG_LEVEL1_NHR_V1;
            HCCL_INFO("server num[%u]: level1:nhr_v1 algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_AHC:
            if (opType == HcclCMDType::HCCL_CMD_ALLGATHER || opType == HcclCMDType::HCCL_CMD_ALLREDUCE
                || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
                algType = AlgTypeLevel1::ALG_LEVEL1_AHC;
                HCCL_INFO("server num[%u]: level1:ahc algo is set.", moduleNum);
                break;
            } else if (opType < HcclCMDType::HCCL_CMD_ALL) {
                algoConfigShadow = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
                HCCL_INFO("server num[%u]: level1:ahc algo is not support, set default.", moduleNum);
                break;
            } else {
                algType = AlgTypeLevel1::ALG_LEVEL1_AHC;
                HCCL_INFO("server num[%u]: level1:ahc algo is set.", moduleNum);
                return HCCL_SUCCESS;
            }
        case HcclAlgoType::HCCL_ALGO_TYPE_AHC_BROKE:
            if (opType == HcclCMDType::HCCL_CMD_ALLGATHER || opType == HcclCMDType::HCCL_CMD_ALLREDUCE
                || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
                algType = AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE;
                HCCL_INFO("server num[%u]: level1:ahc broke algo is set.", moduleNum);
                break;
            } else if (opType < HcclCMDType::HCCL_CMD_ALL) {
                algoConfigShadow = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
                HCCL_INFO("server num[%u]: level1:ahc broke algo is not support, set default.", moduleNum);
                break;
            } else {
                algType = AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE;
                HCCL_INFO("server num[%u]: level1:ahc broke algo is set.", moduleNum);
                return HCCL_SUCCESS;
            }
        case HcclAlgoType::HCCL_ALGO_TYPE_NB:
            algType = AlgTypeLevel1::ALG_LEVEL1_NB;
            HCCL_INFO("server num[%u]: level1:nb algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE:
            algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
            HCCL_INFO("server num[%u]: level1:pipeline algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH:
        case HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE:
            HCCL_WARNING("level1:fullmesh algo is not supported. the config is ignored.");
            algoConfigShadow = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
            break;
        default:
            algoConfigShadow = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
            HCCL_INFO("server num[%u]: level1:default algo is set.", moduleNum);
            break;
    }

    HCCL_DEBUG(
        "[AlgConfigurator][SetAlgoLevel1] algType[%u], deviceType_[%u], workflowmode[%u]", algType,
        topoAttr_.deviceType, GetWorkflowMode());
    if (algType == AlgTypeLevel1::ALG_LEVEL1_PIPELINE
        && ((topoAttr_.deviceType != DevType::DEV_TYPE_910B && topoAttr_.deviceType != DevType::DEV_TYPE_910_93)
            || GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)) {
        algoConfigShadow = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
        HCCL_WARNING("hccl algorithm: there are %u server in level1, config pipeline algo failed.", moduleNum);
    }

    if (algoConfigShadow == HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT) {
        if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
            isAlgoLevel1Default_[opType] = true;
        }
        CHK_RET(GetDefaultAlgoLevel1V1(moduleNum, algType));
    }
    return HCCL_SUCCESS;
}

HcclResult AlgConfigurator::GetDefaultAlgoLevel1V1(u32 moduleNum, AlgTypeLevel1& algType) const
{
    if (moduleNum >= HCCL_INTER_SERVER_RING_ALGO_MAX_SUPPORT_SERVER_NUM) {
        // server 数为 8 以上：使用 HD 算法
        algType = AlgTypeLevel1::ALG_LEVEL1_HD;
    } else {
        // server 数为 2 的非整数次幂：使用 RING 算法
        // server 数为 2 的整数次幂：使用 HD 算法
        algType = (((moduleNum & (moduleNum - 1)) != 0) || (moduleNum == 1)) ? AlgTypeLevel1::ALG_LEVEL1_RING :
                                                                               AlgTypeLevel1::ALG_LEVEL1_HD;
    }
    if (algType == AlgTypeLevel1::ALG_LEVEL1_HD && topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        algType = AlgTypeLevel1::ALG_LEVEL1_NHR;
    }
    HCCL_INFO("[AlgConfigurator][GetDefaultAlgoLevel1V1] algType[%u], moduleNum[%u]", algType, moduleNum);
    return HCCL_SUCCESS;
}

HcclResult AlgConfigurator::SetAlgoLevel2(HcclAlgoType algoConfig, AlgTypeLevel2& algType)
{
    u32 superPodNum = topoAttr_.superPodNum;
    switch (algoConfig) {
        case HcclAlgoType::HCCL_ALGO_TYPE_HDR:
            algType = AlgTypeLevel2::ALG_LEVEL2_HD;
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_RING:
            algType = AlgTypeLevel2::ALG_LEVEL2_RING;
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NHR:
            algType = AlgTypeLevel2::ALG_LEVEL2_NHR;
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NB:
            algType = AlgTypeLevel2::ALG_LEVEL2_NB;
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE:
            algType = AlgTypeLevel2::ALG_LEVEL2_PIPELINE;
            break;
        default: {
            // Level2默认选择NHR算法
            algType = AlgTypeLevel2::ALG_LEVEL2_NHR;
            break;
        }
    }
    HCCL_DEBUG(
        "[AlgConfigurator][SetAlgoLevel2]algType[%u], deviceType_[%u], superPodNum_[%u]", algType, topoAttr_.deviceType,
        superPodNum);
    return HCCL_SUCCESS;
}

HcclResult AlgConfigurator::SetAlgoLevel0StandardCard(HcclAlgoType algoConfig, AlgTypeLevel0& algType)
{
    if (algoConfig == HcclAlgoType::HCCL_ALGO_TYPE_NULL) {
        algType = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
        return HCCL_SUCCESS;
    }

    if (algoConfig != HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT && algoConfig != HcclAlgoType::HCCL_ALGO_TYPE_NA) {
        HCCL_WARNING("level0:%d algo is not supported. the config is ignored.", algoConfig);
    }

    CHK_RET(GetDefaultAlgoLevel0StandardCard(algType));
    return HCCL_SUCCESS;
}

HcclResult AlgConfigurator::GetDefaultAlgoLevel0StandardCard(AlgTypeLevel0& algType) const
{
    if (topoAttr_.deviceNumPerAggregation == DEVICE_TWO) {
        if ((topoAttr_.deviceType == DevType::DEV_TYPE_910B)) {
            algType = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
        } else {
            algType = AlgTypeLevel0::ALG_LEVEL0_2P_MESH;
        }
    } else if (topoAttr_.deviceNumPerAggregation > DEVICE_TWO && topoAttr_.deviceNumPerAggregation <= DEVICE_EIGHT) {
        // 随标卡支持rank数变更
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    } else if (topoAttr_.deviceNumPerAggregation == DEVICE_ONE) {
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    } else {
        HCCL_ERROR("in standard card[num %u] there is no supported algo.", topoAttr_.deviceNumPerAggregation);
        return HCCL_E_PARA;
    }
    HCCL_DEBUG("[GetDefaultAlgoLevel0StandardCard] AlgTypeLevel0 is set to [%u].", algType);
    return HCCL_SUCCESS;
}

HcclResult AlgConfigurator::SetAlgoLevel0Module(HcclAlgoType algoConfig, AlgTypeLevel0& algType)
{
    if (algoConfig == HcclAlgoType::HCCL_ALGO_TYPE_NULL) {
        algType = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
        return HCCL_SUCCESS;
    }

    if (algoConfig != HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT && algoConfig != HcclAlgoType::HCCL_ALGO_TYPE_NA) {
        HCCL_WARNING("level0:%d algo is not supported. the config is ignored.", algoConfig);
    }

    CHK_RET(GetDefaultAlgoLevel0Module(algType));
    return HCCL_SUCCESS;
}

HcclResult AlgConfigurator::GetDefaultAlgoLevel0Module(AlgTypeLevel0& algType)
{
    u32 deviceNumPerAggregation = topoAttr_.deviceNumPerAggregation;
    if (deviceNumPerAggregation == DEVICE_EIGHT) {
        algType = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    } else if (deviceNumPerAggregation == DEVICE_FOUR) {
        algType = AlgTypeLevel0::ALG_LEVEL0_4P_MESH;
    } else if (deviceNumPerAggregation == DEVICE_TWO) {
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    } else if (deviceNumPerAggregation == DEVICE_ONE) {
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    } else {
        algType = AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING;
    }

    if ((topoAttr_.pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)]
             == deviceNumPerAggregation * (deviceNumPerAggregation - 1)
         || topoAttr_.pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)]
                == FACTOR_NUM_TWO * deviceNumPerAggregation * (deviceNumPerAggregation - 1))
        && topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
        HCCL_DEBUG("[GetDefaultAlgoLevel0Module] AlgTypeLevel0 is set to ALG_LEVEL0_NP_MESH (HCCS links are enabled).");
    }

    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        algType = IsHCCSSWNumEqualToTwiceSIONum() ? AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING :
                                                    AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    }
    if (topoAttr_.isDiffDeviceType) {
        HCCL_DEBUG(
            "[GetDefaultAlgoLevel0Module]gcdDeviceNumPerAggregation [%u] isDiffDeviceType [%u] algType [%u]",
            topoAttr_.gcdDeviceNumPerAggregation, topoAttr_.deviceType, algType);
    }
    return HCCL_SUCCESS;
}

bool AlgConfigurator::IsHCCSSWNumEqualToTwiceSIONum()
{
    u32 hccsSWNum = topoAttr_.pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)];
    u32 sioNum = topoAttr_.pairLinkCounter[static_cast<u32>(LinkTypeInServer::SIO_TYPE)];
    HCCL_DEBUG(
        "In pairLinkCounter_, the hccsSWNum is [%lu], the sioNum is [%lu], the deviceNumPerAggregation is [%lu]",
        hccsSWNum, sioNum, topoAttr_.deviceNumPerAggregation);
    if (hccsSWNum == 0) {
        return false;
    }
    if (sioNum == 0) {
        return false;
    }
    // The following 2 means that the device has no HCCS_SW link with itself and its companion linked by same SIO link.
    return (hccsSWNum == ((topoAttr_.deviceNumPerAggregation - 2) * topoAttr_.deviceNumPerAggregation))
           && (sioNum == topoAttr_.deviceNumPerAggregation);
}

HcclResult AlgConfigurator::CheckAlgType(const AlgType algType)
{
    if (algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING
        && algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD) {
        HCCL_ERROR(
            "[Check][AlgType]errNo[0x%016llx] algType[%s] is not supported", HCCL_ERROR_CODE(HCCL_E_PARA),
            AlgTypeToStr(algType).c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

AlgTypeLevel0 AlgConfigurator::GetLevel0AlgType(const AlgType algType) const
{
    if (algType.algoLevel0 != AlgTypeLevel0::ALG_LEVEL0_NP_STAR
        && algType.algoLevel1 != AlgTypeLevel1::ALG_LEVEL1_STAR) {
        return algType.algoLevel0;
    }

    return AlgTypeLevel0::ALG_LEVEL0_NP_STAR;
}

HcclResult AlgConfigurator::GetTopoTypeByAlgType(const AlgType& algType, const DevType deviceType, TopoType& topoType)
{
    CHK_RET(CheckAlgType(algType));
    CHK_RET(CheckDeviceType(deviceType));

    switch (algType.algoLevel0) {
        case AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING:
            topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_8P_RING:
            topoType = TopoType::TOPO_TYPE_8P_RING;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_4P_MESH:
            topoType = TopoType::TOPO_TYPE_4P_MESH;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_2P_MESH:
            topoType = TopoType::TOPO_TYPE_2P_MESH;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING:
            topoType = TopoType::TOPO_TYPE_NP_SINGLE_RING;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_1P_MESH:
            topoType = TopoType::TOPO_TYPE_1P_MESH;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_4P_RING:
            topoType = TopoType::TOPO_TYPE_4P_RING;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_NP_MESH:
            topoType = TopoType::TOPO_TYPE_NP_MESH;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING:
        case AlgTypeLevel0::ALG_LEVEL0_RESERVED:
            topoType = TopoType::TOPO_TYPE_COMMON;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_NP_STAR:
            topoType = TopoType::TOPO_TYPE_ES_MESH;
            break;
        default:
            HCCL_ERROR(
                "[AlgConfigurator][GetTopoTypeByAlgType]errNo[0x%016llx] case: device type[%d](0~1:V910),"
                " algorithm[%s] is not support",
                HCCL_ERROR_CODE(HCCL_E_PARA), deviceType, AlgTypeToStr(algType).c_str());
            return HCCL_E_PARA;
    }

    HCCL_INFO(
        "[AlgConfigurator][GetTopoTypeByAlgType]algtype[%s], devicetype[%d],topotype[%d] is selected",
        AlgTypeToStr(algType).c_str(), deviceType, topoType);
    return HCCL_SUCCESS;
}

HcclResult AlgConfigurator::GetAlgType(AlgType& algType, HcclCMDType opType)
{
    opType = (algType_.find(opType) == algType_.end() ? HcclCMDType::HCCL_CMD_INVALID : opType);
    algType = algType_[opType];
    CHK_RET(CheckAlgType(algType));
    return HCCL_SUCCESS;
}

HcclResult AlgConfigurator::SetAlgType(AlgType algType, HcclCMDType opType)
{
    CHK_RET(CheckAlgType(algType));
    algType_[opType] = algType;
    return HCCL_SUCCESS;
}

bool AlgConfigurator::SupportDeterministicOptim() const
{
    bool support = topoAttr_.isSingleMeshAggregation && topoAttr_.deviceNumPerAggregation > DEVICE_TWO
                   && topoAttr_.deviceType == DevType::DEV_TYPE_910B && deterministic_ != DETERMINISTIC_DISABLE;
    return support;
}

void AlgConfigurator::GetTopoType(TopoType& topoType)
{
    topoType = topoType_;
    return;
}

void AlgConfigurator::GetAlgTypeDirect(AlgType& algType, HcclCMDType opType)
{
    opType = (algType_.find(opType) == algType_.end() ? HcclCMDType::HCCL_CMD_INVALID : opType);
    algType = algType_[opType];
    return;
}

HcclResult AlgConfigurator::GetAlgoLevel1DefaultSwitch(bool& isAlgoLevel1Default, HcclCMDType opType)
{
    isAlgoLevel1Default = isAlgoLevel1Default_[opType];
    return HCCL_SUCCESS;
}

const HcclTopoAttr& AlgConfigurator::GetTopoAttr() { return topoAttr_; }

const HcclAlgoAttr& AlgConfigurator::GetAlgoAttr() { return algoAttr_; }
} // namespace hccl
