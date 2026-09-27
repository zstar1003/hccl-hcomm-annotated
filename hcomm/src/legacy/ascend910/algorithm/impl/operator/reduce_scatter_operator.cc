/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_operator.h"
#include "device_capacity.h"
#include "coll_alg_utils.h"
#include "hccl_aiv.h"
#include "coll_alg_op_registry.h"
#include <algorithm>

constexpr u32 MODULE_NUM_FOUR = 4;
// 保序规约仅支持多于2卡场景(2卡时收发对端唯一, 无规约顺序差异)
constexpr u32 DETERMINISTIC_MIN_USER_RANK_SIZE = 2;
constexpr u32 HCCL_310P_DATA_SIZE_MID_COUNT = 320 * 1024;
constexpr u32 HCCL_310P_DATA_SIZE_SMALL_COUNT = 1024;
constexpr u32 HCCL_310P_SLIM_RING_MAX_SIZE = 8;

// Pipeline并行比串行更优的总数据量临界点（基于910_93 2 SuperPod, 1 Server, 12 Rank 性能基线测试，
// 数据量低于此值时调度开销超过流水收益）
constexpr u64 HCCL_PIPELINE_TOTAL_DATA_SIZE_THRESHOLD = 608 * 1024 * 1024;

namespace hccl {

ReduceScatterOperator::ReduceScatterOperator(
    AlgConfigurator* algConfigurator, CCLBufferManager& cclBufferManager, HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_REDUCE_SCATTER)
{}

ReduceScatterOperator::~ReduceScatterOperator() {}

HcclResult ReduceScatterOperator::SelectAlg(
    const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag)
{
    ResourceLimit limit;
    return SelectAlg(tag, param, algName, newTag, limit);
}

HcclResult ReduceScatterOperator::SelectAlg(
    const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag, const ResourceLimit& limit)
{
    if (userRankSize_ == 1) {
        algName = "ReduceScatterSingleExecutor";
        return HCCL_SUCCESS;
    }
    HcclResult ret;
    if (isDiffDeviceType_) {
        ret = SelectAlgforMix(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        ret = SelectAlgfor310P3(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        ret = SelectAlgfor910A(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910_93) {
        ret = SelectAlgfor91093(param, algName, limit);
    } else {
        HCCL_ERROR("[SelectAlg] device type[%d] is out of range for selector.", deviceType_);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[ReduceScatterSelector][SelectAlg]tag[%s], ReduceScatter failed, return[%d]", tag.c_str(), ret),
        ret);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        newTag = tag;
    } else {
        if (deviceType_ == DevType::DEV_TYPE_310P3) {
            newTag = tag + algName;
        } else {
            auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType_.algoLevel1);
            CHK_PRT_RET(
                level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(),
                HCCL_ERROR("level1: algType1[%u] is invalid.", algType_.algoLevel1), HCCL_E_INTERNAL);
            newTag = tag + level1Iter->second + algName;
        }

        bool isInlineReduce = IsSupportSDMAReduce(
            cclBufferManager_.GetInCCLbuffer().ptr(), cclBufferManager_.GetOutCCLbuffer().ptr(), param.DataDes.dataType,
            param.reduceType);
        bool isRdmaReduce = IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType);
        const std::string REDUCE_SCATTER_NO_INLINE = "_no_inline";
        newTag = (isInlineReduce && isRdmaReduce) ? newTag : newTag + REDUCE_SCATTER_NO_INLINE;
    }
    if (algName == "ReduceScatterARSFor91093Executor") {
        u32 ringSize = CalcOptimalIntraRingsize(
            param.DataDes.count, param.DataDes.dataType, HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
        newTag += std::to_string(ringSize);
    }
    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    return ret;
}

HcclResult ReduceScatterOperator::SelectAlgforMix(const OpParam& param, std::string& algName)
{
    (void)param;

    // 混合组网场景不支持规约保序
    if (IsNeedStrictMode(param)) {
        HCCL_ERROR("[ReduceScatterOperator][SelectAlgforMix] not support DETERMINISTIC_STRICT mode.");
        return HCCL_E_NOT_SUPPORT;
    }

    if (gcdDeviceNumPerAggregation_ > 1) {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
        HCCL_WARNING("[ReduceScatterOperator][SelectAlgforMix] only support NHR in AlgoLevel1 yet, "
                     "default is algType=NHR.");
        algName = "ReduceScatterMixExecutor";
    } else {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
        ;
        HCCL_WARNING("[ReduceScatterOperator][SelectAlgforMix] only support ring in AlgoComm yet, "
                     "default is algType=ring.");
        algName = "ReduceScatterComm";
    }

    HCCL_INFO("[SelectAlgforMix] ReduceScatter SelectAlgforMix is algName [%s].", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOperator::SelectAlgfor310P3(const OpParam& param, std::string& algName)
{
    if (HCCL_310P_DATA_SIZE_SMALL_COUNT < param.DataDes.count && param.DataDes.count <= HCCL_310P_DATA_SIZE_MID_COUNT
        && userRankSize_ <= HCCL_310P_SLIM_RING_MAX_SIZE) {
        algName = "ReduceScatterSlimRing";
    } else {
        algName = "ReduceScatterRing";
    }

    HCCL_INFO(
        "[SelectAlgfor310P3] ReduceScatter SelectAlgfor310P3 is algName [%s] DataDesCount [%llu]", algName.c_str(),
        param.DataDes.count);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOperator::SelectAlgfor910A(const OpParam& param, std::string& algName)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_4P_MESH || topoType_ == TopoType::TOPO_TYPE_2P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;

    const u32 RANK_SIZE_FOUR = 4;
    const u32 RANK_SIZE_EIGHT = 8;
    bool isOpbase = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    u64 dataSize = SIZE_TABLE[param.DataDes.dataType] * param.DataDes.count;
    if (isOpbase && serverNum_ == 1 && dataSize <= HCCL_SMALL_COUNT_256_KB
        && (userRankSize_ == RANK_SIZE_FOUR || userRankSize_ == RANK_SIZE_EIGHT)) {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
        algName = "ReduceScatterComm";
    } else if (isMeshTopo) {
        algName = "ReduceScatterMeshExecutor";
    } else if (isRingTopo) {
        algName = "ReduceScatterRingExecutor";
    } else {
        algName = "ReduceScatterComm";
    }
    HCCL_INFO("[SelectAlgfor910A] ReduceScatter SelectAlgfor910A is algName [%s].", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];

    bool isOnlyAiv = topoMatcher_->GetIsOnlyAivConfig();
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH
                      || topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING;

    u64 dataSize = param.DataDes.count * unitSize; // 单位：字节
    u64 cclBufferSize = cclBufferManager_.GetInCCLbufferSize() / userRankSize_;

    void* commInputPtr = nullptr;
    void* commOutputPtr = nullptr;
    u64 commInputSize = 0;
    u64 commOutputSize = 0;

    CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));
    CHK_RET(cclBufferManager_.GetOutCCLbuffer(commOutputPtr, commOutputSize));
    bool isServNumPowOfTwo = (serverNum_ > 0) && ((serverNum_ & (serverNum_ - 1)) == 0);
    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    bool isInlineReduce = IsSupportSDMAReduce(
        cclBufferManager_.GetInCCLbuffer().ptr(), cclBufferManager_.GetOutCCLbuffer().ptr(), param.DataDes.dataType,
        param.reduceType);

    if (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_STRICT
        && userRankSize_ > DETERMINISTIC_MIN_USER_RANK_SIZE) {
        if (multiModuleDiffDeviceNumMode_) {
            // 保序规约场景（多batch一致），当前不支持A2非对称场景
            HCCL_ERROR(
                "[SelectAlgfor910B] reduce order preservation only support"
                " Symmetry(multiModuleDiffDeviceNumMode_[%d]).",
                multiModuleDiffDeviceNumMode_);
            return HCCL_E_NOT_SUPPORT;
        }
        if (param.DataDes.dataType == HCCL_DATA_TYPE_FP16 || param.DataDes.dataType == HCCL_DATA_TYPE_FP32
            || param.DataDes.dataType == HCCL_DATA_TYPE_BFP16) {
            // 只有浮点数存在多batch不一致的可能，整数天然一致
            if (param.aicpuUnfoldMode || topoMatcher_->GetAivModeConfig()) {
                // AIV / AICPU场景，规约保序优先级更高
                HCCL_WARNING(
                    "[SelectAlgfor910B]aicpuMode[%d], AivModeConfig[%d], "
                    "the AIV/AICPU mode does not support when the reduce order preservation is enabled.",
                    param.aicpuUnfoldMode, topoMatcher_->GetAivModeConfig());
            }
            algName = "ReduceScatterOrderPreservedExecutor";
            HCCL_INFO("[SelectAlgfor910B] ReduceScatterSelectAlgfor910B is algName [%s].", algName.c_str());
            return HCCL_SUCCESS;
        }
    }

    // 暂只支持单算子模式
    bool isCCLBufferGE16M
        = isOpbase && (commInputSize >= HCCL_MID_COUNT_16_MB && commOutputSize >= HCCL_MID_COUNT_16_MB);

    bool isSupportAivRdmaCount = !isSingleMeshAggregation_ && !multiModuleDiffDeviceNumMode_ && isMeshTopo
                                 && (((isServNumPowOfTwo || dataSize <= HCCL_SMALL_COUNT_128_KB)
                                      && dataSize * userRankSize_ <= HCCL_MID_COUNT_16_MB && isCCLBufferGE16M
                                      && dataSize <= HCCL_SMALL_COUNT_256_KB)
                                     || isOnlyAiv);

    bool isSupportAivDeter = isSingleMeshAggregation_ && serverNum_ == 1
                             && (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_ENABLE)
                             && ((dataSize * userRankSize_ <= HCCL_SMALL_COUNT_8_MB) || isOnlyAiv);

    bool isAivMode
        = topoMatcher_->GetAivModeConfig() && IsSupportAIVReduce(param.DataDes.dataType, param.reduceType)
          && (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE || isSupportAivDeter)
          && ((isSingleMeshAggregation_ && (dataSize <= AIV_BIG_SIZE || isOnlyAiv)) || isSupportAivRdmaCount);
    if (isAivMode) {
        if (isSupportAivDeter) {
            if (dataSize * userRankSize_ <= HCCL_SMALL_COUNT_8_MB) {
                algName = "ReduceScatterAivDeterSmallExecutor";
            } else {
                algName = "ReduceScatterAivDeterExecutor";
            }
            HCCL_INFO("[SelectAlgfor910B] ReduceScatter SelectAlgfor910B is algName [%s].", algName.c_str());
            return HCCL_SUCCESS;
        }
        if (isSupportAivRdmaCount) {
            algName = "ReduceScatterAivRdmaExecutor";
        } else if (
            workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && dataSize <= AIV_REDUCE_SCATTER_MID_SIZE) {
            algName = "ReduceScatterMeshAivSmallCountExecutor";
        } else {
            algName = "ReduceScatterMeshAivExecutor";
        }
        HCCL_INFO("[SelectAlgfor910BAIV] ReduceScatterSelectAlgfor910B is algName [%s].", algName.c_str());
        return HCCL_SUCCESS;
    }

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        bool isRdmaReduce = IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType);

        std::string algTypeLevel1Tag;
        CHK_RET(AutoSelectAlgTypeLevel1(
            HcclCMDType::HCCL_CMD_REDUCE_SCATTER, dataSize, cclBufferSize, algTypeLevel1Tag, isInlineReduce,
            isRdmaReduce));
        if (GetExternalInputHcclEnableEntryLog() && param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(algTypeLevel1Tag, param.tag));
        }
    }

    // AHC 算法选择逻辑
    if (((algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC)
         || (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE))) {
        CHK_RET(SelectAlgforAHC(dataSize, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER));
    }

    // pipeline算法task数量多，如果超出FFTS子图限制，则重定向到HD算法
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
        u32 contextNum = CalcContextNumForPipeline(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
        if (contextNum > HCCL_FFTS_CAPACITY) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
            HCCL_WARNING(
                "[ReduceScatterOperator][SelectAlgfor910B] context num[%u] is out of capacity of FFTS+ "
                "graph[%u], reset algorithm to HD.",
                contextNum, HCCL_FFTS_CAPACITY);
        }
    }

    if (isMeshTopo) {
        if (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_ENABLE
            && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
            && algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE && deviceNumPerAggregation_ > DEVICE_TWO) {
            algName = "ReduceScatterDeterPipelineExecutor";
        } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            bool enableSmallCountDeterministicAlgo
                = !isSingleMeshAggregation_
                  && IsSupportSDMAReduce(
                      cclBufferManager_.GetInCCLbuffer().ptr(), cclBufferManager_.GetOutCCLbuffer().ptr(),
                      param.DataDes.dataType, param.reduceType);
            if (SingleMeshInlineReduce(
                    cclBufferManager_.GetInCCLbuffer().ptr(), cclBufferManager_.GetOutCCLbuffer().ptr(),
                    param.DataDes.dataType, param.reduceType)) {
                if (topoMatcher_->GetDeterministicConfig() != DETERMINISTIC_DISABLE) {
                    algName = "ReduceScatterDeterExecutor";
                } else {
                    algName = "ReduceScatterMeshDmaEliminationExecutor";
                }
            } else if (
                topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE
                && algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE
                && IsMultiMeshInlineReduce(
                    cclBufferManager_.GetInCCLbuffer().ptr(), cclBufferManager_.GetOutCCLbuffer().ptr(),
                    param.DataDes.dataType, param.reduceType)) {
                algName = "ReduceScatterMeshOpbasePipelineExecutor";
            } else if (
                enableSmallCountDeterministicAlgo
                && ((dataSize <= HCCL_SMALL_COUNT_512_KB
                     && topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_ENABLE)
                    || dataSize * userRankSize_ < HCCL_SMALL_COUNT_512_KB)) {
                algName = "ReduceScatterMeshOpbaseSmallCountDeterministicExecutor";
            }
        } else {
            if (SingleMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType)) {
                if (topoMatcher_->GetDeterministicConfig() != DETERMINISTIC_DISABLE
                    && deviceNumPerAggregation_ > DEVICE_TWO) {
                    algName = "ReduceScatterDeterExecutor";
                } else {
                    if (dataSize <= HCCL_SMALL_COUNT_1_MB
                        && workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
                        algName = "ReduceScatterMeshGraphExecutor";
                    } else {
                        algName = "ReduceScatterMeshExecutor";
                    }
                }
            }
        }
        if (algName.empty()) {
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB && moduleNum_ > 1
                && deviceNumPerAggregation_ > 1 && topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE
                && IsMultiMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType)
                && (dataSize > HCCL_SMALL_COUNT_1_MB || moduleNum_ <= MODULE_NUM_FOUR
                    || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE)) {
                algName = "ReduceScatterMeshGraphPipelineExecutor";
            } else if (
                workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE || dataSize > HCCL_SMALL_COUNT_1_MB) {
                algName = "ReduceScatterMeshExecutor";
            } else if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
                algName = "ReduceScatterMeshGraphExecutor";
            }
        }
    } else if (isRingTopo) {
        algName = "ReduceScatterRingExecutor";
    } else {
        algName = "ReduceScatterComm";
    }
    // 如果配置了aiv only,但是实际没有选择aiv算法,需要通过DFX打印出具体原因
    if (isOnlyAiv && !isAivMode) {
        HCCL_ERROR("The current conditions do not meet the aiv only execution criteria because:");
        CHK_PRT_RET(
            !IsSupportAIVReduce(param.DataDes.dataType, param.reduceType),
            HCCL_ERROR(
                "current data type[%s] or reduceType[%s] not supported, "
                "data type support range:[int8, int16, int32, float16, float32, bfloat16] reduce type support "
                "range:[sum, max, min]",
                GetDataTypeEnumStr(param.DataDes.dataType).c_str(), GetReduceOpEnumStr(param.reduceType).c_str()),
            HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(
            !isSupportAivDeter,
            HCCL_ERROR(
                "is not support aiv deter.isSingleMeshAggregation_[%d] isOpbase[%d] "
                "deterministic config[%u] dataSize[%llu], serverNum_[%u]",
                isSingleMeshAggregation_, isOpbase, topoMatcher_->GetDeterministicConfig(), dataSize, serverNum_),
            HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(
            !isSingleMeshAggregation_ && multiModuleDiffDeviceNumMode_,
            HCCL_ERROR(
                "The number of cards between servers in a multi-server setup must be consistent. "
                "isSingleMeshAggregation_[%d] multiModuleDiffDeviceNumMode_[%d]",
                isSingleMeshAggregation_, multiModuleDiffDeviceNumMode_),
            HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(!isMeshTopo, HCCL_ERROR("current topo type[%d] not supported", topoType_), HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(!isServNumPowOfTwo, HCCL_ERROR("server num[%u] is pow of two.", serverNum_), HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(
            !isCCLBufferGE16M,
            HCCL_ERROR(
                "current isOpbase[%d] or commInputSize[%llu] or commOutputSize[%llu] not supported", isOpbase,
                commInputSize, commOutputSize),
            HCCL_E_NOT_SUPPORT);
        HCCL_ERROR(
            "isSingleMeshAggregation_[%d] multiModuleDiffDeviceNumMode_[%d] dataSize[%llu]", isSingleMeshAggregation_,
            multiModuleDiffDeviceNumMode_, dataSize);
        return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO(
        "[SelectAlgfor910B] ReduceScatter SelectAlgfor910B is algName [%s], current mode is [%u].", algName.c_str(),
        workflowMode_);
    return HCCL_SUCCESS;
}

HcclResult
ReduceScatterOperator::SelectAlgfor91093(const OpParam& param, std::string& algName, const ResourceLimit& limit)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 dataSize = param.DataDes.count * unitSize; // 单位：字节
    if (dataSize >= cclBufferManager_.GetInCCLbufferSize()) {
        HCCL_WARNING(
            "The current inCCLbufferSize is [%llu] bytes, change the HCCL_BUFFSIZE environment variable "
            "to be greater than the current data volume[%llu] bytes to improve the performance of the 91093 "
            "environment.",
            cclBufferManager_.GetInCCLbufferSize(), dataSize);
    }
    bool isOnlyAiv = topoMatcher_->GetIsOnlyAivConfig();
    bool isOpbase = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;

    bool isAivCrossNode = superPodNum_ == 1 && serverNum_ > 1 && !GetExternalInputInterHccsDisable()
                          && ((
                              (userRankSize_ <= ONE_EIGHTH_MAX_NUM_BLOCKS
                               && dataSize <= AIV_REDUCE_SCATTER_A3_SMALL_RANKSIZE_ENTRY_SIZE)
                              || (userRankSize_ <= ONE_THIRD_MAX_NUM_BLOCKS
                                  && dataSize <= AIV_REDUCE_SCATTER_A3_MID_RANKSIZE_ENTRY_SIZE)
                              || (dataSize <= AIV_REDUCE_SCATTER_A3_LARGE_RANKSIZE_ENTRY_SIZE) || isOnlyAiv));

    bool isAivSingleNode = serverNum_ == 1
                           && ((isOpbase && (dataSize <= AIV_REDUCE_SCATTER_A3_ENTRY_SIZE || isOnlyAiv))
                               || (!isOpbase && (dataSize <= AIV_REDUCE_SCATTER_A3_GRAPH_ENTRY_SIZE || isOnlyAiv)));

    // A3 AIV 确定性 超节点内(单机与跨机) 支持单算子与图模式 限制单卡数据量8MB
    bool isSupportAivDeter = (superPodNum_ == 1) && topoMatcher_->GetAivModeConfig()
                             && IsSupportAIVReduce(param.DataDes.dataType, param.reduceType)
                             && (topoMatcher_->GetDeterministicConfig() != DETERMINISTIC_DISABLE)
                             && ((dataSize * userRankSize_ < HCCL_SMALL_COUNT_8_MB) || isOnlyAiv) && (!retryEnable_)
                             && userRankSize_ > 1 && !multiModuleDiffDeviceNumMode_;

    bool isAivMode = topoMatcher_->GetAivModeConfig() && IsSupportAIVReduce(param.DataDes.dataType, param.reduceType)
                     && (isAivSingleNode || isAivCrossNode)
                     && (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE) && (!retryEnable_)
                     && !multiModuleDiffDeviceNumMode_;

    if (isSupportAivDeter) {
        algName = "ReduceScatterMeshAivFor91093Executor";
        HCCL_INFO("[SelectAlgfor91093] reduce_scatter SelectAlgfor91093 algName [%s]", algName.c_str());
        return HCCL_SUCCESS;
    }

    if (IsNeedStrictMode(param)) {
        CHK_PRT_RET(
            !CheckStrictCondition(param),
            HCCL_ERROR("[ReduceScatterOperator][SelectAlgfor91093] not support DETERMINISTIC_STRICT mode."),
            HCCL_E_NOT_SUPPORT);

        algName = "ReduceScatterOrderPreservedFor91093Executor";
        HCCL_INFO("[SelectAlgfor91093] reduce_scatter SelectAlgfor91093 algName [%s].", algName.c_str());
        return HCCL_SUCCESS;
    }

    if (isAivMode) {
        if (isAivCrossNode) {
            algName = "ReduceScatterMeshAivFor91093Executor";
        } else if (
            (isOpbase && dataSize <= AIV_REDUCE_SCATTER_SMALL_SIZE)
            || (!isOpbase
                && dataSize <= std::min(limit.aivCoreLimit / userRankSize_, NUM_BLOCKS_FACTOR_FOUR)
                                   * AIV_REDUCE_SCATTER_BIG_SIZE)) {
            algName = "ReduceScatterMeshAivSmallCountExecutor";
        } else {
            algName = "ReduceScatterMeshAivExecutor";
        }
        HCCL_INFO("[SelectAlgfor91093] ReduceScatter SelectAlgfor91093 is algName [%s]", algName.c_str());
        return HCCL_SUCCESS;
    }
    // ARS 算法选择
    bool isARSAlgo = multiModuleDiffDeviceNumMode_ && !multiSuperPodDiffDeviceNumMode_;
    if (isARSAlgo) {
        if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB
              || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING)) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_WARNING("[ReduceScatterOperator][SelectAlgfor91093] ARS only support NHR or RING in AlgoLevel1 "
                         "yet, default is NHR.");
        }
    }
    // AHC 算法选择逻辑
    bool isAHCAlgo = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC)
                     || (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);
    if (isAHCAlgo) {
        CHK_RET(SelectAlgforAHC(dataSize, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER));
    }

    bool isSupportInlineReduce
        = IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);
    bool isPowOfTwo = ((userRankSize_ - 1) & userRankSize_) == 0;
    void* commInputPtr = nullptr;
    u64 commInputSize = 0;
    s32 HCCS_PORT_NUM_910_93_7 = 7;
    u64 smallCountSingleServerThreshold
        = (hccsPortNum_ == HCCS_PORT_NUM_910_93_7) ? HCCL_SMALL_COUNT_512_KB : HCCL_SMALL_COUNT_1_MB;
    u64 smallCountMultiServerThreshold
        = (hccsPortNum_ == HCCS_PORT_NUM_910_93_7) ? HCCL_SMALL_COUNT_1_MB : HCCL_SMALL_COUNT_2_MB;
    CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));
    u64 maxPipelineBlockSize = 0;
    if (userRankSize_ != 0) {
        maxPipelineBlockSize
            = commInputSize / userRankSize_ / HCCL_DEVICE_NUM_TWO / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
    }
    bool isSupportPipelineFor91093 = (maxPipelineBlockSize >= HCCL_SMALL_COUNT_4_MB)
                                     && (dataSize * userRankSize_ > HCCL_PIPELINE_TOTAL_DATA_SIZE_THRESHOLD);
    HCCL_INFO(
        "[ReduceScatterOperator][SelectAlgfor91093] dataSize[%llu] commInputSize[%llu] "
        "userRankSize[%u] maxPipelineBlockSize[%llu] isSupportPipelineFor91093[%d]",
        dataSize, commInputSize, userRankSize_, maxPipelineBlockSize, isSupportPipelineFor91093);
    bool dmaReduceLimit = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) && isPowOfTwo
                          && ((commInputSize * HCCL_DEVICE_NUM_TWO
                               < param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] * userRankSize_)
                              || retryEnable_);
    bool smallCountOptimSingleServer
        = (!retryEnable_) && (serverNum_ == 1)
          && ((workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
              || (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !param.aicpuUnfoldMode))
          && isSupportInlineReduce && (deviceNumPerAggregation_ > HCCL_DEVICE_NUM_TWO)
          && (param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] <= smallCountSingleServerThreshold)
          && !GetExternalInputInterHccsDisable() && !dmaReduceLimit;
    bool smallCountOptimMultiServer = isSupportInlineReduce && (deviceNumPerAggregation_ > HCCL_DEVICE_NUM_TWO)
                                      && (serverNum_ != 1) && (superPodNum_ == 1) && !dmaReduceLimit
                                      && !GetExternalInputInterHccsDisable();
    bool isHccsPlusSio = userRankSize_ == 2 && pairLinkCounter_[static_cast<u32>(LinkTypeInServer::SIO_TYPE)] == 2
                         && pairLinkCounter_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)] == 0;
    bool useHostComm
        = !isSupportInlineReduce
          && ((serverNum_ != 1 && superPodNum_ == 1 && !GetExternalInputInterHccsDisable())
              || ((superPodNum_ > 1 || GetExternalInputInterHccsDisable()) && !retryEnable_
                  && ((isPowOfTwo && param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] <= HCCL_SMALL_COUNT_4_MB)
                      || (!isPowOfTwo
                          && param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] <= HCCL_SMALL_COUNT_2_MB))));
    bool smallCountOptimMultiPod = false;

    isHccsPlusSio = false;
    if (isHccsPlusSio && isSupportHccsAndSio_) {
        algName = "ReduceScatterHccsSioExecutor";
    } else if (multiModuleDiffDeviceNumMode_ && multiSuperPodDiffDeviceNumMode_) {
        algName = "ReduceScatterComm";
    } else if (multiModuleDiffDeviceNumMode_ && !multiSuperPodDiffDeviceNumMode_) {
        algName = "ReduceScatterARSFor91093Executor";
    } else if (
        smallCountOptimMultiPod || useHostComm
        || (smallCountOptimMultiServer && !isPowOfTwo
            && (param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] <= HCCL_SMALL_COUNT_256_KB))) {
        algName = "ReduceScatterComm";
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
    } else if (
        smallCountOptimSingleServer
        || (smallCountOptimMultiServer && isPowOfTwo
            && (param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] * serverNum_
                <= smallCountMultiServerThreshold))) {
        algName = "ReduceScatterDeterExecutor";
    } else if (
        isSupportInlineReduce && (param.supportSymmetricMemory || param.supportZeroCopy)
        && // isSupportInlineReduce：不申请scratch ==> 不支持非InlineReduce
        (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING
         || param.DataDes.count * unitSize * deviceNumPerAggregation_ > HCCL_MID_COUNT_16_MB)) {
        const u32 SEVER_NUM_FOUR = 4;
        constexpr u64 RING_EXCHANGE_PIPELINE_DATA_SIZE_MIN = 2 * 1024 * 1024;
        HcclAlgoType configAlgTypeLevel2
            = topoMatcher_->GetAlgoConfig(HcclCMDType::HCCL_CMD_REDUCE_SCATTER)[HCCL_ALGO_LEVEL_2];
        if ((superPodNum_ > 1) && (userRankSize_ / superPodNum_ > 1)
            && ((configAlgTypeLevel2 == HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE)
                || ((configAlgTypeLevel2 == HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT)
                    && (dataSize >= RING_EXCHANGE_PIPELINE_DATA_SIZE_MIN)))) {
            // 单算子, 超节点数大于1， 每个超节点的rank数大于1
            algName = "ReduceScatterRingZerocopyExchangePipelineExecutor"; // 连续数据通信+数据交换+Pipeline
            algType_.algoLevel2 = AlgTypeLevel2::ALG_LEVEL2_PIPELINE;
        } else if (serverNum_ < SEVER_NUM_FOUR || isAHCAlgo) {
            algName = "ReduceScatterRingZerocopyExecutor"; // 非连续数据通信（限制Server数，避免数据切太碎）
        } else {
            algName = "ReduceScatterRingZerocopyExchangeExecutor"; // 连续数据通信+数据交换（AHC不支持）
        }
    } else if (
        isOpbase && superPodNum_ > 1 && !isAHCAlgo && !multiSuperPodDiffDeviceNumMode_ && isSupportInlineReduce
        && (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING)
        && isSupportPipelineFor91093) {
        algName = "ReduceScatterPipelineFor91093Executor";
    } else {
        if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING) {
            algName = "ReduceScatterRingFor91093Executor";
        } else if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
            if (IsSupportUnifiedMarch(param, topoType_, serverNum_, superPodNum_)) {
                algName = "ReduceScatterSemiRingExecutor";
            } else {
                algName = "ReduceScatterFastDoubleRingFor91093Executor";
            }
        } else {
            algName = "ReduceScatterComm";
        }
    }

    if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB
          || (algType_.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING
              && algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING)
          || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
          || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE)
        && (algName != "ReduceScatterComm" && algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD)) {
        // 910_93超节点只支持server间ring,NB和NHR，默认需继续使用NHR
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
        HCCL_WARNING("[ReduceScatterOperator][SelectAlgfor91093] only support ring, NB AHC and NHR in AlgoLevel1 yet, "
                     "default is algType=NHR.");
    }
    // 如果配置了aiv only,但是实际没有选择aiv算法,需要通过DFX打印出具体原因
    if (isOnlyAiv && !isAivMode && !isSupportAivDeter) {
        HCCL_ERROR("The current conditions do not meet the aiv only execution criteria because:");
        CHK_PRT_RET(
            !IsSupportAIVReduce(param.DataDes.dataType, param.reduceType),
            HCCL_ERROR(
                "current data type[%s] or reduceType[%s] not supported, "
                "data type support range:[int8, int16, int32, float16, float32, bfloat16] reduce type support "
                "range:[sum, max, min]",
                GetDataTypeEnumStr(param.DataDes.dataType).c_str(), GetReduceOpEnumStr(param.reduceType).c_str()),
            HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(retryEnable_, HCCL_ERROR("retryEnable [%d] not supported", retryEnable_), HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(
            superPodNum_ != 1, HCCL_ERROR("multi superpod [%u] not supported", superPodNum_), HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(
            multiModuleDiffDeviceNumMode_,
            HCCL_ERROR("multiModuleDiffDeviceNumMode [%d] not supported", multiModuleDiffDeviceNumMode_),
            HCCL_E_NOT_SUPPORT);
        return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO("[SelectAlgfor91093] ReduceScatter SelectAlgfor91093 is algName [%s]", algName.c_str());

    HCCL_INFO(
        "[SelectAlgfor91093] isOpbase[%d] superPodNum_[%u] isAHCAlgo[%d] multiSuperPodDiffDeviceNumMode_[%d] "
        "isSupportInlineReduce[%d] topoType_[%d] dataSize[%llu]",
        isOpbase, superPodNum_, isAHCAlgo, multiSuperPodDiffDeviceNumMode_, isSupportInlineReduce, topoType_, dataSize);
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, ReduceScatter, ReduceScatterOperator);

} // namespace hccl
