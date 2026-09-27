/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_operator.h"
#include "device_capacity.h"
#include "coll_alg_utils.h"
#include "hccl_aiv.h"
#include "coll_alg_op_registry.h"

constexpr u32 MODULE_NUM_FOUR = 4;
constexpr u32 HCCL_310P_DATA_SIZE_MID_COUNT = 320 * 1024;
constexpr u32 HCCL_310P_DATA_SIZE_SMALL_COUNT = 1024;
constexpr u32 HCCL_310P_SLIM_RING_MAX_SIZE = 8;
constexpr u32 HCCL_91093_TOTAL_DATA_SIZE_FOR_PIPELINE = 637534208; // 608M
constexpr u64 ALLGATHER_PIPELINE_THRESHOLD = 4194304;              // 4MB

namespace hccl {
AllGatherOperator::AllGatherOperator(
    AlgConfigurator* algConfigurator, CCLBufferManager& cclBufferManager, HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLGATHER)
{}

AllGatherOperator::~AllGatherOperator() {}

HcclResult
AllGatherOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag)
{
    if (userRankSize_ == 1) {
        algName = "AllGatherSingleExecutor";
        HCCL_INFO("[SelectAlg] AllGather SelectAlg is algName [%s]", algName.c_str());
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
        ret = SelectAlgfor91093(param, algName);
    } else {
        HCCL_ERROR("[AllGatherSelector][SelectAlg] device type[%d] is out of range for selector.", deviceType_);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherSelector][SelectAlg]tag[%s], AllGather failed, return[%d]", tag.c_str(), ret), ret);
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        newTag = tag;
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        newTag = tag + algName;
    } else {
        AlgTypeLevel1 algType1 = algType_.algoLevel1;
        auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
        CHK_PRT_RET(
            level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(),
            HCCL_ERROR("[AllGatherSelector]level1: algType1[%u] is invalid.", algType1), HCCL_E_INTERNAL);
        newTag = tag + level1Iter->second + algName;
    }
    if (algName == "AllGatherARSFor91093Executor") {
        u32 ringSize
            = CalcOptimalIntraRingsize(param.DataDes.count, param.DataDes.dataType, HcclCMDType::HCCL_CMD_ALLGATHER);
        newTag += std::to_string(ringSize);
    }
    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    HCCL_DEBUG("[AllGatherSelector][SelectAlg]newTag is [%s].", newTag.c_str());
    return ret;
}

HcclResult AllGatherOperator::SelectAlgforMix(const OpParam& param, std::string& algName)
{
    (void)param;
    if (gcdDeviceNumPerAggregation_ > 1) {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
        HCCL_WARNING("[AllGatherOperator][SelectAlgforMix]only support NHR in AlgoLevel1 yet, "
                     "default is algType=NHR.");
        algName = "AllGatherMixExecutor";
    } else {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
        HCCL_WARNING("[AllGatherOperator][SelectAlgforMix]only support ring in AlgoComm yet, "
                     "default is algType=ring.");
        algName = "AllGatherComm";
    }

    HCCL_INFO("[SelectAlgforMix] AllGather SelectAlgforMix is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::SelectAlgfor310P3(const OpParam& param, std::string& algName)
{
    if (HCCL_310P_DATA_SIZE_SMALL_COUNT < param.DataDes.count && param.DataDes.count <= HCCL_310P_DATA_SIZE_MID_COUNT
        && userRankSize_ <= HCCL_310P_SLIM_RING_MAX_SIZE) {
        algName = "AllGatherSlimRingFor310PExecutor";
    } else {
        algName = "AllGatherFor310PExecutor";
    }
    HCCL_INFO("[SelectAlgfor310P3] AllGather SelectAlgfor310P3 is algName [%s].", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::SelectAlgfor910A(const OpParam& param, std::string& algName)
{
    (void)param;
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_4P_MESH || topoType_ == TopoType::TOPO_TYPE_2P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;

    if (isMeshTopo) {
        algName = "AllGatherMeshExecutor";
    } else if (isRingTopo) {
        algName = "AllGatherRingExecutor";
    } else {
        algName = "AllGatherComm";
    }
    HCCL_INFO("[SelectAlgfor910A] AllGather SelectAlgfor910A is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 dataSize = param.DataDes.count * unitSize; // 单位：字节
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH
                      || topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING;
    bool isOnlyAiv = topoMatcher_->GetIsOnlyAivConfig();
    bool isAivMode = topoMatcher_->GetAivModeConfig() && isSingleMeshAggregation_
                     && IsSupportAIVCopy(param.DataDes.dataType) && (dataSize <= AIV_BIG_SIZE || isOnlyAiv);
    bool isA2APreAG = (param.tag == HCCL_ALLTOALL_PARA_ALLGATHER);
    if (isAivMode && !isA2APreAG) {
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && dataSize <= AIV_ALL_GATHER_SMALL_SIZE) {
            algName = "AllGatherMeshAivSmallCountExecutor";
            HCCL_INFO("[SelectAlgfor910BAIV] AllGather SelectAlgfor910B is algName [%s]", algName.c_str());
        } else {
            algName = "AllGatherMeshAivExecutor";
            HCCL_INFO("[SelectAlgfor910BAIV] AllGather SelectAlgfor910B is algName [%s]", algName.c_str());
        }
        return HCCL_SUCCESS;
    }

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !isSingleMeshAggregation_) {
        u64 cclBufferSize = cclBufferManager_.GetOutCCLbufferSize() / userRankSize_;
        std::string algTypeLevel1Tag;
        CHK_RET(AutoSelectAlgTypeLevel1(HcclCMDType::HCCL_CMD_ALLGATHER, dataSize, cclBufferSize, algTypeLevel1Tag));
        if (GetExternalInputHcclEnableEntryLog() && param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(algTypeLevel1Tag, param.tag));
        }
    }

    // AHC 算法选择逻辑
    if (((algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC)
         || (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE))) {
        CHK_RET(SelectAlgforAHC(dataSize, AHCOpType::AHC_OP_TYPE_ALLGATHER));
    }

    // pipeline算法task数量多，如果超出FFTS子图限制，则重定向到HD算法
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
        u32 contextNum = CalcContextNumForPipeline(HcclCMDType::HCCL_CMD_ALLGATHER);
        if (contextNum > HCCL_FFTS_CAPACITY) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
            HCCL_WARNING(
                "[AllGatherOperator][SelectAlgfor910B] context num[%u] is out of capacity of FFTS+ graph[%u], "
                "reset algorithm to HD.",
                contextNum, HCCL_FFTS_CAPACITY);
        }
    }

    // 多机场景下aiv支持情况
    void* commInputPtr = nullptr;
    void* commOutputPtr = nullptr;
    u64 commInputSize = 0;
    u64 commOutputSize = 0;

    CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));
    CHK_RET(cclBufferManager_.GetOutCCLbuffer(commOutputPtr, commOutputSize));
    bool isServNumPowOfTwo = (serverNum_ > 0) && ((serverNum_ & (serverNum_ - 1)) == 0);
    bool isSupportAivRdmaCount
        = !isSingleMeshAggregation_ && !multiModuleDiffDeviceNumMode_
          && (((isServNumPowOfTwo || dataSize <= HCCL_SMALL_COUNT_128_KB)
               && dataSize * userRankSize_ <= HCCL_MID_COUNT_16_MB && dataSize <= HCCL_SMALL_COUNT_256_KB)
              || isOnlyAiv);

    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    // 暂只支持单算子模式
    bool isCCLBufferGE16M = isOpbase && commInputSize >= HCCL_MID_COUNT_16_MB && commOutputSize >= HCCL_MID_COUNT_16_MB;

    bool isAivRdmaMode = topoMatcher_->GetAivModeConfig() && IsSupportAIVCopy(param.DataDes.dataType) && isMeshTopo
                         && isCCLBufferGE16M && isSupportAivRdmaCount;
    if (isAivRdmaMode && !isA2APreAG) {
        algName = "AllGatherAivRdmaExecutor";
    } else if (isMeshTopo) {
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            if (isSingleMeshAggregation_) {
                algName = "AllGatherMeshOpbaseExecutor";
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
                algName = "AllGatherMeshOpbasePipelineExecutor";
            }
        }
        if (algName.empty()) {
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB && moduleNum_ > 1
                && deviceNumPerAggregation_ > 1
                && (dataSize > HCCL_SMALL_COUNT_1_MB || moduleNum_ <= MODULE_NUM_FOUR
                    || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE)) {
                algName = "AllGatherMeshGraphPipelineExecutor";
            } else if (
                workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE || dataSize > HCCL_SMALL_COUNT_1_MB) {
                algName = "AllGatherMeshExecutor";
            } else if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
                algName = "AllGatherMeshGraphExecutor";
            }
        }
    } else if (isRingTopo) {
        algName = "AllGatherRingExecutor";
    } else {
        algName = "AllGatherComm";
    }
    // 如果配置了aiv only,但是实际没有选择aiv算法,需要通过DFX打印出具体原因
    if (isOnlyAiv && !isAivRdmaMode) {
        HCCL_ERROR("The current conditions do not meet the aiv only execution criteria because:");
        CHK_PRT_RET(
            !IsSupportAIVCopy(param.DataDes.dataType),
            HCCL_ERROR(
                "current data type[%s] not supported, support range: "
                "[int8, int16, int32, uint8, uint16, uint32, float16, float32, bfloat16]",
                GetDataTypeEnumStr(param.DataDes.dataType).c_str()),
            HCCL_E_NOT_SUPPORT);
        CHK_PRT_RET(!isMeshTopo, HCCL_ERROR("current topo type[%d] not supported", topoType_), HCCL_E_NOT_SUPPORT);
        CHK_PRT_RET(
            !isCCLBufferGE16M,
            HCCL_ERROR(
                "current isOpbase[%d] or commInputSize[%llu] or commOutputSize[%llu] not supported", isOpbase,
                commInputSize, commOutputSize),
            HCCL_E_NOT_SUPPORT);
        CHK_PRT_RET(
            !isSingleMeshAggregation_ && multiModuleDiffDeviceNumMode_,
            HCCL_ERROR(
                "The number of cards between servers in a multi-server setup must be consistent. "
                "isSingleMeshAggregation_[%d] multiModuleDiffDeviceNumMode_[%d]",
                isSingleMeshAggregation_, multiModuleDiffDeviceNumMode_),
            HCCL_E_NOT_SUPPORT);
        return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO(
        "[SelectAlgfor910B] AllGather SelectAlgfor910B is algName [%s], current mode is [%u].", algName.c_str(),
        workflowMode_);
    return HCCL_SUCCESS;
}

bool AllGatherOperator::SmallCountOptimSinglePod(const OpParam& param)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 totalSize = param.DataDes.count * unitSize * userRankSize_;
    void* commInputPtr = nullptr;
    u64 commInputSize = 0;
    CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));

    bool smallCountOptimSingleServer
        = (serverNum_ == 1)
          && ((workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && commInputSize >= totalSize)
              || (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !param.aicpuUnfoldMode))
          && (param.DataDes.count * unitSize <= HCCL_SMALL_COUNT_512_KB)
          && (deviceNumPerAggregation_ > HCCL_DEVICE_NUM_TWO) && !GetExternalInputInterHccsDisable();

    bool dmaReduceLimit
        = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
          && (((deviceNumPerAggregation_ % HCCL_DEVICE_NUM_FOUR == 0)
               && (commInputSize * HCCL_DEVICE_NUM_FOUR < totalSize))
              || ((deviceNumPerAggregation_ % HCCL_DEVICE_NUM_TWO == 0)
                  && (commInputSize * HCCL_DEVICE_NUM_TWO < totalSize))
              || ((deviceNumPerAggregation_ % HCCL_DEVICE_NUM_TWO != 0) && (commInputSize < totalSize)));
    bool smallCountOptimMultiServer
        = (deviceNumPerAggregation_ > HCCL_DEVICE_NUM_TWO) && (serverNum_ != 1) && (superPodNum_ == 1)
          && (((deviceNumPerAggregation_ % HCCL_DEVICE_NUM_FOUR == 0)
               && (param.DataDes.count * unitSize * serverNum_ <= HCCL_SMALL_COUNT_1_MB))
              || ((deviceNumPerAggregation_ % HCCL_DEVICE_NUM_FOUR != 0)
                  && (param.DataDes.count * unitSize * serverNum_ <= HCCL_SMALL_COUNT_512_KB)))
          && !dmaReduceLimit && !GetExternalInputInterHccsDisable();
    return smallCountOptimSingleServer || smallCountOptimMultiServer;
}

HcclResult AllGatherOperator::SelectAlgfor91093(const OpParam& param, std::string& algName)
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

    bool isAivCrossNode
        = superPodNum_ == 1 && serverNum_ > 1 && !GetExternalInputInterHccsDisable()
          && (((userRankSize_ <= ONE_EIGHTH_MAX_NUM_BLOCKS && dataSize <= AIV_ALL_GATHER_A3_SMALL_RANKSIZE_ENTRY_SIZE)
               || (userRankSize_ <= ONE_THIRD_MAX_NUM_BLOCKS && dataSize <= AIV_ALL_GATHER_A3_MID_RANKSIZE_ENTRY_SIZE)
               || (dataSize <= AIV_ALL_GATHER_A3_LARGE_RANKSIZE_ENTRY_SIZE))
              || isOnlyAiv);

    bool isAivSingleNode = (serverNum_ == 1)
                           && ((isOpbase && (dataSize <= AIV_ALL_GATHER_A3_ENTRY_SIZE || isOnlyAiv))
                               || (!isOpbase && (dataSize <= AIV_ALL_GATHER_A3_GRAPH_ENTRY_SIZE || isOnlyAiv)));

    bool isAivMode = topoMatcher_->GetAivModeConfig() && IsSupportAIVCopy(param.DataDes.dataType)
                     && (isAivSingleNode || isAivCrossNode) && !retryEnable_ && !multiModuleDiffDeviceNumMode_;
    if (isAivMode) {
        if (isAivCrossNode) {
            algName = "AllGatherMeshAivFor91093Executor";
        } else if (
            (isOpbase && dataSize <= AIV_ALL_GATHER_SMALL_SIZE)
            || (!isOpbase && dataSize <= AIV_A3_ALL_GATHER_GRAPH_GUIYI_SIZE)) {
            algName = "AllGatherMeshAivSmallCountExecutor"; // 目前a3 aivmode下单算子模式正好全走小数据
        } else {
            algName = "AllGatherMeshAivExecutor";
        }
        HCCL_INFO("[SelectAlgfor91093] AllGather SelectAlgfor91093 is algName [%s].", algName.c_str());
        return HCCL_SUCCESS;
    }

    bool smallCountOptimSinglePod = SmallCountOptimSinglePod(param);
    bool is2Pod2ServerTopo = (superPodNum_ == 2 && serverNum_ == 2); // 针对 A3背靠背机型
    bool smallCountOptimMultiPod = (superPodNum_ > 1 || (GetExternalInputInterHccsDisable() && serverNum_ > 1))
                                   && !is2Pod2ServerTopo && (param.DataDes.count * unitSize <= HCCL_SMALL_COUNT_16_KB)
                                   && !retryEnable_; // 涉及ROCE平面
    // 多超节点的中等数据量
    bool midCountOptimMultiPod = (superPodNum_ > 1) && isOpbase
                                 && (param.DataDes.count * unitSize <= HCCL_SMALL_COUNT_256_KB)
                                 && !retryEnable_; // 涉及ROCE平面

    // ARS 算法选择
    bool isARSAlgo = multiModuleDiffDeviceNumMode_ && !multiSuperPodDiffDeviceNumMode_;
    if (isARSAlgo) {
        if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB
              || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING)) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_WARNING("[AllGatherOperator][SelectAlgfor91093] ARS only support NHR or RING in AlgoLevel1 "
                         "yet, default is NHR.");
        }
    }
    // AHC 算法选择逻辑
    bool isAHCAlgo = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC)
                     || (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);
    if (isAHCAlgo) {
        CHK_RET(SelectAlgforAHC(dataSize, AHCOpType::AHC_OP_TYPE_ALLGATHER));
    }

    u64 maxSizePerLoop = cclBufferManager_.GetInCCLbufferSize() / HCCL_DEVICE_NUM_TWO / userRankSize_
                         / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;

    bool isHccsPlusSio = userRankSize_ == 2 && pairLinkCounter_[static_cast<u32>(LinkTypeInServer::SIO_TYPE)] == 2
                         && pairLinkCounter_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)] == 0;
    isHccsPlusSio = false;
    if (isHccsPlusSio && isSupportHccsAndSio_) {
        algName = "AllGatherHccsSioExecutor";
    } else if (multiModuleDiffDeviceNumMode_ && multiSuperPodDiffDeviceNumMode_) {
        algName = "AllGatherComm";
    } else if (multiModuleDiffDeviceNumMode_ && !multiSuperPodDiffDeviceNumMode_) {
        if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB
              || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING)) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_WARNING("[AllGatherOperator][SelectAlgfor91093] ARS only support NHR or RING in AlgoLevel1 "
                         "yet, default is NHR.");
        }
        algName = "AllGatherARSFor91093Executor";
    } else if (smallCountOptimMultiPod) {
        algName = "AllGatherComm";
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
    } else if (smallCountOptimSinglePod) {
        algName = "AllGatherSmallCount";
    } else if (midCountOptimMultiPod) {
        algName = "AllGatherMidCountFor91093Executor";
    } else if (
        (param.supportSymmetricMemory || param.supportZeroCopy)
        && (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING
            || param.DataDes.count * unitSize * deviceNumPerAggregation_ > HCCL_MID_COUNT_16_MB)) {
        const u32 SEVER_NUM_FOUR = 4;
        constexpr u64 RING_EXCHANGE_PIPELINE_DATA_SIZE_MIN = 2 * 1024 * 1024;
        HcclAlgoType configAlgTypeLevel2
            = topoMatcher_->GetAlgoConfig(HcclCMDType::HCCL_CMD_ALLGATHER)[HCCL_ALGO_LEVEL_2];
        bool setPipelineAlgo
            = ((configAlgTypeLevel2 == HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE)
               || (configAlgTypeLevel2 == HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT
                   && dataSize >= RING_EXCHANGE_PIPELINE_DATA_SIZE_MIN));
        if (superPodNum_ > 1 && userRankSize_ / superPodNum_ > 1 && setPipelineAlgo) {
            algName = "AllGatherRingZerocopyPipelineExecutor"; // 连续数据通信+额外的数据交换，Level2和level0+1并发流水
            algType_.algoLevel2 = AlgTypeLevel2::ALG_LEVEL2_PIPELINE;
        } else if (serverNum_ < SEVER_NUM_FOUR || isAHCAlgo) {
            algName = "AllGatherRingZerocopyExecutor"; // 非连续数据通信（限制Server数，避免数据切太碎）
        } else {
            algName = "AllGatherRingZerocopyExchangeExecutor"; // 连续数据通信+额外的数据交换（AHC不支持）
        }
    } else if (
        superPodNum_ > 1 && maxSizePerLoop >= ALLGATHER_PIPELINE_THRESHOLD
        && dataSize * userRankSize_ > HCCL_91093_TOTAL_DATA_SIZE_FOR_PIPELINE && isOpbase && !isAHCAlgo
        && !multiModuleDiffDeviceNumMode_
        && (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING)) {
        algName = "AllGatherPipelineFor91093Executor";
    } else {
        if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING
              || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB
              || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING
              || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
              || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE)) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_WARNING("[AllGatherOperator][SelectAlgfor91093] only support ring, NB AHC and NHR in AlgoLevel1 yet, "
                         "default is algType=NHR.");
        }
        if (IsSupportUnifiedMarch(param, topoType_, serverNum_, superPodNum_)) {
            algName = "AllGatherSemiRingExecutor";
        } else if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
            algName = "AlignedAllGatherDoubleRingFor91093Executor";
        } else if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING) {
            algName = "AllGatherRingFor91093Executor";
        } else {
            algName = "AllGatherComm";
        }
    }
    // 如果配置了aiv only,但是实际没有选择aiv算法,需要通过DFX打印出具体原因
    if (isOnlyAiv && !isAivMode) {
        HCCL_ERROR("The current conditions do not meet the aiv only execution criteria because:");
        CHK_PRT_RET(
            !IsSupportAIVCopy(param.DataDes.dataType),
            HCCL_ERROR(
                "current data type[%s] not supported, support range: "
                "[int8, int16, int32, uint8, uint16, uint32, float16, float32, bfloat16]",
                GetDataTypeEnumStr(param.DataDes.dataType).c_str()),
            HCCL_E_NOT_SUPPORT);
        CHK_PRT_RET(
            !isAivSingleNode && !isAivCrossNode,
            HCCL_ERROR(
                "not is aiv single or cross node. serverNum_[%u] isOpbase[%d] superPodNum_[%u]", serverNum_, isOpbase,
                superPodNum_),
            HCCL_E_NOT_SUPPORT);
        CHK_PRT_RET(retryEnable_, HCCL_ERROR("retryEnable_[%d] is true.", retryEnable_), HCCL_E_NOT_SUPPORT);
        CHK_PRT_RET(
            multiModuleDiffDeviceNumMode_,
            HCCL_ERROR("multiModuleDiffDeviceNumMode [%d] not supported", multiModuleDiffDeviceNumMode_),
            HCCL_E_NOT_SUPPORT);
        return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO("[SelectAlgfor91093] AllGather SelectAlgfor91093 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_ALLGATHER, AllGather, AllGatherOperator);

} // namespace hccl
