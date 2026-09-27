/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_reduce_operator.h"
#include "device_capacity.h"
#include "coll_alg_utils.h"
#include "hccl_aiv.h"
#include "coll_alg_op_registry.h"

namespace hccl {

// 保序规约仅支持多于2卡场景(2卡时收发对端唯一, 无规约顺序差异)
constexpr u32 DETERMINISTIC_MIN_USER_RANK_SIZE = 2;

AllReduceOperator::AllReduceOperator(
    AlgConfigurator* algConfigurator, CCLBufferManager& cclBufferManager, HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLREDUCE)
{}

AllReduceOperator::~AllReduceOperator() {}

// 如果逻辑有修改，需同步修改GetAllReduceScratchMemSize()
HcclDataCountType AllReduceOperator::GetCountTypeForDeterAllReduce(const u64 count, const HcclDataType dataType)
{
    u64 dataSize = SIZE_TABLE[dataType] * count;
    if ((GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB)) {
        if (dataSize <= HCCL_SMALL_COUNT_GRAPH_64_KB) {
            return HcclDataCountType::HCCL_COUNT_SMALL;
        } else if ((dataSize <= HCCL_MEDIUM_COUNT_GRAPH_4_MB) && (deviceNumPerAggregation_ == DEVICE_EIGHT)) {
            return HcclDataCountType::HCCL_COUNT_MEDIUM;
        } else {
            return HcclDataCountType::HCCL_COUNT_HUGE;
        }
    } else {
        if (dataSize <= HCCL_SMALL_COUNT_128_KB) {
            return HcclDataCountType::HCCL_COUNT_SMALL;
        } else {
            if (deviceNumPerAggregation_ == DEVICE_EIGHT) {
                return HcclDataCountType::HCCL_COUNT_MEDIUM;
            } else {
                return HcclDataCountType::HCCL_COUNT_HUGE;
            }
        }
    }
}

// 如果逻辑有修改，需同步修改GetAllReduceScratchMemSize()
HcclResult AllReduceOperator::GetScratchSizeForDeterAllReduce(
    const u64 count, const HcclDataType dataType, const u32 rankSize, u64& outScratchSize)
{
    // 两卡不需要申请额外内存
    if (rankSize == DEVICE_TWO) {
        outScratchSize = 0;
        return HCCL_SUCCESS;
    }

    HcclDataCountType countType = GetCountTypeForDeterAllReduce(count, dataType);
    u64 memSize = SIZE_TABLE[dataType] * count;
    switch (countType) {
        case HcclDataCountType::HCCL_COUNT_SMALL:
            // 小数据量下，八卡选择HD算法、非八卡选择Reduce-Bcast算法
            if (rankSize == DEVICE_EIGHT) {
                // one shot HD算法，需要额外的(log2(N)-1)倍内存避免读写冲突
                outScratchSize = 0;
            } else {
                // Reduce-Bcast算法，需要N-1倍内存来暂存来自其他卡的数据（先收集数据，再本地Reduce到目的内存上）
                outScratchSize = memSize * (rankSize - 1);
            }
            break;
        case HcclDataCountType::HCCL_COUNT_MEDIUM:
            // 中数据量下，八卡选择Local Reduce算法，非八卡选择MeshChunk算法，都不要额外内存
            outScratchSize = 0;
            break;
        case HcclDataCountType::HCCL_COUNT_HUGE:
            // 大数据量下，统一选择MeshChunk算法，不需要额外内存
            outScratchSize = 0;
            break;
        default:
            return HCCL_E_NOT_SUPPORT;
    }

    HCCL_DEBUG(
        "[GetScratchSizeForDeterAllReduce] countType=%u, rankSize=%u, memSize=%llu, outScratchSize=%llu", countType,
        rankSize, memSize, outScratchSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::GetAllReduceScratchSize(const u64 count, const HcclDataType dataType, u64& scratchSize)
{
    // 针对 单机、910B、确定性计算、图模式 的特殊优化
    if (algConfigurator_->SupportDeterministicOptim()) {
        CHK_RET(GetScratchSizeForDeterAllReduce(count, dataType, deviceNumPerAggregation_, scratchSize));
    } else {
        u64 reservedSize = (userRankSize_ + 1) * (userRankSize_ + 1) * SIZE_TABLE[dataType];

        scratchSize = count * SIZE_TABLE[dataType] * DEVICE_TWO + reservedSize;
    }

    HCCL_INFO("[AllReduceOperator][GetAllReduceScratchSize] scratchSize %llu, count %llu", scratchSize, count);
    return HCCL_SUCCESS;
}

HcclResult
AllReduceOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag)
{
    if (userRankSize_ == 1) {
        algName = "AllReduceSingleExecutor";
        HCCL_INFO("[SelectAlg] AllReduce SelectAlg is algName [%s]", algName.c_str());
        return HCCL_SUCCESS;
    }
    HcclResult ret;
    if (isDiffDeviceType_) {
        ret = SelectAlgforMix(param, algName);
    } else if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        if (is310PDuoCard_) {
            ret = SelectAlgfor310P3DUO(param, algName);
        } else {
            ret = SelectAlgfor310P3(param, algName);
        }
    } else if (Is310PDevice()) {
        ret = SelectAlgfor310PHelper(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        ret = SelectAlgfor910A(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910_93) {
        ret = SelectAlgfor91093(param, algName);
    } else {
        HCCL_ERROR("[AllReduceOperator][SelectAlg] device type[%d] is out of range for selector.", deviceType_);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllReduceSelector][SelectAlg]tag[%s], AllReduce failed, return[%d]", tag.c_str(), ret), ret);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
            newTag = tag + algName;
        } else {
            AlgTypeLevel1 algType1 = algType_.algoLevel1;
            auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
            CHK_PRT_RET(
                level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), HCCL_ERROR("level1: algType1[%u] is invalid.", algType1),
                HCCL_E_INTERNAL);
            newTag = tag + level1Iter->second + algName;
        }

        bool isInlineReduce = IsSupportSDMAReduce(
            cclBufferManager_.GetInCCLbuffer().ptr(), cclBufferManager_.GetOutCCLbuffer().ptr(), param.DataDes.dataType,
            param.reduceType);
        bool isRdmaReduce = IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType);
        const std::string ALL_REDUCE_NO_INLINE = "_no_inline";
        newTag = (!isDiffDeviceType_ || (isDiffDeviceType_ && isInlineReduce && isRdmaReduce)) ?
                     newTag :
                     newTag + ALL_REDUCE_NO_INLINE;
    } else {
        newTag = tag;
    }
    if (algName == "AllReduceARSFor91093Executor") {
        u32 ringSize
            = CalcOptimalIntraRingsize(param.DataDes.count, param.DataDes.dataType, HcclCMDType::HCCL_CMD_ALLREDUCE);
        newTag += std::to_string(ringSize);
    }
    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    return ret;
}

HcclResult AllReduceOperator::SelectAlgforMix(const OpParam& param, std::string& algName)
{
    (void)param;

    // 混合组网场景不支持规约保序
    if (IsNeedStrictMode(param)) {
        HCCL_ERROR("[AllReduceOperator][SelectAlgforMix] not support DETERMINISTIC_STRICT mode.");
        return HCCL_E_NOT_SUPPORT;
    }

    if (gcdDeviceNumPerAggregation_ > 1) {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
        HCCL_WARNING("[AllReduceOperator][SelectAlgforMix] only support NHR in AlgoLevel1 yet, "
                     "default is algType=NHR.");
        algName = "AllReduceMixExecutor";
    } else {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
        ;
        HCCL_WARNING("[AllReduceOperator][SelectAlgforMix] only support ring in AlgoComm yet, "
                     "default is algType=ring.");
        algName = "AllReduceComm";
    }

    HCCL_INFO("[SelectAlgforMix] AllReduce SelectAlgforMix is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::SelectAlgfor310P3DUO(const OpParam& param, std::string& algName)
{
    bool isInlineReduce
        = IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);
    u64 dataSize = SIZE_TABLE[param.DataDes.dataType] * param.DataDes.count;

    bool isPowOfTwo = ((userRankSize_ - 1) & userRankSize_) == 0;
    const u32 RANK_SIZE_TWO = 2;
    const u32 RANK_SIZE_EIGHT = 8;

    if (isInlineReduce) {
        if ((dataSize <= HCCL_SMALL_COUNT_256_KB && isPowOfTwo && userRankSize_ <= RANK_SIZE_EIGHT)
            || userRankSize_ == RANK_SIZE_TWO) {
            algType_.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_HD;
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
            if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                algName = "AllReduceDoublingDirect";
            } else {
                algName = "AllReduceDoubling";
            }
        }
    }
    if (algName.empty()) {
        algType_.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING;
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING;
        algName = "AllReduceRing";
    }
    HCCL_INFO("[SelectAlgfor310P3DUO] AllReduce SelectAlgfor310P3DUO is algName [%s].", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::SelectAlgfor310P3(const OpParam& param, std::string& algName)
{
    bool isPowOfTwo = ((userRankSize_ - 1) & userRankSize_) == 0;
    u64 dataSize = SIZE_TABLE[param.DataDes.dataType] * param.DataDes.count;

    bool isInlineReduce
        = IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);
    if (isInlineReduce) {
        if (dataSize <= HCCL_SMALL_COUNT_256_KB && isPowOfTwo) {
            algType_.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_HD;
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
            algName = "AllReduceDoubling";
        }
    }
    if (algName.empty()) {
        algType_.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING;
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING;
        algName = "AllReduceRing";
    }
    HCCL_INFO("[SelectAlgfor310P3] AllReduce SelectAlgfor310P3 is algName [%s].", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::SelectAlgfor310PHelper(const OpParam& param, std::string& algName)
{
    (void)param;
    algName = "AllReduceReducePlusBcast";
    HCCL_INFO("[SelectAlgfor310PHelper] AllReduce SelectAlgfor310PHelper is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::SelectAlgfor910A(const OpParam& param, std::string& algName)
{
    const u32 RANK_SIZE_FOUR = 4;
    const u32 RANK_SIZE_EIGHT = 8;
    u64 dataSize = SIZE_TABLE[param.DataDes.dataType] * param.DataDes.count;
    bool isOpbase = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    bool isInlineReduce
        = IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_4P_MESH || topoType_ == TopoType::TOPO_TYPE_2P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;

    if (isOpbase && serverNum_ == 1 && dataSize <= HCCL_SMALL_COUNT_1_MB
        && (userRankSize_ == RANK_SIZE_FOUR || userRankSize_ == RANK_SIZE_EIGHT)) {
        algType_.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_HD;
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
        if (isInlineReduce && userRankSize_ == RANK_SIZE_FOUR) {
            algName = "AllReduceDoublingDirect";
        } else if (isInlineReduce && userRankSize_ == RANK_SIZE_EIGHT) {
            algName = "AllReduceDoubling";
        } else {
            algName = "AllReduceSmallCountFor910";
        }
    } else if (isMeshTopo) {
        algName = "AllReduceMeshExecutor";
    } else if (isRingTopo) {
        algName = "AllReduceRingExecutor";
    } else {
        algName = "AllReduceComm";
    }
    HCCL_INFO("[SelectAlgfor910A] AllReduce SelectAlgfor910A is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];

    bool isOnlyAiv = topoMatcher_->GetIsOnlyAivConfig();
    bool isInlineReduce
        = IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);
    bool isRdmaReduce = IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType);

    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH
                      || topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING;

    u64 dataSize = param.DataDes.count * unitSize; // 单位：字节

    void* commInputPtr = nullptr;
    void* commOutputPtr = nullptr;
    u64 commInputSize = 0;
    u64 commOutputSize = 0;

    CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));
    CHK_RET(cclBufferManager_.GetOutCCLbuffer(commOutputPtr, commOutputSize));

    // aiv场景单独判断逻辑，满足AIV模式打开+支持AIVReduce+非确定性场景+外层为mesh+（单机/跨机小数据/跨机中数据）时进入分支
    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    bool isMesh = IsAlgTypeLevel0Mesh(algType_.algoLevel0);
    u64 rankCountSize = dataSize / deviceNumPerAggregation_;
    bool isServNumPowOfTwo = (serverNum_ > 0) && ((serverNum_ & (serverNum_ - 1)) == 0);

    bool isSupportAivRdmaSmallCount = !isSingleMeshAggregation_ && !multiModuleDiffDeviceNumMode_ && isServNumPowOfTwo
                                      && ((rankCountSize <= HCCL_SMALL_COUNT_190_KB || isOnlyAiv));

    bool isSupportAivRdmaMidCount
        = !isSingleMeshAggregation_ && !multiModuleDiffDeviceNumMode_ && (dataSize <= HCCL_MID_COUNT_16_MB);

    bool isSupportAivDeter = isSingleMeshAggregation_
                             && (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_ENABLE)
                             && (dataSize <= HCCL_SMALL_COUNT_8_MB);

    bool isCCLBufferGE16M
        = !isOpbase || (commInputSize >= HCCL_MID_COUNT_16_MB && commOutputSize >= HCCL_MID_COUNT_16_MB);

    bool isBarrierOp = param.syncMode == SyncMode::UNLIMITED_TIMEWAITSYNCMODE; // Barrier算子不使能AIV
    bool isAivMode = (topoMatcher_->GetAivModeConfig() && !isBarrierOp)
                     && IsSupportAIVReduce(param.DataDes.dataType, param.reduceType) && isMesh && isCCLBufferGE16M
                     && (isSingleMeshAggregation_ || isSupportAivRdmaSmallCount || isSupportAivRdmaMidCount)
                     && (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE || isSupportAivDeter);
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        std::string algTypeLevel1Tag;
        CHK_RET(AutoSelectAlgTypeLevel1(
            HcclCMDType::HCCL_CMD_ALLREDUCE, dataSize, commInputSize, algTypeLevel1Tag, isInlineReduce, isRdmaReduce,
            isAivMode));
        if (GetExternalInputHcclEnableEntryLog() && param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(algTypeLevel1Tag, param.tag));
        }
    }

    // AHC 算法选择逻辑
    if (((algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC)
         || (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE))) {
        CHK_RET(SelectAlgforAHC(dataSize, AHCOpType::AHC_OP_TYPE_ALLREDUCE));
    }

    // pipeline算法task数量多，如果超出FFTS子图限制，则重定向到HD算法
    // 图模式不会重定向到HD算法
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE
        && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        u32 contextNum = CalcContextNumForPipeline(HcclCMDType::HCCL_CMD_ALLREDUCE);
        if (contextNum > HCCL_FFTS_CAPACITY) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
            HCCL_WARNING(
                "[AllReduceOperator][SelectAlgfor910B] context num[%u] is out of capacity of FFTS+ graph[%u], "
                "reset algorithm to HD.",
                contextNum, HCCL_FFTS_CAPACITY);
        }
    }

    if (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_STRICT && multiModuleDiffDeviceNumMode_) {
        // 保序规约场景（多batch一致），当前不支持非对称场景
        HCCL_ERROR(
            "[SelectAlgfor910B] reduce order preservation only support Symmetry("
            "multiModuleDiffDeviceNumMode_[%d]).",
            multiModuleDiffDeviceNumMode_);
        return HCCL_E_NOT_SUPPORT;
    }

    if (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_STRICT
        && userRankSize_ > DETERMINISTIC_MIN_USER_RANK_SIZE
        && (param.DataDes.dataType == HCCL_DATA_TYPE_FP16 || param.DataDes.dataType == HCCL_DATA_TYPE_FP32
            || param.DataDes.dataType == HCCL_DATA_TYPE_BFP16)) {
        if (param.aicpuUnfoldMode || (topoMatcher_->GetAivModeConfig() && !isBarrierOp)) {
            // AIV / AICPU场景，规约保序优先级更高
            HCCL_WARNING(
                "[SelectAlgfor910B]aicpuMode[%d], AivModeConfig[%d], "
                "the Aiv/AICPU mode does not support when the reduce order preservation is enabled.",
                param.aicpuUnfoldMode, topoMatcher_->GetAivModeConfig());
        }
        // 只有浮点数存在多batch不一致的可能，整数天然一致
        algName = "AllReduceOrderPreservedExecutor";
    } else if (isAivMode) {
        if (isSupportAivDeter) {
            if (dataSize <= HCCL_SMALL_COUNT_8_MB) {
                algName = "AllReduceAivDeterSmallExecutor";
            } else {
                algName = "AllReduceAivDeterExecutor";
            }
            HCCL_INFO("[SelectAlgfor910B] AllReduce SelectAlgfor910B is algName [%s].", algName.c_str());
            return HCCL_SUCCESS;
        }
        bool isOpbaseBigCount = isOpbase && (dataSize >= AIV_ALL_REDUCE_BIG_SIZE);
        HCCL_INFO(
            "[SelectAlgfor910B] Select AivMode Alg: DataSize[%llu], RankCountSize[%llu], DeviceNumPerAgg [%u]",
            dataSize, rankCountSize, deviceNumPerAggregation_);
        if (isSupportAivRdmaSmallCount) {
            algName = "AllReduceSmallCountAivRdmaExecutor"; // 多server，满足二次幂，小数据量（单卡190K以内）
        } else if (isSupportAivRdmaMidCount) {
            algName = "AllReduceMidCountAivRdmaExecutor"; // 多server，中小数据量（总数据量16M以内）
        } else if (isOpbaseBigCount || !isOpbase) {
            algName = "AllReduceMeshAivExecutor"; // 单server，单算子AIV模式大数据 和 图模式AIV 共用一个Executor
        } else {
            algName = "AllReduceMeshAivSmallCountExecutor"; // 单server，单算子AIV模式小数据单独一个Executor
        }
        // 小于等于两卡场景单独判断逻辑
    } else if (deviceNumPerAggregation_ <= DEVICE_TWO) {
        // 动态图算子融合场景？
        if ((param.inputPtr == commInputPtr)
            && (param.outputPtr == commOutputPtr && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
            && isMeshTopo) {
            algName = "AllReduceMeshExecutor";
            // 两卡不存在确定性问题 server内
        } else if (SingleMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType)) {
            ret = MeshTopoSelector(algName, dataSize);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[SelectAlgfor910B] AllReduce MeshTopoSelector failed, return[%d]", ret), ret);
            // 标卡场景（只有2p）
        } else if (Is2U2PInfer()) {
            if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && isInlineReduce) {
                algName = "AllReduceMeshOneshotLoopExecutor";
            } else {
                algName = "AllReduceRingExecutor";
            }
            // 多机单卡/两卡 pipeline需单独做判断(pipeline无确定性算法，并只支持单算子模式）
        } else if (
            topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE
            && algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE
            && IsMultiMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType)) {
            if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                algName = "AllReduceMeshOpbasePipelineExecutor";
            } else {
                algName = "AllReduceMeshGraphPipelineExecutor";
            }
            // 常规910B为mesh拓扑
        } else if (isMeshTopo) {
            algName = "AllReduceMeshExecutor";
            // 多机单卡topo为ring
        } else if (isRingTopo) {
            algName = "AllReduceRingExecutor";
            // 通信域打平场景
        } else {
            algName = "AllReduceComm";
        }
        // 多卡场景
    } else {
        if (isMeshTopo) {
            if ((param.inputPtr == commInputPtr)
                && (param.outputPtr == commOutputPtr
                    && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)) {
                algName = "AllReduceMeshExecutor";
                // 非确定性算法
            } else if (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE) {
                ret = NonDeterministicSelector(param, algName, dataSize);
                // 确定性算法
            } else {
                ret = DeterministicSelector(param, algName);
            }
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[SelectAlgfor910B] AllReduce SelectAlgfor910B failed, return[%d]", ret), ret);
            if (algName.empty()) {
                algName = "AllReduceMeshExecutor";
            }
        } else {
            algName = "AllReduceComm";
        }
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
            !isMesh, HCCL_ERROR("current algoLevel0Mesh[%d] not supported", algType_.algoLevel0), HCCL_E_NOT_SUPPORT);

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

        CHK_PRT_RET(
            !isServNumPowOfTwo, HCCL_ERROR("server num[%u] is not pow of two.", serverNum_), HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(
            !isSupportAivRdmaMidCount, HCCL_ERROR("current data size[%llu] not support aiv rdma mid count.", dataSize),
            HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(
            !isSupportAivDeter,
            HCCL_ERROR(
                "is not support aiv deter.isSingleMeshAggregation_[%d] isOpbase[%d] deterministic config[%u], "
                "dataSize[%llu]",
                isSingleMeshAggregation_, isOpbase, topoMatcher_->GetDeterministicConfig(), dataSize),
            HCCL_E_NOT_SUPPORT);
        return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO("[SelectAlgfor910B] AllReduce SelectAlgfor910B is algName [%s].", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::MeshTopoSelector(std::string& algName, u64 unitSize)
{
    // 单算子选择逻辑
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        if (unitSize <= HCCL_SMALL_COUNT_256_KB) {
            algName = "AllReduceMeshSmallCountExecutor";
        } else {
            algName = "AllReduceMeshOpbaseLoopExecutor";
        }
        // 图模式选择逻辑
    } else {
        if (unitSize <= HCCL_SMALL_COUNT_GRAPH_64_KB) {
            algName = "AllReduceMeshSmallCountExecutor";
        } else {
            algName = "AllReduceMeshExecutor";
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::NonDeterministicSelector(const OpParam& param, std::string& algName, u64 dataSize)
{
    const bool isOpbase = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    if (isOpbase) {
        if (IsMultiMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType)
            && algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
            algName = "AllReduceMeshOpbasePipelineExecutor";
        } else if (SingleMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType)) {
            if (dataSize <= HCCL_SMALL_COUNT_256_KB) {
                algName = "AllReduceMeshSmallCountExecutor";
            } else {
                algName = "AllReduceMeshOpbaseLoopExecutor";
            }
        }
    } else if (
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB
        && IsMultiMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType)
        && algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
        algName = "AllReduceMeshGraphPipelineExecutor";
    }
    if (!algName.empty() || !isOpbase) {
        return HCCL_SUCCESS;
    }
    const bool isInlineReduce
        = IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);
    // 单算子 + 数据量小于512kB
    if (dataSize < HCCL_SMALL_COUNT_512_KB && !isSingleMeshAggregation_ && isInlineReduce) {
        algName = "AllReduceMeshOpbaseSmallCountDeterministicExecutor";
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::DeterministicSelector(const OpParam& param, std::string& algName)
{
    // 确定性图和单算子归一流程
    HcclDataCountType countType = GetCountTypeForDeterAllReduce(param.DataDes.count, param.DataDes.dataType);
    const bool isOpbase = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    const bool isInlineReduce
        = IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);

    if (isOpbase && algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE
        && deviceNumPerAggregation_ > DEVICE_TWO) {
        u64 dataSize = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
        if (dataSize >= deviceNumPerAggregation_ * HCCL_MIN_SLICE_ALIGN) {
            algName = "AllReduceDeterPipelineExecutor";
            return HCCL_SUCCESS;
        }
    }
    if (SingleMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType)) {
        if (countType == HcclDataCountType::HCCL_COUNT_SMALL) {
            algName = "AllReduceMeshSmallCountExecutor";
        } else if (countType == HcclDataCountType::HCCL_COUNT_MEDIUM) {
            algName = "AllReduceMeshMidCountLoopExecutor";
        } else {
            algName = "AllReduceMeshOneshotLoopExecutor";
        }
    } else {
        u64 dataSize = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
        if (isOpbase && !isSingleMeshAggregation_ && isInlineReduce) {
            if (dataSize <= HCCL_SMALL_COUNT_512_KB) {
                // 单算子 + 确定性 + 数据量小于512kB
                algName = "AllReduceMeshOpbaseSmallCountDeterministicExecutor";
            } else {
                algName = "AllReduceMeshOpbaseMidCountDeterministicExecutor";
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceOperator::SelectAlgfor91093(const OpParam& param, std::string& algName)
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

    u64 dataSizePerRank = dataSize / deviceNumPerAggregation_;
    bool isOpbase = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    bool isOnlyAiv = topoMatcher_->GetIsOnlyAivConfig();
    // A3 AIV确定性 超节点内(单机与跨机) 支持单算子与图模式 限制单卡数据量8MB
    bool isBarrierOp = param.syncMode == SyncMode::UNLIMITED_TIMEWAITSYNCMODE; // Barrier算子不使能AIV
    bool isSupportAivDeter = (superPodNum_ == 1) && (topoMatcher_->GetAivModeConfig() && !isBarrierOp)
                             && IsSupportAIVReduce(param.DataDes.dataType, param.reduceType)
                             && ((topoMatcher_->GetDeterministicConfig() != DETERMINISTIC_DISABLE) || (serverNum_ > 1))
                             && ((userRankSize_ > DEVICE_EIGHT && dataSize < HCCL_SMALL_COUNT_8_MB)
                                 || (userRankSize_ <= DEVICE_EIGHT && dataSize <= HCCL_SMALL_COUNT_512_KB) || isOnlyAiv)
                             && (!retryEnable_) && userRankSize_ > 1 && !multiModuleDiffDeviceNumMode_
                             && ((!GetExternalInputInterHccsDisable()) || (serverNum_ == 1));

    bool isAivMode = (topoMatcher_->GetAivModeConfig() && !isBarrierOp)
                     && IsSupportAIVReduce(param.DataDes.dataType, param.reduceType) && serverNum_ == 1
                     && ((isOpbase && (dataSizePerRank <= AIV_ALL_REDUCE_A3_ENTRY_SIZE || isOnlyAiv))
                         || (!isOpbase && (dataSizePerRank <= AIV_ALL_REDUCE_A3_GRAPH_ENTRY_SIZE || isOnlyAiv)))
                     && (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE) && (!retryEnable_)
                     && !multiModuleDiffDeviceNumMode_;

    if (isSupportAivDeter) {
        algName = "AllReduceMeshAivFor91093Executor";
        HCCL_INFO("[SelectAlgfor91093] allreduce SelectAlgfor91093 algName [%s].", algName.c_str());
        return HCCL_SUCCESS;
    }

    if (IsNeedStrictMode(param)) {
        CHK_PRT_RET(
            !CheckStrictCondition(param),
            HCCL_ERROR("[AllReduceOperator][SelectAlgfor91093] not support DETERMINISTIC_STRICT mode."),
            HCCL_E_NOT_SUPPORT);

        algName = "AllReduceOrderPreservedFor91093Executor";
        HCCL_INFO("[SelectAlgfor91093] allreduce SelectAlgfor91093 algName [%s].", algName.c_str());
        return HCCL_SUCCESS;
    }

    if (isAivMode) {
        HCCL_INFO(
            "[SelectAlgfor91093] dataSize[%llu], dataSizePerRank[%llu], deviceNumPerAggregation[%u]", dataSize,
            dataSizePerRank, deviceNumPerAggregation_);
        if ((isOpbase && dataSize < AIV_ALL_REDUCE_BIG_SIZE)
            || (!isOpbase && dataSize <= AIV_A3_ALL_REDUCE_GRAPH_GUIYI_SIZE)) {
            algName = "AllReduceMeshAivSmallCountExecutor"; // 单server小数据
        } else {
            algName = "AllReduceMeshAivExecutor"; // 单server大数据
        }
        HCCL_INFO("[SelectAlgfor91093] AllReduce SelectAlgfor91093 is algName [%s].", algName.c_str());
        return HCCL_SUCCESS;
    }
    // ARS 算法选择
    bool isARSAlgo = multiModuleDiffDeviceNumMode_ && !multiSuperPodDiffDeviceNumMode_;
    if (isARSAlgo) {
        if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB
              || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING)) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_WARNING("[AllReduceOperator][SelectAlgfor91093] ARS only support NHR or RING in AlgoLevel1 "
                         "yet, default is NHR.");
        }
    }
    // AHC 算法选择逻辑
    if ((algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC)
        || (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE)) {
        CHK_RET(SelectAlgforAHC(dataSize, AHCOpType::AHC_OP_TYPE_ALLREDUCE));
    }
    void* commInputPtr = nullptr;
    u64 commInputSize = 0;
    CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));
    bool cclLimit
        = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
          && (param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] > (commInputSize / HCCL_MEMSIZE_HD_FACTOR));

    bool isSupportInlineReduce
        = IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);
    bool smallCountOptimSingleServer
        = (!retryEnable_) && (serverNum_ == 1)
          && ((workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
              || (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !param.aicpuUnfoldMode))
          && isSupportInlineReduce && (deviceNumPerAggregation_ > HCCL_DEVICE_NUM_TWO)
          && (param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] <= HCCL_SMALL_COUNT_512_KB * userRankSize_)
          && !cclLimit;
    bool smallCountOptimMultiServer = (deviceNumPerAggregation_ > HCCL_DEVICE_NUM_TWO) && (serverNum_ != 1)
                                      && (superPodNum_ == 1)
                                      && (param.DataDes.count * SIZE_TABLE[param.DataDes.dataType]
                                          <= HCCL_SMALL_COUNT_1_MB * deviceNumPerAggregation_);
    bool useHostComm = !isSupportInlineReduce
                       && ((serverNum_ != 1 && superPodNum_ == 1 && !GetExternalInputInterHccsDisable())
                           || ((superPodNum_ > 1 || GetExternalInputInterHccsDisable()) && !retryEnable_
                               && param.DataDes.count * SIZE_TABLE[param.DataDes.dataType]
                                      <= HCCL_SMALL_COUNT_4_MB * deviceNumPerAggregation_));
    bool is2Pod2ServerTopo = (superPodNum_ == 2 && serverNum_ == 2); // 针对 A3背靠背机型
    bool smallCountOptimMultiPod
        = (superPodNum_ > 1 || (GetExternalInputInterHccsDisable() && serverNum_ > 1)) && !is2Pod2ServerTopo
          && (param.DataDes.count * unitSize <= HCCL_SMALL_COUNT_16_KB * deviceNumPerAggregation_)
          && !retryEnable_; // 涉及ROCE平面
    // 多超节点 的中等数据量
    bool midCountOptimMultiPod
        = (superPodNum_ > 1) && isOpbase && !multiSuperPodDiffDeviceNumMode_ && !multiModuleDiffDeviceNumMode_
          && (param.DataDes.count * unitSize <= HCCL_SMALL_COUNT_256_KB) && !retryEnable_; // 涉及ROCE平面

    if (multiModuleDiffDeviceNumMode_ && multiSuperPodDiffDeviceNumMode_) {
        algName = "AllReduceComm";
    } else if (multiModuleDiffDeviceNumMode_ && !multiSuperPodDiffDeviceNumMode_) {
        algName = "AllReduceARSFor91093Executor";
    } else if (midCountOptimMultiPod) {
        algName = "AllReduceMidCountFor91093Executor";
    } else if (useHostComm || smallCountOptimMultiServer || smallCountOptimMultiPod) {
        algName = "AllReduceComm";
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
    } else if (smallCountOptimSingleServer) {
        algName = "AllReduceMeshSmallCountExecutor";
    } else if (
        (param.supportSymmetricMemory || param.supportZeroCopy)
        && (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING
            || param.DataDes.count * unitSize > HCCL_MID_COUNT_16_MB * serverNum_)) {
        algName = "AllReduceRingZerocopyExecutor";
    } else {
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_WARNING("[AllReduceOperator][SelectAlgfor91093] only support ring, NB and NHR in AlgoLevel1 yet, "
                         "default is algType=NHR.");
        }
        if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
            algName = "AllReduceFastDoubleRingFor91093Executor";
        } else if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING) {
            algName = "AllReduceRingFor91093Executor";
        } else {
            algName = "AllReduceComm"; // 支持91093全通信域
        }
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
    HCCL_INFO("[SelectAlgfor91093] AllReduce SelectAlgfor91093 is algName [%s].", algName.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_ALLREDUCE, AllReduce, AllReduceOperator);

} // namespace hccl
