/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <algorithm>
#include "device_capacity.h"
#include "coll_executor_base.h"
#include "coll_alg_exec_registry.h"
#include "coll_alg_operator.h"

namespace hccl {
using namespace std;
constexpr float GB2B = 1024 * 1024 * 1024;
constexpr float SECOND2MICROSECOND = 1000000;
constexpr float RHD_FACTOR_TWO = 2.0;
constexpr float RHD_FACTOR_ONE = 1.0;
constexpr float DOUBLE_SUB_HCCLCMD = 2.0; // The hcclCMD can be considered as combination of two hcclCMDs.
constexpr float COPY_TIME_IN_RHD = 1.0;
constexpr double NHR_FACTOR_TWO = 2.0;
constexpr double NHR_FACTOR_THREE = 3.0;
constexpr double NHR_FACTOR_FOUR = 4.0;
constexpr double NHR_SUB_TWO = 2.0;
constexpr float LATENCY = 60;                // 静态时延 60 us;
constexpr u64 PIPELINE_MIN_SIZE = 32 * 1024; // 当数据量大于等于32KB时，reduce_scatter和all_gather使能pipeline模式
constexpr u64 PIPELINE_ALLREDUCE_MIN_SIZE = 1024 * 1024; // 当数据量大于等于1MB时，allreduce使能pipeline模式
constexpr u64 PIPELINE_MIN_SIZE_NO_LITE = 2 * 1024 * 1024; // 如不支持RDMALite，当数据量大于等于2MB时，使能pipeline模式
constexpr u64 HCCL_FFTS_CAPACITY = 65535; // FFTS+子图最大容量
constexpr u32 AHC_MIN_SUBGROUP_SPLIT_DIVISOR = 2;
constexpr u32 AHC_LEVEL0_GROUP_SIZE_THRESHOLD = 3;
constexpr u32 SERVER_COUNT_THRESHOLD_FOR_MULTI_DETER_PIPELINE = 2;
constexpr u32 MIN_STRICT_RANK_NUM = 3;

CollAlgOperator::CollAlgOperator(
    AlgConfigurator* algConfigurator, CCLBufferManager& cclBufferManager, HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher>& topoMatcher, HcclCMDType opType)
    : algConfigurator_(algConfigurator),
      cclBufferManager_(cclBufferManager),
      dispatcher_(dispatcher),
      topoMatcher_(topoMatcher),
      workflowMode_(GetWorkflowMode())
{
    SetTopoAttr(algConfigurator_);
    SetAlgoAttr(algConfigurator_);
    algConfigurator->GetAlgTypeDirect(algType_, opType);
    algConfigurator->GetAlgoLevel1DefaultSwitch(isAlgoLevel1Default_, opType);
    algConfigurator->GetTopoType(topoType_);
}

HcclResult CollAlgOperator::SelectAlg(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] const OpParam& param,
    [[maybe_unused]] std::string& algName, [[maybe_unused]] std::string& newTag)
{
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlg(
    const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag,
    [[maybe_unused]] const ResourceLimit& limit)
{
    return SelectAlg(tag, param, algName, newTag);
}

HcclResult CollAlgOperator::GetAivExecParam(
    std::string& algName, const OpParam& param, AlgResourceResponse& algRes, AivSuperKernelArgs& args)
{
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance().GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(
            executor_.get() == nullptr,
            HCCL_ERROR("[CollAlgOperator][GetAivExecParam]Fail to find executor for algName[%s]", algName.c_str()),
            HCCL_E_PARA);
    }
    return executor_->GetAivExecParam(param, algRes, args);
}

HcclResult
CollAlgOperator::CalNumBlocks(std::string& algName, const OpParam& param, u32& numBlocks, int32_t aivCoreLimit)
{
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance().GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(
            executor_.get() == nullptr,
            HCCL_ERROR("[CollAlgOperator][CalNumBlocks]Fail to find executor for algName[%s]", algName.c_str()),
            HCCL_E_PARA);
        CHK_RET(SetExecutorAttr(param));
    }

    if (aivCoreLimit != 0) {
        CHK_RET(executor_->SetNumBlocks(aivCoreLimit));
    }

    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        return executor_->CalNumBlocks(
            numBlocks, userRankSize_, param.All2AllDataDes.sendCount * SIZE_TABLE[param.All2AllDataDes.sendType],
            param.opType);
    } else if (
        param.opType == HcclCMDType::HCCL_CMD_ALLREDUCE || param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER
        || param.opType == HcclCMDType::HCCL_CMD_ALLGATHER || param.opType == HcclCMDType::HCCL_CMD_BROADCAST) {
        return executor_->CalNumBlocks(
            numBlocks, userRankSize_, param.DataDes.count * SIZE_TABLE[param.DataDes.dataType], param.opType);
    } else {
        return executor_->CalNumBlocks(numBlocks, userRankSize_);
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::GetOpExpansionStr(const OpParam& param, const AlgDesc& algDesc, std::string& opExpansionStr)
{
    if (algDesc.isAivMode) {
        opExpansionStr = "AIV";
    } else if (param.aicpuUnfoldMode) {
        opExpansionStr = "AI_CPU";
    } else if (static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())) {
        opExpansionStr = "HOST";
    } else {
        opExpansionStr = "HOST_TS";
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlg(
    const std::string& tag, const OpParam& param, const ResourceLimit& limit, std::string& algName, AlgDesc& algDesc,
    std::string& newTag)
{
    bool isOnlyAiv = topoMatcher_->GetIsOnlyAivConfig();
    bool supportOnlyAiv
        = (param.opType == HcclCMDType::HCCL_CMD_ALLGATHER || param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER
           || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC
           || param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLREDUCE);
    CHK_PRT_RET(
        isOnlyAiv && !supportOnlyAiv,
        HCCL_ERROR(
            "[CollAlgOperator][SelectAlg] opType[%s] currently do not support aivonly",
            GetCMDTypeEnumStr(param.opType).c_str()),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(
        isOnlyAiv && userRankSize_ == 1 && supportOnlyAiv,
        HCCL_ERROR("[CollAlgOperator][SelectAlg] aivonly not support, please ensure rankNum is greater than one"),
        HCCL_E_NOT_SUPPORT);

    // 兼容老接口
    if (limit.ifLimit) {
        CHK_RET(SelectAlg(tag, param, algName, newTag, limit));
    } else {
        CHK_RET(SelectAlg(tag, param, algName, newTag));
    }

    // 非AIV算法提前返回, 采用兜底Executor
    if (algName.empty()) {
        executor_ = CollAlgExecRegistry::Instance().GetAlgExec("SendExecutor", dispatcher_, topoMatcher_);
        CHK_PRT_RET(
            executor_.get() == nullptr,
            HCCL_ERROR("[CollAlgOperator][SelectAlg]Fail to find executor for algName[DefaultExecutor]"), HCCL_E_PARA);
    } else {
        // 校验控核
        if (limit.ifLimit && deviceType_ == DevType::DEV_TYPE_910_93 && topoMatcher_->GetAivModeConfig()) {
            CHK_RET(SelectAlgFor91093WithCoreLimit(param, limit, algName));
        }

        // 从对应executor获取算法描述
        if (executor_.get() == nullptr) {
            executor_ = CollAlgExecRegistry::Instance().GetAlgExec(algName, dispatcher_, topoMatcher_);
            CHK_PRT_RET(
                executor_.get() == nullptr,
                HCCL_ERROR("[CollAlgOperator][SelectAlg]Fail to find executor for algName[%s]", algName.c_str()),
                HCCL_E_PARA);
            CHK_RET(SetExecutorAttr(param));
        }
    }

    bool isLastSelect = algDesc.isLastSelect;
    algDesc = executor_->GetAlgDesc();
    // 打印维测日志
    if (UNLIKELY(GetDebugConfig() & HCCL_ALG) && isLastSelect) {
        // 获取展开模式，转换成字符串
        std::string opExpansionStr;
        CHK_RET(GetOpExpansionStr(param, algDesc, opExpansionStr));
        // 尝试获取确定性属性（如果Executor有声明自己是否为确定性）
        std::string appendStr = "";
        if (algDesc.deterministic >= 0) {
            appendStr += "deterministic[" + std::to_string(algDesc.deterministic) + "]";
        }
        // 打印关键维测内容
        bool isOpBase = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
        HCCL_CONFIG_INFO(
            HCCL_ALG,
            "[%s] newTag[%s] algName[%s] userRank[%u] topoType[%d] algType[%s] "
            "userRankSize[%u] level0Size[%u] moduleNum_[%u] level2Size[%u] ",
            __func__, newTag.c_str(), algName.c_str(), userRank_, topoType_, AlgTypeToStr(algDesc.algType).c_str(),
            userRankSize_, deviceNumPerAggregation_, moduleNum_, superPodNum_);
        HCCL_CONFIG_INFO(
            HCCL_ALG,
            "[%s] newTag[%s] "
            "opExpansionMode[%s] isZeroCopy[%u] retryEnable[%u] isOpBase[%u] isCapture[%u] aivCoreLimit[%u] %s.",
            __func__, newTag.c_str(), opExpansionStr.c_str(), algDesc.isZeroCopy, retryEnable_, isOpBase,
            param.isCapture, limit.aivCoreLimit, appendStr.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult
CollAlgOperator::SelectAlgFor91093WithCoreLimit(const OpParam& param, const ResourceLimit& limit, std::string& algName)
{
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance().GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(
            executor_.get() == nullptr,
            HCCL_ERROR(
                "[CollAlgOperator][SelectAlgFor91093WithCoreLimit]Fail to find executor for algName[%s]",
                algName.c_str()),
            HCCL_E_PARA);
        CHK_RET(SetExecutorAttr(param));
    }

    CHK_RET(SetNumBlocks(limit.aivCoreLimit));

    std::string reSelName;
    switch (param.opType) {
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
            reSelName = "AllReduceMeshAivFor91093Executor";
            break;
        case HcclCMDType::HCCL_CMD_ALLGATHER:
            reSelName = "AllGatherMeshAivFor91093Executor";
            break;
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
            reSelName = "ReduceScatterMeshAivFor91093Executor";
            break;
        case HcclCMDType::HCCL_CMD_ALLTOALLV:
        case HcclCMDType::HCCL_CMD_ALLTOALL:
        case HcclCMDType::HCCL_CMD_ALLTOALLVC:
            reSelName = "AlltoAllMeshAivFor91093Executor";
            break;
        default:
            break;
    }

    u32 numBlocks;
    HcclResult ret = CalNumBlocks(algName, param, numBlocks);
    if (ret != HCCL_SUCCESS) {
        CHK_PRT_RET(
            reSelName.empty() || reSelName == algName,
            HCCL_ERROR(
                "[CollAlgOperator][SelectAlgFor91093WithCoreLimit]Fail to check CalNumBlocks for algName[%s]",
                algName.c_str()),
            HCCL_E_PARA);

        algName = reSelName;
        executor_ = nullptr;
        HCCL_INFO("[CollAlgOperator][SelectAlgFor91093WithCoreLimit]Re select to algName[%s]", reSelName.c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult
CollAlgOperator::CalcResRequest(const std::string& algName, const OpParam& param, AlgResourceRequest& resourceRequest)
{
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance().GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(
            executor_.get() == nullptr,
            HCCL_ERROR("[CollAlgOperator][CalcResRequest]Fail to find executor for algName[%s]", algName.c_str()),
            HCCL_E_PARA);
        CHK_RET(SetExecutorAttr(param));
    }
    return executor_->CalcResRequest(param, resourceRequest);
}

HcclResult CollAlgOperator::Orchestrate(const std::string& algName, OpParam& param, AlgResourceResponse& algResource)
{
    HCCL_INFO("[CollAlgOperator][Orchestrate]algName[%s]", algName.c_str());
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance().GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(
            executor_.get() == nullptr,
            HCCL_ERROR("[CollAlgOperator][Orchestrate]Fail to find executor for algName[%s]", algName.c_str()),
            HCCL_E_PARA);
        CHK_RET(SetExecutorAttr(param));
    }
    executor_->SetAivClearEnable(aivClearEnable_);
    executor_->SetAlgOpContext(algOpContext_);
    executor_->SetOpCounter(opCounter_);
    return executor_->Orchestrate(param, algResource);
}

HcclResult CollAlgOperator::GetAdjInfo(
    const std::string& algName, OpParam& param, AlgResourceResponse& algResource, AdjInfo& nslbAdjInfo)
{
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance().GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(
            executor_.get() == nullptr,
            HCCL_ERROR("[CollAlgOperator][Orchestrate]Fail to find executor for algName[%s]", algName.c_str()),
            HCCL_E_PARA);
        CHK_RET(SetExecutorAttr(param));
    }

    return executor_->GetAdjInfo(algResource, nslbAdjInfo);
}

HcclResult CollAlgOperator::PrepareCommInfoToDevice(const std::string& algName, AlgResourceResponse& algResource)
{
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance().GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(
            executor_.get() == nullptr,
            HCCL_ERROR(
                "[CollAlgOperator][PrepareCommInfoToDevice]Fail to find executor for algName[%s]", algName.c_str()),
            HCCL_E_PARA);
    }
    return executor_->PrepareCommInfoToDevice(algResource);
}

HcclResult CollAlgOperator::CalcIncreLinkRequest(
    const std::string& algName, const OpParam& param, std::set<u32>& ranksHasLinked,
    AlgResourceRequest& resourceRequest, bool& needIncreLink)
{
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance().GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(
            executor_.get() == nullptr,
            HCCL_ERROR("[CollAlgOperator][CalcIncreLinkRequest]Fail to find executor for algName[%s]", algName.c_str()),
            HCCL_E_PARA);
    }
    return executor_->CalcIncreLinkRequest(param, ranksHasLinked, resourceRequest, needIncreLink);
}

void CollAlgOperator::SetTopoAttr(AlgConfigurator* algConfigurator)
{
    const HcclTopoAttr& topoAttr = algConfigurator->GetTopoAttr();
    serverNum_ = topoAttr.serverNum;
    moduleNum_ = topoAttr.moduleNum;
    superPodNum_ = topoAttr.superPodNum;
    deviceNumPerServer_ = topoAttr.deviceNumPerServer;
    deviceNumPerAggregation_ = topoAttr.deviceNumPerAggregation;
    multiModuleDiffDeviceNumMode_ = topoAttr.multiModuleDiffDeviceNumMode;
    multiSuperPodDiffServerNumMode_ = topoAttr.multiSuperPodDiffServerNumMode;
    multiSuperPodDiffDeviceNumMode_ = topoAttr.multiSuperPodDiffDeviceNumMode;
    isDiffDeviceType_ = topoAttr.isDiffDeviceType;
    gcdDeviceNumPerAggregation_ = topoAttr.gcdDeviceNumPerAggregation;

    meshAggregationRankSize_ = topoAttr.meshAggregationRankSize;
    isDiffDeviceModule_ = topoAttr.isDiffDeviceModule;
    isSingleMeshAggregation_ = topoAttr.isSingleMeshAggregation;
    isAllRankSamePlane_ = topoAttr.isAllRankSamePlane;
    is310PDuoCard_ = topoAttr.is310PDuoCard;
    isCommon310P3DUO_ = topoAttr.isCommon310P3DUO;
    hccsPortNum_ = topoAttr.hccsPortNum;

    userRank_ = topoAttr.userRank;
    realUserRank_ = topoAttr.realUserRank;
    userRankSize_ = topoAttr.userRankSize;

    devicePhyId_ = topoAttr.devicePhyId;
    deviceLogicId_ = topoAttr.deviceLogicId;
    deviceType_ = topoAttr.deviceType;

    nicList_ = topoAttr.nicList;
    pairLinkCounter_ = topoAttr.pairLinkCounter;
    isSupportRdmaLite_ = topoAttr.isSupportRdmaLite;
    isSupportHccsAndSio_ = topoAttr.isSupportHccsAndSio;
    useSuperPodMode_ = topoAttr.useSuperPodMode;
    isARSDoubleRing_ = topoAttr.isARSDoubleRing;
    return;
}

void CollAlgOperator::SetAlgoAttr(AlgConfigurator* algConfigurator)
{
    const HcclAlgoAttr& algoAttr = algConfigurator->GetAlgoAttr();
    isHaveCpuRank_ = algoAttr.isHaveCpuRank;
    inlineReduceSwitchOn_ = algoAttr.inlineReduceSwitchOn;
    identifier_ = algoAttr.identifier;
    return;
}

HcclResult CollAlgOperator::SetExecutorAttr(const OpParam& param)
{
    CHK_RET(executor_->SetAlgType(algType_));
    CHK_RET(executor_->SetCCLInBuffer(cclBufferManager_.GetInCCLbufferSize()));

    if (param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        bool isSupportSDMAReduce = false;
        if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            isSupportSDMAReduce
                = IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);
        } else {
            isSupportSDMAReduce = IsSupportSDMAReduce(
                cclBufferManager_.GetInCCLbuffer().ptr(), cclBufferManager_.GetOutCCLbuffer().ptr(),
                param.DataDes.dataType, param.reduceType);
        }
        CHK_RET(executor_->SetIsSupportSDMAReduce(isSupportSDMAReduce));
    }
    return HCCL_SUCCESS;
}

std::string CollAlgOperator::GenerateNewTagByAlgTypeLevel1(std::string tag, std::string algTypeLevel1Tag) const
{
    if (algTypeLevel1Tag == "") {
        return tag;
    } else {
        return tag + "_" + algTypeLevel1Tag;
    }
}

HcclResult CollAlgOperator::AppendTag(const AlgTypeLevel1& algTypeLevel1, std::string& tag)
{
    switch (algTypeLevel1) {
        case AlgTypeLevel1::ALG_LEVEL1_RING:
            tag = "ALG_LEVEL1_RING";
            break;
        case AlgTypeLevel1::ALG_LEVEL1_HD:
            tag = "ALG_LEVEL1_HD";
            break;
        case AlgTypeLevel1::ALG_LEVEL1_NHR:
            tag = "ALG_LEVEL1_NHR";
            break;
        case AlgTypeLevel1::ALG_LEVEL1_PIPELINE:
            tag = "ALG_LEVEL1_PIPELINE";
            break;
        default:
            HCCL_WARNING("[CollAlgOperator][AppendTag] The algTypeLevel1 %d is not supported.", algTypeLevel1);
            break;
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::AutoSelectAlgTypeLevel1(
    HcclCMDType hcclCMDType, u64 countSize, u64 cclBufferSize, std::string& algTypeLevel1Tag, bool isInlineReduce,
    bool isRdmaReduce, bool isAivMode)
{
    if (isSingleMeshAggregation_) {
        HCCL_INFO(
            "[AutoSelectAlgTypeLevel1] there are %u server(%u module) in level1, no need to choose algo.", serverNum_,
            moduleNum_);
        return HCCL_SUCCESS;
    }

    // auto algo selection process
    if (isAlgoLevel1Default_) {
        // parse algType_ and get algTypeLevel1 and algTypeLevel0
        auto originalAlgTypeLevel0 = algType_.algoLevel0;
        // set algTypeLevel1
        AlgTypeLevel1 algTypeLevel1;
        CHK_RET(GetDefaultAlgoLevel1V2(
            hcclCMDType, countSize, cclBufferSize, algTypeLevel1, isInlineReduce, isRdmaReduce, isAivMode));
        auto iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algTypeLevel1);
        CHK_PRT_RET(
            iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(),
            HCCL_ERROR("[AutoSelectAlgTypeLevel1] level1: algType[%u] is invalid.", algTypeLevel1), HCCL_E_INTERNAL);
        HCCL_INFO(
            "[AutoSelectAlgTypeLevel1] there are %u server(%u module) in level1, using %s algo", serverNum_, moduleNum_,
            iter->second.c_str());
        algType_.algoLevel0 = originalAlgTypeLevel0;
        algType_.algoLevel1 = algTypeLevel1;
        // tag 增加所选的算法
        AppendTag(algTypeLevel1, algTypeLevel1Tag);
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoForComm(
    HcclCMDType hcclCMDType, float delay, u64 curSize, float bandWidth, AlgTypeLevel1& algType)
{
    // 从map中查找对应的计算函数
    auto it = selectFuncMap_.find(hcclCMDType);
    if (it == selectFuncMap_.end()) {
        HCCL_ERROR("[Get][AlgTypeLevel1] The hcclCMDType %d is not supported.", hcclCMDType);
        return HCCL_E_NOT_SUPPORT;
    }
    return (it->second)(delay, curSize, bandWidth, algType);
}

// 保守估计Pipeline算法所需context数量
u32 CollAlgOperator::CalcContextNumForPipeline(HcclCMDType hcclCMDType)
{
    bool isDeterPipeline
        = topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_ENABLE
          && (hcclCMDType == HcclCMDType::HCCL_CMD_ALLREDUCE || hcclCMDType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    const u32 stepNum = moduleNum_;                                       // 通信步数
    const u32 hccsContextNumPerStep = 5 * (deviceNumPerAggregation_ - 1); // SDMA跨片每步所需context数
    const u32 roceContextNumPerStep = 7;                                  // RDMA每步所需context数
    const u32 copyContextNumPerStep = 1;                                  // SDMA片内每步所需context数
    const u32 localReduceNumPerStep = isDeterPipeline ? (deviceNumPerAggregation_ - 1) : 0;
    const u32 contextNumPerStep
        = hccsContextNumPerStep + roceContextNumPerStep + copyContextNumPerStep + localReduceNumPerStep; // 小计
    const u32 barrierContextNum = 4; // 通信结束时barrier操作所需context数

    switch (hcclCMDType) {
        case HcclCMDType::HCCL_CMD_ALLREDUCE:      // fall-through
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER: // fall-through
        case HcclCMDType::HCCL_CMD_ALLGATHER:
        case HcclCMDType::HCCL_CMD_ALLGATHER_V: {
            const u32 copyContextNum = 1; // 通信首尾所需context数量
            u32 contextNum = stepNum * contextNumPerStep + barrierContextNum + copyContextNum;
            if (hcclCMDType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
                contextNum += contextNum;
            }
            if (isDeterPipeline) {
                contextNum += stepNum - 1; // 最后的local reduce
            }
            return contextNum;
        }
        case HcclCMDType::HCCL_CMD_ALLTOALLV:  // fall-through
        case HcclCMDType::HCCL_CMD_ALLTOALLVC: // fall-through
        case HcclCMDType::HCCL_CMD_ALLTOALL: {
            const u32 copyContextNum = 1 + moduleNum_; // 通信首尾所需context数量
            return stepNum * contextNumPerStep + barrierContextNum + copyContextNum;
        }
        default:
            return 0;
    }
}

HcclResult CollAlgOperator::GetDefaultAlgoLevel1V2(
    HcclCMDType hcclCMDType, u64 curSize, u64 cclBufferSize, AlgTypeLevel1& algType, bool isInlineReduce,
    bool isRdmaReduce, bool isAivMode)
{
    // pipeline mode is deployed,where there is multi-server multi-device(in server) now,
    // since RDMA is not reduced by normal serial orchestration of tasks.
    // So pipeline mode is more dominant than normal serial orchestration now.
    auto originalAlgTypeLevel0 = algType_.algoLevel0;
    bool disdeterniminsticWithInlineReduce
        = isInlineReduce && isRdmaReduce && topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE;
    bool deterniminsticWithInlineReduce
        = isInlineReduce && isRdmaReduce && topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_ENABLE;

    // 对于不支持Rdma Lite的场景，下发性能较差，RS和AG需要一个很大的数据量（AR的一半）才能掩盖下发时间
    u64 pipelineMinSize = (isSupportRdmaLite_) ? (PIPELINE_MIN_SIZE) : (PIPELINE_MIN_SIZE_NO_LITE);
    if (((hcclCMDType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER && disdeterniminsticWithInlineReduce)
         || hcclCMDType == HcclCMDType::HCCL_CMD_ALLGATHER || hcclCMDType == HcclCMDType::HCCL_CMD_ALLGATHER_V)
        && deviceNumPerAggregation_ != 1 && curSize >= pipelineMinSize && IsAlgTypeLevel0Mesh(originalAlgTypeLevel0)
        && CalcContextNumForPipeline(hcclCMDType) <= HCCL_FFTS_CAPACITY) {
        algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
        return HCCL_SUCCESS;
    }
    if (hcclCMDType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER && deterniminsticWithInlineReduce
        && deviceNumPerAggregation_ > 1 && curSize >= pipelineMinSize && IsAlgTypeLevel0Mesh(originalAlgTypeLevel0)
        && CalcContextNumForPipeline(hcclCMDType) <= HCCL_FFTS_CAPACITY && moduleNum_ > 1
        && curSize >= HCCL_SMALL_COUNT_256_KB) {
        algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
        return HCCL_SUCCESS;
    }

    // 对于不支持Rdma Lite的场景，下发性能较差，AllReduce需要一个较大的数据量才能掩盖下发时间
    pipelineMinSize = (isSupportRdmaLite_) ? (PIPELINE_ALLREDUCE_MIN_SIZE) : (PIPELINE_MIN_SIZE_NO_LITE);
    if (hcclCMDType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        // 计算每个slice的大小
        u64 allreduceCurSize = 0;
        allreduceCurSize = curSize / (moduleNum_ * deviceNumPerAggregation_);
        if (disdeterniminsticWithInlineReduce && deviceNumPerAggregation_ != 1 && allreduceCurSize >= pipelineMinSize
            && !isAivMode && IsAlgTypeLevel0Mesh(originalAlgTypeLevel0)
            && CalcContextNumForPipeline(hcclCMDType) <= HCCL_FFTS_CAPACITY) {
            algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
            return HCCL_SUCCESS;
        }
        if (deterniminsticWithInlineReduce && deviceNumPerAggregation_ > 1 && allreduceCurSize >= HCCL_SMALL_COUNT_1_MB
            && !isAivMode && IsAlgTypeLevel0Mesh(originalAlgTypeLevel0)
            && CalcContextNumForPipeline(hcclCMDType) <= HCCL_FFTS_CAPACITY) {
            algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
            return HCCL_SUCCESS;
        }
    }
    u64 dataSizePerLoop = curSize > cclBufferSize ? cclBufferSize : curSize;
    float delay = LATENCY; // 静态时延 60 us;
    float bandWidth;
    CHK_RET(GetBandWidthPerNPU(1, userRankSize_, deviceNumPerAggregation_, bandWidth)); // 单位：GB/s
    bandWidth = bandWidth * GB2B;                                                       // 单位：B/s
    CHK_RET(SelectAlgoForComm(hcclCMDType, delay, dataSizePerLoop, bandWidth, algType));
    return HCCL_SUCCESS;
}

HcclResult
CollAlgOperator::SelectAlgoTypeForReduceScatter(float delay, u64 recvCurSize, float bandWidth, AlgTypeLevel1& algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost
        = static_cast<double>(steps) * delay
          + static_cast<double>(steps) / moduleNum_ * recvCurSize * userRankSize_ / bandWidth * SECOND2MICROSECOND;

    // theoretical time cost of NHR
    double nhrCost = ceil(log2(moduleNum_)) * delay
                     + static_cast<double>(moduleNum_ - 1) / moduleNum_ * recvCurSize * userRankSize_ / bandWidth
                           * SECOND2MICROSECOND;

    // compare costs between NHR and Ring, if same cost, Ring > NHR > HD
    algType = (nhrCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_NHR : AlgTypeLevel1::ALG_LEVEL1_RING;
    double interMinCost = min(nhrCost, ringCost);

    // theoretical time cost of HD/RHD
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost
            = log2(moduleNum_) * delay
              + static_cast<double>(steps) / moduleNum_ * recvCurSize * userRankSize_ / bandWidth * SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD,
        // the (RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ is equal to 1 + (moduleNum_ -1) / moduleNum_
        hdCost = ceil(log2(moduleNum_)) * delay
                 + static_cast<double>(RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ * recvCurSize
                       * userRankSize_ / bandWidth * SECOND2MICROSECOND;
    }

    // compare cost among NHR, HD and Ring
    algType = (hdCost < interMinCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : algType;
    return HCCL_SUCCESS;
}

HcclResult
CollAlgOperator::SelectAlgoTypeForAllGather(float delay, u64 sendCurSize, float bandWidth, AlgTypeLevel1& algType) const
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost
        = static_cast<double>(steps) * delay
          + static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth * SECOND2MICROSECOND;

    // theoretical time cost of NHR
    double nhrCost = ceil(log2(moduleNum_)) * delay
                     + static_cast<double>(moduleNum_ - 1) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth
                           * SECOND2MICROSECOND;

    // compare costs between NHR and Ring, if same cost, Ring > NHR > HD
    algType = (nhrCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_NHR : AlgTypeLevel1::ALG_LEVEL1_RING;
    double interMinCost = min(nhrCost, ringCost);

    // theoretical time cost of HD/RHD
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost
            = log2(moduleNum_) * delay
              + static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth * SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // the (RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ is equal to 1 + (moduleNum_ -1) / moduleNum_
        hdCost = ceil(log2(moduleNum_)) * delay
                 + static_cast<double>(RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ * sendCurSize
                       * userRankSize_ / bandWidth * SECOND2MICROSECOND;
    }

    // compare cost among NHR, HD and Ring
    algType = (hdCost < interMinCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : algType;
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoTypeForAllGatherV(
    float delay, u64 sendCurSize, float bandWidth, AlgTypeLevel1& algType) const
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost
        = static_cast<double>(steps) * delay
          + static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth * SECOND2MICROSECOND;

    HCCL_DEBUG("[%s] CollAlgOperator for SelectAlgoTypeForAllGatherV", __func__);
    // theoretical time cost of NHR
    double nhrCost = ceil(log2(moduleNum_)) * delay
                     + static_cast<double>(moduleNum_ - 1) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth
                           * SECOND2MICROSECOND;

    // compare costs between NHR and Ring, if same cost, Ring > NHR > HD
    algType = (nhrCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_NHR : AlgTypeLevel1::ALG_LEVEL1_RING;

    return HCCL_SUCCESS;
}

HcclResult
CollAlgOperator::SelectAlgoTypeForGather(float delay, u64 sendCurSize, float bandWidth, AlgTypeLevel1& algType) const
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost
        = static_cast<double>(steps) * delay
          + static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth * SECOND2MICROSECOND;
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost
            = log2(moduleNum_) * delay
              + static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth * SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // the (RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ is equal to 1 + (moduleNum_ -1) / moduleNum_
        hdCost = ceil(log2(moduleNum_)) * delay
                 + static_cast<double>(RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ * sendCurSize
                       * userRankSize_ / bandWidth * SECOND2MICROSECOND;
    }
    algType = (hdCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : AlgTypeLevel1::ALG_LEVEL1_RING;
    return HCCL_SUCCESS;
}

HcclResult
CollAlgOperator::SelectAlgoTypeForAllReduce(float delay, u64 curSize, float bandWidth, AlgTypeLevel1& algType) const
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) * delay
                      + DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ * curSize
                            / deviceNumPerAggregation_ / bandWidth * SECOND2MICROSECOND;

    // theoretical time cost of NHR
    double nhrCost = NHR_FACTOR_TWO * ceil(log2(moduleNum_)) * delay
                     + NHR_FACTOR_TWO * static_cast<double>(moduleNum_ - 1) / moduleNum_ * curSize
                           / deviceNumPerAggregation_ / bandWidth * SECOND2MICROSECOND;

    // compare costs between NHR and Ring, if same cost, Ring > NHR > HD
    algType = (nhrCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_NHR : AlgTypeLevel1::ALG_LEVEL1_RING;
    double interMinCost = min(nhrCost, ringCost);

    // theoretical time cost of HD/RHD
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = DOUBLE_SUB_HCCLCMD * log2(moduleNum_) * delay
                 + DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ * curSize / deviceNumPerAggregation_
                       / bandWidth * SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // the (RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ is equal to 1 + (moduleNum_ -1) / moduleNum_
        hdCost = DOUBLE_SUB_HCCLCMD * ceil(log2(moduleNum_)) * delay
                 + DOUBLE_SUB_HCCLCMD * static_cast<double>(RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_
                       * curSize / deviceNumPerAggregation_ / bandWidth * SECOND2MICROSECOND;
    }

    // compare cost among NHR, HD and Ring
    algType = (hdCost < interMinCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : algType;
    return HCCL_SUCCESS;
}

HcclResult
CollAlgOperator::SelectAlgoTypeForBroadcast(float delay, u64 curSize, float bandWidth, AlgTypeLevel1& algType) const
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) * delay
                      + DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ * curSize
                            / deviceNumPerAggregation_ / bandWidth * SECOND2MICROSECOND;
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = DOUBLE_SUB_HCCLCMD * log2(moduleNum_) * delay
                 + DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ * curSize / deviceNumPerAggregation_
                       / bandWidth * SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // rhd-broadcast = scatter + allgather + copy
        hdCost = (COPY_TIME_IN_RHD + DOUBLE_SUB_HCCLCMD * floor(log2(moduleNum_))) * delay
                 + (COPY_TIME_IN_RHD + DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_) * curSize
                       / deviceNumPerAggregation_ / bandWidth * SECOND2MICROSECOND;
    }
    algType = (hdCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : AlgTypeLevel1::ALG_LEVEL1_RING;
    return HCCL_SUCCESS;
}

HcclResult
CollAlgOperator::SelectAlgoTypeForReduce(float delay, u64 curSize, float bandWidth, AlgTypeLevel1& algType) const
{
    HCCL_DEBUG("[CollAlgOperator]SelectAlgoTypeForReduce start");
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) * delay
                      + DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ * curSize
                            / deviceNumPerAggregation_ / bandWidth * SECOND2MICROSECOND;
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = DOUBLE_SUB_HCCLCMD * log2(moduleNum_) * delay
                 + DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ * curSize / deviceNumPerAggregation_
                       / bandWidth * SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // rhd-broadcast = reducescatter + gather + copy
        hdCost = (COPY_TIME_IN_RHD + DOUBLE_SUB_HCCLCMD * floor(log2(moduleNum_))) * delay
                 + (COPY_TIME_IN_RHD + DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_) * curSize
                       / deviceNumPerAggregation_ / bandWidth * SECOND2MICROSECOND;
    }
    algType = (hdCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : AlgTypeLevel1::ALG_LEVEL1_RING;
    return HCCL_SUCCESS;
}

AlgType CollAlgOperator::GetAlgType() { return algType_; }

bool CollAlgOperator::Is2U2PInfer()
{
    return (
        (deviceNumPerAggregation_ == HCCL_DEVICE_NUM_TWO) && (serverNum_ == 1)
        && (deviceType_ == DevType::DEV_TYPE_910B) && (meshAggregationRankSize_ == HCCL_DEVICE_NUM_TWO)
        && (pairLinkCounter_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)] == 0));
}

bool CollAlgOperator::Is910BSingleMesh()
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH
                      || topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;

    bool isSingleMesh
        = (deviceType_ == DevType::DEV_TYPE_910B) && (isMeshTopo || Is2U2PInfer()) && (userRankSize_ != 1);
    return isSingleMesh;
}

bool CollAlgOperator::NeedCreateSingleMeshPlane(const bool isInlineReduce)
{
    // 910B 图模式非确定计算，inlineReduce使能，MESH拓扑场景下，创建一个mesh平面
    bool meshSinglePlane = Is910BSingleMesh() && topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE
                           && isInlineReduce && (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    return meshSinglePlane;
}

bool CollAlgOperator::SingleMeshInlineReduce(void* inputPtr, void* outputPtr, HcclDataType dataType, HcclReduceOp op)
{
    bool isInlineReduce = IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op);
    bool singleMeshInlineReduce = Is910BSingleMesh() && isInlineReduce && isSingleMeshAggregation_;
    return singleMeshInlineReduce;
}

bool CollAlgOperator::IsMultiMeshInlineReduce(void* inputPtr, void* outputPtr, HcclDataType dataType, HcclReduceOp op)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH
                      || topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;

    bool isInlineReduce = IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op);
    bool isRdmaReduce = IsSupportRDMAReduce(dataType, op);
    bool multiMeshInlineReduce = (deviceType_ == DevType::DEV_TYPE_910B) && isMeshTopo && isInlineReduce && isRdmaReduce
                                 && (!isSingleMeshAggregation_);
    return multiMeshInlineReduce;
}

void CollAlgOperator::SetLegacyHcclImpl(std::unique_ptr<hcclImpl>& impl)
{
    hcclImpl_ = impl.get();
    return;
}

HcclResult CollAlgOperator::SetRetryEnable(bool retryEnable)
{
    retryEnable_ = retryEnable;
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SetAivClearEnable(bool aivClearEnable)
{
    aivClearEnable_ = aivClearEnable;
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SetAlgOpContext(AlgOpContext algOpContext)
{
    algOpContext_ = algOpContext;
    return HCCL_SUCCESS;
}

bool CollAlgOperator::SupportRetryWithInplaceCheck(
    const HcclCMDType& opType, OpParam& param, std::string& algName, u8& isInplaceStatus,
    InplaceSupportRetryStatus& inPlaceSupportRetryStatus)
{
    // 不支持inplace的通信算子重执行
    if (IsHcclOpInplace(opType, param, userRank_, userRankSize_, isInplaceStatus)) {
        void* commInputPtr = nullptr;
        u64 commInputSize = 0;
        CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));
        if (!FitRetryConditionforInPlaceOp(
                opType, param, algName, commInputSize, userRankSize_, retryEnable_, inPlaceSupportRetryStatus)) {
            HCCL_DEBUG(
                "[CollAlgOperator][OpRetry][AICPU]hccl aicpu can not retry, opType[%s], inputPtr[%p], "
                "outputPtr[%p].",
                GetCMDTypeEnumStr(opType).c_str(), param.inputPtr, param.outputPtr);
            return false;
        }
    }
    // true 存在两种情况：
    // 1. 非inplace场景
    // 2. 是inplace但同时符合retry条件的场景
    return true;
}

HcclResult CollAlgOperator::GetNumBlocks(u32& numBlocks)
{
    CHK_SMART_PTR_NULL(executor_);
    return executor_->GetNumBlocks(numBlocks);
}

HcclResult CollAlgOperator::SetNumBlocks(const u32& numBlocks)
{
    CHK_SMART_PTR_NULL(executor_);
    return executor_->SetNumBlocks(numBlocks);
}

HcclResult CollAlgOperator::GetCache(HcclCacheInfo& cacheInfo)
{
    CHK_SMART_PTR_NULL(executor_);
    return executor_->GetCache(cacheInfo);
}

HcclResult CollAlgOperator::SetOpCounter(const OpCounterInfo& opCounter)
{
    opCounter_ = opCounter;
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SetRmaInfo(void* rmaInfo)
{
    CHK_SMART_PTR_NULL(executor_);
    CHK_PTR_NULL(rmaInfo);
    return executor_->SetRmaInfo(rmaInfo);
}

HcclResult CollAlgOperator::SelectAlgforAHC(u64 dataSize, AHCOpType ahcOpType)
{
    if (multiModuleDiffDeviceNumMode_) {
        return HCCL_SUCCESS;
    }

    bool isAHCWholeConfig
        = (algType_.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_RESERVED
           && (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
               || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE));

    CommPlane ahcSubGroupLevel = COMM_LEVEL1_AHC;
    if (isAHCWholeConfig) {
        if (deviceType_ != DevType::DEV_TYPE_910_93) {
            ahcSubGroupLevel = COMM_COMBINE;
        } else {
            ahcSubGroupLevel = COMM_COMBINE_ORDER;
        }
    } else if (deviceType_ != DevType::DEV_TYPE_910_93) {
        HCCL_DEBUG("[AHCAlgSelect] hccl algorithm: 910B not support level1 ahc, return ERROR.");
        return HCCL_E_PARA;
    }

    HCCL_INFO(
        "[SelectAlgforAHC] ahcOpType[%u] isAHCWholeConfig[%u] AHClevel[%u] algType_[%u] deviceType_[%u]", ahcOpType,
        isAHCWholeConfig, ahcSubGroupLevel, algType_.algoLevel1, deviceType_);

    AlgTypeLevel1 algTypeLevel1;

    std::vector<std::vector<std::vector<u32>>> globalSubGroups;
    std::map<AHCConcOpType, TemplateType> ahcAlgOption;
    CHK_RET(topoMatcher_->GetGlobalSubGroups(ahcSubGroupLevel, globalSubGroups));
    topoMatcher_->GetAHCAlgOption(ahcAlgOption);

    AHCAlgSelectParam ahcAlgSelectParam;
    ahcAlgSelectParam.opType = ahcOpType;
    ahcAlgSelectParam.dataSize = dataSize;

    // AHC 封装算法选择逻辑
    CHK_RET(AHCAlgSelect(algTypeLevel1, globalSubGroups, ahcAlgOption, ahcAlgSelectParam));

    topoMatcher_->SetAHCAlgOption(ahcAlgOption);

    auto iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algTypeLevel1);
    CHK_PRT_RET(
        iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(),
        HCCL_ERROR("[AHCAlgSelect] level1: algType_[%u] is invalid.", algTypeLevel1), HCCL_E_INTERNAL);

    // 支持 AHC 自适应调节为 BROKE 类型
    if (algType_.algoLevel1 != algTypeLevel1 && algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
        algType_.algoLevel1 = algTypeLevel1;
    }

    HCCL_INFO(
        "[AHCAlgSelect] hccl algorithm: there are %u server(%u module) in level1, using %s algo", serverNum_,
        moduleNum_, iter->second.c_str());

    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::AHCAlgSelect(
    AlgTypeLevel1& algType, std::vector<std::vector<std::vector<u32>>>& globalSubGroups,
    std::map<AHCConcOpType, TemplateType>& ahcAlgOption, AHCAlgSelectParam& ahcAlgSelectParam)
{
    // globalSubGroups 参数检查
    CHK_RET(CommAHCBaseInfo::CheckGlobalGroups(globalSubGroups));

    bool isAHCType = false;
    u32 minSubGroupSize = globalSubGroups[0][0].size();
    u32 maxSubGroupSize = globalSubGroups[0][0].size();
    for (u32 i = 1; i < globalSubGroups[0].size(); ++i) {
        if (globalSubGroups[0][i].size() < minSubGroupSize) {
            minSubGroupSize = globalSubGroups[0][i].size();
        }
        if (globalSubGroups[0][i].size() > maxSubGroupSize) {
            maxSubGroupSize = globalSubGroups[0][i].size();
        }
    }
    for (u32 i = 0; i < globalSubGroups[0].size(); ++i) {
        if (globalSubGroups[0][i].size() != minSubGroupSize) {
            isAHCType = true;
            break;
        }
    }

    // 多平面 reduce scatter 和 all gather 算子，强制写死成BROKE类型
    if (deviceNumPerServer_ != 1 && ahcAlgSelectParam.opType != AHCOpType::AHC_OP_TYPE_ALLREDUCE) {
        isAHCType = false;
    }

    // add AHC Conc Type logic here,  modify init Type  depend on the input para
    CHK_RET(AHCAlgOptionSelect(algType, globalSubGroups, ahcAlgOption, ahcAlgSelectParam));

    if (ahcAlgSelectParam.enableAlgAutoSelect == false) { // 关闭算法自适应功能时，默认设置AHC算法
        algType = AlgTypeLevel1::ALG_LEVEL1_AHC;
        return HCCL_SUCCESS;
    }

    if (isAHCType) {
        algType = AlgTypeLevel1::ALG_LEVEL1_AHC; // 设置为 AHC 类型
    } else {
        algType = AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE; // 设置为 BROKE 类型
    }

    HCCL_DEBUG(
        "[AHCAlgSelect] end minSubGroupSize = %u maxSubGroupSize = %u isAHCType = %u", minSubGroupSize, maxSubGroupSize,
        isAHCType);

    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::AHCAlgOptionSelect(
    const AlgTypeLevel1& algType, std::vector<std::vector<std::vector<u32>>>& globalSubGroups,
    std::map<AHCConcOpType, TemplateType>& ahcAlgOption, const AHCAlgSelectParam& ahcAlgSelectParam) const
{
    (void)algType;
    (void)ahcAlgSelectParam;
    AHCConcOpType ahcConcOpType;
    // 一层组间拼接时，分组数大于设定阈值则修改默认算法为NHR
    if (globalSubGroups[0].size() <= AHC_LEVEL0_GROUP_SIZE_THRESHOLD) {
        HCCL_DEBUG("[AHCAlgSelect] conc inter select type RING ");
        ahcConcOpType = {AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER};
        ahcAlgOption[ahcConcOpType] = TemplateType::TEMPLATE_REDUCESCATTER_RING;

        ahcConcOpType = {AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_ALLREDUCE};
        ahcAlgOption[ahcConcOpType] = TemplateType::TEMPLATE_ALL_REDUCE_RING;

        ahcConcOpType = {AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_ALLGATHER};
        ahcAlgOption[ahcConcOpType] = TemplateType::TEMPLATE_ALL_GATHER_RING;
    } else {
        HCCL_DEBUG("[AHCAlgSelect] conc inter select type NHR ");
        ahcConcOpType = {AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER};
        ahcAlgOption[ahcConcOpType] = TemplateType::TEMPLATE_REDUCESCATTER_NHR;

        ahcConcOpType = {AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_ALLREDUCE};
        ahcAlgOption[ahcConcOpType] = TemplateType::TEMPLATE_ALL_REDUCE_NHR;

        ahcConcOpType = {AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_ALLGATHER};
        ahcAlgOption[ahcConcOpType] = TemplateType::TEMPLATE_ALL_GATHER_NHR;
    }
    return HCCL_SUCCESS;
}

u32 CollAlgOperator::CalcOptimalIntraRingsize(u64 count, HcclDataType dataType, HcclCMDType opType)
{
    if (!topoMatcher_->GetARSFlag())
        return 0;

    u32 level0RankSize = topoMatcher_->GetCommPlaneRanks(COMM_LEVEL0)[0].size();
    u32 rankSizeInSuperPod = topoMatcher_->GetCommPlaneRanks(COMM_ARS)[0].size();
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));
    // 不支持 ARS 或环内卡数不是 2 的倍数
    u32 level0RingSize = 1;
    if (!isARSDoubleRing_ || (level0RankSize % FACTOR_TWO != 0)) {
        HCCL_INFO(
            "not Support ARS doubleRing, level0RingSize:[%u], level0RankSize[%u].", level0RingSize, level0RankSize);
        return level0RingSize;
    }
    // --- 1. 带宽 & 基本参数 ---
    float bwHCCS, bwHBM, bwSIO;
    constexpr u32 level0 = 0;
    constexpr u32 level2 = 2;
    constexpr u32 level3 = 3;
    CHK_RET(GetBandWidthPerNPU(level0, userRankSize_, deviceNumPerAggregation_, bwHCCS));
    CHK_RET(GetBandWidthPerNPU(level2, userRankSize_, deviceNumPerAggregation_, bwHBM));
    CHK_RET(GetBandWidthPerNPU(level3, userRankSize_, deviceNumPerAggregation_, bwSIO));
    float latency = BASE_COMM_LATENCY / MULTIPLIER_MS2US; // ms
    // --- 2. 数据总量 (GB) ---
    float baseSizeGB = static_cast<double>(count) * perDataSize / GB2B;
    float totalSize = baseSizeGB;
    HCCL_INFO(
        "CalcOptimalIntraRingsize: count[%u], totalSize:[%lf]GB, perDataSize[%u].", count, totalSize, perDataSize);
    if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER || opType == HcclCMDType::HCCL_CMD_ALLGATHER) {
        totalSize *= rankSizeInSuperPod;
    }
    // --- 3. 枚举可能的环大小 ---
    std::vector<u32> factors;
    for (u32 i = 1; i <= rankSizeInSuperPod / i; ++i) {
        if (rankSizeInSuperPod % i == 0) {
            if (i != rankSizeInSuperPod / i) {
                factors.push_back(rankSizeInSuperPod / i);
            }
            factors.push_back(i);
        }
    }
    std::sort(factors.begin(), factors.end());
    // --- 4. 计算最优带宽 ---
    double maxBwARS = 0.0;
    for (u32 N1 : factors) {
        // 传输时延 (ms)
        double latencyIntra;
        if (N1 == FACTOR_TWO) {
            latencyIntra = totalSize * MULTIPLIER_S2MS / FACTOR_TWO / bwSIO;
        } else if ((N1 % FACTOR_TWO == 0) && (N1 > FACTOR_TWO)) {
            latencyIntra = (N1 - 1) * totalSize * MULTIPLIER_S2MS / N1 / bwHCCS / FACTOR_TWO;
        } else {
            latencyIntra = (N1 - 1) * totalSize * MULTIPLIER_S2MS / N1 / bwHCCS;
        }
        u32 N2 = rankSizeInSuperPod / N1;
        // 静态时延 (ms)
        double interStep = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) ? (N2 - 1) : log2(N2);
        double latencyStep = (interStep + (N1 - 1)) * latency;
        double latencyInter = (N2 - 1) * totalSize * MULTIPLIER_S2MS / N1 / N2 / bwHCCS;
        // HBM 拷贝时延 (ms)
        double latencyCopy = totalSize * MULTIPLIER_S2MS / bwHBM;
        u8 mul = (HcclCMDType::HCCL_CMD_ALLREDUCE == opType) ? FACTOR_TWO : 1;
        double timeCost = mul * (latencyStep + latencyIntra + latencyInter) + latencyCopy;
        double bwARS = totalSize / timeCost; // GB/ms
        if (bwARS > maxBwARS) {
            level0RingSize = N1;
            maxBwARS = bwARS;
        }
    }
    HCCL_INFO("level0RingSize:[%u], level0RankSize[%u], totalSize:[%lf]GB", level0RingSize, level0RankSize, totalSize);
    return level0RingSize;
}

bool CollAlgOperator::IsNeedStrictMode(const OpParam& param)
{
    bool isStrictMode
        = (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_STRICT)
          && (param.DataDes.dataType == HCCL_DATA_TYPE_FP16 || param.DataDes.dataType == HCCL_DATA_TYPE_FP32
              || param.DataDes.dataType == HCCL_DATA_TYPE_BFP16 || param.DataDes.dataType == HCCL_DATA_TYPE_FP64)
          && (param.reduceType == HCCL_REDUCE_SUM || param.reduceType == HCCL_REDUCE_PROD)
          && userRankSize_ >= MIN_STRICT_RANK_NUM;

    return isStrictMode;
}

bool CollAlgOperator::CheckStrictCondition(const OpParam& param) const
{
    CHK_PRT_RET(
        multiModuleDiffDeviceNumMode_ || multiSuperPodDiffDeviceNumMode_ || multiSuperPodDiffServerNumMode_,
        HCCL_ERROR("[CollAlgOperator][CheckStrictCondition] DETERMINISTIC_STRICT mode not support asymmetrical topo."),
        false);

    CHK_PRT_RET(
        param.reduceType == HCCL_REDUCE_PROD,
        HCCL_ERROR("[CollAlgOperator][CheckStrictCondition] DETERMINISTIC_STRICT mode not support PROD."), false);

    CHK_PRT_RET(
        param.DataDes.dataType == HCCL_DATA_TYPE_FP64,
        HCCL_ERROR("[CollAlgOperator][CheckStrictCondition] DETERMINISTIC_STRICT mode not support FP64."), false);

    CHK_PRT_RET(
        GetExternalInputInterHccsDisable(),
        HCCL_ERROR("[CollAlgOperator][CheckStrictCondition] DETERMINISTIC_STRICT mode not support HCCS disable."),
        false);

    return true;
}

} // namespace hccl
