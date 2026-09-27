/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "auto_selector_base.h"
#include "coll_operator.h"
#include "coll_alg_params.h"

namespace Hccl {

SelectorStatus AutoSelectorBase::Select(const CollAlgOperator& op, CollAlgParams& params, std::string& primQueueGenName)
{
    HCCL_DEBUG("[AutoSelectorBase][%s] start", __func__);
    TopoInfo topoInfo;
    HCCL_DEBUG("[AutoSelectorBase][%s] CalcTopoShape start", __func__);
    CalcTopoShape(topoInfo);
    HCCL_DEBUG("[AutoSelectorBase][%s] end, levelNum[%u]", __func__, topoInfo.levelNum);
    std::map<OpType, std::vector<HcclAlgoType>> configAlgMap = EnvConfig::GetInstance().GetAlgoConfig().GetAlgoConfig();
    SelectorStatus ret = SelectorStatus::NOT_MATCH;
    HCCL_DEBUG(
        "[AutoSelectorBase][%s] params.opExecuteConfig.accelerator[%s]", __func__,
        params.opExecuteConfig.accState.Describe().c_str());
    dataSize_ = op.dataCount * DataTypeSizeGet(op.dataType);
    ;
    if (params.opExecuteConfig.accState == AcceleratorState::CCU_MS) {
        ret = SelectCcuMsAlgo(topoInfo, op, configAlgMap, primQueueGenName);
        if (ret == SelectorStatus::NOT_MATCH) {
            params.opExecuteConfig.accState = AcceleratorState::CCU_SCHED;
        } else {
            return ret;
        }
    }
    if (params.opExecuteConfig.accState == AcceleratorState::CCU_SCHED) {
        ret = SelectCcuScheduleAlgo(topoInfo, op, configAlgMap, primQueueGenName);
        if (ret == SelectorStatus::NOT_MATCH) {
            params.opExecuteConfig.accState = AcceleratorState::CCU_FALLBACK;
        } else {
            return ret;
        }
    }
    if (params.opExecuteConfig.accState == AcceleratorState::AIV) {
        if (op.opType != OpType::BARRIER) {
            ret = SelectAivAlgo(topoInfo, op, configAlgMap, primQueueGenName);
        }
        if (ret == SelectorStatus::MATCH) {
            return ret;
        }
        params.opExecuteConfig.accState = AcceleratorState::CCU_FALLBACK;
    }

    if (params.opExecuteConfig.accState == AcceleratorState::AIV_ONLY) {
        return (op.opType == OpType::BARRIER) ? SelectorStatus::NOT_MATCH :
                                                SelectAivAlgo(topoInfo, op, configAlgMap, primQueueGenName);
    }
    if (IsStarsState(params.opExecuteConfig)) {
        // level0是PCIE混合的场景，且CLOS规模大于8，选择AIV_ONLY算法
        if (topoInfo.level0PcieMix && topoInfo.level0BigClosRange) {
            params.opExecuteConfig.accState = AcceleratorState::AIV_ONLY;
            return (op.opType == OpType::BARRIER) ? SelectorStatus::NOT_MATCH :
                                                    SelectAivAlgo(topoInfo, op, configAlgMap, primQueueGenName);
        }
        ret = SelectAicpuAlgo(topoInfo, op, configAlgMap, primQueueGenName);
        if ((ret == SelectorStatus::MATCH) && (params.opExecuteConfig.accState == AcceleratorState::CCU_FALLBACK)) {
            params.opExecuteConfig.accState = AcceleratorState::AICPU_TS;
        }
        return ret;
    }
    return SelectorStatus::NOT_MATCH;
}

bool AutoSelectorBase::IsStarsState(const OpExecuteConfig& opExecuteConfig) const
{
    return (
        opExecuteConfig.accState == AcceleratorState::AICPU_TS
        || opExecuteConfig.accState == AcceleratorState::HOSTCPU_TS
        || opExecuteConfig.accState == AcceleratorState::CCU_FALLBACK);
}

bool AutoSelectorBase::IsDefaultAlg(const HcclAlgoType algoType) const
{
    return (algoType == HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT) || (algoType == HcclAlgoType::HCCL_ALGO_TYPE_NA);
}

HcclAlgoType AutoSelectorBase::GetLevel0AlgoType(
    const CollAlgOperator& op, const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap) const
{
    HcclAlgoType levle0Algo = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
    auto it = configAlgMap.find(op.opType);
    if ((it != configAlgMap.end()) && (it->second.size() > 0)) {
        levle0Algo = it->second[0];
    }
    return levle0Algo;
}

bool AutoSelectorBase::IsSmallData(const u64 dataSize) const { return dataSize < SMALL_COUNT_512KB; }

bool AutoSelectorBase::IsLargeData(const u64 dataSize) const { return dataSize >= LARGE_COUNT_1024KB; }

bool AutoSelectorBase::IsSmallDataCCU(const u64 dataSize, const u64 rankSize) const
{
    if (rankSize == 0) {
        HCCL_WARNING("the selector is not set RankSize");
    }
    return (dataSize <= CCU_PARALLEL_MAX_DATA_SIZE) ? true : false;
}

SelectorStatus AutoSelectorBase::SelectCcuMsAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)topoInfo;
    (void)op;
    (void)configAlgMap;
    (void)primQueueGenName;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AutoSelectorBase::SelectCcuScheduleAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)topoInfo;
    (void)op;
    (void)configAlgMap;
    (void)primQueueGenName;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AutoSelectorBase::SelectAicpuAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)topoInfo;
    (void)op;
    (void)configAlgMap;
    (void)primQueueGenName;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AutoSelectorBase::SelectAivAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)topoInfo;
    (void)op;
    (void)configAlgMap;
    (void)primQueueGenName;
    return SelectorStatus::NOT_MATCH;
}

} // namespace Hccl
