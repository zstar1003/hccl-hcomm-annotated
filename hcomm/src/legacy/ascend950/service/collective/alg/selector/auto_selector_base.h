/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_AUTO_SELECTOR
#define HCCLV2_AUTO_SELECTOR

#include "base_selector.h"
#include "coll_alg_params.h"
#include "coll_operator.h"
#include "virtual_topo.h"

namespace Hccl {

constexpr uint64_t SMALL_COUNT_512KB = 512 * 1024;   // Byte, UB协议一次传输的最大size
constexpr uint64_t LARGE_COUNT_1024KB = 1024 * 1024; // Byte, 可掩盖多mission尾块开销
constexpr uint64_t AIV_ALL_REDUCE_SMALL_COUNT = 64 * 1024;
constexpr uint64_t CCU_PARALLEL_MAX_DATA_SIZE = 64 * 1024 * 1024;
constexpr double DEFAULT_RANK_SIZE = 8.0;
class AutoSelectorBase : public BaseSelector {
public:
    SelectorStatus Select(const CollAlgOperator& op, CollAlgParams& params, std::string& primQueueGenName) override;
    bool IsDefaultAlg(const HcclAlgoType algoType) const;
    HcclAlgoType
    GetLevel0AlgoType(const CollAlgOperator& op, const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap) const;
    bool IsSmallData(const u64 dataSize) const;
    bool IsLargeData(const u64 dataSize) const;
    virtual SelectorStatus SelectCcuMsAlgo(
        const TopoInfo& topoInfo, const CollAlgOperator& op,
        const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const;
    virtual SelectorStatus SelectCcuScheduleAlgo(
        const TopoInfo& topoInfo, const CollAlgOperator& op,
        const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const;
    virtual SelectorStatus SelectAicpuAlgo(
        const TopoInfo& topoInfo, const CollAlgOperator& op,
        const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const;
    virtual SelectorStatus SelectAivAlgo(
        const TopoInfo& topoInfo, const CollAlgOperator& op,
        const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const;
    bool IsStarsState(const OpExecuteConfig& opExecuteConfig) const;
    bool IsSmallDataCCU(const u64 dataSize, const u64 rankSize) const;

protected:
    u64 dataSize_;
};

} // namespace Hccl
#endif
