/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASYMMETRIC_HIERARCHICAL_CONCATENATE_ALG_TEMPLATE_BASE_PUB_H
#define ASYMMETRIC_HIERARCHICAL_CONCATENATE_ALG_TEMPLATE_BASE_PUB_H

#include <cmath>
#include <algorithm>
#include "alg_template_base_pub.h"
#include "asymmetric_hierarchical_concatenate_base_pub.h"
#include "comm_ahc_base_pub.h"
#include "device_capacity.h"

namespace hccl {

constexpr u32 NSLBDP_MAX_PHASE = 255;

class AHCAlgTemplateBase : public AlgTemplateBase {
public:
    explicit AHCAlgTemplateBase(const HcclDispatcher dispatcher);
    ~AHCAlgTemplateBase() override;

    HcclResult Prepare(u64 reduceAttrBitMap, HcomCollOpInfo* opInfo = nullptr) override;

    HcclResult Prepare(
        u64 totalCount, const std::vector<std::vector<std::vector<u32>>>& globalSubGroups,
        std::map<AHCConcOpType, TemplateType>& ahcAlgOption, bool extendFlag = false,
        AHCExtendPreparePara extendPara = AHCExtendPreparePara()) override;

    HcclResult GetFftsPhase(u32& fftsPhase) const;
    HcclResult SetFftsPhase(u32& fftsPhase);
    HcclResult
    GetNslbAdjInfoPro(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo);
    u64 reduceAttr_;
    bool needTraslateSliceAddr_; // 是否需要地址映射，当前仅 reduce scatter 和 allgather 需要设置
    std::vector<Slice>
        physicalSlices_; // 记录传入的物理slice切片，算法计算的 slices_为逻辑切片，执行时需要转换为物理切片
protected:
    virtual HcclResult CommAHCInfoInit();
    virtual HcclResult DisposeSubGroups(const u32 rank);

    HcclResult PrepareRunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;
    HcclResult
    PrepareAlgTemplate(std::unique_ptr<AlgTemplateBase>& tempAlg, const std::vector<Slice>& slices, AHCOpType opType);
    HcclResult RunInstance(
        const u32 rank, const std::vector<LINK>& links, std::vector<Slice>& slices,
        std::unique_ptr<AlgTemplateBase>& tempAlg, AHCOpType opType);
    HcclResult MemcpyForSingleOp(const u32 rank, AHCOpType opType);

    u32 rankSize_;
    std::unique_ptr<CommAHCBaseInfo> commAHCBaseInfo_;
    std::map<AHCConcOpType, TemplateType> ahcAlgOption_;
    std::vector<std::vector<u32>> level0SubGroups_;
    std::vector<std::vector<u32>> level1SubGroups_;
    std::vector<std::vector<std::vector<u32>>> globalSubGroups_;
    u64 totalCount_; // 完整数据量大小，用于判断是否为hugeData
    bool extendFlag_;
    AHCExtendPreparePara ahcExtendPreparePara_;

private:
};

class AllGatherAHCBase : public AHCAlgTemplateBase {
public:
    explicit AllGatherAHCBase(const HcclDispatcher dispatcher);
    ~AllGatherAHCBase() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;
    HcclResult
    GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo) override;

private:
    HcclResult RunIntraAllGather(
        const u32 rank, const std::vector<LINK>& links, const std::unique_ptr<CommAHCBaseInfo>& commAHCBaseInfo);
    virtual HcclResult RunInterAllGather(
        const u32 rank, const std::vector<LINK>& links, const std::unique_ptr<CommAHCBaseInfo>& commAHCBaseInfo)
        = 0;
};

class AllReduceAHCBase : public AHCAlgTemplateBase {
public:
    explicit AllReduceAHCBase(const HcclDispatcher dispatcher);
    ~AllReduceAHCBase() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;
    HcclResult
    GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo) override;

private:
    HcclResult RunIntraReduceScatter(
        const u32 rank, const std::vector<LINK>& links, const std::unique_ptr<CommAHCBaseInfo>& commAHCBaseInfo);
    HcclResult RunIntraAllGather(
        const u32 rank, const std::vector<LINK>& links, const std::unique_ptr<CommAHCBaseInfo>& commAHCBaseInfo);
    virtual HcclResult RunInterAllReduce(
        const u32 rank, const std::vector<LINK>& links, const std::unique_ptr<CommAHCBaseInfo>& commAHCBaseInfo)
        = 0;
};

class ReduceScatterAHCBase : public AHCAlgTemplateBase {
public:
    explicit ReduceScatterAHCBase(const HcclDispatcher dispatcher);
    ~ReduceScatterAHCBase() override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;
    HcclResult
    GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo) override;

private:
    HcclResult RunIntraReduceScatter(
        const u32 rank, const std::vector<LINK>& links, const std::unique_ptr<CommAHCBaseInfo>& commAHCBaseInfo);
    virtual HcclResult RunInterReduceScatter(
        const u32 rank, const std::vector<LINK>& links, const std::unique_ptr<CommAHCBaseInfo>& commAHCBaseInfo)
        = 0;
};
} // namespace hccl

#endif /* ASYMMETRIC_HIERARCHICAL_CONCATENATE_ALG_TEMPLATE_BASE_PUB_H */
