/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_AHC_BASE_PUB
#define COMM_AHC_BASE_PUB

#include <cmath>
#include "alg_template_base_pub.h"

namespace hccl {

class CommAHCBaseInfo {
public:
    explicit CommAHCBaseInfo(const std::vector<std::vector<u32>>& subGroups);
    virtual ~CommAHCBaseInfo();

    static HcclResult CheckGlobalGroups(std::vector<std::vector<std::vector<u32>>>& globalSubGroups);
    static HcclResult CheckSubGroups(std::vector<std::vector<u32>>& subGroups);
    static HcclResult DisposeSubGroups(
        const u32 rank, const std::vector<std::vector<std::vector<u32>>>& globalSubGroups,
        std::vector<std::vector<u32>>& level0SubGroups, std::vector<std::vector<u32>>& level1SubGroups);
    static HcclResult DisposeSubGroups(
        const u32 rank, const std::vector<std::vector<std::vector<u32>>>& globalSubGroups,
        std::vector<std::vector<u32>>& level0SubGroups, std::vector<std::vector<u32>>& level1SubGroups,
        u64& globalTotalSliceSegment, u32& rankSizeLevel0);
    static HcclResult InitConcAlgOption(std::map<AHCConcOpType, TemplateType>& ahcAlgOption);

    virtual HcclResult Init(AHCOpType opType, std::map<AHCConcOpType, TemplateType>& ahcAlgOption);
    virtual HcclResult SetGlobalTotalSliceSegment(u64 globalTotalSliceSegment);
    HcclResult SetAlgTemplateParam(AHCExtendPreparePara& ahcExtendPreparePara);
    HcclResult SetIsAlignBound(bool isAlignBound);
    HcclResult ParseInputSlice(const std::vector<Slice>& physicalSlices);
    HcclResult TrasLogicSliceToPhysical(std::vector<Slice>& slices, const std::vector<Slice>& physicalSlices) const;
    HcclResult GetNslbDstRanks(u32 rank, std::vector<u32>& dstRanks);
    HcclResult GetRingNslbDstRanks(const u32 rank, const std::vector<u32> commGroups, std::vector<u32>& dstRanks) const;
    HcclResult GetNHRNslbDstRanks(const u32 rank, const std::vector<u32> commGroups, std::vector<u32>& dstRanks) const;
    HcclResult GetNBNslbDstRanks(const u32 rank, const std::vector<u32> commGroups, std::vector<u32>& dstRanks) const;
    HcclResult GetDstRanksByType(
        AHCTemplateType type, const u32 rank, const std::vector<u32> commGroups, std::vector<u32>& dstRanks);

    virtual HcclResult CalcDstRanks(u32 rank, std::set<u32>& dstRanks, AHCLevel ahcLevel = AHCLevel::AHC_LEVEL_0);
    // Reduce-Scatter 及 All-Gather 组内切片逻辑
    virtual HcclResult CalcIntraSlicesAndLinks(
        const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
        std::vector<std::vector<LINK>>& intraLinksVector, std::vector<std::vector<Slice>>& intraSlicesVector);
    // All-Reduce 组内切片逻辑
    virtual HcclResult CalcIntraSlicesAndLinks(
        const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
        std::vector<LINK>& intraLinks, std::vector<Slice>& intraSlices);
    // AHC 算法组间切片逻辑
    virtual HcclResult CalcInterSlicesAndLinks(
        const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
        std::vector<std::vector<LINK>>& interLinksVector, std::vector<std::vector<Slice>>& interSlicesVector,
        std::vector<u32>& logicCardList);
    virtual bool IsNeedInterProc(const u32 rank);

    u32 GetIntraRank(const u32 rank);
    u32 GetInterRank(const u32 groupIdx, const u32 rank);
    u32 GetCommRank(const u32 rank);
    HcclResult GetIntraAlgTemplateOpInstance(
        const AHCOpType opType, std::unique_ptr<AlgTemplateBase>& tempAlg, const HcclDispatcher& dispatcher,
        const u64 reduceAttr, bool extendFlag, AHCExtendPreparePara extendPara,
        AHCLevel ahcLevel = AHCLevel::AHC_LEVEL_0);
    HcclResult GetInterAlgTemplateOpInstance(
        const AHCOpType opType, std::unique_ptr<AlgTemplateBase>& tempAlg, const HcclDispatcher& dispatcher,
        const u64 reduceAttr, bool extendFlag, AHCExtendPreparePara extendPara,
        AHCLevel ahcLevel = AHCLevel::AHC_LEVEL_0);
    void GetIntraCommGroup(u32 rank, std::vector<u32>& intraCommGroup);
    void GetInterCommGroupList(u32 rank, std::vector<std::vector<u32>>& interCommGroupList);

protected:
    AHCOpType opType_;   // 当前处理的算子类型
    u32 minSubGroupIdx_; // 第一层分组最小的分组下标
    u32 maxSubGroupIdx_; // 第二层分组最大的分组下标
    u32 rankSize_;
    bool isAlignBound_;                                  // 切片时是否需要严格对齐 Bound
    bool isContinusSlice_;                               // 物理切片是否连续
    u64 totalSize_;                                      // 当前处理的总数据量
    std::map<AHCConcOpType, TemplateType> ahcAlgOption_; // 内部使用算子的相关配置
    std::map<u32, u32> rankGroupMap_;                    // rank 到分组 group index 的 map
    std::map<u32, u32> rankCommMap_; // rank 在全通信域中的位置，用于Reduce-Scatter/All-Gather的数据搬运
    std::map<u32, u32> groupOriginOffset_;    // broke对齐方式中Reduce-Scatter/All-Gather每个分组起始Offset
    std::vector<std::vector<u32>> subGroups_; // 第一层组内分组信息
    std::vector<std::map<u32, u32>> interRankList_;     // rank 在执行计算时组内的位置
    std::vector<u32> subGroupMaxStreams_;               // 分组最大并发流
    std::vector<std::vector<u32>> logicCardCommGroups_; // 第二层组间逻辑同号卡通信域
private:
    HcclResult GetAlgTemplateOpInstance(
        const AHCOpType opType, std::unique_ptr<AlgTemplateBase>& tempAlg, const HcclDispatcher& dispatcher,
        const u64 reduceAttr, bool extendFlag, AHCExtendPreparePara extendPara, AHCLevel ahcLevel, ConcType concType);
    void GetInterCommGroupIdxList(u32 rank, std::vector<u32>& interCommGroupIdxList);
};

class CommBrokeAlignInfo : public CommAHCBaseInfo {
public:
    explicit CommBrokeAlignInfo(const std::vector<std::vector<u32>>& subGroups);
    ~CommBrokeAlignInfo() override;
    HcclResult Init(AHCOpType opType, std::map<AHCConcOpType, TemplateType>& ahcAlgOption) override;

    bool IsNeedInterProc(const u32 rank) override;
    // Reduce-Scatter 及 All-Gather 组内切片逻辑
    HcclResult CalcIntraSlicesAndLinks(
        const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
        std::vector<std::vector<LINK>>& intraLinksVector, std::vector<std::vector<Slice>>& intraSlicesVector) override;
    // All-Reduce 组内切片逻辑
    HcclResult CalcIntraSlicesAndLinks(
        const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
        std::vector<LINK>& intraLinks, std::vector<Slice>& intraSlices) override;
    // AHC 算法组间切片逻辑
    HcclResult CalcInterSlicesAndLinks(
        const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
        std::vector<std::vector<LINK>>& interLinksVector, std::vector<std::vector<Slice>>& interSlicesVector,
        std::vector<u32>& logicCardList) override;

private:
    HcclResult
    PrepareIntraSlices(const u32 rank, const u32 dataUnitSize, const u64 count, std::vector<Slice>& intraSlices) const;
    HcclResult CalcInterSlicesAndLinksForRS(
        const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
        std::vector<std::vector<LINK>>& interLinksVector, std::vector<std::vector<Slice>>& interSlicesVector,
        std::vector<u32>& logicCardList);
    HcclResult CalcInterSlicesAndLinksForAR(
        const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
        std::vector<std::vector<LINK>>& interLinksVector, std::vector<std::vector<Slice>>& interSlicesVector);

    std::map<u32, std::vector<u32>> completeGroupOrder_;
    std::map<u32, std::vector<u32>> emptyGroupOrder_;
};

class CommAHCAlignInfo : public CommAHCBaseInfo {
public:
    explicit CommAHCAlignInfo(const std::vector<std::vector<u32>>& subGroups);
    ~CommAHCAlignInfo() override;
    HcclResult Init(AHCOpType opType, std::map<AHCConcOpType, TemplateType>& ahcAlgOption) override;
    HcclResult SetGlobalTotalSliceSegment(u64 globalTotalSliceSegment) override;

    // Reduce-Scatter 及 All-Gather 组内切片逻辑
    HcclResult CalcIntraSlicesAndLinks(
        const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
        std::vector<std::vector<LINK>>& intraLinksVector, std::vector<std::vector<Slice>>& intraSlicesVector) override;
    // All-Reduce 组内切片逻辑
    HcclResult CalcIntraSlicesAndLinks(
        const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
        std::vector<LINK>& intraLinks, std::vector<Slice>& intraSlices) override;
    // AHC 算法组间切片逻辑
    HcclResult CalcInterSlicesAndLinks(
        const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
        std::vector<std::vector<LINK>>& interLinksVector, std::vector<std::vector<Slice>>& interSlicesVector,
        std::vector<u32>& logicCardList) override;

private:
    HcclResult InitSliceInfo();
    HcclResult InitLogicCardInfo();
    bool CompareLogicCardExcuteOrder(u32 i, u32 j);
    HcclResult GetLogicCardExecuteOrder(u32 rank, std::vector<u32>& executeOrder);
    HcclResult InitMapInfo();
    HcclResult SliceSizeAlignBound(
        Slice& slice, u64 offsetCount, u64 sliceSizeCalculated, const u64 boundSize, u32 boundOffsetCount,
        u32& curOffset) const;
    HcclResult PrepareIntraSlices(
        const u32 rank, const u32 dataUnitSize, const u64 count, std::vector<std::vector<Slice>>& intraSlicesVector);
    HcclResult CalcInterSlicesAndLinksForRS(
        const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
        std::vector<std::vector<LINK>>& interLinksVector, std::vector<std::vector<Slice>>& interSlicesVector,
        std::vector<u32>& logicCardList);
    HcclResult PrepareWholeLogicSlices(
        const Slice& intraSlice, const u64 sliceSizeAligned, const u32 originOffset,
        std::vector<Slice>& logicGroupSlice, std::vector<u32>& logicCardList);
    HcclResult PreparePartialLogicSlices(
        const Slice& intraSlice, const u64 sliceSizeAligned, const u32 originOffset,
        std::vector<Slice>& logicGroupSlice, std::vector<u32>& logicCardList);
    HcclResult
    PrepareEmptyLogicSlices(std::vector<Slice>& logicGroupSlice, const std::vector<u32>& logicCardList) const;
    HcclResult CalcLogicSlicesAndLinks(
        std::vector<Slice>& logicGroupSlice, std::vector<u32>& logicCardList, const std::vector<LINK>& links,
        std::vector<std::vector<LINK>>& interLinksVector, std::vector<std::vector<Slice>>& interSlicesVector);
    HcclResult CalcInterSlicesAndLinksForAR(
        const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
        std::vector<std::vector<LINK>>& interLinksVector, std::vector<std::vector<Slice>>& interSlicesVector,
        std::vector<u32>& logicCardList);

    std::vector<std::vector<u32>> logicCardGroup_;     // 逻辑同号卡并发流分组
    std::vector<u32> logicCardExecuteOffset_;          // 逻辑同号卡在并发流分组中的 offset
    std::vector<u32> logicCardSliceSize_;              // 逻辑同号卡切片大小
    std::map<u32, std::vector<u32>> rankLogicCardMap_; // rank 号到逻辑同号卡的映射
    std::map<u32, std::vector<u32>> rankLogicCardOrderMap_; // rank 号到并发流执行的多个非对称同号卡序列映射
    std::vector<u32> logicCardSliceOffset_;                 // 逻辑同号卡对齐后的边界数组
    u32 totalSliceSegment_;                                 // 切片总大小，分组数的最小公倍数
    u32 globalTotalSliceSegment_;                           // 全局切片总大小
};
} // namespace hccl

#endif /* COMM_AHC_BASE_PUB */
