/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_ahc_base_pub.h"
#include "alg_template_register.h"
#include "calc_ahc_template_register.h"

#include <iostream>
#include <fstream>

namespace hccl {

// AHC 通信关系注册
AHCCommCalcFuncRegistry::AHCCommCalcFuncRegistry()
{
    commCalcFuncCreators_.resize(static_cast<u32>(AHCTemplateType::AHC_TEMPLATE_RESERVED), nullptr);
}

AHCCommCalcFuncRegistry& AHCCommCalcFuncRegistry::Instance()
{
    static AHCCommCalcFuncRegistry globalAlgTemplateRegistry;
    return globalAlgTemplateRegistry;
}

HcclResult AHCCommCalcFuncRegistry::Register(AHCTemplateType type, AHCCommCalcFuncPtr funPtr)
{
    if (type >= AHCTemplateType::AHC_TEMPLATE_RESERVED) {
        HCCL_ERROR("[AHCCommCalcFuncRegistry]template type[%d] out of range.", type);
        return HcclResult::HCCL_E_INTERNAL;
    }

    const std::lock_guard<std::mutex> lock(mu_);
    if (commCalcFuncCreators_[static_cast<u32>(type)] != nullptr) {
        HCCL_ERROR("[AHCCommCalcFuncRegistry]template type[%d] already registered.", type);
        return HcclResult::HCCL_E_INTERNAL;
    }
    commCalcFuncCreators_[static_cast<u32>(type)] = funPtr;
    return HcclResult::HCCL_SUCCESS;
}

AHCCommCalcFuncPtr AHCCommCalcFuncRegistry::GetCommCalcFunction(AHCTemplateType type)
{
    if (type >= AHCTemplateType::AHC_TEMPLATE_RESERVED) {
        HCCL_ERROR("[AHCCommCalcFuncRegistry]template type[%d] out of range.", type);
        return nullptr;
    }

    if (commCalcFuncCreators_[static_cast<u32>(type)] == nullptr) {
        HCCL_DEBUG("[AHCCommCalcFuncRegistry]Creator for template type[%d] has not registered.", type);
        return nullptr;
    }
    HCCL_DEBUG("[AHCCommCalcFuncRegistry][GetCommCalcFunction]get template by type[%d]", type);
    return commCalcFuncCreators_[static_cast<u32>(type)];
}

// AHC 核心算法逻辑
CommAHCBaseInfo::CommAHCBaseInfo(const std::vector<std::vector<u32>>& subGroups)
    : minSubGroupIdx_(0),
      maxSubGroupIdx_(0),
      rankSize_(0),
      isAlignBound_(true),
      isContinusSlice_(true),
      subGroups_(subGroups)
{
    // rank 到 group index 的map 初始化以及最大最小分组下标的初始化
    u32 minSubGroupSize = subGroups_[0].size();
    u32 maxSubGroupSize = subGroups_[0].size();
    u32 curIdx = 0;
    u32 curOffset = 0;

    for (u32 i = 0; i < subGroups_.size(); ++i) {
        rankSize_ = rankSize_ + subGroups[i].size();
        if (subGroups_[i].size() < minSubGroupSize) {
            minSubGroupSize = subGroups_[i].size();
            minSubGroupIdx_ = i;
        }
        if (subGroups_[i].size() > maxSubGroupSize) {
            maxSubGroupSize = subGroups_[i].size();
            maxSubGroupIdx_ = i;
        }
        for (u32 j = 0; j < subGroups_[i].size(); ++j) {
            rankGroupMap_.insert(std::make_pair(subGroups_[i][j], i));
            rankCommMap_.insert(std::make_pair(subGroups_[i][j], curIdx));
            curIdx++;
        }
        groupOriginOffset_.insert(std::make_pair(i, curOffset));
        curOffset = curOffset + subGroups_[i].size();
    }

    HCCL_DEBUG("[CommAHCBaseInfo] minSubGroupSize[%u] maxSubGroupSize[%u]", minSubGroupSize, maxSubGroupSize);
}

CommAHCBaseInfo::~CommAHCBaseInfo() {}

HcclResult CommAHCBaseInfo::Init(AHCOpType opType, [[maybe_unused]] std::map<AHCConcOpType, TemplateType>& ahcAlgOption)
{
    (void)opType;
    return HCCL_SUCCESS;
}

HcclResult CommAHCBaseInfo::DisposeSubGroups(
    const u32 rank, const std::vector<std::vector<std::vector<u32>>>& globalSubGroups,
    std::vector<std::vector<u32>>& level0SubGroups, std::vector<std::vector<u32>>& level1SubGroups)
{
    bool isRankLevel0SubGroup = false;
    for (u32 i = 0; i < globalSubGroups.size(); i++) {
        std::vector<u32> level1SubGroup;
        for (u32 j = 0; j < globalSubGroups[i].size(); j++) {
            std::vector<u32> curSubGroup = globalSubGroups[i][j];
            for (u32 k = 0; k < curSubGroup.size(); k++) {
                if (curSubGroup[k] == rank) {
                    isRankLevel0SubGroup = true;
                }
                level1SubGroup.push_back(curSubGroup[k]);
            }
        }
        if (isRankLevel0SubGroup) {
            level0SubGroups = globalSubGroups[i];
            isRankLevel0SubGroup = false;
        }
        level1SubGroups.push_back(level1SubGroup);
    }
    return HCCL_SUCCESS;
}

HcclResult CommAHCBaseInfo::DisposeSubGroups(
    const u32 rank, const std::vector<std::vector<std::vector<u32>>>& globalSubGroups,
    std::vector<std::vector<u32>>& level0SubGroups, std::vector<std::vector<u32>>& level1SubGroups,
    u64& globalTotalSliceSegment, u32& rankSizeLevel0)
{
    u64 tmpTotalSliceSegmentLevel1 = 1;
    bool isRankLevel0SubGroup = false;
    for (u32 i = 0; i < globalSubGroups.size(); i++) {
        std::vector<u32> level1SubGroup;
        u64 tmpTotalSliceSegmentLevel0 = 1;
        for (u32 j = 0; j < globalSubGroups[i].size(); j++) {
            std::vector<u32> curSubGroup = globalSubGroups[i][j];
            u64 level0GroupSize = static_cast<u64>(curSubGroup.size());
            tmpTotalSliceSegmentLevel0 = tmpTotalSliceSegmentLevel0 * level0GroupSize
                                         / std::__gcd(tmpTotalSliceSegmentLevel0, level0GroupSize);
            for (u32 k = 0; k < curSubGroup.size(); k++) {
                if (curSubGroup[k] == rank) {
                    isRankLevel0SubGroup = true;
                }
                level1SubGroup.push_back(curSubGroup[k]);
            }
        }
        if (isRankLevel0SubGroup) {
            level0SubGroups = globalSubGroups[i];
            rankSizeLevel0 = level1SubGroup.size();
            isRankLevel0SubGroup = false;
        }
        tmpTotalSliceSegmentLevel0 = tmpTotalSliceSegmentLevel0 * static_cast<u64>(level1SubGroup.size());
        globalTotalSliceSegment = globalTotalSliceSegment * tmpTotalSliceSegmentLevel0
                                  / std::__gcd(globalTotalSliceSegment, tmpTotalSliceSegmentLevel0);
        u64 level1GroupSize = static_cast<u64>(level1SubGroup.size());
        tmpTotalSliceSegmentLevel1
            = tmpTotalSliceSegmentLevel1 * level1GroupSize / std::__gcd(tmpTotalSliceSegmentLevel1, level1GroupSize);
        level1SubGroups.push_back(level1SubGroup);
    }
    tmpTotalSliceSegmentLevel1 = tmpTotalSliceSegmentLevel1 * static_cast<u64>(level1SubGroups.size());
    globalTotalSliceSegment = globalTotalSliceSegment * tmpTotalSliceSegmentLevel1
                              / std::__gcd(globalTotalSliceSegment, tmpTotalSliceSegmentLevel1);
    return HCCL_SUCCESS;
}

HcclResult CommAHCBaseInfo::InitConcAlgOption(std::map<AHCConcOpType, TemplateType>& ahcAlgOption)
{
    // 初始化设置拼接算法，intra NHR,inter RING ; 每个 level+conc 类型对应的算子类型约束一致
    std::map<AHCConcOpType, TemplateType> ahcAlgOptionInstance
        = {{{AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTRA, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER},
            TemplateType::TEMPLATE_REDUCESCATTER_NHR},
           {{AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTRA, AHCOpType::AHC_OP_TYPE_ALLREDUCE},
            TemplateType::TEMPLATE_ALL_REDUCE_NHR},
           {{AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTRA, AHCOpType::AHC_OP_TYPE_ALLGATHER},
            TemplateType::TEMPLATE_ALL_GATHER_NHR},

           {{AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER},
            TemplateType::TEMPLATE_REDUCESCATTER_RING},
           {{AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_ALLREDUCE},
            TemplateType::TEMPLATE_ALL_REDUCE_RING},
           {{AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_ALLGATHER},
            TemplateType::TEMPLATE_ALL_GATHER_RING},

           {{AHCLevel::AHC_LEVEL_1, ConcType::CONC_INTRA, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER},
            TemplateType::TEMPLATE_REDUCESCATTER_NHR},
           {{AHCLevel::AHC_LEVEL_1, ConcType::CONC_INTRA, AHCOpType::AHC_OP_TYPE_ALLREDUCE},
            TemplateType::TEMPLATE_ALL_REDUCE_NHR},
           {{AHCLevel::AHC_LEVEL_1, ConcType::CONC_INTRA, AHCOpType::AHC_OP_TYPE_ALLGATHER},
            TemplateType::TEMPLATE_ALL_GATHER_NHR},

           {{AHCLevel::AHC_LEVEL_1, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER},
            TemplateType::TEMPLATE_REDUCESCATTER_RING},
           {{AHCLevel::AHC_LEVEL_1, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_ALLREDUCE},
            TemplateType::TEMPLATE_ALL_REDUCE_RING},
           {{AHCLevel::AHC_LEVEL_1, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_ALLGATHER},
            TemplateType::TEMPLATE_ALL_GATHER_RING}};
    ahcAlgOption = ahcAlgOptionInstance;
    return HCCL_SUCCESS;
}

HcclResult CommAHCBaseInfo::SetIsAlignBound(bool isAlignBound)
{
    isAlignBound_ = isAlignBound;
    HCCL_DEBUG("[CommAHCBaseInfo][SetIsAlignBound] isAlignBound_ set [%d].", isAlignBound_);
    return HCCL_SUCCESS;
}

HcclResult CommAHCBaseInfo::SetGlobalTotalSliceSegment(u64 globalTotalSliceSegment)
{
    (void)globalTotalSliceSegment;
    return HCCL_SUCCESS;
}

HcclResult CommAHCBaseInfo::ParseInputSlice(const std::vector<Slice>& physicalSlices)
{
    totalSize_ = 0;

    for (u32 i = 0; i < physicalSlices.size(); i++) {
        HCCL_DEBUG(
            "[CommAHCBaseInfo][ParseInputSlice] physicalSlices index[%u] offset[%llu] size[%llu]", i,
            physicalSlices[i].offset, physicalSlices[i].size);

        if (i >= 1 && physicalSlices[i].size != 0) {
            if (physicalSlices[i - 1].offset + physicalSlices[i - 1].size != physicalSlices[i].offset) {
                isContinusSlice_ = false;
            }
        }
        totalSize_ = totalSize_ + physicalSlices[i].size;
    }
    HCCL_DEBUG(
        "[CommAHCBaseInfo][ParseInputSlice] totalSize_[%llu] isContinusSlice_[%u]", totalSize_, isContinusSlice_);
    return HCCL_SUCCESS;
}

HcclResult
CommAHCBaseInfo::TrasLogicSliceToPhysical(std::vector<Slice>& slices, const std::vector<Slice>& physicalSlices) const
{
    for (u32 i = 0; i < slices.size(); i++) {
        u64 logicOffset = 0;
        bool translateSuccess = false;

        HCCL_DEBUG(
            "[CommAHCBaseInfo][TrasLogicSliceToPhysical] translate slice offset[%llu] size[%llu]", slices[i].offset,
            slices[i].size);

        for (u32 j = 0; j < physicalSlices.size(); j++) {
            bool startOffsetInRange
                = ((slices[i].offset >= logicOffset)
                   && (slices[i].offset <= (logicOffset + physicalSlices[j].size - 1)));
            bool endOffsetInRange
                = (((slices[i].offset + slices[i].size - 1) >= logicOffset)
                   && ((slices[i].offset + slices[i].size - 1) <= (logicOffset + physicalSlices[j].size - 1)));

            if (isContinusSlice_ && startOffsetInRange) { // 连续silie，检查逻辑slice起始边界在物理slice范围,则正常翻译
                translateSuccess = true;
                HCCL_DEBUG(
                    "[CommAHCBaseInfo][TrasLogicSliceToPhysical] translate to continuous slice offset[%llu] size[%llu] "
                    "in physical slice offset[%llu] size[%llu]",
                    slices[i].offset, slices[i].size, physicalSlices[j].offset, physicalSlices[j].size);
                break;
            } else if (
                startOffsetInRange
                && endOffsetInRange) { // 非连续slice，检查逻辑slice起始和结束边界在物理slice范围,则正常翻译
                slices[i].offset = physicalSlices[j].offset + slices[i].offset - logicOffset;
                translateSuccess = true;
                HCCL_DEBUG(
                    "[CommAHCBaseInfo][TrasLogicSliceToPhysical] translate to slice offset[%llu] size[%llu] in "
                    "physical slice offset[%llu] size[%llu]",
                    slices[i].offset, slices[i].size, physicalSlices[j].offset, physicalSlices[j].size);
                break;
            } else if (
                !isContinusSlice_ && startOffsetInRange && !endOffsetInRange
                && slices[i].size != 0) { // 逻辑slice跨越非连续物理slice边界，异常退出
                HCCL_ERROR(
                    "[CommAHCBaseInfo][TrasLogicSliceToPhysical] logic slice index[%u] offset[%llu] size[%llu], "
                    "physical index[%u] start offset[%llu] end offset[%llu]",
                    i, slices[i].offset, slices[i].size, j, logicOffset, (logicOffset + physicalSlices[j].size));
                return HCCL_E_PARA;
            }

            logicOffset = logicOffset + physicalSlices[j].size;
        }

        // 检查翻译结果
        if (slices[i].size == 0) {
            // 0 切片特殊处理
            slices[i].offset = logicOffset;
        } else if (!translateSuccess) {
            HCCL_ERROR(
                "[CommAHCBaseInfo][TrasLogicSliceToPhysical] slice index[%u] offset[%llu] size[%llu] translate ERROR",
                i, slices[i].offset, slices[i].size);
            return HCCL_E_PARA;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult CommAHCBaseInfo::CheckGlobalGroups(std::vector<std::vector<std::vector<u32>>>& globalSubGroups)
{
    if (globalSubGroups.size() == 0) {
        HCCL_ERROR("[CommAHCBaseInfo][globalSubGroups] globalSubGroups.size() == 0, globalSubGroups init ERROR");
        return HCCL_E_PARA;
    }

    for (u32 i = 0; i < globalSubGroups.size(); i++) {
        CHK_RET(CheckSubGroups(globalSubGroups[i]));
    }
    return HCCL_SUCCESS;
}

HcclResult CommAHCBaseInfo::CheckSubGroups(std::vector<std::vector<u32>>& subGroups)
{
    if (subGroups.size() == 0) {
        HCCL_ERROR("[CommAHCBaseInfo][CheckSubGroups] subGroups.size() == 0, subGroups init ERROR");
        return HCCL_E_PARA;
    }

    for (u32 i = 0; i < subGroups.size(); i++) {
        if (subGroups[i].size() == 0) {
            HCCL_ERROR("[CommAHCBaseInfo][CheckSubGroups] subGroups[%u].size() == 0, subGroups[] init ERROR", i);
            return HCCL_E_PARA;
        }
        for (u32 j = 0; j < subGroups[i].size(); j++) {
            HCCL_DEBUG("[CommAHCBaseInfo][CheckSubGroups] subGroups[%u][%u] = %u", i, j, subGroups[i][j]);
        }
    }
    return HCCL_SUCCESS;
}

void CommAHCBaseInfo::GetIntraCommGroup(u32 rank, std::vector<u32>& intraCommGroup)
{
    u32 groupIndex = rankGroupMap_[rank];
    intraCommGroup = subGroups_[groupIndex];
}

void CommAHCBaseInfo::GetInterCommGroupIdxList(u32 rank, std::vector<u32>& interCommGroupIdxList)
{
    // broke 方式的合法vector大小为0或1,AHC 方式的合法vector大小大于等于1
    for (u32 i = 0; i < logicCardCommGroups_.size(); ++i) {
        for (u32 j = 0; j < logicCardCommGroups_[i].size(); ++j) {
            if (rank == logicCardCommGroups_[i][j]) {
                interCommGroupIdxList.push_back(i);
            }
        }
    }
}

void CommAHCBaseInfo::GetInterCommGroupList(u32 rank, std::vector<std::vector<u32>>& interCommGroupList)
{
    // broke 方式的合法vector大小为0或1,AHC 方式的合法vector大小大于等于1
    for (u32 i = 0; i < logicCardCommGroups_.size(); ++i) {
        for (u32 j = 0; j < logicCardCommGroups_[i].size(); ++j) {
            if (rank == logicCardCommGroups_[i][j]) {
                interCommGroupList.push_back(logicCardCommGroups_[i]);
            }
        }
    }
}

HcclResult CommAHCBaseInfo::CalcDstRanks(u32 rank, std::set<u32>& dstRanks, AHCLevel ahcLevel)
{
    // 组内和组间通信域计算
    std::vector<u32> intraCommGroup;
    std::vector<u32> interCommGroupIdxList;

    GetIntraCommGroup(rank, intraCommGroup);
    GetInterCommGroupIdxList(rank, interCommGroupIdxList);
    for (u32 i = 0; i < intraCommGroup.size(); i++) {
        HCCL_DEBUG("[CommAHCBaseInfo][CalcDstRanks] intraCommGroup[%u] = [%u]", i, intraCommGroup[i]);
    }
    for (u32 i = 0; i < interCommGroupIdxList.size(); i++) {
        for (u32 j = 0; j < logicCardCommGroups_[interCommGroupIdxList[i]].size(); j++) {
            HCCL_DEBUG(
                "[CommAHCBaseInfo][CalcDstRanks] Rank[%u] logicCardCommGroups_[%u][%u] = [%u]", rank,
                interCommGroupIdxList[i], j, logicCardCommGroups_[interCommGroupIdxList[i]][j]);
        }
    }

    // 组内通信关系计算
    AHCConcOpType concOpType;
    concOpType.ahcLevel = ahcLevel;
    concOpType.concType = ConcType::CONC_INTRA;
    concOpType.ahcOpType = AHCOpType::AHC_OP_TYPE_ALLREDUCE;

    TemplateType algType = ahcAlgOption_[concOpType];
    HCCL_DEBUG(
        "[CommAHCBaseInfo][CalcDstRanks] Level[%u] ConcType[%u] choose algType[%u]", ahcLevel, ConcType::CONC_INTRA,
        algType);

    auto iterAHCCaclTemplateType = templateToAHCCalcTemplateMap.find(algType);
    if (iterAHCCaclTemplateType == templateToAHCCalcTemplateMap.end()) {
        HCCL_ERROR("[CommAHCBaseInfo][CalcDstRanks] intra algo type[%u] is invalid, is not register.", algType);
        return HCCL_E_PARA;
    }

    AHCCommCalcFuncPtr intraFunctionPtr
        = AHCCommCalcFuncRegistry::Instance().GetCommCalcFunction(iterAHCCaclTemplateType->second);
    CHK_PTR_NULL(intraFunctionPtr);
    intraFunctionPtr(GetIntraRank(rank), intraCommGroup, dstRanks);

    // 组间通信关系计算
    concOpType.concType = ConcType::CONC_INTER;
    algType = ahcAlgOption_[concOpType];
    HCCL_DEBUG(
        "[CommAHCBaseInfo][CalcDstRanks] Level[%u] ConcType[%u] choose algType[%u]", ahcLevel, ConcType::CONC_INTER,
        algType);

    iterAHCCaclTemplateType = templateToAHCCalcTemplateMap.find(algType);
    if (iterAHCCaclTemplateType == templateToAHCCalcTemplateMap.end()) {
        HCCL_ERROR("[CommAHCBaseInfo][CalcDstRanks] inter algo type[%u] is invalid, is not register.", algType);
        return HCCL_E_PARA;
    }

    AHCCommCalcFuncPtr interFunctionPtr
        = AHCCommCalcFuncRegistry::Instance().GetCommCalcFunction(iterAHCCaclTemplateType->second);
    CHK_PTR_NULL(interFunctionPtr);
    for (u32 i = 0; i < interCommGroupIdxList.size(); ++i) {
        interFunctionPtr(
            GetInterRank(interCommGroupIdxList[i], rank), logicCardCommGroups_[interCommGroupIdxList[i]], dstRanks);
    }

    return HCCL_SUCCESS;
}

HcclResult CommAHCBaseInfo::GetNslbDstRanks(u32 rank, std::vector<u32>& dstRanks)
{
    HCCL_DEBUG("[NSLB-AHC] entry GetNslbDstRanks rank[%u]", rank);
    std::vector<u32> intraCommGroup;
    std::vector<u32> interCommGroupIdxList;

    GetIntraCommGroup(rank, intraCommGroup);
    GetInterCommGroupIdxList(rank, interCommGroupIdxList);

    // 组间通信关系计算
    AHCConcOpType concOpType;
    concOpType.ahcLevel = AHCLevel::AHC_LEVEL_0;
    concOpType.concType = ConcType::CONC_INTER;
    concOpType.ahcOpType = AHCOpType::AHC_OP_TYPE_ALLREDUCE;

    TemplateType algType = ahcAlgOption_[concOpType];
    auto iterAHCCaclTemplateType = templateToAHCCalcTemplateMap.find(algType);
    if (iterAHCCaclTemplateType == templateToAHCCalcTemplateMap.end()) {
        HCCL_ERROR("[CommAHCBaseInfo][CalcDstRanks] inter algo type[%u] is invalid, is not register.", algType);
        return HCCL_E_PARA;
    }

    AHCCommCalcFuncPtr interFunctionPtr
        = AHCCommCalcFuncRegistry::Instance().GetCommCalcFunction(iterAHCCaclTemplateType->second);
    CHK_PTR_NULL(interFunctionPtr);
    for (u32 i = 0; i < interCommGroupIdxList.size(); ++i) {
        AHCTemplateType type = iterAHCCaclTemplateType->second;
        GetDstRanksByType(
            type, GetInterRank(interCommGroupIdxList[i], rank), logicCardCommGroups_[interCommGroupIdxList[i]],
            dstRanks);
    }

    return HCCL_SUCCESS;
}

u32 CommAHCBaseInfo::GetIntraRank(const u32 rank)
{
    u32 intraRank = 0;
    for (u32 i = 0; i < subGroups_[rankGroupMap_[rank]].size(); i++) {
        if (subGroups_[rankGroupMap_[rank]][i] == rank) {
            intraRank = i;
            return intraRank;
        }
    }
    HCCL_DEBUG("[CommAHCBaseInfo][GetIntraRank] rank[%u] not found", rank);
    return intraRank;
}

u32 CommAHCBaseInfo::GetInterRank(const u32 groupIdx, const u32 rank)
{
    u32 subGroupsIdx = rankGroupMap_[rank];
    u32 interRank = interRankList_[groupIdx][subGroupsIdx];
    HCCL_DEBUG("[CommAHCBaseInfo][GetInterRank] rank[%u] group[%u] interRank[%u]", rank, groupIdx, interRank);
    return interRank;
}

u32 CommAHCBaseInfo::GetCommRank(const u32 rank) { return rankCommMap_[rank]; }

HcclResult CommAHCBaseInfo::CalcIntraSlicesAndLinks(
    const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
    std::vector<std::vector<LINK>>& intraLinksVector, std::vector<std::vector<Slice>>& intraSlicesVector)
{
    (void)rank;
    (void)dataUnitSize;
    (void)count;
    (void)links;
    (void)intraLinksVector;
    (void)intraSlicesVector;
    return HCCL_SUCCESS;
}

HcclResult CommAHCBaseInfo::CalcIntraSlicesAndLinks(
    const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
    std::vector<LINK>& intraLinks, std::vector<Slice>& intraSlices)
{
    (void)rank;
    (void)dataUnitSize;
    (void)count;
    (void)links;
    (void)intraLinks;
    (void)intraSlices;
    return HCCL_SUCCESS;
}

HcclResult CommAHCBaseInfo::CalcInterSlicesAndLinks(
    const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
    std::vector<std::vector<LINK>>& interLinksVector, std::vector<std::vector<Slice>>& interSlicesVector,
    std::vector<u32>& logicCardList)
{
    (void)rank;
    (void)dataUnitSize;
    (void)count;
    (void)links;
    (void)interLinksVector;
    (void)interSlicesVector;
    (void)logicCardList;
    return HCCL_SUCCESS;
}

HcclResult CommAHCBaseInfo::GetIntraAlgTemplateOpInstance(
    const AHCOpType opType, std::unique_ptr<AlgTemplateBase>& tempAlg, const HcclDispatcher& dispatcher,
    const u64 reduceAttr, bool extendFlag, AHCExtendPreparePara extendPara, AHCLevel ahcLevel)
{
    return GetAlgTemplateOpInstance(
        opType, tempAlg, dispatcher, reduceAttr, extendFlag, extendPara, ahcLevel, ConcType::CONC_INTRA);
}

HcclResult CommAHCBaseInfo::GetInterAlgTemplateOpInstance(
    const AHCOpType opType, std::unique_ptr<AlgTemplateBase>& tempAlg, const HcclDispatcher& dispatcher,
    const u64 reduceAttr, bool extendFlag, AHCExtendPreparePara extendPara, AHCLevel ahcLevel)
{
    return GetAlgTemplateOpInstance(
        opType, tempAlg, dispatcher, reduceAttr, extendFlag, extendPara, ahcLevel, ConcType::CONC_INTER);
}

HcclResult CommAHCBaseInfo::GetAlgTemplateOpInstance(
    const AHCOpType opType, std::unique_ptr<AlgTemplateBase>& tempAlg, const HcclDispatcher& dispatcher,
    const u64 reduceAttr, bool extendFlag, AHCExtendPreparePara extendPara, AHCLevel ahcLevel, ConcType concType)
{
    AHCConcOpType ahcConcOpType;
    ahcConcOpType.ahcLevel = ahcLevel;
    ahcConcOpType.concType = concType;
    ahcConcOpType.ahcOpType = opType;

    TemplateType algType = ahcAlgOption_[ahcConcOpType];

    HCCL_DEBUG(
        "[CommAHCBaseInfo][GetAlgTemplateOpInstance] Level[%u] ConcType[%u] choose algType[%u]", ahcLevel, concType,
        algType);

    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(algType, dispatcher);
    CHK_SMART_PTR_NULL(tempAlg);

    /*特殊属性传递*/
    // reduceAttr 传递
    if (opType == AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER || opType == AHCOpType::AHC_OP_TYPE_ALLREDUCE) {
        if (algType == TemplateType::TEMPLATE_REDUCESCATTER_NHR) {
            CHK_RET(tempAlg->Prepare(reduceAttr, false));
        } else {
            CHK_RET(tempAlg->Prepare(reduceAttr));
        }
    }

    // AHC 扩展属性传递
    if (extendFlag) {
        CHK_RET(tempAlg->Prepare(extendPara));
    }

    return HCCL_SUCCESS;
}

bool CommAHCBaseInfo::IsNeedInterProc(const u32 rank)
{
    (void)rank;
    return true;
}

CommBrokeAlignInfo::CommBrokeAlignInfo(const std::vector<std::vector<u32>>& subGroups) : CommAHCBaseInfo(subGroups) {}

CommBrokeAlignInfo::~CommBrokeAlignInfo() {}

HcclResult CommBrokeAlignInfo::Init(AHCOpType opType, std::map<AHCConcOpType, TemplateType>& ahcAlgOption)
{
    ahcAlgOption_ = ahcAlgOption;

    // 参数检查
    opType_ = opType;
    CHK_RET(CheckSubGroups(subGroups_));

    // 初始化broke 对齐的组间通信域相关信息
    for (u32 i = 0; i < subGroups_[minSubGroupIdx_].size(); ++i) {
        std::map<u32, u32> interRankOrder;
        std::vector<u32> logicGroup;
        for (u32 j = 0; j < subGroups_.size(); ++j) {
            logicGroup.push_back(subGroups_[j][i]);
            interRankOrder.insert(std::make_pair(j, j));
        }
        interRankList_.push_back(interRankOrder);
        logicCardCommGroups_.push_back(logicGroup);
    }

    // Reduce-Scatter 及 All-Gather 增加建链信息
    if (opType_ != AHCOpType::AHC_OP_TYPE_ALLREDUCE) {
        // 生成 broke中Reduce-scatter的执行顺序及通信关系分组
        for (u32 i = subGroups_[minSubGroupIdx_].size(); i < subGroups_[maxSubGroupIdx_].size(); ++i) {
            std::vector<u32> logicGroup;
            std::vector<u32> tmpCompleteGroupOrder;
            std::vector<u32> tmpEmptyGroupOrder;
            std::map<u32, u32> interRankOrder;
            u32 curCompleteIdx = 0;
            for (u32 j = 0; j < subGroups_.size(); ++j) {
                // 填充需要得到数据的分组信息
                if (subGroups_[j].size() > i) {
                    tmpCompleteGroupOrder.push_back(j);
                    interRankOrder.insert(std::make_pair(j, curCompleteIdx));
                    logicGroup.insert(logicGroup.begin() + curCompleteIdx, subGroups_[j][i % subGroups_[j].size()]);
                    curCompleteIdx++;
                } else {
                    tmpEmptyGroupOrder.push_back(j);
                    logicGroup.push_back(subGroups_[j][i % subGroups_[j].size()]);
                }
            }
            // 填充用空片参与运算的分组信息
            for (u32 j = 0; j < tmpEmptyGroupOrder.size(); ++j) {
                interRankOrder.insert(std::make_pair(tmpEmptyGroupOrder[j], curCompleteIdx));
                curCompleteIdx++;
            }
            interRankList_.push_back(interRankOrder);
            logicCardCommGroups_.push_back(logicGroup);
            completeGroupOrder_.insert(std::make_pair(i, tmpCompleteGroupOrder));
            emptyGroupOrder_.insert(std::make_pair(i, tmpEmptyGroupOrder));
        }
    }

    return HCCL_SUCCESS;
}

bool CommBrokeAlignInfo::IsNeedInterProc(const u32 rank)
{
    u32 intraRank = GetIntraRank(rank);
    HCCL_DEBUG(
        "[CommBrokeAlignInfo][IsNeedInterProc] rank[%u] intraRank[%u] minSize[%u]", rank, intraRank,
        subGroups_[minSubGroupIdx_].size());
    if (intraRank > (subGroups_[minSubGroupIdx_].size() - 1)) {
        return false;
    }
    return true;
}

// Reduce-Scatter 及 All-Gather 组内切片逻辑
HcclResult CommBrokeAlignInfo::CalcIntraSlicesAndLinks(
    const u32 rank, [[maybe_unused]] const u32 dataUnitSize, [[maybe_unused]] const u64 count,
    const std::vector<LINK>& links, std::vector<std::vector<LINK>>& intraLinksVector,
    std::vector<std::vector<Slice>>& intraSlicesVector)
{
    HCCL_DEBUG("[CommBrokeAlignInfo][CalcIntraSlicesAndLinks] begin calc intra slices and links rank[%u]", rank);

    u64 sliceSizeAligned = totalSize_ / rankSize_;
    u64 curoffset = 0;

    HCCL_DEBUG("[CommBrokeAlignInfo][CalcIntraSlicesAndLinks] calculate sliceSizeAligned[%llu]", sliceSizeAligned);

    for (u32 k = 0; k < subGroups_.size(); ++k) {
        // 满片分组处理过程
        for (u32 j = 0; j < subGroups_[k].size() / subGroups_[rankGroupMap_[rank]].size(); ++j) {
            std::vector<Slice> intraSlices;
            std::vector<LINK> intraLinks;
            for (u32 i = 0; i < subGroups_[rankGroupMap_[rank]].size(); ++i) {
                u32 curRank = subGroups_[rankGroupMap_[rank]][i];
                intraLinks.push_back(links[curRank]);
                Slice slice;
                slice.size = sliceSizeAligned;
                slice.offset = curoffset;
                curoffset = curoffset + slice.size;
                intraSlices.push_back(slice);
                HCCL_DEBUG(
                    "[CommBrokeAlignInfo][CalcIntraSlicesAndLinks] rank[%u], link[%u] slices[%u].offset=%llu, "
                    "slices[%u].size=%llu",
                    rank, curRank, i, slice.offset, i, slice.size);
            }
            intraLinksVector.push_back(intraLinks);
            intraSlicesVector.push_back(intraSlices);
        }
        std::vector<Slice> intraSlices;
        std::vector<LINK> intraLinks;
        // 涉及空片分组非零切片处理过程
        for (u32 i = 0; i < subGroups_[rankGroupMap_[rank]].size(); ++i) {
            u32 curRank = subGroups_[rankGroupMap_[rank]][i];
            intraLinks.push_back(links[curRank]);
            Slice slice;
            slice.size = i < subGroups_[k].size() % subGroups_[rankGroupMap_[rank]].size() ? sliceSizeAligned : 0;
            slice.offset = curoffset;
            curoffset = curoffset + slice.size;
            intraSlices.push_back(slice);
            HCCL_DEBUG(
                "[CommBrokeAlignInfo][CalcIntraSlicesAndLinks] rank[%u], link[%u] slices[%u].offset=%llu, "
                "slices[%u].size=%llu",
                rank, curRank, i, slice.offset, i, slice.size);
        }
        intraLinksVector.push_back(intraLinks);
        intraSlicesVector.push_back(intraSlices);
    }

    HCCL_DEBUG("[CommBrokeAlignInfo][CalcIntraSlicesAndLinks] end calc intra slices and links rank[%u]", rank);
    return HCCL_SUCCESS;
}

// All-Reduce 组内切片逻辑
HcclResult CommBrokeAlignInfo::CalcIntraSlicesAndLinks(
    const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
    std::vector<LINK>& intraLinks, std::vector<Slice>& intraSlices)
{
    // 计算组内每个rank结果上的offset和size
    HCCL_DEBUG("[CommBrokeAlignInfo][CalcIntraSlicesAndLinks] begin calc intra slices and links rank[%u]", rank);

    u64 sliceSizeCalculated = (count + (static_cast<u32>(subGroups_[minSubGroupIdx_].size()) - 1))
                              / subGroups_[minSubGroupIdx_].size() * dataUnitSize;
    u64 totalSize = count * dataUnitSize;
    u64 residueSize = totalSize;
    u64 sliceSizeAligned;
    const u64 sizeAlignedMinSize = 128 * 1024; // 优化小包性能，小于128k不切片
    if (sliceSizeCalculated > sizeAlignedMinSize) {
        sliceSizeAligned = AlgTemplateBase::RoundUpWithDivisor(sliceSizeCalculated, HCCL_MIN_SLICE_ALIGN);
    } else {
        sliceSizeAligned = AlgTemplateBase::RoundUpWithDivisor(sliceSizeCalculated, sizeAlignedMinSize);
    }

    for (u32 i = 0; i < subGroups_[rankGroupMap_[rank]].size(); ++i) {
        intraLinks.push_back(links[subGroups_[rankGroupMap_[rank]][i]]);
        Slice slice;
        if (i < subGroups_[minSubGroupIdx_].size()) {
            slice.size = (residueSize > sliceSizeAligned) ? sliceSizeAligned : residueSize;
            slice.offset = totalSize - residueSize;
            residueSize -= slice.size;
        } else {
            slice.size = 0;
            slice.offset = totalSize - residueSize;
        }
        HCCL_DEBUG(
            "[CommBrokeAlignInfo][CalcIntraSlicesAndLinks] rank[%u], slices[%u].offset=%llu, slices[%u].size=%llu",
            rank, i, slice.offset, i, slice.size);
        intraSlices.push_back(slice);
    }

    HCCL_DEBUG("[CommBrokeAlignInfo][CalcIntraSlicesAndLinks] end calc intra slices and links rank[%u]", rank);

    return HCCL_SUCCESS;
}

// 组间切片逻辑统一对外接口
HcclResult CommBrokeAlignInfo::CalcInterSlicesAndLinks(
    const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
    std::vector<std::vector<LINK>>& interLinksVector, std::vector<std::vector<Slice>>& interSlicesVector,
    std::vector<u32>& logicCardList)
{
    HcclResult ret = HCCL_SUCCESS;
    switch (opType_) {
        case AHCOpType::AHC_OP_TYPE_ALLREDUCE:
            ret = CalcInterSlicesAndLinksForAR(rank, dataUnitSize, count, links, interLinksVector, interSlicesVector);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CommBrokeAlignInfo][CalcInterSlicesAndLinks]rank[%u] count[%llu] failed in "
                    "CalcInterSlicesAndLinks step",
                    rank, count),
                ret);
            break;
        case AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER:
        case AHCOpType::AHC_OP_TYPE_ALLGATHER:
            ret = CalcInterSlicesAndLinksForRS(
                rank, dataUnitSize, count, links, interLinksVector, interSlicesVector, logicCardList);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CommBrokeAlignInfo][CalcInterSlicesAndLinks]rank[%u] count[%llu] failed in "
                    "CalcInterSlicesAndLinks step",
                    rank, count),
                ret);
            break;
        default:
            ret = HCCL_SUCCESS;
    }
    return ret;
}

HcclResult CommBrokeAlignInfo::PrepareIntraSlices(
    const u32 rank, const u32 dataUnitSize, const u64 count, std::vector<Slice>& intraSlices) const
{
    (void)dataUnitSize;
    (void)count;

    // 计算组内每个rank结果上的offset和size
    HCCL_DEBUG(
        "[CommBrokeAlignInfo][PrepareIntraSlices] begin calc intra slices and links rank[%u] ranksize[%u]", rank,
        rankSize_);

    u64 sliceSizeAligned = totalSize_ / rankSize_;
    u64 curoffset = 0;

    for (u32 i = 0; i < rankSize_; ++i) {
        Slice slice;
        slice.size = sliceSizeAligned;
        slice.offset = curoffset;
        curoffset = curoffset + slice.size;
        intraSlices.push_back(slice);
        HCCL_DEBUG(
            "[CommBrokeAlignInfo][PrepareIntraSlices] rank[%u], slices[%u].offset=%llu, slices[%u].size=%llu", rank, i,
            slice.offset, i, slice.size);
    }
    HCCL_DEBUG("[CommBrokeAlignInfo][PrepareIntraSlices] end calc intra slices and links rank[%u]", rank);
    return HCCL_SUCCESS;
}

HcclResult CommBrokeAlignInfo::CalcInterSlicesAndLinksForRS(
    const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
    std::vector<std::vector<LINK>>& interLinksVector, std::vector<std::vector<Slice>>& interSlicesVector,
    std::vector<u32>& logicCardList)
{
    std::vector<Slice> intraSlices;

    CHK_RET(PrepareIntraSlices(rank, dataUnitSize, count, intraSlices));
    HCCL_DEBUG("[CommBrokeAlignInfo][CalcInterSlicesAndLinksForRS] rank[%u] begin inter", rank);
    u32 intraRank = GetIntraRank(rank);
    u32 groupCountForRank = subGroups_[maxSubGroupIdx_].size() / subGroups_[rankGroupMap_[rank]].size();
    if (subGroups_[maxSubGroupIdx_].size() % subGroups_[rankGroupMap_[rank]].size() > intraRank) {
        groupCountForRank++;
    }

    for (u32 k = 0; k < groupCountForRank; ++k) {
        std::vector<Slice> interSlices;
        std::vector<LINK> interLinks;
        u32 curGroupIdx = intraRank + k * subGroups_[rankGroupMap_[rank]].size();
        if (curGroupIdx < subGroups_[minSubGroupIdx_].size()) { // 参与运算的所有 slice 都是有数据的
            logicCardList.push_back(rankGroupMap_[rank]);
            for (u32 i = 0; i < subGroups_.size(); i++) {
                Slice curSlice = intraSlices[groupOriginOffset_[i] + curGroupIdx];
                interLinks.push_back(links[subGroups_[i][intraRank]]);
                interSlices.push_back(curSlice);
                HCCL_DEBUG(
                    "[CommBrokeAlignInfo][CalcInterSlicesAndLinksForRS] rank[%u], link[%u], curIdx[%u], subGroup[%u], "
                    "groupIdx[%u], slices[%u].offset=%llu, slices[%u].size=%llu",
                    rank, subGroups_[i][intraRank], groupOriginOffset_[i] + curGroupIdx, i, curGroupIdx,
                    groupOriginOffset_[i], curSlice.offset, groupOriginOffset_[i], curSlice.size);
            }
        } else { // 部分空片参与运算
            Slice emptySlice;
            emptySlice.size = 0;
            emptySlice.offset = 0;
            for (u32 i = 0; i < subGroups_.size(); i++) {
                u32 curSubgroupsIdx = i < completeGroupOrder_[curGroupIdx].size() ?
                                          completeGroupOrder_[curGroupIdx][i] :
                                          emptyGroupOrder_[curGroupIdx][i - completeGroupOrder_[curGroupIdx].size()];
                if (curSubgroupsIdx == rankGroupMap_[rank]) {
                    logicCardList.push_back(i);
                }
                Slice curSlice = i < completeGroupOrder_[curGroupIdx].size() ?
                                     intraSlices[groupOriginOffset_[curSubgroupsIdx] + curGroupIdx] :
                                     emptySlice;
                interLinks.push_back(
                    links[subGroups_[curSubgroupsIdx][curGroupIdx % subGroups_[curSubgroupsIdx].size()]]);
                interSlices.push_back(curSlice);
                HCCL_DEBUG(
                    "[CommBrokeAlignInfo][CalcInterSlicesAndLinksForRS] rank[%u], link[%u], curIdx[%u], subGroup[%u], "
                    "groupIdx[%u], slices[%u].offset=%llu, slices[%u].size=%llu",
                    rank, subGroups_[curSubgroupsIdx][curGroupIdx % subGroups_[curSubgroupsIdx].size()],
                    groupOriginOffset_[curSubgroupsIdx] + curGroupIdx, curSubgroupsIdx, curGroupIdx,
                    groupOriginOffset_[curSubgroupsIdx], curSlice.offset, groupOriginOffset_[curSubgroupsIdx],
                    curSlice.size);
            }
        }
        interLinksVector.push_back(interLinks);
        interSlicesVector.push_back(interSlices);
    }

    HCCL_DEBUG("[CommBrokeAlignInfo][CalcInterSlicesAndLinksForRS] rank[%u] end inter", rank);
    return HCCL_SUCCESS;
}

HcclResult CommBrokeAlignInfo::CalcInterSlicesAndLinksForAR(
    const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
    std::vector<std::vector<LINK>>& interLinksVector, std::vector<std::vector<Slice>>& interSlicesVector)
{
    // 查找自己位于组内的第几个rank
    HCCL_DEBUG("[CommBrokeAlignInfo][CalcInterSlicesAndLinksForAR] begin calc inter slices and links rank[%u]", rank);

    u32 intraRank = GetIntraRank(rank);

    std::vector<Slice> intraSlices;
    std::vector<LINK> intraLinks;
    CHK_RET(CalcIntraSlicesAndLinks(rank, dataUnitSize, count, links, intraLinks, intraSlices));

    // 计算组间每个rank结果上的offset和size
    u64 sliceSizeCalculated = (intraSlices[intraRank].size / dataUnitSize + (static_cast<u32>(subGroups_.size()) - 1))
                              / subGroups_.size() * dataUnitSize;
    u64 totalSize = intraSlices[intraRank].size;
    u64 residueSize = totalSize;
    u64 sliceSizeAligned;
    const u64 sizeAlignedMinSize = 128 * 1024; // 优化小包性能，小于128k不切片
    if (sliceSizeCalculated > sizeAlignedMinSize) {
        sliceSizeAligned = AlgTemplateBase::RoundUpWithDivisor(sliceSizeCalculated, HCCL_MIN_SLICE_ALIGN);
    } else {
        sliceSizeAligned = AlgTemplateBase::RoundUpWithDivisor(sliceSizeCalculated, sizeAlignedMinSize);
    }

    std::vector<LINK> interLinks;
    std::vector<Slice> interSlices;
    for (u32 i = 0; i < subGroups_.size(); ++i) {
        interLinks.push_back(links[subGroups_[i][intraRank]]);
        Slice slice;
        slice.size = (residueSize > sliceSizeAligned) ? sliceSizeAligned : residueSize;
        slice.offset = intraSlices[intraRank].offset + totalSize - residueSize;
        residueSize -= slice.size;
        HCCL_DEBUG(
            "[CommBrokeAlignInfo][CalcInterSlicesAndLinksForAR] rank[%u], slices[%u].offset=%llu, slices[%u].size=%llu",
            rank, i, slice.offset, i, slice.size);
        interSlices.push_back(slice);
    }
    interLinksVector.push_back(interLinks);
    interSlicesVector.push_back(interSlices);

    HCCL_DEBUG("[CommBrokeAlignInfo][CalcInterSlicesAndLinksForAR] end calc inter slices and links rank[%u]", rank);
    return HCCL_SUCCESS;
}

CommAHCAlignInfo::CommAHCAlignInfo(const std::vector<std::vector<u32>>& subGroups) : CommAHCBaseInfo(subGroups) {}

CommAHCAlignInfo::~CommAHCAlignInfo() {}

HcclResult CommAHCAlignInfo::Init(AHCOpType opType, std::map<AHCConcOpType, TemplateType>& ahcAlgOption)
{
    ahcAlgOption_ = ahcAlgOption;

    // 参数检查
    opType_ = opType;
    CHK_RET(CheckSubGroups(subGroups_));

    // 初始化slice相关信息
    CHK_RET(InitSliceInfo());

    // 计算 logicCard 相关信息;
    InitLogicCardInfo();

    // 初始化相关Map信息
    CHK_RET(InitMapInfo());

    return HCCL_SUCCESS;
}

HcclResult CommAHCAlignInfo::InitSliceInfo()
{
    // 计算 totalSliceSegment_ ，即所有分组大小的最小公倍数， 以及 interRankOrder
    totalSliceSegment_ = subGroups_[0].size();
    // u32 groupSizeGcd;
    for (u32 i = 1; i < subGroups_.size(); ++i) {
        u32 groupSize = static_cast<u32>(subGroups_[i].size());
        totalSliceSegment_ = totalSliceSegment_ * groupSize / std::__gcd(totalSliceSegment_, groupSize);
    }
    globalTotalSliceSegment_ = rankSize_ * totalSliceSegment_;
    HCCL_DEBUG("[CommAHCAlignInfo][InitSliceInfo] totalSliceSegment [%u]", totalSliceSegment_);

    // 计算 logicCardSliceSize_ ;
    std::set<u32> sliceOffset;
    for (u32 i = 0; i < subGroups_.size(); ++i) {
        for (u32 j = 0; j < subGroups_[i].size(); ++j) {
            u32 rankSliceSize = (totalSliceSegment_ / subGroups_[i].size()) * (j + 1);
            sliceOffset.insert(rankSliceSize);
            HCCL_DEBUG("[CommAHCAlignInfo][InitSliceInfo] sliceOffset [%u]", rankSliceSize);
        }
    }
    sliceOffset.insert(static_cast<u32>(0));
    logicCardSliceOffset_.resize(sliceOffset.size());
    std::copy(sliceOffset.begin(), sliceOffset.end(), logicCardSliceOffset_.begin());

    std::vector<u32>::iterator itPre = logicCardSliceOffset_.begin();
    std::vector<u32>::iterator itNext = logicCardSliceOffset_.begin();
    itNext++;
    while (itNext != logicCardSliceOffset_.end()) {
        auto boundDiff = (*itNext) - (*itPre);
        logicCardSliceSize_.push_back(boundDiff);
        itPre++;
        itNext++;
    }

    CHK_PRT_RET(
        logicCardSliceSize_.size() != (logicCardSliceOffset_.size() - 1),
        HCCL_ERROR(
            "[CommAHCAlignInfo][InitSliceInfo] cardOffset size [%u] cardSize size [%u] check error",
            logicCardSliceSize_.size(), logicCardSliceOffset_.size()),
        HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

HcclResult CommAHCAlignInfo::InitLogicCardInfo()
{
    // 计算 logicCardCommGroups_;
    for (std::vector<u32>::iterator it = (logicCardSliceOffset_.begin() + 1); it != logicCardSliceOffset_.end(); ++it) {
        std::vector<u32> logicGroup;
        for (u32 i = 0; i < subGroups_.size(); ++i) {
            u32 logicRank;
            if ((*it) % (totalSliceSegment_ / subGroups_[i].size()) != 0) {
                logicRank = (*it) / (totalSliceSegment_ / subGroups_[i].size()) + 1;
            } else {
                logicRank = (*it) / (totalSliceSegment_ / subGroups_[i].size());
            }
            logicGroup.push_back(subGroups_[i][logicRank - 1]);
        }
        logicCardCommGroups_.push_back(logicGroup);
    }

    // 计算 logicCardGroup_
    u32 curRank = subGroups_[minSubGroupIdx_][0];
    u32 curOffset = 0;
    std::vector<u32>::iterator it = logicCardSliceOffset_.begin();
    u32 curIdx = 0;
    u32 curLogicIdx = 0;
    logicCardGroup_.resize(static_cast<u32>(subGroups_[minSubGroupIdx_].size()));
    for (u32 i = 0; i < logicCardCommGroups_.size(); ++i) {
        if (logicCardCommGroups_[i][minSubGroupIdx_] != curRank) {
            logicCardGroup_[curLogicIdx].resize(i - curIdx);
            for (u32 j = 0; j < i - curIdx; j++) {
                logicCardGroup_[curLogicIdx][j] = curIdx + j;
            }
            curIdx = i;
            curLogicIdx++;
            curRank = logicCardCommGroups_[i][minSubGroupIdx_];
            curOffset = *it;
        }
        logicCardExecuteOffset_.push_back(*it - curOffset);
        it++;
    }
    if (curIdx != logicCardCommGroups_.size() - 1) {
        logicCardGroup_[curLogicIdx].resize(logicCardCommGroups_.size() - curIdx);
        for (u32 i = 0; i < logicCardCommGroups_.size() - curIdx; i++) {
            logicCardGroup_[curLogicIdx][i] = curIdx + i;
        }
    }
    return HCCL_SUCCESS;
}

bool CommAHCAlignInfo::CompareLogicCardExcuteOrder(u32 i, u32 j)
{
    return logicCardExecuteOffset_[i] < logicCardExecuteOffset_[j];
}

HcclResult CommAHCAlignInfo::InitMapInfo()
{
    // rank 到 logicCardOrder  初始化
    std::map<u32, u32> interRankOrder;
    for (u32 i = 0; i < subGroups_.size(); ++i) {
        interRankOrder.insert(std::make_pair(i, i));
    }
    for (u32 i = 0; i < logicCardCommGroups_.size(); ++i) {
        interRankList_.push_back(interRankOrder);
        for (u32 j = 0; j < logicCardCommGroups_[i].size(); ++j) {
            rankLogicCardOrderMap_[logicCardCommGroups_[i][j]].push_back(i);
            rankLogicCardMap_[logicCardCommGroups_[i][j]].push_back(i);
        }
    }

    // 定义 lambda 将对象指针传递到成员函数
    auto sortLambda = [this](u32 i, u32 j) {
        return this->CompareLogicCardExcuteOrder(i, j);
    };

    // rankLogicCardOrderMap_ 内的逻辑同号卡list按照 logicCardExecuteOffset_ 并发流开始时间排序
    for (auto iter = rankLogicCardOrderMap_.begin(); iter != rankLogicCardOrderMap_.end(); iter++) {
        std::vector<u32>& rankLogicCardList = iter->second;
        std::sort(rankLogicCardList.begin(), rankLogicCardList.end(), sortLambda);
    }

    return HCCL_SUCCESS;
}

// 配置当前需要的 globalTotalSliceSegment_，用于 Multi-AllReduce 中
HcclResult CommAHCAlignInfo::SetGlobalTotalSliceSegment(u64 globalTotalSliceSegment)
{
    globalTotalSliceSegment_ = globalTotalSliceSegment;
    HCCL_DEBUG(
        "[CommAHCAlignInfo][setGlobalTotalSliceSegment] globalTotalSliceSegment set to [%llu]",
        globalTotalSliceSegment_);
    return HCCL_SUCCESS;
}

// 获取当前rank对应的多个逻辑同号卡,并且按照并发流的开始执行时间排序
HcclResult CommAHCAlignInfo::GetLogicCardExecuteOrder(u32 rank, std::vector<u32>& executeOrder)
{
    executeOrder = rankLogicCardOrderMap_[rank];
    return HCCL_SUCCESS;
}

HcclResult CommAHCAlignInfo::SliceSizeAlignBound(
    Slice& slice, u64 offsetCount, u64 sliceSizeCalculated, const u64 boundSize, u32 boundOffsetCount,
    u32& curOffset) const
{
    u64 sliceSize = offsetCount * sliceSizeCalculated;
    if (!isAlignBound_) {
        // 对于 All-Reduce 中的 Reduce-Scatter 以及 All-Gather，不需要严格对齐bound
        slice.size = slice.size + sliceSize;
        curOffset = curOffset + offsetCount;
        return HCCL_SUCCESS;
    }
    if (offsetCount < boundOffsetCount) {
        if (sliceSize <= ((curOffset / boundOffsetCount + 1) * boundSize - (slice.size + slice.offset))) {
            slice.size = slice.size + sliceSize;
        } else {
            slice.size = slice.size + ((curOffset / boundOffsetCount + 1) * boundSize - (slice.size + slice.offset));
        }
    } else {
        slice.size = slice.size + (offsetCount / boundOffsetCount) * boundSize;
    }
    curOffset = curOffset + offsetCount;
    return HCCL_SUCCESS;
}

// Reduce-Scatter 及 All-Gather 组内切片逻辑
HcclResult CommAHCAlignInfo::CalcIntraSlicesAndLinks(
    const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
    std::vector<std::vector<LINK>>& intraLinksVector, std::vector<std::vector<Slice>>& intraSlicesVector)
{
    // Boundary 指在 RS 及 AG 中单卡应有的数据量的 offset， 如八卡跑8K，boundary 为 1024
    HCCL_DEBUG("[CommAHCAlignInfo][CalcIntraSlicesAndLinks] begin calc intra slices and links rank[%u]", rank);

    u32 singleRankOffset = globalTotalSliceSegment_ / rankSize_; // 每个 rank 最后结果应有的小块数据份数
    u64 sliceSizeCalculated = (totalSize_ / dataUnitSize + globalTotalSliceSegment_ - 1) / globalTotalSliceSegment_
                              * dataUnitSize * (globalTotalSliceSegment_ / rankSize_ / totalSliceSegment_);
    u64 totalSize = totalSize_;
    u64 residueSize = totalSize;
    HCCL_DEBUG(
        "[CommAHCAlignInfo][AHCDEBUG] count[%u] ranksize[%u] sliceSizeCalculated[%u] totalSize[%u] "
        "globalTotalSliceSegment[%u] totalSliceSegment[%u] dataUnitSize[%u]",
        count, rankSize_, sliceSizeCalculated, totalSize, globalTotalSliceSegment_, totalSliceSegment_, dataUnitSize);

    u32 curOffset = 0;
    for (u32 k = 0; k < subGroups_.size(); ++k) {
        std::vector<Slice> intraSlices;
        std::vector<LINK> intraLinks;
        std::vector<u32> curLogicCardGroup
            = rankLogicCardMap_[subGroups_[rankGroupMap_[rank]][0]]; // 获取当前rank对应的逻辑同号组
        u32 singleSliceOffset = logicCardSliceSize_[curLogicCardGroup[0]];
        for (u32 j = 1; j < curLogicCardGroup.size(); ++j) {
            singleSliceOffset = singleSliceOffset + logicCardSliceSize_[curLogicCardGroup[j]];
        }
        for (u32 i = 0; i < subGroups_[rankGroupMap_[rank]].size(); ++i) {
            u32 curRank = subGroups_[rankGroupMap_[rank]][i];
            intraLinks.push_back(links[curRank]);
            Slice slice;
            slice.size = 0;
            slice.offset = totalSize - residueSize;
            u64 targeOffset = singleSliceOffset * subGroups_[k].size();
            u64 offsetCountBeforeBoundary
                = ((curOffset + singleRankOffset - 1) / singleRankOffset * singleRankOffset - curOffset) < targeOffset ?
                      ((curOffset + singleRankOffset - 1) / singleRankOffset * singleRankOffset - curOffset) :
                      targeOffset;
            SliceSizeAlignBound(
                slice, offsetCountBeforeBoundary, sliceSizeCalculated, totalSize_ / rankSize_, singleRankOffset,
                curOffset);
            u64 offsetCountCrossBoundary
                = (targeOffset - offsetCountBeforeBoundary) / singleRankOffset * singleRankOffset;
            SliceSizeAlignBound(
                slice, offsetCountCrossBoundary, sliceSizeCalculated, totalSize_ / rankSize_, singleRankOffset,
                curOffset);
            u64 offsetCountBehindBoundary
                = (targeOffset - offsetCountBeforeBoundary - offsetCountCrossBoundary) % singleRankOffset;
            SliceSizeAlignBound(
                slice, offsetCountBehindBoundary, sliceSizeCalculated, totalSize_ / rankSize_, singleRankOffset,
                curOffset);
            slice.size = (residueSize > slice.size) ? slice.size : residueSize;
            residueSize -= slice.size;
            intraSlices.push_back(slice);
            HCCL_DEBUG(
                "[CommAHCAlignInfo][CalcIntraSlicesAndLinks] rank[%u], singleSliceOffset[%u], "
                "subGroups_[%u].size()[%u], slices[%u].offset=%llu, slices[%u].size=%llu",
                rank, singleSliceOffset, k, subGroups_[k].size(), i, slice.offset, i, slice.size);
        }
        intraLinksVector.push_back(intraLinks);
        intraSlicesVector.push_back(intraSlices);
    }

    HCCL_DEBUG("[CommAHCAlignInfo][CalcIntraSlicesAndLinks] end calc intra slices and links rank[%u]", rank);
    return HCCL_SUCCESS;
}

// All-Reduce 组内切片逻辑
HcclResult CommAHCAlignInfo::CalcIntraSlicesAndLinks(
    const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
    std::vector<LINK>& intraLinks, std::vector<Slice>& intraSlices)
{
    // 计算组内每个rank结果上的offset和size
    HCCL_DEBUG("[CommAHCAlignInfo][CalcIntraSlicesAndLinks] begin calc intra slices and links rank[%u]", rank);

    u64 sliceSizeCalculated = (count + (totalSliceSegment_ * static_cast<u32>(subGroups_.size()) - 1))
                              / (totalSliceSegment_ * subGroups_.size()) * dataUnitSize;
    u64 totalSize = count * dataUnitSize;
    u64 residueSize = totalSize;
    u64 sliceSizeAligned = sliceSizeCalculated;
    const u64 sizeAlignedMinSize = 128 * 1024; // 优化小包性能，小于128k不切片
    if (sliceSizeCalculated > sizeAlignedMinSize) {
        sliceSizeAligned = AlgTemplateBase::RoundUpWithDivisor(sliceSizeCalculated, HCCL_MIN_SLICE_ALIGN);
    } else {
        sliceSizeAligned = AlgTemplateBase::RoundUpWithDivisor(sliceSizeCalculated, sizeAlignedMinSize);
    }
    sliceSizeAligned = sliceSizeAligned * static_cast<u32>(subGroups_.size())
                       * (totalSliceSegment_ / static_cast<u32>(subGroups_[rankGroupMap_[rank]].size()));

    for (u32 i = 0; i < subGroups_[rankGroupMap_[rank]].size(); ++i) {
        intraLinks.push_back(links[subGroups_[rankGroupMap_[rank]][i]]);
        Slice slice;
        slice.size = (residueSize > sliceSizeAligned) ? sliceSizeAligned : residueSize;
        slice.offset = totalSize - residueSize;
        residueSize -= slice.size;
        HCCL_DEBUG(
            "[CommAHCAlignInfo][CalcIntraSlicesAndLinks] rank[%u], slices[%u].offset=%llu, slices[%u].size=%llu", rank,
            i, slice.offset, i, slice.size);
        intraSlices.push_back(slice);
    }

    HCCL_DEBUG("[CommAHCAlignInfo][CalcIntraSlicesAndLinks] end calc intra slices and links rank[%u]", rank);
    return HCCL_SUCCESS;
}

// 组间切片逻辑统一对外接口
HcclResult CommAHCAlignInfo::CalcInterSlicesAndLinks(
    const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
    std::vector<std::vector<LINK>>& interLinksVector, std::vector<std::vector<Slice>>& interSlicesVector,
    std::vector<u32>& logicCardList)
{
    HcclResult ret = HCCL_SUCCESS;
    switch (opType_) {
        case AHCOpType::AHC_OP_TYPE_ALLREDUCE:
            ret = CalcInterSlicesAndLinksForAR(
                rank, dataUnitSize, count, links, interLinksVector, interSlicesVector, logicCardList);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CommAHCAlignInfo][CalcInterSlicesAndLinks]rank[%u] count[%llu] failed in CalcInterSlicesAndLinks "
                    "step",
                    rank, count),
                ret);
            break;
        case AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER:
        case AHCOpType::AHC_OP_TYPE_ALLGATHER:
            ret = CalcInterSlicesAndLinksForRS(
                rank, dataUnitSize, count, links, interLinksVector, interSlicesVector, logicCardList);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[CommAHCAlignInfo][CalcInterSlicesAndLinks]rank[%u] count[%llu] failed in CalcInterSlicesAndLinks "
                    "step",
                    rank, count),
                ret);
            break;
        default:
            ret = HCCL_SUCCESS;
    }
    return ret;
}

HcclResult CommAHCAlignInfo::PrepareIntraSlices(
    const u32 rank, const u32 dataUnitSize, const u64 count, std::vector<std::vector<Slice>>& intraSlicesVector)
{
    // 计算组内每个rank结果上的offset和size
    HCCL_DEBUG("[CommAHCAlignInfo][PrepareIntraSlices] begin calc intra slices and links rank[%u]", rank);

    u32 singleRankOffset = globalTotalSliceSegment_ / rankSize_;
    u64 sliceSizeCalculated = (totalSize_ / dataUnitSize + globalTotalSliceSegment_ - 1) / globalTotalSliceSegment_
                              * dataUnitSize * (globalTotalSliceSegment_ / rankSize_ / totalSliceSegment_);
    u64 totalSize = totalSize_;
    u64 residueSize = totalSize;
    HCCL_DEBUG(
        "[CommAHCAlignInfo][AHCDEBUG] count[%u] ranksize[%u] sliceSizeCalculated[%u] totalSize[%u]", count, rankSize_,
        sliceSizeCalculated, totalSize);

    for (u32 i = 0; i < logicCardCommGroups_.size(); ++i) {
        std::vector<Slice> intraSlices;
        intraSlicesVector.push_back(intraSlices);
    }

    u32 curOffset = 0;
    for (u32 i = 0; i < subGroups_.size(); i++) {
        for (u32 j = 0; j < logicCardCommGroups_.size(); ++j) {
            Slice slice;
            slice.size = 0;
            slice.offset = totalSize - residueSize;
            u64 targeOffset = logicCardSliceSize_[j] * subGroups_[i].size();
            u64 offsetCountBeforeBoundary
                = ((curOffset + singleRankOffset - 1) / singleRankOffset * singleRankOffset - curOffset) < targeOffset ?
                      ((curOffset + singleRankOffset - 1) / singleRankOffset * singleRankOffset - curOffset) :
                      targeOffset;
            SliceSizeAlignBound(
                slice, offsetCountBeforeBoundary, sliceSizeCalculated, totalSize_ / rankSize_, singleRankOffset,
                curOffset);
            u64 offsetCountCrossBoundary
                = (targeOffset - offsetCountBeforeBoundary) / singleRankOffset * singleRankOffset;
            SliceSizeAlignBound(
                slice, offsetCountCrossBoundary, sliceSizeCalculated, totalSize_ / rankSize_, singleRankOffset,
                curOffset);
            u64 offsetCountBehindBoundary
                = (targeOffset - offsetCountBeforeBoundary - offsetCountCrossBoundary) % singleRankOffset;
            SliceSizeAlignBound(
                slice, offsetCountBehindBoundary, sliceSizeCalculated, totalSize_ / rankSize_, singleRankOffset,
                curOffset);
            slice.size = (residueSize > slice.size) ? slice.size : residueSize;
            residueSize -= slice.size;
            intraSlicesVector[j].push_back(slice);
            HCCL_DEBUG(
                "[CommAHCAlignInfo][PrepareIntraSlices] rank[%u], round[%u], logicGroup[%u], slices[%u].offset=%llu, "
                "slices[%u].size=%llu",
                rank, i, j, j, slice.offset, j, slice.size);
        }
    }
    HCCL_DEBUG("[CommAHCAlignInfo][PrepareIntraSlices] end calc intra slices and links rank[%u]", rank);
    return HCCL_SUCCESS;
}

HcclResult CommAHCAlignInfo::CalcInterSlicesAndLinksForRS(
    const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
    std::vector<std::vector<LINK>>& interLinksVector, std::vector<std::vector<Slice>>& interSlicesVector,
    std::vector<u32>& logicCardList)
{
    HCCL_DEBUG("[CommAHCAlignInfo][CalcInterSlicesAndLinksForRS] begin calc inter slices and links rank[%u]", rank);
    std::vector<std::vector<Slice>> intraSlicesVecotr;

    CHK_RET(PrepareIntraSlices(rank, dataUnitSize, count, intraSlicesVecotr));
    GetLogicCardExecuteOrder(rank, logicCardList);

    for (u32 i = 0; i < logicCardList.size(); i++) {
        u32 logicGroupIdx = logicCardList[i]; // 获取当前处理的目标逻辑同号卡组的下标
        std::vector<Slice> curIntraSliceVector = intraSlicesVecotr[logicGroupIdx];
        std::vector<Slice> interSlices;
        std::vector<LINK> interLinks;
        for (u32 j = 0; j < subGroups_.size(); j++) {
            Slice curSlice = curIntraSliceVector[j];
            interLinks.push_back(links[logicCardCommGroups_[logicGroupIdx][j]]); // 当前处理的逻辑同号卡
            interSlices.push_back(curSlice);
            HCCL_DEBUG(
                "[CommAHCAlignInfo][CalcInterSlicesAndLinksForRS] rank[%u], link[%u], logicGroup[%u], "
                "slices[%u].offset=%llu, slices[%u].size=%llu",
                rank, logicCardCommGroups_[logicGroupIdx][j], logicGroupIdx, i, curSlice.offset, i, curSlice.size);
        }
        interSlicesVector.push_back(interSlices);
        interLinksVector.push_back(interLinks);
    }
    HCCL_DEBUG("[CommAHCAlignInfo][CalcInterSlicesAndLinksForRS] calc inter slices and links rank[%u] end", rank);
    return HCCL_SUCCESS;
}

HcclResult CommAHCAlignInfo::PrepareWholeLogicSlices(
    const Slice& intraSlice, const u64 sliceSizeAligned, const u32 originOffset, std::vector<Slice>& logicGroupSlice,
    std::vector<u32>& logicCardList)
{
    for (u32 i = 0; i < logicCardList.size(); i++) {
        Slice logicSlice;
        u32 logicRank = logicCardList[i];
        // 计算当前逻辑同号组的offset大小
        u32 offsetDiff = logicCardSliceOffset_[logicRank + 1] - logicCardSliceOffset_[logicRank];
        HCCL_DEBUG(
            "[CommAHCAlignInfo][CalcInterSlicesAndLinks] logicGroupSlice begin, logicRank : [%u],"
            "offsetDiff : [%u], offset_next : [%u], offset_cur[%u]",
            logicRank, offsetDiff, logicCardSliceOffset_[logicRank + 1], logicCardSliceOffset_[logicRank]);

        logicSlice.size = sliceSizeAligned * offsetDiff;
        logicSlice.offset = intraSlice.offset + sliceSizeAligned * (logicCardSliceOffset_[logicRank] - originOffset);
        HCCL_DEBUG(
            "[CommAHCAlignInfo][PrepareFullLogicSlices] logicGroupSlice end, logicRank : [%u] ,"
            "size : [%u], offset : [%u] ",
            logicRank, logicSlice.size, logicSlice.offset);
        logicGroupSlice.push_back(logicSlice);
    }
    return HCCL_SUCCESS;
}

HcclResult CommAHCAlignInfo::PreparePartialLogicSlices(
    const Slice& intraSlice, const u64 sliceSizeAligned, const u32 originOffset, std::vector<Slice>& logicGroupSlice,
    std::vector<u32>& logicCardList)
{
    for (u32 i = 0; i < logicCardList.size(); i++) {
        Slice logicSlice;
        u32 logicRank = logicCardList[i];
        // 计算当前逻辑同号组的offset大小
        u32 offsetDiff = logicCardSliceOffset_[logicRank + 1] - logicCardSliceOffset_[logicRank];
        HCCL_DEBUG(
            "[CommAHCAlignInfo][CalcInterSlicesAndLinks] logicGroupSlice begin, logicRank : [%u],"
            "offsetDiff : [%u], offset_next : [%u], offset_cur[%u]",
            logicRank, offsetDiff, logicCardSliceOffset_[logicRank + 1], logicCardSliceOffset_[logicRank]);

        // 当前rank在组内对应的offset能获取到完全的数据，即前几个逻辑同号卡
        if ((logicCardSliceOffset_[logicRank + 1] - originOffset) <= intraSlice.size / sliceSizeAligned) {
            logicSlice.size = sliceSizeAligned * offsetDiff;
            logicSlice.offset
                = intraSlice.offset + sliceSizeAligned * (logicCardSliceOffset_[logicRank] - originOffset);
            // 当前rank在组内对应的offset能获取到部分的数据，即边界上的逻辑同号卡
        } else if ((logicCardSliceOffset_[logicRank] - originOffset) <= intraSlice.size / sliceSizeAligned) {
            logicSlice.size = intraSlice.size - (logicCardSliceOffset_[logicRank] - originOffset) * sliceSizeAligned;
            logicSlice.offset
                = intraSlice.offset + sliceSizeAligned * (logicCardSliceOffset_[logicRank] - originOffset);
            // 当前rank在组内对应的offset不能获取到数据，即最后的逻辑同号卡
        } else {
            logicSlice.size = 0;
            logicSlice.offset = 0;
        }
        HCCL_DEBUG(
            "[CommAHCAlignInfo][CalcInterSlicesAndLinks] logicGroupSlice end, logicRank : [%u] ,"
            "size : [%u], offset : [%u] ",
            logicRank, logicSlice.size, logicSlice.offset);

        logicGroupSlice.push_back(logicSlice);
    }
    return HCCL_SUCCESS;
}

HcclResult CommAHCAlignInfo::PrepareEmptyLogicSlices(
    std::vector<Slice>& logicGroupSlice, const std::vector<u32>& logicCardList) const
{
    for (u32 i = 0; i < logicCardList.size(); i++) {
        Slice logicSlice;
        logicSlice.size = 0;
        logicSlice.offset = 0;
        logicGroupSlice.push_back(logicSlice);
    }
    return HCCL_SUCCESS;
}

HcclResult CommAHCAlignInfo::CalcLogicSlicesAndLinks(
    std::vector<Slice>& logicGroupSlice, std::vector<u32>& logicCardList, const std::vector<LINK>& links,
    std::vector<std::vector<LINK>>& interLinksVector, std::vector<std::vector<Slice>>& interSlicesVector)
{
    for (u32 i = 0; i < logicGroupSlice.size(); i++) {
        Slice logicSlice = logicGroupSlice[i];
        std::vector<Slice> interSlices;
        std::vector<LINK> interLinks;
        u32 logicRank = logicCardList[i];
        u64 totalSize = logicSlice.size;
        u64 residueSize = totalSize;
        u64 logicSliceSizeAligned;
        if (logicSlice.size % subGroups_.size() == 0 && logicSlice.size % HCCL_MIN_SLICE_ALIGN == 0) {
            logicSliceSizeAligned = logicSlice.size / subGroups_.size();
        } else {
            u64 sliceSizeCalculated
                = (logicSlice.size + static_cast<u32>(subGroups_.size()) - 1) / static_cast<u32>(subGroups_.size());
            logicSliceSizeAligned = AlgTemplateBase::RoundUpWithDivisor(sliceSizeCalculated, HCCL_MIN_SLICE_ALIGN);
        }
        for (u32 j = 0; j < subGroups_.size(); j++) {
            u32 curRank = logicCardCommGroups_[logicRank][j];
            Slice slice;
            interLinks.push_back(links[curRank]);
            slice.size = (residueSize > logicSliceSizeAligned) ? logicSliceSizeAligned : residueSize;
            slice.offset = logicSlice.offset + totalSize - residueSize;
            residueSize -= slice.size;
            interSlices.push_back(slice);
        }
        interLinksVector.push_back(interLinks);
        interSlicesVector.push_back(interSlices);
    }
    return HCCL_SUCCESS;
}

HcclResult CommAHCAlignInfo::CalcInterSlicesAndLinksForAR(
    const u32 rank, const u32 dataUnitSize, const u64 count, const std::vector<LINK>& links,
    std::vector<std::vector<LINK>>& interLinksVector, std::vector<std::vector<Slice>>& interSlicesVector,
    std::vector<u32>& logicCardList)
{
    HCCL_DEBUG("[CommAHCAlignInfo][CalcInterSlicesAndLinksForAR] begin calc inter slices and links rank[%u]", rank);

    u32 intraRank = GetIntraRank(rank);

    std::vector<Slice> intraSlices;
    std::vector<LINK> intraLinks;

    CHK_RET(CalcIntraSlicesAndLinks(rank, dataUnitSize, count, links, intraLinks, intraSlices));
    GetLogicCardExecuteOrder(rank, logicCardList);

    // 计算当前rank逻辑同号卡之间最小slice的大小
    u64 sliceSizeCalculated = (count + (totalSliceSegment_ * static_cast<u32>(subGroups_.size()) - 1))
                              / (totalSliceSegment_ * subGroups_.size()) * dataUnitSize;
    const u64 sizeAlignedMinSize = 128 * 1024; // 优化小包性能，小于128k不切片
    u64 sliceSizeAligned = sliceSizeCalculated;
    if (sliceSizeCalculated > sizeAlignedMinSize) {
        sliceSizeAligned = AlgTemplateBase::RoundUpWithDivisor(sliceSizeCalculated, HCCL_MIN_SLICE_ALIGN);
    } else {
        sliceSizeAligned = AlgTemplateBase::RoundUpWithDivisor(sliceSizeCalculated, sizeAlignedMinSize);
    }

    sliceSizeAligned = sliceSizeAligned * static_cast<u32>(subGroups_.size());
    u32 originOffset = intraRank * totalSliceSegment_ / subGroups_[rankGroupMap_[rank]].size(); // 当前rank起始offset

    HCCL_DEBUG(
        "[CommAHCAlignInfo][CalcInterSlicesAndLinksForAR] rank : [%u], intraslice.size : [%u], intraslice.offset : "
        "[%u],"
        "sliceSizeAligned : [%u], originOffset : [%u]",
        rank, intraSlices[intraRank].size, intraSlices[intraRank].offset, sliceSizeAligned, originOffset);

    // 进行逻辑同号组对应的slice切分
    std::vector<Slice> logicGroupSlice; // 逻辑同号组对应的slice
    HCCL_DEBUG(
        "[CommAHCAlignInfo][CalcInterSlicesAndLinksForAR] check rank : [%u], intraSlices[%u].size : [%u], "
        "sliceSizeAligned : [%u],"
        "totalSliceSegment_ : [%u], subGroups_[rankGroupMap_[rank]].size : [%u]",
        rank, intraRank, intraSlices[intraRank].size, sliceSizeAligned, totalSliceSegment_,
        subGroups_[rankGroupMap_[rank]].size());

    // 当前rank有完整的对齐后的数据量
    if (intraSlices[intraRank].size / sliceSizeAligned == totalSliceSegment_ / subGroups_[rankGroupMap_[rank]].size()) {
        CHK_RET(PrepareWholeLogicSlices(
            intraSlices[intraRank], sliceSizeAligned, originOffset, logicGroupSlice, logicCardList));
        // 当前rank有不完整的数据量
    } else if (intraSlices[intraRank].size != 0) {
        CHK_RET(PreparePartialLogicSlices(
            intraSlices[intraRank], sliceSizeAligned, originOffset, logicGroupSlice, logicCardList));
    } else {
        CHK_RET(PrepareEmptyLogicSlices(logicGroupSlice, logicCardList));
    }

    // 计算当前rank逻辑同号组之间的slice大小
    CHK_RET(CalcLogicSlicesAndLinks(logicGroupSlice, logicCardList, links, interLinksVector, interSlicesVector));

    HCCL_DEBUG("[CommAHCAlignInfo][CalcInterSlicesAndLinks] end calc inter slices and links rank[%u]", rank);
    return HCCL_SUCCESS;
}

} // namespace hccl
