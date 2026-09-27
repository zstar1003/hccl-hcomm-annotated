/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_BROADCAST_PARALLEL_APICPU_EXECUTOR_H
#define HCCLV2_INS_BROADCAST_PARALLEL_APICPU_EXECUTOR_H

#include "ins_coll_alg_base.h"

namespace Hccl {

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
class InsBroadcastParallelAiCpuExecutor : public InsCollAlgBase {
public:
    explicit InsBroadcastParallelAiCpuExecutor() = default;
    ~InsBroadcastParallelAiCpuExecutor() override = default;

    std::string Describe() const override { return "Instruction based BroadCast Parallel AICPU Executor."; }

    HcclResult CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq) override;

    HcclResult CalcResOffload(const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq) override;
    // HOST 接口
    HcclResult Orchestrate(
        const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue) override;
    // AICPU 接口
    HcclResult Orchestrate(
        const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
        InsQuePtr insQue) override;

private:
    struct ScratchMultiple {
        u32 interScatter;
        u32 intraScatter;
        u32 interAllGather;
        u32 intraAllGather;
        float maxMultiple;
    };
    struct SliceConfig {
        u32 loopTimes = 0;
        u64 sliceCount = 0;           // 正常切分个数
        u64 sliceCountPart0 = 0;      // 正常切块第一部分个数
        u64 sliceCountPart1 = 0;      // 正常切块第二部分个数
        u64 finalSliceCount = 0;      // 尾块切分个数
        u64 finalSliceCountPart0 = 0; // 尾块第一部分切分个数
        u64 finalSliceCountPart1 = 0; // 尾块第二部分切分个数
        u64 finalTailCountPart0 = 0;  // 尾块非卡整数倍尾巴
        u64 finalTailCountPart1 = 0;  // 尾块非卡整数倍尾巴
    };
    struct ScratchOffset {
        u64 interScatterStage0;
        u64 intraScatterStage0;
        u64 intraScatterStage1;
        u64 interScatterStage1;
        u64 intraAllGatherStage2;
        u64 interAllGatherStage2;
        u64 interAllGatherStage3;
        u64 intraAllGatherStage3;
    };
    struct DataParameters {
        u64 dataOffset[2] = {0, 0};                     // 每个part数据偏移
        std::vector<std::vector<u64>> sliceSize{2};     // 正常分块part每个阶段数据大小
        std::vector<std::vector<u64>> inputStride{2};   // 正常分块partInputStride大小
        std::vector<std::vector<u64>> scratchOffset{2}; // 每个分块scratchoffset
        std::vector<std::vector<u64>> tailSize{2};      // 尾片的整数倍数据大小
    };

    struct StageProcAlgPara {
        std::function<HcclResult(TempFuncs&, TemplateDataParams&, ResLinks&, std::vector<InsQuePtr>&)> part0FuncPtr;
        ResLinks part0links;
        std::vector<InsQuePtr> part0Que;
        std::function<HcclResult(TempFuncs&, TemplateDataParams&, ResLinks&, std::vector<InsQuePtr>&)> part1FuncPtr;
        ResLinks part1links;
        std::vector<InsQuePtr> part1Que;
    };
    HcclResult WrapPrepResLinks(const RankGraph* type, const LinkReq& linkReq, ResLinks& resLinks)
    {
        return PrepResLinks(myRank_, type, linkPriority_, linkReq, resLinks);
    }
    HcclResult WrapPrepResLinks(ConnectedLinkMgr* type, const LinkReq& linkReq, ResLinks& resLinks) const
    {
        return PrepResLinks(myRank_, linkReq, type, resLinks);
    }
    HcclResult PreCalcRes(
        const RankGraph* rankGraph, AlgTempResReq& resReqIntraScatter, AlgTempResReq& resReqInterScatter,
        AlgTempResReq& resReqIntraAllGather, AlgTempResReq& resReqInterAllGather);
    template <typename T>
    HcclResult CalcSingleAlgRes(
        InsAlgTemplate0& intraScatter, InsAlgTemplate1& interScatter, InsAlgTemplate2& intraAllGather,
        InsAlgTemplate3& interAllGather, T* type, AlgTempResReq& resReqIntraScatter, AlgTempResReq& resReqInterScatter,
        AlgTempResReq& resReqIntraAllGather, AlgTempResReq& resReqInterAllGather) const;

    template <typename T>
    HcclResult PrepareRes(
        T* type, AlgTempResReq& resReqIntraScatter, AlgTempResReq& resReqInterScatter,
        AlgTempResReq& resReqIntraAllGather, AlgTempResReq& resReqInterAllGather);

    HcclResult CalcLocalRankSize()
    {
        uint64_t virtRanks_2 = 2;
        CHK_PRT_RET(
            virtRanks_.size() < virtRanks_2, HCCL_ERROR("[CalcLocalRankSize] virtRanks level num is smaller than 2."),
            HcclResult::HCCL_E_INTERNAL);

        intraLocalRankSize_ = virtRanks_.at(0).size();
        interLocalRankSize_ = virtRanks_.at(1).size();

        HCCL_INFO(
            "[CalcLocalRankSize] localRankSize: myRank[%d] intraLocalRankSize[%u] interLocalRankSize[%u]", myRank_,
            intraLocalRankSize_, interLocalRankSize_);
        return HcclResult::HCCL_SUCCESS;
    };
    void GetParallelDataSplit(std::vector<double>& splitDataSize) const
    {
        // to do 先做等分，后续根据性能做调整
        double splitData = 0.5;
        splitDataSize.push_back(splitData);
        splitDataSize.push_back(splitData);
        return;
    }
    void InitDataParameters(SliceConfig& slice, ScratchMultiple& scratchMultiple, DataParameters& dataParameters) const
    {
        dataParameters.sliceSize.at(0)
            = {slice.sliceCountPart0 * dataTypeSize_ / interLocalRankSize_,
               slice.sliceCountPart0 * dataTypeSize_ / interLocalRankSize_ / intraLocalRankSize_,
               slice.sliceCountPart0 * dataTypeSize_ / interLocalRankSize_ / intraLocalRankSize_,
               slice.sliceCountPart0 * dataTypeSize_ / interLocalRankSize_};
        dataParameters.sliceSize.at(1)
            = {slice.sliceCountPart1 * dataTypeSize_ / intraLocalRankSize_,
               slice.sliceCountPart1 * dataTypeSize_ / intraLocalRankSize_ / interLocalRankSize_,
               slice.sliceCountPart1 * dataTypeSize_ / intraLocalRankSize_ / interLocalRankSize_,
               slice.sliceCountPart1 * dataTypeSize_ / intraLocalRankSize_};
        dataParameters.inputStride.at(0)
            = {slice.sliceCountPart0 * dataTypeSize_ / interLocalRankSize_,
               slice.sliceCountPart0 * dataTypeSize_ / interLocalRankSize_ / intraLocalRankSize_, 0, 0};
        dataParameters.inputStride.at(1)
            = {slice.sliceCountPart1 * dataTypeSize_ / intraLocalRankSize_,
               slice.sliceCountPart1 * dataTypeSize_ / intraLocalRankSize_ / interLocalRankSize_, 0, 0};
        // 计算Scratch偏移，数据尾块必然小于常规块，不用额外计算尾块时的Scratch偏移
        dataParameters.scratchOffset.at(0) = {0, 0, 0, 0};
        dataParameters.scratchOffset.at(1)
            = {slice.sliceCountPart0 * scratchMultiple.interScatter * dataTypeSize_,
               (slice.sliceCountPart0 / interLocalRankSize_) * scratchMultiple.intraScatter * dataTypeSize_,
               (slice.sliceCountPart0 / interLocalRankSize_ / intraLocalRankSize_) * scratchMultiple.intraAllGather
                   * dataTypeSize_,
               (slice.sliceCountPart0 / interLocalRankSize_) * scratchMultiple.interAllGather * dataTypeSize_};
        dataParameters.tailSize = dataParameters.sliceSize;
        return;
    }
    void InitFinalSliceDataParameters(
        SliceConfig& slice, ScratchMultiple& scratchMultiple, DataParameters& dataParameters) const
    {
        dataParameters.sliceSize.at(0)
            = {slice.finalSliceCountPart0 * dataTypeSize_ / interLocalRankSize_,
               slice.finalSliceCountPart0 * dataTypeSize_ / interLocalRankSize_ / intraLocalRankSize_,
               slice.finalSliceCountPart0 * dataTypeSize_ / interLocalRankSize_ / intraLocalRankSize_,
               slice.finalSliceCountPart0 * dataTypeSize_ / interLocalRankSize_};
        dataParameters.sliceSize.at(1)
            = {slice.finalSliceCountPart1 * dataTypeSize_ / intraLocalRankSize_,
               slice.finalSliceCountPart1 * dataTypeSize_ / intraLocalRankSize_ / interLocalRankSize_,
               slice.finalSliceCountPart1 * dataTypeSize_ / intraLocalRankSize_ / interLocalRankSize_,
               slice.finalSliceCountPart1 * dataTypeSize_ / intraLocalRankSize_};
        dataParameters.inputStride.at(0)
            = {slice.finalSliceCountPart0 * dataTypeSize_ / interLocalRankSize_,
               slice.finalSliceCountPart0 * dataTypeSize_ / interLocalRankSize_ / intraLocalRankSize_, 0, 0};
        dataParameters.inputStride.at(1)
            = {slice.finalSliceCountPart1 * dataTypeSize_ / intraLocalRankSize_,
               slice.finalSliceCountPart1 * dataTypeSize_ / intraLocalRankSize_ / interLocalRankSize_, 0, 0};
        dataParameters.scratchOffset.at(0) = {0, 0, 0, 0};
        dataParameters.scratchOffset.at(1)
            = {slice.finalSliceCountPart0 * scratchMultiple.interScatter * dataTypeSize_,
               (slice.finalSliceCountPart0 / interLocalRankSize_) * scratchMultiple.intraScatter * dataTypeSize_,
               (slice.finalSliceCountPart0 / interLocalRankSize_ / intraLocalRankSize_) * scratchMultiple.intraAllGather
                   * dataTypeSize_,
               (slice.finalSliceCountPart0 / interLocalRankSize_) * scratchMultiple.interAllGather * dataTypeSize_};
        // 只有最后一片数据的part1部分存在尾片数据，scatter算子和allgather算子都需要支持该数据收集
        for (size_t i = 0; i < dataParameters.sliceSize.at(0).size(); i++) {
            dataParameters.tailSize.at(0).at(i)
                = dataParameters.sliceSize.at(0).at(i) + slice.finalTailCountPart0 * dataTypeSize_;
        }
        for (size_t i = 0; i < dataParameters.sliceSize.at(1).size(); i++) {
            dataParameters.tailSize.at(1).at(i)
                = dataParameters.sliceSize.at(1).at(i) + slice.finalTailCountPart1 * dataTypeSize_;
        }
        return;
    }
    HcclResult CalcLocalRoot()
    {
        CHK_PRT_RET(
            root_ >= rankSize_, HCCL_ERROR("[CalcLocalRoot] root[%u] is out of rankSize[%u]", root_, rankSize_),
            HcclResult::HCCL_E_INTERNAL);

        u32 intraLocalRootIdx = root_ % intraLocalRankSize_;
        intraLocalRoot_ = static_cast<u32>(vTopo_.at(0).at(0).at(intraLocalRootIdx));
        u32 interLocalRootIdx = root_ / intraLocalRankSize_;
        interLocalRoot_ = static_cast<u32>(vTopo_.at(1).at(0).at(interLocalRootIdx));

        HCCL_INFO(
            "[CalcLocalRoot] localRoot: myRank[%d] intraLocalRoot[%u] interLocalRoot[%u]", myRank_, intraLocalRoot_,
            interLocalRoot_);
        return HcclResult::HCCL_SUCCESS;
    }

    void CalcScratchMultiple(
        std::vector<double>& splitDataSize, ScratchMultiple& scratchMultiple, InsAlgTemplate0& intraScatterTempAlg,
        InsAlgTemplate1& interScatterTempAlg, InsAlgTemplate2& intraAllGatherTempAlg,
        InsAlgTemplate3& interAllGatherTempAlg) const
    {
        scratchMultiple.intraScatter = intraScatterTempAlg.CalcScratchMultiple(BufferType::INPUT, BufferType::INPUT);
        scratchMultiple.interScatter = interScatterTempAlg.CalcScratchMultiple(BufferType::INPUT, BufferType::INPUT);
        scratchMultiple.intraAllGather
            = intraAllGatherTempAlg.CalcScratchMultiple(BufferType::INPUT, BufferType::INPUT);
        scratchMultiple.interAllGather
            = interAllGatherTempAlg.CalcScratchMultiple(BufferType::INPUT, BufferType::INPUT);
        // 计算第一步需要的倍数和最后一步所需要的数据缓存倍数，取multiple最大需求
        float multiple0 = splitDataSize.at(0) * float(scratchMultiple.interScatter)
                          + splitDataSize.at(1) * float(scratchMultiple.intraScatter);
        float multiple1 = splitDataSize.at(0) * float(scratchMultiple.interAllGather / interLocalRankSize_)
                          + splitDataSize.at(1) * float(scratchMultiple.intraAllGather / intraLocalRankSize_);
        scratchMultiple.maxMultiple = std::max(multiple0, multiple1);
        return;
    }
    void CalcSlice(std::vector<double>& splitDataSize, float scratchMaxMultiple, SliceConfig& slice);
    void LogAlgInfo(
        InsAlgTemplate0& intraScatterTempAlg, InsAlgTemplate1& interScatterTempAlg,
        InsAlgTemplate2& intraAllGatherTempAlg, InsAlgTemplate3& interAllGatherTempAlg) const
    {
        HCCL_INFO("[InsBroadcastParallelAiCpuExecutor] Alg0 is [%s]", intraScatterTempAlg.Describe().c_str());
        HCCL_INFO("[InsBroadcastParallelAiCpuExecutor] Alg1 is [%s]", interScatterTempAlg.Describe().c_str());
        HCCL_INFO("[InsBroadcastParallelAiCpuExecutor] Alg2 is [%s]", intraAllGatherTempAlg.Describe().c_str());
        HCCL_INFO("[InsBroadcastParallelAiCpuExecutor] Alg3 is [%s]", interAllGatherTempAlg.Describe().c_str());
        return;
    }
    HcclResult StageProcess(DataParameters& dataParameters, std::vector<StageProcAlgPara>& algParaVec);
    void AlgTemplateInitPara(
        const CollAlgOperator& op, InsAlgTemplate0& intraScatterTempAlg, InsAlgTemplate1& interScatterTempAlg,
        InsAlgTemplate2& intraAllGatherTempAlg, InsAlgTemplate3& interAllGatherTempAlg)
    {
        intraScatterTempAlg.SetDmaMode(dmaMode_);
        intraScatterTempAlg.SetCollOp(op);
        intraScatterTempAlg.SetDataType(dataType_);
        intraScatterTempAlg.SetRoot(intraLocalRoot_);

        interScatterTempAlg.SetDmaMode(dmaMode_);
        interScatterTempAlg.SetCollOp(op);
        interScatterTempAlg.SetDataType(dataType_);
        interScatterTempAlg.SetRoot(interLocalRoot_);

        intraAllGatherTempAlg.SetDmaMode(dmaMode_);
        intraAllGatherTempAlg.SetCollOp(op);
        intraAllGatherTempAlg.SetDataType(dataType_);
        intraAllGatherTempAlg.SetRoot(intraLocalRoot_);

        interAllGatherTempAlg.SetDmaMode(dmaMode_);
        interAllGatherTempAlg.SetCollOp(op);
        interAllGatherTempAlg.SetDataType(dataType_);
        interAllGatherTempAlg.SetRoot(intraLocalRoot_);
        return;
    }
    // Host
    HcclResult PrepareResForTemplate(
        const RankGraph* rankGraph, InsAlgTemplate0& intraScatterTempAlg, InsAlgTemplate1& interScatterTempAlg,
        InsAlgTemplate2& intraAllGatherTempAlg, InsAlgTemplate3& interAllGatherTempAlg);
    // Aicpu
    HcclResult PrepareResForTemplate(
        ConnectedLinkMgr* linkMgr, InsAlgTemplate0& intraScatterTempAlg, InsAlgTemplate1& interScatterTempAlg,
        InsAlgTemplate2& intraAllGatherTempAlg, InsAlgTemplate3& interAllGatherTempAlg);

    void GenDataParamsStage(
        const u32 part, const u32 stage, DataParameters& dataParameters, TemplateDataParams& dataParams) const
    {
        dataParams.buffInfo.inBuffType = BufferType::INPUT;
        dataParams.buffInfo.outBuffType = BufferType::INPUT;
        dataParams.buffInfo.scratBuffType = BufferType::SCRATCH;
        dataParams.buffInfo.inBuffBaseOff = dataParameters.dataOffset[part];
        dataParams.buffInfo.outBuffBaseOff = dataParameters.dataOffset[part];
        dataParams.buffInfo.scratchBuffBaseOff = dataParameters.scratchOffset.at(part).at(stage);
        dataParams.sliceSize = dataParameters.sliceSize.at(part).at(stage);
        dataParams.inputSliceStride = dataParameters.inputStride.at(part).at(stage);
        dataParams.outputSliceStride = dataParameters.sliceSize.at(part).at(stage);
        dataParams.repeatNum = 1;
        dataParams.inputRepeatStride = 0;
        dataParams.outputRepeatStride = 0;
        dataParams.tailSize = dataParameters.tailSize.at(part).at(stage);
        return;
    }
    void InitStageProcAlgParaVec(
        std::vector<StageProcAlgPara>& stageProcAlgParaVec, InsAlgTemplate0& intraScatterTempAlg,
        InsAlgTemplate1& interScatterTempAlg, InsAlgTemplate2& intraAllGatherTempAlg,
        InsAlgTemplate3& interAllGatherTempAlg)
    {
        // 以此输入第1个阶段的part0 GenExtIns, scratchOffset, links,que以及 part1部分对应信息
        stageProcAlgParaVec = {
            {[&](auto&... args) {
                 return interScatterTempAlg.GenExtIns(args...);
             },
             scatterInterLinks_,
             interQue_, // stage0 part0
             [&](auto&... args) {
                 return intraScatterTempAlg.GenExtIns(args...);
             },
             scatterIntraLinks_, intraQue_}, // stage0 part1
            {[&](auto&... args) {
                 return intraScatterTempAlg.GenExtIns(args...);
             },
             scatterIntraLinks_,
             intraQue_, // stage1 part0
             [&](auto&... args) {
                 return interScatterTempAlg.GenExtIns(args...);
             },
             scatterInterLinks_, interQue_}, // stage1 part1
            {[&](auto&... args) {
                 return intraAllGatherTempAlg.GenExtIns(args...);
             },
             allGatherIntraLinks_,
             intraQue_, // stage2 part0
             [&](auto&... args) {
                 return interAllGatherTempAlg.GenExtIns(args...);
             },
             allGatherInterLinks_, interQue_}, // stage2 part1
            {[&](auto&... args) {
                 return interAllGatherTempAlg.GenExtIns(args...);
             },
             allGatherInterLinks_,
             interQue_, // stage3 part0
             [&](auto&... args) {
                 return intraAllGatherTempAlg.GenExtIns(args...);
             },
             allGatherIntraLinks_, intraQue_}, // stage3 part1
        };
    }
    HcclResult GenInsQues(
        InsAlgTemplate0& intraScatterTempAlg, InsAlgTemplate1& interScatterTempAlg,
        InsAlgTemplate2& intraAllGatherTempAlg, InsAlgTemplate3& interAllGatherTempAlg);

    u32 intraLocalRankSize_{0}; // server内算法rankSize
    u32 interLocalRankSize_{0}; // server间算法rankSize

    RankId intraLocalRank_{INVALID_RANKID}; // server内算法rank
    RankId interLocalRank_{INVALID_RANKID}; // server间算法rank

    u32 intraLocalRoot_{0}; // server内算法root
    u32 interLocalRoot_{0}; // server间算法root

    const RankGraph* rankGraph_ = nullptr;

    std::vector<std::vector<std::vector<RankId>>> vTopo_;
    std::vector<std::vector<RankId>> virtRanks_;
    std::vector<std::map<RankId, u32>> virtRankMap_; // map<virtRank, virtRankOrder>

    std::vector<InsQuePtr> requiredQue_;
    std::vector<InsQuePtr> intraQue_;
    std::vector<InsQuePtr> interQue_;
    std::vector<InsQuePtr> syncQueues_;
    ResLinks scatterIntraLinks_;
    ResLinks scatterInterLinks_;
    ResLinks allGatherIntraLinks_;
    ResLinks allGatherInterLinks_;

    const RankGraph* rankGraphPtr_ = nullptr;
};

} // namespace Hccl

#endif
