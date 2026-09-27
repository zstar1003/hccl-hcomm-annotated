/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_ALL_REDUCE_PARALLEL_EXECUTOR_OPT_H
#define HCCLV2_INS_ALL_REDUCE_PARALLEL_EXECUTOR_OPT_H
#include "ins_coll_alg_base.h"

namespace Hccl {

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
class InsAllReduceParallelExecutorV2 : public InsCollAlgBase {
public:
    explicit InsAllReduceParallelExecutorV2();
    ~InsAllReduceParallelExecutorV2() override;

    std::string Describe() const override { return "Instruction based All Reduce Parallel Executor."; }

    // HOST 接口
    HcclResult Orchestrate(
        const RankGraph* rankGraph, const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr insQue) override;
    // AICPU 接口
    HcclResult Orchestrate(
        const AlgTopoInfo& topoInfo, const CollAlgOperator& op, const CollAlgParams& params, ConnectedLinkMgr* linkMgr,
        InsQuePtr insQue) override;

    HcclResult CalcResOffload(const RankGraph* rankGraph, const u64& dataSize, CollOffloadOpResReq& resReq) override;

    HcclResult CalcRes(const RankGraph* rankGraph, CollAlgResReq& algResReq) override;

private:
    HcclResult CalcLocalRankSize();
    HcclResult GenInsQues(
        InsAlgTemplate0& tempAlgIntraRS, InsAlgTemplate1& tempAlgInterRS, InsAlgTemplate2& tempAlgIntraAG,
        InsAlgTemplate3& tempAlgInterAG);
    void GetParallelDataSplitRate(std::vector<float>& splitDataSize) const;
    HcclResult PrepareResForTemplate(
        const RankGraph* rankGraph, InsAlgTemplate0& tempAlgIntraRS, InsAlgTemplate1& tempAlgInterRS,
        InsAlgTemplate2& tempAlgIntraAG, InsAlgTemplate3& tempAlgInterAG);
    HcclResult PrepareResForTemplate(
        ConnectedLinkMgr* linkMgr, InsAlgTemplate0& tempAlgIntraRS, InsAlgTemplate1& tempAlgInterRS,
        InsAlgTemplate2& tempAlgIntraAG, InsAlgTemplate3& tempAlgInterAG);

    void GenRSIntraParams0(
        const u64 dataOffset, const u64 dataCount, const u64 scratchOff, TemplateDataParams& params) const;

    void GenRSInterParams0(
        const u64 dataOffset, const u64 dataCount, const u64 scratchOff, TemplateDataParams& params) const;

    void GenAGInterParams0(
        const u64 dataOffset, const u64 dataCount, const u64 scratchOff, TemplateDataParams& params) const;

    void GenAGIntraParams0(
        const u64 dataOffset, const u64 dataCount, const u64 scratchOff, TemplateDataParams& params) const;

    void GenRSInterParams1(
        const u64 dataOffset, const u64 dataCount, const u64 scratchOff, TemplateDataParams& params) const;

    void GenRSIntraParams1(
        const u64 dataOffset, const u64 dataCount, const u64 scratchOff, TemplateDataParams& params) const;

    void GenAGIntraParams1(
        const u64 dataOffset, const u64 dataCount, const u64 scratchOff, TemplateDataParams& params) const;

    void GenAGInterParams1(
        const u64 dataOffset, const u64 dataCount, const u64 scratchOff, TemplateDataParams& params) const;

    inline void InitAlgCommonParams(
        InsAlgTemplate0& tempAlgIntraRS, InsAlgTemplate1& tempAlgInterRS, InsAlgTemplate2& tempAlgIntraAG,
        InsAlgTemplate3& tempAlgInterAG, const CollAlgOperator& op) const
    {
        tempAlgIntraRS.SetDmaMode(dmaMode_);
        tempAlgIntraRS.InitReduceInfo(redOp_, dataType_);
        tempAlgIntraRS.SetCollOp(op);

        tempAlgInterRS.SetDmaMode(dmaMode_);
        tempAlgInterRS.InitReduceInfo(redOp_, dataType_);
        tempAlgInterRS.SetCollOp(op);

        tempAlgIntraAG.SetDmaMode(dmaMode_);
        tempAlgIntraAG.SetCollOp(op);
        tempAlgIntraAG.SetDataType(dataType_);

        tempAlgInterAG.SetDmaMode(dmaMode_);
        tempAlgInterAG.SetCollOp(op);
        tempAlgInterAG.SetDataType(dataType_);
    }

    // 统一设置 TemplateDataParams 的公共字段
    inline void SetTemplateDataParams(
        TemplateDataParams& params, BufferType inBuffType, BufferType outBuffType, u64 inBuffBaseOff,
        u64 outBuffBaseOff, u64 scratchBuffBaseOff, u64 sliceSize, u64 inputSliceStride, u64 outputSliceStride,
        u32 repeatNum, u64 inputRepeatStride, u64 outputRepeatStride, u64 tailSize) const
    {
        params.buffInfo.inBuffType = inBuffType;
        params.buffInfo.outBuffType = outBuffType;
        params.buffInfo.scratBuffType = BufferType::SCRATCH;
        params.buffInfo.inBuffBaseOff = inBuffBaseOff;
        params.buffInfo.outBuffBaseOff = outBuffBaseOff;
        params.buffInfo.scratchBuffBaseOff = scratchBuffBaseOff;
        params.sliceSize = sliceSize;
        params.inputSliceStride = inputSliceStride;
        params.outputSliceStride = outputSliceStride;
        params.repeatNum = repeatNum;
        params.inputRepeatStride = inputRepeatStride;
        params.outputRepeatStride = outputRepeatStride;
        params.tailSize = tailSize;
    }

    // 计算 Gen*Params1 中 intra 函数的公共 dataCountTmp
    inline u64 CalcDataCountTmp1(u64 dataCount) const
    {
        return (rankIdxLevel1_ != rankSizeLevel1_ - 1) ?
                   dataCount / rankSize_ * rankSizeLevel0_ :
                   dataCount - dataCount / rankSize_ * rankSizeLevel0_ * (rankSizeLevel1_ - 1);
    }

    inline HcclResult CalcQue(
        AlgTempResReq& resReqIntraRS, AlgTempResReq& resReqInterRS, AlgTempResReq& resReqIntraAG,
        AlgTempResReq& resReqInterAG)
    {
        // 申请算法模板所需资源
        if (!(resReqIntraRS.queNum > 0 && resReqInterRS.queNum > 0 && resReqIntraAG.queNum > 0
              && resReqInterAG.queNum > 0)) {
            HCCL_ERROR("resReqIntra.queNum and resReqInter.queNum must be larger than 0.");
            return HcclResult::HCCL_E_INTERNAL;
        }
        u32 intraQueNum = std::max(resReqIntraRS.queNum, resReqIntraAG.queNum);
        u32 interQueNum = std::max(resReqInterRS.queNum, resReqInterAG.queNum);
        u32 totalQueNum = intraQueNum + interQueNum;
        CHK_RET(InitQueue(totalQueNum, requiredQue_));
        for (u32 i = 0; i < requiredQue_.size(); i++) {
            if (i < intraQueNum) {
                intraQue_.push_back(requiredQue_[i]);
            } else {
                interQue_.push_back(requiredQue_[i]);
            }
        }
        HCCL_INFO(
            "[CalcQue] requiredQue_.size is [%llu]. intraQue_.size is [%llu]. interQue_.size is [%llu].",
            requiredQue_.size(), intraQue_.size(), interQue_.size());
        syncQueues_.emplace_back(intraQue_[0]);
        syncQueues_.emplace_back(interQue_[0]);
        return HCCL_SUCCESS;
    }

    uint64_t rankSizeLevel0_{0};
    uint64_t rankSizeLevel1_{0};
    uint64_t rankSize_{0};

    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};

    u64 sliceCount_;

    std::vector<std::vector<RankId>> virtRanks_;
    std::vector<std::map<RankId, u32>> virtRankMap_; // map<virtRank, virtRankOrder>
    std::vector<std::vector<std::vector<RankId>>> vTopo_;

    std::vector<InsQuePtr> requiredQue_;
    std::vector<InsQuePtr> intraQue_;
    std::vector<InsQuePtr> interQue_;
    std::vector<InsQuePtr> syncQueues_;

    ResLinks intraRSLinks_;
    ResLinks interRSLinks_;
    ResLinks intraAGLinks_;
    ResLinks interAGLinks_;
};

} // namespace Hccl

#endif // HCCLV2_INS_ALL_REDUCE_PARALLEL_EXECUTOR_H
