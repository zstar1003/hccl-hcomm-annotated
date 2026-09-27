/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_MESH_2D
#define INS_TEMP_REDUCE_MESH_2D

#include "string_util.h"
#include "executor_utils.h"
#include "ins_alg_template_base.h"

namespace Hccl {

// 通信方向常量
constexpr u32 AXIS_X = 0;
constexpr u32 AXIS_Y = 1;
constexpr u32 AXIS_NUM = 2;

// 数据分片常量（数据分为A/B两部分并行处理）
constexpr u32 SLICE_A = 0;
constexpr u32 SLICE_B = 1;
constexpr u32 SLICE_NUM = 2;

class InsTempReduceMesh2D : public InsAlgTemplateBase {
public:
    explicit InsTempReduceMesh2D(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempReduceMesh2D() override;

    std::string Describe() const override
    {
        return StringFormat("Template of reduce Mesh2D with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    u32 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType);

    HcclResult GenExtIns(
        const TempFuncs& tempFuncs, const TemplateDataParams& templateDataParams, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& tempInsQues);

private:
    HcclResult CalcParams(const TemplateDataParams& templateDataParams);
    HcclResult SplitInsQues(
        std::vector<InsQuePtr>& tempInsQues, std::vector<InsQuePtr>& ctrlTempInsQues,
        std::vector<InsQuePtr>& xTempInsQues, std::vector<InsQuePtr>& yTempInsQues);
    std::vector<std::tuple<QId, QId, u32>> CreateNotifiesRequest(u32 xQueueNum, u32 yQueueNum) const;
    HcclResult CalcResLinksConcurrMesh(
        const RankId myRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const u32 linkNumBtwPeers, AlgTempResReq& tempResReq) const;

    HcclResult
    LocalCopyFromInputToOutput(const TemplateDataParams& templateDataParams, std::vector<InsQuePtr>& tempInsQues) const;
    HcclResult GatherFromInput(
        const u32 slice, const u32 axis, const ResLinks& tempLinks, std::vector<InsQuePtr>& axisTempInsQues);
    HcclResult GatherFromScratch(
        const u32 slice, const u32 axis, const ResLinks& tempLinks, std::vector<InsQuePtr>& axisTempInsQues);
    HcclResult
    SendFromInput(const u32 slice, const u32 axis, const ResLinks& tempLinks, std::vector<InsQuePtr>& axisTempInsQues);
    HcclResult SendFromScratch(
        const u32 slice, const u32 axis, const ResLinks& tempLinks, std::vector<InsQuePtr>& axisTempInsQues);
    HcclResult ReduceToScratch(const u32 slice, const u32 axis, std::vector<InsQuePtr>& axisTempInsQues);
    HcclResult ReduceToOutput(const u32 slice, const u32 axis, std::vector<InsQuePtr>& axisTempInsQues);

    u32 axisRankSize_[AXIS_NUM]{0}; // 轴向RankSize大小
    u32 axisRank_[AXIS_NUM]{0};     // 当前rank在轴向上的序号
    u32 axisRoot_[AXIS_NUM]{0};     // root在轴向上的序号

    u64 sliceSize_[SLICE_NUM]{0}; // 数据分片大小（双轴并行通信）
    u64 sliceInputBaseOffset_[SLICE_NUM]{0};
    u64 sliceScratchBaseOffset_[SLICE_NUM]{0};
    u64 sliceOutputBaseOffset_[SLICE_NUM]{0};
};

} // namespace Hccl

#endif // INS_TEMP_REDUCE_MESH_2D
