/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALL_REDUCE_RING_FOR_910_93_EXECUTOR_H
#define COLL_ALL_REDUCE_RING_FOR_910_93_EXECUTOR_H

#include "coll_all_reduce_executor.h"

namespace hccl {
class CollAllReduceRingFor91093Executor : public CollAllReduceExecutor {
public:
    CollAllReduceRingFor91093Executor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollAllReduceRingFor91093Executor() override = default;

private:
    /* *************** 资源计算 *************** */
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel0CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel2CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const;

    /* *************** 算法编排 *************** */
    bool IsHugeData(const u64 curSize) override;
    bool IsSmallData(const u64 totalSize, const u64 curSize) override;
    virtual HcclResult RunIntraSeverReduceScatter(
        const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, const u64 count,
        const HcclDataType& dataType, const HcclReduceOp& reductionOp,
        const std::vector<std::vector<Slice>>& multRingsSliceZero, const Stream& stream, s32 profStage,
        const u64 baseOffset = 0, const HcomCollOpInfo* opInfo = nullptr,
        const std::vector<std::vector<Slice>>& multRingsUserMemSlice = std::vector<std::vector<Slice>>(0),
        const bool disableDMAReduce = false);
    virtual HcclResult RunIntraSeverAllGather(
        const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, const u64 count,
        const HcclDataType& dataType, const std::vector<std::vector<Slice>>& multRingsSliceZero, const Stream& stream,
        s32 profStage, const u64 baseOffset = 0, const HcomCollOpInfo* opInfo = nullptr,
        const std::vector<std::vector<Slice>>& multRingsUserMemSlice = std::vector<std::vector<Slice>>(0));
    HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;
    HcclResult Getlevel1CommRank(SubCommInfo& level1CommInfo) override;
    HcclResult SelectTempAlg(std::unique_ptr<AlgTemplateBase>& level1TempAlg, u32 level1RankSize) override;
    virtual HcclResult GetLevelCommInfo();
    HcclResult GetNicList(std::vector<u32>& mockNicList);
    HcclResult PrepareARSLevel1CommInfo(
        u32& segmentIdx, u32& commIndex, u64& hdSize, const SubCommInfo& commInfo,
        const std::vector<std::vector<Slice>>& multRingsSliceZero, const std::string& tag,
        const std::vector<u32>& nicList);

protected:
    SubCommInfo logicalLevel0CommInfo_ = {0, 0, {}, {}};
    SubCommInfo logicalLevel1CommInfo_ = {0, 0, {}, {}};
    CommPlane logicalLevel0plane_ = COMM_LEVEL_RESERVED;
    CommPlane logicalLevel1plane_ = COMM_LEVEL_RESERVED;
};

} // namespace hccl

#endif
