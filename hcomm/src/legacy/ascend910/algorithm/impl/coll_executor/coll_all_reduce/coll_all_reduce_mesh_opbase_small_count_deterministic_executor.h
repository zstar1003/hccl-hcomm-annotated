/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALLREDUCE_MESH_OPBASE_SMALL_COUNT_DETERMINISTIC_EXECUTOR_H
#define COLL_ALLREDUCE_MESH_OPBASE_SMALL_COUNT_DETERMINISTIC_EXECUTOR_H

#include "coll_all_reduce_executor.h"

namespace hccl {
class CollAllReduceMeshOpbaseSmallCountDeterministicExecutor : public CollAllReduceExecutor {
public:
    CollAllReduceMeshOpbaseSmallCountDeterministicExecutor(
        const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollAllReduceMeshOpbaseSmallCountDeterministicExecutor() override = default;

private:
    /**************** 资源计算 *************** */
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel0CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel1CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const;
    /* *************** 任务编排 *************** */
    u64 CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize) override;
    bool IsHugeData(const u64 curSize) override;
    bool IsSmallData(const u64 totalSize, const u64 curSize) override;
    HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;
    bool IsPowerOfTwo(u32 num) const;
    HcclResult
    RunDoublingSingleLevel(const OpParam& param, u64 reduceAttr, ExecMem& execMem, SubCommInfo& levelCommInfo) const;
    HcclResult RunReduceBcastSingleLevel(
        const OpParam& param, HcomCollOpInfo& opInfo, u64 reduceAttr, ExecMem& execMem,
        SubCommInfo& levelCommInfo) const;
    HcclResult RunTempLevel1(
        const TemplateType type, const OpParam& param, u64 reduceAttr, ExecMem& execMem,
        SubCommInfo& level1CommInfo) const;
    HcclResult RunRingLevel1(const OpParam& param, u64 reduceAttr, ExecMem& execMem, SubCommInfo& level1CommInfo);
    HcclResult RunNHRLevel1(const OpParam& param, u64 reduceAttr, ExecMem& execMem, SubCommInfo& level1CommInfo);
    HcclResult RunRHDLevel1(const OpParam& param, u64 reduceAttr, ExecMem& execMem, SubCommInfo& level1CommInfo);
};

} // namespace hccl

#endif
