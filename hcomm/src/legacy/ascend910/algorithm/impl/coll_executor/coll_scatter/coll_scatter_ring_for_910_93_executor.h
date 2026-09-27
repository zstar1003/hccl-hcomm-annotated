/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_SCATTER_RING_FOR_910_93_EXECUTOR_H
#define COLL_SCATTER_RING_FOR_910_93_EXECUTOR_H
#include "coll_scatter_executor.h"

namespace hccl {
class CollScatterRingFor91093Executor : public CollScatterExecutor {
public:
    explicit CollScatterRingFor91093Executor(
        const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollScatterRingFor91093Executor() override = default;

private:
    using CollScatterExecutor::KernelRunLevel1;
    /* *************** 资源计算 *************** */
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcLevel0CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel1CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel2CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;

    /* *************** 算法编排 *************** */
    HcclResult KernelRunLevel2(const OpParam& param, ExecMem& execMem, Stream& stream);
    HcclResult KernelRunLevel1(const OpParam& param, ExecMem& execMem, Stream& stream);
    HcclResult KernelRunLevel0(const OpParam& param, ExecMem& execMem, Stream& stream);
    HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;
    HcclResult Getlevel1CommRank(SubCommInfo& level1CommInfo) override;
    HcclResult SelectTempAlg(std::unique_ptr<AlgTemplateBase>& level1TempAlg, u32 level1RankSize) override;
    /* *************** 算法参数 *************** */
    u32 subRoot_ = 0;
    u32 commIndex_ = 0;
    u32 perDataSize_ = 0;
    u64 level1SliceOffset_ = 0;
    u64 serverSliceOffset_ = 0;
    u32 subUserRankRootSupperPod_ = 0;
    SubCommInfo level0CommInfo_;
    SubCommInfo level1CommInfo_;
    SubCommInfo level2CommInfo_;
};

} // namespace hccl

#endif
