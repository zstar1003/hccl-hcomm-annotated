/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_RUN_ALLTOALLV_DIRECT_FULLMESH_EXECUTOR_H
#define COLL_RUN_ALLTOALLV_DIRECT_FULLMESH_EXECUTOR_H
#include "coll_all_to_all_executor.h"
namespace hccl {
class CollRunAlltoAllDirectFullmesh : public CollAlltoAllExecutor {
public:
    CollRunAlltoAllDirectFullmesh(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollRunAlltoAllDirectFullmesh() override = default;
    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;
    HcclResult GetAdjInfo(AlgResourceResponse& algRes, AdjInfo& adjInfo) override;

    // 用于alltoallv算子的aicpu展开cache
    HcclResult MarkNeedAlltoallvCache() override;
    HcclResult
    GetHcclOffsetDstRanksMap(std::unordered_map<uint64_t, std::vector<uint32_t>>& hcclOffsetDstRanksMap) const override;

private:
    HcclOpMetaInfo GetOpMeta(HcclCMDType opType, const u64 size) override;
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcLevel0CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;
    HcclResult Getlevel1CommRank(SubCommInfo& level1CommInfo) override;
    HcclResult SelectTempAlg(std::unique_ptr<AlgTemplateBase>& level1TempAlg, u32 level1RankSize) override;
    HcclResult GetDevNumInlocalPod(u32& devNumInlocalPod) override;
    HcclResult GetAlltoAllvTmpRankSendRecvInfo(const OpParam& param);
    HcclResult GetLocalSendRecvInfoforAlltoall(const OpParam& param);
    HcclResult GetLocalSendRecvInfoforAlltoallV(const OpParam& param);
    HcclResult GetLocalSendRecvInfoforAlltoallVC(const OpParam& param);
    HcclResult CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const;
    HcclResult GetLocalSDMAGroupInfo(const u32 userRank, u32& devNumInlocalPod, u32& rankIdxInPod) const;

    // 用于alltoallv算子的aicpu展开cache
    bool needAlltoallvCache_ = false; // 是否需要对当前alltoallv算子做aicpu cache
    std::unordered_map<uint64_t, std::vector<uint32_t>>
        hcclOffsetDstRanksMap_; // Local hccl input buffer中的local hccl offset到remote dst ranks的映射
                                // (用于PrepareIntraData)
};
} // namespace hccl
#endif
