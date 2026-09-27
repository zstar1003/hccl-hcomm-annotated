/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_TEMP_ALL_TO_ALL_MESH
#define HCCLV2_INS_TEMP_ALL_TO_ALL_MESH

#include "string_util.h"

#include "ins_alg_template_base.h"
#include "alg_data_trans_wrapper.h"

namespace Hccl {

const uint32_t ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE = 16; // fullmesh最大的并发数量

class InsTempAlltoAllMesh : public InsAlgTemplateBase {
public:
    explicit InsTempAlltoAllMesh(
        const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
        const std::map<RankId, u32>& tempVirtRankMap);
    ~InsTempAlltoAllMesh() override;

    std::string Describe() const override
    {
        return StringFormat("Instruction based Template of alltoall mesh with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult CalcRes(AlgTempResReq& tempResReq) override;
    HcclResult
    Run(const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo,
        const ResLinks& tempLinks, std::vector<InsQuePtr>& tempInsQues) override;
    void SetA2ASendRecvInfo(const A2ASendRecvInfo& sendRecvInfo);
    HcclResult GetScratchBufferInfo(const u64& scratchBufferSize, DataType dataType);

private:
    u32 CalcStepNum(); // 计算step数
    HcclResult CalcSendRecvAllSliceInfo(
        std::unordered_map<u32, UsrData>& sendSliceInfoMap,
        std::unordered_map<u32, UsrData>& recvSliceInfoMap); // 计算每个rank当前step的send/recv slice

    HcclResult CalcSendSliceInfo(u32 remoteRank, UsrData& sendSliceInfo);

    HcclResult CalcRecvSliceInfo(u32 remoteRank, UsrData& readSliceInfo);

    HcclResult RunSendRecvForAllRanks(
        u32 step, std::unordered_map<u32, UsrData>& sendSliceInfoMap,
        std::unordered_map<u32, UsrData>& recvSliceInfoMap, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& queues); // 所有对端send/Recv循环一次

    HcclResult RunSendRecvBufferLoop(
        u32 step, const std::vector<u32>& commRanks, std::unordered_map<u32, UsrData>& sendSliceInfoMap,
        std::unordered_map<u32, UsrData>& recvSliceInfoMap, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& queues) const; // 循环一次buffer

    HcclResult CalcCommRankSetforOneLoop(
        u32 roundIdx, const u32 groupRankSize,
        std::vector<u32>& commRanks) const; // 计算当前循环的send/recv rank set

    HcclResult CopySendDataToScratch(
        u32 step, const std::vector<u32>& commRanks, std::unordered_map<u32, UsrData>& sendSliceInfo,
        const ResLinks& tempLinks, std::vector<InsQuePtr>& queues) const;

    HcclResult SendRecvData(
        u32 step, const std::vector<u32>& commRanks, std::unordered_map<u32, UsrData>& sendSliceInfo,
        std::unordered_map<u32, UsrData>& readSliceInfo, const ResLinks& tempLinks,
        std::vector<InsQuePtr>& queues) const;

    HcclResult CopyRecvDataFromScratch(
        u32 step, const std::vector<u32>& commRanks, std::unordered_map<u32, UsrData>& readSliceInfo,
        const ResLinks& tempLinks, std::vector<InsQuePtr>& queues) const;

    HcclResult LocalDataCopy(InsQuePtr tempInsQue);

    HcclResult SetBuffBlockSize(const u64 buffBlockSize);

    HcclResult SetConcurrentSendRecvNum(const u32 concurrentSendRecvNum);

    u32 concurrentSendRecvNum_ = 8;
    u64 buffBlockSize_ = 0;
    A2ASendRecvInfo localSendRecvInfo_;
    BuffInfo buffInfo_;
    u32 maxPathNum = 0;
};

} // namespace Hccl

#endif // !HCCLV2_INS_TEMP_ALL_TO_ALL_MESH
