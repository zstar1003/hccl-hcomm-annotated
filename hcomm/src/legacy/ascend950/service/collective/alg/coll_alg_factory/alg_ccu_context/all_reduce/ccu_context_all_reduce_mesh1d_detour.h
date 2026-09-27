/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_ALL_REDUCE_MESH_DETOUR_1D_H_
#define HCCLV2_CCU_CONTEXT_ALL_REDUCE_MESH_DETOUR_1D_H_

#include <vector>
#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"
#include "ccu_assist.h"

namespace Hccl {

class CcuContextAllReduceMeshDetour1D : public CcuContextAlgBase {
public:
    CcuContextAllReduceMeshDetour1D(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextAllReduceMeshDetour1D() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

private:
    void CreateMultiOpReduceDetour(DataType& dataType, DataType& outputDataType, ReduceOp& opType);
    void GroupReduceDetour(
        std::vector<CcuRep::Memory>& src, std::vector<CcuRep::Memory>& dst, DataType& dataType,
        DataType& outputDataType, ReduceOp& opType);
    void CreateMultiOpBroadcastDetour();
    void GroupBroadcastDetour(
        std::vector<CcuRep::Variable>& lengths, std::vector<CcuRep::Memory>& src, std::vector<CcuRep::Memory>& dst);
    void ReduceScatterFirstStep();
    void ReduceScatterSecondStep();
    void AllGatherFirstStep();
    void AllGatherSecondStep();

    uint64_t singleTransportSize{0}; // 每个loop单次传输的总数据量，通信域级别
    uint64_t detourPathNum{0};
    uint64_t pathNumPerPeer{0}; // 到每个rank有几个transport，包括重复的
    std::vector<std::vector<CcuTransport*>> detourTransports_;
    CcuRep::Variable tailOffset_; // 尾块相对偏移，singleTransportSize*128*iterNum

    DataType dataType_;
    DataType outputDataType_;
    std::vector<CcuRep::Variable> input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable offset_;
    CcuRep::Variable iterNum_;
    CcuRep::Variable tailSize_;
    GroupOpSize groupOpSize_;
    std::vector<CcuRep::Variable> lengths_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_REDUCE_MESH_DETOUR_1D_H_
