/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_REDUCE_SCATTER_V_MESH_1D_MEM2MEM_H_
#define HCCLV2_CCU_CONTEXT_REDUCE_SCATTER_V_MESH_1D_MEM2MEM_H_

#include <vector>
#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"

namespace Hccl {

class CcuContextReduceScatterVMeshMem2Mem1D : public CcuContextAlgBase {
public:
    CcuContextReduceScatterVMeshMem2Mem1D(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextReduceScatterVMeshMem2Mem1D() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

protected:
    void CollectAllRanksSlice(
        std::vector<CcuRep::Memory>& tmpSrc, std::vector<CcuRep::Memory>& tmpDst, const CcuRep::MaskSignal& locMask);
    void InitResources();
    void PrepareReduceScatterVData(
        std::vector<CcuRep::Memory>& reduceScatterVSrc, std::vector<CcuRep::Memory>& reduceScatterVDst);

private:
    DataType dataType_;
    DataType outputDataType_;
    std::vector<CcuRep::Variable> input_;
    CcuRep::Variable output_;
    std::vector<CcuRep::Variable> scratch_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable sliceSize_;
    CcuRep::Variable scratchInterval_;
    CcuRep::Variable offset_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_REDUCE_SCATTER_V_MESH_1D_MEM2MEM_H_
