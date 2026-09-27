/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_ALL_REDUCE_MESH_1D_H_
#define HCCLV2_CCU_CONTEXT_ALL_REDUCE_MESH_1D_H_

#include <vector>
#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"
#include "ccu_device_manager.h"
#include "ccu_context_alg_base.h"

namespace Hccl {

class CcuContextAllReduceMesh1D : public CcuContextAlgBase {
public:
    CcuContextAllReduceMesh1D(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextAllReduceMesh1D() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

private:
    void RunBroadcast(std::vector<CcuRep::Memory>& dst, CcuRep::Memory& src);
    void RunReduce(CcuRep::Memory& dst, std::vector<CcuRep::Memory>& src);

    CcuVersion ccuVersion_ = CcuVersion::CCU_INVALID;
    DataType dataType_;
    DataType outputDataType_;
    std::vector<CcuRep::Variable> input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable offset_;
    GroupOpSize groupOpSize_;
    GroupOpSizeV2 groupOpSizeV2_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_REDUCE_MESH_1D_H_
