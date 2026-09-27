/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_BROADCAST_MESH_1D_H_
#define HCCLV2_CCU_CONTEXT_BROADCAST_MESH_1D_H_

#include <vector>
#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"

namespace Hccl {

class CcuContextBroadcastMesh1D : public CcuContextAlgBase {
public:
    CcuContextBroadcastMesh1D(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextBroadcastMesh1D() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

private:
    void CreateAllVariables();
    void LoadAndExchangeData();
    void BroadcastFromRootToAll();

    uint32_t rootId_{0}; // 当rankid == rootid时，为root节点 则跳过write操作
    DataType dataType_;
    DataType outputDataType_;
    CcuRep::Variable input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable offset_;
    CcuRep::Variable slicesize_;
    GroupOpSize groupOpSize_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_BROADCAST_MESH_1D_H_
