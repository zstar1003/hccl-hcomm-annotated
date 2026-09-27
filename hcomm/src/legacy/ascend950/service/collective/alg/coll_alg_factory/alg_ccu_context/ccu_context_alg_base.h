/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_ALG_BASE_H_
#define HCCLV2_CCU_CONTEXT_ALG_BASE_H_

#include <vector>
#include <ios>
#include "log.h"
#include "ccu_ctx.h"
#include "ccu_datatype.h"

namespace Hccl {

class CcuContextAlgBase : public CcuContext {
public:
    CcuContextAlgBase(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
        : CcuContext(arg, transports, group)
    {}
    ~CcuContextAlgBase() override {}

    // 子类实现
    void Algorithm() override = 0;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override = 0;

protected:
    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    ReduceOp reduceOp_;

    struct GroupOpSizeV2 {
        CcuRep::Variable baseIterNum;
        CcuRep::Variable tailLoopId;
        CcuRep::Variable tail;
    };

    GroupOpSizeV2 CreateGroupOpSizeV2() { return GroupOpSizeV2{CreateVariable(), CreateVariable(), CreateVariable()}; }

    void GroupBroadcastV2(
        const std::vector<CcuTransport*>& transports, const std::vector<CcuRep::Memory>& dst, const CcuRep::Memory& src,
        const GroupOpSizeV2& goSize) const;
    void GroupReduceV2(
        const std::vector<CcuTransport*>& transports, const CcuRep::Memory& dst, const std::vector<CcuRep::Memory>& src,
        const GroupOpSizeV2& goSize, DataType dataType, DataType outputDataType, ReduceOp reduceType) const;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_ALG_BASE_H_
