/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_REPRESENTATION_BUFREDUCE_H
#define HCCL_CCU_REPRESENTATION_BUFREDUCE_H

#include <vector>

#include "ccu_rep_base.h"
#include "ccu_datatype.h"
#include "ccu_error_handler.h"

namespace Hccl {
namespace CcuRep {

    class CcuRepBufReduce : public CcuRepBase {
    public:
        CcuRepBufReduce(
            const std::vector<CcuBuffer>& mem, uint16_t count, uint16_t dataType, uint16_t outputDataType,
            uint16_t opType, MaskSignal sem, const CcuRep::Variable& len, uint16_t mask = 1);
        bool Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;

    private:
        std::vector<CcuBuffer> mem;
        uint16_t count;
        uint16_t dataType;
        uint16_t outputDataType;
        uint16_t opType;
        MaskSignal sem;
        CcuRep::Variable xnIdLength_;
        uint16_t mask{0};

        friend class Hccl::CcuErrorHandler;
    };

}; // namespace CcuRep
}; // namespace Hccl
#endif // HCCL_CCU_REPRESENTATION_BUFREDUCE_H
