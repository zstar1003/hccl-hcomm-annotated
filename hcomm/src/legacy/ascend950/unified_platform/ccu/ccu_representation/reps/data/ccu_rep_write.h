/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_REPRESENTATION_WRITE_H
#define HCCL_CCU_REPRESENTATION_WRITE_H

#include "ccu_rep_base.h"
#include "ccu_datatype.h"
#include "ccu_transport.h"
#include "ccu_error_handler.h"

namespace Hccl {
namespace CcuRep {

    class CcuRepWrite : public CcuRepBase {
    public:
        CcuRepWrite(const CcuTransport& transport, Memory rem, Memory loc, Variable len, MaskSignal sem, uint16_t mask);
        CcuRepWrite(
            const CcuTransport& transport, Memory rem, Memory loc, Variable len, uint16_t dataType, uint16_t opType,
            MaskSignal sem, uint16_t mask);
        bool Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
        uint32_t GetTransportChannelId() { return transport.GetChannelId(); }

    private:
        const CcuTransport& transport;

        Memory rem;
        Memory loc;
        Variable len;

        MaskSignal sem;
        uint16_t mask{0};

        uint16_t dataType{0};
        uint16_t opType{0};
        uint16_t reduceFlag{0};

        friend class Hccl::CcuErrorHandler;
    };

}; // namespace CcuRep
}; // namespace Hccl
#endif // HCCL_CCU_REPRESENTATION_WRITE_H
