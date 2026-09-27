/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_REPRESENTATION_BASE
#define HCCL_CCU_REPRESENTATION_BASE

#include <string>

#include "ccu_microcode_v1.h"

namespace hcomm {
namespace CcuRep {
    struct TransDep {
        int32_t logicalId;
        uint16_t dieId;
        uint16_t reserveXnId;
        uint16_t reserveGsaId;
        uint16_t reserveCkeId;
        uint16_t reserveChannalId[2]; //  0: selfLoopBack; 1: inter die, 0xffff为无效值，rep翻译时检查
        uint64_t xnBaseAddr;
        uint64_t ccuResSpaceTokenInfo;
        uint64_t memTokenInfo;
        uint16_t commXn[3];  // 3个Xn
        uint16_t commGsa[2]; // 2个GSA
        uint16_t commSignal; // 1个CKE
        uint16_t loadXnId;
        bool isFuncBlock;
    };

    class CcuRepBase {
    public:
        explicit CcuRepBase();
        virtual ~CcuRepBase();
        virtual bool Translate(hcomm::CcuRep::CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) = 0;
        virtual std::string Describe() = 0;

        CcuRepType Type() const;
        bool Translated() const;
        uint16_t StartInstrId() const;
        virtual uint16_t InstrCount();

    protected:
        CcuRepType type{CcuRepType::BASE};
        bool translated{false};
        uint16_t instrId{0};
        uint16_t instrCount{0};
    };
}; // namespace CcuRep
}; // namespace hcomm
#endif // HCCL_CCU_REPRESENTATION_BASE
