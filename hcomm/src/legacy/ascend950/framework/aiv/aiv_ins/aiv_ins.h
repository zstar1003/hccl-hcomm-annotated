/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_INSTRUCTION_H
#define AIV_INSTRUCTION_H

#include "instruction.h"
#include "hccl_aiv_utils.h"

namespace Hccl {

class AivInstruction : public Instruction {
public:
    AivInstruction(const std::vector<LinkData>& links, const AivOpArgs& aivOpArgs)
        : Instruction(InstructionType::AIV_INS),
          links_(links),
          aivOpArgs_(aivOpArgs)
    {}

    std::string Describe() const override;
    const std::vector<LinkData> GetLinks() const;
    HcclResult GetAivInsArgs(AivOpArgs& aivOpArgs) const;

private:
    std::vector<LinkData> links_;
    AivOpArgs aivOpArgs_;
};

} // namespace Hccl

#endif // AIV_INSTRUCTION_H
