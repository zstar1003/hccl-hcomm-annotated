/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_INSTRUCTION_H
#define AICPU_INSTRUCTION_H

#include "instruction.h"
#include "virtual_topo.h"
#include "coll_alg_params.h"

#include "template_utils.h"

namespace Hccl {

class AicpuInstruction : public Instruction {
public:
    AicpuInstruction(const std::string& algName, const CollAlgResReq& collAlgResReq, const TemplateInfo& tmpInfo)
        : Instruction(InstructionType::AICPU_INS),
          algName(algName),
          collAlgResReq(collAlgResReq),
          tmpInfo(tmpInfo)
    {}

    std::string GetAlgName() const { return algName; }

    CollAlgResReq GetCollAlgResReq() const { return collAlgResReq; }

    virtual std::vector<LinkData> GetLinks() const { return collAlgResReq.links; }

    std::string Describe() const override
    {
        return StringFormat(
            "AicpuInstruction[algName=%s, linkNum=%zu, workQueueNum=%u]", algName.c_str(), collAlgResReq.links.size(),
            collAlgResReq.primQueueNum);
    }

protected:
    std::string algName{""};
    CollAlgResReq collAlgResReq{};

    TemplateInfo tmpInfo;
};

} // namespace Hccl

#endif // AICPU_INSTRUCTION_H
