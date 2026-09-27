/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_INSTRUCTION_GROUP_H
#define CCU_INSTRUCTION_GROUP_H

#include "ccu_ins.h"
#include "types.h"
#include "instruction.h"

namespace Hccl {

class CcuInsGroup : public CcuInstruction {
public:
    CcuInsGroup() : CcuInstruction() {}

    void SetExecId(u64 id) override;
    void Append(std::unique_ptr<CcuInstruction> ins);
    const std::vector<std::unique_ptr<CcuInstruction>>& GetCcuInstructions() const;
    CcuCtxSignature GetCtxSignature() const override;
    u64 GetExecId() const override;
    std::string Describe() const override;
    CcuInstType GetInstType() const override;
    std::unique_ptr<CcuCtxArg> GetCtxArg() const override;
    std::unique_ptr<CcuTaskArg> GetTaskArg() const override;
    std::vector<LinkData> GetLinks() const override;
    RankGroup GetRankGroup() const override;

private:
    std::vector<std::unique_ptr<CcuInstruction>> ccuInstructions;
};

} // namespace Hccl

#endif // CCU_INSTRUCTION_GROUP_H
