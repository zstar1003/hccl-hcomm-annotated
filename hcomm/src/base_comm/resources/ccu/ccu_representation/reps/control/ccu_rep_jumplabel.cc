/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_v1.h"

#include "string_util.h"
#include "ccu_ins_generator_base.h"

namespace hcomm {
namespace CcuRep {

    CcuRepJumpLabel::CcuRepJumpLabel(CcuInsGeneratorBase* insGeneratorPtr, const std::string& label)
        : CcuRepBlock(insGeneratorPtr, label)
    {
        type = CcuRepType::JUMP_LABEL;
        Append(std::make_shared<CcuRepNop>(insGeneratorPtr));
    }

    std::string CcuRepJumpLabel::Describe() { return Hccl::StringFormat("JumpLabel[%s]", GetLabel().c_str()); }

}; // namespace CcuRep
}; // namespace hcomm
