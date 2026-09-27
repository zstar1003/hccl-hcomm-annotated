/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_REPRESENTATION_JUMPLABEL_H
#define HCCL_CCU_REPRESENTATION_JUMPLABEL_H

#include "ccu_rep_block.h"

namespace Hccl {
namespace CcuRep {

    class CcuRepJumpLabel : public CcuRepBlock {
    public:
        explicit CcuRepJumpLabel(const std::string& label);
        std::string Describe() override;
    };

}; // namespace CcuRep
}; // namespace Hccl
#endif // HCCL_CCU_REPRESENTATION_JUMPLABEL_H
