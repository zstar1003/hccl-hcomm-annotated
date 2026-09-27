/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_LOOP_BLOCK_H
#define CCU_LOOP_BLOCK_H

#include "ccu_rep_block_v1.h"
#include "ccu_rep_loopblock_v1.h"
#include "ccu_rep_context_v1.h"

namespace hcomm {
namespace CcuRep {

    class LoopBlock {
    public:
        LoopBlock(CcuRepContext* context, std::string label);
        ~LoopBlock();

        template <typename... Arguments>
        LoopBlock& operator()(const Arguments&... args)
        {
            DefineInArgHelper(args...);
            return *this;
        }

    private:
        template <typename First>
        void DefineInArgHelper(const First& first)
        {
            repLoopBlock->DefineArg(first);
        }

        template <typename First, typename... Rest>
        void DefineInArgHelper(const First& first, const Rest&... rest)
        {
            repLoopBlock->DefineArg(first);
            DefineInArgHelper(rest...);
        }

        CcuRepContext* context{nullptr};
        std::string label;

        std::shared_ptr<CcuRepLoopBlock> repLoopBlock{nullptr};
        std::shared_ptr<CcuRepBlock> curActiveBlock{nullptr};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // _CCU_LOOP_BLOCK_H
