/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_REGISTER_OP_CT_IMPL_KERNEL_REGISTRY_H_
#define INC_EXTERNAL_REGISTER_OP_CT_IMPL_KERNEL_REGISTRY_H_
#include "exe_graph/runtime/base_type.h"
#include "graph/ascend_string.h"
#include "exe_graph/runtime/exe_res_generation_context.h"

#define OP_CT_IMPL_MAIM_VERSION 1

namespace gert {
struct OpCtImplKernelRegistry {
    using OpType = ge::AscendString;
    using OpCalcParamKernelFunc = UINT32 (*)(ExeResGenerationContext* context);
    using OpGenTaskKernelFunc
        = UINT32 (*)(const ExeResGenerationContext* context, std::vector<std::vector<uint8_t>>& tasks);
    using OP_CHECK_FUNC_V2 = ge::graphStatus (*)(const OpCheckContext* context, ge::AscendString& result);
    struct OpCtImplFunctions {
        uint32_t st_size = sizeof(OpCtImplFunctions);
        uint32_t version = OP_CT_IMPL_MAIM_VERSION;
        OpCalcParamKernelFunc calc_op_param = nullptr;
        OpGenTaskKernelFunc gen_task = nullptr;
        OP_CHECK_FUNC_V2 check_support = nullptr;
        OP_CHECK_FUNC_V2 op_select_format = nullptr;
        OP_CHECK_FUNC_V2 get_op_support_info = nullptr;
        OP_CHECK_FUNC_V2 get_op_specific_info = nullptr;
    };
};
} // namespace gert
#endif
