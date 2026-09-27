/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_50EA5B1AAF3341A28036E698708ADB64_H
#define INC_50EA5B1AAF3341A28036E698708ADB64_H
#include <cstdint>
#include <unordered_set>
#include "graph/ge_error_codes.h"
#include "exe_graph/runtime/base_type.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/op_execute_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "graph/ascend_string.h"
#include "register/op_impl_registry.h"

namespace gert {
class TilingParseContext;
class InferSymbolShapeContext;
class ExeResGenerationContext;
class OpCheckContext;
#define OP_IMPL_MAIN_VERSION 2
struct OpImplKernelRegistry {
    // for other code repo, they use those alias, but will delete later
    using InferShapeKernelFunc = UINT32 (*)(InferShapeContext*);
    using InferShapeRangeKernelFunc = UINT32 (*)(InferShapeRangeContext*);
    using TilingKernelFunc = UINT32 (*)(TilingContext*);
    using InferDataTypeKernelFunc = UINT32 (*)(InferDataTypeContext*);
    using GenSimplifiedKeyKernelFunc = UINT32 (*)(TilingContext*, ge::char_t*);
    // aclnn接口的原型，入参含义：
    // OpExecuteContext：保存算子的Input，Output，Attr信息
    using OpExecuteFunc = UINT32 (*)(OpExecuteContext*);
    using OpType = ge::AscendString;
    using PrivateAttrList = std::vector<std::pair<ge::AscendString, ge::AnyValue>>;
    using PrivateAttrSet = std::unordered_set<ge::AscendString>;
    using CompileInfoCreatorFunc = void* (*)();
    using CompileInfoDeleterFunc = void (*)(void*);
    using KernelFunc = UINT32 (*)(KernelContext* context);
    using TilingParseFunc = UINT32 (*)(TilingParseContext* context);
    struct OpImplFunctionsV2;
    struct OpImplFunctions {
        OpImplFunctions& operator=(OpImplFunctionsV2& func);
        bool HasDataDependency() const { return (inputs_dependency != 0U); }
        /*
         * param index: must be ir index
         */
        bool IsInputDataDependency(const size_t index) const
        {
            if (index >= sizeof(inputs_dependency) * kByteBitCount) {
                return false;
            }
            return static_cast<bool>(inputs_dependency & static_cast<uint64_t>(1) << index);
        }
        ge::graphStatus SetInputDataDependency(const size_t index)
        {
            if (index >= sizeof(inputs_dependency) * kByteBitCount) {
                return ge::GRAPH_FAILED;
            }
            inputs_dependency |= 1UL << index;
            return ge::GRAPH_SUCCESS;
        }

        bool HasHostInput() const { return (host_inputs != 0UL); }
        /*
         * param index: must be ir index
         */
        bool IsHostInput(const size_t index) const
        {
            if (index >= (sizeof(host_inputs) * kByteBitCount)) {
                return false;
            }
            return static_cast<bool>(host_inputs & (static_cast<uint64_t>(1) << index));
        }
        ge::graphStatus SetHostInputs(const size_t index)
        {
            if (index >= (sizeof(host_inputs) * kByteBitCount)) {
                return ge::GRAPH_FAILED;
            }
            host_inputs |= 1UL << index;
            return ge::GRAPH_SUCCESS;
        }

        bool HasTilingInputDataDependency() const { return (tiling_dependency != 0UL); }
        /*
         * param index: must be ir index
         */
        bool IsTilingInputDataDependency(const size_t index) const
        {
            if (index >= (sizeof(tiling_dependency) * kByteBitCount)) {
                return false;
            }
            return static_cast<bool>(tiling_dependency & (static_cast<uint64_t>(1) << index));
        }
        ge::graphStatus SetTilingInputDataDependency(const size_t index)
        {
            if (index >= (sizeof(tiling_dependency) * kByteBitCount)) {
                return ge::GRAPH_FAILED;
            }
            tiling_dependency |= 1UL << index;
            return ge::GRAPH_SUCCESS;
        }

        bool IsSupportTilingDependencyPlacement(const uint32_t placement) const
        {
            if (static_cast<size_t>(placement) >= (sizeof(tiling_dependency_placements) * kByteBitCount)) {
                return false;
            }

            return static_cast<bool>(tiling_dependency_placements & (static_cast<uint8_t>(1U) << placement));
        }

        ge::graphStatus SetTilingDependencyPlacement(const uint32_t placement)
        {
            if (static_cast<size_t>(placement) >= (sizeof(tiling_dependency_placements) * kByteBitCount)) {
                return ge::GRAPH_FAILED;
            }
            tiling_dependency_placements |= (static_cast<uint8_t>(1U) << placement);
            return ge::GRAPH_SUCCESS;
        }

        bool IsOutputShapeDependOnCompute() const { return (output_shape_depend_compute != 0UL); }
        /*
         * param index: must be ir index
         */
        bool IsOutputShapeDependOnCompute(const size_t index) const
        {
            if (index >= (sizeof(output_shape_depend_compute) * kByteBitCount)) {
                return false;
            }
            return static_cast<bool>(output_shape_depend_compute & (static_cast<uint64_t>(1) << index));
        }

        ge::graphStatus SetOutputShapeDependOnCompute(const size_t index)
        {
            if (index >= (sizeof(output_shape_depend_compute) * kByteBitCount)) {
                return ge::GRAPH_FAILED;
            }
            output_shape_depend_compute |= 1UL << index;
            return ge::GRAPH_SUCCESS;
        }

        OpImplRegisterV2::InferShapeKernelFunc infer_shape = nullptr;
        OpImplRegisterV2::InferShapeRangeKernelFunc infer_shape_range = nullptr;
        OpImplRegisterV2::InferDataTypeKernelFunc infer_datatype = nullptr;
        OpImplRegisterV2::TilingKernelFunc tiling = nullptr;
        OpImplRegisterV2::KernelFunc tiling_parse = nullptr;
        OpImplRegisterV2::CompileInfoCreatorFunc compile_info_creator = nullptr;
        OpImplRegisterV2::CompileInfoDeleterFunc compile_info_deleter = nullptr;
        size_t max_tiling_data_size = 0UL;
        uint64_t inputs_dependency = 0UL;
        static constexpr size_t kByteBitCount = 8UL;
        OpImplRegisterV2::PrivateAttrList private_attrs;
        // todo 去重和registry没关系，下一步从这里删除，移动到register中实现
        OpImplRegisterV2::PrivateAttrSet unique_private_attrs;
        uint64_t host_inputs = 0UL;
        OpImplRegisterV2::OpExecFunc op_execute_func = nullptr;
        uint64_t tiling_dependency = 0UL;
        OpImplRegisterV2::GenSimplifiedKeyKernelFunc gen_simplifiedkey = nullptr;
        uint8_t tiling_dependency_placements = 0U;
        uint64_t output_shape_depend_compute = 0UL;
    };

    using InferSymbolShapeKernelFunc = UINT32 (*)(InferSymbolShapeContext*);
    struct OpImplFunctionsV2 : OpImplFunctions {
        OpImplFunctionsV2() = default;
        OpImplFunctionsV2(OpImplFunctions& func);
        OpImplFunctionsV2(const OpImplFunctions& func);
        OpImplFunctionsV2& operator=(OpImplFunctions& func);

        InferSymbolShapeKernelFunc infer_symbol_shape = nullptr;
        OpImplRegisterV2::InferFormatFunc infer_format_func = nullptr;

        uint32_t st_size = sizeof(OpImplFunctionsV2);
        uint32_t version = OP_IMPL_MAIN_VERSION;
        OpImplRegisterV2::OpCalcParamKernelFunc calc_op_param = nullptr;
        OpImplRegisterV2::OpGenTaskKernelFunc gen_task = nullptr;
        OpImplRegisterV2::OP_CHECK_FUNC_V2 check_support = nullptr;
        OpImplRegisterV2::OP_CHECK_FUNC_V2 op_select_format = nullptr;
        OpImplRegisterV2::OP_CHECK_FUNC_V2 get_op_support_info = nullptr;
        OpImplRegisterV2::OP_CHECK_FUNC_V2 get_op_specific_info = nullptr;
        OpImplRegisterV2::OpExecPrepareFunc op_execute_prepare_func = nullptr;
        OpImplRegisterV2::OpExecLaunchFunc op_execute_launch_func = nullptr;
        uint64_t reserved_[502] = {0U};
    };
};
} // namespace gert
#endif
