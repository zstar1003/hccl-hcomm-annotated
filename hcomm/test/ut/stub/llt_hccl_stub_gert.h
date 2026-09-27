/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __LLT_HCCL_STUB_GERT_H__
#define __LLT_HCCL_STUB_GERT_H__

#include "graph/ge_error_codes.h"
#include "graph/types.h"
#include "graph/compiler_def.h"
#include "register/op_impl_kernel_registry.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include <initializer_list>
#include <string>
#include <map>

#include "exe_graph/runtime/tensor_data_utils.h"

namespace gert {
namespace bg {
    class GenerateExeGraph {
    public:
        struct ExeGraphGenerator {
            using InferShapeFunc
                = std::vector<ValueHolderPtr> (*)(const ge::NodePtr& node, const std::vector<ValueHolderPtr>& shapes);
            using AllocOutputMemoryFunc = std::vector<DevMemValueHolderPtr> (*)(
                TensorPlacement placement, const ge::NodePtr& node, const std::vector<ValueHolderPtr>& output_sizes,
                LoweringGlobalData& global_data);
            using CalcTensorSizeFunc = std::vector<ValueHolderPtr> (*)(
                const ge::NodePtr& node, const std::vector<ValueHolderPtr>& output_shapes);

            InferShapeFunc infer_shape;
            AllocOutputMemoryFunc alloc_output_memory;
            CalcTensorSizeFunc calc_tensor_size;
        };

    public:
        static std::vector<ValueHolderPtr>
        InferShape(const ge::NodePtr& node, const std::vector<ValueHolderPtr>& shapes)
        {
            if (generator_.infer_shape == nullptr) {
                return {};
            }
            return generator_.infer_shape(node, shapes);
        }
        static std::vector<DevMemValueHolderPtr> AllocOutputMemory(
            TensorPlacement placement, const ge::NodePtr& node, const std::vector<ValueHolderPtr>& output_sizes,
            LoweringGlobalData& global_data)
        {
            if (generator_.alloc_output_memory == nullptr) {
                return {};
            }
            return generator_.alloc_output_memory(placement, node, output_sizes, global_data);
        }
        static std::vector<ValueHolderPtr>
        CalcTensorSize(const ge::NodePtr& node, const std::vector<ValueHolderPtr>& output_shapes)
        {
            std::vector<ValueHolderPtr> holders;
            size_t outputSize = output_shapes.size();
            ValueHolderPtr outputHolder;
            for (size_t i = 0; i < outputSize; i++) {
                holders.push_back(outputHolder);
            }
            return holders;
        }

        static void AddBuilderImplement(ExeGraphGenerator generator) { generator_ = generator; }

    private:
        static ExeGraphGenerator generator_;
    };
} // namespace bg
} // namespace gert
#endif // __LLT_HCCL_STUB_GERT_H__
