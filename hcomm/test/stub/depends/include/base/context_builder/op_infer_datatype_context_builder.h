/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_INFER_DTYPE_CTX_BUILDER_H_
#define METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_INFER_DTYPE_CTX_BUILDER_H_

#include <vector>
#include "graph/types.h"
#include "base/context_builder/context_holder.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "base/context_builder/op_context_builder_base.h"

namespace gert {
/**
 * @brief OpInferDataTypeContextBuilder类，用于构造InferDataTypeContext.
 * @note  OpInferDataTypeContextBuilder类的实例化对象用于构造算子数据类型推导的执行上下文。
 */
class OpInferDataTypeContextBuilder : public OpContextBuilderBase<OpInferDataTypeContextBuilder> {
public:
    OpInferDataTypeContextBuilder();
    ~OpInferDataTypeContextBuilder() override;

    /**
     * @brief 设置第index个实例输入的Tensor Description信息,
     *        用于构造InferDataTypeContext的基类ExtendedKernelContext中的ComputeNodeInfo信息
     * @param index 输入的索引，对应的是Op IR原型中的的输入实例Instance索引
     * @param dtype 输入Tensor的data type
     * @param origin_format 输入Tensor的原始格式
     * @param storage_format 输入Tensor的存储格式
     * @param expand_dims_type 输入Tensor的ExpandDimsType，默认值为{}
     * @return OpInferDataTypeContextBuilder对象引用，用于链式调用
     */
    OpInferDataTypeContextBuilder& InputTensorDesc(
        size_t index, ge::DataType dtype, ge::Format origin_format, ge::Format storage_format,
        const gert::ExpandDimsType& expand_dims_type = {});

    /**
     * @brief 设置第index个实例输出的Tensor Description信息,
     *        用于构造InferDataTypeContext的基类ExtendedKernelContext中的ComputeNodeInfo信息,
     *        无需设置输出data type信息，输出data type由算子实现类根据输入DataType计算推导得到
     * @param index 输出的索引，对应的是Op IR原型中的的输出实例Instance索引
     * @param origin_format 输出Tensor的原始格式
     * @param storage_format 输出Tensor的存储格式
     * @param expand_dims_type 输出Tensor的ExpandDimsType，默认值为{}
     * @return OpInferDataTypeContextBuilder对象引用，用于链式调用
     */
    OpInferDataTypeContextBuilder& OutputTensorDesc(
        size_t index, ge::Format origin_format, ge::Format storage_format,
        const gert::ExpandDimsType& expand_dims_type = {});
    /**
     * @brief 构建InferDataTypeContext对象
     * @return 返回一个ContextHolder对象，包含InferDataTypeContext指针,
     *         注意返回的ContextHolder对象的生命周期需要大于等于InferDataTypeContext对象的生命周期，
     *         才能保证通过InferDataTypeContext获取的所有指针的有效性
     */
    ContextHolder<InferDataTypeContext> Build();
};
} // namespace gert
#endif // METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_INFER_DTYPE_CTX_BUILDER_H_
