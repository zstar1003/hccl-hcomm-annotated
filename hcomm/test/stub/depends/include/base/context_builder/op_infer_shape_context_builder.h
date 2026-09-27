/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_INFER_SHAPE_CTX_BUILDER_H_
#define METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_INFER_SHAPE_CTX_BUILDER_H_

#include <vector>
#include "base/context_builder/op_context_builder_base.h"
#include "base/context_builder/context_holder.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace gert {
/**
 * @brief OpInferShapeContextBuilder类，用于构造InferShapeContext.
 * @note  OpInferShapeContextBuilder类的实例化对象用于构造算子形状推导的执行上下文。
 */
class OpInferShapeContextBuilder : public OpContextBuilderBase<OpInferShapeContextBuilder> {
public:
    OpInferShapeContextBuilder();
    ~OpInferShapeContextBuilder() override;

    /**
     * @brief 用作构造InferShapeContext时Op输出的Tensor Description信息,用于构造
     *        InferShapeContext的基类ExtendedKernelContext中的ComputeNodeInfo等信息
     * @param index 输出的索引，对应的是Op IR原型中的的输出实例Instance索引
     * @param dtype 输出Tensor的数据类型
     * @param origin_format 输出Tensor的原始格式
     * @param storage_format 输出Tensor的存储格式
     * @param expand_dims_type 输出Tensor的ExpandDimsType
     * @return OpInferShapeContextBuilder对象用于链式调用
     */
    OpInferShapeContextBuilder& OutputTensorDesc(
        size_t index, ge::DataType dtype, ge::Format origin_format, ge::Format storage_format,
        const gert::ExpandDimsType& expand_dims_type = {});

    /**
     * @brief
     * 设置输入Tensor指针，用于在shape推导时，可通过该builder类构造的上下文InferShapeContext获取相应的输入tensor指针
     * @note
     * 对于数据依赖的算子，对应数据依赖的输入Tensor中的TensorData是需要有Host地址的正确值；对于非数据依赖算子，Tensor的TensorData为空指针
     * @param inputs
     * 输入指针数组，所有权归调用者管理，调用者需要保证输入指针生命周期指针长于Build产生的ContextHolder对象
     * @return OpInferShapeContextBuilder对象用于链式调用
     */
    OpInferShapeContextBuilder& InputTensors(const std::vector<gert::Tensor*>& inputs);

    /**
     * @brief 构建InferShapeContext对象
     * @return 返回一个ContextHolder对象，包含InferShapeContext指针,
     *  注意返回的ContextHolder对象的生命周期需要大于等于InferShapeContext对象的生命周期，才能保证通过InferShapeContext获取的所有指针的有效性
     */
    ContextHolder<InferShapeContext> Build();
};
} // namespace gert
#endif // METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_INFER_SHAPE_CTX_BUILDER_H_
