/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_KERNEL_CONTEXT_BUILDER_H_
#define METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_KERNEL_CONTEXT_BUILDER_H_
#include <memory>
#include <vector>
#include "base/context_builder/context_holder.h"
#include "exe_graph/runtime/kernel_context.h"
#include "base/context_builder/op_context_builder_base.h"
#include "exe_graph//runtime/expand_dims_type.h"

namespace gert {
/**
 * @brief OpKernelContextBuilder类，用于构造KernelContext.
 * @note OpKernelContextBuilder类的实例化对象用于构造算子host上执行相关交付件的上下文。
 */
class OpKernelContextBuilder : public OpContextBuilderBase<OpKernelContextBuilder> {
public:
    OpKernelContextBuilder();
    ~OpKernelContextBuilder() override;
    /**
     * @brief 设置第index个实例输入的Tensor Description信息,用于构造
     *        KernelContext的基类ExtendedKernelContext中的ComputeNodeInfo信息
     * @param index 输入的索引，对应的是Op IR原型中的的输入实例Instance索引
     * @param dtype 输入Tensor的data type
     * @param origin_format 输入Tensor的原始格式
     * @param storage_format 输入Tensor的存储格式
     * @param expand_dims_type 输入Tensor的ExpandDimsType，默认值为{}
     * @return OpKernelContextBuilder对象引用，用于链式调用
     */
    OpKernelContextBuilder& InputTensorDesc(
        size_t index, ge::DataType dtype, ge::Format origin_format, ge::Format storage_format,
        const gert::ExpandDimsType& expand_dims_type = {});
    /**
     * @brief 设置第index个实例输出的Tensor Description信息,用于构造
     *        KernelContext的基类ExtendedKernelContext中的ComputeNodeInfo信息
     * @param index 输出的索引，对应的是Op IR原型中的的输出实例Instance索引
     * @param dtype 输出Tensor的data type
     * @param origin_format 输出Tensor的原始格式
     * @param storage_format 输出Tensor的存储格式
     * @param expand_dims_type 输出Tensor的ExpandDimsType，默认值为{}
     * @return OpKernelContextBuilder对象引用，用于链式调用
     */
    OpKernelContextBuilder& OutputTensorDesc(
        size_t index, ge::DataType dtype, ge::Format origin_format, ge::Format storage_format,
        const gert::ExpandDimsType& expand_dims_type = {});

    /**
     * @brief 设置context的values的输入指针，values承载的类型为void*的指针数组
     * @param inputs
     * 输入指针数组，所有权归调用者管理，调用者需要保证输入指针生命周期指针长于Build产生的ContextHolder对象
     * @return OpKernelContextBuilder对象引用，用于链式调用
     */
    OpKernelContextBuilder& Inputs(std::vector<void*> inputs);

    /**
     * @brief 设置context的values的输出指针，values承载的类型为void*的指针数组
     * @note
     * 设置的所有输入数据类型，所有权归调用者管理，调用者需要保证输入指针生命周期指针长于Build产生的ContextHolder对象
     * @param outputs 输出指针数组
     * @return OpKernelContextBuilder对象引用，用于链式调用
     */
    OpKernelContextBuilder& Outputs(std::vector<void*> outputs);

    /**
     * @brief 构建KernelRunContext对象
     * @return 返回一个ContextHolder对象，包含KernelRunContext指针,
     * @note 注意返回的ContextHolder对象的生命周期需要大于等于KernelContext对象的生命周期，
     *       才能保证通过KernelContext获取的所有指针的有效性
     */
    ContextHolder<KernelContext> Build();
};
} // namespace gert
#endif // METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_KERNEL_CONTEXT_BUILDER_H_
