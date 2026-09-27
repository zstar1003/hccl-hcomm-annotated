/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_TILING_PARSE_CONTEXT_BUILDER_H_
#define METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_TILING_PARSE_CONTEXT_BUILDER_H_
#include "base/context_builder/context_holder.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "base/context_builder/op_context_builder_base.h"

namespace gert {
/**
 * @brief OpTilingParseContextBuilder类，用于构造TilingParseContext.
 * @note  OpTilingParseContextBuilder类的实例化对象用于构造算子tiling的执行上下文。
 */
class OpTilingParseContextBuilder : public OpContextBuilderBase<OpTilingParseContextBuilder> {
public:
    OpTilingParseContextBuilder();
    ~OpTilingParseContextBuilder() override;
    /**
     * @brief 设置第index个实例输入的Tensor Description信息,
     *        用于构造TilingParseContext的基类ExtendedKernelContext中的ComputeNodeInfo等信息
     * @param index 输入的索引，对应的是Op IR原型中的的输入实例Instance索引
     * @param dtype 输入Tensor的data type
     * @param origin_format 输入Tensor的原始格式
     * @param storage_format 输入Tensor的存储格式
     * @param expand_dims_type 输入Tensor的ExpandDimsType，默认值为{}
     * @return OpTilingParseContextBuilder对象引用，用于链式调用
     */
    OpTilingParseContextBuilder& InputTensorDesc(
        size_t index, ge::DataType dtype, ge::Format origin_format, ge::Format storage_format,
        const gert::ExpandDimsType& expand_dims_type = {});
    /**
     * @brief 设置第index个实例输出的Tensor Description信息,
     *        用于构造TilingParseContext的基类ExtendedKernelContext中的ComputeNodeInfo等信息
     * @param index 输出的索引，对应的是Op IR原型中的的输出实例Instance索引
     * @param dtype 输出Tensor的data type
     * @param origin_format 输出Tensor的原始格式
     * @param storage_format 输出Tensor的存储格式
     * @param expand_dims_type 输出Tensor的ExpandDimsType，默认值为{}
     * @return OpTilingParseContextBuilder对象引用，用于链式调用
     */
    OpTilingParseContextBuilder& OutputTensorDesc(
        size_t index, ge::DataType dtype, ge::Format origin_format, ge::Format storage_format,
        const gert::ExpandDimsType& expand_dims_type = {});
    /**
     * @brief 设置Op的compileJson指针，json格式文件指针, 用于构造通过TilingParseContext获取的的compiled_json字段
     * @note
     * 设置的所有输入数据类型，所有权归调用者管理，调用者需要保证输入指针生命周期指针长于Build产生的ContextHolder对象
     * @param compiled_json 编译信息json文件指针
     * @return OpTilingParseContextBuilder对象用于链式调用
     */
    OpTilingParseContextBuilder& CompiledJson(const ge::char_t* compiled_json);
    /**
     * @brief 设置Op的CompiledInfo指针, 用于构造TilingParseContext中的CompiledInfo字段
     * @note
     * 设置的所有输入数据类型，所有权归调用者管理，调用者需要保证输入指针生命周期指针长于Build产生的ContextHolder对象
     * @param compile_info 编译信息指针
     * @return OpTilingParseContextBuilder对象用于链式调用
     */
    OpTilingParseContextBuilder& CompiledInfo(const void* compile_info);
    /**
     * @brief 设置Op的PlatFormInfo指针, 用于构造TilingParseContext的PlatformInfo字段
     * @note
     * 设置的所有输入数据类型，所有权归调用者管理，调用者需要保证输入指针生命周期指针长于Build产生的ContextHolder对象
     * @param platform_info 平台信息指针
     * @return OpTilingParseContextBuilder对象用于链式调用
     */
    OpTilingParseContextBuilder& PlatformInfo(const void* platform_info);
    /**
     * @brief 构建TilingParseContext对象
     * @return ContextHolder对象，包含TilingParseContext指针,
     * @note 注意返回的ContextHolder对象的生命周期需要大于等于TilingParseContext对象的生命周期，
     *       才能保证通过TilingParseContext获取的所有指针的有效性
     */
    ContextHolder<TilingParseContext> Build();
};
} // namespace gert
#endif // METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_TILING_PARSE_CONTEXT_BUILDER_H_
