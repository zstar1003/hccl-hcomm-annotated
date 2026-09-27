/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_OP_CONTEXT_BUILDER_BASE_H_
#define METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_OP_CONTEXT_BUILDER_BASE_H_
#include <memory>
#include "graph/types.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/shape.h"
#include "graph/ascend_string.h"
#include "exe_graph/runtime//expand_dims_type.h"
namespace gert {
class ContextBuilderImpl;
/**
 * @brief OpContextBuilderBase基类，用于构造Op
 * 子类context中算子信息，包括算子类型、名称、输入输出原型个数、输入输出实例个数、属性等信息。
 *        注意：不可单独构造OpContextBuilderBase基类对象，只能通过子类构造
 * @param T 子类类型，用于返回子类对象的引用，用于支持子类链式调用
 */
template <typename T>
class OpContextBuilderBase {
public:
    /**
     * @brief 设置OpType，用作构造各子类context的基础ComputeNodeInfo信息
     * @param op_type Op的类型
     * @return 返回子类对象T类型的引用，用于支持子类链式调用
     */
    T& OpType(const ge::AscendString& op_type);
    /**
     * @brief 设置OpName，用作构造各子类context的基础ComputeNodeInfo信息
     * @param op_name Op的名称
     * @return 返回子类对象T类型的引用，用于支持子类链式调用
     */
    T& OpName(const ge::AscendString& op_name);
    /**
     * @brief 设置Op输入输出IR原型个数，用作构造各子类context的基础ComputeNodeInfo信息,
     * 默认每个IR原型输入输出的实例个数为1
     * @param input_ir_num 输入IR原型个数
     * @param output_ir_num 输出IR原型个数
     * @attention 此接口与IOInstanceNum接口互斥。仅需调用2种接口的一种即可。
     * @return 返回子类对象T类型的引用，用于支持子类链式调用
     */
    T& IONum(size_t input_ir_num, size_t output_ir_num);
    /**
     * @brief 当输入IR原型实例个数不为1时(一般是可选输入或动态输入场景)，需要设置Op每个输入IR原型的实例个数，
     *        用作构造各子类context的基础ComputeNodeInfo信息
     * @note 当算子存在dynamic input类型输入时，对应input的instance_num需设置为大于1的值
     * @param input_instance_num 每个IR原型输入的实例个数
     * @param output_instance_num 每个IR原型输出的实例个数
     * @attention 此接口与IONum接口互斥。仅需调用2种接口的一种即可。
     * @return 返回子类对象T类型的引用，用于支持子类链式调用
     */
    T& IOInstanceNum(const std::vector<uint32_t>& input_instance_num, const std::vector<uint32_t>& output_instance_num);

    /**
     * @brief 往后追加Op
     * IR原型的属性信息，下标从0开始，用作构造各子类context的基础ExtendedInfo里通过GetAttr接口获取到的的RuntimeAttr属性信息
     * @note
     * 请注意，往后追加的属性，获取到的属性是一个有序列表，属性构造的顺序与通过Context的基类接口GetAttr获取到的RuntimeAttrs中属性的顺序一致.
     *       例如：context_builder.AppendAttr(bool attr0).AppendAttr(int64_t attr1).AppendAttr(vector<int64_t>
     * attr2)，则 ctx->GetAttrs()->GetBool(0) -> attr0, ctx->GetAttrs()->GetInt(1) -> attr1,
     *              ctx->GetAttrs()->GetListInt(2) -> attr2
     * @param attr 属性值，当前仅支持以下确定的几种类型：bool、int64_t、float、AscendString、std::vector<bool>、
     *             std::vector<int64_t>、std::vector<float>、std::vector<AscendString>、std::vector<std::vector<int64_t>>
     * @return 返回子类对象T类型的引用，用于支持子类链式调用
     */
    T& AppendAttr(bool attr);
    T& AppendAttr(int64_t attr);
    T& AppendAttr(float attr);
    T& AppendAttr(const ge::AscendString& attr);
    T& AppendAttr(const std::vector<bool>& attr);
    T& AppendAttr(const std::vector<int64_t>& attr);
    T& AppendAttr(const std::vector<float>& attr);
    T& AppendAttr(const std::vector<ge::AscendString>& attr);
    T& AppendAttr(const std::vector<std::vector<int64_t>>& attr);

    // 禁止拷贝和赋值
    OpContextBuilderBase(const OpContextBuilderBase&) = delete;
    OpContextBuilderBase& operator=(const OpContextBuilderBase&) = delete;
    // 禁止移动构造和移动赋值
    OpContextBuilderBase(OpContextBuilderBase&&) = delete;
    OpContextBuilderBase& operator=(OpContextBuilderBase&&) = delete;

    virtual ~OpContextBuilderBase();

protected:
    [[nodiscard]] ge::DataType& MutableInputDataType(size_t index);
    [[nodiscard]] ge::Format& MutableInputOriginalFormat(size_t index);
    [[nodiscard]] ge::Format& MutableInputStorageFormat(size_t index);
    [[nodiscard]] gert::ExpandDimsType& MutableInputExpandDimsType(size_t index);

    [[nodiscard]] ge::DataType& MutableOutputDataType(size_t index);
    [[nodiscard]] ge::Format& MutableOutputOriginalFormat(size_t index);
    [[nodiscard]] ge::Format& MutableOutputStorageFormat(size_t index);
    [[nodiscard]] gert::ExpandDimsType& MutableOutputExpandDimsType(size_t index);
    std::unique_ptr<ContextBuilderImpl> impl_;

    OpContextBuilderBase();
};
} // namespace gert
#endif // METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_OP_CONTEXT_BUILDER_BASE_H_
