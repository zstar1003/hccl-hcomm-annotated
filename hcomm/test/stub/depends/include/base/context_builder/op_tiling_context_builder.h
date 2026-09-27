/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_TILING_CONTEXT_BUILDER_H_
#define METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_TILING_CONTEXT_BUILDER_H_
#include <memory>
#include <vector>
#include "base/context_builder/context_holder.h"
#include "exe_graph/runtime/tiling_context.h"
#include "base/context_builder/op_context_builder_base.h"

namespace gert {
class TilingData;
/**
 * @brief OpTilingContextBuilder类，用于构造TilingContext.
 * @note  OpTilingContextBuilder类的实例化对象用于构造算子tiling的执行上下文。
 */
class OpTilingContextBuilder : public OpContextBuilderBase<OpTilingContextBuilder> {
public:
    OpTilingContextBuilder();
    ~OpTilingContextBuilder() override;

    /**
     * @brief 设置Op的compileInfo指针
     * @note
     * 设置的所有输入数据类型，所有权归调用者管理，调用者需要保证输入指针生命周期指针长于Build产生的ContextHolder对象
     * @param compile_info 编译信息指针
     * @return TilingContextBuilder对象用于链式调用
     */
    OpTilingContextBuilder& CompileInfo(const void* compile_info);
    /**
     * @brief 设置Op的平台信息
     * @note
     * 设置的所有输入数据类型，所有权归调用者管理，调用者需要保证输入指针生命周期指针长于Build产生的ContextHolder对象
     * @param platform_info 平台信息指针
     * @return TilingContextBuilder对象用于链式调用
     */
    OpTilingContextBuilder& PlatformInfo(const void* platform_info);

    /**
     * @brief 设置Op的deterministic标志
     * @note
     * 设置的所有输入数据类型，所有权归调用者管理，调用者需要保证输入指针生命周期指针长于Build产生的ContextHolder对象
     * @param deterministic 是否为确定性计算，当前只支持两种配置，0：未开启确定性配置选项。1：开启确定性配置选项。
     * @return TilingContextBuilder对象用于链式调用
     */
    OpTilingContextBuilder& Deterministic(int32_t deterministic);
    /**
     * @brief 设置Op的tilingData指针, 用于构造TilingContext的中的TilingData字段
     * @note
     * 设置的所有输入数据类型，所有权归调用者管理，调用者需要保证输入指针生命周期指针长于Build产生的ContextHolder对象
     * @param tiling_data tiling数据指针
     * @param deleter
     * tiling数据的删除器，如果用户显示传入删除器，ContextHolder析构时会调用删除器释放tiling数据，默认无需传入，
     *       建议使用TilingDataSize接口
     * @return TilingContextBuilder对象用于链式调用
     */
    OpTilingContextBuilder& TilingData(const gert::TilingData* tiling_data, gert::Chain::Deleter deleter = nullptr);

    /**
     * @brief 设置Op的tilingData的大小, 用于构造TilingContext的中的TilingData字段，
     *        相较于TilingData而言，调用此接口时生成的TilingData指针所有权归属ContextHolder，调用者无需关注TilingData的生命周期
     * @note
     * 注意该接口与TilingData互斥，如果同时调用TilingDataSize和TilingData接口，后调用的会覆盖前一次调用的结果，以最新的为准
     * @param tiling_data_size tiling数据大小
     * @return TilingContextBuilder对象用于链式调用
     */
    OpTilingContextBuilder& TilingDataSize(size_t tiling_data_size);
    /**
     * @brief 设置Op的所有workspace内存指针
     * @note
     * 设置的所有输入数据类型，所有权归调用者管理，调用者需要保证输入指针生命周期指针长于Build产生的ContextHolder对象
     * @param workspace workspace内存指针
     * @return OpTilingContextBuilder对象用于链式调用
     */
    OpTilingContextBuilder& Workspace(const gert::ContinuousVector* workspace);
    /**
     * @brief 设置输入Tensor指针，用于在tiling计算时，可通过该builder类构造的上下文TilingContext获取相应的输入tensor指针
     * @note  对于数据依赖的算子，对应数据依赖的输入Tensor中的TensorData是需要有Host地址的正确值；
     *        对于非数据依赖算子，输入Tensor的TensorData为空指针
     * @param inputs
     * 输入指针数组，所有权归调用者管理，调用者需要保证输入指针生命周期指针长于Build产生的ContextHolder对象
     * @return OpTilingContextBuilder对象用于链式调用
     */
    OpTilingContextBuilder& InputTensors(const std::vector<gert::Tensor*>& inputs);
    /**
     * @brief 设置输出Tensor指针，用于在tiling计算时，可通过该builder类构造的上下文TilingContext获取相应的输出tensor指针
     * @param outputs
     * 输出Tensor指针数组，所有权归调用者管理，调用者需要保证输出指针生命周期指针长于Build产生的ContextHolder对象
     * @return OpTilingContextBuilder对象用于链式调用
     */
    OpTilingContextBuilder& OutputTensors(const std::vector<gert::Tensor*>& outputs);

    /**
     * @brief 构建TilingContext对象
     * @return ContextHolder对象，包含TilingContext指针,
     * @note   注意返回的ContextHolder对象的生命周期需要大于等于TilingContext对象的生命周期，
     *        才能保证通过TilingContext获取的所有指针的有效性
     */
    ContextHolder<TilingContext> Build();
};
} // namespace gert
#endif // METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_TILING_CONTEXT_BUILDER_H_
