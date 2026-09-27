/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "llt_hccl_stub.h"
#include "register/op_impl_registry.h"
#include "register/op_impl_kernel_registry.h"
#include "base/err_mgr.h"
#include <stack>
#include <securec.h>
#include <cstdint>
#include "graph/ge_error_codes.h"
#include "base/context_builder/op_tiling_parse_context_builder.h"
#include "base/context_builder/op_tiling_context_builder.h"
#include "exe_graph/runtime/storage_shape.h"

void DlogErrorInner(int module_id, const char* fmt, ...) { return; }

int32_t DlogSetAttr(LogAttr logAttrInfo) { return 0; }
void DlogWarnInner(int module_id, const char* fmt, ...) { return; }

void DlogInfoInner(int module_id, const char* fmt, ...) { return; }

void DlogDebugInner(int module_id, const char* fmt, ...) { return; }

void DlogEventInner(int module_id, const char* fmt, ...) { return; }

namespace ge {
bool AscendString::operator<(const AscendString& d) const
{
    if ((name_ == nullptr) && (d.name_ == nullptr)) {
        return false;
    } else if (name_ == nullptr) {
        return true;
    } else if (d.name_ == nullptr) {
        return false;
    } else {
        return (*name_) < (*(d.name_));
    }
}

bool AscendString::operator>(const AscendString& d) const
{
    if ((name_ == nullptr) && (d.name_ == nullptr)) {
        return false;
    } else if (name_ == nullptr) {
        return false;
    } else if (d.name_ == nullptr) {
        return true;
    } else {
        return (*name_) > (*(d.name_));
    }
}

bool AscendString::operator==(const AscendString& d) const
{
    if ((name_ == nullptr) && (d.name_ == nullptr)) {
        return true;
    } else if (name_ == nullptr) {
        return false;
    } else if (d.name_ == nullptr) {
        return false;
    } else {
        return (*name_) == (*(d.name_));
    }
}

bool AscendString::operator<=(const AscendString& d) const
{
    if (name_ == nullptr) {
        return true;
    } else if (d.name_ == nullptr) {
        return false;
    } else {
        return (*name_) <= (*(d.name_));
    }
}

bool AscendString::operator>=(const AscendString& d) const
{
    if (d.name_ == nullptr) {
        return true;
    } else if (name_ == nullptr) {
        return false;
    } else {
        return (*name_) >= (*(d.name_));
    }
}

bool AscendString::operator!=(const AscendString& d) const
{
    if ((name_ == nullptr) && (d.name_ == nullptr)) {
        return false;
    } else if (name_ == nullptr) {
        return true;
    } else if (d.name_ == nullptr) {
        return true;
    } else {
        return (*name_) != (*(d.name_));
    }
}

AscendString::AscendString(const char_t* const name) {}
} // namespace ge

namespace gert {
template <typename T>
struct ComGraphMakeUniq {
    using unique_object = std::unique_ptr<T>;
};

template <typename T>
struct ComGraphMakeUniq<T[]> {
    using unique_array = std::unique_ptr<T[]>;
};

template <typename T, size_t B>
struct ComGraphMakeUniq<T[B]> {
    struct invalid_type {};
};

template <typename T, typename... Args>
static inline typename ComGraphMakeUniq<T>::unique_object ComGraphMakeUnique(Args&&... args)
{
    using T_nc = typename std::remove_const<T>::type;
    return std::unique_ptr<T>(new (std::nothrow) T_nc(std::forward<Args>(args)...));
}

template <typename T>
static inline typename ComGraphMakeUniq<T>::unique_array ComGraphMakeUnique(const size_t num)
{
    return std::unique_ptr<T>(new (std::nothrow) typename std::remove_extent<T>::type[num]());
}

template <typename T, typename... Args>
static inline typename ComGraphMakeUniq<T>::invalid_type ComGraphMakeUnique(Args&&...) = delete;

namespace {
    uint32_t GeMemcpy(uint8_t* dst_ptr, size_t dst_size, const uint8_t* src_ptr, const size_t src_size) { return 0; }
} // namespace

class ContextHolderImpl {
public:
    ContextHolderImpl() {};
    ~ContextHolderImpl() {};
};

ContextHolderVoid::ContextHolderVoid() {}
ContextHolderVoid::~ContextHolderVoid() {}
ContextHolderVoid::ContextHolderVoid(ContextHolderVoid&& other) noexcept {}
ContextHolderVoid& ContextHolderVoid::operator=(ContextHolderVoid&& other) noexcept { return *this; }
void* ContextHolderVoid::GetContext() const
{
    int i = 0;
    return &i;
}

class OpImplRegisterV2Impl {
public:
    bool is_private_attr_registered = false;
};

OpImplRegisterV2::OpImplRegisterV2(const ge::char_t* op_type) {}
OpImplRegisterV2::OpImplRegisterV2(const OpImplRegisterV2& register_data) {};
OpImplRegisterV2::~OpImplRegisterV2() {}

OpImplRegisterV2& OpImplRegisterV2::Tiling(TilingKernelFunc tiling_func, size_t max_tiling_data_size) { return *this; }

OpImplRegisterV2& OpImplRegisterV2::TilingParse(
    KernelFunc tiling_parse_func, CompileInfoCreatorFunc creator_func, CompileInfoDeleterFunc deleter_func)
{
    return *this;
}

class ContextBuilderImpl {
public:
    ContextBuilderImpl() {};
    ~ContextBuilderImpl() {};
};

template class OpContextBuilderBase<OpTilingContextBuilder>;
template class OpContextBuilderBase<OpTilingParseContextBuilder>;

template <typename T>
OpContextBuilderBase<T>::OpContextBuilderBase() : impl_()
{}
template <typename T>
OpContextBuilderBase<T>::~OpContextBuilderBase() = default;

template <typename T>
T& OpContextBuilderBase<T>::OpType(const ge::AscendString& op_type)
{
    return static_cast<T&>(*this);
}

template <typename T>
T& OpContextBuilderBase<T>::OpName(const ge::AscendString& op_name)
{
    return static_cast<T&>(*this);
}

template <typename T>
T& OpContextBuilderBase<T>::IONum(size_t input_ir_num, size_t output_ir_num)
{
    return static_cast<T&>(*this);
}

OpTilingParseContextBuilder::OpTilingParseContextBuilder() {}
OpTilingParseContextBuilder::~OpTilingParseContextBuilder() {};

OpTilingParseContextBuilder& OpTilingParseContextBuilder::InputTensorDesc(
    size_t index, ge::DataType dtype, ge::Format origin_format, ge::Format storage_format,
    const gert::ExpandDimsType& expand_dims_type)
{
    return *this;
}

OpTilingParseContextBuilder& OpTilingParseContextBuilder::OutputTensorDesc(
    size_t index, ge::DataType dtype, ge::Format origin_format, ge::Format storage_format,
    const gert::ExpandDimsType& expand_dims_type)
{
    return *this;
}

OpTilingParseContextBuilder& OpTilingParseContextBuilder::CompiledJson(const ge::char_t* compiled_json)
{
    return *this;
}

OpTilingParseContextBuilder& OpTilingParseContextBuilder::CompiledInfo(const void* compile_info) { return *this; }

OpTilingParseContextBuilder& OpTilingParseContextBuilder::PlatformInfo(const void* platform_info) { return *this; }

ContextHolder<TilingParseContext> OpTilingParseContextBuilder::Build() { return ContextHolder<TilingParseContext>(); }

OpTilingContextBuilder::OpTilingContextBuilder() {}
OpTilingContextBuilder::~OpTilingContextBuilder() {};

OpTilingContextBuilder& OpTilingContextBuilder::CompileInfo(const void* compile_info) { return *this; }

OpTilingContextBuilder& OpTilingContextBuilder::PlatformInfo(const void* platform_info) { return *this; }
OpTilingContextBuilder& OpTilingContextBuilder::Deterministic(int32_t deterministic) { return *this; }

OpTilingContextBuilder&
OpTilingContextBuilder::TilingData(const gert::TilingData* tiling_data, gert::Chain::Deleter deleter)
{
    return *this;
}

OpTilingContextBuilder& OpTilingContextBuilder::TilingDataSize(size_t tiling_data_size) { return *this; }

OpTilingContextBuilder& OpTilingContextBuilder::Workspace(const gert::ContinuousVector* workspace) { return *this; }

OpTilingContextBuilder& OpTilingContextBuilder::InputTensors(const std::vector<gert::Tensor*>& inputs) { return *this; }

OpTilingContextBuilder& OpTilingContextBuilder::OutputTensors(const std::vector<gert::Tensor*>& outputs)
{
    return *this;
}

ContextHolder<TilingContext> OpTilingContextBuilder::Build() { return ContextHolder<TilingContext>(); }

ge::graphStatus TilingData::AppendConvertedAttrVal(
    const RuntimeAttrs* attrs, const size_t attr_index, const AttrDataType src_type, const AttrDataType dst_type)
{
    return ge::GRAPH_SUCCESS;
}

const Shape g_vec_1_shape;
int64_t GetInputCDim(TilingContext* kernel_context, const size_t index) { return 0; }

namespace bg {

    bool AppendAttr(const ge::AnyValue& attr, std::vector<std::vector<uint8_t>>& attrs) { return true; }
} // namespace bg
} // namespace gert
