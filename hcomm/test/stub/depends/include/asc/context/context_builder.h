/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file context_builder.h
 * \brief Api to build tiling context
 */

#ifndef CONTEXT_BUILDER_H
#define CONTEXT_BUILDER_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <cstring>
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/kernel_run_context.h"
#include "exe_graph/runtime/context_extend.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"
#include "base/context_builder/op_kernel_run_context_builder.h"
#include "base/context_builder/op_tiling_context_builder.h"

namespace context_ascendc {
class ContextBuilderImpl;
class ValueHolderImpl;
using TilingFunc = uint32_t (*)(gert::TilingContext*);
class OpTilingRegistryImpl;
class ContextBuilder;
class OpTilingRegistry {
public:
    OpTilingRegistry() = default;
    ~OpTilingRegistry() = default;
    TilingFunc GetTilingFunc(const char* opType) const;
    bool LoadTilingLibrary(const char* tilingSoPath) const;

private:
    std::shared_ptr<OpTilingRegistryImpl> impl_;
};

struct KernelRunContextHolder {
    KernelRunContextHolder();
    ~KernelRunContextHolder();
    template <typename T>
    T* GetContext() const
    {
        return reinterpret_cast<T*>(kernelContext);
    }
    gert::ComputeNodeInfo* MutableComputeNodeInfo();

    std::unique_ptr<ValueHolderImpl> valueHolder;
    std::unique_ptr<uint8_t[]> computeNodeExtendHolder;

    KernelRunContext* context{nullptr};

private:
    friend class ContextBuilder;
    friend class ContextBuilderImpl;
    gert::KernelContext* kernelContext{nullptr};
    KernelRunContextHolder(
        gert::ContextHolder<gert::KernelContext>&& ctxHolder, gert::KernelContext* kernelContext,
        KernelRunContext* context);
    KernelRunContextHolder(
        gert::ContextHolder<gert::TilingContext>&& ctxHolder,
        std::vector<std::unique_ptr<uint8_t[]>>&& inputTensorHolder, gert::KernelContext* kernelContext);
};

class ContextBuilder {
public:
    ContextBuilder();
    ~ContextBuilder();
    ContextBuilder(ContextBuilder&& kernelRunContextBuilder) = delete;
    ContextBuilder& operator=(ContextBuilder&& kernelRunContextBuilder) = delete;

    // kernel context builder
    ContextBuilder& Inputs(std::vector<void*> inputs);
    ContextBuilder& Outputs(std::vector<void*> outputs);
    std::shared_ptr<KernelRunContextHolder> BuildKernelRunContext();

    // OpInfo
    ContextBuilder& NodeIoNum(size_t inputNum, size_t outputNum);
    ContextBuilder& SetOpNameType(const std::string& opName, const std::string& opType);
    ContextBuilder& IrInstanceNum(std::vector<uint32_t> instanceNum);
    ContextBuilder& AddInputTd(
        int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat,
        gert::StorageShape storageShape);
    ContextBuilder& AddOutputTd(
        int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat,
        gert::StorageShape storageShape);
    ContextBuilder& AddInputTd(
        int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat,
        gert::StorageShape storageShape, void* constValues);
    ContextBuilder& AddInputTd(
        int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat,
        gert::StorageShape storageShape, const std::string& filePath);
    ContextBuilder& AddAttr(const std::string& attrName, int64_t attrValue);
    ContextBuilder& AddAttr(const std::string& attrName, bool attrValue);
    ContextBuilder& AddAttr(const std::string& attrName, const std::string& attrValue);
    ContextBuilder& AddAttr(const std::string& attrName, float attrValue);
    ContextBuilder& AddAttr(const std::string& attrName, const std::vector<float>& attrValue);
    ContextBuilder& AddAttr(const std::string& attrName, const std::vector<bool>& attrValue);
    ContextBuilder& AddAttr(const std::string& attrName, const std::vector<int64_t>& attrValue);
    ContextBuilder& AddAttr(const std::string& attrName, const std::vector<std::string>& attrValue);
    ContextBuilder& AddAttr(const std::string& attrName, const std::vector<std::vector<int64_t>>& attrValue);

    // Tiling Context Builder
    ContextBuilder& CompileInfo(void* compileInfo);
    ContextBuilder& PlatformInfo(void* platformInfo);
    ContextBuilder& AddPlatformInfo(const char* customSocVersion);
    ContextBuilder& TilingData(void* tilingData);
    ContextBuilder& Workspace(gert::ContinuousVector* workspace);
    std::shared_ptr<KernelRunContextHolder> BuildTilingContext();

private:
    std::unique_ptr<ContextBuilderImpl> impl_;
};
} // namespace context_ascendc
#endif
