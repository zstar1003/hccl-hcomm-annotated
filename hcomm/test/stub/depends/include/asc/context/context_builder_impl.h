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
 * \file context_builder_impl.h
 * \brief implementation for context_builder.h
 */

#ifndef CONTEXT_BUILDER_IMPL_H
#define CONTEXT_BUILDER_IMPL_H

#include <memory>
#include <vector>
#include "context_builder.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/context_extend.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/kernel_context.h"
#include "base/context_builder/op_kernel_run_context_builder.h"
#include "base/context_builder/op_tiling_context_builder.h"

namespace context_ascendc {
enum class HolderType : uint8_t { KERNEL_RUN_CTX = 0, TILING_CTX };

class InputHolder;
class ValueHolderImpl {
public:
    ValueHolderImpl(
        gert::ContextHolder<gert::TilingContext>&& ctxHolder,
        std::vector<std::unique_ptr<uint8_t[]>>&& inputTensorHolder)
        : type_(HolderType::TILING_CTX),
          inputTensorHolder_(std::move(inputTensorHolder)),
          ctxTilingHolder_(std::move(ctxHolder)),
          ctxRunHolder_(gert::ContextHolder<gert::KernelContext>())
    {}
    ValueHolderImpl(gert::ContextHolder<gert::KernelContext>&& ctxHolder)
        : type_(HolderType::KERNEL_RUN_CTX),
          inputTensorHolder_(std::vector<std::unique_ptr<uint8_t[]>>()),
          ctxTilingHolder_(gert::ContextHolder<gert::TilingContext>()),
          ctxRunHolder_(std::move(ctxHolder))
    {}
    gert::ComputeNodeInfo* MutableComputeNodeInfo()
    {
        if (type_ == HolderType::KERNEL_RUN_CTX) {
            auto kernelCtx = ctxRunHolder_.GetContext();
            return reinterpret_cast<gert::ComputeNodeInfo*>(const_cast<void*>(kernelCtx->GetComputeNodeExtend()));
        } else if (type_ == HolderType::TILING_CTX) {
            auto tilingCtx = ctxTilingHolder_.GetContext();
            return const_cast<gert::ComputeNodeInfo*>(tilingCtx->GetComputeNodeInfo());
        }
        return nullptr;
    }
    ValueHolderImpl() = default;
    ~ValueHolderImpl() = default;

private:
    HolderType type_;
    std::vector<std::unique_ptr<uint8_t[]>> inputTensorHolder_;
    gert::ContextHolder<gert::TilingContext> ctxTilingHolder_;
    gert::ContextHolder<gert::KernelContext> ctxRunHolder_;
};
class ContextBuilderImpl {
public:
    ContextBuilderImpl();
    ~ContextBuilderImpl() = default;

    // kernel context builder
    void Inputs(std::vector<void*> inputs);
    void Outputs(std::vector<void*> outputs);
    // tiling context builder
    void NodeIoNum(size_t inputNum, size_t outputNum);
    void IrInstanceNum(std::vector<uint32_t> instanceNum);
    void SetOpNameType(const std::string& opName, const std::string& opType);
    void AddInputTd(
        int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat,
        gert::StorageShape storageShape);
    void AddOutputTd(
        int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat,
        gert::StorageShape storageShape);
    void AddInputTd(
        int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat,
        gert::StorageShape storageShape, void* constValues);
    void AddInputTd(
        int32_t index, ge::DataType dtype, ge::Format originFormat, ge::Format storageFormat,
        gert::StorageShape storageShape, const std::string& filePath);
    void AddAttr(const std::string& attrName, int64_t attrValue);
    void AddAttr(const std::string& attrName, bool attrValue);
    void AddAttr(const std::string& attrName, const std::string& attrValue);
    void AddAttr(const std::string& attrName, float attrValue);
    void AddAttr(const std::string& attrName, const std::vector<float>& attrValue);
    void AddAttr(const std::string& attrName, const std::vector<bool>& attrValue);
    void AddAttr(const std::string& attrName, const std::vector<int64_t>& attrValue);
    void AddAttr(const std::string& attrName, const std::vector<std::string>& attrValue);
    void AddAttr(const std::string& attrName, const std::vector<std::vector<int64_t>>& attrValue);

    void CompileInfo(void* compileInfo);
    void PlatformInfo(void* platformInfo);
    void TilingData(void* tilingData);
    void Workspace(gert::ContinuousVector* workspace);

    std::shared_ptr<KernelRunContextHolder> BuildKernelRunContext();
    std::shared_ptr<KernelRunContextHolder> BuildTilingContext();

    bool errFlag_{false};

private:
    std::unique_ptr<gert::OpKernelContextBuilder> kernelCtxBuilder_;
    std::unique_ptr<gert::OpTilingContextBuilder> tilingCtxBuilder_;
    std::unordered_map<int32_t, std::unique_ptr<uint8_t[]>> dependTensorsData_;
    std::unordered_map<int32_t, std::unique_ptr<uint8_t[]>> dependOutputTensorsData_;
    size_t inputNum_ = 0;
    size_t outputNum_ = 0;
};
namespace DataUtils {
    bool ReadBinFile(const std::string& fileName, void* buf, std::size_t bufferLen);
    uint16_t FloatToUint16(const float value);
    uint16_t FloatToBF16(const ge::float32_t value);
    int64_t GetTensorSizeByStorageShape(const gert::StorageShape& storageShape, const ge::DataType& dtype);
    bool
    SetConstDataWithFloat16(void* rawData, int64_t bufferLen, int64_t holderSize, std::unique_ptr<uint8_t[]>& dstData);
    bool
    SetConstDataWithBF16(void* rawData, int64_t bufferLen, int64_t holderSize, std::unique_ptr<uint8_t[]>& dstData);
    template <typename T>
    bool SetConstData(void* rawData, int64_t bufferLen, int64_t holderSize, std::unique_ptr<uint8_t[]>& dstData);
} // namespace DataUtils
} // namespace context_ascendc
#endif
