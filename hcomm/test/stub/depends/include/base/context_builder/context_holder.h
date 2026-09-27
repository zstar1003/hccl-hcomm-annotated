/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_CONTEXT_HOLDER_H_
#define METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_CONTEXT_HOLDER_H_
#include <memory>
#include <vector>
namespace gert {
class ContextHolderImpl;
class ContextHolderVoid {
public:
    ContextHolderVoid();
    ~ContextHolderVoid();
    ContextHolderVoid(ContextHolderVoid&& other) noexcept;
    ContextHolderVoid& operator=(ContextHolderVoid&& other) noexcept;
    void* GetContext() const;

private:
    friend class ContextHolderBuilder;
    std::unique_ptr<ContextHolderImpl> ctx_holder_impl_;
};

/**
 * Builder系列接口最终生成的对象，用来管理context相关资源
 */
template <typename ContextTypeT>
class ContextHolder {
public:
    ContextHolder() = default;
    explicit ContextHolder(ContextHolderVoid&& holder_void) : holder_void_(std::move(holder_void)) {}
    /**
     * @brief 按指定类型获取ctx指针
     * @return ContextTypeT类型的指针
     */
    ContextTypeT* GetContext() { return static_cast<ContextTypeT*>(holder_void_.GetContext()); }

private:
    ContextHolderVoid holder_void_;
};
} // namespace gert
#endif // METADEF_INC_EXTERNAL_BASE_CONTEXT_BUILDER_CONTEXT_HOLDER_H_
