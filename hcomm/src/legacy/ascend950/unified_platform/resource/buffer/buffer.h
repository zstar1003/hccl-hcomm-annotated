/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_BUFFER_H
#define HCCL_BUFFER_H

#include <cstdint>
#include <cstddef>
#include <string>
#include "hccl_mem_defs.h"
namespace Hccl {

class Buffer {
public:
    Buffer(uintptr_t addr, std::size_t size);

    explicit Buffer(std::size_t size);

    explicit Buffer(uintptr_t addr, std::size_t size, HcclMemType memType);

    explicit Buffer(uintptr_t addr, std::size_t size, HcclMemType memType, const char* memInfo);

    explicit Buffer(uintptr_t addr, std::size_t size, const char* memInfo);

    virtual ~Buffer() = default;

    uintptr_t GetAddr() const;

    size_t GetSize() const;

    HcclMemType GetMemType() const;

    const std::string GetMemInfo() const;

    virtual std::string Describe() const;

    bool Contains(Buffer* buf) const;

    bool Contains(uintptr_t bufAddr, size_t bufSize) const;

    Buffer Range(std::size_t offset, std::size_t givenSize) const;

    // "bool"运算符(可执行if(object){...}的操作判断该buffer是否为空)
    operator bool() const { return addr_ != 0; }

    // "=="运算符
    bool operator==(const Buffer& that) const { return (addr_ == that.GetAddr()) && (size_ == that.GetSize()); }

    // "!="运算符
    bool operator!=(const Buffer& that) const { return (addr_ != that.GetAddr()) || (size_ != that.GetSize()); }

protected:
    void SetMemInfo(const char* memInfo);

    uintptr_t addr_{0};
    std::size_t size_{0};
    HcclMemType memType_{HcclMemType::HCCL_MEM_TYPE_DEVICE};
    char memInfo_[256]{}; // memTag上限为HCOMM_RES_TAG_MAX_LEN(255)，此处引用不到该宏，暂用魔数，待后续修改
};

} // namespace Hccl
#endif // HCCL_BUFFER_H
