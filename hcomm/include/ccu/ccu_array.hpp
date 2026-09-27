/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_ARRAY_HPP
#define CCU_ARRAY_HPP

#include <cstdint>
#include <new>
#include <vector>

#include "ccu_types.h"
#include "ccu_primitives_impl.h"
#include "ccu_utils.hpp"
#include "ccu_variable.hpp"
#include "ccu_event.hpp"
#include "ccu_buffer.hpp"

namespace AscendC {
namespace ccu {

    template <typename T>
    struct CcuArrayTraits;

    template <>
    struct CcuArrayTraits<Variable> {
        using Handle = CcuVariableHandle;
        static CcuResult BlockAlloc(Handle* h, uint32_t n) { return CcuBlockVariableAlloc(h, n); }
        static CcuResult CreateByAcquire(Handle acqHandle, uint32_t index, Handle* h)
        {
            return CcuVariableGetByIndex(acqHandle, index, h);
        }
        static void SetHandle(Variable& v, Handle h) { v.handle = h; }
    };

    template <>
    struct CcuArrayTraits<Event> {
        using Handle = CcuEventHandle;
        static CcuResult BlockAlloc(Handle* h, uint32_t n) { return CcuBlockEventAlloc(h, n); }
        static CcuResult CreateByAcquire(Handle acqHandle, uint32_t index, Handle* h)
        {
            return CcuEventGetByIndex(acqHandle, index, h);
        }
        static void SetHandle(Event& e, Handle h) { e.handle = h; }
    };

    template <>
    struct CcuArrayTraits<CcuBuffer> {
        using Handle = CcuBufferHandle;
        static CcuResult BlockAlloc(Handle* h, uint32_t n) { return CcuBlockBufferAlloc(h, n); }
        static void SetHandle(CcuBuffer& b, Handle h) { b.handle = h; }
    };

    template <typename T>
    class Array final {
    public:
        explicit Array(uint32_t count) : count_(count)
        {
            if (count == 0) {
                return;
            }
            using H = typename CcuArrayTraits<T>::Handle;
            std::vector<H> handles(count);
            elems_ = static_cast<T*>(::operator new(sizeof(T) * count));
            for (uint32_t i = 0; i < count; ++i) {
                ::new (static_cast<void*>(&elems_[i])) T(detail::NoAllocTag{});
            }
            auto ret = CcuArrayTraits<T>::BlockAlloc(handles.data(), count);
            if (ret != CcuResult::CCU_SUCCESS) {
                for (uint32_t i = 0; i < count; ++i) {
                    elems_[i].~T();
                }
                ::operator delete(elems_);
                elems_ = nullptr;
                count_ = 0;
                throw ::AscendC::ccu::detail::CcuException(ret, "Array BlockAlloc: failed");
            }
            for (uint32_t i = 0; i < count; ++i) {
                CcuArrayTraits<T>::SetHandle(elems_[i], handles[i]);
            }
        }

        Array(typename CcuArrayTraits<T>::Handle acqHandle, uint32_t count) : count_(count)
        {
            if (count == 0) {
                return;
            }
            using H = typename CcuArrayTraits<T>::Handle;
            elems_ = static_cast<T*>(::operator new(sizeof(T) * count));
            for (uint32_t i = 0; i < count; ++i) {
                ::new (static_cast<void*>(&elems_[i])) T(detail::NoAllocTag{});
            }
            for (uint32_t i = 0; i < count; ++i) {
                H handle{};
                auto ret = CcuArrayTraits<T>::CreateByAcquire(acqHandle, i, &handle);
                if (ret != CcuResult::CCU_SUCCESS) {
                    std::string errMsg = "Array creation failed at index " + std::to_string(i)
                                         + ", requested count=" + std::to_string(count)
                                         + "; the acquire handle likely holds fewer resources than count";
                    for (uint32_t j = 0; j < count; ++j) {
                        elems_[j].~T();
                    }
                    ::operator delete(elems_);
                    elems_ = nullptr;
                    count_ = 0;
                    throw ::AscendC::ccu::detail::CcuException(ret, errMsg.c_str());
                }
                CcuArrayTraits<T>::SetHandle(elems_[i], handle);
            }
        }

        ~Array()
        {
            if (elems_ == nullptr) {
                return;
            }
            for (uint32_t i = 0; i < count_; ++i) {
                elems_[i].~T();
            }
            ::operator delete(elems_);
        }

        Array(const Array&) = delete;
        Array& operator=(const Array&) = delete;

        Array(Array&& other) noexcept : elems_(other.elems_), count_(other.count_)
        {
            other.elems_ = nullptr;
            other.count_ = 0;
        }

        Array& operator=(Array&& other) noexcept
        {
            if (this != &other) {
                this->~Array();
                elems_ = other.elems_;
                count_ = other.count_;
                other.elems_ = nullptr;
                other.count_ = 0;
            }
            return *this;
        }

        T& operator[](uint32_t i) { return elems_[i]; }
        const T& operator[](uint32_t i) const { return elems_[i]; }
        T* data() { return elems_; }
        const T* data() const { return elems_; }
        uint32_t size() const { return count_; }

    private:
        T* elems_{nullptr};
        uint32_t count_{0};
    };

} // namespace ccu
} // namespace AscendC

#endif // CCU_ARRAY_HPP
