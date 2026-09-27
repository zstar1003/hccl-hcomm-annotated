/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_EVENT_HPP
#define CCU_EVENT_HPP

#include <cstdint>
#include <type_traits>

#include "ccu_types.h"
#include "ccu_primitives_impl.h"
#include "ccu_utils.hpp"

namespace AscendC {
namespace ccu {

    template <typename U>
    class Array;

    class Event final {
    public:
        Event() { CCU_THROW_IF_FAILED(CcuEventAlloc(&this->handle), "CcuEventAlloc: failed"); }

        explicit Event(CcuEventHandle acqHandle, uint32_t index = 0)
        {
            CCU_THROW_IF_FAILED(CcuEventGetByIndex(acqHandle, index, &this->handle), "CcuEventGetByIndex: failed");
        }

        Event(const Event& other) : handle(other.handle) {}

        Event(Event&& other) noexcept : handle(other.handle) {}

        void operator=(Event&& other) { this->handle = other.handle; }

        CcuEventHandle handle{0};

    private:
        explicit Event(detail::NoAllocTag) {}
        template <typename U>
        friend class Array;
    };

} // namespace ccu
} // namespace AscendC

#endif // CCU_EVENT_HPP
