/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_STATUS_H
#define TRANSPORT_STATUS_H

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>

namespace Hccl {

// MAKE_ENUM(TransportStatus, INIT, SOCKET_OK, SOCKET_TIMEOUT, READY)

class TransportStatus {
public:
    // 对应宏里的: enum Value : uint8_t { __VA_ARGS__, __COUNT__, INVALID };
    enum Value : uint8_t {
        INIT,
        SOCKET_OK,
        SOCKET_TIMEOUT,
        CONNECT_FAILED,
        READY,
        __COUNT__, // 宏自动生成的计数器
        INVALID    // 宏自动生成的无效值
    };

    // 默认构造函数
    TransportStatus() {}

    // 允许通过 TransportStatus::INIT 等方式构造
    constexpr TransportStatus(Value v) : value(v) {}

    // 类型转换操作符，允许将对象隐式转换为内部枚举值 (用于 switch 等场景)
    constexpr operator Value() const { return value; }

    // ================== 运算符重载 (原样保留) ==================

    constexpr bool operator==(TransportStatus a) const { return value == a.value; }

    constexpr bool operator!=(TransportStatus a) const { return value != a.value; }

    constexpr bool operator<(TransportStatus a) const { return value < a.value; }

    // 针对内部 enum Value 的比较重载
    constexpr bool operator==(Value v) const { return value == v; }

    constexpr bool operator!=(Value v) const { return value != v; }

    constexpr bool operator<(Value v) const { return value < v; }

    // ================== 描述功能 ==================

    // 返回枚举的字符串描述
    // 原宏通过解析字符串实现，这里直接静态定义，性能更好且更直观
    std::string Describe() const
    {
        static const std::vector<std::string> m
            = {"TransportStatus::INIT", "TransportStatus::SOCKET_OK", "TransportStatus::SOCKET_TIMEOUT",
               "TransportStatus::CONNECT_FAILED", "TransportStatus::READY"};

        // 边界检查：如果值超出了定义的字符串范围（例如是 __COUNT__ 或 INVALID）
        // 原宏中的逻辑是 if (value > m.size())，这里修正为 >= 以防止越界崩溃
        if (value >= m.size()) {
            return "TransportStatus::Invalid";
        }
        return m[value];
    }

    // 重载输出流操作符，支持 std::cout << status
    friend std::ostream& operator<<(std::ostream& stream, const TransportStatus& v) { return stream << v.Describe(); }

private:
    // 对应宏里的 private: Value value{INVALID};
    Value value{INVALID};
};

} // namespace Hccl

#endif // TRANSPORT_STATUS_H
