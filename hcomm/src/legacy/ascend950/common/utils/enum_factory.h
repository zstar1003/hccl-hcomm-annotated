/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_ENUM_FACTORY_H
#define HCCLV2_ENUM_FACTORY_H

#include <cstdint>
#include <sstream>
#include <string>

namespace Hccl {
namespace EnumNameDetail {
    struct NameView {
        const char* data;
        unsigned int size;
    };

    inline const char* SkipSpacesAndCommas(const char* cursor)
    {
        while ((*cursor == ' ') || (*cursor == ',')) {
            ++cursor;
        }
        return cursor;
    }

    // "SDMA = 0" -> name is "SDMA"; stop at space or '='.
    inline const char* FindEnumeratorNameEnd(const char* cursor)
    {
        while ((*cursor != '\0') && (*cursor != ',') && (*cursor != ' ') && (*cursor != '=')) {
            ++cursor;
        }
        return cursor;
    }

    // Skip the initializer ("= 0") up to the comma between enumerators.
    inline const char* SkipUntilComma(const char* cursor)
    {
        while ((*cursor != '\0') && (*cursor != ',')) {
            ++cursor;
        }
        return cursor;
    }

    // Split the stringized enumerator list on commas. Pointers stay in that literal.
    inline unsigned int ParseNames(const char* enumeratorList, NameView* enumNames, unsigned int maxEnumCount)
    {
        unsigned int curEnumCount = 0;
        const char* cursor = enumeratorList;
        while (*cursor != '\0') {
            cursor = SkipSpacesAndCommas(cursor);
            if (*cursor == '\0') {
                break;
            }

            const char* nameBegin = cursor;
            const char* nameEnd = FindEnumeratorNameEnd(cursor);
            if ((nameBegin != nameEnd) && (curEnumCount < maxEnumCount)) {
                enumNames[curEnumCount].data = nameBegin;
                enumNames[curEnumCount].size = static_cast<unsigned int>(nameEnd - nameBegin);
                ++curEnumCount;
            }

            cursor = SkipUntilComma(nameEnd);
        }
        return curEnumCount;
    }
} // namespace EnumNameDetail
} // namespace Hccl

#define MAKE_ENUM(enumClass, ...)                                                                                    \
    class enumClass {                                                                                                \
    public:                                                                                                          \
        enum Value : uint8_t { __VA_ARGS__, __COUNT__, INVALID };                                                    \
                                                                                                                     \
        enumClass() {}                                                                                               \
                                                                                                                     \
        constexpr enumClass(Value v) : value(v) {}                                                                   \
                                                                                                                     \
        constexpr operator Value() const { return value; }                                                           \
                                                                                                                     \
        constexpr bool operator==(enumClass a) const { return value == a.value; }                                    \
                                                                                                                     \
        constexpr bool operator!=(enumClass a) const { return value != a.value; }                                    \
                                                                                                                     \
        constexpr bool operator<(enumClass a) const { return value < a.value; }                                      \
                                                                                                                     \
        constexpr bool operator==(Value v) const { return value == v; }                                              \
                                                                                                                     \
        constexpr bool operator!=(Value v) const { return value != v; }                                              \
                                                                                                                     \
        constexpr bool operator<(Value v) const { return value < v; }                                                \
                                                                                                                     \
        std::string Describe() const                                                                                 \
        {                                                                                                            \
            /* POD table: first call parses #__VA_ARGS__; views point at the literal, no heap. */                    \
            static ::Hccl::EnumNameDetail::NameView enumNames[__COUNT__];                                            \
            static const unsigned int enumCount                                                                      \
                = ::Hccl::EnumNameDetail::ParseNames(#__VA_ARGS__, enumNames, static_cast<unsigned int>(__COUNT__)); \
            const unsigned int enumValue = static_cast<unsigned int>(value);                                         \
            if (enumValue >= enumCount) {                                                                            \
                return std::string(#enumClass) + "::Invalid";                                                        \
            }                                                                                                        \
            const ::Hccl::EnumNameDetail::NameView& enumName = enumNames[enumValue];                                 \
            return std::string(#enumClass) + "::" + std::string(enumName.data, enumName.size);                       \
        }                                                                                                            \
                                                                                                                     \
        friend std::ostream& operator<<(std::ostream& stream, const enumClass& v) { return stream << v.Describe(); } \
                                                                                                                     \
    private:                                                                                                         \
        Value value{INVALID};                                                                                        \
    };

namespace std {
struct EnumClassHash {
    template <typename T>
    std::size_t operator()(T t) const
    {
        return static_cast<std::size_t>(t);
    }
};
} // namespace std

#endif // HCCLV2_ENUM_FACTORY_H
