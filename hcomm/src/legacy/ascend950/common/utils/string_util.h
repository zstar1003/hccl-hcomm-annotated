/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_STRING_UTIL_H
#define HCCLV2_STRING_UTIL_H

#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include "securec.h"
#include "log.h"

namespace Hccl {

template <typename... Args>
inline std::string StringFormat(const char* format, Args... args)
{
    using namespace std;
    constexpr size_t bufSize = BUFSIZ;
    char buffer[bufSize];
    int result = snprintf_s(&buffer[0], bufSize, bufSize, format, args...);
    if (result < 0) {
        HCCL_ERROR("[StringFormat] data snprintf_s failed.");
        return "";
    }
    size_t actualSize = static_cast<size_t>(result);
    if (actualSize + 1 > bufSize) {
        actualSize++;
        std::vector<char> newbuffer(actualSize);
        auto ret = snprintf_s(newbuffer.data(), actualSize, actualSize, format, args...);
        if (ret != EOK) {
            HCCL_ERROR("[StringFormat] data snprintf_s failed.");
            return "";
        }
        return newbuffer.data();
    }
    return buffer;
}

template <typename I>
std::string Dec2hex(I i)
{
    static_assert(std::is_integral<I>::value, "type I is not a integral");
    std::stringstream ss;
    ss << std::hex << "0x" << i;
    return ss.str();
}

inline std::vector<std::string> SplitString(const std::string& str, const char c)
{
    std::string::size_type startPos = 0;
    std::string::size_type foundPos = str.find(c);

    std::vector<std::string> strVector;
    while (foundPos != std::string::npos) {
        strVector.push_back(str.substr(startPos, foundPos - startPos));
        startPos = foundPos + 1;
        foundPos = str.find(c, startPos);
    }
    if (startPos != str.length()) {
        strVector.push_back(str.substr(startPos));
    }
    return strVector;
}

template <class T>
T String2T(const std::string& s)
{
    // T must support >>
    T t;
    std::istringstream ist(s);
    ist >> t;
    return t;
}

inline std::string Bytes2hex(const void* data, int size)
{
    constexpr int hexWidth = 2;
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    const auto* tmp = static_cast<const unsigned char*>(data);
    for (int i = 0; i < size; i++) {
        ss << std::setw(hexWidth) << static_cast<int>(*tmp);
        tmp++;
    }
    return ss.str();
}

} // namespace Hccl

#endif // HCCLV2_STRING_UTIL_H
