/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "hash_utils.h"

class HashTestObject {
public:
    const int propertiesInt;
    const std::string propertiesString;

    HashTestObject(const int param1, const std::string param2) : propertiesInt(param1), propertiesString(param2) {};
};

namespace std {
template <>
class hash<HashTestObject> {
public:
    size_t operator()(const HashTestObject& hashTestObject) const
    {
        return Hccl::HashCombine(
            {hash<int>{}(hashTestObject.propertiesInt), hash<string>{}(hashTestObject.propertiesString)});
    }
};
} // namespace std

TEST(hash_combine_test, should_return_the_same_code_when_the_properties_are_same)
{
    HashTestObject first(1, "test");
    HashTestObject second(1, "test");

    EXPECT_NE(&first, &second);
    EXPECT_EQ(std::hash<HashTestObject>{}(first), std::hash<HashTestObject>{}(second));
}

TEST(hash_combine_test, should_return_different_code_when_the_properties_are_change)
{
    HashTestObject first(1, "test");
    HashTestObject second(22, "test");

    EXPECT_NE(&first, &second);
    EXPECT_NE(std::hash<HashTestObject>{}(first), std::hash<HashTestObject>{}(second));
}
