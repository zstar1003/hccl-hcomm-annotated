/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>

#include "gtest/gtest.h"
#include "rma_buffer_mgr.h"

namespace {

using Key = hcomm::BufferKey<uintptr_t, uint64_t>;
using Buffer = std::shared_ptr<int>;
using Manager = hcomm::RmaBufferMgr<Key, Buffer>;

template <typename Manager>
void CheckCommonOperations()
{
    Manager manager;
    Key parentKey(0x1000, 0x100);
    Key childKey(0x1040, 0x20);
    Buffer buffer = std::make_shared<int>(7);

    auto addResult = manager.Add(parentKey, buffer);
    EXPECT_TRUE(addResult.second);
    EXPECT_EQ(manager.size(), 1U);
    EXPECT_TRUE(manager.IsInTree(parentKey));

    auto findResult = manager.Find(childKey);
    ASSERT_TRUE(findResult.first);
    EXPECT_EQ(findResult.second, buffer);

    size_t visited = 0;
    manager.ForEach([&visited, &buffer](const Key&, const Buffer& item) {
        ++visited;
        EXPECT_EQ(item, buffer);
    });
    EXPECT_EQ(visited, 1U);
    EXPECT_EQ(manager.Next(manager.Begin()), manager.End());

    EXPECT_TRUE(manager.Del(parentKey));
    EXPECT_EQ(manager.Begin(), manager.End());
}

TEST(RmaBufferMgrTest, Ut_CommonManager_When_Used_Expect_CommonOperationsAvailable)
{
    CheckCommonOperations<Manager>();
}

} // namespace
