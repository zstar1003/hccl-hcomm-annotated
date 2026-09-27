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

using Key = hccl::BufferKey<uintptr_t, uint64_t>;
using Buffer = std::shared_ptr<int>;
using Manager = hcomm::RmaBufferMgr<Key, Buffer>;

TEST(RmaBufferMgrLogTest, Ut_OverlapBuffer_When_Add_Expect_ErrorLogPrinted)
{
    Manager manager;
    Key parentKey(0x1000, 0x100);
    Key overlapKey(0x1080, 0x100);
    Buffer buffer = std::make_shared<int>(7);

    ASSERT_TRUE(manager.Add(parentKey, buffer).second);

    testing::internal::CaptureStdout();
    auto result = manager.Add(overlapKey, buffer);
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_FALSE(result.second);
    EXPECT_NE(output.find("Error: Buffer key overlaps with existing buffer key."), std::string::npos);
}

TEST(RmaBufferMgrLogTest, Ut_NonHcclScope_When_Log_Expect_ErrorLogPrinted)
{
    testing::internal::CaptureStdout();
    HCCL_ERROR("950 log macro outside Hccl namespace");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_NE(output.find("950 log macro outside Hccl namespace"), std::string::npos);
}

} // namespace
