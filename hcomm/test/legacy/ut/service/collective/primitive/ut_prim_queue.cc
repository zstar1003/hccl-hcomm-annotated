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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "primitive.h"
#include "prim_queue.h"

using namespace Hccl;
using namespace std;
class PrimQueueTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PrimitiveTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "PrimitiveTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        master = make_shared<PrimQueue>();
        slave = master->Fork();
        std::cout << "A Test case in PrimQueueTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in PrimQueueTest TearDown" << std::endl;
    }

    shared_ptr<PrimQueue> master;
    shared_ptr<PrimQueue> slave;
};

TEST_F(PrimQueueTest, test_append_post_to)
{
    unique_ptr<PrimPostTo> ptr = make_unique<PrimPostTo>(slave);

    EXPECT_EQ(INVALID_PRIM_QID, ptr->GetParentQid());

    master->Append(std::move(ptr));

    EXPECT_EQ(1, master->Size());

    auto iter = master->Iter();
    cout << iter->Describe() << endl;

    const PrimPostTo& postTo = static_cast<const PrimPostTo&>(*iter);

    EXPECT_EQ(master->GetId(), postTo.GetParentQid());
}

TEST_F(PrimQueueTest, test_append_wait_from)
{
    unique_ptr<PrimWaitFrom> ptr = make_unique<PrimWaitFrom>(slave);

    EXPECT_EQ(INVALID_PRIM_QID, ptr->GetParentQid());

    master->Append(std::move(ptr));

    EXPECT_EQ(1, master->Size());

    auto iter = master->Iter();
    cout << iter->Describe() << endl;

    const PrimWaitFrom& waitFrom = static_cast<const PrimWaitFrom&>(*iter);

    EXPECT_EQ(master->GetId(), waitFrom.GetParentQid());
}

TEST_F(PrimQueueTest, test_append_wait_group)
{
    unique_ptr<PrimWaitGroup> ptr = make_unique<PrimWaitGroup>(0);

    EXPECT_EQ(INVALID_PRIM_QID, ptr->GetParentQid());

    master->Append(std::move(ptr));

    EXPECT_EQ(1, master->Size());

    auto iter = master->Iter();
    cout << iter->Describe() << endl;

    const PrimWaitGroup& waitGroup = static_cast<const PrimWaitGroup&>(*iter);

    EXPECT_EQ(master->GetId(), waitGroup.GetParentQid());
}
