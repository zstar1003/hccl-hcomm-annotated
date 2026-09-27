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
#include "null_ptr_exception.h"
#define private public
#include "rma_conn_manager.h"
#include "dev_ub_connection.h"
#include "communicator_impl.h"
#undef private
#include "internal_exception.h"
#include "rma_conn_lite.h"
using namespace Hccl;

class RmaConnLiteMgrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RmaConnLiteMgrTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "RmaConnLiteMgrTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in RmaConnLiteMgrTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RmaConnLiteMgrTest TearDown" << std::endl;
    }
};
