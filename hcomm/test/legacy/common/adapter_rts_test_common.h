/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADAPTER_RTS_TEST_COMMON_H
#define ADAPTER_RTS_TEST_COMMON_H

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

class AdapterRtsTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AdapterRts tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AdapterRts tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in AdapterRts SetUP" << std::endl; }

    virtual void TearDown() { GlobalMockObject::verify(); }
};

#endif // ADAPTER_RTS_TEST_COMMON_H
