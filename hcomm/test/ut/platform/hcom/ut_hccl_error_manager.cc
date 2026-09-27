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
#include <mockcpp/mockcpp.hpp>
#include "adapter_error_manager_pub.h"
#include "adapter_rts.h"

class HcclErrManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcclErrManagerTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "HcclErrManagerTest TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

#if 0

TEST_F(HcclErrManagerTest, ut_hccl_error_manager)
{
    bool ret = false;
    int *ptr = new int(10);
    RPT_INNER_ERR_PRT("remote op nic connect failed, please ensure that collective communication execution status "\
        "of each device is consistent(include network TLS configuration)");
    RPT_CALL_ERR(ret, "aclrtIpcMemImportByKey failed. return[%d], ptr[%p], name[%s]", ret, ptr, "name");
    RPT_CALL_ERR_PRT("ra socket batch connect failed. return[%d]", 1);
    delete ptr;
}

#endif
