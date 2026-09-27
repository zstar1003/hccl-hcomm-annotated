/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include "slog.h"
#include "slog_api.h"
#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

using namespace Hccl;
class LogTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LogTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "LogTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in LogTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in LogTest TearDown" << std::endl;
    }
};

TEST_F(LogTest, ut_sal_log_printf_log2)
{
    MODULE_DEBUG("\r\r\r\r \r\r\r\r log 2.0 test"); /*提高覆盖率*/
    MODULE_INFO("log 2.0 test");
    MODULE_WARNING("log 2.0 test");
    MODULE_ERROR("log 2.0 test");
    MODULE_RUN_INFO("<START AllReduce>");
    MODULE_RUN_INFO("<END AllReduce>");
}

// 代码覆盖率 log2.0格式对齐
TEST_F(LogTest, ut_sal_log_printf_log2_destruct)
{
    MODULE_DEBUG(" \r log 2.0 test \n ");
    MODULE_INFO("log 2.0 test");
    MODULE_WARNING("log 2.0 test");
    MODULE_ERROR("log 2.0 test");
    MODULE_RUN_INFO("<START AllReduce>");
}

TEST_F(LogTest, ut_sal_log_printf_error_0)
{
    LOG_PRINT(6 | RUN_LOG_MASK, "test \\n info \n");
    MODULE_DEBUG(nullptr);
    MODULE_INFO(nullptr);
    MODULE_WARNING(nullptr);
    MODULE_ERROR(nullptr);
    MODULE_RUN_INFO(nullptr);
    CallDlogInvalidType(HCCL_LOG_RUN_INFO, 1, "test", 2);
    CallDlogNoSzFormat(HCCL_LOG_RUN_INFO, 1, "test", 2);
    CallDlogMemError(HCCL_LOG_RUN_INFO, "test", 2);
    CallDlogPrintError(HCCL_LOG_RUN_INFO, "test", 2);
    CallDlog(HCCL_LOG_RUN_INFO, 1, "test", "test", 2);
    CallDlogInvalidType(HCCL_LOG_OPLOG, 1, "test", 2);
    CallDlogNoSzFormat(HCCL_LOG_OPLOG, 1, "test", 2);
    CallDlogMemError(HCCL_LOG_OPLOG, "test", 2);
    CallDlogPrintError(HCCL_LOG_OPLOG, "test", 2);
    CallDlog(HCCL_LOG_OPLOG, 1, "test", "test", 2);
}
