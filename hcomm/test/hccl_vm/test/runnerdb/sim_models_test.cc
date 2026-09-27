/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "sim_models.h"

using namespace sim;

class SimModelsTest : public testing::Test {
protected:
};

TEST_F(SimModelsTest, Server_StructSize)
{
    static_assert(sizeof(Server) > 0, "Server struct should have size");
    Server s{1, 100, 0, "v1"};
    EXPECT_EQ(s.id, 1);
    EXPECT_EQ(s.pod_id, 100);
}

TEST_F(SimModelsTest, Device_StructSize)
{
    static_assert(sizeof(Device) > 0, "Device struct should have size");
    Device d{1, 100, 255, 0, 0, 1, 0, 0};
    EXPECT_EQ(d.id, 1);
    EXPECT_EQ(d.server_id, 100);
}

TEST_F(SimModelsTest, Runner_StructSize)
{
    static_assert(sizeof(Runner) > 0, "Runner struct should have size");
    Runner r{1, 100, 1, 1, 1000, 1};
    EXPECT_EQ(r.id, 1);
}
