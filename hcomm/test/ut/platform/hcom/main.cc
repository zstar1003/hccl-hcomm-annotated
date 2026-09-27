/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <gtest/gtest.h>

#include "comm.h"
#include "../../stub/llt_hccl_stub_pub.h"

GTEST_API_ int main(int argc, char** argv)
{
    printf("Running main() from gtest_main.cc\n");
    setTargetPort(16452, 31114);
    testing::InitGoogleTest(&argc, argv);
    setenv("HCCL_DFS_CONFIG", "connection_fault_detection_time:0", 1);
    return RUN_ALL_TESTS();
}
