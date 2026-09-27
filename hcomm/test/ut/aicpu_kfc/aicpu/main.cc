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
#include "gtest/gtest.h"
#include "comm.h"

GTEST_API_ int main(int argc, char** argv)
{
    printf("Running main() from gtest_main.cc\n");
    testing::InitGoogleTest(&argc, argv);
    setenv(
        "LD_LIBRARY_PATH",
        "./llt/ace/comop/hccl/aicpu_kfc/stub/workspace/fwkacllib/lib64/:"
        "./llt/ace/comop/hccl/aicpu_kfc/stub/workspace/hccl/lib64/",
        1);
    return RUN_ALL_TESTS();
}
