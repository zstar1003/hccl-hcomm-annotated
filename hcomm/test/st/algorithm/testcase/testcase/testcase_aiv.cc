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
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include "stream_pub.h"
#define private public
#define protected public
#include "topo_matcher.h"
#include "coll_alg_operator.h"
#include "coll_native_executor_base.h"
#include "ccl_buffer_manager.h"
#include "alg_configurator.h"
#include "hccl_aiv.h"
#include "topoinfo_struct.h"
#include "log.h"
#include "checker_def.h"
#include "topo_meta.h"
#include "testcase_utils.h"
#include "checker.h"
#include "checker_def.h"
#include "dispatcher.h"
#undef protected
#undef private
using namespace checker;

using namespace hccl;
constexpr u32 MAX_BIN_FILE_SIZE = 100 * 1024 * 1024;

class AlgAivTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AlgAivTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "AlgAivTest TearDown" << std::endl; }
};

void TestConstructParam(AivOpArgs& opArgs, AivTopoArgs& topoArgs, AivAlgArgs& algArgs, AivResourceArgs& resourceArgs)
{
    opArgs.cmdType = HcclCMDType::HCCL_CMD_ALLTOALL;
    opArgs.isOpBase = true;
    opArgs.dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    opArgs.count = 100;
    opArgs.op = HcclReduceOp::HCCL_REDUCE_SUM;
    opArgs.input = nullptr;
    opArgs.output = nullptr;
    opArgs.root = 1;
    topoArgs.rank = 1;
    topoArgs.rankSize = 8;
    topoArgs.serverNum = 2;
    topoArgs.devType = DevType::DEV_TYPE_910_93;
    algArgs.step = -1;
    algArgs.isSmallCount = true;
    resourceArgs.stream = nullptr;
    u64 bufferSize = 1024;
    resourceArgs.bufferSize = bufferSize;
}

TEST_F(AlgAivTest, executeKernelLaunch_test)
{
    AivOpArgs opArgs;
    AivTopoArgs topoArgs(1, 8);
    AivAlgArgs algArgs;
    std::string commTag = "exampleTag";
    AivResourceArgs resourceArgs{commTag};
    auto v1 = std::vector<u64>(16, 1);
    resourceArgs.buffersIn = reinterpret_cast<void**>(v1.data());
    resourceArgs.buffersOut = reinterpret_cast<void**>(v1.data());
    s32 tag = 1000;
    void* args;
    u32 argsSize = 0;
    AivProfilingInfo aivProInfo;
    ExtraArgs exArgs;
    ExtraArgsV2 exArgsV2;
    uint64_t beginTime = 0;
    SetAivProfilingInfoBeginTime(beginTime);
    SetAivProfilingInfoBeginTime(aivProInfo);
    TestConstructParam(opArgs, topoArgs, algArgs, resourceArgs);
    ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProInfo);
    ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, exArgs, aivProInfo);
    ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, exArgsV2, aivProInfo);
    // GlobalMockObject::verify();
}

TEST_F(AlgAivTest, ReadBinFile_test)
{
    std::string fileName = "sss ss";
    std::string buffer;
    ReadBinFile(fileName, buffer);
}
