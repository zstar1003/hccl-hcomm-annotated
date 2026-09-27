/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include <string>
#include <cstring>

#define private public
#define protected public
#include "coll_comm_aicpu_destroy_func.h"
#include "coll_comm_aicpu_mgr.h"
#include "coll_comm_aicpu.h"
#undef private
#undef protected

using namespace hccl;

class CollCommAicpuDestroyFuncTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CollCommAicpuDestroyFuncTest set up." << std::endl; }
    static void TearDownTestCase() { std::cout << "CollCommAicpuDestroyFuncTest tear down." << std::endl; }
    void SetUp() override { GlobalMockObject::reset(); }
    void TearDown() override { GlobalMockObject::verify(); }
};

// 测试 Call() - stopCall_ 为 true 时直接返回，不执行 Process
TEST_F(CollCommAicpuDestroyFuncTest, Ut_Call_When_StopCallTrue_Expect_NoProcess)
{
    auto& func = CollCommAicpuDestroyFunc::GetInstance();
    func.stopCall_ = true;
    func.Call();
    EXPECT_TRUE(func.stopCall_);
    func.stopCall_ = false;
}

// 测试 Call() - 无通信域时 Process 正常返回，覆盖 shared_lock 读路径
TEST_F(CollCommAicpuDestroyFuncTest, Ut_Call_When_NoComm_Expect_Success)
{
    auto& func = CollCommAicpuDestroyFunc::GetInstance();
    func.stopCall_ = false;
    func.Call();
    EXPECT_FALSE(func.stopCall_);
}

// 测试 Process() - 有通信域但状态为 INVALID（continue 分支），覆盖 shared_lock 读路径
TEST_F(CollCommAicpuDestroyFuncTest, Ut_Process_When_CommStatusInvalid_Expect_SkipAndSuccess)
{
    // Mock CollCommAicpu::InitAicpuIndOp 跳过实际初始化
    MOCKER_CPP(&CollCommAicpu::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));

    CommAicpuParam commAicpuParam;
    std::string commName = "destroy_func_test_group";
    strncpy(commAicpuParam.hcomId, commName.c_str(), HCOMID_MAX_SIZE - 1);
    ASSERT_EQ(CollCommAicpuMgr::GetInstance().InitComm(&commAicpuParam), HCCL_SUCCESS);

    // 直接调用 Process()，通信域状态为 INVALID，会 continue 跳过
    auto& func = CollCommAicpuDestroyFunc::GetInstance();
    HcclResult ret = func.Process();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 清理
    EXPECT_EQ(CollCommAicpuMgr::GetInstance().DestroyComm(commAicpuParam.hcomId), HCCL_SUCCESS);
    GlobalMockObject::verify();
}

// 测试 Call() - 有通信域且状态正常但无 DESTROY_AICPU_COMM 命令，覆盖完整遍历路径
TEST_F(CollCommAicpuDestroyFuncTest, Ut_Call_When_CommReadyButNoDestroyCmd_Expect_Success)
{
    MOCKER_CPP(&CollCommAicpu::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));

    CommAicpuParam commAicpuParam;
    std::string commName = "destroy_func_no_cmd_group";
    strncpy(commAicpuParam.hcomId, commName.c_str(), HCOMID_MAX_SIZE - 1);
    ASSERT_EQ(CollCommAicpuMgr::GetInstance().InitComm(&commAicpuParam), HCCL_SUCCESS);

    auto& func = CollCommAicpuDestroyFunc::GetInstance();
    func.stopCall_ = false;
    func.Call();
    EXPECT_FALSE(func.stopCall_);

    // 清理
    EXPECT_EQ(CollCommAicpuMgr::GetInstance().DestroyComm(commAicpuParam.hcomId), HCCL_SUCCESS);
    GlobalMockObject::verify();
}
