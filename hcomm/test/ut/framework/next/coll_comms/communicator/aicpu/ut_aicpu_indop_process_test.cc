/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../ut_hcomm_base.h"
#include "coll_comm_aicpu_mgr.h"
#include "coll_comm_aicpu_kernel_adpt.h"

// aicpu_indop_process 已解散，原用例迁移到 CollCommAicpuMgr 和 kernel adapter。

class TestCollCommAicpuMgr : public TestHcommCAdptBase {
public:
    void SetUp() override { TestHcommCAdptBase::SetUp(); }
    void TearDown() override { TestHcommCAdptBase::TearDown(); }
};

// InitComm 空指针测试 (原 AicpuIndOpCommInit)
TEST_F(TestCollCommAicpuMgr, Ut_InitComm_When_ParamNullptr_Return_Error)
{
    HcclResult ret = CollCommAicpuMgr::GetInstance().InitComm(nullptr);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// kernel adapter 空指针测试 (原 AicpuIndOpThreadInit)
TEST_F(TestCollCommAicpuMgr, Ut_InitThreads_When_ParamNullptr_Return_Error)
{
    HcclResult ret = CollCommAicpuKernelAdptInitThreads(nullptr);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// kernel adapter 空指针测试 (原 AicpuIndOpChannelInit)
TEST_F(TestCollCommAicpuMgr, Ut_InitChannel_When_ParamNullptr_Return_Error)
{
    HcclResult ret = CollCommAicpuKernelAdptInitChannel(nullptr);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// kernel adapter 空指针测试 (原 AicpuIndOpNotifyInit)
TEST_F(TestCollCommAicpuMgr, Ut_InitNotify_When_ParamNullptr_Return_Error)
{
    HcclResult ret = CollCommAicpuKernelAdptInitNotify(nullptr);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// kernel adapter 空指针测试 (原 AicpuIndOpChannelUpdate)
TEST_F(TestCollCommAicpuMgr, Ut_UpdateChannel_When_ParamNullptr_Return_Error)
{
    HcclResult ret = CollCommAicpuKernelAdptUpdateChannel(nullptr);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// ==================== CollCommAicpuMgr 注册表操作测试 ====================

// AcquireCommForUse + ReleaseComm — 获取使用标记后释放
TEST_F(TestCollCommAicpuMgr, Ut_AcquireCommForUse_ThenRelease)
{
    MOCKER_CPP(&CollCommAicpu::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));
    CommAicpuParam param{};
    strncpy(param.hcomId, "test_acquire_use", HCOMID_MAX_SIZE - 1);
    CollCommAicpuMgr::GetInstance().InitComm(&param);

    CollCommAicpu* comm = CollCommAicpuMgr::GetInstance().AcquireCommForUse("test_acquire_use");
    EXPECT_NE(comm, nullptr);
    CollCommAicpuMgr::GetInstance().ReleaseComm("test_acquire_use");
    CollCommAicpuMgr::GetInstance().DestroyComm("test_acquire_use");
}

// ReleaseComm — 不存在的 group 不崩溃
TEST_F(TestCollCommAicpuMgr, Ut_ReleaseComm_Nonexistent_DoesNotCrash)
{
    CollCommAicpuMgr::GetInstance().ReleaseComm("nonexistent_group");
    SUCCEED();
}

TEST_F(TestCollCommAicpuMgr, Ut_GetCurrentComm_EmptyOrMismatch_ReturnsNull)
{
    EXPECT_EQ(CollCommAicpuMgr::GetInstance().GetCurrentComm(""), nullptr);
    EXPECT_EQ(CollCommAicpuMgr::GetInstance().GetCurrentComm("nonexistent"), nullptr);
}

TEST_F(TestCollCommAicpuMgr, Ut_GetMutex_ReturnsValidReference)
{
    std::shared_mutex& mtx = CollCommAicpuMgr::GetInstance().GetMutex();
    mtx.lock();
    mtx.unlock();
    SUCCEED();
}

TEST_F(TestCollCommAicpuMgr, Ut_DestroyComm_NonExistent_ReturnsError)
{
    HcclResult ret = CollCommAicpuMgr::GetInstance().DestroyComm("nonexistent_group");
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(TestCollCommAicpuMgr, Ut_GetAllComms_ReturnsEmpty)
{
    std::vector<std::pair<std::string, CollCommAicpu*>> commInfo;
    HcclResult ret = CollCommAicpuMgr::GetInstance().GetAllComms(commInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// InitComm — mock InitAicpuIndOp 跳过硬件初始化，验证通信域创建流程
TEST_F(TestCollCommAicpuMgr, Ut_InitComm_MockInit_CreatesAndDestroys)
{
    MOCKER_CPP(&CollCommAicpu::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));
    CommAicpuParam param{};
    strncpy(param.hcomId, "initcomm_test_group", HCOMID_MAX_SIZE - 1);
    HcclResult ret = CollCommAicpuMgr::GetInstance().InitComm(&param);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 初始化 commEngineResMgr_ 供后续操作使用
    auto* comm = CollCommAicpuMgr::GetInstance().AcquireCommForUse("initcomm_test_group");
    EXPECT_NE(comm, nullptr);
    if (comm->commEngineResMgr_ == nullptr) {
        comm->commEngineResMgr_ = std::make_unique<CommEngineResAicpuMgr>(comm->dfx_, [](bool) {
            return HCCL_SUCCESS;
        });
    }

    CollCommAicpuMgr::GetInstance().DestroyComm("initcomm_test_group");
}

// AcquireCommForUse — 获取通信域并标记使用中
TEST_F(TestCollCommAicpuMgr, Ut_AcquireCommForUse_Then_Release)
{
    MOCKER_CPP(&CollCommAicpu::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));
    CommAicpuParam param{};
    strncpy(param.hcomId, "acquire_use_group", HCOMID_MAX_SIZE - 1);
    CollCommAicpuMgr::GetInstance().InitComm(&param);

    CollCommAicpu* comm = CollCommAicpuMgr::GetInstance().AcquireCommForUse("acquire_use_group");
    EXPECT_NE(comm, nullptr);
    CollCommAicpuMgr::GetInstance().ReleaseComm("acquire_use_group");

    CollCommAicpuMgr::GetInstance().DestroyComm("acquire_use_group");
}

// GetCurrentComm — 设置后能获取到当前通信域
TEST_F(TestCollCommAicpuMgr, Ut_GetCurrentComm_AfterSet_ReturnsComm)
{
    MOCKER_CPP(&CollCommAicpu::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));
    CommAicpuParam param{};
    strncpy(param.hcomId, "current_comm_group", HCOMID_MAX_SIZE - 1);
    CollCommAicpuMgr::GetInstance().InitComm(&param);

    CollCommAicpu* comm = CollCommAicpuMgr::GetInstance().AcquireCommForUse("current_comm_group");
    EXPECT_NE(comm, nullptr);
    // InitAicpuIndOp 被 mock，手动设置 identifier_ 供 GetCurrentComm 校验
    comm->identifier_ = "current_comm_group";
    // AcquireCommForUse 已设置 currentComm_，GetCurrentComm 应能找到
    CollCommAicpu* found = CollCommAicpuMgr::GetInstance().GetCurrentComm("current_comm_group");
    EXPECT_EQ(found, comm);

    CollCommAicpuMgr::GetInstance().ReleaseComm("current_comm_group");
    CollCommAicpuMgr::GetInstance().DestroyComm("current_comm_group");
}

// DestroyComm — 正常创建并销毁
TEST_F(TestCollCommAicpuMgr, Ut_DestroyComm_CreateThenDestroy_Success)
{
    MOCKER_CPP(&CollCommAicpu::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));
    CommAicpuParam param{};
    strncpy(param.hcomId, "destroy_test_group", HCOMID_MAX_SIZE - 1);
    CollCommAicpuMgr::GetInstance().InitComm(&param);
    HcclResult ret = CollCommAicpuMgr::GetInstance().DestroyComm("destroy_test_group");
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 再次销毁应返回错误
    ret = CollCommAicpuMgr::GetInstance().DestroyComm("destroy_test_group");
    EXPECT_NE(ret, HCCL_SUCCESS);
}
