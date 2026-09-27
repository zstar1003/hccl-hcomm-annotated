/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../hccl_api_base_test.h"
#include "hccl/hccl_res.h"
#include "hcomm_res_defs.h"
#include "aicpu_launch_manager.h"
#include "llt_hccl_stub_rank_graph.h"
#include "launch_aicpu.h"
#include "coll_comm_profiling.h"

#define private public
#define protected public
#include "coll_comm.h"
#include "comm_engine_res_manager.h"
#undef protected
#undef private

class TestHcclThreadAcquireWithConfig : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        unsetenv("HCCL_DFS_CONFIG");
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }

    void InitValidConfig(ThreadConfig* config, uint32_t num) { (void)ThreadConfigInit(config, num); }

    std::shared_ptr<hccl::hcclComm> CreateV2Comm()
    {
        void* commV2 = (void*)0x2000;
        RankGraphStub rankGraphStub;
        std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
        u32 rank = 1;
        HcclMem cclBuffer;
        cclBuffer.size = 1;
        cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
        cclBuffer.addr = (void*)0x1000;
        char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
        auto hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
        HcclCommConfig config;
        UtInitHcclCommConfig(config);
        config.hcclOpExpansionMode = 1;
        config.hcclRdmaTrafficClass = 0xFFFFFFFF;
        config.hcclRdmaServiceLevel = 0xFFFFFFFF;
        HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
        EXPECT_EQ(ret, 0);
        return hcclCommPtr;
    }
};

TEST_F(TestHcclThreadAcquireWithConfig, Ut_When_PtrNull_Expect_HCCL_E_PTR)
{
    ThreadConfig config;
    InitValidConfig(&config, 1);
    ThreadHandle thread;

    HcclResult ret = HcclThreadAcquireWithConfig(nullptr, COMM_ENGINE_CPU, 1, THREAD_TYPE_TS, &config, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    auto hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    ret = HcclThreadAcquireWithConfig(comm, COMM_ENGINE_CPU, 1, THREAD_TYPE_TS, &config, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclThreadAcquireWithConfig(comm, COMM_ENGINE_CPU, 1, THREAD_TYPE_TS, nullptr, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThreadAcquireWithConfig, Ut_When_TypeInvalid_Expect_HCCL_E_PARA)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    auto hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    ThreadConfig config;
    InitValidConfig(&config, 1);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquireWithConfig(comm, COMM_ENGINE_CPU, 1, THREAD_TYPE_INVALID, &config, &thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThreadAcquireWithConfig, Ut_When_ConfigMagicWordMismatch_Expect_HCCL_E_PARA)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    auto hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    ThreadConfig config;
    (void)memset_s(&config, sizeof(ThreadConfig), 0, sizeof(ThreadConfig));
    config.header.magicWord = 0x0;
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquireWithConfig(comm, COMM_ENGINE_CPU, 1, THREAD_TYPE_TS, &config, &thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThreadAcquireWithConfig, Ut_When_UnsupportedEngine_Expect_HCCL_E_PARA)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    auto hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    ThreadConfig config;
    InitValidConfig(&config, 1);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    CommEngine unsupportedEngines[] = {COMM_ENGINE_AICPU_TS, COMM_ENGINE_CPU_TS, COMM_ENGINE_AIV, COMM_ENGINE_CCU};
    for (auto engine : unsupportedEngines) {
        HcclResult ret = HcclThreadAcquireWithConfig(comm, engine, 1, THREAD_TYPE_TS, &config, &thread);
        EXPECT_EQ(ret, HCCL_E_PARA);
    }
}

TEST_F(TestHcclThreadAcquireWithConfig, Ut_When_V2_CollCommNullptr_Expect_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));
    MOCKER_CPP(&hcclComm::IsCommunicatorV2).stubs().will(returnValue(true));

    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    auto hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    ThreadConfig config;
    InitValidConfig(&config, 1);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquireWithConfig(comm, COMM_ENGINE_CPU, 1, THREAD_TYPE_TS, &config, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThreadAcquireWithConfig, Ut_When_V2_EngineResMgrNullptr_Expect_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));
    MOCKER_CPP(&CollComm::Init).stubs().will(returnValue(0));
    MOCKER_CPP(&CollComm::GetHDCommunicate).stubs().will(returnValue(0));

    auto hcclCommPtr = CreateV2Comm();
    ThreadConfig threadConfig;
    InitValidConfig(&threadConfig, 1);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquireWithConfig(comm, COMM_ENGINE_CPU, 1, THREAD_TYPE_TS, &threadConfig, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThreadAcquireWithConfig, Ut_When_V2_HcclThreadAcquireV2Fail_Expect_Error)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommEngineResMgr::HcclThreadAcquireV2).stubs().will(returnValue(HCCL_E_UNAVAIL));

    auto hcclCommPtr = CreateV2Comm();
    ThreadConfig threadConfig;
    InitValidConfig(&threadConfig, 1);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquireWithConfig(comm, COMM_ENGINE_AICPU, 1, THREAD_TYPE_TS, &threadConfig, &thread);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);
}

TEST_F(TestHcclThreadAcquireWithConfig, Ut_When_V2_CpuEngine_Expect_HCCL_SUCCESS)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommEngineResMgr::HcclThreadAcquireV2).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommThreadRegisterDfx).stubs().will(returnValue(0));

    auto hcclCommPtr = CreateV2Comm();
    ThreadConfig threadConfig;
    InitValidConfig(&threadConfig, 1);
    ThreadHandle thread = 0;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquireWithConfig(comm, COMM_ENGINE_CPU, 1, THREAD_TYPE_TS, &threadConfig, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcclThreadAcquireWithConfig, Ut_When_V2_AicpuEngine_Expect_HCCL_SUCCESS)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommEngineResMgr::HcclThreadAcquireV2).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommDfx::ReportMc2CommInfo).stubs();
    MOCKER_CPP(&HcclCommDfx::ReportKernel).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollComm::GetMyRankId).stubs().will(returnValue(0u));

    auto hcclCommPtr = CreateV2Comm();
    ThreadConfig threadConfig;
    InitValidConfig(&threadConfig, 1);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquireWithConfig(comm, COMM_ENGINE_AICPU, 1, THREAD_TYPE_TS, &threadConfig, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcclThreadAcquireWithConfig, Ut_When_V1_CpuEngine_Expect_HCCL_SUCCESS)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910B)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));
    MOCKER_CPP(&hcclComm::IsCommunicatorV2).stubs().will(returnValue(false));
    MOCKER_CPP(&CommEngineResMgr::HcclThreadAcquire).stubs().will(returnValue(HCCL_SUCCESS));

    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    auto hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    ThreadConfig config;
    InitValidConfig(&config, 1);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquireWithConfig(comm, COMM_ENGINE_CPU, 1, THREAD_TYPE_TS, &config, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcclThreadAcquireWithConfig, Ut_When_V1_AicpuEngine_Expect_HCCL_SUCCESS)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910B)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));
    MOCKER_CPP(&hcclComm::IsCommunicatorV2).stubs().will(returnValue(false));
    std::vector<uint32_t> threadIdVal = {0};
    MOCKER_CPP(&CommEngineResMgr::HcclThreadAcquire)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(threadIdVal))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclStreamProfilingReport).stubs().will(returnValue(HCCL_SUCCESS));

    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    auto hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    ThreadConfig config;
    InitValidConfig(&config, 1);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquireWithConfig(comm, COMM_ENGINE_AICPU, 1, THREAD_TYPE_TS, &config, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcclThreadAcquireWithConfig, Ut_When_V1_AicpuEngineThreadNumMismatch_Expect_HCCL_E_PARA)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910B)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));
    MOCKER_CPP(&hcclComm::IsCommunicatorV2).stubs().will(returnValue(false));
    MOCKER_CPP(&CommEngineResMgr::HcclThreadAcquire).stubs().will(returnValue(HCCL_SUCCESS));

    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    auto hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);

    ThreadConfig config[2];
    InitValidConfig(config, 2);
    ThreadHandle threads[2];
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquireWithConfig(comm, COMM_ENGINE_AICPU, 2, THREAD_TYPE_TS, config, threads);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThreadAcquireWithConfig, Ut_When_V1_AicpuEngineStreamProfilingFail_Expect_Error)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910B)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));
    MOCKER_CPP(&hcclComm::IsCommunicatorV2).stubs().will(returnValue(false));
    std::vector<uint32_t> threadIdVal = {0};
    MOCKER_CPP(&CommEngineResMgr::HcclThreadAcquire)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(threadIdVal))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclStreamProfilingReport).stubs().will(returnValue(HCCL_E_INTERNAL));

    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    auto hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    ThreadConfig config;
    InitValidConfig(&config, 1);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquireWithConfig(comm, COMM_ENGINE_AICPU, 1, THREAD_TYPE_TS, &config, &thread);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(TestHcclThreadAcquireWithConfig, Ut_When_V1_HcclThreadAcquireFail_Expect_Error)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910B)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));
    MOCKER_CPP(&hcclComm::IsCommunicatorV2).stubs().will(returnValue(false));
    MOCKER_CPP(&CommEngineResMgr::HcclThreadAcquire).stubs().will(returnValue(HCCL_E_UNAVAIL));

    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    auto hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    ThreadConfig config;
    InitValidConfig(&config, 1);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquireWithConfig(comm, COMM_ENGINE_CPU, 1, THREAD_TYPE_TS, &config, &thread);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);
}

TEST_F(TestHcclThreadAcquireWithConfig, Ut_When_V2_DfxFail_Expect_Error)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommEngineResMgr::HcclThreadAcquireV2).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommThreadRegisterDfx).stubs().will(returnValue(1));

    auto hcclCommPtr = CreateV2Comm();
    ThreadConfig threadConfig;
    InitValidConfig(&threadConfig, 1);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquireWithConfig(comm, COMM_ENGINE_CPU, 1, THREAD_TYPE_TS, &threadConfig, &thread);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}
