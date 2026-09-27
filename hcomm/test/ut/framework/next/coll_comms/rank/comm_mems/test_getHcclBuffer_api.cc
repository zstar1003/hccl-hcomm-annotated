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
#include "hcomm_c_adpt.h"
#include "local_notify_impl.h"
#include "aicpu_launch_manager.h"
#include "llt_hccl_stub_rank_graph.h"

class TestHcclGetHcclBuffer : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        const char* fakeA5SocName = "Ascend950PR_958b";
        MOCKER(aclrtGetSocName).stubs().will(returnValue(fakeA5SocName));
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }

    // 创建并初始化 V2 通信域：mock 设备类型与 V2 支持，按 singleRank 选择 1P/2P 拓扑，
    // 完成 InitCollComm，返回 HcclComm 句柄。rankGraph_ 保持 rank graph 生命周期，避免悬垂。
    HcclComm InitV2Comm(bool singleRank, HcclMem cclBuffer)
    {
        MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
        MOCKER(IsSupportHCCLV2).stubs().will(returnValue(true));
        setenv("HCCL_INDEPENDENT_OP", "1", 1);

        void* commV2 = (void*)0x2000;
        RankGraphStub rankGraphStub;
        rankGraph_ = singleRank ? rankGraphStub.Create1PGraph() : rankGraphStub.Create2PGraph();
        u32 rank = singleRank ? 0 : 1;
        char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
        hcclCommPtr_ = make_shared<hccl::hcclComm>(1, 1, commName);
        HcclCommConfig config;
        UtInitHcclCommConfig(config);
        config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
        config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
        config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
        unsetenv("HCCL_DFS_CONFIG");
        HcclResult ret = hcclCommPtr_->InitCollComm(commV2, rankGraph_.get(), rank, cclBuffer, commName, &config);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        return static_cast<HcclComm>(hcclCommPtr_.get());
    }

private:
    std::shared_ptr<Hccl::RankGraph> rankGraph_;
    std::shared_ptr<hccl::hcclComm> hcclCommPtr_;
};

TEST_F(TestHcclGetHcclBuffer, Ut_HcclGetHcclBuffer_When_Normal_Return_HCCL_Success)
{
    HcclMem cclBuffer;
    cclBuffer.size = 2;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    HcclComm comm = InitV2Comm(false, cclBuffer);

    void* buffer;
    uint64_t size;
    HcclResult ret = HcclGetHcclBuffer(comm, &buffer, &size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(size, 2);
}

TEST_F(TestHcclGetHcclBuffer, Ut_HcclGetHcclBufferCleared_When_Normal_Return_HCCL_Success)
{
    u64 bufferStub = 0;
    HcclMem cclBuffer;
    cclBuffer.size = sizeof(bufferStub);
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = &bufferStub;
    HcclComm comm = InitV2Comm(false, cclBuffer);

    void* buffer;
    uint64_t size;
    HcclResult ret = HcclGetHcclBufferCleared(comm, &buffer, &size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(size, sizeof(bufferStub));
}

TEST_F(TestHcclGetHcclBuffer, Ut_HcclGetHcclBuffer_When_SingleRank_Return_NullptrAndZeroSize)
{
    HcclMem cclBuffer;
    cclBuffer.size = 2;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    HcclComm comm = InitV2Comm(true, cclBuffer);

    void* buffer = (void*)0xDEAD;
    uint64_t size = 0xDEAD;
    HcclResult ret = HcclGetHcclBuffer(comm, &buffer, &size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(buffer, nullptr);
    EXPECT_EQ(size, 0);
}

TEST_F(TestHcclGetHcclBuffer, Ut_HcclGetHcclBufferCleared_When_SingleRank_Return_NullptrAndZeroSize)
{
    HcclMem cclBuffer;
    cclBuffer.size = 2;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    HcclComm comm = InitV2Comm(true, cclBuffer);

    void* buffer = (void*)0xDEAD;
    uint64_t size = 0xDEAD;
    HcclResult ret = HcclGetHcclBufferCleared(comm, &buffer, &size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(buffer, nullptr);
    EXPECT_EQ(size, 0);
}

TEST_F(TestHcclGetHcclBuffer, Ut_HcclGetHcclBuffer_When_CommNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    void* comm = nullptr;
    void* buffer;
    uint64_t size;
    HcclResult ret = HcclGetHcclBuffer(comm, &buffer, &size);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclGetHcclBuffer, Ut_HcclGetHcclBuffer_When_bufferNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    void* comm = (void*)0x123456;
    void** buffer{nullptr};
    uint64_t size;
    HcclResult ret = HcclGetHcclBuffer(comm, buffer, &size);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclGetHcclBuffer, Ut_HcclGetHcclBuffer_When_sizeNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    void* comm = (void*)0x123456;
    void* buffer;
    uint64_t* size{nullptr};
    HcclResult ret = HcclGetHcclBuffer(comm, &buffer, size);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclGetHcclBuffer, Ut_HcclGetHcclBuffer_When_CollCommNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(IsSupportHCCLV2).stubs().will(returnValue(true));
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);

    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    void* buffer;
    uint64_t size;
    HcclResult ret = HcclGetHcclBuffer(comm, &buffer, &size);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclGetHcclBuffer, Ut_HcclGetHcclBuffer_When_MyRankNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollComm::Init).stubs().will(returnValue(0));
    MOCKER_CPP(&CollComm::GetHDCommunicate).stubs().will(returnValue(0));
    MOCKER(IsSupportHCCLV2).stubs().will(returnValue(true));
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    unsetenv("HCCL_DFS_CONFIG");
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);

    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    void* buffer;
    uint64_t size;
    ret = HcclGetHcclBuffer(comm, &buffer, &size);
    EXPECT_EQ(ret, HCCL_E_PTR);
}
TEST_F(TestHcclGetHcclBuffer, Ut_HcclGetHcclBuffer_When_CommMemsNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&MyRank::Init).stubs().will(returnValue(0));
    MOCKER(IsSupportHCCLV2).stubs().will(returnValue(true));
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    unsetenv("HCCL_DFS_CONFIG");
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);

    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    void* buffer;
    uint64_t size;
    ret = HcclGetHcclBuffer(comm, &buffer, &size);
    EXPECT_EQ(ret, HCCL_E_PTR);
}
