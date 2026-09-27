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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "test_mock_setup.h"
#define private public
#define protected public
#include "op_type.h"
#include "hccl_communicator.h"
#include "communicator_impl.h"
#include "hccp_hdc_manager.h"
#include "rank_table.h"
#include "hccl_result.h"
#include "coll_operator.h"
#include "binary_stream.h"
#include "snap_shot_parse.h"
#include "binary_stream.h"
#include "mc2_type.h"
#include <hccl/hccl_types.h>
#include "hccl_comm.h"
#undef private
#undef protected

using namespace Hccl;

class HcclCommunicatorTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcclCommunicatorTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "HcclCommunicatorTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in HcclCommunicatorTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in HcclCommunicatorTest TearDown" << std::endl;
    }
};

#define SETUP_COMM_INIT_PUB_MOCKS()                                                                                 \
    do {                                                                                                            \
        MOCKER_CPP(static_cast<HcclResult (CommunicatorImpl::*)(                                                    \
                       const CommParams&, const std::string&, const HcclCommConfig&)>(&CommunicatorImpl::Init))     \
            .stubs()                                                                                                \
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())                                                   \
            .will(returnValue(HCCL_SUCCESS));                                                                       \
        MOCKER(HrtGetDevice).stubs().with().will(returnValue(0));                                                   \
        MOCKER(HrtGetDeviceCount).stubs().with().will(returnValue(8));                                              \
        MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(static_cast<DevId>(0))); \
    } while (0)

TEST_F(HcclCommunicatorTest, should_success_when_calling_init_with_valid_params)
{
    SETUP_COMM_INIT_PUB_MOCKS();
    MOCKER(HrtSetDevice).stubs().with(mockcpp::any());
    MOCKER_CPP(&HccpHdcManager::Init).stubs().with(mockcpp::any());
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    GenRankTableFile1Ser8Dev();
    CommParams commParams;
    auto comm = std::make_unique<HcclCommunicator>(commParams);
    auto res = comm->Init("ranktable.json");
    DelRankTableFile();
    EXPECT_EQ(HCCL_SUCCESS, res);
}

TEST_F(HcclCommunicatorTest, should_failed_when_calling_init_with_invalid_params)
{
    MOCKER_CPP(
        static_cast<HcclResult (CommunicatorImpl::*)(const CommParams&, const std::string&, const HcclCommConfig&)>(
            &CommunicatorImpl::Init))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PARA));
    MOCKER(HrtGetDevice).stubs().with().will(returnValue(0));
    MOCKER(HrtGetDeviceCount).stubs().with().will(returnValue(8));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(static_cast<DevId>(0)));
    MOCKER(HrtSetDevice).stubs().with(mockcpp::any());
    MOCKER_CPP(&HccpHdcManager::Init).stubs().with(mockcpp::any());
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    GenRankTableFile1Ser8Dev();
    CommParams commParams;
    auto comm = new HcclCommunicator(commParams);
    auto res = comm->Init("ranktable.json");
    delete comm;
    DelRankTableFile();

    EXPECT_EQ(HCCL_E_PARA, res);
}

TEST_F(HcclCommunicatorTest, should_success_when_calling_collop_with_valid_params)
{
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    CommParams commParams;
    auto comm = new HcclCommunicator(commParams);
    comm->Init("ranktable.json");
    CollOpParams opParams;
    auto res = comm->LoadOpbasedCollOp(opParams, nullptr);
    delete comm;
    EXPECT_EQ(HCCL_SUCCESS, res);
}

TEST_F(HcclCommunicatorTest, should_failed_when_calling_collop_with_invalid_params)
{
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PARA));
    CommParams commParams;
    auto comm = new HcclCommunicator(commParams);
    comm->Init("ranktable.json");
    CollOpParams opParams;
    auto res = comm->LoadOpbasedCollOp(opParams, nullptr);
    delete comm;
    EXPECT_EQ(HCCL_E_PARA, res);
}

TEST_F(HcclCommunicatorTest, should_success_when_calling_calc_coll_offload_op_res_with_valid_params)
{
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::CalcCollOffloadOpRes).stubs().will(returnValue(HCCL_SUCCESS));
    CommParams commParams;
    auto comm = new HcclCommunicator(commParams);
    comm->Init("ranktable.json");
    CollOpParams opParams;
    CollOffloadOpResReq resReq;
    auto res = comm->CalcCollOffloadOpRes(OpType::ALLREDUCE, 0, HCCL_DATA_TYPE_INT8, resReq);
    delete comm;
    EXPECT_EQ(HCCL_SUCCESS, res);
}

TEST_F(HcclCommunicatorTest, should_success_when_calling_set_coll_offload_slave_streams_with_valid_params)
{
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCollOffloadSlaveStreams).stubs().will(returnValue(HCCL_SUCCESS));
    CommParams commParams;
    auto comm = new HcclCommunicator(commParams);
    comm->Init("ranktable.json");
    CollOpParams opParams;
    std::vector<void*> stubStreams;
    std::string opTag = "test";
    auto res = comm->SetCollOffloadSlaveStreams(opTag, stubStreams);
    delete comm;
    EXPECT_EQ(HCCL_SUCCESS, res);
}

TEST_F(HcclCommunicatorTest, should_success_when_calling_set_coll_offload_scratch_buf_with_valid_params)
{
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCollOffloadScratchBuf).stubs().will(returnValue(HCCL_SUCCESS));
    CommParams commParams;
    auto comm = new HcclCommunicator(commParams);
    comm->Init("ranktable.json");
    CollOpParams opParams;
    std::string opTag = "opTag";
    auto res = comm->SetCollOffloadScratchBuf(opTag, nullptr, 0);
    delete comm;
    EXPECT_EQ(HCCL_SUCCESS, res);
}

TEST_F(HcclCommunicatorTest, should_success_when_calling_colloffloadop_with_valid_params)
{
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::LoadOffloadCollOp).stubs().will(returnValue(HCCL_SUCCESS));
    CommParams commParams;
    auto comm = new HcclCommunicator(commParams);
    comm->Init("ranktable.json");
    CollOpParams opParams;
    std::string opTag = "opTag";
    auto res = comm->LoadOffloadCollOp(opTag, opParams, nullptr);
    delete comm;
    EXPECT_EQ(HCCL_SUCCESS, res);
}

TEST_F(HcclCommunicatorTest, should_return_success_when_calling_communicator_create_sub_group)
{
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(static_cast<HcclResult (CommunicatorImpl::*)(
                   const CommParams& subCommParams, const std::vector<u32>& rankIds, CommunicatorImpl* subCommImpl,
                   HcclCommConfig& subConfig)>(&CommunicatorImpl::CreateSubComm))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    CommParams commParams;
    auto comm = new HcclCommunicator(commParams);
    comm->Init("ranktable.json");

    CommParams subCommParams;
    std::shared_ptr<HcclCommunicator> subHcclComm;
    std::vector<u32> rankIds;
    HcclCommConfig subConfig;

    auto res = comm->CreateSubComm(subCommParams, rankIds, subHcclComm, subConfig);
    EXPECT_EQ(HCCL_SUCCESS, res);
    EXPECT_NE(nullptr, subHcclComm);
    delete comm;
}

TEST_F(HcclCommunicatorTest, should_return_success_when_calling_communicator_create_sub_group_1)
{
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(static_cast<HcclResult (CommunicatorImpl::*)(
                   const CommParams& subCommParams, const std::vector<u32>& rankIds, CommunicatorImpl* subCommImpl)>(
                   &CommunicatorImpl::CreateSubComm))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    CommParams commParams;
    auto comm = new HcclCommunicator(commParams);
    comm->Init("ranktable.json");

    CommParams subCommParams;
    std::shared_ptr<HcclCommunicator> subHcclComm;
    std::vector<u32> rankIds;

    auto res = comm->CreateSubComm(subCommParams, rankIds, subHcclComm);
    EXPECT_EQ(HCCL_SUCCESS, res);
    EXPECT_NE(nullptr, subHcclComm);
    delete comm;
}

TEST_F(HcclCommunicatorTest, alloc_comm_resource_test)
{
    CommParams params;
    HcclCommunicator hcclCommunicator(params);
    void* commContext = nullptr;
    MOCKER_CPP(&CommunicatorImpl::AllocCommResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    Mc2Tiling tiling;
    HcclResult ret = hcclCommunicator.AllocCommResource(&tiling, &commContext);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcclCommunicatorTest, should_return_success_when_calling_communicator_suspend)
{
    MOCKER_CPP(&CommunicatorImpl::Suspend).stubs().will(returnValue(HCCL_SUCCESS));
    CommParams params;
    HcclCommunicator comm(params);
    EXPECT_EQ(HCCL_SUCCESS, comm.Suspend());
}

TEST_F(HcclCommunicatorTest, should_return_success_when_calling_communicator_clean)
{
    MOCKER_CPP(&CommunicatorImpl::Clean).stubs().will(returnValue(HCCL_SUCCESS));
    CommParams params;
    HcclCommunicator comm(params);
    EXPECT_EQ(HCCL_SUCCESS, comm.Clean());
}

TEST_F(HcclCommunicatorTest, should_return_success_when_calling_communicator_resume)
{
    MOCKER_CPP(&CommunicatorImpl::Resume).stubs().will(returnValue(HCCL_SUCCESS));
    CommParams params;
    HcclCommunicator comm(params);
    EXPECT_EQ(HCCL_SUCCESS, comm.Resume());
}

TEST_F(HcclCommunicatorTest, should_return_fail_when_calling_communicator_suspend)
{
    MOCKER_CPP(&CommunicatorImpl::Suspend).stubs().will(returnValue(HCCL_E_INTERNAL));
    CommParams params;
    HcclCommunicator comm(params);
    EXPECT_EQ(HCCL_E_INTERNAL, comm.Suspend());
}

TEST_F(HcclCommunicatorTest, should_return_fail_when_calling_communicator_resume)
{
    MOCKER_CPP(&CommunicatorImpl::Resume).stubs().will(returnValue(HCCL_E_INTERNAL));
    CommParams params;
    HcclCommunicator comm(params);
    EXPECT_EQ(HCCL_E_INTERNAL, comm.Resume());
}

TEST_F(HcclCommunicatorTest, should_get_rank_size_when_calling_communicator_get_rank_size)
{
    CommParams params;
    params.rankSize = 0;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    HcclCommunicator comm(params, &config);
    MOCKER_CPP(
        static_cast<HcclResult (CommunicatorImpl::*)(const CommParams&, const std::string&, const HcclCommConfig&)>(
            &CommunicatorImpl::Init))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    u32 rankSize = 0;
    comm.GetRankSize(&rankSize);
    EXPECT_EQ(params.rankSize, rankSize);
    EXPECT_EQ(HCCL_SUCCESS, comm.GetRankSize(&rankSize));
}

TEST_F(HcclCommunicatorTest, should_get_rank_id_when_calling_communicator_get_rank_id)
{
    CommParams params;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    HcclCommunicator comm(params, &config);
    u32 rankId = 0;
    EXPECT_EQ(HCCL_SUCCESS, comm.GetRankId(rankId));
}

TEST_F(HcclCommunicatorTest, should_return_seccess)
{
    CommParams params;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    HcclCommunicator comm(params, &config);
}

TEST_F(HcclCommunicatorTest, CollOpParams_desc)
{
    CollOpParams opParams;
    CollOpParams opParams1;
    opParams.DescAlltoall(opParams1);
    opParams.DescAlltoallV(opParams1);
    opParams.DescAlltoallVC(opParams1);
    opParams.DescBroadcast(opParams1);
    opParams.DescBatchSendRecv(opParams1);
}

TEST_F(HcclCommunicatorTest, should_return_true)
{
    CommParams commParams;
    commParams.isWorldGroup = true;
    auto comm = new HcclCommunicator(commParams);
    comm->Init("ranktable.json");
    EXPECT_EQ(comm->IsWorldGroup(), true);
    delete comm;
}

TEST_F(HcclCommunicatorTest, CollOperator_CollOpToString)
{
    CollOperator collOp;
    vector<OpType> optypes
        = {OpType::REDUCESCATTER, OpType::ALLREDUCE,  OpType::ALLGATHER, OpType::SCATTER, OpType::ALLTOALL,
           OpType::ALLTOALLV,     OpType::ALLTOALLVC, OpType::SEND,      OpType::RECV,    OpType::REDUCE};
    for (auto optype : optypes) {
        collOp.opType = optype;
        std::cout << CollOpToString(collOp);
        EXPECT_NO_THROW(CollOpToString(collOp));
    }
}

TEST_F(HcclCommunicatorTest, is_comm_ready_should_success)
{
    CommParams params;
    HcclCommConfig config;
    HcclCommunicator comm(params, &config);

    MOCKER_CPP(&CommunicatorImpl::IsCommReady).stubs().will(returnValue(true));
    EXPECT_TRUE(comm.IsCommReady());
}

TEST_F(HcclCommunicatorTest, get_snap_shot_dynamic_buf_should_success)
{
    CommParams params;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    HcclCommunicator comm(params, &config);

    comm.pimpl->collService = new CollServiceDefaultImpl(comm.pimpl.get());

    BinaryStream bs{};
    void* buf = &bs;
    EXPECT_EQ(comm.GetSnapShotDynamicBuf(buf), HcclResult::HCCL_SUCCESS);
    delete comm.pimpl->collService;
}

TEST_F(HcclCommunicatorTest, recover_comm_should_success)
{
    CommParams params;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    HcclCommunicator comm(params, &config);

    MOCKER_CPP(
        static_cast<HcclResult (CommunicatorImpl::*)(SnapShotComm&, u32, const char*)>(&CommunicatorImpl::RecoverComm))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    SnapShotComm snapShotComm{};
    void* ptr = &snapShotComm;
    const char* filePath = "test";
    EXPECT_EQ(comm.RecoverComm(ptr, 0, filePath), HcclResult::HCCL_SUCCESS);
}

TEST_F(HcclCommunicatorTest, recover_sub_comm_should_success)
{
    CommParams params;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    HcclCommunicator comm(params, &config);

    MOCKER_CPP(&CommunicatorImpl::RecoverSubComm)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    SnapShotSubComm snapShotSubComm{};
    void* ptr = &snapShotSubComm;
    std::shared_ptr<HcclCommunicator> subCommPtr;
    EXPECT_EQ(comm.RecoverSubComm(ptr, subCommPtr, 0), HcclResult::HCCL_SUCCESS);
}

TEST_F(HcclCommunicatorTest, get_static_binary_info_should_success)
{
    CommParams params;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    HcclCommunicator comm(params, &config);

    EXPECT_NO_THROW(comm.GetStaticBinaryInfo());
}

TEST_F(HcclCommunicatorTest, should_success_when_IsUsingCcu)
{
    SETUP_COMM_INIT_PUB_MOCKS();
    MOCKER(HrtSetDevice).stubs().with(mockcpp::any());
    MOCKER_CPP(&HccpHdcManager::Init).stubs().with(mockcpp::any());
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    GenRankTableFile1Ser8Dev();
    CommParams commParams;
    auto comm = std::make_unique<HcclCommunicator>(commParams);
    auto res = comm->Init("ranktable.json");

    EXPECT_EQ(comm->IsUsingCcuMs(), true);
    EXPECT_EQ(comm->IsUsingCcuSched(), false);

    DelRankTableFile();
    EXPECT_EQ(HCCL_SUCCESS, res);
}

TEST_F(HcclCommunicatorTest, Ut_GetLocalCclBuffer_When_Normal_Expect_OK)
{
    void* bufAddr = reinterpret_cast<void*>(0x12345678);
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(bufAddr));
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_950)));

    shared_ptr<DevBuffer> cclBuf = std::make_shared<DevBuffer>(10);
    GenRankTableFile1Ser8Dev();
    CommParams commParams;
    auto comm = std::make_unique<HcclCommunicator>(commParams);
    auto res = comm->Init("ranktable.json");
    comm->pimpl->inCclBuffer = std::make_shared<DevBuffer>(10);
    void* addr = nullptr;
    uint64_t size = 0;
    auto res1 = comm->GetLocalCclBuffer(&addr, &size);

    EXPECT_EQ(10, size);
    EXPECT_NE(nullptr, addr);
    EXPECT_EQ(HCCL_SUCCESS, res1);
}

TEST_F(HcclCommunicatorTest, Ut_GetDevMemWorkSpace_When_Tag_Fill_Expect_OK)
{
    void* bufAddr = reinterpret_cast<void*>(0x12345678);
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(bufAddr));
    GenRankTableFile1Ser8Dev();
    CommParams commParams;
    auto comm = std::make_unique<HcclCommunicator>(commParams);
    auto res = comm->Init("ranktable.json");

    std::string memTag = "";
    uint64_t size = 100;
    void* addr = nullptr;
    bool newCreated = false;
    auto res1 = comm->GetDevMemWorkSpace(memTag, &size, &addr, &newCreated);
    EXPECT_EQ(100, size);
    EXPECT_NE(nullptr, addr);
    EXPECT_EQ(true, newCreated);
    EXPECT_EQ(HCCL_SUCCESS, res1);
}

TEST_F(HcclCommunicatorTest, Ut_GetDevMemWorkSpace_When_Tag_Empty_Expect_OK)
{
    void* bufAddr = reinterpret_cast<void*>(0x12345678);
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(bufAddr));
    GenRankTableFile1Ser8Dev();
    CommParams commParams;
    auto comm = std::make_unique<HcclCommunicator>(commParams);
    auto res = comm->Init("ranktable.json");

    std::string memTag = "";
    uint64_t size = 100;
    void* addr = nullptr;
    auto res1 = comm->GetDevMemWorkSpace(memTag, &size, &addr, nullptr);
    EXPECT_EQ(100, size);
    EXPECT_NE(nullptr, addr);
    EXPECT_EQ(HCCL_SUCCESS, res1);

    size = 200;
    auto res2 = comm->GetDevMemWorkSpace(memTag, &size, &addr, nullptr);
    EXPECT_EQ(200, size);
    EXPECT_EQ(HCCL_E_PARA, res2);
}

// 测试 GetStreamId - 正常情况
TEST_F(HcclCommunicatorTest, Ut_GetStreamId_When_Normal_Expect_ReturnSuccess)
{
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::GetDpuStreamId).stubs().will(returnValue(12345U));

    CommParams commParams;
    auto comm = std::make_unique<HcclCommunicator>(commParams);
    comm->Init("ranktable.json");

    u32 streamId = 0;
    auto res = comm->GetStreamId(streamId);
    EXPECT_EQ(HCCL_SUCCESS, res);
    EXPECT_EQ(12345U, streamId);
}

// 测试 GetStreamId - 多次调用
TEST_F(HcclCommunicatorTest, Ut_GetStreamId_When_MultipleCalls_Expect_ReturnSameValue)
{
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::GetDpuStreamId).stubs().will(returnValue(54321U));

    CommParams commParams;
    auto comm = std::make_unique<HcclCommunicator>(commParams);
    comm->Init("ranktable.json");

    u32 streamId1 = 0;
    u32 streamId2 = 0;
    comm->GetStreamId(streamId1);
    comm->GetStreamId(streamId2);

    EXPECT_EQ(54321U, streamId1);
    EXPECT_EQ(54321U, streamId2);
}

// 测试 GetStreamId - 流 ID 为 0
TEST_F(HcclCommunicatorTest, Ut_GetStreamId_When_StreamIdZero_Expect_ReturnSuccess)
{
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::GetDpuStreamId).stubs().will(returnValue(0U));

    CommParams commParams;
    auto comm = std::make_unique<HcclCommunicator>(commParams);
    comm->Init("ranktable.json");

    u32 streamId = 0;
    auto res = comm->GetStreamId(streamId);
    EXPECT_EQ(HCCL_SUCCESS, res);
    EXPECT_EQ(0U, streamId);
}

TEST_F(HcclCommunicatorTest, Ut_HcclGetCclBufferSharedPtr_When_CommImplIsNull_Expect_ReturnHCCL_E_PTR)
{
    CommParams commParams;
    auto comm = std::make_unique<HcclCommunicator>(commParams);
    comm->Init("ranktable.json");

    // 模拟CommunicatorImpl为空
    CommunicatorImpl* commImpl = nullptr;
    MOCKER_CPP(&HcclCommunicator::GetCommImpl).stubs().will(returnValue(commImpl));

    std::shared_ptr<Hccl::DevBuffer> resultBuffer = nullptr;
    HcclResult ret = comm->GetCclBufferSharedPtr(resultBuffer);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclCommunicatorTest, Ut_HcclGetCclBufferSharedPtr_When_ValidComm_Expect_ReturnHCCL_SUCCESS)
{
    CommParams commParams;
    auto comm = std::make_unique<HcclCommunicator>(commParams);
    comm->Init("ranktable.json");

    // 在CommunicatorImpl中设置一个有效的cclBuffer
    auto testBuffer = std::make_shared<Hccl::DevBuffer>(1024);
    MOCKER_CPP(&CommunicatorImpl::GetCclBuffer).stubs().will(returnValue(testBuffer));

    std::shared_ptr<Hccl::DevBuffer> resultBuffer = nullptr;
    HcclResult ret = comm->GetCclBufferSharedPtr(resultBuffer);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(resultBuffer, testBuffer);
}

TEST_F(HcclCommunicatorTest, Ut_HcclGetCclBufferSharedPtr_When_CclBufferIsNull_Expect_ReturnHCCL_SUCCESS)
{
    CommParams commParams;
    auto comm = std::make_unique<HcclCommunicator>(commParams);
    comm->Init("ranktable.json");

    std::shared_ptr<Hccl::DevBuffer> testBuffer = nullptr;
    MOCKER_CPP(&CommunicatorImpl::GetCclBuffer).stubs().will(returnValue(testBuffer));
    std::shared_ptr<Hccl::DevBuffer> resultBuffer = nullptr;
    HcclResult ret = comm->GetCclBufferSharedPtr(resultBuffer);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(resultBuffer, nullptr);
}
