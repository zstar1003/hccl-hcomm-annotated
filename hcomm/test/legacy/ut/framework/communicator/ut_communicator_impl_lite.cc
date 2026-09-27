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

#define private public
#include "communicator_impl_lite_manager.h"
#include "communicator_impl_lite.h"
#include "ins_executor.h"
#include "aicpu_res_package_helper.h"
#include "alg_topo_package_helper.h"
#include "connected_link_mgr.h"
#include "rtsq_a5.h"
#include "rtsq_base.h"
#include "orion_adapter_rts.h"
#include "one_sided_component_lite.h"
#undef private
#include "internal_exception.h"
using namespace Hccl;
class CommunicatorImplLiteTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CommunicatorImplLiteTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "CommunicatorImplLiteTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_950));
        MOCKER_CPP(&RtsqBase::QuerySqBaseAddr)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(reinterpret_cast<u64>(&mockSq)));
        MOCKER_CPP(&RtsqBase::QuerySqDepth)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(static_cast<u32>(AC_SQE_MAX_CNT)));
        MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(0)));
        MOCKER_CPP(&RtsqBase::ConfigSqStatusByType).stubs();
        MOCKER_CPP(&CommunicatorImplLite::CheckNeedUpdateRes).defaults().will(returnValue(false));
        std::cout << "A Test case in CommunicatorImplLiteTest SetUp" << std::endl;
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in CommunicatorImplLiteTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }
    u8 mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT]{0};
};

TEST_F(CommunicatorImplLiteTest, test_load_with_hccl_exception)
{
    MOCKER_CPP(&InsExecutor::Execute).stubs().with(mockcpp::any()).will(throws(InternalException("")));
    MOCKER_CPP(&PrimTranslator::Translate).stubs().with(mockcpp::any()).will(returnValue(std::make_shared<InsQueue>()));
    MOCKER_CPP(&InsExecutor::ExecuteV82).stubs().with(mockcpp::any()).will(throws(InternalException("")));
    MOCKER_CPP(&CommunicatorImplLite::RegisterRtsqCallback).stubs();
    CommunicatorImplLite service(0);
    service.primTranslator = std::make_unique<PrimTranslator>();
    service.insExecutor = std::make_unique<InsExecutor>(&service);
    std::vector<char> uniqueId = {'0', '0', '0'};
    service.GetStreamLiteMgr()->streams.push_back(std::make_unique<StreamLite>(uniqueId));
    HcclSendRecvItem* items = new HcclSendRecvItem[5];
    u32 inttobuff = 100;
    for (int i = 0; i < 5; i++) {
        items[i].remoteRank = i;
        items[i].count = inttobuff;
        items[i].buf = static_cast<void*>(&inttobuff);
    }

    SendRecvItemTokenInfo* sendRecvItems = new SendRecvItemTokenInfo[5];
    u32 idvalue = 10;
    for (int i = 0; i < 5; i++) {
        sendRecvItems->tokenId = idvalue;
        sendRecvItems->tokenValue = idvalue;
    }

    HcclKernelParamLite param;
    param.op.algOperator.opMode = OpMode::OPBASE;
    param.op.algOperator.batchSendRecvDataDes.sendRecvItemsPtr = items;
    auto ret = service.LoadWithOpBasedMode(&param);
    EXPECT_EQ(ret, 1);

    HcclKernelParamLite param1;
    param1.op.algOperator.opMode = OpMode::OPBASE;
    param1.comm.devType = DevType::DEV_TYPE_950;
    param1.op.algOperator.batchSendRecvDataDes.sendRecvItemsPtr = items;

    service.isUpdateComm = false;
    service.UpdateCommParam(&param1);
    ret = service.LoadWithOpBasedMode(&param1);
    EXPECT_EQ(ret, 1);

    HcclKernelParamLite param2;
    param2.op.algOperator.opMode = OpMode::OPBASE;
    param2.comm.devType = DevType::DEV_TYPE_910A2;
    param2.op.algOperator.batchSendRecvDataDes.sendRecvItemsPtr = items;

    service.isUpdateComm = false;
    service.UpdateCommParam(&param2);
    ret = service.LoadWithOpBasedMode(&param2);
    EXPECT_EQ(ret, 1);
    delete[] items;
    delete[] sendRecvItems;
}

TEST_F(CommunicatorImplLiteTest, Ut_LoadWithOpBasedMode_When_910A2_Expect_ReturnKERNEL_SUCCESS)
{
    MOCKER_CPP(&CommunicatorImplLite::GetInsQueue)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(std::make_shared<InsQueue>()));
    CommunicatorImplLite service(0);
    service.primTranslator = std::make_unique<PrimTranslator>();
    service.insExecutor = std::make_unique<InsExecutor>(&service);
    std::vector<char> uniqueId = {'0', '0', '0'};
    service.GetStreamLiteMgr()->streams.push_back(std::make_unique<StreamLite>(uniqueId));
    HcclSendRecvItem* items = new HcclSendRecvItem[5];
    u32 inttobuff = 100;
    for (int i = 0; i < 5; i++) {
        items[i].remoteRank = i;
        items[i].count = inttobuff;
        items[i].buf = static_cast<void*>(&inttobuff);
    }

    SendRecvItemTokenInfo* sendRecvItems = new SendRecvItemTokenInfo[5];
    u32 idvalue = 10;
    for (int i = 0; i < 5; i++) {
        sendRecvItems->tokenId = idvalue;
        sendRecvItems->tokenValue = idvalue;
    }

    HcclKernelParamLite param2;
    param2.op.algOperator.opMode = OpMode::OPBASE;
    param2.comm.devType = DevType::DEV_TYPE_910A2;
    param2.op.algOperator.batchSendRecvDataDes.sendRecvItemsPtr = items;

    service.isUpdateComm = false;
    service.UpdateCommParam(&param2);
    auto ret = service.LoadWithOpBasedMode(&param2);
    EXPECT_EQ(ret, 1);
    delete[] items;
    delete[] sendRecvItems;
}

TEST_F(CommunicatorImplLiteTest, test_load_with_any_exception)
{
    MOCKER_CPP(&InsExecutor::Execute).stubs().with(mockcpp::any()).will(throws(1));
    MOCKER_CPP(&PrimTranslator::Translate).stubs().with(mockcpp::any()).will(returnValue(std::make_shared<InsQueue>()));
    MOCKER_CPP(&CommunicatorImplLite::RegisterRtsqCallback).stubs();
    CommunicatorImplLite service(0);
    service.primTranslator = std::make_unique<PrimTranslator>();
    service.insExecutor = std::make_unique<InsExecutor>(&service);
    std::vector<char> uniqueId = {'0', '0', '0'};
    service.GetStreamLiteMgr()->streams.push_back(std::make_unique<StreamLite>(uniqueId));
    HcclSendRecvItem* items = new HcclSendRecvItem[5];
    u32 inttobuff = 100;
    for (int i = 0; i < 5; i++) {
        items[i].remoteRank = i;
        items[i].count = inttobuff;
        items[i].buf = static_cast<void*>(&inttobuff);
    }

    SendRecvItemTokenInfo* sendRecvItems = new SendRecvItemTokenInfo[5];
    u32 idvalue = 10;
    for (int i = 0; i < 5; i++) {
        sendRecvItems->tokenId = idvalue;
        sendRecvItems->tokenValue = idvalue;
    }

    HcclKernelParamLite param;
    param.op.algOperator.opMode = OpMode::OPBASE;
    param.op.algOperator.batchSendRecvDataDes.sendRecvItemsPtr = items;

    auto ret = service.LoadWithOpBasedMode(&param);
    EXPECT_EQ(ret, 1);
    delete[] items;
    delete[] sendRecvItems;
}

TEST_F(CommunicatorImplLiteTest, test_get_method)
{
    CommunicatorImplLite service(0);

    EXPECT_NE(nullptr, service.GetHostDeviceSyncNotifyLiteMgr());
    EXPECT_NE(nullptr, service.GetStreamLiteMgr());
    EXPECT_NE(nullptr, service.GetQueueNotifyLiteMgr());
    EXPECT_NE(nullptr, service.GetCnt1tonNotifyLiteMgr());
    EXPECT_NE(nullptr, service.GetCntNto1NotifyLiteMgr());
    EXPECT_NE(nullptr, service.GetConnectedLinkMgr());
}

TEST_F(CommunicatorImplLiteTest, test_update_comm_with_hccl_exception)
{
    CommunicatorImplLite service(0);
    service.isSuspended = false;
    HcclKernelParamLite param;

    MOCKER(memcpy_s).stubs().will(returnValue(-1));
    auto ret = service.UpdateComm(&param);
    EXPECT_EQ(ret, 1);

    MOCKER_CPP(&CommunicatorImplLite::UpdateTransports)
        .stubs()
        .with(mockcpp::any())
        .will(throws(InternalException("")));
    ret = service.UpdateComm(&param);
    EXPECT_EQ(ret, 1);

    service.isSuspended = true;
    ret = service.UpdateComm(&param);
    EXPECT_EQ(ret, 1);
}

TEST_F(CommunicatorImplLiteTest, test_update_comm_with_any_exception)
{
    MOCKER_CPP(&CommunicatorImplLite::UpdateTransports).stubs().with(mockcpp::any()).will(throws(1));

    CommunicatorImplLite service(0);
    service.isSuspended = true;
    HcclKernelParamLite param;
    auto ret = service.UpdateComm(&param);
    EXPECT_EQ(ret, 1);
}

TEST_F(CommunicatorImplLiteTest, test_update_comm_success)
{
    CommunicatorImplLite service(0);

    u32 fakeDevPhyId = 1;
    u32 fakeNotifyId1 = 1;
    u32 fakeNotifyId2 = 2;
    BinaryStream notifyStream1;
    notifyStream1 << fakeNotifyId1;
    notifyStream1 << fakeDevPhyId;
    std::vector<char> notifyUniqueId1;
    notifyStream1.Dump(notifyUniqueId1);
    service.GetHostDeviceSyncNotifyLiteMgr()->notifys[0] = std::make_unique<NotifyLite>(notifyUniqueId1);

    BinaryStream notifyStream2;
    notifyStream2 << fakeNotifyId2;
    notifyStream2 << fakeDevPhyId;
    std::vector<char> notifyUniqueId2;
    notifyStream2.Dump(notifyUniqueId2);
    service.GetHostDeviceSyncNotifyLiteMgr()->notifys[1] = std::make_unique<NotifyLite>(notifyUniqueId2);

    u8 mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT]{0};
    MOCKER_CPP(&RtsqBase::QuerySqBaseAddr)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(reinterpret_cast<u64>(&mockSq)));
    MOCKER_CPP(&RtsqBase::QuerySqDepth)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(static_cast<u32>(AC_SQE_MAX_CNT)));
    MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(0)));
    MOCKER_CPP(&RtsqBase::ConfigSqStatusByType).stubs();

    u32 fakeStreamId = 0;
    u32 fakeSqId = 0;
    BinaryStream liteBinaryStream;
    liteBinaryStream << fakeStreamId;
    liteBinaryStream << fakeSqId;
    liteBinaryStream << fakeDevPhyId;
    std::vector<char> uniqueId{};
    liteBinaryStream.Dump(uniqueId);
    service.GetStreamLiteMgr()->streams.emplace_back(std::make_unique<StreamLite>(uniqueId));

    auto rtsq = static_cast<RtsqA5*>(service.GetStreamLiteMgr()->GetMaster()->GetRtsq());
    MOCKER_CPP_VIRTUAL(*rtsq, static_cast<void (RtsqA5::*)(u32)>(&RtsqA5::NotifyWait)).stubs().with(mockcpp::any());
    MOCKER_CPP_VIRTUAL(*rtsq, static_cast<void (RtsqA5::*)(u32, u32)>(&RtsqA5::NotifyWait))
        .stubs()
        .with(mockcpp::any());
    MOCKER_CPP_VIRTUAL(*rtsq, &RtsqA5::NotifyRecordLoc).stubs().with(mockcpp::any());

    HcclKernelParamLite kernelParam;
    // Ns快恢模式，必须更新资源
    service.isSuspended = true;
    MOCKER_CPP(&CommunicatorImplLite::CheckNeedUpdateRes).stubs().will(returnValue(true));

    std::vector<ModuleData> dataVec;
    dataVec.resize(AicpuResMgrType::__COUNT__);
    AicpuResPackageHelper tool;
    auto buffer = tool.GetPackedData(dataVec);
    uint64_t bufferAddress = reinterpret_cast<uintptr_t>(buffer.data());
    kernelParam.binaryResAddr = bufferAddress;
    kernelParam.binaryResSize = buffer.size();

    MOCKER_CPP(&MemTransportLiteMgr::ParseAllPackedData).stubs().with(mockcpp::any());
    auto ret = service.UpdateComm(&kernelParam);
    EXPECT_EQ(ret, 0);
}

TEST_F(CommunicatorImplLiteTest, test_UpdateLocBuffer_ranksize1_batchsendrecv)
{
    CommunicatorImplLite service(0);
    service.currentOp.opMode = OpMode::OPBASE;
    AlgTopoInfo algTopoInfo;
    service.algTopoInfoMap.insert(std::make_pair("TestAlgorithm", algTopoInfo));

    HcclKernelParamLite kernelParam;

    kernelParam.comm.rankSize = 4;
    MOCKER_CPP(&CommunicatorImplLite::CheckNeedUpdateRes).stubs().will(returnValue(true));
    strncpy(kernelParam.algName, "TestAlgorithm", MAX_NAME_LEN);
    strncpy(kernelParam.opTag, "TestTag", MAX_OP_TAG_LEN);
    kernelParam.op.algOperator.opType == OpType::BATCHSENDRECV;

    HcclAicpuLocBufLite input;
    input.addr = 0x12345678;
    input.size = 20;
    input.tokenId = 100;
    input.tokenValue = 123456;
    HcclAicpuLocBufLite output;
    input.addr = 0x12345678;
    input.size = 20;
    input.tokenId = 100;
    input.tokenValue = 123456;
    kernelParam.op.input = input;
    kernelParam.op.output = output;
    kernelParam.comm.opBaseScratch = output;

    SendRecvItemTokenInfo sendRecvTokens[5];
    int buf[5] = {1, 2, 3, 4, 5};
    int i = 0;
    for (i = 0; i < 5; i++) {
        sendRecvTokens[i].tokenId = i;
        sendRecvTokens[i].tokenValue = i;
    }
    HcclSendRecvItem sendRecvItems[5];
    for (i = 0; i < 5; i++) {
        sendRecvItems[i].remoteRank = i;
        sendRecvItems[i].buf = static_cast<void*>(&buf);
        sendRecvItems[i].count = i;
    }
    kernelParam.op.algOperator.batchSendRecvDataDes.sendRecvItemsPtr = sendRecvItems;

    kernelParam.op.algOperator.opType == OpType::BATCHSENDRECV;
    service.UpdateLocBuffer(&kernelParam);
}

TEST_F(CommunicatorImplLiteTest, test_UpdateLocBuffer_ranksize1_batchsendrecv_exp)
{
    CommunicatorImplLite service(0);
    HcclKernelParamLite kernelParam;
    Hccl::CollAlgOperator collAlgOperator;
    collAlgOperator.opType = OpType::BATCHSENDRECV;
    HcclAicpuOpLite opLite;
    opLite.algOperator = collAlgOperator;

    kernelParam.op = opLite;
    service.UpdateLocBuffer(&kernelParam);
}

TEST_F(CommunicatorImplLiteTest, test_UpdateLocBuffer_ranksize1_batchsendrecv_1)
{
    CommunicatorImplLite service(0);
    HcclKernelParamLite kernelParam;
    Hccl::CollAlgOperator collAlgOperator;
    collAlgOperator.opType = OpType::BATCHSENDRECV;
    HcclAicpuOpLite opLite;
    opLite.algOperator = collAlgOperator;

    SendRecvItemTokenInfo sendRecvTokens[5];
    int buf[5] = {1, 2, 3, 4, 5};
    int i = 0;
    for (i = 0; i < 5; i++) {
        sendRecvTokens[i].tokenId = i;
        sendRecvTokens[i].tokenValue = i;
    }
    HcclSendRecvItem sendRecvItems[5];
    for (i = 0; i < 5; i++) {
        sendRecvItems[i].remoteRank = i;
        sendRecvItems[i].buf = static_cast<void*>(&buf);
        sendRecvItems[i].count = i;
    }
    opLite.algOperator.batchSendRecvDataDes.sendRecvItemsPtr = sendRecvItems;

    kernelParam.op = opLite;
    service.UpdateLocBuffer(&kernelParam);
}

TEST_F(CommunicatorImplLiteTest, test_UnfoldOneSidedOp)
{
    HcclKernelParamLite kernelParamLite;
    kernelParamLite.comm.myRank = 3;
    kernelParamLite.comm.rankSize = 10;
    kernelParamLite.comm.devPhyId = 1;
    kernelParamLite.comm.opBaseScratch.size = 3;
    kernelParamLite.comm.opCounterAddr = 0;
    kernelParamLite.op.algOperator.opMode = OpMode::OPBASE;
    kernelParamLite.op.algOperator.opType = OpType::BATCHPUT;
    kernelParamLite.op.algOperator.sendRecvRemoteRank = 3;

    CommunicatorImplLite communicatorImplLite{0};
    BasePortType basePortType{PortDeploymentType::P2P, ConnectProtoType::HCCS};
    LinkData linkData{basePortType, 0, 1, 0, 1};
    communicatorImplLite.connectedLinkMgr->levelRankPairLinkDataMap[0][3] = {linkData};
    MOCKER_CPP(&CommunicatorImplLite::UpdateLocBuffer).stubs();
    MOCKER_CPP(&CommunicatorImplLite::UpdateRes).stubs();
}
