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
#include <mockcpp/mockcpp.hpp>
#include "workflow_pub.h"
#include "topoinfo_struct.h"
#include "hccl_ip_address.h"
#include "dlra_function.h"
#include "dltdt_function.h"
#include "acl/acl.h"
#include "hccl_comm.h"
#include "hccl_inner.h"
#include "hccl_impl.h"
#include "config.h"
#include "hccl_communicator.h"
#include "hccl_communicator_attrs.h"
#include "network_manager_pub.h"
#include "transport_base_pub.h"
#include "transport_p2p_pub.h"
#include "broadcast_operator.h"

#define private public
#define protected public

#undef private
#undef protected

using namespace hccl;
using namespace std;
class HcclImplTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        DlRaFunction::GetInstance().DlRaFunctionInit();
        std::cout << "HcclImplTest SetUP" << std::endl;
    }
    static void TearDownTestCase() { std::cout << "HcclImplTest TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HcclCommunicator::InitPreResource).stubs().will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
        unsetenv("HCCL_ALGO");
    }
    static HcclResult TestConstructParam(HcclCommParams& params, RankTable_t& rankTable, int serverNum)
    {
        string commId = "comm ";
        CHK_SAFETY_FUNC_RET(memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1));

        // 1. 更新总 Rank 数
        params.rank = 0;
        params.totalRanks = serverNum;
        params.isHeterogComm = false;
        params.logicDevId = 0;
        params.commWorkMode = WorkMode::HCCL_MODE_NORMAL;
        params.deviceType = DevType::DEV_TYPE_910;

        rankTable.collectiveId = "192.168.0.101-8000-8001";

        // 2. 将 vector 容量改为 3
        vector<RankInfo_t> rankVec(serverNum);

        // Rank 0
        rankVec[0].rankId = 0;
        rankVec[0].deviceInfo.devicePhyId = 0;
        HcclIpAddress ipAddr1(1694542016); // 192.168.0.101 (主机字节序视具体实现而定)
        rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1);
        rankVec[0].serverIdx = 0;
        rankVec[0].serverId = "192.168.0.101";

        if (serverNum == 3) {
            // Rank 1
            rankVec[1].rankId = 1;
            rankVec[1].deviceInfo.devicePhyId = 0;
            HcclIpAddress ipAddr2(1711319232); // 192.168.0.102
            rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2);
            rankVec[1].serverIdx = 1;
            rankVec[1].serverId = "192.168.0.102";

            //  Rank 2
            rankVec[2].rankId = 2;
            rankVec[2].deviceInfo.devicePhyId = 0;
            HcclIpAddress ipAddr3(1728096448);
            rankVec[2].deviceInfo.deviceIp.push_back(ipAddr3);
            rankVec[2].serverIdx = 2;
            rankVec[2].serverId = "192.168.0.103";
        }

        // 4. 更新 rankTable 统计信息
        rankTable.rankList.assign(rankVec.begin(), rankVec.end());
        rankTable.deviceNum = serverNum;
        rankTable.serverNum = serverNum;

        return HCCL_SUCCESS;
    }
};

void InitSendRecvTestData(SingleSubCommTransport& singleTrans, std::vector<HcclSendRecvItem>& items)
{
    TransportRequest transportReq1;
    transportReq1.isValid = true;
    transportReq1.remoteUserRank = 1;
    transportReq1.inputMemType = CCL_INPUT;
    transportReq1.outputMemType = CCL_OUTPUT;
    transportReq1.linkType = TransportLinkType::HCCS;

    TransportRequest transportReq2;
    transportReq2.isValid = true;
    transportReq2.remoteUserRank = 2;
    transportReq2.inputMemType = CCL_INPUT;
    transportReq2.outputMemType = CCL_OUTPUT;
    transportReq2.linkType = TransportLinkType::SIO;

    singleTrans.transportRequests.emplace_back(transportReq1);
    singleTrans.transportRequests.emplace_back(transportReq2);
    singleTrans.links.resize(2, nullptr);
    singleTrans.status.resize(2, TransportStatus::INIT);

    items = {
        {HCCL_SEND, nullptr, 512, HCCL_DATA_TYPE_INT32, 1}, // 发给1，对端1是接收者
        {HCCL_RECV, nullptr, 512, HCCL_DATA_TYPE_INT32, 1},
        {HCCL_RECV, nullptr, 512, HCCL_DATA_TYPE_INT32, 2} // 从2收，对端2是发送者
    };
}

TEST_F(HcclImplTest, ut_AllocBatchSendRecvLinks_When_Normal_Return_HCCL_SUCCESS)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable, 3);
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());

    MOCKER(hrtProfRegisterCtrlCallback).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    communicator->Init(params, rankTable);

    SingleSubCommTransport singleTrans;
    std::vector<HcclSendRecvItem> items;
    TransportIOMem transMem;
    InitSendRecvTestData(singleTrans, items);

    MOCKER_CPP(&TransportManager::GetIOMem).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    // stubs in CreateDestSockets
    MOCKER_CPP(&TransportManager::UpdateIsInterRdma)
        .stubs()
        .with(mockcpp::any(), outBound(false), mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&TransportManager::MakeRemoteLinkInfo).stubs().will(returnValue(HCCL_SUCCESS));

    // stubs in IsHccsTransport
    MOCKER(hrtGetPairDeviceLinkType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(LinkTypeInServer::HCCS_SW_TYPE))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(Is310PDevice).stubs().will(returnValue(false));
    MOCKER_CPP(&HcclSocketManager::CreateSingleLinkSocket).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    // stubs in CreateLink
    MOCKER(hrtErrMSetErrorContextPub).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(hrtSetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::TransportInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtResetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::CheckBatchSendRecvLinkStatus)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::DestroySockets, void(HcclSocketManager::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());

    std::string tag = "batch_send_recv_test";
    ret = communicator->transportManager_->AllocBatchSendRecvLinks(
        items.data(), items.size(), tag, transMem, singleTrans, false, false, 0, false,
        HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_AllocBatchSendRecvLinks_When_hrtSetDevice_Failed_Return_HCCL_E_INTERNAL)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable, 3);
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());

    MOCKER(hrtProfRegisterCtrlCallback).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    communicator->Init(params, rankTable);

    SingleSubCommTransport singleTrans;
    std::vector<HcclSendRecvItem> items;
    TransportIOMem transMem;
    InitSendRecvTestData(singleTrans, items);

    MOCKER(hrtSetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_RUNTIME));

    MOCKER(hrtResetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::CheckBatchSendRecvLinkStatus)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    std::string tag = "batch_send_recv_test";
    ret = communicator->transportManager_->AllocBatchSendRecvLinks(
        items.data(), items.size(), tag, transMem, singleTrans, false, false, 0, false,
        HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, false);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_AllocBatchSendRecvLinks_When_CreateSingleLinkSocket_Failed_Return_HCCL_E_INTERNAL)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable, 3);
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());

    MOCKER(hrtProfRegisterCtrlCallback).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    communicator->Init(params, rankTable);

    SingleSubCommTransport singleTrans;
    std::vector<HcclSendRecvItem> items;
    TransportIOMem transMem;
    InitSendRecvTestData(singleTrans, items);

    MOCKER_CPP(&TransportManager::GetIOMem).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    // stubs in CreateDestSockets
    MOCKER_CPP(&TransportManager::UpdateIsInterRdma)
        .stubs()
        .with(mockcpp::any(), outBound(false), mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&TransportManager::MakeRemoteLinkInfo).stubs().will(returnValue(HCCL_SUCCESS));

    // stubs in IsHccsTransport
    MOCKER(hrtGetPairDeviceLinkType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(LinkTypeInServer::HCCS_SW_TYPE))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(Is310PDevice).stubs().will(returnValue(false));
    MOCKER_CPP(&HcclSocketManager::CreateSingleLinkSocket)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));

    // stubs in CreateLink
    MOCKER(hrtErrMSetErrorContextPub).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(hrtSetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::TransportInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtResetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::CheckBatchSendRecvLinkStatus)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::DestroySockets, void(HcclSocketManager::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());

    std::string tag = "batch_send_recv_test";
    ret = communicator->transportManager_->AllocBatchSendRecvLinks(
        items.data(), items.size(), tag, transMem, singleTrans, false, false, 0, false,
        HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, false);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_CheckBatchSendRecvLinkStatus_When_LinkIsNullPtr_Return_HCCL_E_NOT_FOUND)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable, 3);
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());

    MOCKER(hrtProfRegisterCtrlCallback).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    communicator->Init(params, rankTable);

    TransportRequest transportReq1;
    transportReq1.isValid = true;
    transportReq1.remoteUserRank = 0;
    transportReq1.inputMemType = CCL_INPUT;
    transportReq1.outputMemType = CCL_OUTPUT;
    transportReq1.linkType = TransportLinkType::HCCS;
    TransportRequest transportReq2;
    transportReq2.isValid = true;
    transportReq2.remoteUserRank = 1;
    transportReq2.inputMemType = CCL_INPUT;
    transportReq2.outputMemType = CCL_OUTPUT;
    transportReq2.linkType = TransportLinkType::SIO;

    SingleSubCommTransport singleTrans;
    singleTrans.transportRequests.emplace_back(transportReq1);
    singleTrans.transportRequests.emplace_back(transportReq2);
    singleTrans.links.resize(2, nullptr);
    singleTrans.status.resize(2, TransportStatus::INIT);

    MOCKER_CPP(&NotifyPool::UnregisterOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    std::string tag = "test";
    ret = communicator->transportManager_->CheckBatchSendRecvLinkStatus(tag, singleTrans, false);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_SelectAlg_when_broadcast_910C_Expect_ReturnIs_BroadcastMeshAivExecutor)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable, 1);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER(hrtProfRegisterCtrlCallback).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    impl->deviceLogicId_ = 0;
    impl->devicePhyId_ = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_BROADCAST].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_4P_MESH;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_BROADCAST].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
    impl->topoType_ = TopoType::TOPO_TYPE_4P_MESH;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_4P_MESH;
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(2048);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 2048;
    opParam.DataDes.count = 2048 / 4;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);
    opParam.root = 0;
    std::string algName;
    std::string newTag;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    topoMatcher->SetAivModeConfig(true);
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<BroadCastOperator> operation(
        new (std::nothrow) BroadCastOperator(algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher));
    ret = operation->SelectAlg("", opParam, algName, newTag);
    EXPECT_TRUE(algName == "BroadcastMeshAivExecutor");
    operation = nullptr;
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, Ut_SplitBsrData_When_countEqualZero_Expect_ReturnIsHCCL_SUCCESS)
{
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());
    hcclCommunicator->userRankSize_ = 2;
    hcclCommunicator->userRank_ = 0;
    OpParam opParam;
    opParam.BatchSendRecvDataDes.itemNum = 2;
    HcclSendRecvItem sendRecvItems[2]
        = {{HcclSendRecvType::HCCL_SEND, nullptr, 0, HcclDataType::HCCL_DATA_TYPE_INT16, 0},
           {HcclSendRecvType::HCCL_RECV, nullptr, 0, HcclDataType::HCCL_DATA_TYPE_INT16, 1}};

    opParam.BatchSendRecvDataDes.sendRecvItemsPtr = sendRecvItems;
    std::vector<u8> isDirectRemoteRank;
    std::vector<HcclSendRecvItem> hostSendRecvInfo;
    std::vector<HcclSendRecvItem> deviceSendRecvInfo;
    hcclCommunicator->SplitBsrData(opParam, isDirectRemoteRank, hostSendRecvInfo, deviceSendRecvInfo);
    EXPECT_EQ(hostSendRecvInfo.size(), 0);
    GlobalMockObject::verify();
}

TEST_F(
    HcclImplTest,
    ut_HcclCommunicator_BuildZeroCopyParamSuccess_When_ZeroCopyLocalBuffer_Is_Not_Nullptr_Expect_Return_Success)
{
    MOCKER_CPP(&ZeroCopyMemoryAgent::GetRingBufferAddr).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());
    EXPECT_NE(hcclCommunicator, nullptr);

    void* fakePtr = (void*)0x12345678; // 给定非空的地址用于打桩控制条件
    u64 fakeSize = 50;
    auto deviceMem = DeviceMem(fakePtr, fakeSize);
    hcclCommunicator->zeroCopyLocalBuffer_ = std::move(deviceMem);
    EXPECT_EQ(hcclCommunicator->BuildZeroCopyParam(), HcclResult::HCCL_SUCCESS);
}

TEST_F(
    HcclImplTest,
    ut_HcclCommunicator_BuildZeroCopyParamFailed_When_ZeroCopyLocalBuffer_Is_Nullptr_Expect_Return_Success)
{
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());
    EXPECT_NE(hcclCommunicator, nullptr);

    auto deviceMem = DeviceMem(); // 生成空的地址用于打桩控制条件
    hcclCommunicator->zeroCopyLocalBuffer_ = std::move(deviceMem);
    EXPECT_EQ(hcclCommunicator->BuildZeroCopyParam(), HcclResult::HCCL_SUCCESS);
}

TEST_F(
    HcclImplTest, ut_HcclCommunicator_BuildZeroCopyParamFailed_When_GetRingBufferAddr_Failed_Expect_Return_E_INTERNAL)
{
    // 控制内部流程失败返回特定返回码
    MOCKER_CPP(&ZeroCopyMemoryAgent::GetRingBufferAddr).stubs().will(returnValue(HcclResult::HCCL_E_INTERNAL));
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());
    EXPECT_NE(hcclCommunicator, nullptr);

    void* fakePtr = (void*)0x12345678; // 给定非空的地址用于打桩控制条件
    u64 fakeSize = 50;
    auto deviceMem = DeviceMem(fakePtr, fakeSize);
    hcclCommunicator->zeroCopyLocalBuffer_ = std::move(deviceMem);
    EXPECT_EQ(hcclCommunicator->BuildZeroCopyParam(), HcclResult::HCCL_E_INTERNAL);
}
