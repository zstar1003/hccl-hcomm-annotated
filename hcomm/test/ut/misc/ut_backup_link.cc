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
#include "llt_hccl_stub_sal_pub.h"
#include "adapter_tdt.h"
#include "adapter_hal.h"
#include "dlra_function.h"

#define private public
#define protected public
#include "hccl_communicator.h"
#include "transport_manager.h"
#include "network_manager_pub.h"
#include "framework/aicpu_communicator.h"
#include "peterson_lock.h"
#include "coll_all_reduce_ring_executor.h"
#include "framework/aicpu_hdc.h"
#include "platform/resource/socket/hccl_network.h"
#undef private

using namespace std;
using namespace hccl;

class BackupLinkTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        DlRaFunction::GetInstance().DlRaFunctionInit();
        cout << "\033[36m--BackupLinkTest SetUP--\033[0m" << endl;
    }
    static void TearDownTestCase() { cout << "\033[36m--BackupLinkTest TearDown--\033[0m" << endl; }
    virtual void SetUp()
    {
        cout << "A Test SetUP" << endl;
        setenv("HCCL_OP_RETRY_ENABLE", "L0:1,L1:1,L2:1", 1);
        DevType deviceType = DevType::DEV_TYPE_910_93;
        MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        unsetenv("HCCL_OP_RETRY_ENABLE");
        cout << "A Test TearDown" << endl;
    }
};

static void TestConstructParam_SurperPod(HcclCommParams& params, RankTable_t& rankTable)
{
    string commId = "comm ";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = 2;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.commWorkMode = WorkMode::HCCL_MODE_NORMAL;
    params.deviceType = DevType::DEV_TYPE_910_93;

    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(2);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.0.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankVec[0].superPodId = "192.168.0.103";
    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr2(1711319232);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
    rankVec[1].serverIdx = 1;
    rankVec[1].serverId = "192.168.0.102";
    rankVec[1].superPodId = "192.168.0.104";
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = 2;
    rankTable.serverNum = 2;
}

#if 1
TEST_F(BackupLinkTest, ut_DestroyAlgResource)
{
    TransportRequest transportReq1;
    transportReq1.isValid = true;
    transportReq1.remoteUserRank = 0;
    transportReq1.remoteUserRank = 2;
    TransportRequest transportReq2;
    transportReq2.isValid = true;
    transportReq2.remoteUserRank = 0;
    transportReq2.remoteUserRank = 4;

    SingleSubCommTransport singleTrans;
    singleTrans.transportRequests.emplace_back(transportReq1);
    singleTrans.transportRequests.emplace_back(transportReq2);

    LevelNSubCommTransport levelTrans;
    levelTrans.emplace_back(singleTrans);

    AlgResourceResponse algResRsp;
    algResRsp.opTransportResponseBackUp.emplace_back(levelTrans);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_SurperPod(params, rankTable);
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    communicator->Init(params, rankTable);

    MOCKER_CPP(&HcclCommunicator::DestroyOpTransportResponse).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::IsEnableBackupLink).stubs().will(returnValue(true));

    communicator->DestroyAlgResource(algResRsp);
    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(BackupLinkTest, ut_DestroyNetworkResources)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_SurperPod(params, rankTable);
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    communicator->Init(params, rankTable);

    MOCKER_CPP(&HcclCommunicator::IsEnableBackupLink).stubs().will(returnValue(true));

    MOCKER_CPP(&NetworkManager::DeInit).stubs().will(returnValue(HCCL_SUCCESS));

    ret = communicator->DestroyNetworkResources();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(BackupLinkTest, ut_DeinitNic)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_SurperPod(params, rankTable);
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());
    std::unique_ptr<HcclSocketManager> socketManager(
        new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::IsEnableBackupLink).stubs().will(returnValue(true));

    MOCKER_CPP(&HcclSocketManager::ServerDeInit, HcclResult(HcclSocketManager::*)(const HcclNetDevCtx, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&NetDevContext::Deinit).stubs().will(returnValue(HCCL_SUCCESS));

    communicator->Init(params, rankTable);

    u32 devicePhyId = 0;
    HcclNetDevCtx vnicPortCtx;
    ret = HcclNetOpenDev(&vnicPortCtx, NicType::DEVICE_NIC_TYPE, devicePhyId, devicePhyId, HcclIpAddress(devicePhyId));
    EXPECT_EQ(ret, HCCL_SUCCESS);

    communicator->nicDeployment_ = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    communicator->netDevCtxMap_.insert(make_pair(HcclIpAddress(devicePhyId), vnicPortCtx));
    communicator->devBackupIpAddr_.push_back(HcclIpAddress(devicePhyId));
    communicator->socketManager_ = std::move(socketManager);

    ret = communicator->DeinitNic();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(BackupLinkTest, ut_GetRemoteRankList)
{
    HcclResult ret = HCCL_SUCCESS;

    TransportType transportType = TransportType::TRANS_TYPE_ROCE;
    MOCKER_CPP(&TransportManager::GetTransportType).stubs().will(returnValue(transportType));

    TransportRequest transportReq1;
    transportReq1.isValid = true;
    transportReq1.remoteUserRank = 0;
    transportReq1.remoteUserRank = 2;
    TransportRequest transportReq2;
    transportReq2.isValid = true;
    transportReq2.remoteUserRank = 0;
    transportReq2.remoteUserRank = 4;

    SingleSubCommTransport singleTrans;
    singleTrans.transportRequests.emplace_back(transportReq1);
    singleTrans.transportRequests.emplace_back(transportReq2);

    LevelNSubCommTransport levelTrans;
    levelTrans.emplace_back(singleTrans);

    OpCommTransport opTrans;
    opTrans.emplace_back(levelTrans);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_SurperPod(params, rankTable);
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    communicator->Init(params, rankTable);

    std::vector<u32> rankList;
    ret = communicator->transportManager_->GetRemoteRankList(opTrans, rankList, TransportType::TRANS_TYPE_ROCE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    communicator->ClearOpTransportResponseLinks(opTrans);
    std::string tag = "test";
    TransportIOMem transMem;
    ret = communicator->transportManager_->Alloc(tag, transMem, opTrans, true, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(BackupLinkTest, ut_GetIncreRemoteRankList)
{
    HcclResult ret = HCCL_SUCCESS;

    TransportType transportType = TransportType::TRANS_TYPE_ROCE;
    MOCKER_CPP(&TransportManager::GetTransportType).stubs().will(returnValue(transportType));

    TransportRequest transportReq1;
    transportReq1.isValid = true;
    transportReq1.remoteUserRank = 0;
    transportReq1.remoteUserRank = 2;
    TransportRequest transportReq2;
    transportReq2.isValid = true;
    transportReq2.remoteUserRank = 0;
    transportReq2.remoteUserRank = 4;

    SingleSubCommTransport singleTrans;
    singleTrans.transportRequests.emplace_back(transportReq1);
    singleTrans.transportRequests.emplace_back(transportReq2);

    LevelNSubCommTransport levelTrans;
    levelTrans.emplace_back(singleTrans);

    OpCommTransport opTrans;
    opTrans.emplace_back(levelTrans);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_SurperPod(params, rankTable);
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    communicator->Init(params, rankTable);

    std::vector<u32> rankList;
    ret = communicator->transportManager_->GetIncreRemoteRankList(opTrans, rankList, TransportType::TRANS_TYPE_ROCE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(BackupLinkTest, ut_SetMachinePara)
{
    HcclResult ret = HCCL_SUCCESS;

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_SurperPod(params, rankTable);
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    communicator->Init(params, rankTable);

    std::vector<std::shared_ptr<HcclSocket>> socketList;
    DeviceMem inMem = DeviceMem::alloc(128);
    DeviceMem outMem = DeviceMem::alloc(128);
    DeviceMem expMem = DeviceMem::alloc(128);
    MachinePara machinePara;

    MOCKER(hrtGetPairDevicePhyId).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    RankInfo loaclRankInfo;
    RankInfo remoteRankInfo;

    MOCKER(hrtRaGetInterfaceVersion).stubs().will(returnValue(HCCL_SUCCESS));

    HcclNetDevCtx netDevCtx;
    ret = communicator->transportManager_->SetMachinePara(
        "test", MachineType::MACHINE_SERVER_TYPE, "192.0.0.0", 0, 0, LinkMode::LINK_DUPLEX_MODE, socketList, inMem,
        outMem, expMem, 1, 1, 0, 0, 132, 4, machinePara, loaclRankInfo, remoteRankInfo, netDevCtx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(BackupLinkTest, ut_RefreshAlgResponseTransportRes)
{
    HcclResult ret = HCCL_SUCCESS;

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommAicpu::CalcResRequest).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommAicpu::CreateLink).stubs().will(returnValue(HCCL_SUCCESS));

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_SurperPod(params, rankTable);
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());
    communicator->Init(params, rankTable);

    // std::unique_ptr<hcclImpl> &impl = communicator->implAlg_->pimpl_;
    // std::unique_ptr<TopoMatcher> &topoMatcher = communicator->implAlg_->topoMatcher_;
    // std::unique_ptr<CollExecutorBase> executor(new CollAllReduceRingExecutor(impl->dispatcher_, topoMatcher));

    HcclCommAicpu* hcclCommAicpu = new HcclCommAicpu();
    AlgResourceResponse tempAlgResRep;
    hcclCommAicpu->resMap_["test"] = tempAlgResRep;
    DeviceMem devMem = DeviceMem::alloc(128);
    HcclOpResParam commParam;
    commParam.lockAddr = reinterpret_cast<u64>(devMem.ptr());
    hcclCommAicpu->InitHostDeviceLock(&commParam);

    OpParam opParam;
    opParam.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    AlgResourceResponse algResResponse;
    std::map<u32, bool> remoteRankPortMap;
    bool isChangeLinkFlag = true;
    ret = hcclCommAicpu->RefreshAlgResponseTransportRes(
        "test", algResResponse, remoteRankPortMap, isChangeLinkFlag, &commParam, opParam);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    isChangeLinkFlag = false;
    ret = hcclCommAicpu->RefreshAlgResponseTransportRes(
        "test", algResResponse, remoteRankPortMap, isChangeLinkFlag, &commParam, opParam);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete hcclCommAicpu;
    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(BackupLinkTest, ut_HcclOpExecChangeLinkProcess)
{
    HcclResult ret = HCCL_SUCCESS;

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ChangeLinkInfo changeLinkInfo;
    changeLinkInfo.remoteRankNum = 1;
    changeLinkInfo.remoteRankList[0] = 0;
    changeLinkInfo.isUseDefaultPort[0] = true;
    MOCKER_CPP(&AicpuHdc::GetOpExecChangeLink)
        .stubs()
        .with(mockcpp::any(), outBound(changeLinkInfo))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommAicpu::RefreshAlgResponseTransportRes).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(
        &HcclCommAicpu::UpdateOpExecStatus,
        HcclResult(HcclCommAicpu::*)(HcclOpExecFSM&, KfcStatus, KfcError&, uint32_t))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_SurperPod(params, rankTable);
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());
    communicator->Init(params, rankTable);

    // std::unique_ptr<hcclImpl> &impl = communicator->implAlg_->pimpl_;
    // std::unique_ptr<TopoMatcher> &topoMatcher = communicator->implAlg_->topoMatcher_;
    // std::unique_ptr<CollExecutorBase> executor(new CollAllReduceRingExecutor(impl->dispatcher_, topoMatcher));

    HcclCommAicpu* hcclCommAicpu = new HcclCommAicpu();
    OpParam opParam;
    opParam.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    AlgResourceResponse algResResponse;
    HcclOpResParam commParam;
    HcclOpExecFSM state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_CHANGE_LINK;
    KfcError errorCode = KfcError::kRdma;
    uint32_t retryCnt = 0;
    hcclCommAicpu->retryEnable_ = true;
    ret = hcclCommAicpu->HcclOpExecChangeLinkProcess(
        "test", state, errorCode, retryCnt, algResResponse, &commParam, opParam);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete hcclCommAicpu;
    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(BackupLinkTest, ut_CloseHccpProcess)
{
    HcclResult ret = HCCL_SUCCESS;
    MOCKER(hrtCloseNetService).stubs().will(returnValue(HCCL_SUCCESS));

    ret = NetworkManager::GetInstance(0).CloseHccpProcess();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(BackupLinkTest, ut_TsdProcessOpen)
{
    HcclResult ret = HCCL_SUCCESS;
    MOCKER(hrtOpenNetService).stubs().will(returnValue(HCCL_SUCCESS));

    // 老版本driver（hccp备进程日志无法获取）
    ret = NetworkManager::GetInstance(0).TsdProcessOpen(false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 新版本driver（主备hccp进程日志可以获取）
    s32 apiVersion = 0x72318; // MAJOR:0x07, MINOR:0x23, PATCH:0x18 新版本号
    MOCKER(hrtHalGetAPIVersion).stubs().with(outBound(apiVersion)).will(returnValue(ret));
    MOCKER(hrtGetPairDevicePhyId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = NetworkManager::GetInstance(0).TsdProcessOpen(false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(BackupLinkTest, ut_ReAllocTransportResource)
{
    /*
     *  借轨重新刷新资源 ReAllocTransportResource 用例
     */
    HcclResult ret = HCCL_SUCCESS;
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommAicpu::CalcResRequest).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommAicpu::CreateLink).stubs().will(returnValue(HCCL_SUCCESS));

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_SurperPod(params, rankTable);
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());
    communicator->Init(params, rankTable);

    // std::unique_ptr<hcclImpl> &impl = communicator->implAlg_->pimpl_;
    // std::unique_ptr<TopoMatcher> &topoMatcher = communicator->implAlg_->topoMatcher_;
    // std::unique_ptr<CollExecutorBase> executor(new CollAllReduceRingExecutor(impl->dispatcher_, topoMatcher));

    HcclCommAicpu* hcclCommAicpu = new HcclCommAicpu();
    AlgResourceResponse tempAlgResRep;
    hcclCommAicpu->resMap_["test"] = tempAlgResRep;
    DeviceMem devMem = DeviceMem::alloc(128);
    HcclOpResParam commParam;
    commParam.lockAddr = reinterpret_cast<u64>(devMem.ptr());
    hcclCommAicpu->InitHostDeviceLock(&commParam);

    OpParam opParam;
    AlgResourceResponse algResResponse;
    opParam.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    algResResponse.opTransportResponse.resize(1);
    algResResponse.opTransportResponse[COMM_LEVEL0].resize(1);
    algResResponse.opTransportResponse[COMM_LEVEL0][0].links.resize(1);
    algResResponse.opTransportResponse[COMM_LEVEL0][0].transportRequests.resize(1);
    algResResponse.opTransportResponse[COMM_LEVEL0][0].transportRequests[0].isValid = true;
    algResResponse.opTransportResponse[COMM_LEVEL0][0].transportRequests[0].isUsedRdma = true;
    std::map<u32, bool> remoteRankPortMap;

    ret = hcclCommAicpu->CleanRoceResource("test", algResResponse, remoteRankPortMap, opParam);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u8 tmp = 0;
    opParam.BatchSendRecvDataDes.isDirectRemoteRank = &tmp;
    ret = hcclCommAicpu->ReAllocTransportResource("test", algResResponse, remoteRankPortMap, &commParam, opParam);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete hcclCommAicpu;
    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(BackupLinkTest, ut_InitRaResource_notSupportChangelink)
{
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());
    std::unique_ptr<HcclSocketManager> socketManager(
        new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    u32 devicePhyId = 0;
    HcclNetDevCtx vnicPortCtx;
    HcclResult ret
        = HcclNetOpenDev(&vnicPortCtx, NicType::DEVICE_NIC_TYPE, devicePhyId, devicePhyId, HcclIpAddress(devicePhyId));
    EXPECT_EQ(ret, HCCL_SUCCESS);
    communicator->devicePhyId_ = devicePhyId;
    communicator->netDevCtxMap_.insert(make_pair(HcclIpAddress(devicePhyId), vnicPortCtx));
    communicator->socketManager_ = std::move(socketManager);
    communicator->userRankSize_ = 2;
    communicator->nicDeployment_ = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    communicator->isHaveCpuRank_ = false;

    MOCKER(IsHostUseDevNic).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::IsEnableBackupLink).stubs().will(returnValue(true));
    MOCKER(Is310PDevice).stubs().will(returnValue(true));
    MOCKER_CPP(&HcclCommunicator::InitNic).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetPairDevicePhyId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    // rts无法访问备用die（客户自定义） -> 不支持借轨
    MOCKER(hrtGetDeviceIndexByPhyId).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_RUNTIME));
    // 网络资源初始化失败（不初始化备用资源）
    ret = communicator->InitRaResource();
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    HcclNetCloseDev(vnicPortCtx);
    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(BackupLinkTest, ut_InitRaResource_SupportChangelink)
{
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());
    std::unique_ptr<HcclSocketManager> socketManager(
        new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    u32 devicePhyId = 0;
    HcclNetDevCtx vnicPortCtx;
    HcclResult ret
        = HcclNetOpenDev(&vnicPortCtx, NicType::DEVICE_NIC_TYPE, devicePhyId, devicePhyId, HcclIpAddress(devicePhyId));
    EXPECT_EQ(ret, HCCL_SUCCESS);
    communicator->devicePhyId_ = devicePhyId;
    communicator->netDevCtxMap_.insert(make_pair(HcclIpAddress(devicePhyId), vnicPortCtx));
    communicator->socketManager_ = std::move(socketManager);
    communicator->userRankSize_ = 2;
    communicator->nicDeployment_ = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    communicator->isHaveCpuRank_ = false;

    MOCKER(IsHostUseDevNic).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::IsEnableBackupLink).stubs().will(returnValue(true));
    MOCKER(Is310PDevice).stubs().will(returnValue(true));
    MOCKER_CPP(&HcclCommunicator::InitNic).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetPairDevicePhyId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    // rts可以访问备用die（默认） -> 支持借轨
    MOCKER(hrtGetDeviceIndexByPhyId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    // 网络资源初始化成功（不初始化备用资源）
    ret = communicator->InitRaResource();
    ;
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}
#endif
