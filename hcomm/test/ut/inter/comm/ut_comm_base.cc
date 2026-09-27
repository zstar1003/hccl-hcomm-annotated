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
#include <stdio.h>

#include "hccl/base.h"
#include <hccl/hccl_types.h>

#define private public
#define protected public
#include "comm_base_pub.h"
#include "comm_star_pub.h"
#include "network_manager_pub.h"
#undef protected
#undef private
#include "sal.h"
#include <vector>

#include "llt_hccl_stub_pub.h"
#include "llt_hccl_stub.h"
#include <externalinput_pub.h>
#include "remote_notify.h"
#include "alg_template_base_pub.h"
#include "dlra_function.h"

using namespace std;
using namespace hccl;

class CommInnerTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        s32 ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, 0, &dispatcherPtr);
        if (ret != HCCL_SUCCESS)
            return;
        if (dispatcherPtr == nullptr)
            return;
        dispatcher = reinterpret_cast<DispatcherPub*>(dispatcherPtr);
        std::cout << "\033[36m--CommInnerTest SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase()
    {
        if (dispatcherPtr != nullptr) {
            s32 ret = HcclDispatcherDestroy(dispatcherPtr);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            dispatcherPtr = nullptr;
            dispatcher = nullptr;
        }
        std::cout << "\033[36m--CommInnerTest TearDown--\033[0m" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
    static HcclDispatcher dispatcherPtr;
    static DispatcherPub* dispatcher;
};
HcclDispatcher CommInnerTest::dispatcherPtr = nullptr;
DispatcherPub* CommInnerTest::dispatcher = nullptr;

typedef struct innerpara_struct {
    std::string collectiveId;
    u32 userRank;
    u32 user_rank_size;
    u32 rank;
    u32 rank_size;
    u32 devicePhyId;
    std::vector<s32> device_ids;
    std::vector<u32> device_ips;
    std::vector<u32> user_ranks;
    std::string tag;
    HcclDispatcher dispatcher;
    IntraExchanger* exchanger;
    std::vector<RankInfo> para_vector;
    std::shared_ptr<CommBase> comm_inner;
} innerpara_t;

TEST_F(CommInnerTest, ut_destructor_D0)
{
    std::string rootInfo = "test_collective";
    IntraExchanger exchanger{};
    std::vector<RankInfo> para_vector(1);

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    CommBase* comm_inner = new CommBase(
        rootInfo, 0, 1, 0, 1, para_vector, topoFlag, dispatcher, nullptr, netDevCtxMap, exchanger, DeviceMem(),
        DeviceMem(), true);

    delete comm_inner;
}

TEST_F(CommInnerTest, ut_gget_transport_by_rank_err)
{
    std::string rootInfo = "test_collective";
    IntraExchanger exchanger{};
    std::vector<RankInfo> para_vector(1);

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;

    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    CommBase* comm_inner = new CommBase(
        rootInfo, 0, 1, 0, 1, para_vector, topoFlag, dispatcher, nullptr, netDevCtxMap, exchanger, DeviceMem(),
        DeviceMem(), true);

    std::shared_ptr<Transport> link = comm_inner->GetTransportByRank(3);
    EXPECT_EQ(link, nullptr);

    delete comm_inner;
}

TEST_F(CommInnerTest, ut_get_rank_by_userrank)
{
    s32 ret = HCCL_SUCCESS;

    u32 userRank = 1;
    u32 user_rank_size = 5;

    RankInfo tmp_para_1;
    RankInfo tmp_para_2;
    RankInfo tmp_para_3;

    tmp_para_1.userRank = userRank;
    tmp_para_2.userRank = userRank + 1;
    tmp_para_3.userRank = userRank - 1;

    s32 devicePhyId = 0;
    ret = hrtGetDevice(&devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    tmp_para_1.devicePhyId = devicePhyId;
    tmp_para_2.devicePhyId = devicePhyId;
    tmp_para_3.devicePhyId = devicePhyId;

    tmp_para_1.serverIdx = 0;
    tmp_para_1.serverId = "10.21.78.208";
    tmp_para_2.serverIdx = 0;
    tmp_para_2.serverId = "10.21.78.208";
    tmp_para_3.serverIdx = 1;
    tmp_para_3.serverId = "10.21.78.209";

    tmp_para_1.nicIp.push_back(HcclIpAddress("10.21.78.207"));
    tmp_para_2.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    tmp_para_3.nicIp.push_back(HcclIpAddress("10.21.78.209"));

    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para_1);
    para_vector.push_back(tmp_para_2);
    para_vector.push_back(tmp_para_3);

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;
    IntraExchanger exchanger{};

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    CommBase* comm_inner = new CommBase(
        collective_id_tmp, userRank, user_rank_size, 0, 1, para_vector, topoFlag, dispatcher, nullptr, netDevCtxMap,
        exchanger, DeviceMem::alloc(1024), DeviceMem::alloc(1024), true);

    ret = comm_inner->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 rank = INVALID_VALUE_RANKID;
    ret = comm_inner->GetRankByUserRank(5, rank);
    EXPECT_EQ(rank, INVALID_VALUE_RANKID);

    ret = comm_inner->GetRankByUserRank(1, rank);
    EXPECT_EQ(rank, 0);

    ret = comm_inner->GetRankByUserRank(3, rank);
    EXPECT_EQ(rank, INVALID_VALUE_RANKID);

    delete comm_inner;
}

TEST_F(CommInnerTest, ut_get_userrank_by_rank)
{
    s32 ret = HCCL_SUCCESS;

    u32 userRank = 0;
    u32 user_rank_size = 5;

    RankInfo tmp_para;

    tmp_para.userRank = userRank;

    s32 devicePhyId = 0;
    ret = hrtGetDevice(&devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    tmp_para.devicePhyId = devicePhyId;

    tmp_para.serverIdx = 0;
    tmp_para.serverId = "10.21.78.208";
    tmp_para.nicIp.push_back(HcclIpAddress("10.21.78.208"));

    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para);

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;
    IntraExchanger exchanger{};

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    CommBase* comm_inner = new CommBase(
        collective_id_tmp, userRank, user_rank_size, userRank, 1, para_vector, topoFlag, dispatcher, nullptr,
        netDevCtxMap, exchanger, DeviceMem::alloc(1024), DeviceMem::alloc(1024), true);

    ret = comm_inner->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 user_rank = INVALID_VALUE_RANKID;
    ret = comm_inner->GetUserRankByRank(5, user_rank);
    EXPECT_EQ(user_rank, INVALID_VALUE_RANKID);

    ret = comm_inner->GetUserRankByRank(0, user_rank);
    EXPECT_EQ(user_rank, 0);

    delete comm_inner;
}

TEST_F(CommInnerTest, ut_get_userrank_by_rank_with_err)
{
    s32 ret = HCCL_SUCCESS;

    u32 userRank = 0;
    u32 user_rank_size = 5;

    RankInfo tmp_para;

    tmp_para.userRank = userRank;

    s32 devicePhyId = 0;
    ret = hrtGetDevice(&devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    tmp_para.devicePhyId = devicePhyId;

    tmp_para.serverIdx = 0;
    tmp_para.serverId = "10.21.78.208";
    tmp_para.nicIp.push_back(HcclIpAddress("10.21.78.208"));

    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para);

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;
    IntraExchanger exchanger{};

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    CommBase* comm_inner = new CommBase(
        collective_id_tmp, userRank, user_rank_size, 0, 1, para_vector, topoFlag, dispatcher, nullptr, netDevCtxMap,
        exchanger, DeviceMem::alloc(1024), DeviceMem::alloc(1024), true);

    ret = comm_inner->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 user_rank = INVALID_VALUE_RANKID;
    ret = comm_inner->GetUserRankByRank(5, user_rank);
    EXPECT_EQ(user_rank, INVALID_VALUE_RANKID);

    delete comm_inner;
}

TEST_F(CommInnerTest, ut_SetTransportType)
{
    s32 ret = HCCL_SUCCESS;

    u32 userRank = 0;
    u32 user_rank_size = 5;

    RankInfo tmp_para;
    RankInfo tmp_para1;

    tmp_para.userRank = userRank;
    tmp_para1.userRank = 1;

    s32 devicePhyId = 0;
    ret = hrtGetDevice(&devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    tmp_para.devicePhyId = devicePhyId;
    tmp_para1.devicePhyId = devicePhyId;

    tmp_para.serverIdx = 0;
    tmp_para.serverId = "10.21.78.208";
    tmp_para1.serverIdx = 1;
    tmp_para1.serverId = "10.21.78.206";
    tmp_para.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    tmp_para1.nicIp.push_back(HcclIpAddress("10.21.78.206"));
    tmp_para.deviceType = DevType::DEV_TYPE_310P1;
    tmp_para1.deviceType = DevType::DEV_TYPE_310P1;

    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para);
    para_vector.push_back(tmp_para1);

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;
    IntraExchanger exchanger{};

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    CommBase* comm_inner = new CommBase(
        collective_id_tmp, userRank, user_rank_size, 0, 2, para_vector, topoFlag, dispatcher, nullptr, netDevCtxMap,
        exchanger, DeviceMem::alloc(1024), DeviceMem::alloc(1024), true);

    ret = comm_inner->SetTransportType(1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete comm_inner;
}

TEST_F(CommInnerTest, ut_TransportInit)
{
    MOCKER_CPP(&CommBase::SetTransportType).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    s32 ret = HCCL_SUCCESS;

    u32 userRank = 0;
    u32 user_rank_size = 5;

    RankInfo tmp_para;

    tmp_para.userRank = userRank;

    s32 devicePhyId = 0;
    ret = hrtGetDevice(&devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    tmp_para.devicePhyId = devicePhyId;

    tmp_para.serverIdx = 0;
    tmp_para.serverId = "10.21.78.208";
    tmp_para.nicIp.push_back(HcclIpAddress("10.21.78.208"));

    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para);

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;
    IntraExchanger exchanger{};

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    ConstructNetDevCtx(
        netDevCtxMap, NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId, devicePhyId, NicType::DEVICE_NIC_TYPE,
        tmp_para.nicIp[0]);
    CommBase* comm_inner = new CommBase(
        collective_id_tmp, userRank, user_rank_size, userRank, 1, para_vector, topoFlag, dispatcher, nullptr,
        netDevCtxMap, exchanger, DeviceMem::alloc(1024), DeviceMem::alloc(1024), true);

    MachinePara machinePara;
    HcclIpAddress invalidIp;
    machinePara.localIpAddr = tmp_para.nicIp[0];
    machinePara.remoteIpAddr = invalidIp;
    IpSocket socket;
    socket.listenedPort.insert(1);
    machinePara.deviceLogicId = 0;
    RaResourceInfo& raResourceInfo = NetworkManager::GetInstance(machinePara.deviceLogicId).raResourceInfo_;
    raResourceInfo.nicSocketMap[invalidIp] = socket;
    machinePara.inputMem = DeviceMem::alloc(1024);
    machinePara.outputMem = DeviceMem::alloc(1024);

    comm_inner->interSocketManager_.reset(new (std::nothrow)
                                              HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    comm_inner->interSocketManager_->ServerInit(netDevCtxMap[tmp_para.nicIp[0]], 18000);

    ret = comm_inner->TransportInit(0, machinePara);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    CommBase* comm_star = new CommStar(
        collective_id_tmp, userRank, user_rank_size, userRank, 1, topoFlag, dispatcher, nullptr, netDevCtxMap,
        exchanger, para_vector, DeviceMem::alloc(1024), DeviceMem::alloc(1024), true);
    comm_star->transportType_[0] = TransportType::TRANS_TYPE_RESERVED;
    comm_star->isHaveCpuRank_ = true;
    comm_star->interSocketManager_.reset(new (std::nothrow)
                                             HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    comm_star->interSocketManager_->ServerInit(netDevCtxMap[tmp_para.nicIp[0]], 16666);
    ret = comm_star->TransportInit(0, machinePara);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    raResourceInfo.nicSocketMap.erase(invalidIp);

    comm_inner->interSocketManager_->ServerDeInit(netDevCtxMap[tmp_para.nicIp[0]], 0);
    comm_star->interSocketManager_->ServerDeInit(netDevCtxMap[tmp_para.nicIp[0]], 16666);
    DeConstructNetDevCtx(netDevCtxMap, NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId, devicePhyId);
    delete comm_inner;
    delete comm_star;
    GlobalMockObject::verify();
}

TEST_F(CommInnerTest, ut_AiCpuNotifyDataGet)
{
    HcclResult ret = HCCL_SUCCESS;
    std::chrono::milliseconds kdefaultTimeout = std::chrono::seconds(120);
    MachinePara linkPara;
    std::shared_ptr<Transport> link(
        new Transport(new (std::nothrow) TransportBase(nullptr, nullptr, linkPara, kdefaultTimeout)));

    std::unique_ptr<NotifyPool> notifyPool = nullptr;
    notifyPool.reset(new (std::nothrow) NotifyPool());
    EXPECT_NE(notifyPool, nullptr);
    ret = notifyPool->Init(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string tag = "ut_AiCpuNotifyDataGet";
    ret = notifyPool->RegisterOp(tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::shared_ptr<LocalIpcNotify> localNotify = nullptr;
    std::shared_ptr<RemoteNotify> remoteNotify = nullptr;
    s32 pid = 0;
    ret = SalGetBareTgid(&pid); // 当前进程id
    RemoteRankInfo info(0, 0, pid);
    ret = notifyPool->Alloc(tag, info, localNotify, NotifyLoadType::DEVICE_NOTIFY);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
    ret = localNotify->Serialize(data);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remoteNotify.reset(new (std::nothrow) RemoteNotify());

    ret = remoteNotify->Init(data);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = remoteNotify->Open();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    link->pimpl_->remoteSendDoneDeviceNotify_ = remoteNotify;
    link->pimpl_->localSendDoneDeviceNotify_ = localNotify;
    link->pimpl_->remoteSendReadyDeviceNotify_ = remoteNotify;
    link->pimpl_->localSendReadyDeviceNotify_ = localNotify;
    HcclSignalInfo tmp;

    MOCKER(rtGetNotifyAddress).stubs().will(returnValue(RT_ERROR_NONE));

    MOCKER(aclrtGetNotifyId).stubs().will(returnValue(ACL_SUCCESS));

    ret = link->GetTxAckDevNotifyInfo(tmp);
    EXPECT_EQ(tmp.devId, 1);
    EXPECT_EQ(tmp.tsId, 3);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    tmp.devId = 0;
    tmp.tsId = 0;
    ret = link->GetRxAckDevNotifyInfo(tmp);
    EXPECT_EQ(tmp.devId, 1);
    EXPECT_EQ(tmp.tsId, 3);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    tmp.devId = 0;
    tmp.tsId = 0;
    ret = link->GetTxDataSigleDevNotifyInfo(tmp);
    EXPECT_EQ(tmp.devId, 1);
    EXPECT_EQ(tmp.tsId, 3);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    tmp.devId = 0;
    tmp.tsId = 0;
    ret = link->GetRxDataSigleDevNotifyInfo(tmp);
    EXPECT_EQ(tmp.devId, 1);
    EXPECT_EQ(tmp.tsId, 3);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CommInnerTest, ut_get_intra_rank_ipinfo)
{
    std::string rootInfo = "test_collective";
    IntraExchanger exchanger{};

    RankInfo tmp_para;
    tmp_para.devicePhyId = 0;

    tmp_para.serverIdx = 0;
    tmp_para.serverId = "10.21.78.208";
    tmp_para.nicIp.push_back(HcclIpAddress("10.21.78.208"));

    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para);

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;

    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    CommBase* comm_inner = new CommBase(
        rootInfo, 0, 1, 0, 1, para_vector, topoFlag, dispatcher, nullptr, netDevCtxMap, exchanger, DeviceMem(),
        DeviceMem(), true);

    std::vector<u32> dstIntraVec;
    HcclIpAddress localIP(0);
    std::map<u32, HcclRankLinkInfo> dstServerMap;
    std::map<u32, HcclRankLinkInfo> dstClientMap;
    dstIntraVec.push_back(0);
    HcclResult ret = comm_inner->GetIntraRankIPInfo(dstIntraVec, localIP, dstServerMap, dstClientMap);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete comm_inner;
}

TEST_F(CommInnerTest, ut_create_dest_link_memorry_error)
{
    MOCKER_CPP(&CommBase::TransportInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_MEMORY));

    s32 ret = HCCL_SUCCESS;

    u32 userRank = 0;
    u32 user_rank_size = 5;

    RankInfo tmp_para;

    tmp_para.userRank = userRank;

    s32 devicePhyId = 0;
    ret = hrtGetDevice(&devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    tmp_para.devicePhyId = devicePhyId;

    tmp_para.serverIdx = 0;
    tmp_para.serverId = "10.21.78.208";
    tmp_para.nicIp.push_back(HcclIpAddress("10.21.78.208"));

    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para);

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;
    IntraExchanger exchanger{};

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;

    CommBase* comm_inner = new CommBase(
        collective_id_tmp, userRank, user_rank_size, userRank, 1, para_vector, topoFlag, dispatcher, nullptr,
        netDevCtxMap, exchanger, DeviceMem::alloc(1024), DeviceMem::alloc(1024), true);
    ErrContextPub error_context;
    error_context.work_stream_id = 1234567890;
    std::vector<std::shared_ptr<HcclSocket>> sockets;
    ret = comm_inner->CreateDestLink(
        error_context, MachineType::MACHINE_SERVER_TYPE, "10.21.78.208", 0, "threadStr", sockets);
    EXPECT_EQ(ret, HCCL_E_MEMORY);
    delete comm_inner;
    GlobalMockObject::verify();
}

TEST_F(CommInnerTest, ut_PrepareDeInit_fail)
{
    s32 ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 device_id = 0;
    s32 ref = 0;
    hrtSetDevice(device_id);

    NetworkManager::GetInstance(device_id).hostNicInitRef_.Unref();

    ret = NetworkManager::GetInstance(device_id).PrepareDeInit(ref, NICDeployment::NIC_DEPLOYMENT_HOST);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    GlobalMockObject::verify();
}

TEST_F(CommInnerTest, ut_HeterogDeinit_fail)
{
    s32 ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 port = 16666;
    u32 device_id = 0;
    HcclIpAddress ipAddr("192.168.0.11");

    hrtSetDevice(device_id);
    ret = NetworkManager::GetInstance(device_id).Init(NICDeployment::NIC_DEPLOYMENT_HOST);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    NetworkManager::GetInstance(device_id).hostNicInitRef_.Unref();

    ret = NetworkManager::GetInstance(device_id).HeterogDeinit(device_id, ipAddr, port);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    GlobalMockObject::verify();
}
