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
#include <stdlib.h>
#include <assert.h>
#include <securec.h>
#include <ifaddrs.h>
#include <sys/socket.h>
#include <netdb.h>

#include <sys/types.h>
#include <stddef.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "dlra_function.h"

#define private public
#define protected public
#include "hccl_communicator.h"
#include "hccl/hccl/hccl_comm/inc/hccl_impl.h"
#include "hccl_comm_pub.h"
#include "transport_heterog_ibv_pub.h"
#undef protected
#undef private

#include <hccl/hccl_comm.h>
#include <hccl/hccl_inner.h>
#include <hccl/hccl_ex.h>
#include "llt_hccl_stub_pub.h"
#include "llt_hccl_stub_gdr.h"
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include "hccl/base.h"
#include "hccl/hcom.h"
#include <hccl/hccl_types.h>
#include "topoinfo_ranktableParser_pub.h"
#include "v80_rank_table.h"
#include "tsd/tsd_client.h"
#include "dltdt_function.h"
#include "externalinput_pub.h"

using namespace std;
using namespace hccl;

class MPI_TCP_SENDRECV_TEST : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        s32 ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, 0, &dispatcherPtr);
        if (ret != HCCL_SUCCESS)
            return;
        if (dispatcherPtr == nullptr)
            return;
        dispatcher = reinterpret_cast<DispatcherPub*>(dispatcherPtr);
        std::cout << "MPI_TCP_SENDRECV_TEST SetUP" << std::endl;
    }
    static void TearDownTestCase()
    {
        if (dispatcherPtr != nullptr) {
            s32 ret = HcclDispatcherDestroy(dispatcherPtr);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            dispatcherPtr = nullptr;
            dispatcher = nullptr;
        }
        std::cout << "MPI_TCP_SENDRECV_TEST TearDown" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        MPI_Barrier(MPI_COMM_WORLD);
        static s32 call_cnt = 0;
        string name = std::to_string(call_cnt++) + "_" + __PRETTY_FUNCTION__;
        DlRaFunction::GetInstance().DlRaFunctionInit();
        ra_set_shm_name(name.c_str());
        ra_set_test_type(1, "MPI_TCP_SENDRECV_TEST");
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A TestCase SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "A TestCase TearDown" << std::endl;
    }
    static HcclDispatcher dispatcherPtr;
    static DispatcherPub* dispatcher;
};
HcclDispatcher MPI_TCP_SENDRECV_TEST::dispatcherPtr = nullptr;
DispatcherPub* MPI_TCP_SENDRECV_TEST::dispatcher = nullptr;

TEST_F(MPI_TCP_SENDRECV_TEST, ut_mpi_tcp_send_link)
{
    SetTcpMode(true);
    HcclResult ret;
    string data = "MPI_TCP_SENDRECV_TEST";
    string* buffer = &data;
    u32 peerRank = 0;
    u32 tag = 0;
    void* request;
    string collectiveId = "192.168.3.3-9527-0001";

    MOCKER_CPP(&TcpSendThreadPool::AddSendTask).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportHeterogIbv::GetState)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(ConnState::CONN_STATE_COMPLETE));

    hcclImpl impl;
    impl.commFactory_.reset(new (std::nothrow) CommFactory(collectiveId, 0, 2, dispatcher));
    impl.ranksPort_.push_back(1);
    impl.userRank_ = 0;
    impl.userRankSize_ = 1;
    impl.collectiveId_ = collectiveId;
    impl.pReqInfosMem_.reset(new (std::nothrow) LocklessRingMemoryAllocate<HcclRequestInfo>(4096));
    impl.pReqInfosMem_->Init();

    std::string commPair = std::to_string(peerRank) + "_" + std::to_string(impl.userRank_);
    unique_ptr<TransportHeterogIbv> link(
        new (nothrow) TransportHeterogIbv("192.168.3.3-9527-0001", 0, 0, 0, nullptr, nullptr, &impl));

    CommRankTagKey commRankTagInfo(0, peerRank, tag);
    std::unique_lock<std::mutex> lock(g_transportStorageMutex);
    g_commRankTagMap[commRankTagInfo].hcclHandle = link.get();
    lock.unlock();

    ret = impl.Isend(buffer, sizeof(data), HCCL_DATA_TYPE_INT8, peerRank, tag, request);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    impl.pReqInfosMem_->Free(reinterpret_cast<HcclRequestInfo*>(request));
    std::unique_lock<std::mutex> lockend(g_transportStorageMutex);
    g_commRankTagMap.erase(g_commRankTagMap.begin(), g_commRankTagMap.end());
    SetTcpMode(false);
    GlobalMockObject::verify();
}

TEST_F(MPI_TCP_SENDRECV_TEST, ut_mpi_tcp_improbe)
{
    SetTcpMode(true);
    HcclResult ret;
    u32 commId = 0;
    u32 flag = 0;
    u32 peerRank = 0;
    u32 tag = 0;
    string collectiveId = "192.168.3.3-9527-0001";
    HcclMessage* msg;
    HcclStatus* status;

    MOCKER_CPP(&TransportHeterogIbv::GetState)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(ConnState::CONN_STATE_COMPLETE));

    hcclImpl impl;
    impl.commFactory_.reset(new (std::nothrow) CommFactory(collectiveId, 0, 2, dispatcher));
    impl.ranksPort_.push_back(1);
    impl.userRank_ = 0;
    impl.userRankSize_ = 1;
    impl.collectiveId_ = collectiveId;
    unique_ptr<TransportHeterogIbv> link(
        new (nothrow) TransportHeterogIbv("192.168.3.3-9527-0001", 0, 0, 0, nullptr, nullptr, &impl));
    std::string commPair = std::to_string(peerRank) + "_" + std::to_string(impl.userRank_);
    CommRankTagKey commRankTagInfo(0, peerRank, tag);
    std::unique_lock<std::mutex> lock(g_transportStorageMutex);
    g_commRankTagMap[commRankTagInfo].hcclHandle = link.get();
    lock.unlock();

    ret = impl.Improbe(peerRank, tag, flag, msg, status);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    std::unique_lock<std::mutex> endlock(g_transportStorageMutex);
    g_commRankTagMap.erase(g_commRankTagMap.begin(), g_commRankTagMap.end());
    SetTcpMode(false);
}

TEST_F(MPI_TCP_SENDRECV_TEST, ut_mpi_tcp_imrecv)
{
    SetTcpMode(true);
    u32 commId = 0;
    u32 peerRank = 0;
    u32 tag = 0;
    string data = "MPI_TCP_SENDRECV_TEST";
    string* buffer = &data;
    void* request = nullptr;
    void** requestPtr = &request;

    hcclImpl impl;
    impl.pReqInfosMem_.reset(new (std::nothrow) LocklessRingMemoryAllocate<HcclRequestInfo>(4096));
    impl.pMsgInfosMem_.reset(new (std::nothrow) LocklessRingMemoryAllocate<HcclMessageInfo>(4096));
    impl.pReqInfosMem_->Init();
    impl.pMsgInfosMem_->Init();

    EnvelopeStatusInfo envelope;
    envelope.src_addr = reinterpret_cast<u64>(buffer);
    HcclMessageInfo* msgInfo = impl.pMsgInfosMem_->Alloc();
    msgInfo->srcRank = peerRank;
    msgInfo->tag = tag;
    msgInfo->comm = nullptr;
    msgInfo->envelope = envelope;
    void* msg = msgInfo;

    unique_ptr<TransportHeterogIbv> link(new (nothrow) TransportHeterogIbv(
        "192.168.3.3-9527-0001", 0, 0, 0, nullptr, nullptr, nullptr, nullptr, true, &impl));
    CommRankTagKey commRankTagInfo(commId, peerRank, tag);
    CommRankTagHash channel;
    channel.commId = commId;
    channel.dstRank = peerRank;
    channel.tag = tag;
    std::unique_lock<std::mutex> lock(g_transportStorageMutex);
    g_commRankTagMap[commRankTagInfo] = channel;
    g_commRankTagMap[commRankTagInfo].hcclHandle = link.get();
    lock.unlock();

    MOCKER_CPP(&TcpRecvTask::SetRecvEntry).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtEpollCtlMod).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    impl.Imrecv(buffer, sizeof(data), HCCL_DATA_TYPE_INT8, msg, requestPtr);
    impl.pReqInfosMem_->Free(reinterpret_cast<HcclRequestInfo*>(request));
    GlobalMockObject::verify();
    std::unique_lock<std::mutex> endlock(g_transportStorageMutex);
    g_commRankTagMap.erase(g_commRankTagMap.begin(), g_commRankTagMap.end());
    SetTcpMode(false);
}

TEST_F(MPI_TCP_SENDRECV_TEST, ut_mpi_tcp_TcpfdHandleCheck)
{
    SetTcpMode(true);

    MOCKER(halEschedSubmitEvent).stubs().with(mockcpp::any()).will(returnValue(1));

    MOCKER_CPP(&TransportHeterogIbv::GetState)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(ConnState::CONN_STATE_COMPLETE));

    unsigned int devId = 0;
    unsigned int grpId = 0;
    unsigned int eventId = 0;
    unsigned int subeventId = 0;
    gCompCounterEvent[eventId].counter = 1;

    EschedFinishCallback(devId, grpId, 1, subeventId);
    EschedFinishCallback(devId, grpId, eventId, subeventId);
    SetTcpMode(false);
    GlobalMockObject::verify();
}

TEST_F(MPI_TCP_SENDRECV_TEST, ut_mpi_tcp_tcpEschedFinishProcess)
{
    SetTcpMode(true);

    MOCKER(halEschedSubmitEvent).stubs().with(mockcpp::any()).will(returnValue(1));

    MOCKER_CPP(&TransportHeterogIbv::GetState)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(ConnState::CONN_STATE_COMPLETE));

    unsigned int devId = 0;
    unsigned int grpId = 0;
    unsigned int eventId = 0;
    unsigned int subeventId = 0;
    gCompCounterEvent[eventId].counter = 1;

    EschedFinishCallback(devId, grpId, 1, subeventId);
    EschedFinishCallback(devId, grpId, eventId, subeventId);
    SetTcpMode(false);
    GlobalMockObject::verify();
}

TEST_F(MPI_TCP_SENDRECV_TEST, ut_mpi_tcp_sendWork)
{
    MOCKER_CPP(&TransportHeterogIbv::TcpSendImm).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret;
    hcclImpl impl;
    TcpSendThreadPool tcpThreadPool(&impl, 0);
    TransportHeterogIbv link("192.168.3.3-9527-0001", 0, 0, 0, nullptr, nullptr, nullptr, nullptr, true, &impl);

    HcclRequestInfo request;
    SREntry entry;
    int data = 32;
    entry.buffer = &data;
    entry.userHandle = &request;
    entry.hcclHandle = &link;
    gCompCounterEvent[HCCL_EVENT_SEND_COMPLETION_MSG].flag.clear();

    MOCKER_CPP(&TransportHeterogIbv::GetState)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(ConnState::CONN_STATE_COMPLETE));

    MOCKER(halEschedSubmitEvent).stubs().with(mockcpp::any()).will(returnValue(1));

    ret = tcpThreadPool.SendWork(entry);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    SetTcpMode(false);
    GlobalMockObject::verify();
}

TEST_F(MPI_TCP_SENDRECV_TEST, ut_mpi_tcp_TcpSearchRequestStatus)
{
    MOCKER_CPP(&TransportHeterogIbv::GetState)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(ConnState::CONN_STATE_COMPLETE));

    HcclResult ret;
    uint32_t compCount = 0;
    HcclStatus compState;
    hcclImpl impl;
    impl.pReqInfosMem_.reset(new (std::nothrow) LocklessRingMemoryAllocate<HcclRequestInfo>(4096));
    impl.pReqInfosMem_->Init();
    gCompCounterEvent[HCCL_EVENT_SEND_COMPLETION_MSG].counter++;
    gCompCounterEvent[HCCL_EVENT_RECV_COMPLETION_MSG].counter++;

    TransportHeterogIbv link("192.168.3.3-9527-0001", 0, 0, 0, nullptr, nullptr, nullptr, nullptr, true, &impl);

    HcclRequestInfo* request = impl.pReqInfosMem_->Alloc();
    request->requestType = HcclRequestType::HCCL_REQUEST_SEND;
    request->status = 0;
    request->hcclHandle = &link;
    ret = impl.TcpSearchRequestStatus(request, &compCount, compState);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    request = impl.pReqInfosMem_->Alloc();
    request->requestType = HcclRequestType::HCCL_REQUEST_RECV;
    request->status = 0;
    request->hcclHandle = &link;
    ret = impl.TcpSearchRequestStatus(request, &compCount, compState);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    request = impl.pReqInfosMem_->Alloc();
    request->requestType = HcclRequestType::HCCL_REQUEST_INVAIL;
    request->status = 0;
    request->hcclHandle = &link;
    ret = impl.TcpSearchRequestStatus(request, &compCount, compState);
    EXPECT_EQ(ret, HCCL_E_PARA);
    impl.pReqInfosMem_->Free(request);
    GlobalMockObject::verify();
}
