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

#include <memory>
#include <iostream>
#include <fstream>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_inner.h>
#include "llt_hccl_stub_pub.h"
#include "v80_rank_table.h"
#include "hccl/base.h"
#include <hccl/hccl_types.h>
#include "sal.h"
#include "llt_hccl_stub_gdr.h"
#include "network_manager_pub.h"
#include <externalinput_pub.h>
#include "tsd/tsd_client.h"
#include "dltdt_function.h"
#include "dlra_function.h"
#include "externalinput.h"
#include "adapter_rts.h"
#define private public
#define protected public
#include "hccl_socket.h"
#include "hccl_socket_manager.h"
#include "hccl_communicator.h"
#include "zero_copy/zero_copy_address_mgr.h"
#undef private
#undef protected
#include "socket/hccl_network.h"
#include <queue>
#include <mutex>
using namespace std;
using namespace hccl;

s32 stub_SocketManagerTest_hrtRaSocketNonBlockSendHB(
    const FdHandle fdHandle, const void* data, u64 size, u64* sent_size)
{
    *sent_size = size;
    return 0;
}

template <typename T>
HcclResult ConstructData(u8*& exchangeDataPtr, u32& exchangeDataBlankSize, T& value)
{
    CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, &value, sizeof(T)));
    exchangeDataPtr += sizeof(T);
    exchangeDataBlankSize -= sizeof(T);
    return HCCL_SUCCESS;
}

static std::queue<std::vector<u8>> exchangeDataForAck_;
static std::unordered_map<u32, std::array<uint64_t, 2 * 1024 * 1024 / sizeof(uint64_t)>> vir_ptr_map;
u32 devicePhyId_ = 1;
u64 addr;
size_t size = 2 * 1024 * 1024;
size_t lenth = 1;
size_t alignment = 2 * 1024 * 1024;
uint64_t flags = 1;
std::mutex stub_ZeroCopyMemoryAgentUt_mutex;
HcclResult stub_ZeroCopyMemoryAgentSt_Send(hccl::HcclSocket* socket, const void* data, u64 size)
{
    std::unique_lock<std::mutex> lock(stub_ZeroCopyMemoryAgentUt_mutex);
    std::vector<u8> temp;
    temp.resize(size);
    memcpy_s(temp.data(), size, data, size);
    exchangeDataForAck_.push(temp);
    lock.unlock();
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgentRecv(hccl::HcclSocket* socket, void* recvBuf, u32 recvBufLen, u64& compSize)
{
    RequestType requestType = RequestType::RESERVED;
    std::vector<u8> temp = exchangeDataForAck_.front();
    exchangeDataForAck_.pop();
    memcpy_s(&requestType, sizeof(RequestType), temp.data(), sizeof(RequestType));
    switch (requestType) {
        case RequestType::SET_MEMORY_RANGE: {
            static std::vector<u8> exchangeDataForAck_reserve_ipc_memory;
            exchangeDataForAck_reserve_ipc_memory.resize(recvBufLen);
            RequestType requestType = RequestType::SET_MEMORY_RANGE;
            u32 buf_len = recvBufLen;
            auto data = exchangeDataForAck_reserve_ipc_memory.data();
            CHK_RET(ConstructData(data, buf_len, requestType));

            CHK_RET(ConstructData(data, buf_len, devicePhyId_));
            vir_ptr_map[devicePhyId_];
            u64 addr = reinterpret_cast<u64>(vir_ptr_map[devicePhyId_].data());
            CHK_RET(ConstructData(data, buf_len, addr));

            CHK_RET(ConstructData(data, buf_len, lenth));

            CHK_RET(ConstructData(data, buf_len, alignment));

            CHK_RET(ConstructData(data, buf_len, flags));
            memcpy_s(
                recvBuf, recvBufLen, exchangeDataForAck_reserve_ipc_memory.data(),
                exchangeDataForAck_reserve_ipc_memory.size());
            compSize = recvBufLen;
        } break;
        case RequestType::UNSET_MEMORY_RANGE: {
            static std::vector<u8> exchangeDataForAck_reserve_ipc_memory;
            exchangeDataForAck_reserve_ipc_memory.resize(recvBufLen);
            RequestType requestType = RequestType::UNSET_MEMORY_RANGE;
            u32 buf_len = recvBufLen;
            auto data = exchangeDataForAck_reserve_ipc_memory.data();
            CHK_RET(ConstructData(data, buf_len, requestType));

            CHK_RET(ConstructData(data, buf_len, devicePhyId_));
            vir_ptr_map[devicePhyId_];
            u64 addr = reinterpret_cast<u64>(vir_ptr_map[devicePhyId_].data());
            CHK_RET(ConstructData(data, buf_len, addr));
            memcpy_s(
                recvBuf, recvBufLen, exchangeDataForAck_reserve_ipc_memory.data(),
                exchangeDataForAck_reserve_ipc_memory.size());
            compSize = recvBufLen;
        } break;
        case RequestType::SET_MEMORY_RANGE_ACK: {
            static std::vector<u8> exchangeDataForAck_release_ipc_memory;
            exchangeDataForAck_release_ipc_memory.resize(recvBufLen);
            RequestType requestType = RequestType::SET_MEMORY_RANGE_ACK;
            u32 buf_len = recvBufLen;
            auto data = exchangeDataForAck_release_ipc_memory.data();
            CHK_RET(ConstructData(data, buf_len, requestType));

            CHK_RET(ConstructData(data, buf_len, devicePhyId_));
            vir_ptr_map[devicePhyId_];
            u64 addr = reinterpret_cast<u64>(vir_ptr_map[devicePhyId_].data());
            CHK_RET(ConstructData(data, buf_len, addr));
            memcpy_s(
                recvBuf, recvBufLen, exchangeDataForAck_release_ipc_memory.data(),
                exchangeDataForAck_release_ipc_memory.size());
            compSize = recvBufLen;
        } break;
        case RequestType::SET_REMOTE_BARE_TGID: {
            static std::vector<u8> exchangeDataForAck_bare_tgid;
            exchangeDataForAck_bare_tgid.resize(recvBufLen);
            u8* exchangeDataPtr = exchangeDataForAck_bare_tgid.data();
            u32 exchangeDataBlankSize = recvBufLen;
            RequestType requestType = RequestType::SET_REMOTE_BARE_TGID;
            CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, requestType));
            CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId_));
            memcpy_s(recvBuf, recvBufLen, exchangeDataForAck_bare_tgid.data(), exchangeDataForAck_bare_tgid.size());
            compSize = recvBufLen;
        } break;
        case RequestType::SET_REMOTE_BARE_TGID_ACK: {
            static std::vector<u8> exchangeDataForAck_bare_tgid;
            exchangeDataForAck_bare_tgid.resize(recvBufLen);
            u8* exchangeDataPtr = exchangeDataForAck_bare_tgid.data();
            u32 exchangeDataBlankSize = recvBufLen;
            RequestType requestType = RequestType::SET_REMOTE_BARE_TGID_ACK;
            CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, requestType));
            CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId_));
            u64 addr = reinterpret_cast<u64>(vir_ptr_map[devicePhyId_].data());
            CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, addr));
            memcpy_s(recvBuf, recvBufLen, exchangeDataForAck_bare_tgid.data(), exchangeDataForAck_bare_tgid.size());
            compSize = recvBufLen;
        } break;
        case RequestType::ACTIVATE_COMM_MEMORY: {
            static std::vector<u8> exchangeDataForAck_validate_ipc_memory;
            exchangeDataForAck_validate_ipc_memory.resize(recvBufLen);
            u8* exchangeDataPtr = exchangeDataForAck_validate_ipc_memory.data();
            u32 exchangeDataBlankSize = recvBufLen;

            RequestType requestType = RequestType::ACTIVATE_COMM_MEMORY;
            CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, requestType));

            CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId_));

            u64 addr = reinterpret_cast<u64>(vir_ptr_map[devicePhyId_].data());
            CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, addr));
            long unsigned int vv = size;
            CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, lenth));
            int offset = 0;
            CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, offset));
            uint64_t shareableHandle = 0x01;
            CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, shareableHandle));
            CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, flags));
            memcpy_s(
                recvBuf, recvBufLen, exchangeDataForAck_validate_ipc_memory.data(),
                exchangeDataForAck_validate_ipc_memory.size());
            compSize = recvBufLen;
        } break;
        case RequestType::DEACTIVATE_COMM_MEMORY: {
            static std::vector<u8> exchangeDataForAck_invalidate_ipc_memory;
            exchangeDataForAck_invalidate_ipc_memory.resize(recvBufLen);
            u8* exchangeDataPtr = exchangeDataForAck_invalidate_ipc_memory.data();
            u32 exchangeDataBlankSize = recvBufLen;

            RequestType requestType = RequestType::DEACTIVATE_COMM_MEMORY;
            CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, requestType));

            CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId_));

            u64 addr = reinterpret_cast<u64>(vir_ptr_map[devicePhyId_].data());
            CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, addr));
            memcpy_s(
                recvBuf, recvBufLen, exchangeDataForAck_invalidate_ipc_memory.data(),
                exchangeDataForAck_invalidate_ipc_memory.size());
            compSize = recvBufLen;
        } break;
        case RequestType::BARRIER_CLOSE: {
            static std::vector<u8> exchangeDataForAck_bare_close;
            exchangeDataForAck_bare_close.resize(recvBufLen);
            u8* exchangeDataPtr = exchangeDataForAck_bare_close.data();
            u32 exchangeDataBlankSize = recvBufLen;

            RequestType requestType = RequestType::BARRIER_CLOSE;
            CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, requestType));

            CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId_));

            u64 addr = reinterpret_cast<u64>(vir_ptr_map[devicePhyId_].data());
            CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, addr));
            memcpy_s(recvBuf, recvBufLen, exchangeDataForAck_bare_close.data(), exchangeDataForAck_bare_close.size());
            compSize = recvBufLen;
        } break;
        default: {
            memcpy_s(recvBuf, recvBufLen, temp.data(), temp.size());
            compSize = temp.size();
        }
    }
    return HCCL_SUCCESS;
}

HcclResult stub_ZeroCopyMemoryAgent_IRecv(hccl::HcclSocket* socket, void* recvBuf, u32 recvBufLen, u64& compSize)
{
    std::unique_lock<std::mutex> lock(stub_ZeroCopyMemoryAgentUt_mutex);
    while (exchangeDataForAck_.empty()) {
        compSize = 0;
        return HCCL_SUCCESS;
    }
    return ZeroCopyMemoryAgentRecv(socket, recvBuf, recvBufLen, compSize);
}

s32 stub_SocketManagerTest_hrtRaGetSockets(u32 role, struct SocketInfoT conn[], u32 num, u32* connectedNum)
{
    static std::vector<int> fdHandle;
    for (int i = 0; i < num; i++) {
        fdHandle.push_back(0);
        conn[i].fdHandle = 0;
        conn[i].status = CONNECT_OK;
    }
    *connectedNum = num;
    return 0;
}

HcclResult stub_SocketManagerTest_GetIsSupSockBatchCloseImmed(u32 phyId, bool& isSupportBatchClose)
{
    isSupportBatchClose = true;
    return HCCL_SUCCESS;
}

HcclResult stub_exchangerSocketTest_hrtRaBlockGetSockets(u32 role, struct SocketInfoT conn[], u32 num)
{
    static std::vector<int> fdHandle;
    for (int i = 0; i < num; i++) {
        fdHandle.push_back(0);
        conn[i].fdHandle = &fdHandle[fdHandle.size() - 1];
        conn[i].status = CONNECT_OK;
    }
    return HCCL_SUCCESS;
}

HcclResult stub_GetRaResourceInfo_exchangerSocketTest(NetworkManager* that, RaResourceInfo& raResourceInfo)
{
    static bool initialized = false;
    static RaResourceInfo fake_raResourceInfo;
    static int fake_handle = 1;
    HcclIpAddress ipAddr = HcclIpAddress(1684515008);
    if (!initialized) {
        IpSocket tmpIpSocket;
        tmpIpSocket.nicSocketHandle = &fake_handle;
        for (int i = 0; i < 8; i++) {
            fake_raResourceInfo.vnicSocketMap[ipAddr] = tmpIpSocket;
            fake_raResourceInfo.nicSocketMap[ipAddr] = tmpIpSocket;
        }
    }
    raResourceInfo = fake_raResourceInfo;
    return HCCL_SUCCESS;
}

s32 stub_SocketManagerTest_hrtRaSocketNonBlockRecvHB(const FdHandle fdHandle, void* data, u64 size, u64* recvSize)
{
    static u32 count = 0;
    if (count++ % 5 != 0) {
        *recvSize = size;
        count = 0;
    }
    return 0;
}

class ZeroCopyMemoryAgentUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        DlTdtFunction::GetInstance().DlTdtFunctionInit();
        DlRaFunction::GetInstance().DlRaFunctionInit();
        TsdOpen(0, 2);
        std::cout << "\033[36m--OneSidedUt SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase()
    {
        TsdClose(0);
        std::cout << "\033[36m--OneSidedUt TearDown--\033[0m" << std::endl;
    }
    virtual void SetUp()
    {
        MOCKER(hrtRaSocketNonBlockRecv).stubs().will(invoke(stub_SocketManagerTest_hrtRaSocketNonBlockRecvHB));

        MOCKER(hrtRaSocketWhiteListAdd).stubs().will(returnValue(HCCL_SUCCESS));

        MOCKER(hrtRaSocketWhiteListDel).stubs().will(returnValue(HCCL_SUCCESS));

        MOCKER(hrtRaSocketBatchConnect).stubs().will(returnValue(HCCL_SUCCESS));

        MOCKER(hrtRaGetSockets).stubs().will(invoke(stub_SocketManagerTest_hrtRaGetSockets));

        MOCKER(hrtRaSocketBatchClose).stubs().will(returnValue(HCCL_SUCCESS));

        MOCKER(hrtRaSocketNonBlockSend).stubs().will(invoke(stub_SocketManagerTest_hrtRaSocketNonBlockSendHB));

        MOCKER(hrtRaBlockGetSockets).stubs().will(invoke(stub_exchangerSocketTest_hrtRaBlockGetSockets));

        MOCKER_CPP(&NetworkManager::GetRaResourceInfo).stubs().will(invoke(stub_GetRaResourceInfo_exchangerSocketTest));
        hrtSetDevice(0);
        ResetInitState();
        DlRaFunction::GetInstance().DlRaFunctionInit();
        ClearHalEvent();
        struct RaInitConfig config;

        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

void get_ranks_1server_2dev(std::vector<RankInfo>& rank_vector)
{
    RankInfo tmp_para_0;

    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverIdx = 0;
    tmp_para_0.serverId = "10.0.0.10";
    tmp_para_0.nicIp.push_back(HcclIpAddress("192.168.0.11"));
    tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_1;

    tmp_para_1.userRank = 1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverIdx = 0;
    tmp_para_1.serverId = "10.0.0.10";
    tmp_para_1.nicIp.push_back(HcclIpAddress("192.168.0.12"));
    tmp_para_1.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    rank_vector.push_back(tmp_para_0);
    rank_vector.push_back(tmp_para_1);
    return;
}

aclError aclrtReserveMemAddress_stub(void** virPtr, size_t size, size_t alignment, void* expectPtr, uint64_t flags)
{
    CHK_PTR_NULL(virPtr);
    vir_ptr_map[1];
    *virPtr = reinterpret_cast<void*>(reinterpret_cast<u64>(vir_ptr_map[1].data()));
    expectPtr = reinterpret_cast<void*>(reinterpret_cast<u64>(vir_ptr_map[1].data()));
    return ACL_SUCCESS;
}

rtError_t aclrtMemImportFromShareableHandle_stub(uint64_t shareableHandle, int32_t deviceId, aclrtDrvMemHandle* handle)
{
    *handle = reinterpret_cast<void*>(reinterpret_cast<u64>(vir_ptr_map[deviceId].data()));
    return ACL_SUCCESS;
}

TEST_F(ZeroCopyMemoryAgentUt, ut_agent_test)
{
    MOCKER(GetIsSupSockBatchCloseImmed).stubs().will(invoke(stub_SocketManagerTest_GetIsSupSockBatchCloseImmed));
    u32 interfaceVersion = 1;
    MOCKER(hrtRaGetInterfaceVersion)
        .expects(atMost(2))
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&interfaceVersion))
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret;
    u32 recvBufLen = 64;
    u64 compSize = 64;
    MOCKER_CPP(&HcclSocket::Send, HcclResult(HcclSocket::*)(const void*, u64))
        .stubs()
        .with(mockcpp::any())
        .will(invoke(stub_ZeroCopyMemoryAgentSt_Send));
    MOCKER_CPP(&HcclSocket::IRecv).stubs().will(invoke(stub_ZeroCopyMemoryAgent_IRecv));
    MOCKER(aclrtMapMem).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtUnmapMem).stubs().will(returnValue(ACL_SUCCESS));
    std::unique_ptr<HcclSocketManager> socketManager;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    std::string commTag = "SocketManagerTest";
    bool isInterLink = false;
    u32 socketsPerLink = 1;
    NicType socketType = NicType::VNIC_TYPE;
    HcclSocketRole localRole = HcclSocketRole::SOCKET_ROLE_SERVER;
    HcclIpAddress localIPs(0x01);
    ret = HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::vector<RankInfo> rank_vector;
    get_ranks_1server_2dev(rank_vector);
    ZeroCopyMemoryAgent ZeroCopyMemoryAgent(
        socketManager, 0, 0, localIPs, rank_vector, 0, true, "ZeroCopyMemoryAgentTest");
    EXPECT_EQ(ZeroCopyMemoryAgent.Init(), HCCL_SUCCESS);
    MOCKER(aclrtReserveMemAddress).stubs().will(invoke(aclrtReserveMemAddress_stub));
    MOCKER(aclrtMemImportFromShareableHandle).stubs().will(invoke(aclrtMemImportFromShareableHandle_stub));
    MOCKER(aclrtMapMem).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtUnmapMem).stubs().will(returnValue(ACL_SUCCESS));
    EXPECT_EQ(
        ZeroCopyMemoryAgent.SetMemoryRange(
            reinterpret_cast<void*>(reinterpret_cast<u64>(vir_ptr_map[0].data())), lenth, alignment, flags),
        HCCL_SUCCESS);
    EXPECT_EQ(
        ZeroCopyMemoryAgent.ActivateCommMemory(
            reinterpret_cast<void*>(reinterpret_cast<u64>(vir_ptr_map[0].data())), lenth, 0,
            reinterpret_cast<void*>(reinterpret_cast<u64>(vir_ptr_map[3].data())), flags),
        HCCL_SUCCESS);
    EXPECT_EQ(
        ZeroCopyMemoryAgent.DeactivateCommMemory(reinterpret_cast<void*>(reinterpret_cast<u64>(vir_ptr_map[0].data()))),
        HCCL_SUCCESS);
    EXPECT_EQ(
        ZeroCopyMemoryAgent.UnsetMemoryRange(reinterpret_cast<void*>(reinterpret_cast<u64>(vir_ptr_map[0].data()))),
        HCCL_SUCCESS);

    u64 baseSetAddr = 0x1000;
    u64 baseSetLen = 2 * 1024 * 1024; // 2MB
    int dummyHandle = 1;
    void* handle = &dummyHandle;

    EXPECT_EQ(ZeroCopyMemoryAgent.BarrierClose(), HCCL_SUCCESS);
    EXPECT_EQ(ZeroCopyMemoryAgent.DeInit(), HCCL_SUCCESS);
    ZeroCopyMemoryAgent.mapDevPhyIdconnectedSockets_.clear();
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0);
}

TEST_F(ZeroCopyMemoryAgentUt, ut_agent_wait_timeout)
{
    std::unique_ptr<HcclSocketManager> socketManager;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    HcclIpAddress localIPs(0x01);
    std::vector<RankInfo> rankInfo;
    ZeroCopyMemoryAgent agent(socketManager, 0, 0, localIPs, rankInfo, 0, false, "wait_timeout");

    s32 timeout = 0;
    MOCKER(GetExternalInputHcclLinkTimeOut).stubs().will(returnValue(timeout));

    agent.mapDevPhyIdconnectedSockets_[0] = std::make_shared<HcclSocket>(nullptr, 0);
    agent.mapDevPhyIdconnectedSockets_[1] = std::make_shared<HcclSocket>(nullptr, 0);
    agent.reqMsgCounter_[static_cast<int>(RequestType::SET_MEMORY_RANGE)] = 100;
    EXPECT_NE(agent.WaitForAllRemoteComplete(RequestType::SET_MEMORY_RANGE), HCCL_SUCCESS);

    agent.reqMsgCounter_[static_cast<int>(RequestType::SET_MEMORY_RANGE)] = 1;
    EXPECT_NE(agent.WaitForAllRemoteComplete(RequestType::SET_MEMORY_RANGE), HCCL_SUCCESS);
}

HcclResult stub_ZeroCopyMemoryAgent_SendAsync(
    hccl::HcclSocket* socket, const void* data, u64 size, u64* sentSize, void** reqHandle)
{
    std::unique_lock<std::mutex> lock(stub_ZeroCopyMemoryAgentUt_mutex);
    std::vector<u8> temp;
    temp.resize(size);
    memcpy_s(temp.data(), size, data, size);
    exchangeDataForAck_.push(temp);
    *sentSize = size;
    *reqHandle = (void*)0x01;
    return HCCL_SUCCESS;
}

HcclResult stub_ZeroCopyMemoryAgent_RecvAsync(
    hccl::HcclSocket* socket, void* recvBuf, u64 recvBufLen, u64* receivedSize, void** reqHandle)
{
    *reqHandle = (void*)0x02;
    std::unique_lock<std::mutex> lock(stub_ZeroCopyMemoryAgentUt_mutex);
    while (exchangeDataForAck_.empty()) {
        *receivedSize = 0;
        return HCCL_SUCCESS;
    }
    u64 compSize = 0;
    HcclResult ret = ZeroCopyMemoryAgentRecv(socket, recvBuf, recvBufLen, compSize);
    *receivedSize = compSize;
    return ret;
}

HcclResult stub_ZeroCopyMemoryAgent_GetAsyncReqResult(hccl::HcclSocket* socket, void* reqHandle, HcclResult& reqResult)
{
    reqResult = HCCL_SUCCESS;
    return HCCL_SUCCESS;
}

TEST_F(ZeroCopyMemoryAgentUt, Ut_AgentFunc_When_UseAsyncSocketApi_ExpectNorm)
{
    MOCKER(GetIsSupSockBatchCloseImmed).stubs().will(invoke(stub_SocketManagerTest_GetIsSupSockBatchCloseImmed));
    u32 interfaceVersion = 1;
    MOCKER(hrtRaGetInterfaceVersion)
        .expects(atMost(2))
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&interfaceVersion))
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret;

    MOCKER(HcclSocket::IsSupportAsync).stubs().will(returnValue(true));
    MOCKER_CPP(&HcclSocket::SendAsync).stubs().will(invoke(stub_ZeroCopyMemoryAgent_SendAsync));
    MOCKER_CPP(&HcclSocket::RecvAsync).stubs().will(invoke(stub_ZeroCopyMemoryAgent_RecvAsync));
    MOCKER_CPP(&HcclSocket::GetAsyncReqResult).stubs().will(invoke(stub_ZeroCopyMemoryAgent_GetAsyncReqResult));

    MOCKER(aclrtMapMem).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtUnmapMem).stubs().will(returnValue(ACL_SUCCESS));
    std::unique_ptr<HcclSocketManager> socketManager;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    std::string commTag = "SocketManagerTest";
    bool isInterLink = false;
    u32 socketsPerLink = 1;
    NicType socketType = NicType::VNIC_TYPE;
    HcclSocketRole localRole = HcclSocketRole::SOCKET_ROLE_SERVER;
    HcclIpAddress localIPs(0x01);
    ret = HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::vector<RankInfo> rank_vector;
    get_ranks_1server_2dev(rank_vector);
    ZeroCopyMemoryAgent agent(socketManager, 0, 0, localIPs, rank_vector, 0, true, "ZeroCopyMemoryAgentTest");
    EXPECT_EQ(agent.Init(), HCCL_SUCCESS);
    MOCKER(aclrtReserveMemAddress).stubs().will(invoke(aclrtReserveMemAddress_stub));
    MOCKER(aclrtMemImportFromShareableHandle).stubs().will(invoke(aclrtMemImportFromShareableHandle_stub));
    MOCKER(aclrtMapMem).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtUnmapMem).stubs().will(returnValue(ACL_SUCCESS));

    EXPECT_EQ(
        agent.SetMemoryRange(
            reinterpret_cast<void*>(reinterpret_cast<u64>(vir_ptr_map[0].data())), lenth, alignment, flags),
        HCCL_SUCCESS);
    EXPECT_EQ(
        agent.ActivateCommMemory(
            reinterpret_cast<void*>(reinterpret_cast<u64>(vir_ptr_map[0].data())), lenth, 0,
            reinterpret_cast<void*>(reinterpret_cast<u64>(vir_ptr_map[3].data())), flags),
        HCCL_SUCCESS);
    EXPECT_EQ(
        agent.DeactivateCommMemory(reinterpret_cast<void*>(reinterpret_cast<u64>(vir_ptr_map[0].data()))),
        HCCL_SUCCESS);
    EXPECT_EQ(
        agent.UnsetMemoryRange(reinterpret_cast<void*>(reinterpret_cast<u64>(vir_ptr_map[0].data()))), HCCL_SUCCESS);

    EXPECT_EQ(agent.BarrierClose(), HCCL_SUCCESS);
    EXPECT_EQ(agent.DeInit(), HCCL_SUCCESS);
    agent.mapDevPhyIdconnectedSockets_.clear();
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0);
}

TEST_F(ZeroCopyMemoryAgentUt, Ut_RequestBatchSendAsync_When_CombineAckAndReq_ExpectNorm)
{
    MOCKER(GetIsSupSockBatchCloseImmed).stubs().will(invoke(stub_SocketManagerTest_GetIsSupSockBatchCloseImmed));
    u32 interfaceVersion = 1;
    MOCKER(hrtRaGetInterfaceVersion)
        .expects(atMost(2))
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&interfaceVersion))
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret;

    MOCKER(HcclSocket::IsSupportAsync).stubs().will(returnValue(true));
    MOCKER(aclrtMapMem).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtUnmapMem).stubs().will(returnValue(ACL_SUCCESS));
    std::unique_ptr<HcclSocketManager> socketManager;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    std::string commTag = "SocketManagerTest";
    bool isInterLink = false;
    u32 socketsPerLink = 1;
    NicType socketType = NicType::VNIC_TYPE;
    HcclSocketRole localRole = HcclSocketRole::SOCKET_ROLE_SERVER;
    HcclIpAddress localIPs(0x01);
    ret = HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::vector<RankInfo> rank_vector;
    get_ranks_1server_2dev(rank_vector);
    ZeroCopyMemoryAgent agent(socketManager, 0, 0, localIPs, rank_vector, 0, true, "ZeroCopyMemoryAgentTest");

    MOCKER_CPP(&ZeroCopyMemoryAgent::InitInnerThread).stubs().will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(agent.Init(), HCCL_SUCCESS);
    MOCKER(aclrtReserveMemAddress).stubs().will(invoke(aclrtReserveMemAddress_stub));
    MOCKER(aclrtMemImportFromShareableHandle).stubs().will(invoke(aclrtMemImportFromShareableHandle_stub));
    MOCKER(aclrtMapMem).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtUnmapMem).stubs().will(returnValue(ACL_SUCCESS));

    MOCKER_CPP(&HcclSocket::SendAsync).stubs().will(returnValue(HCCL_SUCCESS));
    agent.sendMgrs_[1].AddRequest(true, agent.exchangeDataForAck_[1]);
    agent.sendMgrs_[1].AddRequest(false, agent.exchangeDataForSend_);
    agent.RequestBatchSendAsync();
    EXPECT_FALSE(agent.sendMgrs_[1].hasReq_[0].load());
    EXPECT_TRUE(agent.sendMgrs_[1].hasReq_[1].load());
    EXPECT_EQ(agent.sendMgrs_[1].reqDataSize_, 128U);

    EXPECT_EQ(agent.DeInit(), HCCL_SUCCESS);
    agent.mapDevPhyIdconnectedSockets_.clear();
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0);
}

TEST_F(ZeroCopyMemoryAgentUt, Ut_RequestBatchSendAsync_When_SocketSendFailed_ExpectTryMore)
{
    MOCKER(GetIsSupSockBatchCloseImmed).stubs().will(invoke(stub_SocketManagerTest_GetIsSupSockBatchCloseImmed));
    u32 interfaceVersion = 1;
    MOCKER(hrtRaGetInterfaceVersion)
        .expects(atMost(2))
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&interfaceVersion))
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret;

    MOCKER(HcclSocket::IsSupportAsync).stubs().will(returnValue(true));
    MOCKER(aclrtMapMem).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtUnmapMem).stubs().will(returnValue(ACL_SUCCESS));
    std::unique_ptr<HcclSocketManager> socketManager;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    std::string commTag = "SocketManagerTest";
    bool isInterLink = false;
    u32 socketsPerLink = 1;
    NicType socketType = NicType::VNIC_TYPE;
    HcclSocketRole localRole = HcclSocketRole::SOCKET_ROLE_SERVER;
    HcclIpAddress localIPs(0x01);
    ret = HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::vector<RankInfo> rank_vector;
    get_ranks_1server_2dev(rank_vector);
    ZeroCopyMemoryAgent agent(socketManager, 0, 0, localIPs, rank_vector, 0, true, "ZeroCopyMemoryAgentTest");

    MOCKER_CPP(&ZeroCopyMemoryAgent::InitInnerThread).stubs().will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(agent.Init(), HCCL_SUCCESS);
    MOCKER(aclrtReserveMemAddress).stubs().will(invoke(aclrtReserveMemAddress_stub));
    MOCKER(aclrtMemImportFromShareableHandle).stubs().will(invoke(aclrtMemImportFromShareableHandle_stub));
    MOCKER(aclrtMapMem).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtUnmapMem).stubs().will(returnValue(ACL_SUCCESS));

    // SendAsync调用失败场景
    MOCKER_CPP(&HcclSocket::SendAsync).stubs().will(returnValue(HCCL_E_NETWORK));
    agent.sendMgrs_[1].AddRequest(false, agent.exchangeDataForSend_);
    agent.sendMgrs_[1].reqDataSize_ = 0;
    agent.sendMgrs_[1].sentSize_ = 0;
    agent.RequestBatchSendAsync();
    EXPECT_FALSE(agent.sendMgrs_[1].hasReq_[0].load());
    EXPECT_TRUE(agent.sendMgrs_[1].hasReq_[1].load());
    EXPECT_EQ(agent.sendMgrs_[1].reqDataSize_, 64U);
    EXPECT_EQ(agent.sendMgrs_[1].sentSize_, 0);

    // SendAsync后，GetResult表明send失败场景
    HcclResult sendReqRet = HCCL_E_TCP_TRANSFER;
    MOCKER_CPP(&HcclSocket::GetAsyncReqResult)
        .stubs()
        .with(mockcpp::any(), outBound(sendReqRet))
        .will(returnValue(HCCL_E_AGAIN))
        .then(returnValue(HCCL_SUCCESS));
    agent.sendMgrs_[1].AddRequest(false, agent.exchangeDataForSend_);
    // 构造SendAsync的结果
    void* handle = (void*)0x01;
    agent.sendMgrs_[1].lastSendSize_ = 0;
    agent.sendMgrs_[1].sentSize_ = 0;
    agent.sendMgrs_[1].reqDataSize_ = 64;
    agent.sendMgrs_[1].lastSendHandle_ = handle;
    agent.CheckBatchSendAsyncResult();
    EXPECT_EQ(agent.sendMgrs_[1].lastSendHandle_, handle);

    agent.CheckBatchSendAsyncResult();
    EXPECT_FALSE(agent.sendMgrs_[1].hasReq_[0].load());
    EXPECT_TRUE(agent.sendMgrs_[1].hasReq_[1].load());
    EXPECT_EQ(agent.sendMgrs_[1].reqDataSize_, 64U);
    EXPECT_EQ(agent.sendMgrs_[1].sentSize_, 0);
    EXPECT_EQ(agent.sendMgrs_[1].lastSendSize_, 0);
    EXPECT_EQ(agent.sendMgrs_[1].lastSendHandle_, nullptr);

    EXPECT_EQ(agent.DeInit(), HCCL_SUCCESS);
    agent.mapDevPhyIdconnectedSockets_.clear();
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0);
}

TEST_F(ZeroCopyMemoryAgentUt, Ut_RequestBatchRecvAsync_When_SocketRecvFailed_ExpectTryMore)
{
    MOCKER(GetIsSupSockBatchCloseImmed).stubs().will(invoke(stub_SocketManagerTest_GetIsSupSockBatchCloseImmed));
    u32 interfaceVersion = 1;
    MOCKER(hrtRaGetInterfaceVersion)
        .expects(atMost(2))
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&interfaceVersion))
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret;

    MOCKER(HcclSocket::IsSupportAsync).stubs().will(returnValue(true));
    MOCKER(aclrtMapMem).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtUnmapMem).stubs().will(returnValue(ACL_SUCCESS));
    std::unique_ptr<HcclSocketManager> socketManager;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    std::string commTag = "SocketManagerTest";
    bool isInterLink = false;
    u32 socketsPerLink = 1;
    NicType socketType = NicType::VNIC_TYPE;
    HcclSocketRole localRole = HcclSocketRole::SOCKET_ROLE_SERVER;
    HcclIpAddress localIPs(0x01);
    ret = HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::vector<RankInfo> rank_vector;
    get_ranks_1server_2dev(rank_vector);
    ZeroCopyMemoryAgent agent(socketManager, 0, 0, localIPs, rank_vector, 0, true, "ZeroCopyMemoryAgentTest");

    MOCKER_CPP(&ZeroCopyMemoryAgent::InitInnerThread).stubs().will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(agent.Init(), HCCL_SUCCESS);
    MOCKER(aclrtReserveMemAddress).stubs().will(invoke(aclrtReserveMemAddress_stub));
    MOCKER(aclrtMemImportFromShareableHandle).stubs().will(invoke(aclrtMemImportFromShareableHandle_stub));
    MOCKER(aclrtMapMem).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtUnmapMem).stubs().will(returnValue(ACL_SUCCESS));

    // RecvAsync后，GetResult表明Recv失败场景
    void* handle = (void*)0x02;
    MOCKER_CPP(&HcclSocket::RecvAsync)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&handle))
        .will(returnValue(HCCL_SUCCESS));
    agent.recvMgrs_[1].recvIndex_ = 0;
    agent.recvMgrs_[1].lastRecvSize_ = 0;
    agent.RequestBatchRecvAsync();
    EXPECT_EQ(agent.recvMgrs_[1].lastRecvHandle_, handle);
    EXPECT_EQ(agent.recvMgrs_[1].lastRecvSize_, 0);

    HcclResult recvReqRet = HCCL_E_TCP_TRANSFER;
    MOCKER_CPP(&HcclSocket::GetAsyncReqResult)
        .stubs()
        .with(mockcpp::any(), outBound(recvReqRet))
        .will(returnValue(HCCL_E_AGAIN))
        .then(returnValue(HCCL_SUCCESS));
    agent.CheckBatchRecvAsyncResult();
    EXPECT_EQ(agent.recvMgrs_[1].recvIndex_, 0);
    EXPECT_EQ(agent.recvMgrs_[1].lastRecvHandle_, handle);
    agent.CheckBatchRecvAsyncResult();
    EXPECT_EQ(agent.recvMgrs_[1].recvIndex_, 0);
    EXPECT_EQ(agent.recvMgrs_[1].lastRecvHandle_, nullptr);

    EXPECT_EQ(agent.DeInit(), HCCL_SUCCESS);
    agent.mapDevPhyIdconnectedSockets_.clear();
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0);
}
