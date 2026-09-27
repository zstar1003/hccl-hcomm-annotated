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
#include "hccl_communicator.h"

#define private public
#define protected public
#include "adapter_rts.h"
#include "exception_handler.h"
#include "hccl_socket_manager.h"
#include "notify_pool.h"
#include "i_hccl_one_sided_service.h"
#include "hccl_one_sided_service.h"
#include <hccl/hccl_one_sided_services.h>
#include "hccl_one_sided_conn.h"
#include "rma_buffer_mgr.h"
#include "local_rdma_rma_buffer.h"
#include "local_ipc_rma_buffer.h"
#include "remote_rdma_rma_buffer.h"
#include "remote_ipc_rma_buffer.h"
#include "local_rdma_rma_buffer_impl.h"
#include "local_ipc_rma_buffer_impl.h"
#include "remote_rdma_rma_buffer_impl.h"
#include "remote_ipc_rma_buffer_impl.h"
#include "hccl_network.h"
#include "transport_roce_mem.h"
#include "transport_ipc_mem.h"
#include "dispatcher_pub.h"
#include "op_base.h"
#include "global_mem_record.h"
#include "global_mem_manager.h"
#include "externalinput.h"
#undef protected
#undef private

using namespace std;
using namespace hccl;

using LocalIpcRmaBufferMgr = hccl::NetDevContext::LocalIpcRmaBufferMgr;
using LocalRdmaRmaBufferMgr = hccl::NetDevContext::LocalRdmaRmaBufferMgr;

class OneSidedSt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        unsetenv("HCCL_INTRA_PCIE_ENABLE");
        setenv("HCCL_INTRA_ROCE_ENABLE", "1", 1);
        std::cout << "\033[36m--OneSidedSt SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase()
    {
        unsetenv("HCCL_INTRA_ROCE_ENABLE");
        std::cout << "\033[36m--OneSidedSt TearDown--\033[0m" << std::endl;
    }
    virtual void SetUp()
    {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

HcclResult ExceptionRuntimeTest()
{
    EXCEPTION_HANDLE_BEGIN
    throw runtime_error("failed");
    EXCEPTION_HANDLE_END
}

HcclResult ExceptionLogicalTest()
{
    EXCEPTION_HANDLE_BEGIN
    throw logic_error("failed");
    EXCEPTION_HANDLE_END
}

HcclResult ExceptionOutRangeTest()
{
    EXCEPTION_HANDLE_BEGIN
    throw out_of_range("failed");
    EXCEPTION_HANDLE_END
}

HcclResult ExceptionNormalTest()
{
    EXCEPTION_HANDLE_BEGIN
    throw exception();
    EXCEPTION_HANDLE_END
}

HcclResult ExceptionBadAllocTest()
{
    EXCEPTION_HANDLE_BEGIN
    throw bad_alloc();
    EXCEPTION_HANDLE_END
}

TEST_F(OneSidedSt, ut_exception_handler)
{
    try {
        EXCEPTION_THROW_IF_ERR(ExceptionRuntimeTest(), "ExceptionRuntimeTest");
    } catch (const exception& e) {
        HCCL_ERROR("exception, what: %s", e.what());
    }

    try {
        EXCEPTION_THROW_IF_ERR(ExceptionLogicalTest(), "ExceptionLogicalTest");
    } catch (const exception& e) {
        HCCL_ERROR("exception, what: %s", e.what());
    }

    try {
        EXCEPTION_THROW_IF_ERR(ExceptionOutRangeTest(), "ExceptionOutRangeTest");
    } catch (const exception& e) {
        HCCL_ERROR("exception, what: %s", e.what());
    }

    try {
        EXCEPTION_THROW_IF_ERR(ExceptionNormalTest(), "ExceptionNormalTest");
    } catch (const exception& e) {
        HCCL_ERROR("exception, what: %s", e.what());
    }

    try {
        EXCEPTION_THROW_IF_ERR(ExceptionBadAllocTest(), "ExceptionBadAllocTest");
    } catch (const exception& e) {
        HCCL_ERROR("exception, what: %s", e.what());
    }

    std::unique_ptr<HcclSocketManager> socketManager;
    std::unique_ptr<NotifyPool> notifyPool;
    IHcclOneSidedService iHcclOneSidedService(socketManager, notifyPool);

    HcclDispatcher dispatcher = &notifyPool;
    HcclRankLinkInfo localRankInfo{};
    RankTable_t rankTable{};
    map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap{};

    iHcclOneSidedService.Config(dispatcher, localRankInfo, &rankTable);
}

HcclResult st_rma_buffer_mgr_test()
{
    EXCEPTION_HANDLE_BEGIN

    using AType = uintptr_t;
    using SType = u64;
    using BufferType = string;

    hcomm::RmaBufferMgr<BufferKey<AType, SType>, BufferType> bufferMgr;

    // 1. Add接口
    // 1.1 添加
    BufferKey<AType, SType> key1((AType)0x1000, (SType)0x100);
    string str1 = "Buffer1";
    auto result1 = bufferMgr.Add(key1, str1);
    EXPECT_EQ(result1.second, true);
    EXPECT_NE(result1.first, bufferMgr.End());
    EXPECT_EQ(result1.first->second.ref, 1);

    // 1.2 添加第二个
    BufferKey<AType, SType> key2((AType)0x2000, (SType)0x100);
    string str2 = "Buffer2";
    auto result2 = bufferMgr.Add(key2, str2);
    EXPECT_EQ(result2.second, true);
    EXPECT_NE(result2.first, bufferMgr.End());
    EXPECT_EQ(result2.first->second.ref, 1);

    // 1.3 重复添加
    result2 = bufferMgr.Add(key2, str2);
    EXPECT_EQ(result2.second, false);
    EXPECT_NE(result2.first, bufferMgr.End());
    EXPECT_EQ(result2.first->second.ref, 2);
    EXPECT_EQ(result2.first->second.buffer, str2);

    // 1.4 添加左边子集
    BufferKey<AType, SType> key3((AType)0x2000, (SType)0x1);
    string str3 = "Buffer3";
    auto result3 = bufferMgr.Add(key3, str3);
    EXPECT_EQ(result3.second, false);
    EXPECT_EQ(result3.first, bufferMgr.End());

    // 1.5 添加超集
    BufferKey<AType, SType> key4((AType)0x2000, (SType)0x1000);
    string str4 = "Buffer4";
    auto result4 = bufferMgr.Add(key4, str4);
    EXPECT_EQ(result4.second, false);
    EXPECT_EQ(result4.first, bufferMgr.End());

    // 1.6 添加左交集
    BufferKey<AType, SType> key5((AType)(0x2000 - 1), (SType)0x100);
    string str5 = "Buffer5";
    auto result5 = bufferMgr.Add(key5, str5);
    EXPECT_EQ(result5.second, false);
    EXPECT_EQ(result5.first, bufferMgr.End());

    // 1.7 添加右交集
    BufferKey<AType, SType> key6((AType)(0x2000 + 1), (SType)0x100);
    string str6 = "Buffer6";
    auto result6 = bufferMgr.Add(key6, str6);
    EXPECT_EQ(result6.second, false);
    EXPECT_EQ(result6.first, bufferMgr.End());

    // 1.8 添加右边子集
    BufferKey<AType, SType> key7((AType)(0x2000 + 1), (SType)(0x100 - 1));
    string str7 = "Buffer7";
    auto result7 = bufferMgr.Add(key7, str7);
    EXPECT_EQ(result7.second, false);
    EXPECT_EQ(result7.first, bufferMgr.End());

    // 2. Find接口
    // 2.1 Find成功
    auto findResult1 = bufferMgr.Find(key1);
    EXPECT_EQ(findResult1.first, true);
    EXPECT_EQ(findResult1.second, str1);

    // 2.2 Find第二个
    auto findResult2 = bufferMgr.Find(key2);
    EXPECT_EQ(findResult2.first, true);
    EXPECT_EQ(findResult2.second, str2);

    // 2.4 find左边子集
    auto findResult3 = bufferMgr.Find(key3);
    EXPECT_EQ(findResult3.first, true);
    EXPECT_EQ(findResult3.second, str2);

    // 2.5 查找超集
    auto findResult4 = bufferMgr.Find(key4);
    EXPECT_EQ(findResult4.first, false);
    EXPECT_EQ(findResult4.second, string(""));

    // 2.6 查找左交集
    auto findResult5 = bufferMgr.Find(key5);
    EXPECT_EQ(findResult5.first, false);
    EXPECT_EQ(findResult5.second, string(""));

    // 2.7 查找右交集
    auto findResult6 = bufferMgr.Find(key6);
    EXPECT_EQ(findResult6.first, false);
    EXPECT_EQ(findResult6.second, string(""));

    // 2.8 查找右边子集
    auto findResult7 = bufferMgr.Find(key7);
    EXPECT_EQ(findResult7.first, true);
    EXPECT_EQ(findResult7.second, str2);

    // 3 删除异常
    // 3.4 删除左边子集
    bool delResult3{};
    try {
        delResult3 = bufferMgr.Del(key3);
    } catch (const out_of_range& e) {
        EXPECT_EQ(delResult3, false);
    } catch (const exception& e) {
        HCCL_ERROR("Standard exception, is not expected. what: %s", e.what());
        EXPECT_EQ(delResult3, -1);
    }

    // 3.5 删除超集
    bool delResult4{};
    try {
        delResult4 = bufferMgr.Del(key4);
    } catch (const out_of_range& e) {
        EXPECT_EQ(delResult4, false);
    } catch (const exception& e) {
        HCCL_ERROR("Standard exception, is not expected. what: %s", e.what());
        EXPECT_EQ(delResult4, -1);
    }

    // 3.6 删除左交集
    bool delResult5{};
    try {
        delResult5 = bufferMgr.Del(key5);
    } catch (const out_of_range& e) {
        EXPECT_EQ(delResult5, false);
    } catch (const exception& e) {
        HCCL_ERROR("Standard exception, is not expected. what: %s", e.what());
        EXPECT_EQ(delResult5, -1);
    }

    // 3.7 删除右交集
    bool delResult6{};
    try {
        delResult6 = bufferMgr.Del(key6);
    } catch (const out_of_range& e) {
        EXPECT_EQ(delResult6, false);
    } catch (const exception& e) {
        HCCL_ERROR("Standard exception, is not expected. what: %s", e.what());
        EXPECT_EQ(delResult6, -1);
    }

    // 4 删除正常
    // 4.3 删除重复的第二个
    bool delResult2{};
    try {
        delResult2 = bufferMgr.Del(key2);
        EXPECT_EQ(delResult2, false);
    } catch (const out_of_range& e) {
        EXPECT_EQ(delResult2, -1);
    } catch (const exception& e) {
        HCCL_ERROR("Standard exception, is not expected. what: %s", e.what());
        EXPECT_EQ(delResult2, -1);
    }

    // 4.2 删除重复的第一个
    try {
        delResult2 = bufferMgr.Del(key2);
        EXPECT_EQ(delResult2, true);
    } catch (const out_of_range& e) {
        EXPECT_EQ(delResult2, -1);
    } catch (const exception& e) {
        HCCL_ERROR("Standard exception, is not expected. what: %s", e.what());
        EXPECT_EQ(delResult2, -1);
    }

    // 4.2 删除不存在的
    bool delResult2_1{};
    try {
        delResult2_1 = bufferMgr.Del(key2);
    } catch (const out_of_range& e) {
        EXPECT_EQ(delResult2_1, false);
    } catch (const exception& e) {
        HCCL_ERROR("Standard exception, is not expected. what: %s", e.what());
        EXPECT_EQ(delResult2_1, -1);
    }

    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

TEST_F(OneSidedSt, ut_rma_buffer_mgr) { EXPECT_EQ(st_rma_buffer_mgr_test(), HCCL_SUCCESS); }

TEST_F(OneSidedSt, ut_one_sided_service_mem_regDereg_enable_disable_roce)
{
    typedef HcclResult (*HcclOneSideServiceCallBack)(
        std::unique_ptr<hccl::IHcclOneSidedService>&, std::unique_ptr<hccl::HcclSocketManager>&,
        std::unique_ptr<hccl::NotifyPool>&);
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    s8* localbuf;
    s8* remotebuf;
    s32 count = 1024;
    void* comm;
    const char* rank_table_file = "./ut_opbase_test.json";

    localbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    remotebuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(remotebuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    NetDevContext devContext;
    devContext.nicType_ = NicType::DEVICE_NIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &devContext;
    HcclRankLinkInfo localLinkInfo{};
    HcclRankLinkInfo remoteLinkInfo{};
    remoteLinkInfo.userRank = 1;
    std::unique_ptr<HcclSocketManager> socketManager = nullptr;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    std::unique_ptr<NotifyPool> notifyPool;
    HcclDispatcher dispatcher;
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);

    IHcclOneSidedService* iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    EXPECT_NE(iService, nullptr);
    iService->netDevRdmaCtx_ = devCtx;
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);

    std::shared_ptr<HcclOneSidedConn> connPtr = make_shared<HcclOneSidedConn>(
        devCtx, localLinkInfo, remoteLinkInfo, socketManager, notifyPool, dispatcher, true, 0U, 0U);
    service->oneSidedConns_.insert({1, connPtr});
    MOCKER_CPP(&HcclOneSidedConn::GetMemType).stubs().will(returnValue(HCCL_SUCCESS));
    std::string localdesc = "ld";
    MOCKER_CPP(&LocalRdmaRmaBuffer::Serialize).stubs().will(returnValue(localdesc));
    MOCKER_CPP(&LocalRdmaRmaBuffer::Init).stubs().will(returnValue(HCCL_SUCCESS));
    u32 remoteRankId = 1;
    bool useRdma = true;
    MOCKER_CPP(&HcclOneSidedService::IsUsedRdma)
        .stubs()
        .with(eq(remoteRankId), outBound(useRdma))
        .will(returnValue(HCCL_SUCCESS));

    HcclMemDesc localMemDesc, remoteMemDesc;

    ret = HcclRegisterMem(comm, remoteRankId, 0, localbuf, 1024, &localMemDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    memcpy_s(&(remoteMemDesc.desc[0]), sizeof(remoteMemDesc.desc), &remoteRankId, sizeof(u32));
    remoteMemDesc.desc[8] = static_cast<int>(RmaType::RDMA_RMA);
    MOCKER_CPP(&RemoteRdmaRmaBuffer::Deserialize).stubs().will(returnValue(HCCL_SUCCESS));
    CommMem remoteMem{};
    ret = HcclEnableMemAccess(comm, &remoteMemDesc, &remoteMem);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclEnableMemAccess(comm, &remoteMemDesc, &remoteMem);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclDisableMemAccess(comm, &remoteMemDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclDisableMemAccess(comm, &remoteMemDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclDeregisterMem(comm, &localMemDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(localbuf);
    sal_free(remotebuf);
    remove(file_name_t);
    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, ut_test_rma_buffer_impl)
{
    int ret = HCCL_SUCCESS;
    s8* localbuf;
    s8* remotebuf;
    s32 count = 1024;

    localbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    remotebuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(remotebuf, count * sizeof(s8), 0, count * sizeof(s8));

    hccl::NetDevContext remoteNetDevCtx{};
    HcclIpAddress ipAddr{};
    remoteNetDevCtx.Init(NicType::HOST_NIC_TYPE, 1, 1, ipAddr);

    std::shared_ptr<RemoteIpcRmaBufferImpl> remoteImplPtr
        = make_shared<RemoteIpcRmaBufferImpl>(reinterpret_cast<void*>(&remoteNetDevCtx));
    std::shared_ptr<LocalIpcRmaBufferImpl> localImplPtr = make_shared<LocalIpcRmaBufferImpl>(
        reinterpret_cast<void*>(&remoteNetDevCtx), localbuf, count * sizeof(s8), RmaMemType::DEVICE);

    char name[21] = "aaaaabbbbbcccccddddd";
    memcpy_s(remoteImplPtr->memName.ipcName, HCCL_IPC_MEM_NAME_LEN, name, sizeof(name));
    memcpy_s(localImplPtr->memName.ipcName, HCCL_IPC_MEM_NAME_LEN, name, sizeof(name));
    u64 offset = 0;

    MOCKER_CPP(&MemNameRepository::OpenIpcMem).stubs().will(returnValue(HCCL_E_PTR));
    MOCKER_CPP(&MemNameRepository::CloseIpcMem).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtDevMemAlignWithPage).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&MemNameRepository::DestroyIpcMem).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtIpcSetMemoryPid).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtIpcSetMemoryName).stubs().will(returnValue(HCCL_SUCCESS));

    ret = localImplPtr->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::string serialStr = localImplPtr->Serialize();
    u32 remotePid = 1;
    u32 remoteSdid = INVALID_INT;
    ret = localImplPtr->Grant(remotePid, remoteSdid);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = localImplPtr->Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = remoteImplPtr->Open();
    EXPECT_EQ(ret, HCCL_E_PTR);
    remoteImplPtr->memType = RmaMemType::HOST;
    ret = remoteImplPtr->Open();
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = remoteImplPtr->Close();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    const std::string temp = serialStr;
    ret = remoteImplPtr->Deserialize(temp);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(localbuf);
    sal_free(remotebuf);
    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, ut_test_rma_buffer_SetIpcMem_noOffset)
{
    int ret = HCCL_SUCCESS;
    s8* localbuf;
    s8* remotebuf;
    s32 count = 1024;

    localbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    remotebuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(remotebuf, count * sizeof(s8), 0, count * sizeof(s8));

    hccl::NetDevContext remoteNetDevCtx{};
    HcclIpAddress ipAddr{};
    remoteNetDevCtx.Init(NicType::HOST_NIC_TYPE, 1, 1, ipAddr);

    std::shared_ptr<RemoteIpcRmaBufferImpl> remoteImplPtr
        = make_shared<RemoteIpcRmaBufferImpl>(reinterpret_cast<void*>(&remoteNetDevCtx));
    std::shared_ptr<LocalIpcRmaBufferImpl> localImplPtr = make_shared<LocalIpcRmaBufferImpl>(
        reinterpret_cast<void*>(&remoteNetDevCtx), localbuf, count * sizeof(s8), RmaMemType::DEVICE);

    char name[21] = "aaaaabbbbbcccccddddd";
    memcpy_s(remoteImplPtr->memName.ipcName, HCCL_IPC_MEM_NAME_LEN, name, sizeof(name));
    memcpy_s(localImplPtr->memName.ipcName, HCCL_IPC_MEM_NAME_LEN, name, sizeof(name));
    u64 offset = 0;

    MOCKER_CPP(&MemNameRepository::OpenIpcMem).stubs().will(returnValue(HCCL_E_PTR));
    MOCKER_CPP(&MemNameRepository::CloseIpcMem).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtDevMemAlignWithPage).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&MemNameRepository::DestroyIpcMem).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtIpcSetMemoryPid).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtIpcSetMemoryName).stubs().will(returnValue(HCCL_SUCCESS));

    IpcMemInfo ipcMemInfo = {nullptr};
    ipcMemInfo.ptr = localImplPtr->addr;
    ipcMemInfo.size = localImplPtr->size;
    MemNameRepository::GetInstance(1)->setNameMap_.insert(std::make_pair(ipcMemInfo, localImplPtr->memName));

    ret = localImplPtr->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::string serialStr = localImplPtr->Serialize();
    u32 remotePid = 1;
    u32 remoteSdid = INVALID_INT;
    ret = localImplPtr->Grant(remotePid, remoteSdid);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = localImplPtr->Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = remoteImplPtr->Open();
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = remoteImplPtr->Close();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    const std::string temp = serialStr;
    ret = remoteImplPtr->Deserialize(temp);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(localbuf);
    sal_free(remotebuf);
    GlobalMockObject::verify();
}
#if 0
TEST_F(OneSidedSt, ut_one_sided_service_mem_regDereg_enable_disable_ipc)
{
    typedef HcclResult (*HcclOneSideServiceCallBack)(std::unique_ptr<hccl::IHcclOneSidedService> &,
    std::unique_ptr<hccl::HcclSocketManager> &, std::unique_ptr<hccl::NotifyPool> &);
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    s8* localbuf;
    s8* remotebuf;
    s32 count = 1024;
    void* comm;
    const char* rank_table_file = "./ut_opbase_test.json";

    localbuf= (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    remotebuf= (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(remotebuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    NetDevContext devContext;
    devContext.nicType_ = NicType::VNIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &devContext;
    HcclRankLinkInfo localLinkInfo {};
    HcclRankLinkInfo remoteLinkInfo {};
    remoteLinkInfo.userRank = 1;
    std::unique_ptr<HcclSocketManager> socketManager = nullptr;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    std::unique_ptr<NotifyPool> notifyPool;
    HcclDispatcher dispatcher;
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);

    IHcclOneSidedService *iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    EXPECT_NE(iService, nullptr);
    iService->netDevIpcCtx_ = devCtx;
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);

    std::shared_ptr<HcclOneSidedConn> connPtr = make_shared<HcclOneSidedConn>(devCtx, localLinkInfo,
        remoteLinkInfo, socketManager, notifyPool, dispatcher, false, 0U, 0U);
    service->oneSidedConns_.insert({1, connPtr});
    MOCKER_CPP(&HcclOneSidedConn::GetMemType)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    std::string localdesc = "ld";
    MOCKER_CPP(&LocalIpcRmaBuffer::Serialize)
        .stubs()
        .will(returnValue(localdesc));
    MOCKER_CPP(&LocalIpcRmaBuffer::Init)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&LocalRdmaRmaBuffer::Init)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType)
        .stubs()
        .with(outBound(deviceType))
        .will(returnValue(HCCL_SUCCESS));
    LinkTypeInServer linkType = LinkTypeInServer::HCCS_TYPE;
    u32 invalid = 0xFFFFFFFF;
    MOCKER(hrtGetPairDeviceLinkType)
        .stubs()
        .with(neq(invalid), neq(invalid), outBound(linkType))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclOneSidedConn::ExchangeIpcProcessInfo)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclOneSidedService::Grant, HcclResult(HcclOneSidedService::*)(const HcclMemDesc&, const HcclOneSidedConn::ProcessInfo&))
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));
    u32 remoteRankId = 1;
    service->isUsedRdmaMap_[1] = false;
    HcclMemDesc localMemDesc, remoteMemDesc;
    ret = HcclRegisterMem(comm, remoteRankId, 0, localbuf, 1024, &localMemDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    memcpy_s(&(remoteMemDesc.desc[0]), sizeof(remoteMemDesc.desc), &remoteRankId, sizeof(u32));
    remoteMemDesc.desc[8] = static_cast<int>(RmaType::IPC_RMA);

    MOCKER_CPP(&RemoteIpcRmaBuffer::Deserialize)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RemoteIpcRmaBuffer::Open)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    service->localMemDescs_[1].push_back(localMemDesc);
    CommMem remoteMem{};
    ret = HcclEnableMemAccess(comm, &remoteMemDesc, &remoteMem);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER_CPP(&RemoteIpcRmaBuffer::Close)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    ret = HcclDisableMemAccess(comm, &remoteMemDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    service->isUsedRdmaMap_[1] = true;
    ret = HcclDeregisterMem(comm, &localMemDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(localbuf);
    sal_free(remotebuf);
    remove(file_name_t);
    GlobalMockObject::verify();
}


TEST_F(OneSidedSt, ut_HcclBatchGetPut_When_SdmaAicpuUnflod_Expect_Success)
{
    nlohmann::json rank_table = rank_table_910_1server_4rank;
    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);
    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();

    HcclResult ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* localbuf;
    s8* remotebuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = 1024;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = rtStreamCreate(&stream, 0);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    localbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    remotebuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(remotebuf, count * sizeof(s8), 0, count * sizeof(s8));

    void *comm;
    const char *rankTableFile = "./ut_opbase_test.json";
    ret = HcclCommInitClusterInfo(rankTableFile, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(GetExternalInputHcclAicpuUnfold).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER(GetExternalInputIntraRoceSwitch).stubs().will(returnValue(0));

    const DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaSendWrV2).stubs().will(returnValue(HCCL_SUCCESS));

    for (int j = 0; j < count; j++) {
        localbuf[j] = 2;
    }
    u32 itemNum = 1;
    HcclOneSideOpDesc desc[itemNum];
    desc[0].count = 1024;
    desc[0].dataType = HCCL_DATA_TYPE_INT8;
    desc[0].localAddr = localbuf;
    desc[0].remoteAddr = remotebuf;

    NetDevContext devContext;
    devContext.nicType_ = NicType::DEVICE_NIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &devContext;

    const u32 remoteRankId = 1;
    HcclRankLinkInfo remoteLinkInfo {};
    remoteLinkInfo.userRank = remoteRankId;

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    IHcclOneSidedService *iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    EXPECT_NE(iService, nullptr);
    std::string commIdentifier = hcclComm->GetIdentifier();
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);
    service->netDevIpcCtx_ = devCtx;
    service->isUsedRdmaMap_[remoteRankId] = false;
    ret = service->CreateConnection(remoteRankId, remoteLinkInfo, service->oneSidedConns_[remoteRankId]);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(service->aicpuUnfoldMode_);
    std::shared_ptr<hccl::HcclOneSidedConn> connPtr = service->oneSidedConns_[remoteRankId];
    EXPECT_NE(connPtr, nullptr);

    TransportIpcMem *transport = dynamic_cast<TransportIpcMem *>(connPtr->transportMemPtr_.get());
    BufferKey<uintptr_t, u64> tempLocalKey(reinterpret_cast<uintptr_t>(localbuf), count * sizeof(s8));
    auto tempLocalBufferPtr = make_shared<LocalIpcRmaBuffer>(devCtx, localbuf, count * sizeof(s8));
    tempLocalBufferPtr->devAddr = localbuf;
    devContext.localIpcRmaBufferMgr_->Add(tempLocalKey, tempLocalBufferPtr);

    BufferKey<uintptr_t, u64> tempRemoteKey(reinterpret_cast<uintptr_t>(remotebuf), count * sizeof(s8));
    RemoteIpcRmaBuffer tempRemoteBufferPtr(devCtx);
    tempRemoteBufferPtr.addr = remotebuf;
    tempRemoteBufferPtr.size = count * sizeof(s8);
    tempRemoteBufferPtr.devAddr = remotebuf;
    connPtr->remoteRmaBufferMgr_.Add(tempRemoteKey, reinterpret_cast<void *>(&tempRemoteBufferPtr));

    HcclIpAddress ipAddr;
    auto socketPtr = make_shared<HcclSocket>("tag", devCtx, ipAddr, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT);
    std::vector<std::shared_ptr<HcclSocket>> connectSockets;
    connectSockets.push_back(socketPtr);

    MOCKER_CPP(&HcclSocketManager::CreateSingleLinkSocket).stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(connectSockets), mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP_VIRTUAL(*transport, &TransportIpcMem::ExchangeMemDesc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(*transport, &TransportIpcMem::SetSocket).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&LocalNotify::Wait).stubs().will(returnValue(HCCL_SUCCESS));

    ret = connPtr->Connect(commIdentifier, 10);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclBatchGet(comm, 1, desc, itemNum, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = hcclStreamSynchronize(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    ret = HcclBatchPut(comm, 1, desc, itemNum, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = hcclStreamSynchronize(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(localbuf);
    sal_free(remotebuf);

    rt_ret = rtStreamDestroy(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name_t);
    GlobalMockObject::verify();
}
#endif
TEST_F(OneSidedSt, ut_one_sided_service_mem_regDereg_enable_disable_a3_rdma)
{
    typedef HcclResult (*HcclOneSideServiceCallBack)(
        std::unique_ptr<hccl::IHcclOneSidedService>&, std::unique_ptr<hccl::HcclSocketManager>&,
        std::unique_ptr<hccl::NotifyPool>&);
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    s8* localbuf;
    s8* remotebuf;
    s32 count = 1024;
    void* comm;
    const char* rank_table_file = "./ut_opbase_test.json";

    localbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    remotebuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(remotebuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    NetDevContext devContext;
    devContext.nicType_ = NicType::VNIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &devContext;
    HcclRankLinkInfo localLinkInfo{};
    HcclRankLinkInfo remoteLinkInfo{};
    remoteLinkInfo.userRank = 1;
    std::unique_ptr<HcclSocketManager> socketManager = nullptr;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    std::unique_ptr<NotifyPool> notifyPool;
    HcclDispatcher dispatcher;
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);

    IHcclOneSidedService* iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    EXPECT_NE(iService, nullptr);
    iService->netDevRdmaCtx_ = devCtx;
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);

    std::shared_ptr<HcclOneSidedConn> connPtr = make_shared<HcclOneSidedConn>(
        devCtx, localLinkInfo, remoteLinkInfo, socketManager, notifyPool, dispatcher, false, 0U, 0U);
    service->oneSidedConns_.insert({1, connPtr});
    MOCKER_CPP(&HcclOneSidedConn::GetMemType).stubs().will(returnValue(HCCL_SUCCESS));
    std::string localdesc = "ld";
    MOCKER_CPP(&LocalIpcRmaBuffer::Serialize).stubs().will(returnValue(localdesc));
    MOCKER_CPP(&LocalIpcRmaBuffer::Init).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&LocalRdmaRmaBuffer::Init).stubs().will(returnValue(HCCL_SUCCESS));
    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));
    u32 intraRoceSwitch = 1;
    MOCKER(GetExternalInputIntraRoceSwitch).stubs().will(returnValue(intraRoceSwitch));
    u32 remoteRankId = 1;

    HcclMemDesc localMemDesc, remoteMemDesc;
    ret = HcclRegisterMem(comm, remoteRankId, 0, localbuf, 1024, &localMemDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    memcpy_s(&(remoteMemDesc.desc[0]), sizeof(remoteMemDesc.desc), &remoteRankId, sizeof(u32));
    remoteMemDesc.desc[8] = static_cast<int>(RmaType::IPC_RMA);

    MOCKER_CPP(&RemoteIpcRmaBuffer::Deserialize).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RemoteIpcRmaBuffer::Open).stubs().will(returnValue(HCCL_SUCCESS));
    CommMem remoteMem{};
    ret = HcclEnableMemAccess(comm, &remoteMemDesc, &remoteMem);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER_CPP(&RemoteIpcRmaBuffer::Close).stubs().will(returnValue(HCCL_SUCCESS));
    ret = HcclDisableMemAccess(comm, &remoteMemDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclDeregisterMem(comm, &localMemDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(localbuf);
    sal_free(remotebuf);
    remove(file_name_t);
    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, ut_one_sided_service_mem_exchange)
{
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    void* comm;
    const char* rank_table_file = "./ut_opbase_test.json";

    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclMemDesc localDesc, remoteDesc;
    char str[21] = "aaaabbbbccccddddeeee";
    memcpy_s(localDesc.desc, sizeof(localDesc.desc), str, sizeof(str));
    memcpy_s(remoteDesc.desc, sizeof(remoteDesc.desc), str, sizeof(str));
    HcclMemDescs local;
    local.arrayLength = 1;
    local.array = &localDesc;
    HcclMemDescs remote;
    remote.arrayLength = 1;
    remote.array = &remoteDesc;
    u32 actualNum;
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    IHcclOneSidedService* iService = nullptr;
    hcclComm->GetOneSidedService(&iService);

    NetDevContext devContext;
    devContext.nicType_ = NicType::VNIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &devContext;
    iService->netDevIpcCtx_ = &devContext;
    EXPECT_NE(iService, nullptr);
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);
    service->isUsedRdmaMap_.insert({1, false});

    MOCKER(hrtRaGetSingleSocketVnicIpInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclRankLinkInfo localLinkInfo{};
    HcclRankLinkInfo remoteLinkInfo{};
    remoteLinkInfo.userRank = 1;
    std::unique_ptr<HcclSocketManager> socketManager = nullptr;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    std::unique_ptr<NotifyPool> notifyPool;
    HcclDispatcher dispatcher;
    std::shared_ptr<HcclOneSidedConn> connPtr = make_shared<HcclOneSidedConn>(
        devCtx, localLinkInfo, remoteLinkInfo, socketManager, notifyPool, dispatcher, false, 0U, 0U);
    service->oneSidedConns_.insert({1, connPtr});
    HcclIpAddress ipAddr;
    std::shared_ptr<HcclSocket> socketPtr1
        = make_shared<HcclSocket>("tag", devCtx, ipAddr, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT);
    connPtr->socket_ = socketPtr1;
    connPtr->transportMemPtr_->SetDataSocket(socketPtr1);

    MOCKER_CPP(&HcclSocket::Send, HcclResult(HcclSocket::*)(const void*, u64))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclSocket::Recv, HcclResult(HcclSocket::*)(void*, u32))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    ret = HcclExchangeMemDesc(comm, 1, &local, 120, &remote, &actualNum);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    std::shared_ptr<HcclSocket> socketPtr2
        = make_shared<HcclSocket>("tag", devCtx, ipAddr, 16666, HcclSocketRole::SOCKET_ROLE_SERVER);
    connPtr->socket_ = socketPtr2;
    connPtr->transportMemPtr_->SetDataSocket(socketPtr2);
    ret = HcclExchangeMemDesc(comm, 1, &local, 120, &remote, &actualNum);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name_t);
    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, ut_one_sided_service_mem_exchange_create_socket_failed)
{
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    void* comm;
    const char* rank_table_file = "./ut_opbase_test.json";

    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    IHcclOneSidedService* iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    EXPECT_NE(iService, nullptr);

    NetDevContext devContext;
    devContext.nicType_ = NicType::VNIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &devContext;
    iService->netDevIpcCtx_ = &devContext;
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);
    service->isUsedRdmaMap_.insert({1, false});

    MOCKER_CPP(&HcclSocketManager::CreateSingleLinkSocket).stubs().will(returnValue(HCCL_E_PTR));

    HcclMemDescs local;
    HcclMemDescs remote;
    u32 actualNum;
    ret = HcclExchangeMemDesc(comm, 1, &local, 120, &remote, &actualNum);
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name_t);
    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, ut_one_sided_service_mem_exchange_sendReceive_roce_0)
{
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    int ret = HCCL_SUCCESS;
    void* comm;
    const char* rank_table_file = "./ut_opbase_test.json";

    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    NetDevContext devContext;
    devContext.nicType_ = NicType::DEVICE_NIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &devContext;

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    IHcclOneSidedService* iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    EXPECT_NE(iService, nullptr);
    iService->netDevRdmaCtx_ = devCtx;
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);

    HcclRankLinkInfo localLinkInfo{};
    HcclRankLinkInfo remoteLinkInfo{};
    remoteLinkInfo.userRank = 1;
    std::unique_ptr<HcclSocketManager> socketManager = nullptr;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    std::unique_ptr<NotifyPool> notifyPool;
    HcclDispatcher dispatcher;
    std::shared_ptr<HcclOneSidedConn> connPtr = make_shared<HcclOneSidedConn>(
        devCtx, localLinkInfo, remoteLinkInfo, socketManager, notifyPool, dispatcher, true, 0U, 0U);
    service->oneSidedConns_.insert({1, connPtr});
    service->isUsedRdmaMap_.insert({1, true});

    TransportRoceMem* transport = dynamic_cast<TransportRoceMem*>(connPtr->transportMemPtr_.get());
    MOCKER_CPP_VIRTUAL(*transport, &TransportRoceMem::ExchangeMemDesc).stubs().will(returnValue(HCCL_SUCCESS));

    HcclMemDesc localMemDesc;
    HcclMemDesc remoteMemDesc;
    HcclMemDescs local = {&localMemDesc, 1};
    HcclMemDescs remote = {&remoteMemDesc, 1};
    u32 actualNum = 0;
    std::string commIdentifier = "hccl_world_group";
    ret = service->ExchangeMemDesc(1, local, remote, actualNum, commIdentifier, 10);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, ut_one_sided_service_BatchPut_BatchPut_error_param)
{
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* localbuf;
    s8* remotebuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = 1024;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;
    s32 ndev = 8;
    // 走1910 8pring
    const char* rank_table_file = "./ut_opbase_test.json";

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    localbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    remotebuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(remotebuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++) {
        localbuf[j] = 2;
    }
    u32 itemNum = 1;
    HcclOneSideOpDesc desc[itemNum];
    desc[0].count = 1024;
    desc[0].dataType = HCCL_DATA_TYPE_INT8;
    desc[0].localAddr = localbuf;
    desc[0].remoteAddr = remotebuf;
    ret = HcclBatchPut(comm, 1, nullptr, itemNum, stream);

    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclBatchGet(comm, 1, nullptr, itemNum, stream);

    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclBatchPut(nullptr, 1, desc, itemNum, stream);

    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclBatchGet(nullptr, 1, desc, itemNum, stream);

    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclBatchPut(comm, 1, desc, itemNum, nullptr);

    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclBatchGet(comm, 1, desc, itemNum, nullptr);

    EXPECT_EQ(ret, HCCL_E_PTR);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(localbuf);
    sal_free(remotebuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, one_sided_service_HcclBatchPut_HcclBatchGet_success_roce)
{
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* localbuf;
    s8* remotebuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = 1024;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;
    s32 ndev = 8;
    // 走1910 8pring
    const char* rank_table_file = "./ut_opbase_test.json";

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    localbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    remotebuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(remotebuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++) {
        localbuf[j] = 2;
    }
    u32 itemNum = 1;
    HcclOneSideOpDesc desc[itemNum];
    desc[0].count = 1024;
    desc[0].dataType = HCCL_DATA_TYPE_INT8;
    desc[0].localAddr = localbuf;
    desc[0].remoteAddr = remotebuf;

    NetDevContext devContext;
    devContext.nicType_ = NicType::DEVICE_NIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &devContext;
    HcclRankLinkInfo localLinkInfo{};
    HcclRankLinkInfo remoteLinkInfo{};
    remoteLinkInfo.userRank = 1;
    std::unique_ptr<HcclSocketManager> socketManager = nullptr;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    std::unique_ptr<NotifyPool> notifyPool;
    DispatcherPub dispatcherPub(0);
    HcclDispatcher dispatcher = &dispatcherPub;

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);

    IHcclOneSidedService* iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    EXPECT_NE(iService, nullptr);
    std::string commIdentifier = hcclComm->GetIdentifier();
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);
    std::shared_ptr<hccl::HcclOneSidedConn> connPtr = nullptr;
    connPtr.reset(new hccl::HcclOneSidedConn(
        devCtx, localLinkInfo, remoteLinkInfo, socketManager, notifyPool, dispatcher, true, 0U, 0U));
    service->oneSidedConns_.insert({1, connPtr});
    TransportRoceMem* transport = dynamic_cast<TransportRoceMem*>(connPtr->transportMemPtr_.get());

    BufferKey<uintptr_t, u64> tempLocalKey(reinterpret_cast<uintptr_t>(localbuf), count * sizeof(s8));
    std::shared_ptr<LocalRdmaRmaBuffer> tempLocalBufferPtr
        = make_shared<LocalRdmaRmaBuffer>(devCtx, localbuf, count * sizeof(s8));
    tempLocalBufferPtr->devAddr = localbuf;
    devContext.localRdmaRmaBufferMgr_->Add(tempLocalKey, tempLocalBufferPtr);

    BufferKey<uintptr_t, u64> tempRemoteKey(reinterpret_cast<uintptr_t>(remotebuf), count * sizeof(s8));
    RemoteRdmaRmaBuffer tempRemoteBufferPtr{};
    tempRemoteBufferPtr.addr = remotebuf;
    tempRemoteBufferPtr.size = count * sizeof(s8);
    tempRemoteBufferPtr.devAddr = remotebuf;
    connPtr->remoteRmaBufferMgr_.Add(tempRemoteKey, reinterpret_cast<void*>(&tempRemoteBufferPtr));

    MOCKER(HrtRaSendWrV2).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(
        &DispatcherPub::RdmaSend,
        HcclResult(DispatcherPub::*)(u32, u64, const struct send_wr&, HcclRtStream, hccl::RdmaType, u64, u64, bool))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(*transport, &TransportRoceMem::AddOpFence, HcclResult(TransportRoceMem::*)(const rtStream_t&))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));

    ret = HcclBatchPut(comm, 1, desc, itemNum, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclBatchGet(comm, 1, desc, itemNum, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(localbuf);
    sal_free(remotebuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, ut_one_sided_service_HcclBatchPut_HcclBatchGet_success_ipc)
{
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* localbuf;
    s8* remotebuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = 1024;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;
    s32 ndev = 8;
    // 走1910 8pring
    const char* rank_table_file = "./ut_opbase_test.json";

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    localbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    remotebuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(remotebuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++) {
        localbuf[j] = 2;
    }
    u32 itemNum = 1;
    HcclOneSideOpDesc desc[itemNum];
    desc[0].count = 1024;
    desc[0].dataType = HCCL_DATA_TYPE_INT8;
    desc[0].localAddr = localbuf;
    desc[0].remoteAddr = remotebuf;

    NetDevContext devContext;
    devContext.nicType_ = NicType::VNIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &devContext;
    HcclRankLinkInfo localLinkInfo{};
    HcclRankLinkInfo remoteLinkInfo{};
    remoteLinkInfo.userRank = 1;
    std::unique_ptr<HcclSocketManager> socketManager = nullptr;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    std::unique_ptr<NotifyPool> notifyPool;
    DispatcherPub dispatcherPub(0);
    HcclDispatcher dispatcher = &dispatcherPub;

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);

    IHcclOneSidedService* iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    EXPECT_NE(iService, nullptr);
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);
    std::shared_ptr<hccl::HcclOneSidedConn> connPtr = nullptr;
    connPtr.reset(new hccl::HcclOneSidedConn(
        devCtx, localLinkInfo, remoteLinkInfo, socketManager, notifyPool, dispatcher, false, 0U, 0U));
    service->oneSidedConns_.insert({1, connPtr});

    TransportIpcMem* transport = dynamic_cast<TransportIpcMem*>(connPtr->transportMemPtr_.get());
    BufferKey<uintptr_t, u64> tempLocalKey(reinterpret_cast<uintptr_t>(localbuf), count * sizeof(s8));
    std::shared_ptr<LocalIpcRmaBuffer> tempLocalBufferPtr
        = make_shared<LocalIpcRmaBuffer>(devCtx, localbuf, count * sizeof(s8));
    tempLocalBufferPtr->devAddr = localbuf;
    devContext.localIpcRmaBufferMgr_->Add(tempLocalKey, tempLocalBufferPtr);

    BufferKey<uintptr_t, u64> tempRemoteKey(reinterpret_cast<uintptr_t>(remotebuf), count * sizeof(s8));
    RemoteIpcRmaBuffer tempRemoteBufferPtr{devCtx};
    tempRemoteBufferPtr.addr = remotebuf;
    tempRemoteBufferPtr.size = count * sizeof(s8);
    tempRemoteBufferPtr.devAddr = remotebuf;
    tempRemoteBufferPtr.memType = RmaMemType::DEVICE;
    connPtr->remoteRmaBufferMgr_.Add(tempRemoteKey, reinterpret_cast<void*>(&tempRemoteBufferPtr));

    MOCKER_CPP(
        &DispatcherPub::MemcpyAsync,
        HcclResult(DispatcherPub::*)(
            void*, uint64_t, const void*, u64, HcclRtMemcpyKind, hccl::Stream&, u32, hccl::LinkType))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    ret = HcclBatchPut(comm, 1, desc, itemNum, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclBatchGet(comm, 1, desc, itemNum, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(localbuf);
    sal_free(remotebuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, ut_one_sided_service_HcclBatchPut_HcclBatchGet_mem_not_found)
{
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* localbuf;
    s8* remotebuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = 1024;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;
    s32 ndev = 8;
    // 走1910 8pring
    const char* rank_table_file = "./ut_opbase_test.json";

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    localbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    remotebuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(remotebuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++) {
        localbuf[j] = 2;
    }
    u32 itemNum = 1;
    HcclOneSideOpDesc desc[itemNum];
    desc[0].count = 1024;
    desc[0].dataType = HCCL_DATA_TYPE_INT8;
    desc[0].localAddr = localbuf;
    desc[0].remoteAddr = remotebuf;

    NetDevContext devContext;
    devContext.nicType_ = NicType::DEVICE_NIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &devContext;
    HcclRankLinkInfo localLinkInfo{};
    HcclRankLinkInfo remoteLinkInfo{};
    remoteLinkInfo.userRank = 1;
    std::unique_ptr<HcclSocketManager> socketManager = nullptr;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    std::unique_ptr<NotifyPool> notifyPool;
    HcclDispatcher dispatcher;

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);

    IHcclOneSidedService* iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    EXPECT_NE(iService, nullptr);
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);
    std::shared_ptr<hccl::HcclOneSidedConn> connPtr = nullptr;
    connPtr.reset(new hccl::HcclOneSidedConn(
        devCtx, localLinkInfo, remoteLinkInfo, socketManager, notifyPool, dispatcher, true, 0U, 0U));
    service->oneSidedConns_.insert({1, connPtr});

    TransportRoceMem* transport = dynamic_cast<TransportRoceMem*>(connPtr->transportMemPtr_.get());
    MOCKER_CPP_VIRTUAL(*transport, &TransportRoceMem::AddOpFence, HcclResult(TransportRoceMem::*)(const rtStream_t&))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportRoceMem::TransportRdmaWithType).stubs().will(returnValue(HCCL_SUCCESS));

    ret = HcclBatchPut(comm, 1, desc, itemNum, stream);

    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    ret = HcclBatchGet(comm, 1, desc, itemNum, stream);

    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(localbuf);
    sal_free(remotebuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    GlobalMockObject::verify();
}
#if 0 // 执行失败内存泄漏
TEST_F(OneSidedSt, ut_one_sided_service_DeinitOneSidedCtx)
{
    typedef HcclResult (*HcclOneSideServiceCallBack)(std::unique_ptr<hccl::IHcclOneSidedService> &,
    std::unique_ptr<hccl::HcclSocketManager> &, std::unique_ptr<hccl::NotifyPool> &);
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    s8* localbuf;
    s32 count = 1024;
    void* comm;
    const char* rank_table_file = "./ut_opbase_test.json";

    localbuf= (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    IHcclOneSidedService *iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    EXPECT_NE(iService, nullptr);
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);

    std::string localdesc = "localdesc1";
    MOCKER_CPP(&LocalRdmaRmaBuffer::Serialize)
    .stubs()
    .will(returnValue(localdesc));
    MOCKER_CPP(&LocalRdmaRmaBuffer::Init)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputIntraRoceSwitch).stubs().will(returnValue(1U));

    constexpr u64 ONE_SIDE_HOST_MEM_MAX_SIZE = 1024llu * 1024 * 1024 * 1024;  // host侧支持内存注册大小上限为1TB
    HcclMemDesc regMemDesc;
    ret = HcclRegisterMem(comm, 1, 1, localbuf, ONE_SIDE_HOST_MEM_MAX_SIZE, &regMemDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclDeregisterMem(comm, &regMemDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(localbuf);
    remove(file_name_t);
    GlobalMockObject::verify();
}
#endif
TEST_F(OneSidedSt, ut_one_sided_service_RegHostMemMax)
{
    typedef HcclResult (*HcclOneSideServiceCallBack)(
        std::unique_ptr<hccl::IHcclOneSidedService>&, std::unique_ptr<hccl::HcclSocketManager>&,
        std::unique_ptr<hccl::NotifyPool>&);
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    s8* localbuf;
    s32 count = 1024;
    void* comm;
    const char* rank_table_file = "./ut_opbase_test.json";

    localbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    NetDevContext devContext;
    devContext.nicType_ = NicType::HOST_NIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    IHcclOneSidedService* iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    iService->netDevRdmaCtx_ = &devContext;
    EXPECT_NE(iService, nullptr);
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);

    std::string localdesc = "ld";
    MOCKER_CPP(&LocalRdmaRmaBuffer::Serialize).stubs().will(returnValue(localdesc));
    MOCKER_CPP(&LocalRdmaRmaBuffer::Init).stubs().will(returnValue(HCCL_SUCCESS));

    constexpr u64 ONE_SIDE_HOST_MEM_MAX_SIZE = 1024llu * 1024 * 1024 * 1024; // host侧支持内存注册大小上限为1TB
    HcclMemDesc regMemDesc;
    HcclMemDesc regMemDesc1;
    ret = HcclRegisterMem(comm, 1, 1, localbuf, ONE_SIDE_HOST_MEM_MAX_SIZE, &regMemDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclRegisterMem(comm, 2, 1, localbuf, ONE_SIDE_HOST_MEM_MAX_SIZE, &regMemDesc1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclRegisterMem(comm, 1, 1, localbuf, ONE_SIDE_HOST_MEM_MAX_SIZE + 1, &regMemDesc);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcclDeregisterMem(comm, &regMemDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclDeregisterMem(comm, &regMemDesc1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(localbuf);
    remove(file_name_t);
    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, ut_one_sided_service_RegDevMemMax)
{
    typedef HcclResult (*HcclOneSideServiceCallBack)(
        std::unique_ptr<hccl::IHcclOneSidedService>&, std::unique_ptr<hccl::HcclSocketManager>&,
        std::unique_ptr<hccl::NotifyPool>&);
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    s8* localbuf;
    s32 count = 1024;
    void* comm;
    const char* rank_table_file = "./ut_opbase_test.json";

    localbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    NetDevContext devContext;
    devContext.nicType_ = NicType::DEVICE_NIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    IHcclOneSidedService* iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    iService->netDevRdmaCtx_ = &devContext;
    EXPECT_NE(iService, nullptr);
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);

    std::string localdesc = "ld";
    MOCKER_CPP(&LocalRdmaRmaBuffer::Serialize).stubs().will(returnValue(localdesc));
    MOCKER_CPP(&LocalRdmaRmaBuffer::Init).stubs().will(returnValue(HCCL_SUCCESS));

    constexpr u64 ONE_SIDE_DEVICE_MEM_MAX_SIZE = 64llu * 1024 * 1024 * 1024; // device侧支持内存注册大小上限为64GB
    HcclMemDesc regMemDesc;
    HcclMemDesc regMemDesc1;

    ret = HcclRegisterMem(comm, 1, 0, localbuf, ONE_SIDE_DEVICE_MEM_MAX_SIZE, &regMemDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclRegisterMem(comm, 1, 0, localbuf, ONE_SIDE_DEVICE_MEM_MAX_SIZE, &regMemDesc1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclRegisterMem(comm, 1, 0, localbuf, ONE_SIDE_DEVICE_MEM_MAX_SIZE + 1, &regMemDesc);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcclDeregisterMem(comm, &regMemDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclDeregisterMem(comm, &regMemDesc1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    sal_free(localbuf);
    remove(file_name_t);
    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, ut_one_sided_service_RegUnRegDevMemMaxCnt)
{
    typedef HcclResult (*HcclOneSideServiceCallBack)(
        std::unique_ptr<hccl::IHcclOneSidedService>&, std::unique_ptr<hccl::HcclSocketManager>&,
        std::unique_ptr<hccl::NotifyPool>&);
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    s8* localbuf;
    s32 count = 1024;
    void* comm;
    const char* rank_table_file = "./ut_opbase_test.json";

    localbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    NetDevContext devContext;
    devContext.nicType_ = NicType::DEVICE_NIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    IHcclOneSidedService* iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    iService->netDevRdmaCtx_ = &devContext;
    EXPECT_NE(iService, nullptr);
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);
    service->isUsedRdmaMap_.insert({1, true});

    MOCKER_CPP(&LocalRdmaRmaBuffer::Init).stubs().will(returnValue(HCCL_SUCCESS));

    constexpr u64 ONE_SIDE_DEVICE_MEM_SIZE = 1024 * 1024 * 1024;
    constexpr u64 ONE_SIDE_DEVICE_MEM_SIZE_1 = 1024 * 1024;
    HcclMemDesc regMemDesc;
    HcclMemDesc regMemDesc1;

    // 无资源非法释放
    ret = HcclDeregisterMem(comm, &regMemDesc1);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);

    // 申请重叠内存
    ret = HcclRegisterMem(comm, 1, 0, localbuf, ONE_SIDE_DEVICE_MEM_SIZE, &regMemDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclRegisterMem(comm, 1, 0, localbuf, ONE_SIDE_DEVICE_MEM_SIZE_1, &regMemDesc1);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    ret = HcclDeregisterMem(comm, &regMemDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 申请超次
    u32 regCntMax = 256;
    s8* subBuffers[regCntMax];
    HcclMemDesc subRegMemDesc[regCntMax];
    for (int i = 0; i < regCntMax; i++) {
        subBuffers[i] = localbuf + i * ONE_SIDE_DEVICE_MEM_SIZE_1; // 每块内存1M
        ret = HcclRegisterMem(comm, 1, 0, subBuffers[i], ONE_SIDE_DEVICE_MEM_SIZE_1, &subRegMemDesc[i]);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
    ret = HcclRegisterMem(
        comm, 1, 0, subBuffers[regCntMax] + ONE_SIDE_DEVICE_MEM_SIZE_1, ONE_SIDE_DEVICE_MEM_SIZE_1, &regMemDesc1);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);

    for (int i = 0; i < regCntMax; i++) {
        ret = HcclDeregisterMem(comm, &subRegMemDesc[i]);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    ret = HcclCommDestroy(comm);
    sal_free(localbuf);
    remove(file_name_t);
    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, ut_one_sided_service_RegUnRegDevMemMaxCnt_multi_remoteRank)
{
    typedef HcclResult (*HcclOneSideServiceCallBack)(
        std::unique_ptr<hccl::IHcclOneSidedService>&, std::unique_ptr<hccl::HcclSocketManager>&,
        std::unique_ptr<hccl::NotifyPool>&);
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    s8* localbuf;
    s32 count = 1024;
    void* comm;
    const char* rank_table_file = "./ut_opbase_test.json";

    localbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    NetDevContext devContext;
    devContext.nicType_ = NicType::DEVICE_NIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &devContext;

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    IHcclOneSidedService* iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    iService->netDevRdmaCtx_ = devCtx;
    EXPECT_NE(iService, nullptr);
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);
    bool useRdma = true;
    MOCKER_CPP(&HcclOneSidedService::IsUsedRdma)
        .stubs()
        .with(mockcpp::any(), outBound(useRdma))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&LocalRdmaRmaBuffer::Init).stubs().will(returnValue(HCCL_SUCCESS));

    constexpr u64 ONE_SIDE_DEVICE_MEM_SIZE = 1024 * 1024 * 1024;
    constexpr u64 ONE_SIDE_DEVICE_MEM_SIZE_1 = 1024 * 1024;
    HcclMemDesc regMemDesc;
    HcclMemDesc regMemDesc1;

    // 申请超次
    u32 regCntMax = 256;
    s8* subBuffers[regCntMax];
    HcclMemDesc subRegMemDesc[regCntMax];
    HcclMemDesc subRegMemDesc2[regCntMax];
    for (int i = 0; i < 255; i++) {
        subBuffers[i] = localbuf + i * ONE_SIDE_DEVICE_MEM_SIZE_1; // 每块内存1M
        ret = service->RegMem(subBuffers[i], ONE_SIDE_DEVICE_MEM_SIZE_1, HCCL_MEM_TYPE_DEVICE, 1, subRegMemDesc[i]);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        ret = service->RegMem(subBuffers[i], ONE_SIDE_DEVICE_MEM_SIZE_1, HCCL_MEM_TYPE_DEVICE, 2, subRegMemDesc2[i]);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
    ret = service->RegMem(subBuffers[255], ONE_SIDE_DEVICE_MEM_SIZE_1, HCCL_MEM_TYPE_DEVICE, 1, subRegMemDesc[255]);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = service->RegMem(subBuffers[255], ONE_SIDE_DEVICE_MEM_SIZE_1, HCCL_MEM_TYPE_DEVICE, 2, subRegMemDesc2[255]);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);

    for (int i = 0; i < 255; i++) {
        ret = service->DeregMem(subRegMemDesc[i]);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        ret = service->DeregMem(subRegMemDesc2[i]);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
    ret = service->DeregMem(subRegMemDesc[255]);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(comm);
    sal_free(localbuf);
    remove(file_name_t);
    GlobalMockObject::verify();
}
#if 0 // 执行失败
TEST_F(OneSidedSt, ut_one_sided_service_InitNetDevCtx_Fail)
{
    typedef HcclResult (*HcclOneSideServiceCallBack)(std::unique_ptr<hccl::IHcclOneSidedService> &,
    std::unique_ptr<hccl::HcclSocketManager> &, std::unique_ptr<hccl::NotifyPool> &);
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    s8* localbuf;
    void* comm;
    s32 count = 1024;

    const char* rank_table_file = "./ut_opbase_test.json";
    localbuf = (s8*)sal_malloc(count * sizeof(s8));

    sal_memset(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);

    IHcclOneSidedService *iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    EXPECT_NE(iService, nullptr);
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);

    std::string localdesc = "ld";
    MOCKER_CPP(&LocalRdmaRmaBuffer::Serialize)
    .stubs()
    .will(returnValue(localdesc));
    MOCKER_CPP(&LocalRdmaRmaBuffer::Init)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::InitNic)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputIntraRoceSwitch).stubs().will(returnValue(1U));
    HcclMemDesc regMemDesc;
    ret = HcclRegisterMem(comm, 1, 0, localbuf, 1024, &regMemDesc);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(localbuf);
    remove(file_name_t);
    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, ut_one_sided_service_InitNetDevCtx_HostDeployment)
{
    typedef HcclResult (*HcclOneSideServiceCallBack)(std::unique_ptr<hccl::IHcclOneSidedService> &,
    std::unique_ptr<hccl::HcclSocketManager> &, std::unique_ptr<hccl::NotifyPool> &);
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    s8* localbuf;
    void* comm;
    s32 count = 1024;

    const char* rank_table_file = "./ut_opbase_test.json";
    localbuf = (s8*)sal_malloc(count * sizeof(s8));

    sal_memset(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    hcclComm->communicator_->nicDeployment_ = NICDeployment::NIC_DEPLOYMENT_HOST;

    IHcclOneSidedService *iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    EXPECT_NE(iService, nullptr);
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);

    std::string localdesc = "ld";
    MOCKER_CPP(&LocalRdmaRmaBuffer::Serialize)
    .stubs()
    .will(returnValue(localdesc));
    MOCKER_CPP(&LocalRdmaRmaBuffer::Init)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::InitNic)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    HcclMemDesc regMemDesc;
    ret = HcclRegisterMem(comm, 1, 0, localbuf, 1024, &regMemDesc);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(localbuf);
    remove(file_name_t);
    GlobalMockObject::verify();
}
#endif
TEST_F(OneSidedSt, ut_one_sided_service_setNetDevCtx)
{
    typedef HcclResult (*HcclOneSideServiceCallBack)(
        std::unique_ptr<hccl::IHcclOneSidedService>&, std::unique_ptr<hccl::HcclSocketManager>&,
        std::unique_ptr<hccl::NotifyPool>&);
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    void* comm;
    s32 count = 1024;

    const char* rank_table_file = "./ut_opbase_test.json";
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);

    IHcclOneSidedService* iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    EXPECT_NE(iService, nullptr);

    HcclNetDevCtx* devCtx;
    bool useRdma = false;
    devCtx = (HcclNetDevCtx*)sal_malloc(sizeof(HcclNetDevCtx));
    sal_memset(devCtx, sizeof(HcclNetDevCtx), 0, sizeof(HcclNetDevCtx));
    ret = iService->SetNetDevCtx(devCtx, useRdma);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(devCtx);
    remove(file_name_t);
    GlobalMockObject::verify();
}
#if 0 // 执行失败内存泄漏
TEST_F(OneSidedSt, ut_one_sided_service_init_devIpAddrEmpty)
{
    typedef HcclResult (*HcclOneSideServiceCallBack)(std::unique_ptr<hccl::IHcclOneSidedService> &,
    std::unique_ptr<hccl::HcclSocketManager> &, std::unique_ptr<hccl::NotifyPool> &);
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    void* comm;
    s32 count = 1024;

    const char* rank_table_file = "./ut_opbase_test.json";
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    hcclComm->communicator_->devIpAddr_.clear();

    RankTable_t rankTable;
    ret = hcclComm->communicator_->InitOneSidedService(rankTable);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name_t);
    GlobalMockObject::verify();
}
#endif
TEST_F(OneSidedSt, ut_one_sided_service_BatchPut_Rma_Buffer_test)
{
    nlohmann::json rank_table = rank_table_910_1server_4rank;
    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* localbuf;
    s8* remotebuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = 1024;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;
    s32 ndev = 8;
    // 走1910 8pring
    const char* rank_table_file = "./ut_opbase_test.json";

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    localbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    remotebuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(remotebuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++) {
        localbuf[j] = 2;
    }
    u32 itemNum = 1;
    HcclOneSideOpDesc desc[itemNum];
    desc[0].count = 1024;
    desc[0].dataType = HCCL_DATA_TYPE_INT8;
    desc[0].localAddr = localbuf;
    desc[0].remoteAddr = remotebuf;

    NetDevContext devContext;
    devContext.nicType_ = NicType::DEVICE_NIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &devContext;
    HcclRankLinkInfo localLinkInfo{};
    HcclRankLinkInfo remoteLinkInfo{};
    remoteLinkInfo.userRank = 1;
    std::unique_ptr<HcclSocketManager> socketManager = nullptr;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));

    std::unique_ptr<NotifyPool> notifyPool;
    HcclDispatcher dispatcher;
    BufferKey<uintptr_t, u64> tempLocalKey(reinterpret_cast<uintptr_t>(localbuf), count * sizeof(s8));
    std::shared_ptr<LocalRdmaRmaBuffer> tempLocalBufferPtr;
    tempLocalBufferPtr.reset(new LocalRdmaRmaBuffer(devCtx, localbuf, count * sizeof(s8)));
    tempLocalBufferPtr->devAddr = localbuf;
    tempLocalBufferPtr->pimpl_.reset(
        new LocalRdmaRmaBufferImpl(devCtx, localbuf, count * sizeof(s8), RmaMemType::DEVICE));

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);

    IHcclOneSidedService* iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    EXPECT_NE(iService, nullptr);
    std::string commIdentifier = hcclComm->GetIdentifier();
    HcclOneSidedService* service = reinterpret_cast<HcclOneSidedService*>(iService);
    std::shared_ptr<hccl::HcclOneSidedConn> connPtr = nullptr;
    connPtr.reset(new hccl::HcclOneSidedConn(
        devCtx, localLinkInfo, remoteLinkInfo, socketManager, notifyPool, dispatcher, true, 0U, 0U));

    BufferKey<uintptr_t, u64> tempRemoteKey(reinterpret_cast<uintptr_t>(remotebuf), count * sizeof(s8));
    RemoteRdmaRmaBuffer tempRemoteBufferPtr{};
    tempRemoteBufferPtr.pimpl_.reset(new RemoteRdmaRmaBufferImpl());
    tempRemoteBufferPtr.addr = remotebuf;
    tempRemoteBufferPtr.size = count * sizeof(s8);
    tempRemoteBufferPtr.devAddr = remotebuf;
    service->oneSidedConns_.insert({1, connPtr});
    TransportRoceMem* transport = dynamic_cast<TransportRoceMem*>(connPtr->transportMemPtr_.get());
    connPtr->remoteRmaBufferMgr_.Add(tempRemoteKey, reinterpret_cast<void*>(&tempRemoteBufferPtr));
    devContext.localRdmaRmaBufferMgr_->Add(tempLocalKey, tempLocalBufferPtr);

    MOCKER_CPP_VIRTUAL(*transport, &TransportRoceMem::AddOpFence, HcclResult(TransportRoceMem::*)(const rtStream_t&))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportRoceMem::TransportRdmaWithType).stubs().will(returnValue(HCCL_SUCCESS));

    ret = HcclBatchPut(comm, 1, desc, itemNum, stream);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));
    ret = HcclBatchGet(comm, 1, desc, itemNum, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(localbuf);
    sal_free(remotebuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    GlobalMockObject::verify();
}
#if 0 // 执行失败ut_opbase_test.json
TEST_F(OneSidedSt, ut_one_sided_service_mem_RemapRegistedMemory_roce_01)
{
    typedef HcclResult (*HcclOneSideServiceCallBack)(std::unique_ptr<hccl::IHcclOneSidedService> &,
    std::unique_ptr<hccl::HcclSocketManager> &, std::unique_ptr<hccl::NotifyPool> &);
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./llt/ace/comop/hccl/stub/workspace/ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    s8* localbuf[2];
    s8* remotebuf[2];
    s32 count = 1024;
    void* comm[2];
    char* rank_table_file = "./llt/ace/comop/hccl/stub/workspace/ut_opbase_test.json";

    CommMem memInfoArray[2];
    int test = 3;
    for (int j = 0; j < 2; j++) {
        memInfoArray[j].size = 2;
        memInfoArray[j].type = COMM_MEM_TYPE_DEVICE;
    }

    NetDevContext devContext;
    devContext.nicType_ = NicType::DEVICE_NIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    hccl::hcclComm* hcclComm[2];
    HcclMemDesc localMemDesc[2];
    for (int i = 0; i < 1; i++) {
        localbuf[i]= (s8*)sal_malloc(count * sizeof(s8));
        sal_memset(localbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        remotebuf[i]= (s8*)sal_malloc(count * sizeof(s8));
        sal_memset(remotebuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        memInfoArray[i].addr = localbuf[i];
        ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm[i]);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        HcclNetDevCtx devCtx = &devContext;
        HcclRankLinkInfo localLinkInfo {};
        HcclRankLinkInfo remoteLinkInfo {};
        remoteLinkInfo.userRank = 1;
        std::unique_ptr<HcclSocketManager> socketManager = nullptr;
        socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
        std::unique_ptr<NotifyPool> notifyPool;
        HcclDispatcher dispatcher;
        hcclComm[i] = static_cast<hccl::hcclComm *>(comm[i]);

        IHcclOneSidedService *iService = nullptr;
        hcclComm[i]->GetOneSidedService(&iService);
        EXPECT_NE(iService, nullptr);
        std::string commIdentifier = hcclComm[i]->GetIdentifier();
        iService->netDevRdmaCtx_ = devCtx;
        HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);

        std::shared_ptr<HcclOneSidedConn> connPtr = make_shared<HcclOneSidedConn>(devCtx, localLinkInfo,
        remoteLinkInfo, socketManager,  notifyPool, dispatcher, true, 0U, 0U);
        service->oneSidedConns_.insert({1, connPtr});

        std::string localdesc = "ld";
        MOCKER_CPP(&LocalRdmaRmaBuffer::Serialize)
        .stubs()
        .will(returnValue(localdesc));
        MOCKER_CPP(&LocalRdmaRmaBuffer::Init)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
        u32 remoteRankId = 1;
        bool useRdma = true;
        MOCKER_CPP(&HcclOneSidedService::IsUsedRdma)
        .stubs()
        .with(eq(remoteRankId), outBound(useRdma))
        .will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&RemoteRdmaRmaBuffer::Deserialize)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

        ret = HcclRegisterMem(comm[i], remoteRankId, 0, localbuf[i], 1024, &localMemDesc[i]);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    MOCKER(ra_remap_mr)
    .expects(atMost(10))
    .will(returnValue(0));

    MOCKER(ra_remap_mr)
    .expects(atMost(10))
    .will(returnValue(0));

    ret = HcclRemapRegistedMemory(comm, memInfoArray, 1, 2);
    EXPECT_EQ(ret, HCCL_E_PTR);

    for (int j = 0; j < 1; j++) {
        ret = HcclDeregisterMem(comm[j], &localMemDesc[j]);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        ret = HcclCommDestroy(comm[j]);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        sal_free(localbuf[j]);
        sal_free(remotebuf[j]);

    }
    remove(file_name_t);
    GlobalMockObject::verify();

}

TEST_F(OneSidedSt, ut_one_sided_service_mem_RemapRegistedMemory_roce_02)
{
    typedef HcclResult (*HcclOneSideServiceCallBack)(std::unique_ptr<hccl::IHcclOneSidedService> &,
    std::unique_ptr<hccl::HcclSocketManager> &, std::unique_ptr<hccl::NotifyPool> &);
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./llt/ace/comop/hccl/stub/workspace/ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    s8* localbuf[2];
    s8* remotebuf[2];
    s32 count = 1024;
    void* comm[2];
    char* rank_table_file = "./llt/ace/comop/hccl/stub/workspace/ut_opbase_test.json";

    CommMem memInfoArray[2];
    int test = 3;
    for (int j = 0; j < 2; j++) {
        memInfoArray[j].addr = &test;
        memInfoArray[j].size = 2;
        memInfoArray[j].type = COMM_MEM_TYPE_DEVICE;
    }

    NetDevContext devContext;
    devContext.nicType_ = NicType::DEVICE_NIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    hccl::hcclComm* hcclComm[2];
    HcclMemDesc localMemDesc[2];
    for (int i = 0; i < 1; i++) {
        localbuf[i]= (s8*)sal_malloc(count * sizeof(s8));
        sal_memset(localbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        remotebuf[i]= (s8*)sal_malloc(count * sizeof(s8));
        sal_memset(remotebuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        memInfoArray[i].addr = localbuf[i];
        ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm[i]);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        HcclNetDevCtx devCtx = &devContext;
        HcclRankLinkInfo localLinkInfo {};
        HcclRankLinkInfo remoteLinkInfo {};
        remoteLinkInfo.userRank = 1;
        std::unique_ptr<HcclSocketManager> socketManager = nullptr;
        socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
        std::unique_ptr<NotifyPool> notifyPool;
        HcclDispatcher dispatcher;
        hcclComm[i] = static_cast<hccl::hcclComm *>(comm[i]);

        IHcclOneSidedService *iService = nullptr;
        hcclComm[i]->GetOneSidedService(&iService);
        EXPECT_NE(iService, nullptr);
        std::string commIdentifier = hcclComm[i]->GetIdentifier();
        iService->netDevRdmaCtx_ = devCtx;
        HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);

        std::shared_ptr<HcclOneSidedConn> connPtr = make_shared<HcclOneSidedConn>(devCtx, localLinkInfo,
        remoteLinkInfo, socketManager, notifyPool, dispatcher, true, 0U, 0U);

        service->oneSidedConns_.insert({1, connPtr});

        std::string localdesc = "ld";
        MOCKER_CPP(&LocalRdmaRmaBuffer::Serialize)
        .stubs()
        .will(returnValue(localdesc));
        MOCKER_CPP(&LocalRdmaRmaBuffer::Init)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
        u32 remoteRankId = 1;
        bool useRdma = true;
        MOCKER_CPP(&HcclOneSidedService::IsUsedRdma)
        .stubs()
        .with(eq(remoteRankId), outBound(useRdma))
        .will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&RemoteRdmaRmaBuffer::Deserialize)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
        ret = HcclRegisterMem(comm[i], remoteRankId, 0, localbuf[i], 1024, &localMemDesc[i]);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    MOCKER(ra_remap_mr)
    .expects(atMost(10))
    .will(returnValue(0));

    MOCKER_CPP(&LocalRdmaRmaBuffer::Remap)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    ret = HcclRemapRegistedMemory(comm, memInfoArray, 1, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < 1; j++) {
        ret = HcclDeregisterMem(comm[j], &localMemDesc[j]);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        ret = HcclCommDestroy(comm[j]);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        sal_free(localbuf[j]);
        sal_free(remotebuf[j]);

    }
    remove(file_name_t);
    GlobalMockObject::verify();
}
#endif
TEST_F(OneSidedSt, ut_one_sided_service_conn_conect)
{
    NetDevContext devContext;
    devContext.nicType_ = NicType::DEVICE_NIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &devContext;
    HcclRankLinkInfo localLinkInfo{};
    HcclRankLinkInfo remoteLinkInfo{};
    localLinkInfo.userRank = 0;
    remoteLinkInfo.userRank = 1;
    std::unique_ptr<HcclSocketManager> socketManager
        = make_unique<HcclSocketManager>(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0);
    std::unique_ptr<NotifyPool> notifyPool;
    HcclDispatcher dispatcher;
    std::shared_ptr<HcclOneSidedConn> connPtr = make_shared<HcclOneSidedConn>(
        devCtx, localLinkInfo, remoteLinkInfo, socketManager, notifyPool, dispatcher, true, 0U, 0U);

    HcclIpAddress ipAddr;
    std::shared_ptr<HcclSocket> socketPtr
        = make_shared<HcclSocket>("tag", devCtx, ipAddr, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT);
    std::vector<std::shared_ptr<HcclSocket>> connectSockets;
    connectSockets.push_back(socketPtr);
    MOCKER_CPP(&HcclSocketManager::CreateSingleLinkSocket)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(connectSockets), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    TransportRoceMem* transport = dynamic_cast<TransportRoceMem*>(connPtr->transportMemPtr_.get());
    MOCKER_CPP_VIRTUAL(*transport, &TransportRoceMem::ExchangeMemDesc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(*transport, &TransportRoceMem::Connect).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(*transport, &TransportRoceMem::SetSocket).stubs().will(returnValue(HCCL_SUCCESS));

    std::string commIdentifier("dejj");
    HcclResult ret = connPtr->Connect(commIdentifier, 10);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, regmem_without_deregmem)
{
    int ret = HCCL_SUCCESS;

    NetDevContext netDevCtx;
    ;
    netDevCtx.nicType_ = NicType::DEVICE_NIC_TYPE;
    netDevCtx.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &netDevCtx;

    NetDevContext vNetDevCtx;
    vNetDevCtx.nicType_ = NicType::VNIC_TYPE;
    vNetDevCtx.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    HcclNetDevCtx vDevCtx = &vNetDevCtx;

    unique_ptr<HcclSocketManager> socketManager
        = std::make_unique<HcclSocketManager>(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0);
    unique_ptr<NotifyPool> notifyPool = std::make_unique<NotifyPool>();
    HcclCommConfig commConfig("hccl_world_group");
    unique_ptr<HcclOneSidedService> service
        = std::make_unique<HcclOneSidedService>(socketManager, notifyPool, commConfig);
    service->SetNetDevCtx(devCtx, true);
    service->SetNetDevCtx(vDevCtx, false);

    u64 count = 1024;
    u64 bufSize = 1024 * sizeof(s8);
    s8* localbuf = (s8*)sal_malloc(bufSize);
    sal_memset(localbuf, bufSize, 0, bufSize);

    MOCKER_CPP(&LocalIpcRmaBuffer::Init).stubs().will(returnValue(HCCL_SUCCESS));
    std::string localdesc = "ipc_desc";
    MOCKER_CPP(&LocalIpcRmaBuffer::Serialize).stubs().will(returnValue(localdesc));

    MOCKER_CPP(&LocalRdmaRmaBuffer::Init).stubs().will(returnValue(HCCL_SUCCESS));
    std::string localdesc2 = "roce_desc";
    MOCKER_CPP(&LocalRdmaRmaBuffer::Serialize).stubs().will(returnValue(localdesc2));

    MOCKER_CPP(&HcclOneSidedService::IsUsedRdma).stubs().with(eq(1U), outBound(false)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclOneSidedService::IsUsedRdma).stubs().with(eq(2U), outBound(false)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclOneSidedService::IsUsedRdma).stubs().with(eq(3U), outBound(true)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclOneSidedService::IsUsedRdma).stubs().with(eq(4U), outBound(true)).will(returnValue(HCCL_SUCCESS));

    HcclMemDesc localMemDesc1;
    ret = service->RegMem(localbuf, bufSize, HCCL_MEM_TYPE_DEVICE, 1, localMemDesc1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclMemDesc localMemDesc2;
    ret = service->RegMem(localbuf, bufSize, HCCL_MEM_TYPE_DEVICE, 2, localMemDesc2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclMemDesc localMemDesc3;
    ret = service->RegMem(localbuf, bufSize, HCCL_MEM_TYPE_DEVICE, 3, localMemDesc3);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclMemDesc localMemDesc4;
    ret = service->RegMem(localbuf, bufSize, HCCL_MEM_TYPE_DEVICE, 4, localMemDesc4);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    service.reset(nullptr);
    sal_free(localbuf);
    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, remap_ipc_mem)
{
    int ret = HCCL_SUCCESS;

    NetDevContext vNetDevCtx;
    vNetDevCtx.nicType_ = NicType::VNIC_TYPE;
    vNetDevCtx.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    HcclNetDevCtx vDevCtx = &vNetDevCtx;

    unique_ptr<HcclSocketManager> socketManager
        = std::make_unique<HcclSocketManager>(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0);
    unique_ptr<NotifyPool> notifyPool = std::make_unique<NotifyPool>();
    HcclCommConfig commConfig("hccl_world_group");
    unique_ptr<HcclOneSidedService> service
        = std::make_unique<HcclOneSidedService>(socketManager, notifyPool, commConfig);
    service->SetNetDevCtx(vDevCtx, false);

    u64 count = 1024;
    u64 bufSize = 1024 * sizeof(s8);
    s8* localbuf = (s8*)sal_malloc(bufSize);
    sal_memset(localbuf, bufSize, 0, bufSize);

    MOCKER_CPP(&LocalIpcRmaBuffer::Init).stubs().will(returnValue(HCCL_SUCCESS));
    std::string localdesc = "ipc_desc";
    MOCKER_CPP(&LocalIpcRmaBuffer::Serialize).stubs().will(returnValue(localdesc));

    MOCKER_CPP(&HcclOneSidedService::IsUsedRdma).stubs().with(eq(1U), outBound(false)).will(returnValue(HCCL_SUCCESS));

    HcclMemDesc localMemDesc1;
    ret = service->RegMem(localbuf, bufSize, HCCL_MEM_TYPE_DEVICE, 1, localMemDesc1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclMem memInfoArray = {HCCL_MEM_TYPE_DEVICE, localbuf, 1};
    ret = service->ReMapMem(&memInfoArray, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    service.reset(nullptr);
    sal_free(localbuf);
    GlobalMockObject::verify();
}
#if 0
TEST_F(OneSidedSt, ipc_mem_regmem_deregmem)
{
    NetDevContext devContext;
    devContext.nicType_ = NicType::VNIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &devContext;

    u64 count = 1024;
    u64 bufSize = 1024 * sizeof(s8);
    s8* localbuf = (s8*)sal_malloc(bufSize);
    sal_memset(localbuf, bufSize, 0, bufSize);

    MOCKER_CPP(&LocalIpcRmaBuffer::Init)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    std::string localdesc = "lipcd";
    MOCKER_CPP(&LocalIpcRmaBuffer::Serialize)
    .stubs()
    .will(returnValue(localdesc));

    HcclBuf hcclBuf{};
    HcclMem localMem = {HCCL_MEM_TYPE_DEVICE, localbuf, bufSize};
    HcclResult ret = HcclMemReg(devCtx, &localMem, &hcclBuf);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(hcclBuf.addr, ((void*)localbuf));
    EXPECT_EQ(hcclBuf.len, 1024U);

    // 重复注册
    HcclBuf hcclBuf1{};
    HcclMem localMem1 = {HCCL_MEM_TYPE_DEVICE, localbuf, bufSize};
    ret = HcclMemReg(devCtx, &localMem1, &hcclBuf1);
    EXPECT_EQ(ret, HCCL_E_AGAIN);
    EXPECT_EQ(hcclBuf1.addr, ((void*)localbuf));
    EXPECT_EQ(hcclBuf1.len, 1024U);

    // 注册子集
    HcclBuf hcclBuf2{};
    HcclMem localMem2 = {HCCL_MEM_TYPE_DEVICE, localbuf, bufSize - 1};
    ret = HcclMemReg(devCtx, &localMem2, &hcclBuf2);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    // 注册交集
    HcclBuf hcclBuf3{};
    HcclMem localMem3 = {HCCL_MEM_TYPE_DEVICE, localbuf, bufSize + 1};
    ret = HcclMemReg(devCtx, &localMem3, &hcclBuf3);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    // 2次注册，第1次解注册
    ret = HcclMemDereg(&hcclBuf1);
    EXPECT_EQ(ret, HCCL_E_AGAIN);

    // 2次注册，第2次解注册
    ret = HcclMemDereg(&hcclBuf);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclMemReg(nullptr, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcclMemReg(devCtx, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcclMemReg(devCtx, &localMem2, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);

    HcclMem localMemE = {HCCL_MEM_TYPE_NUM, localbuf, bufSize};
    ret = HcclMemReg(devCtx, &localMemE, &hcclBuf);
    EXPECT_EQ(ret, HCCL_E_PARA);
    localMemE = {HCCL_MEM_TYPE_DEVICE, nullptr, bufSize};
    ret = HcclMemReg(devCtx, &localMemE, &hcclBuf);
    EXPECT_EQ(ret, HCCL_E_PARA);
    localMemE = {HCCL_MEM_TYPE_DEVICE, localbuf, 0U};
    ret = HcclMemReg(devCtx, &localMemE, &hcclBuf);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcclMemDereg(nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);

    hcclBuf3.addr = localbuf;
    hcclBuf3.len = 0;
    ret = HcclMemDereg(&hcclBuf3);
    EXPECT_EQ(ret, HCCL_E_PARA);

    hcclBuf3.addr = nullptr;
    hcclBuf3.len = 1;
    ret = HcclMemDereg(&hcclBuf3);
    EXPECT_EQ(ret, HCCL_E_PARA);

    sal_free(localbuf);
}

TEST_F(OneSidedSt, roce_mem_regmem_deregmem)
{
    NetDevContext devContext;
    devContext.nicType_ = NicType::DEVICE_NIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &devContext;

    u64 count = 1024;
    u64 bufSize = 1024 * sizeof(s8);
    s8* localbuf = (s8*)sal_malloc(bufSize);
    sal_memset(localbuf, bufSize, 0, bufSize);

    MOCKER_CPP(&LocalRdmaRmaBuffer::Init)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    std::string localdesc = "lipcd";
    MOCKER_CPP(&LocalRdmaRmaBuffer::Serialize)
    .stubs()
    .will(returnValue(localdesc));

    HcclBuf hcclBuf{};
    HcclMem localMem = {HCCL_MEM_TYPE_DEVICE, localbuf, bufSize};
    HcclResult ret = HcclMemReg(devCtx, &localMem, &hcclBuf);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(hcclBuf.addr, ((void*)localbuf));
    EXPECT_EQ(hcclBuf.len, 1024U);

    // 重复注册
    HcclBuf hcclBuf1{};
    HcclMem localMem1 = {HCCL_MEM_TYPE_DEVICE, localbuf, bufSize};
    ret = HcclMemReg(devCtx, &localMem1, &hcclBuf1);
    EXPECT_EQ(ret, HCCL_E_AGAIN);
    EXPECT_EQ(hcclBuf1.addr, ((void*)localbuf));
    EXPECT_EQ(hcclBuf1.len, 1024U);

    // 注册子集
    HcclBuf hcclBuf2{};
    HcclMem localMem2 = {HCCL_MEM_TYPE_DEVICE, localbuf, bufSize - 1};
    ret = HcclMemReg(devCtx, &localMem2, &hcclBuf2);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    // 注册交集
    HcclBuf hcclBuf3{};
    HcclMem localMem3 = {HCCL_MEM_TYPE_DEVICE, localbuf, bufSize + 1};
    ret = HcclMemReg(devCtx, &localMem3, &hcclBuf3);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    // 2次注册，第1次解注册
    ret = HcclMemDereg(&hcclBuf1);
    EXPECT_EQ(ret, HCCL_E_AGAIN);

    // 2次注册，第2次解注册
    ret = HcclMemDereg(&hcclBuf);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclMemReg(nullptr, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcclMemReg(devCtx, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcclMemReg(devCtx, &localMem2, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);

    HcclMem localMemE = {HCCL_MEM_TYPE_NUM, localbuf, bufSize};
    ret = HcclMemReg(devCtx, &localMemE, &hcclBuf);
    EXPECT_EQ(ret, HCCL_E_PARA);
    localMemE = {HCCL_MEM_TYPE_DEVICE, nullptr, bufSize};
    ret = HcclMemReg(devCtx, &localMemE, &hcclBuf);
    EXPECT_EQ(ret, HCCL_E_PARA);
    localMemE = {HCCL_MEM_TYPE_DEVICE, localbuf, 0U};
    ret = HcclMemReg(devCtx, &localMemE, &hcclBuf);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcclMemDereg(nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);

    hcclBuf3.addr = localbuf;
    hcclBuf3.len = 0;
    ret = HcclMemDereg(&hcclBuf3);
    EXPECT_EQ(ret, HCCL_E_PARA);

    hcclBuf3.addr = nullptr;
    hcclBuf3.len = 1;
    ret = HcclMemDereg(&hcclBuf3);
    EXPECT_EQ(ret, HCCL_E_PARA);

    sal_free(localbuf);
}
#endif
TEST_F(OneSidedSt, ut_one_sided_service_batchput_batchget_err)
{
    typedef HcclResult (*HcclOneSideServiceCallBack)(
        std::unique_ptr<hccl::IHcclOneSidedService>&, std::unique_ptr<hccl::HcclSocketManager>&,
        std::unique_ptr<hccl::NotifyPool>&);
    nlohmann::json rank_table = rank_table_910_1server_4rank;
    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();

    int ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    const char* rank_table_file = "./ut_opbase_test.json";
    HcclComm comm;
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);

    IHcclOneSidedService* iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    EXPECT_NE(iService, nullptr);
    iService->netDevIpcCtx_ = nullptr;
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);

    RankId remoteRankId = 10; // invalid value
    HcclOneSideOpDesc desc;
    u32 descNum;
    rtStream_t stream;
    try {
        service->BatchPut(remoteRankId, &desc, descNum, stream);
    } catch (const std::out_of_range& e) {
    }
    try {
        service->BatchGet(remoteRankId, &desc, descNum, stream);
    } catch (const std::out_of_range& e) {
    }

    ret = HcclCommDestroy(comm);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    GlobalMockObject::verify();
}
#if 0 // 路径失效launch_aicpu168
TEST_F(OneSidedSt, ut_one_sided_service_batchput_aicpu_rdma)
{
    nlohmann::json rank_table = rank_table_910_1server_4rank;
    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);
    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();

    HcclResult ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* localbuf;
    s8* remotebuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = 1024;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = rtStreamCreate(&stream, 0);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    localbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    remotebuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(remotebuf, count * sizeof(s8), 0, count * sizeof(s8));

    void *comm;
    const char *rankTableFile = "./ut_opbase_test.json";
    ret = HcclCommInitClusterInfo(rankTableFile, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(GetExternalInputHcclAicpuUnfold).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER(GetExternalInputIntraRoceSwitch).stubs().will(returnValue(1));

    const DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaSendWrV2).stubs().will(returnValue(HCCL_SUCCESS));

    for (int j = 0; j < count; j++) {
        localbuf[j] = 2;
    }
    u32 itemNum = 1;
    HcclOneSideOpDesc desc[itemNum];
    desc[0].count = 1024;
    desc[0].dataType = HCCL_DATA_TYPE_INT8;
    desc[0].localAddr = localbuf;
    desc[0].remoteAddr = remotebuf;

    NetDevContext devContext;
    devContext.nicType_ = NicType::DEVICE_NIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &devContext;

    const u32 remoteRankId = 1;
    HcclRankLinkInfo remoteLinkInfo {};
    remoteLinkInfo.userRank = remoteRankId;

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    IHcclOneSidedService *iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    EXPECT_NE(iService, nullptr);
    std::string commIdentifier = hcclComm->GetIdentifier();
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);
    service->netDevRdmaCtx_ = devCtx;
    service->isUsedRdmaMap_[remoteRankId] = true;
    ret = service->CreateConnection(remoteRankId, remoteLinkInfo, service->oneSidedConns_[remoteRankId]);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(service->aicpuUnfoldMode_);
    std::shared_ptr<hccl::HcclOneSidedConn> connPtr = service->oneSidedConns_[remoteRankId];
    EXPECT_NE(connPtr, nullptr);

    TransportRoceMem *transport = dynamic_cast<TransportRoceMem *>(connPtr->transportMemPtr_.get());
    BufferKey<uintptr_t, u64> tempLocalKey(reinterpret_cast<uintptr_t>(localbuf), count * sizeof(s8));
    auto tempLocalBufferPtr = make_shared<LocalRdmaRmaBuffer>(devCtx, localbuf, count * sizeof(s8));
    tempLocalBufferPtr->devAddr = localbuf;
    devContext.localRdmaRmaBufferMgr_->Add(tempLocalKey, tempLocalBufferPtr);

    BufferKey<uintptr_t, u64> tempRemoteKey(reinterpret_cast<uintptr_t>(remotebuf), count * sizeof(s8));
    RemoteRdmaRmaBuffer tempRemoteBufferPtr{};
    tempRemoteBufferPtr.addr = remotebuf;
    tempRemoteBufferPtr.size = count * sizeof(s8);
    tempRemoteBufferPtr.devAddr = remotebuf;
    connPtr->remoteRmaBufferMgr_.Add(tempRemoteKey, reinterpret_cast<void *>(&tempRemoteBufferPtr));

    HcclIpAddress ipAddr;
    auto socketPtr = make_shared<HcclSocket>("tag", devCtx, ipAddr, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT);
    std::vector<std::shared_ptr<HcclSocket>> connectSockets;
    connectSockets.push_back(socketPtr);

    MOCKER_CPP(&HcclSocketManager::CreateSingleLinkSocket).stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(connectSockets), mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&DispatcherPub::RdmaSend,
        HcclResult(DispatcherPub::*)(u32, u64, const struct send_wr&, HcclRtStream, hccl::RdmaType, u64, u64, bool))
        .stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&LocalNotify::GetNotifyData).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP_VIRTUAL(*transport, &TransportRoceMem::ExchangeMemDesc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(*transport, &TransportRoceMem::Connect).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(*transport, &TransportRoceMem::SetSocket).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(*transport, &TransportRoceMem::WaitOpFence).stubs().will(returnValue(HCCL_SUCCESS));

    ret = connPtr->Connect(commIdentifier, 10);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclBatchPut(comm, 1, desc, itemNum, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = hcclStreamSynchronize(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(localbuf);
    sal_free(remotebuf);

    rt_ret = rtStreamDestroy(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name_t);
    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, ut_one_sided_service_batchput_batchget_aicpu_rdma)
{
    nlohmann::json rank_table = rank_table_910_1server_4rank;
    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);
    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();

    HcclResult ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* localbuf;
    s8* remotebuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = 1024;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = rtStreamCreate(&stream, 0);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    localbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    remotebuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(remotebuf, count * sizeof(s8), 0, count * sizeof(s8));

    void *comm;
    const char *rankTableFile = "./ut_opbase_test.json";
    ret = HcclCommInitClusterInfo(rankTableFile, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(GetExternalInputHcclAicpuUnfold).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER(GetExternalInputIntraRoceSwitch).stubs().will(returnValue(1));

    const DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaSendWrV2).stubs().will(returnValue(HCCL_SUCCESS));

    for (int j = 0; j < count; j++) {
        localbuf[j] = 2;
    }
    u32 itemNum = 1;
    HcclOneSideOpDesc desc[itemNum];
    desc[0].count = 1024;
    desc[0].dataType = HCCL_DATA_TYPE_INT8;
    desc[0].localAddr = localbuf;
    desc[0].remoteAddr = remotebuf;

    NetDevContext devContext;
    devContext.nicType_ = NicType::DEVICE_NIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &devContext;

    const u32 remoteRankId = 1;
    HcclRankLinkInfo remoteLinkInfo {};
    remoteLinkInfo.userRank = remoteRankId;

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    IHcclOneSidedService *iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    EXPECT_NE(iService, nullptr);
    std::string commIdentifier = hcclComm->GetIdentifier();
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);
    service->netDevRdmaCtx_ = devCtx;
    service->isUsedRdmaMap_[remoteRankId] = true;
    ret = service->CreateConnection(remoteRankId, remoteLinkInfo, service->oneSidedConns_[remoteRankId]);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(service->aicpuUnfoldMode_);
    std::shared_ptr<hccl::HcclOneSidedConn> connPtr = service->oneSidedConns_[remoteRankId];
    EXPECT_NE(connPtr, nullptr);

    TransportRoceMem *transport = dynamic_cast<TransportRoceMem *>(connPtr->transportMemPtr_.get());
    BufferKey<uintptr_t, u64> tempLocalKey(reinterpret_cast<uintptr_t>(localbuf), count * sizeof(s8));
    auto tempLocalBufferPtr = make_shared<LocalRdmaRmaBuffer>(devCtx, localbuf, count * sizeof(s8));
    tempLocalBufferPtr->devAddr = localbuf;
    devContext.localRdmaRmaBufferMgr_->Add(tempLocalKey, tempLocalBufferPtr);

    BufferKey<uintptr_t, u64> tempRemoteKey(reinterpret_cast<uintptr_t>(remotebuf), count * sizeof(s8));
    RemoteRdmaRmaBuffer tempRemoteBufferPtr{};
    tempRemoteBufferPtr.addr = remotebuf;
    tempRemoteBufferPtr.size = count * sizeof(s8);
    tempRemoteBufferPtr.devAddr = remotebuf;
    connPtr->remoteRmaBufferMgr_.Add(tempRemoteKey, reinterpret_cast<void *>(&tempRemoteBufferPtr));

    HcclIpAddress ipAddr;
    auto socketPtr = make_shared<HcclSocket>("tag", devCtx, ipAddr, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT);
    std::vector<std::shared_ptr<HcclSocket>> connectSockets;
    connectSockets.push_back(socketPtr);

    MOCKER_CPP(&HcclSocketManager::CreateSingleLinkSocket).stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(connectSockets), mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&DispatcherPub::RdmaSend,
        HcclResult(DispatcherPub::*)(u32, u64, const struct send_wr&, HcclRtStream, hccl::RdmaType, u64, u64, bool))
        .stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&LocalNotify::GetNotifyData).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP_VIRTUAL(*transport, &TransportRoceMem::ExchangeMemDesc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(*transport, &TransportRoceMem::Connect).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(*transport, &TransportRoceMem::SetSocket).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(*transport, &TransportRoceMem::WaitOpFence).stubs().will(returnValue(HCCL_SUCCESS));

    ret = connPtr->Connect(commIdentifier, 10);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclBatchPut(comm, 1, desc, itemNum, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclBatchGet(comm, 1, desc, itemNum, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = hcclStreamSynchronize(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(localbuf);
    sal_free(remotebuf);

    rt_ret = rtStreamDestroy(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name_t);
    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, ut_one_sided_service_bind_mem)
{
    typedef HcclResult (*HcclOneSideServiceCallBack)(std::unique_ptr<hccl::IHcclOneSidedService> &,
    std::unique_ptr<hccl::HcclSocketManager> &, std::unique_ptr<hccl::NotifyPool> &);

    nlohmann::json rank_table = rank_table_910_1server_2rank;
    std::string clusterString = rank_table.dump();

    int ret = HCCL_SUCCESS;
    void* comm;
    u32 rank_ID = 0;

    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    commConfig.hcclBufferSize=800;
    strcpy_s(commConfig.hcclCommName, COMM_NAME_MAX_LENGTH, "comm1");

    unsetenv("HCCL_INTRA_PCIE_ENABLE");
    setenv("HCCL_INTRA_ROCE_ENABLE", "1", 1);
    ret = HcclCommInitClusterInfoMemConfig(const_cast<char*>(clusterString.c_str()), rank_ID, &commConfig, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);

    IHcclOneSidedService *iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    EXPECT_NE(iService, nullptr);
    iService->netDevIpcCtx_ = nullptr;
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);

    // 注册全局内存
    GlobalMemRegMgr mgr;
    auto buffer1 = std::vector<int8_t>(10);
    HcclMem mem1{HCCL_MEM_TYPE_DEVICE, buffer1.data(), buffer1.size()};
    void* memHandle1 = nullptr;
    ret = mgr.Reg(&mem1, &memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    auto buffer2 = std::vector<int8_t>(10);
    HcclMem mem2{HCCL_MEM_TYPE_DEVICE, buffer2.data(), buffer2.size()};
    void* memHandle2 = nullptr;
    ret = mgr.Reg(&mem2, &memHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // Service绑定一块内存
    std::string commIdentifier = hcclComm->GetIdentifier();
    ret = service->BindMem(memHandle1, commIdentifier);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 不能重复绑定同一块内存
    ret = service->BindMem(memHandle1, commIdentifier);
    EXPECT_EQ(ret, HCCL_E_PARA);

    // 不能解绑未绑定的内存
    ret = service->UnbindMem(memHandle2, commIdentifier);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = service->BindMem(memHandle2, commIdentifier);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = service->UnbindMem(memHandle2, commIdentifier);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 未解绑所有内存，不能destroy通信域
    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = service->UnbindMem(memHandle1, commIdentifier);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, ut_one_sided_globally_bind_mem)
{
    typedef HcclResult (*HcclOneSideServiceCallBack)(std::unique_ptr<hccl::IHcclOneSidedService> &,
    std::unique_ptr<hccl::HcclSocketManager> &, std::unique_ptr<hccl::NotifyPool> &);

    nlohmann::json rank_table = rank_table_910_1server_2rank;
    std::string clusterString = rank_table.dump();

    int ret = HCCL_SUCCESS;
    void* comm;
    u32 rank_ID = 0;

    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    commConfig.hcclBufferSize=800;
    strcpy_s(commConfig.hcclCommName, COMM_NAME_MAX_LENGTH, "comm1");

    unsetenv("HCCL_INTRA_PCIE_ENABLE");
    setenv("HCCL_INTRA_ROCE_ENABLE", "1", 1);
    ret = HcclCommInitClusterInfoMemConfig(const_cast<char*>(clusterString.c_str()), rank_ID, &commConfig, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);

    IHcclOneSidedService *iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    EXPECT_NE(iService, nullptr);
    iService->netDevIpcCtx_ = nullptr;

    // 注册全局内存
    GlobalMemRegMgr mgr;
    auto buffer1 = std::vector<int8_t>(10);
    CommMem mem1{COMM_MEM_TYPE_DEVICE, buffer1.data(), buffer1.size()};
    void* memHandle1 = nullptr;

    // 异常入参
    ret = HcclRegisterGlobalMem(nullptr, &memHandle1);
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclRegisterGlobalMem(&mem1, &memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    auto buffer2 = std::vector<int8_t>(10);
    CommMem mem2{COMM_MEM_TYPE_DEVICE, buffer2.data(), buffer2.size()};
    void* memHandle2 = nullptr;
    ret = HcclRegisterGlobalMem(&mem2, &memHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // Service绑定一块内存
    // 异常入参
    ret = HcclCommBindMem(nullptr, memHandle1);
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = HcclCommBindMem(comm, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcclCommBindMem(comm, memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 不能重复绑定同一块内存
    ret = HcclCommBindMem(comm, memHandle1);
    EXPECT_EQ(ret, HCCL_E_PARA);

    // 异常入参
    ret = HcclCommUnbindMem(nullptr, memHandle1);
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = HcclCommUnbindMem(comm, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);

    // 不能解绑未绑定的内存
    ret = HcclCommUnbindMem(comm, memHandle2);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcclCommBindMem(comm, memHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 绑定着通信域的内存不能直接注销
    ret = HcclDeregisterGlobalMem(memHandle1);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcclCommUnbindMem(comm, memHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 未解绑所有内存，不能destroy通信域
    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcclCommUnbindMem(comm, memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 异常入参
    ret = HcclDeregisterGlobalMem(nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcclDeregisterGlobalMem(memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclDeregisterGlobalMem(memHandle2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 没注册就析构报Not found
    GlobalMemRecord record(mem1);
    ret = HcclDeregisterGlobalMem(&record);
    EXPECT_EQ(ret, HCCL_E_PARA);
    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, ut_one_sided_service_mem_test_prepare)
{
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    void* comm;
    const char* rank_table_file = "./ut_opbase_test.json";

    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclMemDesc localDesc, remoteDesc;
    char str[21] = "aaaabbbbccccddddeeee";
    memcpy_s(localDesc.desc, sizeof(localDesc.desc), str, sizeof(str));
    memcpy_s(remoteDesc.desc, sizeof(remoteDesc.desc), str, sizeof(str));
    HcclMemDescs local;
    local.arrayLength = 1;
    local.array = &localDesc;
    HcclMemDescs remote;
    remote.arrayLength = 1;
    remote.array = &remoteDesc;
    u32 actualNum;
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    IHcclOneSidedService *iService = nullptr;
    hcclComm->GetOneSidedService(&iService);

    NetDevContext devContext;
    devContext.nicType_ = NicType::VNIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &devContext;
    iService->netDevIpcCtx_ = &devContext;
    EXPECT_NE(iService, nullptr);
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);
    service->isUsedRdmaMap_.insert({1, false});

    MOCKER(hrtRaGetSingleSocketVnicIpInfo)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    for (u32 i = 1; i < 4; i++) {
        HcclRankLinkInfo localLinkInfo {};
        HcclRankLinkInfo remoteLinkInfo {};
        remoteLinkInfo.userRank = i;
        std::unique_ptr<HcclSocketManager> socketManager = nullptr;
        socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
        std::unique_ptr<NotifyPool> notifyPool;
        HcclDispatcher dispatcher;
        std::shared_ptr<HcclOneSidedConn> connPtr = make_shared<HcclOneSidedConn>(devCtx, localLinkInfo,
            remoteLinkInfo, socketManager, notifyPool, dispatcher, false, 0U, 0U);
        service->oneSidedConns_.insert({i, connPtr});
        HcclIpAddress ipAddr;
        std::shared_ptr<HcclSocket> socketPtr1 = make_shared<HcclSocket>("tag", devCtx, ipAddr, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT);
        connPtr->socket_ = socketPtr1;
        connPtr->transportMemPtr_->SetDataSocket(socketPtr1);
    }

    MOCKER_CPP(&HcclSocket::Send, HcclResult(HcclSocket::*)(const void *, u64))
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclSocket::Recv, HcclResult(HcclSocket::*)(void *, u32))
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    HcclPrepareConfig config;
    config.topoType = HcclTopoType::HCCL_TOPO_FULLMESH;

    MOCKER_CPP(&HcclOneSidedService::CreateLinkFullmesh)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclOneSidedService::RegisterBoundMems)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclOneSidedService::ExchangeMemDescFullMesh)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclOneSidedService::EnableMemAccess, HcclResult(HcclOneSidedService::*)())
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    ret = HcclCommPrepare(comm, &config, 120);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = GlobalMemRegMgr::GetInstance().Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name_t);
    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, ut_one_sided_service_mem_test_RegisterBoundMems)
{
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    void* comm;
    const char* rank_table_file = "./ut_opbase_test.json";

    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclMemDesc localDesc, remoteDesc;
    char str[21] = "aaaabbbbccccddddeeee";
    memcpy_s(localDesc.desc, sizeof(localDesc.desc), str, sizeof(str));
    memcpy_s(remoteDesc.desc, sizeof(remoteDesc.desc), str, sizeof(str));
    HcclMemDescs local;
    local.arrayLength = 1;
    local.array = &localDesc;
    HcclMemDescs remote;
    remote.arrayLength = 1;
    remote.array = &remoteDesc;
    u32 actualNum;
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    IHcclOneSidedService *iService = nullptr;
    hcclComm->GetOneSidedService(&iService);

    NetDevContext devContext;
    devContext.nicType_ = NicType::VNIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();

    NetDevContext devContextRdma;
    devContextRdma.nicType_ = NicType::DEVICE_NIC_TYPE;
    devContextRdma.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContextRdma.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtxIpc = &devContext;
    HcclNetDevCtx devCtxRdma = &devContextRdma;
    iService->netDevIpcCtx_ = &devContext;
    iService->netDevRdmaCtx_ = &devContextRdma;
    EXPECT_NE(iService, nullptr);
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);
    service->isUsedRdmaMap_.insert({1, false});

    for (u32 i = 1; i < 4; i++) {
        HcclRankLinkInfo localLinkInfo {};
        HcclRankLinkInfo remoteLinkInfo {};
        remoteLinkInfo.userRank = i;
        std::unique_ptr<HcclSocketManager> socketManager = nullptr;
        socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
        std::unique_ptr<NotifyPool> notifyPool;
        HcclDispatcher dispatcher;
        std::shared_ptr<HcclOneSidedConn> connPtr = make_shared<HcclOneSidedConn>(devCtxIpc, localLinkInfo,
            remoteLinkInfo, socketManager, notifyPool, dispatcher, false, 0U, 0U);
        service->oneSidedConns_.insert({i, connPtr});
        HcclIpAddress ipAddr;
        std::shared_ptr<HcclSocket> socketPtr1 = make_shared<HcclSocket>("tag", devCtxIpc, ipAddr, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT);
        connPtr->socket_ = socketPtr1;
        connPtr->transportMemPtr_->SetDataSocket(socketPtr1);
    }

    MOCKER(hrtRaGetSingleSocketVnicIpInfo)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    void* localbuf = (void*)0x11;
    u64 count = 1024;

    BufferKey<uintptr_t, u64> tempLocalKey(reinterpret_cast<uintptr_t>(localbuf), count * sizeof(s8));
    std::shared_ptr<LocalIpcRmaBuffer> tempLocalBufferPtr = make_shared<LocalIpcRmaBuffer>(devCtxIpc, localbuf, count * sizeof(s8));
    tempLocalBufferPtr->devAddr = localbuf;
    devContext.localIpcRmaBufferMgr_->Add(tempLocalKey, tempLocalBufferPtr);


    std::shared_ptr<LocalRdmaRmaBuffer> tempLocalBufferPtr2 = make_shared<LocalRdmaRmaBuffer>(devCtxRdma, localbuf, count * sizeof(s8));
    tempLocalBufferPtr2->devAddr = localbuf;
    devContextRdma.localRdmaRmaBufferMgr_->Add(tempLocalKey, tempLocalBufferPtr2);

    HcclMem mem1;
    mem1.addr = (void*)0x11;
    mem1.size = 1024;
    mem1.type = HCCL_MEM_TYPE_HOST;

    std::shared_ptr<GlobalMemRecord> ptr = make_shared<GlobalMemRecord>(mem1);
    service->boundMemPtrSet_.insert(ptr.get());
    service->needRegIpcMem_ = true;
    service->needRegRoceMem_ = true;
    ret = service->RegisterBoundMems();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name_t);
    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, ut_one_sided_service_mem_test_Exchange_and_enable_Mem)
{
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();


    int ret = HCCL_SUCCESS;
    void* comm;
    const char* rank_table_file = "./ut_opbase_test.json";

    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclMemDesc localDesc, remoteDesc;
    char str[21] = "aaaabbbbccccddddeeee";
    memcpy_s(localDesc.desc, sizeof(localDesc.desc), str, sizeof(str));
    memcpy_s(remoteDesc.desc, sizeof(remoteDesc.desc), str, sizeof(str));
    HcclMemDescs local;
    local.arrayLength = 1;
    local.array = &localDesc;
    HcclMemDescs remote;
    remote.arrayLength = 1;
    remote.array = &remoteDesc;
    u32 actualNum;
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    IHcclOneSidedService *iService = nullptr;
    hcclComm->GetOneSidedService(&iService);

    NetDevContext devContext;
    devContext.nicType_ = NicType::VNIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();

    NetDevContext devContextRdma;
    devContextRdma.nicType_ = NicType::DEVICE_NIC_TYPE;
    devContextRdma.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContextRdma.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtxIpc = &devContext;
    HcclNetDevCtx devCtxRdma = &devContextRdma;
    iService->netDevIpcCtx_ = &devContext;
    iService->netDevRdmaCtx_ = &devContextRdma;
    EXPECT_NE(iService, nullptr);
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);
    service->isUsedRdmaMap_.insert({1, false});


    void* localbuf = (void*)0x11;

    void* remotebuf = (void*)0x22;
    u64 count = 1024;

    u32 itemNum = 1;
    HcclOneSideOpDesc desc[itemNum];
    desc[0].count = 1024;
    desc[0].dataType = HCCL_DATA_TYPE_INT8;
    desc[0].localAddr = localbuf;
    desc[0].remoteAddr = remotebuf;

    for (u32 i = 1; i < 4; i++) {
        HcclRankLinkInfo localLinkInfo {};
        HcclRankLinkInfo remoteLinkInfo {};
        remoteLinkInfo.userRank = i;
        std::unique_ptr<HcclSocketManager> socketManager = nullptr;
        socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
        std::unique_ptr<NotifyPool> notifyPool;
        HcclDispatcher dispatcher;
        std::shared_ptr<HcclOneSidedConn> connPtr = make_shared<HcclOneSidedConn>(devCtxIpc, localLinkInfo,
            remoteLinkInfo, socketManager, notifyPool, dispatcher, false, 0U, 0U);
        service->oneSidedConns_.insert({i, connPtr});
        HcclIpAddress ipAddr;
        std::shared_ptr<HcclSocket> socketPtr1 = make_shared<HcclSocket>("tag", devCtxIpc, ipAddr, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT);
        connPtr->socket_ = socketPtr1;
        connPtr->transportMemPtr_->SetDataSocket(socketPtr1);
        connPtr->actualNumOfRemote_ = 1;

        TransportIpcMem *transport = dynamic_cast<TransportIpcMem *>(connPtr->transportMemPtr_.get());
        BufferKey<uintptr_t, u64> tempLocalKey(reinterpret_cast<uintptr_t>(localbuf), count * sizeof(s8));
        std::shared_ptr<LocalIpcRmaBuffer> tempLocalBufferPtr = make_shared<LocalIpcRmaBuffer>(devCtxIpc, localbuf, count * sizeof(s8));
        tempLocalBufferPtr->devAddr = localbuf;
        devContext.localIpcRmaBufferMgr_->Add(tempLocalKey, tempLocalBufferPtr);

        std::shared_ptr<LocalRdmaRmaBuffer> tempLocalBufferPtr2 = make_shared<LocalRdmaRmaBuffer>(devCtxRdma, localbuf, count * sizeof(s8));
        tempLocalBufferPtr2->devAddr = localbuf;
        devContextRdma.localRdmaRmaBufferMgr_->Add(tempLocalKey, tempLocalBufferPtr2);

        BufferKey<uintptr_t, u64> tempRemoteKey(reinterpret_cast<uintptr_t>(remotebuf), count * sizeof(s8));
        std::shared_ptr<RemoteIpcRmaBuffer> tempRemoteBufferPtr = make_shared<RemoteIpcRmaBuffer>(devCtxIpc);
        tempRemoteBufferPtr->addr = remotebuf;
        tempRemoteBufferPtr->size = count * sizeof(s8);
        tempRemoteBufferPtr->devAddr = remotebuf;
        tempRemoteBufferPtr->memType =  RmaMemType::DEVICE;
        transport->remoteIpcRmaBufferMgr_.Add(tempRemoteKey, tempRemoteBufferPtr);
    }

    MOCKER(hrtRaGetSingleSocketVnicIpInfo)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    HcclMem mem1;
    mem1.addr = (void*)0x11;
    mem1.size = 1024;
    mem1.type = HCCL_MEM_TYPE_HOST;

    std::shared_ptr<GlobalMemRecord> ptr = make_shared<GlobalMemRecord>(mem1);
    service->boundMemPtrSet_.insert(ptr.get());
    service->needRegIpcMem_ = true;
    service->needRegRoceMem_ = true;

    MOCKER_CPP(&HcclOneSidedService::ConnectByThread)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclSocket::Send, HcclResult(HcclSocket::*)(const void *, u64))
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclSocket::Recv, HcclResult(HcclSocket::*)(void *, u32))
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclOneSidedConn::Connect)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclOneSidedConn::ExchangeIpcProcessInfo)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportMem::DoExchangeMemDesc)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclOneSidedConn::EnableMemAccess, HcclResult(HcclOneSidedConn::*)())
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclOneSidedConn::DisableMemAccess, HcclResult(HcclOneSidedConn::*)())
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    ret = service->RegisterBoundMems();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = service->ExchangeMemDescFullMesh();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = service->EnableMemAccess();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = service->DisableMemAccess();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name_t);
    GlobalMockObject::verify();
}
#endif
TEST_F(OneSidedSt, regmem_without_memExport)
{
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    int ret = HCCL_SUCCESS;
    void* comm;
    const char* rank_table_file = "./ut_opbase_test.json";

    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    NetDevContext devContext;
    devContext.nicType_ = NicType::DEVICE_NIC_TYPE;
    devContext.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    devContext.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &devContext;

    u64 bufSize = 1024 * sizeof(s8);
    s8* localbuf = (s8*)sal_malloc(bufSize);
    sal_memset(localbuf, bufSize, 0, bufSize);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    IHcclOneSidedService* iService = nullptr;
    hcclComm->GetOneSidedService(&iService);
    iService->netDevRdmaCtx_ = devCtx;
    EXPECT_NE(iService, nullptr);
    HcclOneSidedService* service = dynamic_cast<HcclOneSidedService*>(iService);

    constexpr u64 ONE_SIDE_DEVICE_MEM_SIZE_1 = 1024 * 1024;
    u32 regCntMax = 256;
    s8* subBuffers[regCntMax];
    HcclMemDesc subRegMemDesc[regCntMax];

    MOCKER(HcclMemReg).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(HcclMemExport).stubs().will(returnValue(HCCL_E_UNAVAIL));
    HcclMemDesc localMemDesc1;
    try {
        ret = service->RegMem(localbuf, bufSize, HCCL_MEM_TYPE_DEVICE, 1, localMemDesc1);
    } catch (...) {
        HCCL_ERROR("[HcclOneSidedService][RegMem] get mem desc failed");
    }

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(file_name_t);
    sal_free(localbuf);
}

HcclResult hrtGetPairDevicePhyIdForTest(u32 localDevPhyId, u32& pairDevPhyId)
{
    pairDevPhyId = 1;
    return HCCL_SUCCESS;
}

HcclResult hrtGetDeviceTypeForTest(DevType& devType)
{
    devType = DevType::DEV_TYPE_910_93;
    return HCCL_SUCCESS;
}

HcclResult hrtGetDeviceIndexByPhyIdForTest(u32 devicePhyId, u32& deviceLogicId)
{
    deviceLogicId = 1;
    return HCCL_SUCCESS;
}

HcclResult hrtRaGetDeviceAllNicIPForTest(std::vector<std::vector<HcclIpAddress>>& ipAddr)
{
    ipAddr.clear();
    HcclIpAddress testIp1{"10.10.10.11"};
    std::vector<HcclIpAddress> vec1;
    vec1.push_back(testIp1);
    HcclIpAddress testIp2{"10.10.10.12"};
    std::vector<HcclIpAddress> vec2;
    vec2.push_back(testIp2);
    ipAddr.push_back(vec1);
    ipAddr.push_back(vec2);
    GTEST_LOG_(INFO) << "lyy ipAddr.size: " << ipAddr.size();

    return HCCL_SUCCESS;
}

HcclResult hrtRaGetDeviceIPForTest(u32 devicePhyId, std::vector<hccl::HcclIpAddress>& ipAddr)
{
    ipAddr.clear();
    hccl::HcclIpAddress testIp1{"10.10.10.11"};
    ipAddr.push_back(testIp1);
    return HCCL_SUCCESS;
}

TEST_F(OneSidedSt, ut_hcclComm_InitNic_IsOneSidedBackupInit)
{
    setenv("HCCL_INTRA_ROCE_ENABLE", "1", 1);
    MOCKER(hrtGetPairDevicePhyId).stubs().will(invoke(hrtGetPairDevicePhyIdForTest));
    MOCKER(hrtGetDeviceIndexByPhyId).stubs().will(invoke(hrtGetDeviceIndexByPhyIdForTest));
    MOCKER(hrtGetDeviceType).stubs().will(invoke(hrtGetDeviceTypeForTest));
    MOCKER(hrtRaGetDeviceIP).stubs().will(invoke(hrtRaGetDeviceIPForTest));
    MOCKER(hrtRaGetDeviceAllNicIP).stubs().will(invoke(hrtRaGetDeviceAllNicIPForTest));
    MOCKER_CPP(&HcclCommunicator::IsEnableBackupLink).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&HcclSocketManager::ServerInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclNetInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclNetOpenDev).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::ServerDeInit, HcclResult(HcclSocketManager::*)(const HcclNetDevCtx, u32))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicatorAttrs::SetNeedInitNicFlag).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(Is310PDevice).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER_CPP(&HcclCommunicator::ReleasePreemptSocket).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclIpAddress remoteIp{"10.10.10.11"};
    std::shared_ptr<HcclSocket> listenSocket(
        new (std::nothrow) HcclSocket("my tag2", nullptr, remoteIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER));

    HcclNetDevCtx ctx1;
    HcclResult ret = HcclNetOpenDev(&ctx1, NicType::DEVICE_NIC_TYPE, 0, 0, HcclIpAddress("1.1.1.1"));

    HcclIpAddress remoteIp2{"10.10.10.12"};
    std::shared_ptr<HcclSocket> listenSocket2(
        new (std::nothrow) HcclSocket("my tag2", nullptr, remoteIp2, 0, HcclSocketRole::SOCKET_ROLE_SERVER));
    HcclNetDevCtx ctx2;

    HcclCommunicator hcclCommunicator;
    hcclCommunicator.nicDeployment_ = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    hcclCommunicator.devicePhyId_ = 0;
    hcclCommunicator.devIpAddr_.clear();
    hcclCommunicator.devIpAddr_.push_back(remoteIp2);
    hcclCommunicator.devBackupIpAddr_.clear();
    hcclCommunicator.devBackupIpAddr_.push_back(remoteIp2);
    hcclCommunicator.commPortConfig_.devNicListen = std::make_pair(listenSocket, ctx1);
    HcclCommConfig commConfig("hccl_world_group");
    hcclCommunicator.oneSideService_ = std::make_unique<HcclOneSidedService>(
        hcclCommunicator.socketManager_, hcclCommunicator.notifyPool_, commConfig);

    ret = hcclCommunicator.InitNic();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_IF_BASE_PORT", "50000", 1);
    InitExternalInput();
    hcclCommunicator.GetHostPort(0);

    unsetenv("HCCL_INTRA_ROCE_ENABLE");
    InitExternalInput();
    hcclCommunicator.GetHostPort(0);

    hcclCommunicator.nicInitialized_ = 0;
    hcclCommunicator.raResourceInit_ = false;
    ResetInitState();

    GlobalMockObject::verify();
}

TEST_F(OneSidedSt, ut_one_sided_service_prepare_fail)
{
    int ret = HCCL_SUCCESS;

    NetDevContext netDevCtx;
    ;
    netDevCtx.nicType_ = NicType::DEVICE_NIC_TYPE;
    netDevCtx.localRdmaRmaBufferMgr_ = std::make_shared<LocalRdmaRmaBufferMgr>();
    HcclNetDevCtx devCtx = &netDevCtx;

    NetDevContext vNetDevCtx;
    vNetDevCtx.nicType_ = NicType::VNIC_TYPE;
    vNetDevCtx.localIpcRmaBufferMgr_ = std::make_shared<LocalIpcRmaBufferMgr>();
    HcclNetDevCtx vDevCtx = &vNetDevCtx;

    unique_ptr<HcclSocketManager> socketManager
        = std::make_unique<HcclSocketManager>(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0);
    unique_ptr<NotifyPool> notifyPool = std::make_unique<NotifyPool>();
    HcclCommConfig commConfig("hccl_world_group");
    unique_ptr<HcclOneSidedService> service
        = std::make_unique<HcclOneSidedService>(socketManager, notifyPool, commConfig);

    MOCKER_CPP(&HcclOneSidedService::PrepareFullMesh).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_TIMEOUT));

    HcclDispatcher dispatcher = &notifyPool;
    HcclRankLinkInfo localRankInfo{};
    RankTable_t rankTable{};
    RankInfo_t rankInfo;
    rankTable.rankList.push_back(rankInfo);
    map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap{};

    service->Config(dispatcher, localRankInfo, &rankTable);

    std::string commIdentifier("test");
    HcclPrepareConfig config;
    config.topoType = HcclTopoType::HCCL_TOPO_FULLMESH;
    ret = service->Prepare(commIdentifier, &config, 1);

    GlobalMockObject::verify();
}
