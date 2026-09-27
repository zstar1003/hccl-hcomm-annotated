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
#include <stdexcept>
#include <future>
#include <chrono>
#include <vector>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_inner.h>

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

class OneSidedUt : public testing::Test {
protected:
    void SetUp() override
    {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }

    void TearDown() override { std::cout << "A Test TearDown" << std::endl; }
};

TEST_F(OneSidedUt, Prepare_When_PrepareFullMeshFail_With_NullptrConn_Expect_SkipClean)
{
    unique_ptr<HcclSocketManager> socketManager
        = std::make_unique<HcclSocketManager>(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0);
    unique_ptr<NotifyPool> notifyPool = std::make_unique<NotifyPool>();
    CommConfig commConfig;
    unique_ptr<HcclOneSidedService> service
        = std::make_unique<HcclOneSidedService>(socketManager, notifyPool, commConfig);

    MOCKER_CPP(&HcclOneSidedService::CreateLinkFullmesh).stubs().will(returnValue(HCCL_E_INTERNAL));
    MOCKER_CPP(&HcclOneSidedConn::CleanSocketResource).stubs().will(returnValue(HCCL_SUCCESS));

    // 构造ranktable
    HcclDispatcher dispatcher = &notifyPool;
    HcclRankLinkInfo localRankInfo{};
    localRankInfo.userRank = 0;
    RankTable_t rankTable{};
    RankInfo_t rankInfo0;
    rankInfo0.rankId = 0;
    RankInfo_t rankInfo1;
    rankInfo1.rankId = 1;
    rankTable.rankList.push_back(rankInfo0);
    rankTable.rankList.push_back(rankInfo1);
    service->Config(dispatcher, localRankInfo, &rankTable);

    service->oneSidedConns_[1] = nullptr;

    std::string commIdentifier("test");
    HcclPrepareConfig config;
    config.topoType = HcclTopoType::HCCL_TOPO_FULLMESH;
    HcclResult ret = service->Prepare(commIdentifier, &config, 1);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    GlobalMockObject::verify();
}

TEST_F(OneSidedUt, HcclRegisterGlobalMem_ValidMem_Expect_ReinterpretCastCovered)
{
    auto buffer = std::vector<int8_t>(10);
    CommMem mem{COMM_MEM_TYPE_DEVICE, buffer.data(), buffer.size()};
    void* memHandle = nullptr;
    HcclResult ret = HcclRegisterGlobalMem(&mem, &memHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(memHandle, nullptr);

    HcclDeregisterGlobalMem(memHandle);
}

TEST_F(OneSidedUt, HcclRemapRegistedMemory_ValidParams_Expect_ReinterpretCastCovered)
{
    auto commObj = std::make_unique<hccl::hcclComm>();
    void* commArr[1] = {static_cast<void*>(commObj.get())};

    unique_ptr<HcclSocketManager> socketManager
        = std::make_unique<HcclSocketManager>(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0);
    unique_ptr<NotifyPool> notifyPool = std::make_unique<NotifyPool>();
    CommConfig commConfig;
    hccl::HcclOneSidedService service(socketManager, notifyPool, commConfig);

    hccl::IHcclOneSidedService* servicePtr = &service;

    CommMem memInfoArray[1];
    memInfoArray[0].type = COMM_MEM_TYPE_DEVICE;
    memInfoArray[0].addr = reinterpret_cast<void*>(0x1000);
    memInfoArray[0].size = 1024;

    MOCKER_CPP(&hccl::hcclComm::GetOneSidedService)
        .stubs()
        .with(outBoundP(&servicePtr))
        .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HcclRemapRegistedMemory(commArr, memInfoArray, 1, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    commObj.reset();
    GlobalMockObject::verify();
}

TEST_F(OneSidedUt, HcclEnableMemAccess_LegacyPath_Expect_ReinterpretCastCovered)
{
    auto commObj = std::make_unique<hccl::hcclComm>();

    unique_ptr<HcclSocketManager> socketManager
        = std::make_unique<HcclSocketManager>(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0);
    unique_ptr<NotifyPool> notifyPool = std::make_unique<NotifyPool>();
    CommConfig commConfig;
    hccl::HcclOneSidedService service(socketManager, notifyPool, commConfig);

    hccl::IHcclOneSidedService* servicePtr = &service;

    MOCKER_CPP(&hccl::hcclComm::GetOneSidedService)
        .stubs()
        .with(outBoundP(&servicePtr))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclOneSidedService::EnableMemAccess, void(HcclOneSidedService::*)(const HcclMemDesc&, HcclMem&))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    HcclMemDesc remoteMemDesc{};
    CommMem remoteMem{};
    HcclResult ret = HcclEnableMemAccess(static_cast<HcclComm>(commObj.get()), &remoteMemDesc, &remoteMem);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    commObj.reset();
    GlobalMockObject::verify();
}

TEST_F(OneSidedUt, Ut_Prepare_When_PrepareFullMeshFailAndRankMissing_Expect_NoSpuriousEntry)
{
    unique_ptr<HcclSocketManager> socketManager
        = std::make_unique<HcclSocketManager>(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0);
    unique_ptr<NotifyPool> notifyPool = std::make_unique<NotifyPool>();
    CommConfig commConfig;
    unique_ptr<HcclOneSidedService> service
        = std::make_unique<HcclOneSidedService>(socketManager, notifyPool, commConfig);

    MOCKER_CPP(&HcclOneSidedService::PrepareFullMesh).stubs().will(returnValue(HCCL_E_INTERNAL));
    MOCKER_CPP(&HcclOneSidedConn::CleanSocketResource).stubs().will(returnValue(HCCL_SUCCESS));

    HcclDispatcher dispatcher = &notifyPool;
    HcclRankLinkInfo localRankInfo{};
    localRankInfo.userRank = 1;
    RankTable_t rankTable{};
    RankInfo_t rankInfo0;
    rankInfo0.rankId = 0;
    RankInfo_t rankInfo1;
    rankInfo1.rankId = 1;
    RankInfo_t rankInfo2;
    rankInfo2.rankId = 2;
    rankTable.rankList.push_back(rankInfo0);
    rankTable.rankList.push_back(rankInfo1);
    rankTable.rankList.push_back(rankInfo2);
    service->Config(dispatcher, localRankInfo, &rankTable);

    service->oneSidedConns_[2] = nullptr;

    std::string commIdentifier("test");
    HcclPrepareConfig config;
    config.topoType = HcclTopoType::HCCL_TOPO_FULLMESH;
    HcclResult ret = service->Prepare(commIdentifier, &config, 1);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_EQ(service->oneSidedConns_.count(0), 0u);
    EXPECT_EQ(service->oneSidedConns_.size(), 1u);

    GlobalMockObject::verify();
}

TEST_F(OneSidedUt, Ut_Prepare_When_AlreadyPrepared_Expect_ReturnSuccessImmediately)
{
    unique_ptr<HcclSocketManager> socketManager
        = std::make_unique<HcclSocketManager>(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0);
    unique_ptr<NotifyPool> notifyPool = std::make_unique<NotifyPool>();
    CommConfig commConfig;
    unique_ptr<HcclOneSidedService> service
        = std::make_unique<HcclOneSidedService>(socketManager, notifyPool, commConfig);

    MOCKER_CPP(&HcclOneSidedService::PrepareFullMesh).stubs().will(returnValue(HCCL_E_INTERNAL));

    HcclDispatcher dispatcher = &notifyPool;
    HcclRankLinkInfo localRankInfo{};
    localRankInfo.userRank = 0;
    RankTable_t rankTable{};
    RankInfo_t rankInfo0;
    rankInfo0.rankId = 0;
    rankTable.rankList.push_back(rankInfo0);
    service->Config(dispatcher, localRankInfo, &rankTable);

    service->prepared_ = true;

    std::string commIdentifier("test");
    HcclPrepareConfig config;
    config.topoType = HcclTopoType::HCCL_TOPO_FULLMESH;
    HcclResult ret = service->Prepare(commIdentifier, &config, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(OneSidedUt, Ut_Prepare_When_TopoNotFullMesh_Expect_SkipFullMeshAndSuccess)
{
    unique_ptr<HcclSocketManager> socketManager
        = std::make_unique<HcclSocketManager>(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0);
    unique_ptr<NotifyPool> notifyPool = std::make_unique<NotifyPool>();
    CommConfig commConfig;
    unique_ptr<HcclOneSidedService> service
        = std::make_unique<HcclOneSidedService>(socketManager, notifyPool, commConfig);

    MOCKER_CPP(&HcclOneSidedService::PrepareFullMesh).stubs().will(returnValue(HCCL_E_INTERNAL));

    HcclDispatcher dispatcher = &notifyPool;
    HcclRankLinkInfo localRankInfo{};
    localRankInfo.userRank = 0;
    RankTable_t rankTable{};
    RankInfo_t rankInfo0;
    rankInfo0.rankId = 0;
    rankTable.rankList.push_back(rankInfo0);
    service->Config(dispatcher, localRankInfo, &rankTable);

    std::string commIdentifier("test");
    HcclPrepareConfig config;
    config.topoType = HcclTopoType::HCCL_TOPO_NUM;
    HcclResult ret = service->Prepare(commIdentifier, &config, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(service->prepared_, true);

    GlobalMockObject::verify();
}

TEST_F(OneSidedUt, Ut_Prepare_When_PrepareFullMeshSuccess_Expect_PreparedTrue)
{
    unique_ptr<HcclSocketManager> socketManager
        = std::make_unique<HcclSocketManager>(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0);
    unique_ptr<NotifyPool> notifyPool = std::make_unique<NotifyPool>();
    CommConfig commConfig;
    unique_ptr<HcclOneSidedService> service
        = std::make_unique<HcclOneSidedService>(socketManager, notifyPool, commConfig);

    MOCKER_CPP(&HcclOneSidedService::PrepareFullMesh).stubs().will(returnValue(HCCL_SUCCESS));

    HcclDispatcher dispatcher = &notifyPool;
    HcclRankLinkInfo localRankInfo{};
    localRankInfo.userRank = 0;
    RankTable_t rankTable{};
    RankInfo_t rankInfo0;
    rankInfo0.rankId = 0;
    rankTable.rankList.push_back(rankInfo0);
    service->Config(dispatcher, localRankInfo, &rankTable);

    std::string commIdentifier("test");
    HcclPrepareConfig config;
    config.topoType = HcclTopoType::HCCL_TOPO_FULLMESH;
    HcclResult ret = service->Prepare(commIdentifier, &config, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(service->prepared_, true);

    GlobalMockObject::verify();
}

TEST_F(OneSidedUt, Ut_ExchangeMemDesc_When_SetupRemoteRankInfoFail_Expect_ReturnError)
{
    unique_ptr<HcclSocketManager> socketManager
        = std::make_unique<HcclSocketManager>(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0);
    unique_ptr<NotifyPool> notifyPool = std::make_unique<NotifyPool>();
    CommConfig commConfig;
    unique_ptr<HcclOneSidedService> service
        = std::make_unique<HcclOneSidedService>(socketManager, notifyPool, commConfig);

    MOCKER_CPP(&HcclOneSidedService::SetupRemoteRankInfo).stubs().will(returnValue(HCCL_E_NOT_FOUND));

    HcclMemDesc localDesc{};
    HcclMemDescs localMemDescs{&localDesc, 1};
    HcclMemDescs remoteMemDescs{nullptr, 0};
    u32 actualNum = 0;
    HcclResult ret = service->ExchangeMemDesc(0, localMemDescs, remoteMemDescs, actualNum, std::string("test"), 1);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    EXPECT_EQ(service->oneSidedConns_.empty(), true);

    GlobalMockObject::verify();
}

TEST_F(OneSidedUt, Ut_EnableMemAccess_When_ConnNotFound_Expect_ThrowLogicError)
{
    unique_ptr<HcclSocketManager> socketManager
        = std::make_unique<HcclSocketManager>(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0);
    unique_ptr<NotifyPool> notifyPool = std::make_unique<NotifyPool>();
    CommConfig commConfig;
    unique_ptr<HcclOneSidedService> service
        = std::make_unique<HcclOneSidedService>(socketManager, notifyPool, commConfig);

    HcclMemDesc desc{};
    const u32 notFoundRank = 999;
    memcpy_s(desc.desc, sizeof(desc.desc), &notFoundRank, sizeof(notFoundRank));
    HcclMem remoteMem{};
    EXPECT_THROW(service->EnableMemAccess(desc, remoteMem), std::logic_error);
}

TEST_F(OneSidedUt, Ut_DisableMemAccess_When_ConnNotFound_Expect_ThrowLogicError)
{
    unique_ptr<HcclSocketManager> socketManager
        = std::make_unique<HcclSocketManager>(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0);
    unique_ptr<NotifyPool> notifyPool = std::make_unique<NotifyPool>();
    CommConfig commConfig;
    unique_ptr<HcclOneSidedService> service
        = std::make_unique<HcclOneSidedService>(socketManager, notifyPool, commConfig);

    HcclMemDesc desc{};
    const u32 notFoundRank = 999;
    memcpy_s(desc.desc, sizeof(desc.desc), &notFoundRank, sizeof(notFoundRank));
    EXPECT_THROW(service->DisableMemAccess(desc), std::logic_error);
}

TEST_F(OneSidedUt, Ut_BatchPut_When_ConnNotFound_Expect_ThrowOutOfRange)
{
    unique_ptr<HcclSocketManager> socketManager
        = std::make_unique<HcclSocketManager>(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0);
    unique_ptr<NotifyPool> notifyPool = std::make_unique<NotifyPool>();
    CommConfig commConfig;
    unique_ptr<HcclOneSidedService> service
        = std::make_unique<HcclOneSidedService>(socketManager, notifyPool, commConfig);

    HcclOneSideOpDesc opDesc{};
    rtStream_t stream = nullptr;
    EXPECT_THROW(service->BatchPut(999, &opDesc, 1, stream), std::out_of_range);
}

TEST_F(OneSidedUt, Ut_BatchGet_When_ConnNotFound_Expect_ThrowOutOfRange)
{
    unique_ptr<HcclSocketManager> socketManager
        = std::make_unique<HcclSocketManager>(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0);
    unique_ptr<NotifyPool> notifyPool = std::make_unique<NotifyPool>();
    CommConfig commConfig;
    unique_ptr<HcclOneSidedService> service
        = std::make_unique<HcclOneSidedService>(socketManager, notifyPool, commConfig);

    HcclOneSideOpDesc opDesc{};
    rtStream_t stream = nullptr;
    EXPECT_THROW(service->BatchGet(999, &opDesc, 1, stream), std::out_of_range);
}

TEST_F(OneSidedUt, Ut_ConcurrentReadWrite_When_MixedReadWrite_Expect_NoDeadlock)
{
    auto socketManager = std::make_unique<HcclSocketManager>(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0);
    auto notifyPool = std::make_unique<NotifyPool>();
    CommConfig commConfig;
    std::shared_ptr<HcclOneSidedService> service
        = std::make_shared<HcclOneSidedService>(socketManager, notifyPool, commConfig);

    MOCKER_CPP(&HcclOneSidedService::SetupRemoteRankInfo).stubs().will(returnValue(HCCL_E_NOT_FOUND));

    auto writer = std::async(std::launch::async, [service]() {
        HcclMemDesc localDesc{};
        HcclMemDescs localMemDescs{&localDesc, 1};
        HcclMemDescs remoteMemDescs{nullptr, 0};
        u32 actualNum = 0;
        for (int i = 0; i < 50; ++i) {
            (void)service->ExchangeMemDesc(0, localMemDescs, remoteMemDescs, actualNum, std::string("test"), 1);
        }
    });
    std::vector<std::future<void>> readers;
    for (int t = 0; t < 8; ++t) {
        readers.emplace_back(std::async(std::launch::async, [service]() {
            HcclOneSideOpDesc opDesc{};
            rtStream_t stream = nullptr;
            for (int i = 0; i < 50; ++i) {
                try {
                    service->BatchPut(0, &opDesc, 1, stream);
                } catch (const std::out_of_range&) {
                }
            }
        }));
    }

    EXPECT_EQ(writer.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    for (auto& f : readers) {
        EXPECT_EQ(f.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    }
    GlobalMockObject::verify();
}
