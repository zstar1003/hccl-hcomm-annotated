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
#include <thread>
#define private public
#include "ns_recovery_handler_func.h"
#include "communicator_impl.h"
#include "kfc.h"
#undef private
#include "internal_exception.h"
using namespace Hccl;

constexpr u32 h2dBufferSize = sizeof(KfcCommand);
constexpr u32 d2hBufferSize = sizeof(KfcExecStatus);

static HcclResult HrtDrvMemCpyStub(void* dst, uint64_t destMax, const void* src, uint64_t count)
{
    memcpy(dst, src, count);
    return HCCL_SUCCESS;
}

class NsRecoveryHandlerFuncTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "NsRecoveryHandlerFuncTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "NsRecoveryHandlerFuncTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtDrvMemCpy).stubs().with().will(invoke(HrtDrvMemCpyStub));
        MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
        MOCKER_CPP(&RtsqBase::QuerySqBaseAddr)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(reinterpret_cast<u64>(&mockSq)));
        MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().with(mockcpp::any()).will(returnValue(0));
        MOCKER_CPP(&RtsqBase::ConfigSqStatusByType).stubs();

        std::cout << "A Test case in NsRecoveryHandlerFuncTest SetUp" << std::endl;
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in NsRecoveryHandlerFuncTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }

    u8 mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT]{0};
};

TEST_F(NsRecoveryHandlerFuncTest, test_handle_stop_launch)
{
    // 实现host侧对应的内容
    CommunicatorImpl comm;
    comm.kfcControlTransferH2D = std::make_unique<HDCommunicate>(0, HCCL_HDC_TYPE_H2D, h2dBufferSize);
    comm.kfcStatusTransferD2H = std::make_unique<HDCommunicate>(0, HCCL_HDC_TYPE_D2H, d2hBufferSize);
    comm.kfcControlTransferH2D->Init();
    comm.kfcStatusTransferD2H->Init();
    auto kfcControlTransferH2DParams = comm.kfcControlTransferH2D->GetCommunicateParams();
    auto kfcStatusTransferD2HParams = comm.kfcStatusTransferD2H->GetCommunicateParams();
    // 实现device侧对应的内容->保证device侧共享内存和host侧共享内存是一个
    CommunicatorImplLite serviceLite(0);
    serviceLite.kfcControlTransferH2D = std::make_unique<HDCommunicateLite>();
    serviceLite.kfcStatusTransferD2H = std::make_unique<HDCommunicateLite>();
    serviceLite.kfcControlTransferH2D->Init(kfcControlTransferH2DParams);
    serviceLite.kfcStatusTransferD2H->Init(kfcStatusTransferD2HParams);
    serviceLite.hdcHandler
        = make_unique<AicpuHdcHandler>(*serviceLite.kfcControlTransferH2D, *serviceLite.kfcStatusTransferD2H);

    KfcCommand cmd = KfcCommand::NONE;
    memset_s(&cmd, sizeof(KfcCommand), 0, sizeof(KfcCommand));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    // 模拟host侧行为
    thread threadHandle([&] {
        cmd = KfcCommand::NS_STOP_LAUNCH;
        comm.kfcControlTransferH2D->Put(0, sizeof(KfcCommand), (u8*)&cmd); // 把命令NS_STOP_LAUNCH发到device侧
        auto timeout = std::chrono::milliseconds(100);
        auto startTime = std::chrono::steady_clock::now();
        while (true) {
            comm.kfcStatusTransferD2H->Get(
                0, sizeof(KfcExecStatus), (u8*)&response); // 从device侧拿STOP_LAUNCH_DONE的命令字
            if (response.kfcStatus != KfcStatus::NONE) {
                break;
            }
            if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
                break;
            }
        }
        EXPECT_EQ(response.kfcStatus, KfcStatus::STOP_LAUNCH_DONE); // 希望从device侧拿到STOP_LAUNCH_DONE
    });
    usleep(1000);

    serviceLite.isSuspended = false;
    serviceLite.isCommReady = true;

    auto timeout = std::chrono::milliseconds(100);
    auto startTime = std::chrono::steady_clock::now();
    while (true) {
        NsRecoveryHandlerFunc::GetInstance().HandleStopLaunch(&serviceLite);
        if (serviceLite.isSuspended) {
            break;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            break;
        }
    }
    threadHandle.join();
    EXPECT_EQ(serviceLite.isSuspended, true); // 希望serviceLite进入暂停状态
    EXPECT_EQ(serviceLite.needClean, true);   // 希望serviceLite需要清理
}

TEST_F(NsRecoveryHandlerFuncTest, test_handle_clean)
{
    // 实现host侧对应的内容
    CommunicatorImpl comm;
    comm.kfcControlTransferH2D = std::make_unique<HDCommunicate>(0, HCCL_HDC_TYPE_H2D, h2dBufferSize);
    comm.kfcStatusTransferD2H = std::make_unique<HDCommunicate>(0, HCCL_HDC_TYPE_D2H, d2hBufferSize);
    comm.kfcControlTransferH2D->Init();
    comm.kfcStatusTransferD2H->Init();
    auto kfcControlTransferH2DParams = comm.kfcControlTransferH2D->GetCommunicateParams();
    auto kfcStatusTransferD2HParams = comm.kfcStatusTransferD2H->GetCommunicateParams();
    // 实现device侧对应的内容->保证device侧共享内存和host侧共享内存是一个
    CommunicatorImplLite serviceLite(0);
    serviceLite.kfcControlTransferH2D = std::make_unique<HDCommunicateLite>();
    serviceLite.kfcStatusTransferD2H = std::make_unique<HDCommunicateLite>();
    serviceLite.kfcControlTransferH2D->Init(kfcControlTransferH2DParams);
    serviceLite.kfcStatusTransferD2H->Init(kfcStatusTransferD2HParams);
    serviceLite.hdcHandler
        = make_unique<AicpuHdcHandler>(*serviceLite.kfcControlTransferH2D, *serviceLite.kfcStatusTransferD2H);

    KfcCommand cmd = KfcCommand::NONE;
    memset_s(&cmd, sizeof(KfcCommand), 0, sizeof(KfcCommand));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    // 模拟host侧行为
    thread threadHandle([&] {
        cmd = KfcCommand::NS_CLEAN;
        comm.kfcControlTransferH2D->Put(0, sizeof(KfcCommand), (u8*)&cmd); // 把命令NS_CLEAN发到device侧
        auto timeout = std::chrono::milliseconds(100);
        auto startTime = std::chrono::steady_clock::now();
        while (true) {
            comm.kfcStatusTransferD2H->Get(0, sizeof(KfcExecStatus), (u8*)&response); // 从device侧拿CLEAN_DONE的命令字
            if (response.kfcStatus != KfcStatus::NONE) {
                break;
            }
            if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
                break;
            }
        }
        EXPECT_EQ(response.kfcStatus, KfcStatus::CLEAN_DONE); // 希望从device侧拿到CLEAN_DONE
    });
    usleep(1000);

    serviceLite.isCommReady = true;
    serviceLite.isSuspended = true;
    serviceLite.needClean = true;

    u32 fakeStreamId = 0;
    u32 fakeSqId = 0;
    u32 fakedevPhyId = 0;
    BinaryStream liteBinaryStream;
    liteBinaryStream << fakeStreamId;
    liteBinaryStream << fakeSqId;
    liteBinaryStream << fakedevPhyId;
    std::vector<char> uniqueId{};
    liteBinaryStream.Dump(uniqueId);
    StreamLite stream(uniqueId);
    MOCKER_CPP(&StreamLiteMgr::GetMaster).stubs().with().will(returnValue(&stream));
    MOCKER_CPP(&NsRecoveryHandlerFunc::DeviceQuery)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    auto timeout = std::chrono::milliseconds(100);
    auto startTime = std::chrono::steady_clock::now();
    while (true) {
        NsRecoveryHandlerFunc::GetInstance().HandleClean(&serviceLite);
        if (serviceLite.needClean == false) {
            break;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            break;
        }
    }
    threadHandle.join();
    EXPECT_EQ(serviceLite.isSuspended, true); // 希望serviceLite维持暂停状态
    EXPECT_EQ(serviceLite.needClean, false);  // 希望serviceLite取消需要清理的状态
}

TEST_F(NsRecoveryHandlerFuncTest, test_device_query)
{
    MOCKER(halTsdrvCtl).stubs().with(mockcpp::any()).will(returnValue(DRV_ERROR_NOT_SUPPORT));
    auto ret = NsRecoveryHandlerFunc::GetInstance().DeviceQuery(0, 0, 0);
    EXPECT_EQ(ret, HcclResult::HCCL_E_DRV);
    GlobalMockObject::verify();

    MOCKER(halTsdrvCtl).stubs().with(mockcpp::any()).will(returnValue(DRV_ERROR_NONE));
    ret = NsRecoveryHandlerFunc::GetInstance().DeviceQuery(0, 0, 0);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}
