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

#ifndef private
#define private public
#define protected public
#endif

#include "transport_device_ibverbs.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "sal.h"

#include "adapter_rts.h"
#include "ascend_hal.h"
#include "dispatcher_pub.h"

#undef private
#undef protected

using namespace std;
using namespace hccl;

class TransportIbverbs_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--TransportIbverbs_UT SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--TransportIbverbs_UT TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        // 初始化dispatcher
        dispatcher = new (std::nothrow) DispatcherPub(s32(0));

        // 初始化notifyPool
        std::unique_ptr<NotifyPool> notifyPool;
        notifyPool.reset(new (std::nothrow) NotifyPool());

        // 初始化MachinePara
        machinePara.deviceLogicId = 0;
        if (deviceMem.ptr() == nullptr) {
            DeviceMem::alloc(deviceMem, devSize);
        }
        machinePara.inputMem = deviceMem;
    }
    virtual void TearDown()
    {
        std::cout << "A Test TearDown" << std::endl;
        delete dispatcher;
        if (deviceMem.ptr() != nullptr) {
            deviceMem.free();
        }
        GlobalMockObject::verify();
    }

    DispatcherPub* dispatcher;
    std::unique_ptr<NotifyPool> notifyPool = nullptr;
    MachinePara machinePara;
    DeviceMem deviceMem;
    std::chrono::milliseconds timeout = std::chrono::milliseconds(3000);
    int devSize = 1024;
};

TEST_F(TransportIbverbs_UT, RegUserMem)
{
    MOCKER_CPP(&TransportIbverbs::UseMultiQp).stubs().will(returnValue(false));
    HcclUs startut = TIME_NOW();
    s32 ret = HCCL_SUCCESS;

    std::shared_ptr<TransportIbverbs> ibverbs
        = std::make_shared<TransportIbverbs>(dispatcher, notifyPool, machinePara, timeout);
    std::vector<u8> exchangeDataForSend_;
    exchangeDataForSend_.resize(devSize);
    u64 exchangeDataBlankSize = devSize;
    u8* exchangeDataPtr = exchangeDataForSend_.data();
    ret = ibverbs->RegUserMem(MemType::USER_INPUT_MEM, exchangeDataPtr, exchangeDataBlankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TransportIbverbs_UT, BatchTransferAsync_WriteSuccess)
{
    MOCKER_CPP(&TransportDeviceIbverbs::UseMultiQp).stubs().will(returnValue(false));

    TransportDeviceIbverbsData transDevIbverbsData;
    transDevIbverbsData.qpInfo.resize(1);
    transDevIbverbsData.qpsPerConnection = 1;
    transDevIbverbsData.multiQpThreshold = HCCL_MULTI_QP_THRESHOLD_DEFAULT;

    std::shared_ptr<TransportDeviceIbverbs> ibverbs
        = std::make_shared<TransportDeviceIbverbs>(dispatcher, notifyPool, machinePara, timeout, transDevIbverbsData);

    Stream stream;
    HcommBatchTransferDesc transferDescs[1];
    transferDescs[0].transType = HCOMM_TRANSFER_TYPE_WRITE;
    transferDescs[0].transferInfo.write.dst = reinterpret_cast<void*>(0x1000);
    transferDescs[0].transferInfo.write.src = reinterpret_cast<void*>(0x2000);
    transferDescs[0].transferInfo.write.len = 1024;

    HcclResult ret = ibverbs->BatchTransferAsync(transferDescs, 1, stream);
    GlobalMockObject::verify();
}

TEST_F(TransportIbverbs_UT, BatchTransferAsync_ReadSuccess)
{
    MOCKER_CPP(&TransportDeviceIbverbs::UseMultiQp).stubs().will(returnValue(false));

    TransportDeviceIbverbsData transDevIbverbsData;
    transDevIbverbsData.qpInfo.resize(1);
    transDevIbverbsData.qpsPerConnection = 1;
    transDevIbverbsData.multiQpThreshold = HCCL_MULTI_QP_THRESHOLD_DEFAULT;

    std::shared_ptr<TransportDeviceIbverbs> ibverbs
        = std::make_shared<TransportDeviceIbverbs>(dispatcher, notifyPool, machinePara, timeout, transDevIbverbsData);

    Stream stream;
    HcommBatchTransferDesc transferDescs[1];
    transferDescs[0].transType = HCOMM_TRANSFER_TYPE_READ;
    transferDescs[0].transferInfo.read.dst = reinterpret_cast<void*>(0x2000);
    transferDescs[0].transferInfo.read.src = reinterpret_cast<void*>(0x1000);
    transferDescs[0].transferInfo.read.len = 1024;

    HcclResult ret = ibverbs->BatchTransferAsync(transferDescs, 1, stream);
    GlobalMockObject::verify();
}

TEST_F(TransportIbverbs_UT, BatchTransferAsync_NullPtr)
{
    MOCKER_CPP(&TransportDeviceIbverbs::UseMultiQp).stubs().will(returnValue(false));

    TransportDeviceIbverbsData transDevIbverbsData;
    transDevIbverbsData.qpInfo.resize(1);
    transDevIbverbsData.qpsPerConnection = 1;

    std::shared_ptr<TransportDeviceIbverbs> ibverbs
        = std::make_shared<TransportDeviceIbverbs>(dispatcher, notifyPool, machinePara, timeout, transDevIbverbsData);

    Stream stream;
    HcommBatchTransferDesc transferDescs[1];
    transferDescs[0].transType = HCOMM_TRANSFER_TYPE_WRITE;
    transferDescs[0].transferInfo.write.dst = nullptr;
    transferDescs[0].transferInfo.write.src = reinterpret_cast<void*>(0x2000);
    transferDescs[0].transferInfo.write.len = 1024;

    HcclResult ret = ibverbs->BatchTransferAsync(transferDescs, 1, stream);
    EXPECT_EQ(ret, HCCL_E_PTR);
    GlobalMockObject::verify();
}

TEST_F(TransportIbverbs_UT, test_TransportIbverb_Drain_will_return_SUCCESS)
{
    MOCKER_CPP(&TransportDeviceIbverbs::SendWrlistExt).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(
        &DispatcherPub::SignalWait, HcclResult(DispatcherPub::*)(HcclRtNotify, HcclRtStream, u32, u32, s32, u32, bool))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(
        *dispatcher, &DispatcherPub::RdmaSend, HcclResult(DispatcherPub::*)(u32, u64, hccl::Stream&, RdmaTaskInfo&))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    TransportDeviceIbverbsData transDevIbverbsData;
    transDevIbverbsData.qpInfo.resize(1);
    transDevIbverbsData.qpsPerConnection = 1;

    std::shared_ptr<TransportDeviceIbverbs> ibverbsDev
        = std::make_shared<TransportDeviceIbverbs>(dispatcher, notifyPool, machinePara, timeout, transDevIbverbsData);

    ibverbsDev->dataNotify_ = std::make_shared<LocalIpcNotify>();
    ibverbsDev->memMsg_[MemType::DATA_NOTIFY_MEM].addr = (void*)(0x9999);
    ibverbsDev->remoteMemMsg_[MemType::NOTIFY_SRC_MEM].addr = (void*)(0x8899);

    Stream stream;
    HcclResult ret = ibverbsDev->Drain(stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<TransportIbverbs> ibverbs
        = std::make_shared<TransportIbverbs>(dispatcher, notifyPool, machinePara, timeout);
    ibverbs->dataNotify_ = std::make_shared<LocalIpcNotify>();
    ibverbs->memMsg_[MemType::DATA_NOTIFY_MEM].addr = (void*)(0x9999);
    ibverbs->remoteMemMsg_[MemType::NOTIFY_SRC_MEM].addr = (void*)(0x8899);

    ret = ibverbs->Drain(stream);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    GlobalMockObject::verify();
}
