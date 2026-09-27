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

#include "transport_device_p2p.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "sal.h"

#include "adapter_rts.h"
#include "ascend_hal.h"
#include "dispatcher_aicpu_pub.h"

#undef private
#undef protected

using namespace std;
using namespace hccl;

class TransportDeviceP2pAiCpu_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--TransportDeviceP2pAiCpu_UT SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase()
    {
        std::cout << "\033[36m--TransportDeviceP2pAiCpu_UT TearDown--\033[0m" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        // 初始化dispatcher
        DispatcherPub disPatcherTemp(s32(1));
        dispatcher = &disPatcherTemp;

        // 初始化notifyPool
        notifyPool.reset(new (std::nothrow) NotifyPool());

        // 初始化MachinePara

        // 初始化TransportDeviceP2pData
        HcclSignalInfo notifyInfo;
        notifyInfo.addr = 100;
        notifyInfo.devId = 1;
        notifyInfo.rankId = 2;
        notifyInfo.resId = 3;
        notifyInfo.tsId = 4;
        notifyInfo.flag = 1;

        transDevP2pData.inputBufferPtr = nullptr;
        transDevP2pData.outputBufferPtr = nullptr;
        transDevP2pData.ipcPreWaitNotify = std::make_shared<LocalIpcNotify>();
        transDevP2pData.ipcPreWaitNotify->Init(notifyInfo, NotifyLoadType::DEVICE_NOTIFY);
        transDevP2pData.ipcPostWaitNotify = std::make_shared<LocalIpcNotify>();
        transDevP2pData.ipcPostWaitNotify->Init(notifyInfo, NotifyLoadType::DEVICE_NOTIFY);
        transDevP2pData.ipcPreRecordNotify = std::make_shared<RemoteNotify>();
        transDevP2pData.ipcPreRecordNotify->Init(notifyInfo, NotifyLoadType::DEVICE_NOTIFY);
        transDevP2pData.ipcPostRecordNotify = std::make_shared<RemoteNotify>();
        transDevP2pData.ipcPostRecordNotify->Init(notifyInfo, NotifyLoadType::DEVICE_NOTIFY);
        transDevP2pData.transportAttr.linkType = LinkType::LINK_HCCS_SW;

        machinePara.deviceLogicId = 0;

        signalBuff = DeviceMem::alloc(4);
        transDevP2pData.transportAttr.signalRecordBuff.length = 4;
        transDevP2pData.transportAttr.signalRecordBuff.address = reinterpret_cast<u64>(signalBuff.ptr());

        dispatcher = new (std::nothrow) DispatcherAiCpu(0);
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        std::cout << "A Test TearDown" << std::endl;
        delete dispatcher;
        GlobalMockObject::verify();
    }

    DispatcherPub* dispatcher;
    std::unique_ptr<NotifyPool> notifyPool = nullptr;
    MachinePara machinePara;
    std::chrono::milliseconds timeout;
    TransportDeviceP2pData transDevP2pData;
    DeviceMem signalBuff;
};

TEST_F(TransportDeviceP2pAiCpu_UT, constructor_and_init)
{
    transDevP2pData.transportAttr.linkType = LinkType::LINK_HCCS_SW;
    transDevP2pData.transportAttr.relationship |= HCCL_TRANSPORT_RELATIONSHIP_SAME_SERVER;
    transDevP2pData.transportAttr.relationship |= HCCL_TRANSPORT_RELATIONSHIP_SAME_SUPERPOD;

    s32 ret = HCCL_SUCCESS;

    TransportDeviceP2p transDevP2p(dispatcher, notifyPool, machinePara, timeout, transDevP2pData);
    EXPECT_EQ(transDevP2p.remoteInputPtr_, transDevP2pData.inputBufferPtr);
    EXPECT_EQ(transDevP2p.remoteOutputPtr_, transDevP2pData.outputBufferPtr);
    EXPECT_EQ(transDevP2p.transportAttr_.linkType, transDevP2pData.transportAttr.linkType);
    EXPECT_EQ(transDevP2p.transportAttr_.linkType, transDevP2pData.transportAttr.linkType);

    // init
    ret = transDevP2p.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(transDevP2p.remoteSendReadyAddress_, u64(100));

    // signal record
    MOCKER_CPP_VIRTUAL(
        *dispatcher, &DispatcherPub::SignalRecord,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u64, s32, bool, u64, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    Stream stream;
    ret = transDevP2p.SignalRecord(transDevP2p.remoteSendReadyNotify_, transDevP2p.remoteSendReadyAddress_, 0, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TransportDeviceP2pAiCpu_UT, transport_init_A3_between_servers)
{
    transDevP2pData.transportAttr.linkType = LinkType::LINK_HCCS_SW;
    transDevP2pData.transportAttr.relationship |= HCCL_TRANSPORT_RELATIONSHIP_SAME_SUPERPOD;

    s32 ret = HCCL_SUCCESS;

    TransportDeviceP2p transDevP2p(dispatcher, notifyPool, machinePara, timeout, transDevP2pData);
    EXPECT_EQ(transDevP2p.remoteInputPtr_, transDevP2pData.inputBufferPtr);
    EXPECT_EQ(transDevP2p.remoteOutputPtr_, transDevP2pData.outputBufferPtr);
    EXPECT_EQ(transDevP2p.transportAttr_.linkType, transDevP2pData.transportAttr.linkType);
    EXPECT_EQ(transDevP2p.transportAttr_.relationship, transDevP2pData.transportAttr.relationship);

    unsigned long notifyAddr = 0xFFFFFFFF00000000;
    unsigned int notifyLen = 4;
    MOCKER(halResAddrMap)
        .stubs()
        .with(
            mockcpp::any(), mockcpp::any(), outBoundP(&notifyAddr, sizeof(notifyAddr)),
            outBoundP(&notifyLen, sizeof(notifyLen)))
        .will(returnValue(0));

    // init
    ret = transDevP2p.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(transDevP2p.remoteSendReadyAddress_, notifyAddr);

    // signal record
    MOCKER_CPP_VIRTUAL(
        *dispatcher, &DispatcherPub::SignalRecord,
        HcclResult(DispatcherPub::*)(hccl::DeviceMem&, hccl::DeviceMem&, hccl::Stream&, u32, hccl::LinkType, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    Stream stream;
    ret = transDevP2p.SignalRecord(transDevP2p.remoteSendReadyNotify_, transDevP2p.remoteSendReadyAddress_, 0, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
