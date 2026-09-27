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
#include "communicator_impl.h"
#include "lite_res_mgr_fetcher.h"
#include <vector>

using namespace Hccl;

class ResPackagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ResPackagerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "ResPackagerTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        string testTag = "test";
        memcpy_s(tag.raw, sizeof(tag.raw), testTag.c_str(), testTag.size());
        std::cout << "A Test case in ResPackagerTest SetUp" << std::endl;
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in ResPackagerTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }
    AicpuOpTag tag;
};

TEST_F(ResPackagerTest, test_package_res)
{
    CommunicatorImpl impl;

    impl.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&impl);
    impl.aicpuQueueNotifyManager_ = std::make_unique<QueueNotifyManager>(impl);
    impl.queueWaitGroupCntNotifyManager = std::make_unique<QueueWaitGroupCntNotifyManager>();
    impl.queueBcastPostCntNotifyManager = std::make_unique<QueueBcastPostCntNotifyManager>();
    impl.localRmaBufManager = std::make_unique<LocalRmaBufManager>(impl);
    impl.remoteRmaBufManager = std::make_unique<RemoteRmaBufManager>(impl);
    impl.dataBufferManager = std::make_unique<DataBufManager>();

    u64 addr = 0;
    u64 size = 0;

    vector<char> buffer = {'0'};
    AlgTopoElemMgr algTopoElemMgr("InsAllreduceMesh", 0);
}

class MockResFetcher : public ResMgrFetcher {
public:
    HostDeviceSyncNotifyLiteMgr* GetHostDeviceSyncNotifyLiteMgr() { return &hostDeviceSyncNotifyLiteMgr; }

    StreamLiteMgr* GetStreamLiteMgr() { return &streamLiteMgr; }

    u32 GetExecTimeOut() override { return 1836; }

    QueueNotifyLiteMgr* GetQueueNotifyLiteMgr() { return &queueNotifyLiteMgr; }

    Cnt1tonNotifyLiteMgr* GetCnt1tonNotifyLiteMgr() { return &cnt1tonNotifyLiteMgr; }

    CntNto1NotifyLiteMgr* GetCntNto1NotifyLiteMgr() { return &cntNto1NotifyLiteMgr; }

    ConnectedLinkMgr* GetConnectedLinkMgr() { return &connectedLinkMgr; }

    DevId GetDevPhyId() { return 0; }

    u64 GetLocAddr(BufferType type) { return 0xffffffff; }

private:
    HostDeviceSyncNotifyLiteMgr hostDeviceSyncNotifyLiteMgr;
    StreamLiteMgr streamLiteMgr;
    QueueNotifyLiteMgr queueNotifyLiteMgr;
    Cnt1tonNotifyLiteMgr cnt1tonNotifyLiteMgr;
    CntNto1NotifyLiteMgr cntNto1NotifyLiteMgr;
    ConnectedLinkMgr connectedLinkMgr;
};
