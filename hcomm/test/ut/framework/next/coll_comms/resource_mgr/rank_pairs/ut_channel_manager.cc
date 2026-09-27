/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

// 允许 UT 直接访问 ChannelManager 的 private 成员（对齐 ut_notify_manager.cc 的写法）
#define private public

#include "channel_manager.h"
#include "hccl/hccl_res.h"

using namespace hccl;

class ChannelManagerTest : public testing::Test {
protected:
    ChannelManager mgr;

    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(ChannelManagerTest, Init_Normal)
{
    ManagerCallbacks callbacks;
    HcclResult ret = mgr.Init((aclrtBinHandle)0x1, 10, callbacks);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, SetHcclQos_Normal)
{
    HcclResult ret = mgr.SetHcclQos(42);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, ReleaseChannel_Empty)
{
    HcclResult ret = mgr.ReleaseChannel();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, ChannelCommDestroy_EmptyList)
{
    ChannelHandle channelList[1] = {0};
    HcclResult ret = mgr.ChannelCommDestroy(channelList, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, RegisterHandle_NewChannel)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclChannelDesc desc{};
    memset(&desc, 0, sizeof(desc));
    desc.remoteRank = 1;
    desc.channelProtocol = COMM_PROTOCOL_HCCS;

    HcclResult ret = mgr.RegisterHandle("tag1", COMM_ENGINE_CPU, desc, (ChannelHandle)0x100);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, RegisterHandle_DuplicateChannel)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclChannelDesc desc{};
    memset(&desc, 0, sizeof(desc));
    desc.remoteRank = 1;
    desc.channelProtocol = COMM_PROTOCOL_HCCS;

    mgr.RegisterHandle("tag1", COMM_ENGINE_CPU, desc, (ChannelHandle)0x100);
    HcclResult ret = mgr.RegisterHandle("tag1", COMM_ENGINE_CPU, desc, (ChannelHandle)0x200);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, IsChannelExist_NotExist)
{
    HcclResult ret = mgr.IsChannelExist((ChannelHandle)0x999);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, IsChannelExist_Exist)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclChannelDesc desc{};
    memset(&desc, 0, sizeof(desc));
    desc.remoteRank = 1;
    desc.channelProtocol = COMM_PROTOCOL_HCCS;

    mgr.RegisterHandle("tag1", COMM_ENGINE_CPU, desc, (ChannelHandle)0x100);
    HcclResult ret = mgr.IsChannelExist((ChannelHandle)0x100);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, UnregisterHandle_NotExist)
{
    HcclResult ret = mgr.UnregisterHandle((ChannelHandle)0x999);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, UnregisterHandle_Exist)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclChannelDesc desc{};
    memset(&desc, 0, sizeof(desc));
    desc.remoteRank = 1;
    desc.channelProtocol = COMM_PROTOCOL_HCCS;

    mgr.RegisterHandle("tag1", COMM_ENGINE_CPU, desc, (ChannelHandle)0x100);
    HcclResult ret = mgr.UnregisterHandle((ChannelHandle)0x100);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = mgr.IsChannelExist((ChannelHandle)0x100);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, RegisterHandleHDPair_ZeroHandle)
{
    HcclResult ret = mgr.RegisterHandleHDPair(0, (ChannelHandle)0x100);
    EXPECT_NE(ret, HCCL_SUCCESS);

    ret = mgr.RegisterHandleHDPair((ChannelHandle)0x100, 0);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, RegisterHandleHDPair_Normal)
{
    HcclResult ret = mgr.RegisterHandleHDPair((ChannelHandle)0x100, (ChannelHandle)0x200);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, RegisterHandleHDPair_Duplicate)
{
    mgr.RegisterHandleHDPair((ChannelHandle)0x100, (ChannelHandle)0x200);
    HcclResult ret = mgr.RegisterHandleHDPair((ChannelHandle)0x100, (ChannelHandle)0x300);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, CheckNotifyOrQPMaxNum_Sufficient)
{
    u64 existNum = 10;
    HcclResult ret = mgr.CheckNotifyOrQPMaxNum(existNum, 100, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, CheckNotifyOrQPMaxNum_Insufficient)
{
    u64 existNum = 100;
    HcclResult ret = mgr.CheckNotifyOrQPMaxNum(existNum, 100, false);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, CheckNotifyOrQPMaxNum_Equal)
{
    u64 existNum = 99;
    HcclResult ret = mgr.CheckNotifyOrQPMaxNum(existNum, 100, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, CheckChannelParam_ValidHccs)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclChannelDesc desc[2] = {};
    memset(desc, 0, sizeof(desc));
    desc[0].remoteRank = 1;
    desc[0].channelProtocol = COMM_PROTOCOL_HCCS;
    desc[0].notifyNum = 2;
    desc[1].remoteRank = 2;
    desc[1].channelProtocol = COMM_PROTOCOL_HCCS;
    desc[1].notifyNum = 4;

    HcclResult ret = mgr.CheckChannelParam(COMM_ENGINE_CPU, desc, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, CheckChannelParam_NotifyNumExceedMax)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclChannelDesc desc[1] = {};
    memset(desc, 0, sizeof(desc));
    desc[0].remoteRank = 1;
    desc[0].channelProtocol = COMM_PROTOCOL_HCCS;
    desc[0].notifyNum = 65;

    HcclResult ret = mgr.CheckChannelParam(COMM_ENGINE_CPU, desc, 1);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, CheckChannelParam_SelfRank)
{
    mgr.Init((aclrtBinHandle)0x1, 5, ManagerCallbacks{});
    HcclChannelDesc desc[1] = {};
    memset(desc, 0, sizeof(desc));
    desc[0].remoteRank = 5;
    desc[0].channelProtocol = COMM_PROTOCOL_HCCS;
    desc[0].notifyNum = 2;

    HcclResult ret = mgr.CheckChannelParam(COMM_ENGINE_CPU, desc, 1);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, CheckChannelParam_DuplicateDesc)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclChannelDesc desc[2] = {};
    memset(desc, 0, sizeof(desc));
    desc[0].remoteRank = 1;
    desc[0].channelProtocol = COMM_PROTOCOL_HCCS;
    desc[0].notifyNum = 2;
    desc[1].remoteRank = 1;
    desc[1].channelProtocol = COMM_PROTOCOL_HCCS;
    desc[1].notifyNum = 2;

    HcclResult ret = mgr.CheckChannelParam(COMM_ENGINE_CPU, desc, 2);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, CheckChannelParam_UnsupportedProtocol)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclChannelDesc desc[1] = {};
    memset(desc, 0, sizeof(desc));
    desc[0].remoteRank = 1;
    desc[0].channelProtocol = (CommProtocol)99;
    desc[0].notifyNum = 2;

    HcclResult ret = mgr.CheckChannelParam(COMM_ENGINE_CPU, desc, 1);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, CheckChannelParam_UnsupportedEngine)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclChannelDesc desc[1] = {};
    memset(desc, 0, sizeof(desc));
    desc[0].remoteRank = 1;
    desc[0].channelProtocol = COMM_PROTOCOL_HCCS;
    desc[0].notifyNum = 2;

    HcclResult ret = mgr.CheckChannelParam(COMM_ENGINE_CCU, desc, 1);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, PrepareHandleArray_AllNew)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclChannelDesc desc[2] = {};
    memset(desc, 0, sizeof(desc));
    desc[0].remoteRank = 1;
    desc[0].channelProtocol = COMM_PROTOCOL_HCCS;
    desc[1].remoteRank = 2;
    desc[1].channelProtocol = COMM_PROTOCOL_ROCE;

    ChannelHandle handleArray[2] = {0};
    std::vector<HcclChannelDesc> needCreateDescs;
    std::vector<uint32_t> needCreateIndices;

    HcclResult ret
        = mgr.PrepareHandleArray("tag", COMM_ENGINE_CPU, desc, 2, handleArray, needCreateDescs, needCreateIndices);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(needCreateDescs.size(), 2u);
    EXPECT_EQ(needCreateIndices.size(), 2u);
    EXPECT_EQ(handleArray[0], (ChannelHandle)0);
    EXPECT_EQ(handleArray[1], (ChannelHandle)0);
}

TEST_F(ChannelManagerTest, PrepareHandleArray_PartialExist)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclChannelDesc desc[2] = {};
    memset(desc, 0, sizeof(desc));
    desc[0].remoteRank = 1;
    desc[0].channelProtocol = COMM_PROTOCOL_HCCS;
    desc[1].remoteRank = 2;
    desc[1].channelProtocol = COMM_PROTOCOL_ROCE;

    mgr.RegisterHandle("tag", COMM_ENGINE_CPU, desc[0], (ChannelHandle)0x100);

    ChannelHandle handleArray[2] = {0};
    std::vector<HcclChannelDesc> needCreateDescs;
    std::vector<uint32_t> needCreateIndices;

    HcclResult ret
        = mgr.PrepareHandleArray("tag", COMM_ENGINE_CPU, desc, 2, handleArray, needCreateDescs, needCreateIndices);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(needCreateDescs.size(), 1u);
    EXPECT_EQ(needCreateIndices.size(), 1u);
    EXPECT_EQ(handleArray[0], (ChannelHandle)0x100);
    EXPECT_EQ(handleArray[1], (ChannelHandle)0);
}

TEST_F(ChannelManagerTest, BuildChannelRequests_Normal)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    std::vector<HcclChannelDesc> descs;
    HcclChannelDesc desc1{};
    memset(&desc1, 0, sizeof(desc1));
    desc1.remoteRank = 1;
    desc1.channelProtocol = COMM_PROTOCOL_HCCS;
    desc1.notifyNum = 2;
    descs.push_back(desc1);

    HcclChannelDesc desc2{};
    memset(&desc2, 0, sizeof(desc2));
    desc2.remoteRank = 2;
    desc2.channelProtocol = COMM_PROTOCOL_ROCE;
    desc2.notifyNum = 3;
    descs.push_back(desc2);

    OpCommTransport result = mgr.BuildChannelRequests(descs);
    EXPECT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].size(), 1u);
    EXPECT_EQ(result[0][0].transportRequests.size(), 2u);
    EXPECT_EQ(result[0][0].transportRequests[0].remoteUserRank, 1u);
    EXPECT_EQ(result[0][0].transportRequests[1].remoteUserRank, 2u);
    EXPECT_FALSE(result[0][0].transportRequests[0].isUsedRdma);
    EXPECT_TRUE(result[0][0].transportRequests[1].isUsedRdma);
}

TEST_F(ChannelManagerTest, BuildChannelRequests_Empty)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    std::vector<HcclChannelDesc> descs;

    OpCommTransport result = mgr.BuildChannelRequests(descs);
    EXPECT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].size(), 1u);
    EXPECT_EQ(result[0][0].transportRequests.size(), 0u);
}

TEST_F(ChannelManagerTest, ClearOpTransportResponseLinks_Normal)
{
    OpCommTransport opTransport;
    LevelNSubCommTransport level0;
    SingleSubCommTransport commTransport;

    TransportRequest req1;
    req1.isValid = true;
    req1.localUserRank = 0;
    req1.remoteUserRank = 1;
    commTransport.transportRequests.push_back(req1);

    TransportRequest req2;
    req2.isValid = true;
    req2.localUserRank = 0;
    req2.remoteUserRank = 2;
    commTransport.transportRequests.push_back(req2);

    commTransport.links.resize(2, nullptr);
    commTransport.status.resize(2, TransportStatus::INIT);

    level0.push_back(commTransport);
    opTransport.push_back(level0);

    mgr.ClearOpTransportResponseLinks(opTransport);
    EXPECT_EQ(opTransport[0][0].links.size(), 2u);
    EXPECT_EQ(opTransport[0][0].status.size(), 2u);
    for (const auto& s : opTransport[0][0].status) {
        EXPECT_EQ(s, TransportStatus::INIT);
    }
}

TEST_F(ChannelManagerTest, GetHostChannel_NonAicpuEngine)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclChannelDesc desc{};
    memset(&desc, 0, sizeof(desc));
    desc.remoteRank = 1;
    desc.channelProtocol = COMM_PROTOCOL_HCCS;

    mgr.RegisterHandle("tag", COMM_ENGINE_CPU, desc, (ChannelHandle)0x100);

    ChannelHandle hostChannel;
    HcclResult ret = mgr.GetHostChannel((ChannelHandle)0x100, hostChannel);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(hostChannel, (ChannelHandle)0x100);
}

TEST_F(ChannelManagerTest, GetHostChannel_AicpuWithoutPair)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclChannelDesc desc{};
    memset(&desc, 0, sizeof(desc));
    desc.remoteRank = 1;
    desc.channelProtocol = COMM_PROTOCOL_HCCS;

    mgr.RegisterHandle("tag", COMM_ENGINE_AICPU, desc, (ChannelHandle)0x100);

    ChannelHandle hostChannel;
    HcclResult ret = mgr.GetHostChannel((ChannelHandle)0x100, hostChannel);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, GetHostChannel_AicpuWithPair)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclChannelDesc desc{};
    memset(&desc, 0, sizeof(desc));
    desc.remoteRank = 1;
    desc.channelProtocol = COMM_PROTOCOL_HCCS;

    mgr.RegisterHandle("tag", COMM_ENGINE_AICPU, desc, (ChannelHandle)0x100);
    mgr.RegisterHandleHDPair((ChannelHandle)0x100, (ChannelHandle)0x200);

    ChannelHandle hostChannel;
    HcclResult ret = mgr.GetHostChannel((ChannelHandle)0x100, hostChannel);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(hostChannel, (ChannelHandle)0x200);
}

TEST_F(ChannelManagerTest, ChannelCommDestroy_RegisteredChannel)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclChannelDesc desc{};
    memset(&desc, 0, sizeof(desc));
    desc.remoteRank = 1;
    desc.channelProtocol = COMM_PROTOCOL_HCCS;

    mgr.RegisterHandle("tag", COMM_ENGINE_CPU, desc, (ChannelHandle)0x100);

    ChannelHandle channelList[1] = {(ChannelHandle)0x100};
    HcclResult ret = mgr.ChannelCommDestroy(channelList, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(channelList[0], (ChannelHandle)0);

    ret = mgr.IsChannelExist((ChannelHandle)0x100);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(ChannelManagerTest, ChannelCommDestroy_MixedChannels)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclChannelDesc desc1{};
    memset(&desc1, 0, sizeof(desc1));
    desc1.remoteRank = 1;
    desc1.channelProtocol = COMM_PROTOCOL_HCCS;

    HcclChannelDesc desc2{};
    memset(&desc2, 0, sizeof(desc2));
    desc2.remoteRank = 2;
    desc2.channelProtocol = COMM_PROTOCOL_ROCE;

    mgr.RegisterHandle("tag", COMM_ENGINE_CPU, desc1, (ChannelHandle)0x100);
    mgr.RegisterHandle("tag", COMM_ENGINE_CPU, desc2, (ChannelHandle)0x200);

    ChannelHandle channelList[3] = {(ChannelHandle)0x100, (ChannelHandle)0x999, (ChannelHandle)0x200};
    HcclResult ret = mgr.ChannelCommDestroy(channelList, 3);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(channelList[0], (ChannelHandle)0);
    EXPECT_EQ(channelList[1], (ChannelHandle)0);
    EXPECT_EQ(channelList[2], (ChannelHandle)0);
}

// ============ BuildChannelKey (memHandles 纳入 key) 相关测试 ============
// BuildChannelKey 是文件内 static 函数，通过 RegisterHandle + PrepareHandleArray 间接验证：
// 相同 memHandle 的 desc 命中复用；不同 memHandle 的 desc 不命中、进入 needCreateDescs。

TEST_F(ChannelManagerTest, PrepareHandleArray_SameMemHandle_Reused)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclMemHandle handle = reinterpret_cast<HcclMemHandle>(0x1000);
    HcclChannelDesc desc{};
    memset(&desc, 0, sizeof(desc));
    desc.remoteRank = 1;
    desc.channelProtocol = COMM_PROTOCOL_HCCS;
    desc.memHandles = &handle;
    desc.memHandleNum = 1;

    // 先注册一个 channel
    mgr.RegisterHandle("tag", COMM_ENGINE_CPU, desc, (ChannelHandle)0x100);

    // 用相同 memHandle 再次准备：应命中复用，needCreateDescs 为空
    ChannelHandle handleArray[1] = {0};
    std::vector<HcclChannelDesc> needCreateDescs;
    std::vector<uint32_t> needCreateIndices;
    HcclResult ret
        = mgr.PrepareHandleArray("tag", COMM_ENGINE_CPU, &desc, 1, handleArray, needCreateDescs, needCreateIndices);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(needCreateDescs.size(), 0u);
    EXPECT_EQ(handleArray[0], (ChannelHandle)0x100);
}

TEST_F(ChannelManagerTest, PrepareHandleArray_DifferentMemHandle_NotReused)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclMemHandle handleA = reinterpret_cast<HcclMemHandle>(0x1000);
    HcclChannelDesc descA{};
    memset(&descA, 0, sizeof(descA));
    descA.remoteRank = 1;
    descA.channelProtocol = COMM_PROTOCOL_HCCS;
    descA.memHandles = &handleA;
    descA.memHandleNum = 1;

    // 先用 handleA 注册
    mgr.RegisterHandle("tag", COMM_ENGINE_CPU, descA, (ChannelHandle)0x100);

    // 用不同 memHandle 的 desc 再次准备：不应命中，应进入 needCreateDescs
    HcclMemHandle handleB = reinterpret_cast<HcclMemHandle>(0x2000);
    HcclChannelDesc descB{};
    memset(&descB, 0, sizeof(descB));
    descB.remoteRank = 1;
    descB.channelProtocol = COMM_PROTOCOL_HCCS;
    descB.memHandles = &handleB;
    descB.memHandleNum = 1;

    ChannelHandle handleArray[1] = {0};
    std::vector<HcclChannelDesc> needCreateDescs;
    std::vector<uint32_t> needCreateIndices;
    HcclResult ret
        = mgr.PrepareHandleArray("tag", COMM_ENGINE_CPU, &descB, 1, handleArray, needCreateDescs, needCreateIndices);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(needCreateDescs.size(), 1u);
    EXPECT_EQ(handleArray[0], (ChannelHandle)0);
}

TEST_F(ChannelManagerTest, PrepareHandleArray_ZeroMemHandleNum_TreatedAsNoMem)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclChannelDesc desc{};
    memset(&desc, 0, sizeof(desc));
    desc.remoteRank = 1;
    desc.channelProtocol = COMM_PROTOCOL_HCCS;
    desc.memHandles = nullptr;
    desc.memHandleNum = 0;

    // memHandleNum=0 注册
    mgr.RegisterHandle("tag", COMM_ENGINE_CPU, desc, (ChannelHandle)0x100);

    // 再次用 memHandleNum=0 准备：应命中复用（都走 ":0" 分支）
    ChannelHandle handleArray[1] = {0};
    std::vector<HcclChannelDesc> needCreateDescs;
    std::vector<uint32_t> needCreateIndices;
    HcclResult ret
        = mgr.PrepareHandleArray("tag", COMM_ENGINE_CPU, &desc, 1, handleArray, needCreateDescs, needCreateIndices);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(needCreateDescs.size(), 0u);
    EXPECT_EQ(handleArray[0], (ChannelHandle)0x100);
}

TEST_F(ChannelManagerTest, PrepareHandleArray_ZeroMemHandleNumThenNonZero_NotReused)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclChannelDesc descNoMem{};
    memset(&descNoMem, 0, sizeof(descNoMem));
    descNoMem.remoteRank = 1;
    descNoMem.channelProtocol = COMM_PROTOCOL_HCCS;
    descNoMem.memHandles = nullptr;
    descNoMem.memHandleNum = 0;

    // 先用 memHandleNum=0 注册
    mgr.RegisterHandle("tag", COMM_ENGINE_CPU, descNoMem, (ChannelHandle)0x100);

    // 再用 memHandleNum=1 准备：key 不同（":0" vs ":handle"），不应命中
    HcclMemHandle handle = reinterpret_cast<HcclMemHandle>(0x1000);
    HcclChannelDesc descWithMem{};
    memset(&descWithMem, 0, sizeof(descWithMem));
    descWithMem.remoteRank = 1;
    descWithMem.channelProtocol = COMM_PROTOCOL_HCCS;
    descWithMem.memHandles = &handle;
    descWithMem.memHandleNum = 1;

    ChannelHandle handleArray[1] = {0};
    std::vector<HcclChannelDesc> needCreateDescs;
    std::vector<uint32_t> needCreateIndices;
    HcclResult ret = mgr.PrepareHandleArray(
        "tag", COMM_ENGINE_CPU, &descWithMem, 1, handleArray, needCreateDescs, needCreateIndices);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(needCreateDescs.size(), 1u);
    EXPECT_EQ(handleArray[0], (ChannelHandle)0);
}

TEST_F(ChannelManagerTest, PrepareHandleArray_MultipleMemHandles_AllCompared)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclMemHandle handlesA[2] = {reinterpret_cast<HcclMemHandle>(0x1000), reinterpret_cast<HcclMemHandle>(0x2000)};
    HcclChannelDesc descA{};
    memset(&descA, 0, sizeof(descA));
    descA.remoteRank = 1;
    descA.channelProtocol = COMM_PROTOCOL_HCCS;
    descA.memHandles = handlesA;
    descA.memHandleNum = 2;

    // 用 handlesA 注册
    mgr.RegisterHandle("tag", COMM_ENGINE_CPU, descA, (ChannelHandle)0x100);

    // 第二个 handle 不同，应不命中
    HcclMemHandle handlesB[2] = {reinterpret_cast<HcclMemHandle>(0x1000), reinterpret_cast<HcclMemHandle>(0x3000)};
    HcclChannelDesc descB{};
    memset(&descB, 0, sizeof(descB));
    descB.remoteRank = 1;
    descB.channelProtocol = COMM_PROTOCOL_HCCS;
    descB.memHandles = handlesB;
    descB.memHandleNum = 2;

    ChannelHandle handleArray[1] = {0};
    std::vector<HcclChannelDesc> needCreateDescs;
    std::vector<uint32_t> needCreateIndices;
    HcclResult ret
        = mgr.PrepareHandleArray("tag", COMM_ENGINE_CPU, &descB, 1, handleArray, needCreateDescs, needCreateIndices);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(needCreateDescs.size(), 1u);
    EXPECT_EQ(handleArray[0], (ChannelHandle)0);
}

TEST_F(ChannelManagerTest, PrepareHandleArray_MultipleMemHandles_AllSame_Reused)
{
    mgr.Init((aclrtBinHandle)0x1, 0, ManagerCallbacks{});
    HcclMemHandle handlesA[2] = {reinterpret_cast<HcclMemHandle>(0x1000), reinterpret_cast<HcclMemHandle>(0x2000)};
    HcclChannelDesc descA{};
    memset(&descA, 0, sizeof(descA));
    descA.remoteRank = 1;
    descA.channelProtocol = COMM_PROTOCOL_HCCS;
    descA.memHandles = handlesA;
    descA.memHandleNum = 2;

    mgr.RegisterHandle("tag", COMM_ENGINE_CPU, descA, (ChannelHandle)0x100);

    // 两个 handle 全部相同，应命中复用
    HcclMemHandle handlesB[2] = {reinterpret_cast<HcclMemHandle>(0x1000), reinterpret_cast<HcclMemHandle>(0x2000)};
    HcclChannelDesc descB{};
    memset(&descB, 0, sizeof(descB));
    descB.remoteRank = 1;
    descB.channelProtocol = COMM_PROTOCOL_HCCS;
    descB.memHandles = handlesB;
    descB.memHandleNum = 2;

    ChannelHandle handleArray[1] = {0};
    std::vector<HcclChannelDesc> needCreateDescs;
    std::vector<uint32_t> needCreateIndices;
    HcclResult ret
        = mgr.PrepareHandleArray("tag", COMM_ENGINE_CPU, &descB, 1, handleArray, needCreateDescs, needCreateIndices);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(needCreateDescs.size(), 0u);
    EXPECT_EQ(handleArray[0], (ChannelHandle)0x100);
}

// ============ CollectMemHandles (memHandles 去重收集) 相关测试 ============
// CollectMemHandles 从 descs 中收集去重后的 memHandles，用于建链时一次性透传。

TEST_F(ChannelManagerTest, CollectMemHandles_EmptyDescs)
{
    std::vector<HcclChannelDesc> descs;
    std::vector<HcclMemHandle> handles = mgr.CollectMemHandles(descs);
    EXPECT_TRUE(handles.empty());
}

TEST_F(ChannelManagerTest, CollectMemHandles_Deduplicate)
{
    HcclMemHandle handle = reinterpret_cast<HcclMemHandle>(0x1000);
    std::vector<HcclChannelDesc> descs(2);
    for (auto& d : descs) {
        memset(&d, 0, sizeof(d));
        d.memHandles = &handle;
        d.memHandleNum = 1;
    }
    std::vector<HcclMemHandle> handles = mgr.CollectMemHandles(descs);
    EXPECT_EQ(handles.size(), 1u);
    EXPECT_EQ(handles[0], handle);
}

TEST_F(ChannelManagerTest, CollectMemHandles_DistinctKept)
{
    HcclMemHandle h1 = reinterpret_cast<HcclMemHandle>(0x1000);
    HcclMemHandle h2 = reinterpret_cast<HcclMemHandle>(0x2000);
    std::vector<HcclChannelDesc> descs(2);
    memset(&descs[0], 0, sizeof(descs[0]));
    descs[0].memHandles = &h1;
    descs[0].memHandleNum = 1;
    memset(&descs[1], 0, sizeof(descs[1]));
    descs[1].memHandles = &h2;
    descs[1].memHandleNum = 1;

    std::vector<HcclMemHandle> handles = mgr.CollectMemHandles(descs);
    EXPECT_EQ(handles.size(), 2u);
    EXPECT_EQ(handles[0], h1);
    EXPECT_EQ(handles[1], h2);
}

TEST_F(ChannelManagerTest, CollectMemHandles_NullMemHandlesSkipped)
{
    std::vector<HcclChannelDesc> descs(2);
    memset(&descs[0], 0, sizeof(descs[0]));
    descs[0].memHandles = nullptr;
    descs[0].memHandleNum = 1;

    HcclMemHandle handle = reinterpret_cast<HcclMemHandle>(0x1000);
    memset(&descs[1], 0, sizeof(descs[1]));
    descs[1].memHandles = &handle;
    descs[1].memHandleNum = 1;

    std::vector<HcclMemHandle> handles = mgr.CollectMemHandles(descs);
    EXPECT_EQ(handles.size(), 1u);
    EXPECT_EQ(handles[0], handle);
}

TEST_F(ChannelManagerTest, CollectMemHandles_ZeroNumSkipped)
{
    HcclMemHandle handle = reinterpret_cast<HcclMemHandle>(0x1000);
    std::vector<HcclChannelDesc> descs(2);
    memset(&descs[0], 0, sizeof(descs[0]));
    descs[0].memHandles = &handle;
    descs[0].memHandleNum = 0;

    memset(&descs[1], 0, sizeof(descs[1]));
    descs[1].memHandles = &handle;
    descs[1].memHandleNum = 1;

    std::vector<HcclMemHandle> handles = mgr.CollectMemHandles(descs);
    EXPECT_EQ(handles.size(), 1u);
    EXPECT_EQ(handles[0], handle);
}

TEST_F(ChannelManagerTest, CollectMemHandles_MultipleHandlesOneDesc)
{
    HcclMemHandle handles[2] = {reinterpret_cast<HcclMemHandle>(0x1000), reinterpret_cast<HcclMemHandle>(0x2000)};
    std::vector<HcclChannelDesc> descs(1);
    memset(&descs[0], 0, sizeof(descs[0]));
    descs[0].memHandles = handles;
    descs[0].memHandleNum = 2;

    std::vector<HcclMemHandle> result = mgr.CollectMemHandles(descs);
    EXPECT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], handles[0]);
    EXPECT_EQ(result[1], handles[1]);
}
