/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../hccl_api_base_test.h"
#include "endpoint_pair.h"
using namespace hcomm;
class TestEndpointPair : public BaseInit {
public:
    void SetUp() override { BaseInit::SetUp(); }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
    Hccl::RankIpPortMapPtr rankIpPortMap;
};

TEST_F(TestEndpointPair, Ut_EndpointPair_Construct_Expect_HCCL_SUCCESS)
{
    EndpointDesc localEndpointDesc{};
    localEndpointDesc.protocol = COMM_PROTOCOL_ROCE;
    localEndpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    Hccl::IpAddress localIp("192.168.100.100");
    localEndpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
    localEndpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    EndpointDesc remoteEndpointDesc{};
    remoteEndpointDesc.protocol = COMM_PROTOCOL_ROCE;
    remoteEndpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    Hccl::IpAddress remoteIp("192.168.100.101");
    remoteEndpointDesc.commAddr.addr = remoteIp.GetBinaryAddress().addr;
    remoteEndpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;

    MOCKER_CPP(&SocketMgr::GetSocket).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    EndpointPair endpointPair(localEndpointDesc, remoteEndpointDesc, rankIpPortMap);
    HcclResult ret = endpointPair.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Hccl::Socket* socket = nullptr;
    ret = endpointPair.GetSocket(0, 1, "Hccl_Test_Group", 60001, 0, socket, 0, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = endpointPair.GetSocket(1, 0, "Hccl_Test_Group", 60001, 0, socket, 0, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestEndpointPair, Ut_EndpointPair_Create_Host_Socket_Expect_HCCL_SUCCESS)
{
    EndpointDesc localEndpointDesc{};
    localEndpointDesc.protocol = COMM_PROTOCOL_ROCE;
    localEndpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    Hccl::IpAddress localIp("192.168.100.100");
    localEndpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
    localEndpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    EndpointDesc remoteEndpointDesc{};
    remoteEndpointDesc.protocol = COMM_PROTOCOL_ROCE;
    remoteEndpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    Hccl::IpAddress remoteIp("192.168.100.101");
    remoteEndpointDesc.commAddr.addr = remoteIp.GetBinaryAddress().addr;
    remoteEndpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;

    MOCKER_CPP(&SocketMgr::GetSocket).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    EndpointPair endpointPair(localEndpointDesc, remoteEndpointDesc, rankIpPortMap);
    HcclResult ret = endpointPair.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Hccl::Socket* socket = nullptr;
    ret = endpointPair.ServerInit(0, 1, "Hccl_Test_Group", 0, 0, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = endpointPair.GetConnectedSocket(0, 1, "Hccl_Test_Group", 0, 60001, socket, 0, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试销毁不存在的channel，返回HCCL_SUCCESS
TEST_F(TestEndpointPair, Ut_DestroyChannel_When_Channel_Not_Exist_Expect_SUCCESS)
{
    MOCKER(HcommChannelDestroy).stubs().will(returnValue(static_cast<int>(HCCL_SUCCESS)));

    EndpointDesc localEndpointDesc{};
    localEndpointDesc.protocol = COMM_PROTOCOL_UB_CTP;
    localEndpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    Hccl::IpAddress localIp("192.168.100.100");
    localEndpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
    localEndpointDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    EndpointDesc remoteEndpointDesc{};
    remoteEndpointDesc.protocol = COMM_PROTOCOL_UB_CTP;
    remoteEndpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    Hccl::IpAddress remoteIp("192.168.100.101");
    remoteEndpointDesc.commAddr.addr = remoteIp.GetBinaryAddress().addr;
    remoteEndpointDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    EndpointPair endpointPair(localEndpointDesc, remoteEndpointDesc, rankIpPortMap);
    HcclResult ret = endpointPair.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(endpointPair.channelHandles_.size(), 0);

    ret = endpointPair.DestroyChannel(COMM_ENGINE_CCU, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试SocketConfig构造函数 connectMode=0 (hostNic2DeviceNicMode=0)
// 场景：角色由IP比较决定(100<101为true->SERVER)，tag需要拼接port
TEST_F(TestEndpointPair, Ut_SocketConfig_ConnectMode_0_Expect_Success)
{
    Hccl::IpAddress localIp("192.168.100.100");
    Hccl::IpAddress remoteIp("192.168.100.101");

    Hccl::LinkData link(Hccl::PortDeploymentType::P2P, Hccl::LinkProtocol::ROCE, 0, 1, localIp, remoteIp, 0, 0, 0);

    uint32_t listenPort = 12345;
    std::string tag = "test_tag";
    uint32_t hostNic2DeviceNicMode = 0;
    uint32_t myRank = 0;
    uint32_t rmtRank = 1;

    Hccl::SocketConfig socketConfig(link, listenPort, tag, hostNic2DeviceNicMode, myRank, rmtRank);

    EXPECT_EQ(socketConfig.remoteRank, rmtRank);
    EXPECT_EQ(socketConfig.listeningPort, listenPort);
    EXPECT_EQ(socketConfig.GetRole(), Hccl::SocketRole::SERVER);

    std::string expectedTag
        = tag + "_" + localIp.GetIpStr() + "_" + remoteIp.GetIpStr() + "_" + std::to_string(listenPort);
    EXPECT_EQ(socketConfig.GetHccpTag(), expectedTag);
}

// 测试SocketConfig构造函数 connectMode=1 (hostNic2DeviceNicMode=1)
// 场景：角色由rank比较决定(1<0为false->CLIENT)，tag不需要拼接port
TEST_F(TestEndpointPair, Ut_SocketConfig_ConnectMode_1_Expect_Success)
{
    Hccl::IpAddress localIp("192.168.100.100");
    Hccl::IpAddress remoteIp("192.168.100.101");

    Hccl::LinkData link(Hccl::PortDeploymentType::P2P, Hccl::LinkProtocol::ROCE, 1, 0, localIp, remoteIp, 0, 0, 0);

    uint32_t listenPort = 54321;
    std::string tag = "hccp_test";
    uint32_t hostNic2DeviceNicMode = 1;
    uint32_t myRank = 1;
    uint32_t rmtRank = 0;

    Hccl::SocketConfig socketConfig(link, listenPort, tag, hostNic2DeviceNicMode, myRank, rmtRank);

    EXPECT_EQ(socketConfig.remoteRank, rmtRank);
    EXPECT_EQ(socketConfig.listeningPort, listenPort);
    EXPECT_EQ(socketConfig.hostNic2DeviceNicMode_, hostNic2DeviceNicMode);
    EXPECT_EQ(socketConfig.GetRole(), Hccl::SocketRole::CLIENT);

    std::string expectedTag = tag + "_" + std::to_string(rmtRank) + "_" + std::to_string(myRank) + "_"
                              + remoteIp.GetIpStr() + "_" + localIp.GetIpStr();
    EXPECT_EQ(socketConfig.GetHccpTag(), expectedTag);
}

// 测试SocketConfig构造函数 hostNic2DeviceNicMode=1 从socketTag解析commTag
// 场景：socketTag格式为"commTag_engine_X"，解析后commTag应为"commTag"
TEST_F(TestEndpointPair, Ut_SocketConfig_CommTagParse_FromSocketTag_Expect_Success)
{
    Hccl::IpAddress localIp("192.168.100.100");
    Hccl::IpAddress remoteIp("192.168.100.101");

    Hccl::LinkData link(Hccl::PortDeploymentType::P2P, Hccl::LinkProtocol::ROCE, 0, 1, localIp, remoteIp, 0, 0, 0);

    uint32_t listenPort = 54321;
    std::string socketTag = "testCommTag_engine_5"; // socketTag格式: commTag_engine_X
    uint32_t hostNic2DeviceNicMode = 1;
    uint32_t myRank = 0;
    uint32_t rmtRank = 1;

    Hccl::SocketConfig socketConfig(link, listenPort, socketTag, hostNic2DeviceNicMode, myRank, rmtRank);

    EXPECT_EQ(socketConfig.remoteRank, rmtRank);
    EXPECT_EQ(socketConfig.hostNic2DeviceNicMode_, hostNic2DeviceNicMode);
    EXPECT_EQ(socketConfig.GetRole(), Hccl::SocketRole::SERVER);

    // 解析出的commTag应为"testCommTag"，hccpTag格式: commTag_rankId_rankId_ip_ip
    std::string expectedCommTag = "testCommTag";
    std::string expectedTag = expectedCommTag + "_" + std::to_string(myRank) + "_" + std::to_string(rmtRank) + "_"
                              + localIp.GetIpStr() + "_" + remoteIp.GetIpStr();
    EXPECT_EQ(socketConfig.GetHccpTag(), expectedTag);
}

// 测试SocketConfig构造函数 hostNic2DeviceNicMode=1 从socketTag解析commTag (带protocol后缀)
// 场景：socketTag格式为"commTag_engine_X_protocol_Y"，解析后commTag应为"commTag"
TEST_F(TestEndpointPair, Ut_SocketConfig_CommTagParse_FromSocketTagWithProtocol_Expect_Success)
{
    Hccl::IpAddress localIp("192.168.100.100");
    Hccl::IpAddress remoteIp("192.168.100.101");

    Hccl::LinkData link(Hccl::PortDeploymentType::P2P, Hccl::LinkProtocol::ROCE, 1, 0, localIp, remoteIp, 0, 0, 0);

    uint32_t listenPort = 54321;
    std::string socketTag = "myCommTag_engine_2_protocol_3"; // socketTag格式: commTag_engine_X_protocol_Y
    uint32_t hostNic2DeviceNicMode = 1;
    uint32_t myRank = 1;
    uint32_t rmtRank = 0;

    Hccl::SocketConfig socketConfig(link, listenPort, socketTag, hostNic2DeviceNicMode, myRank, rmtRank);

    EXPECT_EQ(socketConfig.remoteRank, rmtRank);
    EXPECT_EQ(socketConfig.hostNic2DeviceNicMode_, hostNic2DeviceNicMode);
    EXPECT_EQ(socketConfig.GetRole(), Hccl::SocketRole::CLIENT);

    // 解析出的commTag应为"myCommTag"，hccpTag格式: commTag_rankId_rankId_ip_ip
    std::string expectedCommTag = "myCommTag";
    std::string expectedTag = expectedCommTag + "_" + std::to_string(rmtRank) + "_" + std::to_string(myRank) + "_"
                              + remoteIp.GetIpStr() + "_" + localIp.GetIpStr();
    EXPECT_EQ(socketConfig.GetHccpTag(), expectedTag);
}

// 测试SocketConfig构造函数 hostNic2DeviceNicMode=1 异常场景：socketTag无_engine_后缀
// 场景：socketTag不包含"_engine_"模式，记录WARNING日志，使用原tag作为commTag（非法场景）
TEST_F(TestEndpointPair, Ut_SocketConfig_CommTagParse_NoEngineSuffix_Expect_Warning)
{
    Hccl::IpAddress localIp("192.168.100.100");
    Hccl::IpAddress remoteIp("192.168.100.101");

    Hccl::LinkData link(Hccl::PortDeploymentType::P2P, Hccl::LinkProtocol::ROCE, 0, 1, localIp, remoteIp, 0, 0, 0);

    uint32_t listenPort = 54321;
    std::string socketTag = "plainTag"; // 不包含"_engine_"后缀
    uint32_t hostNic2DeviceNicMode = 1;
    uint32_t myRank = 0;
    uint32_t rmtRank = 1;

    Hccl::SocketConfig socketConfig(link, listenPort, socketTag, hostNic2DeviceNicMode, myRank, rmtRank);

    EXPECT_EQ(socketConfig.remoteRank, rmtRank);
    EXPECT_EQ(socketConfig.hostNic2DeviceNicMode_, hostNic2DeviceNicMode);
    EXPECT_EQ(socketConfig.GetRole(), Hccl::SocketRole::SERVER);

    // 无_engine_后缀时，commTag保持原样
    std::string expectedTag = socketTag + "_" + std::to_string(myRank) + "_" + std::to_string(rmtRank) + "_"
                              + localIp.GetIpStr() + "_" + remoteIp.GetIpStr();
    EXPECT_EQ(socketConfig.GetHccpTag(), expectedTag);
}

// 测试SocketConfig构造函数 hostNic2DeviceNicMode=1 异常场景：socketTag包含多个"_engine_"
// 场景：socketTag格式为"commTag_engine_1_engine_2"，解析后commTag应为"commTag"
TEST_F(TestEndpointPair, Ut_SocketConfig_CommTagParse_MultiEngine_Expect_FirstEngineOnly)
{
    Hccl::IpAddress localIp("192.168.100.100");
    Hccl::IpAddress remoteIp("192.168.100.101");

    Hccl::LinkData link(Hccl::PortDeploymentType::P2P, Hccl::LinkProtocol::ROCE, 0, 1, localIp, remoteIp, 0, 0, 0);

    uint32_t listenPort = 54321;
    std::string socketTag = "commTag_engine_1_engine_2"; // 多个_engine_，取第一个之前的部分
    uint32_t hostNic2DeviceNicMode = 1;
    uint32_t myRank = 0;
    uint32_t rmtRank = 1;

    Hccl::SocketConfig socketConfig(link, listenPort, socketTag, hostNic2DeviceNicMode, myRank, rmtRank);

    EXPECT_EQ(socketConfig.GetRole(), Hccl::SocketRole::SERVER);

    // 解析出的commTag应为"commTag"（第一个"_engine_"之前的部分）
    std::string expectedCommTag = "commTag";
    std::string expectedTag = expectedCommTag + "_" + std::to_string(myRank) + "_" + std::to_string(rmtRank) + "_"
                              + localIp.GetIpStr() + "_" + remoteIp.GetIpStr();
    EXPECT_EQ(socketConfig.GetHccpTag(), expectedTag);
}

// 测试SocketConfig构造函数 hostNic2DeviceNicMode=1 异常场景：socketTag以"_engine_"开头
// 场景：socketTag格式为"_engine_5"，_engine_在位置0，substr(0,0)返回空字符串
// 实际行为：commTag为空字符串，这是非法场景
TEST_F(TestEndpointPair, Ut_SocketConfig_CommTagParse_EnginePrefix_Expect_EmptyCommTag)
{
    Hccl::IpAddress localIp("192.168.100.100");
    Hccl::IpAddress remoteIp("192.168.100.101");

    Hccl::LinkData link(Hccl::PortDeploymentType::P2P, Hccl::LinkProtocol::ROCE, 0, 1, localIp, remoteIp, 0, 0, 0);

    uint32_t listenPort = 54321;
    std::string socketTag = "_engine_5"; // 以_engine_开头，_engine_在位置0
    uint32_t hostNic2DeviceNicMode = 1;
    uint32_t myRank = 0;
    uint32_t rmtRank = 1;

    Hccl::SocketConfig socketConfig(link, listenPort, socketTag, hostNic2DeviceNicMode, myRank, rmtRank);

    EXPECT_EQ(socketConfig.GetRole(), Hccl::SocketRole::SERVER);

    // 以_engine_开头时，commTag为空字符串（substr(0,0)返回空）
    std::string expectedTag = "_" + std::to_string(myRank) + "_" + std::to_string(rmtRank) + "_" + localIp.GetIpStr()
                              + "_" + remoteIp.GetIpStr();
    EXPECT_EQ(socketConfig.GetHccpTag(), expectedTag);
}

// 测试SocketConfig构造函数 hostNic2DeviceNicMode=1 异常场景：解析出的commTag不符合预期
// 场景：socketTag为"test_engine_abc"，按数字解析逻辑会得到错误结果
TEST_F(TestEndpointPair, Ut_SocketConfig_CommTagParse_InvalidFormat_Expect_WrongParse)
{
    Hccl::IpAddress localIp("192.168.100.100");
    Hccl::IpAddress remoteIp("192.168.100.101");

    Hccl::LinkData link(Hccl::PortDeploymentType::P2P, Hccl::LinkProtocol::ROCE, 0, 1, localIp, remoteIp, 0, 0, 0);

    uint32_t listenPort = 54321;
    std::string socketTag = "test_engine_abc"; // engine后不是数字
    uint32_t hostNic2DeviceNicMode = 1;
    uint32_t myRank = 0;
    uint32_t rmtRank = 1;

    Hccl::SocketConfig socketConfig(link, listenPort, socketTag, hostNic2DeviceNicMode, myRank, rmtRank);

    EXPECT_EQ(socketConfig.GetRole(), Hccl::SocketRole::SERVER);

    // 解析出的commTag为"test"（_engine_之前的部分）
    // 但实际期望的commTag可能是完整的"test_engine_abc"
    // 这里验证实际解析结果：commTag应该是"test"
    std::string actualCommTag = "test";
    std::string wrongExpectedTag = socketTag + "_" + std::to_string(myRank) + "_" + std::to_string(rmtRank) + "_"
                                   + localIp.GetIpStr() + "_" + remoteIp.GetIpStr();
    // 验证解析出的commTag确实不是完整的socketTag
    EXPECT_STRNE(socketTag.c_str(), actualCommTag.c_str());
    // 但hccpTag中用的是解析后的commTag
    std::string expectedTag = actualCommTag + "_" + std::to_string(myRank) + "_" + std::to_string(rmtRank) + "_"
                              + localIp.GetIpStr() + "_" + remoteIp.GetIpStr();
    EXPECT_EQ(socketConfig.GetHccpTag(), expectedTag);
}

// 测试SocketConfig构造函数 hostNic2DeviceNicMode=1 空字符串socketTag
// 场景：socketTag为空字符串
TEST_F(TestEndpointPair, Ut_SocketConfig_CommTagParse_EmptyTag_Expect_Success)
{
    Hccl::IpAddress localIp("192.168.100.100");
    Hccl::IpAddress remoteIp("192.168.100.101");

    Hccl::LinkData link(Hccl::PortDeploymentType::P2P, Hccl::LinkProtocol::ROCE, 0, 1, localIp, remoteIp, 0, 0, 0);

    uint32_t listenPort = 54321;
    std::string socketTag = ""; // 空字符串
    uint32_t hostNic2DeviceNicMode = 1;
    uint32_t myRank = 0;
    uint32_t rmtRank = 1;

    Hccl::SocketConfig socketConfig(link, listenPort, socketTag, hostNic2DeviceNicMode, myRank, rmtRank);

    EXPECT_EQ(socketConfig.GetRole(), Hccl::SocketRole::SERVER);

    // 空字符串时，commTag保持为空
    std::string expectedTag = "_" + std::to_string(myRank) + "_" + std::to_string(rmtRank) + "_" + localIp.GetIpStr()
                              + "_" + remoteIp.GetIpStr();
    EXPECT_EQ(socketConfig.GetHccpTag(), expectedTag);
}

// 测试SocketConfig构造函数 hostNic2DeviceNicMode=1 复杂commTag带下划线
// 场景：socketTag格式为"comm_tag_with_underscore_engine_3"
TEST_F(TestEndpointPair, Ut_SocketConfig_CommTagParse_ComplexCommTag_Expect_Success)
{
    Hccl::IpAddress localIp("192.168.100.100");
    Hccl::IpAddress remoteIp("192.168.100.101");

    Hccl::LinkData link(Hccl::PortDeploymentType::P2P, Hccl::LinkProtocol::ROCE, 0, 1, localIp, remoteIp, 0, 0, 0);

    uint32_t listenPort = 54321;
    std::string socketTag = "comm_tag_with_underscore_engine_3"; // commTag带下划线
    uint32_t hostNic2DeviceNicMode = 1;
    uint32_t myRank = 0;
    uint32_t rmtRank = 1;

    Hccl::SocketConfig socketConfig(link, listenPort, socketTag, hostNic2DeviceNicMode, myRank, rmtRank);

    EXPECT_EQ(socketConfig.GetRole(), Hccl::SocketRole::SERVER);

    // 解析出的commTag应为"comm_tag_with_underscore"
    std::string expectedCommTag = "comm_tag_with_underscore";
    std::string expectedTag = expectedCommTag + "_" + std::to_string(myRank) + "_" + std::to_string(rmtRank) + "_"
                              + localIp.GetIpStr() + "_" + remoteIp.GetIpStr();
    EXPECT_EQ(socketConfig.GetHccpTag(), expectedTag);
}

TEST_F(TestEndpointPair, Ut_DestroyChannel_When_Channel_Exist_Expect_SUCCESS)
{
    MOCKER(HcommChannelDestroy).stubs().will(returnValue(static_cast<int>(HCCL_SUCCESS)));

    EndpointDesc localEndpointDesc{};
    localEndpointDesc.protocol = COMM_PROTOCOL_UB_CTP;
    localEndpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    Hccl::IpAddress localIp("192.168.100.100");
    localEndpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
    localEndpointDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    EndpointDesc remoteEndpointDesc{};
    remoteEndpointDesc.protocol = COMM_PROTOCOL_UB_CTP;
    remoteEndpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    Hccl::IpAddress remoteIp("192.168.100.101");
    remoteEndpointDesc.commAddr.addr = remoteIp.GetBinaryAddress().addr;
    remoteEndpointDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    EndpointPair endpointPair(localEndpointDesc, remoteEndpointDesc, rankIpPortMap);
    HcclResult ret = endpointPair.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // channelHandles_[COMM_ENGINE_CCU]中添加三个不同handle
    ChannelHandle fakeChannelHandle = 100;
    endpointPair.channelHandles_[COMM_ENGINE_CCU].push_back(fakeChannelHandle);
    endpointPair.channelHandles_[COMM_ENGINE_CCU].push_back(fakeChannelHandle + 1);
    endpointPair.channelHandles_[COMM_ENGINE_CCU].push_back(fakeChannelHandle + 2);

    // 删除index为0的channel
    ret = endpointPair.DestroyChannel(COMM_ENGINE_CCU, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(endpointPair.channelHandles_[COMM_ENGINE_CCU].size(), 2);
    EXPECT_EQ(endpointPair.channelHandles_[COMM_ENGINE_CCU][0], fakeChannelHandle + 1);
    EXPECT_EQ(endpointPair.channelHandles_[COMM_ENGINE_CCU][1], fakeChannelHandle + 2);
}
