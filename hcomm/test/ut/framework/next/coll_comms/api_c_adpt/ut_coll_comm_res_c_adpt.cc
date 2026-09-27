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
#include "../../hccl_api_base_test.h"
#include "hccl/hccl_res.h"
#include "hccl_common.h"
#include "config/env_config.h"
#include "independent_op_context_manager.h"
#include "log.h"
#include "hccl_comm_pub.h"
#include "independent_op.h"
#include "llt_hccl_stub_rank_graph.h"
#include <cstring>
#include <string>
#include "mockcpp/mockcpp.hpp"
#include "dfx/cluster_monitor/cluster_monitor.h"
#include "host/host_cpu_roce_channel.h"
#include "param_check_pub.h"
#include "hccl/hccl_types.h"
#include "hccp.h"
#include "my_rank.h"
#include "roce_channel_desc_configurator.h"
#include "aiv_urma_channel.h"
#include "channel_process.h"
#include "coll_comm_res_c_adpt.h"
#include "hcomm_c_adpt.h"
#include "channel_config.h"
#include "hccl/hccl_channel.h"
#include "shared_jetty_channel_pool.h"

#define private public

using namespace hccl;
using namespace hcomm;

HcclResult ProcessUbChannelDesc(
    const HcclChannelDesc& channelDesc, const HcclChannelDesc& channelDescFinal, const hcclComm* hcclComm);
HcclResult
ProcessHcclResPackReq(const HcclChannelDesc& channelDesc, HcclChannelDesc& channelDescFinal, hcclComm* hcclComm);

static int StubRaGetHccnCfgRoceQosDscp(struct RaInfo* info, enum HccnCfgKey key, char* value, unsigned int* valueLen)
{
    (void)info;
    (void)key;
    if (value == nullptr || valueLen == nullptr) {
        return -1;
    }
    const char* cfg = "0:33,1:20,4:50,6:70";
    const unsigned int len = static_cast<unsigned int>(std::strlen(cfg));
    if (*valueLen < len) {
        return -1;
    }
    (void)std::memcpy(value, cfg, len);
    *valueLen = len;
    return 0;
}

namespace {
// UT stub state is only used by the single-threaded ResolveQueueNum test.
s32 g_hostConfigDeviceLogicId = 0;
u32 g_hostConfigDevicePhyId = 0;
HcclResult g_getDeviceResult = HCCL_SUCCESS;
HcclResult g_getDevicePhyIdResult = HCCL_SUCCESS;
std::string g_hostMultiQpMode;
std::string g_hostMultiQpCount;
std::vector<HccnCfgKey> g_hostConfigReadKeys;
u32 g_getDeviceCallCount = 0;
u32 g_getDevicePhyIdCallCount = 0;

void ResetHostMultiQpCountStub()
{
    g_hostConfigDeviceLogicId = 6;
    g_hostConfigDevicePhyId = 3;
    g_getDeviceResult = HCCL_SUCCESS;
    g_getDevicePhyIdResult = HCCL_SUCCESS;
    g_hostMultiQpMode = "multi_qp";
    g_hostMultiQpCount = "4";
    g_hostConfigReadKeys.clear();
    g_getDeviceCallCount = 0;
    g_getDevicePhyIdCallCount = 0;
}

HcclResult StubGetDeviceForHostMultiQp(s32* deviceLogicId)
{
    ++g_getDeviceCallCount;
    if (g_getDeviceResult == HCCL_SUCCESS && deviceLogicId != nullptr) {
        *deviceLogicId = g_hostConfigDeviceLogicId;
    }
    return g_getDeviceResult;
}

HcclResult StubGetDevicePhyIdForHostMultiQp(u32 deviceLogicId, u32& devicePhyId, bool isRefresh)
{
    ++g_getDevicePhyIdCallCount;
    EXPECT_EQ(deviceLogicId, static_cast<u32>(g_hostConfigDeviceLogicId));
    EXPECT_FALSE(isRefresh);
    if (g_getDevicePhyIdResult == HCCL_SUCCESS) {
        devicePhyId = g_hostConfigDevicePhyId;
    }
    return g_getDevicePhyIdResult;
}

int StubRaGetHostMultiQpCount(RaInfo* info, HccnCfgKey key, char* value, unsigned int* valueLen)
{
    if (info == nullptr || value == nullptr || valueLen == nullptr) {
        return -1;
    }
    EXPECT_EQ(info->mode, NETWORK_PEER_ONLINE);
    EXPECT_EQ(info->phyId, g_hostConfigDevicePhyId);
    EXPECT_EQ(*valueLen, RoceChannelDescConfigurator::HOST_NIC_CONFIG_BUFFER_SIZE);
    g_hostConfigReadKeys.emplace_back(key);

    const std::string* configValue = nullptr;
    if (key == HCCN_CFG_UDP_PORT_MODE) {
        configValue = &g_hostMultiQpMode;
    } else if (key == HCCN_CFG_MULTI_QP_COUNT) {
        configValue = &g_hostMultiQpCount;
    }
    if (configValue == nullptr) {
        *valueLen = 0;
        return 0;
    }
    if (configValue->empty()) {
        *valueLen = 0;
        return 0;
    }
    const std::size_t requiredValueLen = configValue->size() + 1U;
    if (*valueLen < requiredValueLen) {
        return -1;
    }
    std::copy(configValue->begin(), configValue->end(), value);
    value[configValue->size()] = '\0';
    *valueLen = static_cast<unsigned int>(requiredValueLen);
    return 0;
}
} // namespace

static HcclMemHandle g_userMemHandle = reinterpret_cast<HcclMemHandle>(0x1111);
static HcclMemHandle g_symMemHandle = reinterpret_cast<HcclMemHandle>(0x2222);
static const std::string g_symMemTag = "__hccl_sym_win__ut";
static ChannelHandle g_testChannel = static_cast<ChannelHandle>(0x3333);
static void* g_testDevChannelEntity = reinterpret_cast<void*>(0x5678);
// UT stub state, used in single-threaded test execution and reset in SetUp.
static HcclChannelDesc g_capturedChannelDesc{};
static std::vector<HcclMemHandle> g_capturedMemHandles;
static bool g_remoteHasSymMemTag = false;

void StubSetRemoteTags(const std::vector<std::string>& remoteMemTags)
{
    g_remoteHasSymMemTag = false;
    for (const std::string& memTag : remoteMemTags) {
        if (memTag == g_symMemTag) {
            g_remoteHasSymMemTag = true;
            break;
        }
    }
}

void StubFillSymHandles(std::vector<HcclMemHandle>& memHandles)
{
    memHandles.clear();
    if (!g_remoteHasSymMemTag) {
        memHandles.emplace_back(g_symMemHandle);
    }
}

void MockRemoteMems(uint32_t& remoteMemNum, CommMem*& remoteMems, const std::vector<std::string>& remoteMemTags)
{
    MOCKER_CPP(
        &MyRank::ChannelGetRemoteMems,
        HcclResult(MyRank::*)(ChannelHandle, uint32_t*, CommMem**, std::vector<std::string>&) const)
        .expects(once())
        .with(mockcpp::any(), outBoundP(&remoteMemNum), outBoundP(&remoteMems), outBound(remoteMemTags))
        .will(returnValue(HCCL_SUCCESS));
}

HcclResult StubQueryExistingChannels(
    MyRank* self, CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum, ChannelHandle* channels)
{
    (void)self;
    (void)engine;
    (void)channelDescs;
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        channels[idx] = g_testChannel;
    }
    return HCCL_SUCCESS;
}

HcclResult StubCreateChannelsCapture(
    MyRank* self, CommEngine engine, const std::string& commTag, const HcclChannelDesc* channelDescs,
    uint32_t channelNum, ChannelHandle* channels)
{
    (void)self;
    (void)engine;
    (void)commTag;
    if (channelNum > 0) {
        g_capturedChannelDesc = channelDescs[0];
        g_capturedMemHandles.clear();
        for (uint32_t idx = 0; idx < channelDescs[0].memHandleNum; ++idx) {
            g_capturedMemHandles.emplace_back(channelDescs[0].memHandles[idx]);
        }
        channels[0] = g_testChannel;
    }
    return HCCL_SUCCESS;
}

HcclResult StubChannelGet(ChannelHandle channelHandle, void** channel)
{
    EXPECT_EQ(channelHandle, g_testChannel);
    *channel = reinterpret_cast<void*>(g_testChannel);
    return HCCL_SUCCESS;
}

HcclResult StubBuildChannelEntityToDevice(AivUrmaChannel* self, void** devChannelEntity)
{
    EXPECT_EQ(reinterpret_cast<ChannelHandle>(self), g_testChannel);
    *devChannelEntity = g_testDevChannelEntity;
    return HCCL_SUCCESS;
}

HcclResult StubChannelGetRemoteMems(ChannelHandle channel, uint32_t* memNum, CommMem** remoteMems, char*** memTags)
{
    static CommMem remoteMem{};
    static char memTag[] = "__hccl_sym_win__ut";
    static char* tagList[] = {memTag};
    remoteMem.type = COMM_MEM_TYPE_DEVICE;
    remoteMem.addr = reinterpret_cast<void*>(0x4444);
    remoteMem.size = 0x1000;
    EXPECT_EQ(channel, g_testChannel);
    *memNum = 1;
    *remoteMems = &remoteMem;
    *memTags = tagList;
    return HCCL_SUCCESS;
}

HcclResult StubUpdateSymmetricRemoteMem(uint32_t remoteRank, const CommMem* remoteMems, char** memTags, uint32_t memNum)
{
    (void)remoteRank;
    (void)remoteMems;
    (void)memTags;
    (void)memNum;
    return HCCL_SUCCESS;
}

static uint32_t g_sharedJettyGetStatusCallCount = 0;
static int32_t StubHcommChannelGetStatus(const ChannelHandle* channelList, uint32_t listNum, int32_t* statusList)
{
    (void)channelList;
    g_sharedJettyGetStatusCallCount++;
    if (g_sharedJettyGetStatusCallCount == 1) {
        for (uint32_t i = 0; i < listNum; ++i) {
            statusList[i] = static_cast<int32_t>(HCOMM_CHANNEL_STATUS_CONNECTING);
        }
        return static_cast<int32_t>(HCCL_E_AGAIN);
    }
    for (uint32_t i = 0; i < listNum; ++i) {
        statusList[i] = static_cast<int32_t>(HCOMM_CHANNEL_STATUS_READY);
    }
    return 0;
}

class HcclChannelDescTest : public testing::Test {
public:
    void SetUp() override
    {
        g_capturedChannelDesc = {};
        g_capturedMemHandles.clear();
        g_remoteHasSymMemTag = false;
        g_testChannel = static_cast<ChannelHandle>(0x3333);
        const char* fakeA5SocName = "Ascend950PR_958b";
        MOCKER(aclrtGetSocName).stubs().will(returnValue(fakeA5SocName));
        MOCKER(&HcclCommDfx::ReportKernel).stubs().will(returnValue(HCCL_SUCCESS));
        SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
    void TearDown() override { GlobalMockObject::verify(); }

protected:
    void SetUpCommAndGraph(
        std::shared_ptr<hccl::hcclComm>& hcclCommPtr, std::shared_ptr<Hccl::RankGraph>& rankGraphV2, void*& comm,
        HcclResult& ret)
    {
        MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

        bool isDeviceSide{false};
        MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
        MOCKER(IsSupportHCCLV2).stubs().will(returnValue(true));
        setenv("HCCL_INDEPENDENT_OP", "1", 1);
        setenv("HCCL_RDMA_RETRY_CNT", "7", 1);
        setenv("HCCL_RDMA_TIMEOUT", "20", 1);
        setenv("HCCL_RDMA_TC", "120", 1);
        setenv("HCCL_RDMA_SL", "2", 1);
        setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
        RankGraphStub rankGraphStub;
        rankGraphV2 = rankGraphStub.Create2PGraph();
        void* commV2 = (void*)0x2000;
        uint32_t rank = 1;
        HcclMem cclBuffer;
        cclBuffer.size = 1;
        cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
        cclBuffer.addr = (void*)0x1000;
        char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
        hcclCommPtr = std::make_shared<hccl::hcclComm>(1, 1, commName);
        HcclCommConfig config;
        UtInitHcclCommConfig(config);
        config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
        config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
        config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
        unsetenv("HCCL_DFS_CONFIG");
        ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
        CollComm* collComm = hcclCommPtr->GetCollComm();
        comm = static_cast<HcclComm>(hcclCommPtr.get());
    }

    void GetChannelDesc(std::vector<HcclChannelDesc>& channelDesc)
    {
        HcclChannelDescInit(channelDesc.data(), 1);
        channelDesc[0].remoteRank = 2;
        channelDesc[0].channelProtocol = CommProtocol::COMM_PROTOCOL_ROCE;
        channelDesc[0].notifyNum = 50;
        channelDesc[0].roceAttr.queueNum = 3;
        channelDesc[0].roceAttr.retryCnt = 3;
        channelDesc[0].roceAttr.retryInterval = 20;
        channelDesc[0].roceAttr.tc = 120;
        channelDesc[0].roceAttr.sl = 3;
    }

    void InitRoceChannelDesc(HcclChannelDesc& channelDesc) const
    {
        ASSERT_EQ(HcclChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
        channelDesc.channelProtocol = COMM_PROTOCOL_ROCE;
        channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE; // CheckRoceAttr 以 remoteEndpoint.protocol 判定
        channelDesc.roceAttr.queueNum = 3;
        channelDesc.roceAttr.retryCnt = 3;
        channelDesc.roceAttr.retryInterval = 20;
        channelDesc.localEndpoint.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        channelDesc.localEndpoint.loc.device.devPhyId = 0U;
    }

    void InitSymMemChannelDesc(HcclChannelDesc& channelDesc) const
    {
        ASSERT_EQ(HcclChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
        channelDesc.remoteRank = 2;
        channelDesc.channelProtocol = COMM_PROTOCOL_UBOE;
        channelDesc.notifyNum = 1;
        channelDesc.memHandles = &g_userMemHandle;
        channelDesc.memHandleNum = 1;
    }

    void MockConsumedPendingSymMem()
    {
        MOCKER(&hcomm::ClusterMonitor::RegisterToClusterMonitor).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&CollComm::RegisterPendingSymmetricMemHandles).expects(once()).will(returnValue(HCCL_SUCCESS));
    }

    void ExpectRoceSlTcInHcommChannelDesc(uint32_t hcclQos, uint8_t expectSl, uint8_t expectTc)
    {
        // ApplyRoceQosCompatToSlTc：hrtGetDevice(userDevId) → aclrtGetPhyDevIdByUserDevId → 查 HCCN
        s32 userDevId = 0;
        s32 phyDevId = 0;
        MOCKER(hrtGetDevice).stubs().with(outBoundP(&userDevId)).will(returnValue(HCCL_SUCCESS));
        MOCKER(aclrtGetPhyDevIdByUserDevId)
            .stubs()
            .with(mockcpp::any(), outBoundP(&phyDevId))
            .will(returnValue(ACL_SUCCESS));

        hcclCommPtr->GetCollComm()->GetCommConfig().SetConfigHcclQos(hcclQos);

        HcclChannelDesc in{};
        HcclChannelDesc out{};
        InitRoceChannelDesc(in);
        ASSERT_EQ(HcclChannelDescInit(&out, 1), HCCL_SUCCESS);

        ret = ProcessHcclResPackReq(in, out, hcclCommPtr.get());
        ASSERT_EQ(ret, HCCL_SUCCESS);

        HcommChannelDesc hcommDesc
            = MyRankUtils::ChannelDescHccl2Hcomm(out, hcclCommPtr->GetCollComm()->GetCommConfig());
        ASSERT_EQ(CheckRoceAttr(hcommDesc, ENDPOINT_LOC_TYPE_HOST), HCCL_SUCCESS);
        EXPECT_EQ(hcommDesc.roceAttr.sl, expectSl) << "hcclQos=" << hcclQos;
        EXPECT_EQ(hcommDesc.roceAttr.tc, expectTc) << "hcclQos=" << hcclQos;
    }

protected:
    // 共享 jetty 用例公共逻辑：仅协议和 tag 不同，提取消除重复代码
    void DoSharedJettyAcquireTest(CommProtocol protocol, const char* tag)
    {
        HcclChannelConfig config = nullptr;
        ASSERT_EQ(HcclChannelConfigCreate(&config), HCCL_SUCCESS);
        ASSERT_EQ(HcclChannelConfigSetInt(config, HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1), HCCL_SUCCESS);
        ASSERT_EQ(HcclChannelConfigSetStr(config, HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG, tag), HCCL_SUCCESS);

        std::vector<HcclChannelDesc> channelDesc(1);
        std::vector<ChannelHandle> channels(1);
        ASSERT_EQ(HcclChannelDescInit(channelDesc.data(), 1), HCCL_SUCCESS);
        channelDesc[0].remoteRank = 2;
        channelDesc[0].channelProtocol = protocol;
        channelDesc[0].notifyNum = 1;
        channelDesc[0].localEndpoint.protocol = protocol;
        channelDesc[0].localEndpoint.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        channelDesc[0].localEndpoint.loc.device.devPhyId = 0U;
        channelDesc[0].localEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        channelDesc[0].localEndpoint.commAddr.addr.s_addr = 0x01000001U;
        channelDesc[0].remoteEndpoint.protocol = protocol;
        channelDesc[0].remoteEndpoint.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        channelDesc[0].remoteEndpoint.loc.device.devPhyId = 1U;
        channelDesc[0].remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        channelDesc[0].remoteEndpoint.commAddr.addr.s_addr = 0x02000002U;

        MOCKER(&hcomm::ClusterMonitor::RegisterToClusterMonitor).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&MyRank::GetOpExpansionMode).stubs().will(returnValue(0u));
        MOCKER_CPP(&EndpointMgr::GetWithTag).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&MyRank::PrepareMemHandles).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&MyRank::BatchCreateSockets).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(HcommChannelCreateWithConfig).stubs().will(returnValue(0));
        g_sharedJettyGetStatusCallCount = 0;
        MOCKER(HcommChannelGetStatus).stubs().will(invoke(StubHcommChannelGetStatus));
        MOCKER_CPP(&Hccl::EnvSocketConfig::GetLinkTimeOut).stubs().will(returnValue(static_cast<s32>(5)));
        MOCKER_CPP(&MyRank::BatchExchangeAndCheckConsistency).stubs().will(returnValue(HCCL_SUCCESS));

        ret = HcclChannelAcquireWithConfig(
            comm, CommEngine::COMM_ENGINE_AIV, channelDesc.data(), 1, config, channels.data());
        EXPECT_EQ(ret, HCCL_SUCCESS);

        // 清理共享 jetty 池中本 MyRank 的条目，避免 MyRank 析构时以 null handle 调用真实 HcommChannelDestroy
        MOCKER(HcommChannelDestroy).stubs().will(returnValue(0));
        (void)SharedJettyChannelPool::GetInstance().DestroyAllByMyRank(hcclCommPtr->GetCollComm()->GetMyRank());
        HcclChannelConfigDestroy(config);
    }

private:
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
};

TEST_F(HcclChannelDescTest, Ut_ProcessRoceChannelDesc_When_IsCommunicatorV2_Is_True_RetrunHCCLSUCCESS)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);
    MOCKER(&MyRank::CreateChannels).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommDpuChannelRegisterDfx).stubs().will(returnValue(0));
    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CPU, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclChannelDescTest, Ut_ProcessRoceChannelDesc_HostCountPriority)
{
    auto& portConfig = const_cast<Hccl::MultiQpSrcPortConfig&>(
        Hccl::EnvConfig::GetInstance().GetRdmaConfig().GetMultiQpSrcPortConfig());
    portConfig.ipPairToPorts.clear();
    portConfig.ipPairToPorts["1.0.0.0,2.0.0.0"] = {10001, 10002, 10003};
    ResetHostMultiQpCountStub();
    MOCKER(hrtGetDevice).stubs().will(invoke(StubGetDeviceForHostMultiQp));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(invoke(StubGetDevicePhyIdForHostMultiQp));
    MOCKER(RaGetHccnCfg).stubs().will(invoke(StubRaGetHostMultiQpCount));

    EndpointLocType localLocType = ENDPOINT_LOC_TYPE_HOST;
    auto resolveQueueNum = [this, &localLocType](uint32_t requestedQueueNum) {
        HcclChannelDesc input{};
        HcclChannelDesc output{};
        InitRoceChannelDesc(input);
        EXPECT_EQ(HcclChannelDescInit(&output, 1), HCCL_SUCCESS);
        input.roceAttr.queueNum = requestedQueueNum;
        input.localEndpoint.loc.locType = localLocType;
        input.localEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        input.localEndpoint.commAddr.addr = Hccl::IpAddress("1.0.0.0").GetBinaryAddress().addr;
        input.remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        input.remoteEndpoint.commAddr.addr = Hccl::IpAddress("2.0.0.0").GetBinaryAddress().addr;
        EXPECT_EQ(ProcessHcclResPackReq(input, output, hcclCommPtr.get()), HCCL_SUCCESS);
        return output.roceAttr.queueNum;
    };

    EXPECT_EQ(resolveQueueNum(7), 7U);
    EXPECT_EQ(g_getDeviceCallCount, 0U);
    EXPECT_TRUE(g_hostConfigReadKeys.empty());

    EXPECT_EQ(resolveQueueNum(INVALID_UINT), 4U);
    EXPECT_EQ(g_getDeviceCallCount, 1U);
    EXPECT_EQ(g_getDevicePhyIdCallCount, 1U);
    EXPECT_EQ(g_hostConfigReadKeys, (std::vector<HccnCfgKey>{HCCN_CFG_UDP_PORT_MODE, HCCN_CFG_MULTI_QP_COUNT}));

    localLocType = ENDPOINT_LOC_TYPE_DEVICE;
    g_hostConfigReadKeys.clear();
    const u32 getDeviceCallCount = g_getDeviceCallCount;
    EXPECT_EQ(resolveQueueNum(INVALID_UINT), 3U);
    EXPECT_EQ(g_getDeviceCallCount, getDeviceCallCount);
    EXPECT_TRUE(g_hostConfigReadKeys.empty());

    localLocType = ENDPOINT_LOC_TYPE_HOST;
    g_hostConfigReadKeys.clear();
    g_hostMultiQpMode.clear();
    EXPECT_EQ(resolveQueueNum(INVALID_UINT), 3U);
    EXPECT_EQ(g_hostConfigReadKeys, (std::vector<HccnCfgKey>{HCCN_CFG_UDP_PORT_MODE}));

    g_hostConfigReadKeys.clear();
    g_getDevicePhyIdResult = HCCL_E_RUNTIME;
    EXPECT_EQ(resolveQueueNum(INVALID_UINT), 3U);
    EXPECT_TRUE(g_hostConfigReadKeys.empty());

    g_getDevicePhyIdResult = HCCL_SUCCESS;
    g_getDeviceResult = HCCL_E_RUNTIME;
    const u32 phyIdCallCount = g_getDevicePhyIdCallCount;
    EXPECT_EQ(resolveQueueNum(INVALID_UINT), 3U);
    EXPECT_EQ(g_getDevicePhyIdCallCount, phyIdCallCount);
    EXPECT_TRUE(g_hostConfigReadKeys.empty());

    g_getDeviceResult = HCCL_SUCCESS;
    portConfig.ipPairToPorts.clear();
    EXPECT_EQ(resolveQueueNum(INVALID_UINT), Hccl::EnvConfig::GetInstance().GetRdmaConfig().GetRdmaQueueNum());

    MOCKER_CPP(&hccl::hcclComm::IsCommunicatorV2).stubs().will(returnValue(false));
    EXPECT_EQ(resolveQueueNum(6), 6U);

    portConfig.ipPairToPorts.clear();
}

TEST_F(HcclChannelDescTest, Ut_ProcessRoceChannelDesc_When_TcIsInvaild_ReturnHCCLEPARA)
{
    hcclCommPtr->collComm_->config_.trafficClass_ = 10; // 单独赋非法值
    comm = static_cast<HcclComm>(hcclCommPtr.get());    // 重新给comm

    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);

    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CPU, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclChannelDescTest, Ut_ProcessRoceChannelDesc_When_SlIsInvaild_ReturnHCCLEPARA)
{
    hcclCommPtr->collComm_->config_.serviceLevel_ = 10; // 单独赋非法值
    comm = static_cast<HcclComm>(hcclCommPtr.get());    // 重新给comm

    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);

    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CPU, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclChannelDescTest, Ut_ProcessRoceChannelDesc_When_RetryIntervalIsInvaild_ReturnHCCLEPARA)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);
    channelDesc[0].roceAttr.retryInterval = 30; // 单独赋非法值

    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CPU, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclChannelDescTest, Ut_ProcessRoceChannelDesc_When_RetryCntIsInvaild_ReturnHCCLEPARA)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);
    channelDesc[0].roceAttr.retryCnt = 10; // 单独赋非法值
    MOCKER(&MyRank::BatchCreateSockets).stubs().will(returnValue(HCCL_SUCCESS));
    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CPU, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelAcquire_When_Notifynum_Exceeds_Return_Error)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);
    channelDesc[0].notifyNum = 65;
    MOCKER(&hcomm::ClusterMonitor::RegisterToClusterMonitor).stubs().will(returnValue(HCCL_SUCCESS));

    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelAcquire_When_BuildConnection_Fails_Return_Error)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);

    // Mock BuildConnection 失败
    MOCKER(&HostCpuRoceChannel::BuildConnection).stubs().will(returnValue(HCCL_E_NETWORK));

    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelAcquire_When_IbvPostRecv_Fails_Return_Error)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);

    // Mock BuildConnection 成功
    MOCKER(&HostCpuRoceChannel::BuildConnection).stubs().will(returnValue(HCCL_SUCCESS));
    // Mock IbvPostRecv 失败
    MOCKER(&HostCpuRoceChannel::IbvPostRecv).stubs().will(returnValue(HCCL_E_INTERNAL));

    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_E_PTR);
}
TEST_F(HcclChannelDescTest, Ut_ProcessUbChannelDesc_When_WrongProtocol_Expect_E_PARA)
{
    HcclChannelDesc in{};
    HcclChannelDesc out{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    ASSERT_EQ(HcclChannelDescInit(&out, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_ROCE;
    ret = ProcessUbChannelDesc(in, out, hcclCommPtr.get());
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclChannelDescTest, Ut_ProcessUbChannelDesc_When_Hccs_Expect_E_PARA)
{
    HcclChannelDesc in{};
    HcclChannelDesc out{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    ASSERT_EQ(HcclChannelDescInit(&out, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_HCCS;
    ret = ProcessUbChannelDesc(in, out, hcclCommPtr.get());
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclChannelDescTest, Ut_ProcessUbChannelDesc_When_UbcCtp_QosUnset_UsesCommHcclQos)
{
    hcclCommPtr->GetCollComm()->GetCommConfig().SetConfigHcclQos(5u);
    comm = static_cast<HcclComm>(hcclCommPtr.get());

    HcclChannelDesc in{};
    HcclChannelDesc out{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    ASSERT_EQ(HcclChannelDescInit(&out, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UB_CTP;
    ret = ProcessUbChannelDesc(in, out, hcclCommPtr.get());
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclChannelDescTest, Ut_ProcessUbChannelDesc_When_UbcTp_QosUnset_UsesUbQosDefault)
{
    hcclCommPtr->GetCollComm()->GetCommConfig().SetConfigHcclQos(HCCL_COMM_QOS_CONFIG_NOT_SET);
    comm = static_cast<HcclComm>(hcclCommPtr.get());

    HcclChannelDesc in{};
    HcclChannelDesc out{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    ASSERT_EQ(HcclChannelDescInit(&out, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UBC_TP;
    ret = ProcessUbChannelDesc(in, out, hcclCommPtr.get());
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclChannelDescTest, Ut_ProcessUbChannelDesc_When_UbcCtp_Valid_Expect_Success)
{
    hcclCommPtr->GetCollComm()->GetCommConfig().SetConfigHcclQos(1u);
    comm = static_cast<HcclComm>(hcclCommPtr.get());

    HcclChannelDesc in{};
    HcclChannelDesc out{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    ASSERT_EQ(HcclChannelDescInit(&out, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UB_CTP;
    ret = ProcessUbChannelDesc(in, out, hcclCommPtr.get());
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclChannelDescTest, Ut_ProcessUbChannelDesc_When_Uboe_QosUnset_UsesCommHcclQos)
{
    hcclCommPtr->GetCollComm()->GetCommConfig().SetConfigHcclQos(3u);
    comm = static_cast<HcclComm>(hcclCommPtr.get());

    HcclChannelDesc in{};
    HcclChannelDesc out{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    ASSERT_EQ(HcclChannelDescInit(&out, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UBOE;
    ret = ProcessUbChannelDesc(in, out, hcclCommPtr.get());
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ProcessHcclChannelDesc 的 UB_MEM/HCCS 分支仅做 pathMode 拷贝，不依赖 hcclComm，
// 使用轻量 fixture 避免重型通信域初始化，便于在无设备环境独立运行。
class ProcessHcclChannelDescTest : public testing::Test {
protected:
    HcclResult ret{HCCL_SUCCESS};
};

TEST_F(ProcessHcclChannelDescTest, Ut_ProcessHcclChannelDesc_When_UbMem_CopiesPathMode)
{
    HcclChannelDesc in{};
    HcclChannelDesc out{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    ASSERT_EQ(HcclChannelDescInit(&out, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UB_MEM;
    in.ubMemAttr.pathMode = 2u;

    ret = ProcessHcclChannelDesc(in, out, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(out.channelProtocol, COMM_PROTOCOL_UB_MEM);
    EXPECT_EQ(out.ubMemAttr.pathMode, 2u);
}

TEST_F(ProcessHcclChannelDescTest, Ut_ProcessHcclChannelDesc_When_UbMem_PathModeZero_CopiesZero)
{
    HcclChannelDesc in{};
    HcclChannelDesc out{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    ASSERT_EQ(HcclChannelDescInit(&out, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_UB_MEM;
    in.ubMemAttr.pathMode = 0u;

    ret = ProcessHcclChannelDesc(in, out, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(out.ubMemAttr.pathMode, 0u);
}

TEST_F(ProcessHcclChannelDescTest, Ut_ProcessHcclChannelDesc_When_Hccs_DoesNotCopyUnion)
{
    HcclChannelDesc in{};
    HcclChannelDesc out{};
    ASSERT_EQ(HcclChannelDescInit(&in, 1), HCCL_SUCCESS);
    ASSERT_EQ(HcclChannelDescInit(&out, 1), HCCL_SUCCESS);
    in.channelProtocol = COMM_PROTOCOL_HCCS;
    in.ubMemAttr.pathMode = 1u;

    ret = ProcessHcclChannelDesc(in, out, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(out.channelProtocol, COMM_PROTOCOL_HCCS);
    EXPECT_EQ(out.ubMemAttr.pathMode, 0xFFu);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelAcquire_When_PendingConsumed_AppendHistoricalSymmetricMemHandle)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    InitSymMemChannelDesc(channelDesc[0]);

    MockConsumedPendingSymMem();
    MOCKER_CPP(&CollComm::GetAllRegisteredSymMemHandles)
        .expects(once())
        .with(processWith(StubFillSymHandles))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&MyRank::QueryChannels).expects(once()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&MyRank::CreateChannels).stubs().will(invoke(StubCreateChannelsCapture));
    uint32_t remoteMemNum = 1;
    CommMem remoteMem{};
    CommMem* remoteMems = &remoteMem;
    const std::vector<std::string> remoteMemTags{g_symMemTag};
    MockRemoteMems(remoteMemNum, remoteMems, remoteMemTags);
    MOCKER_CPP(&CollComm::UpdateSymmetricRemoteMem).expects(once()).will(returnValue(HCCL_SUCCESS));

    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(g_capturedMemHandles.size(), 2U);
    EXPECT_EQ(g_capturedMemHandles[0], g_userMemHandle);
    EXPECT_EQ(g_capturedMemHandles[1], g_symMemHandle);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelAcquire_When_ExistingChannelHasSymmetricTag_NotAppendHandle)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    InitSymMemChannelDesc(channelDesc[0]);

    MockConsumedPendingSymMem();
    MOCKER_CPP(&CollComm::GetAllRegisteredSymMemHandles)
        .expects(once())
        .with(processWith(StubFillSymHandles))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollComm::GetRemoteMissingSymMemHandles)
        .expects(once())
        .with(processWith(StubSetRemoteTags), processWith(StubFillSymHandles))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&MyRank::QueryChannels).expects(once()).will(invoke(StubQueryExistingChannels));
    uint32_t remoteMemNum = 1;
    CommMem remoteMem{};
    CommMem* remoteMems = &remoteMem;
    const std::vector<std::string> remoteMemTags{g_symMemTag};
    MockRemoteMems(remoteMemNum, remoteMems, remoteMemTags);
    MOCKER_CPP(&MyRank::CreateChannels).stubs().will(invoke(StubCreateChannelsCapture));
    MOCKER_CPP(&CollComm::UpdateSymmetricRemoteMem).expects(never());

    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(g_capturedMemHandles.size(), 1U);
    EXPECT_EQ(g_capturedMemHandles[0], g_userMemHandle);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelAcquire_When_CpuUrma_NotAppendSymmetricMemHandle)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    ASSERT_EQ(HcclChannelDescInit(channelDesc.data(), 1), HCCL_SUCCESS);
    channelDesc[0].remoteRank = 2;
    channelDesc[0].channelProtocol = COMM_PROTOCOL_UBOE;
    channelDesc[0].notifyNum = 1;
    channelDesc[0].memHandles = &g_userMemHandle;
    channelDesc[0].memHandleNum = 1;

    MOCKER_CPP(&CollComm::RegisterPendingSymmetricMemHandles).expects(never());
    MOCKER_CPP(&MyRank::CreateChannels).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommDpuChannelRegisterDfx).stubs().will(returnValue(0));

    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CPU, channelDesc.data(), 1, channels.data());
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelAcquire_When_AivUbRtp_ConvertsHostHandleToDeviceEntity)
{
    HcclChannelDesc channelDesc{};
    ASSERT_EQ(HcclChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.remoteRank = 2;
    channelDesc.channelProtocol = COMM_PROTOCOL_UB_RTP;

    hcomm::AivUrmaChannel channel(reinterpret_cast<EndpointHandle>(0x1), HcommChannelDesc{});
    g_testChannel = reinterpret_cast<ChannelHandle>(&channel);

    MOCKER(&hcomm::ClusterMonitor::RegisterToClusterMonitor).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&MyRank::CreateChannels).stubs().will(invoke(StubCreateChannelsCapture));
    MOCKER_CPP(&hcomm::ChannelProcess::ChannelGet).expects(once()).will(invoke(StubChannelGet));
    MOCKER_CPP(&hcomm::AivUrmaChannel::BuildChannelEntityToDevice, HcclResult(hcomm::AivUrmaChannel::*)(void**))
        .expects(once())
        .will(invoke(StubBuildChannelEntityToDevice));
    MOCKER_CPP(&hcomm::ChannelProcess::RegisterChannelD2HMap).expects(once()).will(returnValue(HCCL_SUCCESS));

    ChannelHandle outputChannel = 0;
    ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AIV, &channelDesc, 1, &outputChannel);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(outputChannel, static_cast<ChannelHandle>(reinterpret_cast<uintptr_t>(g_testDevChannelEntity)));
}

TEST_F(HcclChannelDescTest, Ut_ProcessRoceChannelDesc_When_HcclQosUnset_UsesRdmaEnvSlTc)
{
    // EnvRdmaConfig 默认 SL=4、TC=132（CI 中 setenv 晚于 EnvConfig 解析时不生效）
    ExpectRoceSlTcInHcommChannelDesc(HCCL_COMM_QOS_CONFIG_NOT_SET, 4U, 132U);
}

TEST_F(HcclChannelDescTest, Ut_ProcessRoceChannelDesc_When_HcclQos1_MapsSlAndTcFromHccn)
{
    MOCKER(RaGetHccnCfg).stubs().will(invoke(StubRaGetHccnCfgRoceQosDscp));
    ExpectRoceSlTcInHcommChannelDesc(1U, 1U, static_cast<uint8_t>(20U << 2U));
}

TEST_F(HcclChannelDescTest, Ut_ProcessRoceChannelDesc_When_HcclQos4_MapsSlAndTcFromHccn)
{
    MOCKER(RaGetHccnCfg).stubs().will(invoke(StubRaGetHccnCfgRoceQosDscp));
    ExpectRoceSlTcInHcommChannelDesc(4U, 4U, static_cast<uint8_t>(50U << 2U));
}

TEST_F(HcclChannelDescTest, Ut_ProcessRoceChannelDesc_When_HcclQos6_MapsSlAndTcFromHccn)
{
    MOCKER(RaGetHccnCfg).stubs().will(invoke(StubRaGetHccnCfgRoceQosDscp));
    ExpectRoceSlTcInHcommChannelDesc(6U, 6U, static_cast<uint8_t>(70U << 2U));
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelQuery_When_CommNull_Expect_E_PTR)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);
    EXPECT_EQ(
        HcclChannelQuery(nullptr, CommEngine::COMM_ENGINE_CCU, channelDesc.data(), 1, channels.data()), HCCL_E_PTR);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelQuery_When_DescsNull_Expect_E_PTR)
{
    std::vector<ChannelHandle> channels(1);
    EXPECT_EQ(HcclChannelQuery(comm, CommEngine::COMM_ENGINE_CCU, nullptr, 1, channels.data()), HCCL_E_PTR);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelQuery_When_ChannelsNull_Expect_E_PTR)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    GetChannelDesc(channelDesc);
    EXPECT_EQ(HcclChannelQuery(comm, CommEngine::COMM_ENGINE_CCU, channelDesc.data(), 1, nullptr), HCCL_E_PTR);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelQuery_When_NumZero_Expect_E_PARA)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);
    EXPECT_EQ(HcclChannelQuery(comm, CommEngine::COMM_ENGINE_CCU, channelDesc.data(), 0, channels.data()), HCCL_E_PARA);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelQuery_When_LegacyComm_Expect_NOT_SUPPORT)
{
    MOCKER_CPP(&hccl::hcclComm::IsCommunicatorV2).stubs().will(returnValue(false));
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);
    EXPECT_EQ(
        HcclChannelQuery(comm, CommEngine::COMM_ENGINE_CCU, channelDesc.data(), 1, channels.data()),
        HCCL_E_NOT_SUPPORT);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelQuery_When_Normal_Expect_SUCCESS)
{
    MOCKER_CPP(&MyRank::GetOpExpansionMode).stubs().will(returnValue(0u));
    MOCKER_CPP(&MyRank::QueryChannels).stubs().will(returnValue(HCCL_SUCCESS));
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);
    EXPECT_EQ(
        HcclChannelQuery(comm, CommEngine::COMM_ENGINE_CCU, channelDesc.data(), 1, channels.data()), HCCL_SUCCESS);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelQuery_When_MyRankFailed_Expect_ErrorPropagated)
{
    MOCKER_CPP(&MyRank::GetOpExpansionMode).stubs().will(returnValue(0u));
    MOCKER_CPP(&MyRank::QueryChannels).stubs().will(returnValue(HCCL_E_INTERNAL));
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);
    EXPECT_EQ(
        HcclChannelQuery(comm, CommEngine::COMM_ENGINE_CCU, channelDesc.data(), 1, channels.data()), HCCL_E_INTERNAL);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelDestroy_When_CommNull_Expect_E_PTR)
{
    ChannelHandle channels[1] = {g_testChannel};
    EXPECT_EQ(HcclChannelDestroy(nullptr, channels, 1), HCCL_E_PTR);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelDestroy_When_ChannelsNull_Expect_E_PTR)
{
    EXPECT_EQ(HcclChannelDestroy(comm, nullptr, 1), HCCL_E_PTR);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelDestroy_When_NumZero_Expect_E_PARA)
{
    ChannelHandle channels[1] = {g_testChannel};
    EXPECT_EQ(HcclChannelDestroy(comm, channels, 0), HCCL_E_PARA);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelDestroy_When_LegacyComm_Expect_NOT_SUPPORT)
{
    MOCKER_CPP(&hccl::hcclComm::IsCommunicatorV2).stubs().will(returnValue(false));
    ChannelHandle channels[1] = {g_testChannel};
    EXPECT_EQ(HcclChannelDestroy(comm, channels, 1), HCCL_E_NOT_SUPPORT);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelDestroy_When_Normal_Expect_SUCCESS)
{
    MOCKER_CPP(&MyRank::DestroyChannels).stubs().will(returnValue(HCCL_SUCCESS));
    ChannelHandle channels[1] = {g_testChannel};
    EXPECT_EQ(HcclChannelDestroy(comm, channels, 1), HCCL_SUCCESS);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelDestroy_When_MyRankFailed_Expect_ErrorPropagated)
{
    MOCKER_CPP(&MyRank::DestroyChannels).stubs().will(returnValue(HCCL_E_INTERNAL));
    ChannelHandle channels[1] = {g_testChannel};
    EXPECT_EQ(HcclChannelDestroy(comm, channels, 1), HCCL_E_INTERNAL);
}

// 适配层: GetCollComm 返回 nullptr 时 Query 返回 E_PTR
TEST_F(HcclChannelDescTest, Ut_HcclChannelQuery_When_CollCommNull_Expect_E_PTR)
{
    MOCKER_CPP(&hccl::hcclComm::GetCollComm).stubs().will(returnValue(static_cast<hccl::CollComm*>(nullptr)));
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);
    EXPECT_EQ(HcclChannelQuery(comm, CommEngine::COMM_ENGINE_CCU, channelDesc.data(), 1, channels.data()), HCCL_E_PTR);
}

// 适配层: channelNum 超过上限(1024*1024)时返回 E_PARA
TEST_F(HcclChannelDescTest, Ut_HcclChannelQuery_When_NumExceedsMax_Expect_E_PARA)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);
    EXPECT_EQ(
        HcclChannelQuery(comm, CommEngine::COMM_ENGINE_CCU, channelDesc.data(), 1024 * 1024 + 1, channels.data()),
        HCCL_E_PARA);
}

TEST_F(HcclChannelDescTest, Ut_HcclChannelDestroy_When_NumExceedsMax_Expect_E_PARA)
{
    ChannelHandle channels[1] = {g_testChannel};
    EXPECT_EQ(HcclChannelDestroy(comm, channels, 1024 * 1024 + 1), HCCL_E_PARA);
}

// 适配层: channelDesc 的 magicWord 非法时, ProcessHcclResPackReq 失败并透传 E_PARA
TEST_F(HcclChannelDescTest, Ut_HcclChannelQuery_When_DescMagicWordInvalid_Expect_E_PARA)
{
    std::vector<HcclChannelDesc> channelDesc(1);
    std::vector<ChannelHandle> channels(1);
    GetChannelDesc(channelDesc);
    channelDesc[0].header.magicWord = 0xDEADBEEF;
    EXPECT_EQ(HcclChannelQuery(comm, CommEngine::COMM_ENGINE_CCU, channelDesc.data(), 1, channels.data()), HCCL_E_PARA);
}

// 共享 jetty 路径覆盖：CreateSharedJettyChannelsForGroup（line 997/1000）与
// WaitForSharedJettyChannelsReady（line 1256）通过 HcclChannelAcquireWithConfig 公共 API 触发。
// 参数化封装：UB_CTP 与 UB_RTP 用例仅协议和 tag 不同，提取公共逻辑消除重复代码。
TEST_F(HcclChannelDescTest, Ut_HcclChannelAcquireWithConfig_When_SharedJetty_Expect_Success)
{
    DoSharedJettyAcquireTest(COMM_PROTOCOL_UB_CTP, "ut_sj_4900");
}

// 共享 jetty 支持 UB_RTP(UBG) 协议：门控放行 + AivUrmaChannel 路径
TEST_F(HcclChannelDescTest, Ut_HcclChannelAcquireWithConfig_When_SharedJetty_UbRtp_Expect_Success)
{
    DoSharedJettyAcquireTest(COMM_PROTOCOL_UB_RTP, "ut_sj_ubrtp");
}
