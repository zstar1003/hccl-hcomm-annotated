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
#include "mockcpp/mokc.h"
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mockcpp/mockcpp.hpp>
#include <string>
#include "endpoint.h"
#include "hcomm_c_adpt.h"
#include "hcomm_res_defs.h"
#include "hcomm_channel.h"
#include "channel_process.h"
#include "../../../../../../src/base_comm/hcomm_res_mgr.h"
#include "../../../../../../src/base_comm/resources/endpoints/mgr/endpoint_mgr.h"
#include "hcomm_c_adpt_common.h"
#include "env_config/env_config_v2.h"
#include "nic_plugin_manager.h"
#include "orion_adpt_utils.h"

// 内存方法从 Endpoint 移至 RegedMemMgr；EndpointMgr 收编 HcommEndpointMap；
// SetNicEndpointCtx 移至 PluginEndpointHolder。StubEndpoint 不再 override 内存方法。
[[maybe_unused]] static HcommNicEndpointOps g_stubEndpointOps = {
    {HCOMM_NIC_ENDPOINT_OPS_VERSION, HCOMM_NIC_ENDPOINT_OPS_MAGIC_WORD, sizeof(HcommNicEndpointOps), 0},
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};
#include "hccp_peer_manager.h"
#include "rdma_handle_manager.h"
#include "hccp_nda.h"
#include "hcomm_adapter_hccp.h"
#include "hccp_hdc_manager.h"
#include "adapter_rts_common.h"

using namespace hcomm;

HcclResult StubServerSocketGetListenPort(Endpoint* /*endpoint*/, uint32_t* port)
{
    *port = 12345;
    return HCCL_SUCCESS;
}

class HcommCAdptTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcommCAdptTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "HcommCAdptTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        const char* dfsConfig = std::getenv("HCCL_DFS_CONFIG");
        hasOldDfsConfig_ = (dfsConfig != nullptr);
        oldDfsConfig_ = hasOldDfsConfig_ ? dfsConfig : "";
        setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
        std::cout << "A Test case in HcommCAdptTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        if (hasOldDfsConfig_) {
            setenv("HCCL_DFS_CONFIG", oldDfsConfig_.c_str(), 1);
        } else {
            unsetenv("HCCL_DFS_CONFIG");
        }
        std::cout << "A Test case in HcommCAdptTest TearDown" << std::endl;
    }

    bool hasOldDfsConfig_ = false;
    std::string oldDfsConfig_;
};

TEST_F(HcommCAdptTest, ut_HcommChannelGet_When_Normal_Expect_Success)
{
    ChannelHandle channelHandle = 0x12345;
    void* channel = nullptr;
    MOCKER(ChannelProcess::ChannelGet).stubs().will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommChannelGet(channelHandle, &channel);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_When_Normal_Expect_Success)
{
    // 假 handle 未注册到 channel map，GetChannelsInfo 找不到 channel 返回错误
    ChannelHandle channelList[2] = {0x12345, 0x12346};
    int32_t statusList[2] = {0, 0};
    HcommResult ret = HcommChannelGetStatus(channelList, 2, statusList);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_When_ChannelListNull_Expect_E_PTR)
{
    int32_t statusList[2] = {0, 0};
    HcommResult ret = HcommChannelGetStatus(nullptr, 2, statusList);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_When_StatusListNull_Expect_E_PTR)
{
    ChannelHandle channelList[2] = {0x12345, 0x12346};
    HcommResult ret = HcommChannelGetStatus(channelList, 2, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_When_ListNumZero_Expect_E_PARA)
{
    ChannelHandle channelList[2] = {0x12345, 0x12346};
    int32_t statusList[2] = {0, 0};
    HcommResult ret = HcommChannelGetStatus(channelList, 0, statusList);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// ===================== 非阻塞建链 HandleAicpuStatus/HandleAivStatus UT =====================

// FakeChannel for HcommChannelGetStatus 端到端测试
class HcommCAdptFakeChannel : public hcomm::Channel {
public:
    HcommCAdptFakeChannel(CommEngine eng, hcomm::ChannelStatus sta)
    {
        engine_ = eng;
        channelKind_ = hcomm::HcommChannelKind::AICPU_TS_UBOE;
        fakeStatus_ = sta;
    }
    ~HcommCAdptFakeChannel() = default;

    hcomm::HcommChannelKind GetChannelKind() const override { return channelKind_; }
    HcclResult Init() override { return HCCL_SUCCESS; }
    HcclResult GetNotifyNum(uint32_t* notifyNum) const override { return HCCL_SUCCESS; }
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override { return HCCL_SUCCESS; }
    hcomm::ChannelStatus GetStatus() override { return fakeStatus_; }
    const HcommChannelDesc& GetChannelDesc() const override { return desc_; }
    HcclResult Clean() override { return HCCL_SUCCESS; }
    HcclResult Resume() override { return HCCL_SUCCESS; }
    HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override { return HCCL_SUCCESS; }
    HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override { return HCCL_SUCCESS; }
    HcclResult WriteWithNotify(void* dst, const void* src, const uint64_t len, uint32_t remoteNotifyIdx) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult Write(void* dst, const void* src, uint64_t len) override { return HCCL_SUCCESS; }
    HcclResult Read(void* dst, const void* src, uint64_t len) override { return HCCL_SUCCESS; }
    HcclResult ChannelFence() override { return HCCL_SUCCESS; }

private:
    hcomm::ChannelStatus fakeStatus_{hcomm::ChannelStatus::INIT};
    HcommChannelDesc desc_{};
};

// 辅助：注册 channel 到全局 map
static ChannelHandle HcommCAdptRegisterChannel(std::shared_ptr<HcommCAdptFakeChannel>& ch)
{
    ChannelHandle handle = reinterpret_cast<ChannelHandle>(ch.get());
    hcomm::DeviceChannelKey key{0, handle};
    std::lock_guard<std::mutex> lock(hcomm::ChannelProcess::g_ChannelMapMtx);
    hcomm::ChannelProcess::g_ChannelMap.emplace(handle, ch);
    hcomm::ChannelProcess::g_ChannelD2HMap.emplace(key, handle);
    return handle;
}

// 辅助：清理全局 map
static void HcommCAdptClearChannelMap()
{
    std::lock_guard<std::mutex> lock(hcomm::ChannelProcess::g_ChannelMapMtx);
    hcomm::ChannelProcess::g_ChannelMap.clear();
    hcomm::ChannelProcess::g_ChannelD2HMap.clear();
}

// HandleAicpuStatus: 未全 READY（含 CONNECTING）→ 返回 CONNECTING
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_Aicpu_When_Connecting_Return_Connecting)
{
    auto ch = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AICPU, hcomm::ChannelStatus::INIT);
    ChannelHandle handle = HcommCAdptRegisterChannel(ch);

    int32_t statusList[1] = {0};
    HcommResult ret = HcommChannelGetStatus(&handle, 1, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], hcomm::HCOMM_CHANNEL_STATUS_CONNECTING);

    HcommCAdptClearChannelMap();
}

// HandleAicpuStatus: 含 FAILED → FAILED 真实返回
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_Aicpu_When_Failed_Return_Failed)
{
    auto ch = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AICPU, hcomm::ChannelStatus::FAILED);
    ChannelHandle handle = HcommCAdptRegisterChannel(ch);

    int32_t statusList[1] = {0};
    HcommResult ret = HcommChannelGetStatus(&handle, 1, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], hcomm::HCOMM_CHANNEL_STATUS_FAILED);

    HcommCAdptClearChannelMap();
}

// HandleAicpuStatus: 含 SOCKET_TIMEOUT → TIMEOUT
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_Aicpu_When_SocketTimeout_Return_Timeout)
{
    auto ch = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AICPU, hcomm::ChannelStatus::SOCKET_TIMEOUT);
    ChannelHandle handle = HcommCAdptRegisterChannel(ch);

    int32_t statusList[1] = {0};
    HcommResult ret = HcommChannelGetStatus(&handle, 1, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], hcomm::HCOMM_CHANNEL_STATUS_TIMEOUT);

    HcommCAdptClearChannelMap();
}

// HandleAicpuStatus: 混合状态（READY+FAILED）→ 未全 READY，FAILED 真实返回，READY 报 CONNECTING
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_Aicpu_When_MixedReadyFailed_Return_FailedAndConnecting)
{
    auto ch1 = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AICPU, hcomm::ChannelStatus::READY);
    auto ch2 = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AICPU, hcomm::ChannelStatus::FAILED);
    ChannelHandle handles[2] = {HcommCAdptRegisterChannel(ch1), HcommCAdptRegisterChannel(ch2)};

    int32_t statusList[2] = {0, 0};
    HcommResult ret = HcommChannelGetStatus(handles, 2, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], hcomm::HCOMM_CHANNEL_STATUS_CONNECTING);
    EXPECT_EQ(statusList[1], hcomm::HCOMM_CHANNEL_STATUS_FAILED);

    HcommCAdptClearChannelMap();
}

// HandleAivStatus: 未全 READY（含 CONNECTING）→ 返回 CONNECTING
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_Aiv_When_Connecting_Return_Connecting)
{
    auto ch = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AIV, hcomm::ChannelStatus::INIT);
    ChannelHandle handle = HcommCAdptRegisterChannel(ch);

    int32_t statusList[1] = {0};
    HcommResult ret = HcommChannelGetStatus(&handle, 1, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], hcomm::HCOMM_CHANNEL_STATUS_CONNECTING);

    HcommCAdptClearChannelMap();
}

// HandleAivStatus: 含 FAILED → FAILED 真实返回
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_Aiv_When_Failed_Return_Failed)
{
    auto ch = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AIV, hcomm::ChannelStatus::FAILED);
    ChannelHandle handle = HcommCAdptRegisterChannel(ch);

    int32_t statusList[1] = {0};
    HcommResult ret = HcommChannelGetStatus(&handle, 1, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], hcomm::HCOMM_CHANNEL_STATUS_FAILED);

    HcommCAdptClearChannelMap();
}

// 非 AICPU/AIV engine（如 CPU）→ 走通用 CopyLinkStatusToOutput
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_Cpu_When_Connecting_Return_Connecting)
{
    auto ch = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_CPU, hcomm::ChannelStatus::INIT);
    ChannelHandle handle = HcommCAdptRegisterChannel(ch);

    int32_t statusList[1] = {0};
    HcommResult ret = HcommChannelGetStatus(&handle, 1, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], hcomm::HCOMM_CHANNEL_STATUS_CONNECTING);

    HcommCAdptClearChannelMap();
}

// GetChannelsInfo 失败（channel not found）→ HcommChannelGetStatus 返回 E_INTERNAL
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_When_ChannelNotFound_Return_E_INTERNAL)
{
    ChannelHandle handle = static_cast<ChannelHandle>(0x8888);
    int32_t statusList[1] = {0};
    HcommResult ret = HcommChannelGetStatus(&handle, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    HcommCAdptClearChannelMap();
}

// HandleAivStatus: 全 READY + 已 deviceEntityReady → FillAivDevEntities 跳过 → 返回 READY
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_Aiv_When_AllReady_Return_Ready)
{
    auto ch = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AIV, hcomm::ChannelStatus::READY);
    ch->SetDeviceEntityReady();
    ChannelHandle handle = HcommCAdptRegisterChannel(ch);

    int32_t statusList[1] = {0};
    HcommResult ret = HcommChannelGetStatus(&handle, 1, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], hcomm::HCOMM_CHANNEL_STATUS_READY);

    HcommCAdptClearChannelMap();
}

// HandleAicpuStatus: 全 READY + 已 deviceEntityReady → LaunchAicpuKernel 跳过 → 返回 READY
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_Aicpu_When_AllReady_Return_Ready)
{
    auto ch = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AICPU, hcomm::ChannelStatus::READY);
    ch->SetDeviceEntityReady();
    ChannelHandle handle = HcommCAdptRegisterChannel(ch);

    int32_t statusList[1] = {0};
    HcommResult ret = HcommChannelGetStatus(&handle, 1, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], hcomm::HCOMM_CHANNEL_STATUS_READY);

    HcommCAdptClearChannelMap();
}

// 混合 engine：AICPU + AIV 同时传入，各自独立处理 → 各自返回对应状态
TEST_F(HcommCAdptTest, ut_HcommChannelGetStatus_MixedEngine_When_MixedStatus_Return_EachStatus)
{
    auto chAicpuReady = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AICPU, hcomm::ChannelStatus::READY);
    chAicpuReady->SetDeviceEntityReady();
    auto chAivConnecting = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_AIV, hcomm::ChannelStatus::INIT);
    auto chCpuFailed = std::make_shared<HcommCAdptFakeChannel>(COMM_ENGINE_CPU, hcomm::ChannelStatus::FAILED);
    ChannelHandle handles[3] = {
        HcommCAdptRegisterChannel(chAicpuReady),
        HcommCAdptRegisterChannel(chAivConnecting),
        HcommCAdptRegisterChannel(chCpuFailed),
    };

    int32_t statusList[3] = {0, 0, 0};
    HcommResult ret = HcommChannelGetStatus(handles, 3, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], hcomm::HCOMM_CHANNEL_STATUS_READY);      // AICPU: READY
    EXPECT_EQ(statusList[1], hcomm::HCOMM_CHANNEL_STATUS_CONNECTING); // AIV: INIT → CONNECTING
    EXPECT_EQ(statusList[2], hcomm::HCOMM_CHANNEL_STATUS_FAILED);     // CPU: FAILED

    HcommCAdptClearChannelMap();
}

TEST_F(HcommCAdptTest, ut_HcommChannelGetNotifyNum_When_Normal_Expect_Success)
{
    ChannelHandle channelHandle = 0x12345;
    uint32_t notifyNum = 0;
    MOCKER(ChannelProcess::ChannelGetNotifyNum).stubs().will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommChannelGetNotifyNum(channelHandle, &notifyNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommChannelDestroy_When_Normal_Expect_Success)
{
    ChannelHandle channels[2] = {0x12345, 0x12346};
    MOCKER(ChannelProcess::ChannelDestroy).stubs().will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommChannelDestroy(channels, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommChannelGetRemoteMems_When_Normal_Expect_Success)
{
    ChannelHandle channelHandle = 0x12345;
    CommMem* remoteMem = nullptr;
    uint32_t memNum = 0;
    char* memInfosStorage[2] = {nullptr, nullptr};
    char** memInfos = memInfosStorage;
    MOCKER(ChannelProcess::ChannelGetRemoteMems).stubs().will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommChannelGetRemoteMems(channelHandle, &memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

namespace {

class ChannelAdptStubEndpoint final : public Endpoint {
public:
    explicit ChannelAdptStubEndpoint(const EndpointDesc& desc) : Endpoint(desc) {}
    HcclResult Init() override { return HCCL_SUCCESS; }
    RegedMemMgr* GetRegedMemMgr() override { return nullptr; }
    void* GetRdmaHandle() override { return nullptr; }
    bool IsCtxHandleValid() const override { return false; }
};

class ScopedChannelAdptStubEndpoint {
public:
    explicit ScopedChannelAdptStubEndpoint(EndpointLocType locType = ENDPOINT_LOC_TYPE_DEVICE)
    {
        EndpointDesc desc{};
        desc.loc.locType = locType;
        auto ep = std::make_unique<ChannelAdptStubEndpoint>(desc);
        handle_ = reinterpret_cast<EndpointHandle>(ep.get());
        HcommResMgr::GetInstance().GetEndpointMgr().Add(handle_, std::move(ep));
    }

    ~ScopedChannelAdptStubEndpoint()
    {
        if (handle_ != nullptr) {
            HcommResMgr::GetInstance().GetEndpointMgr().Remove(handle_);
        }
    }

    EndpointHandle Get() const { return handle_; }

private:
    EndpointHandle handle_ = nullptr;
};

uint32_t gCapturedChannelDescQos = 0U;
uint32_t gCapturedChannelDescSqDepth = 0U;
const char* gCapturedChannelDescChannelName = nullptr;
// 捕获 roceAttr 队列深度与 srcPortList，用于 v4 兼容性验证
uint32_t gCapturedRoceSqDepth = 0U;
uint32_t gCapturedRoceScqDepth = 0U;
uint16_t* gCapturedRoceSrcPortList = nullptr;

HcclResult CaptureCreateChannelsLoop(
    EndpointHandle, CommEngine, HcommChannelDesc* channelDescs, uint32_t channelNum, ChannelHandle*)
{
    if (channelDescs != nullptr && channelNum > 0U) {
        gCapturedChannelDescQos = channelDescs[0].qos;
        gCapturedChannelDescSqDepth = channelDescs[0].ubAttr.sqDepth;
        gCapturedChannelDescChannelName = channelDescs[0].channelName;
        gCapturedRoceSqDepth = channelDescs[0].roceAttr.sqDepth;
        gCapturedRoceScqDepth = channelDescs[0].roceAttr.scqDepth;
        gCapturedRoceSrcPortList = channelDescs[0].roceAttr.srcPortList;
    }
    return HCCL_SUCCESS;
}

} // namespace

TEST_F(HcommCAdptTest, ut_HcommCollectiveChannelCreate_When_Normal_Expect_Success)
{
    gCapturedChannelDescSqDepth = 0U;
    ScopedChannelAdptStubEndpoint stubEndpoint;
    EndpointHandle endpointHandle = stubEndpoint.Get();
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_UBC_TP;
    channelDesc.ubAttr.sqDepth = 300U;
    ChannelHandle channels[1] = {0};
    MOCKER(ChannelProcess::CreateChannelsLoop).stubs().will(invoke(CaptureCreateChannelsLoop));
    HcommResult ret = HcommCollectiveChannelCreate(endpointHandle, COMM_ENGINE_AIV, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(gCapturedChannelDescSqDepth, 512U);
}

TEST_F(HcommCAdptTest, ut_HcommCollectiveChannelCreate_When_ChannelDescsNull_Expect_E_PTR)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    ChannelHandle channels[1] = {0};
    HcommResult ret = HcommCollectiveChannelCreate(endpointHandle, COMM_ENGINE_AICPU_TS, nullptr, 1, channels);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommCAdptTest, ut_HcommCollectiveChannelCreate_When_ChannelNumZero_Expect_E_PARA)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0};
    HcommResult ret = HcommCollectiveChannelCreate(endpointHandle, COMM_ENGINE_AICPU_TS, &channelDesc, 0, channels);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcommCAdptTest, ut_HcommChannelCreate_When_ChannelDescsNull_Expect_E_PTR)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    ChannelHandle channels[1] = {0};
    HcommResult ret = HcommChannelCreate(endpointHandle, COMM_ENGINE_AICPU_TS, nullptr, 1, channels);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommCAdptTest, ut_HcommChannelCreate_When_ChannelNumZero_Expect_E_PARA)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0};
    HcommResult ret = HcommChannelCreate(endpointHandle, COMM_ENGINE_AICPU_TS, &channelDesc, 0, channels);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcommCAdptTest, ut_HcommEndpointGet_When_NotFound_Expect_E_NOT_FOUND)
{
    EndpointHandle handle = reinterpret_cast<EndpointHandle>(0x12345678);
    void* endpoint = nullptr;
    HcommResult ret = HcommEndpointGet(handle, &endpoint);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(HcommCAdptTest, ut_HcommEndpointGet_When_EndpointPtrNull_Expect_E_PTR)
{
    EndpointHandle handle = reinterpret_cast<EndpointHandle>(0x12345678);
    HcommResult ret = HcommEndpointGet(handle, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommCAdptTest, ut_HcommEndpointGetDescNum_When_UnsupportedDeviceType_Expect_E_NOT_SUPPORT)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910)).will(returnValue(HCCL_SUCCESS));

    uint32_t descNum = 0;
    EXPECT_EQ(HcommEndpointGetDescNum(0, &descNum), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(descNum, 0U);
}

TEST_F(HcommCAdptTest, ut_HcommEndpointGetDescNum_When_DevType960_Expect_E_NOT_SUPPORT)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_960)).will(returnValue(HCCL_SUCCESS));

    uint32_t descNum = 0;
    EXPECT_EQ(HcommEndpointGetDescNum(0, &descNum), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(descNum, 0U);
}

TEST_F(HcommCAdptTest, ut_HcommEndpointGetDescNum_When_NoEidConfiguredTwice_Expect_E_NOT_FOUND)
{
    constexpr uint32_t devicePhyId = 12U;
    unsigned int eidNum = 0U;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(devicePhyId))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::HccpHdcManager::Init).stubs().will(ignoreReturnValue());
    MOCKER(RaGetDevEidInfoNum).stubs().with(mockcpp::any(), outBoundP(&eidNum)).will(returnValue(0));
    MOCKER(RaGetDevEidInfoList).stubs().will(returnValue(0));

    uint32_t descNum = 1U;
    EXPECT_EQ(HcommEndpointGetDescNum(0, &descNum), HCCL_E_NOT_FOUND);
    EXPECT_EQ(descNum, 0U);

    descNum = 1U;
    EXPECT_EQ(HcommEndpointGetDescNum(0, &descNum), HCCL_E_NOT_FOUND);
    EXPECT_EQ(descNum, 0U);
}

namespace {
int EndpointQueryGetEidInfoListStub(struct RaInfo, struct HccpDevEidInfo infoList[], unsigned int* num)
{
    constexpr uint32_t fakeEidNum = 2U;
    if (*num < fakeEidNum) {
        return -1;
    }
    infoList[0].eid.raw[15] = 1U;
    infoList[0].eidIndex = 3U;
    infoList[1].eid.raw[15] = 2U;
    infoList[1].eidIndex = 4U;
    *num = fakeEidNum;
    return 0;
}

int EndpointQueryGetUboeEidInfoListStub(struct RaInfo, struct HccpDevEidInfo infoList[], unsigned int* num)
{
    if (*num < 1U) {
        return -1;
    }
    infoList[0].eid.raw[15] = 3U;
    infoList[0].eidIndex = 5U;
    infoList[0].devFeature = 1U << UBOE_DEV_FLAG_RIGHT_SHIFT;
    *num = 1U;
    return 0;
}

HcclResult EndpointQueryGetIpByEidStub(void* ctxHandle, const CommAddr& eidAddr, CommAddr& ipAddr)
{
    EXPECT_EQ(ctxHandle, reinterpret_cast<void*>(0x12345678));
    EXPECT_EQ(eidAddr.type, COMM_ADDR_TYPE_EID);
    ipAddr.type = COMM_ADDR_TYPE_IP_V4;
    ipAddr.addr.s_addr = htonl(0xC0A80367U);
    return HCCL_SUCCESS;
}

uint32_t gEndpointQueryCtpEnableCallCount = 0;
HcclResult EndpointQueryGetCtpEnableStub(void* ctxHandle, bool& ctpEnable)
{
    EXPECT_EQ(ctxHandle, reinterpret_cast<void*>(0x12345678));
    ctpEnable = (gEndpointQueryCtpEnableCallCount++ == 0);
    return HCCL_SUCCESS;
}
} // namespace

TEST_F(HcommCAdptTest, ut_HcommEndpointGetDescs_When_CtpAndUbg_Expect_Success)
{
    constexpr uint32_t devicePhyId = 14U;
    constexpr unsigned int fakeEidNum = 2U;
    unsigned int eidNum = fakeEidNum;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(devicePhyId))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::HccpHdcManager::Init).stubs().will(ignoreReturnValue());
    MOCKER(RaGetDevEidInfoNum).stubs().with(mockcpp::any(), outBoundP(&eidNum)).will(returnValue(0));
    MOCKER(RaGetDevEidInfoList).stubs().will(invoke(EndpointQueryGetEidInfoListStub));

    void* fakeRdmaHandle = reinterpret_cast<void*>(0x12345678);
    MOCKER_CPP(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(fakeRdmaHandle));
    gEndpointQueryCtpEnableCallCount = 0;
    MOCKER(hcomm::HccpGetCtpEnable).stubs().will(invoke(EndpointQueryGetCtpEnableStub));

    uint32_t descNum = 0;
    ASSERT_EQ(HcommEndpointGetDescNum(0, &descNum), HCCL_SUCCESS);
    ASSERT_EQ(descNum, fakeEidNum);

    EndpointDesc endpointDescs[fakeEidNum]{};
    ASSERT_EQ(HcommEndpointGetDescs(0, &descNum, endpointDescs), HCCL_SUCCESS);
    ASSERT_EQ(descNum, fakeEidNum);
    EXPECT_EQ(endpointDescs[0].protocol, COMM_PROTOCOL_UBC_CTP);
    EXPECT_EQ(endpointDescs[1].protocol, COMM_PROTOCOL_UBG);
    for (const auto& endpointDesc : endpointDescs) {
        EXPECT_EQ(endpointDesc.commAddr.type, COMM_ADDR_TYPE_EID);
        EXPECT_EQ(endpointDesc.loc.locType, ENDPOINT_LOC_TYPE_DEVICE);
        EXPECT_EQ(endpointDesc.loc.device.devPhyId, devicePhyId);
    }
}

TEST_F(HcommCAdptTest, ut_HcommEndpointGetDescs_When_Uboe_Expect_Ipv4Desc)
{
    constexpr uint32_t devicePhyId = 13U;
    unsigned int eidNum = 1U;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(devicePhyId))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::HccpHdcManager::Init).stubs().will(ignoreReturnValue());
    MOCKER(RaGetDevEidInfoNum).stubs().with(mockcpp::any(), outBoundP(&eidNum)).will(returnValue(0));
    MOCKER(RaGetDevEidInfoList).stubs().will(invoke(EndpointQueryGetUboeEidInfoListStub));
    void* fakeRdmaHandle = reinterpret_cast<void*>(0x12345678);
    MOCKER_CPP(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(fakeRdmaHandle));
    MOCKER(hcomm::HccpGetIpByEid).stubs().will(invoke(EndpointQueryGetIpByEidStub));

    uint32_t descNum = 1U;
    EndpointDesc endpointDesc{};
    ASSERT_EQ(HcommEndpointGetDescs(0, &descNum, &endpointDesc), HCCL_SUCCESS);
    EXPECT_EQ(descNum, 1U);
    EXPECT_EQ(endpointDesc.protocol, COMM_PROTOCOL_UBOE);
    EXPECT_EQ(endpointDesc.commAddr.type, COMM_ADDR_TYPE_IP_V4);
    EXPECT_EQ(endpointDesc.commAddr.addr.s_addr, htonl(0xC0A80367U));
    EXPECT_EQ(endpointDesc.loc.locType, ENDPOINT_LOC_TYPE_DEVICE);
    EXPECT_EQ(endpointDesc.loc.device.devPhyId, devicePhyId);
}

TEST_F(HcommCAdptTest, ut_HcommChannelCreate_When_NotAiCpu_Expect_Success)
{
    ScopedChannelAdptStubEndpoint stubEndpoint;
    EndpointHandle endpointHandle = stubEndpoint.Get();
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0};
    MOCKER(ChannelProcess::CreateChannelsLoop).stubs().will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommChannelCreate(endpointHandle, COMM_ENGINE_CPU, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommEngineCtxCopy_When_CPU_Expect_Success)
{
    void* dstCtx = malloc(1024);
    void* srcCtx = malloc(1024);
    ASSERT_NE(dstCtx, nullptr);
    ASSERT_NE(srcCtx, nullptr);
    memset(dstCtx, 0, 1024);
    memset(srcCtx, 1, 1024);
    HcommResult ret = HcommEngineCtxCopy(COMM_ENGINE_CPU, dstCtx, srcCtx, 1024);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    free(dstCtx);
    free(srcCtx);
}

TEST_F(HcommCAdptTest, ut_HcommEngineCtxCopy_When_CPU_TS_Expect_Success)
{
    void* dstCtx = malloc(1024);
    void* srcCtx = malloc(1024);
    ASSERT_NE(dstCtx, nullptr);
    ASSERT_NE(srcCtx, nullptr);
    memset(dstCtx, 0, 1024);
    memset(srcCtx, 1, 1024);
    HcommResult ret = HcommEngineCtxCopy(COMM_ENGINE_CPU_TS, dstCtx, srcCtx, 1024);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    free(dstCtx);
    free(srcCtx);
}

TEST_F(HcommCAdptTest, ut_HcommEngineCtxCopy_When_CCU_Expect_Success)
{
    void* dstCtx = malloc(1024);
    void* srcCtx = malloc(1024);
    ASSERT_NE(dstCtx, nullptr);
    ASSERT_NE(srcCtx, nullptr);
    memset(dstCtx, 0, 1024);
    memset(srcCtx, 1, 1024);
    HcommResult ret = HcommEngineCtxCopy(COMM_ENGINE_CCU, dstCtx, srcCtx, 1024);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    free(dstCtx);
    free(srcCtx);
}

TEST_F(HcommCAdptTest, ut_HcommChannelDescInit_When_Normal_Expect_Success)
{
    HcommChannelDesc channelDesc{};
    HcommResult ret = HcommChannelDescInit(&channelDesc, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommChannelDescInit_When_Normal_Expect_Version2AndFullSize)
{
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    EXPECT_EQ(channelDesc.header.version, HCOMM_CHANNEL_VERSION);
    EXPECT_EQ(channelDesc.header.size, sizeof(HcommChannelDesc));
    EXPECT_GE(sizeof(HcommChannelDesc), HCOMM_CHANNEL_DESC_ABI_V1_SIZE + sizeof(uint32_t));
}

TEST_F(HcommCAdptTest, ut_HcommChannelCreate_AICPU_When_CreateLoopMocked_PreAllocFail_Return_Error)
{
    // CreateChannelsLoop 被 mock 返回 SUCCESS 但未填充 targetChannels，
    // PreAllocAicpuChannels 拿到空指针 channel 返回 E_PTR
    ScopedChannelAdptStubEndpoint stubEndpoint;
    EndpointHandle endpointHandle = stubEndpoint.Get();
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0};
    MOCKER(ChannelProcess::CreateChannelsLoop).stubs().will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommChannelCreate(endpointHandle, COMM_ENGINE_AICPU, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommCAdptTest, ut_HcommCollectiveChannelCreate_CPU_Expect_Success)
{
    ScopedChannelAdptStubEndpoint stubEndpoint;
    EndpointHandle endpointHandle = stubEndpoint.Get();
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0};
    MOCKER(ChannelProcess::CreateChannelsLoop).stubs().will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommCollectiveChannelCreate(endpointHandle, COMM_ENGINE_CPU, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommCollectiveChannelCreate_CCU_Expect_Success)
{
    ScopedChannelAdptStubEndpoint stubEndpoint;
    EndpointHandle endpointHandle = stubEndpoint.Get();
    HcommChannelDesc channelDesc{};
    (void)HcommChannelDescInit(&channelDesc, 1);
    ChannelHandle channels[1] = {0};
    MOCKER(ChannelProcess::CreateChannelsLoop).stubs().will(returnValue(HCCL_SUCCESS));
    HcommResult ret = HcommCollectiveChannelCreate(endpointHandle, COMM_ENGINE_CCU, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommCollectiveChannelCreate_V1Desc_ClearsQosField)
{
    gCapturedChannelDescQos = 0U;
    ScopedChannelAdptStubEndpoint stubEndpoint;
    EndpointHandle endpointHandle = stubEndpoint.Get();
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.header.version = HCOMM_CHANNEL_VERSION_ONE;
    channelDesc.header.size = HCOMM_CHANNEL_DESC_ABI_V1_SIZE;
    channelDesc.qos = 5U;
    ChannelHandle channels[1] = {0};

    MOCKER(ChannelProcess::CreateChannelsLoop).stubs().will(invoke(CaptureCreateChannelsLoop));

    HcommResult ret = HcommCollectiveChannelCreate(endpointHandle, COMM_ENGINE_CPU, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(gCapturedChannelDescQos, 0xFFFFFFFFU);
}

TEST_F(HcommCAdptTest, ut_HcommCollectiveChannelCreate_CurrentDescWithSmallSize_KeepsFieldsByVersion)
{
    static const char channelName[] = "channel-name";
    gCapturedChannelDescQos = 0U;
    gCapturedChannelDescChannelName = nullptr;
    ScopedChannelAdptStubEndpoint stubEndpoint;
    EndpointHandle endpointHandle = stubEndpoint.Get();
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.header.version = HCOMM_CHANNEL_VERSION;
    channelDesc.header.size = HCOMM_CHANNEL_DESC_ABI_V1_SIZE;
    channelDesc.qos = 5U;
    channelDesc.channelName = channelName;
    ChannelHandle channels[1] = {0};

    MOCKER(ChannelProcess::CreateChannelsLoop).stubs().will(invoke(CaptureCreateChannelsLoop));

    HcommResult ret = HcommCollectiveChannelCreate(endpointHandle, COMM_ENGINE_CPU, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(gCapturedChannelDescQos, 5U);
    EXPECT_STREQ(gCapturedChannelDescChannelName, channelName);
}

TEST_F(HcommCAdptTest, ut_HcommResMgrInit_When_Normal_Expect_Success)
{
    HcommResult ret = HcommResMgrInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommResMgrInit_MultiDevice_Expect_Success)
{
    HcommResult ret1 = HcommResMgrInit();
    HcommResult ret2 = HcommResMgrInit();
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    EXPECT_EQ(ret2, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommEndpointGetListenPort_When_PortNull_Expect_E_PTR)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    HcommResult ret = HcommEndpointGetListenPort(endpointHandle, nullptr);
    // 新增 ENABLE_EXPERIMENTAL 路径先校验 endpoint 是否存在再校验 port 是否为空
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(HcommCAdptTest, ut_HcommEndpointGetListenPort_When_HandleInvalid_Expect_E_NOT_FOUND)
{
    uint32_t port = 0;
    HcommResult ret = HcommEndpointGetListenPort(nullptr, &port);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(HcommCAdptTest, ut_HcommEndpointGetListenPort_When_ServerSocketNotSupport_Expect_E_NOT_SUPPORT)
{
    uint32_t port = 0;
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    ChannelAdptStubEndpoint stubEndpoint(EndpointDesc{});

    // HcommEndpointMap 收编为 hcomm::EndpointMgr::Get
    MOCKER_CPP(&hcomm::EndpointMgr::Get, Endpoint * (hcomm::EndpointMgr::*)(EndpointHandle))
        .stubs()
        .will(returnValue(static_cast<Endpoint*>(&stubEndpoint)));

    HcommResult ret = HcommEndpointGetListenPort(endpointHandle, &port);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

// endpoint 不持有 ServerSocketContext（GetServerSocketContext 返回 nullptr）时：
// StartListen 历史上为 no-op SUCCESS，StopListen 历史上返回 NOT_SUPPORT（保持迁移前语义）
TEST_F(HcommCAdptTest, ut_HcommEndpointStartListen_When_NoServerSocketContext_Expect_NoOpSuccess)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    ChannelAdptStubEndpoint stubEndpoint(EndpointDesc{});

    MOCKER_CPP(&hcomm::EndpointMgr::Get, Endpoint * (hcomm::EndpointMgr::*)(EndpointHandle))
        .stubs()
        .will(returnValue(static_cast<Endpoint*>(&stubEndpoint)));

    HcommResult ret = HcommEndpointStartListen(endpointHandle, 100, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommCAdptTest, ut_HcommEndpointStopListen_When_NoServerSocketContext_Expect_E_NOT_SUPPORT)
{
    EndpointHandle endpointHandle = reinterpret_cast<EndpointHandle>(0x12345);
    ChannelAdptStubEndpoint stubEndpoint(EndpointDesc{});

    MOCKER_CPP(&hcomm::EndpointMgr::Get, Endpoint * (hcomm::EndpointMgr::*)(EndpointHandle))
        .stubs()
        .will(returnValue(static_cast<Endpoint*>(&stubEndpoint)));

    HcommResult ret = HcommEndpointStopListen(endpointHandle, 100);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(HcommCAdptTest, ut_HcommEndpointCheckFeature_When_SupportedFeature_Expect_True)
{
    EndpointDesc endpointDesc{};
    (void)memset_s(&endpointDesc, sizeof(endpointDesc), 0, sizeof(endpointDesc));
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;

    MOCKER(hcomm::CommAddrToIpAddress).stubs().will(returnValue(HCCL_SUCCESS));

    s32 devId = 0;
    MOCKER(hrtGetDevice).stubs().with(outBoundP(&devId)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Hccl::HccpPeerManager::Init).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    u32 devPhyId = 0;
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any(), outBound(devPhyId)).will(returnValue(HCCL_SUCCESS));

    void* fakeRdmaHandle = reinterpret_cast<void*>(0x12345678);
    MOCKER_CPP(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(fakeRdmaHandle));

    s32 directFlag = DIRECT_FLAG_PCIE;
    MOCKER(RaNdaGetDirectFlag).stubs().with(mockcpp::any(), outBoundP(&directFlag)).will(returnValue(0));

    bool value = false;
    HcommResult ret = HcommEndpointCheckFeature(HCOMM_ENDPOINT_FEATURE_NDA, &endpointDesc, &value);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(value, true);
}

// ===================== CheckRoceAttr 队列深度校验 UT =====================

/**
 * 场景: RoCE协议，队列深度字段均为0（使用默认值哨兵）
 * 预期: 校验通过，返回 HCCL_SUCCESS
 */
TEST_F(HcommCAdptTest, Ut_CheckRoceAttr_When_AllDepthsZero_Expect_Success)
{
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    channelDesc.qos = 0xFFFFFFFFU; // 不触发 ApplyRoceQosCompatToSlTc 的外部依赖
    channelDesc.roceAttr.sqDepth = 0U;
    channelDesc.roceAttr.scqDepth = 0U;

    HcommResult ret = CheckRoceAttr(channelDesc, ENDPOINT_LOC_TYPE_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

/**
 * 场景: RoCE协议，队列深度字段均为 INVALID_UINT（0xFFFFFFFF，使用默认值哨兵）
 * 预期: 校验通过，返回 HCCL_SUCCESS
 */
TEST_F(HcommCAdptTest, Ut_CheckRoceAttr_When_AllDepthsInvalidUint_Expect_Success)
{
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    channelDesc.qos = 0xFFFFFFFFU;
    channelDesc.roceAttr.sqDepth = 0xFFFFFFFFU;
    channelDesc.roceAttr.scqDepth = 0xFFFFFFFFU;

    HcommResult ret = CheckRoceAttr(channelDesc, ENDPOINT_LOC_TYPE_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

/**
 * 场景: RoCE协议，队列深度字段均为合法2的幂值（128/2048）
 * 预期: 校验通过，返回 HCCL_SUCCESS
 */
TEST_F(HcommCAdptTest, Ut_CheckRoceAttr_When_AllDepthsValidPowerOfTwo_Expect_Success)
{
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    channelDesc.qos = 0xFFFFFFFFU;
    channelDesc.roceAttr.sqDepth = 128U;
    channelDesc.roceAttr.scqDepth = 2048U;

    HcommResult ret = CheckRoceAttr(channelDesc, ENDPOINT_LOC_TYPE_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

/**
 * 场景: RoCE协议，sqDepth 小于最小值64（如32）
 * 预期: 校验失败，返回 HCCL_E_PARA
 */
TEST_F(HcommCAdptTest, Ut_CheckRoceAttr_When_SqDepthTooSmall_Expect_E_PARA)
{
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    channelDesc.qos = 0xFFFFFFFFU;
    channelDesc.roceAttr.sqDepth = 32U; // 小于64
    channelDesc.roceAttr.scqDepth = 0U;

    HcommResult ret = CheckRoceAttr(channelDesc, ENDPOINT_LOC_TYPE_DEVICE);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

/**
 * 场景: RoCE协议，sqDepth 等于最小值64
 * 预期: 校验通过，返回 HCCL_SUCCESS
 */
TEST_F(HcommCAdptTest, Ut_CheckRoceAttr_When_SqDepthIsMin64_Expect_Success)
{
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    channelDesc.qos = 0xFFFFFFFFU;
    channelDesc.roceAttr.sqDepth = 64U; // 等于最小值64
    channelDesc.roceAttr.scqDepth = 0U;

    HcommResult ret = CheckRoceAttr(channelDesc, ENDPOINT_LOC_TYPE_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

/**
 * 场景: RoCE协议，sqDepth 大于最大值32768（如65536）
 * 预期: 校验失败，返回 HCCL_E_PARA
 */
TEST_F(HcommCAdptTest, Ut_CheckRoceAttr_When_SqDepthTooLarge_Expect_E_PARA)
{
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    channelDesc.qos = 0xFFFFFFFFU;
    channelDesc.roceAttr.sqDepth = 65536U; // 大于32768
    channelDesc.roceAttr.scqDepth = 0U;

    HcommResult ret = CheckRoceAttr(channelDesc, ENDPOINT_LOC_TYPE_DEVICE);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

/**
 * 场景: RoCE协议，sqDepth 非2的幂（如300）
 * 预期: 校验失败，返回 HCCL_E_PARA
 */
TEST_F(HcommCAdptTest, Ut_CheckRoceAttr_When_SqDepthNotPowerOfTwo_Expect_E_PARA)
{
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    channelDesc.qos = 0xFFFFFFFFU;
    channelDesc.roceAttr.sqDepth = 300U; // 非2的幂
    channelDesc.roceAttr.scqDepth = 0U;

    HcommResult ret = CheckRoceAttr(channelDesc, ENDPOINT_LOC_TYPE_DEVICE);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

/**
 * 场景: 非RoCE协议（如HCCS），深度字段为非法值
 * 预期: 不触发深度校验，返回 HCCL_SUCCESS
 */
TEST_F(HcommCAdptTest, Ut_CheckRoceAttr_When_NonRoCEProtocol_Expect_Success_NoDepthCheck)
{
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_HCCS; // 非RoCE协议
    channelDesc.roceAttr.sqDepth = 64U;                       // 非法值，但不应被校验

    HcommResult ret = CheckRoceAttr(channelDesc, ENDPOINT_LOC_TYPE_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ===================== ProcessHcommChannelDescs v4 兼容性 UT =====================

/**
 * 场景: 低版本(VERSION_ONE)desc，设置合法的队列深度值
 * 预期: 经 ProcessHcommChannelDescs 处理后，深度字段被置为 INVALID_UINT（默认哨兵）
 */
TEST_F(HcommCAdptTest, Ut_HcommCollectiveChannelCreate_LowerVersionDesc_ClearsRoceQueueDepth)
{
    gCapturedRoceSqDepth = 0U;
    gCapturedRoceScqDepth = 0U;
    ScopedChannelAdptStubEndpoint stubEndpoint;
    EndpointHandle endpointHandle = stubEndpoint.Get();
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.header.version = HCOMM_CHANNEL_VERSION_ONE;
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    channelDesc.roceAttr.sqDepth = 256U;
    channelDesc.roceAttr.scqDepth = 1024U;
    ChannelHandle channels[1] = {0};

    MOCKER(ChannelProcess::CreateChannelsLoop).stubs().will(invoke(CaptureCreateChannelsLoop));

    HcommResult ret = HcommCollectiveChannelCreate(endpointHandle, COMM_ENGINE_CPU, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 低版本 desc 经处理后，深度字段应被置为 INVALID_UINT（0xFFFFFFFF）
    EXPECT_EQ(gCapturedRoceSqDepth, 0xFFFFFFFFU);
    EXPECT_EQ(gCapturedRoceScqDepth, 0xFFFFFFFFU);
}

/**
 * 场景: 当前版本desc，设置合法的队列深度值
 * 预期: 经 ProcessHcommChannelDescs 处理后，深度字段保持输入值（透传）
 */
TEST_F(HcommCAdptTest, Ut_HcommCollectiveChannelCreate_CurrentVersionDesc_KeepsRoceQueueDepth)
{
    gCapturedRoceSqDepth = 0U;
    gCapturedRoceScqDepth = 0U;
    ScopedChannelAdptStubEndpoint stubEndpoint;
    EndpointHandle endpointHandle = stubEndpoint.Get();
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.header.version = HCOMM_CHANNEL_VERSION;
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    channelDesc.roceAttr.sqDepth = 128U;
    channelDesc.roceAttr.scqDepth = 1024U;
    ChannelHandle channels[1] = {0};

    MOCKER(ChannelProcess::CreateChannelsLoop).stubs().will(invoke(CaptureCreateChannelsLoop));

    HcommResult ret = HcommCollectiveChannelCreate(endpointHandle, COMM_ENGINE_CPU, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 当前版本 desc 经处理后，深度字段应保持输入值
    EXPECT_EQ(gCapturedRoceSqDepth, 128U);
    EXPECT_EQ(gCapturedRoceScqDepth, 1024U);
}

/**
 * 场景: 当前版本desc，设置 srcPortList 指针
 * 预期: 经 ProcessHcommChannelDescs 处理后，srcPortList 保持输入值（透传）
 */
TEST_F(HcommCAdptTest, Ut_HcommCollectiveChannelCreate_V4Desc_KeepsSrcPortList)
{
    uint16_t portList[2] = {100, 200};
    gCapturedRoceSrcPortList = nullptr;
    ScopedChannelAdptStubEndpoint stubEndpoint;
    EndpointHandle endpointHandle = stubEndpoint.Get();
    HcommChannelDesc channelDesc{};
    ASSERT_EQ(HcommChannelDescInit(&channelDesc, 1), HCCL_SUCCESS);
    channelDesc.header.version = HCOMM_CHANNEL_VERSION;
    channelDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    channelDesc.roceAttr.srcPortList = portList;
    ChannelHandle channels[1] = {0};

    MOCKER(ChannelProcess::CreateChannelsLoop).stubs().will(invoke(CaptureCreateChannelsLoop));

    HcommResult ret = HcommCollectiveChannelCreate(endpointHandle, COMM_ENGINE_CPU, &channelDesc, 1, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 当前版本 desc 的 srcPortList 应被正确透传
    EXPECT_EQ(gCapturedRoceSrcPortList, portList);
}
