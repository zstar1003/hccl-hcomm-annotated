/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mockcpp/mockcpp.hpp>

#include "gtest/gtest.h"
#include "ccu_device_pub.h"
#include "ccu_instance_mgr.h"
#include "ccu_kernel.h"
#include "ccu_kernel_mgr.h"
#include "ccu_primitives.hpp"
#include "ccu_res.h"
#include "ccu_urma_channel.h"
#include "hcom_common.h"
#include "hcomm_c_adpt.h"
#include "op_base.h"
#include "src/base_comm/resources/ccu/ccu_device/ccu_res_batch_allocator.h"
#include "ccu_pfe_cfg_mgr.h"
#include "src/base_comm/resources/ccu/ccu_device/ccu_res_specs.h"
#include "ccu_comp.h"
#include "mocks/ccu_device_mock_utils.h"

namespace {
constexpr int32_t TEST_DEVICE_LOGIC_ID = 0;
constexpr int32_t OTHER_TEST_DEVICE_LOGIC_ID = 1;
constexpr uint32_t VALID_DIE_ID = 0;
constexpr uint32_t OTHER_VALID_DIE_ID = 1;
constexpr uint32_t INVALID_DIE_ID = hcomm::CCU_MAX_IODIE_NUM;
constexpr HcommCcuResDescHandle MISSING_DESC_HANDLE = 0xBADCAFEULL;

uint32_t g_noArgKernelCalls = 0;
uint32_t g_oneArgKernelCalls = 0;
uint32_t g_channelGetCalls = 0;
uint32_t g_mockLocCkeId = 10;
uint32_t g_mockLocXnId = 20;
CcuKernelArg g_lastKernelArg = nullptr;
CcuResult g_kernelReturn = CcuResult::CCU_SUCCESS;
std::vector<std::pair<Hccl::ResType::Value, uint32_t>> g_setResCalls;
uint32_t g_failSetResCall = 0;
std::unordered_map<ChannelHandle, hcomm::CcuUrmaChannel*> g_channels;
int32_t g_runtimeDeviceLogicId = TEST_DEVICE_LOGIC_ID;
int32_t g_kernelObservedDeviceLogicId = INVALID_INT;
uint32_t g_deviceRefreshCalls = 0;
HcclResult g_deviceRefreshResult = HcclResult::HCCL_SUCCESS;

struct ChannelQueryArg {
    ChannelHandle first;
    ChannelHandle second;
};

HcclResult MockHrtGetDeviceRefresh(int32_t* deviceLogicId)
{
    ++g_deviceRefreshCalls;
    if (g_deviceRefreshResult != HcclResult::HCCL_SUCCESS) {
        return g_deviceRefreshResult;
    }
    if (deviceLogicId != nullptr) {
        *deviceLogicId = g_runtimeDeviceLogicId;
        return HcclResult::HCCL_SUCCESS;
    }
    return HcclResult::HCCL_E_PTR;
}

HcclResult MockHrtGetDevicePhyIdByUserDevId(uint32_t deviceLogicId, uint32_t& devicePhyId, bool)
{
    devicePhyId = deviceLogicId;
    return HcclResult::HCCL_SUCCESS;
}

HcommResult MockChannelGetNotFound(ChannelHandle, void** channel)
{
    ++g_channelGetCalls;
    if (channel != nullptr) {
        *channel = nullptr;
    }
    return HCCL_E_NOT_FOUND;
}

HcommResult MockChannelGet(ChannelHandle handle, void** channel)
{
    ++g_channelGetCalls;
    if (channel == nullptr) {
        return HCCL_E_PTR;
    }
    auto it = g_channels.find(handle);
    if (it == g_channels.end()) {
        *channel = nullptr;
        return HCCL_E_NOT_FOUND;
    }
    *channel = it->second;
    return HCCL_SUCCESS;
}

hcomm::CcuResReq MakeBlockAndNonBlockResReq(uint32_t dieId = VALID_DIE_ID)
{
    hcomm::CcuResReq resReq{};
    resReq.loopEngineReq[dieId] = 1;
    resReq.blockLoopEngineReq[dieId] = 6;
    resReq.msReq[dieId] = 3;
    resReq.blockMsReq[dieId] = 5;
    resReq.xnReq[dieId] = 7;
    resReq.blockXnReq[dieId] = 8;
    resReq.gsaReq[dieId] = 9;
    resReq.blockGsaReq[dieId] = 10;
    resReq.ckeReq[dieId] = 2;
    resReq.blockCkeReq[dieId] = 4;
    resReq.missionReq.req[dieId] = 1;
    return resReq;
}

// 验证无参 Kernel 在 dry-run 中被执行并返回可控结果。
CcuResult NoArgKernel()
{
    ++g_noArgKernelCalls;
    return g_kernelReturn;
}

// 验证单参 Kernel 在 dry-run 中接收调用者传入的参数并返回可控结果。
CcuResult OneArgKernel(CcuKernelArg arg)
{
    ++g_oneArgKernelCalls;
    g_lastKernelArg = arg;
    return g_kernelReturn;
}

// 验证 Device 刷新后真实 primitive 与查询入口使用同一个线程 Device。
CcuResult DeviceRefreshPrimitiveKernel()
{
    ++g_noArgKernelCalls;
    g_kernelObservedDeviceLogicId = HcclGetThreadDeviceId();
    AscendC::ccu::Variable scalar;
    scalar = 1;
    return CcuResult::CCU_SUCCESS;
}

// 验证 Kernel 在 dry-run 中抛出 CCU 异常时查询接口正确返回错误。
CcuResult ThrowingKernel(CcuKernelArg)
{
    CCU_THROW_IF_FAILED(CcuResult::CCU_E_INTERNAL, "query resource test exception");
    return CcuResult::CCU_SUCCESS;
}

// 验证各类 CC primitive 在 dry-run 中产生的资源诉求被完整统计。
CcuResult ResourceCensusKernel(CcuKernelArg arg)
{
    namespace ccu = AscendC::ccu;
    auto* channel = static_cast<ChannelHandle*>(arg);
    if (channel == nullptr) {
        return CcuResult::CCU_E_PTR;
    }

    ccu::Variable scalar;
    ccu::Array<ccu::Variable> vars(2);
    ccu::Address address;
    ccu::Event event;
    ccu::Array<ccu::Event> blockEvents(2);
    ccu::CcuBuffer buffer;
    ccu::Array<ccu::CcuBuffer> blockBuffers(2);
    ccu::LocalAddr local;
    ccu::RemoteAddr remote;

    scalar = 1;
    address = 0x1000;

    CcuResult ret = ccu::Load(0x2000, vars, 2);
    if (ret != CcuResult::CCU_SUCCESS) {
        return ret;
    }
    ret = ccu::Store(0x3000, vars, 2);
    if (ret != CcuResult::CCU_SUCCESS) {
        return ret;
    }
    ret = ccu::EventRecord(event);
    if (ret != CcuResult::CCU_SUCCESS) {
        return ret;
    }
    ret = ccu::EventWait(event);
    if (ret != CcuResult::CCU_SUCCESS) {
        return ret;
    }

    auto channelVar = ccu::GetResByChannel<ccu::Variable>(*channel, 0);
    ret = ccu::NotifyRecord(*channel, 0, 1);
    if (ret != CcuResult::CCU_SUCCESS) {
        return ret;
    }
    ret = ccu::NotifyWait(*channel, 0, 1);
    if (ret != CcuResult::CCU_SUCCESS) {
        return ret;
    }
    ret = ccu::WriteVariableWithNotify(*channel, channelVar, 0, 0, 1);
    if (ret != CcuResult::CCU_SUCCESS) {
        return ret;
    }
    ret = ccu::Read(*channel, local, remote, scalar, event);
    if (ret != CcuResult::CCU_SUCCESS) {
        return ret;
    }
    ret = ccu::Read(*channel, buffer, remote, scalar, event);
    if (ret != CcuResult::CCU_SUCCESS) {
        return ret;
    }
    ret = ccu::ReadReduce(*channel, local, remote, scalar, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, event);
    if (ret != CcuResult::CCU_SUCCESS) {
        return ret;
    }
    ret = ccu::Write(*channel, remote, local, scalar, event);
    if (ret != CcuResult::CCU_SUCCESS) {
        return ret;
    }
    ret = ccu::Write(*channel, remote, buffer, scalar, event);
    if (ret != CcuResult::CCU_SUCCESS) {
        return ret;
    }
    ret = ccu::WriteReduce(*channel, remote, local, scalar, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, event);
    if (ret != CcuResult::CCU_SUCCESS) {
        return ret;
    }

    ccu::Func body([]() {
        ccu::Variable loopVar;
        loopVar = 1;
    });
    ccu::LoopConfig loopCfg{};
    loopCfg.iterNum = 1;
    ccu::Loop loop(loopCfg, body);
    ccu::LoopGroupConfig groupCfg{};
    groupCfg.cloneNum = 1;
    ccu::LoopGroup group(groupCfg, 1, {loop});

    (void)blockEvents;
    (void)buffer;
    (void)blockBuffers;
    (void)local;
    (void)remote;
    (void)group;
    return CcuResult::CCU_SUCCESS;
}

// 验证非法 ChannelHandle 在 dry-run 中触发预期错误。
CcuResult InvalidChannelKernel(CcuKernelArg arg)
{
    auto* channel = static_cast<ChannelHandle*>(arg);
    if (channel == nullptr) {
        return CcuResult::CCU_E_PTR;
    }
    return AscendC::ccu::NotifyWait(*channel, 0, 1);
}

// 验证依赖 ChannelHandle 的 primitive 在 dry-run 中正确解析并统计资源。
CcuResult ChannelBackedKernel(CcuKernelArg arg)
{
    auto* queryArg = static_cast<ChannelQueryArg*>(arg);
    if (queryArg == nullptr) {
        return CcuResult::CCU_E_PTR;
    }

    auto channelVar = AscendC::ccu::GetResByChannel<AscendC::ccu::Variable>(queryArg->first, 0);
    CcuResult ret = AscendC::ccu::NotifyWait(queryArg->first, 0, 1);
    if (ret != CcuResult::CCU_SUCCESS) {
        return ret;
    }
    if (queryArg->second != 0) {
        ret = AscendC::ccu::NotifyRecord(queryArg->second, 0, 1);
        if (ret != CcuResult::CCU_SUCCESS) {
            return ret;
        }
    }

    AscendC::ccu::Array<AscendC::ccu::Variable> continuousVars(2);
    (void)channelVar;
    return AscendC::ccu::Load(0x1000, continuousVars, 2);
}

// g_failSetResCall 指定第几次 SetResNum 调用返回 CCU_E_INTERNAL；为 0 时不注入失败。
CcuResult
CaptureSetResNum(hcomm::CcuResDescMgr* mgr, HcommCcuResDescHandle handle, Hccl::ResType resType, uint32_t resNum)
{
    g_setResCalls.emplace_back(static_cast<Hccl::ResType::Value>(resType), resNum);
    if (g_failSetResCall != 0 && g_setResCalls.size() == g_failSetResCall) {
        return CcuResult::CCU_E_INTERNAL;
    }
    auto* desc = const_cast<hcomm::CcuResDesc*>(mgr->Get(handle));
    if (desc == nullptr) {
        return CcuResult::CCU_E_NOT_FOUND;
    }
    return desc->SetResNum(resType, resNum);
}
} // namespace

class HcommCcuKernelQueryResReqTest : public testing::Test {
public:
    void SetUp() override
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
        ResetDeviceState(TEST_DEVICE_LOGIC_ID);
        ResetDeviceState(OTHER_TEST_DEVICE_LOGIC_ID);
        g_noArgKernelCalls = 0;
        g_oneArgKernelCalls = 0;
        g_channelGetCalls = 0;
        g_lastKernelArg = nullptr;
        g_kernelReturn = CcuResult::CCU_SUCCESS;
        g_setResCalls.clear();
        g_failSetResCall = 0;
        g_channels.clear();
        g_runtimeDeviceLogicId = TEST_DEVICE_LOGIC_ID;
        g_kernelObservedDeviceLogicId = INVALID_INT;
        g_deviceRefreshCalls = 0;
        g_deviceRefreshResult = HcclResult::HCCL_SUCCESS;

        MOCKER(hrtGetDeviceRefresh).stubs().with(mockcpp::any()).will(invoke(MockHrtGetDeviceRefresh));
        int32_t seededDeviceLogicId = INVALID_INT;
        ASSERT_EQ(HcclDeviceRefresh(seededDeviceLogicId), HcclResult::HCCL_SUCCESS);
        ASSERT_EQ(seededDeviceLogicId, TEST_DEVICE_LOGIC_ID);
        ASSERT_EQ(HcclGetThreadDeviceId(), TEST_DEVICE_LOGIC_ID);
        g_deviceRefreshCalls = 0;

        ASSERT_EQ(ResDescMgr().Create(VALID_DIE_ID, resDesc_), CcuResult::CCU_SUCCESS);
        ASSERT_NE(resDesc_, 0U);
        int32_t fakeDeviceLogicId = static_cast<int32_t>(TEST_DEVICE_LOGIC_ID);
        MOCKER(hrtGetDevice).stubs().with(outBoundP(&fakeDeviceLogicId)).will(returnValue(HcclResult::HCCL_SUCCESS));
        MOCKER(hrtGetDevicePhyIdByIndex)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(invoke(MockHrtGetDevicePhyIdByUserDevId));
        constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
        MockCcuNetworkDeviceDefault(fakeDeviceLogicId);
        EXPECT_EQ(MockCcuResourcesDefault(fakeDeviceLogicId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(hcomm::CcuKernelMgr::GetInstance(TEST_DEVICE_LOGIC_ID).Init(), HcclResult::HCCL_SUCCESS);
    }

    void TearDown() override
    {
        g_runtimeDeviceLogicId = TEST_DEVICE_LOGIC_ID;
        g_deviceRefreshResult = HcclResult::HCCL_SUCCESS;
        int32_t restoredDeviceLogicId = INVALID_INT;
        EXPECT_EQ(HcclDeviceRefresh(restoredDeviceLogicId), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(restoredDeviceLogicId, TEST_DEVICE_LOGIC_ID);
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
        ResetDeviceState(TEST_DEVICE_LOGIC_ID);
        ResetDeviceState(OTHER_TEST_DEVICE_LOGIC_ID);
    }

protected:
    static hcomm::CcuResDescMgr& ResDescMgr(int32_t deviceLogicId = TEST_DEVICE_LOGIC_ID)
    {
        return hcomm::CcuInstanceMgr::GetInstance(deviceLogicId).GetResDescMgr();
    }

    static void ResetDeviceState(int32_t deviceLogicId)
    {
        (void)hcomm::CcuInstanceMgr::GetInstance(deviceLogicId).Deinit();
        (void)hcomm::CcuKernelMgr::GetInstance(deviceLogicId).Deinit();
        (void)hcomm::CcuResBatchAllocator::GetInstance(deviceLogicId).Deinit();
        (void)hcomm::CcuComponent::GetInstance(deviceLogicId).Deinit();
        (void)hcomm::CcuPfeCfgMgr::GetInstance(deviceLogicId).Deinit();
        (void)hcomm::CcuResSpecifications::GetInstance(deviceLogicId).Deinit();
    }

    static void InitAdditionalKernelDevice(int32_t deviceLogicId)
    {
        constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
        ASSERT_EQ(InitMockCcuResourcesForDevice(deviceLogicId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
        ASSERT_EQ(hcomm::CcuKernelMgr::GetInstance(deviceLogicId).Init(), HcclResult::HCCL_SUCCESS);
    }

    static HcommCcuResDescHandle CreateDesc(int32_t deviceLogicId)
    {
        HcommCcuResDescHandle handle = 0;
        EXPECT_EQ(ResDescMgr(deviceLogicId).Create(VALID_DIE_ID, handle), CcuResult::CCU_SUCCESS);
        EXPECT_NE(handle, 0U);
        return handle;
    }

    void MockKernelOutputs(const hcomm::CcuResReq& resReq = MakeBlockAndNonBlockResReq(), uint32_t instrNum = 8)
    {
        MOCKER_CPP(&hcomm::CcuKernel::GetResourceRequest).stubs().will(returnValue(resReq));
        MOCKER_CPP(&hcomm::CcuKernel::GetInstrCount).stubs().will(returnValue(instrNum));
    }

    void MockChannels(bool differentDies = false)
    {
        MOCKER(HcommChannelGet).stubs().will(invoke(MockChannelGet));
        if (differentDies) {
            MOCKER_CPP(&hcomm::CcuUrmaChannel::GetDieId)
                .expects(exactly(2))
                .will(returnValue(VALID_DIE_ID))
                .then(returnValue(OTHER_VALID_DIE_ID));
        } else {
            MOCKER_CPP(&hcomm::CcuUrmaChannel::GetDieId).stubs().will(returnValue(VALID_DIE_ID));
        }
        MOCKER_CPP(&hcomm::CcuUrmaChannel::GetChannelId).expects(atLeast(1)).will(returnValue(100U));
        MOCKER_CPP(&hcomm::CcuUrmaChannel::GetLocCkeByIndex)
            .expects(atLeast(1))
            .with(mockcpp::any(), outBound(g_mockLocCkeId))
            .will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&hcomm::CcuUrmaChannel::GetLocXnByIndex)
            .expects(atLeast(1))
            .with(mockcpp::any(), outBound(g_mockLocXnId))
            .will(returnValue(HCCL_SUCCESS));
    }

    void PrefillAll(uint32_t base) { PrefillAll(TEST_DEVICE_LOGIC_ID, resDesc_, base); }

    static void PrefillAll(int32_t deviceLogicId, HcommCcuResDescHandle handle, uint32_t base)
    {
        const std::array<Hccl::ResType, 7> types
            = {Hccl::ResType::LOOP, Hccl::ResType::MS,      Hccl::ResType::XN, Hccl::ResType::GSA,
               Hccl::ResType::CKE,  Hccl::ResType::MISSION, Hccl::ResType::INS};
        for (size_t i = 0; i < types.size(); ++i) {
            ASSERT_EQ(ResDescMgr(deviceLogicId).SetResNum(handle, types[i], base + i), CcuResult::CCU_SUCCESS);
        }
    }

    uint32_t QueryNum(Hccl::ResType type) { return QueryNum(TEST_DEVICE_LOGIC_ID, resDesc_, type); }

    static uint32_t QueryNum(int32_t deviceLogicId, HcommCcuResDescHandle handle, Hccl::ResType type)
    {
        uint32_t num = 0;
        EXPECT_EQ(ResDescMgr(deviceLogicId).QueryResNum(handle, type, num), CcuResult::CCU_SUCCESS);
        return num;
    }

    void ExpectAll(uint32_t loop, uint32_t ms, uint32_t xn, uint32_t gsa, uint32_t cke, uint32_t mission, uint32_t ins)
    {
        EXPECT_EQ(QueryNum(Hccl::ResType::LOOP), loop);
        EXPECT_EQ(QueryNum(Hccl::ResType::MS), ms);
        EXPECT_EQ(QueryNum(Hccl::ResType::XN), xn);
        EXPECT_EQ(QueryNum(Hccl::ResType::GSA), gsa);
        EXPECT_EQ(QueryNum(Hccl::ResType::CKE), cke);
        EXPECT_EQ(QueryNum(Hccl::ResType::MISSION), mission);
        EXPECT_EQ(QueryNum(Hccl::ResType::INS), ins);
    }

    static void ExpectAll(
        int32_t deviceLogicId, HcommCcuResDescHandle handle, uint32_t loop, uint32_t ms, uint32_t xn, uint32_t gsa,
        uint32_t cke, uint32_t mission, uint32_t ins)
    {
        EXPECT_EQ(QueryNum(deviceLogicId, handle, Hccl::ResType::LOOP), loop);
        EXPECT_EQ(QueryNum(deviceLogicId, handle, Hccl::ResType::MS), ms);
        EXPECT_EQ(QueryNum(deviceLogicId, handle, Hccl::ResType::XN), xn);
        EXPECT_EQ(QueryNum(deviceLogicId, handle, Hccl::ResType::GSA), gsa);
        EXPECT_EQ(QueryNum(deviceLogicId, handle, Hccl::ResType::CKE), cke);
        EXPECT_EQ(QueryNum(deviceLogicId, handle, Hccl::ResType::MISSION), mission);
        EXPECT_EQ(QueryNum(deviceLogicId, handle, Hccl::ResType::INS), ins);
    }

    HcommCcuResDescHandle resDesc_{0};
};

// 验证无参 Kernel dry-run 成功后，接口正确写入七类资源诉求。
TEST_F(HcommCcuKernelQueryResReqTest, Ut_HcommCcuKernelQueryResReq_When_ArgNumIsZero_Expect_ReturnIsCCU_SUCCESS)
{
    MockKernelOutputs();
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(NoArgKernel), nullptr, 0, resDesc_),
        CcuResult::CCU_SUCCESS);
    EXPECT_EQ(g_noArgKernelCalls, 1U);
    ExpectAll(7, 8, 15, 19, 6, 1, 12);
}

// 验证单参 Kernel 接收正确入参并将 dry-run 资源诉求写入描述符。
TEST_F(HcommCcuKernelQueryResReqTest, Ut_HcommCcuKernelQueryResReq_When_ArgNumIsOne_Expect_ReturnIsCCU_SUCCESS)
{
    MockKernelOutputs();
    int32_t kernelArg = 7;
    const void* kernelArgs[] = {&kernelArg};
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(OneArgKernel), kernelArgs, 1, resDesc_),
        CcuResult::CCU_SUCCESS);
    EXPECT_EQ(g_oneArgKernelCalls, 1U);
    EXPECT_EQ(g_lastKernelArg, &kernelArg);
    ExpectAll(7, 8, 15, 19, 6, 1, 12);
}

// 验证阻塞与非阻塞资源正确聚合，并按约定顺序写入七类资源数量。
TEST_F(
    HcommCcuKernelQueryResReqTest,
    Ut_HcommCcuKernelQueryResReq_When_ResourceRequestHasBlockAndNonBlock_Expect_SetAggregatedResNumsInOrder)
{
    MockKernelOutputs();
    MOCKER_CPP(&hcomm::CcuResDescMgr::SetResNum).expects(exactly(7)).will(invoke(CaptureSetResNum));
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(NoArgKernel), nullptr, 0, resDesc_),
        CcuResult::CCU_SUCCESS);
    const std::vector<std::pair<Hccl::ResType::Value, uint32_t>> expected
        = {{Hccl::ResType::LOOP, 7}, {Hccl::ResType::MS, 8},      {Hccl::ResType::XN, 15}, {Hccl::ResType::GSA, 19},
           {Hccl::ResType::CKE, 6},  {Hccl::ResType::MISSION, 1}, {Hccl::ResType::INS, 12}};
    EXPECT_EQ(g_setResCalls, expected);
}

// 验证 Kernel 实际执行基于 Channel 的资源查询路径并获取资源诉求。
TEST_F(
    HcommCcuKernelQueryResReqTest,
    Ut_HcommCcuKernelQueryResReq_When_KernelUsesAcquiredChannel_Expect_ExecuteChannelPathAndGetResReq)
{
    HcommChannelDesc channelDesc{};
    hcomm::CcuUrmaChannel channel(nullptr, channelDesc);
    constexpr ChannelHandle channelHandle = 0xCCAA5501ULL;
    g_channels[channelHandle] = &channel;
    MockChannels();

    ChannelQueryArg queryArg{channelHandle, 0};
    const void* kernelArgs[] = {&queryArg};
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(ChannelBackedKernel), kernelArgs, 1, resDesc_),
        CcuResult::CCU_SUCCESS);
    EXPECT_GT(g_channelGetCalls, 0U);
    EXPECT_GT(QueryNum(Hccl::ResType::XN), 0U);
    EXPECT_EQ(QueryNum(Hccl::ResType::CKE), 0U);
    EXPECT_GT(QueryNum(Hccl::ResType::INS), 0U);
    EXPECT_EQ(hcomm::CcuKernelMgr::GetInstance(TEST_DEVICE_LOGIC_ID).GetCurrentKernel(), nullptr);
}

// 验证查询结果覆盖描述符中的旧资源数量，同时保持 dieId 不变。
TEST_F(
    HcommCcuKernelQueryResReqTest,
    Ut_HcommCcuKernelQueryResReq_When_DescriptorHasExistingValues_Expect_OverwriteAllResNumsAndKeepDieId)
{
    PrefillAll(100);
    MockKernelOutputs();
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(NoArgKernel), nullptr, 0, resDesc_),
        CcuResult::CCU_SUCCESS);
    ExpectAll(7, 8, 15, 19, 6, 1, 12);
    const hcomm::CcuResDesc* desc = ResDescMgr().Get(resDesc_);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->dieId, VALID_DIE_ID);
}

// 验证不打资源统计桩时，本地及 ChannelHandle primitive 的真实资源诉求和指令数均可被精确统计。
TEST_F(
    HcommCcuKernelQueryResReqTest,
    Ut_HcommCcuKernelQueryResReq_When_KernelUsesMultiplePrimitives_Expect_ExactRealResourceRequirements)
{
    HcommChannelDesc channelDesc{};
    hcomm::CcuUrmaChannel channel(nullptr, channelDesc);
    constexpr ChannelHandle channelHandle = 0xCCAA5501ULL;
    g_channels[channelHandle] = &channel;
    MockChannels();

    const void* kernelArgs[] = {&channelHandle};
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(ResourceCensusKernel), kernelArgs, 1, resDesc_),
        CcuResult::CCU_SUCCESS);
    EXPECT_GT(g_channelGetCalls, 0U);
    // ins=40: GetRepNeedToAddLatency 统计三种 wait 类 rep (LOC_WAIT_EVENT / LOC_WAIT_NOTIFY /
    // REM_WAIT_SEM) 并按 CCU_CKE_RAW_LATENCY 预留. ResourceCensusKernel 顶层的 EventWait /
    // NotifyWait 走 record/wait 高层 API, 未在 GetRepSequence 里落成上述三种 rep, 故预留数为 0,
    // 指令总数为无预留时的基准 40.
    ExpectAll(1, 3, 9, 3, 3, 1, 40);
    EXPECT_EQ(hcomm::CcuKernelMgr::GetInstance(TEST_DEVICE_LOGIC_ID).GetCurrentKernel(), nullptr);
}

// 验证 kernelFunc 为空时返回 CCU_E_PTR，且不修改描述符原有资源。
TEST_F(HcommCcuKernelQueryResReqTest, Ut_HcommCcuKernelQueryResReq_When_KernelFuncIsNull_Expect_ReturnIsCCU_E_PTR)
{
    PrefillAll(100);
    EXPECT_EQ(HcommCcuKernelQueryResReq(nullptr, nullptr, 0, resDesc_), CcuResult::CCU_E_PTR);
    ExpectAll(100, 101, 102, 103, 104, 105, 106);
}

// 验证资源描述符句柄为 0 时返回 CCU_E_PARA，且不执行 Kernel。
TEST_F(HcommCcuKernelQueryResReqTest, Ut_HcommCcuKernelQueryResReq_When_ResDescHandleIsZero_Expect_ReturnIsCCU_E_PARA)
{
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(NoArgKernel), nullptr, 0, 0), CcuResult::CCU_E_PARA);
    EXPECT_EQ(g_noArgKernelCalls, 0U);
}

// 验证单参调用的 kernelArgs 为空时返回 CCU_E_PTR，且不执行 Kernel。
TEST_F(
    HcommCcuKernelQueryResReqTest,
    Ut_HcommCcuKernelQueryResReq_When_ArgNumIsOneAndKernelArgsIsNull_Expect_ReturnIsCCU_E_PTR)
{
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(OneArgKernel), nullptr, 1, resDesc_),
        CcuResult::CCU_E_PTR);
    EXPECT_EQ(g_oneArgKernelCalls, 0U);
}

// 验证单参调用的 kernelArgs[0] 为空时返回 CCU_E_PTR，且不执行 Kernel。
TEST_F(
    HcommCcuKernelQueryResReqTest,
    Ut_HcommCcuKernelQueryResReq_When_ArgNumIsOneAndKernelArg0IsNull_Expect_ReturnIsCCU_E_PTR)
{
    const void* kernelArgs[] = {nullptr};
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(OneArgKernel), kernelArgs, 1, resDesc_),
        CcuResult::CCU_E_PTR);
    EXPECT_EQ(g_oneArgKernelCalls, 0U);
}

// 验证 argNum 大于 1 时返回 CCU_E_PARA，且不执行 Kernel。
TEST_F(HcommCcuKernelQueryResReqTest, Ut_HcommCcuKernelQueryResReq_When_ArgNumGreaterThanOne_Expect_ReturnIsCCU_E_PARA)
{
    int32_t arg0 = 0;
    int32_t arg1 = 1;
    const void* kernelArgs[] = {&arg0, &arg1};
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(OneArgKernel), kernelArgs, 2, resDesc_),
        CcuResult::CCU_E_PARA);
    EXPECT_EQ(g_oneArgKernelCalls, 0U);
}

// 验证描述符中的 dieId 非法时返回 CCU_E_PARA，且不执行 dry-run。
TEST_F(HcommCcuKernelQueryResReqTest, Ut_HcommCcuKernelQueryResReq_When_ResDescDieIdIsInvalid_Expect_ReturnIsCCU_E_PARA)
{
    ResDescMgr().Deinit();
    ASSERT_EQ(ResDescMgr().Create(INVALID_DIE_ID, resDesc_), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(NoArgKernel), nullptr, 0, resDesc_),
        CcuResult::CCU_E_PARA);
    EXPECT_EQ(g_noArgKernelCalls, 0U);
}

// 验证 Kernel dry-run 失败时透传错误、清理临时 Kernel 且不修改描述符。
TEST_F(
    HcommCcuKernelQueryResReqTest,
    Ut_HcommCcuKernelQueryResReq_When_KernelDryRunFails_Expect_ReturnErrorAndCleanTempKernel)
{
    PrefillAll(100);
    g_kernelReturn = CcuResult::CCU_E_INTERNAL;
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(NoArgKernel), nullptr, 0, resDesc_),
        CcuResult::CCU_E_INTERNAL);
    EXPECT_EQ(g_noArgKernelCalls, 1U);
    EXPECT_EQ(hcomm::CcuKernelMgr::GetInstance(TEST_DEVICE_LOGIC_ID).GetCurrentKernel(), nullptr);
    ExpectAll(100, 101, 102, 103, 104, 105, 106);
}

// 验证资源描述符不存在时返回 CCU_E_NOT_FOUND，且不执行 Kernel。
TEST_F(HcommCcuKernelQueryResReqTest, Ut_HcommCcuKernelQueryResReq_When_ResDescNotFound_Expect_ReturnIsCCU_E_NOT_FOUND)
{
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(NoArgKernel), nullptr, 0, MISSING_DESC_HANDLE),
        CcuResult::CCU_E_NOT_FOUND);
    EXPECT_EQ(g_noArgKernelCalls, 0U);
}

// 验证后续资源写入失败时立即返回错误，并保留此前已成功写入的资源。
TEST_F(
    HcommCcuKernelQueryResReqTest,
    Ut_HcommCcuKernelQueryResReq_When_LaterSetResNumFails_Expect_ReturnErrorAndKeepEarlierWrites)
{
    PrefillAll(100);
    MockKernelOutputs();
    g_failSetResCall = 4;
    MOCKER_CPP(&hcomm::CcuResDescMgr::SetResNum).expects(exactly(4)).will(invoke(CaptureSetResNum));
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(NoArgKernel), nullptr, 0, resDesc_),
        CcuResult::CCU_E_INTERNAL);
    ExpectAll(7, 8, 15, 103, 104, 105, 106);
}

// 验证 Kernel 通过 CCU_THROW_IF_FAILED 抛出 CCU 异常时转换为 CCU_E_INTERNAL 并清理临时 Kernel。
TEST_F(
    HcommCcuKernelQueryResReqTest,
    Ut_HcommCcuKernelQueryResReq_When_KernelThrowsException_Expect_ReturnIsCCU_E_INTERNAL)
{
    PrefillAll(100);
    int32_t kernelArg = 0;
    const void* kernelArgs[] = {&kernelArg};
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(ThrowingKernel), kernelArgs, 1, resDesc_),
        CcuResult::CCU_E_INTERNAL);
    EXPECT_EQ(hcomm::CcuKernelMgr::GetInstance(TEST_DEVICE_LOGIC_ID).GetCurrentKernel(), nullptr);
    ExpectAll(100, 101, 102, 103, 104, 105, 106);
}

// 验证使用未获取的 Channel 时返回 Channel 错误，且不修改描述符。
TEST_F(
    HcommCcuKernelQueryResReqTest,
    Ut_HcommCcuKernelQueryResReq_When_ChannelWasNotAcquired_Expect_ReturnChannelErrorAndNotWriteDesc)
{
    PrefillAll(100);
    ChannelHandle invalidChannel = 0xCCAA55AAULL;
    const void* kernelArgs[] = {&invalidChannel};
    MOCKER(HcommChannelGet).stubs().will(invoke(MockChannelGetNotFound));
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(InvalidChannelKernel), kernelArgs, 1, resDesc_),
        CcuResult::CCU_E_NOT_FOUND);
    EXPECT_GT(g_channelGetCalls, 0U);
    EXPECT_EQ(hcomm::CcuKernelMgr::GetInstance(TEST_DEVICE_LOGIC_ID).GetCurrentKernel(), nullptr);
    ExpectAll(100, 101, 102, 103, 104, 105, 106);
}

// 验证 Kernel 使用不同 die 的 Channel 时返回 CCU_E_PARA，且不修改描述符。
TEST_F(
    HcommCcuKernelQueryResReqTest,
    Ut_HcommCcuKernelQueryResReq_When_ChannelsUseDifferentDies_Expect_ReturnParaAndNotWriteDesc)
{
    PrefillAll(100);
    HcommChannelDesc channelDesc{};
    hcomm::CcuUrmaChannel firstChannel(nullptr, channelDesc);
    hcomm::CcuUrmaChannel secondChannel(nullptr, channelDesc);
    constexpr ChannelHandle firstHandle = 0xCCAA5501ULL;
    constexpr ChannelHandle secondHandle = 0xCCAA5502ULL;
    g_channels[firstHandle] = &firstChannel;
    g_channels[secondHandle] = &secondChannel;
    MockChannels(true);

    ChannelQueryArg queryArg{firstHandle, secondHandle};
    const void* kernelArgs[] = {&queryArg};
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(ChannelBackedKernel), kernelArgs, 1, resDesc_),
        CcuResult::CCU_E_PARA);
    EXPECT_GT(g_channelGetCalls, 1U);
    ExpectAll(100, 101, 102, 103, 104, 105, 106);
}

// 验证无 Channel Kernel 使用描述符指定的 die1，并成功写入 die1 上的资源诉求。
TEST_F(
    HcommCcuKernelQueryResReqTest,
    Ut_HcommCcuKernelQueryResReq_When_NoChannelAndDescriptorDieIsOne_Expect_UseDescriptorDieAndSuccess)
{
    ResDescMgr().Deinit();
    ASSERT_EQ(ResDescMgr().Create(OTHER_VALID_DIE_ID, resDesc_), CcuResult::CCU_SUCCESS);
    PrefillAll(100);
    MockKernelOutputs(MakeBlockAndNonBlockResReq(OTHER_VALID_DIE_ID));
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(NoArgKernel), nullptr, 0, resDesc_),
        CcuResult::CCU_SUCCESS);
    ExpectAll(7, 8, 15, 19, 6, 1, 12);
}

// 验证单个 Channel 所属 die 与描述符指定 die 不一致时返回 CCU_E_PARA，且不修改描述符。
TEST_F(
    HcommCcuKernelQueryResReqTest,
    Ut_HcommCcuKernelQueryResReq_When_ChannelDieDiffersFromDescriptorDie_Expect_ReturnIsCCU_E_PARA)
{
    ResDescMgr().Deinit();
    ASSERT_EQ(ResDescMgr().Create(OTHER_VALID_DIE_ID, resDesc_), CcuResult::CCU_SUCCESS);
    PrefillAll(100);

    HcommChannelDesc channelDesc{};
    hcomm::CcuUrmaChannel channel(nullptr, channelDesc);
    constexpr ChannelHandle channelHandle = 0xCCAA5501ULL;
    g_channels[channelHandle] = &channel;
    MockChannels();

    ChannelQueryArg queryArg{channelHandle, 0};
    const void* kernelArgs[] = {&queryArg};
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(ChannelBackedKernel), kernelArgs, 1, resDesc_),
        CcuResult::CCU_E_PARA);
    EXPECT_GT(g_channelGetCalls, 0U);
    ExpectAll(100, 101, 102, 103, 104, 105, 106);
}

// 验证 Runtime 从 descriptor 所属 Device 切走再切回时，切走阶段拒绝，切回后恢复成功。
TEST_F(
    HcommCcuKernelQueryResReqTest,
    Ut_HcommCcuKernelQueryResReq_When_RuntimeDeviceSwitchesAwayAndBack_Expect_RejectAwayThenSucceedOnOwnerDevice)
{
    PrefillAll(100);
    MockKernelOutputs();

    // 模拟 aclrtSetDevice 将当前 Runtime Device 切换到 Device 1。
    g_runtimeDeviceLogicId = OTHER_TEST_DEVICE_LOGIC_ID;
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(NoArgKernel), nullptr, 0, resDesc_),
        CcuResult::CCU_E_NOT_FOUND);
    EXPECT_EQ(g_deviceRefreshCalls, 1U);
    EXPECT_EQ(g_noArgKernelCalls, 0U);
    EXPECT_EQ(ResDescMgr(OTHER_TEST_DEVICE_LOGIC_ID).Get(resDesc_), nullptr);
    ExpectAll(100, 101, 102, 103, 104, 105, 106);

    // 模拟 aclrtSetDevice 将当前 Runtime Device 切换回 Device 0。
    g_runtimeDeviceLogicId = TEST_DEVICE_LOGIC_ID;
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(NoArgKernel), nullptr, 0, resDesc_),
        CcuResult::CCU_SUCCESS);
    EXPECT_EQ(g_deviceRefreshCalls, 2U);
    EXPECT_EQ(g_noArgKernelCalls, 1U);
    ExpectAll(7, 8, 15, 19, 6, 1, 12);
}

// 验证刷新后的 Device 同时用于入口 manager 和 dry-run 内部真实 primitive。
TEST_F(
    HcommCcuKernelQueryResReqTest,
    Ut_HcommCcuKernelQueryResReq_When_RefreshedDeviceRunsPrimitiveKernel_Expect_UseSameDeviceThroughoutDryRun)
{
    PrefillAll(100);
    InitAdditionalKernelDevice(OTHER_TEST_DEVICE_LOGIC_ID);

    HcommCcuResDescHandle currentDeviceDesc = CreateDesc(OTHER_TEST_DEVICE_LOGIC_ID);
    if (ResDescMgr(TEST_DEVICE_LOGIC_ID).Get(currentDeviceDesc) != nullptr) {
        currentDeviceDesc = CreateDesc(OTHER_TEST_DEVICE_LOGIC_ID);
    }
    ASSERT_EQ(ResDescMgr(TEST_DEVICE_LOGIC_ID).Get(currentDeviceDesc), nullptr);

    // 模拟 aclrtSetDevice 将当前 Runtime Device 切换到 Device 1。
    g_runtimeDeviceLogicId = OTHER_TEST_DEVICE_LOGIC_ID;
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(
            reinterpret_cast<const void*>(DeviceRefreshPrimitiveKernel), nullptr, 0, currentDeviceDesc),
        CcuResult::CCU_SUCCESS);
    EXPECT_EQ(g_deviceRefreshCalls, 1U);
    EXPECT_EQ(g_noArgKernelCalls, 1U);
    EXPECT_EQ(g_kernelObservedDeviceLogicId, OTHER_TEST_DEVICE_LOGIC_ID);
    EXPECT_GT(QueryNum(OTHER_TEST_DEVICE_LOGIC_ID, currentDeviceDesc, Hccl::ResType::INS), 0U);
    ExpectAll(100, 101, 102, 103, 104, 105, 106);
    EXPECT_EQ(hcomm::CcuKernelMgr::GetInstance(TEST_DEVICE_LOGIC_ID).GetCurrentKernel(), nullptr);
    EXPECT_EQ(hcomm::CcuKernelMgr::GetInstance(OTHER_TEST_DEVICE_LOGIC_ID).GetCurrentKernel(), nullptr);
}

// 验证刷新当前 Device 失败时直接返回错误，不执行 Kernel，也不修改 descriptor。
TEST_F(
    HcommCcuKernelQueryResReqTest,
    Ut_HcommCcuKernelQueryResReq_When_DeviceRefreshFails_Expect_ReturnInternalWithoutRunningKernel)
{
    PrefillAll(100);
    MockKernelOutputs();

    g_deviceRefreshResult = HcclResult::HCCL_E_INTERNAL;
    EXPECT_EQ(
        HcommCcuKernelQueryResReq(reinterpret_cast<const void*>(NoArgKernel), nullptr, 0, resDesc_),
        CcuResult::CCU_E_INTERNAL);
    EXPECT_EQ(g_deviceRefreshCalls, 1U);
    EXPECT_EQ(g_noArgKernelCalls, 0U);
    ExpectAll(100, 101, 102, 103, 104, 105, 106);
}
