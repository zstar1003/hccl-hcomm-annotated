/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define private public
#define protected public

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <chrono>

#include "ccu_component.h"

#include "hccl_common_v2.h"
#include "ccu_res_specs_legacy.h"
#include "rdma_handle_manager.h"
#include "ccu_api_exception.h"

#undef private
#undef protected

using namespace Hccl;

class CcuComponentTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "CcuComponentTest tests set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "CcuComponentTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "A Test case in CcuComponentTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "A Test case in CcuComponentTest TearDown" << std::endl;
    }
};

constexpr uint32_t LOOP_CHANNEL_NUM = 1; // 每个die环回channel数量，如果创建数量为1
constexpr uint32_t CCU_V1_MAX_CHANNEL_NUM = 128;

// 为 ccu 两个die添加寄存器资源信息
void MockCcuResources(const int32_t devLogicId, const CcuVersion ccuVersion)
{
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(MAX_MODULE_DEVICE_NUM));

    auto& ccuResSpecs = CcuResSpecifications::GetInstance(devLogicId);
    ccuResSpecs.ccuVersion = ccuVersion;
    for (uint8_t dieId = 0; dieId < MAX_CCU_IODIE_NUM; dieId++) {
        ccuResSpecs.dieEnableFlags[dieId] = true;

        ccuResSpecs.resSpecs[dieId].loopEngineNum = 200;

        ccuResSpecs.resSpecs[dieId].msNum = 1536;
        ccuResSpecs.resSpecs[dieId].ckeNum = 1024;

        ccuResSpecs.resSpecs[dieId].xnNum = 3072;

        ccuResSpecs.resSpecs[dieId].gsaNum = 3072;

        ccuResSpecs.resSpecs[dieId].instructionNum = 32768;
        ccuResSpecs.resSpecs[dieId].missionNum = 16;

        ccuResSpecs.resSpecs[dieId].channelNum = 128;

        ccuResSpecs.resSpecs[dieId].jettyNum = 128;
        ccuResSpecs.resSpecs[dieId].wqeBBNum = 4096;

        ccuResSpecs.resSpecs[dieId].pfeNum = 10;

        ccuResSpecs.resSpecs[dieId].resourceAddr = 0xE7FFBF800000;

        ccuResSpecs.resSpecs[dieId].memInfoList[0].memVa = 0xE7FFBF800000;
    }
}

// 为 ccu 两个die添加网络通信设备
void MockCcuNetworkDevice(const int32_t devLogicId)
{
    vector<HrtDevEidInfo> eidInfoListStbu;
    HrtDevEidInfo eidInfo;
    eidInfo.name = "udma0";
    eidInfo.dieId = 0;
    eidInfo.funcId = 3;
    eidInfo.chipId = static_cast<uint32_t>(devLogicId);
    eidInfoListStbu.push_back(eidInfo);

    eidInfo.name = "udma1";
    eidInfo.dieId = 1;
    eidInfo.funcId = 4;
    eidInfo.chipId = static_cast<uint32_t>(devLogicId);
    eidInfoListStbu.push_back(eidInfo);

    MOCKER(HrtRaGetDevEidInfoList).stubs().with(mockcpp::any()).will(returnValue(eidInfoListStbu));
    MOCKER(HraGetRtpEnable).stubs().with(mockcpp::any()).will(returnValue(true));

    MOCKER_CPP(&RdmaHandleManager::GetByIp).stubs().will(returnValue((void*)0x12345678));
    std::pair<uint32_t, uint32_t> fakeDieFuncPair = std::make_pair(1, 4);
    MOCKER_CPP(&RdmaHandleManager::GetDieAndFuncId).stubs().will(returnValue(fakeDieFuncPair));
    std::pair<TokenIdHandle, uint32_t> fakeTokenInfo = std::make_pair(0x12345678, 1);
    MOCKER_CPP(&RdmaHandleManager::GetTokenIdInfo).stubs().will(returnValue(fakeTokenInfo));
}

TEST_F(CcuComponentTest, St_Init_When_CcuV1_Expect_Return_Ok)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuResources(devLogicId, ccuVersion);
    MockCcuNetworkDevice(devLogicId);

    CcuComponent ccuComponent;
    ccuComponent.devLogicId = devLogicId;

    EXPECT_NO_THROW(ccuComponent.Init());
}

TEST_F(CcuComponentTest, St_Init_When_NoUsbleFeEid_Expect_Throw_CcuApiException)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 2; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuResources(devLogicId, ccuVersion);

    CcuComponent ccuComponent;
    ccuComponent.devLogicId = devLogicId;

    // 没有可用EID时，两个die都表示为不启用
    EXPECT_THROW(ccuComponent.Init(), CcuApiException);
}

TEST_F(CcuComponentTest, St_AllocInsAndCkeAndXn_When_ResNumIsOk_Expect_Return_Ok)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuResources(devLogicId, ccuVersion);
    MockCcuNetworkDevice(devLogicId);

    CcuComponent ccuComponent;
    ccuComponent.devLogicId = devLogicId;

    EXPECT_NO_THROW(ccuComponent.Init());

    uint8_t dieId = 1;
    HcclResult ret = HcclResult::HCCL_E_RESERVED;

    // 申请少量资源，分配成功
    // Ins 资源
    ResInfo resInfo;
    uint32_t req = 100;
    ret = ccuComponent.AllocIns(dieId, req, resInfo);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(resInfo.num, req);

    ret = ccuComponent.ReleaseIns(dieId, resInfo);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    // Cke 资源
    vector<ResInfo> resInfos;
    ret = ccuComponent.AllocCke(dieId, req, resInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(resInfos.size(), 1);
    if (resInfos.size() > 1) {
        EXPECT_EQ(resInfos[0].num, req);
    }

    ret = ccuComponent.ReleaseCke(dieId, resInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    // Xn 资源
    ret = ccuComponent.AllocXn(dieId, req, resInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(resInfos.size(), 1);
    if (resInfos.size() > 1) {
        EXPECT_EQ(resInfos[0].num, req);
    }

    ret = ccuComponent.ReleaseXn(dieId, resInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuComponentTest, St_AllocInsAndCkeAndXn_When_ResNumExceedsLeftNum_Expect_Return_NotOk)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuResources(devLogicId, ccuVersion);
    MockCcuNetworkDevice(devLogicId);

    CcuComponent ccuComponent;
    ccuComponent.devLogicId = devLogicId;

    EXPECT_NO_THROW(ccuComponent.Init());

    const uint8_t dieId = 1;
    const uint8_t errorDieId = MAX_CCU_IODIE_NUM;
    HcclResult ret = HcclResult::HCCL_E_RESERVED;

    // 1.申请超过资源规格一半的资源，第二次申请资源不足
    // Ins 资源
    ResInfo resInfo;
    uint32_t req = 16385;
    ret = ccuComponent.AllocIns(dieId, req, resInfo);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(resInfo.num, req);

    ResInfo errorResInfo;
    ret = ccuComponent.AllocIns(dieId, req, errorResInfo);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    ret = ccuComponent.ReleaseIns(dieId, resInfo);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    // Cke 资源
    vector<ResInfo> resInfos;
    req = 513;
    ret = ccuComponent.AllocCke(dieId, req, resInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(resInfos.size(), 1);
    if (resInfos.size() > 1) {
        EXPECT_EQ(resInfos[0].num, req);
    }

    vector<ResInfo> errorResInfos;
    ret = ccuComponent.AllocCke(dieId, req, errorResInfos);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    ret = ccuComponent.ReleaseCke(dieId, resInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    // Xn 资源
    req = 1537;
    ret = ccuComponent.AllocXn(dieId, req, resInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(resInfos.size(), 1);
    if (resInfos.size() > 1) {
        EXPECT_EQ(resInfos[0].num, req);
    }

    ret = ccuComponent.AllocXn(dieId, req, errorResInfos);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    ret = ccuComponent.ReleaseXn(dieId, resInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    // 2.申请超过规格的资源
    req = 32769;
    ret = ccuComponent.AllocIns(dieId, req, errorResInfo);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    req = 1537;
    ret = ccuComponent.AllocCke(dieId, req, errorResInfos);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    req = 3073;
    ret = ccuComponent.AllocXn(dieId, req, resInfos);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    // 3. 申请错误的die的资源
    ret = ccuComponent.AllocIns(errorDieId, req, errorResInfo);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    ret = ccuComponent.ReleaseIns(errorDieId, errorResInfo);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    ret = ccuComponent.AllocCke(errorDieId, req, errorResInfos);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    ret = ccuComponent.ReleaseCke(errorDieId, errorResInfos);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    ret = ccuComponent.AllocXn(errorDieId, req, errorResInfos);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    ret = ccuComponent.ReleaseXn(errorDieId, errorResInfos);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuComponentTest, St_ReleaseRes_When_ResNumIsInvalid_Expect_Return_NotOk)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuResources(devLogicId, ccuVersion);
    MockCcuNetworkDevice(devLogicId);

    CcuComponent ccuComponent;
    ccuComponent.devLogicId = devLogicId;

    EXPECT_NO_THROW(ccuComponent.Init());

    const uint8_t dieId = 1;
    HcclResult ret = HcclResult::HCCL_E_RESERVED;
    vector<ResInfo> resInfos;

    // 申请资源后，释放部分资源，构造错误的释放出参
    uint32_t req = 40;
    ret = ccuComponent.AllocCke(dieId, req, resInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    resInfos[0].num = req / 2;
    ret = ccuComponent.ReleaseCke(dieId, resInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    resInfos[0].startId += req / 2; // 恢复正确值
    resInfos[0].startId -= 1;       // 构造错误用例
    ret = ccuComponent.ReleaseCke(dieId, resInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);

    resInfos[0].startId += 1;               // 恢复正确值
    resInfos[0].startId += resInfos[0].num; // 构造错误用例
    ret = ccuComponent.ReleaseCke(dieId, resInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);

    resInfos[0].startId -= resInfos[0].num;
    ret = ccuComponent.ReleaseCke(dieId, resInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuComponentTest, St_AllocInsAndCkeAndXn_When_ResNumIsBoundary_Expect_Return_NotOk)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuResources(devLogicId, ccuVersion);
    MockCcuNetworkDevice(devLogicId);

    CcuComponent ccuComponent;
    ccuComponent.devLogicId = devLogicId;

    EXPECT_NO_THROW(ccuComponent.Init());

    const uint8_t dieId = 1;
    HcclResult ret = HcclResult::HCCL_E_RESERVED;
    vector<ResInfo> resInfos;

    // 1. 申请最大规格的资源
    ResInfo resInfo;
    uint32_t req = 32768;
    ret = ccuComponent.AllocIns(dieId, req, resInfo);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(resInfo.num, req);

    ret = ccuComponent.ReleaseIns(dieId, resInfo);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    req = 1024;
    ret = ccuComponent.AllocCke(dieId, req, resInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(resInfos.size(), 1);
    if (resInfos.size() > 1) {
        EXPECT_EQ(resInfos[0].num, req);
    }

    ret = ccuComponent.ReleaseCke(dieId, resInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    req = 3072;
    ret = ccuComponent.AllocXn(dieId, req, resInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(resInfos.size(), 1);
    if (resInfos.size() > 1) {
        EXPECT_EQ(resInfos[0].num, req);
    }

    ret = ccuComponent.ReleaseXn(dieId, resInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    // 2. 申请空资源
    ResInfo errorResInfo;
    req = 0;
    ret = ccuComponent.AllocIns(dieId, req, errorResInfo);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    vector<ResInfo> errorResInfos;
    req = 0;
    ret = ccuComponent.AllocCke(dieId, req, errorResInfos);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    req = 0;
    ret = ccuComponent.AllocXn(dieId, req, errorResInfos);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuComponentTest, St_AllocChannels_When_CcuV1ResNumIsBoundary_Expect_Return_NotOk)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuResources(devLogicId, ccuVersion);
    MockCcuNetworkDevice(devLogicId);

    const uint8_t dieId = 1;
    HcclResult ret = HcclResult::HCCL_E_RESERVED;
    CcuComponent ccuComponent;
    ccuComponent.devLogicId = devLogicId;

    EXPECT_NO_THROW(ccuComponent.Init());

    ChannelPara channelPara{4, 1, 16}; // feId, jettyNum, sqSize，要和打桩对应
    vector<ChannelInfo> channelInfos;
    ret = ccuComponent.AllocChannels(dieId, channelPara, channelInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(1, channelInfos.size());

    auto& channelInfo = channelInfos[0];
    EXPECT_EQ(channelInfo.jettyInfos.size(), 1);
    EXPECT_EQ(channelInfo.jettyInfos[0].taJettyId, CCU_START_TA_JETTY_ID + LOOP_CHANNEL_NUM);

    ChannelCfg channelCfg;
    channelCfg.channelId = LOOP_CHANNEL_NUM; // 环回Channel占用 die 数量个
    channelCfg.remoteEid.in6 = {0x1234, 0x5678};
    channelCfg.remoteCcuVa = 0x87654321;
    channelCfg.memTokenId = 1;
    channelCfg.memTokenValue = 1;
    channelCfg.tpn = 1;

    JettyCfg cfg1 = {LOOP_CHANNEL_NUM, 0x87654321, 1, 1}; // jettyCtxId, dbVa, dbTokenId, dbTokenValue
    channelCfg.jettyCfgs.push_back(cfg1);
    ret = ccuComponent.ConfigChannel(dieId, channelCfg);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    ret = ccuComponent.ReleaseChannel(dieId, channelCfg.channelId);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuComponentTest, St_AllocChannels_When_CcuV1AndParaError_Expect_Return_NotOk)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuResources(devLogicId, ccuVersion);
    MockCcuNetworkDevice(devLogicId);

    const uint8_t dieId = 1;
    HcclResult ret = HcclResult::HCCL_E_RESERVED;
    CcuComponent ccuComponent;
    ccuComponent.devLogicId = devLogicId;

    EXPECT_NO_THROW(ccuComponent.Init());

    ChannelPara channelPara{3, 3, 1}; // feId, jettyNum, sqSize
    vector<ChannelInfo> channelInfos;
    ret = ccuComponent.AllocChannels(dieId, channelPara, channelInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    channelPara.feId = 4;
    ret = ccuComponent.AllocChannels(dieId, channelPara, channelInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    // 配置错误的channel
    ChannelCfg channelCfg;
    channelCfg.channelId = CCU_V1_MAX_CHANNEL_NUM;
    channelCfg.remoteEid.in6 = {0x1234, 0x5678};
    channelCfg.remoteCcuVa = 0x87654321;
    channelCfg.memTokenId = 5;
    channelCfg.memTokenValue = 8;
    channelCfg.tpn = 4;

    ret = ccuComponent.ConfigChannel(dieId, channelCfg); // 错误Channel范围
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    // 配置数量不对等的jetty
    channelCfg.channelId = LOOP_CHANNEL_NUM; // 除环回外首个分配的channel
    uint32_t baseId = LOOP_CHANNEL_NUM;
    JettyCfg cfg1 = {baseId, 1, 2, 3};
    JettyCfg cfg2 = {baseId + 1, 1, 2, 3};

    channelCfg.jettyCfgs.push_back(cfg1);
    channelCfg.jettyCfgs.push_back(cfg2);

    ret = ccuComponent.ConfigChannel(dieId, channelCfg);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    // 配置错误的jettyId
    JettyCfg cfg3 = {baseId + 1, 1, 2, 3};
    channelCfg.jettyCfgs.push_back(cfg3);
    ret = ccuComponent.ConfigChannel(dieId, channelCfg); // 检查JettyId
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    channelCfg.jettyCfgs[2].jettyCtxId = baseId + 2; // 修改为正确ID
    ret = ccuComponent.ConfigChannel(dieId, channelCfg);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    // 申请超过数量上限的jetty
    ChannelPara channelPara2{4, 3, 16}; // feId, jettyNum, sqSize
    ChannelInfo channelInfo2;
    ret = ccuComponent.AllocChannels(dieId, channelPara, channelInfos);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    auto channelInfo = channelInfos[0];
    ret = ccuComponent.ReleaseChannel(dieId, channelInfo.channelId);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    // 重复释放Channel
    ret = ccuComponent.ReleaseChannel(dieId, channelInfo.channelId);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    // 配置环回Channel
    channelCfg.channelId = 0;
    ret = ccuComponent.ConfigChannel(dieId, channelCfg);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    // 释放环回Channel
    ret = ccuComponent.ReleaseChannel(dieId, channelCfg.channelId);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuComponentTest, St_GetLoopChannelId_When_CcuV1_Expect_Return_Ok)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuResources(devLogicId, ccuVersion);
    MockCcuNetworkDevice(devLogicId);

    const uint8_t dieId = 1;
    HcclResult ret = HcclResult::HCCL_E_RESERVED;
    CcuComponent ccuComponent;
    ccuComponent.devLogicId = devLogicId;

    EXPECT_NO_THROW(ccuComponent.Init());
    uint32_t chanId = 0;
    for (int i = 0; i < MAX_CCU_IODIE_NUM; i++) {
        for (int j = 0; j < MAX_CCU_IODIE_NUM; j++) {
            auto ret = ccuComponent.GetLoopChannelId(i, j, chanId);
            EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
            EXPECT_EQ(chanId, 0); // 环回初始化应为0
        }
    }
}

TEST_F(CcuComponentTest, St_CleanDieCkes_When_CcuV1_Expect_Return_Ok)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuResources(devLogicId, ccuVersion);
    MockCcuNetworkDevice(devLogicId);

    const uint8_t dieId = 1;
    HcclResult ret = HcclResult::HCCL_E_RESERVED;
    CcuComponent ccuComponent;
    ccuComponent.devLogicId = devLogicId;

    EXPECT_NO_THROW(ccuComponent.Init());

    ret = ccuComponent.CleanDieCkes(dieId);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuComponentTest, St_SetTaskKillAndSetTaskKillDone_When_CcuV1_Expect_Return_Ok)
{
    const int32_t devLogicId = MAX_MODULE_DEVICE_NUM - 1; // 避免影响其他用例
    const CcuVersion ccuVersion = CcuVersion::CCU_V1;
    MockCcuResources(devLogicId, ccuVersion);
    MockCcuNetworkDevice(devLogicId);

    const uint8_t dieId = 1;
    HcclResult ret = HcclResult::HCCL_E_RESERVED;
    CcuComponent ccuComponent;
    ccuComponent.devLogicId = devLogicId;

    EXPECT_NO_THROW(ccuComponent.Init());

    EXPECT_EQ(ccuComponent.SetTaskKill(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ccuComponent.status, CcuComponent::CcuTaskKillStatus::TASK_KILL);

    EXPECT_EQ(ccuComponent.SetTaskKillDone(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ccuComponent.status, CcuComponent::CcuTaskKillStatus::INIT);
}
