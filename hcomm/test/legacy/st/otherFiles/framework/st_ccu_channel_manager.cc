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
#define private public
#define protected public
#include "channel_manager.h"
#include "pfe_manager.h"
#include "local_jetty_context_manager.h"
#include "env_config_v2.h"
#undef private
#undef protected

using namespace Hccl;
using namespace Ccu;

class ChannelManagerTest : public ::testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ChannelManagerTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "ChannelManagerTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in ChannelManagerTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in ChannelManagerTest TearDown" << std::endl;
    }

    struct ChannelData channelData_;
};

TEST(ChannelManagerTest, Initialize)
{
    ChannelManager cm;

    // 检查totalChannelNum_是否被正确设置
    EXPECT_EQ(cm.totalChannelNum_, 128);

    // 检查oneChannelUseJettyNum_是否被正确设置
    EXPECT_EQ(cm.oneChannelUseJettyNum_, ONE_CHANNEL_USE_JETTY_NUM);

    // 检查每个通道是否被正确初始化
    u32 ioDieNumTmp = EnvConfig::GetInstance().GetCcuConfig().GetIODieNum();
    for (uint32_t ioDie = 0; ioDie < ioDieNumTmp; ioDie++) {
        for (uint32_t i = 0; i < cm.totalChannelNum_; i++) {
            EXPECT_EQ(cm.channelInfo_[ioDie][i].channelId, i);
            EXPECT_EQ(cm.channelInfo_[ioDie][i].isUsed, false);
            EXPECT_EQ(cm.channelInfo_[ioDie][i].jettyNum, ONE_CHANNEL_USE_JETTY_NUM - 1);
            EXPECT_EQ(cm.channelInfo_[ioDie][i].ioDieId, ioDie);
        }
    }
}

TEST(ChannelManagerTest, TestChannelCreate_PFECreateFailed)
{
    ChannelManager channelManager;
    // 模拟PFEManager::PfeCreate返回未使用的PFE信息
    CfgBasePara para;
    para.dieId = 0;
    MOCKER_CPP(&PfeManager::PfeCreate).stubs().with(mockcpp::any()).will(returnValue(CcuPfeInfo{false, 0, 0, 0}));

    // 调用ChannelManager::ChannelCreate
    struct ChannelInfo info = channelManager.ChannelCreate(para);

    // 验证结果
    EXPECT_EQ(info.id, INVILD_U8);
}

TEST(ChannelManagerTest, TestChannelCreate_ChannelCreateFailed)
{
    ChannelManager channelManager;
    // 模拟PFEManager::PfeCreate返回使用的PFE信息
    CfgBasePara para;
    para.dieId = 0;

    MOCKER_CPP(&PfeManager::PfeCreate).stubs().with(mockcpp::any()).will(returnValue(CcuPfeInfo{true, 0, 0, 0}));

    // 模拟ChannelManager中所有通道都已使用
    channelManager.channelInfo_[0][0].isUsed = true;

    // 调用ChannelManager::ChannelCreate
    struct ChannelInfo info = channelManager.ChannelCreate(para);

    // 验证结果
    EXPECT_EQ(info.id, INVILD_U8);
}

TEST(ChannelManagerTest, TestChannelCreate_JettyCtxCreateFailed)
{
    ChannelManager channelManager;
    // 模拟PFEManager::PfeCreate返回使用的PFE信息
    CfgBasePara para;
    para.dieId = 0;

    MOCKER_CPP(&PfeManager::PfeCreate).stubs().with(mockcpp::any()).will(returnValue(CcuPfeInfo{true, 0, 0, 0}));

    // 模拟ChannelManager中有未使用的通道资源
    channelManager.channelInfo_[0][0].isUsed = false;

    // 模拟LocalJettyContextManager::JettyCtxCreate返回失败
    MOCKER_CPP(&LocalJettyContextManager::JettyCtxCreate).stubs().with(mockcpp::any()).will(returnValue(INVILD_U8));

    // 调用ChannelManager::ChannelCreate
    struct ChannelInfo info = channelManager.ChannelCreate(para);

    // 验证结果
    EXPECT_EQ(info.id, INVILD_U8);
}

TEST(ChannelManagerTest, TestChannelCreate_Success)
{
    ChannelManager channelManager;
    // 模拟PFEManager::PfeCreate返回使用的PFE信息
    CfgBasePara para;
    para.dieId = 0;
    MOCKER_CPP(&PfeManager::PfeCreate).stubs().with(mockcpp::any()).will(returnValue(CcuPfeInfo{true, 0, 0, 0}));

    // 模拟ChannelManager中有未使用的通道资源
    channelManager.channelInfo_[0][0].isUsed = false;

    // 模拟LocalJettyContextManager::JettyCtxCreate返回成功
    MOCKER_CPP(&LocalJettyContextManager::JettyCtxCreate).stubs().with(mockcpp::any()).will(returnValue(0));

    // 调用ChannelManager::ChannelCreate
    struct ChannelInfo info = channelManager.ChannelCreate(para);

    // 验证结果
    EXPECT_NE(info.id, 0xFF);
}

TEST(ChannelManagerTest, ConfigChannelDataTest)
{
    ChannelManager channelManager;
    struct ChannelCfg cfg;
    // 设置cfg的值
    cfg.basePara.dieId = 1;
    cfg.basePara.networkMode = HrtNetworkMode::HDC;
    cfg.basePara.phyDeviceId = 1;
    cfg.id = 1;

    // 调用待测试函数 channelManager.ConfigChannelData(cfg);

    // 验证结果
}

TEST(ChannelManagerTest, TestChannelConfig)
{
    ChannelManager channelManager;
    // Arrange
    ChannelCfg cfg;
    cfg.basePara.dieId = 1;
    cfg.id = 2;

    // Act
    HcclResult result = channelManager.ChannelConfig(cfg);

    // Assert
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

TEST(ChannelManagerTest, TestChannelConfigCheckFail)
{
    ChannelManager channelManager;
    // Arrange
    ChannelCfg cfg;
    cfg.basePara.dieId = 0; // Set to an invalid value
    cfg.id = 2;

    // Act
    HcclResult result = channelManager.ChannelConfig(cfg);

    // Assert
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

TEST(ChannelManagerTest, ChannelDestroy_ValidInput_ChannelUsed)
{
    ChannelManager channelManager;
    // 假设ioDie和channelId都在有效范围内，且通道已被使用
    uint8_t ioDie = 0;
    uint8_t channelId = 1;
    channelManager.channelInfo_[ioDie][channelId].isUsed = true;
    HcclResult result = channelManager.ChannelDestroy(ioDie, channelId);
    EXPECT_EQ(result, HcclResult::HCCL_SUCCESS);
    EXPECT_FALSE(channelManager.channelInfo_[ioDie][channelId].isUsed);
}

TEST(ChannelManagerTest, ChannelDestroy_ValidInput_ChannelNotUsed)
{
    ChannelManager channelManager;
    // 假设ioDie和channelId都在有效范围内，但通道未被使用
    uint8_t ioDie = 0;
    uint8_t channelId = 1;
    channelManager.channelInfo_[ioDie][channelId].isUsed = false;
    HcclResult result = channelManager.ChannelDestroy(ioDie, channelId);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

TEST(ChannelManagerTest, ChannelDestroy_InvalidIoDie)
{
    ChannelManager channelManager;
    // 假设ioDie不在有效范围内
    uint8_t ioDie = MAX_CCU_IODIE_NUM;
    uint8_t channelId = 1;
    HcclResult result = channelManager.ChannelDestroy(ioDie, channelId);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}

TEST(ChannelManagerTest, ChannelDestroy_InvalidChannelId)
{
    ChannelManager channelManager;
    // 假设channelId不在有效范围内
    uint8_t ioDie = 1;
    uint8_t channelId = 128; //  128 无效通道
    HcclResult result = channelManager.ChannelDestroy(ioDie, channelId);
    EXPECT_EQ(result, HcclResult::HCCL_E_PARA);
}
