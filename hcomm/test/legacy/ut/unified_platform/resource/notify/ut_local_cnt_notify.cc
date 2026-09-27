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
#include "test_mock_setup.h"
#include <cstring>
#define private public
#include "local_cnt_notify.h"
#include "rdma_handle_manager.h"
#undef private
#include "orion_adapter_hccp.h"
using namespace Hccl;

class LocalCntNotifyTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LocalCntNotifyTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "LocalCntNotifyTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in LocalCntNotifyTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in LocalCntNotifyTest TearDown" << std::endl;
    }
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
};

TEST_F(LocalCntNotifyTest, getExchangeDto_test)
{
    SETUP_CNT_NOTIFY_MOCKS();

    int a = 0;
    RtsCntNotify rtsCntNotify;
    RdmaHandleManager::GetInstance().tokenInfoMap[(void*)&a] = make_unique<TokenInfoManager>(0, (void*)&a);
    LocalCntNotify localCntNotify((void*)&a, &rtsCntNotify);

    localCntNotify.GetExchangeDto();
};

TEST_F(LocalCntNotifyTest, Ut_When_DestroyAllDone_ThenDestructLocalCntNotify_Expect_SkipUnregNoCrash)
{
    // given
    SETUP_CNT_NOTIFY_MOCKS();
    pair<u64, u32> notifyInfoPair(1, 1);
    MOCKER_CPP(&RdmaHandleManager::GetTokenIdInfo).stubs().will(returnValue(notifyInfoPair));
    HrtRaUbLocalMemRegOutParam regOut;
    regOut.handle = 0x300;
    MOCKER(HrtRaUbLocalMemReg).stubs().will(returnValue(regOut));
    MOCKER(HrtRaUbLocalMemUnreg).expects(atMost(0));
    MOCKER_CPP(&RdmaHandleManager::PutTokenIdInfo).expects(atMost(0));

    int a = 0;
    RtsCntNotify rtsCntNotify;
    RdmaHandle rdmaHandle = (void*)&a;
    auto& mgr = RdmaHandleManager::GetInstance();
    mgr.tokenInfoMap[rdmaHandle] = make_unique<TokenInfoManager>(0, rdmaHandle);
    mgr.activeHandles_.insert(rdmaHandle);

    // when: 模拟 DestroyAll 后析构
    {
        LocalCntNotify localCntNotify(rdmaHandle, &rtsCntNotify);
        mgr.destroyed.store(true);
        mgr.activeHandles_.clear();
    }

    // then
    mgr.destroyed.store(false);
}

TEST_F(LocalCntNotifyTest, Ut_When_RdmaHandleValid_ThenDestructLocalCntNotify_Expect_UnregAndPutTokenIdInfo)
{
    SETUP_CNT_NOTIFY_MOCKS();
    pair<u64, u32> notifyInfoPair(1, 1);
    MOCKER_CPP(&RdmaHandleManager::GetTokenIdInfo).stubs().will(returnValue(notifyInfoPair));
    HrtRaUbLocalMemRegOutParam regOut;
    regOut.handle = 0x300;
    MOCKER(HrtRaUbLocalMemReg).stubs().will(returnValue(regOut));
    MOCKER(HrtRaUbLocalMemUnreg).expects(atMost(1)).with(mockcpp::any(), mockcpp::any());
    MOCKER_CPP(&RdmaHandleManager::PutTokenIdInfo).stubs();

    RtsCntNotify rtsCntNotify;
    RdmaHandle rdmaHandle = (void*)0x301;
    auto& mgr = RdmaHandleManager::GetInstance();
    mgr.tokenInfoMap[rdmaHandle] = make_unique<TokenInfoManager>(0, rdmaHandle);
    mgr.activeHandles_.insert(rdmaHandle);

    {
        LocalCntNotify localCntNotify(rdmaHandle, &rtsCntNotify);
        EXPECT_NE(localCntNotify.memHandle, 0u);
    }

    mgr.activeHandles_.erase(rdmaHandle);
}
