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
#include <cstring>
#define private public
#define protected public
#include "rdma_handle_manager.h"
#include "connections_builder.h"
#include "conn_local_cnt_notify_manager.h"
#include "local_cnt_notify.h"
#include "communicator_impl.h"
#undef protected
#undef private
#include "orion_adapter_hccp.h"
#include "null_ptr_exception.h"
#include "invalid_params_exception.h"
#include "ins_rules.h"
#include "interpreter.h"
#include "instruction.h"
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
};

class InsRulesTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "InsRulesTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "InsRulesTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in InsRulesTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in InsRulesTest TearDown" << std::endl;
    }
};

class ConnLocalCntNotifyManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ConnLocalCntNotifyManagerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "ConnLocalCntNotifyManagerTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in ConnLocalCntNotifyManagerTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in ConnLocalCntNotifyManagerTest TearDown" << std::endl;
    }
};

TEST_F(ConnLocalCntNotifyManagerTest, applyfor_and_get_success)
{
    int a = 0;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtCntNotifyCreate).stubs().will(returnValue((void*)&a));
    MOCKER(HrtGetCntNotifyId).stubs().will(returnValue(0));
    HrtDevResAddrInfo resAddrInfo;
    MOCKER(HrtGetDevResAddress).stubs().with(mockcpp::any()).will(returnValue(resAddrInfo));
    CommunicatorImpl comm;
    comm.devPhyId = 0;
    ConnLocalCntNotifyManager connLocalCntNotifyManager(&comm);
    vector<LinkData> links;
    LinkData link(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);
    links.push_back(link);

    auto rtsCntNotifys0 = connLocalCntNotifyManager.Get(0);
    EXPECT_EQ(0, rtsCntNotifys0.size());

    MOCKER(HrtRaRdmaInit).stubs().will(returnValue((void*)&a));
    RdmaHandleManager::GetInstance().tokenInfoMap[(void*)&a] = make_unique<TokenInfoManager>(0, (void*)&a);
    RdmaHandleManager::GetInstance().Create(comm.GetDevicePhyId(), link.GetLocalPort());
    connLocalCntNotifyManager.ApplyFor(0, links);
    auto rtsCntNotifys = connLocalCntNotifyManager.Get(0);

    EXPECT_EQ(rtsCntNotifys.size(), 2);

    connLocalCntNotifyManager.ApplyFor(0, links);

    LinkData link2(BasePortType(PortDeploymentType::P2P, ConnectProtoType::RDMA), 0, 1, 0, 1);
    vector<LinkData> links2;
    links2.push_back(link2);
    EXPECT_THROW(connLocalCntNotifyManager.ApplyFor(1, links2), InvalidParamsException);
}
