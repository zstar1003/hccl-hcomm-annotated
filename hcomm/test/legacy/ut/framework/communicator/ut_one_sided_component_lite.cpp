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
#include "one_sided_component_lite.h"
#include "connected_link_mgr.h"
#include "rmt_data_buffer_mgr.h"
#include "mem_transport_lite_mgr.h"
#include "coll_alg_info.h"
#include "kernel_param_lite.h"
#include "virtual_topo.h"
#undef private
#undef protected

using namespace Hccl;
using namespace std;

class OneSidedComponentLiteTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "OneSidedComponentLiteTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "OneSidedComponentLiteTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in OneSidedComponentLiteTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in OneSidedComponentLiteTest TearDown" << std::endl;
    }
};

TEST_F(OneSidedComponentLiteTest, test_Orchestrate_should_failed)
{
    ConnectedLinkMgr connectedLinkMgr{};
    MemTransportLiteMgr transportLiteMgr{nullptr};
    CollAlgInfo collAlgInfo{OpMode::OPBASE, "tag"};
    RmtDataBufferMgr rmtDataBufferMgr{&transportLiteMgr, &collAlgInfo};
    OneSidedComponentLite oneSidedComponentLite{0, 1, DevType::DEV_TYPE_950, 0, &connectedLinkMgr, &rmtDataBufferMgr};

    HcclAicpuOpLite aiCpuOpLite{};
    std::shared_ptr<InsQueue> insQueue = std::make_shared<InsQueue>();
    EXPECT_EQ(oneSidedComponentLite.Orchestrate(aiCpuOpLite, insQueue), HCCL_E_PARA);
}

TEST_F(OneSidedComponentLiteTest, test_Orchestrate_should_success)
{
    ConnectedLinkMgr connectedLinkMgr{};
    MemTransportLiteMgr transportLiteMgr{nullptr};
    CollAlgInfo collAlgInfo{OpMode::OPBASE, "tag"};
    RmtDataBufferMgr rmtDataBufferMgr{&transportLiteMgr, &collAlgInfo};
    OneSidedComponentLite oneSidedComponentLite{0, 1, DevType::DEV_TYPE_950, 0, &connectedLinkMgr, &rmtDataBufferMgr};

    std::shared_ptr<InsQueue> insQueue = std::make_shared<InsQueue>();
    HcclAicpuOpLite aiCpuOpLite{};
    aiCpuOpLite.algOperator.opType = OpType::BATCHPUT;
    aiCpuOpLite.algOperator.sendRecvRemoteRank = 3;
    aiCpuOpLite.algOperator.opType = OpType::BATCHGET;
    BasePortType basePortType{PortDeploymentType::P2P, ConnectProtoType::UB};
    LinkData linkData{basePortType, 0, 1, 0, 1};
    connectedLinkMgr.levelRankPairLinkDataMap[0][0] = {linkData};
    EXPECT_EQ(oneSidedComponentLite.Orchestrate(aiCpuOpLite, insQueue), HCCL_SUCCESS);
}
