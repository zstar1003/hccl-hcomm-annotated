/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_api_base_test.h"

#ifndef private
#define private public
#define protected public
#endif
#include "topoinfo_detect.h"
#undef private
#undef protected

using namespace std;
using namespace hccl;

// mockcpp 的 invoke 只接受函数指针,不能传 lambda,所以抽成静态函数
static HcclResult StubHcclNetOpenDev(
    HcclNetDevCtx* netDevCtx, NicType /*nicType*/, s32 /*devicePhyId*/, s32 /*deviceLogicId*/,
    HcclIpAddress /*localIp*/, HcclIpAddress /*backupIp*/)
{
    // 必须回填 ctx,否则后续 HcclSocket 拿着空指针 segfault
    *netDevCtx = reinterpret_cast<HcclNetDevCtx>(0x1);
    return HCCL_SUCCESS;
}

class HcclGroupLeaderListenTest : public BaseInit {
public:
    static void SetUpTestCase() { std::cout << "HcclGroupLeaderListenTest SetUpTestCase" << std::endl; }
    static void TearDownTestCase() { std::cout << "HcclGroupLeaderListenTest TearDownTestCase" << std::endl; }
    virtual void SetUp()
    {
        BaseInit::SetUp();

        MOCKER(HcclNetOpenDev).stubs().with(mockcpp::any()).will(invoke(&StubHcclNetOpenDev));
        MOCKER(HcclNetInit).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&TopoInfoDetect::GetRootHostIP).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HcclSocket::Init).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        // 业务代码调的是无参 Listen(),必须显式指定无参重载
        MOCKER_CPP(&HcclSocket::Listen, HcclResult(HcclSocket::*)())
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HcclSocket::Listen, HcclResult(HcclSocket::*)(u32))
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&TopoInfoDetect::GenerateRootInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&TopoInfoDetect::Teardown).stubs().will(returnValue(HCCL_SUCCESS));
    }
    virtual void TearDown()
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(HcclGroupLeaderListenTest, Ut_GroupLeaderListen_When_AutoPort_ReturnIsHCCL_SUCCESS)
{
    // 占住 60000 端口,验证 group leader 能从 [60000, 60031] 中抢占其他端口
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        HCCL_ERROR("create socket failed");
        return;
    }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(HOST_CONTROL_BASE_PORT);

    if (bind(server_fd, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        HCCL_ERROR("bind failed");
        close(server_fd);
        return;
    }

    if (listen(server_fd, 3) < 0) {
        HCCL_ERROR("listen failed");
        close(server_fd);
        return;
    }

    unsetenv("HCCL_IF_BASE_PORT");
    unsetenv("HCCL_HOST_SOCKET_PORT_RANGE");

    HcclRankHandle rankHandle;
    std::vector<HcclIpAddress> whitelist;
    std::shared_ptr<TopoInfoDetect> topoDetectGroupLeader;
    EXCEPTION_CATCH((topoDetectGroupLeader = std::make_shared<TopoInfoDetect>()), return);

    HcclResult ret = topoDetectGroupLeader->GroupLeaderListen(rankHandle, whitelist);
    close(server_fd);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclGroupLeaderListenTest, Ut_GroupLeaderListen_When_IfBasePort_ReturnIsHCCL_SUCCESS)
{
    setenv("HCCL_IF_BASE_PORT", "30000", 1);
    HcclResult ret;
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclRankHandle rankHandle;
    std::vector<HcclIpAddress> whitelist;
    std::shared_ptr<TopoInfoDetect> topoDetectGroupLeader;
    EXCEPTION_CATCH((topoDetectGroupLeader = std::make_shared<TopoInfoDetect>()), return);

    ret = topoDetectGroupLeader->GroupLeaderListen(rankHandle, whitelist);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclGroupLeaderListenTest, Ut_GroupLeaderListen_When_PortRange_ReturnIsHCCL_SUCCESS)
{
    setenv("HCCL_HOST_SOCKET_PORT_RANGE", "60000-60050", 1);
    HcclResult ret;
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclRankHandle rankHandle;
    std::vector<HcclIpAddress> whitelist;
    std::shared_ptr<TopoInfoDetect> topoDetectGroupLeader;
    EXCEPTION_CATCH((topoDetectGroupLeader = std::make_shared<TopoInfoDetect>()), return);

    ret = topoDetectGroupLeader->GroupLeaderListen(rankHandle, whitelist);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
