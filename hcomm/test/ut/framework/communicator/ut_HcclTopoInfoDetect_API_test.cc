/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_api_base_test.h"
#include "hccl_network.h"

class HcclTopoInfoDetectTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();

        NetDevContext devContext;
        HcclNetDevCtx devCtx = &devContext;
        MOCKER(HcclNetOpenDev)
            .stubs()
            .with(outBoundP(&devCtx), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));

        MOCKER(HcclNetInit).stubs().will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&TopoInfoDetect::GetRootHostIP).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&HcclSocket::Listen, HcclResult(HcclSocket::*)(u32 port))
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&TopoInfoDetect::GenerateRootInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&TopoInfoDetect::SetupTopoExchangeServer)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&TopoInfoDetect::Teardown).stubs().will(returnValue(HCCL_SUCCESS));
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(HcclTopoInfoDetectTest, Ut_SetupServer_When_AutoPort_ReturnIsHCCL_SUCCESS)
{
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
        HCCL_ERROR("create socket failed");
        close(server_fd);
        return;
    }

    if (listen(server_fd, 3) < 0) {
        HCCL_ERROR("socket listen failed");
        close(server_fd);
        return;
    }

    unsetenv("HCCL_IF_BASE_PORT");
    unsetenv("HCCL_HOST_SOCKET_PORT_RANGE");

    HcclRootHandle rootHandle;
    std::shared_ptr<TopoInfoDetect> topoDetectServer;
    EXCEPTION_CATCH((topoDetectServer = std::make_shared<TopoInfoDetect>()), return);
    HcclResult ret = topoDetectServer->SetupServer(rootHandle);
    close(server_fd);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclTopoInfoDetectTest, Ut_SetupServer_When_IfBasePort_ReturnIsHCCL_SUCCESS)
{
    setenv("HCCL_IF_BASE_PORT", "30000", 1);
    HcclResult ret;
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclRootHandle rootHandle;
    std::shared_ptr<TopoInfoDetect> topoDetectServer;
    EXCEPTION_CATCH((topoDetectServer = std::make_shared<TopoInfoDetect>()), return);

    ret = topoDetectServer->SetupServer(rootHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclTopoInfoDetectTest, Ut_SetupServer_When_PortRange_ReturnIsHCCL_SUCCESS)
{
    setenv("HCCL_HOST_SOCKET_PORT_RANGE", "60000-60050", 1);
    HcclResult ret;
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclRootHandle rootHandle;
    std::shared_ptr<TopoInfoDetect> topoDetectServer;
    EXCEPTION_CATCH((topoDetectServer = std::make_shared<TopoInfoDetect>()), return);

    ret = topoDetectServer->SetupServer(rootHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclTopoInfoDetectTest, Ut_SetupServer_When_HostNicIsDown_ReturnIsHCCL_E_NETWORK)
{
    MOCKER_CPP(&TopoInfoDetect::CheckHostNicLinkUp).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_NETWORK));

    HcclRootHandle rootHandle{};
    TopoInfoDetect topoDetectServer;

    EXPECT_EQ(topoDetectServer.SetupServer(rootHandle), HCCL_E_NETWORK);
}

TEST_F(HcclTopoInfoDetectTest, Ut_SetupAgent_When_HostNicIsDown_ReturnIsHCCL_E_NETWORK)
{
    const HcclIpAddress localHostIP("127.0.0.1%lo");
    MOCKER(GetLocalHostIP).stubs().with(outBound(localHostIP), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TopoInfoDetect::CheckHostNicLinkUp).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_NETWORK));

    HcclRootHandle rootHandle{};
    ASSERT_EQ(strcpy_s(rootHandle.ip, sizeof(rootHandle.ip), "127.0.0.1"), EOK);
    rootHandle.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    HcclRankHandle rankHandle{};
    const CommConfig commConfig;
    TopoInfoDetect topoDetectAgent;

    EXPECT_EQ(topoDetectAgent.SetupAgent(1, 0, rootHandle, rankHandle, commConfig), HCCL_E_NETWORK);
}

TEST_F(HcclTopoInfoDetectTest, Ut_SetupServerByMasterInfo_When_HostNicIsDown_ReturnIsHCCL_E_NETWORK)
{
    MOCKER_CPP(&TopoInfoDetect::CheckHostNicLinkUp).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_NETWORK));

    const HcclIpAddress hostIP("127.0.0.1%lo");
    HcclRootHandle rootHandle{};
    TopoInfoDetect topoDetectServer;

    EXPECT_EQ(topoDetectServer.SetupServerByMasterInfo(hostIP, 60000, rootHandle), HCCL_E_NETWORK);
}

TEST_F(HcclTopoInfoDetectTest, Ut_SetupAgentByMasterInfo_When_HostNicIsDown_ReturnIsHCCL_E_NETWORK)
{
    MOCKER_CPP(&TopoInfoDetect::CheckHostNicLinkUp).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_NETWORK));

    HcclIpAddress hostIP("127.0.0.1%lo");
    HcclRootHandle rootHandle{};
    TopoInfoDetect topoDetectAgent;

    EXPECT_EQ(topoDetectAgent.SetupAgentByMasterInfo(hostIP, rootHandle), HCCL_E_NETWORK);
}
