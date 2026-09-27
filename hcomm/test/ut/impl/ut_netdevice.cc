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
#include <mockcpp/mockcpp.hpp>
#include <cmath>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

#include "workflow_pub.h"
#include "topoinfo_struct.h"
#include "hccl_ip_address.h"
#include "dlra_function.h"
#include "hccl_net_dev.h"
#define private public
#define protected public
#include "hccl_alg.h"
#include "hccl_impl.h"
#include "hccl_aiv.h"
#include "config.h"
#include "externalinput.h"
#include "hccl_communicator.h"
#include "hccl_communicator_attrs.h"
#include "hccl_comm_pub.h"
#include "transport_base_pub.h"
#include "comm_impl.h"
#include "comm_mesh_pub.h"
#include "coll_alg_operator.h"
#include "all_gather_operator.h"
#include "reduce_operator.h"
#include "transport_pub.h"
#include "hccl_common.h"
#include "broadcast_operator.h"
#include "reduce_scatter_operator.h"
#include "notify_pool.h"
#include "comm_base_pub.h"
#include "task_abort_handler_pub.h"
#include "coll_comm_executor.h"
#include "adapter_rts.h"
#include "heartbeat.h"
#include "acl/acl.h"
#include "hccl_comm.h"
#include "hccl_inner.h"
#undef private
#undef protected
#include "remote_notify.h"
#include "profiling_manager.h"
#include "base.h"
#include "adapter_rts_common.h"
#include "hdc_pub.h"
#include "aicpu_hdc_utils.h"
#include "common/aicpu_kfc_def.h"
#include "llt_hccl_stub_mc2.h"
#include "llt_hccl_stub.h"

#include "hccl_network.h"
using namespace std;
class NetDevUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        u32 deviceLogicId;
        hrtGetDeviceIndexByPhyId(0, deviceLogicId);
        hccl::NetworkManager::GetInstance(deviceLogicId).Destroy();
    }
    static void TearDownTestCase() {}
    virtual void SetUp()
    {
        HcclIpAddress localIp(1711319232);
        info.devicePhyId = 0;
        info.addr.type = HcclAddressType::HCCL_ADDR_TYPE_IP_V4;
        info.addr.addr = localIp.GetBinaryAddress().addr;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
    HcclNetDevInfos info;
    HcclNetDev netDev;
};

TEST_F(NetDevUt, TestDeviceDeploymentBusAddr1)
{
    info.netdevDeployment = HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_DEVICE;
    info.isBackup = false;
    MOCKER(Is310PDevice).stubs().with(mockcpp::any()).will(returnValue(false));
    bool supportGetVincip = true;
    MOCKER(IsSuppportRaGetSocketVnicIps).stubs().with(outBound(supportGetVincip)).will(returnValue(HCCL_SUCCESS));

    // 测试查询地址

    HcclAddress busAddr;
    HcclDeviceId info1;
    info1.devicePhyId = info.devicePhyId;
    HcclNetDevGetBusAddr(info1, &busAddr);
    EXPECT_EQ(busAddr.addr.s_addr, 2);

    GlobalMockObject::verify();
}

TEST_F(NetDevUt, TestDeviceDeploymentBusAddr2)
{
    GlobalMockObject::verify();
    info.netdevDeployment = HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_DEVICE;
    info.isBackup = false;
    MOCKER(Is310PDevice).stubs().with(mockcpp::any()).will(returnValue(false));
    u32 deviceLogicId;
    hrtGetDeviceIndexByPhyId(info.devicePhyId, deviceLogicId);
    bool supportGetVincip = false;
    MOCKER(IsSuppportRaGetSocketVnicIps).stubs().with(outBound(supportGetVincip)).will(returnValue(HCCL_SUCCESS));
    HcclAddress busAddr;
    HcclDeviceId info1;
    info1.devicePhyId = info.devicePhyId;
    HcclNetDevGetBusAddr(info1, &busAddr);
    EXPECT_EQ(busAddr.addr.s_addr, 0);

    GlobalMockObject::verify();
}

TEST_F(NetDevUt, TestDeviceDeploymentNicAddr)
{
    info.netdevDeployment = HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_DEVICE;
    info.isBackup = false;
    MOCKER(Is310PDevice).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER(GetExternalInputHcclIsTcpMode).stubs().with(mockcpp::any()).will(returnValue(true));
    HcclAddress* addr;
    uint32_t addrNum = 0;
    HcclNetDevGetNicAddr(info.devicePhyId, &addr, &addrNum);
    EXPECT_NE(addr, nullptr);
    GlobalMockObject::verify();
}

TEST_F(NetDevUt, TestHostDeploymentTcp)
{
    info.netdevDeployment = HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_HOST;
    info.isBackup = false;
    info.addr.protoType = HcclProtoType::HCCL_PROTO_TYPE_TCP;
    MOCKER(Is310PDevice).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER(GetExternalInputHcclIsTcpMode).stubs().with(mockcpp::any()).will(returnValue(true));
    EXPECT_EQ(HcclNetDevOpen(&info, &netDev), HCCL_SUCCESS);

    HcclAddress getaddr;
    NetDevContext* pNetDevCtx = static_cast<hccl::NetDevContext*>(netDev);
    EXPECT_EQ(pNetDevCtx->GetNetDevDeployment(), info.netdevDeployment);
    EXPECT_EQ(pNetDevCtx->GetProtoType(), info.addr.protoType);

    HcclNetDevGetAddr(netDev, &getaddr);
    EXPECT_EQ(getaddr.addr.s_addr, info.addr.addr.s_addr);
    EXPECT_NE(pNetDevCtx->handle_, nullptr);

    EXPECT_EQ(HcclNetDevClose(netDev), HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(NetDevUt, TestHostDeploymentTcpERR)
{
    info.netdevDeployment = HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_HOST;
    info.isBackup = false;
    info.addr.protoType = HcclProtoType::HCCL_PROTO_TYPE_TCP;
    MOCKER(Is310PDevice).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER(GetExternalInputHcclIsTcpMode).stubs().with(mockcpp::any()).will(returnValue(true));
    EXPECT_EQ(HcclNetDevOpen(&info, &netDev), HCCL_SUCCESS);

    HcclAddress getaddr;
    NetDevContext* pNetDevCtx = static_cast<hccl::NetDevContext*>(netDev);
    EXPECT_EQ(pNetDevCtx->GetNetDevDeployment(), info.netdevDeployment);
    EXPECT_EQ(pNetDevCtx->GetProtoType(), info.addr.protoType);

    HcclNetDevGetAddr(netDev, &getaddr);
    EXPECT_EQ(getaddr.addr.s_addr, info.addr.addr.s_addr);
    EXPECT_NE(pNetDevCtx->handle_, nullptr);
    pNetDevCtx->deviceLogicId_ = 100;
    EXPECT_NE(HcclNetDevClose(netDev), HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(NetDevUt, TestHostDeploymentRoce)
{
    info.netdevDeployment = HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_HOST;
    info.isBackup = false;
    info.addr.protoType = HcclProtoType::HCCL_PROTO_TYPE_ROCE;
    MOCKER(Is310PDevice).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER(GetExternalInputHcclIsTcpMode).stubs().with(mockcpp::any()).will(returnValue(true));
    HcclNetDevInfos infotest = info;
    infotest.addr.protoType = HcclProtoType::HCCL_PROTO_TYPE_TCP;
    HcclNetDev netDevtest;
    EXPECT_EQ(HcclNetDevOpen(&infotest, &netDevtest), HCCL_SUCCESS);
    GlobalMockObject::verify();
    MOCKER(Is310PDevice).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER(GetExternalInputHcclIsTcpMode).stubs().with(mockcpp::any()).will(returnValue(false));
    EXPECT_EQ(HcclNetDevOpen(&info, &netDev), HCCL_SUCCESS);

    EXPECT_EQ(HcclNetDevClose(netDev), HCCL_SUCCESS);
    GlobalMockObject::verify();
    MOCKER(Is310PDevice).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER(GetExternalInputHcclIsTcpMode).stubs().with(mockcpp::any()).will(returnValue(true));
    EXPECT_EQ(HcclNetDevClose(netDevtest), HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(NetDevUt, TestHostDeploymentBus)
{
    //
    info.netdevDeployment = HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_HOST;
    info.isBackup = false;
    info.addr.protoType = HcclProtoType::HCCL_PROTO_TYPE_BUS;

    EXPECT_EQ(HcclNetDevOpen(&info, &netDev), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(netDev, nullptr);
    EXPECT_EQ(HcclNetDevClose(netDev), HCCL_E_PTR);
    GlobalMockObject::verify();
}

TEST_F(NetDevUt, TestHostDeploymentReserved)
{
    //
    info.netdevDeployment = HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_HOST;
    info.isBackup = false;
    info.addr.protoType = HcclProtoType::HCCL_PROTO_TYPE_RESERVED;

    EXPECT_EQ(HcclNetDevOpen(&info, &netDev), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(netDev, nullptr);
    EXPECT_EQ(HcclNetDevClose(netDev), HCCL_E_PTR);
    GlobalMockObject::verify();
}

// 当前桩有问题  只能先初始化 tcp才能初始化rdma
TEST_F(NetDevUt, TestDeviceDeploymentRoce)
{
    info.netdevDeployment = HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_DEVICE;
    info.isBackup = false;
    info.addr.protoType = HcclProtoType::HCCL_PROTO_TYPE_ROCE;
    MOCKER(Is310PDevice).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER(GetExternalInputHcclIsTcpMode).stubs().with(mockcpp::any()).will(returnValue(true));
    HcclNetDevInfos infotest = info;
    infotest.addr.protoType = HcclProtoType::HCCL_PROTO_TYPE_TCP;
    HcclNetDev netDevtest;
    EXPECT_EQ(HcclNetDevOpen(&infotest, &netDevtest), HCCL_SUCCESS);
    GlobalMockObject::verify();
    MOCKER(Is310PDevice).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER(GetExternalInputHcclIsTcpMode).stubs().with(mockcpp::any()).will(returnValue(false));

    EXPECT_EQ(HcclNetDevOpen(&info, &netDev), HCCL_SUCCESS);

    EXPECT_EQ(HcclNetDevClose(netDev), HCCL_SUCCESS);
    GlobalMockObject::verify();
    MOCKER(Is310PDevice).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER(GetExternalInputHcclIsTcpMode).stubs().with(mockcpp::any()).will(returnValue(true));
    EXPECT_EQ(HcclNetDevClose(netDevtest), HCCL_SUCCESS);
    GlobalMockObject::verify();
}

// 当前桩有问题  只能先初始化 tcp才能初始化rdma
TEST_F(NetDevUt, TestDeviceDeploymentRoceBackup)
{
    info.netdevDeployment = HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_DEVICE;
    info.isBackup = true;
    info.addr.protoType = HcclProtoType::HCCL_PROTO_TYPE_ROCE;
    MOCKER(Is310PDevice).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER(GetExternalInputHcclIsTcpMode).stubs().with(mockcpp::any()).will(returnValue(true));
    HcclNetDevInfos infotest = info;
    infotest.addr.protoType = HcclProtoType::HCCL_PROTO_TYPE_TCP;
    HcclNetDev netDevtest;
    EXPECT_EQ(HcclNetDevOpen(&infotest, &netDevtest), HCCL_SUCCESS);
    GlobalMockObject::verify();
    MOCKER(Is310PDevice).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER(GetExternalInputHcclIsTcpMode).stubs().with(mockcpp::any()).will(returnValue(false));
    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    LinkTypeInServer linkType = LinkTypeInServer::SIO_TYPE;
    MOCKER(hrtGetPairDeviceLinkType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(linkType))
        .will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(HcclNetDevOpen(&info, &netDev), HCCL_SUCCESS);

    EXPECT_EQ(HcclNetDevClose(netDev), HCCL_SUCCESS);
    GlobalMockObject::verify();
    MOCKER(Is310PDevice).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER(GetExternalInputHcclIsTcpMode).stubs().with(mockcpp::any()).will(returnValue(true));
    EXPECT_EQ(HcclNetDevClose(netDevtest), HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(NetDevUt, TestDeviceDeploymentTcp)
{
    //
    info.netdevDeployment = HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_DEVICE;
    info.isBackup = false;
    info.addr.protoType = HcclProtoType::HCCL_PROTO_TYPE_TCP;
    MOCKER(Is310PDevice).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER(GetExternalInputHcclIsTcpMode).stubs().with(mockcpp::any()).will(returnValue(true));

    EXPECT_EQ(HcclNetDevOpen(&info, &netDev), HCCL_SUCCESS);

    HcclAddress getaddr;
    NetDevContext* pNetDevCtx = static_cast<hccl::NetDevContext*>(netDev);
    EXPECT_EQ(pNetDevCtx->GetNetDevDeployment(), info.netdevDeployment);
    EXPECT_EQ(pNetDevCtx->GetProtoType(), info.addr.protoType);
    EXPECT_NE(pNetDevCtx->handle_, nullptr);

    HcclNetDevGetAddr(netDev, &getaddr);
    EXPECT_EQ(getaddr.addr.s_addr, info.addr.addr.s_addr);

    EXPECT_EQ(HcclNetDevClose(netDev), HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(NetDevUt, TestDeviceDeploymentTcpERR)
{
    //
    info.netdevDeployment = HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_DEVICE;
    info.isBackup = false;
    info.addr.protoType = HcclProtoType::HCCL_PROTO_TYPE_TCP;
    MOCKER(Is310PDevice).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER(GetExternalInputHcclIsTcpMode).stubs().with(mockcpp::any()).will(returnValue(true));

    EXPECT_EQ(HcclNetDevOpen(&info, &netDev), HCCL_SUCCESS);

    HcclAddress getaddr;
    NetDevContext* pNetDevCtx = static_cast<hccl::NetDevContext*>(netDev);
    EXPECT_EQ(pNetDevCtx->GetNetDevDeployment(), info.netdevDeployment);
    EXPECT_EQ(pNetDevCtx->GetProtoType(), info.addr.protoType);
    EXPECT_NE(pNetDevCtx->handle_, nullptr);

    HcclNetDevGetAddr(netDev, &getaddr);
    EXPECT_EQ(getaddr.addr.s_addr, info.addr.addr.s_addr);
    pNetDevCtx->deviceLogicId_ = 100;
    EXPECT_NE(HcclNetDevClose(netDev), HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(NetDevUt, TestDeviceDeploymentReserved)
{
    //
    info.netdevDeployment = HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_DEVICE;
    info.isBackup = false;
    info.addr.protoType = HcclProtoType::HCCL_PROTO_TYPE_RESERVED;

    EXPECT_EQ(HcclNetDevOpen(&info, &netDev), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(netDev, nullptr);
    EXPECT_EQ(HcclNetDevClose(netDev), HCCL_E_PTR);
    GlobalMockObject::verify();
}

TEST_F(NetDevUt, TestReservedDeploymentReserved)
{
    info.netdevDeployment = HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_RESERVED;
    info.isBackup = false;
    info.addr.protoType = HcclProtoType::HCCL_PROTO_TYPE_RESERVED;

    EXPECT_EQ(HcclNetDevOpen(&info, &netDev), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(netDev, nullptr);
    EXPECT_EQ(HcclNetDevClose(netDev), HCCL_E_PTR);
    GlobalMockObject::verify();
}

TEST_F(NetDevUt, TestDeviceDeploymentBus)
{
    info.netdevDeployment = HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_DEVICE;
    info.isBackup = false;
    info.addr.protoType = HcclProtoType::HCCL_PROTO_TYPE_BUS;
    MOCKER(Is310PDevice).stubs().with(mockcpp::any()).will(returnValue(false));
    u32 deviceLogicId;
    hrtGetDeviceIndexByPhyId(info.devicePhyId, deviceLogicId);

    bool supportGetVincip = true;
    MOCKER(IsSuppportRaGetSocketVnicIps).stubs().with(outBound(supportGetVincip)).will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(HcclNetDevOpen(&info, &netDev), HCCL_SUCCESS);
    NetDevContext* pNetDevCtx = static_cast<hccl::NetDevContext*>(netDev);
    HcclAddress getaddr;
    EXPECT_EQ(pNetDevCtx->GetNetDevDeployment(), info.netdevDeployment);
    EXPECT_EQ(pNetDevCtx->GetProtoType(), info.addr.protoType);
    EXPECT_NE(pNetDevCtx->handle_, nullptr);

    HcclNetDevGetAddr(netDev, &getaddr);
    EXPECT_EQ(getaddr.addr.s_addr, info.addr.addr.s_addr);
    EXPECT_EQ(HcclNetDevClose(netDev), HCCL_SUCCESS);
    GlobalMockObject::verify();
}
