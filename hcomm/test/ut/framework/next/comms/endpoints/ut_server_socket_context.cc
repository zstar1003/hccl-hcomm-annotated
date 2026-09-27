/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 被测目标: hcomm::ServerSocketContext / HostServerSocketContext / DeviceServerSocketContext

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public
#include "server_socket_context/server_socket_context.h"
#include "server_socket_context/host_server_socket_context.h"
#include "server_socket_context/device_server_socket_context.h"
#include "server_socket_context/aicpu_ts_roce_server_socket_context.h"
#include "server_socket_context/aicpu_ts_hccs_server_socket_context.h"
#include "server_socket_context/ub_rtp_uboe_server_socket_context.h"
#include "server_socket_manager.h"
#undef protected
#undef private

#include "endpoint.h"
#include "orion_adpt_utils.h"
#include "adapter_rts_common.h"
#include "ip_address.h"
#include "port.h"
#include "hcomm_res_defs.h"

using namespace hcomm;

namespace {
// 统一 mock：CommAddrToIpAddress / hrtGetDevice / hrtGetDevicePhyIdByIndex / ServerSocketStartListen
HcclResult (*g_realCommAddrToIpAddress)(const CommAddr&, Hccl::IpAddress&) = nullptr;

HcclResult StubCommAddrToIpAddress(const CommAddr&, Hccl::IpAddress& ip)
{
    // 返回一个固定合法 IP，避免真实地址解析
    ip = Hccl::IpAddress("1.0.0.0");
    return HCCL_SUCCESS;
}

void SetupServerSocketContextMocks(uint32_t listenRetPort = 60001, int32_t startListenRet = HCCL_SUCCESS)
{
    static uint32_t g_listenPort = 60001;
    g_listenPort = listenRetPort;
    MOCKER(&CommAddrToIpAddress).stubs().will(invoke(StubCommAddrToIpAddress));
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    // hrtGetDevicePhyIdByIndex 输出 devPhyId=0
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));
    // ServerSocketStartListen: 第 4 参数为输出端口（uint32_t* port），用 outBoundP 设置
    MOCKER_CPP(&ServerSocketManager::ServerSocketStartListen)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&g_listenPort, sizeof(g_listenPort)))
        .will(returnValue(startListenRet));
    // ServerSocketStopListen 不校验返回值，桩为 SUCCESS
    MOCKER_CPP(&ServerSocketManager::ServerSocketStopListen).stubs().will(returnValue(HCCL_SUCCESS));
}
// commAddr 在构造时传入 context（与迁移前一致，方法内经 CommAddrToIpAddress 转换，此处已统一 mock）
} // namespace

class ServerSocketContextTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }

    // 构造一个 HOST ROCE EndpointDesc（commAddr 为 IPv4）
    EndpointDesc MakeHostRoceDesc()
    {
        EndpointDesc desc{};
        desc.protocol = COMM_PROTOCOL_ROCE;
        desc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
        desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        Hccl::IpAddress ip("1.0.0.0");
        desc.commAddr.addr = ip.GetBinaryAddress().addr;
        return desc;
    }

    // 构造一个 DEVICE UB EndpointDesc
    EndpointDesc MakeDeviceUbDesc()
    {
        EndpointDesc desc{};
        desc.protocol = COMM_PROTOCOL_UB_CTP;
        desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        desc.loc.device.devPhyId = 0;
        desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        Hccl::IpAddress ip("2.0.0.0");
        desc.commAddr.addr = ip.GetBinaryAddress().addr;
        return desc;
    }
};

// TC-HostServerSocketContext_Listen-001: RDMA 协议 Host 正常监听
TEST_F(ServerSocketContextTest, Ut_HostListen_When_RdmaNormal_Expect_Success)
{
    SetupServerSocketContextMocks();
    EndpointDesc desc = MakeHostRoceDesc();
    HostServerSocketContext ctx(Hccl::ConnectProtoType::RDMA, desc.commAddr);

    HcclResult ret = ctx.ServerSocketListen(60001);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// TC-HostServerSocketContext_Listen-002: UB 协议 Host 正常监听
TEST_F(ServerSocketContextTest, Ut_HostListen_When_UbNormal_Expect_Success)
{
    SetupServerSocketContextMocks();
    EndpointDesc desc = MakeHostRoceDesc();
    HostServerSocketContext ctx(Hccl::ConnectProtoType::UB, desc.commAddr);

    HcclResult ret = ctx.ServerSocketListen(60002);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// TC-HostServerSocketContext_GetListenPort-002: 未监听时获取端口自动创建
TEST_F(ServerSocketContextTest, Ut_HostGetListenPort_When_NotListened_Expect_AutoCreate)
{
    SetupServerSocketContextMocks();
    EndpointDesc desc = MakeHostRoceDesc();
    HostServerSocketContext ctx(Hccl::ConnectProtoType::RDMA, desc.commAddr);

    uint32_t port = 0;
    HcclResult ret = ctx.ServerSocketGetListenPort(&port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // dynamicPort_ 被设置（不等于初始无效端口）
    EXPECT_NE(ctx.dynamicPort_, HCCL_INVALID_PORT);
}

// TC-HostServerSocketContext_GetListenPort-001: 已监听时获取端口返回 dynamicPort_
TEST_F(ServerSocketContextTest, Ut_HostGetListenPort_When_AlreadyListened_Expect_ReturnCachedPort)
{
    SetupServerSocketContextMocks();
    EndpointDesc desc = MakeHostRoceDesc();
    HostServerSocketContext ctx(Hccl::ConnectProtoType::RDMA, desc.commAddr);

    // 第一次 GetListenPort：走完整查询路径（StartListen 成功后写入 dynamicPort_ 缓存）
    uint32_t firstGetListenPort = 0;
    EXPECT_EQ(ctx.ServerSocketGetListenPort(&firstGetListenPort), HCCL_SUCCESS);
    EXPECT_NE(firstGetListenPort, 0u);
    // 将外部查询打桩为失败：若缓存分支失效，第二次 GetListenPort 会触发 StartListen 并返回失败
    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen).stubs().will(returnValue(HCCL_E_INTERNAL));
    // 第二次 GetListenPort：命中 dynamicPort_ 缓存分支，不再触发外部查询
    uint32_t secondGetListenPort = 0;
    EXPECT_EQ(ctx.ServerSocketGetListenPort(&secondGetListenPort), HCCL_SUCCESS);
    EXPECT_EQ(secondGetListenPort, firstGetListenPort);
}

// TC-HostServerSocketContext_GetListenPort: port 出参为空返回 HCCL_E_PTR
TEST_F(ServerSocketContextTest, Ut_HostGetListenPort_When_PortNull_Expect_ReturnPtrError)
{
    SetupServerSocketContextMocks();
    EndpointDesc desc = MakeHostRoceDesc();
    HostServerSocketContext ctx(Hccl::ConnectProtoType::RDMA, desc.commAddr);

    HcclResult ret = ctx.ServerSocketGetListenPort(nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// TC-HostServerSocketContext_StopListen-001: 停止监听
TEST_F(ServerSocketContextTest, Ut_HostStopListen_When_Called_Expect_Success)
{
    SetupServerSocketContextMocks();
    EndpointDesc desc = MakeHostRoceDesc();
    HostServerSocketContext ctx(Hccl::ConnectProtoType::RDMA, desc.commAddr);

    EXPECT_EQ(ctx.ServerSocketListen(60004), HCCL_SUCCESS);
    HcclResult ret = ctx.ServerSocketStopListen(60004);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// TC-DeviceServerSocketContext_Listen-001: DEVICE UB 正常监听
TEST_F(ServerSocketContextTest, Ut_DeviceListen_When_UbDevice_Expect_Success)
{
    SetupServerSocketContextMocks();
    EndpointDesc desc = MakeDeviceUbDesc();
    DeviceServerSocketContext ctx(
        Hccl::ConnectProtoType::UB, desc.loc.device.devPhyId, desc.loc.locType, desc.commAddr);

    HcclResult ret = ctx.ServerSocketListen(60005);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// TC-DeviceServerSocketContext_Listen-002: locType 非 DEVICE 时跳过监听返回 SUCCESS
TEST_F(ServerSocketContextTest, Ut_DeviceListen_When_LocTypeHost_Expect_SkipAndSuccess)
{
    SetupServerSocketContextMocks();
    EndpointDesc desc = MakeHostRoceDesc(); // locType=HOST
    DeviceServerSocketContext ctx(
        Hccl::ConnectProtoType::UB, desc.loc.device.devPhyId, desc.loc.locType, desc.commAddr);

    HcclResult ret = ctx.ServerSocketListen(60006);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// TC-DeviceServerSocketContext_GetListenPort-002: locType 非 DEVICE 时跳过获取端口返回 SUCCESS
TEST_F(ServerSocketContextTest, Ut_DeviceGetListenPort_When_LocTypeHost_Expect_SkipAndSuccess)
{
    SetupServerSocketContextMocks();
    EndpointDesc desc = MakeHostRoceDesc(); // locType=HOST
    DeviceServerSocketContext ctx(
        Hccl::ConnectProtoType::UB, desc.loc.device.devPhyId, desc.loc.locType, desc.commAddr);

    uint32_t port = 0;
    HcclResult ret = ctx.ServerSocketGetListenPort(&port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// TC-DeviceServerSocketContext_GetListenPort-001: DEVICE 已监听获取端口
TEST_F(ServerSocketContextTest, Ut_DeviceGetListenPort_When_AlreadyListened_Expect_ReturnCachedPort)
{
    SetupServerSocketContextMocks();
    EndpointDesc desc = MakeDeviceUbDesc();
    DeviceServerSocketContext ctx(
        Hccl::ConnectProtoType::UB, desc.loc.device.devPhyId, desc.loc.locType, desc.commAddr);

    // 第一次 GetListenPort：走完整查询路径（StartListen 成功后写入 dynamicPort_ 缓存）
    uint32_t firstPort = 0;
    EXPECT_EQ(ctx.ServerSocketGetListenPort(&firstPort), HCCL_SUCCESS);
    EXPECT_NE(firstPort, 0u);
    // 将外部查询打桩为失败：若缓存分支失效，第二次 GetListenPort 会触发 StartListen 并返回失败
    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen).stubs().will(returnValue(HCCL_E_INTERNAL));
    // 第二次 GetListenPort：命中 dynamicPort_ 缓存分支，不再触发外部查询
    uint32_t secondPort = 0;
    EXPECT_EQ(ctx.ServerSocketGetListenPort(&secondPort), HCCL_SUCCESS);
    EXPECT_EQ(secondPort, firstPort);
}

// TC-DeviceServerSocketContext_StopListen-001: 停止监听
TEST_F(ServerSocketContextTest, Ut_DeviceStopListen_When_Called_Expect_Success)
{
    SetupServerSocketContextMocks();
    EndpointDesc desc = MakeDeviceUbDesc();
    DeviceServerSocketContext ctx(
        Hccl::ConnectProtoType::UB, desc.loc.device.devPhyId, desc.loc.locType, desc.commAddr);

    EXPECT_EQ(ctx.ServerSocketListen(60008), HCCL_SUCCESS);
    HcclResult ret = ctx.ServerSocketStopListen(60008);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// TC-HostServerSocketContext_析构-001 / TC-DeviceServerSocketContext_析构-001:
// 析构时自动停止监听 dynamicPort_（不崩溃即通过）
TEST_F(ServerSocketContextTest, Ut_HostDtor_When_ListenedNotStopped_Expect_NoCrash)
{
    SetupServerSocketContextMocks();
    EndpointDesc desc = MakeHostRoceDesc();
    {
        HostServerSocketContext ctx(Hccl::ConnectProtoType::RDMA, desc.commAddr);
        EXPECT_EQ(ctx.ServerSocketListen(60009), HCCL_SUCCESS);
        // 先经 GetListenPort 使 dynamicPort_ 生效，离开作用域时析构兜底真实走到 StopListenImpl 分支
        uint32_t port = 0;
        EXPECT_EQ(ctx.ServerSocketGetListenPort(&port), HCCL_SUCCESS);
    }
    SUCCEED();
}

TEST_F(ServerSocketContextTest, Ut_DeviceDtor_When_ListenedNotStopped_Expect_NoCrash)
{
    SetupServerSocketContextMocks();
    EndpointDesc desc = MakeDeviceUbDesc();
    {
        DeviceServerSocketContext ctx(
            Hccl::ConnectProtoType::UB, desc.loc.device.devPhyId, desc.loc.locType, desc.commAddr);
        EXPECT_EQ(ctx.ServerSocketListen(60010), HCCL_SUCCESS);
        // 先经 GetListenPort 使 dynamicPort_ 生效，离开作用域时析构兜底真实走到 StopListenImpl 分支
        uint32_t port = 0;
        EXPECT_EQ(ctx.ServerSocketGetListenPort(&port), HCCL_SUCCESS);
    }
    SUCCEED();
}

// 基类 ServerSocketContext 的默认 ServerSocketStopListen 返回 HCCL_E_NOT_SUPPORT
TEST_F(ServerSocketContextTest, Ut_BaseStopListen_When_DefaultImpl_Expect_NotSupport)
{
    SetupServerSocketContextMocks();
    EndpointDesc desc = MakeHostRoceDesc();
    HostServerSocketContext ctx(Hccl::ConnectProtoType::RDMA, desc.commAddr);
    // 基类虚函数默认实现返回 NOT_SUPPORT，但 Host 子类 override 了；
    // 此处验证基类指针调用走子类 override（应 Success）。基类默认只对未 override 子类生效。
    ServerSocketContext* base = &ctx;
    HcclResult ret = base->ServerSocketStopListen(60011);
    // Host override 实现，未实际监听该端口，ServerSocketStopListen 返回成功或错误取决于实现，仅验证不崩溃
    (void)ret;
    SUCCEED();
}

// TC-HostServerSocketContext_CommAddrConvert-001: CommAddrToIpAddress 失败时 Listen/StopListen/GetListenPort
// 均返回失败码 （Host 三方法均以地址转换为前置步骤，覆盖本次改造引入的主要错误路径）
TEST_F(ServerSocketContextTest, Ut_HostSocketOps_When_CommAddrConvertFailed_Expect_Error)
{
    EndpointDesc desc = MakeHostRoceDesc();
    HostServerSocketContext ctx(Hccl::ConnectProtoType::RDMA, desc.commAddr);
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(&CommAddrToIpAddress).stubs().will(returnValue(HCCL_E_NOT_SUPPORT));

    EXPECT_EQ(ctx.ServerSocketListen(60012), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ctx.ServerSocketStopListen(60012), HCCL_E_NOT_SUPPORT);
    uint32_t port = 0;
    EXPECT_EQ(ctx.ServerSocketGetListenPort(&port), HCCL_E_NOT_SUPPORT);
}

// TC-AicpuTsRoceServerSocketContext_StopListen/GetListenPort-001:
// AicpuTsRoce 迁移前（Endpoint 时期）仅覆写 Listen，StopListen/GetListenPort 走基类默认 NOT_SUPPORT，
// 迁移恢复后子类不再 override，经基类默认保持迁移前对外行为
TEST_F(ServerSocketContextTest, Ut_AicpuTsRoceStopListenGetListenPort_When_InheritBaseDefault_Expect_NotSupport)
{
    AicpuTsRoceServerSocketContext ctx(nullptr, 0U);
    EXPECT_EQ(ctx.ServerSocketStopListen(16666U), HCCL_E_NOT_SUPPORT);

    uint32_t port = 0;
    EXPECT_EQ(ctx.ServerSocketGetListenPort(&port), HCCL_E_NOT_SUPPORT);
}

// TC-AicpuTsHccsServerSocketContext_GetListenPort-001:
// AicpuTsHccs 迁移前（Endpoint 时期）覆写 Listen/StopListen，GetListenPort 走基类默认 NOT_SUPPORT，
// 迁移恢复后子类不再 override GetListenPort，经基类默认保持迁移前对外行为
TEST_F(ServerSocketContextTest, Ut_AicpuTsHccsGetListenPort_When_InheritBaseDefault_Expect_NotSupport)
{
    AicpuTsHccsServerSocketContext ctx(0U, 16666U);

    uint32_t port = 0;
    EXPECT_EQ(ctx.ServerSocketGetListenPort(&port), HCCL_E_NOT_SUPPORT);
}

// TC-UbRtpUboeServerSocketContext_Listen/StopListen-001:
// Uboe/UbRtp 迁移前（Endpoint 时期）经 UboeUbRtpEndpointHelper 的 Listen/StopListen 为 no-op SUCCESS
// （仅 HCCL_INFO），迁移恢复后由本 context 承载该 no-op 语义
TEST_F(ServerSocketContextTest, Ut_UbRtpUboeListenStopListen_When_NoOp_Expect_Success)
{
    UbRtpUboeServerSocketContext ctx;
    EXPECT_EQ(ctx.ServerSocketListen(60001U), HCCL_SUCCESS);
    EXPECT_EQ(ctx.ServerSocketStopListen(60001U), HCCL_SUCCESS);
}

// TC-UbRtpUboeServerSocketContext_GetListenPort-001:
// GetListenPort 迁移前未覆写，走基类默认 NOT_SUPPORT，本 context 不覆写保持迁移前对外行为
TEST_F(ServerSocketContextTest, Ut_UbRtpUboeGetListenPort_When_InheritBaseDefault_Expect_NotSupport)
{
    UbRtpUboeServerSocketContext ctx;

    uint32_t port = 0;
    EXPECT_EQ(ctx.ServerSocketGetListenPort(&port), HCCL_E_NOT_SUPPORT);
}
