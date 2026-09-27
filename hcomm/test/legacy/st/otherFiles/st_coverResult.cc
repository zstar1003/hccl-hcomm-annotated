/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define protected public
#define private public
#include "dev_capability.h"
#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "test_mock_setup.h"
#include "base_config_legacy.h"
#include "cfg_field.h"
#include "env_config_v2.h"
#include "env_func.h"
#include "dev_capability.h"
#include "orion_adapter_hccp.h"
#include "ins_rules.h"
#include "instruction.h"
#include "coll_service_base.h"
#include "log.h"
#include "whitelist.h"
#include "invalid_params_exception.h"
#include "internal_exception.h"
#include "whitelist_test.h"
#include "hccp_peer_manager.h"
#include "dev_type.h"
#include "json_parser.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_tsd.h"
#include "network_api_exception.h"
#include "instruction.h"
#include "orion_adapter_hccp.h"
#include "socket.h"
#include "hccp_peer_manager.h"
#include "socket_agent.h"
#include "host_socket_handle_manager.h"
#include "coll_service_default_impl.h"
#include "communicator_impl.h"
#include "local_rma_buffer.h"
#include "remote_rma_buffer.h"
#include "rts_cnt_notify.h"
#include "host_ip_not_found_exception.h"
#include "sal.h"
#include "coll_alg_component.h"
#include "virtual_topo_stub.h"
#include "local_notify_v2.h"
#include "remote_notify.h"
#include "dev_ub_connection.h"
#include "rma_conn_manager.h"
#include "not_support_exception.h"
#include "socket_manager.h"
#include "rma_connection.h"
#include "conn_local_notify_manager.h"
#include "exchange_ub_buffer_dto.h"
#include "exchange_ipc_buffer_dto.h"
#include "exchange_ipc_notify_dto.h"
#include "../fwk/ranktable/stub_rank_table.h"
#include "hccp.h"
#include "hccp_ctx.h"
#include "hccp_common.h"
#include "hccp_async.h"
#include "local_rdma_rma_buffer_v2.h"
#include "local_ipc_rma_buffer_v2.h"
#include "local_ub_rma_buffer.h"
#include "topo_common_types.h"
#include "ccu_ins_preprocessor.h"
#include "aicpu_ins_preprocessor.h"
#undef private
#undef protected

#define private public
#include "rdma_handle_manager.h"
#undef private

using namespace Hccl;

class CoverageResult : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CoverageResult set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CoverageResult tear down" << std::endl; }

    static std::unique_ptr<Socket> mockProducer(
        IpAddress& localIpAddress, IpAddress& remoteIpAddress, u32 listenPort, SocketHandle socketHandle,
        const std::string& tag, SocketRole socketRole, NicType nicType)
    {
        return std::make_unique<Socket>(
            socketHandle, localIpAddress, listenPort, remoteIpAddress, "stub", socketRole, nicType);
    }

    IpAddress GetAnIpAddress()
    {
        IpAddress ipAddress("1.0.0.0");
        return ipAddress;
    }
};

TEST(ST_SocketListenStop, st_HrtRaUbCtxInit_ok)
{
    IpAddress addr(1);

    HrtRaUbCtxInitParam initParam(HrtNetworkMode::HDC, 0, addr);

    RdmaHandle rdmaHandle = HrtRaUbCtxInit(initParam);
}

TEST(ST_SocketListenStop, st_HrtRaUbCtxDestroy_ok) { HrtRaUbCtxDestroy(nullptr); }

TEST(ST_SocketListenStop, st_HrtRaUbLocalMemReg_ok)
{
    uint64_t fakeAddr = 0;
    uint64_t fakeSize = 0;
    uint32_t fakeTokenValue = 0;
    uint64_t fakeTokenIdHandle = 0;

    HrtRaUbLocMemRegParam inParam(fakeAddr, fakeSize, fakeTokenValue, fakeTokenIdHandle);

    RdmaHandle fakeRdmaHandle = nullptr;

    HrtRaUbLocalMemRegOutParam result = HrtRaUbLocalMemReg(fakeRdmaHandle, inParam);

    EXPECT_EQ(0, result.tokenId);
}

TEST(ST_SocketListenStop, st_HrtRaUbLocalMemUnreg_ok)
{
    RdmaHandle fakeRdmaHandle = nullptr;
    HrtRaUbLocalMemUnreg(fakeRdmaHandle, 0);
}

TEST(ST_SocketListenStop, st_HrtRaUbRemoteMemImport_ok)
{
    uint8_t value[128];
    HrtRaUbRemMemImportedOutParam result = HrtRaUbRemoteMemImport(nullptr, value, 128, 0);

    EXPECT_EQ(0, result.targetSegVa);
}

TEST(ST_SocketListenStop, st_HrtRaUbRemoteMemUnimport_ok)
{
    RdmaHandle fakeRdmaHandle = nullptr;
    HrtRaUbRemoteMemUnimport(fakeRdmaHandle, 0);
}

TEST(ST_SocketListenStop, st_HrtRaUbCreateCq_ok)
{
    auto result = HrtRaUbCreateJfc(nullptr, HrtUbJfcMode::NORMAL);
    EXPECT_EQ(0, result);
}

TEST(ST_SocketListenStop, st_HrtRaUbDestroyCq_ok) { HrtRaUbDestroyJfc(nullptr, 0); }

TEST(ST_SocketListenStop, st_HrtRaUbCreateJetty_ok)
{
    HrtRaUbCreateJettyParam inParam{100, 100, 100, 100, HrtJettyMode::HOST_OPBASE, 0, 100, 100, 100, 100};

    HrtRaUbJettyCreatedOutParam result = HrtRaUbCreateJetty(nullptr, inParam);

    EXPECT_EQ(0, result.jettyVa);
}

TEST(ST_SocketListenStop, st_HrtRaUbDestroyJetty_ok) { HrtRaUbDestroyJetty(0); }

TEST(ST_SocketListenStop, st_RaUbImportJetty_ok)
{
    JettyImportCfg cfg;
    cfg.protocol = Hccl::TpProtocol::CTP;
    HrtRaUbJettyImportedOutParam result = RaUbTpImportJetty(nullptr, nullptr, 0, 0, cfg);
    EXPECT_EQ(0, result.targetJettyVa);
}

TEST(ST_SocketListenStop, st_HrtRaUbUnimportJetty_ok) { HrtRaUbUnimportJetty(nullptr, 0); }

TEST(ST_SocketListenStop, st_HrtRaUbJettyBind_ok) { HrtRaUbJettyBind(0, 0); }

TEST(ST_SocketListenStop, st_HrtRaUbJettyUnbind_ok) { HrtRaUbJettyUnbind(0); }

TEST(ST_SocketListenStop, st_HrtRaUbPostSend_ok)
{
    HrtRaUbSendWrReqParam in;
    in.opcode = HrtUbSendWrOpCode::WRITE_WITH_NOTIFY;
    HrtRaUbSendWrRespParam result = HrtRaUbPostSend(0, in);
    EXPECT_EQ(0, result.dieId);
}

TEST(LogTest, st_sal_log_printf_log2)
{
    MODULE_DEBUG("\r\r\r\r \r\r\r\r log 2.0 test"); /*提高覆盖率*/
    MODULE_INFO("log 2.0 test");
    MODULE_WARNING("log 2.0 test");
    MODULE_ERROR("log 2.0 test");
    MODULE_RUN_INFO("<START AllReduce>");
    MODULE_RUN_INFO("<END AllReduce>");
}

// 代码覆盖率 log2.0格式对齐
TEST(LogTest, st_sal_log_printf_log2_destruct)
{
    MODULE_DEBUG(" \r log 2.0 test \n ");
    MODULE_INFO("log 2.0 test");
    MODULE_WARNING("log 2.0 test");
    MODULE_ERROR("log 2.0 test");
    MODULE_RUN_INFO("<START AllReduce>");
}

TEST(LogTest, st_sal_log_printf_error_0)
{
    LOG_PRINT(6 | RUN_LOG_MASK, "test \\n info \n");
    MODULE_DEBUG(nullptr);
    MODULE_INFO(nullptr);
    MODULE_WARNING(nullptr);
    MODULE_ERROR(nullptr);
    MODULE_RUN_INFO(nullptr);
    CallDlogInvalidType(HCCL_LOG_RUN_INFO, 1, "test", 2);
    CallDlogNoSzFormat(HCCL_LOG_RUN_INFO, 1, "test", 2);
    CallDlogMemError(HCCL_LOG_RUN_INFO, "test", 2);
    CallDlogPrintError(HCCL_LOG_RUN_INFO, "test", 2);
    CallDlog(HCCL_LOG_RUN_INFO, 1, "test", "test", 2);
    CallDlogInvalidType(HCCL_LOG_OPLOG, 1, "test", 2);
    CallDlogNoSzFormat(HCCL_LOG_OPLOG, 1, "test", 2);
    CallDlogMemError(HCCL_LOG_OPLOG, "test", 2);
    CallDlogPrintError(HCCL_LOG_OPLOG, "test", 2);
    CallDlog(HCCL_LOG_OPLOG, 1, "test", "test", 2);
}

TEST(ST_WhiteListTest, st_get_host_whitelist)
{
    IpAddress ipAddress("1.0.0.0");
    std::vector<IpAddress> whiteList;
    whiteList.push_back(ipAddress);
    Whitelist::GetInstance().GetHostWhiteList(whiteList);
}

TEST(ST_WhiteListTest, st_whitelist_load_config_file)
{
    std::string name;
    EXPECT_THROW(Whitelist::GetInstance().LoadConfigFile(name), InvalidParamsException);

    name = "whitelist";
    EXPECT_THROW(Whitelist::GetInstance().LoadConfigFile(name), InternalException);

    name = "whitelist.json";
    GenWhiteListFile();
    Whitelist::GetInstance().LoadConfigFile(name);

    IpAddress ipAddress("1.0.0.0");
    std::vector<IpAddress> whiteList;
    whiteList.push_back(ipAddress);
    Whitelist::GetInstance().GetHostWhiteList(whiteList);
    DelWhiteListFile();
}

TEST(ST_HccpPeerManagerTest, st_hccp_peer_manager_getInstance)
{
    // Given
    SETUP_HCCP_PEER_MGR_MOCKS(3, 4);
    // when
    s32 deviceLogicId = 0;
    HccpPeerManager::GetInstance().Init(deviceLogicId);
    s32 deviceLogicId1 = 1;
    HccpPeerManager::GetInstance().Init(deviceLogicId1);
    auto res = HccpPeerManager::GetInstance().instances_;

    // then
    EXPECT_EQ(2, res.size());
}

TEST(ST_HccpPeerManagerTest, st_hccp_peer_manager_init)
{
    // Given
    s32 deviceLogicId = 0;
    s32 deviceLogicId1 = 1;
    s32 deviceLogicId2 = 2;
    s32 fakedevPhyId = 3;
    MOCKER(HrtGetDevicePhyIdByUserDevId)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(static_cast<DevId>(fakedevPhyId)));
    MOCKER(HrtRaDeInit).stubs().with();
    // when
    HccpPeerManager::GetInstance().Init(deviceLogicId);
    auto res1 = HccpPeerManager::GetInstance().instances_;
    HccpPeerManager::GetInstance().Init(deviceLogicId);
    auto res2 = HccpPeerManager::GetInstance().instances_;

    // then
    EXPECT_EQ(res1[deviceLogicId].Count() + 1, res2[deviceLogicId].Count());

    // when
    HccpPeerManager::GetInstance().Init(deviceLogicId);
    HccpPeerManager::GetInstance().Init(deviceLogicId);
    HccpPeerManager::GetInstance().Init(deviceLogicId1);
    HccpPeerManager::GetInstance().Init(deviceLogicId1);
    HccpPeerManager::GetInstance().Init(deviceLogicId2);
    HccpPeerManager::GetInstance().Init(deviceLogicId2);
    auto res = HccpPeerManager::GetInstance().instances_;

    // then
    EXPECT_EQ(3, res.size());
}

TEST(AdapterHccpTest, HrtRaSocketWhiteListDel_nok)
{
    // Given
    MOCKER(RaSocketWhiteListDel).stubs().will(returnValue(-1));
    vector<RaSocketWhitelist> whiteList(1);
    // when

    // then
    EXPECT_THROW(HrtRaSocketWhiteListDel(nullptr, whiteList), NetworkApiException);
}

TEST(AdapterHccpTest, HrtGetHosIf_nok_ra_get_ifnum_error)
{
    MOCKER(RaGetIfnum).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1));
    EXPECT_THROW(HrtGetHostIf(0), NetworkApiException);
}

TEST(AdapterHccpTest, HrtRaMrReg_deReg_NOK)
{
    // Given
    MOCKER(RaMrReg).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1));

    QpHandle qpHandle;
    struct RaMrInfo mrInfo;
    // when

    EXPECT_THROW(HrtRaMrReg(qpHandle, mrInfo), NetworkApiException);

    MOCKER(RaMrDereg).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1));
    EXPECT_THROW(HrtRaMrDereg(qpHandle, mrInfo), NetworkApiException);
}

TEST(AdapterHccpTest, HrtHrtRaRdmaInit_NOK)
{
    // Given
    u32* num = new u32[1];
    RdmaHandle rdmaHandle = static_cast<void*>(num);
    MOCKER(RaRdevInit)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&rdmaHandle, sizeof(rdmaHandle)))
        .will(returnValue(1));

    struct RaInterface rdevInfo;
    // when

    EXPECT_THROW(HrtRaRdmaInit(HrtNetworkMode::HDC, rdevInfo), NetworkApiException);
    delete[] num;
}

TEST_F(AdapterHccpTest, HrtHrtRaRdmaInit_return_HCCP_ELINKDOWN_NOK)
{
    // Given
    u32* num = new u32[1];
    RdmaHandle rdmaHandle = static_cast<void*>(num);
    MOCKER(RaRdevInit)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&rdmaHandle, sizeof(rdmaHandle)))
        .will(returnValue(HCCP_ELINKDOWN));

    struct RaInterface rdevInfo;
    // when

    EXPECT_THROW(HrtRaRdmaInit(HrtNetworkMode::HDC, rdevInfo), NetworkApiException);
    delete[] num;
}

TEST(AdapterHccpTest, HrtGetHosIf_nok_ra_get_ifaddrs_error)
{
    unsigned int fakeNum = 1;
    MOCKER(RaGetIfnum).stubs().with(mockcpp::any(), outBoundP(&fakeNum, sizeof(fakeNum))).will(returnValue(0));
    MOCKER(RaGetIfaddrs).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(1));

    EXPECT_THROW(HrtGetHostIf(0), NetworkApiException);
}

TEST(ST_AdapterHccpTest, st_HrtGetDeviceIp_nok)
{
    u32 devPhyId = 0;

    EXPECT_THROW(HrtGetDeviceIp(devPhyId), NetworkApiException);
}

TEST(ST_AdapterHccpTest, st_HrtRaUbPostNops_ok) { HrtRaUbPostNops(0, 0, 1); }

TEST(ST_AdapterHccpTest, st_HrtRaUbPostNops_exception)
{
    MOCKER(RaBatchSendWr).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_THROW(HrtRaUbPostNops(0, 0, 1), NetworkApiException);
}

TEST(AdapterHccpTest, RaUbUpdateCi_ok)
{
    JettyHandle jettyHandle = 0;
    RaUbUpdateCi(jettyHandle, 100);
}

TEST(AdapterHccpTest, RaUbUpdateCi_exception)
{
    JettyHandle jettyHandle = 0;
    MOCKER(RaCtxUpdateCi).stubs().will(returnValue(1));
    EXPECT_THROW(RaUbUpdateCi(jettyHandle, 100), NetworkApiException);
}

TEST(AdapterHccpTest, HrtRaSocketWhiteListAdd_nok)
{
    // Given
    MOCKER(RaSocketWhiteListAdd).stubs().will(returnValue(-1));
    vector<RaSocketWhitelist> whiteList(1);
    // when

    // then
    EXPECT_THROW(HrtRaSocketWhiteListAdd(nullptr, whiteList), NetworkApiException);
}

TEST(AdapterHccpTest, HrtRaSocketWhiteListAdd_strcpy_nok)
{
    // Given
    MOCKER(strcpy_s).stubs().will(returnValue(-1));
    vector<RaSocketWhitelist> whiteList(1);
    // when

    // then
    EXPECT_THROW(HrtRaSocketWhiteListAdd(nullptr, whiteList), InternalException);
}

TEST(InstructionTest, test_ins_wait_group)
{
    QId postQid = 0;
    u32 topicId = 0;
    QId waitQid = 1;
    InsLocalWaitGroup insWaitGroup(topicId);
    cout << insWaitGroup.Describe() << endl;
    insWaitGroup.Append(postQid);
    insWaitGroup.SetWaitQid(waitQid);
    EXPECT_EQ(postQid, *(insWaitGroup.Iter()));
    EXPECT_EQ(waitQid, insWaitGroup.GetWaitQid());
    EXPECT_EQ(topicId, insWaitGroup.GetTopicId());

    EXPECT_THROW(insWaitGroup.SetWaitQid(postQid), InvalidParamsException);
}

TEST(CountNotifyTest, test_rts_cnt_notify)
{
    u64 fakeNotifyHandleAddr = 300;
    u32 fakeNotifyId = 1;

    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtCntNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
    MOCKER(HrtGetCntNotifyId).stubs().will(returnValue(fakeNotifyId));

    RtsCntNotify rtsCntNotify;
    std::string des = rtsCntNotify.Describe();

    QueueWaitGroupCntNotifyManager queueWaitGroupCntNotifyManager;

    QId qid = 1;
    u32 index = 1;

    auto result = queueWaitGroupCntNotifyManager.Get(qid, index);
    queueWaitGroupCntNotifyManager.ApplyFor(qid, index);
    auto result1 = queueWaitGroupCntNotifyManager.Get(qid, index);

    GlobalMockObject::reset();
}

TEST(CountNotifyTest, test_rts_1ton_cnt_notify)
{
    u64 fakeNotifyHandleAddr = 300;
    u32 fakeNotifyId = 1;

    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtCntNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
    MOCKER(HrtGetCntNotifyId).stubs().will(returnValue(fakeNotifyId));

    Rts1ToNCntNotify rts1ToNCntNotify;
    std::string des = rts1ToNCntNotify.Describe();

    QueueBcastPostCntNotifyManager queueBcastPostCntNotifyManager;

    QId qid = 1;
    u32 index = 1;

    auto result = queueBcastPostCntNotifyManager.Get(qid, index);
    queueBcastPostCntNotifyManager.ApplyFor(qid, index);
    auto result1 = queueBcastPostCntNotifyManager.Get(qid, index);

    GlobalMockObject::reset();
}

TEST(LocalNotifyTest, ipc_local_notify_test)
{
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    u64 fakeAddress = 300;
    u32 fakePid = 100;
    char fakeName[RTS_IPC_MEM_NAME_LEN] = "testRtsNotify";

    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_910A2)));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));
    MOCKER(HrtIpcSetNotifyName).stubs().with(mockcpp::any(), outBoundP(fakeName, sizeof(fakeName)), mockcpp::any());
    MOCKER(HrtGetNotifyID).stubs().will(returnValue(fakeNotifyId));
    MOCKER(HrtNotifyGetAddr).stubs().with(mockcpp::any()).will(returnValue(fakeAddress));
    MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(fakeOffset));
    MOCKER(HrtDeviceGetBareTgid).stubs().will(returnValue(fakePid));
};

TEST(LocalRmaBufferTest, localubrmabuffer_construct_error)
{
    EXPECT_THROW(LocalUbRmaBuffer localUbRmaBuffer(nullptr, nullptr), NullPtrException);
};

TEST(LocalRmaBufferTest, getExchangeDto_test)
{
    MOCKER(GetUbToken).stubs().will(returnValue(1));
    std::shared_ptr<DevBuffer> devBuf = DevBuffer::Create(0x100, 0x100);
    RdmaHandle rdmaHandle = (void*)0x1000000;
    LocalUbRmaBuffer localUbRmaBuffer(devBuf, rdmaHandle);
    localUbRmaBuffer.GetExchangeDto();

    MOCKER(HrtIpcSetMemoryName).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
    MOCKER(HrtDevMemAlignWithPage)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
    MOCKER(HrtIpcDestroyMemoryName).stubs().with(mockcpp::any());
};

TEST(LocalRmaBufferTest, localubrmabuffer_serialize)
{
    HrtRaUbLocalMemRegOutParam localMemRegInfo;
    u64 fakeSegVa = 0x200;
    u8 fakeKey[HRT_UB_MEM_KEY_MAX_LEN]{0};
    u32 fakeTokenId = 1;
    u64 fakeMemHandle = 0x200;

    localMemRegInfo.handle = fakeMemHandle;
    memcpy_s(localMemRegInfo.key, HRT_UB_MEM_KEY_MAX_LEN, fakeKey, HRT_UB_MEM_KEY_MAX_LEN);
    localMemRegInfo.tokenId = fakeTokenId;
    localMemRegInfo.targetSegVa = fakeSegVa;
    localMemRegInfo.keySize = 4;

    RequestHandle fakeReqHandle = 1;

    vector<char_t> out;
    out.resize(sizeof(struct MrRegInfoT));
    struct MrRegInfoT* info = reinterpret_cast<struct MrRegInfoT*>(out.data());
    memcpy_s(info->out.key.value, HRT_UB_MEM_KEY_MAX_LEN, fakeKey, HRT_UB_MEM_KEY_MAX_LEN);
    info->out.key.size = 4;
    info->out.ub.tokenId = fakeTokenId;
    info->out.ub.targetSegHandle = fakeSegVa;

    MOCKER(RaUbLocalMemRegAsync)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(out), outBound(reinterpret_cast<void*>(fakeMemHandle)))
        .will(returnValue(fakeReqHandle));

    MOCKER(HrtRaUbLocalMemUnreg).stubs();
    MOCKER(GetUbToken).stubs().will(returnValue(1));
    BufferType type = BufferType::INPUT;
    u32 a = 0;
    void* ptr = static_cast<void*>(&a);
    u64 size = 0;
    bool remoteAccess = true;

    std::shared_ptr<DevBuffer> buf = DevBuffer::Create(0x100, 0x100);
    RdmaHandle rdmaHandle = (void*)0x1000000;
    LocalUbRmaBuffer localUbRmaBuffer(buf, rdmaHandle);
    localUbRmaBuffer.Describe();

    UbRmaBufferExchangeData exchangeData;
    exchangeData.addr = buf->GetAddr();
    exchangeData.size = buf->GetSize();
    exchangeData.tokenValue = 1;
    exchangeData.tokenId = 1;
    exchangeData.keySize = 4;
    u8 key[HRT_UB_MEM_KEY_MAX_LEN]{};
    memcpy_s(exchangeData.key, HRT_UB_MEM_KEY_MAX_LEN, key, HRT_UB_MEM_KEY_MAX_LEN);

    GlobalMockObject::verify();
};

TEST(LocalRmaBufferTest, generate_safe_random_number)
{
    MOCKER(HrtGetDevice).stubs().will(returnValue(1));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(1)));
    MOCKER(HrtRaGetSecRandom).stubs().with(mockcpp::any(), mockcpp::any());
    u32 token = GetUbToken();
};

TEST(RemoteRmaBufferTest, remoteubrmabuffer_construct_error)
{
    EXPECT_THROW(RemoteUbRmaBuffer remoteUbRmaBuffer(nullptr), NullPtrException);
};

TEST(RemoteRmaBufferTest, remoteubrmabuffer_deserialize_success)
{
    // construct buffer
    BufferType type = BufferType::INPUT;
    void* ptr = nullptr;
    u64 size = 0;
    bool remoteAccess = true;

    std::shared_ptr<DevBuffer> devBuf = DevBuffer::Create(0x100, 0x100);
    Buffer* buf = devBuf.get();

    u32 tokenValue = 1;
    u32 tokenId = 0;
    u64 keySize = 4;
    u8 key[HRT_UB_MEM_KEY_MAX_LEN]{};

    BinaryStream binaryStream;
    HcclMemType memType = HcclMemType::HCCL_MEM_TYPE_DEVICE;
    binaryStream << buf->GetAddr() << buf->GetSize() << memType << tokenValue << tokenId << keySize << key;
    ExchangeUbBufferDto dto;
    dto.Deserialize(binaryStream);

    RdmaHandle rdmaHandle = (void*)0x1000000;
    RemoteUbRmaBuffer remoteUbRmaBuffer(rdmaHandle, dto);
    remoteUbRmaBuffer.Describe();

    UbRmaBufferExchangeData exchangeData;
    exchangeData.addr = buf->GetAddr();
    exchangeData.size = buf->GetSize();
    exchangeData.tokenValue = tokenValue;
    exchangeData.tokenId = tokenId;
    memcpy_s(exchangeData.key, HRT_UB_MEM_KEY_MAX_LEN, key, HRT_UB_MEM_KEY_MAX_LEN);

    EXPECT_EQ(exchangeData.tokenValue, remoteUbRmaBuffer.tokenValue);
    EXPECT_EQ(exchangeData.tokenId, remoteUbRmaBuffer.tokenId);
    EXPECT_EQ(memcmp(remoteUbRmaBuffer.key, exchangeData.key, HRT_UB_MEM_KEY_MAX_LEN), 0);
    EXPECT_EQ(buf->GetAddr(), remoteUbRmaBuffer.addr);
    EXPECT_EQ(buf->GetSize(), remoteUbRmaBuffer.size);
};

TEST(RemoteRmaBufferTest, remoterdmarmabuffer_describe_size)
{
    RdmaHandle rdmaHandle = (void*)0x1000000;
    RemoteRdmaRmaBuffer remoteRdmaRmaBuffer(rdmaHandle);

    std::string fakeKeyDesc = "fakeKeyDesc";
    MOCKER(HrtRaGetKeyDescribe).stubs().will(returnValue(fakeKeyDesc));
    remoteRdmaRmaBuffer.Describe();
}

TEST(DevUbConnectionTest, rma_ub_connection_get_status_return_exchanging_and_ok)
{
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    std::string tag = "test";
    fakeSocket = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    // Given
    MOCKER_CPP(&Socket::GetAsyncStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));

    RdmaHandle rdmaHandle = (void*)0x1000000;
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // construct DevUbConnection
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    devUbConnection.tpProtocol = TpProtocol::TP;
    unsigned long long sentSize = 0;
    //  When:
    u32 tokenValue = 1;
    MOCKER_CPP(&Socket::SendAsync).stubs().will(returnValue(true));
    char* responseMsg = "connect ready!";
    MOCKER_CPP(&Socket::RecvAsync)
        .stubs()
        .with(outBoundP(reinterpret_cast<u8*>(responseMsg), (u32)15), mockcpp::any())
        .will(returnValue(true));
    // Then
    auto res = devUbConnection.GetStatus();
    EXPECT_EQ(RmaConnStatus::INIT, res);
    EXPECT_EQ(DevUbConnection::UbConnStatus::TP_INFO_GETTING, devUbConnection.ubConnStatus);
    res = devUbConnection.GetStatus();
    EXPECT_EQ(RmaConnStatus::EXCHANGEABLE, res);
    EXPECT_EQ(DevUbConnection::UbConnStatus::JETTY_CREATED, devUbConnection.ubConnStatus);

    auto rmtDto = devUbConnection.GetExchangeDto();
    devUbConnection.ParseRmtExchangeDto(*rmtDto);
    devUbConnection.ImportRmtDto();

    res = devUbConnection.GetStatus();
    EXPECT_EQ(RmaConnStatus::READY, res);
    EXPECT_EQ(DevUbConnection::UbConnStatus::READY, devUbConnection.ubConnStatus);

    res = devUbConnection.GetStatus();
    EXPECT_EQ(RmaConnStatus::READY, res);

    string msg = devUbConnection.Describe();

    delete fakeSocket;
}

TEST(DevUbConnectionTest, rma_net_connection_prepare_write_task)
{
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    string tag = "SENDRECV";
    fakeSocket = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    MOCKER(HrtRaQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(fakeQpHandle));

    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    // Given
    MemoryBuffer localMemBuffer1(0, 0, 0);
    MemoryBuffer remoteMemBuffer1(2000, 0, 0);
    SqeConfig config{};
    // When
    auto result1 = devUbConnection.PrepareWrite(remoteMemBuffer1, localMemBuffer1, config);
    // Then
    EXPECT_EQ(nullptr, result1);

    MemoryBuffer localMemBuffer2(0, 100, 0);
    MemoryBuffer remoteMemBuffer2(2000, 100, 0);
    auto result2 = devUbConnection.PrepareWrite(remoteMemBuffer2, localMemBuffer2, config);
    EXPECT_NE(nullptr, result2);

    MemoryBuffer localMemBuffer10(0, 0, 0);
    MemoryBuffer remoteMemBuffer10(2000, 10, 0);
    EXPECT_THROW(devUbConnection.PrepareWrite(remoteMemBuffer10, localMemBuffer10, config), InvalidParamsException);
    delete fakeSocket;
}

TEST(DevUbConnectionTest, rma_net_connection_prepare_write_task_with_dwqe)
{
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    string tag = "SENDRECV";
    fakeSocket = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    MOCKER(HrtRaQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(fakeQpHandle));
    HrtRaUbSendWrRespParam postSendRes;
    postSendRes.dwqeSize = 128;
    MOCKER(HrtRaUbPostSend).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(postSendRes));

    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    // Given
    MemoryBuffer localMemBuffer1(0, 0, 0);
    MemoryBuffer remoteMemBuffer1(2000, 0, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DWQE;
    // When
    auto result1 = devUbConnection.PrepareWrite(remoteMemBuffer1, localMemBuffer1, config);
    // Then
    EXPECT_EQ(nullptr, result1);

    MemoryBuffer localMemBuffer2(0, 100, 0);
    MemoryBuffer remoteMemBuffer2(2000, 100, 0);
    auto result2 = devUbConnection.PrepareWrite(remoteMemBuffer2, localMemBuffer2, config);
    EXPECT_NE(nullptr, result2);

    MemoryBuffer localMemBuffer10(0, 0, 0);
    MemoryBuffer remoteMemBuffer10(2000, 10, 0);
    EXPECT_THROW(devUbConnection.PrepareWrite(remoteMemBuffer10, localMemBuffer10, config), InvalidParamsException);
    delete fakeSocket;
    GlobalMockObject::verify();
}

TEST(DevUbConnectionTest, rma_net_connection_prepare_read_task)
{
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    string tag = "SENDRECV";
    fakeSocket = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    // Given
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    SqeConfig config{};
    auto task = devUbConnection.PrepareRead(remoteMemBuffer, localMemBuffer, config);
    EXPECT_NE(nullptr, task);
    delete fakeSocket;
}

TEST(DevUbConnectionTest, rma_net_connection_prepare_read_task_with_dwqe)
{
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    string tag = "SENDRECV";
    fakeSocket = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    // Given
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    HrtRaUbSendWrRespParam postSendRes;
    postSendRes.dwqeSize = 128;
    MOCKER(HrtRaUbPostSend).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(postSendRes));

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DWQE;
    auto task = devUbConnection.PrepareRead(remoteMemBuffer, localMemBuffer, config);
    EXPECT_NE(nullptr, task);
    delete fakeSocket;
    GlobalMockObject::verify();
}

TEST(DevUbConnectionTest, rma_net_connection_prepare_read_reduce_task)
{
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    string tag = "SENDRECV";
    fakeSocket = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    // Given
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;

    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    SqeConfig config{};
    auto task
        = devUbConnection.PrepareReadReduce(remoteMemBuffer, localMemBuffer, DataType::INT8, ReduceOp::SUM, config);
    EXPECT_NE(nullptr, task);
    delete fakeSocket;
}

TEST(DevUbConnectionTest, rma_net_connection_prepare_read_reduce_task_with_dwqe)
{
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    string tag = "SENDRECV";
    fakeSocket = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    // Given
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;

    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    HrtRaUbSendWrRespParam postSendRes;
    postSendRes.dwqeSize = 128;
    MOCKER(HrtRaUbPostSend).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(postSendRes));

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DWQE;
    auto task
        = devUbConnection.PrepareReadReduce(remoteMemBuffer, localMemBuffer, DataType::INT8, ReduceOp::SUM, config);
    EXPECT_NE(nullptr, task);
    delete fakeSocket;
    GlobalMockObject::verify();
}

TEST(DevUbConnectionTest, rma_ub_connection_prepare_write_reduce_task)
{
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    string tag = "SENDRECV";
    fakeSocket = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    // Given
    RdmaHandle rdmaHandle = (void*)0x1000000;
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    SqeConfig config{};

    auto task
        = devUbConnection.PrepareWriteReduce(remoteMemBuffer, localMemBuffer, DataType::INT8, ReduceOp::SUM, config);
    EXPECT_NE(nullptr, task);

    delete fakeSocket;
}

TEST(DevUbConnectionTest, rma_ub_connection_prepare_write_reduce_task_with_dwqe)
{
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    string tag = "SENDRECV";
    fakeSocket = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    // Given
    RdmaHandle rdmaHandle = (void*)0x1000000;
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    HrtRaUbSendWrRespParam postSendRes;
    postSendRes.dwqeSize = 128;
    MOCKER(HrtRaUbPostSend).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(postSendRes));

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DWQE;

    auto task
        = devUbConnection.PrepareWriteReduce(remoteMemBuffer, localMemBuffer, DataType::INT8, ReduceOp::SUM, config);
    EXPECT_NE(nullptr, task);

    delete fakeSocket;
    GlobalMockObject::verify();
}

TEST(DevUbConnectionTest, rma_net_connection_prepare_write_with_notify_task)
{
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    string tag = "SENDRECV";
    fakeSocket = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    MOCKER(HrtRaQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(fakeQpHandle));

    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    // Given
    MemoryBuffer localMemBuffer(0, 100, 0);
    MemoryBuffer remoteMemBuffer(2000, 100, 0);
    MemoryBuffer remoteNotifyMemBuffer(8000, 100, 0);
    SqeConfig config{};

    auto task
        = devUbConnection.PrepareWriteWithNotify(remoteMemBuffer, localMemBuffer, 1, remoteNotifyMemBuffer, config);
    EXPECT_NE(nullptr, task);

    delete fakeSocket;
}

TEST(DevUbConnectionTest, rma_net_connection_prepare_write_with_notify_task_with_dwqe)
{
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    string tag = "SENDRECV";
    fakeSocket = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    MOCKER(HrtRaQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(fakeQpHandle));
    HrtRaUbSendWrRespParam postSendRes;
    postSendRes.dwqeSize = 128;
    MOCKER(HrtRaUbPostSend).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(postSendRes));

    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    // Given
    MemoryBuffer localMemBuffer(0, 100, 0);
    MemoryBuffer remoteMemBuffer(2000, 100, 0);
    MemoryBuffer remoteNotifyMemBuffer(8000, 100, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DWQE;

    auto task
        = devUbConnection.PrepareWriteWithNotify(remoteMemBuffer, localMemBuffer, 1, remoteNotifyMemBuffer, config);
    EXPECT_NE(nullptr, task);

    delete fakeSocket;
    GlobalMockObject::verify();
}

TEST(DevUbConnectionTest, rma_ub_connection_prepare_write_reduce_with_notify_task)
{
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    string tag = "SENDRECV";
    fakeSocket = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    // Given
    RdmaHandle rdmaHandle = (void*)0x1000000;
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    MemoryBuffer remoteNotifyMemBuffer(8000, 100, 0);
    SqeConfig config{};

    auto task = devUbConnection.PrepareWriteReduceWithNotify(
        remoteMemBuffer, localMemBuffer, DataType::INT8, ReduceOp::SUM, 1, remoteNotifyMemBuffer, config);
    EXPECT_NE(nullptr, task);

    delete fakeSocket;
}

TEST(DevUbConnectionTest, rma_ub_connection_prepare_write_reduce_with_notify_task_with_dwqe)
{
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    string tag = "SENDRECV";
    fakeSocket = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    // Given
    RdmaHandle rdmaHandle = (void*)0x1000000;
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    HrtRaUbSendWrRespParam postSendRes;
    postSendRes.dwqeSize = 128;
    MOCKER(HrtRaUbPostSend).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(postSendRes));

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    MemoryBuffer remoteNotifyMemBuffer(8000, 100, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DWQE;

    auto task = devUbConnection.PrepareWriteReduceWithNotify(
        remoteMemBuffer, localMemBuffer, DataType::INT8, ReduceOp::SUM, 1, remoteNotifyMemBuffer, config);
    EXPECT_NE(nullptr, task);

    delete fakeSocket;
    GlobalMockObject::verify();
}

TEST(DevUbConnectionTest, rma_ub_connection_prepare_inline_write_task)
{
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    string tag = "SENDRECV";
    fakeSocket = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);

    // Given
    RdmaHandle rdmaHandle = (void*)0x1000000;
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    SqeConfig config{};

    auto task = devUbConnection.PrepareInlineWrite(remoteMemBuffer, 1, config);
    EXPECT_NE(nullptr, task);

    delete fakeSocket;
}

TEST(ConnLocalNotifyManagerTest, apply_for_ub_notify_ok)
{
    CommunicatorImpl comm;
    ConnLocalNotifyManager connLocalNotifyManager(&comm);
    // Given
    MOCKER(HrtGetDevice).stubs().will(returnValue(1));
    MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(0)));
    MOCKER(HrtIpcSetNotifyName).stubs();
    MOCKER(HrtGetNotifyID).stubs().will(returnValue(1));
    MOCKER(HrtNotifyGetAddr).stubs().will(returnValue((u64)0));
    MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(1));
    MOCKER(HrtGetSocVer).stubs();
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_950));

    RankId fakeLocalRankID = 1;
    RankId fakeRemoteRankID = 4;
    u32 fakeLocalPortId = 3;
    u32 fakeRemotePortId = 2;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData fakeLinkData(basePortType, fakeLocalRankID, fakeRemoteRankID, fakeLocalPortId, fakeRemotePortId);
    pair<TokenIdHandle, uint32_t> fakeTokenInfo = make_pair(0x12345678, 1);
    MOCKER_CPP(&RdmaHandleManager::GetTokenIdInfo).stubs().will(returnValue(fakeTokenInfo));

    EXPECT_NO_THROW(connLocalNotifyManager.ApplyFor(fakeRemoteRankID, fakeLinkData));
}

TEST(TaskTest, test_task_ub_direct_send_create)
{
    u32 funcId = 0;
    u32 dieId = 0;
    u32 piVal = 0;
    u32 jettyId = 18;
    u32 dwqeSize = 128;
    u8 dwqe[128]{0};

    auto ubDirectSend = new TaskUbDirectSend(funcId, dieId, jettyId, dwqeSize, dwqe);

    cout << ubDirectSend->Describe() << endl;

    delete ubDirectSend;
}

TEST(TaskTest, test_task_ub_direct_send_exception)
{
    u32 funcId = 0;
    u32 dieId = 0;
    u32 piVal = 0;
    u32 jettyId = 18;
    u32 dwqeSize = 128;
    u8 dwqe[128]{0};

    EXPECT_THROW(new TaskUbDirectSend(funcId, dieId, jettyId, 0, dwqe), InternalException);
}

class StubRmaConnectionSync : public RmaConnection {
public:
    StubRmaConnectionSync(const LinkData& linkData, const RmaConnType rmaConnType)
        : link(linkData),
          RmaConnection(nullptr, rmaConnType)
    {}

    unique_ptr<BaseTask>
    PrepareRead(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config) override
    {
        return nullptr;
    }

    unique_ptr<BaseTask> PrepareReadReduce(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType datatype, ReduceOp reduceOp,
        const SqeConfig& config) override
    {
        return nullptr;
    }

    unique_ptr<BaseTask>
    PrepareWrite(const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, const SqeConfig& config) override
    {
        if (link.GetType() == PortDeploymentType::P2P) {
            return make_unique<TaskP2pMemcpy>(remoteMemBuf.addr, localMemBuf.addr, localMemBuf.size, MemcpyKind::D2D);
        }

        if (link.GetType() == PortDeploymentType::DEV_NET && link.GetLinkProtocol() == LinkProtocol::ROCE) {
            u32 dbIndex = 100;
            u64 dbInfo = 100;
            return make_unique<TaskRdmaSend>(dbIndex, dbInfo);
        }

        if (link.GetType() == PortDeploymentType::DEV_NET
            && (link.GetLinkProtocol() == LinkProtocol::UB_TP || link.GetLinkProtocol() == LinkProtocol::UB_CTP)) {
            u32 dieId = 1;
            u32 funcId = 1;
            u32 jettyId = 1;
            u32 piVal = 1;
            return make_unique<TaskUbDbSend>(dieId, funcId, jettyId, piVal);
        }
        return nullptr;
    }

    unique_ptr<BaseTask> PrepareWriteReduce(
        const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, DataType datatype, ReduceOp reduceOp,
        const SqeConfig& config) override
    {
        return nullptr;
    }

    unique_ptr<BaseTask>
    PrepareInlineWrite(const MemoryBuffer& remoteMemBuf, u64 data, const SqeConfig& config) override
    {
        if (link.GetType() == PortDeploymentType::DEV_NET
            && (link.GetLinkProtocol() == LinkProtocol::UB_TP || link.GetLinkProtocol() == LinkProtocol::UB_CTP)) {
            u32 dieId = 1;
            u32 funcId = 1;
            u32 jettyId = 1;
            u32 piVal = 1;
            return make_unique<TaskUbDbSend>(dieId, funcId, jettyId, piVal);
        }
        return nullptr;
    }

    string Describe() const override { return "StubRmaConnectionSync"; }

    void Connect() override {}

private:
    LinkData link;
};

const map<ReduceOp, bool> CAP_INLINE_REDUCE_OP_V82
    = {{ReduceOp::SUM, true},
       {ReduceOp::PROD, false},
       {ReduceOp::MAX, true},
       {ReduceOp::MIN, true},
       {ReduceOp::EQUAL, true}};

const map<DataType, bool> CAP_INLINE_REDUCE_DATATYPE_V82 = {
    {DataType::INT8, true},    {DataType::INT16, true},    {DataType::INT32, true},   {DataType::FP16, true},
    {DataType::FP32, true},    {DataType::INT64, false},   {DataType::UINT64, false}, {DataType::UINT8, true},
    {DataType::UINT16, true},  {DataType::UINT32, true},   {DataType::FP64, false},   {DataType::BFP16, true},
    {DataType::INT128, false}, {DataType::BF16_SAT, true},
};

const u32 CAP_NOTIFY_SIZE_V82 = 8;
const u32 CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_V82 = 32;

const u64 RDMA_SEND_MAX_SIZE = 0x80000000;  // 节点间RDMA发送数据单个WQE支持的最大数据量
const u64 SDMA_SEND_MAX_SIZE = 0x100000000; // 节点内单个SDMA任务发送数据支持的最大数据量

TEST(DevCapabilityTest, test_dev_cap_v82)
{
    const u32 CAP_NOTIFY_SIZE_V82 = 8;
    const u32 CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_V82 = 32;
    const u64 RDMA_SEND_MAX_SIZE = 0x80000000;  // 节点间RDMA发送数据单个WQE支持的最大数据量
    const u64 SDMA_SEND_MAX_SIZE = 0x100000000; // 节点内单个SDMA任务发送数据支持的最大数据量
    const map<ReduceOp, bool> CAP_INLINE_REDUCE_OP_V82
        = {{ReduceOp::SUM, true},
           {ReduceOp::PROD, false},
           {ReduceOp::MAX, true},
           {ReduceOp::MIN, true},
           {ReduceOp::EQUAL, true}};
    const map<DataType, bool> CAP_INLINE_REDUCE_DATATYPE_V82 = {
        {DataType::INT8, true},    {DataType::INT16, true},    {DataType::INT32, true},   {DataType::FP16, true},
        {DataType::FP32, true},    {DataType::INT64, false},   {DataType::UINT64, false}, {DataType::UINT8, true},
        {DataType::UINT16, true},  {DataType::UINT32, true},   {DataType::FP64, false},   {DataType::BFP16, true},
        {DataType::INT128, false}, {DataType::BF16_SAT, true},
    };
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    DevCapability& devCap = DevCapability::GetInstance();
    devCap.LoadV82Cap();
    EXPECT_EQ(CAP_NOTIFY_SIZE_V82, devCap.GetNotifySize());
    EXPECT_EQ(SDMA_SEND_MAX_SIZE, devCap.GetSdmaSendMaxSize());
    EXPECT_EQ(RDMA_SEND_MAX_SIZE, devCap.GetRdmaSendMaxSize());
    EXPECT_EQ(CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_V82, devCap.GetSdmaInlineReduceAlignBytes());
    EXPECT_EQ(CAP_INLINE_REDUCE_OP_V82, devCap.GetInlineReduceOpMap());
    EXPECT_EQ(CAP_INLINE_REDUCE_DATATYPE_V82, devCap.GetInlineReduceDataTypeMap());
    EXPECT_EQ(true, devCap.IsSupportStarsPollNetCq());
    EXPECT_EQ(true, devCap.IsSupportDevNetInlineReduce());
    devCap.Load910A3Cap();
    devCap.Load910ACap();
}

TEST(CommunicatorImplTest, should_return_success_when_calling_suspend)
{
    GlobalMockObject::verify();
    CommunicatorImpl comm;
    comm.InitRmaConnManager();
    EXPECT_EQ(false, comm.isSuspended);
    EXPECT_EQ(HCCL_SUCCESS, comm.Suspend());
    EXPECT_EQ(true, comm.isSuspended);

    CollOpParams opParams;
    opParams.staticAddr = true;
    opParams.staticShape = true;
    opParams.dataType = DataType::FP32;
    EXPECT_EQ(HCCL_E_SUSPENDING, comm.LoadOpbasedCollOp(opParams, nullptr));
}

TEST(LocalRmaBufferTest, Serialize_test)
{
    std::shared_ptr<DevBuffer> devBuf = DevBuffer::Create(0x100, 0x100);
    RdmaHandle rdmaHandle = (void*)0x1000000;
    LocalRdmaRmaBuffer localRdmaRmaBuffer(devBuf, rdmaHandle);

    localRdmaRmaBuffer.Describe();
};

TEST(LocalRmaBufferTest, getExchangeDto_ipc_test)
{
    std::shared_ptr<DevBuffer> devBuf = DevBuffer::Create(0x100, 0x100);
    LocalIpcRmaBuffer localIpcRmaBuffer(devBuf);

    localIpcRmaBuffer.Describe();
    localIpcRmaBuffer.GetExchangeDto();

    u32 pid = 1;
    MOCKER(HrtDeviceGetBareTgid).stubs().will(returnValue(pid));
    localIpcRmaBuffer.Grant(pid);
};

TEST(AdapterHccpTest, RaGetOneSocket_return_err_2)
{
    u32 connectedNum = 2;
    MOCKER(RaGetSockets)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&connectedNum))
        .will(returnValue(0));

    SocketHandle socketHandle = nullptr;
    IpAddress ipAddr = IpAddress();
    std::string tag = "";
    FdHandle fdHandle = nullptr;
    u32 role = 0;
    RaSocketGetParam param(socketHandle, ipAddr, tag, fdHandle);

    RequestHandle reqHandle = 12;
    EXPECT_THROW(RaGetOneSocket(role, param), NetworkApiException);
}

TEST(AdapterHccpTest, RaSocketCloseOneAsync_return_ok)
{
    MOCKER(RaSocketBatchCloseAsync).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(0));

    SocketHandle socketHandle = nullptr;
    FdHandle fdHandle = nullptr;
    RaSocketCloseParam param(socketHandle, fdHandle);

    RaSocketCloseOneAsync(param);
}

TEST(AdapterHccpTest, RaSocketListenOneStopAsync_return_ok)
{
    MOCKER(RaSocketListenStopAsync).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(0));

    SocketHandle socketHandle = nullptr;
    unsigned int port = 100;
    RaSocketListenParam param(socketHandle, port, IpAddress());

    RaSocketListenOneStopAsync(param);
};

TEST(AdapterHccpTest, RaUbAllocTokenIdHanlde_ok)
{
    std::pair<TokenIdHandle, uint32_t> result = RaUbAllocTokenIdHandle(nullptr);
    std::pair<TokenIdHandle, uint32_t> expectResult(0, 0);
    EXPECT_EQ(result, expectResult);
}

TEST(AdapterHccpTest, RaUbFreeTokenIdHandle_exception)
{
    MOCKER(RaCtxTokenIdFree).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_THROW(RaUbFreeTokenIdHandle(0, 0), NetworkApiException);
}

TEST(CommunicatorImplTest, should_fail_when_comm_status_error)
{
    CommunicatorImpl comm;
    comm.status = CommStatus::COMM_ERROR;
    comm.opExecuteConfig.accState = AcceleratorState::CCU_MS;
    CollOpParams param = {};
    param.opType = OpType::ALLREDUCE;
    param.dataType = DataType::INT32;
    auto res = comm.LoadOpbasedCollOp(param, nullptr);
    EXPECT_EQ(res, HcclResult::HCCL_E_INTERNAL);
    GlobalMockObject::verify();
}

TEST(CommunicatorImplTest, should_success_when_comm_LoadOpbasedCollOp_ccu)
{
    CommunicatorImpl fakeComm;
    u32 fakeDevPhyId = 1;
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    char fakeName[65] = "testRtsNotify";
    SETUP_COMM_NOTIFY_MOCKS();

    // 资源初始化
    MOCKER_CPP(&CcuInsPreprocessor::Preprocess).stubs().with().will(ignoreReturnValue());
    MOCKER_CPP(&AicpuInsPreprocessor::Preprocess).stubs().with().will(ignoreReturnValue());

    Buffer* buf = nullptr;
    LocalRmaBuffer* rmaBuf = nullptr;
    MOCKER_CPP(&DataBufManager::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(buf));
    MOCKER_CPP(
        &LocalRmaBufManager::Reg, LocalRmaBuffer
                                      * (LocalRmaBufManager::*)(const string&, BufferType, std::shared_ptr<Buffer>,
                                                                const PortData&, LinkProtocol))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(rmaBuf));
    RtsNotify notify(false);
    RtsNotify notify1(false);
    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetHostWaitNotify).stubs().with().will(returnValue(&notify));
    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetDeviceWaitNotify).stubs().with().will(returnValue(&notify1));
    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetPackedData)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(std::vector<char>{'1', '2'}));
    void* ptr1 = (void*)1;
    MOCKER(HrtStreamCreateWithFlags).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(ptr1));
    MOCKER(HrtGetStreamId).stubs().with(mockcpp::any()).will(returnValue(0));

    fakeComm.cclBuffer = DevBuffer::Create(0x100, 0x100);
    fakeComm.status = CommStatus::COMM_READY;
    fakeComm.opExecuteConfig.accState = AcceleratorState::CCU_MS;
    fakeComm.InitNotifyManager();
    fakeComm.InitSocketManager();
    fakeComm.InitRmaConnManager();
    fakeComm.InitStreamManager();
    fakeComm.InitMemTransportManager();
    fakeComm.InitMirrorTaskManager();
    fakeComm.InitProfilingReporter();
    fakeComm.myRank = 0;
    fakeComm.rankSize = 2;
    fakeComm.id = "testTag";
    fakeComm.streamManager->opbase = make_unique<OpbaseStreamManager>(&fakeComm);
    std::shared_ptr<Buffer> buffer = DevBuffer::Create(0x100, 10);
    fakeComm.dataBufferManager = std::make_unique<DataBufManager>();
    fakeComm.dataBufferManager->Register("testTag", BufferType::SCRATCH, buffer);
    fakeComm.rankGraph = std::make_unique<RankGraph>(0);
    fakeComm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&fakeComm);
    fakeComm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&fakeComm);
    fakeComm.rmaConnectionManager = std::make_unique<RmaConnManager>(fakeComm);
    fakeComm.currentCollOperator = std::make_unique<CollOperator>();
    fakeComm.currentCollOperator->opMode = OpMode::OPBASE;
    fakeComm.currentCollOperator->opType = OpType::DEBUGCASE;
    fakeComm.currentCollOperator->inputMem = DevBuffer::Create(0x100, 10);
    fakeComm.currentCollOperator->outputMem = DevBuffer::Create(0x100, 10);
    fakeComm.queueWaitGroupCntNotifyManager = std::make_unique<QueueWaitGroupCntNotifyManager>();
    fakeComm.queueBcastPostCntNotifyManager = std::make_unique<QueueBcastPostCntNotifyManager>();
    fakeComm.hostDeviceSyncNotifyManager = std::make_unique<HostDeviceSyncNotifyManager>();
    fakeComm.memTransportManager = make_unique<MemTransportManager>(fakeComm);

    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    IpAddress inputAddr(0);
    std::set<std::string> ports = {"0/1"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    shared_ptr<NetInstance::Peer> peer0 = std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    shared_ptr<NetInstance::ConnInterface> connInterface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocols);
    peer0->AddConnInterface(connInterface);
    fakeComm.rankGraph->AddPeer(peer0);
    fakeComm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(fakeComm);
    fakeComm.trace = std::make_unique<Trace>();

    fakeComm.InitCollService();
    fakeComm.CollAlgComponentInit();
    MOCKER_CPP(&CollAlgComponent::ExecAlgSelect)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    // 算法组件初始化
    CollAlgOpReq collAlgOpReq;
    collAlgOpReq.algName = "testAlg";
    collAlgOpReq.resReq.primQueueNum = 1;
    CollAlgComponent collAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1);
    MOCKER_CPP_VIRTUAL(
        collAlgComponent, &CollAlgComponent::Orchestrate,
        HcclResult(CollAlgComponent::*)(
            const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(
        collAlgComponent, &CollAlgComponent::CalcResOffload,
        HcclResult(CollAlgComponent::*)(
            const OpType& opType, const u64& dataSize, const HcclDataType& dataType, const OpExecuteConfig& opConfig,
            CollOffloadOpResReq& resReq))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(collAlgComponent, &CollAlgComponent::GetCollAlgOpReq)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(collAlgOpReq));
    MOCKER_CPP(&Trace::Save).stubs();
    MOCKER_CPP(&CommunicatorImpl::ReportProfInfo).stubs();
    MOCKER_CPP(&CollServiceAiCpuImpl::AllocOpMem).stubs();
    MOCKER_CPP(&Stream::InitDevPhyId).stubs();
    MOCKER_CPP(&CollServiceBase::SaveMirrorDfxOpInfo).stubs();
    MOCKER_CPP(&CollServiceAiCpuImpl::AddPostToUserStream).stubs().with(mockcpp::any());
    MOCKER_CPP(&CollServiceAiCpuImpl::AddWaitToUserStream).stubs().with(mockcpp::any());
    MOCKER_CPP(&CollServiceAiCpuImpl::SetHcclKernelLaunchParam).stubs().with(mockcpp::any(), mockcpp::any());
    CollOpParams param = {};
    param.opType = OpType::ALLREDUCE;
    param.dataType = DataType::INT32;
    auto res = fakeComm.LoadOpbasedCollOp(param, nullptr);
    GlobalMockObject::verify();
}

TEST(CommunicatorImplTest, should_success_when_comm_LoadOpbasedCollOp_aicpu)
{
    CommunicatorImpl fakeComm;
    u32 fakeDevPhyId = 1;
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    char fakeName[65] = "testRtsNotify";
    SETUP_COMM_NOTIFY_MOCKS();

    // 资源初始化
    MOCKER_CPP(&CcuInsPreprocessor::Preprocess).stubs().with().will(ignoreReturnValue());
    MOCKER_CPP(&AicpuInsPreprocessor::Preprocess).stubs().with().will(ignoreReturnValue());

    Buffer* buf = nullptr;
    LocalRmaBuffer* rmaBuf = nullptr;
    MOCKER_CPP(&DataBufManager::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(buf));
    MOCKER_CPP(
        &LocalRmaBufManager::Reg, LocalRmaBuffer
                                      * (LocalRmaBufManager::*)(const string&, BufferType, std::shared_ptr<Buffer>,
                                                                const PortData&, LinkProtocol))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(rmaBuf));
    RtsNotify notify(false);
    RtsNotify notify1(false);
    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetHostWaitNotify).stubs().with().will(returnValue(&notify));
    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetDeviceWaitNotify).stubs().with().will(returnValue(&notify1));
    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetPackedData)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(std::vector<char>{'1', '2'}));
    void* ptr1 = (void*)1;
    MOCKER(HrtStreamCreateWithFlags).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(ptr1));
    MOCKER(HrtGetStreamId).stubs().with(mockcpp::any()).will(returnValue(0));
    fakeComm.rankSize = 2;
    fakeComm.cclBuffer = DevBuffer::Create(0x100, 0x100);
    fakeComm.status = CommStatus::COMM_READY;
    fakeComm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;
    fakeComm.InitNotifyManager();
    fakeComm.InitSocketManager();
    fakeComm.InitRmaConnManager();
    fakeComm.InitStreamManager();
    fakeComm.InitMemTransportManager();
    fakeComm.InitMirrorTaskManager();
    fakeComm.InitProfilingReporter();
    fakeComm.myRank = 0;
    fakeComm.rankSize = 2;
    fakeComm.id = "testTag";
    fakeComm.streamManager->opbase = make_unique<OpbaseStreamManager>(&fakeComm);
    std::shared_ptr<Buffer> buffer = DevBuffer::Create(0x100, 10);
    fakeComm.dataBufferManager = std::make_unique<DataBufManager>();
    fakeComm.dataBufferManager->Register("testTag", BufferType::SCRATCH, buffer);
    fakeComm.rankGraph = std::make_unique<RankGraph>(0);
    fakeComm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&fakeComm);
    fakeComm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&fakeComm);
    fakeComm.rmaConnectionManager = std::make_unique<RmaConnManager>(fakeComm);
    fakeComm.currentCollOperator = std::make_unique<CollOperator>();
    fakeComm.currentCollOperator->opMode = OpMode::OPBASE;
    fakeComm.currentCollOperator->opType = OpType::DEBUGCASE;
    fakeComm.currentCollOperator->inputMem = DevBuffer::Create(0x100, 10);
    fakeComm.currentCollOperator->outputMem = DevBuffer::Create(0x100, 10);
    fakeComm.queueWaitGroupCntNotifyManager = std::make_unique<QueueWaitGroupCntNotifyManager>();
    fakeComm.queueBcastPostCntNotifyManager = std::make_unique<QueueBcastPostCntNotifyManager>();
    fakeComm.hostDeviceSyncNotifyManager = std::make_unique<HostDeviceSyncNotifyManager>();
    fakeComm.memTransportManager = make_unique<MemTransportManager>(fakeComm);

    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    IpAddress inputAddr(0);
    std::set<std::string> ports = {"0/1"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    shared_ptr<NetInstance::Peer> peer0 = std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    shared_ptr<NetInstance::ConnInterface> connInterface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocols);
    peer0->AddConnInterface(connInterface);
    fakeComm.rankGraph->AddPeer(peer0);
    fakeComm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(fakeComm);
    fakeComm.trace = std::make_unique<Trace>();

    fakeComm.InitCollService();
    fakeComm.CollAlgComponentInit();
    MOCKER_CPP(&CollAlgComponent::ExecAlgSelect)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    // 算法组件初始化
    CollAlgOpReq collAlgOpReq;
    collAlgOpReq.algName = "testAlg";
    collAlgOpReq.resReq.primQueueNum = 1;
    CollAlgComponent collAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1);
    MOCKER_CPP_VIRTUAL(
        collAlgComponent, &CollAlgComponent::Orchestrate,
        HcclResult(CollAlgComponent::*)(
            const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(
        collAlgComponent, &CollAlgComponent::CalcResOffload,
        HcclResult(CollAlgComponent::*)(
            const OpType& opType, const u64& dataSize, const HcclDataType& dataType, const OpExecuteConfig& opConfig,
            CollOffloadOpResReq& resReq))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(collAlgComponent, &CollAlgComponent::GetCollAlgOpReq)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(collAlgOpReq));
    MOCKER_CPP(&Trace::Save).stubs();
    MOCKER_CPP(&CommunicatorImpl::ReportProfInfo).stubs();
    MOCKER_CPP(&CollServiceAiCpuImpl::AllocOpMem).stubs();
    MOCKER_CPP(&Stream::InitDevPhyId).stubs();
    MOCKER_CPP(&CollServiceBase::SaveMirrorDfxOpInfo).stubs();
    MOCKER_CPP(&CollServiceAiCpuImpl::AddPostToUserStream).stubs().with(mockcpp::any());
    MOCKER_CPP(&CollServiceAiCpuImpl::AddWaitToUserStream).stubs().with(mockcpp::any());
    MOCKER_CPP(&CollServiceAiCpuImpl::SetHcclKernelLaunchParam).stubs().with(mockcpp::any(), mockcpp::any());
    CollOpParams param = {};
    param.opType = OpType::ALLREDUCE;
    param.dataType = DataType::INT32;
    auto res = fakeComm.LoadOpbasedCollOp(param, nullptr);
    GlobalMockObject::verify();
}

TEST(CommunicatorImplTest, should_fail_when_comm_status_error4)
{
    GlobalMockObject::verify();
    MOCKER_CPP(&CommunicatorImpl::CovertToCurrentCollOperator).stubs().will(throws(InternalException("")));
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(ignoreReturnValue());

    CommunicatorImpl comm;
    CollOpParams param = {};
    std::string opTag = "";
    param.opType = OpType::ALLREDUCE;
    param.dataType = DataType::INT32;
    comm.status = CommStatus::COMM_READY;
    comm.rankSize = 2;
    MOCKER_CPP(&CommunicatorImpl::UpdateProfStat).stubs();
    auto res = comm.LoadOffloadCollOp(opTag, param, nullptr);
    EXPECT_EQ(res, HcclResult::HCCL_E_INTERNAL);
    GlobalMockObject::verify();
}

TEST(CommunicatorImplTest, should_fail_when_comm_status_error5)
{
    GlobalMockObject::verify();
    MOCKER_CPP(&CommunicatorImpl::CovertToCurrentCollOperator).stubs().will(throws(1));
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(ignoreReturnValue());

    CommunicatorImpl comm;
    CollOpParams param = {};
    std::string opTag = "";
    param.opType = OpType::ALLREDUCE;
    param.dataType = DataType::INT32;
    comm.status = CommStatus::COMM_READY;
    comm.rankSize = 2;
    MOCKER_CPP(&CommunicatorImpl::UpdateProfStat).stubs();
    auto res = comm.LoadOffloadCollOp(opTag, param, nullptr);
    EXPECT_EQ(res, HcclResult::HCCL_E_INTERNAL);
    GlobalMockObject::verify();
}

TEST(CommunicatorImplTest, should_fail_when_comm_status_error6)
{
    GlobalMockObject::verify();
    CommunicatorImpl comm;
    comm.status = CommStatus::COMM_ERROR;
    CollOpParams param = {};
    std::string opTag = "";
    param.opType = OpType::ALLREDUCE;
    param.dataType = DataType::INT32;
    MOCKER_CPP(&CommunicatorImpl::UpdateProfStat).stubs();
    auto res = comm.LoadOffloadCollOp(opTag, param, nullptr);
    EXPECT_EQ(res, HcclResult::HCCL_E_INTERNAL);
    GlobalMockObject::verify();
}

TEST(ST_AdapterRtsTest, DevCapabilityT_Init)
{
    DevType devType = DevType::DEV_TYPE_V51_310_P3;

    DevCapability& devCap = DevCapability::GetInstance();
    devCap.Reset();
    EXPECT_THROW(devCap.Init(devType), NotSupportException);
}

void MockRaSocketRecv(int ret, unsigned long long recvSize)
{
    MOCKER(RaSocketRecv)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&recvSize, sizeof(recvSize)))
        .will(returnValue(ret));
}

void MockEnvLinkTimeoutGet(int timeout)
{
    // 设置超时时间
    MOCKER(EnvLinkTimeoutGet).stubs().will(returnValue(timeout));
}

TEST(AdapterHccpTest, St_HrtRaSocketBlockRecv_When_SockClosed_Expect_ThrowException)
{
    FdHandle fakeFdHandle = nullptr;
    void* fakeData = (void*)0x100;
    MockRaSocketRecv(SOCK_ESOCKCLOSED, 0);
    MockEnvLinkTimeoutGet(1);
    EXPECT_THROW(HrtRaSocketBlockRecv(fakeFdHandle, fakeData, 100), NetworkApiException);
}

TEST(AdapterHccpTest, St_HrtRaSocketBlockRecv_When_SockClose_Expect_ThrowException)
{
    FdHandle fakeFdHandle = nullptr;
    void* fakeData = (void*)0x100;
    MockRaSocketRecv(SOCK_CLOSE, 0);
    MockEnvLinkTimeoutGet(1);
    EXPECT_THROW(HrtRaSocketBlockRecv(fakeFdHandle, fakeData, 100), NetworkApiException);
}
