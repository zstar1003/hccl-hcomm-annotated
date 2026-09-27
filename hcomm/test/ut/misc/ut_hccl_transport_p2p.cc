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
#include <stdio.h>

#define private public
#define protected public
#include "dispatcher_pub.h"
#include "transport_p2p_pub.h"
#include "transport_device_ibverbs_pub.h"
#include "dlhns_function.h"
#undef protected
#undef private

#include "hccl/base.h"
#include <hccl/hccl_types.h>

#include "sal.h"

#include "llt_hccl_stub_pub.h"
#include "remote_notify.h"

#include "adapter_rts.h"

using namespace std;
using namespace hccl;

class LinkPcieTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "\033[36m--LinkPcieTest SetUP--\033[0m" << std::endl;
        localNotify.reset(new (std::nothrow) LocalIpcNotify());
        HcclResult ret = localNotify->Init(0, 0);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
        ret = localNotify->Serialize(data);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        remoteNotify.reset(new (std::nothrow) RemoteNotify());

        ret = remoteNotify->Init(data);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        ret = remoteNotify->Open();
        EXPECT_EQ(ret, HCCL_SUCCESS);

        userLocalNotify.resize(8);
        userRemoteNotify.resize(8);
        for (u32 i = 0; i < 8; i++) {
            userLocalNotify[i].reset(new (std::nothrow) LocalIpcNotify());
            HcclResult ret = userLocalNotify[i]->Init(0, 0);
            EXPECT_EQ(ret, HCCL_SUCCESS);

            ret = userLocalNotify[i]->Serialize(data);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            userRemoteNotify[i].reset(new (std::nothrow) RemoteNotify());

            ret = userRemoteNotify[i]->Init(data);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            ret = userRemoteNotify[i]->Open();
            EXPECT_EQ(ret, HCCL_SUCCESS);
        }

        ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, 0, &dispatcherPtr);
        if (ret != HCCL_SUCCESS)
            return;
        if (dispatcherPtr == nullptr)
            return;
        dispatcher = reinterpret_cast<DispatcherPub*>(dispatcherPtr);
    }
    static void TearDownTestCase()
    {
        if (dispatcherPtr != nullptr) {
            s32 ret = HcclDispatcherDestroy(dispatcherPtr);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            dispatcherPtr = nullptr;
            dispatcher = nullptr;
        }
        std::cout << "\033[36m--LinkPcieTest TearDown--\033[0m" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }

    static std::shared_ptr<LocalIpcNotify> localNotify;
    static std::shared_ptr<RemoteNotify> remoteNotify;
    static std::vector<std::shared_ptr<LocalIpcNotify>> userLocalNotify;
    static std::vector<std::shared_ptr<RemoteNotify>> userRemoteNotify;
    static HcclDispatcher dispatcherPtr;
    static DispatcherPub* dispatcher;
};
std::shared_ptr<LocalIpcNotify> LinkPcieTest::localNotify = nullptr;
std::shared_ptr<RemoteNotify> LinkPcieTest::remoteNotify = nullptr;
std::vector<std::shared_ptr<LocalIpcNotify>> LinkPcieTest::userLocalNotify;
std::vector<std::shared_ptr<RemoteNotify>> LinkPcieTest::userRemoteNotify;
HcclDispatcher LinkPcieTest::dispatcherPtr = nullptr;
DispatcherPub* LinkPcieTest::dispatcher = nullptr;

class LinkPcieTmp : public TransportP2p {
public:
    explicit LinkPcieTmp(HcclDispatcher dispatcher, MachinePara& machine_para, std::chrono::milliseconds timeout);

    virtual ~LinkPcieTmp();

    HcclResult get_remote_mem_tmp(UserMemType mem_type, void** remote_ptr)
    {
        return get_remote_mem_tmp(mem_type, remote_ptr);
    }

    HcclResult wait_peer_mem_config_tmp(void** mem_ptr, u8* mem_name, uint64_t size, u64 offset)
    {
        return WaitPeerMemConfig(mem_ptr, mem_name, 0, offset);
    }
};

LinkPcieTmp::LinkPcieTmp(HcclDispatcher dispatcher, MachinePara& machine_para, std::chrono::milliseconds timeout)
    : TransportP2p(reinterpret_cast<DispatcherPub*>(dispatcher), nullptr, machine_para, timeout)
{}

LinkPcieTmp::~LinkPcieTmp() {}

TEST_F(LinkPcieTest, ut_wait_peer_config_timeout)
{
    s32 ret = HCCL_SUCCESS;

    std::string port_name = "mlx5_0";

    std::string collectiveId = "test_collective";

    s32 device_id = 0;
    DevType chipType = DevType::DEV_TYPE_910;

    MachinePara machine_para;
    machine_para.deviceLogicId = device_id;
    machine_para.serverId = "127.0.0.1";
    machine_para.collectiveId = collectiveId;
    machine_para.localUserrank = 0;
    machine_para.remoteUserrank = 1;

    std::shared_ptr<LinkPcieTmp> linktmp = nullptr;
    std::chrono::milliseconds timeout = std::chrono::milliseconds(10);

    const std::string tag = "tag";
    linktmp.reset(new LinkPcieTmp(dispatcher, machine_para, timeout));

    ret = linktmp->wait_peer_mem_config_tmp(NULL, NULL, 0, 0);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(LinkPcieTest, ut_RxAsync)
{
    std::string collectiveId = "test_collective";

    s32 device_id = 0;

    MachinePara machine_para;
    machine_para.deviceLogicId = device_id;
    machine_para.supportDataReceivedAck = true;

    MOCKER_CPP_VIRTUAL(
        *dispatcher, &DispatcherPub::SignalRecord,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u64, s32, bool, u64, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(
        *dispatcher, &DispatcherPub::SignalWait,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u32, s32, bool, u32, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    std::shared_ptr<LinkPcieTmp> linktmp = nullptr;
    std::chrono::milliseconds timeout = std::chrono::milliseconds(10);
    linktmp.reset(new LinkPcieTmp(dispatcher, machine_para, timeout));
    Stream streamObj(StreamType::STREAM_TYPE_OFFLINE);

    linktmp->localSendReadyNotify_ = localNotify;
    linktmp->localSendDoneNotify_ = localNotify;
    linktmp->remoteSendReadyNotify_ = remoteNotify;
    linktmp->remoteSendDoneNotify_ = remoteNotify;

    HcclResult ret = linktmp->RxAsync(UserMemType::INPUT_MEM, 0, nullptr, 0, streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    const char* msg = "hello server from client";
    s32 msg_len = SalStrLen(msg) + 1;
    std::vector<TxMemoryInfo> txMems;
    u32 addr = 123;
    txMems.emplace_back(TxMemoryInfo{UserMemType::INPUT_MEM, 0, &addr, msg_len});

    MOCKER(&HcclD2DMemcpyAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = linktmp->TxAsync(txMems, streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(LinkPcieTest, ut_createNotifyRecordBuff)
{
    MachinePara machine_para;
    machine_para.deviceLogicId = 0;
    std::shared_ptr<LinkPcieTmp> linktmp = nullptr;
    std::chrono::milliseconds timeout = std::chrono::milliseconds(10);
    linktmp.reset(new LinkPcieTmp(dispatcher, machine_para, timeout));
    Stream streamObj(StreamType::STREAM_TYPE_OFFLINE);

    linktmp->localSendReadyNotify_ = localNotify;
    linktmp->localSendDoneNotify_ = localNotify;
    linktmp->remoteSendReadyNotify_ = remoteNotify;
    linktmp->remoteSendDoneNotify_ = remoteNotify;
    linktmp->useSdmaToSignalRecord_ = true;

    u32 notifySize = 4;
    MOCKER(hrtGetNotifySize).stubs().with(outBound(notifySize)).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = linktmp->CreateNotifyValueBuffer();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(linktmp->transportAttr_.signalRecordBuff.length, notifySize);
    EXPECT_NE(linktmp->transportAttr_.signalRecordBuff.address, 0);
    GlobalMockObject::verify();
}

TEST_F(LinkPcieTest, ut_function_for_sendrecv_p2p)
{
    std::string collectiveId = "test_collective";

    s32 device_id = 0;

    MachinePara machine_para;
    machine_para.deviceLogicId = device_id;
    machine_para.supportDataReceivedAck = true;
    machine_para.linkAttribute = 0x1;
    machine_para.notifyNum = 8;

    MOCKER_CPP_VIRTUAL(
        *dispatcher, &DispatcherPub::SignalRecord,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u64, s32, bool, u64, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(
        *dispatcher, &DispatcherPub::SignalWait,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u32, s32, bool, u32, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    DeviceMem input = DeviceMem::alloc(1);
    DeviceMem output = DeviceMem::alloc(1);

    std::shared_ptr<LinkPcieTmp> linktmp = nullptr;
    std::chrono::milliseconds timeout = std::chrono::milliseconds(10);
    linktmp.reset(new LinkPcieTmp(dispatcher, machine_para, timeout));

    linktmp->localSendReadyNotify_ = localNotify;
    linktmp->localSendDoneNotify_ = localNotify;
    linktmp->remoteSendReadyNotify_ = remoteNotify;
    linktmp->remoteSendDoneNotify_ = remoteNotify;

    std::cout << "userLocalNotify_.size()" << linktmp->userLocalNotify_.size() << std::endl;
    std::cout << "userRemoteNotify_.size()" << linktmp->userRemoteNotify_.size() << std::endl;
    std::cout << "userLocalNotify.size()" << userLocalNotify.size() << std::endl;
    std::cout << "userRemoteNotify.size()" << userLocalNotify.size() << std::endl;
    for (u32 i = 0; i < 8; i++) {
        linktmp->userLocalNotify_[i] = userLocalNotify[i];
        linktmp->userRemoteNotify_[i] = userRemoteNotify[i];
    }

    Stream streamObj(StreamType::STREAM_TYPE_OFFLINE);
    HcclResult ret = linktmp->TxPrepare(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->RxPrepare(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->TxData(UserMemType::OUTPUT_MEM, 0, input.ptr(), 0, streamObj);

    ret = linktmp->RxData(UserMemType::OUTPUT_MEM, 0, output.ptr(), 0, streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->TxDone(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->RxDone(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = linktmp->PostReady(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->WaitReady(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = linktmp->Post(1, streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->Wait(1, streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<HcclSignalInfo> localNotify;
    std::vector<HcclSignalInfo> remoteNotifyAddrKey;
    linktmp->GetLocalNotify(localNotify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    linktmp->GetRemoteNotify(remoteNotifyAddrKey);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* remoteAddr = nullptr;
    linktmp->GetRemoteMem(hccl::UserMemType::OUTPUT_MEM, &remoteAddr);
    struct Transport::Buffer remoteBuf(remoteAddr, 0);
    struct Transport::Buffer localBuf(output.ptr(), 0);
    ret = linktmp->ReadSync(localBuf, remoteBuf, streamObj);
    EXPECT_NE(ret, HCCL_SUCCESS);
    ret = linktmp->ReadReduceSync(
        localBuf, remoteBuf, HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, streamObj);
    EXPECT_NE(ret, HCCL_SUCCESS);
    ret = linktmp->PostFin(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->WaitFin(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = linktmp->PostReady(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->PostFin(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->WaitReady(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->WaitFin(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = linktmp->PostFinAck(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->WaitFinAck(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(LinkPcieTest, ut_function_for_device)
{
    std::string collectiveId = "test_collective";

    s32 device_id = 0;

    MachinePara machine_para;
    machine_para.deviceLogicId = device_id;
    machine_para.supportDataReceivedAck = true;
    machine_para.linkAttribute = 0x1;

    TransportDeviceIbverbsData transDevIbverbsData;
    HcclSignalInfo notifyInfo;
    notifyInfo.addr = 100;
    notifyInfo.devId = 1;
    notifyInfo.rankId = 2;
    notifyInfo.resId = 3;
    notifyInfo.tsId = 4;
    transDevIbverbsData.inputBufferPtr = nullptr;
    transDevIbverbsData.outputBufferPtr = nullptr;
    transDevIbverbsData.localInputMem.size = 8192;
    transDevIbverbsData.localInputMem.addr = 0x00;
    transDevIbverbsData.localInputMem.key = 10;
    transDevIbverbsData.localOutputMem.size = 8192;
    transDevIbverbsData.localOutputMem.addr = 0xF000;
    transDevIbverbsData.localOutputMem.key = 15;
    transDevIbverbsData.ackNotify = std::make_shared<LocalIpcNotify>();
    transDevIbverbsData.ackNotify->Init(notifyInfo, NotifyLoadType::DEVICE_NOTIFY);
    transDevIbverbsData.dataAckNotify = std::make_shared<LocalIpcNotify>();
    transDevIbverbsData.dataAckNotify->Init(notifyInfo, NotifyLoadType::DEVICE_NOTIFY);
    transDevIbverbsData.dataNotify = std::make_shared<LocalIpcNotify>();
    transDevIbverbsData.dataNotify->Init(notifyInfo, NotifyLoadType::DEVICE_NOTIFY);
    transDevIbverbsData.localNotifyValueAddr = 0x1000;
    transDevIbverbsData.remoteAckNotifyDetails.addr = 0x2000;
    transDevIbverbsData.remoteDataNotifyDetails.addr = 0x3000;
    transDevIbverbsData.remoteDataAckNotifyDetails.addr = 0x4000;
    transDevIbverbsData.remoteAckNotifyDetails.key = 20;
    transDevIbverbsData.remoteDataNotifyDetails.key = 20;
    transDevIbverbsData.remoteDataAckNotifyDetails.key = 20;
    transDevIbverbsData.notifyValueKey = 10;
    transDevIbverbsData.qpInfo.resize(1);
    transDevIbverbsData.qpInfo[0].qpPtr = 0x5000;
    transDevIbverbsData.qpInfo[0].sqIndex = 1;
    transDevIbverbsData.qpInfo[0].dbIndex = 2;
    transDevIbverbsData.remoteInputKey = 3;
    transDevIbverbsData.remoteOutputKey = 4;
    transDevIbverbsData.qpsPerConnection = 1;
    transDevIbverbsData.notifySize = 0;
    transDevIbverbsData.userRemoteNotifyDetails.resize(1);
    transDevIbverbsData.userLocalNotify.resize(1);

    MOCKER_CPP_VIRTUAL(
        *dispatcher, &DispatcherPub::SignalRecord,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u64, s32, bool, u64, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(
        *dispatcher, &DispatcherPub::SignalWait,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u32, s32, bool, u32, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(
        *dispatcher, &DispatcherPub::RdmaSend, HcclResult(DispatcherPub::*)(u32, u64, hccl::Stream&, RdmaTaskInfo&))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&DlHnsFunction::DlHnsFunctionRoceInit).stubs().will(returnValue(HCCL_SUCCESS));
    unsigned int temp = 1;
    MOCKER_CPP(&TransportDeviceIbverbs::TxSendWrlistExt)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&temp))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRDMADBSend).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    DeviceMem input = DeviceMem::alloc(1);
    DeviceMem output = DeviceMem::alloc(1);

    std::shared_ptr<TransportDeviceIbverbs> linktmp = nullptr;
    std::chrono::milliseconds timeout = std::chrono::milliseconds(10);
    linktmp.reset(new TransportDeviceIbverbs(dispatcher, nullptr, machine_para, timeout, transDevIbverbsData));
    linktmp->Init();
    linktmp->localSendReadyNotify_ = localNotify;
    linktmp->localSendDoneNotify_ = localNotify;
    linktmp->remoteSendReadyNotify_ = remoteNotify;
    linktmp->remoteSendDoneNotify_ = remoteNotify;

    Stream streamObj(StreamType::STREAM_TYPE_OFFLINE);
    HcclResult ret = linktmp->TxPrepare(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->RxPrepare(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->TxData(UserMemType::OUTPUT_MEM, 0, input.ptr(), 0, streamObj);

    ret = linktmp->RxData(UserMemType::OUTPUT_MEM, 0, output.ptr(), 0, streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->TxDone(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->RxDone(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->TxAck(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->RxAck(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->PostReady(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->WaitReady(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* remoteAddr = nullptr;
    ret = linktmp->GetRemoteMem(hccl::UserMemType::OUTPUT_MEM, &remoteAddr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    struct Transport::Buffer remoteBuf(remoteAddr, 0);
    struct Transport::Buffer localBuf((void*)(transDevIbverbsData.localInputMem.addr), 0);
    ret = linktmp->WriteAsync(remoteBuf, localBuf, streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->PostFin(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->WaitFin(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->TxDataSignal(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->RxDataSignal(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->PostFinAck(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->WaitFinAck(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = linktmp->DataReceivedAck(streamObj);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    s32 host_mem_size = 256;
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    DeviceMem inputMem = DeviceMem::alloc(host_mem_size);
    DeviceMem outputMem = DeviceMem::alloc(host_mem_size);
    sal_memset(inputMem.ptr(), sizeof(host_mem_size), 0, sizeof(host_mem_size));
    sal_memset(outputMem.ptr(), sizeof(host_mem_size), 0, sizeof(host_mem_size));
    HostMem host_mem = HostMem::alloc(host_mem_size);
    sal_memset(host_mem.ptr(), host_mem_size, 0, host_mem_size);

    const char* msg = "hello client from server";
    s32 msg_len = SalStrLen(msg) + 1;
    DeviceMem tx_buf = inputMem.range(1, msg_len);
    ret = sal_memcpy(host_mem.ptr(), msg_len, msg, msg_len);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = dispatcher->MemcpyAsync(host_mem, tx_buf, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::vector<TxMemoryInfo> txMems;
    txMems.emplace_back(TxMemoryInfo{UserMemType::INPUT_MEM, 0, tx_buf.ptr(), msg_len});
    HcclDataType datatype = HcclDataType::HCCL_DATA_TYPE_INT32;
    HcclReduceOp redOp = HcclReduceOp::HCCL_REDUCE_SUM;

    MOCKER_CPP(&TransportDeviceIbverbs::TxPayLoad).stubs().will(returnValue(HCCL_SUCCESS));

    ret = linktmp->TxWithReduce(txMems, datatype, redOp, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 dstOffset = 1000;
    void* src = &dstOffset;
    ret = linktmp->TxAsync(UserMemType::INPUT_MEM, dstOffset, nullptr, dstOffset, stream);
    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = linktmp->TxAsync(txMems, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = linktmp->TxWithReduce(UserMemType::INPUT_MEM, dstOffset, src, dstOffset, datatype, redOp, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<WrInformation> wrInfoVec;
    bool useOneDoorbell;

    ret = linktmp->TxSendDataAndNotifyWithSingleQP(wrInfoVec, stream, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = linktmp->TxSendDataAndNotify(wrInfoVec, stream, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = linktmp->RdmaSendAsync(wrInfoVec, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
    u64 qpIndex = 1;
    struct WrInfo wr_info_ptr;
    wr_info_ptr.sendFlags = 1;
    wr_info_ptr.rkey = 12345;
    wr_info_ptr.op = RA_WR_RDMA_READ;
    wr_info_ptr.immData = 0;
    wr_info_ptr.wrId = 67890;
    wr_info_ptr.dstAddr = 0xabcdef;
    u32 sendNum = 0;
    struct SendWrRsp my_send_wr_rsp;
    my_send_wr_rsp.wqeTmp.sqIndex = 1;
    my_send_wr_rsp.wqeTmp.wqeIndex = 2;
    my_send_wr_rsp.db.dbIndex = 3;
    my_send_wr_rsp.db.dbInfo = 0x12345678;
    unsigned int* completeNum = &sendNum;

    struct AiQpInfo qpInfo;
    qpInfo.aiQpAddr = 0x12345678;
    qpInfo.sqIndex = 0;
    qpInfo.dbIndex = 1;
    linktmp->combineAiQpInfos_.resize(1);
    linktmp->combineAiQpInfos_[0].aiQpInfo = qpInfo;
    WrInformation wrInfo;
    wrInfo.wrData = wr_info_ptr;
    ret = linktmp->TxSendWrlistExt(&wrInfo, sendNum, &my_send_wr_rsp, completeNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(LinkPcieTest, ut_SpecifyLink)
{
    MachinePara machine_para;
    machine_para.deviceLogicId = 0;
    machine_para.specifyLink = LinkTypeInServer::HCCS_SW_TYPE;
    std::shared_ptr<LinkPcieTmp> linktmp = nullptr;
    std::chrono::milliseconds timeout = std::chrono::milliseconds(10);
    linktmp.reset(new LinkPcieTmp(dispatcher, machine_para, timeout));

    // 设置成功
    linktmp->recvPid_ = INVALID_INT;
    LinkTypeInServer linkType = LinkTypeInServer::SIO_TYPE;
    MOCKER(hrtGetPairDeviceLinkType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(linkType))
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = linktmp->SetLinkType();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    s32 pid = 0;
    SalGetBareTgid(&pid);
    linktmp->recvPid_ = pid;
    ret = linktmp->SetLinkType();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    linkType = LinkTypeInServer::HCCS_TYPE;
    MOCKER(hrtGetPairDeviceLinkType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(linkType))
        .will(returnValue(HCCL_SUCCESS));
    ret = linktmp->SetLinkType();
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    u8 name = 1;
    ret = hrtIpcSetMemoryAttr(&name, ACL_RT_IPC_MEM_ATTR_ACCESS_LINK, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(aclrtIpcMemSetAttr).stubs().with(mockcpp::any()).will(returnValue(1));
    ret = hrtIpcSetMemoryAttr(&name, ACL_RT_IPC_MEM_ATTR_ACCESS_LINK, 1);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}
