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

#include "hccl/base.h"
#include <hccl/hccl_types.h>
#include "llt_hccl_stub_pub.h"

#define private public
#define protected public
#include "hccl_alg.h"
#include "hccl_impl.h"
#include "hccl_communicator.h"
#include "alg_template_base_pub.h"
#include "coll_all_reduce_reduce_plus_bcast_executor.h"
#include "coll_all_reduce_for_310p_doubling_direct_executor.h"
#include "common/externalinput.h"
#include "dlra_function.h"
#include "adapter_hal.h"
#include "adapter_rts.h"
#include "dltdt_function.h"
#include "dlhal_function.h"
#undef private
#undef protected
using namespace std;
using namespace hccl;

class HcclCommAicpuTest_UT : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        s32 ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, 0, &dispatcherPtr);
        if (ret != HCCL_SUCCESS)
            return;
        if (dispatcherPtr == nullptr)
            return;
        dispatcher = reinterpret_cast<DispatcherPub*>(dispatcherPtr);
        DlRaFunction::GetInstance().DlRaFunctionInit();
        std::cout << "\033[36m--HcclCommAicpuTest_UT SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase()
    {
        if (dispatcherPtr != nullptr) {
            s32 ret = HcclDispatcherDestroy(dispatcherPtr);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            dispatcherPtr = nullptr;
            dispatcher = nullptr;
        }
        std::cout << "\033[36m--HcclCommAicpuTest_UT TearDown--\033[0m" << std::endl;
    }
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
    static HcclDispatcher dispatcherPtr;
    static DispatcherPub* dispatcher;
};
HcclDispatcher HcclCommAicpuTest_UT::dispatcherPtr = nullptr;
DispatcherPub* HcclCommAicpuTest_UT::dispatcher = nullptr;

static u64 notifyid_ = 0;
static u32 key_ = 0;
static u64 addr_ = 0;
static u64 size_ = 8;
static s64 chipId_ = 0;
static u32 sqIndex_ = 0;
static u32 dbIndex_ = 0;

class StubTransportBase : public TransportBase {
public:
    StubTransportBase(
        DispatcherPub* dispatcher, MachinePara& machinePara, std::chrono::milliseconds timeout, void* remoteInPtr,
        void* remoteOutPtr, std::vector<void*> remoteMemPtrVector, u64 inSize, u64 outSize, u32 inKey, u32 outKey)
        : TransportBase(dispatcher, nullptr, machinePara, timeout),
          remoteInPtr_(remoteInPtr),
          remoteOutPtr_(remoteOutPtr),
          remoteIpcMemPtrVector_(remoteMemPtrVector),
          remoteInSize_(inSize),
          remoteOutSize_(outSize),
          remoteInKey_(inKey),
          remoteOutKey_(outKey)
    {}

    HcclResult GetRemoteMem(UserMemType memType, void** remotePtr)
    {
        if (memType == UserMemType::INPUT_MEM) {
            *remotePtr = remoteInPtr_;
        } else {
            *remotePtr = remoteOutPtr_;
        }
        return HCCL_SUCCESS;
    }

    HcclResult GetRemoteMem(std::vector<void*>* remotePtr)
    {
        *remotePtr = remoteIpcMemPtrVector_;
        return HCCL_SUCCESS;
    }

    HcclResult GetRemoteMemSize(UserMemType memType, u64& size)
    {
        if (memType == UserMemType::INPUT_MEM) {
            size = remoteInSize_;
        } else {
            size = remoteOutSize_;
        }
        return HCCL_SUCCESS;
    }
    HcclResult GetRemoteMemKey(UserMemType memType, u32* key)
    {
        if (memType == UserMemType::INPUT_MEM) {
            *key = remoteInKey_;
        } else {
            *key = remoteOutKey_;
        }
        return HCCL_SUCCESS;
    }

private:
    void* remoteInPtr_;
    void* remoteOutPtr_;
    u64 remoteInSize_;
    u64 remoteOutSize_;
    u32 remoteInKey_;
    u32 remoteOutKey_;
    std::vector<void*> remoteIpcMemPtrVector_;
};
void GetNotifyInfo(HcclSignalInfo& notifyInfo)
{
    notifyInfo.addr = notifyid_;
    notifyInfo.devId = notifyid_;
    notifyInfo.resId = notifyid_;
    notifyInfo.tsId = notifyid_;
    notifyid_++;
}
void GetMemDetails(UserMemType memType, MemDetails& memDetails)
{
    memDetails.addr = addr_;
    memDetails.key = key_;
    memDetails.size = size_;
    addr_ += size_;
    key_++;
    size_ += 8;
}
void GetAddrKey(std::vector<AddrKey>& addrKey)
{
    AddrKey tempAddrKey;
    tempAddrKey.addr = addr_;
    tempAddrKey.key = key_;
    addrKey.push_back(tempAddrKey);
    addr_ += size_;
    key_++;
}
void GetNotify(std::vector<HcclSignalInfo>& notify)
{
    HcclSignalInfo notifyInfo;
    GetNotifyInfo(notifyInfo);
    notify.push_back(notifyInfo);
}
void GetQpInfo(std::vector<HcclQpInfoV2>& AiQpInfo)
{
    AiQpInfo.resize(1);
    AiQpInfo[0].qpPtr = addr_;
    AiQpInfo[0].sqIndex = sqIndex_;
    AiQpInfo[0].dbIndex = dbIndex_;
    addr_ += size_;
    sqIndex_++;
    dbIndex_++;
}
void GetchipId(s64& chipId)
{
    chipId = chipId_;
    chipId_++;
}

HcclResult GetTxAckDevNotifyInfo(TransportBase* This, HcclSignalInfo& notifyInfo)
{
    GetNotifyInfo(notifyInfo);
    return HCCL_SUCCESS;
}
HcclResult GetTxDataSigleDevNotifyInfo(TransportBase* This, HcclSignalInfo& notifyInfo)
{
    GetNotifyInfo(notifyInfo);
    return HCCL_SUCCESS;
}
HcclResult GetRxAckDevNotifyInfo(TransportBase* This, HcclSignalInfo& notifyInfo)
{
    GetNotifyInfo(notifyInfo);
    return HCCL_SUCCESS;
}
HcclResult GetRxDataSigleDevNotifyInfo(TransportBase* This, HcclSignalInfo& notifyInfo)
{
    GetNotifyInfo(notifyInfo);
    return HCCL_SUCCESS;
}
HcclResult GetLocalMemDetails(TransportBase* This, UserMemType memType, MemDetails& memDetails)
{
    GetMemDetails(memType, memDetails);
    return HCCL_SUCCESS;
}
HcclResult GetRemoteRdmaNotifyAddrKey(TransportBase* This, std::vector<AddrKey>& rdmaNotifyAddr)
{
    GetAddrKey(rdmaNotifyAddr);
    GetAddrKey(rdmaNotifyAddr);
    GetAddrKey(rdmaNotifyAddr);
    return HCCL_SUCCESS;
}
HcclResult GetLocalRdmaNotify(TransportBase* This, std::vector<HcclSignalInfo>& rdmaNotify)
{
    GetNotify(rdmaNotify);
    GetNotify(rdmaNotify);
    GetNotify(rdmaNotify);
    return HCCL_SUCCESS;
}
HcclResult GetLocalNotifyValueAddrKey(TransportBase* This, std::vector<AddrKey>& notifyValue)
{
    GetAddrKey(notifyValue);
    return HCCL_SUCCESS;
}

HcclResult GetLocalNotify(TransportBase* This, std::vector<HcclSignalInfo>& localNotify) { return HCCL_SUCCESS; }

HcclResult GetRemoteNotify(TransportBase* This, std::vector<HcclSignalInfo>& localNotify) { return HCCL_SUCCESS; }

HcclResult GetAiQpInfo(TransportBase* This, std::vector<HcclQpInfoV2>& AiQpInfo)
{
    GetQpInfo(AiQpInfo);
    return HCCL_SUCCESS;
}
HcclResult GetChipId(TransportBase* This, s64& chipId)
{
    GetchipId(chipId);
    return HCCL_SUCCESS;
}

static void TestConstructParam(HcclCommParams& params, RankTable_t& rankTable)
{
    string commId = "comm ";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = 4;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.commWorkMode = WorkMode::HCCL_MODE_NORMAL;
    params.deviceType = DevType::DEV_TYPE_910;

    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(4);

    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.0.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";

    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 1;
    HcclIpAddress ipAddr2(1711319232);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
    rankVec[1].serverIdx = 0;
    rankVec[1].serverId = "192.168.0.101";

    rankVec[2].rankId = 2;
    rankVec[2].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr3(1694542017);
    rankVec[2].deviceInfo.deviceIp.push_back(ipAddr3); // 101.0.168.192
    rankVec[2].serverIdx = 1;
    rankVec[2].serverId = "192.168.0.102";

    rankVec[3].rankId = 3;
    rankVec[3].deviceInfo.devicePhyId = 1;
    HcclIpAddress ipAddr4(1711319233);
    rankVec[3].deviceInfo.deviceIp.push_back(ipAddr4); // 101.0.168.192
    rankVec[3].serverIdx = 1;
    rankVec[3].serverId = "192.168.0.102";

    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = 4;
    rankTable.serverNum = 2;
}
#define STREAM_NUM 16
#define NOTIFY_NUM 16
#define MAX_LOOP_NUM 16
#define DEVICE_MEM_SIZE 128
#define SINGAL_SUB_COMM_NUM 2
#define LEVEL_SUB_COMM_NUM 2
#define OP_COM_NUM 1
static u32 rankId = 0;
static void TestConstructResponse(AlgResourceResponse& algResource, DispatcherPub* dispatcher)
{
    rankId = 0;
    u32 inkey = 0;
    u32 outkey = 0;
    std::chrono::milliseconds timeout;
    MachinePara machinePara;
    LevelNSubCommTransport levelNSubCommTransport;
    DeviceMem output = DeviceMem::alloc(DEVICE_MEM_SIZE);
    DeviceMem input = DeviceMem::alloc(DEVICE_MEM_SIZE);
    DeviceMem expMem = DeviceMem::alloc(DEVICE_MEM_SIZE);
    std::vector<void*> memPtrVec = {expMem.ptr()};
    std::vector<std::shared_ptr<Transport>> link;
    algResource.opTransportResponse.resize(OP_COM_NUM);
    for (int opIdx = 0; opIdx < OP_COM_NUM; opIdx++) {
        levelNSubCommTransport.resize(LEVEL_SUB_COMM_NUM);
        for (int levelIdx = 0; levelIdx < LEVEL_SUB_COMM_NUM; levelIdx++) {
            SingleSubCommTransport singleSubCommTransport;
            singleSubCommTransport.links.resize(SINGAL_SUB_COMM_NUM);
            singleSubCommTransport.transportRequests.resize(SINGAL_SUB_COMM_NUM);
            for (int i = 0; i < SINGAL_SUB_COMM_NUM; i++) {
                singleSubCommTransport.transportRequests[i].isValid = true;
                singleSubCommTransport.transportRequests[i].remoteUserRank = rankId;
                singleSubCommTransport.transportRequests[i].inputMemType = TransportMemType::SCRATCH;
                singleSubCommTransport.transportRequests[i].isUsedRdma = i % 2;
                singleSubCommTransport.transportRequests[i].outputMemType = TransportMemType::CCL_OUTPUT;
                singleSubCommTransport.links[i].reset(new (std::nothrow) Transport(new (std::nothrow) StubTransportBase(
                    dispatcher, machinePara, timeout, input.ptr(), output.ptr(), memPtrVec, input.size(), output.size(),
                    inkey, outkey)));
                singleSubCommTransport.links[i]->Init();
                rankId++;
            }
            levelNSubCommTransport[levelIdx] = singleSubCommTransport;
        }
        algResource.opTransportResponse[opIdx] = (levelNSubCommTransport);
    }
}
HcclResult TestConstructAlgResourceResponse(AlgResourceResponse& resourceResponse, DispatcherPub* dispatcher)
{
    HcclResult ret = HCCL_SUCCESS;

    for (size_t i = 0; i < STREAM_NUM; i++) {
        resourceResponse.slaveDevStreams.emplace_back(Stream(StreamType::STREAM_TYPE_DEVICE));
    }

    for (size_t i = 0; i < NOTIFY_NUM; i++) {
        std::shared_ptr<LocalNotify> localNotify;
        localNotify = std::make_shared<LocalNotify>();
        ret = localNotify->Init(NotifyLoadType::DEVICE_NOTIFY);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("localNotify init fail");
            return ret;
        }
        if (i % 2 == 0) {
            resourceResponse.notifiesDevMain.push_back(localNotify);
        } else {
            resourceResponse.notifiesDevAux.push_back(localNotify);
        }
    }
    TestConstructResponse(resourceResponse, dispatcher);
    return HCCL_SUCCESS;
}
void verifyList(u64 head)
{
    ListCommon* curPtr = reinterpret_cast<ListCommon*>(head);
    int idx = 0;
    while (curPtr->nextHost != head) {
        ListCommon* nextPtr = reinterpret_cast<ListCommon*>(curPtr->nextHost);
        HCCL_ERROR(
            "head addr[%p], curPtr[%p], curPtr nextHost[%p], nextPtr preHost[%p], nextPtr nextHost[%p], nextPtr "
            "preDevice[%p], nextPtr nextDevice[%p]",
            head, curPtr, curPtr->nextHost, nextPtr->preHost, nextPtr->nextHost, nextPtr->preDevice,
            nextPtr->nextDevice);
        EXPECT_EQ(reinterpret_cast<u64>(curPtr), nextPtr->preHost);
        curPtr = nextPtr;
        idx++;
        if (idx > MAX_LOOP_NUM) {
            break;
        }
    };
}

TEST_F(HcclCommAicpuTest_UT, BuildOpRetryParam)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    implBase->retryEnable_ = true;
    implBase->deviceType_ = DevType::DEV_TYPE_910_93;
    implBase->opRetryStreamPtr_ = std::make_shared<HcclOpStreamRes>();
    MOCKER(hrtHalMemCtl).stubs().will(returnValue(0));

    AlgResourceResponse algResource;
    std::string newTag;
    ASSERT_EQ(implBase->InitHDCommunicate(), HCCL_SUCCESS);
    ASSERT_EQ(implBase->BuildOpRetryParam(algResource, newTag), HCCL_SUCCESS);
    EXPECT_EQ(implBase->retryEnable_, true);
    EXPECT_EQ(implBase->opResPara_.config.retryHoldTime, HCCL_RETRY_HOLD_TIME_DEFAULT);
    EXPECT_EQ(implBase->opResPara_.config.retryIntervalTime, HCCL_RETRY_INTERVAL_DEFAULT);
}

TEST_F(HcclCommAicpuTest_UT, hcclImpl_BuildOpResParam_ok)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910;
    params.identifier = "tag";
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;

    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    DeviceMem scratchMem = DeviceMem::alloc(8192);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 1024;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 1024;
    opParam.DataDes.count = 4096;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;

    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    resourceResponse.scratchMem = scratchMem;
    string algName = "allreduce_mesh";
    string newTag = "allreduce_my_hcom_id_allreduce_mesh";
    ret = TestConstructAlgResourceResponse(resourceResponse, dispatcher);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->BuildOpLocalResParam(resourceResponse, newTag);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(implBase->opResPara_.localRes.signalNum, NOTIFY_NUM);
    EXPECT_EQ(implBase->opResPara_.localRes.streamNum, STREAM_NUM);
    HCCL_ERROR(
        "head addr[%p], nextHost[%p], preHost[%p]", &implBase->opResPara_.localRes.nextTagRes,
        implBase->opResPara_.localRes.nextTagRes.nextHost, implBase->opResPara_.localRes.nextTagRes.preHost);
    verifyList(reinterpret_cast<u64>(&implBase->opResPara_.localRes.nextTagRes));

    MOCKER_CPP(&TransportBase::GetTxAckDevNotifyInfo).stubs().will(invoke(GetTxAckDevNotifyInfo));
    MOCKER_CPP(&TransportBase::GetTxDataSigleDevNotifyInfo).stubs().will(invoke(GetTxDataSigleDevNotifyInfo));
    MOCKER_CPP(&TransportBase::GetRxAckDevNotifyInfo).stubs().will(invoke(GetRxAckDevNotifyInfo));
    MOCKER_CPP(&TransportBase::GetRxDataSigleDevNotifyInfo).stubs().will(invoke(GetRxDataSigleDevNotifyInfo));
    const std::unique_ptr<NotifyPool> notifyPool;
    std::chrono::milliseconds timeout;
    MachinePara machinePara;
    hccl::TransportBase transportBase(dispatcher, notifyPool, machinePara, timeout);
    MOCKER_CPP_VIRTUAL(transportBase, &TransportBase::GetLocalMemDetails).stubs().will(invoke(GetLocalMemDetails));
    MOCKER_CPP_VIRTUAL(transportBase, &TransportBase::GetRemoteRdmaNotifyAddrKey)
        .stubs()
        .will(invoke(GetRemoteRdmaNotifyAddrKey));
    MOCKER_CPP_VIRTUAL(transportBase, &TransportBase::GetLocalRdmaNotify).stubs().will(invoke(GetLocalRdmaNotify));
    MOCKER_CPP_VIRTUAL(transportBase, &TransportBase::GetLocalNotifyValueAddrKey)
        .stubs()
        .will(invoke(GetLocalNotifyValueAddrKey));
    MOCKER_CPP_VIRTUAL(transportBase, &TransportBase::GetLocalNotify).stubs().will(invoke(GetLocalNotify));
    MOCKER_CPP_VIRTUAL(transportBase, &TransportBase::GetRemoteNotify).stubs().will(invoke(GetRemoteNotify));
    MOCKER_CPP_VIRTUAL(transportBase, &TransportBase::GetAiQpInfo).stubs().will(invoke(GetAiQpInfo));
    MOCKER_CPP(&TransportBase::GetChipId).stubs().will(invoke(GetChipId));

    // delete executor;
    ret = implBase->BuildOpRemoteResParam(resourceResponse, newTag, HcclCMDType::HCCL_CMD_GATHER);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->BuildOpRemoteResParam(resourceResponse, newTag, HcclCMDType::HCCL_CMD_GATHER);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    AlgResourceResponse resourceResponseRefresh;
    ret = TestConstructAlgResourceResponse(resourceResponseRefresh, dispatcher);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    string newTagRefresh = "allreduce_my_hcom_id_allreduce_mesh_refresh";
    ret = implBase->BuildOpRemoteResParam(resourceResponse, newTagRefresh, HcclCMDType::HCCL_CMD_GATHER);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    AlgResourceResponse resourceResponseRefresh_next;
    ret = TestConstructAlgResourceResponse(resourceResponseRefresh_next, dispatcher);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    string newTagRefresh_next = "allreduce_my_hcom_id_allreduce_mesh_refresh_next";
    ret = implBase->BuildOpRemoteResParam(
        resourceResponseRefresh_next, newTagRefresh_next, HcclCMDType::HCCL_CMD_GATHER);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER_CPP(&HcclCommunicator::IsEnableBackupLink).stubs().will(returnValue(true));
    ret = implBase->BuildOpRemoteResParam(resourceResponse, newTag, HcclCMDType::HCCL_CMD_GATHER);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HCCL_ERROR("*************** Remote ******************");
    for (u64 i = 0; i < AICPU_MAX_RANK_NUM; i++) {
        if (implBase->opResPara_.remoteRes[i].nextHostPtr != 0) {
            HCCL_ERROR("remoteRes i[%lu] nextHostPtr[%lu]", i, implBase->opResPara_.remoteRes[i].nextHostPtr);
            HcclRankRelationResV2* remotePtr
                = reinterpret_cast<HcclRankRelationResV2*>(implBase->opResPara_.remoteRes[i].nextHostPtr);
            verifyList(reinterpret_cast<u64>(&remotePtr->nextTagRes));
        }
    }

    ret = implBase->BuildOpTopoResParam(algName, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(implBase->opResPara_.topoInfo.nicNum, 2);
    EXPECT_EQ(implBase->opResPara_.topoInfo.bridgeRankNum, 2);
    ret = implBase->CopyHostOpResToDeviceParam(newTag);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcclCommAicpuTest_UT, hcclImpl_BuildOpRemoteLinkP2pResParam_ok)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910;
    params.identifier = "tag";
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    LINK link;
    MachinePara machinePara;
    u32 inkey = 0;
    u32 outkey = 0;
    std::chrono::milliseconds timeout;
    DeviceMem output = DeviceMem::alloc(DEVICE_MEM_SIZE);
    DeviceMem input = DeviceMem::alloc(DEVICE_MEM_SIZE);
    DeviceMem expMem = DeviceMem::alloc(DEVICE_MEM_SIZE);
    std::vector<void*> memPtrVec = {expMem.ptr()};
    TransportBase* tpBase = new (std::nothrow) StubTransportBase(
        dispatcher, machinePara, timeout, input.ptr(), output.ptr(), memPtrVec, input.size(), output.size(), inkey,
        outkey);
    Transport* tp = new (std::nothrow) Transport(tpBase);
    link.reset(tp);
    link->Init();

    HcclSignalInfo locIpcSignal{1, 2, 3, 4, 5, 6};
    HcclSignalInfo rmtIpcSignal{7, 8, 9, 10, 11, 12};
    std::vector<HcclSignalInfo> locIpcSignals;
    locIpcSignals.emplace_back(locIpcSignal);
    std::vector<HcclSignalInfo> rmtIpcSignals;
    rmtIpcSignals.emplace_back(rmtIpcSignal);
    MOCKER_CPP_VIRTUAL(*tpBase, &TransportBase::GetLocalNotify)
        .stubs()
        .with(outBound(locIpcSignals))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(*tpBase, &TransportBase::GetRemoteNotify)
        .stubs()
        .with(outBound(rmtIpcSignals))
        .will(returnValue(HCCL_SUCCESS));
    HccltagRemoteResV2 tagRemoteResPtr;
    tagRemoteResPtr.linkP2p.localIpcSignal[0].resId = INVALID_U64;
    tagRemoteResPtr.linkP2pSio.localIpcSignal[0].resId = INVALID_U64;
    HccltagRemoteResV3 tagRemoteRes;
    tagRemoteRes.tagRemoteResPtr = &tagRemoteResPtr;
    ret = implBase->BuildOpRemoteLinkP2pResParam(link, tagRemoteRes, TransportLinkType::RESERVED);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(tagRemoteResPtr.linkP2p.remoteMem[INPUT].addr, reinterpret_cast<u64>(input.ptr()));
    EXPECT_EQ(tagRemoteResPtr.linkP2p.remoteMem[OUTPUT].addr, reinterpret_cast<u64>(output.ptr()));
    EXPECT_EQ(tagRemoteRes.p2pNotifyNum, 1);
    EXPECT_EQ(tagRemoteResPtr.linkP2p.localIpcSignal[0].resId, 1U);
    EXPECT_EQ(tagRemoteResPtr.linkP2p.remoteIpcSignal[0].resId, 7U);
    EXPECT_EQ(tagRemoteResPtr.linkP2pSio.localIpcSignal[0].resId, INVALID_U64);

    HccltagRemoteResV2 tagRemoteResPtr2;
    tagRemoteResPtr2.linkP2p.localIpcSignal[0].resId = INVALID_U64;
    tagRemoteResPtr2.linkP2pSio.localIpcSignal[0].resId = INVALID_U64;
    HccltagRemoteResV3 tagRemoteRes2;
    tagRemoteRes2.tagRemoteResPtr = &tagRemoteResPtr2;
    ret = implBase->BuildOpRemoteLinkP2pResParam(link, tagRemoteRes2, TransportLinkType::SIO);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(tagRemoteResPtr2.linkP2pSio.remoteMem[INPUT].addr, reinterpret_cast<u64>(input.ptr()));
    EXPECT_EQ(tagRemoteResPtr2.linkP2pSio.remoteMem[OUTPUT].addr, reinterpret_cast<u64>(output.ptr()));
    EXPECT_EQ(tagRemoteRes2.p2pNotifyNum, 1);
    EXPECT_EQ(tagRemoteResPtr2.linkP2pSio.localIpcSignal[0].resId, 1U);
    EXPECT_EQ(tagRemoteResPtr2.linkP2pSio.remoteIpcSignal[0].resId, 7U);
    EXPECT_EQ(tagRemoteResPtr2.linkP2p.localIpcSignal[0].resId, INVALID_U64);

    GlobalMockObject::verify();
}

TEST_F(HcclCommAicpuTest_UT, AiCpuCreateAndGetNotify)
{
    HcclResult ret = HCCL_SUCCESS;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    std::shared_ptr<LocalNotify> localNotify;
    localNotify = std::make_shared<LocalNotify>();
    ret = localNotify->Init(NotifyLoadType::DEVICE_NOTIFY);

    HcclSignalInfo notifyInfo;
    notifyInfo.addr = 100;
    notifyInfo.devId = 1;
    notifyInfo.rankId = 2;
    notifyInfo.resId = 3;
    notifyInfo.tsId = 4;
    ASSERT_EQ(implBase->CreateAndGetAiCpuNotify(localNotify, notifyInfo), HCCL_SUCCESS);
}

HcclResult
stub_hrtGetDeviceInfo(u32 deviceId, HcclRtDeviceModuleType moduleType, HcclRtDeviceInfoType infoType, s64& val)
{
    val = 1;
    return HCCL_SUCCESS;
}

TEST_F(HcclCommAicpuTest_UT, CommunicatorCustomTest)
{
    char c = '1';
    MOCKER(realpath).stubs().with(mockcpp::any()).will(returnValue(&c));

    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    int inputPtr = 0;
    int outputPtr = 0;
    rtStream_t stm;
    u64 addr = 0;
    int tilingDataPtr;
    u32 tilingDataSize = 33 * 1024;
    std::string kernelName = "test";
    std::string tag = "test1";
    implBase->binCustomHandle_ = &inputPtr;
    implBase->binHandle_ = &inputPtr;

    AlgResourceResponse algResource;
    std::string newTag = "test111";

    MOCKER_CPP(&HcclCommunicator::BuildOpResParam).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::BuildCustomOpResParam).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::SetMC2EnvFlag).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hcclStreamSynchronize).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceInfo).stubs().will(invoke(stub_hrtGetDeviceInfo));
    std::string algName = "allreduce_mesh";

    HcclResult ret = implBase->AicpuResourceInit(algName, algResource, newTag, stm, HcclCMDType::HCCL_CMD_SEND, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    char binPath[10] = "hccl.json";
    uint32_t cpuKernelMode = 10;
    aclrtBinHandle binHandle;
    ret = implBase->LoadCustomFile(binPath, ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD, cpuKernelMode, binHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(memcpy_s).stubs().with().will(returnValue(0));
    ret = implBase->AicpuUnfoldKernelLaunchV2(
        &inputPtr, &outputPtr, stm, addr, &tilingDataPtr, tilingDataSize, kernelName,
        HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE, tag, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    binHandle = &inputPtr;
    implBase->UnloadBinary(binHandle);

    GlobalMockObject::verify();
}

HcclResult
stub_hrtGetDeviceInfo_close(u32 deviceId, HcclRtDeviceModuleType moduleType, HcclRtDeviceInfoType infoType, s64& val)
{
    val = 0;
    return HCCL_SUCCESS;
}

TEST_F(HcclCommAicpuTest_UT, CommunicatorCustomTest_CloseSwitch)
{
    char c = '1';
    MOCKER(realpath).stubs().with(mockcpp::any()).will(returnValue(&c));

    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    int inputPtr = 0;
    int outputPtr = 0;
    rtStream_t stm;
    u64 addr = 0;
    int tilingDataPtr;
    u32 tilingDataSize = 33 * 1024;
    std::string kernelName = "test";
    std::string tag = "test1";
    implBase->binCustomHandle_ = &inputPtr;
    implBase->binHandle_ = &inputPtr;

    AlgResourceResponse algResource;
    std::string newTag = "test111";

    MOCKER_CPP(&HcclCommunicator::BuildOpResParam).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::BuildCustomOpResParam).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::SetMC2EnvFlag).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hcclStreamSynchronize).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceInfo).stubs().will(invoke(stub_hrtGetDeviceInfo_close));
    std::string algName = "allreduce_mesh";

    HcclResult ret = implBase->AicpuResourceInit(algName, algResource, newTag, stm, HcclCMDType::HCCL_CMD_SEND, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    char binPath[10] = "hccl.json";
    uint32_t cpuKernelMode = 10;
    aclrtBinHandle binHandle;
    ret = implBase->LoadCustomFile(binPath, ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD, cpuKernelMode, binHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(memcpy_s).stubs().with().will(returnValue(0));
    ret = implBase->AicpuUnfoldKernelLaunchV2(
        &inputPtr, &outputPtr, stm, addr, &tilingDataPtr, tilingDataSize, kernelName,
        HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE, tag, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    binHandle = &inputPtr;
    implBase->UnloadBinary(binHandle);

    GlobalMockObject::verify();
}
