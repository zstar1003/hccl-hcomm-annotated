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
#include <stdlib.h>
#include <assert.h>
#include <securec.h>
#include <ifaddrs.h>
#include <sys/socket.h>
#include <netdb.h>
#include <string>
#include <sys/types.h>
#include <stddef.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "dlra_function.h"

#define private public
#define protected public
#include "externalinput.h"
#include "adapter_rts.h"
#include "network_manager_pub.h"
#include "adapter_rts.h"
#include "typical_qp_manager.h"
#include "externalinput_pub.h"
#include "interface_hccl.h"
#undef private

#include <hccl/hccl_comm.h>
#include <hccl/hccl_inner.h>
#include <hccl/hccl_ex.h>
#include "llt_hccl_stub_pub.h"
#include "llt_hccl_stub_gdr.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace hccl;

class MultiThreadNpuGpu : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        ClearRdevNicInfo();
        std::cout << "MultiThreadNpuGpu SetUP" << std::endl;
    }
    static void TearDownTestCase()
    {
        ClearRdevNicInfo();
        std::cout << "MultiThreadNpuGpu TearDown" << std::endl;
    }

    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

HcclResult stub_HrtRaGetNotifyBaseAddr_3(RdmaHandle handle, u64* va, u64* size)
{
    *va = 0x20000000;
    *size = 4;
    return HCCL_SUCCESS;
}

HcclResult stub_HrtRaGetHccnCfg(s32 networkMode, u32 devicePhyId, enum HccnCfgKeyT key, std::string& value)
{
    value = "0_1_128_4";
    return HCCL_SUCCESS;
}

struct StubQpInfo {
    u32 qpn = 0;
};

thread_local static u32 gQpn = 1;
#define DEV_NUM 1

HcclResult
stub_hrtRaTypicalQpCreate(RdmaHandle rdmaHandle, int flag, int qpMode, struct TypicalQp* qpInfo, QpHandle& qpHandle)
{
    StubQpInfo* info = new StubQpInfo();
    info->qpn = gQpn++;
    HCCL_ERROR("QPN:%u", gQpn);
    qpHandle = (void*)info;
    qpInfo->qpn = info->qpn;
    return HCCL_SUCCESS;
}

HcclResult stub_hrtRaQpCreateWithAttrs(RdmaHandle rdmaHandle, struct QpExtAttrs* attrs, QpHandle& qpHandle)
{
    StubQpInfo* info = new StubQpInfo();
    info->qpn = gQpn++;
    HCCL_ERROR("QPN:%u", gQpn);
    qpHandle = (void*)info;
    return HCCL_SUCCESS;
}

HcclResult stub_hrtRaGetQpAttr(QpHandle qpHandle, struct QpAttr* attr)
{
    attr->qpn = gQpn++;
    return HCCL_SUCCESS;
}

HcclResult
stub_hrtRaGetInterfaceVersion_support(unsigned int phyId, unsigned int interfaceOpcode, unsigned int* interfaceVersion)
{
    *interfaceVersion = 2;
    return HCCL_SUCCESS;
}

HcclResult stub_hrtRaQpDestroy_1(QpHandle handle)
{
    delete (StubQpInfo*)handle;
    handle = nullptr;
    return HCCL_SUCCESS;
}

TEST_F(MultiThreadNpuGpu, EndtoEndOneProcess)
{
    MOCKER(hrtRaTypicalQpCreate).stubs().will(invoke(stub_hrtRaTypicalQpCreate));
    MOCKER(HrtRaQpDestroy).stubs().will(invoke(stub_hrtRaQpDestroy_1));
    MOCKER(HrtRaGetNotifyBaseAddr).stubs().will(invoke(stub_HrtRaGetNotifyBaseAddr_3));

    MOCKER(GetExternalInputRdmaTrafficClass).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaServerLevel).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaRetryCnt).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaTimeOut).stubs().will(returnValue(1));

    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtNotifyWaitWithTimeOut).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRDMADBSend).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HCCL_SUCCESS;
    EXPECT_EQ(hrtSetDevice(0), HCCL_SUCCESS);
    EXPECT_EQ(hcclAscendRdmaInit(), HCCL_SUCCESS);
    AscendQPInfo localQPInfo;
    EXPECT_EQ(hcclCreateAscendQP(&localQPInfo), HCCL_SUCCESS);
    AscendQPInfo remoteQpInfo;
    AscendQPQos qpQos;
    qpQos.sl = 4;
    qpQos.tc = 4;
    EXPECT_EQ(hcclModifyAscendQPEx(&localQPInfo, &remoteQpInfo, &qpQos), HCCL_SUCCESS);

    AscendMrInfo localSyncMemPrepare;
    localSyncMemPrepare.addr = 0x2;
    localSyncMemPrepare.size = 4;
    localSyncMemPrepare.key = 4;
    AscendMrInfo localSyncMemDone;
    localSyncMemDone.addr = 0x3;
    localSyncMemDone.size = 4;
    localSyncMemDone.key = 4;
    AscendMrInfo localSyncMemAck;
    localSyncMemAck.addr = 0x5;
    localSyncMemAck.size = 4;
    localSyncMemAck.key = 4;
    EXPECT_EQ(hcclAllocSyncMem(reinterpret_cast<int32_t**>(&(localSyncMemPrepare.addr))), HCCL_SUCCESS);
    EXPECT_EQ(hcclGetSyncMemRegKey(&localSyncMemPrepare), HCCL_SUCCESS);

    EXPECT_EQ(hcclAllocSyncMem(reinterpret_cast<int32_t**>(&(localSyncMemDone.addr))), HCCL_SUCCESS);
    EXPECT_EQ(hcclGetSyncMemRegKey(&localSyncMemDone), HCCL_SUCCESS);

    EXPECT_EQ(hcclAllocSyncMem(reinterpret_cast<int32_t**>(&(localSyncMemAck.addr))), HCCL_SUCCESS);
    EXPECT_EQ(hcclGetSyncMemRegKey(&localSyncMemAck), HCCL_SUCCESS);

    AscendMrInfo localMr;
    localMr.addr = 0x11111111;
    localMr.size = 8;
    localMr.key = 4;
    EXPECT_EQ(hcclAllocWindowMem((void**)(&localMr.addr), localMr.size), HCCL_SUCCESS);
    EXPECT_EQ(hcclRegisterMem(&localMr), HCCL_SUCCESS);

    AscendSendRecvLinkInfo linkInfo;
    linkInfo.localSyncMemAck = &localSyncMemAck;
    linkInfo.localQPinfo = &localQPInfo;
    linkInfo.localSyncMemDone = &localSyncMemDone;
    linkInfo.localSyncMemPrepare = &localSyncMemPrepare;
    linkInfo.remoteSyncMemAck = &localSyncMemAck;
    linkInfo.remoteSyncMemDone = &localSyncMemDone;
    linkInfo.remoteSyncMemPrepare = &localSyncMemPrepare;

    linkInfo.wqePerDoorbell = 2;
    aclrtStream stream = (aclrtStream)0x87654321;
    EXPECT_EQ(HcclBatchPutMRByAscendQP(1, &localMr, &localMr, &linkInfo, stream), HCCL_SUCCESS);
    EXPECT_EQ(HcclWaitPutMRByAscendQP(&linkInfo, stream), HCCL_SUCCESS);

    EXPECT_EQ(hcclDeRegisterMem(&localMr), HCCL_SUCCESS);
    EXPECT_EQ(hcclFreeWindowMem((void*)localMr.addr), HCCL_SUCCESS);

    EXPECT_EQ(hcclFreeSyncMem(reinterpret_cast<int32_t*>(localSyncMemPrepare.addr)), HCCL_SUCCESS);
    EXPECT_EQ(hcclFreeSyncMem(reinterpret_cast<int32_t*>(localSyncMemDone.addr)), HCCL_SUCCESS);
    EXPECT_EQ(hcclFreeSyncMem(reinterpret_cast<int32_t*>(localSyncMemAck.addr)), HCCL_SUCCESS);

    EXPECT_EQ(hcclDestroyAscendQP(&localQPInfo), HCCL_SUCCESS);
    EXPECT_EQ(hrtResetDevice(0), HCCL_SUCCESS);
    EXPECT_EQ(hcclAscendRdmaDeInit(), HCCL_SUCCESS);
    GlobalMockObject::verify();
}

void* ThreadHandleTypIcalQP(void* args)
{
    s32 devId = *(s32*)args;
    HcclResult ret = HCCL_SUCCESS;
    EXPECT_EQ(hrtSetDevice(devId), HCCL_SUCCESS);
    EXPECT_EQ(hcclAscendRdmaInit(), HCCL_SUCCESS);
    AscendQPInfo localQPInfo;
    localQPInfo.qpn = 1;
    EXPECT_EQ(hcclCreateAscendQP(&localQPInfo), HCCL_SUCCESS);
    AscendQPQos qpQos;
    qpQos.sl = 4;
    qpQos.tc = 4;
    EXPECT_EQ(hcclModifyAscendQPEx(&localQPInfo, &localQPInfo, &qpQos), HCCL_SUCCESS);

    AscendMrInfo localSyncMemPrepare;
    localSyncMemPrepare.addr = 0x2;
    localSyncMemPrepare.size = 4;
    localSyncMemPrepare.key = 4;
    AscendMrInfo localSyncMemDone;
    localSyncMemDone.addr = 0x3;
    localSyncMemDone.size = 4;
    localSyncMemDone.key = 4;
    AscendMrInfo localSyncMemAck;
    localSyncMemAck.addr = 0x5;
    localSyncMemAck.size = 4;
    localSyncMemAck.key = 4;
    EXPECT_EQ(hcclAllocSyncMem(reinterpret_cast<int32_t**>(&(localSyncMemPrepare.addr))), HCCL_SUCCESS);
    EXPECT_EQ(hcclGetSyncMemRegKey(&localSyncMemPrepare), HCCL_SUCCESS);

    EXPECT_EQ(hcclAllocSyncMem(reinterpret_cast<int32_t**>(&(localSyncMemDone.addr))), HCCL_SUCCESS);
    EXPECT_EQ(hcclGetSyncMemRegKey(&localSyncMemDone), HCCL_SUCCESS);

    EXPECT_EQ(hcclAllocSyncMem(reinterpret_cast<int32_t**>(&(localSyncMemAck.addr))), HCCL_SUCCESS);
    EXPECT_EQ(hcclGetSyncMemRegKey(&localSyncMemAck), HCCL_SUCCESS);

    AscendMrInfo localMr;
    localMr.addr = 0x11111111;
    localMr.size = 8;
    localMr.key = 4;
    EXPECT_EQ(hcclAllocWindowMem((void**)(&localMr.addr), localMr.size), HCCL_SUCCESS);
    EXPECT_EQ(hcclRegisterMem(&localMr), HCCL_SUCCESS);

    AscendSendRecvLinkInfo linkInfo;
    linkInfo.localSyncMemAck = &localSyncMemAck;
    linkInfo.localQPinfo = &localQPInfo;
    linkInfo.localSyncMemDone = &localSyncMemDone;
    linkInfo.localSyncMemPrepare = &localSyncMemPrepare;
    linkInfo.remoteSyncMemAck = &localSyncMemAck;
    linkInfo.remoteSyncMemDone = &localSyncMemDone;
    linkInfo.remoteSyncMemPrepare = &localSyncMemPrepare;

    linkInfo.wqePerDoorbell = 2;
    aclrtStream stream = (aclrtStream)0x87654321;
    EXPECT_EQ(HcclBatchPutMRByAscendQP(1, &localMr, &localMr, &linkInfo, stream), HCCL_SUCCESS);
    EXPECT_EQ(HcclWaitPutMRByAscendQP(&linkInfo, stream), HCCL_SUCCESS);

    EXPECT_EQ(hcclDeRegisterMem(&localMr), HCCL_SUCCESS);
    EXPECT_EQ(hcclFreeWindowMem((void*)localMr.addr), HCCL_SUCCESS);

    EXPECT_EQ(hcclFreeSyncMem(reinterpret_cast<int32_t*>(localSyncMemPrepare.addr)), HCCL_SUCCESS);
    EXPECT_EQ(hcclFreeSyncMem(reinterpret_cast<int32_t*>(localSyncMemDone.addr)), HCCL_SUCCESS);
    EXPECT_EQ(hcclFreeSyncMem(reinterpret_cast<int32_t*>(localSyncMemAck.addr)), HCCL_SUCCESS);

    EXPECT_EQ(hcclDestroyAscendQP(&localQPInfo), HCCL_SUCCESS);
    EXPECT_EQ(hcclAscendRdmaDeInit(), HCCL_SUCCESS);
    EXPECT_EQ(hrtResetDevice(devId), HCCL_SUCCESS);
    return (nullptr);
}

void* ThreadHandleQPWithAttr(void* args)
{
    s32 devId = *(s32*)args;
    HcclResult ret = HCCL_SUCCESS;
    EXPECT_EQ(hrtSetDevice(devId), HCCL_SUCCESS);
    EXPECT_EQ(hcclAscendRdmaInit(), HCCL_SUCCESS);
    AscendQPInfo localQPInfo;
    localQPInfo.qpn = 1;
    localQPInfo.rq_depth = 128;
    localQPInfo.sq_depth = 128;
    localQPInfo.scq_depth = 128;
    localQPInfo.rcq_depth = 128;
    EXPECT_EQ(hcclCreateAscendQPWithAttr(&localQPInfo), HCCL_SUCCESS);
    AscendQPQos qpQos;
    qpQos.sl = 4;
    qpQos.tc = 4;
    EXPECT_EQ(hcclModifyAscendQPEx(&localQPInfo, &localQPInfo, &qpQos), HCCL_SUCCESS);

    AscendMrInfo localSyncMemPrepare;
    localSyncMemPrepare.addr = 0x2;
    localSyncMemPrepare.size = 4;
    localSyncMemPrepare.key = 4;
    AscendMrInfo localSyncMemDone;
    localSyncMemDone.addr = 0x3;
    localSyncMemDone.size = 4;
    localSyncMemDone.key = 4;
    AscendMrInfo localSyncMemAck;
    localSyncMemAck.addr = 0x5;
    localSyncMemAck.size = 4;
    localSyncMemAck.key = 4;
    EXPECT_EQ(hcclAllocSyncMem(reinterpret_cast<int32_t**>(&(localSyncMemPrepare.addr))), HCCL_SUCCESS);
    EXPECT_EQ(hcclGetSyncMemRegKey(&localSyncMemPrepare), HCCL_SUCCESS);

    EXPECT_EQ(hcclAllocSyncMem(reinterpret_cast<int32_t**>(&(localSyncMemDone.addr))), HCCL_SUCCESS);
    EXPECT_EQ(hcclGetSyncMemRegKey(&localSyncMemDone), HCCL_SUCCESS);

    EXPECT_EQ(hcclAllocSyncMem(reinterpret_cast<int32_t**>(&(localSyncMemAck.addr))), HCCL_SUCCESS);
    EXPECT_EQ(hcclGetSyncMemRegKey(&localSyncMemAck), HCCL_SUCCESS);

    AscendMrInfo localMr;
    localMr.addr = 0x11111111;
    localMr.size = 8;
    localMr.key = 4;
    EXPECT_EQ(hcclAllocWindowMem((void**)(&localMr.addr), localMr.size), HCCL_SUCCESS);
    EXPECT_EQ(hcclRegisterMem(&localMr), HCCL_SUCCESS);

    AscendSendRecvLinkInfo linkInfo;
    linkInfo.localSyncMemAck = &localSyncMemAck;
    linkInfo.localQPinfo = &localQPInfo;
    linkInfo.localSyncMemDone = &localSyncMemDone;
    linkInfo.localSyncMemPrepare = &localSyncMemPrepare;
    linkInfo.remoteSyncMemAck = &localSyncMemAck;
    linkInfo.remoteSyncMemDone = &localSyncMemDone;
    linkInfo.remoteSyncMemPrepare = &localSyncMemPrepare;

    linkInfo.wqePerDoorbell = 2;
    aclrtStream stream = (aclrtStream)0x87654321;
    EXPECT_EQ(HcclBatchPutMRByAscendQP(1, &localMr, &localMr, &linkInfo, stream), HCCL_SUCCESS);
    EXPECT_EQ(HcclWaitPutMRByAscendQP(&linkInfo, stream), HCCL_SUCCESS);

    EXPECT_EQ(hcclDeRegisterMem(&localMr), HCCL_SUCCESS);
    EXPECT_EQ(hcclFreeWindowMem((void*)localMr.addr), HCCL_SUCCESS);

    EXPECT_EQ(hcclFreeSyncMem(reinterpret_cast<int32_t*>(localSyncMemPrepare.addr)), HCCL_SUCCESS);
    EXPECT_EQ(hcclFreeSyncMem(reinterpret_cast<int32_t*>(localSyncMemDone.addr)), HCCL_SUCCESS);
    EXPECT_EQ(hcclFreeSyncMem(reinterpret_cast<int32_t*>(localSyncMemAck.addr)), HCCL_SUCCESS);

    EXPECT_EQ(hcclDestroyAscendQP(&localQPInfo), HCCL_SUCCESS);
    EXPECT_EQ(hcclAscendRdmaDeInit(), HCCL_SUCCESS);
    EXPECT_EQ(hrtResetDevice(devId), HCCL_SUCCESS);
    return (nullptr);
}

TEST_F(MultiThreadNpuGpu, EndtoEndMutiThread)
{
    MOCKER(hrtRaTypicalQpCreate).stubs().will(invoke(stub_hrtRaTypicalQpCreate));
    MOCKER(HrtRaQpDestroy).stubs().will(invoke(stub_hrtRaQpDestroy_1));
    MOCKER(HrtRaGetNotifyBaseAddr).stubs().will(invoke(stub_HrtRaGetNotifyBaseAddr_3));

    MOCKER(GetExternalInputRdmaTrafficClass).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaServerLevel).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaRetryCnt).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaTimeOut).stubs().will(returnValue(1));

    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtNotifyWaitWithTimeOut).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRDMADBSend).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    sal_thread_t tid[DEV_NUM];
    for (int devId = 0; devId < DEV_NUM; devId++) {
        tid[devId] = sal_thread_create("thread", ThreadHandleTypIcalQP, (void*)&devId);
        EXPECT_NE(tid[devId], (sal_thread_t) nullptr);
    }
    for (s32 devId = 0; devId < DEV_NUM; ++devId) {
        while (sal_thread_is_running(tid[devId])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }
    SaluSleep(SAL_MILLISECOND_USEC * 2);
    for (int devId = 0; devId < DEV_NUM; devId++) {
        (void)sal_thread_destroy(tid[devId]);
    }
    GlobalMockObject::verify();
}

TEST_F(MultiThreadNpuGpu, EndtoEndMutiThreadSwitchDevice)
{
    MOCKER(hrtRaTypicalQpCreate).stubs().will(invoke(stub_hrtRaTypicalQpCreate));
    MOCKER(HrtRaQpDestroy).stubs().will(invoke(stub_hrtRaQpDestroy_1));
    MOCKER(HrtRaGetNotifyBaseAddr).stubs().will(invoke(stub_HrtRaGetNotifyBaseAddr_3));

    MOCKER(GetExternalInputRdmaTrafficClass).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaServerLevel).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaRetryCnt).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaTimeOut).stubs().will(returnValue(1));

    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtNotifyWaitWithTimeOut).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRDMADBSend).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    for (int devId = 0; devId < 8; devId++) {
        ThreadHandleTypIcalQP((void*)&devId);
    }
    GlobalMockObject::verify();
}

TEST_F(MultiThreadNpuGpu, EndtoEndOneProcessWithAttr)
{
    MOCKER(hrtRaGetQpAttr).stubs().will(invoke(stub_hrtRaGetQpAttr));
    MOCKER(hrtRaGetInterfaceVersion).stubs().will(invoke(stub_hrtRaGetInterfaceVersion_support));
    MOCKER(hrtRaQpCreateWithAttrs).stubs().will(invoke(stub_hrtRaQpCreateWithAttrs));
    MOCKER(HrtRaQpDestroy).stubs().will(invoke(stub_hrtRaQpDestroy_1));
    MOCKER(HrtRaGetNotifyBaseAddr).stubs().will(invoke(stub_HrtRaGetNotifyBaseAddr_3));

    MOCKER(GetExternalInputRdmaTrafficClass).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaServerLevel).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaRetryCnt).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaTimeOut).stubs().will(returnValue(1));

    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtNotifyWaitWithTimeOut).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRDMADBSend).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    for (int devId = 0; devId < 8; devId++) {
        ThreadHandleQPWithAttr((void*)&devId);
    }
    GlobalMockObject::verify();
}

TEST_F(MultiThreadNpuGpu, EndtoEndOneProcessWithAttrMultiThread)
{
    MOCKER(hrtRaGetQpAttr).stubs().will(invoke(stub_hrtRaGetQpAttr));
    MOCKER(hrtRaGetInterfaceVersion).stubs().will(invoke(stub_hrtRaGetInterfaceVersion_support));
    MOCKER(hrtRaQpCreateWithAttrs).stubs().will(invoke(stub_hrtRaQpCreateWithAttrs));
    MOCKER(HrtRaQpDestroy).stubs().will(invoke(stub_hrtRaQpDestroy_1));
    MOCKER(HrtRaGetNotifyBaseAddr).stubs().will(invoke(stub_HrtRaGetNotifyBaseAddr_3));

    MOCKER(GetExternalInputRdmaTrafficClass).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaServerLevel).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaRetryCnt).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaTimeOut).stubs().will(returnValue(1));

    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtNotifyWaitWithTimeOut).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRDMADBSend).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    sal_thread_t tid[DEV_NUM];
    for (int devId = 0; devId < DEV_NUM; devId++) {
        tid[devId] = sal_thread_create("thread", ThreadHandleQPWithAttr, (void*)&devId);
        EXPECT_NE(tid[devId], (sal_thread_t) nullptr);
    }
    for (s32 devId = 0; devId < DEV_NUM; ++devId) {
        while (sal_thread_is_running(tid[devId])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }
    SaluSleep(SAL_MILLISECOND_USEC * 2);
    for (int devId = 0; devId < DEV_NUM; devId++) {
        (void)sal_thread_destroy(tid[devId]);
    }
    GlobalMockObject::verify();
}

TEST_F(MultiThreadNpuGpu, CreateQPWithAttrNotSupport)
{
    QpConfigInfo qpConfig;
    struct TypicalQp qpInfoTmp;
    QpHandle qpHandle;
    RdmaHandle rdmaHandle;

    qpConfig.rq_depth = 128;
    qpConfig.sq_depth = 128;
    qpConfig.scq_depth = 128;
    qpConfig.rcq_depth = 128;
    EXPECT_EQ(CreateQpWithDepthConfig(rdmaHandle, OPBASE_QP_MODE, qpConfig, qpHandle, qpInfoTmp), HCCL_E_NOT_SUPPORT);

    MOCKER(HrtRaGetNotifyBaseAddr).stubs().will(invoke(stub_HrtRaGetNotifyBaseAddr_3));
    EXPECT_EQ(hrtSetDevice(0), HCCL_SUCCESS);
    EXPECT_EQ(hcclAscendRdmaInit(), HCCL_SUCCESS);
    AscendQPInfo localQPInfo;
    localQPInfo.qpn = 1;
    localQPInfo.rq_depth = 128;
    localQPInfo.sq_depth = 128;
    localQPInfo.scq_depth = 128;
    localQPInfo.rcq_depth = 128;
    EXPECT_EQ(hcclCreateAscendQPWithAttr(&localQPInfo), HCCL_E_INTERNAL);

    EXPECT_EQ(hcclAscendRdmaDeInit(), HCCL_SUCCESS);
    EXPECT_EQ(hrtResetDevice(0), HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(MultiThreadNpuGpu, CreateQPWithAttrConfigError)
{
    MOCKER(HrtRaGetNotifyBaseAddr).stubs().will(invoke(stub_HrtRaGetNotifyBaseAddr_3));
    EXPECT_EQ(hrtSetDevice(0), HCCL_SUCCESS);
    EXPECT_EQ(hcclAscendRdmaInit(), HCCL_SUCCESS);
    AscendQPInfo localQPInfo;
    localQPInfo.qpn = 1;
    localQPInfo.rq_depth = 128;
    localQPInfo.scq_depth = 128;
    localQPInfo.rcq_depth = 128;

    localQPInfo.sq_depth = 32789;
    EXPECT_EQ(hcclCreateAscendQPWithAttr(&localQPInfo), HCCL_E_PARA);
    localQPInfo.sq_depth = 2;
    EXPECT_EQ(hcclCreateAscendQPWithAttr(&localQPInfo), HCCL_E_PARA);
    localQPInfo.sq_depth = 253;
    EXPECT_EQ(hcclCreateAscendQPWithAttr(&localQPInfo), HCCL_E_PARA);
    localQPInfo.sq_depth = 128;

    localQPInfo.rq_depth = 32789;
    EXPECT_EQ(hcclCreateAscendQPWithAttr(&localQPInfo), HCCL_E_PARA);
    localQPInfo.rq_depth = 2;
    EXPECT_EQ(hcclCreateAscendQPWithAttr(&localQPInfo), HCCL_E_PARA);
    localQPInfo.rq_depth = 253;
    EXPECT_EQ(hcclCreateAscendQPWithAttr(&localQPInfo), HCCL_E_PARA);
    localQPInfo.rq_depth = 128;

    localQPInfo.scq_depth = 32789;
    EXPECT_EQ(hcclCreateAscendQPWithAttr(&localQPInfo), HCCL_E_PARA);
    localQPInfo.scq_depth = 2;
    EXPECT_EQ(hcclCreateAscendQPWithAttr(&localQPInfo), HCCL_E_PARA);
    localQPInfo.scq_depth = 253;
    EXPECT_EQ(hcclCreateAscendQPWithAttr(&localQPInfo), HCCL_E_PARA);
    localQPInfo.scq_depth = 128;

    localQPInfo.rcq_depth = 32789;
    EXPECT_EQ(hcclCreateAscendQPWithAttr(&localQPInfo), HCCL_E_PARA);
    localQPInfo.rcq_depth = 2;
    EXPECT_EQ(hcclCreateAscendQPWithAttr(&localQPInfo), HCCL_E_PARA);
    localQPInfo.rcq_depth = 253;
    EXPECT_EQ(hcclCreateAscendQPWithAttr(&localQPInfo), HCCL_E_PARA);
    localQPInfo.rcq_depth = 128;

    EXPECT_EQ(hcclAscendRdmaDeInit(), HCCL_SUCCESS);
    EXPECT_EQ(hrtResetDevice(0), HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(MultiThreadNpuGpu, OneSideEndtoEndOneProcess)
{
    MOCKER(hrtRaTypicalQpCreate).stubs().will(invoke(stub_hrtRaTypicalQpCreate));
    MOCKER(HrtRaQpDestroy).stubs().will(invoke(stub_hrtRaQpDestroy_1));
    MOCKER(HrtRaGetNotifyBaseAddr).stubs().will(invoke(stub_HrtRaGetNotifyBaseAddr_3));
    MOCKER(HrtRaGetHccnCfg).stubs().will(invoke(stub_HrtRaGetHccnCfg));
    MOCKER(GetExternalInputRdmaTrafficClass).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaServerLevel).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaRetryCnt).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaTimeOut).stubs().will(returnValue(1));

    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtNotifyWaitWithTimeOut).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRDMADBSend).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HCCL_SUCCESS;
    EXPECT_EQ(hrtSetDevice(0), HCCL_SUCCESS);
    EXPECT_EQ(hcclAscendRdmaInit(), HCCL_SUCCESS);
    AscendQPInfo localQPInfo;
    EXPECT_EQ(hcclCreateAscendQP(&localQPInfo), HCCL_SUCCESS);
    AscendQPInfo remoteQpInfo;
    AscendQPQos qpQos;
    qpQos.sl = 4;
    qpQos.tc = 4;
    EXPECT_EQ(hcclModifyAscendQPEx(&localQPInfo, &remoteQpInfo, &qpQos), HCCL_SUCCESS);

    AscendMrInfo localSyncMemPrepare;
    localSyncMemPrepare.addr = 0x2;
    localSyncMemPrepare.size = 4;
    localSyncMemPrepare.key = 4;
    AscendMrInfo localSyncMemDone;
    localSyncMemDone.addr = 0x3;
    localSyncMemDone.size = 4;
    localSyncMemDone.key = 4;
    AscendMrInfo localSyncMemAck;
    localSyncMemAck.addr = 0x5;
    localSyncMemAck.size = 4;
    localSyncMemAck.key = 4;
    EXPECT_EQ(hcclAllocSyncMem(reinterpret_cast<int32_t**>(&(localSyncMemPrepare.addr))), HCCL_SUCCESS);
    EXPECT_EQ(hcclGetSyncMemRegKey(&localSyncMemPrepare), HCCL_SUCCESS);

    EXPECT_EQ(hcclAllocSyncMem(reinterpret_cast<int32_t**>(&(localSyncMemDone.addr))), HCCL_SUCCESS);
    EXPECT_EQ(hcclGetSyncMemRegKey(&localSyncMemDone), HCCL_SUCCESS);

    EXPECT_EQ(hcclAllocSyncMem(reinterpret_cast<int32_t**>(&(localSyncMemAck.addr))), HCCL_SUCCESS);
    EXPECT_EQ(hcclGetSyncMemRegKey(&localSyncMemAck), HCCL_SUCCESS);

    AscendMrInfo localMr;
    localMr.addr = 0x11111111;
    localMr.size = 8;
    localMr.key = 4;
    EXPECT_EQ(hcclAllocWindowMem((void**)(&localMr.addr), localMr.size), HCCL_SUCCESS);
    EXPECT_EQ(hcclRegisterMem(&localMr), HCCL_SUCCESS);

    AscendSendLinkInfo linkInfo;
    linkInfo.localSyncMemAck = &localSyncMemAck;
    linkInfo.localQPinfo = &localQPInfo;
    linkInfo.remoteNotifyValueMem = &localSyncMemAck;

    linkInfo.wqePerDoorbell = 2;
    aclrtStream stream = (aclrtStream)0x87654321;
    EXPECT_EQ(HcclOneSideBatchPutByAscendQP(1, &localMr, &localMr, &linkInfo, stream), HCCL_SUCCESS);

    EXPECT_EQ(hcclDeRegisterMem(&localMr), HCCL_SUCCESS);
    EXPECT_EQ(hcclFreeWindowMem((void*)localMr.addr), HCCL_SUCCESS);

    EXPECT_EQ(hcclFreeSyncMem(reinterpret_cast<int32_t*>(localSyncMemPrepare.addr)), HCCL_SUCCESS);
    EXPECT_EQ(hcclFreeSyncMem(reinterpret_cast<int32_t*>(localSyncMemDone.addr)), HCCL_SUCCESS);
    EXPECT_EQ(hcclFreeSyncMem(reinterpret_cast<int32_t*>(localSyncMemAck.addr)), HCCL_SUCCESS);

    EXPECT_EQ(hcclDestroyAscendQP(&localQPInfo), HCCL_SUCCESS);
    EXPECT_EQ(hrtResetDevice(0), HCCL_SUCCESS);
    EXPECT_EQ(hcclAscendRdmaDeInit(), HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(MultiThreadNpuGpu, OneSideEndtoEndOneProcessVerifyFailed)
{
    MOCKER(hrtRaTypicalQpCreate).stubs().will(invoke(stub_hrtRaTypicalQpCreate));
    MOCKER(HrtRaQpDestroy).stubs().will(invoke(stub_hrtRaQpDestroy_1));
    MOCKER(HrtRaGetNotifyBaseAddr).stubs().will(invoke(stub_HrtRaGetNotifyBaseAddr_3));

    MOCKER(GetExternalInputRdmaTrafficClass).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaServerLevel).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaRetryCnt).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaTimeOut).stubs().will(returnValue(1));

    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtNotifyWaitWithTimeOut).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRDMADBSend).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HCCL_SUCCESS;
    EXPECT_EQ(hrtSetDevice(0), HCCL_SUCCESS);
    EXPECT_EQ(hcclAscendRdmaInit(), HCCL_SUCCESS);
    AscendQPInfo localQPInfo;
    EXPECT_EQ(hcclCreateAscendQP(&localQPInfo), HCCL_SUCCESS);
    AscendQPInfo remoteQpInfo;
    AscendQPQos qpQos;
    qpQos.sl = 4;
    qpQos.tc = 4;
    EXPECT_EQ(hcclModifyAscendQPEx(&localQPInfo, &remoteQpInfo, &qpQos), HCCL_SUCCESS);

    AscendMrInfo localSyncMemPrepare;
    localSyncMemPrepare.addr = 0x2;
    localSyncMemPrepare.size = 4;
    localSyncMemPrepare.key = 4;
    AscendMrInfo localSyncMemDone;
    localSyncMemDone.addr = 0x3;
    localSyncMemDone.size = 4;
    localSyncMemDone.key = 4;
    AscendMrInfo localSyncMemAck;
    localSyncMemAck.addr = 0x5;
    localSyncMemAck.size = 4;
    localSyncMemAck.key = 4;
    EXPECT_EQ(hcclAllocSyncMem(reinterpret_cast<int32_t**>(&(localSyncMemPrepare.addr))), HCCL_SUCCESS);
    EXPECT_EQ(hcclGetSyncMemRegKey(&localSyncMemPrepare), HCCL_SUCCESS);

    EXPECT_EQ(hcclAllocSyncMem(reinterpret_cast<int32_t**>(&(localSyncMemDone.addr))), HCCL_SUCCESS);
    EXPECT_EQ(hcclGetSyncMemRegKey(&localSyncMemDone), HCCL_SUCCESS);

    EXPECT_EQ(hcclAllocSyncMem(reinterpret_cast<int32_t**>(&(localSyncMemAck.addr))), HCCL_SUCCESS);
    EXPECT_EQ(hcclGetSyncMemRegKey(&localSyncMemAck), HCCL_SUCCESS);

    AscendMrInfo localMr;
    localMr.addr = 0x11111111;
    localMr.size = 8;
    localMr.key = 4;
    EXPECT_EQ(hcclAllocWindowMem((void**)(&localMr.addr), localMr.size), HCCL_SUCCESS);
    EXPECT_EQ(hcclRegisterMem(&localMr), HCCL_SUCCESS);

    AscendMrInfo remoteNotifyValue;
    remoteNotifyValue.addr = 0x5;
    remoteNotifyValue.size = 9;
    remoteNotifyValue.key = 4;

    AscendSendLinkInfo linkInfo;
    linkInfo.localSyncMemAck = &localSyncMemAck;
    linkInfo.localQPinfo = &localQPInfo;
    linkInfo.remoteNotifyValueMem = &remoteNotifyValue;

    linkInfo.wqePerDoorbell = 2;
    aclrtStream stream = (aclrtStream)0x87654321;
    EXPECT_EQ(HcclOneSideBatchPutByAscendQP(1, &localMr, &localMr, &linkInfo, stream), HCCL_E_PARA);

    EXPECT_EQ(hcclDeRegisterMem(&localMr), HCCL_SUCCESS);
    EXPECT_EQ(hcclFreeWindowMem((void*)localMr.addr), HCCL_SUCCESS);

    EXPECT_EQ(hcclFreeSyncMem(reinterpret_cast<int32_t*>(localSyncMemPrepare.addr)), HCCL_SUCCESS);
    EXPECT_EQ(hcclFreeSyncMem(reinterpret_cast<int32_t*>(localSyncMemDone.addr)), HCCL_SUCCESS);
    EXPECT_EQ(hcclFreeSyncMem(reinterpret_cast<int32_t*>(localSyncMemAck.addr)), HCCL_SUCCESS);

    EXPECT_EQ(hcclDestroyAscendQP(&localQPInfo), HCCL_SUCCESS);
    EXPECT_EQ(hrtResetDevice(0), HCCL_SUCCESS);
    EXPECT_EQ(hcclAscendRdmaDeInit(), HCCL_SUCCESS);
    GlobalMockObject::verify();
}

void* OneSideThreadHandleTypIcalQP(void* args)
{
    s32 devId = *(s32*)args;

    HcclResult ret = HCCL_SUCCESS;
    EXPECT_EQ(hrtSetDevice(devId), HCCL_SUCCESS);
    EXPECT_EQ(hcclAscendRdmaInit(), HCCL_SUCCESS);
    AscendQPInfo localQPInfo;
    EXPECT_EQ(hcclCreateAscendQP(&localQPInfo), HCCL_SUCCESS);
    AscendQPInfo remoteQpInfo;
    EXPECT_EQ(hcclCreateAscendQP(&remoteQpInfo), HCCL_SUCCESS);
    AscendQPQos qpQos;
    qpQos.sl = 4;
    qpQos.tc = 4;
    EXPECT_EQ(hcclModifyAscendQPEx(&localQPInfo, &remoteQpInfo, &qpQos), HCCL_SUCCESS);

    AscendMrInfo localSyncMemPrepare;
    localSyncMemPrepare.addr = 0x2;
    localSyncMemPrepare.size = 4;
    localSyncMemPrepare.key = 4;
    AscendMrInfo localSyncMemDone;
    localSyncMemDone.addr = 0x3;
    localSyncMemDone.size = 4;
    localSyncMemDone.key = 4;
    AscendMrInfo localSyncMemAck;
    localSyncMemAck.addr = 0x5;
    localSyncMemAck.size = 4;
    localSyncMemAck.key = 4;
    EXPECT_EQ(hcclAllocSyncMem(reinterpret_cast<int32_t**>(&(localSyncMemPrepare.addr))), HCCL_SUCCESS);
    EXPECT_EQ(hcclGetSyncMemRegKey(&localSyncMemPrepare), HCCL_SUCCESS);

    EXPECT_EQ(hcclAllocSyncMem(reinterpret_cast<int32_t**>(&(localSyncMemDone.addr))), HCCL_SUCCESS);
    EXPECT_EQ(hcclGetSyncMemRegKey(&localSyncMemDone), HCCL_SUCCESS);

    EXPECT_EQ(hcclAllocSyncMem(reinterpret_cast<int32_t**>(&(localSyncMemAck.addr))), HCCL_SUCCESS);
    EXPECT_EQ(hcclGetSyncMemRegKey(&localSyncMemAck), HCCL_SUCCESS);

    AscendMrInfo localMr;
    localMr.addr = 0x11111111;
    localMr.size = 8;
    localMr.key = 4;
    EXPECT_EQ(hcclAllocWindowMem((void**)(&localMr.addr), localMr.size), HCCL_SUCCESS);
    EXPECT_EQ(hcclRegisterMem(&localMr), HCCL_SUCCESS);

    AscendSendLinkInfo linkInfo;
    linkInfo.localSyncMemAck = &localSyncMemAck;
    linkInfo.localQPinfo = &localQPInfo;
    linkInfo.remoteNotifyValueMem = &localSyncMemAck;

    linkInfo.wqePerDoorbell = 2;
    aclrtStream stream = (aclrtStream)0x87654321;
    EXPECT_EQ(HcclOneSideBatchPutByAscendQP(1, &localMr, &localMr, &linkInfo, stream), HCCL_SUCCESS);

    EXPECT_EQ(hcclDeRegisterMem(&localMr), HCCL_SUCCESS);
    EXPECT_EQ(hcclFreeWindowMem((void*)localMr.addr), HCCL_SUCCESS);

    EXPECT_EQ(hcclFreeSyncMem(reinterpret_cast<int32_t*>(localSyncMemPrepare.addr)), HCCL_SUCCESS);
    EXPECT_EQ(hcclFreeSyncMem(reinterpret_cast<int32_t*>(localSyncMemDone.addr)), HCCL_SUCCESS);
    EXPECT_EQ(hcclFreeSyncMem(reinterpret_cast<int32_t*>(localSyncMemAck.addr)), HCCL_SUCCESS);

    EXPECT_EQ(hcclDestroyAscendQP(&localQPInfo), HCCL_SUCCESS);
    EXPECT_EQ(hrtResetDevice(devId), HCCL_SUCCESS);
    EXPECT_EQ(hcclAscendRdmaDeInit(), HCCL_SUCCESS);
    return (nullptr);
}

TEST_F(MultiThreadNpuGpu, OneSideEndtoEndMutiThreadSwitchDevice)
{
    MOCKER(hrtRaTypicalQpCreate).stubs().will(invoke(stub_hrtRaTypicalQpCreate));
    MOCKER(HrtRaQpDestroy).stubs().will(invoke(stub_hrtRaQpDestroy_1));
    MOCKER(HrtRaGetNotifyBaseAddr).stubs().will(invoke(stub_HrtRaGetNotifyBaseAddr_3));

    MOCKER(GetExternalInputRdmaTrafficClass).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaServerLevel).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaRetryCnt).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaTimeOut).stubs().will(returnValue(1));

    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtNotifyWaitWithTimeOut).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRDMADBSend).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    for (int devId = 0; devId < 8; devId++) {
        OneSideThreadHandleTypIcalQP((void*)&devId);
    }
    GlobalMockObject::verify();
}

TEST_F(MultiThreadNpuGpu, OneSideEndtoEndMutiThread)
{
    MOCKER(hrtRaTypicalQpCreate).stubs().will(invoke(stub_hrtRaTypicalQpCreate));
    MOCKER(HrtRaQpDestroy).stubs().will(invoke(stub_hrtRaQpDestroy_1));
    MOCKER(HrtRaGetNotifyBaseAddr).stubs().will(invoke(stub_HrtRaGetNotifyBaseAddr_3));

    MOCKER(GetExternalInputRdmaTrafficClass).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaServerLevel).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaRetryCnt).stubs().will(returnValue(1));
    MOCKER(GetExternalInputRdmaTimeOut).stubs().will(returnValue(1));

    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtNotifyWaitWithTimeOut).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRDMADBSend).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    sal_thread_t tid[DEV_NUM];
    for (int devId = 0; devId < DEV_NUM; devId++) {
        tid[devId] = sal_thread_create("thread", OneSideThreadHandleTypIcalQP, (void*)&devId);
        EXPECT_NE(tid[devId], (sal_thread_t) nullptr);
    }
    for (s32 devId = 0; devId < DEV_NUM; ++devId) {
        while (sal_thread_is_running(tid[devId])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }
    SaluSleep(SAL_MILLISECOND_USEC * 2);
    for (int devId = 0; devId < DEV_NUM; devId++) {
        (void)sal_thread_destroy(tid[devId]);
    }
    GlobalMockObject::verify();
}
