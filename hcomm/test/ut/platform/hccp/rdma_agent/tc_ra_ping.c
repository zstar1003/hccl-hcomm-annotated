/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <errno.h>
#include <stdlib.h>
#include <pthread.h>
#include "ut_dispatch.h"
#include "hccp_common.h"
#include "hccp_ping.h"
#include "ra.h"
#include "ra_ping.h"
#include "ra_hdc_ping.h"
#include "ra_hdc.h"
#include "ra_client_host.h"
#include "tc_ra_ping.h"

extern int
RaPingInitGetHandle(struct PingInitAttr* initAttr, struct PingInitInfo* initInfo, struct RaPingHandle* pingHandle);
extern int RaPingDeinitParaCheck(struct RaPingHandle* pingHandle);

void TcRaPingInitGetHandleAbnormal()
{
    struct RaPingHandle pingHandle = {0};
    struct PingInitAttr initAttr = {0};
    struct PingInitInfo initInfo = {0};
    int ret;

    ret = RaPingInitGetHandle(&initAttr, NULL, NULL);
    EXPECT_INT_EQ(ret, -EINVAL);

    initAttr.bufferSize = 1;
    ret = RaPingInitGetHandle(&initAttr, &initInfo, NULL);
    EXPECT_INT_EQ(ret, -EINVAL);

    initAttr.bufferSize = 0;
    initAttr.mode = NETWORK_PEER_ONLINE;
    ret = RaPingInitGetHandle(&initAttr, &initInfo, &pingHandle);
    EXPECT_INT_EQ(ret, -EINVAL);

    initAttr.mode = NETWORK_OFFLINE;
    mocker(RaRdevInitCheck, 1, -1);
    ret = RaPingInitGetHandle(&initAttr, &initInfo, &pingHandle);
    EXPECT_INT_EQ(ret, -22);
    mocker_clean();

    mocker(RaRdevInitCheck, 1, 0);
    mocker((stub_fn_t)pthread_mutex_init, 1, -1);
    ret = RaPingInitGetHandle(&initAttr, &initInfo, &pingHandle);
    EXPECT_INT_EQ(ret, -22);
    mocker_clean();

    initAttr.bufferSize = RA_RS_4K_PAGE_SIZE;
    initAttr.commInfo.rdma.udpSport = 65536;
    ret = RaPingInitGetHandle(&initAttr, &initInfo, &pingHandle);
    EXPECT_INT_EQ(ret, -22);
    mocker_clean();
}

void TcRaPingInitAbnormal()
{
    void* pingHandle = NULL;
    int ret;

    ret = RaPingInit(NULL, NULL, NULL);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker((stub_fn_t)calloc, 1, NULL);
    ret = RaPingInit(NULL, NULL, &pingHandle);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker(RaPingInitGetHandle, 1, -1);
    ret = RaPingInit(NULL, NULL, &pingHandle);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();
}

void TcRaPingTargetAddAbnormal()
{
    struct PingTargetInfo target[1] = {0};
    struct RaPingHandle pingHandle = {0};
    struct RaPingOps ops = {0};
    int num;
    int ret;

    num = 0;
    ret = RaPingTargetAdd((void*)(&pingHandle), target, num);
    EXPECT_INT_NE(ret, 0);

    num = 1;
    pingHandle.pingOps = &ops;
    ret = RaPingTargetAdd((void*)(&pingHandle), target, num);
    EXPECT_INT_NE(ret, 0);

    ops.raPingTargetAdd = RaHdcPingTargetAdd;
    pingHandle.phyId = RA_MAX_PHY_ID_NUM;
    ret = RaPingTargetAdd((void*)(&pingHandle), target, num);
    EXPECT_INT_NE(ret, 0);

    pingHandle.phyId = 0;
    pingHandle.taskCnt = 1;
    ret = RaPingTargetAdd((void*)(&pingHandle), target, num);
    EXPECT_INT_NE(ret, 0);

    pingHandle.taskCnt = 0;
    mocker(RaHdcPingTargetAdd, 1, -1);
    ret = RaPingTargetAdd((void*)(&pingHandle), target, num);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();
}

void TcRaPingTaskStartAbnormal()
{
    struct RaPingHandle pingHandle = {0};
    struct PingTaskAttr attr = {0};
    struct RaPingOps ops = {0};
    int ret;

    attr.packetCnt = 1;
    attr.packetInterval = 1;
    attr.timeoutInterval = 1;

    ret = RaPingTaskStart((void*)(&pingHandle), NULL);
    EXPECT_INT_NE(ret, 0);

    pingHandle.pingOps = &ops;
    ret = RaPingTaskStart((void*)(&pingHandle), &attr);
    EXPECT_INT_NE(ret, 0);

    ops.raPingTaskStart = RaHdcPingTaskStart;
    pingHandle.phyId = RA_MAX_PHY_ID_NUM;
    ret = RaPingTaskStart((void*)(&pingHandle), &attr);
    EXPECT_INT_NE(ret, 0);

    pingHandle.phyId = 0;
    pingHandle.taskCnt = 1;
    ret = RaPingTaskStart((void*)(&pingHandle), &attr);
    EXPECT_INT_NE(ret, 0);

    pingHandle.taskCnt = 0;
    pingHandle.targetCnt = 0;
    ret = RaPingTaskStart((void*)(&pingHandle), &attr);
    EXPECT_INT_NE(ret, 0);

    pingHandle.targetCnt = 1;
    pingHandle.bufferSize = 0;
    ret = RaPingTaskStart((void*)(&pingHandle), &attr);
    EXPECT_INT_NE(ret, 0);

    pingHandle.bufferSize = pingHandle.targetCnt * attr.packetCnt * PING_TOTAL_PAYLOAD_MAX_SIZE;
    mocker(RaHdcPingTaskStart, 1, -1);
    ret = RaPingTaskStart((void*)(&pingHandle), &attr);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();
}

void TcRaPingGetResultsAbnormal()
{
    struct PingTargetResult target[1] = {0};
    struct RaPingHandle pingHandle = {0};
    struct RaPingOps ops = {0};
    int num = 0;
    int ret;

    ret = RaPingGetResults((void*)(&pingHandle), target, NULL);
    EXPECT_INT_NE(ret, 0);

    num = 1;
    pingHandle.pingOps = &ops;
    ret = RaPingGetResults((void*)(&pingHandle), target, &num);
    EXPECT_INT_NE(ret, 0);

    ops.raPingGetResults = RaHdcPingGetResults;
    pingHandle.phyId = RA_MAX_PHY_ID_NUM;
    ret = RaPingGetResults((void*)(&pingHandle), target, &num);
    EXPECT_INT_NE(ret, 0);

    pingHandle.phyId = 0;
    pingHandle.targetCnt = 0;
    ret = RaPingGetResults((void*)(&pingHandle), target, &num);
    EXPECT_INT_NE(ret, 0);

    pingHandle.targetCnt = 1;
    mocker(RaHdcPingGetResults, 1, -1);
    ret = RaPingGetResults((void*)(&pingHandle), target, &num);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();
}

void TcRaPingTargetDelAbnoraml()
{
    struct PingTargetResult target[1] = {0};
    struct RaPingHandle pingHandle = {0};
    struct RaPingOps ops = {0};
    int num = 0;
    int ret;

    ret = RaPingTargetDel((void*)(&pingHandle), (struct PingTargetCommInfo*)target, num);
    EXPECT_INT_NE(ret, 0);

    num = 1;
    pingHandle.pingOps = &ops;
    ret = RaPingTargetDel((void*)(&pingHandle), (struct PingTargetCommInfo*)target, num);
    EXPECT_INT_NE(ret, 0);

    ops.raPingTargetDel = RaHdcPingTargetDel;
    pingHandle.taskCnt = 1;
    ret = RaPingTargetDel((void*)(&pingHandle), (struct PingTargetCommInfo*)target, num);
    EXPECT_INT_NE(ret, 0);

    pingHandle.taskCnt = 0;
    pingHandle.targetCnt = 0;
    ret = RaPingTargetDel((void*)(&pingHandle), (struct PingTargetCommInfo*)target, num);
    EXPECT_INT_NE(ret, 0);

    pingHandle.targetCnt = 1;
    pingHandle.phyId = RA_MAX_PHY_ID_NUM;
    ret = RaPingTargetDel((void*)(&pingHandle), (struct PingTargetCommInfo*)target, num);
    EXPECT_INT_NE(ret, 0);

    pingHandle.phyId = 0;
    mocker(RaHdcPingTargetDel, 1, -1);
    ret = RaPingTargetDel((void*)(&pingHandle), (struct PingTargetCommInfo*)target, num);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();
}

void TcRaPingTaskStopAbnormal()
{
    struct RaPingHandle pingHandle = {0};
    struct RaPingOps ops = {0};
    int ret;

    ret = RaPingTaskStop(NULL);
    EXPECT_INT_NE(ret, 0);

    pingHandle.pingOps = &ops;
    ret = RaPingTaskStop((void*)(&pingHandle));
    EXPECT_INT_NE(ret, 0);

    ops.raPingTaskStop = RaHdcPingTaskStop;
    pingHandle.phyId = RA_MAX_PHY_ID_NUM;
    ret = RaPingTaskStop((void*)(&pingHandle));
    EXPECT_INT_NE(ret, 0);

    pingHandle.phyId = 0;
    pingHandle.taskCnt = 0;
    ret = RaPingTaskStop((void*)(&pingHandle));
    EXPECT_INT_NE(ret, 0);

    pingHandle.taskCnt = 1;
    mocker(RaHdcPingTaskStop, 1, -1);
    ret = RaPingTaskStop((void*)(&pingHandle));
    EXPECT_INT_NE(ret, 0);
    mocker_clean();
}

int RaHdcPingDeinitStub(struct RaPingHandle* pingHandle) { return 0; }

void TcRaPingDeinitParaCheckAbnormal()
{
    struct RaPingHandle pingHandle = {0};
    struct RaPingOps ops = {0};
    int ret;

    pingHandle.phyId = RA_MAX_PHY_ID_NUM;
    ret = RaPingDeinitParaCheck(&pingHandle);
    EXPECT_INT_EQ(ret, -EINVAL);

    pingHandle.phyId = 0;
    pingHandle.pingOps = &ops;
    pingHandle.pingOps->raPingDeinit = RaHdcPingDeinitStub;
    mocker(RaInetPton, 1, -1);
    ret = RaPingDeinitParaCheck(&pingHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RaInetPton, 1, 0);
    pingHandle.pingOps->raPingDeinit = NULL;
    ret = RaPingDeinitParaCheck(&pingHandle);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();
}

void TcRaPingDeinitAbnoaml()
{
    struct RaPingHandle* pingHandle = calloc(1, sizeof(struct RaPingHandle));
    struct RaPingOps ops = {0};
    int ret;

    ret = RaPingDeinit(NULL);
    EXPECT_INT_NE(ret, 0);

    mocker(RaPingDeinitParaCheck, 1, -1);
    ret = RaPingDeinit((void*)pingHandle);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    ops.raPingDeinit = RaHdcPingDeinit;
    pingHandle->pingOps = &ops;
    mocker(RaPingDeinitParaCheck, 1, 0);
    mocker(RaHdcPingDeinit, 1, -1);
    ret = RaPingDeinit((void*)pingHandle);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();
}

static void InitAttrFill(struct PingInitAttr* initAttr)
{
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;

    initAttr->mode = NETWORK_OFFLINE;
    initAttr->dev.rdma = rdevInfo;
    initAttr->client.rdma.cqAttr.sendCqDepth = 128;
    initAttr->client.rdma.cqAttr.recvCqDepth = 128;
    initAttr->client.rdma.qpAttr.cap.maxInlineData = 32;
    initAttr->client.rdma.qpAttr.cap.maxSendSge = 1;
    initAttr->client.rdma.qpAttr.cap.maxSendWr = 128;
    initAttr->client.rdma.qpAttr.cap.maxRecvSge = 1;
    initAttr->client.rdma.qpAttr.cap.maxRecvWr = 128;

    initAttr->server.rdma.cqAttr.sendCqDepth = 128;
    initAttr->server.rdma.cqAttr.recvCqDepth = 128;
    initAttr->server.rdma.qpAttr.cap.maxInlineData = 32;
    initAttr->server.rdma.qpAttr.cap.maxSendSge = 1;
    initAttr->server.rdma.qpAttr.cap.maxSendWr = 128;
    initAttr->server.rdma.qpAttr.cap.maxRecvSge = 1;
    initAttr->server.rdma.qpAttr.cap.maxRecvWr = 128;
    initAttr->bufferSize = 8192;
}

void TcRaPing()
{
    struct PingTargetCommInfo targetCommClient = {0};
    struct PingTargetResult targetResultClient = {0};
    struct PingTargetInfo targetInfoClient = {0};
    char payloadClient[20] = "hello, client";
    struct PingInitInfo initInfo = {0};
    struct PingInitAttr initAttr = {0};
    struct PingTaskAttr taskAttr = {0};
    unsigned int targetResultNum = 1;
    struct RaPingOps ops = {0};
    void* pingHandle = NULL;
    int ret;

    InitAttrFill(&initAttr);
    initAttr.bufferSize = 4;
    ret = RaPingInit(&initAttr, &initInfo, &pingHandle);
    EXPECT_INT_NE(ret, 0);
    initAttr.bufferSize = 8192;
    mocker(RaRdevInitCheck, 20, 0);
    mocker((stub_fn_t)RaHdcProcessMsg, 20, 0);
    ret = RaPingInit(&initAttr, &initInfo, &pingHandle);
    EXPECT_INT_EQ(ret, 0);

    targetInfoClient.localInfo.rdma.hopLimit = 64;
    targetInfoClient.localInfo.rdma.qosAttr.tc = (33 & 0x3f) << 2;
    targetInfoClient.localInfo.rdma.qosAttr.sl = 4;
    targetInfoClient.remoteInfo.qpInfo = initInfo.client;
    ret = RaPingTargetAdd(pingHandle, &targetInfoClient, 1);
    EXPECT_INT_EQ(ret, 0);

    taskAttr.packetCnt = 1;
    taskAttr.packetInterval = 10;
    taskAttr.timeoutInterval = 10;
    ret = RaPingTaskStart(pingHandle, &taskAttr);
    EXPECT_INT_EQ(ret, 0);

    targetResultClient.remoteInfo = targetInfoClient.remoteInfo;
    mocker(RaHdcPingGetResults, 20, 0);
    ret = RaPingGetResults(pingHandle, &targetResultClient, &targetResultNum);
    EXPECT_INT_EQ(ret, 0);

    ret = RaPingTaskStop(pingHandle);
    EXPECT_INT_EQ(ret, 0);

    targetCommClient.ip = targetInfoClient.remoteInfo.ip;
    targetCommClient.qpInfo = targetInfoClient.remoteInfo.qpInfo;
    ret = RaPingTargetDel(pingHandle, &targetCommClient, 1);
    EXPECT_INT_EQ(ret, 0);

    ret = RaPingDeinit(pingHandle);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}
