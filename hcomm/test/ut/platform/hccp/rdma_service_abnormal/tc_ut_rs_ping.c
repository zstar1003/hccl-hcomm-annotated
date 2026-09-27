/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>
#include "ascend_hal.h"
#include "ut_dispatch.h"
#include "dl_ibverbs_function.h"
#include "ra_rs_comm.h"
#include "ra_rs_err.h"
#include "rs.h"
#include "hccp_common.h"
#include "rs_inner.h"
#include "rs_ping_inner.h"
#include "hccp_ping.h"
#include "rs_ping.h"
#include "rs_ping_roce.h"
#include "tc_ut_rs_ping.h"

extern int RsPingRocePingCbInit(
    unsigned int phyId, struct PingInitAttr* attr, struct PingInitInfo* info, struct RsPingCtxCb* pingCb);
extern int RsGetPingCb(struct RaRsDevInfo* rdev, struct RsPingCtxCb** pingCb);
extern int
RsPingRoceFindTargetNode(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPingTargetInfo** node);
extern int
RsPingRoceAllocTargetNode(struct RsPingCtxCb* pingCb, struct PingTargetInfo* target, struct RsPingTargetInfo** node);
extern int
RsPingRoceGetTargetResult(struct RsPingCtxCb* pingCb, struct PingTargetCommInfo* target, struct PingResultInfo* result);
extern void RsPingCommonDeinitLocalQp(struct rs_cb* rscb, struct RsPingCtxCb* pingCb, struct RsPingLocalQpCb* qpCb);
extern int RsPingCbGetIbCtxAndIndex(struct rdev* rdevInfo, struct RsPingCtxCb* pingCb);
extern int RsPingCbGetDevRdevIndex(struct RsPingCtxCb* pingCb, int index);
extern int RsPingUrmaCheckFd(struct RsPingCtxCb* pingCb, int fd);
extern int RsPongUrmaCheckFd(struct RsPingCtxCb* pingCb, int fd);

static struct rs_cb gTmpRsCb0;
static struct rs_cb gTmpRsCb1;
static struct rs_cb gPingRsCb;
static struct RsPingCtxCb gTmpPingCb;
static struct RsPingTargetInfo gPingTargetNode;
static struct RsPingTargetInfo gPingTargetNode1;

int RsDev2rscb_stub0(uint32_t devId, struct rs_cb** rsCb, bool initFlag)
{
    *rsCb = &gTmpRsCb0;

    return 0;
}

int RsDev2rscb_stub1(uint32_t devId, struct rs_cb** rsCb, bool initFlag)
{
    pthread_mutex_init(&gTmpRsCb1.pingCb.pingMutex, NULL);
    gTmpRsCb1.pingCb.threadStatus = RS_PING_THREAD_RUNNING;

    *rsCb = &gTmpRsCb1;

    return 0;
}

int RsGetPingCbStub(struct RaRsDevInfo* rdev, struct RsPingCtxCb** pingCb)
{
    *pingCb = &gTmpPingCb;
    (*pingCb)->pingPongOps = RsPingRoceGetOps();
    (*pingCb)->pingPongDfx = RsPingRoceGetDfx();

    return 0;
}

int RsGetRsCbStub(unsigned int phyId, struct rs_cb** rsCb)
{
    *rsCb = &gPingRsCb;
    (*rsCb)->pingCb.pingPongOps = RsPingRoceGetOps();
    (*rsCb)->pingCb.pingPongDfx = RsPingRoceGetDfx();

    return 0;
}

int RsPingRoceFindTargetNodeStub(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPingTargetInfo** node)
{
    *node = &gPingTargetNode;

    return 0;
}

int RsPingRoceFindTargetNodeStub1(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPingTargetInfo** node)
{
    *node = &gPingTargetNode1;

    return -ENODEV;
}

void TcRsPingHandleInit()
{
    unsigned int chipId = 0;
    int hdcType = 0;
    int ret = 0;

    chipId = 0;
    hdcType = HDC_SERVICE_TYPE_RDMA;
    ret = RsPingHandleInit(chipId, hdcType, WHITE_LIST_ENABLE);
    EXPECT_INT_EQ(ret, 0);

    hdcType = HDC_SERVICE_TYPE_RDMA_V2;
    mocker(RsDev2rscb, 1, -1);
    ret = RsPingHandleInit(chipId, hdcType, WHITE_LIST_ENABLE);
    EXPECT_INT_EQ(ret, -ENODEV);
    mocker_clean();

    mocker_invoke(RsDev2rscb, RsDev2rscb_stub0, 1);
    mocker((stub_fn_t)pthread_create, 1, -1);
    ret = RsPingHandleInit(chipId, hdcType, WHITE_LIST_ENABLE);
    EXPECT_INT_EQ(ret, -ESYSFUNC);
    mocker_clean();
}

void TcRsPingHandleDeinit()
{
    unsigned int chipId = 0;
    int ret = 0;

    chipId = 0;
    mocker(RsDev2rscb, 1, -1);
    ret = RsPingHandleDeinit(chipId);
    EXPECT_INT_EQ(ret, -ENODEV);
    mocker_clean();

    mocker_invoke(RsDev2rscb, RsDev2rscb_stub0, 1);
    ret = RsPingHandleDeinit(chipId);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker_invoke(RsDev2rscb, RsDev2rscb_stub1, 1);
    ret = RsPingHandleDeinit(chipId);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    pthread_mutex_destroy(&gTmpRsCb1.pingCb.pingMutex);
}

void TcRsPingInit()
{
    struct PingInitAttr attr = {0};
    struct PingInitInfo info = {0};
    unsigned int rdevIndex = 0;
    int ret = 0;

    ret = RsPingInit(&attr, &info, NULL);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker(RsGetRsCb, 1, -1);
    ret = RsPingInit(&attr, &info, &rdevIndex);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(rsGetLocalDevIDByHostDevID, 1, 0);
    mocker_invoke(RsDev2rscb, RsDev2rscb_stub0, 1);
    gTmpRsCb0.pingCb.initCnt = 1;
    ret = RsPingInit(&attr, &info, &rdevIndex);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker_invoke(RsGetRsCb, RsGetRsCbStub, 1);
    mocker(rsGetLocalDevIDByHostDevID, 1, -1);
    ret = RsPingInit(&attr, &info, &rdevIndex);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker_invoke(RsGetRsCb, RsGetRsCbStub, 1);
    mocker(rsGetLocalDevIDByHostDevID, 1, 0);
    mocker(RsSetupSharemem, 20, -1);
    ret = RsPingInit(&attr, &info, &rdevIndex);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker_invoke(RsGetRsCb, RsGetRsCbStub, 1);
    mocker(rsGetLocalDevIDByHostDevID, 1, 0);
    mocker(RsSetupSharemem, 20, 0);
    mocker(RsPingRocePingCbInit, 1, -1);
    ret = RsPingInit(&attr, &info, &rdevIndex);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRsPingTargetAdd()
{
    struct PingTargetInfo target = {0};
    struct RaRsDevInfo rdev = {0};
    int ret = 0;

    ret = RsPingTargetAdd(&rdev, NULL);
    EXPECT_INT_EQ(ret, -EINVAL);

    target.payload.size = PING_USER_PAYLOAD_MAX_SIZE + 1;
    ret = RsPingTargetAdd(&rdev, &target);
    EXPECT_INT_EQ(ret, -EINVAL);

    target.payload.size = PING_USER_PAYLOAD_MAX_SIZE;
    mocker(RsGetPingCb, 1, -1);
    ret = RsPingTargetAdd(&rdev, &target);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    gTmpPingCb.taskStatus = RS_PING_TASK_RUNNING;
    mocker_invoke(RsGetPingCb, RsGetPingCbStub, 1);
    ret = RsPingTargetAdd(&rdev, &target);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    gTmpPingCb.taskStatus = RS_PING_TASK_RESET;
    mocker_invoke(RsGetPingCb, RsGetPingCbStub, 1);
    mocker_invoke(RsPingRoceFindTargetNode, RsPingRoceFindTargetNodeStub, 1);
    ret = RsPingTargetAdd(&rdev, &target);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker_invoke(RsGetPingCb, RsGetPingCbStub, 1);
    mocker_invoke(RsPingRoceFindTargetNode, RsPingRoceFindTargetNodeStub1, 1);
    mocker(RsPingRoceAllocTargetNode, 1, -1);
    ret = RsPingTargetAdd(&rdev, &target);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRsPingTaskStart()
{
    struct RsPingTargetInfo tmpTarget = {0};
    struct RaRsDevInfo rdev = {0};
    struct PingTaskAttr attr = {0};
    int ret = 0;

    ret = RsPingTaskStart(&rdev, NULL);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker(RsGetPingCb, 1, -1);
    ret = RsPingTaskStart(&rdev, &attr);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    gTmpPingCb.taskStatus = RS_PING_TASK_RUNNING;
    mocker_invoke(RsGetPingCb, RsGetPingCbStub, 1);
    ret = RsPingTaskStart(&rdev, &attr);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    gTmpPingCb.taskStatus = RS_PING_TASK_RESET;
    attr.packetCnt = 1;
    attr.packetInterval = 1;
    attr.timeoutInterval = 1;
    pthread_mutex_init(&gTmpPingCb.pongQp.recvMrCb.mutex, NULL);
    pthread_mutex_init(&gTmpPingCb.pingQp.recvMrCb.mutex, NULL);
    pthread_mutex_init(&gTmpPingCb.pingMutex, NULL);
    RS_INIT_LIST_HEAD(&gTmpPingCb.pingList);
    RsListAddTail(&tmpTarget.list, &gTmpPingCb.pingList);
    mocker_invoke(RsGetPingCb, RsGetPingCbStub, 1);
    ret = RsPingTaskStart(&rdev, &attr);
    EXPECT_INT_EQ(ret, 0);
    RsListDel(&tmpTarget.list);
    mocker_clean();
    pthread_mutex_destroy(&gTmpPingCb.pingMutex);
    pthread_mutex_destroy(&gTmpPingCb.pingQp.recvMrCb.mutex);
    pthread_mutex_destroy(&gTmpPingCb.pongQp.recvMrCb.mutex);
}

void TcRsPingGetResults()
{
    struct PingTargetCommInfo target = {0};
    struct PingResultInfo result = {0};
    struct RaRsDevInfo rdev = {0};
    unsigned int num = 1;
    int ret = 0;

    ret = RsPingGetResults(NULL, &target, &num, &result);
    EXPECT_INT_EQ(ret, -EINVAL);

    num = 1;
    mocker(RsGetPingCb, 1, -1);
    ret = RsPingGetResults(&rdev, &target, &num, &result);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    num = 1;
    gTmpPingCb.taskStatus = RS_PING_TASK_RUNNING;
    mocker_invoke(RsGetPingCb, RsGetPingCbStub, 1);
    ret = RsPingGetResults(&rdev, &target, &num, &result);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    num = 1;
    gTmpPingCb.taskStatus = RS_PING_TASK_RESET;
    mocker_invoke(RsGetPingCb, RsGetPingCbStub, 1);
    mocker(RsPingRoceGetTargetResult, 1, -1);
    ret = RsPingGetResults(&rdev, &target, &num, &result);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRsPingTaskStop()
{
    struct RaRsDevInfo rdev = {0};
    int ret = 0;

    ret = RsPingTaskStop(NULL);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker(RsGetPingCb, 1, -1);
    ret = RsPingTaskStop(&rdev);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRsPingTargetDel()
{
    struct PingTargetCommInfo target = {0};
    struct RaRsDevInfo rdev = {0};
    unsigned int num = 1;
    int ret = 0;

    ret = RsPingTargetDel(&rdev, &target, NULL);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker(RsGetPingCb, 1, -1);
    ret = RsPingTargetDel(&rdev, &target, &num);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    gTmpPingCb.taskStatus = RS_PING_TASK_RUNNING;
    mocker_invoke(RsGetPingCb, RsGetPingCbStub, 1);
    ret = RsPingTargetDel(&rdev, &target, &num);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    gTmpPingCb.taskStatus = RS_PING_TASK_RESET;
    mocker_invoke(RsGetPingCb, RsGetPingCbStub, 1);
    mocker(RsPingRoceFindTargetNode, 1, -1);
    ret = RsPingTargetDel(&rdev, &target, &num);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRsPingDeinit()
{
    struct RaRsDevInfo rdev = {0};
    int ret = 0;

    ret = RsPingDeinit(NULL);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker(RsGetPingCb, 1, -1);
    ret = RsPingDeinit(&rdev);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    gTmpPingCb.initCnt = 0;
    mocker_invoke(RsGetPingCb, RsGetPingCbStub, 1);
    ret = RsPingDeinit(&rdev);
    EXPECT_INT_EQ(ret, -ENODEV);
    mocker_clean();
}

void TcRsPingUrmaCheckFd()
{
    urma_jfce_t jf = {0};
    struct RsPingCtxCb pingCb = {0};
    pingCb.pingJetty.jfce = &jf;
    pingCb.pingJetty.jfce->fd = 1;
    pingCb.pongJetty.jfce = &jf;
    pingCb.pongJetty.jfce->fd = 1;
    int ret = 0;

    ret = RsPingUrmaCheckFd(&pingCb, 1);
    EXPECT_INT_EQ(ret, 1);

    ret = RsPingUrmaCheckFd(&pingCb, 0);
    EXPECT_INT_EQ(ret, 0);

    ret = RsPongUrmaCheckFd(&pingCb, 1);
    EXPECT_INT_EQ(ret, 1);

    ret = RsPongUrmaCheckFd(&pingCb, 0);
    EXPECT_INT_EQ(ret, 0);
}

void TcRsPingCbGetIbCtxAndIndex()
{
    struct RsPingCtxCb pingCb = {0};
    struct ibv_device* devList[1] = {0};
    struct ibv_device devNode = {0};
    struct rdev rdevInfo = {0};
    int ret = 0;

    pingCb.rdevCb.devNum = 1;
    pingCb.rdevCb.devList = devList;
    devList[0] = &devNode;
    mocker(RsQueryGid, 1, 0);
    mocker(RsPingCbGetDevRdevIndex, 1, 0);
    mocker(RsIbvQueryGid, 1, -1);
    ret = RsPingCbGetIbCtxAndIndex(&rdevInfo, &pingCb);
    EXPECT_INT_EQ(ret, -EOPENSRC);
    mocker_clean();
}
