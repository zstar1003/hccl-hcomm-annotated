/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define _GNU_SOURCE
#define SOCK_CONN_TAG_SIZE 192
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <ifaddrs.h>

#include "ut_dispatch.h"
#include "stub/ibverbs.h"
#include "rs.h"
#include "rs_ub_tp.h"
#include "rs_ub_dfx.h"
#include "rs_ub.h"
#include "rs_ctx.h"
#include "rs_common_inner.h"
#include "rs_inner.h"
#include "rs_ctx_inner.h"
#include "tc_ut_rs_ub.h"
#include "rs_drv_rdma.h"
#include "stub/verbs_exp.h"
#include "encrypt.h"
#include "rs_epoll.h"
#include "rs_socket.h"
#include "ra_rs_err.h"

extern uint32_t RsGenerateUeInfo(uint32_t dieId, uint32_t funcId);
extern uint32_t RsGenerateDevIndex(uint32_t devCnt, uint32_t dieId, uint32_t funcId);
extern uint32_t RsGenerateMmapResId(uint32_t id, uint32_t dieId, uint32_t funcId);
extern int RsUbGetRdevCb(struct rs_cb* rsCb, unsigned int rdevIndex, struct RsUbDevCb** devCb);
extern int RsUrmaDeviceApiInit(void);
extern int RsOpenUrmaSo(void);
extern int RsUrmaApiInit(void);
extern void RsUbCtxExtJettyCreateTaCache(struct RsCtxJettyCb* jettyCb, urma_jetty_cfg_t* jettyCfg);
extern int RsUbDevCbInit(
    struct CtxInitAttr* attr, struct RsUbDevCb* devCb, struct rs_cb* rscb, int* devIndex, struct DevBaseAttr* devAttr);
extern void RsJfcCallbackProcess(struct RsCtxJettyCb* jettyCb, urma_cr_t* cr, urma_jfc_t* jfc);
extern int RsUrmaJettyApiInit(void);
extern int RsUrmaJfcApiInit(void);
extern int RsUrmaSegmentApiInit(void);
extern int RsUrmaDataApiInit(void);
extern urma_device_t** RsUrmaGetDeviceList(int* numDevices);
extern urma_eid_info_t* RsUrmaGetEidList(urma_device_t* dev, uint32_t* cnt);
extern void RsUrmaFreeDeviceList(urma_device_t** deviceList);
extern void RsUrmaFreeEidList(urma_eid_info_t* eidList);
extern int RsUrmaGetEidByIp(const urma_context_t* ctx, const urma_net_addr_t* netAddr, urma_eid_t* eid);
extern int RsUrmaGetIpByEid(const urma_context_t* ctx, const urma_eid_t* eid, urma_net_addr_t* netAddr);
extern void RsUbCtxExtJettyCreate(struct RsCtxJettyCb* jettyCb, urma_jetty_cfg_t* jettyCfg);
extern void RsUbCtxExtJettyDelete(struct RsCtxJettyCb* jettyCb);
extern int RsUbCtxRegJettyDb(struct RsCtxJettyCb* jettyCb, struct udma_u_jetty_info* jettyInfo);
extern int RsInitRscbCfg(struct rs_cb* rscb, struct RsInitConfig* cfg);
extern int RsUbCreateCtx(urma_device_t* urmaDev, unsigned int eidIndex, urma_context_t** urmaCtx);
extern int RsUbGetUeInfo(urma_context_t* urmaCtx, struct DevBaseAttr* devBaseAttr);
extern int RsUbGetDevAttr(struct RsUbDevCb* devCb, struct DevBaseAttr* devAttr, unsigned int* devIndex);
extern int RsUbGetJfcCb(struct RsUbDevCb* devCb, unsigned long long addr, struct RsCtxJfcCb** jfcCb);
extern void RsUbFreeSegCbList(struct RsUbDevCb* devCb, struct RsListHead* lsegList, struct RsListHead* rsegList);
extern void RsUbFreeJettyCbList(struct RsUbDevCb* devCb, struct RsListHead* jettyList, struct RsListHead* rjettyList);
extern void RsUbFreeJfcCbList(struct RsUbDevCb* devCb, struct RsListHead* jfcList);
extern void RsUbFreeJfceCbList(struct RsUbDevCb* devCb, struct RsListHead* jfceList);
extern void RsUbFreeTokenIdCbList(struct RsUbDevCb* devCb, struct RsListHead* tokenIdList);
extern int RsUbGetTokenIdCb(struct RsUbDevCb* devCb, unsigned long long addr, struct RsTokenIdCb** tokenIdCb);
extern int RsUbInitSegCb(struct MemRegAttrT* memAttr, struct RsUbDevCb* devCb, struct RsSegCb* segCb);
extern int RsUbCtxJfcCreateExt(struct RsCtxJfcCb* ctxJfcCb, urma_jfc_cfg_t jfcCfg, urma_jfc_t** jfc);
extern int RsUbCtxInitJettyCb(struct RsUbDevCb* devCb, struct CtxQpAttr* attr, struct RsCtxJettyCb** jettyCb);
extern int RsUbQueryJfcCb(
    struct RsUbDevCb* devCb, unsigned long long scqIndex, unsigned long long rcqIndex, struct RsCtxJfcCb** sendJfcCb,
    struct RsCtxJfcCb** recvJfcCb);
extern void RsUbCtxFreeJettyCb(struct RsCtxJettyCb* jettyCb);
extern int
RsUbCtxDrvJettyCreate(struct RsCtxJettyCb* jettyCb, struct RsCtxJfcCb* sendJfcCb, struct RsCtxJfcCb* recvJfcCb);
extern int RsUbFillJettyInfo(struct RsCtxJettyCb* jettyCb, struct QpCreateInfo* jettyInfo);
extern void RsUbCtxDrvJettyDelete(struct RsCtxJettyCb* jettyCb);
extern int
RsUbCtxInitRjettyCb(struct RsUbDevCb* devCb, struct RsJettyImportAttr* importAttr, struct RsCtxRemJettyCb** rjettyCb);
extern int RsUbCtxDrvJettyImport(struct RsCtxRemJettyCb* rjettyCb);
extern int RsUbGetJettyCb(struct RsUbDevCb* devCb, unsigned int jettyId, struct RsCtxJettyCb** jettyCb);
extern void RsCloseUrmaSo(void);
extern int RsUbDestroyJettyCbBatch(struct JettyDestroyBatchInfo* batchInfo, unsigned int* num);
extern int RsUbGetJettyDestroyBatchInfo(
    struct RsUbDevCb* devCb, unsigned int jettyIds[], struct JettyDestroyBatchInfo* batchInfo, unsigned int* num);
extern int RsUbCallocJettyBatchInfo(struct JettyDestroyBatchInfo* batchInfo, unsigned int num);
extern void RsUbFreeJettyCbBatch(
    struct JettyDestroyBatchInfo* batchInfo, unsigned int* num, urma_jetty_t* badJetty, urma_jfr_t* badJfr);
extern int RsUbCtxJfcCreateNormal(struct RsUbDevCb* devCb, urma_jfc_cfg_t* jfcCfg, urma_jfc_t** outJfc);
extern int RsUbGetJfceCb(struct RsUbDevCb* devCb, unsigned long long addr, struct RsCtxJfceCb** jfceCb);
extern int RsHandleEpollPollJfc(struct RsUbDevCb* devCb, urma_jfce_t* jfce);
extern int RsUbJettyCbBuffAlloc(struct RsUbDevCb* devCb, struct RsCtxJettyCb* jettyCb, enum JfcMode jfcType);

struct RsConnInfo gConn = {0};
char gRevBuf[RS_BUF_SIZE] = {0};
extern struct rs_cb stubRsCb;
extern struct RsUbDevCb stubDevCb;
struct RsCtxJettyCb jettyCbStub = {0};
struct RsCtxJfceCb gJfceCb = {0};

int RsUbGetJfceCbStub(struct RsUbDevCb* devCb, unsigned long long addr, struct RsCtxJfceCb** jfceCb)
{
    *jfceCb = &gJfceCb;
    return 0;
}

int RsUbGetJettyCbStub(struct RsUbDevCb* devCb, unsigned int jettyId, struct RsCtxJettyCb** jettyCb)
{
    *jettyCb = &jettyCbStub;
    return 0;
}

void RsUbFreeJettyCbBatchStub(
    struct JettyDestroyBatchInfo* batchInfo, unsigned int* num, urma_jetty_t* badJetty, urma_jfr_t* badJfr)
{
    return;
}

void TcRsUbGetRdevCb()
{
    struct RsUbDevCb* rdevCbOut = NULL;
    struct RsUbDevCb rdevCb = {0};
    unsigned int rdevIndex = 0;
    struct rs_cb rsCb = {0};
    int ret = 0;

    RS_INIT_LIST_HEAD(&rsCb.udevList);
    RsListAddTail(&rdevCb.list, &rsCb.udevList);

    ret = RsUbGetDevCb(&rsCb, rdevIndex, &rdevCbOut);
    EXPECT_INT_EQ(ret, 0);

    rdevIndex = 1;
    ret = RsUbGetDevCb(&rsCb, rdevIndex, &rdevCbOut);
    EXPECT_INT_EQ(ret, -ENODEV);

    return;
}

void TcRsUrmaApiInitAbnormal()
{
    int ret = 0;

    mocker(RsOpenUrmaSo, 100, 0);
    mocker(RsUrmaDeviceApiInit, 100, -1);
    ret = RsUrmaApiInit();
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RsOpenUrmaSo, 100, 0);
    mocker(RsUrmaDeviceApiInit, 100, 0);
    mocker(RsUrmaJettyApiInit, 100, -1);
    ret = RsUrmaApiInit();
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RsOpenUrmaSo, 100, 0);
    mocker(RsUrmaDeviceApiInit, 100, 0);
    mocker(RsUrmaJettyApiInit, 100, 0);
    mocker(RsUrmaJfcApiInit, 100, -1);
    ret = RsUrmaApiInit();
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RsOpenUrmaSo, 100, 0);
    mocker(RsUrmaDeviceApiInit, 100, 0);
    mocker(RsUrmaJettyApiInit, 100, 0);
    mocker(RsUrmaJfcApiInit, 100, 0);
    mocker(RsUrmaSegmentApiInit, 100, -1);
    ret = RsUrmaApiInit();
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RsOpenUrmaSo, 100, 0);
    mocker(RsUrmaDeviceApiInit, 100, 0);
    mocker(RsUrmaJettyApiInit, 100, 0);
    mocker(RsUrmaJfcApiInit, 100, 0);
    mocker(RsUrmaSegmentApiInit, 100, 0);
    mocker(RsUrmaDataApiInit, 100, -1);
    ret = RsUrmaApiInit();
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    return;
}

void TcRsUbV2()
{
    struct RsUbDevCb* devCb = NULL;
    struct DevBaseAttr attr = {0};
    struct RsInitConfig cfg = {0};
    struct CtxInitAttr info = {0};
    struct rs_cb rscb = {0};
    unsigned long long tokenIdAddr = 0;
    unsigned int tokenIdNum = 0;
    unsigned int devIndex = 0;
    int ret = 0;

    struct MemRegAttrT lmemAttr = {0};
    struct MemRegInfoT lmemInfo = {0};
    struct MemImportAttrT rmemAttr = {0};
    struct MemImportInfoT rmemInfo = {0};
    void* addr = malloc(1);
    lmemAttr.mem.addr = (uintptr_t)addr;
    lmemAttr.mem.size = 1;
    lmemAttr.ub.flags.bs.tokenIdValid = 1;

    cfg.chipId = 0;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    RS_INIT_LIST_HEAD(&rscb.udevList);
    ret = RsUbCtxInit(&rscb, &info, &devIndex, &attr);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbGetDevCb(&rscb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxTokenIdAlloc(devCb, &tokenIdAddr, &tokenIdNum);
    EXPECT_INT_EQ(0, ret);

    lmemAttr.ub.tokenIdAddr = tokenIdAddr;

    ret = RsUbCtxLmemReg(devCb, &lmemAttr, &lmemInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxRmemImport(devCb, &rmemAttr, &rmemInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxRmemUnimport(devCb, rmemInfo.ub.targetSegHandle);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxLmemUnreg(devCb, lmemInfo.ub.targetSegHandle);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxDeinit(devCb);
    EXPECT_INT_EQ(0, ret);

    free(addr);
    addr = NULL;

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
}

urma_device_t tcUrmaDev = {0};
urma_device_t* tcUrmaDeviceList[1] = {&tcUrmaDev};
urma_eid_info_t tcUrmaEidList[1] = {0};
urma_eid_info_t tcUrmaEidList2[2] = {0};

urma_device_t** TcRsUrmaGetDeviceListStub(int* numDevices)
{
    *numDevices = 1;
    return tcUrmaDeviceList;
}

void TcRsUbGetDevEidInfoNum()
{
    int ret = 0;
    unsigned int phyId = 0;
    unsigned int num = 0;

    phyId = RS_MAX_DEV_NUM;
    ret = RsUbGetDevEidInfoNum(phyId, &num);
    EXPECT_INT_EQ(0, ret);

    mocker(RsUrmaGetDeviceList, 10, NULL);
    ret = RsUbGetDevEidInfoNum(phyId, &num);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker_invoke(RsUrmaGetDeviceList, TcRsUrmaGetDeviceListStub, 10);
    mocker(RsUrmaGetEidList, 10, NULL);
    mocker(RsUrmaFreeDeviceList, 10, 0);
    mocker(RsUrmaFreeEidList, 10, 0);
    ret = RsUbGetDevEidInfoNum(phyId, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    mocker_invoke(RsUrmaGetDeviceList, TcRsUrmaGetDeviceListStub, 10);
    mocker(RsUrmaGetEidList, 10, 0);
    mocker(RsUrmaFreeDeviceList, 10, 0);
    mocker(RsUrmaFreeEidList, 10, 0);
    ret = RsUbGetDevEidInfoNum(phyId, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

int RsUbGetDevEidInfoNumStub(unsigned int phyId, unsigned int* num)
{
    *num = 1;
    return 0;
}

urma_eid_info_t* RsUrmaGetEidListStub(urma_device_t* dev, uint32_t* cnt)
{
    *cnt = 1;
    return tcUrmaEidList;
}

urma_eid_info_t* RsUrmaGetEidListStub2(urma_device_t* dev, uint32_t* cnt)
{
    *cnt = 2;
    return tcUrmaEidList2;
}

void TcRsUbGetDevEidInfoList()
{
    int ret = 0;
    unsigned int phyId = 0;
    unsigned int startIndex = 0;
    unsigned int count = 0;
    struct HccpDevEidInfo infoList[1] = {0};

    phyId = 0;
    startIndex = 0;
    count = 1;
    ret = RsUbGetDevEidInfoList(phyId, infoList, startIndex, count);
    EXPECT_INT_EQ(0, ret);

    mocker_invoke(RsUbGetDevEidInfoNum, RsUbGetDevEidInfoNumStub, 10);
    mocker(RsUrmaGetDeviceList, 10, NULL);
    ret = RsUbGetDevEidInfoList(phyId, infoList, startIndex, count);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker_invoke(RsUbGetDevEidInfoNum, RsUbGetDevEidInfoNumStub, 10);
    mocker_invoke(RsUrmaGetDeviceList, TcRsUrmaGetDeviceListStub, 10);
    mocker(RsUrmaGetEidList, 10, NULL);
    mocker(RsUrmaFreeDeviceList, 10, 0);
    ret = RsUbGetDevEidInfoList(phyId, infoList, startIndex, count);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    mocker_invoke(RsUbGetDevEidInfoNum, RsUbGetDevEidInfoNumStub, 10);
    mocker_invoke(RsUrmaGetDeviceList, TcRsUrmaGetDeviceListStub, 10);
    mocker(RsUbCreateCtx, 10, 0);
    mocker(RsUbGetUeInfo, 10, 0);
    mocker_invoke(RsUrmaGetEidList, RsUrmaGetEidListStub2, 10);
    mocker(RsUrmaFreeDeviceList, 10, 0);
    mocker(RsUrmaFreeEidList, 10, 0);
    mocker(RsUrmaDeleteContext, 10, 0);
    ret = RsUbGetDevEidInfoList(phyId, infoList, startIndex, count);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();

    mocker_invoke(RsUbGetDevEidInfoNum, RsUbGetDevEidInfoNumStub, 10);
    mocker_invoke(RsUrmaGetDeviceList, TcRsUrmaGetDeviceListStub, 10);
    mocker(RsUbCreateCtx, 10, -ENODEV);
    mocker_invoke(RsUrmaGetEidList, RsUrmaGetEidListStub, 10);
    mocker(RsUrmaFreeDeviceList, 10, 0);
    mocker(RsUrmaFreeEidList, 10, 0);
    ret = RsUbGetDevEidInfoList(phyId, infoList, startIndex, count);
    EXPECT_INT_EQ(-ENODEV, ret);
    mocker_clean();

    mocker_invoke(RsUbGetDevEidInfoNum, RsUbGetDevEidInfoNumStub, 10);
    mocker_invoke(RsUrmaGetDeviceList, TcRsUrmaGetDeviceListStub, 10);
    mocker(RsUbCreateCtx, 10, 0);
    mocker(RsUbGetUeInfo, 10, -259);
    mocker_invoke(RsUrmaGetEidList, RsUrmaGetEidListStub, 10);
    mocker(RsUrmaFreeDeviceList, 10, 0);
    mocker(RsUrmaFreeEidList, 10, 0);
    mocker(RsUrmaDeleteContext, 10, 0);
    ret = RsUbGetDevEidInfoList(phyId, infoList, startIndex, count);
    EXPECT_INT_EQ(-259, ret);
    mocker_clean();

    startIndex = (UINT_MAX / 2) + 1;
    count = (UINT_MAX / 2);
    mocker_invoke(RsUbGetDevEidInfoNum, RsUbGetDevEidInfoNumStub, 10);
    ret = RsUbGetDevEidInfoList(phyId, infoList, startIndex, count);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();

    startIndex = 0;
    count = 2;
    mocker_invoke(RsUbGetDevEidInfoNum, RsUbGetDevEidInfoNumStub, 10);
    ret = RsUbGetDevEidInfoList(phyId, infoList, startIndex, count);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();
}

struct rs_cb* TcRsUbV2Init(int mode, unsigned int* devIndex)
{
    int ret = 0;
    struct DevBaseAttr attr = {0};
    struct RsInitConfig cfg = {0};
    struct CtxInitAttr info = {0};
    struct rs_cb* rsCb = NULL;
    cfg.hccpMode = mode;

    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    ret = RsGetRsCb(0, &rsCb);
    EXPECT_INT_EQ(ret, 0);
    RS_INIT_LIST_HEAD(&rsCb->udevList);

    ret = RsUbCtxInit(rsCb, &info, devIndex, &attr);
    EXPECT_INT_EQ(0, ret);

    return rsCb;
}

void TcRsUbV2Deinit(struct rs_cb* rsCb, int mode, unsigned int devIndex)
{
    int ret = 0;
    struct RsInitConfig cfg = {0};
    cfg.hccpMode = mode;
    struct RsUbDevCb* devCb = NULL;

    ret = RsUbGetDevCb(rsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxDeinit(devCb);
    EXPECT_INT_EQ(ret, 0);
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsUbCtxTokenIdAlloc()
{
    TcRsUbCtxTokenIdAlloc1();
    TcRsUbCtxTokenIdAlloc2();
    TcRsUbCtxTokenIdAlloc3();
}

void TcRsUbCtxTokenIdAlloc1()
{
    unsigned long long addr = 0;
    unsigned int devIndex = 0;
    unsigned int tokenId = 0;
    struct rs_cb* tcRsCb = NULL;
    struct RsUbDevCb* devCb = NULL;
    int ret = 0;

    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxTokenIdAlloc(devCb, &addr, &tokenId);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxTokenIdFree(devCb, addr);
    EXPECT_INT_EQ(0, ret);

    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);
}

void TcRsUbCtxTokenIdAlloc2()
{
    unsigned long long addr = 0;
    unsigned int devIndex = 0;
    unsigned int tokenId = 0;
    struct rs_cb* tcRsCb = NULL;
    struct RsUbDevCb* devCb = NULL;
    int ret = 0;

    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    mocker(RsUrmaAllocTokenId, 10, NULL);
    ret = RsUbCtxTokenIdAlloc(devCb, &addr, &tokenId);
    EXPECT_INT_NE(0, ret);

    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);
}

void TcRsUbCtxTokenIdAlloc3()
{
    unsigned long long addr = 0;
    unsigned long long addr1 = 0;
    unsigned int devIndex = 0;
    unsigned int tokenId = 0;
    unsigned int tokenId1 = 0;
    struct rs_cb* tcRsCb = NULL;
    struct RsUbDevCb* devCb = NULL;
    int ret = 0;

    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxTokenIdAlloc(devCb, &addr, &tokenId);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxTokenIdAlloc(devCb, &addr1, &tokenId1);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxTokenIdFree(devCb, addr1);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxTokenIdFree(devCb, addr);
    EXPECT_INT_EQ(0, ret);

    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);
}

void TcRsUbCtxJfceCreate()
{
    union DataPlaneCstmFlag dataPlaneFlag = {0};
    struct RsUbDevCb* devCb = NULL;
    unsigned long long addr = 0;
    unsigned int devIndex = 0;
    struct rs_cb* tcRsCb = NULL;
    int fd = 0;
    int ret = 0;

    mocker_clean();
    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    dataPlaneFlag.bs.pollCqCstm = 1;
    ret = RsUbCtxChanCreate(devCb, dataPlaneFlag, &addr, &fd);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxChanDestroy(devCb, addr);
    EXPECT_INT_EQ(0, ret);

    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);
}

void TcRsUbCtxJfcCreate()
{
    int ret = 0;
    unsigned int devIndex = 0;
    struct rs_cb* tcRsCb = NULL;
    struct CtxCqAttr attr = {0};
    struct CtxCqInfo info = {0};
    struct RsUbDevCb* devCb = NULL;

    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);
    attr.ub.mode = JFC_MODE_STARS_POLL;

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJfcCreate(devCb, &attr, &info);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJfcDestroy(devCb, info.addr);
    EXPECT_INT_EQ(0, ret);

    attr.ub.mode = JFC_MODE_CCU_POLL;
    attr.ub.ccuExCfg.valid = 1;

    ret = RsUbCtxJfcCreate(devCb, &attr, &info);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJfcDestroy(devCb, info.addr);
    EXPECT_INT_EQ(0, ret);

    attr.ub.mode = JFC_MODE_MAX;
    ret = RsUbCtxJfcCreate(devCb, &attr, &info);
    EXPECT_INT_EQ(-EINVAL, ret);

    mocker_clean();
    attr.ub.mode = JFC_MODE_NORMAL;
    mocker(RsUbCtxJfcCreateNormal, 1, 0);
    ret = RsUbCtxJfcCreate(devCb, &attr, &info);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    mocker(RsUbCtxJfcCreateNormal, 1, -1);
    ret = RsUbCtxJfcCreate(devCb, &attr, &info);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);
}

void TcRsUbCtxJfcCreateNormal()
{
    struct RsUbDevCb devCb = {0};
    urma_jfc_cfg_t jfcCfg = {0};
    urma_jfc_t* outJfc = NULL;
    urma_jfce_t jfce = {0};
    int ret = 0;

    gJfceCb.jfceAddr = 1;
    jfcCfg.jfce = (urma_jfce_t*)(uintptr_t)gJfceCb.jfceAddr;
    mocker_clean();
    mocker(RsUbGetJfceCb, 1, -1);
    ret = RsUbCtxJfcCreateNormal(&devCb, &jfcCfg, &outJfc);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker_invoke(RsUbGetJfceCb, RsUbGetJfceCbStub, 1);
    mocker(RsUrmaCreateJfc, 1, NULL);
    ret = RsUbCtxJfcCreateNormal(&devCb, &jfcCfg, &outJfc);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();

    mocker_invoke(RsUbGetJfceCb, RsUbGetJfceCbStub, 1);
    mocker(RsUrmaRearmJfc, 1, -1);
    ret = RsUbCtxJfcCreateNormal(&devCb, &jfcCfg, &outJfc);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    RsUrmaDeleteJfc(outJfc);
    mocker_clean();

    mocker_invoke(RsUbGetJfceCb, RsUbGetJfceCbStub, 1);
    ret = RsUbCtxJfcCreateNormal(&devCb, &jfcCfg, &outJfc);
    EXPECT_INT_EQ(0, ret);
    RsUrmaDeleteJfc(outJfc);
    mocker_clean();
}

void TcRsUbCtxJettyCreate()
{
    int ret = 0;
    unsigned int devIndex = 0;
    struct rs_cb* tcRsCb = NULL;
    struct CtxQpAttr qpAttr = {0};
    struct QpCreateInfo qpInfo = {0};
    struct CtxCqAttr cqAttr = {0};
    struct CtxCqInfo cqInfo = {0};
    struct RsUbDevCb* devCb = NULL;
    unsigned long long tokenIdAddr = 0;
    unsigned int tokenIdNum = 0;

    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);
    cqAttr.ub.mode = JFC_MODE_STARS_POLL;

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJfcCreate(devCb, &cqAttr, &cqInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyDestroy(devCb, qpInfo.ub.id);
    EXPECT_INT_EQ(0, ret);

    qpAttr.ub.mode = JETTY_MODE_CCU;
    ret = RsUbCtxTokenIdAlloc(devCb, &tokenIdAddr, &tokenIdNum);
    EXPECT_INT_EQ(0, ret);
    qpAttr.ub.tokenIdAddr = tokenIdAddr;
    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_NE(0, ret);

    ret = RsUbCtxJettyDestroy(devCb, qpInfo.ub.id);
    EXPECT_INT_NE(0, ret);

    ret = RsUbCtxJfcDestroy(devCb, cqInfo.addr);
    EXPECT_INT_EQ(0, ret);

    cqAttr.ub.mode = JFC_MODE_CCU_POLL;
    cqAttr.ub.ccuExCfg.valid = 1;
    ret = RsUbCtxJfcCreate(devCb, &cqAttr, &cqInfo);
    EXPECT_INT_EQ(0, ret);

    qpAttr.ub.mode = JETTY_MODE_CCU_TA_CACHE;
    ret = RsUbCtxTokenIdAlloc(devCb, &tokenIdAddr, &tokenIdNum);
    EXPECT_INT_EQ(0, ret);
    qpAttr.ub.tokenIdAddr = tokenIdAddr;
    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_NE(0, ret);

    ret = RsUbCtxJettyDestroy(devCb, qpInfo.ub.id);
    EXPECT_INT_NE(0, ret);

    ret = RsUbCtxJfcDestroy(devCb, cqInfo.addr);
    EXPECT_INT_EQ(0, ret);

    qpAttr.ub.mode = JETTY_MODE_MAX;
    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_EQ(-EINVAL, ret);

    qpAttr.ub.mode = JETTY_MODE_CCU_TA_CACHE;
    qpAttr.ub.taCacheMode.lockFlag = 0;
    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_EQ(-EINVAL, ret);

    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);
}

void TcRsUbCtxJettyImport()
{
    int ret = 0;
    unsigned int devIndex = 0;
    struct rs_cb* tcRsCb = NULL;
    struct RsUbDevCb* devCb = NULL;
    struct CtxQpAttr qpAttr = {0};
    struct QpCreateInfo qpInfo = {0};
    struct CtxCqAttr cqAttr = {0};
    struct CtxCqInfo cqInfo = {0};
    struct RsJettyImportAttr importAttr = {0};
    struct RsJettyImportInfo importData = {0};

    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);
    cqAttr.ub.mode = JFC_MODE_STARS_POLL;

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJfcCreate(devCb, &cqAttr, &cqInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyImport(devCb, &importAttr, &importData);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyUnimport(devCb, importAttr.key.value, REM_JETTY_ID_DEPRECATED_SIZE);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyDestroy(devCb, qpInfo.ub.id);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJfcDestroy(devCb, cqInfo.addr);
    EXPECT_INT_EQ(0, ret);

    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);
}

void TcRsUbCtxJettyBind()
{
    int ret = 0;
    unsigned int devIndex = 0;
    struct rs_cb* tcRsCb = NULL;
    struct RsUbDevCb* devCb = NULL;
    struct CtxQpAttr qpAttr = {0};
    struct QpCreateInfo qpInfo = {0};
    struct CtxCqAttr cqAttr = {0};
    struct CtxCqInfo cqInfo = {0};
    struct RsJettyImportAttr importAttr = {0};
    struct RsJettyImportInfo importData = {0};
    struct RsCtxQpInfo rsQpInfo = {0};

    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);
    cqAttr.ub.mode = JFC_MODE_STARS_POLL;

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJfcCreate(devCb, &cqAttr, &cqInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyUnbind(devCb, rsQpInfo.id);
    EXPECT_INT_NE(0, ret);

    ret = RsUbCtxJettyImport(devCb, &importAttr, &importData);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyBind(devCb, &rsQpInfo, &rsQpInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyBind(devCb, &rsQpInfo, &rsQpInfo);
    EXPECT_INT_NE(0, ret);

    ret = RsUbCtxJettyDestroy(devCb, qpInfo.ub.id);
    EXPECT_INT_NE(0, ret);

    ret = RsUbCtxJettyUnbind(devCb, rsQpInfo.id);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyUnimport(devCb, importAttr.key.value, REM_JETTY_ID_DEPRECATED_SIZE);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyDestroy(devCb, qpInfo.ub.id);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyImport(devCb, &importAttr, &importData);
    EXPECT_INT_EQ(0, ret);

    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);

    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);
    cqAttr.ub.mode = 10000;

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJfcCreate(devCb, &cqAttr, &cqInfo);
    EXPECT_INT_NE(0, ret);

    cqAttr.ub.mode = JFC_MODE_STARS_POLL;
    ret = RsUbCtxJfcCreate(devCb, &cqAttr, &cqInfo);
    EXPECT_INT_EQ(0, ret);

    qpAttr.ub.mode = JFC_MODE_STARS_POLL;
    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_NE(0, ret);

    ret = RsUbCtxJettyImport(devCb, &importAttr, &importData);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyUnimport(devCb, importAttr.key.value, REM_JETTY_ID_DEPRECATED_SIZE);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJfcDestroy(devCb, cqInfo.addr);
    EXPECT_INT_EQ(0, ret);

    struct RsCtxJettyCb jettyCb = {0};
    urma_jetty_t jetty = {0};
    jettyCb.jetty = &jetty;
    jettyCb.devCb = devCb;
    RsUbCtxExtJettyDelete(&jettyCb);

    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);
}

void TcRsUbCtxBatchSendWr()
{
    int ret = 0;
    unsigned int devIndex = 0;
    struct rs_cb* tcRsCb = NULL;
    struct CtxQpAttr qpAttr = {0};
    struct QpCreateInfo qpInfo = {0};
    struct CtxCqAttr cqAttr = {0};
    struct CtxCqInfo cqInfo = {0};
    struct RsJettyImportAttr importAttr = {0};
    struct RsJettyImportInfo importData = {0};
    struct RsCtxQpInfo rsQpInfo = {0};
    struct WrlistBaseInfo baseInfo = {0};
    struct BatchSendWrData wrDataNop[1] = {0};
    struct BatchSendWrData wrData[1] = {0};
    struct SendWrResp wrResp[1] = {0};
    struct WrlistSendCompleteNum wrlistNum = {0};
    struct MemRegAttrT memRegAttr = {0};
    struct MemRegInfoT memRegInfo = {0};
    struct MemImportAttrT memImportAttr = {0};
    struct MemImportInfoT memImportInfo = {0};
    urma_token_id_t* tokenId = NULL;
    unsigned long long tokenIdAddr = 0;
    unsigned int tokenIdNum = 0;
    unsigned int completeNum = 0;
    struct RsUbDevCb* devCb = NULL;

    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);
    cqAttr.ub.mode = JFC_MODE_STARS_POLL;
    wrlistNum.sendNum = 1;
    wrlistNum.completeNum = &completeNum;
    void* addr = malloc(1);
    memRegAttr.mem.addr = (uintptr_t)addr;
    memRegAttr.mem.size = 1;
    memRegAttr.ub.flags.bs.tokenIdValid = 1;

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxTokenIdAlloc(devCb, &tokenIdAddr, &tokenIdNum);
    EXPECT_INT_EQ(0, ret);

    memRegAttr.ub.tokenIdAddr = tokenIdAddr;

    ret = RsUbCtxJfcCreate(devCb, &cqAttr, &cqInfo);
    EXPECT_INT_EQ(0, ret);

    qpAttr.ub.mode = JETTY_MODE_CCU;
    qpAttr.ub.tokenIdAddr = tokenIdAddr;
    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_NE(0, ret);

    qpAttr.ub.mode = 0;
    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyImport(devCb, &importAttr, &importData);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyBind(devCb, &rsQpInfo, &rsQpInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxLmemReg(devCb, &memRegAttr, &memRegInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxRmemImport(devCb, &memImportAttr, &memImportInfo);
    EXPECT_INT_EQ(0, ret);

    wrData[0].devRmemHandle = (uint64_t)(uintptr_t)memImportInfo.ub.targetSegHandle;
    wrData[0].numSge = 1;
    wrData[0].sges[0].addr = (uint64_t)(uintptr_t)addr;
    wrData[0].sges[0].len = 1;
    wrData[0].sges[0].devLmemHandle = (uint64_t)(uintptr_t)memRegInfo.ub.targetSegHandle;

    baseInfo.devIndex = devIndex;
    ret = RsUbCtxBatchSendWr(tcRsCb, &baseInfo, wrData, wrResp, &wrlistNum);
    EXPECT_INT_EQ(0, ret);

    wrData[0].ub.remJetty = 0xfffff;
    ret = RsUbCtxBatchSendWr(tcRsCb, &baseInfo, wrData, wrResp, &wrlistNum);
    EXPECT_INT_NE(0, ret);

    wrDataNop[0].ub.opcode = RA_UB_OPC_NOP;
    ret = RsUbCtxBatchSendWr(tcRsCb, &baseInfo, wrDataNop, wrResp, &wrlistNum);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxRmemUnimport(devCb, memImportInfo.ub.targetSegHandle);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyUpdateCi(devCb, 10000, 0);
    EXPECT_INT_NE(0, ret);

    ret = RsUbCtxJettyUpdateCi(devCb, 0, 0);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxLmemUnreg(devCb, memRegInfo.ub.targetSegHandle);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyUnbind(devCb, rsQpInfo.id);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyUnimport(devCb, importAttr.key.value, REM_JETTY_ID_DEPRECATED_SIZE);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyDestroy(devCb, qpInfo.ub.id);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJfcDestroy(devCb, cqInfo.addr);
    EXPECT_INT_EQ(0, ret);

    free(addr);
    addr = NULL;
    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);
}

void TcRsUbFreeCbList()
{
    int ret = 0;
    unsigned int devIndex = 0;
    struct rs_cb* tcRsCb = NULL;
    struct CtxQpAttr qpAttr = {0};
    struct QpCreateInfo qpInfo = {0};
    struct CtxCqAttr cqAttr = {0};
    struct CtxCqInfo cqInfo = {0};
    struct RsJettyImportAttr importAttr = {0};
    struct RsJettyImportInfo importData = {0};
    struct RsCtxQpInfo rsQpInfo = {0};
    struct MemRegAttrT memRegAttr = {0};
    struct MemRegInfoT memRegInfo = {0};
    struct MemImportAttrT memImportAttr = {0};
    struct MemImportInfoT memImportInfo = {0};
    unsigned long long jfceAddr = 0;
    unsigned long long tokenIdAddr = 0;
    unsigned int tokenId = 0;
    unsigned int completeNum = 0;
    struct RsUbDevCb* devCb = NULL;
    union DataPlaneCstmFlag dataPlaneFlag = {0};
    int fd = 0;

    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);
    cqAttr.ub.mode = JFC_MODE_STARS_POLL;
    void* addr = malloc(1);
    memRegAttr.mem.addr = (uintptr_t)addr;
    memRegAttr.mem.size = 1;
    memRegAttr.ub.flags.bs.tokenIdValid = 1;

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxTokenIdAlloc(devCb, &tokenIdAddr, &tokenId);
    EXPECT_INT_EQ(0, ret);

    memRegAttr.ub.tokenIdAddr = tokenIdAddr;

    ret = RsUbCtxJfcCreate(devCb, &cqAttr, &cqInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyImport(devCb, &importAttr, &importData);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyBind(devCb, &rsQpInfo, &rsQpInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxLmemReg(devCb, &memRegAttr, &memRegInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxRmemImport(devCb, &memImportAttr, &memImportInfo);
    EXPECT_INT_EQ(0, ret);

    dataPlaneFlag.bs.pollCqCstm = 1;
    ret = RsUbCtxChanCreate(devCb, dataPlaneFlag, &jfceAddr, &fd);
    EXPECT_INT_EQ(0, ret);

    free(addr);
    addr = NULL;
    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);
}

void TcRsUbCtxExtJettyCreate()
{
    struct RsCtxJettyCb jettyCb = {0};
    struct RsUbDevCb devCb = {0};
    urma_jetty_cfg_t jettyCfg = {0};
    struct rs_cb rscb = {0};

    devCb.rscb = &rscb;
    jettyCb.devCb = &devCb;
    jettyCb.jettyMode = JETTY_MODE_USER_CTL_NORMAL;
    RsUbCtxExtJettyCreate(&jettyCb, &jettyCfg);
    RsUbCtxExtJettyDelete(&jettyCb);
    mocker_clean();
}

void TcRsUbCtxRmemImport()
{
    struct MemImportAttrT rmemAttr = {0};
    struct MemImportInfoT rmemInfo = {0};
    struct RsUbDevCb devCb = {0};
    int ret = 0;

    mocker(RsUbGetDevCb, 2, 0);
    mocker(memcpy_s, 2, -1);
    ret = RsUbCtxRmemImport(&devCb, &rmemAttr, &rmemInfo);
    EXPECT_INT_EQ(-ESAFEFUNC, ret);
    mocker_clean();
}

void TcRsUbCtxDrvJettyImport()
{
    struct RsCtxRemJettyCb rjettyCb = {0};
    struct RsUbDevCb devCb = {0};
    urma_context_t urmaCtx = {0};
    int ret = 0;

    rjettyCb.mode = JETTY_IMPORT_MODE_EXP;
    devCb.urmaCtx = &urmaCtx;
    rjettyCb.devCb = &devCb;
    ret = RsUbCtxDrvJettyImport(&rjettyCb);
    EXPECT_INT_EQ(0, ret);

    free(rjettyCb.tjetty);
    rjettyCb.tjetty = NULL;
}

void TcRsUbDevCbInit()
{
    struct DevBaseAttr baseAttr = {0};
    struct RsUbDevCb devCb = {0};
    struct CtxInitAttr attr = {0};
    struct rs_cb rscb = {0};
    int devIndex = 0;
    int ret = 0;

    mocker(pthread_mutex_init, 1, 0);
    mocker(RsUbCreateCtx, 1, -1);
    mocker(pthread_mutex_destroy, 1, -1);
    ret = RsUbDevCbInit(&attr, &devCb, &rscb, &devIndex, &baseAttr);
    EXPECT_INT_NE(0, ret);

    mocker_clean();
    mocker(pthread_mutex_init, 1, 0);
    mocker(RsUbCreateCtx, 1, 0);
    mocker(RsUbGetDevAttr, 1, -1);
    mocker(RsUrmaDeleteContext, 1, -1);
    mocker(pthread_mutex_destroy, 1, -1);
    ret = RsUbDevCbInit(&attr, &devCb, &rscb, &devIndex, &baseAttr);
    EXPECT_INT_NE(0, ret);

    mocker_clean();
}

void TcRsUbCtxInit()
{
    struct DevBaseAttr baseAttr = {0};
    struct CtxInitAttr attr = {0};
    struct rs_cb rsCb = {0};
    int devIndex = 0;
    int ret = 0;

    mocker(RsUrmaGetDeviceByEid, 1, NULL);
    ret = RsUbCtxInit(&rsCb, &attr, &devIndex, &baseAttr);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbCtxJfcDestroy()
{
    struct RsUbDevCb devCb = {0};
    unsigned long long addr = 0;
    int ret = 0;

    mocker(RsUbGetJfcCb, 1, -1);
    ret = RsUbCtxJfcDestroy(&devCb, addr);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbCtxExtJettyDelete()
{
    struct RsCtxJettyCb jettyCb = {0};
    struct RsUbDevCb devCb = {0};
    urma_jetty_t* jetty = (urma_jetty_t*)calloc(1, sizeof(urma_jetty_t));
    jettyCb.jetty = jetty;

    jettyCb.devCb = &devCb;
    mocker(RsUrmaUserCtl, 1, -1);
    RsUbCtxExtJettyDelete(&jettyCb);
    mocker_clean();
}

void TcRsUbCtxChanCreate()
{
    union DataPlaneCstmFlag dataPlaneFlag = {0};
    struct RsUbDevCb devCb = {0};
    unsigned long long addr = 0;
    struct rs_cb rsCb = {0};
    int ret = 0;
    int fd = 0;

    devCb.rscb = &rsCb;
    dataPlaneFlag.bs.pollCqCstm = 1;
    mocker(RsUrmaCreateJfce, 1, NULL);
    ret = RsUbCtxChanCreate(&devCb, dataPlaneFlag, &addr, &fd);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    dataPlaneFlag.bs.pollCqCstm = 0;
    ret = RsUbCtxChanCreate(&devCb, dataPlaneFlag, &addr, &fd);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();
}

void TcRsUbCtxDeinit()
{
    struct RsUbDevCb* devCb = (struct RsUbDevCb*)calloc(1, sizeof(struct RsUbDevCb));
    struct rs_cb rsCb = {0};
    int ret = 0;

    devCb->rscb = &rsCb;
    mocker(RsUrmaDeleteContext, 1, -1);
    mocker(RsUbFreeSegCbList, 1, -1);
    mocker(RsUbFreeJettyCbList, 1, -1);
    mocker(RsUbFreeJfcCbList, 1, -1);
    mocker(RsUbFreeJfceCbList, 1, -1);
    mocker(RsUbFreeTokenIdCbList, 1, -1);
    RS_INIT_LIST_HEAD(&devCb->list);
    ret = RsUbCtxDeinit(devCb);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsUbInitSegCb()
{
    struct MemRegAttrT memAttr = {0};
    struct RsUbDevCb devCb = {0};
    struct RsSegCb segCb = {0};
    int ret = 0;

    mocker(RsUbGetTokenIdCb, 1, 0);
    mocker(RsUrmaRegisterSeg, 1, NULL);
    ret = RsUbInitSegCb(&memAttr, &devCb, &segCb);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbCtxLmemReg()
{
    struct MemRegAttrT memAttr = {0};
    struct MemRegInfoT memInfo = {0};
    struct RsUbDevCb devCb = {0};
    int ret = 0;

    memAttr.mem.size = 1;
    mocker(RsUbInitSegCb, 1, -1);
    ret = RsUbCtxLmemReg(&devCb, &memAttr, &memInfo);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(RsUrmaImportSeg, 1, NULL);
    ret = RsUbCtxRmemImport(&devCb, (struct MemImportAttrT*)&memAttr, (struct MemImportInfoT*)&memInfo);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbCtxJfcCreateFail()
{
    struct RsUbDevCb devCb = {0};
    struct CtxCqAttr attr = {0};
    struct CtxCqInfo info = {0};
    struct rs_cb rsCb = {0};
    int ret = 0;

    devCb.rscb = &rsCb;
    attr.ub.mode = JFC_MODE_STARS_POLL;
    mocker(RsUbCtxJfcCreateExt, 1, -1);
    ret = RsUbCtxJfcCreate(&devCb, &attr, &info);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbCtxInitJettyCb()
{
    struct RsCtxJettyCb* jettyCb = NULL;
    struct RsUbDevCb devCb = {0};
    struct CtxQpAttr attr = {0};
    int ret = 0;

    mocker(pthread_mutex_init, 1, -1);
    ret = RsUbCtxInitJettyCb(&devCb, &attr, &jettyCb);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

static struct RsCtxJfcCb gSendJfcCb = {0};

int RsUbQueryJfcCbStub(
    struct RsUbDevCb* devCb, unsigned long long scqIndex, unsigned long long rcqIndex, struct RsCtxJfcCb** sendJfcCb,
    struct RsCtxJfcCb** recvJfcCb)
{
    gSendJfcCb.jfcType = JFC_MODE_NORMAL;
    *sendJfcCb = &gSendJfcCb;
    *recvJfcCb = &gSendJfcCb;
    return 0;
}

void TcRsUbCtxJettyCreateFail()
{
    struct RsUbDevCb devCb = {0};
    struct QpCreateInfo info = {0};
    struct CtxQpAttr attr = {0};
    int ret = 0;

    mocker(RsUbCtxInitJettyCb, 1, 0);
    mocker(RsUbQueryJfcCb, 1, -1);
    mocker(RsUbCtxFreeJettyCb, 1, -1);
    ret = RsUbCtxJettyCreate(&devCb, &attr, &info);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(RsUbCtxInitJettyCb, 1, 0);
    mocker_invoke_p5(
        "RsUbQueryJfcCb", "RsUbQueryJfcCbStub", (stub_fn_t)RsUbQueryJfcCb, (stub_fn_t)RsUbQueryJfcCbStub, 10);
    mocker(RsUbCtxDrvJettyCreate, 1, 0);
    mocker(RsUbFillJettyInfo, 1, -1);
    mocker(RsUbCtxDrvJettyDelete, 1, -1);
    mocker(RsUbCtxFreeJettyCb, 1, -1);
    mocker(RsUbJettyCbBuffAlloc, 10, 0);
    ret = RsUbCtxJettyCreate(&devCb, &attr, &info);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbCtxJettyImportFail()
{
    struct RsJettyImportAttr importAttr = {0};
    struct RsJettyImportInfo importInfo = {0};
    struct RsUbDevCb devCb = {0};
    int ret = 0;

    mocker(RsUbCtxInitRjettyCb, 1, 0);
    mocker(RsUbCtxDrvJettyImport, 1, -1);
    ret = RsUbCtxJettyImport(&devCb, &importAttr, &importInfo);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbCtxBatchSendWrFail()
{
    struct WrlistSendCompleteNum wrlistNum = {0};
    struct WrlistBaseInfo baseInfo = {0};
    struct BatchSendWrData wrData = {0};
    struct SendWrResp wrResp = {0};
    struct rs_cb rsCb = {0};
    int ret = 0;

    wrlistNum.sendNum = 1;
    mocker(RsUbGetDevCb, 1, -1);
    ret = RsUbCtxBatchSendWr(&rsCb, &baseInfo, &wrData, &wrResp, &wrlistNum);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbGetJettyCb, 1, -1);
    ret = RsUbCtxBatchSendWr(&rsCb, &baseInfo, &wrData, &wrResp, &wrlistNum);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbCtxJettyDestroyBatch()
{
    struct JettyDestroyBatchInfo batchInfo = {0};
    struct RsUbDevCb devCb = {0};
    urma_device_t urmaDev = {0};
    unsigned int jettyIds[1] = {0};
    unsigned int num = 0;
    int ret = 0;
    devCb.urmaDev = &urmaDev;

    ret = RsUbCtxJettyDestroyBatch(&devCb, jettyIds, &num);
    EXPECT_INT_EQ(-EINVAL, ret);

    num = 1;
    mocker(RsUbCallocJettyBatchInfo, 1, -1);
    ret = RsUbCtxJettyDestroyBatch(&devCb, jettyIds, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    num = 1;
    mocker(RsUbGetJettyDestroyBatchInfo, 1, -1);
    ret = RsUbCtxJettyDestroyBatch(&devCb, jettyIds, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    num = 1;
    mocker(RsUbGetJettyDestroyBatchInfo, 1, 0);
    mocker(RsUbDestroyJettyCbBatch, 1, -1);
    ret = RsUbCtxJettyDestroyBatch(&devCb, jettyIds, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RsUbGetJettyCb, 1, -1);
    ret = RsUbGetJettyDestroyBatchInfo(&devCb, jettyIds, &batchInfo, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    jettyCbStub.state = RS_JETTY_STATE_INIT;
    mocker_invoke(RsUbGetJettyCb, RsUbGetJettyCbStub, 1);
    mocker(RsUbGetJettyCb, 1, 0);
    ret = RsUbCtxJettyDestroyBatch(&devCb, jettyIds, &num);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();

    num = 1;
    mocker(RsUbGetJettyDestroyBatchInfo, 1, 0);
    mocker(RsUbDestroyJettyCbBatch, 1, 0);
    ret = RsUbCtxJettyDestroyBatch(&devCb, jettyIds, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    num = 1;
    jettyCbStub.state = RS_JETTY_STATE_CREATED;
    pthread_mutex_init(&devCb.mutex, NULL);
    RS_INIT_LIST_HEAD(&devCb.jettyList);
    RsListAddTail(&jettyCbStub.list, &devCb.jettyList);
    devCb.jettyCnt++;
    mocker_invoke(RsUbGetJettyCb, RsUbGetJettyCbStub, 1);
    mocker(RsUbDestroyJettyCbBatch, 1, 0);
    mocker(RsUbGetJettyCb, 1, 0);
    ret = RsUbCtxJettyDestroyBatch(&devCb, jettyIds, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    num = 1;
    jettyCbStub.state = RS_JETTY_STATE_CREATED;
    jettyCbStub.devCb = &devCb;
    RsListAddTail(&jettyCbStub.list, &devCb.jettyList);
    devCb.jettyCnt++;
    mocker_invoke(RsUbGetJettyCb, RsUbGetJettyCbStub, 1);
    mocker_invoke(RsUbFreeJettyCbBatch, RsUbFreeJettyCbBatchStub, 1);
    mocker(RsUrmaDeleteJettyBatch, 1, -1);
    mocker(RsUbGetJettyCb, 1, 0);
    ret = RsUbCtxJettyDestroyBatch(&devCb, jettyIds, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    num = 1;
    jettyCbStub.state = RS_JETTY_STATE_CREATED;
    jettyCbStub.devCb = &devCb;
    RsListAddTail(&jettyCbStub.list, &devCb.jettyList);
    devCb.jettyCnt++;
    mocker_invoke(RsUbGetJettyCb, RsUbGetJettyCbStub, 1);
    mocker_invoke(RsUbFreeJettyCbBatch, RsUbFreeJettyCbBatchStub, 1);
    mocker(RsUrmaDeleteJettyBatch, 1, 0);
    mocker(RsUbGetJettyCb, 1, 0);
    ret = RsUbCtxJettyDestroyBatch(&devCb, jettyIds, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    pthread_mutex_destroy(&devCb.mutex);
}

void TcRsUbCtxQueryJettyBatch()
{
    struct RsUbDevCb devCb = {0};
    unsigned int jettyIds[] = {1, 2, 3};
    struct JettyAttr attr[3] = {{0}};
    unsigned int num = 3;
    int ret = 0;

    mocker(RsUbGetJettyCb, 1, -1);
    ret = RsUbCtxQueryJettyBatch(&devCb, jettyIds, attr, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    num = 3;
    mocker_invoke(RsUbGetJettyCb, RsUbGetJettyCbStub, 1);
    mocker(RsUrmaQueryJetty, 1, -1);
    ret = RsUbCtxQueryJettyBatch(&devCb, jettyIds, attr, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    num = 3;
    mocker_invoke(RsUbGetJettyCb, RsUbGetJettyCbStub, 3);
    ret = RsUbCtxQueryJettyBatch(&devCb, jettyIds, attr, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsGetEidByIp()
{
    struct RaRsDevInfo devInfo = {0};
    union HccpEid eid[32] = {0};
    struct IpInfo ip[32] = {0};
    unsigned int num = 32;
    int ret = 0;

    mocker(RsGetRsCb, 1, -1);
    ret = RsGetEidByIp(&devInfo, ip, eid, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RsGetRsCb, 1, 0);
    mocker(RsUbGetDevCb, 1, -1);
    ret = RsGetEidByIp(&devInfo, ip, eid, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RsGetRsCb, 1, 0);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbGetEidByIp, 1, -1);
    ret = RsGetEidByIp(&devInfo, ip, eid, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RsGetRsCb, 1, 0);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbGetEidByIp, 1, 0);
    ret = RsGetEidByIp(&devInfo, ip, eid, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsUbGetEidByIp()
{
    struct RsUbDevCb devCb = {0};
    union HccpEid eid[32] = {0};
    struct IpInfo ip[32] = {0};
    unsigned int num = 32;
    int ret = 0;
    int i = 0;

    for (i = 0; i < 32; i++) {
        ip[i].family = AF_INET;
    }
    ret = RsUbGetEidByIp(&devCb, ip, eid, &num);
    EXPECT_INT_EQ(0, ret);

    mocker(RsUrmaGetEidByIp, 1, -1);
    for (i = 0; i < 32; i++) {
        ip[i].family = AF_INET6;
    }
    ret = RsUbGetEidByIp(&devCb, ip, eid, &num);
    EXPECT_INT_EQ(-259, ret);
    mocker_clean();
}

void TcRsGetIpByEid()
{
    struct RaRsDevInfo devInfo = {0};
    union HccpEid eid[32] = {0};
    struct IpInfo ip[32] = {0};
    unsigned int num = 32;
    int ret = 0;

    mocker(RsGetRsCb, 1, -1);
    ret = RsGetIpByEid(&devInfo, eid, ip, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RsGetRsCb, 1, 0);
    mocker(RsUbGetDevCb, 1, -1);
    ret = RsGetIpByEid(&devInfo, eid, ip, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RsGetRsCb, 1, 0);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbGetIpByEid, 1, -1);
    ret = RsGetIpByEid(&devInfo, eid, ip, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RsGetRsCb, 1, 0);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbGetIpByEid, 1, 0);
    ret = RsGetIpByEid(&devInfo, eid, ip, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsUbGetIpByEid()
{
    struct RsUbDevCb devCb = {0};
    union HccpEid eid[32] = {0};
    struct IpInfo ip[32] = {0};
    unsigned int num = 32;
    int ret = 0;

    ret = RsUbGetIpByEid(&devCb, eid, ip, &num);
    EXPECT_INT_EQ(0, ret);

    mocker(RsUrmaGetIpByEid, 1, -1);
    ret = RsUbGetIpByEid(&devCb, eid, ip, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();
}

void TcRsUbCtxGetAuxInfo()
{
    struct HccpAuxInfoOut infoOut = {0};
    struct HccpAuxInfoIn infoIn = {0};
    struct RsUbDevCb devCb = {0};
    int ret = 0;

    mocker_clean();
    infoIn.type = AUX_INFO_IN_TYPE_CQE;
    mocker(RsUrmaUserCtl, 1, 0);
    ret = RsUbCtxGetAuxInfo(&devCb, &infoIn, &infoOut);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    mocker(RsUrmaUserCtl, 1, -1);
    ret = RsUbCtxGetAuxInfo(&devCb, &infoIn, &infoOut);
    EXPECT_INT_EQ(-259, ret);
    mocker_clean();

    infoIn.type = AUX_INFO_IN_TYPE_AE;
    mocker(RsUrmaUserCtl, 1, 0);
    ret = RsUbCtxGetAuxInfo(&devCb, &infoIn, &infoOut);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    mocker(RsUrmaUserCtl, 1, -1);
    ret = RsUbCtxGetAuxInfo(&devCb, &infoIn, &infoOut);
    EXPECT_INT_EQ(-259, ret);
    mocker_clean();

    infoIn.type = AUX_INFO_IN_TYPE_MAX;
    ret = RsUbCtxGetAuxInfo(&devCb, &infoIn, &infoOut);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();
}

void TcRsUbGetTpAttr()
{
    struct RsUbDevCb devCb = {0};
    unsigned int attrBitmap = 0b101010;
    uint64_t tpHandle = 12345;
    struct TpAttr attr = {0};
    int ret = 0;

    mocker(RsUrmaGetTpAttr, 1, -1);
    ret = RsUbGetTpAttr(&devCb, &attrBitmap, tpHandle, &attr);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    ret = RsUbGetTpAttr(&devCb, &attrBitmap, tpHandle, &attr);
    EXPECT_INT_EQ(0, ret);
}

void TcRsUbSetTpAttr()
{
    struct RsUbDevCb devCb = {0};
    unsigned int attrBitmap = 0b101010;
    uint64_t tpHandle = 12345;
    struct TpAttr attr = {0};
    int ret = 0;

    mocker(RsUrmaSetTpAttr, 1, -1);
    ret = RsUbSetTpAttr(&devCb, attrBitmap, tpHandle, &attr);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    ret = RsUbSetTpAttr(&devCb, attrBitmap, tpHandle, &attr);
    EXPECT_INT_EQ(0, ret);
}

void TcRsEpollEventJfcInHandle()
{
    struct RsCtxJfceCb jfceCb1 = {0};
    struct RsCtxJfceCb jfceCb2 = {0};
    struct RsUbDevCb devCb1 = {0};
    struct RsUbDevCb devCb2 = {0};
    struct rs_cb rsCb = {0};
    urma_jfce_t jfce1 = {0};
    urma_jfce_t jfce2 = {0};
    int ret = 0;

    RS_INIT_LIST_HEAD(&rsCb.udevList);
    ret = RsEpollEventJfcInHandle(&rsCb, -ENODEV);
    EXPECT_INT_EQ(-ENODEV, ret);

    RsListAddTail(&devCb1.list, &rsCb.udevList);
    RsListAddTail(&devCb2.list, &rsCb.udevList);
    RS_INIT_LIST_HEAD(&devCb1.jfceList);
    RS_INIT_LIST_HEAD(&devCb2.jfceList);
    RsListAddTail(&jfceCb1.list, &devCb2.jfceList);
    RsListAddTail(&jfceCb2.list, &devCb2.jfceList);

    jfce1.fd = 1;
    jfce2.fd = 2;
    jfceCb1.jfceAddr = (uint64_t)(uintptr_t)(&jfce1);
    jfceCb2.jfceAddr = (uint64_t)(uintptr_t)(&jfce2);
    mocker_clean();
    mocker(RsHandleEpollPollJfc, 1, 0);
    ret = RsEpollEventJfcInHandle(&rsCb, 2);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsJfcCallbackProcess()
{
    struct RsCtxJettyCb jettyCb = {0};
    struct RsUbDevCb devCb = {0};
    struct rs_cb rsCb = {0};
    urma_jetty_t jetty = {0};
    urma_jfc_t jfc = {0};
    urma_cr_t cr = {0};

    devCb.rscb = &rsCb;
    jettyCb.devCb = &devCb;
    jettyCb.jetty = &jetty;

    cr.status = URMA_CR_RNR_RETRY_CNT_EXC_ERR;
    RsJfcCallbackProcess(&jettyCb, &cr, &jfc);
}
