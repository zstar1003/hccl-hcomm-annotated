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
#include "securec.h"
#include "ut_dispatch.h"
#include "rs_inner.h"
#include "rs_ub_tp.h"
#include "rs_ub_dfx.h"
#include "rs_ub.h"
#include "rs_ctx.h"
#include "aubdfx_api.h"
#include "ra_rs_err.h"
#include "rs_esched.h"
#include "tc_ut_rs_ctx.h"
#include "ascend_hal_dl.h"
#include "dl_aubdfx_function.h"
#include "dl_hal_function.h"
#include "hccp_msg.h"

extern void RsUbCtxDrvJettyDelete(struct RsCtxJettyCb* jettyCb);
extern void RsUbCtxFreeJettyCb(struct RsCtxJettyCb* jettyCb);
extern int RsCcuDeviceApiInit(void);
extern int RsOpenCcuSo(void);
extern void RsCloseCcuSo(void);
extern int RsEschedProcessEvent(struct rs_cb* rscb, struct event_info* event);
extern void RsEschedHandleEvent(struct rs_cb* rscb);
extern void RsEschedAckEvent(struct rs_cb* rscb, struct event_info* event);
extern int RsNetApiInit(void);
extern void RsNetApiDeinit(void);
extern int RsNetAllocJfcId(const char* udevName, unsigned int jfcMode, unsigned int* jfcId);
extern int RsNetFreeJfcId(const char* udevName, unsigned int jfcMode, unsigned int jfcId);
extern int RsNetAllocJettyId(const char* udevName, unsigned int jettyMode, unsigned int* jettyId);
extern int RsNetFreeJettyId(const char* udevName, unsigned int jettyMode, unsigned int jettyId);
extern int RsNetGetCqeBaseAddr(unsigned int dieId, unsigned long long* cqeBaseAddr);
extern int RsCcuGetCqeBaseAddr(unsigned int dieId, unsigned long long* cqeBaseAddr);

int StubRsUbGetDevCbCrErr(struct rs_cb* rscb, unsigned int devIndex, struct RsUbDevCb** devCb)
{
    static struct RsUbDevCb crErrDevCb = {0};

    *devCb = &crErrDevCb;
    return 0;
}

int StubRsUbGetDevCb(struct rs_cb* rscb, unsigned int devIndex, struct RsUbDevCb** devCb)
{
    static struct RsUbDevCb stubDevCb;

    stubDevCb.rscb = rscb;
    stubDevCb.devAttr.ub.dieId = 0;
    stubDevCb.devAttr.ub.funcId = 0;
    *devCb = &stubDevCb;
    return 0;
}

int StubRsGetRsCb(unsigned int phyId, struct rs_cb** rsCb)
{
    static struct rs_cb stubRsCb = {0};

    stubRsCb.protocol = PROTOCOL_UDMA;
    *rsCb = &stubRsCb;
    return 0;
}

void TcRsGetDevEidInfoNum()
{
    unsigned int num = 0;
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbGetDevEidInfoNum, 1, 0);
    ret = RsGetDevEidInfoNum(0, &num);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsGetDevEidInfoList()
{
    struct HccpDevEidInfo infoList[5] = {0};
    unsigned int count = 0;
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbGetDevEidInfoList, 1, 0);
    ret = RsGetDevEidInfoList(0, infoList, 0, count);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxInit()
{
    struct DevBaseAttr devAttr = {0};
    struct CtxInitAttr attr = {0};
    unsigned int devIndex = 0;
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsSetupSharemem, 1, 0);
    mocker(RsUbCtxInit, 1, 0);
    ret = RsCtxInit(&attr, &devIndex, &devAttr);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxDeinit()
{
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbCtxDeinit, 1, 0);
    ret = RsCtxDeinit(&devInfo);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxTokenIdAlloc()
{
    struct RaRsDevInfo devInfo = {0};
    unsigned long long addr = 0;
    unsigned int tokenId = 0;
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbCtxTokenIdAlloc, 1, 0);
    ret = RsCtxTokenIdAlloc(&devInfo, &addr, &tokenId);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxTokenIdFree()
{
    struct RaRsDevInfo devInfo = {0};
    unsigned long long addr = 0;
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbCtxTokenIdFree, 1, 0);
    ret = RsCtxTokenIdFree(&devInfo, addr);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxLmemReg()
{
    struct RaRsDevInfo devInfo = {0};
    struct MemRegAttrT memAttr = {0};
    struct MemRegInfoT memInfo = {0};
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbCtxLmemReg, 1, 0);
    ret = RsCtxLmemReg(&devInfo, &memAttr, &memInfo);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxLmemUnreg()
{
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbCtxLmemUnreg, 1, 0);
    ret = RsCtxLmemUnreg(&devInfo, 0);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxRmemImport()
{
    struct RaRsDevInfo devInfo = {0};
    struct MemImportAttrT memAttr = {0};
    struct MemImportInfoT memInfo = {0};
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbCtxRmemImport, 1, 0);
    ret = RsCtxRmemImport(&devInfo, &memAttr, &memInfo);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxRmemUnimport()
{
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbCtxRmemUnimport, 1, 0);
    ret = RsCtxRmemUnimport(&devInfo, 0);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxChanCreate()
{
    union DataPlaneCstmFlag dataPlaneFlag = {0};
    struct RaRsDevInfo devInfo = {0};
    unsigned long long addr = 0;
    int ret = 0;
    int fd = 0;

    dataPlaneFlag.bs.pollCqCstm = 1;
    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbCtxChanCreate, 1, 0);
    ret = RsCtxChanCreate(&devInfo, dataPlaneFlag, &addr, &fd);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxChanDestroy()
{
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbCtxChanDestroy, 1, 0);
    ret = RsCtxChanDestroy(&devInfo, 0);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxCqCreate()
{
    struct RaRsDevInfo devInfo = {0};
    struct CtxCqAttr attr = {0};
    struct CtxCqInfo info = {0};
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbCtxJfcCreate, 1, 0);
    ret = RsCtxCqCreate(&devInfo, &attr, &info);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxCqDestroy()
{
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbCtxJfcDestroy, 1, 0);
    ret = RsCtxCqDestroy(&devInfo, 0);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxQpCreate()
{
    struct RaRsDevInfo devInfo = {0};
    struct QpCreateInfo qpInfo = {0};
    struct CtxQpAttr qpAttr = {0};
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbCtxJettyCreate, 1, 0);
    ret = RsCtxQpCreate(&devInfo, &qpAttr, &qpInfo);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxQpDestroy()
{
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbCtxJettyDestroy, 1, 0);
    ret = RsCtxQpDestroy(&devInfo, 0);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxQpImport()
{
    struct RsJettyImportAttr importAttr = {0};
    struct RsJettyImportInfo importInfo = {0};
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbCtxJettyImport, 1, 0);
    ret = RsCtxQpImport(&devInfo, &importAttr, &importInfo);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxQpUnimport()
{
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbCtxJettyUnimport, 1, 0);
    ret = RsCtxQpUnimport(&devInfo, NULL, 0);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxQpBind()
{
    struct RsCtxQpInfo remoteQpInfo = {0};
    struct RsCtxQpInfo localQpInfo = {0};
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbCtxJettyBind, 1, 0);
    ret = RsCtxQpBind(&devInfo, &localQpInfo, &remoteQpInfo);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxQpUnbind()
{
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbCtxJettyUnbind, 1, 0);
    ret = RsCtxQpUnbind(&devInfo, 0);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxBatchSendWr()
{
    struct WrlistSendCompleteNum wrlistNum = {0};
    struct WrlistBaseInfo baseInfo = {0};
    struct BatchSendWrData wrData = {0};
    struct SendWrResp wrResp = {0};
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbCtxBatchSendWr, 1, 0);
    ret = RsCtxBatchSendWr(&baseInfo, &wrData, &wrResp, &wrlistNum);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxUpdateCi()
{
    struct RaRsDevInfo devInfo = {0};
    int ret = 0;

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbCtxJettyUpdateCi, 1, 0);
    ret = RsCtxUpdateCi(&devInfo, 0, 0);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsCtxEsched()
{
    TsCcuTaskReportT ccuTaskInfo = {0};
    TsUbTaskReportT ubTaskInfo = {0};
    struct RsCtxJettyCb jettyCb = {0};
    urma_jetty_t jetty = {0};
    struct RsUbDevCb rdevCb = {0};
    struct event_info event = {0};
    struct rs_cb rscb = {0};
    rscb.protocol = PROTOCOL_UDMA;
    int ret = 0;

    mocker_clean();
    mocker(RsUbFreeJettyCbList, 10, 0);
    mocker(pthread_mutex_lock, 10, 0);
    mocker(pthread_mutex_unlock, 10, 0);
    mocker(DlHalEschedSubmitEvent, 10, 0);
    mocker(RsUrmaUnbindJetty, 10, 0);
    mocker(RsUbCtxDrvJettyDelete, 10, 0);
    mocker(RsUbCtxFreeJettyCb, 10, 0);

    event.priv.msg_len = sizeof(struct TagTsHccpMsg);
    struct TagTsHccpMsg* hccpMsg = (struct TagTsHccpMsg*)event.priv.msg;
    ccuTaskInfo.num = 1;
    ubTaskInfo.num = 1;
    RS_INIT_LIST_HEAD(&rscb.udevList);
    RsListAddTail(&rdevCb.list, &rscb.udevList);
    jettyCb.jetty = &jetty;
    jettyCb.state = RS_JETTY_STATE_BIND;
    RS_INIT_LIST_HEAD(&rdevCb.jettyList);
    RsListAddTail(&jettyCb.list, &rdevCb.jettyList);

    hccpMsg->isAppExit = 0;
    hccpMsg->cmdType = 0;
    ubTaskInfo.array[0].jettyId = 1;
    hccpMsg->u.ubTaskInfo = ubTaskInfo;
    ret = RsEschedProcessEvent(&rscb, &event);
    EXPECT_INT_EQ(ret, 0);

    hccpMsg->isAppExit = 0;
    hccpMsg->cmdType = 0;
    ubTaskInfo.array[0].jettyId = 0;
    hccpMsg->u.ubTaskInfo = ubTaskInfo;
    ret = RsEschedProcessEvent(&rscb, &event);
    EXPECT_INT_EQ(ret, 0);

    hccpMsg->isAppExit = 0;
    hccpMsg->cmdType = 2;
    ret = RsEschedProcessEvent(&rscb, &event);
    EXPECT_INT_EQ(ret, -EINVAL);

    hccpMsg->isAppExit = 1;
    hccpMsg->cmdType = 1;
    ret = RsEschedProcessEvent(&rscb, &event);
    EXPECT_INT_EQ(ret, 0);

    mocker(DlHalEschedWaitEvent, 10, DRV_ERROR_INVALID_DEVICE);
    RsEschedHandleEvent(&rscb);
    mocker_clean();

    mocker(DlHalEschedWaitEvent, 10, 0);
    mocker(RsUbFreeJettyCbList, 10, 0);
    mocker(pthread_mutex_lock, 10, 0);
    mocker(pthread_mutex_unlock, 10, 0);
    mocker(DlHalEschedSubmitEvent, 10, 0);
    mocker(RsUrmaUnbindJetty, 10, 0);
    mocker(RsUbCtxDrvJettyDelete, 10, 0);
    mocker(RsUbCtxFreeJettyCb, 10, 0);

    hccpMsg->isAppExit = 2;
    hccpMsg->cmdType = 1;
    ret = RsEschedProcessEvent(&rscb, &event);
    EXPECT_INT_EQ(ret, -EINVAL);

    RsEschedAckEvent(&rscb, &event);

    event.priv.msg_len = sizeof(struct TagTsHccpMsg) + 1;
    ret = RsEschedProcessEvent(&rscb, &event);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker_clean();
}

void TcDlCcuApiInit()
{
    int ret = 0;

    ret = RsCcuDeviceApiInit();
    EXPECT_INT_EQ(ret, 0);

    ret = RsOpenCcuSo();
    EXPECT_INT_EQ(ret, 0);

    RsCloseCcuSo();
}

void TcRsGetTpInfoList()
{
    struct RaRsDevInfo devInfo = {0};
    struct HccpTpInfo infoList[2] = {0};
    struct GetTpCfg cfg = {0};
    unsigned int num = 1;
    int ret = 0;

    mocker(RsGetRsCb, 1, -EINVAL);
    ret = RsGetTpInfoList(&devInfo, &cfg, infoList, &num);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();

    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 10);
    mocker_invoke(RsUbGetDevCb, StubRsUbGetDevCb, 10);
    ret = RsGetTpInfoList(&devInfo, &cfg, infoList, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsCtxQpDestroyBatch()
{
    struct RaRsDevInfo devInfo = {0};
    unsigned int ids[] = {1, 2, 3};
    unsigned int num = 3;
    int ret = 0;

    ret = RsCtxQpDestroyBatch(NULL, ids, &num);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RsCtxQpDestroyBatch(&devInfo, NULL, &num);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RsCtxQpDestroyBatch(&devInfo, ids, NULL);
    EXPECT_INT_EQ(-EINVAL, ret);

    mocker(RsGetRsCb, 1, -EINVAL);
    ret = RsCtxQpDestroyBatch(&devInfo, ids, &num);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();

    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 10);
    mocker_invoke(RsUbGetDevCb, StubRsUbGetDevCb, 10);
    mocker(RsUbCtxJettyDestroyBatch, 1, 0);
    ret = RsCtxQpDestroyBatch(&devInfo, ids, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsCtxQpQueryBatch()
{
    struct RaRsDevInfo devInfo = {0};
    struct JettyAttr attr[10] = {{0}};
    unsigned int ids[10];
    unsigned int num = 0;
    int ret = 0;

    ret = RsCtxQpQueryBatch(NULL, ids, attr, &num);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RsCtxQpQueryBatch(&devInfo, NULL, attr, &num);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RsCtxQpQueryBatch(&devInfo, ids, NULL, &num);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RsCtxQpQueryBatch(&devInfo, ids, attr, NULL);
    EXPECT_INT_EQ(-EINVAL, ret);

    mocker(RsGetRsCb, 1, -EINVAL);
    ret = RsCtxQpQueryBatch(&devInfo, ids, attr, &num);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();

    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 10);
    mocker_invoke(RsUbGetDevCb, StubRsUbGetDevCb, 10);
    mocker(RsUbCtxQueryJettyBatch, 1, 0);
    num = 1;
    ret = RsCtxQpQueryBatch(&devInfo, ids, attr, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsNetApiInitDeinit()
{
    int ret = 0;

    ret = RsNetApiInit();
    EXPECT_INT_EQ(0, ret);

    RsNetApiDeinit();
}

void TcRsNetAllocJfcId()
{
    int ret = 0;

    ret = RsNetAllocJfcId(NULL, 0, NULL);
    EXPECT_INT_EQ(0, ret);
}

void TcRsNetFreeJfcId()
{
    int ret = 0;

    ret = RsNetFreeJfcId(NULL, 0, 0);
    EXPECT_INT_EQ(0, ret);
}

void TcRsNetAllocJettyId()
{
    int ret = 0;

    ret = RsNetAllocJettyId(NULL, 0, NULL);
    EXPECT_INT_EQ(0, ret);
}

void TcRsNetFreeJettyId()
{
    int ret = 0;

    ret = RsNetFreeJettyId(NULL, 0, 0);
    EXPECT_INT_EQ(0, ret);
}

void TcRsNetGetCqeBaseAddr()
{
    unsigned long long cqeBaseAddr = 0;
    int ret = 0;

    ret = RsNetGetCqeBaseAddr(0, &cqeBaseAddr);
    EXPECT_INT_EQ(0, ret);
}

void TcRsCcuGetCqeBaseAddr()
{
    unsigned long long cqeBaseAddr = 0;
    int ret = 0;

    ret = RsCcuGetCqeBaseAddr(0, &cqeBaseAddr);
    EXPECT_INT_EQ(0, ret);
}

void TcRsCtxGetAuxInfo()
{
    struct RaRsDevInfo devInfo = {0};
    struct HccpAuxInfoIn infoIn = {0};
    struct HccpAuxInfoOut infoOut = {0};
    int ret = 0;

    (void)memset_s(&infoOut, sizeof(struct HccpAuxInfoOut), 0, sizeof(struct HccpAuxInfoOut));
    (void)memset_s(&infoIn, sizeof(struct HccpAuxInfoIn), 0, sizeof(struct HccpAuxInfoIn));

    ret = RsCtxGetAuxInfo(NULL, &infoIn, &infoOut);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RsCtxGetAuxInfo(&devInfo, NULL, &infoOut);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RsCtxGetAuxInfo(&devInfo, &infoIn, NULL);
    EXPECT_INT_EQ(-EINVAL, ret);

    mocker_clean();
    mocker(RsGetRsCb, 1, -EINVAL);
    ret = RsCtxGetAuxInfo(&devInfo, &infoIn, &infoOut);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();

    mocker(RsGetRsCb, 1, 0);
    mocker(RsUbGetDevCb, 1, -ENODEV);
    ret = RsCtxGetAuxInfo(&devInfo, &infoIn, &infoOut);
    EXPECT_INT_EQ(-ENODEV, ret);
    mocker_clean();

    mocker(RsGetRsCb, 1, 0);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbCtxGetAuxInfo, 1, -1);
    ret = RsCtxGetAuxInfo(&devInfo, &infoIn, &infoOut);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RsGetRsCb, 1, 0);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbCtxGetAuxInfo, 1, 0);
    ret = RsCtxGetAuxInfo(&devInfo, &infoIn, &infoOut);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsGetTpAttr()
{
    struct RaRsDevInfo devInfo = {0};
    unsigned int attrBitmap = 0;
    struct TpAttr attr = {0};
    uint64_t tpHandle = 0;
    int ret = 0;

    ret = RsGetTpAttr(NULL, &attrBitmap, tpHandle, &attr);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RsGetTpAttr(&devInfo, NULL, tpHandle, &attr);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RsGetTpAttr(&devInfo, &attrBitmap, tpHandle, NULL);
    EXPECT_INT_EQ(-EINVAL, ret);

    mocker(RsGetRsCb, 1, -EINVAL);
    ret = RsGetTpAttr(&devInfo, &attrBitmap, tpHandle, &attr);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();

    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 10);
    mocker_invoke(RsUbGetDevCb, StubRsUbGetDevCb, 10);
    mocker(RsUbGetTpAttr, 1, 0);
    ret = RsGetTpAttr(&devInfo, &attrBitmap, tpHandle, &attr);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsSetTpAttr()
{
    struct RaRsDevInfo devInfo = {0};
    unsigned int attrBitmap = 0;
    struct TpAttr attr = {0};
    uint64_t tpHandle = 0;
    int ret = 0;

    ret = RsSetTpAttr(NULL, attrBitmap, tpHandle, &attr);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RsSetTpAttr(&devInfo, attrBitmap, tpHandle, NULL);
    EXPECT_INT_EQ(-EINVAL, ret);

    mocker(RsGetRsCb, 1, -EINVAL);
    ret = RsSetTpAttr(&devInfo, attrBitmap, tpHandle, &attr);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();

    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 10);
    mocker_invoke(RsUbGetDevCb, StubRsUbGetDevCb, 10);
    mocker(RsUbSetTpAttr, 1, 0);
    ret = RsSetTpAttr(&devInfo, attrBitmap, tpHandle, &attr);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsCtxGetCrErrInfoList()
{
    struct RsCtxJettyCb jettyCb = {0};
    struct CrErrInfo infoList[1] = {0};
    struct RaRsDevInfo devInfo = {0};
    unsigned int num = 1;
    int ret = 0;

    mocker_clean();
    ret = RsCtxGetCrErrInfoList(NULL, NULL, NULL);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RsCtxGetCrErrInfoList(&devInfo, NULL, NULL);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RsCtxGetCrErrInfoList(&devInfo, infoList, NULL);
    EXPECT_INT_EQ(-EINVAL, ret);

    mocker(RsGetRsCb, 1, -1);
    ret = RsCtxGetCrErrInfoList(&devInfo, infoList, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RsGetRsCb, 1, 0);
    mocker(RsUbGetDevCb, 1, -1);
    ret = RsCtxGetCrErrInfoList(&devInfo, infoList, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RsGetRsCb, 1, 0);
    mocker_invoke(RsUbGetDevCb, StubRsUbGetDevCbCrErr, 10);
    struct RsUbDevCb* crErrDevCb = NULL;
    StubRsUbGetDevCbCrErr(NULL, 0, &crErrDevCb);
    RS_INIT_LIST_HEAD(&crErrDevCb->jettyList);
    ret = RsCtxGetCrErrInfoList(&devInfo, infoList, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    RsListAddTail(&jettyCb.list, &crErrDevCb->jettyList);
    jettyCb.devCb = crErrDevCb;
    jettyCb.crErrInfo.info.status = 1;
    mocker(RsGetRsCb, 1, 0);
    mocker_invoke(RsUbGetDevCb, StubRsUbGetDevCbCrErr, 1);
    ret = RsCtxGetCrErrInfoList(&devInfo, infoList, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsCtxNotifyEvent()
{
    struct RaRsDevInfo devInfo = {0};
    struct CtxNotifyEvent event = {0};

    mocker_clean();
    EXPECT_INT_EQ(RsCtxNotifyEvent(NULL, &event), -EINVAL);
    EXPECT_INT_EQ(RsCtxNotifyEvent(&devInfo, NULL), -EINVAL);

    devInfo.phyId = 0;
    devInfo.devIndex = 0;
    event.serviceType = UBMEM_TYPE;
    event.errorType = 1;
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker_invoke(RsUbGetDevCb, StubRsUbGetDevCb, 1);
    mocker(RsAubdfxNotifyEvent, 1, 0);
    EXPECT_INT_EQ(RsCtxNotifyEvent(&devInfo, &event), 0);
    mocker_clean();
}

void TcRsCtxNotifyEventAubdfxFail()
{
    struct RaRsDevInfo devInfo = {0};
    struct CtxNotifyEvent event = {0};

    mocker_clean();
    mocker_invoke(RsGetRsCb, StubRsGetRsCb, 1);
    mocker_invoke(RsUbGetDevCb, StubRsUbGetDevCb, 1);
    mocker(RsAubdfxNotifyEvent, 1, -EINVAL);
    EXPECT_INT_EQ(RsCtxNotifyEvent(&devInfo, &event), -EINVAL);
    mocker_clean();
}

void TcRsAubdfxApiInitRollback()
{
    mocker_clean();
    mocker(RsAubdfxApiInit, 1, -EINVAL);
    EXPECT_INT_EQ(RsAubdfxApiInit(), -EINVAL);
    mocker_clean();
}
