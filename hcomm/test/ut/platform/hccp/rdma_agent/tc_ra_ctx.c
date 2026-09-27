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
#include "securec.h"
#include "ut_dispatch.h"
#include "hccp_common.h"
#include "ra_rs_err.h"
#include "hccp_ctx.h"
#include "ra.h"
#include "ra_ctx.h"
#include "ra_adp.h"
#include "ra_adp_async.h"
#include "ra_adp_pool.h"
#include "ra_hdc.h"
#include "ra_hdc_ctx.h"
#include "ra_hdc_async.h"
#include "ra_hdc_async_ctx.h"
#include "ra_hdc_async_socket.h"
#include "ra_hdc_socket.h"
#include "ra_peer_ctx.h"
#include "rs_ctx.h"
#include "tc_ra_ctx.h"

extern void HdcAsyncSetReqDone(struct RaRequestHandle* reqHandle, unsigned int phyId, int ret);
extern int DlDrvDeviceGetIndexByPhyId(uint32_t phyId, uint32_t* devIndex);
extern int DlHalGetChipInfo(unsigned int devId, halChipInfo* chipInfo);
extern hdcError_t DlDrvHdcAllocMsg(HDC_SESSION session, struct drvHdcMsg** ppMsg, int count);
extern hdcError_t
DlHalHdcRecv(HDC_SESSION session, struct drvHdcMsg* pMsg, int bufLen, UINT64 flag, int* recvBufCount, UINT32 timeout);
extern hdcError_t DlDrvHdcGetMsgBuffer(struct drvHdcMsg* msg, int index, char** pBuf, int* pLen);
extern hdcError_t DlDrvHdcFreeMsg(struct drvHdcMsg* msg);
extern int RaHdcHandleSendPkt(unsigned int chipId, void* recvBuf, unsigned int recvLen);
extern int DlDrvGetLocalDevIdByHostDevId(uint32_t phyId, uint32_t* devIndex);
extern int DlHalGetChipInfo(unsigned int devId, halChipInfo* chipInfo);
extern int RaCtxPrepareQpCreate(struct QpCreateAttr* qpAttr, struct CtxQpAttr* ctxQpAttr);
extern int QpQueryBatchParamCheck(void* qpHandle[], unsigned int* num, unsigned int phyId, unsigned int ids[]);
extern int
QpDestroyBatchParamCheck(struct RaCtxHandle* ctxHandle, void* qpHandle[], unsigned int ids[], unsigned int* num);

extern struct RaCtxOps gRaHdcCtxOps;

extern int RaRsGetTpInfoList(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern int RaRsCtxQpDestroyBatch(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern int RaRsGetTpAttr(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern int RaRsSetTpAttr(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern int RaRsCtxGetCrErrInfoList(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern int RaRsCtxQpQueryBatch(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern int RaRsGetEidByIp(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern int RaRsGetIpByEid(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern int RaRsCtxGetAuxInfo(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern void HdcAsyncHandleRecvBroken(struct HdcAsyncInfo* asyncInfo);
extern int RaAsyncHandlePkt(int handle, void* recvBuf, int len);

int RaHdcProcessMsgStub(unsigned int opcode, unsigned int phyId, char* data, unsigned int dataSize)
{
    union OpCtxQpQueryBatchData* opData = (union OpCtxQpQueryBatchData*)data;
    opData->rxData.num = 10;
    return 0;
}

void TcRaGetDevEidInfoNum()
{
    struct RaInfo info = {0};
    unsigned int num = 0;
    int ret = 0;

    mocker_clean();
    info.mode = NETWORK_OFFLINE;
    mocker(RaHdcProcessMsg, 1, 0);
    ret = RaGetDevEidInfoNum(info, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    info.mode = NETWORK_PEER_ONLINE;
    mocker(RaPeerGetDevEidInfoNum, 1, 0);
    ret = RaGetDevEidInfoNum(info, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    mocker(RaPeerGetDevEidInfoNum, 1, -1);
    ret = RaGetDevEidInfoNum(info, &num);
    EXPECT_INT_EQ(128100, ret);
    mocker_clean();
}

void TcRaGetDevEidInfoList()
{
    struct HccpDevEidInfo infoList[35] = {0};
    struct RaInfo info = {0};
    unsigned int num = 35;
    int ret = 0;

    mocker_clean();
    info.mode = NETWORK_OFFLINE;
    mocker(RaHdcProcessMsg, 2, 0);
    ret = RaGetDevEidInfoList(info, infoList, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    info.mode = NETWORK_PEER_ONLINE;
    mocker(RaPeerGetDevEidInfoList, 1, 0);
    ret = RaGetDevEidInfoList(info, infoList, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    mocker(RaPeerGetDevEidInfoList, 1, -1);
    ret = RaGetDevEidInfoList(info, infoList, &num);
    EXPECT_INT_EQ(128100, ret);
    mocker_clean();
}

int StubDlHalGetChipInfo(unsigned int devId, halChipInfo* chipInfo)
{
    strncpy_s(chipInfo->name, 32, "960", 7);
    return 0;
}

void TcRaCtxInit()
{
    struct CtxInitAttr attr = {0};
    struct CtxInitCfg cfg = {0};
    void* ctxHandle = NULL;
    int ret = 0;

    mocker_clean();
    mocker(DlDrvGetLocalDevIdByHostDevId, 1, 0);
    mocker_invoke(DlHalGetChipInfo, StubDlHalGetChipInfo, 1);
    mocker(RaHdcProcessMsg, 1, 0);
    cfg.mode = NETWORK_OFFLINE;
    ret = RaCtxInit(&cfg, &attr, &ctxHandle);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
    free(ctxHandle);
}

void TcRaGetDevBaseAttr()
{
    struct RaCtxHandle ctxHandle = {0};
    struct DevBaseAttr attr = {0};
    int ret = 0;

    ret = RaGetDevBaseAttr(&ctxHandle, &attr);
    EXPECT_INT_EQ(0, ret);
}

void TcRaCtxDeinit()
{
    struct RaCtxHandle* ctxHandle = malloc(sizeof(struct RaCtxHandle));
    int ret = 0;

    mocker_clean();
    ctxHandle->ctxOps = &gRaHdcCtxOps;
    mocker(RaHdcProcessMsg, 1, 0);
    ret = RaCtxDeinit(ctxHandle);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRaCtxLmemRegister()
{
    struct RaCtxHandle ctxHandle = {0};
    struct MrRegInfoT lmemInfo = {0};
    void* lmemHandle = NULL;
    int ret = 0;

    mocker_clean();
    ctxHandle.ctxOps = &gRaHdcCtxOps;
    mocker(RaHdcProcessMsg, 3, 0);
    ctxHandle.protocol = PROTOCOL_RDMA;
    ret = RaCtxLmemRegister(&ctxHandle, &lmemInfo, &lmemHandle);
    EXPECT_INT_EQ(0, ret);
    free(lmemHandle);
    ctxHandle.protocol = PROTOCOL_UDMA;
    ret = RaCtxLmemRegister(&ctxHandle, &lmemInfo, &lmemHandle);
    EXPECT_INT_EQ(0, ret);
    ret = RaCtxLmemUnregister(&ctxHandle, lmemHandle);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRaCtxRmemImport()
{
    struct MrImportInfoT rmemInfo = {0};
    struct RaCtxHandle ctxHandle = {0};
    void* rmemHandle = NULL;
    int ret = 0;

    mocker_clean();
    ctxHandle.ctxOps = &gRaHdcCtxOps;
    rmemInfo.in.key.size = 4;
    ctxHandle.protocol = PROTOCOL_UDMA;
    mocker(RaHdcProcessMsg, 2, 0);
    ret = RaCtxRmemImport(&ctxHandle, &rmemInfo, &rmemHandle);
    EXPECT_INT_EQ(0, ret);
    ret = RaCtxRmemUnimport(&ctxHandle, rmemHandle);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRaCtxChanCreate()
{
    struct RaCtxHandle ctxHandle = {0};
    struct ChanInfoT chanInfo = {0};
    void* chanHandle = NULL;
    int ret = 0;

    mocker_clean();
    ctxHandle.ctxOps = &gRaHdcCtxOps;
    mocker(RaHdcProcessMsg, 2, 0);
    ret = RaCtxChanCreate(&ctxHandle, &chanInfo, &chanHandle);
    EXPECT_INT_EQ(0, ret);
    ret = RaCtxChanDestroy(&ctxHandle, chanHandle);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRaCtxCqCreate()
{
    struct RaCtxHandle ctxHandle = {0};
    struct CqInfoT cqInfoT = {0};
    void* cqHandle = NULL;
    int ret = 0;

    mocker_clean();
    ctxHandle.ctxOps = &gRaHdcCtxOps;
    mocker(RaHdcProcessMsg, 5, 0);

    ctxHandle.protocol = PROTOCOL_UDMA;
    ret = RaCtxCqCreate(&ctxHandle, &cqInfoT, &cqHandle);
    EXPECT_INT_EQ(0, ret);
    ret = RaCtxCqDestroy(&ctxHandle, cqHandle);
    EXPECT_INT_EQ(0, ret);

    ctxHandle.protocol = PROTOCOL_UDMA;
    cqInfoT.in.ub.ccuExCfg.valid = 1;
    cqInfoT.in.ub.mode = JFC_MODE_CCU_POLL;
    ret = RaCtxCqCreate(&ctxHandle, &cqInfoT, &cqHandle);
    EXPECT_INT_EQ(0, ret);
    ret = RaCtxCqDestroy(&ctxHandle, cqHandle);
    EXPECT_INT_EQ(0, ret);

    ctxHandle.protocol = PROTOCOL_RDMA;
    ret = RaCtxCqCreate(&ctxHandle, &cqInfoT, &cqHandle);
    EXPECT_INT_EQ(0, ret);

    ret = RaCtxCqCreate(&ctxHandle, &cqInfoT, NULL);
    EXPECT_INT_EQ(ConverReturnCode(RDMA_OP, -EINVAL), ret);

    free(cqHandle);
    mocker_clean();
}

void TcRaCtxTokenIdAlloc()
{
    TcRaCtxTokenIdAlloc1();
    TcRaCtxTokenIdAlloc2();
    TcRaCtxTokenIdAlloc3();
}

void TcRaCtxTokenIdAlloc1()
{
    struct RaCtxHandle ctxHandle = {0};
    struct HccpTokenId info = {0};
    void* tokenIdHandle = NULL;
    int ret;

    mocker_clean();
    ctxHandle.ctxOps = &gRaHdcCtxOps;
    mocker(RaHdcProcessMsg, 3, 0);
    ctxHandle.protocol = PROTOCOL_UDMA;
    ret = RaCtxTokenIdAlloc(&ctxHandle, &info, NULL);
    EXPECT_INT_NE(0, ret);
    ret = RaCtxTokenIdFree(NULL, NULL);
    EXPECT_INT_NE(0, ret);

    ret = RaCtxTokenIdAlloc(&ctxHandle, &info, &tokenIdHandle);
    EXPECT_INT_EQ(0, ret);
    ret = RaCtxTokenIdFree(&ctxHandle, tokenIdHandle);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRaCtxTokenIdAlloc2()
{
    struct RaCtxHandle ctxHandle = {0};
    struct HccpTokenId info = {0};
    void* tokenIdHandle = NULL;
    int ret;

    mocker_clean();
    ctxHandle.ctxOps = &gRaHdcCtxOps;
    mocker(RaHdcProcessMsg, 3, 0);
    ctxHandle.protocol = PROTOCOL_UDMA;

    mocker(RaHdcCtxTokenIdAlloc, 10, -EPERM);
    ret = RaCtxTokenIdAlloc(&ctxHandle, &info, &tokenIdHandle);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRaCtxTokenIdAlloc3()
{
    struct RaCtxHandle ctxHandle = {0};
    struct HccpTokenId info = {0};
    void* tokenIdHandle = NULL;
    int ret;

    mocker_clean();
    ctxHandle.ctxOps = &gRaHdcCtxOps;
    mocker(RaHdcProcessMsg, 3, 0);
    ctxHandle.protocol = PROTOCOL_UDMA;

    ret = RaCtxTokenIdAlloc(&ctxHandle, &info, &tokenIdHandle);
    EXPECT_INT_EQ(0, ret);

    mocker(RaHdcCtxTokenIdFree, 10, -EPERM);
    ret = RaCtxTokenIdFree(&ctxHandle, tokenIdHandle);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRaCtxQpCreate()
{
    struct RaCtxQpHandle* qpHandle = NULL;
    struct RaCtxHandle ctxHandle = {0};
    struct RaCqHandle scqHandle = {0};
    struct RaCqHandle rcqHandle = {0};
    struct QpCreateAttr attr = {0};
    struct QpCreateInfo info = {0};
    void* cqHandle = NULL;
    int ret = 0;

    mocker_clean();
    attr.scqHandle = &scqHandle;
    attr.rcqHandle = &rcqHandle;
    ctxHandle.ctxOps = &gRaHdcCtxOps;
    mocker(RaHdcProcessMsg, 5, 0);

    ctxHandle.protocol = PROTOCOL_UDMA;
    ret = RaCtxQpCreate(&ctxHandle, &attr, &info, (void**)&qpHandle);
    EXPECT_INT_EQ(0, ret);
    ret = RaCtxQpDestroy(qpHandle);
    EXPECT_INT_EQ(0, ret);

    attr.ub.mode = JETTY_MODE_CCU_TA_CACHE;
    attr.ub.taCacheMode.lockFlag = 1;
    ret = RaCtxQpCreate(&ctxHandle, &attr, &info, (void**)&qpHandle);
    EXPECT_INT_EQ(0, ret);
    ret = RaCtxQpDestroy(qpHandle);
    EXPECT_INT_EQ(0, ret);

    ctxHandle.protocol = PROTOCOL_RDMA;
    ret = RaCtxQpCreate(&ctxHandle, &attr, &info, (void**)&qpHandle);
    EXPECT_INT_EQ(0, ret);

    ret = RaCtxQpCreate(&ctxHandle, &attr, &info, NULL);
    EXPECT_INT_EQ(ConverReturnCode(RDMA_OP, -EINVAL), ret);

    free(qpHandle);
    mocker_clean();
}

void TcRaCtxQpImport()
{
    struct RaCtxHandle ctxHandle = {0};
    struct QpImportInfoT qpInfo = {0};
    struct RaCtxRemQpHandle* remQpHandle = NULL;
    int ret = 0;

    mocker_clean();
    ctxHandle.ctxOps = &gRaHdcCtxOps;
    mocker(RaHdcProcessMsg, 2, 0);
    ctxHandle.protocol = PROTOCOL_UDMA;
    ret = RaCtxQpImport(&ctxHandle, &qpInfo, (void**)&remQpHandle);
    EXPECT_INT_EQ(0, ret);
    ret = RaCtxQpUnimport(&ctxHandle, remQpHandle);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRaCtxQpBind()
{
    struct RaCtxRemQpHandle remQpHandle = {0};
    struct RaCtxQpHandle qpHandle = {0};
    struct RaCtxHandle ctxHandle = {0};
    int ret = 0;

    mocker_clean();
    remQpHandle.protocol = PROTOCOL_UDMA;
    qpHandle.ctxHandle = &ctxHandle;
    ctxHandle.ctxOps = &gRaHdcCtxOps;
    mocker(RaHdcProcessMsg, 2, 0);
    ret = RaCtxQpBind(&qpHandle, &remQpHandle);
    EXPECT_INT_EQ(0, ret);
    ret = RaCtxQpUnbind(&qpHandle);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
    mocker(RaHdcProcessMsg, 2, -ENODEV);
    ret = RaCtxQpUnbind(&qpHandle);
    EXPECT_INT_EQ(ConverReturnCode(RDMA_OP, -ENODEV), ret);
    mocker_clean();
}

void TcRaBatchSendWr()
{
    struct RaCtxRemQpHandle remQpHandle = {0};
    struct RaRmemHandle rmemHandle = {0};
    struct RaCtxQpHandle qpHandle = {0};
    struct RaCtxHandle ctxHandle = {0};
    struct SendWrData wrList[1] = {0};
    struct SendWrResp opResp[1] = {0};

    unsigned int completeNum = 0;
    int inlineData = 0;
    int ret = 0;

    mocker_clean();
    qpHandle.ctxHandle = &ctxHandle;
    ctxHandle.ctxOps = &gRaHdcCtxOps;
    wrList[0].rmemHandle = &rmemHandle;
    qpHandle.protocol = PROTOCOL_RDMA;
    wrList[0].rdma.flags = RA_SEND_INLINE;
    wrList[0].inlineData = (uint8_t*)&inlineData;
    mocker(RaHdcProcessMsg, 2, 0);
    ret = RaBatchSendWr(&qpHandle, wrList, opResp, 1, &completeNum);
    EXPECT_INT_EQ(0, ret);
    qpHandle.protocol = PROTOCOL_UDMA;
    wrList[0].ub.flags.bs.inlineFlag = 1;
    wrList[0].ub.remQpHandle = &remQpHandle;
    wrList[0].ub.opcode = RA_UB_OPC_WRITE;
    ret = RaBatchSendWr(&qpHandle, wrList, opResp, 1, &completeNum);
    EXPECT_INT_EQ(0, ret);
}

void TcRaCtxUpdateCi()
{
    struct RaCtxQpHandle qpHandle = {0};
    struct RaCtxHandle ctxHandle = {0};
    int ret = 0;

    mocker_clean();
    mocker(RaHdcProcessMsg, 1, 0);
    qpHandle.ctxHandle = &ctxHandle;
    ctxHandle.ctxOps = &gRaHdcCtxOps;
    ret = RaCtxUpdateCi(&qpHandle, 1);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRaGetTpInfoListAsync()
{
    struct HccpTpInfo infoList[HCCP_MAX_TPID_INFO_NUM] = {0};
    struct RaRequestHandle* reqHandle = NULL;
    struct RaCtxHandle ctxHandle = {0};
    struct GetTpCfg cfg = {0};
    unsigned int num = 0;
    int ret = 0;

    ret = RaGetTpInfoListAsync(NULL, &cfg, infoList, &num, (void**)&reqHandle);
    EXPECT_INT_NE(0, ret);

    ret = RaGetTpInfoListAsync(&ctxHandle, NULL, infoList, &num, (void**)&reqHandle);
    EXPECT_INT_NE(0, ret);

    ret = RaGetTpInfoListAsync(&ctxHandle, &cfg, NULL, &num, (void**)&reqHandle);
    EXPECT_INT_NE(0, ret);

    ret = RaGetTpInfoListAsync(&ctxHandle, &cfg, infoList, NULL, (void**)&reqHandle);
    EXPECT_INT_NE(0, ret);

    ret = RaGetTpInfoListAsync(&ctxHandle, &cfg, infoList, &num, NULL);
    EXPECT_INT_NE(0, ret);

    ret = RaGetTpInfoListAsync(&ctxHandle, &cfg, infoList, &num, (void**)&reqHandle);
    EXPECT_INT_NE(0, ret);

    num = HCCP_MAX_TPID_INFO_NUM + 1;
    ret = RaGetTpInfoListAsync(&ctxHandle, &cfg, infoList, &num, (void**)&reqHandle);
    EXPECT_INT_NE(0, ret);

    num = 1;
    mocker(RaHdcGetTpInfoListAsync, 1, 0);
    ret = RaGetTpInfoListAsync(&ctxHandle, &cfg, infoList, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRaHdcGetTpInfoListAsync()
{
    struct HccpTpInfo infoList[HCCP_MAX_TPID_INFO_NUM] = {0};
    struct RaResponseTpInfoList* asyncRsp = NULL;
    union OpGetTpInfoListData recvBuf = {0};
    struct RaRequestHandle* reqHandle = NULL;
    struct RaCtxHandle ctxHandle = {0};
    struct GetTpCfg cfg = {0};
    unsigned int num = 1;
    int ret = 0;

    mocker(RaHdcSendMsgAsync, 1, -1);
    ret = RaHdcGetTpInfoListAsync(&ctxHandle, &cfg, infoList, &num, (void**)&reqHandle);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(RaHdcSendMsgAsync, 1, 0);
    ret = RaHdcGetTpInfoListAsync(&ctxHandle, &cfg, infoList, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    reqHandle->opRet = -1;
    RaHdcAsyncHandleTpInfoList(reqHandle);

    reqHandle->opRet = 0;
    reqHandle->recvBuf = &recvBuf;
    asyncRsp = (struct RaResponseTpInfoList*)calloc(1, sizeof(struct RaResponseTpInfoList));
    asyncRsp->num = &num;
    asyncRsp->infoList = infoList;
    reqHandle->privData = (void*)asyncRsp;
    RaHdcAsyncHandleTpInfoList(reqHandle);

    reqHandle->opRet = 0;
    recvBuf.rxData.num = 1;
    reqHandle->recvBuf = &recvBuf;
    asyncRsp = (struct RaResponseTpInfoList*)calloc(1, sizeof(struct RaResponseTpInfoList));
    asyncRsp->num = &num;
    asyncRsp->infoList = infoList;
    reqHandle->privData = (void*)asyncRsp;
    mocker(memcpy_s, 1, -1);
    RaHdcAsyncHandleTpInfoList(reqHandle);
    mocker_clean();

    reqHandle->opRet = 0;
    recvBuf.rxData.num = 1;
    reqHandle->recvBuf = &recvBuf;
    asyncRsp = (struct RaResponseTpInfoList*)calloc(1, sizeof(struct RaResponseTpInfoList));
    asyncRsp->num = &num;
    asyncRsp->infoList = infoList;
    reqHandle->privData = (void*)asyncRsp;
    mocker(memcpy_s, 1, 0);
    RaHdcAsyncHandleTpInfoList(reqHandle);
    mocker_clean();

    free(reqHandle);
    reqHandle = NULL;
}

void TcRaRsGetTpInfoList()
{
    union OpGetTpInfoListData dataIn;
    union OpGetTpInfoListData dataOut;
    int rcvBufLen = 0;
    int opResult = 0;
    int outLen = 0;
    int ret;

    char* inBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpGetTpInfoListData));
    char* outBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpGetTpInfoListData));

    dataIn.txData.phyId = 0;
    dataIn.txData.devIndex = 0;
    memcpy_s(
        inBuf + sizeof(struct MsgHead), sizeof(union OpGetTpInfoListData), &dataIn, sizeof(union OpGetTpInfoListData));
    ret = RaRsGetTpInfoList(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker(RsGetTpInfoList, 1, -1);
    ret = RaRsGetTpInfoList(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    free(inBuf);
    inBuf = NULL;
    free(outBuf);
    outBuf = NULL;
}

void TcRaRsAsyncHdcSessionConnect()
{
    mocker_clean();
    RaHwAsyncInit(0, 0);
    union OpAsyncHdcConnectData connectData = {0};
    connectData.txData.phyId = 0;
    connectData.txData.queueSize = MAX_POOL_QUEUE_SIZE;
    connectData.txData.threadNum = MAX_POOL_THREAD_NUM;
    unsigned int connectDataSize = sizeof(union OpAsyncHdcConnectData);

    void* sendRcvBuf = NULL;
    unsigned int sendRcvLen;
    int ret;
    pid_t hostTgid = 0;
    unsigned int opcode = RA_RS_ASYNC_HDC_SESSION_CONNECT;
    sendRcvLen = sizeof(struct MsgHead) + connectDataSize;
    sendRcvBuf = (void*)calloc(sendRcvLen, sizeof(char));
    MsgHeadBuildUp(sendRcvBuf, opcode, 0, connectDataSize, hostTgid);

    ret = memcpy_s(
        sendRcvBuf + sizeof(struct MsgHead), sendRcvLen - sizeof(struct MsgHead), &connectData, connectDataSize);
    if (ret) {
        hccp_err("[process][ra_hdc_msg]memcpy_s failed, ret(%d) phyId(%u)", ret, connectData.txData.phyId);
        return;
    }
    int opRet = 0;
    void* sendBuf = NULL;
    int sndBufLen = 0;

    struct MsgHead* recvMsgHead = (struct MsgHead*)sendRcvBuf;
    sendBuf = (char*)calloc(sizeof(char), recvMsgHead->msgDataLen + sizeof(struct MsgHead));
    if (sendBuf == NULL) {
        hccp_err("calloc fail.");
        return;
    }

    mocker(RaHdcAsyncRecvPkt, 1, -1);
    RaRsAsyncHdcSessionConnect(sendRcvBuf, sendBuf, &sndBufLen, &opRet, sendRcvLen);

    union OpAsyncHdcCloseData closeData = {0};
    unsigned int closeDataSize = sizeof(union OpAsyncHdcCloseData);
    void* sendRcvBuf2 = NULL;
    unsigned int sendRcvLen2;
    unsigned int opcode2 = RA_RS_ASYNC_HDC_SESSION_CLOSE;
    sendRcvLen2 = sizeof(struct MsgHead) + closeDataSize;
    sendRcvBuf2 = (void*)calloc(sendRcvLen, sizeof(char));
    MsgHeadBuildUp(sendRcvBuf2, opcode2, 0, closeDataSize, hostTgid);
    ret = memcpy_s(
        sendRcvBuf2 + sizeof(struct MsgHead), sendRcvLen2 - sizeof(struct MsgHead), &closeData, closeDataSize);

    int opRet2 = 0;
    void* sendBuf2 = NULL;
    int sndBufLen2 = 0;

    RaRsAsyncHdcSessionClose(sendRcvBuf2, sendBuf2, &sndBufLen2, &opRet2, sendRcvLen2);
    RaHwAsyncDeinit();

    free(sendRcvBuf);
    free(sendBuf);
    free(sendRcvBuf2);

    mocker_clean();
}

void TcRaHdcAsyncSendPkt()
{
    char* data;
    data = (char*)calloc(100, sizeof(char));
    unsigned long long size = 100;
    union OpSocketSendData* asyncData = NULL;
    asyncData = (union OpSocketSendData*)calloc(sizeof(union OpSocketSendData), sizeof(char));
    asyncData->txData.fd = 0;
    asyncData->txData.sendSize = size;
    memcpy_s(asyncData->txData.dataSend, SOCKET_SEND_MAXLEN, data, size);

    void* sendBuf = NULL;
    unsigned int sendLen;
    int ret;
    pid_t hostTgid = 0;
    unsigned int opcode = RA_RS_SOCKET_SEND;
    sendLen = sizeof(struct MsgHead) + sizeof(union OpSocketSendData);
    sendBuf = (void*)calloc(sendLen, sizeof(char));
    MsgHeadBuildUp(sendBuf, opcode, 0, sizeof(union OpSocketSendData), hostTgid);

    memcpy_s(
        sendBuf + sizeof(struct MsgHead), sendLen - sizeof(struct MsgHead), asyncData, sizeof(union OpSocketSendData));
    MsgHeadBuildUp(sendBuf, opcode, 0, sizeof(union OpSocketSendData), hostTgid);

    RaHdcHandleSendPkt(0, sendBuf, sendLen);

    free(data);
    free(asyncData);
    free(sendBuf);
}

void TcRaHdcAsyncRecvPkt()
{
    struct RaHdcAsyncInfo asyncInfo = {0};
    void* recvBuf = NULL;

    mocker_clean();
    mocker(DlDrvHdcAllocMsg, 1, 0);
    mocker(DlHalHdcRecv, 1, -1);
    mocker(DlDrvHdcFreeMsg, 1, 0);
    RaHdcAsyncRecvPkt(&asyncInfo, 0, NULL, NULL);
    mocker_clean();

    mocker(DlDrvHdcAllocMsg, 1, 0);
    mocker(DlHalHdcRecv, 1, 0);
    mocker(DlDrvHdcGetMsgBuffer, 1, -1);
    mocker(DlDrvHdcFreeMsg, 1, 0);
    RaHdcAsyncRecvPkt(&asyncInfo, 0, NULL, NULL);
    mocker_clean();

    mocker(DlDrvHdcAllocMsg, 1, 0);
    mocker(DlHalHdcRecv, 1, 0);
    mocker(DlDrvHdcGetMsgBuffer, 1, 0);
    mocker(DlDrvHdcFreeMsg, 1, 0);
    RaHdcAsyncRecvPkt(&asyncInfo, 0, &recvBuf, NULL);
    free(recvBuf);
    mocker_clean();
}

hdcError_t DlDrvHdcGetMsgBufferStub(struct drvHdcMsg* msg, int index, char** pBuf, int* pLen)
{
    *pLen = 1;
    return 0;
}

void TcHdcAsyncRecvPkt()
{
    struct RaRequestHandle stubReqHandle = {0};
    struct HdcAsyncInfo asyncInfo = {0};
    HDC_SESSION stubSession = {0};
    void* recvBuf = NULL;
    unsigned int recvLen;
    int ret;

    asyncInfo.session = &stubSession;
    RA_INIT_LIST_HEAD(&asyncInfo.reqList);
    RaListAddTail(&stubReqHandle.list, &asyncInfo.reqList);

    mocker_clean();
    mocker(pthread_mutex_lock, 10, 0);
    mocker(DlDrvHdcAllocMsg, 10, 0);
    mocker(DlHalHdcRecv, 10, -25);
    mocker(DlDrvHdcGetMsgBuffer, 10, 0);
    mocker(pthread_mutex_unlock, 10, 0);
    mocker(memcpy_s, 10, 0);
    mocker(DlDrvHdcFreeMsg, 10, 0);
    mocker(HdcAsyncSetReqDone, 10, (void*)0);
    ret = HdcAsyncRecvPkt(&asyncInfo, 0, recvBuf, &recvLen);
    EXPECT_INT_EQ(ret, -25);
    HdcAsyncHandleRecvBroken(&asyncInfo);

    mocker_clean();
    mocker(pthread_mutex_lock, 10, 0);
    mocker(DlDrvHdcAllocMsg, 10, 0);
    mocker(DlHalHdcRecv, 10, 0);
    mocker_invoke(DlDrvHdcGetMsgBuffer, DlDrvHdcGetMsgBufferStub, 10);
    mocker(pthread_mutex_unlock, 10, 0);
    mocker(memcpy_s, 10, 0);
    mocker(DlDrvHdcFreeMsg, 10, 0);
    ret = HdcAsyncRecvPkt(&asyncInfo, 0, recvBuf, &recvLen);
    EXPECT_INT_EQ(ret, -25);
    HdcAsyncHandleRecvBroken(&asyncInfo);

    EXPECT_INT_EQ(asyncInfo.lastRecvStatus, -25);
    ret = RaHdcSendMsgAsync(4, 0, NULL, 0, NULL);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();
}

void TcRaHdcPoolAddTask()
{
    struct RaHdcThreadPool pool = {0};
    struct RaHdcTask taskQueue[5] = {0};
    int (*RaHdcHandleSendPkt)(unsigned int chipId, void* recvBuf, unsigned int recvLen);

    mocker_clean();
    pool.taskQueue = taskQueue;
    pool.queuePi = 0;
    pool.taskNum = 2;
    pool.queueSize = 5;
    mocker(pthread_mutex_lock, 1, 0);
    mocker(pthread_mutex_unlock, 1, 0);
    RaHdcPoolAddTask(&pool, RaHdcHandleSendPkt, 0, NULL, 2);
    mocker_clean();
}

int StubRaHdcPoolCreatePthreadCreate(
    pthread_t* thread, const pthread_attr_t* attr, void* (*startRoutine)(void*), void* arg)
{
    return -1;
}

void TcRaHdcPoolCreate()
{
    mocker_clean();
    mocker_invoke(pthread_create, StubRaHdcPoolCreatePthreadCreate, 1);
    RaHdcPoolCreate(1, 1);
    mocker_clean();
}

void TcRaAsyncHandlePkt()
{
    struct MsgHead recvBuf = {0};

    mocker_clean();
    mocker(RaHdcHandleSendPkt, 1, 0);
    mocker(RaHdcCloseSession, 1, 0);
    mocker(RaHdcPoolAddTask, 1, 0);
    mocker(pthread_mutex_lock, 10, 0);
    mocker(pthread_mutex_unlock, 10, 0);
    RaAsyncHandlePkt(1, &recvBuf, 0);
    mocker_clean();
}

void TcRaHdcAsyncHandleSocketListenStart()
{
    struct RaRequestHandle reqHandle = {0};

    mocker_clean();
    reqHandle.privData = malloc(sizeof(struct RaResponseSocketListen));
    mocker(RaGetSocketListenResult, 1, -1);
    RaHdcAsyncHandleSocketListenStart(&reqHandle);
    mocker_clean();
}

void TcRaHdcAsyncHandleQpImport()
{
    struct RaRequestHandle reqHandle = {0};
    union OpCtxQpImportData recvBuf = {0};
    struct QpImportInfo privData = {0};
    struct RaCtxRemQpHandle privHandle = {0};

    reqHandle.recvBuf = &recvBuf;
    reqHandle.privData = &privData;
    reqHandle.privHandle = &privHandle;
    privHandle.protocol = PROTOCOL_UDMA;
    RaHdcAsyncHandleQpImport(&reqHandle);
}

void TcRaPeerCtxInit()
{
    struct DevBaseAttr devBaseAttr = {0};
    struct RaCtxHandle ctxHandle = {0};
    struct CtxInitAttr info = {0};
    unsigned int devIndex = 0;
    int ret = 0;

    ret = RaPeerCtxInit(&ctxHandle, &info, &devIndex, &devBaseAttr);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsCtxInit, 1, -1);
    ret = RaPeerCtxInit(&ctxHandle, &info, &devIndex, &devBaseAttr);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaPeerCtxDeinit()
{
    struct RaCtxHandle ctxHandle = {0};
    int ret = 0;

    ret = RaPeerCtxDeinit(&ctxHandle);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsCtxDeinit, 1, -1);
    ret = RaPeerCtxDeinit(&ctxHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaPeerGetDevEidInfoNum()
{
    struct RaInfo info = {0};
    unsigned int num = 0;
    int ret = 0;

    ret = RaPeerGetDevEidInfoNum(info, &num);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsGetDevEidInfoNum, 1, -1);
    ret = RaPeerGetDevEidInfoNum(info, &num);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaPeerGetDevEidInfoList()
{
    struct HccpDevEidInfo infoList[35] = {0};
    unsigned int phyId = 0;
    unsigned int num = 0;
    int ret = 0;

    ret = RaPeerGetDevEidInfoList(phyId, infoList, &num);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsGetDevEidInfoList, 1, -1);
    ret = RaPeerGetDevEidInfoList(phyId, infoList, &num);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaPeerCtxTokenIdAlloc()
{
    struct RaTokenIdHandle tokenIdHandle = {0};
    struct RaCtxHandle ctxHandle = {0};
    struct HccpTokenId info = {0};
    int ret = 0;

    ret = RaPeerCtxTokenIdAlloc(&ctxHandle, &info, &tokenIdHandle);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsCtxTokenIdAlloc, 1, -1);
    ret = RaPeerCtxTokenIdAlloc(&ctxHandle, &info, &tokenIdHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaPeerCtxTokenIdFree()
{
    struct RaTokenIdHandle tokenIdHandle = {0};
    struct RaCtxHandle ctxHandle = {0};
    int ret = 0;

    ret = RaPeerCtxTokenIdFree(&ctxHandle, &tokenIdHandle);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsCtxTokenIdFree, 1, -1);
    ret = RaPeerCtxTokenIdFree(&ctxHandle, &tokenIdHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaPeerCtxLmemRegister()
{
    struct RaLmemHandle lmemHandle = {0};
    struct RaCtxHandle ctxHandle = {0};
    struct MrRegInfoT lmemInfo = {0};
    int ret = 0;

    lmemInfo.in.ub.flags.bs.tokenIdValid = 1;
    ret = RaPeerCtxLmemRegister(&ctxHandle, &lmemInfo, &lmemHandle);
    EXPECT_INT_EQ(ret, -22);

    lmemInfo.in.ub.flags.bs.tokenIdValid = 0;
    ret = RaPeerCtxLmemRegister(&ctxHandle, &lmemInfo, &lmemHandle);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsCtxLmemReg, 1, -1);
    ret = RaPeerCtxLmemRegister(&ctxHandle, &lmemInfo, &lmemHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaPeerCtxLmemUnregister()
{
    struct RaLmemHandle lmemHandle = {0};
    struct RaCtxHandle ctxHandle = {0};
    int ret = 0;

    ret = RaPeerCtxLmemUnregister(&ctxHandle, &lmemHandle);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsCtxLmemUnreg, 1, -1);
    ret = RaPeerCtxLmemUnregister(&ctxHandle, &lmemHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaPeerCtxRmemImport()
{
    struct MrImportInfoT rmemInfo = {0};
    struct RaCtxHandle ctxHandle = {0};
    int ret = 0;

    ret = RaPeerCtxRmemImport(&ctxHandle, &rmemInfo);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsCtxRmemImport, 1, -1);
    ret = RaPeerCtxRmemImport(&ctxHandle, &rmemInfo);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaPeerCtxRmemUnimport()
{
    struct RaRmemHandle rmemHandle = {0};
    struct RaCtxHandle ctxHandle = {0};
    int ret = 0;

    ret = RaPeerCtxRmemUnimport(&ctxHandle, &rmemHandle);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsCtxRmemUnimport, 1, -1);
    ret = RaPeerCtxRmemUnimport(&ctxHandle, &rmemHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaPeerCtxChanCreate()
{
    struct RaChanHandle chanHandle = {0};
    struct RaCtxHandle ctxHandle = {0};
    struct ChanInfoT chanInfo = {0};
    int ret = 0;

    ret = RaPeerCtxChanCreate(&ctxHandle, &chanInfo, &chanHandle);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsCtxChanCreate, 1, -1);
    ret = RaPeerCtxChanCreate(&ctxHandle, &chanInfo, &chanHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaPeerCtxChanDestroy()
{
    struct RaChanHandle chanHandle = {0};
    struct RaCtxHandle ctxHandle = {0};
    int ret = 0;

    ret = RaPeerCtxChanDestroy(&ctxHandle, &chanHandle);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsCtxChanDestroy, 1, -1);
    ret = RaPeerCtxChanDestroy(&ctxHandle, &chanHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaPeerCtxCqCreate()
{
    struct RaChanHandle chanHandle = {0};
    struct RaCtxHandle ctxHandle = {0};
    struct RaCqHandle cqHandle = {0};
    struct CqInfoT info = {0};
    int ret = 0;

    info.in.ub.mode = JFC_MODE_CCU_POLL;
    info.in.ub.ccuExCfg.valid = true;
    ret = RaPeerCtxCqCreate(&ctxHandle, &info, &cqHandle);
    EXPECT_INT_EQ(ret, -22);

    mocker(RsCtxCqCreate, 1, -1);
    info.in.ub.mode = JFC_MODE_NORMAL;
    info.in.chanHandle = (void*)&chanHandle;
    ret = RaPeerCtxCqCreate(&ctxHandle, &info, &cqHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaPeerCtxCqDestroy()
{
    struct RaCtxHandle ctxHandle = {0};
    struct RaCqHandle cqHandle = {0};
    int ret = 0;

    ret = RaPeerCtxCqDestroy(&ctxHandle, &cqHandle);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsCtxCqDestroy, 1, -1);
    ret = RaPeerCtxCqDestroy(&ctxHandle, &cqHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaPeerCtxQpCreate()
{
    struct RaCtxQpHandle qpHandle = {0};
    struct RaCtxHandle ctxHandle = {0};
    struct QpCreateAttr qpAttr = {0};
    struct QpCreateInfo qpInfo = {0};
    int ret = 0;

    qpAttr.ub.mode = JETTY_MODE_CCU;
    ret = RaPeerCtxQpCreate(&ctxHandle, &qpAttr, &qpInfo, &qpHandle);
    EXPECT_INT_EQ(ret, -22);

    qpAttr.ub.mode = JETTY_MODE_URMA_NORMAL;
    mocker(RaCtxPrepareQpCreate, 1, 0);
    ret = RaPeerCtxQpCreate(&ctxHandle, &qpAttr, &qpInfo, &qpHandle);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker(RaCtxPrepareQpCreate, 1, 0);
    mocker(RsCtxQpCreate, 1, -1);
    ret = RaPeerCtxQpCreate(&ctxHandle, &qpAttr, &qpInfo, &qpHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaCtxPrepareQpCreate()
{
    struct RaTokenIdHandle tokenIdHandle = {0};
    struct CtxQpAttr ctxQpAttr = {0};
    struct QpCreateAttr qpAttr = {0};
    struct RaCqHandle cqHandle = {0};
    int ret = 0;

    ret = RaCtxPrepareQpCreate(&qpAttr, &ctxQpAttr);
    EXPECT_INT_EQ(ret, -22);

    qpAttr.scqHandle = (void*)&cqHandle;
    ret = RaCtxPrepareQpCreate(&qpAttr, &ctxQpAttr);
    EXPECT_INT_EQ(ret, -22);

    qpAttr.rcqHandle = (void*)&cqHandle;

    qpAttr.ub.mode = JETTY_MODE_URMA_NORMAL;
    qpAttr.ub.tokenIdHandle = (void*)&tokenIdHandle;
    ret = RaCtxPrepareQpCreate(&qpAttr, &ctxQpAttr);
    EXPECT_INT_EQ(ret, 0);
}

void TcRaPeerCtxQpDestroy()
{
    struct RaCtxQpHandle qpHandle = {0};
    int ret = 0;

    ret = RaPeerCtxQpDestroy(&qpHandle);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsCtxQpDestroy, 1, -1);
    ret = RaPeerCtxQpDestroy(&qpHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaPeerCtxQpImport()
{
    struct RaCtxRemQpHandle remQpHandle = {0};
    struct RaCtxHandle ctxHandle = {0};
    struct QpImportInfoT qpInfo = {0};
    int ret = 0;

    ret = RaPeerCtxQpImport(&ctxHandle, &qpInfo, &remQpHandle);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsCtxQpImport, 1, -1);
    ret = RaPeerCtxQpImport(&ctxHandle, &qpInfo, &remQpHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaPeerCtxQpUnimport()
{
    struct RaCtxRemQpHandle remQpHandle = {0};
    int ret = 0;

    ret = RaPeerCtxQpUnimport(&remQpHandle);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsCtxQpUnimport, 1, -1);
    ret = RaPeerCtxQpUnimport(&remQpHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaPeerCtxQpBind()
{
    struct RaCtxRemQpHandle remQpHandle = {0};
    struct RaCtxQpHandle qpHandle = {0};
    int ret = 0;

    ret = RaPeerCtxQpBind(&qpHandle, &remQpHandle);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsCtxQpBind, 1, -1);
    ret = RaPeerCtxQpBind(&qpHandle, &remQpHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaPeerCtxQpUnbind()
{
    struct RaCtxQpHandle qpHandle = {0};
    int ret = 0;

    ret = RaPeerCtxQpUnbind(&qpHandle);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsCtxQpUnbind, 1, -1);
    ret = RaPeerCtxQpUnbind(&qpHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaCtxQpDestroyBatchAsync()
{
    struct RaCtxQpHandle* qpHandle[1] = {0};
    struct RaRequestHandle* reqHandle = NULL;
    struct RaCtxHandle ctxHandle = {0};
    unsigned int num = 0;
    int ret;

    mocker_clean();
    ret = RaCtxQpDestroyBatchAsync(NULL, (void**)qpHandle, &num, (void**)&reqHandle);
    EXPECT_INT_NE(0, ret);

    ret = RaCtxQpDestroyBatchAsync(&ctxHandle, NULL, &num, (void**)&reqHandle);
    EXPECT_INT_NE(0, ret);

    ret = RaCtxQpDestroyBatchAsync(&ctxHandle, (void**)qpHandle, NULL, (void**)&reqHandle);
    EXPECT_INT_NE(0, ret);

    ret = RaCtxQpDestroyBatchAsync(&ctxHandle, (void**)qpHandle, &num, NULL);
    EXPECT_INT_NE(0, ret);

    ret = RaCtxQpDestroyBatchAsync(&ctxHandle, (void**)qpHandle, &num, (void**)&reqHandle);
    EXPECT_INT_NE(0, ret);

    num = HCCP_MAX_QP_DESTROY_BATCH_NUM + 1;
    ret = RaCtxQpDestroyBatchAsync(&ctxHandle, (void**)qpHandle, &num, (void**)&reqHandle);
    EXPECT_INT_NE(0, ret);

    num = 1;
    qpHandle[0] = (struct RaCtxQpHandle*)calloc(1, sizeof(struct RaCtxQpHandle));
    mocker(RaHdcCtxQpDestroyBatchAsync, 1, -1);
    ret = RaCtxQpDestroyBatchAsync(&ctxHandle, (void**)qpHandle, &num, (void**)&reqHandle);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    qpHandle[0] = (struct RaCtxQpHandle*)calloc(1, sizeof(struct RaCtxQpHandle));
    mocker(RaHdcCtxQpDestroyBatchAsync, 1, 0);
    ret = RaCtxQpDestroyBatchAsync(&ctxHandle, (void**)qpHandle, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcQpDestroyBatchParamCheck()
{
    struct RaCtxQpHandle qpHandleTmp;
    struct RaCtxHandle ctxHandle = {0};
    unsigned int num = 1;
    unsigned int ids[1];
    void* qpHandle[1];
    int ret;

    mocker_clean();
    qpHandleTmp.ctxHandle = &ctxHandle;
    qpHandleTmp.id = 123;
    qpHandle[0] = &qpHandleTmp;
    ret = QpDestroyBatchParamCheck(&ctxHandle, qpHandle, ids, &num);
    EXPECT_INT_EQ(0, ret);

    qpHandle[0] = NULL;
    ret = QpDestroyBatchParamCheck(&ctxHandle, qpHandle, ids, &num);
    EXPECT_INT_EQ(-EINVAL, ret);

    qpHandleTmp.ctxHandle = NULL;
    qpHandle[0] = &qpHandleTmp;
    ret = QpDestroyBatchParamCheck(&ctxHandle, qpHandle, ids, &num);
    EXPECT_INT_EQ(-EINVAL, ret);

    qpHandleTmp.ctxHandle = (struct RaCtxHandle*)0x1234;
    ret = QpDestroyBatchParamCheck(&ctxHandle, qpHandle, ids, &num);
    EXPECT_INT_EQ(-EINVAL, ret);
}

void TcRaHdcCtxQpDestroyBatchAsync()
{
    union OpCtxQpDestroyBatchData recvBuf = {0};
    struct RaRequestHandle* reqHandle = NULL;
    struct RaCtxQpHandle qpHandleTmp;
    struct RaCtxHandle ctxHandle = {0};
    unsigned int num = 1;
    void* qpHandle[1];
    int ret;

    mocker_clean();
    qpHandleTmp.ctxHandle = &ctxHandle;
    qpHandle[0] = &qpHandleTmp;

    mocker(QpDestroyBatchParamCheck, 1, -EINVAL);
    ret = RaHdcCtxQpDestroyBatchAsync(&ctxHandle, qpHandle, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();

    mocker(calloc, 1, NULL);
    ret = RaHdcCtxQpDestroyBatchAsync(&ctxHandle, qpHandle, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(-ENOMEM, ret);
    mocker_clean();

    mocker(QpDestroyBatchParamCheck, 1, 0);
    mocker(RaHdcSendMsgAsync, 1, -EINVAL);
    ret = RaHdcCtxQpDestroyBatchAsync(&ctxHandle, qpHandle, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();

    mocker(RaHdcSendMsgAsync, 1, 0);
    ret = RaHdcCtxQpDestroyBatchAsync(&ctxHandle, qpHandle, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    reqHandle->recvBuf = &recvBuf;
    RaHdcAsyncHandleQpDestroyBatch(reqHandle);
    free(reqHandle);
    reqHandle = NULL;
}

void TcRaRsCtxQpDestroyBatch()
{
    union OpCtxQpDestroyBatchData dataIn;
    union OpCtxQpDestroyBatchData dataOut;
    int rcvBufLen = 0;
    int opResult = 0;
    int outLen = 0;
    int ret;

    char* inBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpCtxQpDestroyBatchData));
    char* outBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpCtxQpDestroyBatchData));

    mocker_clean();
    dataIn.txData.phyId = 0;
    dataIn.txData.devIndex = 0;
    memcpy_s(
        inBuf + sizeof(struct MsgHead), sizeof(union OpCtxQpDestroyBatchData), &dataIn,
        sizeof(union OpCtxQpDestroyBatchData));
    ret = RaRsCtxQpDestroyBatch(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker(RsCtxQpDestroyBatch, 1, -1);
    ret = RaRsCtxQpDestroyBatch(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    free(inBuf);
    inBuf = NULL;
    free(outBuf);
    outBuf = NULL;
}

void TcRaCtxQpQueryBatch()
{
    struct RaCtxQpHandle qpHandleTmp = {0};
    struct RaCtxQpHandle* qpHandle[2];
    struct RaCtxHandle ctxHandle = {0};
    struct JettyAttr attr[2];
    unsigned int num;
    int ret;

    mocker_clean();
    ret = RaCtxQpQueryBatch(NULL, attr, &num);
    EXPECT_INT_NE(0, ret);

    ret = RaCtxQpQueryBatch((void**)qpHandle, NULL, &num);
    EXPECT_INT_NE(0, ret);

    ret = RaCtxQpQueryBatch((void**)qpHandle, attr, NULL);
    EXPECT_INT_NE(0, ret);

    mocker(QpQueryBatchParamCheck, 1, -1);
    ret = RaCtxQpQueryBatch((void**)qpHandle, attr, &num);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    ctxHandle.ctxOps = &gRaHdcCtxOps;
    qpHandleTmp.ctxHandle = &ctxHandle;
    qpHandle[0] = &qpHandleTmp;
    qpHandle[1] = &qpHandleTmp;
    num = 2;
    mocker(QpQueryBatchParamCheck, 1, 0);
    mocker(RaHdcCtxQpQueryBatch, 1, -1);
    ret = RaCtxQpQueryBatch((void**)qpHandle, attr, &num);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(QpQueryBatchParamCheck, 1, 0);
    mocker(RaHdcCtxQpQueryBatch, 1, 0);
    ret = RaCtxQpQueryBatch((void**)qpHandle, attr, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcQpQueryBatchParamCheck()
{
    struct RaCtxQpHandle qpHandle1 = {0};
    struct RaCtxQpHandle qpHandle2 = {0};
    struct RaCtxHandle ctxHandle = {0};
    struct RaCtxOps ctxOpsTmp = {0};
    unsigned int phyId = 1;
    unsigned int num = 2;
    void* qpHandles[2];
    unsigned int ids[2];
    int ret;

    qpHandle1.id = 1;
    qpHandle2.id = 2;
    qpHandle1.phyId = 1;
    qpHandle2.phyId = 1;
    qpHandles[0] = &qpHandle1;
    qpHandles[1] = &qpHandle2;
    ctxHandle.ctxOps = &gRaHdcCtxOps;
    qpHandle1.ctxHandle = &ctxHandle;
    qpHandle2.ctxHandle = &ctxHandle;

    ret = QpQueryBatchParamCheck(qpHandles, &num, phyId, ids);
    EXPECT_INT_EQ(0, ret);

    qpHandles[0] = NULL;
    ret = QpQueryBatchParamCheck(qpHandles, &num, phyId, ids);
    EXPECT_INT_NE(0, ret);

    qpHandles[0] = &qpHandle1;
    qpHandle1.phyId = 2;
    ret = QpQueryBatchParamCheck(qpHandles, &num, phyId, ids);
    EXPECT_INT_NE(0, ret);

    qpHandle1.phyId = 1;
    qpHandle1.ctxHandle = NULL;
    ret = QpQueryBatchParamCheck(qpHandles, &num, phyId, ids);
    EXPECT_INT_NE(0, ret);

    ctxHandle.ctxOps = NULL;
    qpHandle1.ctxHandle = &ctxHandle;
    ret = QpQueryBatchParamCheck(qpHandles, &num, phyId, ids);
    EXPECT_INT_NE(0, ret);

    ctxHandle.ctxOps = &ctxOpsTmp;
    ret = QpQueryBatchParamCheck(qpHandles, &num, phyId, ids);
    EXPECT_INT_NE(0, ret);
}

void TcRaHdcCtxQpQueryBatch()
{
    unsigned int ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    struct JettyAttr attr[10] = {0};
    unsigned int devIndex = 2;
    unsigned int phyId = 1;
    unsigned int num = 10;
    int ret;

    mocker_clean();
    mocker(RaHdcProcessMsg, 1, 0);
    mocker(memcpy_s, 2, 0);
    ret = RaHdcCtxQpQueryBatch(phyId, devIndex, ids, attr, &num);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();

    mocker(RaHdcProcessMsg, 1, -1);
    ret = RaHdcCtxQpQueryBatch(phyId, devIndex, ids, attr, &num);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();

    mocker_invoke(RaHdcProcessMsg, RaHdcProcessMsgStub, 1);
    ret = RaHdcCtxQpQueryBatch(phyId, devIndex, ids, attr, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRaRsCtxQpQueryBatch()
{
    union OpCtxQpQueryBatchData dataIn;
    union OpCtxQpQueryBatchData dataOut;
    int rcvBufLen = 0;
    int opResult = 0;
    int outLen = 0;
    int ret;

    char* inBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpCtxQpQueryBatchData));
    char* outBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpCtxQpQueryBatchData));

    mocker_clean();
    dataIn.txData.phyId = 0;
    dataIn.txData.devIndex = 0;
    memcpy_s(
        inBuf + sizeof(struct MsgHead), sizeof(union OpCtxQpQueryBatchData), &dataIn,
        sizeof(union OpCtxQpQueryBatchData));
    ret = RaRsCtxQpQueryBatch(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsCtxQpQueryBatch, 1, -1);
    ret = RaRsCtxQpQueryBatch(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    free(inBuf);
    inBuf = NULL;
    free(outBuf);
    outBuf = NULL;
}

void TcRaGetEidByIp()
{
    struct RaCtxHandle ctxHandle = {0};
    union HccpEid eid[32] = {0};
    struct IpInfo ip[32] = {0};
    unsigned int num = 32;
    int ret = 0;

    ret = RaGetEidByIp(NULL, ip, eid, &num);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    num = 33;
    ret = RaGetEidByIp(&ctxHandle, ip, eid, &num);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    num = 32;
    ret = RaGetEidByIp(&ctxHandle, ip, eid, &num);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    ctxHandle.ctxOps = &gRaHdcCtxOps;
    mocker(RaHdcGetEidByIp, 1, 0);
    ret = RaGetEidByIp(&ctxHandle, ip, eid, &num);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker(RaHdcGetEidByIp, 1, -1);
    ret = RaGetEidByIp(&ctxHandle, ip, eid, &num);
    EXPECT_INT_EQ(ret, 128100);
    mocker_clean();
}

void TcRaHdcGetEidByIp()
{
    struct RaCtxHandle ctxHandle = {0};
    union HccpEid eid[32] = {0};
    struct IpInfo ip[32] = {0};
    unsigned int num = 32;
    int ret = 0;

    mocker(RaHdcProcessMsg, 1, -1);
    mocker(RaHdcGetEidResults, 1, 0);
    ret = RaHdcGetEidByIp(&ctxHandle, ip, eid, &num);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RaHdcProcessMsg, 1, 0);
    mocker(RaHdcGetEidResults, 1, 0);
    ret = RaHdcGetEidByIp(&ctxHandle, ip, eid, &num);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRaRsGetEidByIp()
{
    union OpGetEidByIpData dataOut = {0};
    union OpGetEidByIpData dataIn = {0};

    int rcvBufLen = 0;
    int opResult = 0;
    int outLen = 0;
    int ret;

    char* inBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpGetEidByIpData));
    char* outBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpGetEidByIpData));

    dataIn.txData.phyId = 0;
    dataIn.txData.devIndex = 0;
    memcpy_s(inBuf + sizeof(struct MsgHead), sizeof(union OpGetEidByIpData), &dataIn, sizeof(union OpGetEidByIpData));
    dataIn.txData.num = 32;
    mocker(RsGetEidByIp, 1, 0);
    ret = RaRsGetEidByIp(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(opResult, 0);
    mocker_clean();

    mocker(RsGetEidByIp, 1, -1);
    ret = RaRsGetEidByIp(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(opResult, -1);
    mocker_clean();

    free(inBuf);
    inBuf = NULL;
    free(outBuf);
    outBuf = NULL;
}

void TcRaPeerGetEidByIp()
{
    struct RaCtxHandle ctxHandle = {0};
    union HccpEid eid[32] = {0};
    struct IpInfo ip[32] = {0};
    unsigned int num = 32;
    int ret = 0;

    ret = RaPeerGetEidByIp(&ctxHandle, ip, eid, &num);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsGetEidByIp, 1, -1);
    ret = RaPeerGetEidByIp(&ctxHandle, ip, eid, &num);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaGetIpByEid()
{
    struct RaCtxHandle ctxHandle = {0};
    union HccpEid eid[32] = {0};
    struct IpInfo ip[32] = {0};
    unsigned int num = 32;
    int ret = 0;

    ret = RaGetIpByEid(NULL, eid, ip, &num);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    num = 33;
    ret = RaGetIpByEid(&ctxHandle, eid, ip, &num);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    num = 32;
    ret = RaGetIpByEid(&ctxHandle, eid, ip, &num);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    ctxHandle.ctxOps = &gRaHdcCtxOps;
    mocker(RaHdcGetIpByEid, 1, 0);
    ret = RaGetIpByEid(&ctxHandle, eid, ip, &num);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker(RaHdcGetIpByEid, 1, -1);
    ret = RaGetIpByEid(&ctxHandle, eid, ip, &num);
    EXPECT_INT_EQ(ret, 128100);
    mocker_clean();
}

void TcRaHdcGetIpByEid()
{
    struct RaCtxHandle ctxHandle = {0};
    union HccpEid eid[32] = {0};
    struct IpInfo ip[32] = {0};
    unsigned int num = 32;
    int ret = 0;

    mocker(RaHdcProcessMsg, 1, -1);
    mocker(RaHdcGetIpResults, 1, 0);
    ret = RaHdcGetIpByEid(&ctxHandle, eid, ip, &num);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RaHdcProcessMsg, 1, 0);
    mocker(RaHdcGetIpResults, 1, 0);
    ret = RaHdcGetIpByEid(&ctxHandle, eid, ip, &num);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRaRsGetIpByEid()
{
    union OpGetIpByEidData dataOut = {0};
    union OpGetIpByEidData dataIn = {0};

    int rcvBufLen = 0;
    int opResult = 0;
    int outLen = 0;
    int ret;

    char* inBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpGetIpByEidData));
    char* outBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpGetIpByEidData));

    dataIn.txData.phyId = 0;
    dataIn.txData.devIndex = 0;
    memcpy_s(inBuf + sizeof(struct MsgHead), sizeof(union OpGetIpByEidData), &dataIn, sizeof(union OpGetIpByEidData));
    dataIn.txData.num = 32;
    mocker(RsGetIpByEid, 1, 0);
    ret = RaRsGetIpByEid(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(opResult, 0);
    mocker_clean();

    mocker(RsGetIpByEid, 1, -1);
    ret = RaRsGetIpByEid(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(opResult, -1);
    mocker_clean();

    free(inBuf);
    inBuf = NULL;
    free(outBuf);
    outBuf = NULL;
}

void TcRaPeerGetIpByEid()
{
    struct RaCtxHandle ctxHandle = {0};
    union HccpEid eid[32] = {0};
    struct IpInfo ip[32] = {0};
    unsigned int num = 32;
    int ret = 0;

    ret = RaPeerGetIpByEid(&ctxHandle, eid, ip, &num);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsGetIpByEid, 1, -1);
    ret = RaPeerGetIpByEid(&ctxHandle, eid, ip, &num);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaCtxGetAuxInfo()
{
    struct RaCtxHandle ctxHandle = {0};
    struct HccpAuxInfoOut out;
    struct HccpAuxInfoIn in;
    int ret = 0;

    ret = RaCtxGetAuxInfo(NULL, &in, &out);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    in.type = AUX_INFO_IN_TYPE_MAX;
    ret = RaCtxGetAuxInfo(&ctxHandle, &in, &out);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    in.type = AUX_INFO_IN_TYPE_MAX - 1;
    ret = RaCtxGetAuxInfo(&ctxHandle, &in, &out);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    mocker_clean();
    mocker(RaHdcCtxGetAuxInfo, 1, -1);
    ctxHandle.ctxOps = &gRaHdcCtxOps;
    ret = RaCtxGetAuxInfo(&ctxHandle, &in, &out);
    EXPECT_INT_EQ(ret, 128100);
    mocker_clean();

    mocker(RaHdcCtxGetAuxInfo, 1, 0);
    ctxHandle.ctxOps = &gRaHdcCtxOps;
    ret = RaCtxGetAuxInfo(&ctxHandle, &in, &out);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRaHdcCtxGetAuxInfo()
{
    struct RaCtxHandle ctxHandle = {0};
    struct HccpAuxInfoOut out;
    struct HccpAuxInfoIn in;
    int ret = 0;

    mocker_clean();
    mocker(RaHdcProcessMsg, 1, -1);
    ret = RaHdcCtxGetAuxInfo(&ctxHandle, &in, &out);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RaHdcProcessMsg, 1, 0);
    ret = RaHdcCtxGetAuxInfo(&ctxHandle, &in, &out);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRaRsCtxGetAuxInfo()
{
    union OpCtxGetAuxInfoData dataOut = {0};
    union OpCtxGetAuxInfoData dataIn = {0};

    int rcvBufLen = 0;
    int opResult = 0;
    int outLen = 0;
    int ret;

    char* inBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpCtxGetAuxInfoData));
    char* outBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpCtxGetAuxInfoData));

    dataIn.txData.phyId = 0;
    dataIn.txData.devIndex = 0;
    memcpy_s(
        inBuf + sizeof(struct MsgHead), sizeof(union OpCtxGetAuxInfoData), &dataIn, sizeof(union OpCtxGetAuxInfoData));
    mocker(RsCtxGetAuxInfo, 1, 0);
    ret = RaRsCtxGetAuxInfo(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(opResult, 0);
    mocker_clean();

    mocker(RsCtxGetAuxInfo, 1, -1);
    ret = RaRsCtxGetAuxInfo(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(opResult, -1);
    mocker_clean();

    free(inBuf);
    inBuf = NULL;
    free(outBuf);
    outBuf = NULL;
}

void TcRaGetTpAttrAsync()
{
    struct RaRequestHandle* reqHandle = NULL;
    struct RaCtxHandle ctxHandle = {0};
    struct TpAttr attr = {0};
    uint32_t attrBitmap = 1;
    uint64_t tpHandle = 0;
    int ret;

    mocker_clean();
    ret = RaGetTpAttrAsync(NULL, tpHandle, &attrBitmap, &attr, (void**)&reqHandle);
    EXPECT_INT_NE(0, ret);

    ret = RaGetTpAttrAsync(&ctxHandle, tpHandle, NULL, &attr, (void**)&reqHandle);
    EXPECT_INT_NE(0, ret);

    ret = RaGetTpAttrAsync(&ctxHandle, tpHandle, &attrBitmap, NULL, (void**)&reqHandle);
    EXPECT_INT_NE(0, ret);

    ret = RaGetTpAttrAsync(&ctxHandle, tpHandle, &attrBitmap, &attr, NULL);
    EXPECT_INT_NE(0, ret);

    mocker(RaHdcGetTpAttrAsync, 1, -1);
    ret = RaGetTpAttrAsync(&ctxHandle, tpHandle, &attrBitmap, &attr, (void**)&reqHandle);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(RaHdcGetTpAttrAsync, 1, 0);
    ret = RaGetTpAttrAsync(&ctxHandle, tpHandle, &attrBitmap, &attr, (void**)&reqHandle);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRaHdcGetTpAttrAsync()
{
    struct RaResponseGetTpAttr* asyncRsp = NULL;
    union OpGetTpAttrData* asyncData = NULL;
    struct RaRequestHandle* reqHandle = NULL;
    union OpGetTpAttrData recvBuf = {0};
    struct RaCtxHandle ctxHandle = {0};
    uint64_t tpHandle = 1234;
    uint32_t attrBitmap = 0;
    struct TpAttr attr = {0};
    int ret;

    mocker_clean();
    mocker(calloc, 2, NULL);
    ret = RaHdcGetTpAttrAsync(&ctxHandle, tpHandle, &attrBitmap, &attr, (void**)&reqHandle);
    EXPECT_INT_EQ(-ENOMEM, ret);
    mocker_clean();

    mocker(RaHdcSendMsgAsync, 1, -1);
    ret = RaHdcGetTpAttrAsync(&ctxHandle, tpHandle, &attrBitmap, &attr, (void**)&reqHandle);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RaHdcSendMsgAsync, 1, 0);
    ret = RaHdcGetTpAttrAsync(&ctxHandle, tpHandle, &attrBitmap, &attr, (void**)&reqHandle);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    reqHandle->opRet = 0;
    reqHandle->recvBuf = &recvBuf;
    mocker(memcpy_s, 1, 0);
    RaHdcAsyncHandleGetTpAttr(reqHandle);
    mocker_clean();

    reqHandle->opRet = -1;
    asyncRsp = (struct RaResponseGetTpAttr*)calloc(1, sizeof(struct RaResponseGetTpAttr));
    asyncRsp->attr = &attr;
    asyncRsp->attrBitmap = &attrBitmap;
    reqHandle->privData = (void*)asyncRsp;
    RaHdcAsyncHandleGetTpAttr((struct RaRequestHandle*)reqHandle);
    free(reqHandle);
    reqHandle = NULL;
}

void TcRaRsGetTpAttr()
{
    union OpGetTpAttrData dataIn;
    union OpGetTpAttrData dataOut;
    int rcvBufLen = 0;
    int opResult = 0;
    int outLen = 0;
    int ret;

    char* inBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpGetTpAttrData));
    char* outBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpGetTpAttrData));

    mocker_clean();
    dataIn.txData.phyId = 0;
    dataIn.txData.devIndex = 0;
    memcpy_s(inBuf + sizeof(struct MsgHead), sizeof(union OpGetTpAttrData), &dataIn, sizeof(union OpGetTpAttrData));
    ret = RaRsGetTpAttr(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsGetTpAttr, 1, -1);
    ret = RaRsGetTpAttr(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    free(inBuf);
    inBuf = NULL;
    free(outBuf);
    outBuf = NULL;
}

void TcRaSetTpAttrAsync()
{
    struct RaRequestHandle* reqHandle = NULL;
    struct RaCtxHandle ctxHandle = {0};
    struct TpAttr attr = {0};
    uint32_t attrBitmap = 1;
    uint64_t tpHandle = 0;
    int ret;

    mocker_clean();
    ret = RaSetTpAttrAsync(NULL, tpHandle, attrBitmap, &attr, (void**)&reqHandle);
    EXPECT_INT_NE(0, ret);

    ret = RaSetTpAttrAsync(&ctxHandle, tpHandle, attrBitmap, NULL, (void**)&reqHandle);
    EXPECT_INT_NE(0, ret);

    ret = RaSetTpAttrAsync(&ctxHandle, tpHandle, attrBitmap, &attr, NULL);
    EXPECT_INT_NE(0, ret);

    mocker(RaHdcSetTpAttrAsync, 1, -1);
    ret = RaSetTpAttrAsync(&ctxHandle, tpHandle, attrBitmap, &attr, (void**)&reqHandle);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(RaHdcSetTpAttrAsync, 1, 0);
    ret = RaSetTpAttrAsync(&ctxHandle, tpHandle, attrBitmap, &attr, (void**)&reqHandle);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRaHdcSetTpAttrAsync()
{
    struct RaRequestHandle* reqHandle = NULL;
    struct RaCtxHandle ctx = {0};
    uint64_t tpHandle = 1234;
    uint32_t attrBitmap = 0;
    struct TpAttr attr = {0};

    mocker_clean();
    mocker(calloc, 2, NULL);
    int ret = RaHdcSetTpAttrAsync(&ctx, tpHandle, attrBitmap, &attr, (void**)&reqHandle);
    EXPECT_INT_EQ(-ENOMEM, ret);
    mocker_clean();

    mocker(RaHdcSendMsgAsync, 1, -1);
    ret = RaHdcSetTpAttrAsync(&ctx, tpHandle, attrBitmap, &attr, (void**)&reqHandle);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RaHdcSendMsgAsync, 1, 0);
    ret = RaHdcSetTpAttrAsync(&ctx, tpHandle, attrBitmap, &attr, (void**)&reqHandle);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    free(reqHandle);
    reqHandle = NULL;
}

void TcRaRsSetTpAttr()
{
    union OpSetTpAttrData dataIn;
    union OpSetTpAttrData dataOut;
    int rcvBufLen = 0;
    int opResult = 0;
    int outLen = 0;
    int ret;

    char* inBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpSetTpAttrData));
    char* outBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpSetTpAttrData));

    mocker_clean();
    dataIn.txData.phyId = 0;
    dataIn.txData.devIndex = 0;
    memcpy_s(inBuf + sizeof(struct MsgHead), sizeof(union OpSetTpAttrData), &dataIn, sizeof(union OpSetTpAttrData));
    ret = RaRsSetTpAttr(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker(RsSetTpAttr, 1, -1);
    ret = RaRsSetTpAttr(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    free(inBuf);
    inBuf = NULL;
    free(outBuf);
    outBuf = NULL;
}

void TcRaCtxGetCrErrInfoList()
{
    struct RaCtxHandle ctxHandle = {0};
    struct CrErrInfo infoList[1] = {0};
    unsigned int num = 0;
    int ret = 0;

    mocker_clean();
    ret = RaCtxGetCrErrInfoList(NULL, NULL, NULL);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    ret = RaCtxGetCrErrInfoList(&ctxHandle, NULL, NULL);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    ret = RaCtxGetCrErrInfoList(&ctxHandle, infoList, NULL);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    ret = RaCtxGetCrErrInfoList(&ctxHandle, infoList, &num);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    num = CR_ERR_INFO_MAX_NUM + 1;
    ret = RaCtxGetCrErrInfoList(&ctxHandle, infoList, &num);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    num = 1;
    mocker(RaHdcCtxGetCrErrInfoList, 1, -1);
    ret = RaCtxGetCrErrInfoList(&ctxHandle, infoList, &num);
    EXPECT_INT_EQ(ret, 128100);
    mocker_clean();

    mocker(RaHdcCtxGetCrErrInfoList, 1, 0);
    ret = RaCtxGetCrErrInfoList(&ctxHandle, infoList, &num);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRaHdcCtxGetCrErrInfoList()
{
    struct RaCtxHandle ctxHandle = {0};
    struct CrErrInfo infoList[1] = {0};
    unsigned int num = 1;
    int ret = 0;

    mocker_clean();
    mocker(RaHdcProcessMsg, 1, -1);
    ret = RaHdcCtxGetCrErrInfoList(&ctxHandle, infoList, &num);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RaHdcProcessMsg, 1, 0);
    ret = RaHdcCtxGetCrErrInfoList(&ctxHandle, infoList, &num);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRaRsCtxGetCrErrInfoList()
{
    union OpCtxGetCrErrInfoListData dataOut = {0};
    union OpCtxGetCrErrInfoListData dataIn = {0};

    int rcvBufLen = 0;
    int opResult = 0;
    int outLen = 0;
    int ret;

    char* inBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpCtxGetCrErrInfoListData));
    char* outBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpCtxGetCrErrInfoListData));

    dataIn.txData.phyId = 0;
    dataIn.txData.devIndex = 0;
    memcpy_s(
        inBuf + sizeof(struct MsgHead), sizeof(union OpCtxGetCrErrInfoListData), &dataIn,
        sizeof(union OpCtxGetCrErrInfoListData));
    dataIn.txData.num = CQE_ERR_INFO_MAX_NUM;
    mocker(RsCtxGetCrErrInfoList, 1, 0);
    ret = RaRsCtxGetCrErrInfoList(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(opResult, 0);
    mocker_clean();

    mocker(RsCtxGetCrErrInfoList, 1, -1);
    ret = RaRsCtxGetCrErrInfoList(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(opResult, -1);
    mocker_clean();

    free(inBuf);
    inBuf = NULL;
    free(outBuf);
    outBuf = NULL;
}

void TcRaCtxGetTpInfoList()
{
    struct HccpTpInfo infoList[HCCP_MAX_TPID_INFO_NUM] = {0};
    struct RaCtxHandle ctxHandle = {0};
    struct GetTpCfg cfg = {0};
    unsigned int num = 0;
    int ret = 0;

    ret = RaCtxGetTpInfoList(NULL, &cfg, infoList, &num);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    ret = RaCtxGetTpInfoList(&ctxHandle, &cfg, NULL, &num);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    ret = RaCtxGetTpInfoList(&ctxHandle, &cfg, infoList, &num);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    num = 1;
    ret = RaCtxGetTpInfoList(&ctxHandle, &cfg, infoList, &num);
    EXPECT_INT_EQ(ret, 0);
}

void TcRaPeerCtxGetTpInfoList()
{
    struct HccpTpInfo infoList[HCCP_MAX_TPID_INFO_NUM] = {0};
    struct RaCtxHandle ctxHandle = {0};
    struct GetTpCfg cfg = {0};
    unsigned int num = 1;
    int ret = 0;

    mocker_clean();
    mocker(RsGetTpInfoList, 1, -1);
    ret = RaPeerCtxGetTpInfoList(&ctxHandle, &cfg, infoList, &num);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaCtxGetTpAttr()
{
    struct RaCtxHandle ctxHandle = {0};
    struct TpAttr attr = {0};
    uint32_t attrBitmap = 0;
    uint64_t tpHandle = 0;
    int ret = 0;

    ret = RaCtxGetTpAttr(NULL, tpHandle, &attrBitmap, &attr);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    ret = RaCtxGetTpAttr(&ctxHandle, tpHandle, &attrBitmap, &attr);
    EXPECT_INT_EQ(ret, 0);

    mocker_clean();
    mocker(RaPeerCtxGetTpAttr, 1, -1);
    ret = RaCtxGetTpAttr(&ctxHandle, tpHandle, &attrBitmap, &attr);
    EXPECT_INT_EQ(ret, 128100);
    mocker_clean();
}

void TcRaPeerCtxGetTpAttr()
{
    struct RaCtxHandle ctxHandle = {0};
    struct TpAttr attr = {0};
    uint32_t attrBitmap = 0;
    uint64_t tpHandle = 0;
    int ret = 0;

    mocker_clean();
    mocker(RsGetTpAttr, 1, -1);
    ret = RaPeerCtxGetTpAttr(&ctxHandle, tpHandle, &attrBitmap, &attr);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaCtxSetTpAttr()
{
    struct RaCtxHandle ctxHandle = {0};
    struct TpAttr attr = {0};
    uint32_t attrBitmap = 0;
    uint64_t tpHandle = 0;
    int ret = 0;

    ret = RaCtxSetTpAttr(NULL, tpHandle, attrBitmap, &attr);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    ret = RaCtxSetTpAttr(&ctxHandle, tpHandle, attrBitmap, &attr);
    EXPECT_INT_EQ(ret, 0);

    mocker_clean();
    mocker(RaPeerCtxSetTpAttr, 1, -1);
    ret = RaCtxSetTpAttr(&ctxHandle, tpHandle, attrBitmap, &attr);
    EXPECT_INT_EQ(ret, 128100);
    mocker_clean();
}

void TcRaPeerCtxSetTpAttr()
{
    struct RaCtxHandle ctxHandle = {0};
    struct TpAttr attr = {0};
    uint32_t attrBitmap = 0;
    uint64_t tpHandle = 0;
    int ret = 0;

    mocker_clean();
    mocker(RsSetTpAttr, 1, -1);
    ret = RaPeerCtxSetTpAttr(&ctxHandle, tpHandle, attrBitmap, &attr);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaCtxGetJettyContext()
{
    uint8_t context[CONTEXT_MAX_LEN];
    struct RaCtxQpHandle qpHandle = {0};
    struct RaCtxHandle ctxHandle = {0};
    struct RaCtxOps ctxOps = {0};
    unsigned int len = 0;
    int ret = 0;

    mocker_clean();
    ret = RaCtxGetJettyContext(NULL, NULL, NULL);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    ret = RaCtxGetJettyContext(&qpHandle, NULL, NULL);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    ret = RaCtxGetJettyContext(&qpHandle, context, NULL);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    ret = RaCtxGetJettyContext(&qpHandle, context, &len);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    len = CONTEXT_MAX_LEN;
    ret = RaCtxGetJettyContext(&qpHandle, context, &len);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    ctxHandle.ctxOps = NULL;
    qpHandle.ctxHandle = &ctxHandle;
    ret = RaCtxGetJettyContext(&qpHandle, context, &len);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);
    mocker_clean();

    mocker(RaHdcCtxGetJettyContext, 1, -EINVAL);
    ctxHandle.ctxOps = &gRaHdcCtxOps;
    ret = RaCtxGetJettyContext(&qpHandle, context, &len);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);
    mocker_clean();

    mocker(RaHdcCtxGetJettyContext, 1, 0);
    ctxHandle.ctxOps = &gRaHdcCtxOps;
    ret = RaCtxGetJettyContext(&qpHandle, context, &len);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRaHdcCtxGetJettyContext()
{
    struct RaCtxQpHandle qpHandle = {0};
    uint8_t context[CONTEXT_MAX_LEN];
    unsigned int len = 0;
    int ret = 0;

    mocker_clean();
    mocker(RaHdcProcessMsg, 1, -1);
    ret = RaHdcCtxGetJettyContext(&qpHandle, context, &len);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RaHdcProcessMsg, 1, 0);
    mocker(memcpy_s, 1, -1);
    ret = RaHdcCtxGetJettyContext(&qpHandle, context, &len);
    EXPECT_INT_EQ(ret, -ESAFEFUNC);
    mocker_clean();

    mocker(RaHdcProcessMsg, 1, 0);
    mocker(memcpy_s, 1, 0);
    ret = RaHdcCtxGetJettyContext(&qpHandle, context, &len);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRaPeerCtxGetJettyContext()
{
    struct RaCtxQpHandle qpHandle = {0};
    uint8_t context[CONTEXT_MAX_LEN];
    unsigned int len = 0;
    int ret = 0;

    mocker_clean();
    ret = RaPeerCtxGetJettyContext(&qpHandle, context, &len);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker(RsCtxGetUbContext, 1, -1);
    ret = RaPeerCtxGetJettyContext(&qpHandle, context, &len);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaCtxNotifyEvent()
{
    struct RaCtxHandle ctxHandle = {0};
    struct CtxNotifyEvent event = {0};
    int ret = 0;

    mocker_clean();
    ret = RaCtxNotifyEvent(NULL, &event);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    ret = RaCtxNotifyEvent(&ctxHandle, NULL);
    EXPECT_INT_EQ(ret, ROCE_EINVAL);

    ctxHandle.ctxOps = &gRaHdcCtxOps;
    mocker(RaHdcHasCapability, 1, 0);
    ret = RaCtxNotifyEvent(&ctxHandle, &event);
    EXPECT_INT_EQ(ret, ROCE_ENOTSUPP);
    mocker_clean();

    mocker(RaHdcHasCapability, 1, 1);
    mocker(RaHdcCtxNotifyEvent, 1, -1);
    ret = RaCtxNotifyEvent(&ctxHandle, &event);
    EXPECT_INT_EQ(ret, 128100);
    mocker_clean();

    mocker(RaHdcHasCapability, 1, 1);
    mocker(RaHdcCtxNotifyEvent, 1, 0);
    ret = RaCtxNotifyEvent(&ctxHandle, &event);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRaHdcCtxNotifyEvent()
{
    struct RaCtxHandle ctxHandle = {0};
    struct CtxNotifyEvent event = {0};
    int ret = 0;

    mocker_clean();
    mocker(RaHdcProcessMsg, 1, -1);
    ret = RaHdcCtxNotifyEvent(&ctxHandle, &event);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RaHdcProcessMsg, 1, 0);
    ret = RaHdcCtxNotifyEvent(&ctxHandle, &event);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}
