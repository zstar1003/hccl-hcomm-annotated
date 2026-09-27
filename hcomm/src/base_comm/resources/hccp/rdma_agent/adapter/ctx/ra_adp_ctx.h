/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_ADP_CTX_H
#define RA_ADP_CTX_H

#include "hccp_async_ctx.h"
#include "ra_rs_comm.h"
#include "ra_rs_ctx.h"
#include "rs_ctx.h"

struct RsCtxOps {
    int (*getDevEidInfoNum)(unsigned int phyId, unsigned int *num);
    int (*getDevEidInfoList)(unsigned int phyId, struct HccpDevEidInfo infoList[], unsigned int startIndex,
        unsigned int count);
    int (*ctxInit)(struct CtxInitAttr *attr, unsigned int *devIndex, struct DevBaseAttr *devAttr);
    int (*ctxGetAsyncEvents)(struct RaRsDevInfo *devInfo, struct AsyncEvent asyncEvents[], unsigned int *num);
    int (*ctxDeinit)(struct RaRsDevInfo *devInfo);
    int (*getEidByIp)(struct RaRsDevInfo *devInfo, struct IpInfo ip[], union HccpEid eid[], unsigned int *num);
    int (*getIpByEid)(struct RaRsDevInfo *devInfo, union HccpEid eid[], struct IpInfo ip[], unsigned int *num);
    int (*getTpInfoList)(struct RaRsDevInfo *devInfo, struct GetTpCfg *cfg, struct HccpTpInfo infoList[],
        unsigned int *num);
    int (*getTpAttr)(struct RaRsDevInfo *devInfo, unsigned int *attrBitmap, const uint64_t tpHandle,
        struct TpAttr *attr);
    int (*setTpAttr)(struct RaRsDevInfo *devInfo, const unsigned int attrBitmap, const uint64_t tpHandle,
        struct TpAttr *attr);
    int (*ctxTokenIdAlloc)(struct RaRsDevInfo *devInfo, unsigned long long *addr, unsigned int *tokenId);
    int (*ctxTokenIdFree)(struct RaRsDevInfo *devInfo, unsigned long long addr);
    int (*ctxLmemReg)(struct RaRsDevInfo *devInfo, struct MemRegAttrT *memAttr, struct MemRegInfoT *memInfo);
    int (*ctxLmemUnreg)(struct RaRsDevInfo *devInfo, unsigned long long addr);
    int (*ctxRmemImport)(struct RaRsDevInfo *devInfo, struct MemImportAttrT *memAttr, struct MemImportInfoT *memInfo);
    int (*ctxRmemUnimport)(struct RaRsDevInfo *devInfo, unsigned long long addr);
    int (*ctxChanCreate)(struct RaRsDevInfo *devInfo, union DataPlaneCstmFlag dataPlaneFlag, unsigned long long *addr,
        int *fd);
    int (*ctxChanDestroy)(struct RaRsDevInfo *devInfo, unsigned long long addr);
    int (*ctxCqCreate)(struct RaRsDevInfo *devInfo, struct CtxCqAttr *attr, struct CtxCqInfo *info);
    int (*ctxCqDestroy)(struct RaRsDevInfo *devInfo, unsigned long long addr);
    int (*ctxQpCreate)(struct RaRsDevInfo *devInfo, struct CtxQpAttr *qpAttr, struct QpCreateInfo *qpInfo);
    int (*ctxQpDestroy)(struct RaRsDevInfo *devInfo, unsigned int id);
    int (*ctxQpDestroyBatch)(struct RaRsDevInfo *devInfo, unsigned int ids[], unsigned int *num);
    int (*ctxQpImport)(struct RaRsDevInfo *devInfo, struct RsJettyImportAttr *importAttr,
        struct RsJettyImportInfo *importInfo);
    int (*ctxQpUnimport)(struct RaRsDevInfo *devInfo, unsigned char rawRemJettyId[], unsigned int size);
    int (*ctxQpBind)(struct RaRsDevInfo *devInfo, struct RsCtxQpInfo *localQpInfo, struct RsCtxQpInfo *remoteQpInfo);
    int (*ctxQpUnbind)(struct RaRsDevInfo *devInfo, unsigned int qpId);
    int (*ctxBatchSendWr)(struct WrlistBaseInfo *baseInfo, struct BatchSendWrData *wrData, struct SendWrResp *wrResp,
        struct WrlistSendCompleteNum *wrlistNum);
    int (*ctxUpdateCi)(struct RaRsDevInfo *devInfo, unsigned int qpId, uint16_t ci);
    int (*ctxQpQueryBatch)(struct RaRsDevInfo *devInfo, unsigned int ids[], struct JettyAttr attr[], unsigned int *num);
    int (*ctxGetAuxInfo)(struct RaRsDevInfo *devInfo, struct HccpAuxInfoIn *infoIn, struct HccpAuxInfoOut *infoOut);
    int (*ctxGetCrErrInfoList)(struct RaRsDevInfo *devInfo, struct CrErrInfo infoList[], unsigned int *num);
    int (*ctxGetUbContext)(struct RaRsDevInfo *devInfo, unsigned int id, unsigned int contextType, uint8_t context[],
        unsigned int *len);
    int (*ctxNotifyEvent)(struct RaRsDevInfo *devInfo, struct CtxNotifyEvent *event);
};

int RaRsGetDevEidInfoNum(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsGetDevEidInfoList(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxInit(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxGetAsyncEvents(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxDeinit(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsGetEidByIp(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsGetIpByEid(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsGetTpInfoList(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsGetTpAttr(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsSetTpAttr(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxTokenIdAlloc(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxTokenIdFree(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsLmemReg(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsLmemUnreg(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsRmemImport(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsRmemUnimport(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxChanCreate(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxChanDestroy(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxCqCreate(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxCqDestroy(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxQpCreate(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxQpQueryBatch(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxQpDestroy(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxQpDestroyBatch(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxQpImport(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxQpUnimport(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxQpUnimportDeprecated(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxQpBind(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxQpUnbind(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxUpdateCi(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxBatchSendWr(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxGetAuxInfo(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxGetCrErrInfoList(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxNotifyEvent(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
int RaRsCtxGetUbContext(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen);
#endif // RA_ADP_CTX_H
