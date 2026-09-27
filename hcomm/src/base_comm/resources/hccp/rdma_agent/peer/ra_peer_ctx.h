/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_PEER_CTX_H
#define RA_PEER_CTX_H

#include "hccp_ctx.h"
#include "ra_ctx.h"

int RaPeerGetDevEidInfoNum(struct RaInfo info, unsigned int *num);

int RaPeerGetDevEidInfoList(unsigned int phyId, struct HccpDevEidInfo infoList[], unsigned int *num);

int RaPeerCtxInit(struct RaCtxHandle *ctxHandle, struct CtxInitAttr *attr, unsigned int *devIndex,
    struct DevBaseAttr *devBaseAttr);

int RaPeerCtxGetAsyncEvents(struct RaCtxHandle *ctxHandle, struct AsyncEvent events[], unsigned int *num);

int RaPeerCtxDeinit(struct RaCtxHandle *ctxHandle);

int RaPeerGetEidByIp(struct RaCtxHandle *ctxHandle, struct IpInfo ip[], union HccpEid eid[], unsigned int *num);

int RaPeerGetIpByEid(struct RaCtxHandle *ctxHandle, union HccpEid eid[], struct IpInfo ip[], unsigned int *num);

int RaPeerCtxTokenIdAlloc(struct RaCtxHandle *ctxHandle, struct HccpTokenId *info,
    struct RaTokenIdHandle *tokenIdHandle);

int RaPeerCtxTokenIdFree(struct RaCtxHandle *ctxHandle, struct RaTokenIdHandle *tokenIdHandle);

int RaPeerCtxLmemRegister(struct RaCtxHandle *ctxHandle, struct MrRegInfoT *lmemInfo, struct RaLmemHandle *lmemHandle);

int RaPeerCtxLmemUnregister(struct RaCtxHandle *ctxHandle, struct RaLmemHandle *lmemHandle);

int RaPeerCtxRmemImport(struct RaCtxHandle *ctxHandle, struct MrImportInfoT *rmemInfo);

int RaPeerCtxRmemUnimport(struct RaCtxHandle *ctxHandle, struct RaRmemHandle *rmemHandle);

int RaPeerCtxChanCreate(struct RaCtxHandle *ctxHandle, struct ChanInfoT *chanInfo, struct RaChanHandle *chanHandle);

int RaPeerCtxChanDestroy(struct RaCtxHandle *ctxHandle, struct RaChanHandle *chanHandle);

int RaPeerCtxCqCreate(struct RaCtxHandle *ctxHandle, struct CqInfoT *info, struct RaCqHandle *cqHandle);

int RaPeerCtxCqDestroy(struct RaCtxHandle *ctxHandle, struct RaCqHandle *cqHandle);

int RaPeerCtxQpCreate(struct RaCtxHandle *ctxHandle, struct QpCreateAttr *qpAttr, struct QpCreateInfo *qpInfo,
    struct RaCtxQpHandle *qpHandle);

int RaPeerCtxQpDestroy(struct RaCtxQpHandle *qpHandle);

int RaPeerCtxGetTpInfoList(struct RaCtxHandle *ctxHandle, struct GetTpCfg *cfg, struct HccpTpInfo infoList[],
    unsigned int *num);

int RaPeerCtxGetTpAttr(struct RaCtxHandle *ctxHandle, uint64_t tpHandle, uint32_t *attrBitmap, struct TpAttr *attr);

int RaPeerCtxSetTpAttr(struct RaCtxHandle *ctxHandle, uint64_t tpHandle, uint32_t attrBitmap, struct TpAttr *attr);

int RaPeerCtxQpImport(struct RaCtxHandle *ctxHandle, struct QpImportInfoT *qpInfo,
    struct RaCtxRemQpHandle *remQpHandle);

int RaPeerCtxQpUnimport(struct RaCtxRemQpHandle *remQpHandle);

int RaPeerCtxQpBind(struct RaCtxQpHandle *qpHandle, struct RaCtxRemQpHandle *remQpHandle);

int RaPeerCtxQpUnbind(struct RaCtxQpHandle *qpHandle);

int RaPeerCtxGetJettyContext(struct RaCtxQpHandle *qpHandle, uint8_t context[], unsigned int *len);

#endif // RA_PEER_CTX_H
