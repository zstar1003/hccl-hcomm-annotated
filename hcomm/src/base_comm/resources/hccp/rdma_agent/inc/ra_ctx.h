/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_CTX_H
#define RA_CTX_H

#include <stddef.h>
#include "hccp_ctx.h"
#include "ra_rs_ctx.h"

struct RaCtxHandle {
    enum ProtocolTypeT protocol;
    struct RaCtxOps *ctxOps;
    struct CtxInitAttr attr;
    unsigned int devIndex;
    struct DevBaseAttr devAttr;
};

struct RaTokenIdHandle {
    unsigned long long addr;
};

struct RaLmemHandle {
    unsigned long long addr;
};

struct RaRmemHandle {
    struct MemKey key;
    unsigned long long addr;
};

struct RaCtxQpHandle {
    unsigned int id; // qpn(rdma) or jetty_id(udma)
    unsigned int phyId;
    unsigned int devIndex;
    enum ProtocolTypeT protocol;
    struct QpCreateAttr qpAttr;
    struct QpCreateInfo qpInfo;
    struct RaCtxHandle *ctxHandle;
};

/* External accessors (e.g. legacy HrtGetJettyQpNum) read id as the first member; keep this layout stable. */
_Static_assert(offsetof(struct RaCtxQpHandle, id) == 0, "id must be the first member of struct RaCtxQpHandle");

struct RaCtxRemQpHandle {
    unsigned int id; // qpn(rdma) or jetty_id(udma)
    unsigned int phyId;
    unsigned int devIndex;
    enum ProtocolTypeT protocol;
    struct QpKey qpKey; // only for rdma
};

struct RaChanHandle {
    unsigned long long addr; /**< refer to ibv_comp_channel*, urma_jfce_t* for chan_cb index */
};

struct RaCqHandle {
    unsigned long long addr; /**< refer to ibv_cq*, urma_jfc_t* for cq_cb index */
};

struct RaCtxOps {
    int (*raCtxInit)(struct RaCtxHandle *ctxHandle, struct CtxInitAttr *attr, unsigned int *devIndex,
        struct DevBaseAttr *devAttr);
    int (*raCtxGetAsyncEvents)(struct RaCtxHandle *ctxHandle, struct AsyncEvent events[], unsigned int *num);
    int (*raCtxDeinit)(struct RaCtxHandle *ctxHandle);
    int (*raCtxGetEidByIp)(struct RaCtxHandle *ctxHandle, struct IpInfo ip[], union HccpEid eid[], unsigned int *num);
    int (*raCtxGetIpByEid)(struct RaCtxHandle *ctxHandle, union HccpEid eid[], struct IpInfo ip[], unsigned int *num);
    int (*raCtxTokenIdAlloc)(struct RaCtxHandle *ctxHandle, struct HccpTokenId *info,
        struct RaTokenIdHandle *tokenIdHandle);
    int (*raCtxTokenIdFree)(struct RaCtxHandle *ctxHandle, struct RaTokenIdHandle *tokenIdHandle);
    int (*raCtxLmemRegister)(struct RaCtxHandle *ctxHandle, struct MrRegInfoT *lmemInfo,
        struct RaLmemHandle *lmemHandle);
    int (*raCtxLmemUnregister)(struct RaCtxHandle *ctxHandle, struct RaLmemHandle *lmemHandle);
    int (*raCtxRmemImport)(struct RaCtxHandle *ctxHandle, struct MrImportInfoT *rmemInfo);
    int (*raCtxRmemUnimport)(struct RaCtxHandle *ctxHandle, struct RaRmemHandle *rmemHandle);
    int (*raCtxChanCreate)(struct RaCtxHandle *ctxHandle, struct ChanInfoT *chanInfo, struct RaChanHandle *chanHandle);
    int (*raCtxChanDestroy)(struct RaCtxHandle *ctxHandle, struct RaChanHandle *chanHandle);
    int (*raCtxCqCreate)(struct RaCtxHandle *ctxHandle, struct CqInfoT *info, struct RaCqHandle *cqHandle);
    int (*raCtxCqDestroy)(struct RaCtxHandle *ctxHandle, struct RaCqHandle *cqHandle);
    int (*raCtxQpCreate)(struct RaCtxHandle *ctxHandle, struct QpCreateAttr *qpAttr, struct QpCreateInfo *qpInfo,
        struct RaCtxQpHandle *qpHandle);
    int (*raCtxQueryQpBatch)(unsigned int phyId, unsigned int devIndex, unsigned int ids[], struct JettyAttr attr[],
        unsigned int *num);
    int (*raCtxQpDestroy)(struct RaCtxQpHandle *qpHandle);
    int (*raCtxQpImport)(struct RaCtxHandle *ctxHandle, struct QpImportInfoT *qpImportInfo,
        struct RaCtxRemQpHandle *remQpHandle);
    int (*raCtxQpUnimport)(struct RaCtxRemQpHandle *remQpHandle);
    int (*raCtxQpBind)(struct RaCtxQpHandle *qpHandle, struct RaCtxRemQpHandle *remQpHandle);
    int (*raCtxQpUnbind)(struct RaCtxQpHandle *qpHandle);
    int (*raCtxBatchSendWr)(struct RaCtxQpHandle *qpHandle, struct SendWrData wrList[], struct SendWrResp opResp[],
        unsigned int sendNum, unsigned int *completeNum);
    int (*raCtxUpdateCi)(struct RaCtxQpHandle *qpHandle, uint16_t ci);
    int (*raCtxGetAuxInfo)(struct RaCtxHandle *ctxHandle, struct HccpAuxInfoIn *in, struct HccpAuxInfoOut *out);
    int (*raCtxNotifyEvent)(struct RaCtxHandle *ctxHandle, struct CtxNotifyEvent *event);
    int (*raCtxGetJettyContext)(struct RaCtxQpHandle *qpHandle, uint8_t context[], unsigned int *len);
};

#endif // RA_CTX_H
