/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RS_UB_H
#define RS_UB_H

#include <udma_u_ctl.h>
#include <urma_types.h>
#include "dl_urma_function.h"
#include "hccp_async_ctx.h"
#include "hccp_async.h"
#include "hccp_ctx.h"
#include "hccp_common.h"
#include "ra_rs_comm.h"
#include "ra_rs_ctx.h"
#include "rs_ctx.h"
#include "rs_inner.h"
#include "rs_ctx_inner.h"
#include "rs.h"

struct RsJettyKeyInfo {
    urma_jetty_id_t jettyId;
    urma_transport_mode_t transMode;
};

enum RsJettyState {
    RS_JETTY_STATE_INIT = 0,
    RS_JETTY_STATE_CREATED = 1,
    RS_JETTY_STATE_IMPORTED = 2,
    RS_JETTY_STATE_BIND = 3,
    RS_JETTY_STATE_MAX
};

struct JettyDestroyBatchInfo {
    struct RsCtxJettyCb **jettyCbArr;
    urma_jetty_t **jettyArr;
};

int RsUbGetDevEidInfoNum(unsigned int phyId, unsigned int *num);
int RsUbGetUeInfo(urma_context_t *urmaCtx, struct DevBaseAttr *devAttr);
int RsUbGetDevEidInfoList(unsigned int phyId, struct HccpDevEidInfo infoList[], unsigned int startIndex,
    unsigned int count);
int RsUbCtxInit(struct rs_cb *rsCb, struct CtxInitAttr *attr, unsigned int *devIndex, struct DevBaseAttr *devAttr);
int RsUbGetDevCb(struct rs_cb *rscb, unsigned int devIndex, struct RsUbDevCb **devCb);
int RsUbCtxDeinit(struct RsUbDevCb *devCb);
int RsUbGetEidByIp(struct RsUbDevCb *devCb, struct IpInfo ip[], union HccpEid eid[], unsigned int *num);
int RsUbGetIpByEid(struct RsUbDevCb *devCb, union HccpEid eid[], struct IpInfo ip[], unsigned int *num);
int RsUbCtxTokenIdAlloc(struct RsUbDevCb *devCb, unsigned long long *addr, unsigned int *tokenId);
int RsUbCtxTokenIdFree(struct RsUbDevCb *devCb, unsigned long long addr);
int RsUbCtxLmemReg(struct RsUbDevCb *devCb, struct MemRegAttrT *memAttr, struct MemRegInfoT *memInfo);
int RsUbCtxLmemUnreg(struct RsUbDevCb *devCb, unsigned long long addr);
int RsUbCtxRmemImport(struct RsUbDevCb *devCb, struct MemImportAttrT *memAttr, struct MemImportInfoT *memInfo);
int RsUbCtxRmemUnimport(struct RsUbDevCb *devCb, unsigned long long addr);
int RsUbCtxChanCreate(struct RsUbDevCb *devCb, union DataPlaneCstmFlag dataPlaneFlag, unsigned long long *addr,
    int *fd);
int RsUbCtxChanDestroy(struct RsUbDevCb *devCb, unsigned long long addr);
int RsUbCtxJfcCreate(struct RsUbDevCb *devCb, struct CtxCqAttr *attr, struct CtxCqInfo *info);
int RsUbCtxJfcDestroy(struct RsUbDevCb *devCb, unsigned long long addr);
int RsUbCtxJettyCreate(struct RsUbDevCb *devCb, struct CtxQpAttr *attr, struct QpCreateInfo *info);
int RsUbCtxRegJettyDb(struct RsCtxJettyCb *jettyCb, struct udma_u_jetty_info *jettyInfo);
int RsUbCtxQueryJettyBatch(struct RsUbDevCb *devCb, unsigned int jettyIds[], struct JettyAttr attr[],
    unsigned int *num);
int RsUbCtxJettyDestroy(struct RsUbDevCb *devCb, unsigned int jettyId);
int RsUbCtxJettyDestroyBatch(struct RsUbDevCb *devCb, unsigned int jettyIds[], unsigned int *num);
int RsUbCtxJettyImport(struct RsUbDevCb *devCb, struct RsJettyImportAttr *importAttr,
    struct RsJettyImportInfo *importInfo);
int RsUbCtxJettyUnimport(struct RsUbDevCb *devCb, unsigned char rawRemJettyId[], unsigned int size);
int RsUbCtxJettyBind(struct RsUbDevCb *devCb, struct RsCtxQpInfo *jettyInfo, struct RsCtxQpInfo *rjettyInfo);
int RsUbCtxJettyUnbind(struct RsUbDevCb *devCb, unsigned int jettyId);
int RsUbCtxJettyFree(struct rs_cb *rscb, unsigned int ueInfo, unsigned int jettyId);
void RsUbFreeJettyCbList(struct RsUbDevCb *devCb, struct RsListHead *jettyList, struct RsListHead *rjettyList);
int RsUbCtxBatchSendWr(struct rs_cb *rsCb, struct WrlistBaseInfo *baseInfo, struct BatchSendWrData *wrData,
    struct SendWrResp *wrResp, struct WrlistSendCompleteNum *wrlistNum);
int RsUbCtxJettyUpdateCi(struct RsUbDevCb *devCb, unsigned int jettyId, uint16_t ci);
int RsUbGetJettyCb(struct RsUbDevCb *devCb, unsigned int jettyId, struct RsCtxJettyCb **jettyCb);
void RsUbFreeAsyncEventCb(struct RsUbDevCb *devCb, struct RsCtxAsyncEventCb *asyncEventCb);
#endif // RS_UB_H
