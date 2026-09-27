/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RS_CTX_H
#define RS_CTX_H

#include "hccp_async_ctx.h"
#include "hccp_async.h"
#include "hccp_common.h"
#include "ra_rs_ctx.h"
#include "rs.h"

struct RsCtxQpInfo {
    uint32_t id; // qpn(rdma) or jetty_id(udma)
    struct QpKey key;
};

struct RsJettyImportAttr {
    struct QpKey key;
    struct RaRsJettyImportAttr attr;
};

struct RsJettyImportInfo {
    unsigned int remJettyId;
    struct RaRsJettyImportInfo info;
};

int RsGetChipProtocol(unsigned int chipId, enum NetworkMode hccpMode, enum ProtocolTypeT *protocol,
    unsigned int logicId);
int RsCtxApiInit(enum NetworkMode hccpMode, enum ProtocolTypeT protocol);
int RsCtxApiDeinit(enum NetworkMode hccpMode, enum ProtocolTypeT protocol);

RS_ATTRI_VISI_DEF int RsGetDevEidInfoNum(unsigned int phyId, unsigned int *num);
RS_ATTRI_VISI_DEF int RsGetDevEidInfoList(unsigned int phyId, struct HccpDevEidInfo infoList[], unsigned int startIndex,
    unsigned int count);
RS_ATTRI_VISI_DEF int RsCtxInit(struct CtxInitAttr *attr, unsigned int *devIndex, struct DevBaseAttr *devAttr);
RS_ATTRI_VISI_DEF int RsCtxGetAsyncEvents(struct RaRsDevInfo *devInfo, struct AsyncEvent asyncEvents[],
    unsigned int *num);
RS_ATTRI_VISI_DEF int RsCtxDeinit(struct RaRsDevInfo *devInfo);
RS_ATTRI_VISI_DEF int RsGetEidByIp(struct RaRsDevInfo *devInfo, struct IpInfo ip[], union HccpEid eid[],
    unsigned int *num);
RS_ATTRI_VISI_DEF int RsGetIpByEid(struct RaRsDevInfo *devInfo, union HccpEid eid[], struct IpInfo ip[],
    unsigned int *num);
RS_ATTRI_VISI_DEF int RsGetTpInfoList(struct RaRsDevInfo *devInfo, struct GetTpCfg *cfg, struct HccpTpInfo infoList[],
    unsigned int *num);
RS_ATTRI_VISI_DEF int RsGetTpAttr(struct RaRsDevInfo *devInfo, unsigned int *attrBitmap, const uint64_t tpHandle,
    struct TpAttr *attr);
RS_ATTRI_VISI_DEF int RsSetTpAttr(struct RaRsDevInfo *devInfo, const unsigned int attrBitmap, const uint64_t tpHandle,
    struct TpAttr *attr);
RS_ATTRI_VISI_DEF int RsCtxTokenIdAlloc(struct RaRsDevInfo *devInfo, unsigned long long *addr, unsigned int *tokenId);
RS_ATTRI_VISI_DEF int RsCtxTokenIdFree(struct RaRsDevInfo *devInfo, unsigned long long addr);
RS_ATTRI_VISI_DEF int RsCtxLmemReg(struct RaRsDevInfo *devInfo, struct MemRegAttrT *memAttr,
    struct MemRegInfoT *memInfo);
RS_ATTRI_VISI_DEF int RsCtxLmemUnreg(struct RaRsDevInfo *devInfo, unsigned long long addr);
RS_ATTRI_VISI_DEF int RsCtxRmemImport(struct RaRsDevInfo *devInfo, struct MemImportAttrT *memAttr,
    struct MemImportInfoT *memInfo);
RS_ATTRI_VISI_DEF int RsCtxRmemUnimport(struct RaRsDevInfo *devInfo, unsigned long long addr);
RS_ATTRI_VISI_DEF int RsCtxChanCreate(struct RaRsDevInfo *devInfo, union DataPlaneCstmFlag dataPlaneFlag,
    unsigned long long *addr, int *fd);
RS_ATTRI_VISI_DEF int RsCtxChanDestroy(struct RaRsDevInfo *devInfo, unsigned long long addr);
RS_ATTRI_VISI_DEF int RsCtxCqCreate(struct RaRsDevInfo *devInfo, struct CtxCqAttr *attr, struct CtxCqInfo *info);
RS_ATTRI_VISI_DEF int RsCtxCqDestroy(struct RaRsDevInfo *devInfo, unsigned long long addr);
RS_ATTRI_VISI_DEF int RsCtxQpCreate(struct RaRsDevInfo *devInfo, struct CtxQpAttr *qpAttr, struct QpCreateInfo *qpInfo);
RS_ATTRI_VISI_DEF int RsCtxQpDestroy(struct RaRsDevInfo *devInfo, unsigned int id);
RS_ATTRI_VISI_DEF int RsCtxQpDestroyBatch(struct RaRsDevInfo *devInfo, unsigned int ids[], unsigned int *num);
RS_ATTRI_VISI_DEF int RsCtxQpImport(struct RaRsDevInfo *devInfo, struct RsJettyImportAttr *importAttr,
    struct RsJettyImportInfo *importInfo);
RS_ATTRI_VISI_DEF int RsCtxQpUnimport(struct RaRsDevInfo *devInfo, unsigned char rawRemJettyId[], unsigned int size);
RS_ATTRI_VISI_DEF int RsCtxQpBind(struct RaRsDevInfo *devInfo, struct RsCtxQpInfo *localQpInfo,
    struct RsCtxQpInfo *remoteQpInfo);
RS_ATTRI_VISI_DEF int RsCtxQpUnbind(struct RaRsDevInfo *devInfo, unsigned int qpId);
RS_ATTRI_VISI_DEF int RsCtxBatchSendWr(struct WrlistBaseInfo *baseInfo, struct BatchSendWrData *wrData,
    struct SendWrResp *wrResp, struct WrlistSendCompleteNum *wrlistNum);
RS_ATTRI_VISI_DEF int RsCtxUpdateCi(struct RaRsDevInfo *devInfo, unsigned int qpId, uint16_t ci);
RS_ATTRI_VISI_DEF int RsCtxQpQueryBatch(struct RaRsDevInfo *devInfo, unsigned int ids[], struct JettyAttr attr[],
    unsigned int *num);
RS_ATTRI_VISI_DEF int RsCtxGetAuxInfo(struct RaRsDevInfo *devInfo, struct HccpAuxInfoIn *infoIn,
    struct HccpAuxInfoOut *infoOut);
RS_ATTRI_VISI_DEF int RsCtxGetCrErrInfoList(struct RaRsDevInfo *devInfo, struct CrErrInfo infoList[],
    unsigned int *num);
RS_ATTRI_VISI_DEF int RsCtxNotifyEvent(struct RaRsDevInfo *devInfo, struct CtxNotifyEvent *event);
RS_ATTRI_VISI_DEF int RsCtxGetUbContext(struct RaRsDevInfo *devInfo, unsigned int id, unsigned int contextType,
    uint8_t context[], unsigned int *len);
#endif // RS_CTX_H
