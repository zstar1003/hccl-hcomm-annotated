/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_HDC_CTX_H
#define RA_HDC_CTX_H

#include "hccp_common.h"
#include "hccp_ctx.h"
#include "ra_rs_comm.h"
#include "ra_rs_ctx.h"
#include "ra_comm.h"
#include "ra_ctx.h"
#include "ra_hdc.h"

#define MAX_DEV_INFO_TRANS_NUM 30

union OpGetDevEidInfoNumData {
    struct {
        unsigned int phyId;
        unsigned int rsvd;
    } txData;

    struct {
        unsigned int num;
        unsigned int rsvd;
    } rxData;
};

union OpGetDevEidInfoListData {
    struct {
        unsigned int phyId;
        unsigned int startIndex;
        unsigned int count;
        unsigned int rsvd;
    } txData;

    struct {
        struct HccpDevEidInfo infoList[MAX_DEV_INFO_TRANS_NUM];
        unsigned int rsvd;
    } rxData;
};

union OpCtxInitData {
    struct {
        struct CtxInitAttr attr;
        unsigned int rsvd;
    } txData;

    struct {
        struct DevBaseAttr devAttr;
        unsigned int devIndex;
        unsigned int rsvd;
    } rxData;
};

union OpCtxGetAsyncEventsData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        unsigned int num;
        unsigned int rsvd;
    } txData;

    struct {
        struct AsyncEvent events[ASYNC_EVENT_MAX_NUM];
        unsigned int num;
        unsigned int rsvd;
    } rxData;
};

union OpCtxDeinitData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        unsigned int rsvd;
    } txData;

    struct {
        unsigned int rsvd;
    } rxData;
};

union OpGetEidByIpData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        struct IpInfo ip[HCCP_EID_IP_QUERY_MAX_NUM];
        unsigned int num;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } txData;

    struct {
        union HccpEid eid[HCCP_EID_IP_QUERY_MAX_NUM];
        unsigned int num;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

union OpGetIpByEidData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        union HccpEid eid[HCCP_EID_IP_QUERY_MAX_NUM];
        unsigned int num;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } txData;

    struct {
        struct IpInfo ip[HCCP_EID_IP_QUERY_MAX_NUM];
        unsigned int num;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

union OpTokenIdAllocData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } txData;
    struct {
        unsigned long long addr;
        unsigned int tokenId;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

union OpTokenIdFreeData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        unsigned long long addr;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } txData;
    struct {
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

union OpLmemRegInfoData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        struct MemRegAttrT memAttr;
        unsigned int rsvd;
    } txData;

    struct {
        struct MemRegInfoT memInfo;
        unsigned int rsvd;
    } rxData;
};

union OpLmemUnregInfoData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        unsigned long long addr;
        unsigned int rsvd;
    } txData;

    struct {
        unsigned int rsvd;
    } rxData;
};

union OpRmemImportInfoData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        struct MemImportAttrT memAttr;
        unsigned int rsvd;
    } txData;

    struct {
        struct MemImportInfoT memInfo;
        unsigned int rsvd;
    } rxData;
};

union OpRmemUnimportInfoData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        unsigned long long addr;
        unsigned int rsvd;
    } txData;

    struct {
        unsigned int rsvd;
    } rxData;
};

union OpCtxChanCreateData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        union DataPlaneCstmFlag dataPlaneFlag;
        uint32_t resv[RA_RSVD_NUM_4];
    } txData;
    struct {
        unsigned long long addr; /**< refer to ibv_comp_channel*, urma_jfce_t* for chan_cb index */
        int fd;
        uint32_t resv[RA_RSVD_NUM_8];
    } rxData;
};

union OpCtxChanDestroyData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        unsigned long long addr; /**< refer to ibv_comp_channel*, urma_jfce_t* for chan_cb index */
        uint32_t resv[RA_RSVD_NUM_4];
    } txData;
    struct {
        uint32_t resv[RA_RSVD_NUM_4];
    } rxData;
};

union OpCtxCqCreateData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        struct CtxCqAttr attr;
        uint32_t resv[4U];
    } txData;

    struct {
        struct CtxCqInfo info;
        uint32_t resv[4U];
    } rxData;
};

union OpCtxCqDestroyData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        uint64_t addr; /**< refer to ibv_cq*, urma_jfc_t* for cq_cb index */
        uint32_t resv[4U];
    } txData;

    struct {
        uint32_t resv[4U];
    } rxData;
};

union OpCtxQpCreateData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        struct CtxQpAttr qpAttr;
        unsigned int rsvd[RA_RSVD_NUM_3];
    } txData;

    struct {
        struct QpCreateInfo qpInfo;
        unsigned int rsvd[RA_RSVD_NUM_3];
    } rxData;
};

union OpCtxQpDestroyData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        unsigned int id; // qpn(rdma) or jetty_id(udma)
        unsigned int rsvd[RA_RSVD_NUM_3];
    } txData;

    struct {
        unsigned int rsvd;
    } rxData;
};

union OpCtxQpImportData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        struct QpKey key;
        struct RaRsJettyImportAttr attr;
        unsigned int rsvd[RA_RSVD_NUM_3];
    } txData;

    struct {
        unsigned int remJettyId; // only for ub
        struct RaRsJettyImportInfo info;
        unsigned int rsvd[RA_RSVD_NUM_3];
    } rxData;
};

union OpCtxQpUnimportData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        unsigned char rawRemJettyId[REM_JETTY_ID_SIZE]; // refer to urma_jetty_id_t in struct QpKey
        unsigned int rsvd;
    } txData;

    struct {
        unsigned int rsvd[RA_RSVD_NUM_6];
    } rxData;
};

union OpCtxQpBindData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        unsigned int id;    // local qpn(rdma) or local jetty_id(udma)
        unsigned int remId; // only for UB, equivalent to rem_jetty_id
        struct QpKey localQpKey;
        struct QpKey remoteQpKey;
        unsigned int rsvd[RA_RSVD_NUM_6];
    } txData;

    struct {
        unsigned int rsvd[RA_RSVD_NUM_6];
    } rxData;
};

union OpCtxQpUnbindData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        unsigned int id; // local qpn(rdma) or local jetty_id(udma)
        unsigned int rsvd[RA_RSVD_NUM_3];
    } txData;

    struct {
        unsigned int rsvd;
    } rxData;
};

union OpCtxBatchSendWrData {
    struct {
        struct WrlistBaseInfo baseInfo;
        unsigned int sendNum;
        struct BatchSendWrData wrData[MAX_CTX_WR_NUM];
    } txData;

    struct {
        unsigned int completeNum;
        struct SendWrResp wrResp[MAX_CTX_WR_NUM];
    } rxData;
};

union OpCtxUpdateCiData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        unsigned int jettyId;
        uint16_t ci;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } txData;
    struct {
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

union OpCtxQpQueryBatchData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        unsigned int num;
        unsigned int ids[HCCP_MAX_QP_QUERY_NUM];
    } txData;

    struct {
        unsigned int num;
        struct JettyAttr attr[HCCP_MAX_QP_QUERY_NUM];
    } rxData;
};

union OpCtxGetAuxInfoData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        struct HccpAuxInfoIn info;
    } txData;

    struct {
        struct HccpAuxInfoOut info;
    } rxData;
};

union OpCtxGetCrErrInfoListData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        unsigned int num;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } txData;

    struct {
        struct CrErrInfo infoList[CR_ERR_INFO_MAX_NUM];
        unsigned int num;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

union OpCtxGetContextData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        unsigned int id;
        unsigned int contextType;
        unsigned int len;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } txData;

    struct {
        uint8_t context[CONTEXT_MAX_LEN];
        unsigned int len;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

union OpCtxNotifyEventData {
    struct {
        unsigned int phyId;
        unsigned int devIndex;
        uint32_t resv0;
        struct CtxNotifyEvent event;
        uint32_t resv1[RA_RSVD_NUM_512]; /* Reserved extension field for larger payload. */
    } txData;

    struct {
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

int RaHdcGetDevEidInfoNum(struct RaInfo info, unsigned int *num);
int RaHdcGetDevEidInfoList(unsigned int phyId, struct HccpDevEidInfo infoList[], unsigned int *num);
int RaHdcCtxInit(struct RaCtxHandle *ctxHandle, struct CtxInitAttr *attr, unsigned int *devIndex,
    struct DevBaseAttr *devAttr);
int RaHdcCtxGetAsyncEvents(struct RaCtxHandle *ctxHandle, struct AsyncEvent events[], unsigned int *num);
int RaHdcCtxDeinit(struct RaCtxHandle *ctxHandle);
void RaHdcPrepareGetEidByIp(struct RaCtxHandle *ctxHandle, struct IpInfo ip[], unsigned int ipNum,
    union OpGetEidByIpData *opData);
int RaHdcGetEidResults(union OpGetEidByIpData *opData, unsigned int ipNum, union HccpEid eid[], unsigned int *num);
int RaHdcGetEidByIp(struct RaCtxHandle *ctxHandle, struct IpInfo ip[], union HccpEid eid[], unsigned int *num);
void RaHdcPrepareGetIpByEid(struct RaCtxHandle *ctxHandle, union HccpEid eid[], unsigned int eidNum,
    union OpGetIpByEidData *opData);
int RaHdcGetIpResults(union OpGetIpByEidData *opData, unsigned int eidNum, struct IpInfo ip[], unsigned int *num);
int RaHdcGetIpByEid(struct RaCtxHandle *ctxHandle, union HccpEid eid[], struct IpInfo ip[], unsigned int *num);
int RaHdcCtxTokenIdAlloc(struct RaCtxHandle *ctxHandle, struct HccpTokenId *info,
    struct RaTokenIdHandle *tokenIdHandle);
int RaHdcCtxTokenIdFree(struct RaCtxHandle *ctxHandle, struct RaTokenIdHandle *tokenIdHandle);
int RaHdcCtxPrepareLmemRegister(struct RaCtxHandle *ctxHandle, struct MrRegInfoT *lmemInfo,
    union OpLmemRegInfoData *opData);
int RaHdcCtxLmemRegister(struct RaCtxHandle *ctxHandle, struct MrRegInfoT *lmemInfo, struct RaLmemHandle *lmemHandle);
int RaHdcCtxLmemUnregister(struct RaCtxHandle *ctxHandle, struct RaLmemHandle *lmemHandle);
int RaHdcCtxRmemImport(struct RaCtxHandle *ctxHandle, struct MrImportInfoT *rmemInfo);
int RaHdcCtxRmemUnimport(struct RaCtxHandle *ctxHandle, struct RaRmemHandle *rmemHandle);
int RaHdcCtxChanCreate(struct RaCtxHandle *ctxHandle, struct ChanInfoT *chanInfo, struct RaChanHandle *chanHandle);
int RaHdcCtxChanDestroy(struct RaCtxHandle *ctxHandle, struct RaChanHandle *chanHandle);
int RaHdcCtxCqCreate(struct RaCtxHandle *ctxHandle, struct CqInfoT *info, struct RaCqHandle *cqHandle);
int RaHdcCtxCqDestroy(struct RaCtxHandle *ctxHandle, struct RaCqHandle *cqHandle);
int RaHdcCtxPrepareQpCreate(struct RaCtxHandle *ctxHandle, struct QpCreateAttr *qpAttr,
    union OpCtxQpCreateData *opData);
int RaHdcCtxQpCreate(struct RaCtxHandle *ctxHandle, struct QpCreateAttr *qpAttr, struct QpCreateInfo *qpInfo,
    struct RaCtxQpHandle *qpHandle);
int RaHdcCtxQpQueryBatch(unsigned int phyId, unsigned int devIndex, unsigned int ids[], struct JettyAttr attr[],
    unsigned int *num);
int RaHdcCtxQpDestroy(struct RaCtxQpHandle *qpHandle);
int RaHdcCtxPrepareQpImport(struct RaCtxHandle *ctxHandle, struct QpImportInfoT *qpInfo,
    union OpCtxQpImportData *opData);
int RaHdcCtxQpImport(struct RaCtxHandle *ctxHandle, struct QpImportInfoT *qpInfo, struct RaCtxRemQpHandle *remQpHandle);
int RaHdcCtxQpUnimport(struct RaCtxRemQpHandle *remQpHandle);
int RaHdcCtxQpBind(struct RaCtxQpHandle *qpHandle, struct RaCtxRemQpHandle *remQpHandle);
int RaHdcCtxQpUnbind(struct RaCtxQpHandle *qpHandle);
int RaHdcCtxBatchSendWr(struct RaCtxQpHandle *qpHandle, struct SendWrData wrList[], struct SendWrResp opResp[],
    unsigned int sendNum, unsigned int *completeNum);
int RaHdcCtxUpdateCi(struct RaCtxQpHandle *qpHandle, uint16_t ci);
int RaHdcCtxGetAuxInfo(struct RaCtxHandle *ctxHandle, struct HccpAuxInfoIn *in, struct HccpAuxInfoOut *out);
int RaHdcCtxGetCrErrInfoList(struct RaCtxHandle *ctxHandle, struct CrErrInfo *infoList, unsigned int *num);
int RaHdcCtxNotifyEvent(struct RaCtxHandle *ctxHandle, struct CtxNotifyEvent *event);
int RaHdcCtxGetJettyContext(struct RaCtxQpHandle *qpHandle, uint8_t context[], unsigned int *len);
#endif // RA_HDC_CTX_H
