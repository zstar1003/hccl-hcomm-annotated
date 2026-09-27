/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_HDC_LITE_H
#define RA_HDC_LITE_H

#include "ascend_hal.h"
#include "stdio.h"
#include "hccp.h"
#include "hccp_common.h"
#include "ra.h"
#include "ra_rs_comm.h"
#include "ra_hdc_rdma.h"
#include "ra_hdc_rdma_verbs.h"

#define RA_SGLIST_MAX 16
#define RA_QP_32K_DEPTH 32767
#define RA_QP_128_DEPTH 128

#define WRITE_NOTIFY_OFFSET_MASK 0xffffff
#define WRITE_NOTIFY_VALUE_RECORD 0x1000000

#define RA_LITE_POLL_CQE_PERIOD_TIME 10000 // 10ms

#define HDC_LITE_DEFAULT_WR_ID 0

enum {
    LITE_QP_STATE_RESET = 0,
    LITE_QP_STATE_ERR = 6, // refer to IBV_QPS_ERR
};

struct LiteSendWr {
    struct SendWr wr;
    union {
        struct WrAuxInfo aux;
        struct WrExtInfo ext;
    };
};

struct TypicalLiteCqEntry {
    unsigned int phyId;
    unsigned int cqn;
    struct rdma_lite_cq *liteCq;
    struct rdma_lite_device_cq_attr deviceCqAttr;
    struct RaListHead list;
};

union OpLiteSupportData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int rsvd;
    } txData;

    struct {
        int supportLite;
    } rxData;
};

union OpLiteRdevCapData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int rsvd;
    } txData;

    struct {
        struct LiteRdevCapResp resp;
    } rxData;
};

union OpLiteQpCqAttrData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int qpn;
    } txData;

    struct {
        struct LiteQpCqAttrResp resp;
    } rxData;
};

union OpLiteQpAttrData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int qpn;
    } txData;

    struct {
        struct LiteQpAttrResp resp;
    } rxData;
};

union OpGetLiteCqAttrData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int cqn;
    } txData;

    struct {
        struct rdma_lite_device_cq_attr deviceCqAttr;
    } rxData;
};

union OpLiteMemAttrData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int qpn;
    } txData;

    struct {
        struct LiteMemAttrResp resp;
    } rxData;
};

union OpLiteConnectedInfoData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int qpn;
    } txData;

    struct {
        struct LiteConnectedInfoResp resp;
    } rxData;
};

int RaHdcLiteQpCreate(struct RaRdmaHandle *rdmaHandle, struct RaQpHandle *qpHdc, struct rdma_lite_qp_cap *cap);
void RaHdcLiteQpDestroy(struct RaQpHandle *qpHdc);
void RaHdcLiteQpDestroyWithoutCQ(struct RaQpHandle *qpHdc);
int RaHdcLiteInit(struct RaRdmaHandle *rdmaHandle, unsigned int phyId, unsigned int rdevIndex);
void RaHdcLiteDeinit(struct RaRdmaHandle *rdmaHandle);
int RaHdcLiteSendWr(struct RaQpHandle *qpHdc, struct LiteSendWr *wr, struct SendWrRsp *opRsp, unsigned long long wrId);
int RaHdcLiteTypicalSendWr(struct RaQpHandle *qpHdc, struct LiteSendWr *wr, struct SendWrRsp *opRsp,
    unsigned long long wrId);
int RaHdcLiteGetQpAttr(struct RaQpHandle *qpHdc, struct rdma_lite_qp_attr *liteQpAttr);
int RaHdcLiteGetConnectedInfo(struct RaQpHandle *qpHdc);
int RaHdcLiteSendWrlist(struct RaQpHandle *qpHdc, struct SendWrlistData wr[], struct SendWrRsp opRsp[],
    struct WrlistSendCompleteNum wrlistNum);
int RaHdcLiteSendWrlistExt(struct RaQpHandle *qpHdc, struct SendWrlistDataExt wr[], struct SendWrRsp opRsp[],
    struct WrlistSendCompleteNum wrlistNum);
int RaHdcLiteSendNormalWrlist(struct RaQpHandle *qpHdc, struct WrInfo wr[], struct SendWrRsp opRsp[],
    struct WrlistSendCompleteNum wrlistNum);
int RaHdcLiteRecvWrlist(struct RaQpHandle *qpHdc, struct RecvWrlistData *wr, unsigned int recvNum,
    unsigned int *completeNum);
int RaHdcLitePollCq(struct RaQpHandle *qpHdc, bool isSendCq, unsigned int numEntries, struct rdma_lite_wc_v2 *liteWc);
int RaHdcLiteCqCreate(struct RaRdmaHandle *rdmaHandle, unsigned int cqDepth, union OpTypicalCqCreateData *cqData,
    struct rdma_lite_cq **liteCq);
void RaHdcLiteStoreTypicalCq(struct RaRdmaHandle *rdmaHandle, unsigned int cqn, struct rdma_lite_cq *liteCq,
    struct rdma_lite_device_cq_attr *deviceCqAttr);
struct rdma_lite_cq *RaHdcLiteFindTypicalCq(struct RaRdmaHandle *rdmaHandle, unsigned int cqn);
int RaHdcLiteFindTypicalCqAttr(struct RaRdmaHandle *rdmaHandle, unsigned int cqn,
    struct rdma_lite_device_cq_attr **deviceCqAttr);
void RaHdcLiteRemoveTypicalCq(struct RaRdmaHandle *rdmaHandle, unsigned int cqn);
int RaHdcLiteGetCqAttr(struct RaRdmaHandle *rdmaHandle, unsigned int cqn,
    struct rdma_lite_device_cq_attr *deviceCqAttr);
int RaHdcLiteQpCreateWithCQ(struct RaRdmaHandle *rdmaHandle, struct RaQpHandle *qpHdc, struct rdma_lite_qp_cap *cap,
    struct rdma_lite_cq *sendLiteCq, struct rdma_lite_cq *recvLiteCq, unsigned int sendCqn, unsigned int recvCqn);
int RaHdcLiteInitCqeErrInfo(unsigned int phyId);
void RaHdcLiteDeinitCqeErrInfo(unsigned int phyId);
void RaHdcLiteGetCqeErrInfo(unsigned int phyId, struct CqeErrInfo *info);
int RaHdcLiteGetCqeErrInfoList(struct RaRdmaHandle *rdmaHandle, struct CqeErrInfo *infoList, unsigned int *num);
#endif // RA_HDC_LITE_H
