/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_HDC_RDMA_H
#define RA_HDC_RDMA_H
#include "ascend_hal.h"
#include "hccp_common.h"
#include "ra.h"
#include "ra_hdc.h"
#include "ra_rs_comm.h"

#define RA_MAX_BATCH_QP_MODIFY_NUM 768

union OpRdevInitData {
    struct {
        struct rdev rdevInfo;
        unsigned int rsvd;
    } txData;

    struct {
        unsigned int rdevIndex;
        unsigned int rsvd;
    } rxData;
};

union OpRdevInitWithBackupData {
    struct {
        struct rdev rdevInfo;
        struct rdev backupRdevInfo;
        unsigned int rsvd;
    } txData;

    struct {
        unsigned int rdevIndex;
        unsigned int rsvd;
    } rxData;
};

union OpRdevDeinitData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int rsvd;
    } txData;

    struct {
        unsigned int rsvd;
    } rxData;
};

union OpSetTsqpDepthData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int tempDepth;
        unsigned int rsvd;
    } txData;

    struct {
        unsigned int qpNum;
        unsigned int rsvd;
    } rxData;
};

union OpGetTsqpDepthData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int rsvd;
    } txData;

    struct {
        unsigned int tempDepth;
        unsigned int qpNum;
        unsigned int rsvd;
    } rxData;
};

union OpRdevGetPortStatusData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int rsvd;
    } txData;

    struct {
        enum PortStatus status;
        unsigned int rsvd;
    } rxData;
};

union OpQpCreateData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        int qpMode;
        int flag;
        int memAlign; // 0,1:4KB, 2:2MB
        unsigned int rsvd;
    } txData;

    struct {
        unsigned int qpn;
        unsigned int psn;
        unsigned int gidIdx;
        unsigned int rsvd;
    } rxData;
};

union OpQpCreateWithAttrsData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        struct QpExtAttrs extAttrs;
    } txData;

    struct {
        unsigned int qpn;
        unsigned int psn;
        unsigned int gidIdx;
        unsigned int rsvd;
    } rxData;
};

union OpAiQpCreateData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        struct QpExtAttrs extAttrs;
    } txData;

    struct {
        unsigned int qpn;
        unsigned long long aiQpAddr; // refer to struct ibv_qp *
        unsigned int sqIndex;        // index of sq
        unsigned int dbIndex;        // index of db
        unsigned int psn;
    } rxData;
};

union OpAiQpCreateWithAttrsData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        struct QpExtAttrs extAttrs;
    } txData;

    struct {
        unsigned int qpn;
        unsigned int gidIdx;
        unsigned int psn;
        unsigned long long aiQpAddr;  // refer to struct ibv_qp *
        unsigned int sqIndex;         // index of sq
        unsigned int dbIndex;         // index of db
        unsigned long long aiScqAddr; // refer to struct ibv_cq *scq
        unsigned long long aiRcqAddr; // refer to struct ibv_cq *rcq
        struct AiDataPlaneInfo dataPlaneInfo;
        unsigned int rsvd[32U];
    } rxData;
};

union OpTypicalQpCreateData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        int qpMode;
        int flag;
        int memAlign; // 0,1:4KB, 2:2MB
        unsigned int rsvd;
    } txData;

    struct {
        unsigned int qpn;
        unsigned int gidIdx;
        unsigned int psn;
        union ibv_gid gid;
    } rxData;
};

union OpQpDestroyData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int qpn;
    } txData;

    struct {
        unsigned int rsvd;
    } rxData;
};

union OpQpStatusData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int qpn;
    } txData;

    struct {
        int status;
    } rxData;
};

union OpQpInfoData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int qpn;
        unsigned int rsvd[32U];
    } txData;

    struct {
        int status;
        unsigned int udpSport;
        unsigned int rsvd[64U];
    } rxData;
};

union OpTypicalQpModifyData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        struct TypicalQp localQpInfo;
        struct TypicalQp remoteQpInfo;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } txData;

    struct {
        unsigned int udpSport;
        int pathMtu;
        unsigned int rsvd[RA_RSVD_NUM_2];
    } rxData;
};

union OpQpBatchModifyData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        int status;
        int qpn[RA_MAX_BATCH_QP_MODIFY_NUM];
        int qpnNum;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } txData;

    struct {
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

union OpQpConnectData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int qpn;
        unsigned int fd;
    } txData;

    struct {
        unsigned int rsvd[RA_RSVD_NUM_2];
    } rxData;
};

union OpMrRegData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int qpn;
        struct RdmaMrRegAttr mrRegAttr;
    } txData;

    struct {
        unsigned int lkey;
        unsigned int rkey;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

union OpMrDeregData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int qpn;
        unsigned int rsvd;
        char *addr;
    } txData;

    struct {
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

union OpTypicalMrRegData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        struct RdmaMrRegAttr mrRegAttr;
    } txData;

    struct {
        unsigned int lkey;
        unsigned int rkey;
        uint64_t addr;
        unsigned int rsvd[RA_RSVD_NUM_2];
    } rxData;
};

union OpRemapMrData {
    struct {
        struct MemRemapInfo memList[REMAP_MR_MAX_NUM];
        unsigned int memNum;
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } txData;
    struct {
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

union OpTypicalMrDeregData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        uint64_t addr;
    } txData;

    struct {
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

union OpSendWrData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int qpn;
        unsigned short bufNum;
        unsigned short rsvd;
        unsigned long long dstAddr;
        unsigned int op;
        int sendFlags;
        struct SgList memList[MAX_SGE_NUM];
    } txData;

    struct {
        struct SendWrRsp wrRsp;
        unsigned int rsvd[RA_RSVD_NUM_50];
    } rxData;
};

union OpSendWrlistData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int qpn;
        unsigned int sendNum;
        struct SendWrlistData wrlist[MAX_WR_NUM_V1];
    } txData;

    struct {
        unsigned int completeNum;
        struct SendWrRsp wrRsp[MAX_WR_NUM_V1];
        unsigned int rsvd[RA_RSVD_NUM_50];
    } rxData;
};

union OpSendWrlistDataExt {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int qpn;
        unsigned int sendNum;
        struct SendWrlistDataExt wrlist[MAX_WR_NUM_V1];
    } txData;

    struct {
        unsigned int completeNum;
        struct SendWrRsp wrRsp[MAX_WR_NUM_V1];
        unsigned int rsvd[RA_RSVD_NUM_50];
    } rxData;
};

union OpSendWrlistDataV2 {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int qpn;
        unsigned int sendNum;
        struct SendWrlistData wrlist[MAX_WR_NUM];
    } txData;

    struct {
        unsigned int completeNum;
        struct SendWrRsp wrRsp[MAX_WR_NUM];
        unsigned int rsvd[RA_RSVD_NUM_50];
    } rxData;
};

union OpSendWrlistDataExtV2 {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int qpn;
        unsigned int sendNum;
        struct SendWrlistDataExt wrlist[MAX_WR_NUM];
    } txData;

    struct {
        unsigned int completeNum;
        struct SendWrRsp wrRsp[MAX_WR_NUM];
        unsigned int rsvd[RA_RSVD_NUM_50];
    } rxData;
};

union OpSendNormalWrlistData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int qpn;
        unsigned int sendNum;
        struct WrInfo wrlist[MAX_WR_NUM];
        unsigned int rsvd[RA_RSVD_NUM_50];
    } txData;

    struct {
        unsigned int completeNum;
        struct SendWrRsp wrRsp[MAX_WR_NUM];
        unsigned int rsvd[RA_RSVD_NUM_50];
    } rxData;
};

union OpSetQpAttrQosData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int qpn;
        struct QosAttr qosAttr;
    } txData;

    struct {
        unsigned int rsvd[RA_RSVD_NUM_6];
    } rxData;
};

union OpSetQpAttrTimeoutData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int qpn;
        unsigned int timeout; // Rdma超时时间
    } txData;

    struct {
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

union OpSetQpAttrRetryCntData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int qpn;
        unsigned int retryCnt; // Rdma重传次数
    } txData;

    struct {
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

union OpGetCqeErrInfoNumData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } txData;
    struct {
        unsigned int num;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

union OpGetCqeErrInfoListData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int num;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } txData;
    struct {
        struct CqeErrInfo infoList[CQE_ERR_INFO_MAX_NUM];
        unsigned int num;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

int RaHdcQpCreate(struct RaRdmaHandle *rdmaHandle, int flag, int qpMode, void **qpHandle);
int RaHdcQpCreateWithAttrs(struct RaRdmaHandle *rdmaHandle, struct QpExtAttrs *extAttrs, void **qpHandle);
int RaHdcAiQpCreate(struct RaRdmaHandle *rdmaHandle, struct QpExtAttrs *extAttrs, struct AiQpInfo *info,
    void **qpHandle);
int RaHdcAiQpCreateWithAttrs(struct RaRdmaHandle *rdmaHandle, struct QpExtAttrs *extAttrs, struct AiQpInfo *info,
    void **qpHandle);
int RaHdcTypicalQpCreate(struct RaRdmaHandle *rdmaHandle, int flag, int qpMode, struct TypicalQp *qpInfo,
    void **qpHandle);
int RaHdcPollCq(struct RaQpHandle *qpHdc, bool isSendCq, unsigned int numEntries, void *wc);
int RaHdcQpDestroy(struct RaQpHandle *qpHdc);
int RaHdcTypicalQpModify(struct RaQpHandle *qpHdc, struct TypicalQp *localQpInfo, struct TypicalQp *remoteQpInfo);
int RaHdcQpConnectAsync(struct RaQpHandle *qpHdc, const void *sockHandle);
int RaHdcGetQpStatus(struct RaQpHandle *qpHdc, int *status);
int RaHdcMrReg(struct RaQpHandle *qpHdc, struct MrInfoT *info);
int RaHdcMrDereg(struct RaQpHandle *qpHdc, struct MrInfoT *info);
int RaHdcTypicalMrReg(struct RaRdmaHandle *rdmaHandle, struct MrInfoT *info, void **mrHandle);
int RaHdcRemapMr(struct RaRdmaHandle *rdmaHandle, struct MemRemapInfo info[], unsigned int num);
int RaHdcTypicalMrDereg(struct RaRdmaHandle *rdmaHandle, void *mrHandle);
int RaHdcSendWr(struct RaQpHandle *qpHdc, struct SendWr *wr, struct SendWrRsp *opRsp);
int RaHdcTypicalSendWr(struct RaQpHandle *qpHdc, struct SendWr *wr, struct SendWrRsp *opRsp);
int RaHdcSendWrV2(struct RaQpHandle *qpHdc, struct SendWrV2 *wr, struct SendWrRsp *opRsp);
int RaHdcSendWrlist(struct RaQpHandle *qpHdc, struct SendWrlistData wr[], struct SendWrRsp opRsp[],
    struct WrlistSendCompleteNum wrlistNum);
int RaHdcSendWrlistExt(struct RaQpHandle *qpHdc, struct SendWrlistDataExt wr[], struct SendWrRsp opRsp[],
    struct WrlistSendCompleteNum wrlistNum);
int RaHdcSendNormalWrlist(struct RaQpHandle *qpHdc, struct WrInfo wr[], struct SendWrRsp opRsp[],
    struct WrlistSendCompleteNum wrlistNum);
int RaHdcRecvWrlist(struct RaQpHandle *qpHdc, struct RecvWrlistData *wr, unsigned int recvNum,
    unsigned int *completeNum);
int RaHdcRdevInit(struct RaRdmaHandle *rdmaHandle, unsigned int notifyType, struct rdev rdevInfo,
    unsigned int *rdevIndex);
int RaHdcRdevGetPortStatus(struct RaRdmaHandle *rdmaHandle, enum PortStatus *status);
int RaHdcRdevDeinit(struct RaRdmaHandle *rdmaHandle, unsigned int notifyType);
int RaHdcRdevRestoreDeinit(struct RaRdmaHandle *rdmaHandle, unsigned int notifyType);
int RaHdcSetTsqpDepth(struct RaRdmaHandle *rdmaHandle, unsigned int tempDepth, unsigned int *qpNum);
int RaHdcGetTsqpDepth(struct RaRdmaHandle *rdmaHandle, unsigned int *tempDepth, unsigned int *qpNum);
int RaHdcSetQpAttrQos(struct RaQpHandle *qpHdc, struct QosAttr *attr);
int RaHdcSetQpAttrTimeout(struct RaQpHandle *qpHdc, unsigned int *timeout);
int RaHdcSetQpAttrRetryCnt(struct RaQpHandle *qpHdc, unsigned int *retryCnt);
int RaHdcGetCqeErrInfoList(struct RaRdmaHandle *rdmaHandle, struct CqeErrInfo *infoList, unsigned int *num);
int RaHdcQpBatchModify(struct RaRdmaHandle *rdmaHandle, void *qpHdc[], unsigned int num, int expectStatus);
int RaHdcRdmaSetOps(struct RaRdmaHandle *rdmaHandle, struct RaRdmaOps *rdmaOps);
int RaHdcRdmaSaveSnapshot(struct RaRdmaHandle *rdmaHandle, enum SaveSnapshotAction action);
int RaHdcRdmaRestoreSnapshot(struct RaRdmaHandle *rdmaHandle, struct RaRdmaOps *rdmaOps);
#endif // RA_HDC_RDMA_H
