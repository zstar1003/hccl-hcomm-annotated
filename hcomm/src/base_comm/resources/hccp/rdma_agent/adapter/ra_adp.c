/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <errno.h>
#include <sys/prctl.h>
#include "securec.h"
#include "user_log.h"
#include "config_log.h"
#include "dl_hal_function.h"
#include "ra_comm.h"
#include "ra_hdc.h"
#include "ra_hdc_async.h"
#include "ra_hdc_lite.h"
#include "ra_hdc_rdma_notify.h"
#include "ra_hdc_rdma.h"
#include "ra_hdc_socket.h"
#include "ra_hdc_ping.h"
#include "ra_rs_comm.h"
#include "ra_hdc_tlv.h"
#include "ra_rs_err.h"
#include "rs.h"
#include "rs_ping.h"
#include "ra_adp_tlv.h"
#include "ra_adp_ping.h"
#include "ra_adp_socket.h"
#include "ra_hdc_ctx.h"
#include "ra_hdc_async_ctx.h"
#include "ra_adp_ctx.h"
#include "ra_adp_async.h"
#include "ra_adp_verbs.h"
#include "ra_adp.h"

struct RaHdcServer gHdcServer[RA_MAX_PHY_ID_NUM] = {0};
struct RaHdcInitPara gHdcInitPara = {0};
struct RsPthreadInfo gRaThreadInfo = {0};

struct RsOps {
    int (*rdevInit)(struct rdev rdevInfo, unsigned int notifyType, unsigned int *rdevIndex);
    int (*rdevInitWithBackup)(struct rdev rdevInfo, struct rdev backupRdevInfo, unsigned int notifyType,
        unsigned int *rdevIndex);
    int (*rdevGetPortStatus)(unsigned int phyId, unsigned int rdevIndex, enum PortStatus *status);
    int (*rdevDeinit)(unsigned int phyId, unsigned int notifyType, unsigned int rdevIndex);
    int (*qpCreate)(unsigned int phyId, unsigned int rdevIndex, struct RsQpNorm qpNorm, struct RsQpResp *qpResp);
    int (*qpCreateWithAttrs)(unsigned int phyId, unsigned int rdevIndex, struct RsQpNormWithAttrs *qpNorm,
        struct RsQpRespWithAttrs *qpResp);
    int (*qpDestroy)(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn);
    int (*typicalQpModify)(unsigned int phyId, unsigned int rdevIndex, struct TypicalQp localQpInfo,
        struct TypicalQp remoteQpInfo, struct TypicalQpAttr *qpAttr);
    int (*qpBatchModify)(unsigned int phyId, unsigned int rdevIndex, int status, int qpn[], int qpnNum);
    int (*qpConnectAsync)(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, int fd);
    int (*getQpStatus)(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, struct RsQpStatusInfo *qpInfo);
    int (*mrReg)(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, struct RdmaMrRegInfo *mrRegInfo);
    int (*mrDereg)(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, char *addr);
    int (*registerMr)(unsigned int phyId, unsigned int rdevIndex, struct RdmaMrRegInfo *mrRegInfo, void **mrHandle);
    int (*typicalRegisterMr)(unsigned int phyId, unsigned int rdevIndex, struct RdmaMrRegInfo *mrRegInfo,
        void **mrHandle);
    int (*remapMr)(unsigned int phyId, unsigned int rdevIndex, struct MemRemapInfo memList[], unsigned int memNum);
    int (*typicalDeregisterMr)(unsigned int phyId, unsigned int devIndex, unsigned long long addr);
    int (*sendWr)(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, struct SendWr *wr,
        struct SendWrRsp *wrRsp);
    int (*sendWrList)(struct RsWrlistBaseInfo baseInfo, struct WrInfo *wrList, unsigned int sendNum,
        struct SendWrRsp *wrRsp, unsigned int *completeNum);
    int (*getNotifyMrInfo)(unsigned int phyId, unsigned int rdevIndex, struct MrInfoT *info);
    int (*notifyCfgSet)(unsigned int phyId, unsigned long long va, unsigned long long size);
    int (*notifyCfgGet)(unsigned int phyId, unsigned long long *va, unsigned long long *size);
    int (*setHostPid)(unsigned int phyId, pid_t hostPid, const char *pidSign);
    int (*getInterfaceVersion)(unsigned int opcode, unsigned int *version);
    int (*setTsqpDepth)(unsigned int phyId, unsigned int rdevIndex, unsigned int tempDepth, unsigned int *qpNum);
    int (*getTsqpDepth)(unsigned int phyId, unsigned int rdevIndex, unsigned int *tempDepth, unsigned int *qpNum);
    int (*setQpAttrQos)(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, struct QosAttr *attr);
    int (*setQpAttrTimeout)(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, unsigned int *timeout);
    int (*setQpAttrRetryCnt)(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, unsigned int *retryCnt);
    int (*getCqeErrInfo)(struct CqeErrInfo *info);
    int (*getLiteSupport)(unsigned int phyId, unsigned int rdevIndex, int *supportLite);
    int (*getLiteRdevCap)(unsigned int phyId, unsigned int rdevIndex, struct LiteRdevCapResp *resp);
    int (*getLiteQpCqAttr)(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, struct LiteQpCqAttrResp *resp);
    int (*getLiteMemAttr)(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, struct LiteMemAttrResp *resp);
    int (*getLiteConnectedInfo)(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn,
        struct LiteConnectedInfoResp *resp);
    int (*getCqeErrInfoNum)(unsigned int phyId, unsigned int rdevIdx, unsigned int *num);
    int (*getCqeErrInfoList)(unsigned int phyId, unsigned int rdevIdx, struct CqeErrInfo *info, unsigned int *num);
    int (*getTlsEnable)(unsigned int phyId, bool *tlsEnable);
    int (*getSecRandom)(int *value);
    int (*getHccnCfg)(struct RaInfo *info, enum HccnCfgKey key, char *value, unsigned int *valueLen);
};

struct RsOps gRaRsOps = {
    .rdevInit = RsRdevInit,
    .rdevInitWithBackup = RsRdevInitWithBackup,
    .rdevGetPortStatus = RsRdevGetPortStatus,
    .rdevDeinit = RsRdevDeinit,
    .qpCreate = RsQpCreate,
    .qpCreateWithAttrs = RsQpCreateWithAttrs,
    .qpDestroy = RsQpDestroy,
    .typicalQpModify = RsTypicalQpModify,
    .qpBatchModify = RsQpBatchModify,
    .qpConnectAsync = RsQpConnectAsync,
    .getQpStatus = RsGetQpStatus,
    .mrReg = RsMrReg,
    .mrDereg = RsMrDereg,
    .registerMr = RsTypicalRegisterMrV1,
    .typicalRegisterMr = RsTypicalRegisterMr,
    .remapMr = RsRemapMr,
    .typicalDeregisterMr = RsTypicalDeregisterMr,
    .sendWr = RsSendWr,
    .sendWrList = RsSendWrlist,
    .getNotifyMrInfo = RsGetNotifyMrInfo,
    .notifyCfgSet = RsNotifyCfgSet,
    .notifyCfgGet = RsNotifyCfgGet,
    .setHostPid = RsSetHostPid,
    .getInterfaceVersion = RsGetInterfaceVersion,
    .setTsqpDepth = RsSetTsqpDepth,
    .getTsqpDepth = RsGetTsqpDepth,
    .setQpAttrQos = RsSetQpAttrQos,
    .setQpAttrTimeout = RsSetQpAttrTimeout,
    .setQpAttrRetryCnt = RsSetQpAttrRetryCnt,
    .getCqeErrInfo = RsGetCqeErrInfo,
    .getLiteRdevCap = RsGetLiteRdevCap,
    .getLiteQpCqAttr = RsGetLiteQpCqAttr,
    .getLiteConnectedInfo = RsGetLiteConnectedInfo,
    .getLiteMemAttr = RsGetLiteMemAttr,
    .getLiteSupport = RsGetLiteSupport,
    .getCqeErrInfoNum = RsGetCqeErrInfoNum,
    .getCqeErrInfoList = RsGetCqeErrInfoList,
    .getTlsEnable = RsGetTlsEnable,
    .getSecRandom = RsDrvGetRandomNum,
    .getHccnCfg = RsGetHccnCfg,
};

struct HdcOps gRaHdcOps = {
    .getCapacity = DlDrvHdcGetCapacity,
    .clientCreate = DlDrvHdcClientCreate,
    .clientDestroy = DlDrvHdcClientDestroy,
    .sessionConnect = DlDrvHdcSessionConnect,
    .sessionConnectEx = DlHalHdcSessionConnectEx,
    .serverCreate = DlDrvHdcServerCreate,
    .serverDestroy = DlDrvHdcServerDestroy,
    .sessionAccept = DlDrvHdcSessionAccept,
    .sessionClose = DlDrvHdcSessionClose,
    .freeMsg = DlDrvHdcFreeMsg,
    .reuseMsg = DlDrvHdcReuseMsg,
    .addMsgBuffer = DlDrvHdcAddMsgBuffer,
    .getMsgBuffer = DlDrvHdcGetMsgBuffer,
    .recv = DlHalHdcRecv,
    .send = DlHalHdcSend,
    .allocMsg = DlDrvHdcAllocMsg,
    .setSessionReference = DlDrvHdcSetSessionReference,
};

#define RA_HDC_OPS gRaHdcOps

STATIC void MsgHeadBuildUpHw(char *pSendRcvBuf, struct MsgHead *recvMsgHead, int ret, unsigned int msgDataLen)
{
    struct MsgHead *pSendRcvHead = NULL;

    pSendRcvHead = (struct MsgHead *)pSendRcvBuf;
    pSendRcvHead->opcode = recvMsgHead->opcode;
    pSendRcvHead->asyncReqId = recvMsgHead->asyncReqId;
    pSendRcvHead->ret = ret;
    pSendRcvHead->msgDataLen = msgDataLen;

    return;
}

STATIC int OpMsgErr(char **outBuf, struct MsgHead *recvMsgHead, int *outBufLen, int opRight)
{
    unsigned int opcode = recvMsgHead->opcode;
    char *outBufTmp = NULL;
    int msgRet = 0;

    outBufTmp = (char *)calloc(sizeof(struct MsgHead), sizeof(char));
    CHK_PRT_RETURN(outBufTmp == NULL, hccp_err("send_buf calloc failed."), -ENOMEM);

    if (opRight == HAVE_OP_RIGHT) {
        if (opcode >= RA_RS_OP_MAX_NUM || ((opcode < RA_RS_HDC_SESSION_CLOSE) && (opcode >= RA_RS_EXTER_OP_MAX_NUM))) {
            msgRet = -EPROTONOSUPPORT;
        } else {
            msgRet = -EPIPE;
        }
    } else if (opRight == TGID_INVALID) {
        msgRet = -EPERM;
    } else {
        msgRet = -EACCES;
    }

    MsgHeadBuildUpHw(outBufTmp, recvMsgHead, msgRet, 0);

    *outBuf = outBufTmp;
    *outBufLen = sizeof(struct MsgHead);

    return 0;
}

STATIC int RaRsRdevInit(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    unsigned int rdevIndex = 0;
    union OpRdevInitData *rdevInitData = (union OpRdevInitData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpRdevInitData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.rdevInit(rdevInitData->txData.rdevInfo, NOTIFY, &rdevIndex);
    if (*opResult != 0) {
        hccp_err("rdev_init failed ret[%d].", *opResult);
        return 0;
    }

    rdevInitData = (union OpRdevInitData *)(outBuf + sizeof(struct MsgHead));
    rdevInitData->rxData.rdevIndex = rdevIndex;

    return 0;
}

STATIC int RaRsRdevInitWithBackup(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpRdevInitWithBackupData *rdevInitData = (union OpRdevInitWithBackupData *)(inBuf + sizeof(struct MsgHead));
    unsigned int rdevIndex = 0;

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpRdevInitWithBackupData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.rdevInitWithBackup(rdevInitData->txData.rdevInfo, rdevInitData->txData.backupRdevInfo, NOTIFY,
        &rdevIndex);
    if (*opResult != 0) {
        hccp_err("rdev_init_with_backup failed ret[%d].", *opResult);
        return 0;
    }

    rdevInitData = (union OpRdevInitWithBackupData *)(outBuf + sizeof(struct MsgHead));
    rdevInitData->rxData.rdevIndex = rdevIndex;

    return 0;
}

STATIC int RaRsRdevGetPortStatus(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpRdevGetPortStatusData *statusData = NULL;
    enum PortStatus status = PORT_STATUS_DOWN;

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpRdevGetPortStatusData), sizeof(struct MsgHead), rcvBufLen, opResult);

    statusData = (union OpRdevGetPortStatusData *)(inBuf + sizeof(struct MsgHead));
    *opResult = gRaRsOps.rdevGetPortStatus(statusData->txData.phyId, statusData->txData.rdevIndex, &status);
    if (*opResult != 0) {
        hccp_err("rdev_get_port_status failed ret[%d].", *opResult);
        return 0;
    }

    statusData = (union OpRdevGetPortStatusData *)(outBuf + sizeof(struct MsgHead));
    statusData->rxData.status = status;

    return 0;
}

STATIC int RaRsRdevDeinit(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpRdevDeinitData *rdevDeinitData = (union OpRdevDeinitData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpRdevDeinitData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.rdevDeinit(rdevDeinitData->txData.phyId, NOTIFY, rdevDeinitData->txData.rdevIndex);
    if (*opResult != 0) {
        hccp_err("rdev_deinit failed ret[%d].", *opResult);
    }

    return 0;
}

STATIC int RaRsGetTsqpDepth(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    unsigned int tempDepth = 0;
    unsigned int qpNum = 0;
    union OpGetTsqpDepthData *getTsqpDepthData = (union OpGetTsqpDepthData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpGetTsqpDepthData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.getTsqpDepth(getTsqpDepthData->txData.phyId, getTsqpDepthData->txData.rdevIndex, &tempDepth,
        &qpNum);
    if (*opResult != 0) {
        hccp_err("set_tsqp_depth failed ret[%d].", *opResult);
        return 0;
    }

    getTsqpDepthData = (union OpGetTsqpDepthData *)(outBuf + sizeof(struct MsgHead));
    getTsqpDepthData->rxData.tempDepth = tempDepth;
    getTsqpDepthData->rxData.qpNum = qpNum;

    return 0;
}

STATIC int RaRsSetTsqpDepth(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    unsigned int qpNum = 0;
    union OpSetTsqpDepthData *setTsqpDepthData = (union OpSetTsqpDepthData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSetTsqpDepthData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.setTsqpDepth(setTsqpDepthData->txData.phyId, setTsqpDepthData->txData.rdevIndex,
        setTsqpDepthData->txData.tempDepth, &qpNum);
    if (*opResult != 0) {
        hccp_err("set_tsqp_depth failed ret[%d].", *opResult);
        return 0;
    }

    setTsqpDepthData = (union OpSetTsqpDepthData *)(outBuf + sizeof(struct MsgHead));
    setTsqpDepthData->rxData.qpNum = qpNum;

    return 0;
}

STATIC int RaRsQpCreate(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    struct RsQpNorm qpNorm;
    struct RsQpResp qpResp = {0};
    union OpQpCreateData *createData = (union OpQpCreateData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpQpCreateData), sizeof(struct MsgHead), rcvBufLen, opResult);

    int qpMode = createData->txData.qpMode;
    qpNorm.flag = createData->txData.flag;
    qpNorm.isExp = 1;
    qpNorm.isExt = 1;
    if (qpMode == RA_RS_OP_QP_MODE_EXT) {
        qpNorm.qpMode = RA_RS_OP_QP_MODE;
    } else {
        qpNorm.qpMode = qpMode;
    }
    qpNorm.memAlign = createData->txData.memAlign;

    *opResult = gRaRsOps.qpCreate(createData->txData.phyId, createData->txData.rdevIndex, qpNorm, &qpResp);
    if (*opResult != 0) {
        hccp_err("qp create failed ret[%d].", *opResult);
        return 0;
    }

    createData = (union OpQpCreateData *)(outBuf + sizeof(struct MsgHead));
    createData->rxData.qpn = qpResp.qpn;
    createData->rxData.psn = qpResp.psn;
    createData->rxData.gidIdx = qpResp.gidIdx;

    return 0;
}

STATIC int RaRsQpCreateWithAttrs(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpQpCreateWithAttrsData *createData = NULL;
    struct RsQpNormWithAttrs qpNorm = {0};
    struct RsQpRespWithAttrs qpResp = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpQpCreateWithAttrsData), sizeof(struct MsgHead), rcvBufLen, opResult);

    createData = (union OpQpCreateWithAttrsData *)(inBuf + sizeof(struct MsgHead));

    qpNorm.isExp = 1;
    qpNorm.isExt = 1;
    qpNorm.extAttrs = createData->txData.extAttrs;

    *opResult = gRaRsOps.qpCreateWithAttrs(createData->txData.phyId, createData->txData.rdevIndex, &qpNorm, &qpResp);
    if (*opResult != 0) {
        hccp_err("qp create failed ret[%d].", *opResult);
        return 0;
    }

    createData = (union OpQpCreateWithAttrsData *)(outBuf + sizeof(struct MsgHead));
    createData->rxData.qpn = qpResp.qpn;
    createData->rxData.psn = qpResp.psn;
    createData->rxData.gidIdx = qpResp.gidIdx;

    return 0;
}

STATIC int RaRsAiQpCreate(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpAiQpCreateData *createData = NULL;
    struct RsQpNormWithAttrs qpNorm = {0};
    struct RsQpRespWithAttrs qpResp = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpAiQpCreateData), sizeof(struct MsgHead), rcvBufLen, opResult);

    createData = (union OpAiQpCreateData *)(inBuf + sizeof(struct MsgHead));

    qpNorm.isExp = 1;
    qpNorm.isExt = 1;
    qpNorm.extAttrs = createData->txData.extAttrs;
    qpNorm.aiOpSupport = 1;

    *opResult = gRaRsOps.qpCreateWithAttrs(createData->txData.phyId, createData->txData.rdevIndex, &qpNorm, &qpResp);
    if (*opResult != 0) {
        hccp_err("qp create failed ret[%d].", *opResult);
        return 0;
    }

    createData = (union OpAiQpCreateData *)(outBuf + sizeof(struct MsgHead));
    createData->rxData.qpn = qpResp.qpn;
    createData->rxData.aiQpAddr = qpResp.aiQpAddr;
    createData->rxData.sqIndex = qpResp.sqIndex;
    createData->rxData.dbIndex = qpResp.dbIndex;
    createData->rxData.psn = qpResp.psn;

    return 0;
}

STATIC int RaRsAiQpCreateWithData(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpAiQpCreateWithAttrsData *createData = NULL;
    struct RsQpNormWithAttrs qpNorm = {0};
    struct RsQpRespWithAttrs qpResp = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpAiQpCreateWithAttrsData), sizeof(struct MsgHead), rcvBufLen, opResult);

    createData = (union OpAiQpCreateWithAttrsData *)(inBuf + sizeof(struct MsgHead));

    qpNorm.isExp = 1;
    qpNorm.isExt = 1;
    qpNorm.extAttrs = createData->txData.extAttrs;
    qpNorm.aiOpSupport = 1;

    *opResult = gRaRsOps.qpCreateWithAttrs(createData->txData.phyId, createData->txData.rdevIndex, &qpNorm, &qpResp);
    if (*opResult != 0) {
        hccp_err("qp create failed ret[%d].", *opResult);
        return 0;
    }

    createData = (union OpAiQpCreateWithAttrsData *)(outBuf + sizeof(struct MsgHead));
    createData->rxData.qpn = qpResp.qpn;
    createData->rxData.gidIdx = qpResp.gidIdx;
    createData->rxData.psn = qpResp.psn;
    createData->rxData.aiQpAddr = qpResp.aiQpAddr;
    createData->rxData.sqIndex = qpResp.sqIndex;
    createData->rxData.dbIndex = qpResp.dbIndex;
    createData->rxData.aiScqAddr = qpResp.aiScqAddr;
    createData->rxData.aiRcqAddr = qpResp.aiRcqAddr;
    (void)memcpy_s(&createData->rxData.dataPlaneInfo, sizeof(struct AiDataPlaneInfo), &qpResp.dataPlaneInfo,
        sizeof(struct AiDataPlaneInfo));

    return 0;
}

STATIC int RaRsTypicalQpCreate(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpTypicalQpCreateData *createData = (union OpTypicalQpCreateData *)(inBuf + sizeof(struct MsgHead));
    struct RsQpResp qpResp = {0};
    struct RsQpNorm qpNorm;
    int qpMode;

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpTypicalQpCreateData), sizeof(struct MsgHead), rcvBufLen, opResult);

    qpMode = createData->txData.qpMode;
    qpNorm.flag = createData->txData.flag;
    qpNorm.isExp = 1;
    qpNorm.isExt = 1;
    if (qpMode == RA_RS_OP_QP_MODE_EXT) {
        qpNorm.qpMode = RA_RS_OP_QP_MODE;
    } else {
        qpNorm.qpMode = qpMode;
    }
    qpNorm.memAlign = createData->txData.memAlign;

    *opResult = gRaRsOps.qpCreate(createData->txData.phyId, createData->txData.rdevIndex, qpNorm, &qpResp);
    if (*opResult != 0) {
        hccp_err("qp create failed ret[%d].", *opResult);
        return 0;
    }

    createData = (union OpTypicalQpCreateData *)(outBuf + sizeof(struct MsgHead));
    createData->rxData.qpn = qpResp.qpn;
    createData->rxData.gidIdx = qpResp.gidIdx;
    createData->rxData.psn = qpResp.psn;
    createData->rxData.gid = qpResp.gid;

    return 0;
}

STATIC int RaRsQpDestroy(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpQpDestroyData *qpDestroyData = (union OpQpDestroyData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpQpDestroyData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.qpDestroy(qpDestroyData->txData.phyId, qpDestroyData->txData.rdevIndex,
        qpDestroyData->txData.qpn);
    if (*opResult != 0) {
        hccp_err("qp destroy failed ret[%d].", *opResult);
    }

    return 0;
}

STATIC int RaRsTypicalQpModify(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpTypicalQpModifyData *qpModifyData = (union OpTypicalQpModifyData *)(inBuf + sizeof(struct MsgHead));
    struct TypicalQpAttr qpAttr = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpTypicalQpModifyData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.typicalQpModify(qpModifyData->txData.phyId, qpModifyData->txData.rdevIndex,
        qpModifyData->txData.localQpInfo, qpModifyData->txData.remoteQpInfo, &qpAttr);
    if (*opResult != 0) {
        hccp_err("qp info modify failed ret[%d].", *opResult);
        return 0;
    }

    qpModifyData = (union OpTypicalQpModifyData *)(outBuf + sizeof(struct MsgHead));
    qpModifyData->rxData.udpSport = qpAttr.udpSport;
    qpModifyData->rxData.pathMtu = qpAttr.pathMtu;

    return 0;
}

STATIC int RaRsQpBatchModify(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpQpBatchModifyData *qpBatchModifyData = (union OpQpBatchModifyData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpQpBatchModifyData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(qpBatchModifyData->txData.qpnNum, 0, RA_MAX_BATCH_QP_MODIFY_NUM, opResult);

    *opResult = gRaRsOps.qpBatchModify(qpBatchModifyData->txData.phyId, qpBatchModifyData->txData.rdevIndex,
        qpBatchModifyData->txData.status, qpBatchModifyData->txData.qpn, qpBatchModifyData->txData.qpnNum);
    if (*opResult != 0) {
        hccp_err("qp info modify failed ret[%d].", *opResult);
    }

    return 0;
}

STATIC int RaRsQpConnectAsync(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpQpConnectData *qpConnectData = (union OpQpConnectData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpQpConnectData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.qpConnectAsync(qpConnectData->txData.phyId, qpConnectData->txData.rdevIndex,
        qpConnectData->txData.qpn, qpConnectData->txData.fd);
    if (*opResult != 0) {
        hccp_err("qp info async failed ret[%d].", *opResult);
    }

    return 0;
}

STATIC int RaRsGetQpStatus(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpQpStatusData *qpStatusData = (union OpQpStatusData *)(inBuf + sizeof(struct MsgHead));
    struct RsQpStatusInfo qpInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpQpStatusData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.getQpStatus(qpStatusData->txData.phyId, qpStatusData->txData.rdevIndex,
        qpStatusData->txData.qpn, &qpInfo);
    if (*opResult != 0) {
        hccp_err("query qp status async failed ret[%d].", *opResult);
        return 0;
    }

    qpStatusData = (union OpQpStatusData *)(outBuf + sizeof(struct MsgHead));
    qpStatusData->rxData.status = qpInfo.status;

    return 0;
}

STATIC int RaRsGetQpInfo(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpQpInfoData *qpInfoData = (union OpQpInfoData *)(inBuf + sizeof(struct MsgHead));
    struct RsQpStatusInfo qpInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpQpInfoData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.getQpStatus(qpInfoData->txData.phyId, qpInfoData->txData.rdevIndex, qpInfoData->txData.qpn,
        &qpInfo);
    if (*opResult != 0) {
        hccp_err("query qp status async failed ret[%d].", *opResult);
        return 0;
    }

    qpInfoData = (union OpQpInfoData *)(outBuf + sizeof(struct MsgHead));
    qpInfoData->rxData.status = qpInfo.status;
    qpInfoData->rxData.udpSport = qpInfo.udpSport;

    return 0;
}

STATIC int RaRsMrReg(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpMrRegData *regMrData = (union OpMrRegData *)(inBuf + sizeof(struct MsgHead));
    struct RdmaMrRegInfo mrRegInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpMrRegData), sizeof(struct MsgHead), rcvBufLen, opResult);

    mrRegInfo.addr = regMrData->txData.mrRegAttr.addr;
    mrRegInfo.len = regMrData->txData.mrRegAttr.len;
    mrRegInfo.access = regMrData->txData.mrRegAttr.access;
    *opResult = gRaRsOps.mrReg(regMrData->txData.phyId, regMrData->txData.rdevIndex, regMrData->txData.qpn, &mrRegInfo);
    if (*opResult != 0) {
        hccp_err("reg_mr failed ret[%d].", *opResult);
        return 0;
    }

    regMrData = (union OpMrRegData *)(outBuf + sizeof(struct MsgHead));
    regMrData->rxData.lkey = mrRegInfo.lkey;
    regMrData->rxData.rkey = mrRegInfo.rkey;

    return 0;
}

STATIC int RaRsMrDereg(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpMrDeregData *mrDeregData = (union OpMrDeregData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpMrDeregData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.mrDereg(mrDeregData->txData.phyId, mrDeregData->txData.rdevIndex, mrDeregData->txData.qpn,
        mrDeregData->txData.addr);
    if (*opResult != 0) {
        hccp_err("dereg_mr failed ret[%d].", *opResult);
    }

    return 0;
}

STATIC int RaRsTypicalMrRegV1(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpTypicalMrRegData *regMrData = (union OpTypicalMrRegData *)(inBuf + sizeof(struct MsgHead));
    struct RdmaMrRegInfo mrRegInfo = {0};
    struct ibv_mr *raRsMrHandle = NULL;

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpTypicalMrRegData), sizeof(struct MsgHead), rcvBufLen, opResult);

    mrRegInfo.addr = regMrData->txData.mrRegAttr.addr;
    mrRegInfo.len = regMrData->txData.mrRegAttr.len;
    mrRegInfo.access = regMrData->txData.mrRegAttr.access;
    *opResult = gRaRsOps.registerMr(regMrData->txData.phyId, regMrData->txData.rdevIndex, &mrRegInfo,
        (void **)&raRsMrHandle);
    if (*opResult != 0) {
        hccp_err("reg_mr failed ret[%d].", *opResult);
        return 0;
    }

    regMrData = (union OpTypicalMrRegData *)(outBuf + sizeof(struct MsgHead));
    regMrData->rxData.lkey = mrRegInfo.lkey;
    regMrData->rxData.rkey = mrRegInfo.rkey;

    return 0;
}

STATIC int RaRsTypicalMrReg(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpTypicalMrRegData *regMrData = (union OpTypicalMrRegData *)(inBuf + sizeof(struct MsgHead));
    struct RdmaMrRegInfo mrRegInfo = {0};
    struct ibv_mr *raRsMrHandle = NULL;

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpTypicalMrRegData), sizeof(struct MsgHead), rcvBufLen, opResult);

    mrRegInfo.addr = regMrData->txData.mrRegAttr.addr;
    mrRegInfo.len = regMrData->txData.mrRegAttr.len;
    mrRegInfo.access = regMrData->txData.mrRegAttr.access;
    *opResult = gRaRsOps.typicalRegisterMr(regMrData->txData.phyId, regMrData->txData.rdevIndex, &mrRegInfo,
        (void **)&raRsMrHandle);
    if (*opResult != 0) {
        hccp_err("reg_mr failed ret[%d].", *opResult);
        return 0;
    }

    regMrData = (union OpTypicalMrRegData *)(outBuf + sizeof(struct MsgHead));
    regMrData->rxData.lkey = mrRegInfo.lkey;
    regMrData->rxData.rkey = mrRegInfo.rkey;
    regMrData->rxData.addr = (uint64_t)(uintptr_t)raRsMrHandle;

    return 0;
}

STATIC int RaRsRemapMr(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpRemapMrData *opData = (union OpRemapMrData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpRemapMrData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(opData->txData.memNum, 0, REMAP_MR_MAX_NUM, opResult);

    *opResult = gRaRsOps.remapMr(opData->txData.phyId, opData->txData.rdevIndex, opData->txData.memList,
        opData->txData.memNum);
    if (*opResult) {
        hccp_err("remap_mr failed ret[%d]", *opResult);
    }
    return 0;
}

STATIC int RaRsTypicalMrDereg(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpTypicalMrDeregData *mrDeregData = (union OpTypicalMrDeregData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpTypicalMrDeregData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.typicalDeregisterMr(mrDeregData->txData.phyId, mrDeregData->txData.rdevIndex,
        mrDeregData->txData.addr);
    if (*opResult != 0) {
        hccp_err("dereg_mr failed ret[%d].", *opResult);
    }

    return 0;
}

STATIC int RaRsSendWr(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    int ret;

    struct SendWrRsp wrRsp = {0};
    union OpSendWrData *sendWrData = (union OpSendWrData *)(inBuf + sizeof(struct MsgHead));
    struct SendWr sWr = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSendWrData), sizeof(struct MsgHead), rcvBufLen, opResult);

    sWr.bufNum = sendWrData->txData.bufNum;
    sWr.bufList = (struct SgList *)&sendWrData->txData.memList[0];
    sWr.dstAddr = sendWrData->txData.dstAddr;
    sWr.op = sendWrData->txData.op;
    sWr.sendFlag = sendWrData->txData.sendFlags;

    ret = gRaRsOps.sendWr(sendWrData->txData.phyId, sendWrData->txData.rdevIndex, sendWrData->txData.qpn, &sWr, &wrRsp);
    *opResult = ret;
    if (ret) {
        if (ret == -ENOENT) {
            hccp_warn_rma("not found remote mr_info, need try again");
        } else {
            hccp_err("send wr failed ret[%d].", ret);
        }
    }

    sendWrData = (union OpSendWrData *)(outBuf + sizeof(struct MsgHead));
    sendWrData->rxData.wrRsp = wrRsp;

    return 0;
}

STATIC int RaRsSendWrList(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    int ret;
    uint32_t i;
    unsigned int completeNum = 0;
    struct SendWrRsp *wrRsp = NULL;
    struct WrInfo *wrList = NULL;
    struct RsWrlistBaseInfo baseInfo = {0};
    union OpSendWrlistData *sendWrlist = (union OpSendWrlistData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSendWrlistData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(sendWrlist->txData.sendNum, 0, MAX_WR_NUM, opResult);

    wrRsp = calloc(sendWrlist->txData.sendNum, sizeof(struct SendWrRsp));
    CHK_PRT_RETURN(wrRsp == NULL, hccp_err("wr_rsp calloc failed."), -ENOMEM);

    wrList = calloc(sendWrlist->txData.sendNum, sizeof(struct WrInfo));
    if (wrList == NULL) {
        hccp_err("wr_list calloc failed.");
        ret = -ENOMEM;
        goto alloc_wr_list_fail;
    }

    baseInfo.phyId = sendWrlist->txData.phyId;
    baseInfo.rdevIndex = sendWrlist->txData.rdevIndex;
    baseInfo.qpn = sendWrlist->txData.qpn;
    baseInfo.keyFlag = 0;

    for (i = 0; i < sendWrlist->txData.sendNum; i++) {
        wrList[i].op = sendWrlist->txData.wrlist[i].op;
        wrList[i].sendFlags = sendWrlist->txData.wrlist[i].sendFlags;
        wrList[i].dstAddr = sendWrlist->txData.wrlist[i].dstAddr;
        wrList[i].memList.addr = sendWrlist->txData.wrlist[i].memList.addr;
        wrList[i].memList.len = sendWrlist->txData.wrlist[i].memList.len;
        wrList[i].memList.lkey = sendWrlist->txData.wrlist[i].memList.lkey;
    }
    ret = gRaRsOps.sendWrList(baseInfo, wrList, sendWrlist->txData.sendNum, wrRsp, &completeNum);
    *opResult = ret;
    if (ret) {
        if (ret == -ENOENT) {
            hccp_warn_rma("not found remote mr_info, need try again");
        } else {
            hccp_err("send wr failed ret[%d].", ret);
        }
    }
    sendWrlist = (union OpSendWrlistData *)(outBuf + sizeof(struct MsgHead));
    sendWrlist->rxData.completeNum = completeNum;
    ret = memcpy_s(sendWrlist->rxData.wrRsp, sizeof(struct SendWrRsp) * MAX_WR_NUM_V1, wrRsp,
        completeNum * sizeof(struct SendWrRsp));
    if (ret) {
        hccp_err("ra_rs_send_wr_list memcpy_s failed, ret[%d]. ", ret);
        ret = -ESAFEFUNC;
        goto copy_wr_rsp_fail;
    }

copy_wr_rsp_fail:
    free(wrList);
    wrList = NULL;

alloc_wr_list_fail:
    free(wrRsp);
    wrRsp = NULL;
    return 0;
}

STATIC void GetWrListV2(struct WrInfo *wrList, union OpSendWrlistDataV2 *sendWrlist)
{
    uint32_t i;
    for (i = 0; i < sendWrlist->txData.sendNum; i++) {
        wrList[i].op = sendWrlist->txData.wrlist[i].op;
        wrList[i].sendFlags = sendWrlist->txData.wrlist[i].sendFlags;
        wrList[i].dstAddr = sendWrlist->txData.wrlist[i].dstAddr;
        wrList[i].memList.addr = sendWrlist->txData.wrlist[i].memList.addr;
        wrList[i].memList.len = sendWrlist->txData.wrlist[i].memList.len;
        wrList[i].memList.lkey = sendWrlist->txData.wrlist[i].memList.lkey;
    }
    return;
}

STATIC int RaRsSendWrListV2(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    int ret;
    unsigned int completeNum = 0;
    struct WrInfo *wrList = NULL;
    struct SendWrRsp *wrRsp = NULL;
    struct RsWrlistBaseInfo baseInfo = {0};
    union OpSendWrlistDataV2 *sendWrlist = (union OpSendWrlistDataV2 *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSendWrlistDataV2), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(sendWrlist->txData.sendNum, 0, MAX_WR_NUM, opResult);

    wrRsp = calloc(sendWrlist->txData.sendNum, sizeof(struct SendWrRsp));
    CHK_PRT_RETURN(wrRsp == NULL, hccp_err("wr_rsp calloc failed."), -ENOMEM);

    wrList = calloc(sendWrlist->txData.sendNum, sizeof(struct WrInfo));
    if (wrList == NULL) {
        hccp_err("wr_list calloc failed.");
        ret = -ENOMEM;
        goto alloc_wr_list_fail;
    }

    baseInfo.phyId = sendWrlist->txData.phyId;
    baseInfo.rdevIndex = sendWrlist->txData.rdevIndex;
    baseInfo.qpn = sendWrlist->txData.qpn;
    baseInfo.keyFlag = 0;

    GetWrListV2(wrList, sendWrlist);
    ret = gRaRsOps.sendWrList(baseInfo, wrList, sendWrlist->txData.sendNum, wrRsp, &completeNum);
    *opResult = ret;
    if (ret) {
        if (ret == -ENOENT) {
            hccp_warn_rma("not found remote mr_info, need try again");
        } else {
            hccp_err("send wr failed ret[%d].", ret);
        }
    }
    sendWrlist = (union OpSendWrlistDataV2 *)(outBuf + sizeof(struct MsgHead));
    sendWrlist->rxData.completeNum = completeNum;
    ret = memcpy_s(sendWrlist->rxData.wrRsp, sizeof(struct SendWrRsp) * MAX_WR_NUM, wrRsp,
        completeNum * sizeof(struct SendWrRsp));
    if (ret) {
        hccp_err("ra_rs_send_wr_list memcpy_s failed, ret[%d]. ", ret);
        ret = -ESAFEFUNC;
        goto copy_wr_rsp_fail;
    }

copy_wr_rsp_fail:
    free(wrList);
    wrList = NULL;

alloc_wr_list_fail:
    free(wrRsp);
    wrRsp = NULL;
    return 0;
}

STATIC void GetWrList(struct WrInfo *wrList, union OpSendWrlistDataExt *sendWrlist)
{
    uint32_t i;
    for (i = 0; i < sendWrlist->txData.sendNum; i++) {
        wrList[i].op = sendWrlist->txData.wrlist[i].op;
        wrList[i].sendFlags = sendWrlist->txData.wrlist[i].sendFlags;
        wrList[i].immData = sendWrlist->txData.wrlist[i].ext.immData;
        wrList[i].dstAddr = sendWrlist->txData.wrlist[i].dstAddr;
        wrList[i].memList.addr = sendWrlist->txData.wrlist[i].memList.addr;
        wrList[i].memList.len = sendWrlist->txData.wrlist[i].memList.len;
        wrList[i].memList.lkey = sendWrlist->txData.wrlist[i].memList.lkey;
        wrList[i].aux.dataType = sendWrlist->txData.wrlist[i].aux.dataType;
        wrList[i].aux.reduceType = sendWrlist->txData.wrlist[i].aux.reduceType;
        wrList[i].aux.notifyOffset = sendWrlist->txData.wrlist[i].aux.notifyOffset;
    }
    return;
}

STATIC int RaRsSendWrListExt(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    int ret;
    unsigned int completeNum = 0;
    struct SendWrRsp *wrRsp = NULL;
    struct WrInfo *wrList = NULL;
    struct RsWrlistBaseInfo baseInfo = {0};
    union OpSendWrlistDataExt *sendWrlist = (union OpSendWrlistDataExt *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSendWrlistDataExt), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(sendWrlist->txData.sendNum, 0, MAX_WR_NUM, opResult);

    wrRsp = calloc(sendWrlist->txData.sendNum, sizeof(struct SendWrRsp));
    CHK_PRT_RETURN(wrRsp == NULL, hccp_err("wr_rsp calloc failed."), -ENOMEM);

    wrList = calloc(sendWrlist->txData.sendNum, sizeof(struct WrInfo));
    if (wrList == NULL) {
        hccp_err("wr_list calloc failed.");
        ret = -ENOMEM;
        goto alloc_wr_list_fail;
    }

    baseInfo.phyId = sendWrlist->txData.phyId;
    baseInfo.rdevIndex = sendWrlist->txData.rdevIndex;
    baseInfo.qpn = sendWrlist->txData.qpn;
    baseInfo.keyFlag = 0;
    GetWrList(wrList, sendWrlist);
    ret = gRaRsOps.sendWrList(baseInfo, wrList, sendWrlist->txData.sendNum, wrRsp, &completeNum);
    *opResult = ret;
    if (ret) {
        if (ret == -ENOENT) {
            hccp_warn_rma("not found remote mr_info, need try again");
        } else {
            hccp_err("send wr failed ret[%d].", ret);
        }
    }
    sendWrlist = (union OpSendWrlistDataExt *)(outBuf + sizeof(struct MsgHead));
    sendWrlist->rxData.completeNum = completeNum;
    ret = memcpy_s(sendWrlist->rxData.wrRsp, sizeof(struct SendWrRsp) * MAX_WR_NUM_V1, wrRsp,
        completeNum * sizeof(struct SendWrRsp));
    if (ret) {
        hccp_err("ra_rs_send_wr_list_ext memcpy_s failed, ret[%d]. ", ret);
        ret = -ESAFEFUNC;
        goto copy_wr_rsp_fail;
    }

copy_wr_rsp_fail:
    free(wrList);
    wrList = NULL;

alloc_wr_list_fail:
    free(wrRsp);
    wrRsp = NULL;
    return 0;
}

STATIC void GetWrListExtV2(struct WrInfo *wrList, union OpSendWrlistDataExtV2 *sendWrlist)
{
    uint32_t i;
    for (i = 0; i < sendWrlist->txData.sendNum; i++) {
        wrList[i].op = sendWrlist->txData.wrlist[i].op;
        wrList[i].sendFlags = sendWrlist->txData.wrlist[i].sendFlags;
        wrList[i].immData = sendWrlist->txData.wrlist[i].ext.immData;
        wrList[i].dstAddr = sendWrlist->txData.wrlist[i].dstAddr;
        wrList[i].memList.addr = sendWrlist->txData.wrlist[i].memList.addr;
        wrList[i].memList.len = sendWrlist->txData.wrlist[i].memList.len;
        wrList[i].memList.lkey = sendWrlist->txData.wrlist[i].memList.lkey;
        wrList[i].aux.dataType = sendWrlist->txData.wrlist[i].aux.dataType;
        wrList[i].aux.reduceType = sendWrlist->txData.wrlist[i].aux.reduceType;
        wrList[i].aux.notifyOffset = sendWrlist->txData.wrlist[i].aux.notifyOffset;
    }
    return;
}

STATIC int RaRsSendWrListExtV2(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    int ret;
    unsigned int completeNum = 0;
    struct WrInfo *wrList = NULL;
    struct SendWrRsp *wrRsp = NULL;
    struct RsWrlistBaseInfo baseInfo = {0};
    union OpSendWrlistDataExtV2 *sendWrlist = (union OpSendWrlistDataExtV2 *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSendWrlistDataExtV2), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(sendWrlist->txData.sendNum, 0, MAX_WR_NUM, opResult);

    wrRsp = calloc(sendWrlist->txData.sendNum, sizeof(struct SendWrRsp));
    CHK_PRT_RETURN(wrRsp == NULL, hccp_err("wr_rsp calloc failed."), -ENOMEM);

    wrList = calloc(sendWrlist->txData.sendNum, sizeof(struct WrInfo));
    if (wrList == NULL) {
        hccp_err("wr_list calloc failed.");
        ret = -ENOMEM;
        goto alloc_wr_list_fail;
    }

    baseInfo.phyId = sendWrlist->txData.phyId;
    baseInfo.rdevIndex = sendWrlist->txData.rdevIndex;
    baseInfo.qpn = sendWrlist->txData.qpn;
    baseInfo.keyFlag = 0;
    GetWrListExtV2(wrList, sendWrlist);
    ret = gRaRsOps.sendWrList(baseInfo, wrList, sendWrlist->txData.sendNum, wrRsp, &completeNum);
    *opResult = ret;
    if (ret) {
        if (ret == -ENOENT) {
            hccp_warn_rma("not found remote mr_info, need try again");
        } else {
            hccp_err("send wr failed ret[%d].", ret);
        }
    }
    sendWrlist = (union OpSendWrlistDataExtV2 *)(outBuf + sizeof(struct MsgHead));
    sendWrlist->rxData.completeNum = completeNum;
    ret = memcpy_s(sendWrlist->rxData.wrRsp, sizeof(struct SendWrRsp) * MAX_WR_NUM, wrRsp,
        completeNum * sizeof(struct SendWrRsp));
    if (ret) {
        hccp_err("ra_rs_send_wr_list_ext_v2 memcpy_s failed, ret[%d]. ", ret);
        ret = -ESAFEFUNC;
        goto copy_wr_rsp_fail;
    }

copy_wr_rsp_fail:
    free(wrList);
    wrList = NULL;

alloc_wr_list_fail:
    free(wrRsp);
    wrRsp = NULL;
    return 0;
}

STATIC int RaRsSendNormalWrlist(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpSendNormalWrlistData *sendWrlistOut = (union OpSendNormalWrlistData *)(outBuf + sizeof(struct MsgHead));
    union OpSendNormalWrlistData *sendWrlist = (union OpSendNormalWrlistData *)(inBuf + sizeof(struct MsgHead));
    struct RsWrlistBaseInfo baseInfo = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSendNormalWrlistData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(sendWrlist->txData.sendNum, 0, MAX_WR_NUM, opResult);

    baseInfo.phyId = sendWrlist->txData.phyId;
    baseInfo.rdevIndex = sendWrlist->txData.rdevIndex;
    baseInfo.qpn = sendWrlist->txData.qpn;
    baseInfo.keyFlag = 1;

    *opResult = gRaRsOps.sendWrList(baseInfo, sendWrlist->txData.wrlist, sendWrlist->txData.sendNum,
        sendWrlistOut->rxData.wrRsp, &sendWrlistOut->rxData.completeNum);
    if (*opResult != 0) {
        hccp_err("send_wr_list failed ret[%d].", *opResult);
    }
    return 0;
}

STATIC int RaRsGetNotifyBa(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpGetNotifyBaData *getNotifyBaData = (union OpGetNotifyBaData *)(inBuf + sizeof(struct MsgHead));
    struct MrInfoT info = {0};

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpGetNotifyBaData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.getNotifyMrInfo(getNotifyBaData->txData.phyId, getNotifyBaData->txData.rdevIndex, &info);
    if (*opResult != 0) {
        hccp_err("reg_notify_mr failed ret[%d].", *opResult);
        return 0;
    }

    getNotifyBaData = (union OpGetNotifyBaData *)(outBuf + sizeof(struct MsgHead));
    getNotifyBaData->rxData.va = (unsigned long long)(uintptr_t)info.addr;
    getNotifyBaData->rxData.size = info.size;
    getNotifyBaData->rxData.access = info.access;
    getNotifyBaData->rxData.lkey = info.lkey;

    return 0;
}

STATIC int RaRsNotifyCfgSet(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpNotifyCfgSetData *setNotifyBaData = (union OpNotifyCfgSetData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpNotifyCfgSetData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.notifyCfgSet(setNotifyBaData->txData.phyId, setNotifyBaData->txData.va,
        setNotifyBaData->txData.size);
    if (*opResult != 0) {
        hccp_err("notify_cfg_set failed ret[%d].", *opResult);
    }

    return 0;
}

STATIC int RaRsNotifyCfgGet(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    unsigned long long va = 0;
    unsigned long long size = 0;
    union OpNotifyCfgGetData *getNotifyBaData = (union OpNotifyCfgGetData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpNotifyCfgGetData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.notifyCfgGet(getNotifyBaData->txData.phyId, &va, &size);
    if (*opResult != 0) {
        hccp_err("notify_cfg_set failed ret[%d].", *opResult);
        return 0;
    }

    getNotifyBaData = (union OpNotifyCfgGetData *)(outBuf + sizeof(struct MsgHead));
    getNotifyBaData->rxData.va = va;
    getNotifyBaData->rxData.size = size;
    return 0;
}

STATIC int RaSetPid(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpSetPidData *setPidData = (union OpSetPidData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSetPidData), sizeof(struct MsgHead), rcvBufLen, opResult);

    hccp_info_others("ra get pid is [%d]", setPidData->txData.pid);

    *opResult = gRaRsOps.setHostPid(setPidData->txData.phyId, setPidData->txData.pid, setPidData->txData.pidSign);

    hccp_info_others("ra_set_pid finish");
    return 0;
}

STATIC int RaRsCloseHdcSession(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)inBuf;
    (void)outBuf;
    (void)outLen;
    (void)rcvBufLen;
    *opResult = 0;
    hccp_info_socket("ra_rs_close_hdc_session finish");
    return 0;
}

STATIC int RaRsGetInterfaceVersion(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    unsigned int version = 0;
    union OpGetVersionData *versionInfoRet = NULL;
    union OpGetVersionData *versionInfo = (union OpGetVersionData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpGetVersionData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.getInterfaceVersion(versionInfo->txData.opcode, &version);
    if (*opResult != 0) {
        hccp_err("get_interface_version failed, opcode %d, ret %d", versionInfo->txData.opcode, *opResult);
        return 0;
    }

    versionInfoRet = (union OpGetVersionData *)(outBuf + sizeof(struct MsgHead));
    versionInfoRet->rxData.version = version;
    return 0;
}

STATIC int RaRsSetQpAttrQos(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpSetQpAttrQosData *attrQosData = (union OpSetQpAttrQosData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSetQpAttrQosData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.setQpAttrQos(attrQosData->txData.phyId, attrQosData->txData.rdevIndex, attrQosData->txData.qpn,
        &(attrQosData->txData.qosAttr));
    if (*opResult != 0) {
        hccp_err("set_qp_attr_qos failed ret[%d].", *opResult);
    }

    return 0;
}

STATIC int RaRsSetQpAttrTimeout(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpSetQpAttrTimeoutData *attrTimeData = (union OpSetQpAttrTimeoutData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSetQpAttrTimeoutData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.setQpAttrTimeout(attrTimeData->txData.phyId, attrTimeData->txData.rdevIndex,
        attrTimeData->txData.qpn, &(attrTimeData->txData.timeout));
    if (*opResult != 0) {
        hccp_err("set_qp_attr_timeout failed ret[%d].", *opResult);
    }

    return 0;
}

STATIC int RaRsSetQpAttrRetryCnt(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpSetQpAttrRetryCntData *attrRetryCntData = (union OpSetQpAttrRetryCntData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSetQpAttrRetryCntData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.setQpAttrRetryCnt(attrRetryCntData->txData.phyId, attrRetryCntData->txData.rdevIndex,
        attrRetryCntData->txData.qpn, &(attrRetryCntData->txData.retryCnt));
    if (*opResult != 0) {
        hccp_err("set_qp_attr_retry_cnt failed ret[%d].", *opResult);
    }

    return 0;
}

STATIC int RaRsGetCqeErrInfo(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)inBuf;
    (void)outLen;
    int ret;
    struct CqeErrInfo info = {0};
    union OpGetCqeErrInfoData *cqeErrInfoRet = NULL;

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpGetCqeErrInfoData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.getCqeErrInfo(&info);
    if (*opResult != 0) {
        hccp_err("get_cqe_err_info failed, ret %d", *opResult);
        return 0;
    }

    cqeErrInfoRet = (union OpGetCqeErrInfoData *)(outBuf + sizeof(struct MsgHead));
    ret = memcpy_s(&cqeErrInfoRet->rxData.info, sizeof(struct CqeErrInfo), &info, sizeof(struct CqeErrInfo));
    CHK_PRT_RETURN(ret, hccp_err("ra_rs_get_cqe_err_info memcpy_s failed, ret[%d]. ", ret), -ESAFEFUNC);
    return 0;
}

STATIC int RaRsGetLiteSupport(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpLiteSupportData *liteSupportData = (union OpLiteSupportData *)(inBuf + sizeof(struct MsgHead));
    union OpLiteSupportData *liteSupportOut = (union OpLiteSupportData *)(outBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpLiteSupportData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.getLiteSupport(liteSupportData->txData.phyId, liteSupportData->txData.rdevIndex,
        &liteSupportOut->rxData.supportLite);
    if (*opResult != 0) {
        hccp_err("get_lite_support failed ret[%d].", *opResult);
    }

    return 0;
}

STATIC int RaRsGetLiteRdevCap(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpLiteRdevCapData *liteRdevCapData = (union OpLiteRdevCapData *)(inBuf + sizeof(struct MsgHead));
    union OpLiteRdevCapData *liteRdevCapOut = (union OpLiteRdevCapData *)(outBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpLiteRdevCapData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.getLiteRdevCap(liteRdevCapData->txData.phyId, liteRdevCapData->txData.rdevIndex,
        (void *)&liteRdevCapOut->rxData.resp);
    if (*opResult != 0) {
        hccp_err("get_lite_rdev_cap failed ret[%d].", *opResult);
    }

    return 0;
}

STATIC int RaRsGetLiteQpCqAttr(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpLiteQpCqAttrData *liteQpCqAttrData = (union OpLiteQpCqAttrData *)(inBuf + sizeof(struct MsgHead));
    union OpLiteQpCqAttrData *liteQpCqAttrOut = (union OpLiteQpCqAttrData *)(outBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpLiteQpCqAttrData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.getLiteQpCqAttr(liteQpCqAttrData->txData.phyId, liteQpCqAttrData->txData.rdevIndex,
        liteQpCqAttrData->txData.qpn, (void *)&liteQpCqAttrOut->rxData.resp);
    if (*opResult != 0) {
        hccp_err("get_lite_qp_cq_attr failed ret[%d].", *opResult);
    }

    return 0;
}

STATIC int RaRsGetLiteConnectedInfo(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpLiteConnectedInfoData *liteConnectedInfoData = (union OpLiteConnectedInfoData *)(inBuf +
                                                                                             sizeof(struct MsgHead));
    union OpLiteConnectedInfoData *liteConnectedInfoOut = (union OpLiteConnectedInfoData *)(outBuf +
                                                                                            sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpLiteConnectedInfoData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.getLiteConnectedInfo(liteConnectedInfoData->txData.phyId,
        liteConnectedInfoData->txData.rdevIndex, liteConnectedInfoData->txData.qpn,
        (void *)&liteConnectedInfoOut->rxData.resp);
    if (*opResult != 0) {
        hccp_err("get_lite_connected_info failed ret[%d].", *opResult);
    }

    return 0;
}

STATIC int RaRsGetLiteMemAttr(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpLiteMemAttrData *liteMemAttrData = (union OpLiteMemAttrData *)(inBuf + sizeof(struct MsgHead));
    union OpLiteMemAttrData *liteMemAttrOut = (union OpLiteMemAttrData *)(outBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpLiteMemAttrData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.getLiteMemAttr(liteMemAttrData->txData.phyId, liteMemAttrData->txData.rdevIndex,
        liteMemAttrData->txData.qpn, (void *)&liteMemAttrOut->rxData.resp);
    if (*opResult != 0) {
        hccp_err("get_lite_mem_attr failed ret[%d].", *opResult);
    }

    return 0;
}

STATIC int RaRsGetCqeErrInfoNum(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpGetCqeErrInfoNumData *cqeErrInfoNum = (union OpGetCqeErrInfoNumData *)(inBuf + sizeof(struct MsgHead));
    unsigned int num;

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpGetCqeErrInfoNumData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.getCqeErrInfoNum(cqeErrInfoNum->txData.phyId, cqeErrInfoNum->txData.rdevIndex, &num);
    if (*opResult != 0) {
        hccp_err("get_cqe_err_info_num failed, ret %d", *opResult);
        return 0;
    }

    cqeErrInfoNum = (union OpGetCqeErrInfoNumData *)(outBuf + sizeof(struct MsgHead));
    cqeErrInfoNum->rxData.num = num;

    return 0;
}

STATIC int RaRsGetCqeErrInfoList(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpGetCqeErrInfoListData *cqeErrInfoList = (union OpGetCqeErrInfoListData *)(inBuf + sizeof(struct MsgHead));
    union OpGetCqeErrInfoListData *cqeErrInfoListRet = (union OpGetCqeErrInfoListData *)(outBuf +
                                                                                         sizeof(struct MsgHead));
    unsigned int num;

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpGetCqeErrInfoListData), sizeof(struct MsgHead), rcvBufLen, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(cqeErrInfoList->txData.num, 0, CQE_ERR_INFO_MAX_NUM, opResult);

    num = cqeErrInfoList->txData.num;
    *opResult = gRaRsOps.getCqeErrInfoList(cqeErrInfoList->txData.phyId, cqeErrInfoList->txData.rdevIndex,
        cqeErrInfoListRet->rxData.infoList, &num);
    if (*opResult != 0) {
        hccp_err("get_cqe_err_info_list failed, ret %d", *opResult);
        return 0;
    }

    cqeErrInfoListRet->rxData.num = num;

    return 0;
}

STATIC int RaRsGetTlsEnable(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpGetTlsEnableData *opDataRet = (union OpGetTlsEnableData *)(outBuf + sizeof(struct MsgHead));
    union OpGetTlsEnableData *opData = (union OpGetTlsEnableData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpGetTlsEnableData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.getTlsEnable(opData->txData.phyId, &opDataRet->rxData.tlsEnable);
    if (*opResult != 0) {
        hccp_err("get_tls_enable failed, ret %d", *opResult);
    }
    return 0;
}

STATIC int RaRsGetSecRandom(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)inBuf;
    (void)outLen;
    union OpGetSecRandomData *opDataRet = (union OpGetSecRandomData *)(outBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpGetSecRandomData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsOps.getSecRandom((int *)&opDataRet->rxData.value);
    if (*opResult != 0) {
        hccp_err("get sec random failed, ret %d", *opResult);
    }
    return 0;
}

STATIC int RaRsGetHccnCfg(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpGetHccnCfgData *opDataRet = (union OpGetHccnCfgData *)(outBuf + sizeof(struct MsgHead));
    union OpGetHccnCfgData *opData = (union OpGetHccnCfgData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpGetHccnCfgData), sizeof(struct MsgHead), rcvBufLen, opResult);

    struct RaInfo raInfo = {
        .mode = NETWORK_OFFLINE,
        .phyId = opData->txData.phyId,
    };
    opDataRet->rxData.valueLen = HCCN_CFG_MSG_DATA_LEN;
    *opResult = gRaRsOps.getHccnCfg(&raInfo, opData->txData.key, opDataRet->rxData.value, &opDataRet->rxData.valueLen);
    if (*opResult != 0) {
        hccp_err("get hccn cfg failed, ret %d", *opResult);
    }
    return 0;
}

STATIC int RaRsSetDebugConfig(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    union OpSetDebugConfigData *opData = (union OpSetDebugConfigData *)(inBuf + sizeof(struct MsgHead));

    (void)outBuf;
    (void)outLen;
    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpSetDebugConfigData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RsSetDebugConfig(opData->txData.debugConfig);
    HccpSetDebugConfig(opData->txData.debugConfig);
    *opResult = 0;
    return 0;
}

#define US_PRE_SECOND 1000000
#define US_PRE_MSECOND 1000
#define MS_PRE_SECOND 1000

STATIC void RaTimeInterval(struct timeval *endTime, struct timeval *startTime, long *msec)
{
    /* if low position is sufficient, then borrow one from the high position */
    if (endTime->tv_usec < startTime->tv_usec) {
        endTime->tv_sec -= 1;
        endTime->tv_usec += US_PRE_SECOND;
    }

    *msec = (endTime->tv_sec - startTime->tv_sec) * MS_PRE_SECOND +
            (endTime->tv_usec - startTime->tv_usec) / US_PRE_MSECOND;
}

#define OP_TYPE_CFG 0
#define OP_TYPE_QUERY 1

STATIC void RaGetOpRight(struct RaHdcOpSec *opSec, unsigned int opcode, unsigned int asyncReqId, int *right)
{
    long timeInterval = 0;
    struct timeval tCur;
    int exeRight;
    int ret;

    ret = gettimeofday(&tCur, NULL);
    if (ret) {
        *right = OP_RIGHT_QUERY_ERR;
        hccp_err("ra gettimeofday failed ret[%d].", ret);
        return;
    }

    RaTimeInterval(&tCur, &opSec->tLast, &timeInterval);
    opSec->tLast = (timeInterval < 0) ? tCur : opSec->tLast;
    timeInterval = (timeInterval < 0) ? 0 : timeInterval;
    opSec->tLast = (timeInterval == 0) ? opSec->tLast : tCur;
    opSec->tokenNum += timeInterval * TOKEN_RATE;

    if (opSec->tokenNum == 0) {
        *right = HAVE_NOT_OP_RIGHT;
        hccp_err("ra handle have not op right. opcode[%u].", opcode);
        return;
    }

    exeRight = HAVE_OP_RIGHT;

    opSec->tokenNum = opSec->tokenNum - 1;
    opSec->tokenNum = (opSec->tokenNum > BUCKET_DEPTH) ? BUCKET_DEPTH : opSec->tokenNum;
    if (RaIsOpcodeLogSuppressed(opSec->lastOpcode) && opSec->lastOpcode != opcode) {
        hccp_dbg("lastOpcode[%u], exeRight[%d], tokenNum[%llu], cfgOpNum[%u], lastOpcodeCnt[%u]", opSec->lastOpcode,
            exeRight, opSec->tokenNum, opSec->cfgOpNum, opSec->lastOpcodeCnt);
    } else if (RaIsOpcodeLogSuppressed(opcode)) {
        goto out;
    }

    if (opSec->isAsyncOp) {
        hccp_dbg("opcode[%u], reqId[%u], exeRight[%d], tokenNum[%llu], cfgOpNum[%u]", opcode, asyncReqId, exeRight,
            opSec->tokenNum, opSec->cfgOpNum);
    } else {
        hccp_dbg("opcode[%u], exeRight[%d], tokenNum[%llu], cfgOpNum[%u]", opcode, exeRight, opSec->tokenNum,
            opSec->cfgOpNum);
    }

out:
    opSec->lastOpcodeCnt = (opcode == opSec->lastOpcode) ? (opSec->lastOpcodeCnt + 1)
                                                         : (RaIsOpcodeLogSuppressed(opcode) ? 1 : 0);
    opSec->lastOpcode = opcode;
    *right = exeRight;
    return;
}

struct RaOpHandle gRaOpHandle[] = {
    {RA_RS_SOCKET_CONN, RaRsSocketBatchConnect, sizeof(union OpSocketConnectData)},
    {RA_RS_SOCKET_CLOSE, RaRsSocketBatchClose, sizeof(union OpSocketCloseData)},
    {RA_RS_SOCKET_ABORT, RaRsSocketBatchAbort, sizeof(union OpSocketConnectData)},
    {RA_RS_SOCKET_LISTEN_START, RaRsSocketListenStart, sizeof(union OpSocketListenData)},
    {RA_RS_SOCKET_LISTEN_STOP, RaRsSocketListenStop, sizeof(union OpSocketListenData)},
    {RA_RS_GET_SOCKET, RaRsGetSockets, sizeof(union OpSocketInfoData)},
    {RA_RS_SOCKET_SEND, RaRsSocketSend, sizeof(union OpSocketSendData)},
    {RA_RS_SOCKET_RECV, RaRsSocketRecv, sizeof(union OpSocketRecvData)},
    {RA_RS_QP_CREATE, RaRsQpCreate, sizeof(union OpQpCreateData)},
    {RA_RS_QP_CREATE_WITH_ATTRS, RaRsQpCreateWithAttrs, sizeof(union OpQpCreateWithAttrsData)},
    {RA_RS_QP_CREATE_WITH_CQ_WITH_ATTRS, RaRsQpCreateWithCQWithAttrs, sizeof(union OpQpCreateWithCQWithAttrsData)},
    {RA_RS_AI_QP_CREATE, RaRsAiQpCreate, sizeof(union OpAiQpCreateData)},
    {RA_RS_AI_QP_CREATE_WITH_ATTRS, RaRsAiQpCreateWithData, sizeof(union OpAiQpCreateWithAttrsData)},
    {RA_RS_TYPICAL_QP_CREATE, RaRsTypicalQpCreate, sizeof(union OpTypicalQpCreateData)},
    {RA_RS_TYPICAL_CQ_CREATE, RaRsTypicalCqCreate, sizeof(union OpTypicalCqCreateData)},
    {RA_RS_TYPICAL_CQ_DESTROY, RaRsTypicalCqDestroy, sizeof(union OpTypicalCqDestroyData)},
    {RA_RS_QP_DESTROY, RaRsQpDestroy, sizeof(union OpQpDestroyData)},
    {RA_RS_QP_DESTROY_WITHOUT_CQ, RaRsQpDestroyWithoutCQ, sizeof(union OpQpDestroyData)},
    {RA_RS_TYPICAL_QP_MODIFY, RaRsTypicalQpModify, sizeof(union OpTypicalQpModifyData)},
    {RA_RS_QP_BATCH_MODIFY, RaRsQpBatchModify, sizeof(union OpQpBatchModifyData)},
    {RA_RS_QP_CONNECT, RaRsQpConnectAsync, sizeof(union OpQpConnectData)},
    {RA_RS_QP_STATUS, RaRsGetQpStatus, sizeof(union OpQpStatusData)},
    {RA_RS_QP_INFO, RaRsGetQpInfo, sizeof(union OpQpInfoData)},
    {RA_RS_MR_REG, RaRsMrReg, sizeof(union OpMrRegData)},
    {RA_RS_MR_DEREG, RaRsMrDereg, sizeof(union OpMrDeregData)},
    {RA_RS_TYPICAL_MR_REG_V1, RaRsTypicalMrRegV1, sizeof(union OpTypicalMrRegData)},
    {RA_RS_TYPICAL_MR_REG, RaRsTypicalMrReg, sizeof(union OpTypicalMrRegData)},
    {RA_RS_REMAP_MR, RaRsRemapMr, sizeof(union OpRemapMrData)},
    {RA_RS_TYPICAL_MR_DEREG, RaRsTypicalMrDereg, sizeof(union OpTypicalMrDeregData)},
    {RA_RS_SEND_WR, RaRsSendWr, sizeof(union OpSendWrData)},
    {RA_RS_GET_NOTIFY_BA, RaRsGetNotifyBa, sizeof(union OpGetNotifyBaData)},
    {RA_RS_SOCKET_INIT, RaRsSocketInit, sizeof(union OpSocketInitData)},
    {RA_RS_SOCKET_DEINIT, RaRsSocketDeinit, sizeof(union OpSocketDeinitData)},
    {RA_RS_RDEV_INIT, RaRsRdevInit, sizeof(union OpRdevInitData)},
    {RA_RS_RDEV_INIT_WITH_BACKUP, RaRsRdevInitWithBackup, sizeof(union OpRdevInitWithBackupData)},
    {RA_RS_RDEV_GET_PORT_STATUS, RaRsRdevGetPortStatus, sizeof(union OpRdevGetPortStatusData)},
    {RA_RS_RDEV_DEINIT, RaRsRdevDeinit, sizeof(union OpRdevDeinitData)},
    {RA_RS_WLIST_ADD, RaRsSocketWhiteListAdd, sizeof(union OpWlistData)},
    {RA_RS_WLIST_ADD_V2, RaRsSocketWhiteListAddV2, sizeof(union OpWlistDataV2)},
    {RA_RS_WLIST_DEL, RaRsSocketWhiteListDel, sizeof(union OpWlistData)},
    {RA_RS_WLIST_DEL_V2, RaRsSocketWhiteListDelV2, sizeof(union OpWlistDataV2)},
    {RA_RS_ACCEPT_CREDIT_ADD, RaRsSocketCreditAdd, sizeof(union OpAcceptCreditData)},
    {RA_RS_GET_IFNUM, RaRsGetIfnum, sizeof(union OpIfnumData)},
    {RA_RS_GET_IFADDRS, RaRsGetIfaddrs, sizeof(union OpIfaddrData)},
    {RA_RS_GET_IFADDRS_V2, RaRsGetIfaddrsV2, sizeof(union OpIfaddrDataV2)},
    {RA_RS_GET_INTERFACE_VERSION, RaRsGetInterfaceVersion, sizeof(union OpGetVersionData)},
    {RA_RS_SEND_WRLIST, RaRsSendWrList, sizeof(union OpSendWrlistData)},
    {RA_RS_SEND_WRLIST_V2, RaRsSendWrListV2, sizeof(union OpSendWrlistDataV2)},
    {RA_RS_SEND_WRLIST_EXT, RaRsSendWrListExt, sizeof(union OpSendWrlistDataExt)},
    {RA_RS_SEND_WRLIST_EXT_V2, RaRsSendWrListExtV2, sizeof(union OpSendWrlistDataExtV2)},
    {RA_RS_SEND_NORMAL_WRLIST, RaRsSendNormalWrlist, sizeof(union OpSendNormalWrlistData)},
    {RA_RS_SET_TSQP_DEPTH, RaRsSetTsqpDepth, sizeof(union OpSetTsqpDepthData)},
    {RA_RS_GET_TSQP_DEPTH, RaRsGetTsqpDepth, sizeof(union OpGetTsqpDepthData)},
    {RA_RS_HDC_SESSION_CLOSE, RaRsCloseHdcSession, sizeof(union OpSocketRecvData)},
    {RA_RS_GET_VNIC_IP, RaRsGetVnicIp, sizeof(union OpGetVnicIpData)},
    {RA_RS_GET_VNIC_IP_INFOS_V1, RaRsGetVnicIpInfosV1, sizeof(union OpGetVnicIpInfosDataV1)},
    {RA_RS_GET_VNIC_IP_INFOS, RaRsGetVnicIpInfos, sizeof(union OpGetVnicIpInfosData)},
    {RA_RS_NOTIFY_CFG_SET, RaRsNotifyCfgSet, sizeof(union OpNotifyCfgSetData)},
    {RA_RS_NOTIFY_CFG_GET, RaRsNotifyCfgGet, sizeof(union OpNotifyCfgGetData)},
    {RA_RS_SET_PID, RaSetPid, sizeof(union OpSetPidData)},
    {RA_RS_SET_QP_ATTR_QOS, RaRsSetQpAttrQos, sizeof(union OpSetQpAttrQosData)},
    {RA_RS_SET_QP_ATTR_TIMEOUT, RaRsSetQpAttrTimeout, sizeof(union OpSetQpAttrTimeoutData)},
    {RA_RS_SET_QP_ATTR_RETRY_CNT, RaRsSetQpAttrRetryCnt, sizeof(union OpSetQpAttrRetryCntData)},
    {RA_RS_GET_CQE_ERR_INFO, RaRsGetCqeErrInfo, sizeof(union OpGetCqeErrInfoData)},
    {RA_RS_GET_CQE_ERR_INFO_NUM, RaRsGetCqeErrInfoNum, sizeof(union OpGetCqeErrInfoNumData)},
    {RA_RS_GET_CQE_ERR_INFO_LIST, RaRsGetCqeErrInfoList, sizeof(union OpGetCqeErrInfoListData)},
    {RA_RS_GET_LITE_SUPPORT, RaRsGetLiteSupport, sizeof(union OpLiteSupportData)},
    {RA_RS_GET_LITE_RDEV_CAP, RaRsGetLiteRdevCap, sizeof(union OpLiteRdevCapData)},
    {RA_RS_GET_LITE_QP_CQ_ATTR, RaRsGetLiteQpCqAttr, sizeof(union OpLiteQpCqAttrData)},
    {RA_RS_GET_LITE_QP_ATTR, RaRsGetLiteQpAttr, sizeof(union OpLiteQpAttrData)},
    {RA_RS_GET_LITE_CQ_ATTR, RaRsGetLiteCqAttr, sizeof(union OpGetLiteCqAttrData)},
    {RA_RS_GET_LITE_CONNECTED_INFO, RaRsGetLiteConnectedInfo, sizeof(union OpLiteConnectedInfoData)},
    {RA_RS_GET_LITE_MEM_ATTR, RaRsGetLiteMemAttr, sizeof(union OpLiteMemAttrData)},
    {RA_RS_PING_INIT, RaRsPingInit, sizeof(union OpPingInitData)},
    {RA_RS_PING_ADD, RaRsPingTargetAdd, sizeof(union OpPingAddData)},
    {RA_RS_PING_START, RaRsPingTaskStart, sizeof(union OpPingStartData)},
    {RA_RS_PING_GET_RESULTS, RaRsPingGetResults, sizeof(union OpPingResultsData)},
    {RA_RS_PING_STOP, RaRsPingTaskStop, sizeof(union OpPingStopData)},
    {RA_RS_PING_DEL, RaRsPingTargetDel, sizeof(union OpPingDelData)},
    {RA_RS_PING_DEINIT, RaRsPingDeinit, sizeof(union OpPingDeinitData)},
    {RA_RS_TLV_INIT_V1, RaRsTlvInitV1, sizeof(union OpTlvInitDataV1)},
    {RA_RS_TLV_INIT, RaRsTlvInit, sizeof(union OpTlvInitData)},
    {RA_RS_TLV_DEINIT, RaRsTlvDeinit, sizeof(union OpTlvDeinitData)},
    {RA_RS_TLV_REQUEST, RaRsTlvRequest, sizeof(union OpTlvRequestData)},
    {RA_RS_TLV_REQUEST_V2, RaRsTlvRequestV2, sizeof(union OpTlvRequestDataV2)},
    {RA_RS_CUSTOM_CHANNEL_DEPRECATED, RaRsCustomChannel, sizeof(union OpCustomChannelData)},
    {RA_RS_GET_TLS_ENABLE, RaRsGetTlsEnable, sizeof(union OpGetTlsEnableData)},
    {RA_RS_GET_SEC_RANDOM, RaRsGetSecRandom, sizeof(union OpGetSecRandomData)},
    {RA_RS_GET_HCCN_CFG, RaRsGetHccnCfg, sizeof(union OpGetHccnCfgData)},
    {RA_RS_SET_DEBUG_CONFIG, RaRsSetDebugConfig, sizeof(union OpSetDebugConfigData)},
    {RA_RS_ASYNC_HDC_SESSION_CONNECT, RaRsAsyncHdcSessionConnect, sizeof(union OpAsyncHdcConnectData)},
    {RA_RS_ASYNC_HDC_SESSION_CLOSE, RaRsAsyncHdcSessionClose, sizeof(union OpAsyncHdcCloseData)},
    {RA_RS_GET_DEV_EID_INFO_NUM, RaRsGetDevEidInfoNum, sizeof(union OpGetDevEidInfoNumData)},
    {RA_RS_GET_DEV_EID_INFO_LIST, RaRsGetDevEidInfoList, sizeof(union OpGetDevEidInfoListData)},
    {RA_RS_CTX_INIT, RaRsCtxInit, sizeof(union OpCtxInitData)},
    {RA_RS_CTX_GET_ASYNC_EVENTS, RaRsCtxGetAsyncEvents, sizeof(union OpCtxGetAsyncEventsData)},
    {RA_RS_CTX_DEINIT, RaRsCtxDeinit, sizeof(union OpCtxDeinitData)},
    {RA_RS_GET_EID_BY_IP, RaRsGetEidByIp, sizeof(union OpGetEidByIpData)},
    {RA_RS_GET_IP_BY_EID, RaRsGetIpByEid, sizeof(union OpGetIpByEidData)},
    {RA_RS_GET_TP_INFO_LIST, RaRsGetTpInfoList, sizeof(union OpGetTpInfoListData)},
    {RA_RS_GET_TP_ATTR, RaRsGetTpAttr, sizeof(union OpGetTpAttrData)},
    {RA_RS_SET_TP_ATTR, RaRsSetTpAttr, sizeof(union OpSetTpAttrData)},
    {RA_RS_CTX_TOKEN_ID_ALLOC, RaRsCtxTokenIdAlloc, sizeof(union OpTokenIdAllocData)},
    {RA_RS_CTX_TOKEN_ID_FREE, RaRsCtxTokenIdFree, sizeof(union OpTokenIdFreeData)},
    {RA_RS_LMEM_REG, RaRsLmemReg, sizeof(union OpLmemRegInfoData)},
    {RA_RS_LMEM_UNREG, RaRsLmemUnreg, sizeof(union OpLmemUnregInfoData)},
    {RA_RS_RMEM_IMPORT, RaRsRmemImport, sizeof(union OpRmemImportInfoData)},
    {RA_RS_RMEM_UNIMPORT, RaRsRmemUnimport, sizeof(union OpRmemUnimportInfoData)},
    {RA_RS_CTX_CHAN_CREATE, RaRsCtxChanCreate, sizeof(union OpCtxChanCreateData)},
    {RA_RS_CTX_CHAN_DESTROY, RaRsCtxChanDestroy, sizeof(union OpCtxChanDestroyData)},
    {RA_RS_CTX_CQ_CREATE, RaRsCtxCqCreate, sizeof(union OpCtxCqCreateData)},
    {RA_RS_CTX_CQ_DESTROY, RaRsCtxCqDestroy, sizeof(union OpCtxCqDestroyData)},
    {RA_RS_CTX_QP_CREATE, RaRsCtxQpCreate, sizeof(union OpCtxQpCreateData)},
    {RA_RS_CTX_QUERY_QP_BATCH, RaRsCtxQpQueryBatch, sizeof(union OpCtxQpQueryBatchData)},
    {RA_RS_CTX_QP_DESTROY, RaRsCtxQpDestroy, sizeof(union OpCtxQpDestroyData)},
    {RA_RS_CTX_QP_DESTROY_BATCH, RaRsCtxQpDestroyBatch, sizeof(union OpCtxQpDestroyBatchData)},
    {RA_RS_CTX_QP_IMPORT, RaRsCtxQpImport, sizeof(union OpCtxQpImportData)},
    {RA_RS_CTX_QP_UNIMPORT, RaRsCtxQpUnimport, sizeof(union OpCtxQpUnimportData)},
    {RA_RS_CTX_QP_UNIMPORT_DEPRECATED, RaRsCtxQpUnimportDeprecated, sizeof(union OpCtxQpUnimportData)},
    {RA_RS_CTX_QP_BIND, RaRsCtxQpBind, sizeof(union OpCtxQpBindData)},
    {RA_RS_CTX_QP_UNBIND, RaRsCtxQpUnbind, sizeof(union OpCtxQpUnbindData)},
    {RA_RS_CTX_BATCH_SEND_WR, RaRsCtxBatchSendWr, sizeof(union OpCtxBatchSendWrData)},
    {RA_RS_CTX_UPDATE_CI, RaRsCtxUpdateCi, sizeof(union OpCtxUpdateCiData)},
    {RA_RS_CTX_GET_AUX_INFO, RaRsCtxGetAuxInfo, sizeof(union OpCtxGetAuxInfoData)},
    {RA_RS_CTX_GET_CR_ERR_INFO_LIST, RaRsCtxGetCrErrInfoList, sizeof(union OpCtxGetCrErrInfoListData)},
    {RA_RS_CTX_NOTIFY_EVENT, RaRsCtxNotifyEvent, sizeof(union OpCtxNotifyEventData)},
    {RA_RS_CTX_GET_UB_CONTEXT, RaRsCtxGetUbContext, sizeof(union OpCtxGetContextData)},
};

STATIC int RaCheckParam(char *recvBuf, int rcvBufLen, char **sendBuf, int *sndBufLen, int *paramCheckResult)
{
    int i;
    int ret = 0;
    struct MsgHead *recvMsgHead = (struct MsgHead *)recvBuf;
    int num = sizeof(gRaOpHandle) / sizeof(gRaOpHandle[0]);
    unsigned int dataSize = 0;

    *paramCheckResult = 1;
    if (rcvBufLen < (int)sizeof(struct MsgHead)) { // check rcv_buf_len
        hccp_err("rcv_buf_len[%d] from ra is invalid", rcvBufLen);
        ret = OpMsgErr(sendBuf, recvMsgHead, sndBufLen, RECV_BUF_LEN_INVALID);
        return ret;
    }

    if (((recvMsgHead->msgDataLen + sizeof(struct MsgHead)) != (unsigned int)rcvBufLen) ||
        (recvMsgHead->opcode >= RA_RS_OP_MAX_NUM ||
            ((recvMsgHead->opcode < RA_RS_HDC_SESSION_CLOSE) && (recvMsgHead->opcode >= RA_RS_EXTER_OP_MAX_NUM)))) {
        hccp_err("rcv data incomplete, because rcvBufLen[%d] != msg_head_len[%u] + msgDataLen[%u] "
                 "or opcode[%u] is wrong, RA_RS_OP_MAX_NUM:[%d], RA_RS_EXTER_OP_MAX_NUM:[%d]",
            rcvBufLen, sizeof(struct MsgHead), recvMsgHead->msgDataLen, recvMsgHead->opcode, RA_RS_OP_MAX_NUM,
            RA_RS_EXTER_OP_MAX_NUM);
        ret = OpMsgErr(sendBuf, recvMsgHead, sndBufLen, HAVE_OP_RIGHT);
        return ret;
    }
    for (i = 0; i < num; i++) {
        if (gRaOpHandle[i].opcode == recvMsgHead->opcode) {
            dataSize = gRaOpHandle[i].dataSize;
            break;
        }
    }
    if (recvMsgHead->opcode != RA_RS_SOCKET_RECV && recvMsgHead->msgDataLen != dataSize) {
        hccp_err("rcv data incomplete. because msg_data_len[%d] != op_data_len[%u]", recvMsgHead->msgDataLen, dataSize);
        ret = OpMsgErr(sendBuf, recvMsgHead, sndBufLen, RECV_BUF_LEN_INVALID);
        return ret;
    }
    *paramCheckResult = 0;
    return ret;
}

int RaHandle(struct RaHdcOpSec *opSec, char *recvBuf, int rcvBufLen, char **sendBuf, int *sndBufLen,
    unsigned int *closeSession)
{
    int i;
    int ret;
    int opRight = 0;
    int paramCheckRet = 0;
    int opRet = 0;
    struct MsgHead *recvMsgHead = (struct MsgHead *)recvBuf;
    int num = sizeof(gRaOpHandle) / sizeof(gRaOpHandle[0]);

    ret = RaCheckParam(recvBuf, rcvBufLen, sendBuf, sndBufLen, &paramCheckRet);
    CHK_PRT_RETURN(paramCheckRet != 0 || ret != 0,
        hccp_err("ra param check failed. param check ret:[%d] "
                 "function call ret:[%d]",
            paramCheckRet, ret),
        ret);

    RaGetOpRight(opSec, recvMsgHead->opcode, recvMsgHead->asyncReqId, &opRight);
    CHK_PRT_RETURN(opRight != HAVE_OP_RIGHT, ret = OpMsgErr(sendBuf, recvMsgHead, sndBufLen, opRight), ret);

    *sendBuf = (char *)calloc(recvMsgHead->msgDataLen + sizeof(struct MsgHead), sizeof(char));
    CHK_PRT_RETURN(*sendBuf == NULL, hccp_err("calloc failed."), -ENOMEM);

    for (i = 0; i < num; i++) {
        if (gRaOpHandle[i].opcode == recvMsgHead->opcode) {
            ret = gRaOpHandle[i].opHandle(recvBuf, *sendBuf, sndBufLen, &opRet, rcvBufLen);
            if (ret) {
                hccp_err("ra handle failed. ret:[%d]", ret);
                goto out;
            }
            MsgHeadBuildUpHw(*sendBuf, recvMsgHead, opRet, recvMsgHead->msgDataLen);
            *closeSession = recvMsgHead->opcode == RA_RS_HDC_SESSION_CLOSE ? 1 : 0;
            *sndBufLen = recvMsgHead->msgDataLen + sizeof(struct MsgHead);
            return ret;
        }
    }

    hccp_warn_others("not support opcode:%d", recvMsgHead->opcode);
    ret = -EPROTONOSUPPORT;
out:
    free(*sendBuf);
    *sendBuf = NULL;
    return ret;
}

STATIC int RaSendPkt(struct drvHdcMsg *msgRcv, HDC_SESSION session, void *sendBuf, int sndBufLen)
{
    int ret;
    struct drvHdcMsg *msgSnd = NULL;

    ret = RA_HDC_OPS.reuseMsg(msgRcv);
    CHK_PRT_RETURN(ret, hccp_err("reuse msg failed ret %d", ret), ret);

    msgSnd = msgRcv;

    ret = RA_HDC_OPS.addMsgBuffer(msgSnd, sendBuf, sndBufLen);
    CHK_PRT_RETURN(ret, hccp_err("add msg buffer failed ret %d", ret), ret);

    ret = RA_HDC_OPS.send(session, msgSnd, 0, RA_HDC_RECV_SEND_TIMEOUT);
    CHK_PRT_RETURN(ret, hccp_err("send msg failed ret %d", ret), ret);

    return 0;
}

STATIC int RecvHandleSendPkt(HDC_SESSION session, unsigned int *closeSession, unsigned int chipId)
{
    int ret;
    void *recvBuf = NULL;
    void *sendBuf = NULL;
    struct drvHdcMsg *msgRcv = NULL;
    int recvBufCnt, sndBufLen, rcvBufLen;
    RsSetCtx(chipId);
    ret = RA_HDC_OPS.allocMsg(session, &msgRcv, 1);
    CHK_PRT_RETURN(ret, hccp_err("alloc hdc msg failed ret %d", ret), ret);

    ret = RA_HDC_OPS.recv(session, msgRcv, MAX_HDC_DATA, 0, &recvBufCnt, RA_HDC_RECV_SEND_TIMEOUT);
    if (ret) {
        hccp_warn_others("recv hdc msg unsuccessful, ret %d", ret);
        goto out;
    }

    RA_HDC_OPS.getMsgBuffer(msgRcv, 0, (char **)&recvBuf, &rcvBufLen);
    if (recvBuf == NULL) {
        hccp_warn_others("rcv_buf_len is NULL, Session disconnect.");
        goto out;
    }

    if (rcvBufLen == 0) {
        *closeSession = 1;
        hccp_warn_others("rcv_buf_len is 0, Session disconnect.");
        RA_HDC_OPS.freeMsg(msgRcv);
        return 0;
    }

    ret = RaHandle(&gHdcServer[chipId].opSec, recvBuf, rcvBufLen, (char **)&sendBuf, &sndBufLen, closeSession);
    if (ret) {
        hccp_err("ra_handle failed.");
        goto out;
    }

    ret = RaSendPkt(msgRcv, session, sendBuf, sndBufLen);
    if (ret) {
        hccp_err("ra send pkt failed ret %d", ret);
        goto err;
    }

err:
    free(sendBuf);
    sendBuf = NULL;
out:
    RA_HDC_OPS.freeMsg(msgRcv);
    return ret;
}

STATIC void RaHdcRecvHandleSendPkt(const unsigned int chipId)
{
    unsigned int closeSession = 0;
    int ret;

    ret = RecvHandleSendPkt(gHdcServer[chipId].hdcSession, &closeSession, chipId);
    if (closeSession != 0 || ret != 0) {
        hccp_warn_socket("recv_handle_send_pkt close_session[%u] ret[%d]", closeSession, ret);
        RA_PTHREAD_MUTEX_LOCK(&gHdcInitPara.mutex);
        gHdcInitPara.connectStatus = HDC_UNCONNECTED;
        RA_PTHREAD_MUTEX_UNLOCK(&gHdcInitPara.mutex);
    }

    return;
}

STATIC void RaHwHdcCloseSession(HDC_SESSION *session)
{
    int ret;

    RA_PTHREAD_MUTEX_LOCK(&gHdcInitPara.mutex);
    if (session == NULL || *session == NULL) {
        goto out;
    }

    ret = RA_HDC_OPS.sessionClose(*session);
    if (ret != 0) {
        hccp_warn_socket("RA_HDC_OPS.sessionClose unsuccessful, ret:%d", ret);
    }
    *session = NULL;

out:
    RA_PTHREAD_MUTEX_UNLOCK(&gHdcInitPara.mutex);
    return;
}

STATIC void *RaPthread(void *arg)
{
    (void)arg;
    unsigned int chipId = gHdcInitPara.chipId;
    int ret;

    ret = pthread_detach(pthread_self());
    CHK_PRT_RETURN(ret, hccp_err("pthread detach failed ret %d", ret), NULL);

    (void)prctl(PR_SET_NAME, (uintptr_t) "hccp_ra", 0, 0, 0);

    RA_PTHREAD_MUTEX_LOCK(&gHdcInitPara.mutex);
    gHdcInitPara.threadStatus = THREAD_RUNNING;
    RA_PTHREAD_MUTEX_UNLOCK(&gHdcInitPara.mutex);

    RsGetCurTime(&gRaThreadInfo.lastCheckTime);
    ret = strncpy_s((char *)gRaThreadInfo.pthreadName, sizeof(gRaThreadInfo.pthreadName), "ra_thread",
        strlen("ra_thread"));
    CHK_PRT_RETURN(ret, hccp_err("strncpy_s pthread name failed, ret[%d]", ret), NULL);

    hccp_run_info("pthread[%s] is alive!", gRaThreadInfo.pthreadName);
    while (1) {
        if (gHdcInitPara.threadStatus == THREAD_DESTROYING) {
            break;
        }

        if (gHdcInitPara.connectStatus != HDC_CONNECTED) {
            usleep(THREAD_SLEEP_TIME);
            continue;
        }
        RsHeartbeatAlivePrint(&gRaThreadInfo);
        RaHdcRecvHandleSendPkt(chipId);
    }

    hccp_info_others("thread [%d] is out", getpid());
    RaHwHdcCloseSession(&gHdcServer[chipId].hdcSession);
    RA_PTHREAD_MUTEX_LOCK(&gHdcInitPara.mutex);
    gHdcInitPara.threadStatus = THREAD_HALT;
    RA_PTHREAD_MUTEX_UNLOCK(&gHdcInitPara.mutex);
    return NULL;
}

STATIC int RaHdcServerInit(unsigned int chipId, int hdcType)
{
    int ret;

    CHK_PRT_RETURN(chipId > HCCP_MAX_CHIP_ID || gHdcServer[chipId].hdcSession != NULL,
        hccp_err("invalid "
                 "chip id %u, or hdcSession is not NULL",
            chipId),
        -EINVAL);
    CHK_PRT_RETURN(hdcType != HDC_SERVICE_TYPE_RDMA && hdcType != HDC_SERVICE_TYPE_RDMA_V2,
        hccp_err("invalid "
                 "hdc_type %d",
            hdcType),
        -EINVAL);

    RaHdcInitOpSec(&gHdcServer[chipId].opSec, BUCKET_DEPTH, false);

    ret = RA_HDC_OPS.serverCreate(chipId, hdcType, &gHdcServer[chipId].hdcServer);
    CHK_PRT_RETURN(ret, hccp_err("Create Server failed, ret(%d) ", ret), -EINVAL);

    return 0;
}

void RaHdcInitOpSec(struct RaHdcOpSec *opSec, unsigned long long tokenNum, bool isAsyncOp)
{
    opSec->tokenNum = tokenNum;
    opSec->cfgOpNum = 0;
    opSec->tLast.tv_sec = 0;
    opSec->tLast.tv_usec = 0;
    opSec->isAsyncOp = isAsyncOp;
}

int RaHdcSessionAccept(unsigned int chipId, HDC_SESSION *session, int initHostTgid)
{
    int hostTgid;
    int ret = 0;

    ret = RA_HDC_OPS.sessionAccept(gHdcServer[chipId].hdcServer, session);
    if (ret != 0) {
        hccp_warn_socket("Session accept failed, chipId(%u), ret(%d) ", chipId, ret);
        return ret;
    }

    RA_HDC_OPS.setSessionReference(*session);
    ret = DlHalHdcGetSessionAttr(*session, HDC_SESSION_ATTR_PEER_CREATE_PID, &hostTgid);
    if (ret) {
        hccp_err("Session get host_pid failed, chipId(%u), ret(%d)", chipId, ret);
        goto out;
    }

    if (hostTgid != initHostTgid) {
        hccp_warn_others("host_tgid[%d] from ra not equal to the tgid[%d] from hccp_init, invalid", hostTgid,
            initHostTgid);
        goto out;
    }

    return 0;

out:
    RaHdcCloseSession(session);
    return ret;
}

int RaHdcAsyncRecvPkt(struct RaHdcAsyncInfo *asyncInfo, unsigned int chipId, void **recvBuf, unsigned int *recvLen)
{
    (void)chipId;
    struct drvHdcMsg *msgRcv = NULL;
    void *rcvBuf = NULL;
    int rcvLen = 0;
    int ret;

    ret = RA_HDC_OPS.allocMsg(asyncInfo->hdcSession, &msgRcv, 1);
    CHK_PRT_RETURN(ret != 0, hccp_err("alloc hdc msg failed ret %d", ret), ret);

    ret = RA_HDC_OPS.recv(asyncInfo->hdcSession, msgRcv, MAX_HDC_DATA, 0, &rcvLen, RA_HDC_RECV_SEND_TIMEOUT);
    if (ret != 0) {
        hccp_warn_socket("recv hdc msg unsuccessful ret %d", ret);
        goto out;
    }

    RA_HDC_OPS.getMsgBuffer(msgRcv, 0, (char **)&rcvBuf, &rcvLen);
    if (rcvBuf == NULL || rcvLen == 0) {
        hccp_warn_others("get_msg_buffer unsuccessful, rcvBuf is NULL or rcvLen:%d is 0", rcvLen);
        goto out;
    }

    *recvBuf = (char *)calloc(rcvLen, sizeof(char));
    if (*recvBuf == NULL) {
        hccp_err("calloc recv_buf failed, errno:%d rcvLen:%d", errno, rcvLen);
        ret = -ENOMEM;
        goto out;
    }

    (void)memcpy_s(*recvBuf, rcvLen, rcvBuf, rcvLen);
    *recvLen = rcvLen;

out:
    RA_HDC_OPS.freeMsg(msgRcv);
    return ret;
}

int RaHdcAsyncSendPkt(struct RaHdcAsyncInfo *asyncInfo, unsigned int chipId, void *sendBuf, unsigned int sendLen)
{
    struct drvHdcMsg *msgSnd = NULL;
    int ret = -EINVAL;

    RA_PTHREAD_MUTEX_LOCK(&asyncInfo->sendMutex);
    // degrade log level because session will be closed by recv thread and request will be abort
    if (asyncInfo->hdcSession == NULL) {
        hccp_warn_socket("[async][send_pkt]hdc_session is NULL, chipId(%u)", chipId);
        goto alloc_msg_err;
    }

    ret = RA_HDC_OPS.allocMsg(asyncInfo->hdcSession, &msgSnd, 1);
    if (ret != 0) {
        hccp_err("[async][send_pkt]HDC alloc msg err ret(%d) chip_id(%u)", ret, chipId);
        goto alloc_msg_err;
    }

    ret = RA_HDC_OPS.addMsgBuffer(msgSnd, sendBuf, sendLen);
    if (ret != 0) {
        hccp_err("[async][send_pkt]HDC add msg buffer err ret(%d) chip_id(%u)", ret, chipId);
        goto msg_err;
    }

    ret = RA_HDC_OPS.send(asyncInfo->hdcSession, msgSnd, RA_HDC_WAIT_TIMEOUT, RA_HDC_RECV_SEND_TIMEOUT);
    if (ret != 0) {
        hccp_err("[async][send_pkt]HDC send err ret(%d) chip_id(%u)", ret, chipId);
        goto msg_err;
    }

msg_err:
    RA_HDC_OPS.freeMsg(msgSnd);
alloc_msg_err:
    RA_PTHREAD_MUTEX_UNLOCK(&asyncInfo->sendMutex);
    return ret;
}

void RaHdcCloseSession(HDC_SESSION *session)
{
    RA_HDC_OPS.sessionClose(*session);
    *session = NULL;
    return;
}

STATIC void RaHwHdcInit(void *arg)
{
    (void)arg;
    unsigned int chipId = gHdcInitPara.chipId;
    pthread_t tidp;
    int ret;

    ret = pthread_detach(pthread_self());
    if (ret) {
        hccp_err("pthread detach failed ret %d", ret);
        return;
    }

    (void)prctl(PR_SET_NAME, (uintptr_t) "hccp_hw_hdc", 0, 0, 0);

    hccp_info_others("chip_id(%u)", chipId);
    gHdcInitPara.hdcFlag = 1;

    ret = pthread_create(&tidp, NULL, (void *)RaPthread, NULL);
    if (ret) {
        hccp_err("Create pthread failed, chipId(%u), ret(%d) ", chipId, ret);
        return;
    }

    while (1) {
        if (gHdcInitPara.connectStatus != HDC_UNCONNECTED) {
            usleep(HDC_ACCEPT_SLEEP_TIME);
            continue;
        }
        RaHwHdcCloseSession(&gHdcServer[chipId].hdcSession);
        ret = RaHdcSessionAccept(chipId, &gHdcServer[chipId].hdcSession, (int)gHdcInitPara.hostTgid);
        if (ret != 0) {
            hccp_warn_socket("Session Accept unsuccessful, chipId(%u), ret(%d) ", chipId, ret);
            gHdcInitPara.hdcFlag = 0;
            return;
        }
        // original case, should continue to accept: host_tgid != g_hdc_init_para.host_tgid
        if (ret == 0 && gHdcServer[chipId].hdcSession == NULL) {
            continue;
        }
        RA_PTHREAD_MUTEX_LOCK(&gHdcInitPara.mutex);
        gHdcInitPara.connectStatus = HDC_CONNECTED;
        RA_PTHREAD_MUTEX_UNLOCK(&gHdcInitPara.mutex);
    }
}

STATIC void RaHwHdcDeinit(void)
{
    unsigned int chipId = gHdcInitPara.chipId;
    int ret, tryAgain;

    RA_PTHREAD_MUTEX_LOCK(&gHdcInitPara.mutex);
    gHdcInitPara.threadStatus = THREAD_DESTROYING;
    RA_PTHREAD_MUTEX_UNLOCK(&gHdcInitPara.mutex);

    tryAgain = HDC_TRY_TIME;
    while ((gHdcInitPara.threadStatus != THREAD_HALT) && tryAgain != 0) {
        usleep(HDC_USLEEP_TIME);
        tryAgain--;
    }
    if (tryAgain == 0) {
        hccp_warn_others("hdc message thread quit timeout, chipId:%u", chipId);
    }

    if (gHdcServer[chipId].hdcServer != NULL) {
        ret = RA_HDC_OPS.serverDestroy(gHdcServer[chipId].hdcServer);
        if (ret != 0) {
            hccp_warn_socket("RA_HDC_OPS.server_destroy unsuccessful, ret:%d, chipId:%u", ret, chipId);
        }
        gHdcServer[chipId].hdcServer = NULL;
    } else {
        hccp_warn_socket("hdc_server is NULL, chipId:%u", chipId);
    }
    pthread_mutex_destroy(&gHdcInitPara.mutex);
}

STATIC int HccpSetAffinity(unsigned int chipId)
{
    unsigned int phyDevid = chipId;
    int64_t cpuCoreNum;
    int64_t ccpuNum; /* ctrl cpu */
    int64_t dcpuNum; /* data cpu */
    int64_t acpuNum; /* ai cpu */
    cpu_set_t mask;
    int64_t cpuId;
    int ret;

    ret = DlHalGetDeviceInfo(chipId, MODULE_TYPE_CCPU, INFO_TYPE_CORE_NUM, &ccpuNum);
    CHK_PRT_RETURN(ret, hccp_err("get ccpu_num failed, ret(%d)", ret), ret);

    ret = DlHalGetDeviceInfo(chipId, MODULE_TYPE_DCPU, INFO_TYPE_CORE_NUM, &dcpuNum);
    CHK_PRT_RETURN(ret, hccp_err("get dcpu_num failed, ret(%d)", ret), ret);

    ret = DlHalGetDeviceInfo(chipId, MODULE_TYPE_AICPU, INFO_TYPE_CORE_NUM, &acpuNum);
    CHK_PRT_RETURN(ret, hccp_err("get acpu_num failed, ret(%d)", ret), ret);
    cpuCoreNum = ccpuNum + dcpuNum + acpuNum;

    ret = DlHalGetPhyDevIdByudevId(chipId, &phyDevid);
    CHK_PRT_RETURN(ret != 0, hccp_err("DlHalGetPhyDevIdByudevId failed, ret(%d) chipId:%u", ret, chipId), ret);
    CPU_ZERO(&mask);
    cpuId = cpuCoreNum * phyDevid + HCCP_RUN_CPU_CORE;
    /*lint -e574*/
    CPU_SET((size_t)cpuId, &mask); // lint !e573
    /*lint +e574*/
    hccp_run_info("chipId:%u phyDevid:%u ccpuNum:%lld, dcpuNum:%lld, acpuNum:%lld, cpuId:%lld", chipId, phyDevid,
        ccpuNum, dcpuNum, acpuNum, cpuId);
    ret = sched_setaffinity(getpid(), sizeof(mask), &mask); /* hccp use core0 of each chip to set affinity */
    CHK_PRT_RETURN(ret == -1,
        hccp_err("sched_setaffinity failed: ret %d, errno %d chipId:%u phyDevid:%u", ret, errno, chipId, phyDevid),
        -ESYSFUNC);

    return 0;
}

STATIC int RaHwInit(unsigned int chipId, pid_t pid)
{
    int ret;
    pthread_t tidp;
    int timeout = RA_THREAD_TRY_TIME;

    gHdcInitPara.chipId = chipId;
    gHdcInitPara.hostTgid = pid;

    ret = pthread_create(&tidp, NULL, (void *)RaHwHdcInit, NULL);
    CHK_PRT_RETURN(ret, hccp_err("Create pthread failed, ret(%d) ", ret), -ESYSFUNC);

    while (gHdcInitPara.hdcFlag != 1 && timeout > 0) {
        usleep(RA_THREAD_SLEEP_TIME);
        timeout--;
    }

    CHK_PRT_RETURN(gHdcInitPara.hdcFlag == 0 || timeout == 0,
        hccp_err("HDC server thread create timeout,"
                 "flag %d, timeout %d",
            gHdcInitPara.hdcFlag, timeout),
        -ESRCH);

    return 0;
}

RA_ADP_ATTRI_VISI_DEF int HccpInit(unsigned int chipId, pid_t pid, int hdcType, unsigned int whiteListStatus)
{
    struct timeval start, end;
    float timeCost = 0.0;
    int ret, retTmp;

    hccp_info_init("hccp[%u] hdc_type[%d] white_list_status[%u] init start", chipId, hdcType, whiteListStatus);

    ret = DlHalInit();
    if (ret != 0) {
        hccp_err("dl_hal_init failed, ret = %d", ret);
        return ret;
    }

    ret = HccpSetAffinity(chipId);
    if (ret != 0) {
        hccp_err("hccp_set_affinity failed, ret(%d) ", ret);
        goto out;
    }

    RsGetCurTime(&start);
    ret = RaHdcServerInit(chipId, hdcType);
    if (ret != 0) {
        hccp_err("chip_id[%u] hdc_type[%d] ra_hdc_server_init failed, ret[%d] ", chipId, hdcType, ret);
        goto out;
    }

    ret = pthread_mutex_init(&gHdcInitPara.mutex, NULL);
    if (ret != 0) {
        hccp_err("g_hdc_init_para mutex_init failed ret %d!, normal ret 0", ret);
        ret = -ESYSFUNC;
        goto out;
    }

    ret = RaHwInit(chipId, pid);
    if (ret != 0) {
        hccp_err("ra_init failed, ret(%d) ", ret);
        goto hw_init_err;
    }

    ret = RaHwAsyncInit(chipId, pid);
    if (ret != 0) {
        hccp_err("ra_hw_async_init failed, ret(%d) ", ret);
        goto hw_init_err;
    }

    RsGetCurTime(&end);
    HccpTimeInterval(&end, &start, &timeCost);
    hccp_info_init("ra_hw_init ok cost [%f] ms", timeCost);

    struct RsInitConfig offlineConfig = {
        .chipId = chipId,
        .hccpMode = NETWORK_OFFLINE,
        .whiteListStatus = whiteListStatus,
    };

    RsGetCurTime(&start);
    ret = RsInit(&offlineConfig);
    if (ret != 0) {
        hccp_err("rs_init failed (0x%x) ", ret);
        goto init_err;
    }
    RsGetCurTime(&end);
    HccpTimeInterval(&end, &start, &timeCost);
    hccp_info_init("rs_init ok cost [%f] ms", timeCost);

    RsGetCurTime(&start);
    ret = RsBindHostpid(chipId, pid);
    if (ret != 0) {
        hccp_err("rs_bind_hostpid failed, ret=%d ", ret);
        goto bind_hostpid_err;
    }
    RsGetCurTime(&end);
    HccpTimeInterval(&end, &start, &timeCost);
    hccp_info_init("rs_bind_hostpid ok cost [%f] ms", timeCost);

    RsGetCurTime(&start);
    ret = RsPingHandleInit(chipId, hdcType, whiteListStatus);
    if (ret != 0) {
        hccp_err("rs_ping_handle_init failed, ret=%d ", ret);
        goto bind_hostpid_err;
    }
    RsGetCurTime(&end);
    HccpTimeInterval(&end, &start, &timeCost);
    hccp_info_init("rs_ping_handle_init ok cost [%f] ms", timeCost);

    return 0;
bind_hostpid_err:
    retTmp = RsDeinit(&offlineConfig);
    if (retTmp) {
        hccp_err("rs_deinit failed %d ", retTmp);
    }
init_err:
    RaHwAsyncDeinit();
hw_init_err:
    pthread_mutex_destroy(&gHdcInitPara.mutex);
out:
    DlHalDeinit();
    return ret;
}

RA_ADP_ATTRI_VISI_DEF int HccpDeinit(unsigned int chipId)
{
    struct RsInitConfig offlineConfig = {
        .chipId = chipId,
        .hccpMode = NETWORK_OFFLINE,
        .whiteListStatus = WHITE_LIST_ENABLE,
    };
    int ret;

    hccp_info_init("hccp[%u] deinit start", chipId);

    ret = RsPingHandleDeinit(chipId);
    CHK_PRT_RETURN(ret, hccp_err("rs_ping_handle_deinit failed %d ", ret), ret);

    ret = RsDeinit(&offlineConfig);
    CHK_PRT_RETURN(ret, hccp_err("rs_deinit failed %d ", ret), ret);

    RaHwHdcDeinit();

    RaHwAsyncDeinit();
    DlHalDeinit();
    hccp_info_init("hccp [%u] deinit success", chipId);

    return ret;
}
