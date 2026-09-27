/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_CLIENT_HOST_H
#define RA_CLIENT_HOST_H

#include <errno.h>
#include <stdbool.h>
#include <infiniband/verbs.h>
#include "ra_rs_comm.h"
#include "ra.h"

struct RaSocketOps {
    int (*raSocketInit)(struct rdev rdevInfo);
    int (*raSocketDeinit)(struct rdev rdevInfo);
    int (*raSocketBatchConnect)(unsigned int phyId, struct SocketConnectInfoT conn[], unsigned int num);
    int (*raSocketBatchClose)(unsigned int phyId, struct SocketCloseInfoT conn[], unsigned int num);
    int (*raSocketBatchAbort)(unsigned int phyId, struct SocketConnectInfoT conn[], unsigned int num);
    int (*raSocketListenStart)(unsigned int phyId, struct SocketListenInfoT conn[], unsigned int num);
    int (*raSocketListenStop)(unsigned int phyId, struct SocketListenInfoT conn[], unsigned int num);
    int (*raGetSockets)(unsigned int phyId, unsigned int role, struct SocketInfoT conn[], unsigned int num);
    int (*raSocketSend)(unsigned int phyId, const void *handle, const void *data, unsigned long long size);
    int (*raSocketRecv)(unsigned int phyId, const void *handle, void *data, unsigned long long size);
    int (*raGetClientSocketErrInfo)(unsigned int phyId, struct SocketConnectInfoT conn[], struct SocketErrInfo err[],
        unsigned int num);
    int (*raGetServerSocketErrInfo)(unsigned int phyId, struct SocketListenInfoT conn[],
        struct ServerSocketErrInfo err[], unsigned int num);
    int (*raSocketSetWhiteListStatus)(unsigned int enable);
    int (*raSocketGetWhiteListStatus)(unsigned int *enable);
    int (*raSocketWhiteListAdd)(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], unsigned int num);
    int (*raSocketWhiteListDel)(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], unsigned int num);
    int (*raSocketAcceptCreditAdd)(unsigned int phyId, struct SocketListenInfoT conn[], unsigned int num,
        unsigned int creditLimit);
};

struct RaRdmaOps {
    int (*raRdevInit)(struct RaRdmaHandle *rdmaHandle, unsigned int notifyType, struct rdev rdevInfo,
        unsigned int *rdevIndex);
    int (*raRdevGetPortStatus)(struct RaRdmaHandle *rdmaHandle, enum PortStatus *status);
    int (*raGetLbMax)(struct RaRdmaHandle *rdmaHandle, int *lbMax);
    int (*raRdevDeinit)(struct RaRdmaHandle *rdmaHandle, unsigned int notifyType);
    int (*raSetTsqpDepth)(struct RaRdmaHandle *rdmaHandle, unsigned int tempDepth, unsigned int *qpNum);
    int (*raGetTsqpDepth)(struct RaRdmaHandle *rdmaHandle, unsigned int *tempDepth, unsigned int *qpNum);
    int (*raQpCreate)(struct RaRdmaHandle *rdmaHandle, int flag, int qpMode, void **qpHandle);
    int (*raQpCreateWithAttrs)(struct RaRdmaHandle *rdmaHandle, struct QpExtAttrs *extAttrs, void **qpHandle);
    int (*raAiQpCreate)(struct RaRdmaHandle *rdmaHandle, struct QpExtAttrs *extAttrs, struct AiQpInfo *info,
        void **qpHandle);
    int (*raAiQpCreateWithAttrs)(struct RaRdmaHandle *rdmaHandle, struct QpExtAttrs *extAttrs, struct AiQpInfo *info,
        void **qpHandle);
    int (*raTypicalQpCreate)(struct RaRdmaHandle *rdmaHandle, int flag, int qpMode, struct TypicalQp *qpInfo,
        void **qpHandle);
    int (*raTypicalCqCreate)(struct RaRdmaHandle *rdmaHandle, unsigned int cqDepth, unsigned int *cqn, void **cqHandle);
    int (*raTypicalCqDestroy)(struct RaRdmaHandle *rdmaHandle, unsigned int cqn, void *cqHandle);
    int (*raQpCreateWithCQWithAttrs)(struct RaRdmaHandle *rdmaHandle, struct QpExtAttrs *extAttrs, unsigned int sendCqn,
        unsigned int recvCqn, void **qpHandle);
    int (*raLoopbackQpCreate)(struct RaRdmaHandle *rdevHandle, struct LoopbackQpPair *qpPair, void **qpHandle);
    int (*raQpDestroy)(struct RaQpHandle *handle);
    int (*raQpDestroyWithoutCQ)(struct RaQpHandle *handle);
    int (*raTypicalQpModify)(struct RaQpHandle *handle, struct TypicalQp *localQpInfo, struct TypicalQp *remoteQpInfo);
    int (*raQpBatchModify)(struct RaRdmaHandle *handle, void *qpHdc[], unsigned int num, int expectStatus);
    int (*raSetQpLbValue)(struct RaQpHandle *handle, int lbValue);
    int (*raGetQpLbValue)(struct RaQpHandle *handle, int *lbValue);
    int (*raQpConnectAsync)(struct RaQpHandle *handle, const void *sockHandle);
    int (*raGetQpStatus)(struct RaQpHandle *handle, int *status);
    int (*raMrReg)(struct RaQpHandle *handle, struct MrInfoT *info);
    int (*raMrDereg)(struct RaQpHandle *handle, struct MrInfoT *info);
    int (*raRegisterMr)(struct RaRdmaHandle *handle, struct MrInfoT *info, void **mrHandle);
    int (*raRemapMr)(struct RaRdmaHandle *handle, struct MemRemapInfo info[], unsigned int num);
    int (*raDeregisterMr)(struct RaRdmaHandle *handle, void *mrHandle);
    int (*raSendWr)(struct RaQpHandle *handle, struct SendWr *wr, struct SendWrRsp *wrRsp);
    int (*raSendWrV2)(struct RaQpHandle *handle, struct SendWrV2 *wr, struct SendWrRsp *wrRsp);
    int (*raSendWrVerbs)(struct RaQpHandle *handle, struct SendWrVerbs *wr, struct SendWrRsp *wrRsp);
    int (*raRecvWrVerbs)(struct RaQpHandle *handle, struct RecvWrVerbs *wr);
    int (*raTypicalSendWr)(struct RaQpHandle *handle, struct SendWr *wr, struct SendWrRsp *wrRsp);
    int (*raSendWrlist)(struct RaQpHandle *handle, struct SendWrlistData wr[], struct SendWrRsp opRsp[],
        struct WrlistSendCompleteNum wrlistNum);
    int (*raSendWrlistExt)(struct RaQpHandle *handle, struct SendWrlistDataExt wr[], struct SendWrRsp opRsp[],
        struct WrlistSendCompleteNum wrlistNum);
    int (*raSendNormalWrlist)(struct RaQpHandle *handle, struct WrInfo wr[], struct SendWrRsp opRsp[],
        struct WrlistSendCompleteNum wrlistNum);
    int (*raGetNotifyBaseAddr)(struct RaRdmaHandle *handle, unsigned long long *va, unsigned long long *size);
    int (*raGetNotifyMrInfo)(struct RaRdmaHandle *handle, struct MrInfoT *info);
    int (*raRecvWrlist)(struct RaQpHandle *handle, struct RecvWrlistData *wr, unsigned int recvNum,
        unsigned int *completeNum);
    int (*raPollCq)(struct RaQpHandle *handle, bool isSendCq, unsigned int numEntries, void *wc);
    int (*raPollTypicalCq)(struct RaTypicalCqHandle *cqHandle, unsigned int numEntries, void *wc);
    int (*raGetQpContext)(struct RaQpHandle *handle, void **qp, void **sendCq, void **recvCq);
    int (*raNormalQpCreate)(struct RaRdmaHandle *rdmaHandle, struct ibv_qp_init_attr *qpInitAttr, void **qpHandle,
        void **qp);
    int (*raNormalQpDestroy)(struct RaQpHandle *handle);
    int (*raCqCreate)(struct RaRdmaHandle *rdmaHandle, struct CqAttr *attr);
    int (*raCqDestroy)(struct RaRdmaHandle *rdmaHandle, struct CqAttr *attr);
    int (*raSetQpAttrQos)(struct RaQpHandle *handle, struct QosAttr *info);
    int (*raSetQpAttrTimeout)(struct RaQpHandle *handle, unsigned int *timeout);
    int (*raSetQpAttrRetryCnt)(struct RaQpHandle *handle, unsigned int *retryCnt);
    int (*raGetQpHyperFeature)(struct RaQpHandle *qpPeer, struct HyperFeature *hyperFeature);
    int (*raCreateCompChannel)(struct RaRdmaHandle *handle, void **compChannel);
    int (*raDestroyCompChannel)(void *compChannel);
    int (*raCreateSrq)(struct RaRdmaHandle *handle, struct SrqAttr *attr);
    int (*raDestroySrq)(struct RaRdmaHandle *handle, struct SrqAttr *attr);
};

enum ErrTypeDef {
    TYPE_EXE_OK = 0,             // execute successful.
    TYPE_CODE_OR_ENV_ERR = 1,    // code or env error, need to check param or interface call seq or check env, etc.
    TYPE_CODE_MATCH_ENV_ERR = 2, // code does not match env, need to check param or interface to match the env, etc.
    TYPE_SERVICE_ERR = 3,        // service abnormal caused by full or empty queue, etc.
    TYPE_INTERNAL_ERR = 5,       // need to solve the problem on our own
};

struct ErrcodeInfo {
    int origErrcode;
    int errType;
    int moduleErrcode;
};

#define RA_VNIC_MAX 128
#define RA_NOTIFY_TYPE_NUM 2
#define RA_MIN_TEMPTH_DEPTH 8
#define RA_MAX_TEMPTH_DEPTH 4096

#define DEFAULT_ERRCODE_TYPE 3
#define DEFAULT_MODULE_ERRCODE 7
#define HCCP_MODULE_ID 28
#define ACL_ERRCODE_DIGIT 100000

#define CONVER_ERROR_CODE(module, err_type, module_errcode)                                                            \
    ((err_type) * 100000 + (HCCP_MODULE_ID) * 1000 + (module) * 100 +                                                  \
        (module_errcode)) /* Combine a 6-digit ACL error code. */

int RaRdevInitCheck(int mode, struct rdev rdevInfo, char localIp[], unsigned int num, void *rdmaHandle);
int RaInetPton(int family, union HccpIpAddr ip, char netAddr[], unsigned int len);
#endif // RA_CLIENT_HOST_H
