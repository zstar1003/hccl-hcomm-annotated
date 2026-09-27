/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_PEER_H
#define RA_PEER_H

#include <pthread.h>
#include <infiniband/verbs.h>
#include "hccp_common.h"
#include "ra_rs_comm.h"
#include "ra_comm.h"

#define HCCP_CLOSE_RETRY_FOR_EINTR(fd)                                                                                 \
    do {                                                                                                               \
        int ret_;                                                                                                      \
        do {                                                                                                           \
            ret_ = close(fd);                                                                                          \
            if (ret_ < 0) {                                                                                            \
                hccp_warn("close fd[%d] unsuccessful, errno:%d", fd, errno);                                           \
            }                                                                                                          \
        } while ((ret_ < 0) && (errno == EINTR));                                                                      \
        fd = -1;                                                                                                       \
    } while (0)

#define PEER_PTHREAD_MUTEX_LOCK(mutex)                                                                                 \
    do {                                                                                                               \
        int ret_lock = pthread_mutex_lock(mutex);                                                                      \
        if (ret_lock) {                                                                                                \
            hccp_warn("pthread_mutex_lock unsuccessful, ret[%d]", ret_lock);                                           \
        }                                                                                                              \
    } while (0)

#define PEER_PTHREAD_MUTEX_UNLOCK(mutex)                                                                               \
    do {                                                                                                               \
        int ret_ulock = pthread_mutex_unlock(mutex);                                                                   \
        if (ret_ulock) {                                                                                               \
            hccp_warn("pthread_mutex_unlock unsuccessful, ret[%d]", ret_ulock);                                        \
        }                                                                                                              \
    } while (0)

#define RA_SSL_DISABLE 0

struct HostRoceNotifyInfo {
    unsigned int logicId;
    unsigned long long va;
    unsigned long long sz;
};

#define HOST_DEVICE_NAME "/dev/host_rdma"
#define RA_NOTIFY_TYPE_TOTAL_SIZE 1

#define HOST_CDEV_IOC_MAGIC '%'

#define HOST_CDEV_IOC_FREE_NOTIFY _IOWR(HOST_CDEV_IOC_MAGIC, 1, struct HostRoceNotifyInfo)

int RaPeerSocketBatchConnect(unsigned int devId, struct SocketConnectInfoT conn[], unsigned int num);

int RaPeerSocketBatchClose(unsigned int devId, struct SocketCloseInfoT conn[], unsigned int num);

int RaPeerSocketBatchAbort(unsigned int devId, struct SocketConnectInfoT conn[], unsigned int num);

int RaPeerSocketListenStart(unsigned int devId, struct SocketListenInfoT conn[], unsigned int num);

int RaPeerSocketListenStop(unsigned int devId, struct SocketListenInfoT conn[], unsigned int num);

int RaPeerGetSockets(unsigned int phyId, unsigned int role, struct SocketInfoT conn[], unsigned int num);

int RaPeerSocketSend(unsigned int devId, const void *handle, const void *data, unsigned long long size);

int RaPeerSocketRecv(unsigned int devId, const void *handle, void *data, unsigned long long size);

int RaPeerEpollCtlAdd(const void *fdHandle, enum RaEpollEvent event);

int RaPeerEpollCtlMod(const void *fdHandle, enum RaEpollEvent event);

int RaPeerEpollCtlDel(const void *fdHandle);

void RaPeerSetTcpRecvCallback(unsigned int phyId, const void *callback);

int RaPeerSocketWhiteListAdd(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], unsigned int num);

int RaPeerSocketWhiteListDel(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], unsigned int num);

int RaPeerSocketDeinit(struct rdev rdevInfo);

int RaPeerQpCreate(struct RaRdmaHandle *rdmaHandle, int flag, int qpMode, void **qpHandle);

int RaPeerQpCreateWithAttrs(struct RaRdmaHandle *rdmaHandle, struct QpExtAttrs *extAttrs, void **qpHandle);

int RaPeerLoopbackQpCreate(struct RaRdmaHandle *rdmaHandle, struct LoopbackQpPair *qpPair, void **qpHandle);

int RaPeerQpDestroy(struct RaQpHandle *qpPeer);

int RaPeerTypicalQpModify(struct RaQpHandle *qpPeer, struct TypicalQp *localQpInfo, struct TypicalQp *remoteQpInfo);

int RaPeerSetQpLbValue(struct RaQpHandle *qpHandle, int lbValue);

int RaPeerGetQpLbValue(struct RaQpHandle *qpHandle, int *lbValue);

int RaPeerQpConnectAsync(struct RaQpHandle *qpPeer, const void *sockHandle);

int RaPeerGetQpStatus(struct RaQpHandle *qpPeer, int *status);

int RaPeerMrReg(struct RaQpHandle *qpPeer, struct MrInfoT *info);

int RaPeerMrDereg(struct RaQpHandle *qpPeer, struct MrInfoT *info);

int RaPeerRegisterMr(struct RaRdmaHandle *rdmaPeer, struct MrInfoT *info, void **mrHandle);

int RaPeerDeregisterMr(struct RaRdmaHandle *rdmaPeer, void *mrHandle);

int RaPeerSendWr(struct RaQpHandle *qpPeer, struct SendWr *wr, struct SendWrRsp *wrRsp);

int RaPeerSendWrlist(struct RaQpHandle *qpHandle, struct SendWrlistData wr[], struct SendWrRsp opRsp[],
    struct WrlistSendCompleteNum wrlistNum);

int RaPeerRecvWrlist(struct RaQpHandle *qpHandle, struct RecvWrlistData *wr, unsigned int recvNum,
    unsigned int *completeNum);

int RaPeerGetNotifyBaseAddr(struct RaRdmaHandle *handle, unsigned long long *va, unsigned long long *size);

int RaPeerInit(struct RaInitConfig *cfg, unsigned int whiteListStatus);

int RaPeerGetTlsEnable(unsigned int phyId, bool *tlsEnable);

int RaPeerGetHccnCfg(struct RaInfo *info, enum HccnCfgKey key, char *value, unsigned int *valueLen);

int RaPeerGetSecRandom(unsigned int *value);

int RaPeerDeinit(struct RaInitConfig *cfg);

int RaPeerGetIfnum(unsigned int phyId, unsigned int *num);

int RaPeerGetIfaddrs(unsigned int phyId, struct InterfaceInfo interfaceInfos[], unsigned int *num);

int RaPeerRdevInit(struct RaRdmaHandle *rdmaHandle, unsigned int notifyType, struct rdev rdevInfo,
    unsigned int *rdevIndex);

int RaPeerRdevGetPortStatus(struct RaRdmaHandle *rdmaHandle, enum PortStatus *status);

int RaPeerGetLbMax(struct RaRdmaHandle *rdmaHandle, int *lbMax);

int RaPeerRdevDeinit(struct RaRdmaHandle *rdmaHandle, unsigned int notifyType);

int HostNotifyBaseAddrInit(unsigned int phyId);

int RaPeerNotifyBaseAddrInit(unsigned int notifyType, unsigned int phyId);

int HostNotifyBaseAddrUninit(unsigned int phyId);

int NotifyBaseAddrUninit(unsigned int notifyType, unsigned int phyId);

int RaPeerSetTsqpDepth(struct RaRdmaHandle *rdmaHandle, unsigned int tempDepth, unsigned int *qpNum);

int RaPeerGetTsqpDepth(struct RaRdmaHandle *rdmaHandle, unsigned int *tempDepth, unsigned int *qpNum);

int RaPeerGetQpContext(struct RaQpHandle *qpPeer, void **qp, void **sendCq, void **recvCq);

int RaPeerNormalQpCreate(struct RaRdmaHandle *rdmaHandle, struct ibv_qp_init_attr *qpInitAttr, void **qpHandle,
    void **qp);

int RaPeerNormalQpDestroy(struct RaQpHandle *qpPeer);

int RaPeerCqCreate(struct RaRdmaHandle *rdmaHandle, struct CqAttr *attr);

int RaPeerCqDestroy(struct RaRdmaHandle *rdmaHandle, struct CqAttr *attr);

int RaPeerSetQpAttrQos(struct RaQpHandle *qpPeer, struct QosAttr *attr);

int RaPeerSetQpAttrTimeout(struct RaQpHandle *qpPeer, unsigned int *timeout);

int RaPeerSetQpAttrRetryCnt(struct RaQpHandle *qpPeer, unsigned int *retryCnt);

int RaPeerCreateCompChannel(struct RaRdmaHandle *rdmaHandle, void **compChannel);

int RaPeerDestroyCompChannel(void *compChannel);

int RaPeerCreateSrq(struct RaRdmaHandle *rdmaHandle, struct SrqAttr *attr);

int RaPeerDestroySrq(struct RaRdmaHandle *rdmaHandle, struct SrqAttr *attr);

int RaPeerCreateEventHandle(int *eventHandle);

int RaPeerCtlEventHandle(int eventHandle, const void *fdHandle, int opcode, enum RaEpollEvent event);

int RaPeerWaitEventHandle(int eventHandle, struct SocketEventInfoT *eventInfos, int timeout, unsigned int maxevents,
    unsigned int *eventsNum);

int RaPeerDestroyEventHandle(int *eventHandle);

void RaPeerMutexLock(unsigned int phyId);

void RaPeerMutexUnlock(unsigned int phyId);

void RaPeerSetDebugConfig(uint64_t debugConfig);
#endif // RA_PEER_H
