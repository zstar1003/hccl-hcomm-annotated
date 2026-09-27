/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_HDC_H
#define RA_HDC_H

#include <pthread.h>
#include <sys/types.h>
#include "ascend_hal.h"
#include "hccp.h"
#include "hccp_common.h"
#include "ra.h"
#include "ra_async.h"
#include "ra_rs_comm.h"

struct SocketHdcInfo {
    unsigned int phyId;
    int fd;
    void *socketHandle;
    uint32_t rsv; // consistent with socket_peer_info
};

struct HdcInfo {
    HDC_CLIENT client;
    HDC_SESSION session;
    HDC_SESSION snapshotSession;
    int startDeinit;
    int lastRecvStatus;
    pthread_mutex_t lock;
    unsigned int restoreFlag;
};

struct HdcAsyncInfo {
    pid_t hostTgid;
    HDC_SESSION session;
    HDC_SESSION snapshotSession;
    pthread_t tid;
    unsigned int connectStatus; // hdc session connect status
    unsigned int threadStatus;  // recv thread status

    int lastRecvStatus;
    pthread_mutex_t sendMutex;
    pthread_mutex_t recvMutex;

    unsigned int reqId;
    pthread_mutex_t reqMutex;
    struct RaListHead reqList;
    pthread_mutex_t rspMutex;
    struct RaListHead rspList;
    unsigned int restoreFlag;
};

struct MsgHead {
    unsigned int opcode;
    int ret;
    union {
        unsigned int rsvd;
        unsigned int asyncReqId;
    };
    unsigned int msgDataLen;
    pid_t hostTgid;
    char pidSign[PROCESS_RA_SIGN_LENGTH];
};

enum RaHdcRecvMode {
    RA_HDC_WAIT_FOREVER,
    RA_HDC_NOWAIT,
    RA_HDC_WAIT_TIMEOUT,
};

#define RA_RSVD_NUM_2 2
#define RA_RSVD_NUM_3 3
#define RA_RSVD_NUM_4 4
#define RA_RSVD_NUM_5 5
#define RA_RSVD_NUM_6 6
#define RA_RSVD_NUM_8 8
#define RA_RSVD_NUM_16 16
#define RA_RSVD_NUM_33 33
#define RA_RSVD_NUM_50 50
#define RA_RSVD_NUM_53 53
#define RA_RSVD_NUM_61 61
#define RA_RSVD_NUM_62 62
#define RA_RSVD_NUM_63 63
#define RA_RSVD_NUM_64 64
#define RA_RSVD_NUM_512 512
#define RA_RSVD_NUM_801 801

#define MAX_HDC_MSG_DATA (4096 - 16)

union OpSetPidData {
    struct {
        unsigned int phyId;
        pid_t pid;
        unsigned int rsvd[RA_RSVD_NUM_2];
        char pidSign[PROCESS_RA_SIGN_LENGTH];
        char resv[PROCESS_RA_RESV_LENGTH];
    } txData;

    struct {
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

union OpGetCqeErrInfoData {
    struct {
    } txData;

    struct {
        struct CqeErrInfo info;
    } rxData;
};

union OpGetTlsEnableData {
    struct {
        unsigned int phyId;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } txData;
    struct {
        bool tlsEnable;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

union OpGetSecRandomData {
    struct {
        unsigned int rsvd;
    } txData;
    struct {
        unsigned int value;
    } rxData;
};

union OpSetDebugConfigData {
    struct {
        u64 debugConfig;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } txData;

    struct {
        unsigned int rsvd[RA_RSVD_NUM_6];
    } rxData;
};

union OpGetHccnCfgData {
    struct {
        unsigned int phyId;
        enum HccnCfgKey key;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } txData;
    struct {
        char value[HCCN_CFG_MSG_DATA_LEN];
        unsigned int valueLen;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

union OpHdcCloseData {
    struct {
        unsigned int phyId;
        char resv[PROCESS_RA_RESV_LENGTH];
    } txData;

    struct {
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

union OpIfnumData {
    struct {
        unsigned int phyId;
        unsigned int num; /* resv bit 31 for is_all */
    } txData;

    struct {
        unsigned int num;
    } rxData;
};

union OpGetVersionData {
    struct {
        unsigned int opcode;
    } txData;

    struct {
        unsigned int version;
    } rxData;
};

struct HdcOps {
    hdcError_t (*getCapacity)(struct drvHdcCapacity *capacity);
    hdcError_t (*clientCreate)(HDC_CLIENT *client, int maxSessionNum, int serviceType, int flag);
    hdcError_t (*clientDestroy)(HDC_CLIENT client);
    hdcError_t (*sessionConnect)(int peerNode, int peerLogicid, HDC_CLIENT client, HDC_SESSION *session);
    hdcError_t (*sessionConnectEx)(int peerNode, int peerDevid, int peerPid, HDC_CLIENT client, HDC_SESSION *pSession);
    hdcError_t (*serverCreate)(int chipId, int serviceType, HDC_SERVER *server);
    hdcError_t (*serverDestroy)(HDC_SERVER server);
    hdcError_t (*sessionAccept)(HDC_SERVER server, HDC_SESSION *session);
    hdcError_t (*sessionClose)(HDC_SESSION session);
    hdcError_t (*allocMsg)(HDC_SESSION session, struct drvHdcMsg **ppMsg, int count);
    hdcError_t (*freeMsg)(struct drvHdcMsg *msg);
    hdcError_t (*reuseMsg)(struct drvHdcMsg *msg);
    hdcError_t (*addMsgBuffer)(struct drvHdcMsg *msg, char *pBuf, int len);
    hdcError_t (*getMsgBuffer)(struct drvHdcMsg *msg, int index, char **pBuf, int *pLen);
    hdcError_t (*recv)(HDC_SESSION session, struct drvHdcMsg *msg, int bufLen, unsigned long long flag,
        int *recvBufCount, unsigned int timeout);
    hdcError_t (*send)(HDC_SESSION session, struct drvHdcMsg *msg, unsigned long long flag, unsigned int timeout);
    hdcError_t (*setSessionReference)(HDC_SESSION session);
};

#define RA_PTHREAD_MUTEX_LOCK(mutex)                                                                                   \
    do {                                                                                                               \
        int ret_lock = pthread_mutex_lock(mutex);                                                                      \
        if (ret_lock) {                                                                                                \
            hccp_warn("pthread_mutex_lock unsuccessful, ret[%d]", ret_lock);                                           \
        }                                                                                                              \
    } while (0)

#define RA_PTHREAD_MUTEX_UNLOCK(mutex)                                                                                 \
    do {                                                                                                               \
        int ret_ulock = pthread_mutex_unlock(mutex);                                                                   \
        if (ret_ulock) {                                                                                               \
            hccp_warn("pthread_mutex_unlock unsuccessful, ret[%d]", ret_ulock);                                        \
        }                                                                                                              \
    } while (0)

static inline bool RaHdcIsBroken(int lastRecvStatus)
{
    return lastRecvStatus == -DRV_ERROR_SOCKET_CLOSE;
}

int RaHdcInit(struct RaInitConfig *cfg, struct ProcessRaSign pRaSign);
int RaHdcGetTlsEnable(unsigned int phyId, bool *tlsEnable);
int RaHdcDeinit(struct RaInitConfig *cfg);
int RaHdcGetInterfaceVersion(unsigned int phyId, unsigned int interfaceOpcode, unsigned int *interfaceVersion);
void RaHdcGetAllOpcodeVersion(unsigned int phyId);
bool RaHdcHasCapability(unsigned int phyId, unsigned int capability);
int RaHdcGetCqeErrInfo(unsigned int phyId, struct CqeErrInfo *info);
int RaHdcProcessMsg(unsigned int opcode, unsigned int phyId, char *data, unsigned int dataSize);
int RaHdcSetDebugConfig(unsigned int phyId, u64 debugConfig);
int RaHdcInitSession(int peerNode, int peerDevid, unsigned int phyId, int hdcType, HDC_SESSION *session);
void RaHdcDeinitSession(HDC_SESSION *session);
int RaHdcSetSessionReference(HDC_SESSION *session);
void MsgHeadBuildUp(struct MsgHead *pSendRcvHead, unsigned int opcode, unsigned int reqId, unsigned int msgDataLen,
    pid_t hostTgid);
int HdcAsyncSendPkt(struct HdcAsyncInfo *asyncInfo, unsigned int phyId, void *sendBuf, unsigned int sendLen);
int HdcAsyncRecvPkt(struct HdcAsyncInfo *asyncInfo, unsigned int phyId, void *recvBuf, unsigned int *recvLen);
int RaHdcSaveSnapshot(unsigned int phyId, enum SaveSnapshotAction action);
int RaHdcRestoreSnapshot(unsigned int phyId);
int RaHdcGetSecRandom(unsigned int phyId, unsigned int *value);
int RaHdcGetHccnCfg(unsigned int phyId, enum HccnCfgKey key, char *value, unsigned int *valueLen);
#endif // RA_HDC_H
