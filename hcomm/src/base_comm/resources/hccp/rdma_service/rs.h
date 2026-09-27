/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RS_H
#define RS_H

#include <unistd.h>
#include <stdbool.h>
#include <sys/time.h>
#include <infiniband/verbs.h>
#include <infiniband/driver.h>
#include "hccp_common.h"
#include "user_log.h"
#include "ra_rs_comm.h"

#define PROCESS_RS_SIGN_LENGTH 49
#define PROCESS_RS_RESV_LENGTH 4
#define EXP_DEVNUM 2

struct RsMrRegInfo {
    unsigned int phyId;
    char *addr;
    unsigned long long len;
    int access;
};

struct ProcessRsSign {
    int tgid;
    char sign[PROCESS_RS_SIGN_LENGTH];
    char resv[PROCESS_RS_RESV_LENGTH];
};

struct RsQpNorm {
    int flag;
    int qpMode;
    int isExp;
    int isExt;
    int memAlign; // 0,1:4KB, 2:2MB
};

struct RsQpStatusInfo {
    int status;
    unsigned int udpSport;
};

struct RsWrlistBaseInfo {
    unsigned int phyId;
    unsigned int rdevIndex;
    unsigned int qpn;
    unsigned int keyFlag;
};

struct RsLinuxVersionInfo {
    int major;
    int minor;
    int patch;
};

#if defined(HNS_ROCE_LLT) || defined(DEFINE_HNS_LLT)
#define STATIC
#else
#define STATIC static
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ++++++++++++++++++++++++++++++RDMA API for RA++++++++++++++++++++++++++++++++++ */
/*
 * rs_open
 * flag: bit0: 0 = RC, 1= UD
 */
#define MS_PER_SECOND_F 1000.0
#define US_PER_MS_F 1000.0
#define MS_PER_SECOND_I 1000
#define RS_EXPECT_TIME_MAX 200.0 // ms

#define RS_GID_SEQ_NUM 4
#define RS_GID_SEQ_ZERO 0
#define RS_GID_SEQ_ONE 1
#define RS_GID_SEQ_TWO 2
#define RS_GID_SEQ_THREE 3

#define RS_ATTRI_VISI_DEF __attribute__((visibility("default")))

RS_ATTRI_VISI_DEF int RsSetTsqpDepth(unsigned int phyId, unsigned int rdevIndex, unsigned int tempDepth,
    unsigned int *qpNum);
RS_ATTRI_VISI_DEF int RsGetTsqpDepth(unsigned int phyId, unsigned int rdevIndex, unsigned int *tempDepth,
    unsigned int *qpNum);
RS_ATTRI_VISI_DEF int RsQpCreate(unsigned int phyId, unsigned int rdevIndex, struct RsQpNorm qpNorm,
    struct RsQpResp *qpResp);
RS_ATTRI_VISI_DEF int RsQpCreateWithAttrs(unsigned int phyId, unsigned int rdevIndex, struct RsQpNormWithAttrs *qpNorm,
    struct RsQpRespWithAttrs *qpResp);
RS_ATTRI_VISI_DEF int RsQpDestroy(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn);
RS_ATTRI_VISI_DEF int RsQpDestroyWithoutCQ(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn);
RS_ATTRI_VISI_DEF int RsTypicalQpModify(unsigned int phyId, unsigned int rdevIndex, struct TypicalQp localQpInfo,
    struct TypicalQp remoteQpInfo, struct TypicalQpAttr *qpAttr);
RS_ATTRI_VISI_DEF int RsTypicalCqCreate(unsigned int phyId, unsigned int rdevIndex, unsigned int cqDepth,
    unsigned int *cqn);
RS_ATTRI_VISI_DEF int RsTypicalCqDestroy(unsigned int phyId, unsigned int rdevIndex, unsigned int cqn);
RS_ATTRI_VISI_DEF int RsQpBatchModify(unsigned int phyId, unsigned int rdevIndex, int status, int qpn[], int qpnNum);
RS_ATTRI_VISI_DEF int RsSetQpLbValue(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, int lbValue);
RS_ATTRI_VISI_DEF int RsGetQpLbValue(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, int *lbValue);
RS_ATTRI_VISI_DEF int RsQpConnectAsync(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, int fd);

enum RsQpStatus {
    RS_QP_STATUS_DISCONNECT = 0,
    RS_QP_STATUS_CONNECTED = 1,
    RS_QP_STATUS_TIMEOUT = 2,
    RS_QP_STATUS_CONNECTING = 3,
    RS_QP_STATUS_REM_FD_CLOSE = 4,
    RS_QP_STATUS_PAUSE = 5,
};

#define RS_IS_EXP 0
#define RS_NOT_EXP 1

enum RsAccessFlags {
    RS_ACCESS_LOCAL_WRITE = 1,
    RS_ACCESS_REMOTE_WRITE = (1 << 1),
    RS_ACCESS_REMOTE_READ = (1 << 2UL),
    RS_ACCESS_REDUCE = (1 << 8),
};

struct RsInitConfig {
    unsigned int chipId;
    unsigned int hccpMode;
    unsigned int whiteListStatus;
};

struct RsQpConnPara {
    unsigned int phyId;
    unsigned int rdevIndex;
    uint32_t qpn;
};

struct RsBackupInfo {
    bool backupFlag;
    struct rdev rdevInfo;
};

#define RS_HEARTBEAT_TIME (1000.0 * 60) // ms
#define PTHREAD_NAME_LEN 32

struct RsPthreadInfo {
    char pthreadName[PTHREAD_NAME_LEN];
    struct timeval lastCheckTime;
};

RS_ATTRI_VISI_DEF int RsMrReg(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn,
    struct RdmaMrRegInfo *mrRegInfo);
RS_ATTRI_VISI_DEF int RsMrDereg(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, char *addr);

RS_ATTRI_VISI_DEF int RsRegisterMr(unsigned int phyId, unsigned int rdevIndex, struct RdmaMrRegInfo *mrRegInfo,
    void **mrHandle);
RS_ATTRI_VISI_DEF int RsTypicalRegisterMrV1(unsigned int phyId, unsigned int rdevIndex, struct RdmaMrRegInfo *mrRegInfo,
    void **mrHandle);
RS_ATTRI_VISI_DEF int RsTypicalRegisterMr(unsigned int phyId, unsigned int rdevIndex, struct RdmaMrRegInfo *mrRegInfo,
    void **mrHandle);
RS_ATTRI_VISI_DEF int RsRemapMr(unsigned int phyId, unsigned int rdevIndex, struct MemRemapInfo memList[],
    unsigned int memNum);
RS_ATTRI_VISI_DEF int RsTypicalDeregisterMr(unsigned int phyId, unsigned int devIndex, unsigned long long addr);
RS_ATTRI_VISI_DEF int RsDeregisterMr(unsigned int phyId, unsigned int rdevIndex, void *mrHandle);

enum RsSendFlags {
    RS_SEND_FENCE = 1 << 0,
    RS_SEND_SIGNALED = 1 << 1,
    RS_SEND_SOLICITED = 1 << 2,
    RS_SEND_INLINE = 1 << 3,
};
RS_ATTRI_VISI_DEF int RsSendWr(unsigned int phyId, unsigned int rdevIndex, uint32_t qpn, struct SendWr *wr,
    struct SendWrRsp *wrRsp);
RS_ATTRI_VISI_DEF int RsSendWrlist(struct RsWrlistBaseInfo baseInfo, struct WrInfo *wrList, unsigned int sendNum,
    struct SendWrRsp *wrRsp, unsigned int *completeNum);

RS_ATTRI_VISI_DEF int RsRecvWrlist(struct RsWrlistBaseInfo baseInfo, struct RecvWrlistData *wr, unsigned int recvNum,
    unsigned int *completeNum);

RS_ATTRI_VISI_DEF int RsGetNotifyMrInfo(unsigned int phyId, unsigned int rdevIndex, struct MrInfoT *info);
RS_ATTRI_VISI_DEF int RsSetHostPid(uint32_t phyId, pid_t hostPid, const char *pidSign);

RS_ATTRI_VISI_DEF int RsInit(struct RsInitConfig *cfg);
RS_ATTRI_VISI_DEF int RsGetTlsEnable(unsigned int phyId, bool *tlsEnable);
RS_ATTRI_VISI_DEF int RsGetHccnCfg(struct RaInfo *info, enum HccnCfgKey key, char *value, unsigned int *valueLen);
RS_ATTRI_VISI_DEF int RsBindHostpid(unsigned int chipId, pid_t pid);
RS_ATTRI_VISI_DEF int RsDeinit(struct RsInitConfig *cfg);

RS_ATTRI_VISI_DEF int RsSocketInit(const unsigned int *vnicIp, unsigned int num);
RS_ATTRI_VISI_DEF int RsSocketDeinit(struct rdev rdevInfo);

RS_ATTRI_VISI_DEF int RsRdevInit(struct rdev rdevInfo, unsigned int notifyType, unsigned int *rdevIndex);
RS_ATTRI_VISI_DEF int RsRdevInitWithBackup(struct rdev rdevInfo, struct rdev backupRdevInfo, unsigned int notifyType,
    unsigned int *rdevIndex);
RS_ATTRI_VISI_DEF int RsRdevGetPortStatus(unsigned int phyId, unsigned int rdevIndex, enum PortStatus *status);
RS_ATTRI_VISI_DEF int RsGetLbMax(unsigned int phyId, unsigned int rdevIndex, int *lbMax);
RS_ATTRI_VISI_DEF int RsRdevDeinit(unsigned int phyId, unsigned int notifyType, unsigned int rdevIndex);

/* ++++++++++++++++++++++++++++++Epoll API start++++++++++++++++++++++++++++++++++ */
RS_ATTRI_VISI_DEF int RsEpollCtlAdd(const void *fdHandle, enum RaEpollEvent event);
RS_ATTRI_VISI_DEF int RsEpollCtlMod(const void *fdHandle, enum RaEpollEvent event);
RS_ATTRI_VISI_DEF int RsEpollCtlDel(int fd);
RS_ATTRI_VISI_DEF void RsSetTcpRecvCallback(const void *callback);
RS_ATTRI_VISI_DEF int RsCreateEventHandle(int *eventHandle);
RS_ATTRI_VISI_DEF int RsCtlEventHandle(int eventHandle, const void *fdHandle, int opcode, enum RaEpollEvent event);
RS_ATTRI_VISI_DEF int RsWaitEventHandle(int eventHandle, struct SocketEventInfoT *eventInfos, int timeout,
    unsigned int maxevents, unsigned int *eventsNum);
RS_ATTRI_VISI_DEF int RsDestroyEventHandle(int *eventHandle);
/* ++++++++++++++++++++++++++++++Epoll API end++++++++++++++++++++++++++++++++++ */

/* ++++++++++++++++++++++++++++++Socket API++++++++++++++++++++++++++++++++++ */
#define RS_SOCK_PORT_DEF 16666
RS_ATTRI_VISI_DEF int RsSocketListenStart(struct SocketListenInfo conn[], uint32_t num);
RS_ATTRI_VISI_DEF int RsSocketListenStop(struct SocketListenInfo conn[], uint32_t num);

RS_ATTRI_VISI_DEF int RsSocketBatchConnect(struct SocketConnectInfo conn[], uint32_t num);
RS_ATTRI_VISI_DEF int RsSocketBatchAbort(struct SocketConnectInfo conn[], uint32_t num);

RS_ATTRI_VISI_DEF int RsSocketGetClientSocketErrInfo(struct SocketConnectInfo conn[], struct SocketErrInfo err[],
    unsigned int num);
RS_ATTRI_VISI_DEF int RsSocketGetServerSocketErrInfo(struct SocketListenInfo conn[], struct ServerSocketErrInfo err[],
    unsigned int num);

RS_ATTRI_VISI_DEF void RsGetCurTime(struct timeval *time);
RS_ATTRI_VISI_DEF void HccpTimeInterval(struct timeval *endTime, struct timeval *startTime, float *msec);
RS_ATTRI_VISI_DEF int RsSocketWhiteListSwitch(unsigned int phyId, unsigned int enable);
RS_ATTRI_VISI_DEF int RsSocketWhiteListAdd(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], unsigned int num);
RS_ATTRI_VISI_DEF int RsSocketWhiteListDel(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], unsigned int num);
RS_ATTRI_VISI_DEF int RsSocketAcceptCreditAdd(struct SocketListenInfo conn[], uint32_t num, unsigned int creditLimit);
RS_ATTRI_VISI_DEF int RsGetIfnum(unsigned int phyId, bool isAll, unsigned int *num);
RS_ATTRI_VISI_DEF int RsPeerGetIfnum(unsigned int phyId, unsigned int *num);
RS_ATTRI_VISI_DEF int RsGetIfaddrs(struct IfaddrInfo ifaddrInfos[], unsigned int *num, unsigned int phyId);
RS_ATTRI_VISI_DEF int RsGetIfaddrsV2(struct InterfaceInfo interfaceInfos[], unsigned int *num, unsigned int phyId,
    bool isAll);
RS_ATTRI_VISI_DEF int RsPeerGetIfaddrs(struct InterfaceInfo interfaceInfos[], unsigned int *num, unsigned int phyId);
RS_ATTRI_VISI_DEF int RsGetVnicIp(unsigned int phyId, unsigned int *vnicIp);
RS_ATTRI_VISI_DEF int RsGetInterfaceVersion(unsigned int opcode, unsigned int *version);
RS_ATTRI_VISI_DEF int RsGetVnicIpInfos(unsigned int phyId, enum IdType type, unsigned int ids[], unsigned int num,
    struct IpInfo infos[]);
RS_ATTRI_VISI_DEF int RsSocketSetScopeId(unsigned int devId, int scopeId);
struct RsSocketCloseInfoT {
    int fd;
};
RS_ATTRI_VISI_DEF int RsSocketBatchClose(int disuseLinger, struct RsSocketCloseInfoT conn[], uint32_t num);

enum RsConnRole {
    RS_CONN_ROLE_SERVER = 0,
    RS_CONN_ROLE_CLIENT = 1,
};

enum ProductType {
    PRODUCT_TYPE_INVALID = -2,
    PRODUCT_TYPE_NO_VALUE = -1,
    PRODUCT_TYPE_310p = 0,
    PRODUCT_TYPE_910,
    PRODUCT_TYPE_910B,
    PRODUCT_TYPE_910_93,
    PRODUCT_TYPE_950,
    PRODUCT_TYPE_960,
    PRODUCT_TYPE_350,
    PRODUCT_TYPE_OTHERS,
};

enum RsSocketStatus {
    RS_SOCK_STATUS_NA = 0,
    RS_SOCK_STATUS_OK = 1,
    RS_SOCK_STATUS_TIMEOUT = 2,
    RS_SOCK_STATUS_ING = 3,
};

RS_ATTRI_VISI_DEF int RsGetSockets(uint32_t role, struct SocketFdData conn[], uint32_t num);
RS_ATTRI_VISI_DEF int RsGetSslEnable(uint32_t *sslEnable);
RS_ATTRI_VISI_DEF int RsSocketSend(int fd, const void *data, uint64_t size);
RS_ATTRI_VISI_DEF int RsHdcSocketSend(int fd, const void *data, uint64_t size);
RS_ATTRI_VISI_DEF int RsPeerSocketSend(uint32_t sslEnable, int fd, const void *data, uint64_t size);
RS_ATTRI_VISI_DEF int RsSocketRecv(int fd, void *data, uint64_t size);
RS_ATTRI_VISI_DEF int RsPeerSocketRecv(uint32_t sslEnable, int fd, void *data, uint64_t size);
RS_ATTRI_VISI_DEF int RsNotifyCfgSet(unsigned int phyId, unsigned long long va, unsigned long long size);
RS_ATTRI_VISI_DEF int RsNotifyCfgGet(unsigned int phyId, unsigned long long *va, unsigned long long *size);
RS_ATTRI_VISI_DEF void RsHeartbeatAlivePrint(struct RsPthreadInfo *pthreadInfo);

RS_ATTRI_VISI_DEF int RsGetQpContext(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, void **qp,
    void **sendCq, void **recvCq);
RS_ATTRI_VISI_DEF int RsGetQpStatus(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn,
    struct RsQpStatusInfo *qpInfo);

int rsGetLocalDevIDByHostDevID(unsigned int phyId, unsigned int *chipId);
int rsGetDevIDByLocalDevID(unsigned int chipId, unsigned int *phyId);

RS_ATTRI_VISI_DEF int RsCqCreate(unsigned int phyId, unsigned int rdevIndex, struct CqAttr *attr);
RS_ATTRI_VISI_DEF int RsCqDestroy(unsigned int phyId, unsigned int rdevIndex, struct CqAttr *attr);
RS_ATTRI_VISI_DEF int RsNormalQpCreate(unsigned int phyId, unsigned int rdevIndex, struct ibv_qp_init_attr *qpInitAttr,
    struct RsQpResp *qpResp, void **qp);
RS_ATTRI_VISI_DEF int RsNormalQpDestroy(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn);
RS_ATTRI_VISI_DEF int RsSetQpAttrQos(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn,
    struct QosAttr *attr);
RS_ATTRI_VISI_DEF int RsSetQpAttrTimeout(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn,
    unsigned int *timeout);
RS_ATTRI_VISI_DEF int RsSetQpAttrRetryCnt(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn,
    unsigned int *retryCnt);
RS_ATTRI_VISI_DEF int RsCreateCompChannel(unsigned int phyId, unsigned int rdevIndex, void **compChannel);
RS_ATTRI_VISI_DEF int RsDestroyCompChannel(void *compChannel);
RS_ATTRI_VISI_DEF int RsGetCqeErrInfo(struct CqeErrInfo *info);
RS_ATTRI_VISI_DEF int RsCreateSrq(unsigned int phyId, unsigned int rdevIndex, struct SrqAttr *attr);
RS_ATTRI_VISI_DEF int RsDestroySrq(unsigned int phyId, unsigned int rdevIndex, struct SrqAttr *attr);
RS_ATTRI_VISI_DEF int RsGetLiteSupport(unsigned int phyId, unsigned int rdevIndex, int *supportLite);
RS_ATTRI_VISI_DEF int RsGetLiteRdevCap(unsigned int phyId, unsigned int rdevIndex, struct LiteRdevCapResp *resp);
RS_ATTRI_VISI_DEF int RsGetLiteQpCqAttr(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn,
    struct LiteQpCqAttrResp *resp);
RS_ATTRI_VISI_DEF int RsGetLiteQpAttr(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn,
    struct LiteQpAttrResp *resp);
RS_ATTRI_VISI_DEF int RsGetLiteCqAttr(unsigned int phyId, unsigned int rdevIndex, unsigned int cqn,
    struct rdma_lite_device_cq_attr *deviceCqAttr);
RS_ATTRI_VISI_DEF int RsQpCreateWithCQWithAttrs(unsigned int phyId, unsigned int rdevIndex, unsigned int sendCqn,
    unsigned int recvCqn, struct RsQpNormWithAttrs *qpNorm, struct RsQpRespWithAttrs *qpResp);
RS_ATTRI_VISI_DEF int RsGetLiteConnectedInfo(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn,
    struct LiteConnectedInfoResp *resp);
RS_ATTRI_VISI_DEF int RsGetLiteMemAttr(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn,
    struct LiteMemAttrResp *resp);
RS_ATTRI_VISI_DEF void RsSetCtx(unsigned int phyId);
RS_ATTRI_VISI_DEF void RsSetDebugConfig(uint64_t debugConfig);
RS_ATTRI_VISI_DEF int RsGetCqeErrInfoNum(unsigned int phyId, unsigned int rdevIdx, unsigned int *num);
RS_ATTRI_VISI_DEF int RsGetCqeErrInfoList(unsigned int phyId, unsigned int rdevIdx, struct CqeErrInfo *info,
    unsigned int *num);
RS_ATTRI_VISI_DEF int RsDrvGetRandomNum(int *randNum);
RS_ATTRI_VISI_DEF int RsGetSecRandom(unsigned int *value);
// note: The FIRST invocation of this function MAY throw exceptions for each process
RS_ATTRI_VISI_DEF enum ProductType RsGetProductType(int devId);

static inline bool RsIsTlvSupported(void)
{
    enum ProductType productType;
    productType = RsGetProductType(0); // Ensure that RsGetProductType has been called at least once
    return (productType == PRODUCT_TYPE_910B || productType == PRODUCT_TYPE_910_93 || productType == PRODUCT_TYPE_950 ||
            productType == PRODUCT_TYPE_960 || productType == PRODUCT_TYPE_350);
}

static inline bool RsIsRdmaSupported(void)
{
    enum ProductType productType;
    productType = RsGetProductType(0);
    return (productType == PRODUCT_TYPE_910B || productType == PRODUCT_TYPE_910_93 || productType == PRODUCT_TYPE_910);
}

static inline bool RsIsUdmaSupported(void)
{
    enum ProductType productType;
    productType = RsGetProductType(0);
    return (productType == PRODUCT_TYPE_950 || productType == PRODUCT_TYPE_960 || productType == PRODUCT_TYPE_350);
}

static inline bool RsIsCustomInterfaceSupported(void)
{
    enum ProductType productType;
    productType = RsGetProductType(0);
    return (productType != PRODUCT_TYPE_310p);
}
#ifdef __cplusplus
}
#endif
#endif // RS_H
