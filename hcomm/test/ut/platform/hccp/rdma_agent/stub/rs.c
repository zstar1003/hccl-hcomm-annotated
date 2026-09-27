/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdlib.h>
#include <sys/time.h>
#include <sys/epoll.h>
#include <errno.h>
#include "hccp_common.h"
#include "ra_rs_comm.h"
#include "ra_comm.h"

typedef uint32_t u32;
typedef uint16_t u16;
typedef unsigned long long u64;
typedef signed int s32;

struct RsInitConfig {
    u32 port;
    u32 deviceId;
    u32 rsPosition;
};

struct RsSocketListenInfoT {
    uint32_t deviceId;
    uint32_t ipAddr;
    uint32_t phase;
    uint32_t err;
};

struct RsSocketConnectInfoT {
    uint32_t deviceId;
    uint32_t ipAddr;
    char tag[128];
};

struct RsSocketCloseInfoT {
    int fd;
};

struct RsSocketInfoT {
    int fd;
    u32 deviceId;
    u32 serverIpAddr;
    u32 clientIpAddr;
    s32 status;
    char tag[128];
};

struct RsQpNorm {
    int flag;
    int qpMode;
    int isExp;
    int isExt;
    int memAlign;
};

struct RsWrlistBaseInfo {
    unsigned int devId;
    unsigned int rdevIndex;
    unsigned int qpn;
    unsigned int keyFlag;
};

#define MS_PER_SECOND_F 1000.0
#define MS_PER_SECOND_I 1000

void RsGetCurTime(struct timeval* time)
{
    int ret;

    ret = gettimeofday(time, NULL);
    if (ret) {
        memset(time, 0, sizeof(struct timeval));
    }

    return;
}

void HccpTimeInterval(struct timeval* endTime, struct timeval* startTime, float* msec)
{
    /* if low position is sufficient, then borrow one from the high position */
    if (endTime->tv_usec < startTime->tv_usec) {
        endTime->tv_sec -= 1;
        endTime->tv_usec += MS_PER_SECOND_I * MS_PER_SECOND_I;
    }

    *msec = (endTime->tv_sec - startTime->tv_sec) * MS_PER_SECOND_F
            + (endTime->tv_usec - startTime->tv_usec) / MS_PER_SECOND_F;

    return;
}

int RsSocketBatchConnect(struct SocketConnectInfo conn[], uint32_t num) { return 0; }

int RsSocketSetScopeId(unsigned int devId, int scopeId) { return 0; }

int RsSocketBatchClose(int disuseLinger, struct RsSocketCloseInfoT conn[], u32 num) { return 0; }

int RsSocketBatchAbort(int disuseLinger, struct RsSocketCloseInfoT conn[], u32 num) { return 0; }

int RsSocketListenStart(struct SocketListenInfo listen[], uint32_t num) { return 0; }

int RsSocketListenStop(struct SocketListenInfo listen[], uint32_t num) { return 0; }

static int fd = 0;
static int devId = 0;
int RsGetSockets(uint32_t role, struct SocketFdData conn[], uint32_t num)
{
    int i;

    for (i = 0; i < num; i++) {
        conn[i].phyId = devId;
        conn[i].fd = fd;
    }
    return num;
}

int RsGetSslEnable(uint32_t* sslEnable) { return 0; }

int RsPeerSocketSend(uint32_t sslEnable, int fd, const void* data, uint64_t size) { return size; }

int RsPeerSocketRecv(uint32_t sslEnable, int fd, void* data, uint64_t size) { return size; }

int RsGetSocketNum(unsigned int role, struct rdev rdevInfo, unsigned int* socketNum) { return 0; }

int RsGetAllSockets(unsigned int phyId, uint32_t role, struct SocketFdData* conn, uint32_t* socketNum)
{
    conn->fd = 0;
    conn->localIp.addr.s_addr = 0;
    conn->remoteIp.addr.s_addr = 0;
    conn->phyId = 0;
    conn->tag[0] = '0';
    return 0;
}

int RsSocketSend(int fd, void* data, u64 size) { return size; }

int RsHdcSocketSend(int fd, const void* data, uint64_t size) { return size; }

int RsSocketRecv(int fd, void* data, u64 size) { return size; }

int RsQpCreate(unsigned int phyId, unsigned int rdevIndex, struct RsQpNorm qpNorm, struct RsQpResp* qpResp)
{
    qpResp->qpn = 0;
    qpResp->psn = 0;
    qpResp->gidIdx = 0;
    qpResp->gid.global.subnet_prefix = 0;
    qpResp->gid.global.interface_id = 0;

    return 0;
}

int RsQpCreateWithAttrs(
    unsigned int phyId, unsigned int rdevIndex, struct RsQpNormWithAttrs* qpNorm, struct RsQpRespWithAttrs* qpResp)
{
    qpResp->qpn = 0;
    qpResp->aiQpAddr = 0;
    return 0;
}

int RsQpDestroy(unsigned int devId, unsigned int rdevIndex, unsigned int qpn) { return 0; }

int RsTypicalQpModify(
    unsigned int phyId, unsigned int rdevIndex, struct TypicalQp localQpInfo, struct TypicalQp remoteQpInfo,
    struct TypicalQpAttr* qpAttr)
{
    return 0;
}

static struct ibv_mr stubMr = {0};

int RsTypicalRegisterMrV1(unsigned int phyId, unsigned int rdevIndex, struct RdmaMrRegInfo* mrRegInfo, void** mrHandle)
{
    *mrHandle = &stubMr;
    return 0;
}

int RsTypicalRegisterMr(unsigned int phyId, unsigned int rdevIndex, struct RdmaMrRegInfo* mrRegInfo, void** mrHandle)
{
    *mrHandle = &stubMr;
    return 0;
}

int RsTypicalDeregisterMr(unsigned int phyId, unsigned int devIndex, unsigned long long addr) { return 0; }

int RsQpConnectAsync(unsigned int devId, unsigned int rdevIndex, unsigned int qpn, int fd) { return 0; }

int RsGetQpStatus(unsigned int devId, unsigned int rdevIndex, unsigned int qpn, int* status) { return 0; }

int RsMrReg(unsigned int devId, unsigned int rdevIndex, unsigned int qpn, struct RdmaMrRegInfo* mrRegInfo) { return 0; }

int RsMrDereg(unsigned int devId, unsigned int rdevIndex, unsigned int qpn, char* addr) { return 0; }

int RsRegisterMr(unsigned int devId, unsigned int rdevIndex, struct RdmaMrRegInfo* mrRegInfo) { return 0; }

int RsDeregisterMr(unsigned int devId, unsigned int rdevIndex, char* addr) { return 0; }

int RsSendWr(unsigned int devId, unsigned int rdevIndex, uint32_t qpn, struct SendWr* wr, struct SendWrRsp* wrRsp)
{
    return 0;
}

int RsSendWrlist(
    struct RsWrlistBaseInfo baseInfo, struct WrInfo* wr, unsigned int sendNum, struct SendWrRsp* wrRsp,
    unsigned int* completeNum)
{
    return 0;
}

int RsGetNotifyMrInfo(unsigned int phyId, unsigned int rdevIndex, struct mrInfo* info) { return 0; }

int RsSetHostPid(uint32_t devId, uint32_t hostPid) { return 0; }

int RsRdevGetPortStatus(unsigned int phyId, unsigned int rdevIndex, enum PortStatus* status) { return 0; }

int RsInit(struct RaInitConfig* cfg) { return 0; }

int RsBindHostpid(unsigned int chipId, pid_t pid) { return 0; }

int RsDeinit(struct RaInitConfig* cfg) { return 0; }

int RsSocketInit(const unsigned int* vnicIp, unsigned int num) { return 0; }

int RsSocketDeinit(struct rdev rdevInfo) { return 0; }

int RsRdevInit(struct rdev rdevInfo, unsigned int notifyType, unsigned int* rdevIndex) { return 0; }

int RsRdevInitWithBackup(
    struct rdev rdevInfo, struct rdev backupRdevInfo, unsigned int notifyType, unsigned int* rdevIndex)
{
    return 0;
}

int RsRdevDeinit(unsigned int devId, unsigned int notifyType, unsigned int rdevIndex) { return 0; }

int RsSocketWhiteListAdd(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], u32 num) { return 0; }

int RsSocketWhiteListDel(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], u32 num) { return 0; }

int RsPeerGetIfaddrs(struct InterfaceInfo interfaceInfos[], unsigned int* num, unsigned int phyId) { return 0; }

int RsGetIfaddrs(struct IfaddrInfo ifaddrInfos[], unsigned int* num, unsigned int devId) { return 0; }

int RsGetIfaddrsV2(struct InterfaceInfo interfaceInfos[], unsigned int* num, unsigned int phyId, bool isAll)
{
    return 0;
}

int RsGetIfnum(unsigned int phyId, bool isAll, unsigned int* num) { return 0; }

int RsPeerGetIfnum(unsigned int phyId, unsigned int* num) { return 0; }

int RsGetVnicIp(unsigned int phyId, unsigned int* vnicIp) { return 0; }

int RsGetInterfaceVersion(unsigned int opcode, unsigned int* version)
{
    *version = 1;
    return 0;
}

int RsNotifyCfgSet(unsigned int devId, unsigned long long va, unsigned long long size) { return 0; }

int RsNotifyCfgGet(unsigned int devId, unsigned long long* va, unsigned long long* size) { return 0; }

int RsSetTsqpDepth(unsigned int phyId, unsigned int rdevIndex, unsigned int tempDepth, unsigned int* qpNum)
{
    return 0;
}

int RsGetTsqpDepth(unsigned int phyId, unsigned int rdevIndex, unsigned int* tempDepth, unsigned int* qpNum)
{
    return 0;
}

void RsHeartbeatAlivePrint(struct RaHdcAsyncInfo* pthreadInfo) { return; }

int RsRecvWrlist(
    struct RsWrlistBaseInfo baseInfo, struct RecvWrlistData* wr, unsigned int recvNum, unsigned int* completeNum)
{
    *completeNum = recvNum;
    return 0;
}

int RsEpollCtlAdd(const void* fdHandle, enum RaEpollEvent event) { return 0; }

int RsEpollCtlMod(const void* fdHandle, enum RaEpollEvent event) { return 0; }

int RsEpollCtlDel(int fd) { return 0; }

void RsSetTcpRecvCallback(const void* callback) { return; }

int RsGetQpContext(
    unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, void** qp, void** sendCq, void** recvCq)
{
    return 0;
}

int RsCqCreate(unsigned int phyId, unsigned int rdevIndex, struct CqAttr* attr) { return 0; }

int RsCqDestroy(unsigned int phyId, unsigned int rdevIndex, struct CqAttr* attr) { return 0; }

int RsNormalQpCreate(
    unsigned int phyId, unsigned int rdevIndex, struct ibv_qp_init_attr* qpInitAttr, struct RsQpResp* qpResp, void** qp)
{
    return 0;
}

int RsNormalQpDestroy(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn) { return 0; }

int RsSetQpAttrQos(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, struct QosAttr* attr) { return 0; }

int RsSetQpAttrTimeout(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, unsigned int* timeout)
{
    return 0;
}

int RsSetQpAttrRetryCnt(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, unsigned int* retryCnt)
{
    return 0;
}

int RsGetCqeErrInfo(struct CqeErrInfo* info) { return 0; }

int RsGetCqeErrInfoNum(unsigned int phyId, unsigned int rdevIdx, unsigned int* num) { return 0; }

int RsGetCqeErrInfoList(unsigned int phyId, unsigned int rdevIdx, struct CqeErrInfo* info, unsigned int* num)
{
    return 0;
}

int TlsGetUserConfig(
    unsigned int saveMode, unsigned int chipId, const char* name, unsigned char* buf, unsigned int* bufSize)
{
    return 0;
}

void TlsGetEnableInfo(unsigned int saveMode, unsigned int chipId, unsigned char* buf, unsigned int bufSize) { return; }

struct KmcEncInfo {
    unsigned char* cipherText;
    unsigned int cipherTextLen;
    unsigned char* workKey;
    unsigned int workKeyLen;
};

int EncWithSdp(
    unsigned int encAlg, const unsigned char* inbuf, unsigned int sizeIn, unsigned char* cpr, unsigned int* retLen)
{
    return 0;
}

int DecWithKmc(unsigned char* cpr, unsigned int cprLen, unsigned char* outbuf, unsigned int* sizeOut) { return 0; }

int RsCreateCompChannel(unsigned int phyId, unsigned int rdevIndex, void** compChannel) { return 0; }

int RsDestroyCompChannel(void* compChannel) { return 0; }

int RsCreateSrq(unsigned int phyId, unsigned int rdevIndex, struct SrqAttr* attr) { return 0; }

int RsDestroySrq(unsigned int phyId, unsigned int rdevIndex, struct SrqAttr* attr) { return 0; }

int RsGetLiteSupport(unsigned int phyId, unsigned int rdevIndex, int* supportLite) { return 0; }

int RsGetLiteRdevCap(unsigned int phyId, unsigned int rdevIndex, struct LiteRdevCapResp* resp) { return 0; }

int RsGetLiteQpCqAttr(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, struct LiteQpCqAttrResp* resp)
{
    return 0;
}

int RsGetLiteConnectedInfo(
    unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, struct LiteConnectedInfoResp* resp)
{
    return 0;
}

int RsGetLiteMemAttr(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, struct LiteMemAttrResp* resp)
{
    return 0;
}

void RsSetCtx(unsigned int phyId) { return; }

void RsSetDebugConfig(uint64_t debugConfig) { return; }

int RsCreateEventHandle(int* eventHandle) { return 0; }

int RsCtlEventHandle(int eventHandle, const void* fdHandle, int opcode, enum RaEpollEvent event)
{
    int ret;
    int fd = -1;
    int tmpEvent;

    if (eventHandle < 0) {
        hccp_err("[RsCtlEventHandle]event_handle[%d] is invalid", eventHandle);
        return -EINVAL;
    }
    if (fdHandle == NULL) {
        hccp_err("[RsCtlEventHandle]fd_handle is NULL");
        return -EINVAL;
    }
    if (opcode != EPOLL_CTL_ADD && opcode != EPOLL_CTL_DEL && opcode != EPOLL_CTL_MOD) {
        hccp_err(
            "[RsCtlEventHandle]opcode[%d] invalid, valid opcode includes {%d, %d, %d}", opcode, EPOLL_CTL_ADD,
            EPOLL_CTL_DEL, EPOLL_CTL_MOD);
        return -EINVAL;
    }
    if (opcode == EPOLL_CTL_DEL && event != RA_EPOLLIN) {
        hccp_err("[RsCtlEventHandle]param invalid: opcode[%d], event[%d]", opcode, event);
        return -EINVAL;
    }

    if (event == RA_EPOLLONESHOT) {
        tmpEvent = EPOLLIN | EPOLLOUT | EPOLLET | EPOLLONESHOT;
    } else if (event == RA_EPOLLIN) {
        tmpEvent = EPOLLIN;
    } else {
        hccp_err("[RsCtlEventHandle]unkown event[%d]", event);
        return -EINVAL;
    }

    return 0;
}

int RsWaitEventHandle(int eventHandle, struct SocketEventInfoT* eventInfo, int timeout) { return 0; }

int RsDestroyEventHandle(int* eventHandle) { return 0; }

int RsGetVnicIpInfos(unsigned int phyId, enum IdType type, unsigned int ids[], unsigned int num, struct IpInfo infos[])
{
    return 0;
}

int RsQpBatchModify(unsigned int phyId, unsigned int rdevIndex, int status, int qpn[], int qpnNum) { return 0; }

int RsSocketGetClientSocketErrInfo(struct SocketConnectInfo conn[], struct SocketErrInfo err[], unsigned int num)
{
    return 0;
}

int RsSocketGetServerSocketErrInfo(struct SocketListenInfo conn[], struct ServerSocketErrInfo err[], unsigned int num)
{
    return 0;
}

int RsSocketAcceptCreditAdd(struct SocketListenInfo conn[], uint32_t num, unsigned int creditLimit) { return 0; }

int RsRemapMr(unsigned int phyId, unsigned int rdevIndex, struct MemRemapInfo memList[], unsigned int memNum)
{
    return 0;
}

int RsGetTlsEnable(unsigned int phyId, bool* tlsEnable) { return 0; }

int RsGetSecRandom(unsigned int* value) { return 0; }

int RsDrvGetRandomNum(int* randNum) { return 0; }

int RsGetHccnCfg(struct RaInfo* info, enum HccnCfgKey key, char* value, unsigned int* valueLen) { return 0; }

int RsSetQpLbValue(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, int lbValue) { return 0; }

int RsGetQpLbValue(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, int* lbValue) { return 0; }

int RsGetLbMax(unsigned int phyId, unsigned int rdevIndex, int* lbMax) { return 0; }

int RsQpCreateWithCQWithAttrs(
    unsigned int phyId, unsigned int rdevIndex, unsigned int sendCqn, unsigned int recvCqn,
    struct RsQpNormWithAttrs* qpNorm, struct RsQpRespWithAttrs* qpResp)
{
    qpResp->qpn = 0;
    qpResp->aiQpAddr = 0;
    return 0;
}

int RsQpDestroyWithoutCQ(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn) { return 0; }

int RsTypicalCqCreate(unsigned int phyId, unsigned int rdevIndex, unsigned int cqDepth, unsigned int* cqn)
{
    *cqn = 0;
    return 0;
}

int RsTypicalCqDestroy(unsigned int phyId, unsigned int rdevIndex, unsigned int cqn) { return 0; }

int RsGetLiteCqAttr(
    unsigned int phyId, unsigned int rdevIndex, unsigned int cqn, struct rdma_lite_device_cq_attr* deviceCqAttr)
{
    return 0;
}

int RsGetLiteQpAttr(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, struct LiteQpAttrResp* resp)
{
    return 0;
}
