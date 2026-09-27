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
#define SOCK_CONN_TAG_SIZE 192
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <netdb.h>
#include <net/if.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <ifaddrs.h>
#include <fcntl.h>
#include <stdint.h>
#include "ascend_hal.h"
#include "securec.h"
#include "dl_hal_function.h"
#include "dl_ibverbs_function.h"
#include "dl_net_function.h"
#include "rs_socket.h"
#include "rs_tls.h"
#include "ut_dispatch.h"
#include "stub/ibverbs.h"
#include "ibv_extend.h"
#include "rs.h"
#include "rs_nda.h"
#include "rs_common_inner.h"
#include "rs_inner.h"
#include "rs_ping_inner.h"
#include "rs_ping.h"
#include "ra_rs_err.h"
#include "file_opt.h"
#include "rs_drv_rdma.h"
#include "rs_ub_tp.h"
#include "rs_ub_dfx.h"
#include "rs_ub.h"
#include "stub/verbs_exp.h"
#include "encrypt.h"
#include "rs_epoll.h"
#include "tc_ut_rs.h"

extern void RsGetCurTime(struct timeval* time);
extern int memset_s(void* dest, size_t destMax, int c, size_t count);
extern int RsPthreadMutexInit(struct rs_cb* rscb, struct RsInitConfig* cfg);
extern void RsSslRecvTagInHandle(struct RsAcceptInfo* acceptInfo, struct RsConnInfo* connTmp);
extern void RsEpollEventSslRecvTagInHandle(struct rs_cb* rsCb, struct RsAcceptInfo* acceptInfo);
extern int RsRdev2rdevCb(unsigned int chipId, unsigned int rdevIndex, struct RsRdevCb** rdevCb);
extern int RsCompareIpGid(struct rdev rdevInfo, union ibv_gid* gid);
extern int RsGetIbCtxAndRdevIndex(struct rdev rdevInfo, struct RsRdevCb* rdevCb, unsigned int* rdevIndex);
extern int RsQueryGid(struct rdev rdevInfo, struct ibv_context* ibCtxTmp, uint8_t ibPort, int* gidIdx);
extern int RsRdevCbInit(struct rdev rdevInfo, struct RsRdevCb* rdevCb, struct rs_cb* rsCb, unsigned int* rdevIndex);
extern int RsDrvQueryNotifyAndAllocPd(struct RsRdevCb* rdevCb);
extern int RsDrvRegNotifyMr(struct RsRdevCb* rdevCb);
extern int RsGetRsCb(unsigned int phyId, struct rs_cb** rsCb);
extern int RsDrvMrDereg(struct ibv_mr* ibMr);
extern void RsSslFree(struct rs_cb* rscb);
extern int
RsDrvPostRecv(struct RsQpCb* qpCb, struct RecvWrlistData* wr, unsigned int recvNum, unsigned int* completeNum);
extern int RsDrvSslBindFd(struct RsConnInfo* conn, int fd);
extern void rs_ssl_err_string(int fd, int err);
extern void rs_ssl_deinit(struct rs_cb* rscb);
extern int rs_ssl_init(struct rs_cb* rscb);
extern int rs_tls_inner_enable(struct rs_cb* rsCb, unsigned int enable);
extern int rs_ssl_inner_init(struct rs_cb* rscb);
extern int rs_ssl_ca_ky_init(SSL_CTX* sslCtx, struct rs_cb* rscb);
extern int rs_ssl_load_ca(SSL_CTX* sslCtx, struct rs_cb* rscb, struct tls_cert_mng_info* mngInfo);
extern int rs_ssl_crl_init(SSL_CTX* sslCtx, struct rs_cb* rscb, struct tls_cert_mng_info* mngInfo);
extern int rs_ssl_get_crl_data(struct rs_cb* rscb, FILE* fp, struct tls_cert_mng_info* mngInfo, X509_CRL** crl);
extern int rs_check_pridata(SSL_CTX* sslCtx, struct rs_cb* rscb, struct tls_cert_mng_info* mngInfo);
extern int rs_get_pk(struct rs_cb* rscb, struct tls_cert_mng_info* mngInfo, EVP_PKEY** pky);
extern int
rs_ssl_get_ca_data(struct rs_cb* rscb, const char* endFile, const char* caFile, struct tls_cert_mng_info* mngInfo);
extern int rs_remove_certs(const char* endFile, const char* caFile);
extern int rs_ssl_put_certs(
    struct rs_cb* rscb, struct tls_cert_mng_info* mngInfo, struct RsCerts* certs, struct tls_ca_new_certs* newCerts,
    struct CertFile* fileName);
extern int rs_ssl_skid_get_from_chain(
    struct rs_cb* rscb, struct tls_cert_mng_info* mngInfo, struct RsCerts* certs, struct tls_ca_new_certs* newCerts);
extern int
rs_socket_fill_wlist_by_phyID(unsigned int chipId, struct SocketWlistInfoT* whiteListNode, struct RsConnInfo* rsConn);
extern int rs_ssl_check_mng_and_cert_chain(
    struct rs_cb* rscb, struct tls_cert_mng_info* mngInfo, struct RsCerts* certs, struct tls_ca_new_certs* newCerts,
    struct CertFile* fileName);
extern int
rs_ssl_check_cert_chain(struct tls_cert_mng_info* mngInfo, struct RsCerts* certs, struct tls_ca_new_certs* newCerts);
extern int rs_ssl_verify_cert(X509_STORE_CTX* ctx);
extern int rs_ssl_verify_cert_chain(
    X509_STORE_CTX* ctx, X509_STORE* store, struct RsCerts* certs, struct tls_cert_mng_info* mngInfo,
    struct tls_ca_new_certs* newCerts);
extern X509* tls_load_cert(const uint8_t* inbuf, uint32_t bufLen, int type);
extern int rs_ssl_get_leaf_cert(struct RsCerts* certs, X509** leafCert);
extern int tls_get_cert_chain(X509_STORE* store, struct RsCerts* certs, struct tls_cert_mng_info* mngInfo);
extern int rs_tls_peer_cert_verify(SSL* ssl, struct rs_cb* rscb);
extern int RsEpollEventSslAcceptInHandle(struct rs_cb* rsCb, int fd);
extern int NetCommGetSelfHome(char* userNamePath, unsigned int pathLen);
extern int GetTlsConfigPath(char* userNamePath, unsigned int pathLen);
extern int RsDrvQpNormal(struct RsQpCb* qpCb, int qpMode);
extern int
RsDrvNormalQpCreateInit(struct ibv_qp_init_attr* qpInitAttr, struct RsQpCb* qpCb, struct ibv_port_attr* attr);
extern int RsSendExpWrlist(
    struct RsQpCb* qpCb, struct WrInfo* wrList, unsigned int sendNum, struct SendWrRsp* wrRsp,
    unsigned int* completeNum);
extern int RsGetMrcb(struct RsQpCb* qpCb, uint64_t addr, struct RsMrCb** mrCb, struct RsListHead* mrList);
extern void RsWirteAndReadBuildUpWr(
    struct RsMrCb* mrCb, struct RsMrCb* remMrCb, struct WrInfo* wr, struct ibv_sge* list, struct ibv_send_wr* ibWr);
extern struct ibv_mr* RsDrvMrReg(struct ibv_pd* pd, char* addr, size_t length, int access);
extern int RsQueryEvent(int cqEventId, struct event_summary** event);
extern int tls_get_user_config(
    unsigned int saveMode, unsigned int chipId, const char* name, unsigned char* buf, unsigned int* bufSize);
extern void TlsGetEnableInfo(unsigned int saveMode, unsigned int chipId, unsigned char* buf, unsigned int bufSize);
extern int RsGetIpv6ScopeId(struct in6_addr localIp);
extern int verify_callback(int prevOk, X509_STORE_CTX* ctx);
extern int RsOpenBackupIbCtx(struct RsRdevCb* rdevCb);
extern int RsGetSqDepthAndQpMaxNum(struct RsRdevCb* rdevCb, unsigned int rdevIndex);
extern int RsSetupPdAndNotify(struct RsRdevCb* rdevCb);
extern int RsSocketConnectCheckPara(struct SocketConnectInfo* connInfo);
extern int RsSocketNodeid2vnic(uint32_t nodeId, uint32_t* ipAddr);
extern int rsGetLocalDevIDByHostDevID(unsigned int phyId, unsigned int* chipId);
extern int RsDev2conncb(uint32_t chipId, struct RsConnCb** connCb);
extern int RsGetConnInfo(
    struct RsConnCb* connCb, struct SocketConnectInfo* conn, struct RsConnInfo** connInfo, unsigned int serverPort);
extern int RsConvertIpAddr(int family, union HccpIpAddr* ipAddr, struct RsIpAddrInfo* ip);
extern int RsFindListenNode(
    struct RsConnCb* connCb, struct RsIpAddrInfo* ipAddr, uint32_t serverPort, struct RsListenInfo** listenInfo);
extern int RsSocketListenAddToEpoll(struct RsConnCb* connCb, struct RsListenInfo* listenInfo);
extern bool RsSocketIsVnicIp(unsigned int chipId, unsigned int ipAddr);
extern int kmc_dec_data(struct KmcEncInfo* encInfo, unsigned char* outbuf, unsigned int* sizeOut);

typedef uint32_t u32;
typedef uint16_t u16;
typedef unsigned long long u64;
typedef signed int s32;

const char* sTmp = "suc";
struct RsQpCb* qpCbAb2;
struct RsRdevCb gRdevCb = {0};
struct RsListenInfo gListenInfo = {0};
struct RsListenInfo* gPlistenInfo = &gListenInfo;

#define SLEEP_TIME 10000

int RsConnInit(int index);
void RsConnPrepare(void* arg);
int RsConnGetSockets(int index);
int RsConnInfoUpdate(int index);
int RsConnQpInfoUpdate(int index, int i);
int RsConnMrInfoUpdate(int index, int i);
uint64_t str2long(const char* str);
int RsConnSendImm(uint32_t qpn, void* srcAddr, int len, const char* dstAddr, int immData);

int RsConnCloseCheckTimeout(int realDevId);
int dev_read_flash(unsigned int chipId, const char* name, unsigned char* buf, unsigned int* bufSize);

int RsDev2rscb(uint32_t chipId, struct rs_cb** rsCb, bool initFlag);
int memset_s(void* dest, size_t destMax, int c, size_t count);
void RsFreeAcceptOneNode(struct rs_cb* rscb, struct RsAcceptInfo* accept);
int RsEpollEventListenInHandle(struct rs_cb* rsCb, int fd);
int RsEpollEventQpMrInHandle(struct rs_cb* rsCb, int fd);
int RsEpollEventHeterogTcpRecvInHandle(struct rs_cb* rsCb, int fd);
void RsFreeHeterogTcpFdList(struct rs_cb* rsCb);
extern __thread struct rs_cb* gRsCb;
extern struct RsCqeErrInfo gRsCqeErr;
extern void RsDrvSaveCqeErrInfo(uint32_t status, struct RsQpCb* qpCb);
extern int RsDrvNormalQpCreate(struct RsQpCb* qpCb, struct ibv_qp_init_attr* qpInitAttr);
extern int DlHalGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t* value);

extern int rs_ssl_get_cert(
    struct rs_cb* rscb, struct RsCerts* certs, struct tls_cert_mng_info* mngInfo, struct tls_ca_new_certs* newCerts);
extern int rs_ssl_x509_store_init(
    X509_STORE* store, struct RsCerts* certs, struct tls_cert_mng_info* mngInfo, struct tls_ca_new_certs* newCerts);
extern int rs_ssl_skids_subjects_get(
    struct rs_cb* rscb, struct tls_cert_mng_info* mngInfo, struct RsCerts* certs, struct tls_ca_new_certs* newCerts);
extern int rs_ssl_put_cert_ca_pem(
    struct RsCerts* certs, struct tls_cert_mng_info* mngInfo, struct tls_ca_new_certs* newCerts, const char* caFile);
extern int rs_ssl_put_cert_end_pem(struct RsCerts* certs, struct tls_ca_new_certs* newCerts, const char* endFile);
extern int rs_ssl_put_end_cert(struct RsCerts* certs, const char* endFile);
extern int rs_ssl_X509_store_add_cert(char* certInfo, X509_STORE* store);
extern int RsGetLinuxVersion(struct RsLinuxVersionInfo* verInfo);
extern void freeifaddrs(struct ifaddrs* ifa);
extern int DlHalSensorNodeUpdateState(uint32_t devid, uint64_t handle, int val, halGeneralEventType_t assertion);
extern int RsQueryRdevCb(unsigned int phyId, unsigned int rdevIndex, struct RsRdevCb** rdevCb);
extern int RsFindSockets(struct RsConnInfo* connTmp, struct SocketFdData conn[], int num, int role);
extern void RsEpollEventInHandle(struct rs_cb* rsCb, struct epoll_event* events);
extern void RsEpollEventHandleOne(struct rs_cb* rsCb, struct epoll_event* events);
extern void RsCqeCallbackProcess(struct RsQpCb* qpCb, struct ibv_wc* wc, struct ibv_cq* evCq);
extern int rs_get_pridata(struct rs_cb* rscb, struct RsSecPara* rsPara, struct tls_cert_mng_info* mngInfo);
extern void RsSocketTagSync(struct RsConnInfo* conn);
extern int RsRoceUserApiInit(void);
extern int RsSendNormalWrlist(
    struct RsQpCb* qpCb, struct WrInfo* wrList, unsigned int sendNum, unsigned int* completeNum, unsigned int keyFlag);
extern int RsSocketListenBindListen(
    int fd, struct RsConnCb* connCb, struct SocketListenInfo* conn, struct RsListenInfo* listenInfo, int port);
extern int RsEpollTcpRecv(struct rs_cb* rsCb, unsigned int index);
extern int RsInitMemPool(struct RsQpCb* qpCb);
extern void RsDeinitMemPool(struct RsQpCb* qpCb);
extern int RsGetVnicIpInfo(unsigned int phyId, unsigned int rdevIndex, unsigned int vnicIndex, struct IpInfo* info);
extern void RsFreeUdevList(struct rs_cb* rscb);
extern void RsFreeDevList(struct rs_cb* rscb);
extern void RsFreeRdevList(struct rs_cb* rscb);
extern void RsRdmaRetryTimeoutExceptionCheck(struct SensorNode* sensorNode, struct ibv_wc* wc);
extern int RsSocketCheckCredit(struct RsConnCb* connCb, struct RsListenInfo* listenInfo);
extern const char* RsIbvGetDeviceName(struct ibv_device* device);
extern int RsGetHostRdevIndex(struct rdev rdevInfo, struct RsRdevCb* rdevCb, int* rdevIndex, int flag);
extern int RsServerSendWlistCheckResult(struct RsConnInfo* conn, bool flag);

const char* stub_RsIbvGetDeviceName(struct ibv_context* context) { return "910B"; }

int RsQueryRdevCbStub(unsigned int phyId, unsigned int rdevIndex, struct RsRdevCb** rdevCb)
{
    static struct RsRdevCb stub_rdev_cb = {0};

    *rdevCb = &stub_rdev_cb;
    return 0;
}

void TcRsAbnormal()
{
    int ret = 0;
    struct rs_cb* rsCb = NULL;
    uint32_t qpn = 0;
    struct RsQpCb* qpCb = NULL;
    struct RsQpCb qpCbTmp = {0};
    unsigned int phyId = 0;
    unsigned int rdevIndex = 0;
    rs_ut_msg("\n+++++++++ABNORMAL TC Start++++++++\n");
    ret = RsDev2rscb(0, &rsCb, false);
    EXPECT_INT_NE(ret, 0);

    ret = RsDev2rscb(2, &rsCb, false);
    EXPECT_INT_NE(ret, 0);

    ret = RsQpn2qpcb(phyId, rdevIndex, 5, &qpCb);

    rs_ut_msg("---------ABNORMAL TC End----------\n\n");

    return;
}

int stub_dl_hal_get_device_info_pod_910A(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t* value)
{
    *value = 87;
    return 0;
}

int dl_hal_get_device_info_910A(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t* value)
{
    *value = 256;
    return 0;
}

int DlHalGetDeviceInfoSharemem(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t* value)
{
    *value = (5 << 8);
    return 0;
}

int DlHalQueryDevPidSharemem(struct halQueryDevpidInfo info, pid_t* devPid) { return 0; }

extern int SprintfS(char* strDest, size_t destMax, const char* format, ...);
void TcRsInit()
{
    int ret = 0;
    struct RsInitConfig cfg = {0};

    rs_ut_msg("\n%s+++++++++ABNORMAL TC Start++++++++\n", __func__);
    ret = RsInit(NULL);
    EXPECT_INT_NE(ret, 0);

    rs_ut_msg("\n%s---------ABNORMAL TC End----------\n", __func__);

    cfg.chipId = 0;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    RsSetHostPid(cfg.chipId, 0, NULL);
    EXPECT_INT_EQ(ret, 0);

    RsSetHostPid(15, 0, NULL);
    EXPECT_INT_EQ(ret, 0);

    cfg.chipId = 0;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, -17);

    /* ------Resource CLEAN-------- */
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    return;
}

void TcRsDeinit()
{
    int ret = 0;
    uint32_t chipId = 0;
    struct rs_cb* rsCb = NULL;
    int eventfdTmp = 0;
    struct RsInitConfig cfg = {0};

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    RsDev2rscb(chipId, &rsCb, false);

    rs_ut_msg("\n%s+++++++++ABNORMAL TC Start++++++++\n", __func__);
    /* param error */
    ret = RsDeinit(NULL);
    EXPECT_INT_NE(ret, 0);

    /* env store */
    eventfdTmp = rsCb->connCb.eventfd;
    rsCb->connCb.eventfd = -1;
    ret = RsDeinit(&cfg);
    EXPECT_INT_NE(ret, 0);
    /* env recovery */
    rsCb->connCb.eventfd = eventfdTmp;

    rs_ut_msg("\n%s---------ABNORMAL TC End----------\n", __func__);

    /* ------Resource CLEAN-------- */
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    return;
}

/* FREE server/client conn_info, listen_info AUTOMATICALLY */
void TcRsDeinit2()
{
    int ret = 0;
    int i = 0;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listen[2] = {0};
    struct SocketConnectInfo conn[2] = {0};
    struct SocketWlistInfoT whiteList = {0};
    whiteList.remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    whiteList.connLimit = 1;

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 1);

    strcpy(whiteList.tag, "1234");
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");
    RsSocketWhiteListAdd(rdevInfo, &whiteList, 1);
    strcpy(whiteList.tag, "5678");
    RsSocketWhiteListAdd(rdevInfo, &whiteList, 1);

    conn[0].phyId = 0;
    conn[0].family = AF_INET;
    conn[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    conn[0].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(conn[0].tag, "1234");
    conn[1].phyId = 0;
    conn[1].family = AF_INET;
    conn[1].localIp.addr.s_addr = inet_addr("127.0.0.1");
    conn[1].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(conn[1].tag, "5678");
    conn[0].port = 16666;
    conn[1].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 2);

    usleep(SLEEP_TIME);

    /* ------Resource CLEAN-------- */
    struct rs_cb* rsCb = NULL;
    ret = RsGetRsCb(rdevInfo.phyId, &rsCb);
    rsCb->sslEnable = 1;
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    return;
}

void TcRsSocketInit()
{
    int ret = 0;
    int i = 0;
    unsigned int vnicIp[8] = {0};

    ret = RsSocketInit(NULL, 0);
    EXPECT_INT_EQ(ret, -22);

    ret = RsSocketInit(vnicIp, 8);
    EXPECT_INT_EQ(ret, 0);

    return;
}

void TcRsSocketDeinit1()
{
    int ret = 0;
    int i = 0;
    struct RsInitConfig cfg = {0};
    struct rs_cb* rsCb = NULL;
    struct rdev rdevInfo = {0};
    struct RsAcceptInfo* accept = calloc(1, sizeof(struct RsAcceptInfo));
    struct RsListHead list = {0};

    RS_INIT_LIST_HEAD(&list);
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");

    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    ret = RsGetRsCb(rdevInfo.phyId, &rsCb);
    rsCb->sslEnable = 1;

    accept->list = list;
    accept->ssl = NULL;

    RsFreeAcceptOneNode(rsCb, accept);
}

void TcRsSocketDeinit2()
{
    int ret = 0;
    int i = 0;
    struct RsInitConfig cfg = {0};
    struct rs_cb* rsCb = NULL;
    struct rdev rdevInfo = {0};
    struct RsAcceptInfo* accept = calloc(1, sizeof(struct RsAcceptInfo));
    struct RsListHead list = {0};

    RS_INIT_LIST_HEAD(&list);
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");

    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    ret = RsGetRsCb(rdevInfo.phyId, &rsCb);
    rsCb->sslEnable = 1;
    accept->ssl = malloc(sizeof(SSL_CTX));
    accept->list = list;

    free(accept->ssl);
    RsFreeAcceptOneNode(rsCb, accept);
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
}

void TcRsSocketListenIpv6()
{
    int ret = 0;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listen[2] = {0};

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    listen[0].phyId = 0;
    listen[0].family = AF_INET6;
    inet_pton(AF_INET6, "::1", &listen[0].localIp.addr6);
    listen[0].port = 16666;
    ret = RsSocketSetScopeId(0, if_nametoindex("lo"));
    mocker(RsGetIpv6ScopeId, 10, if_nametoindex("lo"));
    EXPECT_INT_EQ(ret, 0);
    ret = RsSocketListenStart(&listen[0], 1);

    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 1);

    /* ------Resource CLEAN-------- */
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    return;
}

void TcRsSocketListen()
{
    int ret = 0;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listen[2] = {{0}};

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 1);

    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 1);

    /* listen 1 will fail, cannot listen same IP twice */
    listen[1].phyId = 0;
    listen[1].family = AF_INET;
    listen[1].localIp.addr.s_addr = inet_addr("127.0.0.1");
    listen[1].port = 16666;
    ret = RsSocketListenStart(&listen[1], 1);

    listen[1].port = 16666;
    ret = RsSocketListenStop(&listen[1], 1);

    /* stop a non-exist node */
    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 1);
    EXPECT_INT_EQ(ret, 0);

    /* ------Resource CLEAN-------- */
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    return;
}

void TcRsSocketConnect()
{
    int ret = 0;
    int i = 0;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listen[2] = {0};
    struct SocketConnectInfo conn[2] = {0};
    struct RsSocketCloseInfoT sockClose[2] = {0};
    struct SocketFdData socketInfo[3] = {0};
    struct SocketWlistInfoT whiteList = {0};
    whiteList.remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    whiteList.connLimit = 1;
    int tryNum = 10;

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 1);

    strcpy(whiteList.tag, "1234");
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");
    RsSocketWhiteListAdd(rdevInfo, &whiteList, 1);
    strcpy(whiteList.tag, "5678");
    RsSocketWhiteListAdd(rdevInfo, &whiteList, 1);

    conn[0].phyId = 0;
    conn[0].family = AF_INET;
    conn[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    conn[0].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(conn[0].tag, "1234");
    conn[1].phyId = 0;
    conn[1].family = AF_INET;
    conn[1].localIp.addr.s_addr = inet_addr("127.0.0.1");
    conn[1].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(conn[1].tag, "5678");
    conn[0].port = 16666;
    conn[1].port = 16666;
    ret = RsSocketBatchConnect(conn, 2);

    strcpy(whiteList.tag, "1234");
    RsSocketWhiteListAdd(rdevInfo, &whiteList, 1);
    /* >>>>>>> RsSocketBatchConnect test case begin <<<<<<<<<<< */
    /* repeat connect */
    conn[0].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 1);

    /* param error - conn NULL */
    ret = RsSocketBatchConnect(NULL, 1);

    /* param error - num error */
    conn[0].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 0);

    /* param error - device id error */
    conn[0].phyId = 64;
    conn[0].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 1);
    /* >>>>>>> RsSocketBatchConnect test case end <<<<<<<<<<< */

    usleep(SLEEP_TIME);

    i = 0;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo[i].tag, "1234");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [client]socket_info[0].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    sockClose[i].fd = socketInfo[i].fd;
    ret = RsSocketBatchClose(0, &sockClose[i], 1);

    i = 1;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo[i].tag, "5678");

    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [client]socket_info[1].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    sockClose[i].fd = socketInfo[i].fd;
    ret = RsSocketBatchClose(0, &sockClose[i], 1);

    /* close a non-exist fd */
    ret = RsSocketBatchClose(0, &sockClose[i], 1);

    i = 0;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo[i].tag, "1234");

    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_SERVER, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg(
        "%s [server]socket_info[0].fd:%d, client if:0x%x, status:%d\n", __func__, socketInfo[i].fd,
        socketInfo[i].remoteIp.addr.s_addr, socketInfo[i].status);

    sockClose[i].fd = socketInfo[i].fd;
    ret = RsSocketBatchClose(0, &sockClose[i], 1);

    /* ------Resource CLEAN-------- */
    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 1);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    return;
}

void TcRsGetSockets()
{
    int ret = 0;
    int i = 0;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listen[2] = {0};
    struct SocketConnectInfo conn[2] = {0};
    struct RsSocketCloseInfoT sockClose[2] = {0};
    struct SocketFdData socketInfo[3] = {0};
    struct SocketWlistInfoT whiteList = {0};
    whiteList.remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    whiteList.connLimit = 1;
    int tryNum = 10;

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 1);
    strcpy(whiteList.tag, "1234");
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");
    RsSocketWhiteListAdd(rdevInfo, &whiteList, 1);
    strcpy(whiteList.tag, "5678");
    RsSocketWhiteListAdd(rdevInfo, &whiteList, 1);

    conn[0].phyId = 0;
    conn[0].family = AF_INET;
    conn[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    conn[0].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(conn[0].tag, "1234");
    conn[1].phyId = 0;
    conn[1].family = AF_INET;
    conn[1].localIp.addr.s_addr = inet_addr("127.0.0.1");
    conn[1].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(conn[1].tag, "5678");
    conn[0].port = 16666;
    conn[1].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 2);

    usleep(SLEEP_TIME);

    i = 0;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo[i].tag, "1234");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        usleep(5000);
        rs_ut_msg(">>**RsGetSockets ret:%d\n", ret);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [client]socket_info[0].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    sockClose[i].fd = socketInfo[i].fd;
    ret = RsSocketBatchClose(0, &sockClose[i], 1);

    i = 1;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo[i].tag, "5678");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [client]socket_info[1].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    /* param error */
    ret = RsGetSockets(RS_CONN_ROLE_CLIENT, NULL, 3);
    EXPECT_INT_NE(ret, 0);

    /* device id error */
    socketInfo[i].phyId = 55555;
    ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
    EXPECT_INT_NE(ret, 0);
    socketInfo[i].phyId = 0;

    /* repeat get */
    ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
    EXPECT_INT_EQ(ret, 0);

    sockClose[i].fd = socketInfo[i].fd;
    ret = RsSocketBatchClose(0, &sockClose[i], 1);

    /* close a non-exist fd */
    ret = RsSocketBatchClose(0, &sockClose[i], 1);
    EXPECT_INT_NE(ret, 0);

    i = 0;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo[i].tag, "1234");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_SERVER, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg(
        "%s [server]socket_info[0].fd:%d, client if:0x%x, status:%d\n", __func__, socketInfo[i].fd,
        socketInfo[i].remoteIp.addr.s_addr, socketInfo[i].status);

    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].status = RS_SOCK_STATUS_OK;
    struct RsConnInfo connTmp = {0};
    connTmp.state = RS_CONN_STATE_VALID_SYNC;
    RsFindSockets(&connTmp, &socketInfo[i], 1, RS_CONN_ROLE_CLIENT);
    sockClose[i].fd = socketInfo[i].fd;
    ret = RsSocketBatchClose(0, &sockClose[i], 1);

    /* ------Resource CLEAN-------- */
    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 1);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    return;
}

void TcRsSetTsqpDepth()
{
    int ret = 0;
    unsigned int phyId = 0;
    unsigned int rdevIndex = 0;
    unsigned int tempDepth = 8;
    unsigned int qpNum = 0;
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");
    struct RsInitConfig cfg = {0};
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;

    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsSetTsqpDepth(phyId, rdevIndex, tempDepth, &qpNum);
    EXPECT_INT_EQ(ret, 0);

    ret = RsRdevDeinit(phyId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    return;
}

void TcRsGetTsqpDepth()
{
    int ret = 0;
    unsigned int phyId = 0;
    unsigned int rdevIndex = 0;
    unsigned int tempDepth = 8;
    unsigned int qpNum = 0;
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");
    struct RsInitConfig cfg = {0};
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;

    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsGetTsqpDepth(phyId, rdevIndex, &tempDepth, &qpNum);
    EXPECT_INT_EQ(ret, 0);
    ret = RsRdevDeinit(phyId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    return;
}

int StubIbvQueryPort(struct ibv_context* context, uint8_t portNum, struct ibv_port_attr* portAttr)
{
    portAttr->gid_tbl_len = 2;
    return 0;
}

extern int ibv_query_port(struct ibv_context* context, uint8_t portNum, struct ibv_port_attr* portAttr);
void TcRsQpCreate()
{
    int ret = 0;
    uint32_t phyId = 0;
    unsigned int rdevIndex = 0;
    int flag = 0; /* RC */
    struct RsQpResp resp = {0};
    struct RsQpResp resp2 = {0};
    int i = 0;
    int tryNum = 10;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listen[2] = {0};
    struct SocketConnectInfo conn[2] = {0};
    struct RsSocketCloseInfoT sockClose[2] = {0};
    struct SocketFdData socketInfo[3] = {0};
    struct SocketWlistInfoT whiteList = {0};
    struct RsQpNorm qpNorm = {0};

    qpNorm.flag = flag;
    qpNorm.qpMode = 1;
    qpNorm.isExp = 1;

    whiteList.remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    whiteList.connLimit = 1;

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 1;
    cfg.hccpMode = NETWORK_OFFLINE;
    mocker((stub_fn_t)drvGetDevNum, 10, -1);
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 1);

    strcpy(whiteList.tag, "1234");
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");
    RsSocketWhiteListAdd(rdevInfo, &whiteList, 1);

    conn[0].phyId = 0;
    conn[0].family = AF_INET;
    conn[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    conn[0].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(conn[0].tag, "1234");
    conn[0].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 1);

    usleep(SLEEP_TIME);

    i = 0;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo[i].tag, "1234");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [client]socket_info[0].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    mocker_invoke((stub_fn_t)ibv_query_port, StubIbvQueryPort, 10);
    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, -ENOLINK);
    mocker_clean();

    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    enum PortStatus status = PORT_STATUS_DOWN;
    ret = RsRdevGetPortStatus(100000, rdevIndex, NULL);
    EXPECT_INT_NE(ret, 0);
    ret = RsRdevGetPortStatus(rdevInfo.phyId, rdevIndex, NULL);
    EXPECT_INT_NE(ret, 0);
    ret = RsRdevGetPortStatus(64, rdevIndex, &status);
    EXPECT_INT_NE(ret, 0);
    ret = RsRdevGetPortStatus(rdevInfo.phyId, 100000, &status);
    EXPECT_INT_NE(ret, 0);
    ret = RsRdevGetPortStatus(rdevInfo.phyId, rdevIndex, &status);
    EXPECT_INT_EQ(ret, 0);
    mocker(ibv_query_port, 20, -1);
    ret = RsRdevGetPortStatus(rdevInfo.phyId, rdevIndex, &status);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    int supportLite = 0;
    ret = RsGetLiteSupport(rdevInfo.phyId, rdevIndex, &supportLite);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(supportLite, 1);

    struct LiteRdevCapResp rdevResp = {0};
    ret = RsGetLiteRdevCap(rdevInfo.phyId, rdevIndex, &rdevResp);
    EXPECT_INT_EQ(ret, 0);

    ret = RsQpCreate(phyId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("RsQpCreate: qpn %d, ret:%d\n", resp.qpn, ret);

    struct LiteQpCqAttrResp qpResp = {0};
    ret = RsGetLiteQpCqAttr(phyId, rdevIndex, resp.qpn, &qpResp);
    EXPECT_INT_EQ(ret, 0);

    struct LiteMemAttrResp memResp = {0};
    ret = RsGetLiteMemAttr(rdevInfo.phyId, rdevIndex, resp.qpn, &memResp);
    EXPECT_INT_EQ(ret, 0);

    struct QosAttr QosAttr = {0};
    QosAttr.tc = 100;
    QosAttr.sl = 3;
    ret = RsSetQpAttrQos(phyId, rdevIndex, resp.qpn, &QosAttr);

    unsigned int timeout = 15;
    unsigned int retryCnt = 6;
    ret = RsSetQpAttrTimeout(phyId, rdevIndex, resp.qpn, &timeout);

    ret = RsSetQpAttrRetryCnt(phyId, rdevIndex, resp.qpn, &retryCnt);

    /* >>>>>>> RsQpConnectAsync test case begin <<<<<<<<<<< */
    /* param error - qpn */
    ret = RsQpConnectAsync(phyId, rdevIndex, 4444, socketInfo[i].fd);
    EXPECT_INT_NE(ret, 0);

    /* param error - fd */
    ret = RsQpConnectAsync(phyId, rdevIndex, resp.qpn, -1);
    EXPECT_INT_NE(ret, 0);
    /* >>>>>>> RsQpConnectAsync test case end <<<<<<<<<<< */

    ret = RsQpConnectAsync(phyId, rdevIndex, resp.qpn, socketInfo[i].fd);
    rs_ut_msg("***RsQpConnectAsync: %d****\n", ret);

    struct RsQpCb* qpCb = NULL;
    ret = RsQpn2qpcb(phyId, rdevIndex, resp.qpn, &qpCb);
    EXPECT_INT_EQ(ret, 0);
    struct LiteConnectedInfoResp connectedResp = {0};
    qpCb->state = RS_QP_STATUS_CONNECTED;
    ret = RsGetLiteConnectedInfo(phyId, rdevIndex, resp.qpn, &connectedResp);
    EXPECT_INT_EQ(ret, 0);

    usleep(SLEEP_TIME);

    i = 1;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo[i].tag, "1234");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_SERVER, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [server]socket_info[1].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    /* >>>>>>> RsQpCreate test case begin <<<<<<<<<<< */
    /* param error - device id */
    ret = RsQpCreate(64, rdevIndex, qpNorm, &resp2);
    EXPECT_INT_NE(ret, 0);

    /* qp number out of boundry */
    struct rs_cb* rsCb = NULL;
    struct RsRdevCb* rdevCb = NULL;
    struct RsAcceptInfo acceptTmp = {0};
    struct RsAcceptInfo* accept = &acceptTmp;
    uint32_t chipId = 0;
    ret = RsDev2rscb(chipId, &rsCb, false);
    EXPECT_INT_EQ(ret, 0);

    ret = RsGetRdevCb(rsCb, rdevIndex, &rdevCb);
    EXPECT_INT_EQ(ret, 0);

    int qpCntTmp = rdevCb->qpCnt;
    rdevCb->qpCnt = 44444;
    ret = RsQpCreate(phyId, rdevIndex, qpNorm, &resp2);
    EXPECT_INT_NE(ret, 0);
    rdevCb->qpCnt = qpCntTmp;
    /* >>>>>>> RsQpCreate test case end <<<<<<<<<<< */

    ret = RsQpCreate(phyId, rdevIndex, qpNorm, &resp2);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("RsQpCreate: qpn2 %d, ret:%d\n", resp2.qpn, ret);

    ret = RsQpConnectAsync(phyId, rdevIndex, resp2.qpn, socketInfo[i].fd);
    usleep(SLEEP_TIME);

    ret = RsQpDestroy(phyId, rdevIndex, resp2.qpn);
    ret = RsQpDestroy(phyId, rdevIndex, resp.qpn);

    /* param error - qpn */
    ret = RsQpDestroy(phyId, rdevIndex, resp.qpn);
    EXPECT_INT_NE(ret, 0);

    sockClose[0].fd = socketInfo[0].fd;
    ret = RsSocketBatchClose(0, &sockClose[0], 1);

    sockClose[1].fd = socketInfo[1].fd;
    ret = RsSocketBatchClose(0, &sockClose[1], 1);

    /* ------Resource CLEAN-------- */
    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 1);

    ret = RsRdevDeinit(phyId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsSocketDeinit(rdevInfo);
    EXPECT_INT_EQ(ret, 0);
    cfg.chipId = 0;
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    /* >>>>>>> 1910 create_qp test case begin <<<<<<<<<<< */
    struct RsInitConfig cfg1910 = {0};
    cfg1910.chipId = 0;
    cfg1910.hccpMode = NETWORK_ONLINE;
    ret = RsInit(&cfg1910);
    EXPECT_INT_EQ(ret, 0);

    rdevInfo.phyId = cfg1910.chipId;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");
    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);
    struct RsQpResp resp1910 = {0};
    ret = RsQpCreate(phyId, rdevIndex, qpNorm, &resp1910);
    EXPECT_INT_EQ(ret, 0);

    ret = RsQpDestroy(phyId, rdevIndex, resp1910.qpn);

    ret = RsRdevDeinit(phyId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);
    ret = RsSocketDeinit(rdevInfo);
    EXPECT_INT_EQ(ret, 0);

    ret = RsRdevInitWithBackup(rdevInfo, rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);
    ret = RsRdevDeinit(phyId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg1910);
    EXPECT_INT_EQ(ret, 0);
    /* >>>>>>> 1910 create_qp test case end <<<<<<<<<<< */

    return;
}

#define RS_DRV_CQ_DEPTH 16384
#define RS_DRV_CQ_128_DEPTH 128
#define RS_DRV_CQ_8K_DEPTH 8192
#define RS_QP_ATTR_MAX_INLINE_DATA 32
#define RS_QP_ATTR_MAX_SEND_SGE 8
#define RS_QP_TX_DEPTH_PEER_ONLINE 4096
#define RS_QP_TX_DEPTH_ONLINE 4096
#define RS_QP_TX_DEPTH 8191
#define RS_QP_TX_DEPTH_OFFLINE 128
void QpExtAttrs(int qpMode, struct QpExtAttrs* extAttrs)
{
    extAttrs->qpMode = qpMode;
    extAttrs->cqAttr.sendCqDepth = RS_DRV_CQ_8K_DEPTH;
    extAttrs->cqAttr.sendCqCompVector = 0;
    extAttrs->cqAttr.recvCqDepth = RS_DRV_CQ_128_DEPTH;
    extAttrs->cqAttr.recvCqCompVector = 0;
    extAttrs->qpAttr.qp_context = NULL;
    extAttrs->qpAttr.send_cq = NULL;
    extAttrs->qpAttr.recv_cq = NULL;
    extAttrs->qpAttr.srq = NULL;
    extAttrs->qpAttr.cap.max_send_wr = RS_QP_TX_DEPTH_OFFLINE;
    extAttrs->qpAttr.cap.max_recv_wr = RS_QP_TX_DEPTH_OFFLINE;
    extAttrs->qpAttr.cap.max_send_sge = 1;
    extAttrs->qpAttr.cap.max_recv_sge = 1;
    extAttrs->qpAttr.cap.max_inline_data = RS_QP_ATTR_MAX_INLINE_DATA;
    extAttrs->qpAttr.qp_type = IBV_QPT_RC;
    extAttrs->qpAttr.sq_sig_all = 0;
    extAttrs->version = QP_CREATE_WITH_ATTR_VERSION;
}

void TcRsQpCreateWithAttrsV1()
{
    int ret = 0;
    uint32_t phyId = 0;
    unsigned int rdevIndex = 0;
    uint32_t qpn, qpn1, qpn2;
    int i = 0;
    int tryNum = 10;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listen[2] = {0};
    struct SocketConnectInfo conn[2] = {0};
    struct RsSocketCloseInfoT sockClose[2] = {0};
    struct SocketFdData socketInfo[3] = {0};
    struct SocketWlistInfoT whiteList = {0};
    struct RsQpNormWithAttrs qpNorm = {0};
    struct RsQpRespWithAttrs qpRespCreate = {0};

    qpNorm.isExp = 1;
    qpNorm.isExt = 1;
    QpExtAttrs(1, &qpNorm.extAttrs);

    whiteList.remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    whiteList.connLimit = 1;

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 1;
    cfg.hccpMode = NETWORK_OFFLINE;
    mocker((stub_fn_t)drvGetDevNum, 10, -1);
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 1);

    strcpy(whiteList.tag, "1234");
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");
    RsSocketWhiteListAdd(rdevInfo, &whiteList, 1);

    conn[0].phyId = 0;
    conn[0].family = AF_INET;
    conn[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    conn[0].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(conn[0].tag, "1234");
    conn[0].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 1);

    usleep(SLEEP_TIME);

    i = 0;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo[i].tag, "1234");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [client]socket_info[0].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    mocker_invoke((stub_fn_t)ibv_query_port, StubIbvQueryPort, 10);
    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, -ENOLINK);
    mocker_clean();

    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    int supportLite = 0;
    ret = RsGetLiteSupport(rdevInfo.phyId, rdevIndex, &supportLite);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(supportLite, 1);

    struct LiteRdevCapResp rdevResp = {0};
    ret = RsGetLiteRdevCap(rdevInfo.phyId, rdevIndex, &rdevResp);
    EXPECT_INT_EQ(ret, 0);

    qpNorm.extAttrs.dataPlaneFlag.bs.cqCstm = 1;
    qpNorm.aiOpSupport = 1;
    ret = RsQpCreateWithAttrs(phyId, rdevIndex, &qpNorm, &qpRespCreate);
    qpn = qpRespCreate.qpn;
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("RsQpCreateWithAttrs: qpn %d, ret:%d\n", qpn, ret);

    struct LiteQpCqAttrResp qpResp = {0};
    ret = RsGetLiteQpCqAttr(phyId, rdevIndex, qpn, &qpResp);
    EXPECT_INT_EQ(ret, 0);

    struct LiteMemAttrResp memResp = {0};
    ret = RsGetLiteMemAttr(rdevInfo.phyId, rdevIndex, qpn, &memResp);
    EXPECT_INT_EQ(ret, 0);

    struct QosAttr QosAttr = {0};
    QosAttr.tc = 100;
    QosAttr.sl = 3;
    ret = RsSetQpAttrQos(phyId, rdevIndex, qpn, &QosAttr);

    unsigned int timeout = 15;
    unsigned int retryCnt = 6;
    ret = RsSetQpAttrTimeout(phyId, rdevIndex, qpn, &timeout);

    ret = RsSetQpAttrRetryCnt(phyId, rdevIndex, qpn, &retryCnt);

    /* >>>>>>> RsQpConnectAsync test case begin <<<<<<<<<<< */
    /* param error - qpn */
    ret = RsQpConnectAsync(phyId, rdevIndex, 4444, socketInfo[i].fd);
    EXPECT_INT_NE(ret, 0);

    /* param error - fd */
    ret = RsQpConnectAsync(phyId, rdevIndex, qpn, -1);
    EXPECT_INT_NE(ret, 0);
    /* >>>>>>> RsQpConnectAsync test case end <<<<<<<<<<< */

    ret = RsQpConnectAsync(phyId, rdevIndex, qpn, socketInfo[i].fd);
    rs_ut_msg("***RsQpConnectAsync: %d****\n", ret);

    struct RsQpCb* qpCb = NULL;
    ret = RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    EXPECT_INT_EQ(ret, 0);
    struct LiteConnectedInfoResp connectedResp = {0};
    qpCb->state = RS_QP_STATUS_CONNECTED;
    ret = RsGetLiteConnectedInfo(phyId, rdevIndex, qpn, &connectedResp);
    EXPECT_INT_EQ(ret, 0);

    usleep(SLEEP_TIME);

    i = 1;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo[i].tag, "1234");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_SERVER, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [server]socket_info[1].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    /* >>>>>>> RsQpCreate test case begin <<<<<<<<<<< */
    /* param error - device id */
    ret = RsQpCreateWithAttrs(64, rdevIndex, &qpNorm, &qpRespCreate);
    qpn2 = qpRespCreate.qpn;
    EXPECT_INT_NE(ret, 0);

    /* qp number out of boundry */
    struct rs_cb* rsCb = NULL;
    struct RsRdevCb* rdevCb = NULL;
    struct RsAcceptInfo acceptTmp = {0};
    struct RsAcceptInfo* accept = &acceptTmp;
    uint32_t chipId = 0;
    ret = RsDev2rscb(chipId, &rsCb, false);
    EXPECT_INT_EQ(ret, 0);

    ret = RsGetRdevCb(rsCb, rdevIndex, &rdevCb);
    EXPECT_INT_EQ(ret, 0);

    int qpCntTmp = rdevCb->qpCnt;
    rdevCb->qpCnt = 44444;
    ret = RsQpCreateWithAttrs(phyId, rdevIndex, &qpNorm, &qpRespCreate);
    EXPECT_INT_NE(ret, 0);
    rdevCb->qpCnt = qpCntTmp;
    /* >>>>>>> RsQpCreateWithAttrs test case end <<<<<<<<<<< */

    ret = RsQpCreateWithAttrs(phyId, rdevIndex, &qpNorm, &qpRespCreate);
    qpn2 = qpRespCreate.qpn;
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("RsQpCreateWithAttrs: qpn2 %d, ret:%d\n", qpn2, ret);
    ret = RsQpConnectAsync(phyId, rdevIndex, qpn2, socketInfo[i].fd);

    qpNorm.extAttrs.qpMode = 0;
    ret = RsQpCreateWithAttrs(phyId, rdevIndex, &qpNorm, &qpRespCreate);
    qpn1 = qpRespCreate.qpn;
    rs_ut_msg("RsQpCreateWithAttrs: qpn1 %d, ret:%d\n", qpn1, ret);

    usleep(SLEEP_TIME);

    ret = RsQpDestroy(phyId, rdevIndex, qpn2);
    ret = RsQpDestroy(phyId, rdevIndex, qpn1);
    ret = RsQpDestroy(phyId, rdevIndex, qpn);

    /* param error - qpn */
    ret = RsQpDestroy(phyId, rdevIndex, qpn);
    EXPECT_INT_NE(ret, 0);

    sockClose[0].fd = socketInfo[0].fd;
    ret = RsSocketBatchClose(0, &sockClose[0], 1);

    sockClose[1].fd = socketInfo[1].fd;
    ret = RsSocketBatchClose(0, &sockClose[1], 1);

    /* ------Resource CLEAN-------- */
    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 1);

    ret = RsRdevDeinit(phyId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsSocketDeinit(rdevInfo);
    EXPECT_INT_EQ(ret, 0);
    cfg.chipId = 0;
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    /* >>>>>>> 1910 create_qp test case begin <<<<<<<<<<< */
    struct RsInitConfig cfg1910 = {0};
    cfg1910.chipId = 0;
    cfg1910.hccpMode = NETWORK_ONLINE;
    ret = RsInit(&cfg1910);
    EXPECT_INT_EQ(ret, 0);

    rdevInfo.phyId = cfg1910.chipId;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");
    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);
    int qpn1910 = 0;
    ret = RsQpCreateWithAttrs(phyId, rdevIndex, &qpNorm, &qpRespCreate);
    qpn1910 = qpRespCreate.qpn;
    EXPECT_INT_EQ(ret, 0);

    ret = RsQpDestroy(phyId, rdevIndex, qpn1910);

    ret = RsRdevDeinit(phyId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);
    ret = RsSocketDeinit(rdevInfo);
    EXPECT_INT_EQ(ret, 0);
    ret = RsDeinit(&cfg1910);
    EXPECT_INT_EQ(ret, 0);
    /* >>>>>>> 1910 create_qp test case end <<<<<<<<<<< */

    return;
}

void TcRsMrSync()
{
    int ret = 0;
    uint32_t phyId = 0;
    uint32_t rdevIndex = 0;
    int flag = 0; /* RC */
    int qpMode = 1;
    struct RsQpResp resp = {0};
    struct RsQpResp resp2 = {0};
    int i = 0;
    int tryNum = 10;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listen[2] = {0};
    struct SocketConnectInfo conn[2] = {0};
    struct RsSocketCloseInfoT sockClose[2] = {0};
    struct SocketFdData socketInfo[3] = {0};
    struct SocketWlistInfoT whiteList = {0};
    whiteList.remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    whiteList.connLimit = 1;

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 1);

    rs_ut_msg("___________________after listen:\n");
    strcpy(whiteList.tag, "1234");
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");
    RsSocketWhiteListAdd(rdevInfo, &whiteList, 1);

    conn[0].phyId = 0;
    conn[0].family = AF_INET;
    conn[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    conn[0].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(conn[0].tag, "1234");
    conn[0].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 1);

    rs_ut_msg("___________________after connect:\n");

    usleep(SLEEP_TIME);

    i = 0;
    socketInfo[i].family = AF_INET;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo[i].tag, "1234");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [client]socket_info[0].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    struct RsQpNorm qpNorm = {0};
    qpNorm.flag = flag;
    qpNorm.qpMode = qpMode;
    qpNorm.isExp = 1;
    ret = RsQpCreate(phyId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("RsQpCreate: qpn %d, ret:%d\n", resp.qpn, ret);

    rs_ut_msg("___________________after qp create:\n");

    /* >>>>>>> RsQpConnectAsync test case begin <<<<<<<<<<< */
    struct RdmaMrRegInfo mrRegInfo = {0};
    mrRegInfo.addr = (void*)0xabcdef;
    mrRegInfo.len = RS_TEST_MEM_SIZE;
    mrRegInfo.access = RS_ACCESS_LOCAL_WRITE;
    ret = RsMrReg(phyId, rdevIndex, resp.qpn, &mrRegInfo);
    EXPECT_INT_EQ(ret, 0);
    /* >>>>>>> RsQpConnectAsync test case end <<<<<<<<<<< */

    rs_ut_msg("___________________after mr reg:\n");

    ret = RsQpConnectAsync(phyId, rdevIndex, resp.qpn, socketInfo[i].fd);
    rs_ut_msg("***RsQpConnectAsync: %d****\n", ret);

    rs_ut_msg("___________________after qp connect async:\n");
    usleep(SLEEP_TIME);
    rs_ut_msg("___________________after qp connect async & sleep:\n");

    i = 1;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo[i].tag, "1234");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_SERVER, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [server]socket_info[1].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    ret = RsQpCreate(phyId, rdevIndex, qpNorm, &resp2);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("RsQpCreate: qpn2 %d, ret:%d\n", resp2.qpn, ret);
    rs_ut_msg("___________________after qp2 create:\n");

    ret = RsQpConnectAsync(phyId, rdevIndex, resp2.qpn, socketInfo[i].fd);

    rs_ut_msg("___________________after qp2 connect async:\n");

    usleep(SLEEP_TIME);

    ret = RsQpDestroy(phyId, rdevIndex, resp2.qpn);
    ret = RsQpDestroy(phyId, rdevIndex, resp.qpn);

    rs_ut_msg("___________________after qp1&2 destroy:\n");

    sockClose[0].fd = socketInfo[0].fd;
    ret = RsSocketBatchClose(0, &sockClose[0], 1);

    rs_ut_msg("___________________after close socket 0:\n");

    sockClose[1].fd = socketInfo[1].fd;
    ret = RsSocketBatchClose(0, &sockClose[1], 1);

    rs_ut_msg("___________________after close socket 1:\n");

    /* ------Resource CLEAN-------- */
    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 1);

    rs_ut_msg("___________________after stop listen:\n");

    ret = RsRdevDeinit(phyId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsSocketDeinit(rdevInfo);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    rs_ut_msg("___________________after deinit:\n");

    return;
}

/* create 2 socket & 2 qp, and connect them */
static int TcRsSockQpCreateNormal(int* fd, uint32_t* qpn, int* fd2, uint32_t* qpn2)
{
    int ret = 0;
    int i = 0;
    int tryNum = 10;
    uint32_t phyId = 0;
    uint32_t rdevIndex = 0;
    int flag = 0; /* RC */
    int qpMode = 0;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listen[1] = {0};
    struct SocketConnectInfo conn[2] = {0};
    struct SocketFdData socketInfo[3] = {0};
    struct SocketWlistInfoT whiteList = {0};
    struct RsQpResp resp = {0};
    struct RsQpResp resp2 = {0};
    whiteList.remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    whiteList.connLimit = 1;

    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");

    rs_ut_msg("resource prepare begin..................\n");

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("RS INIT, ret:%d !\n", ret);

    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 1);
    rs_ut_msg("RS LISTEN, ret:%d !\n", ret);

    strcpy(whiteList.tag, "1234");
    RsSocketWhiteListAdd(rdevInfo, &whiteList, 1);
    ;

    conn[0].phyId = 0;
    conn[0].family = AF_INET;
    conn[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    conn[0].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(conn[0].tag, "1234");
    conn[0].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 1);
    rs_ut_msg("RS CONNECT, ret:%d !\n", ret);

    usleep(SLEEP_TIME);

    i = 0;
    socketInfo[i].family = AF_INET;
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo[i].tag, "1234");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [client]socket_info[0].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    struct RsQpNorm qpNorm = {0};
    qpNorm.flag = flag;
    qpNorm.qpMode = qpMode;
    qpNorm.isExp = 0;

    ret = RsQpCreate(phyId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_EQ(ret, 0);
    *qpn = resp.qpn;
    rs_ut_msg("RS CREATE QP: QPN:%d, ret:%d\n", *qpn, ret);

    ret = RsQpConnectAsync(phyId, rdevIndex, *qpn, socketInfo[i].fd);
    *fd = socketInfo[i].fd;
    rs_ut_msg("RS QP CONNECT ASYNC: ret:%d\n", ret);

    usleep(SLEEP_TIME);

    i = 1;
    socketInfo[i].family = AF_INET;
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo[i].tag, "1234");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_SERVER, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [server]socket_info[1].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    ret = RsQpCreate(phyId, rdevIndex, qpNorm, &resp2);
    EXPECT_INT_EQ(ret, 0);
    *qpn2 = resp2.qpn;
    rs_ut_msg("RS CREATE QP: QPN:%d, ret:%d\n", *qpn2, ret);

    ret = RsQpConnectAsync(phyId, rdevIndex, *qpn2, socketInfo[i].fd);
    *fd2 = socketInfo[i].fd;

    usleep(SLEEP_TIME);

    rs_ut_msg("++++++++++++++ RS QP PREPARE DONE ++++++++++++++\n");

    return ret;
}

/* create 2 socket & 2 qp, and connect them */
static int TcRsSockQpCreate(int* fd, uint32_t* qpn, int* fd2, uint32_t* qpn2)
{
    int ret = 0;
    int i = 0;
    int tryNum = 10;
    uint32_t phyId = 0;
    int flag = 0; /* RC */
    int qpMode = 1;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listen[1] = {0};
    struct SocketConnectInfo conn[2] = {0};
    struct SocketFdData socketInfo[3] = {0};
    struct SocketWlistInfoT whiteList = {0};
    struct RsQpResp resp = {0};
    struct RsQpResp resp2 = {0};
    whiteList.remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    whiteList.connLimit = 1;
    unsigned int rdevIndex = 0;
    struct RsQpNorm qpNorm = {0};

    rs_ut_msg("resource prepare begin..................\n");

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("RS INIT, ret:%d !\n", ret);

    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 1);
    rs_ut_msg("RS LISTEN, ret:%d !\n", ret);
    strcpy(whiteList.tag, "1234");

    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");
    RsSocketWhiteListAdd(rdevInfo, &whiteList, 1);

    conn[0].phyId = 0;
    conn[0].family = AF_INET;
    conn[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    conn[0].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(conn[0].tag, "1234");
    conn[0].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 1);
    rs_ut_msg("RS CONNECT, ret:%d !\n", ret);

    usleep(SLEEP_TIME);

    i = 0;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo[i].tag, "1234");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [client]socket_info[0].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    qpNorm.flag = flag;
    qpNorm.qpMode = qpMode;
    qpNorm.isExp = 1;

    ret = RsQpCreate(phyId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_EQ(ret, 0);
    *qpn = resp.qpn;
    rs_ut_msg("RS CREATE QP: QPN:%d, ret:%d\n", *qpn, ret);

    ret = RsQpConnectAsync(phyId, rdevIndex, *qpn, socketInfo[i].fd);
    *fd = socketInfo[i].fd;
    rs_ut_msg("RS QP CONNECT ASYNC: ret:%d\n", ret);

    usleep(SLEEP_TIME);

    i = 1;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo[i].tag, "1234");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_SERVER, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [server]socket_info[1].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    ret = RsQpCreate(phyId, rdevIndex, qpNorm, &resp2);
    EXPECT_INT_EQ(ret, 0);
    *qpn2 = resp2.qpn;
    rs_ut_msg("RS CREATE QP: QPN:%d, ret:%d\n", *qpn2, ret);

    ret = RsQpConnectAsync(phyId, rdevIndex, *qpn2, socketInfo[i].fd);
    *fd2 = socketInfo[i].fd;

    usleep(SLEEP_TIME * 10);

    rs_ut_msg("++++++++++++++ RS QP PREPARE DONE ++++++++++++++\n");

    return ret;
}

static int TcRsSockQpDestroy(int fd, uint32_t qpn, int fd2, uint32_t qpn2)
{
    int ret = 0;
    uint32_t phyId = 0;
    uint32_t rdevIndex = 0;
    struct RsInitConfig cfg = {0};
    struct RsSocketCloseInfoT sockClose[2] = {0};
    struct SocketListenInfo listen[1] = {0};
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");

    rs_ut_msg("resource free begin..................\n");
    usleep(SLEEP_TIME);

    ret = RsQpDestroy(phyId, rdevIndex, qpn2);
    EXPECT_INT_EQ(ret, 0);
    ret = RsQpDestroy(phyId, rdevIndex, qpn);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("RS destroy QP: ret:%d\n", ret);

    sockClose[0].fd = fd;
    ret = RsSocketBatchClose(0, &sockClose[0], 1);
    rs_ut_msg("RS socket close fd:%d, ret:%d\n", fd, ret);

    sockClose[1].fd = fd2;
    ret = RsSocketBatchClose(0, &sockClose[1], 1);
    rs_ut_msg("RS socket2 close fd:%d, ret:%d\n", fd2, ret);

    /* ------resource CLEAN-------- */
    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 1);
    rs_ut_msg("RS socket listen stop: ret:%d\n", ret);

    ret = RsRdevDeinit(phyId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsSocketDeinit(rdevInfo);
    EXPECT_INT_EQ(ret, 0);

    cfg.chipId = 0;
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    rs_ut_msg("resource free done..................\n");

    return ret;
}

void TcRsMrCreate()
{
    int ret = 0;
    uint32_t phyId = 0;
    uint32_t rdevIndex = 0;
    int flag = 0; /* RC */
    uint32_t qpn, qpn2;
    int fd, fd2;
    void *addr, *addr2;
    int tryNum = 0;
    struct RdmaMrRegInfo mrRegInfo = {0};

    /* +++++Resource Prepare+++++ */
    ret = TcRsSockQpCreate(&fd, &qpn, &fd2, &qpn2);

    addr = malloc(RS_TEST_MEM_SIZE);
    addr2 = malloc(8192);

    tryNum = 3;
    do {
        mrRegInfo.addr = addr;
        mrRegInfo.len = RS_TEST_MEM_SIZE;
        mrRegInfo.access = RS_ACCESS_LOCAL_WRITE;
        ret = RsMrReg(phyId, rdevIndex, qpn, &mrRegInfo);
        EXPECT_INT_EQ(ret, 0);
        if (0 == ret)
            break;
        rs_ut_msg("MR REG1: qpn %d, ret:%d\n", qpn, ret);
        tryNum--;
        sleep(1);
    } while (tryNum && (-EAGAIN == ret));
    EXPECT_INT_EQ(ret, 0);

    /* repeat reg */
    mrRegInfo.addr = addr;
    mrRegInfo.len = RS_TEST_MEM_SIZE;
    mrRegInfo.access = RS_ACCESS_LOCAL_WRITE;
    ret = RsMrReg(phyId, rdevIndex, qpn, &mrRegInfo);
    EXPECT_INT_EQ(ret, 0);

    tryNum = 3;
    do {
        mrRegInfo.addr = addr2;
        mrRegInfo.len = 8192;
        mrRegInfo.access = RS_ACCESS_LOCAL_WRITE;
        ret = RsMrReg(phyId, rdevIndex, qpn2, &mrRegInfo);
        if (0 == ret)
            break;
        rs_ut_msg("MR REG2: qpn2 %d, ret:%d\n", qpn2, ret);
        tryNum--;
        sleep(1);
    } while (tryNum && (-EAGAIN == ret));
    EXPECT_INT_EQ(ret, 0);

    usleep(SLEEP_TIME);

    /* free resource */
    rs_ut_msg("RS MR dereg begin...\n");
    ret = RsMrDereg(phyId, rdevIndex, qpn, addr);
    EXPECT_INT_EQ(ret, 0);
    ret = RsMrDereg(phyId, rdevIndex, qpn2, addr2);
    EXPECT_INT_EQ(ret, 0);

    free(addr);
    free(addr2);

    /* +++++Resource Free+++++ */
    ret = TcRsSockQpDestroy(fd, qpn, fd2, qpn2);
    EXPECT_INT_EQ(ret, 0);

    return;
}
struct RsMrCb* gMrCbA;

int StubRsGetMrcbA(struct RsQpCb* qpCb, uint64_t addr, struct RsMrCb** mrCb, struct RsListHead* mrList)
{
    *mrCb = gMrCbA;
    return -1;
}

void TcRsMrAbnormal()
{
    int ret = 0;
    uint32_t phyId = 0;
    uint32_t rdevIndex = 0;
    int flag = 0; /* RC */
    uint32_t qpn, qpn2;
    int fd, fd2;
    void *addr, *addr2;
    struct RdmaMrRegInfo mrRegInfo = {0};

    /* +++++Resource Prepare+++++ */
    ret = TcRsSockQpCreate(&fd, &qpn, &fd2, &qpn2);

    addr = malloc(RS_TEST_MEM_SIZE);
    addr2 = malloc(RS_TEST_MEM_SIZE);

    mrRegInfo.addr = addr;
    mrRegInfo.len = RS_TEST_MEM_SIZE;
    mrRegInfo.access = RS_ACCESS_LOCAL_WRITE;
    ret = RsMrReg(phyId, rdevIndex, 999999, &mrRegInfo);
    EXPECT_INT_NE(ret, 0);

    mrRegInfo.addr = 0;
    mrRegInfo.len = RS_TEST_MEM_SIZE;
    mrRegInfo.access = RS_ACCESS_LOCAL_WRITE;
    ret = RsMrReg(phyId, rdevIndex, qpn, &mrRegInfo);
    EXPECT_INT_NE(ret, 0);

    ret = RsMrDereg(phyId, rdevIndex, 999999, addr);
    EXPECT_INT_NE(ret, 0);

    ret = RsMrDereg(phyId, rdevIndex, qpn2, 0);
    EXPECT_INT_NE(ret, 0);

    mocker(RsQpn2qpcb, 10, 0);
    mocker_invoke((stub_fn_t)RsGetMrcb, StubRsGetMrcbA, 1);
    ret = RsMrDereg(phyId, rdevIndex, 999999, addr2);
    EXPECT_INT_EQ(ret, -EFAULT);
    mocker_clean();

    free(addr);
    free(addr2);

    /* +++++Resource Free+++++ */
    ret = TcRsSockQpDestroy(fd, qpn, fd2, qpn2);
    EXPECT_INT_EQ(ret, 0);

    return;
}

void TcRsAbnormal2()
{
    int ret = 0;
    int flag = 0; /* RC */
    uint32_t qpn, qpn2;
    int fd, fd2;
    struct rs_cb *rsCb, *rsCb2;
    char cmd, cmd2;
    struct epoll_event events = {0};

    /* +++++Resource Prepare+++++ */
    ret = TcRsSockQpCreate(&fd, &qpn, &fd2, &qpn2);

    /* ABNORMAL TC Start */
    rs_ut_msg("\n+++++++++ABNORMAL TC Start++++++++\n");
    ret = RsDev2rscb(0, &rsCb2, false);
    EXPECT_INT_EQ(ret, 0);
    usleep(1000);
    events.data.fd = 200;
    RsEpollEventInHandle(rsCb2, &events);
    events.data.fd = -1;
    RsEpollEventInHandle(rsCb2, &events);
    events.data.fd = 0;
    RsEpollEventInHandle(rsCb2, &events);
    rs_ut_msg("---------ABNORMAL TC End----------\n\n");
    /* ABNORMAL TC End */

    /* +++++Resource Free+++++ */
    ret = TcRsSockQpDestroy(fd, qpn, fd2, qpn2);
    EXPECT_INT_EQ(ret, 0);

    return;
}

struct RsQpCb* qpCb2;
void TcRsCqHandle()
{
    int ret = 0;
    uint32_t qpn, qpn2;
    int fd, fd2;
    struct RsQpCb qpCb4 = {0};
    unsigned int phyId = 0;
    unsigned int rdevIndex = 0;

    /* +++++Resource Prepare+++++ */
    ret = TcRsSockQpCreate(&fd, &qpn, &fd2, &qpn2);

    qpCb4.channel = NULL;
    RsDrvPollCqHandle(&qpCb4);

    ret = RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCbAb2);
    EXPECT_INT_EQ(ret, 0);
    RsDrvPollCqHandle(qpCbAb2);

    struct RsQpCb qpcbTmp = {0};
    struct ibv_wc wc = {0};
    struct ibv_cq evCqSq = {0};
    struct ibv_cq evCqRq = {0};
    struct RsRdevCb rdevCb = {0};

    qpcbTmp.ibSendCq = &evCqSq;
    qpcbTmp.ibRecvCq = &evCqRq;
    qpcbTmp.rdevCb = &rdevCb;

    RsCqeCallbackProcess(&qpcbTmp, &wc, &evCqSq);
    RsCqeCallbackProcess(&qpcbTmp, &wc, &evCqSq);

    RsCqeCallbackProcess(&qpcbTmp, &wc, &evCqRq);

    wc.status = IBV_WC_WR_FLUSH_ERR;
    RsCqeCallbackProcess(&qpcbTmp, &wc, &evCqRq);

    wc.status = IBV_WC_WR_FLUSH_ERR;
    RsCqeCallbackProcess(&qpcbTmp, &wc, &evCqRq);

    wc.status = IBV_WC_WR_FLUSH_ERR;
    RsCqeCallbackProcess(&qpcbTmp, &wc, &evCqRq);

    wc.status = IBV_WC_SUCCESS;
    RsCqeCallbackProcess(&qpcbTmp, &wc, &evCqRq);

    wc.status = IBV_WC_MW_BIND_ERR;
    RsCqeCallbackProcess(&qpcbTmp, &wc, &evCqRq);

    /* +++++Resource Free+++++ */
    ret = TcRsSockQpDestroy(fd, qpn, fd2, qpn2);
    EXPECT_INT_EQ(ret, 0);

    return;
}

void TcRsEpollHandle()
{
    int ret = 0;
    uint32_t qpn, qpn2;
    int fd, fd2;
    struct epoll_event events3 = {0};

    /* +++++Resource Prepare+++++ */
    ret = TcRsSockQpCreate(&fd, &qpn, &fd2, &qpn2);

    events3.events = 0;
    RsEpollEventHandleOne(NULL, &events3);

    /* +++++Resource Free+++++ */
    ret = TcRsSockQpDestroy(fd, qpn, fd2, qpn2);
    EXPECT_INT_EQ(ret, 0);

    return;
}

int stub_halGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t* value)
{
    *value = 1;
    return 0;
}

void TcRsSocketOps()
{
    int ret = 0;
    int i = 0;
    int tryNum = 10;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listen[2] = {0};
    struct SocketConnectInfo conn[2] = {0};
    struct RsSocketCloseInfoT sockClose[2] = {0};
    struct SocketFdData socketInfo[3] = {0};
    struct SocketWlistInfoT whiteList = {0};
    whiteList.remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    whiteList.connLimit = 1;

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    mocker((stub_fn_t)system, 10, 0);
    mocker((stub_fn_t)access, 10, 0);
    mocker_invoke((stub_fn_t)halGetDeviceInfo, stub_halGetDeviceInfo, 10);
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 1);

    /* >>>>>>> RsSocketListenStart test case begin <<<<<<<<<<< */
    /* param error - close_info NULL */
    ret = RsSocketListenStart(NULL, 1);
    EXPECT_INT_NE(ret, 0);

    /* param error - num = 0 */
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 0);
    EXPECT_INT_NE(ret, 0);

    /* param error - fd */
    listen[0].phyId = 64;
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 1);
    listen[0].phyId = 0;

    /* repeat listen */
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 1);
    /* >>>>>>> RsSocketListenStart test case end <<<<<<<<<<< */

    strcpy_s(whiteList.tag, SOCK_CONN_TAG_SIZE, "1234");
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");
    RsSocketWhiteListAdd(rdevInfo, &whiteList, 1);
    strcpy_s(whiteList.tag, SOCK_CONN_TAG_SIZE, "5678");
    RsSocketWhiteListAdd(rdevInfo, &whiteList, 1);

    conn[0].phyId = 0;
    conn[0].family = AF_INET;
    conn[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    conn[0].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy_s(conn[0].tag, SOCK_CONN_TAG_SIZE, "1234");
    conn[1].phyId = 0;
    conn[1].family = AF_INET;
    conn[1].localIp.addr.s_addr = inet_addr("127.0.0.1");
    conn[1].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy_s(conn[1].tag, SOCK_CONN_TAG_SIZE, "5678");
    conn[0].port = 16666;
    conn[1].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 2);

    usleep(SLEEP_TIME);

    i = 0;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy_s(socketInfo[i].tag, SOCK_CONN_TAG_SIZE, "1234");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [client]socket_info[0].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    sockClose[i].fd = socketInfo[i].fd;
    ret = RsSocketBatchClose(0, &sockClose[i], 1);

    i = 1;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy_s(socketInfo[i].tag, SOCK_CONN_TAG_SIZE, "5678");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [client]socket_info[1].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    /* >>>>>>> RsSocketSend test case begin <<<<<<<<<<< */
    int data = 0;
    int size = sizeof(data);
    /* param error */
    ret = RsSocketSend(socketInfo[i].fd, NULL, 0);
    EXPECT_INT_NE(ret, 0);
    ret = RsPeerSocketSend(0, socketInfo[i].fd, NULL, 0);
    EXPECT_INT_NE(ret, 0);
    ret = RsPeerSocketSend(1, socketInfo[i].fd, NULL, 0);
    EXPECT_INT_NE(ret, 0);

    /* fd error */
    ret = RsSocketSend(1111, &data, size);
    EXPECT_INT_NE(ret, 0);
    ret = RsPeerSocketSend(0, 1111, &data, size);
    EXPECT_INT_NE(ret, 0);
    ret = RsPeerSocketSend(1, 1111, &data, size);
    EXPECT_INT_NE(ret, 0);
    /* >>>>>>> RsSocketSend test case end <<<<<<<<<<< */

    /* >>>>>>> RsSocketRecv test case begin <<<<<<<<<<< */
    /* param error */
    ret = RsSocketRecv(socketInfo[i].fd, NULL, 0);
    EXPECT_INT_NE(ret, 0);
    ret = RsPeerSocketRecv(0, socketInfo[i].fd, NULL, 0);
    EXPECT_INT_NE(ret, 0);
    ret = RsPeerSocketRecv(1, socketInfo[i].fd, NULL, 0);
    EXPECT_INT_NE(ret, 0);
    ret = RsPeerSocketRecv(1, 1111, &data, size);
    EXPECT_INT_NE(ret, 0);
    ret = RsPeerSocketRecv(0, 1111, &data, size);
    EXPECT_INT_NE(ret, 0);
    /* >>>>>>> RsSocketRecv test case end <<<<<<<<<<< */

    /* >>>>>>> RsSocketBatchClose test case begin <<<<<<<<<<< */
    /* param error - close_info NULL */
    ret = RsSocketBatchClose(0, NULL, 1);
    EXPECT_INT_NE(ret, 0);

    /* param error - num = 0 */
    ret = RsSocketBatchClose(0, &sockClose[i], 0);
    EXPECT_INT_NE(ret, 0);

    /* param error - fd */
    sockClose[i].fd = -1;
    ret = RsSocketBatchClose(0, &sockClose[i], 1);
    EXPECT_INT_NE(ret, 0);
    /* >>>>>>> RsSocketBatchClose test case end <<<<<<<<<<< */

    sockClose[i].fd = socketInfo[i].fd;
    ret = RsSocketBatchClose(0, &sockClose[i], 1);

    /* close a non-exist fd */
    ret = RsSocketBatchClose(0, &sockClose[i], 1);

    usleep(1000);

    /* get Server Conn & Close it */
    i = 0;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy_s(socketInfo[i].tag, SOCK_CONN_TAG_SIZE, "1234");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_SERVER, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg(
        "%s [server]socket_info[0].fd:%d, client if:0x%x, status:%d\n", __func__, socketInfo[i].fd,
        socketInfo[i].remoteIp.addr.s_addr, socketInfo[i].status);

    sockClose[i].fd = socketInfo[i].fd;
    ret = RsSocketBatchClose(0, &sockClose[i], 1);

    /* >>>>>>> RsSocketListenStop test case begin <<<<<<<<<<< */
    /* param error - close_info NULL */
    ret = RsSocketListenStop(NULL, 1);
    EXPECT_INT_NE(ret, 0);

    /* param error - num = 0 */
    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 0);
    EXPECT_INT_NE(ret, 0);

    /* param error - fd */
    listen[0].phyId = 64;
    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 1);
    EXPECT_INT_NE(ret, 0);
    listen[0].phyId = 0;
    /* >>>>>>> RsSocketListenStop test case end <<<<<<<<<<< */

    /* ------Resource CLEAN-------- */
    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 1);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    return;
}

int ReplaceRsQpn2qpcb(unsigned int phyId, unsigned int rdevIndex, uint32_t qpn, struct RsQpCb** qpCb)
{
    static struct RsQpCb aQpCb = {0};

    *qpCb = &aQpCb;
    aQpCb.state = 1;
    return 0;
}

void TcRsGetQpStatus()
{
    int ret = 0;
    unsigned int phyId = 0;
    unsigned int rdevIndex = 0;
    uint32_t qpn, qpn2;
    int fd, fd2;
    struct RsQpStatusInfo status = {0};

    struct RsQpCb qpCb = {0};
    struct rs_cb rsCb = {0};
    struct RsRdevCb rdevCb = {0};
    qpCb.rdevCb = &rdevCb;

    mocker((stub_fn_t)ibv_query_port, 1, 1);
    ret = RsDrvSetMtu(&qpCb);
    EXPECT_INT_EQ(ret, -EOPENSRC);
    mocker_clean();

    /* +++++Resource Prepare+++++ */
    mocker((stub_fn_t)RsDrvSetMtu, 10, 5);
    ret = TcRsSockQpCreate(&fd, &qpn, &fd2, &qpn2);
    mocker_clean();

    mocker_invoke(RsQpn2qpcb, ReplaceRsQpn2qpcb, 1);
    mocker(RsRoceQueryQpc, 10, 1);
    ret = RsGetQpStatus(phyId, rdevIndex, qpn, &status);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    /* +++++Resource Free+++++ */
    ret = TcRsSockQpDestroy(fd, qpn, fd2, qpn2);
    EXPECT_INT_EQ(ret, 0);

    return;
}

void TcRsGetNotifyBa()
{
    int ret = 0;
    uint32_t qpn, qpn2;
    int fd, fd2;
    unsigned int phyId = 0;
    struct MrInfoT info = {0};
    unsigned int rdevIndex = 0;

    /* +++++Resource Prepare+++++ */
    ret = TcRsSockQpCreate(&fd, &qpn, &fd2, &qpn2);

    ret = RsGetNotifyMrInfo(phyId, rdevIndex, &info);
    EXPECT_INT_EQ(ret, 0);

    ret = RsGetNotifyMrInfo(100000, rdevIndex, &info);
    EXPECT_INT_NE(ret, 0);

    ret = RsGetNotifyMrInfo(phyId, rdevIndex, NULL);
    EXPECT_INT_NE(ret, 0);

    /* +++++Resource Free+++++ */
    ret = TcRsSockQpDestroy(fd, qpn, fd2, qpn2);
    EXPECT_INT_EQ(ret, 0);

    return;
}

void TcRsSetupSharemem()
{
    struct RsRdevCb rdevCb = {0};
    struct rs_cb rsCb = {0};
    int ret = 0;

    DlHalInit();

    ret = RsBindHostpid(0, getpid());
    EXPECT_INT_NE(ret, 0);

    rsCb.hccpMode = NETWORK_OFFLINE;
    mocker_invoke((stub_fn_t)DlHalGetDeviceInfo, dl_hal_get_device_info_910A, 20);
    ret = RsSetupSharemem(&rsCb, false, 0);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
    ret = RsSetupSharemem(&rsCb, false, 0);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    rsCb.grpSetupFlag = false;
    mocker_invoke((stub_fn_t)DlHalGetDeviceInfo, DlHalGetDeviceInfoSharemem, 20);
    ret = RsSetupSharemem(&rsCb, false, 0);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    rsCb.grpSetupFlag = false;
    mocker_invoke((stub_fn_t)DlHalGetDeviceInfo, DlHalGetDeviceInfoSharemem, 20);
    mocker_invoke((stub_fn_t)DlHalQueryDevPid, DlHalQueryDevPidSharemem, 20);
    ret = RsSetupSharemem(&rsCb, false, 0);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    rsCb.grpSetupFlag = false;
    mocker_invoke((stub_fn_t)DlHalGetDeviceInfo, DlHalGetDeviceInfoSharemem, 20);
    ret = RsSetupSharemem(&rsCb, true, 0);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    ret = RsOpenBackupIbCtx(&rdevCb);
    EXPECT_INT_NE(ret, 0);

    DlHalDeinit();
}

void TcRsPostRecv()
{
    int ret = 0;
    uint32_t qpn, qpn2;
    int fd, fd2;
    uint32_t size = 0;
    int tryNum = 0;
    void *addr, *addr2;
    struct RdmaMrRegInfo mrRegInfo = {0};
    unsigned int phyId = 0;
    unsigned int rdevIndex = 0;

    /* +++++Resource Prepare+++++ */
    ret = TcRsSockQpCreate(&fd, &qpn, &fd2, &qpn2);

    addr = malloc(RS_TEST_MEM_SIZE);
    addr2 = malloc(RS_TEST_MEM_SIZE);

    tryNum = 3;
    do {
        mrRegInfo.addr = addr;
        mrRegInfo.len = RS_TEST_MEM_SIZE;
        mrRegInfo.access = RS_ACCESS_LOCAL_WRITE;
        ret = RsMrReg(phyId, rdevIndex, qpn, &mrRegInfo);
        EXPECT_INT_EQ(ret, 0);
        if (0 == ret)
            break;
        rs_ut_msg("MR REG1: qpn %d, ret:%d\n", qpn, ret);
        tryNum--;
        sleep(1);
    } while (tryNum && (-EAGAIN == ret));
    EXPECT_INT_EQ(ret, 0);

    tryNum = 3;
    do {
        mrRegInfo.addr = addr2;
        mrRegInfo.len = RS_TEST_MEM_SIZE;
        mrRegInfo.access = RS_ACCESS_LOCAL_WRITE;
        ret = RsMrReg(phyId, rdevIndex, qpn2, &mrRegInfo);
        if (0 == ret)
            break;
        rs_ut_msg("MR REG2: qpn2 %d, ret:%d\n", qpn2, ret);
        tryNum--;
        sleep(1);
    } while (tryNum && (-EAGAIN == ret));
    EXPECT_INT_EQ(ret, 0);

    usleep(1000);

    /* free resource */
    rs_ut_msg("RS MR dereg begin...\n");
    ret = RsMrDereg(phyId, rdevIndex, qpn, addr);
    EXPECT_INT_EQ(ret, 0);
    ret = RsMrDereg(phyId, rdevIndex, qpn2, addr2);
    EXPECT_INT_EQ(ret, 0);

    free(addr);
    free(addr2);

    /* +++++Resource Free+++++ */
    ret = TcRsSockQpDestroy(fd, qpn, fd2, qpn2);
    EXPECT_INT_EQ(ret, 0);

    return;
}

void TcRsSendWrlistExp()
{
    int ret = 0;
    uint32_t qpn, qpn2;
    int fd, fd2;
    uint32_t index = 0;
    struct SgList list = {0};
    struct WrInfo wrlist[1] = {{0}};
    int tryNum = 0;
    void *addr, *addr2;
    struct SendWrRsp rsWrInfo[1] = {{0}};
    uint32_t phyId = 0;
    uint32_t rdevIndex = 0;
    struct RsWrlistBaseInfo baseInfo = {0};
    unsigned int sendNum = 1;
    unsigned int completeNum = 0;
    /* +++++Resource Prepare+++++ */
    ret = TcRsSockQpCreate(&fd, &qpn, &fd2, &qpn2);

    addr = malloc(RS_TEST_MEM_SIZE);
    addr2 = malloc(RS_TEST_MEM_SIZE);

    struct RdmaMrRegInfo mrRegInfo = {0};
    mrRegInfo.addr = addr;
    mrRegInfo.len = RS_TEST_MEM_SIZE;
    mrRegInfo.access = RS_ACCESS_LOCAL_WRITE;

    baseInfo.phyId = phyId;
    baseInfo.rdevIndex = rdevIndex;
    baseInfo.qpn = qpn;
    baseInfo.keyFlag = 0;
    tryNum = 3;
    do {
        ret = RsMrReg(phyId, rdevIndex, qpn, &mrRegInfo);
        EXPECT_INT_EQ(ret, 0);
        if (0 == ret)
            break;
        rs_ut_msg("MR REG1: qpn %d, ret:%d\n", qpn, ret);
        tryNum--;
        sleep(1);
    } while (tryNum && (-EAGAIN == ret));
    EXPECT_INT_EQ(ret, 0);

    mrRegInfo.addr = addr2;
    tryNum = 3;
    do {
        ret = RsMrReg(phyId, rdevIndex, qpn2, &mrRegInfo);
        if (0 == ret)
            break;
        rs_ut_msg("MR REG2: qpn2 %d, ret:%d\n", qpn2, ret);
        tryNum--;
        sleep(1);
    } while (tryNum && (-EAGAIN == ret));
    EXPECT_INT_EQ(ret, 0);

    struct RsMrCb* addr2MrCb;
    addr2MrCb = calloc(1, sizeof(struct RsMrCb));
    addr2MrCb->mrInfo.cmd = RS_CMD_MR_INFO;
    addr2MrCb->mrInfo.addr = (uint64_t)mrRegInfo.addr;
    addr2MrCb->mrInfo.len = mrRegInfo.len;
    addr2MrCb->mrInfo.rkey = mrRegInfo.lkey;

    struct RsQpCb* qpCb = NULL;
    RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    RsListAddTail(&addr2MrCb->list, &qpCb->remMrList);

    usleep(1000);

    list.addr = (uint64_t)addr;
    list.len = RS_TEST_MEM_SIZE;
    wrlist[0].memList = list;
    wrlist[0].dstAddr = (uint64_t)addr2;
    wrlist[0].op = 0;
    wrlist[0].sendFlags = RS_SEND_FENCE;

    tryNum = 3;
    do {
        ret = RsSendWrlist(baseInfo, wrlist, sendNum, rsWrInfo, &completeNum);
        if (0 == ret)
            break;
        usleep(SLEEP_TIME);
    } while (tryNum-- && ret == -2);
    EXPECT_INT_EQ(ret, 0);

    wrlist[0].dstAddr = 0;
    tryNum = 3;
    do {
        ret = RsSendWrlist(baseInfo, wrlist, sendNum, rsWrInfo, &completeNum);
        if (0 == ret)
            break;
        usleep(SLEEP_TIME);
    } while (tryNum-- && ret == -2);
    EXPECT_INT_EQ(ret, -ENOENT);
    wrlist[0].dstAddr = (uint64_t)addr2;

    wrlist[0].memList.addr = 0;
    tryNum = 3;
    do {
        ret = RsSendWrlist(baseInfo, wrlist, sendNum, rsWrInfo, &completeNum);
        if (0 == ret)
            break;
        usleep(SLEEP_TIME);
    } while (tryNum-- && ret == -2);
    EXPECT_INT_EQ(ret, -EFAULT);
    wrlist[0].memList.addr = (uint64_t)addr;

    list.len = 2147483649;
    wrlist[0].memList = list;
    tryNum = 3;
    do {
        ret = RsSendWrlist(baseInfo, wrlist, sendNum, rsWrInfo, &completeNum);
        if (0 == ret)
            break;
        usleep(SLEEP_TIME);
    } while (tryNum-- && ret == -2);
    EXPECT_INT_EQ(ret, -EINVAL);
    /* free resource */
    rs_ut_msg("RS MR dereg begin...\n");
    ret = RsMrDereg(phyId, rdevIndex, qpn, addr);
    EXPECT_INT_EQ(ret, 0);
    ret = RsMrDereg(phyId, rdevIndex, qpn2, addr2);
    EXPECT_INT_EQ(ret, 0);

    free(addr);
    free(addr2);

    /* +++++Resource Free+++++ */
    ret = TcRsSockQpDestroy(fd, qpn, fd2, qpn2);
    EXPECT_INT_EQ(ret, 0);

    return;
}

void TcRsSendWrlistNormal()
{
    int ret = 0;
    uint32_t qpn, qpn2;
    int fd, fd2;
    uint32_t index = 0;
    struct SgList list = {0};
    struct WrInfo wrlist[1] = {{0}};
    int tryNum = 0;
    void *addr, *addr2;
    struct SendWrRsp rsWrInfo[1] = {{0}};
    uint32_t phyId = 0;
    uint32_t rdevIndex = 0;
    struct RsWrlistBaseInfo baseInfo = {0};
    unsigned int sendNum = 1;
    unsigned int completeNum = 0;
    /* +++++Resource Prepare+++++ */
    ret = TcRsSockQpCreateNormal(&fd, &qpn, &fd2, &qpn2);

    addr = malloc(RS_TEST_MEM_SIZE);
    addr2 = malloc(RS_TEST_MEM_SIZE);

    struct RdmaMrRegInfo mrRegInfo = {0};
    mrRegInfo.addr = addr;
    mrRegInfo.len = RS_TEST_MEM_SIZE;
    mrRegInfo.access = RS_ACCESS_LOCAL_WRITE;

    baseInfo.phyId = phyId;
    baseInfo.rdevIndex = rdevIndex;
    baseInfo.qpn = qpn;
    baseInfo.keyFlag = 0;
    tryNum = 3;
    do {
        ret = RsMrReg(phyId, rdevIndex, qpn, &mrRegInfo);
        EXPECT_INT_EQ(ret, 0);
        if (0 == ret)
            break;
        rs_ut_msg("MR REG1: qpn %d, ret:%d\n", qpn, ret);
        tryNum--;
        sleep(1);
    } while (tryNum && (-EAGAIN == ret));
    EXPECT_INT_EQ(ret, 0);

    mrRegInfo.addr = addr2;
    tryNum = 3;
    do {
        ret = RsMrReg(phyId, rdevIndex, qpn2, &mrRegInfo);
        if (0 == ret)
            break;
        rs_ut_msg("MR REG2: qpn2 %d, ret:%d\n", qpn2, ret);
        tryNum--;
        sleep(1);
    } while (tryNum && (-EAGAIN == ret));
    EXPECT_INT_EQ(ret, 0);

    struct RsMrCb* addr2MrCb;
    addr2MrCb = calloc(1, sizeof(struct RsMrCb));
    addr2MrCb->mrInfo.cmd = RS_CMD_MR_INFO;
    addr2MrCb->mrInfo.addr = (uint64_t)mrRegInfo.addr;
    addr2MrCb->mrInfo.len = mrRegInfo.len;
    addr2MrCb->mrInfo.rkey = mrRegInfo.lkey;

    struct RsQpCb* qpCb = NULL;
    RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    RsListAddTail(&addr2MrCb->list, &qpCb->remMrList);

    usleep(1000);

    list.addr = (uint64_t)addr;
    list.len = RS_TEST_MEM_SIZE;
    list.addr = (uint64_t)addr;
    list.len = RS_TEST_MEM_SIZE;
    wrlist[0].memList = list;
    wrlist[0].dstAddr = (uint64_t)addr2;
    wrlist[0].op = 0;
    wrlist[0].sendFlags = RS_SEND_FENCE;

    mocker(RsQpn2qpcb, 1, -1);
    ret = RsSendWrlist(baseInfo, wrlist, sendNum, rsWrInfo, &completeNum);
    EXPECT_INT_EQ(ret, -EACCES);
    mocker_clean();

    tryNum = 3;
    do {
        ret = RsSendWrlist(baseInfo, wrlist, sendNum, rsWrInfo, &completeNum);
        if (0 == ret)
            break;
        usleep(SLEEP_TIME);
    } while (tryNum-- && ret == -2);
    EXPECT_INT_EQ(ret, 0);

    wrlist[0].dstAddr = 0;
    tryNum = 3;
    do {
        ret = RsSendWrlist(baseInfo, wrlist, sendNum, rsWrInfo, &completeNum);
        if (0 == ret)
            break;
        usleep(SLEEP_TIME);
    } while (tryNum-- && ret == -2);
    EXPECT_INT_EQ(ret, -ENOENT);

    wrlist[0].dstAddr = (uint64_t)addr2;
    wrlist[0].memList.addr = 0;
    tryNum = 3;
    do {
        ret = RsSendWrlist(baseInfo, wrlist, sendNum, rsWrInfo, &completeNum);
        if (0 == ret)
            break;
        usleep(SLEEP_TIME);
    } while (tryNum-- && ret == -2);
    EXPECT_INT_EQ(ret, -EFAULT);
    wrlist[0].memList.addr = (uint64_t)addr;

    tryNum = 3;
    do {
        ret = RsSendWrlist(baseInfo, wrlist, 1028, rsWrInfo, &completeNum);
        if (0 == ret)
            break;
        usleep(SLEEP_TIME);
    } while (tryNum-- && ret == -2);
    EXPECT_INT_EQ(ret, -EINVAL);

    /* free resource */
    rs_ut_msg("RS MR dereg begin...\n");
    ret = RsMrDereg(phyId, rdevIndex, qpn, addr);
    EXPECT_INT_EQ(ret, 0);
    ret = RsMrDereg(phyId, rdevIndex, qpn2, addr2);
    EXPECT_INT_EQ(ret, 0);

    free(addr);
    free(addr2);

    /* +++++Resource Free+++++ */
    ret = TcRsSockQpDestroy(fd, qpn, fd2, qpn2);
    EXPECT_INT_EQ(ret, 0);

    return;
}

void TcRsSendWr()
{
    int ret = 0;
    uint32_t qpn, qpn2;
    int fd, fd2;
    uint32_t index = 0;
    struct SgList list[2] = {{0}};
    struct SendWr wr = {0};
    int tryNum = 0;
    void *addr, *addr2;
    struct wr_exp_rsp rsWrInfo = {0};
    unsigned int phyId = 0;
    unsigned int rdevIndex = 0;
    struct RdmaMrRegInfo mrRegInfo = {0};

    /* +++++Resource Prepare+++++ */
    ret = TcRsSockQpCreate(&fd, &qpn, &fd2, &qpn2);

    addr = malloc(RS_TEST_MEM_SIZE);
    addr2 = malloc(RS_TEST_MEM_SIZE);

    tryNum = 3;
    do {
        mrRegInfo.addr = addr;
        mrRegInfo.len = RS_TEST_MEM_SIZE;
        mrRegInfo.access = RS_ACCESS_LOCAL_WRITE;
        ret = RsMrReg(phyId, rdevIndex, qpn, &mrRegInfo);
        if (0 == ret)
            break;
        rs_ut_msg("MR REG1: qpn %d, ret:%d\n", qpn, ret);
        tryNum--;
        sleep(1);
    } while (tryNum && (-EAGAIN == ret));

    tryNum = 3;
    do {
        mrRegInfo.addr = addr2;
        mrRegInfo.len = RS_TEST_MEM_SIZE;
        mrRegInfo.access = RS_ACCESS_LOCAL_WRITE;
        ret = RsMrReg(phyId, rdevIndex, qpn2, &mrRegInfo);
        if (0 == ret)
            break;
        rs_ut_msg("MR REG2: qpn2 %d, ret:%d\n", qpn2, ret);
        tryNum--;
        sleep(1);
    } while (tryNum && (-EAGAIN == ret));
    EXPECT_INT_EQ(ret, 0);

    struct RsMrCb* addr2MrCb;
    addr2MrCb = calloc(1, sizeof(struct RsMrCb));
    addr2MrCb->mrInfo.cmd = RS_CMD_MR_INFO;
    addr2MrCb->mrInfo.addr = (uint64_t)mrRegInfo.addr;
    addr2MrCb->mrInfo.len = mrRegInfo.len;
    addr2MrCb->mrInfo.rkey = mrRegInfo.lkey;

    struct RsQpCb* qpCb = NULL;
    RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    RsListAddTail(&addr2MrCb->list, &qpCb->remMrList);

    usleep(1000);

    list[0].addr = (uint64_t)addr;
    list[0].len = RS_TEST_MEM_SIZE;
    list[1].addr = (uint64_t)addr;
    list[1].len = RS_TEST_MEM_SIZE;
    wr.bufList = list;
    wr.bufNum = 2;
    wr.dstAddr = (uint64_t)addr2;
    wr.op = 0;
    wr.sendFlag = RS_SEND_FENCE;

    tryNum = 3;
    do {
        ret = RsSendWr(phyId, rdevIndex, qpn, &wr, (struct SendWrRsp*)&rsWrInfo);
        if (0 == ret)
            break;
        usleep(SLEEP_TIME);
    } while (tryNum-- && ret == -2);
    EXPECT_INT_EQ(ret, 0);

    /* RDMA Write with Notify Test */
    tryNum = 3;
    wr.op = 0x16;
    do {
        ret = RsSendWr(phyId, rdevIndex, qpn, &wr, (struct SendWrRsp*)&rsWrInfo);
        if (ret == 0)
            break;
        usleep(SLEEP_TIME);
    } while (tryNum-- && ret == -2);
    EXPECT_INT_EQ(ret, 0);

    /* qpn error */
    ret = RsSendWr(phyId, rdevIndex, 44444, &wr, (struct SendWrRsp*)&rsWrInfo);
    EXPECT_INT_NE(ret, 0);

    wr.bufNum = MAX_SGE_NUM + 1;
    ret = RsSendWr(phyId, rdevIndex, qpn, &wr, (struct SendWrRsp*)&rsWrInfo);
    EXPECT_INT_NE(ret, 0);
    wr.bufNum = 2;

    list[0].len = 2147483649;
    list[1].len = 2147483649;
    ret = RsSendWr(phyId, rdevIndex, qpn, &wr, (struct SendWrRsp*)&rsWrInfo);
    EXPECT_INT_NE(ret, 0);

    list[0].len = RS_TEST_MEM_SIZE;
    list[1].len = RS_TEST_MEM_SIZE;
    /* addr error, cannot find mrcb */
    list[0].addr = 5555;
    ret = RsSendWr(phyId, rdevIndex, qpn, &wr, (struct SendWrRsp*)&rsWrInfo);
    EXPECT_INT_NE(ret, 0);
    list[0].addr = (uint64_t)addr;

    /* addr error, cannot find remote mrcb */
    wr.dstAddr = 5555;
    ret = RsSendWr(phyId, rdevIndex, qpn, &wr, (struct SendWrRsp*)&rsWrInfo);
    EXPECT_INT_NE(ret, 0);
    wr.dstAddr = (uint64_t)addr2;

    /* free resource */
    rs_ut_msg("RS MR dereg begin...\n");
    ret = RsMrDereg(phyId, rdevIndex, qpn, addr);
    EXPECT_INT_EQ(ret, 0);
    ret = RsMrDereg(phyId, rdevIndex, qpn2, addr2);
    EXPECT_INT_EQ(ret, 0);

    free(addr);
    free(addr2);

    /* +++++Resource Free+++++ */
    ret = TcRsSockQpDestroy(fd, qpn, fd2, qpn2);
    EXPECT_INT_EQ(ret, 0);

    return;
}

void TcRsDfx()
{
    int ret = 0;
    uint32_t qpn, qpn2;
    int fd, fd2;
    struct RsQpCb qpCb4 = {0};

    /* +++++Resource Prepare+++++ */
    ret = TcRsSockQpCreate(&fd, &qpn, &fd2, &qpn2);

    /* +++++Resource Free+++++ */
    ret = TcRsSockQpDestroy(fd, qpn, fd2, qpn2);
    EXPECT_INT_EQ(ret, 0);

    return;
}

void TcRsWhiteList()
{
    int ret = 0;
    int tryNum = 10;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listen = {0};
    struct SocketConnectInfo conn = {0};
    struct SocketConnectInfo conn1 = {0};

    struct RsSocketCloseInfoT sockClose = {0};
    struct RsSocketCloseInfoT sockClose1 = {0};

    struct SocketFdData socketInfo = {0};
    struct SocketFdData socketInfo1 = {0};
    struct SocketWlistInfoT whiteList = {0};
    struct SocketWlistInfoT whiteList1 = {0};
    u32 serverIp = inet_addr("127.0.0.1");

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    listen.phyId = 0;
    listen.family = AF_INET;
    listen.localIp.addr.s_addr = inet_addr("127.0.0.1");
    listen.port = 18888;
    ret = RsSocketListenStart(&listen, 1);

    conn.phyId = 0;
    conn.family = AF_INET;
    conn.remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    conn.localIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(conn.tag, "LinkCheck");
    conn.port = 18888;
    ret = RsSocketBatchConnect(&conn, 1);
    EXPECT_INT_EQ(ret, 0);

    conn1.phyId = 0;
    conn1.family = AF_INET;
    conn1.remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    conn1.localIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(conn1.tag, "2345");
    conn1.port = 18888;
    sleep(1);
    whiteList.remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    whiteList.connLimit = 1;
    strcpy(whiteList.tag, "LinkCheck");
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = serverIp;
    RsSocketWhiteListAdd(rdevInfo, &whiteList, 1);

    whiteList1.remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    whiteList1.connLimit = 1;
    strcpy(whiteList1.tag, "2345");
    RsSocketWhiteListAdd(rdevInfo, &whiteList1, 1);

    socketInfo.phyId = 0;
    socketInfo.family = AF_INET;
    socketInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo.remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo.tag, "LinkCheck");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo, 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [client]socket_info[0].fd:%d, status:%d\n", __func__, socketInfo.fd, socketInfo.status);

    RsSocketWhiteListDel(rdevInfo, &whiteList, 1);
    RsSocketWhiteListDel(rdevInfo, &whiteList1, 1);
    sockClose.fd = socketInfo.fd;
    ret = RsSocketBatchClose(0, &sockClose, 1);

    sockClose1.fd = socketInfo1.fd;

    listen.port = 18888;
    ret = RsSocketListenStop(&listen, 1);

    ret = RsSocketDeinit(rdevInfo);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    return;
}

void TcRsSslTest1()
{
    int ret = 0;
    int tryNum = 10;
    uint32_t devId = 0;
    int flag = 0; /* RC */
    uint32_t qpn, qpn2;
    int i = 0;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listen[2] = {0};
    struct SocketConnectInfo conn[2] = {0};
    struct RsSocketCloseInfoT sockClose[2] = {0};
    struct SocketFdData socketInfo[3] = {0};
    struct SocketWlistInfoT whiteList = {0};
    whiteList.remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    whiteList.connLimit = 1;
    uint32_t sslEnable = 1;

    ret = RsGetSslEnable(NULL);
    EXPECT_INT_NE(ret, 0);

    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    ret = RsGetSslEnable(NULL);
    EXPECT_INT_NE(ret, 0);

    ret = RsGetSslEnable(&sslEnable);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(sslEnable, 0);

    /* pridata tls */
    struct rs_cb rscb = {0};
    struct RsSecPara rsPara = {0};
    struct tls_cert_mng_info mngInfo = {0};

    mngInfo.work_key_len = 10;
    mngInfo.ky_len = 512;
    mngInfo.ky_enc_len = 1678;
    mngInfo.pwd_enc_len = 15;
    mngInfo.pwd_len = 16;

    ret = rs_get_pridata(&rscb, &rsPara, &mngInfo);
    EXPECT_INT_EQ(ret, 0);

    mocker(dev_read_flash, 10, -1);
    ret = rs_get_pridata(&rscb, &rsPara, &mngInfo);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker(kmc_dec_data, 10, -1);
    ret = rs_get_pridata(&rscb, &rsPara, &mngInfo);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker(crypto_decrypt_with_aes_gcm, 10, -1);
    ret = rs_get_pridata(&rscb, &rsPara, &mngInfo);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker(crypto_gen_key_with_pbkdf2, 10, -1);
    ret = rs_get_pridata(&rscb, &rsPara, &mngInfo);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)memset_s, 10, 1);
    ret = rs_get_pridata(&rscb, &rsPara, &mngInfo);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)memset_s, 10, -1);
    ret = rs_get_pridata(&rscb, &rsPara, &mngInfo);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    /* tls pridata end */

    struct rs_cb* tagRsCb = NULL;
    ret = RsGetRsCb(cfg.chipId, &tagRsCb);
    tagRsCb->sslEnable = 1;
    struct RsConnInfo tagConn = {0};
    tagConn.connfd = -1;
    RsSocketTagSync(&tagConn);

    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    listen[0].port = 26666;
    ret = RsSocketListenStart(&listen[0], 1);

    strcpy(whiteList.tag, "1234");
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");
    RsSocketWhiteListAdd(rdevInfo, &whiteList, 1);

    conn[0].phyId = 0;
    conn[0].family = AF_INET;
    conn[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    conn[0].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(conn[0].tag, "1234");
    conn[0].port = 26666;
    ret = RsSocketBatchConnect(&conn[0], 1);
    usleep(SLEEP_TIME);

    i = 0;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo[i].tag, "1234");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [client]socket_info[0].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    usleep(SLEEP_TIME);

    i = 1;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo[i].tag, "1234");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_SERVER, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [server]socket_info[1].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    usleep(SLEEP_TIME);

    rs_ut_msg("++++++++++++++ RS QP PREPARE DONE ++++++++++++++\n");

    ret = RsSocketDeinit(rdevInfo);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    return;
}

extern int RsFillIfaddrInfos(struct IfaddrInfo ifaddrInfos[], unsigned int* num, unsigned int phyId);
extern int RsFillIfaddrInfosV2(struct InterfaceInfo interfaceInfos[], unsigned int* num, unsigned int phyId);
extern enum RsHardwareType RsGetDeviceType(unsigned int phyId);
extern int RsCheckDstInterface(unsigned int phyId, const char* ifaName, enum RsHardwareType type, bool isAll);
extern int snprintf_s(char* strDest, size_t destMax, size_t count, const char* format, ...);
extern int RsFillIfnum(unsigned int phyId, bool isAll, unsigned int* num, unsigned int isPeer);

void TcRsGetInterfaceVersion()
{
    int version = 0;
    int ret = RsGetInterfaceVersion(0, NULL);
    EXPECT_INT_EQ(ret, -EINVAL);

    ret = RsGetInterfaceVersion(0, &version);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsRoceGetApiVersion, 1, 0);
    ret = RsGetInterfaceVersion(96, &version);
    EXPECT_INT_EQ(ret, 0);
}

int StubDlHalGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t* value)
{
    *value = 0x10;
    return 0;
}

int StubDlHalGetDeviceInfoPod(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t* value)
{
    *value = 0x30;
    return 0;
}

int StubDlHalGetDeviceInfoPod16p(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t* value)
{
    *value = 0x50;
    return 0;
}

void TcRsGetIfaddrs()
{
    DlHalInit();
    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->hccpMode = 1;
    int ret = 0;
    int phyId = 0;
    unsigned int ifaddrNum = 4;
    struct IfaddrInfo ifaddrInfos[4] = {0};
    bool isAll = false;

    ret = RsGetIfaddrs(ifaddrInfos, &ifaddrNum, phyId);
    EXPECT_INT_EQ(ret, 0);

    mocker((stub_fn_t)RsFillIfaddrInfos, 1, 1);
    ret = RsGetIfaddrs(ifaddrInfos, &ifaddrNum, phyId);
    EXPECT_INT_EQ(ret, 1);
    mocker_clean();

    ret = RsGetIfaddrs(NULL, &ifaddrNum, phyId);
    EXPECT_INT_EQ(ret, -EINVAL);

    ret = RsGetIfaddrs(ifaddrInfos, &ifaddrNum, 129);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker((stub_fn_t)RsGetDeviceType, 1, 2);
    ret = RsFillIfaddrInfos(ifaddrInfos, &ifaddrNum, phyId);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker_invoke((stub_fn_t)DlHalGetDeviceInfo, StubDlHalGetDeviceInfoPod, 10);
    ret = RsGetDeviceType(0);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker_invoke((stub_fn_t)DlHalGetDeviceInfo, StubDlHalGetDeviceInfoPod16p, 10);
    ret = RsGetDeviceType(0);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)DlHalGetDeviceInfo, 1, -1);
    ret = RsGetDeviceType(0);
    EXPECT_INT_EQ(ret, 3);
    mocker_clean();

    mocker_invoke((stub_fn_t)DlHalGetDeviceInfo, StubDlHalGetDeviceInfo, 10);
    ret = RsGetDeviceType(0);
    EXPECT_INT_EQ(ret, 1);
    mocker_clean();

    mocker((stub_fn_t)rsGetLocalDevIDByHostDevID, 20, 0);
    mocker_invoke((stub_fn_t)DlHalGetDeviceInfo, dl_hal_get_device_info_910A, 20);
    ret = RsGetDeviceType(0);
    EXPECT_INT_EQ(ret, 1);

    mocker((stub_fn_t)getifaddrs, 1, -1);
    ret = RsFillIfaddrInfos(ifaddrInfos, &ifaddrNum, phyId);
    EXPECT_INT_EQ(ret, -ESYSFUNC);
    mocker_clean();

    mocker((stub_fn_t)RsCheckDstInterface, 1, -1);
    ret = RsFillIfaddrInfos(ifaddrInfos, &ifaddrNum, phyId);
    EXPECT_INT_EQ(ret, -EAGAIN);
    mocker_clean();

    ifaddrNum = 0;
    ret = RsFillIfaddrInfos(ifaddrInfos, &ifaddrNum, phyId);

    ret = RsCheckDstInterface(phyId, "eth0", 1, isAll);
    EXPECT_INT_EQ(ret, 1);

    mocker((stub_fn_t)strncmp, 2, 2);
    ret = RsCheckDstInterface(phyId, "eth0", 1, isAll);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)snprintf_s, 2, -1);
    ret = RsCheckDstInterface(phyId, "eth0", 0, isAll);
    EXPECT_INT_EQ(ret, -EAGAIN);
    mocker_clean();

    mocker((stub_fn_t)snprintf_s, 2, 2);
    ret = RsCheckDstInterface(phyId, "bond0", 0, isAll);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)snprintf_s, 2, -1);
    ret = RsCheckDstInterface(phyId, "bond0", 0, isAll);
    EXPECT_INT_EQ(ret, -EAGAIN);
    mocker_clean();

    free(gRsCb);
    gRsCb = NULL;
    DlHalDeinit();
    return;
}

#ifndef IFF_UP
enum {
    IFF_UP = 1 << 0,
    IFF_RUNNING = 1 << 6,
};
#endif

int StubGetifaddrs(struct ifaddrs** ifap)
{
    *ifap = malloc(sizeof(struct ifaddrs));
    if (*ifap == NULL)
        return -ENOMEM;
    memset(*ifap, 0, sizeof(struct ifaddrs));

    struct sockaddr_in* saAddr = malloc(sizeof(struct sockaddr_in));
    if (saAddr == NULL)
        return -ENOMEM;
    memset(saAddr, 0, sizeof(struct sockaddr_in));

    struct sockaddr_in* saNetmask = malloc(sizeof(struct sockaddr_in));
    if (saNetmask == NULL)
        return -ENOMEM;
    memset(saNetmask, 0, sizeof(struct sockaddr_in));

    saAddr->sin_family = AF_INET;
    saAddr->sin_port = 0;
    inet_pton(AF_INET, "192.168.100.50", &(saAddr->sin_addr));

    saNetmask->sin_family = AF_INET;
    saNetmask->sin_port = 0;
    inet_pton(AF_INET, "255.255.255.0", &(saNetmask->sin_addr));
    (*ifap)->ifa_addr = (struct sockaddr*)saAddr;
    (*ifap)->ifa_netmask = (struct sockaddr*)saNetmask;

    (*ifap)->ifa_next = NULL;
    (*ifap)->ifa_name = "eth0";
    (*ifap)->ifa_flags = IFF_UP | IFF_RUNNING;

    return 0;
}

void StubFreeifaddrs(struct ifaddrs* ifa)
{
    if (!ifa)
        return;
    if (ifa->ifa_next)
        StubFreeifaddrs(ifa->ifa_next);
    if (ifa->ifa_addr)
        free(ifa->ifa_addr);
    if (ifa->ifa_netmask)
        free(ifa->ifa_netmask);
    if (ifa->ifa_ifu.ifu_broadaddr)
        free(ifa->ifa_ifu.ifu_broadaddr);
    free(ifa);
}

void TcRsGetIfaddrsV2()
{
    DlHalInit();
    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->hccpMode = 1;
    int ret = 0;
    int phyId = 0;
    unsigned int ifaddrNum = 4;
    struct InterfaceInfo interfaceInfos[4] = {0};
    bool isAll = false;

    ret = RsGetIfaddrsV2(interfaceInfos, &ifaddrNum, phyId, isAll);
    EXPECT_INT_EQ(ret, 0);

    mocker((stub_fn_t)RsFillIfaddrInfosV2, 1, 1);
    ret = RsGetIfaddrsV2(interfaceInfos, &ifaddrNum, phyId, isAll);
    EXPECT_INT_EQ(ret, 1);
    mocker_clean();

    ret = RsGetIfaddrsV2(NULL, &ifaddrNum, phyId, 0);
    EXPECT_INT_EQ(ret, -EINVAL);

    ret = RsGetIfaddrsV2(interfaceInfos, &ifaddrNum, 129, isAll);
    EXPECT_INT_EQ(ret, -EINVAL);

    ifaddrNum = 1;
    mocker((stub_fn_t)RsGetDeviceType, 1, 2);
    mocker_invoke(getifaddrs, StubGetifaddrs, 10);
    mocker_invoke(freeifaddrs, StubFreeifaddrs, 10);
    ret = RsFillIfaddrInfosV2(interfaceInfos, &ifaddrNum, phyId);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker_invoke((stub_fn_t)DlHalGetDeviceInfo, StubDlHalGetDeviceInfoPod, 10);
    ret = RsGetDeviceType(0);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)getifaddrs, 1, -1);
    ret = RsFillIfaddrInfosV2(interfaceInfos, &ifaddrNum, phyId);
    EXPECT_INT_EQ(ret, -ESYSFUNC);
    mocker_clean();

    mocker((stub_fn_t)RsCheckDstInterface, 1, -1);
    ret = RsFillIfaddrInfosV2(interfaceInfos, &ifaddrNum, phyId);
    EXPECT_INT_EQ(ret, -EAGAIN);
    mocker_clean();

    ifaddrNum = 0;
    ret = RsFillIfaddrInfosV2(interfaceInfos, &ifaddrNum, phyId);

    ret = RsCheckDstInterface(phyId, "eth0", 1, isAll);
    EXPECT_INT_EQ(ret, 1);

    mocker((stub_fn_t)strncmp, 2, 2);
    ret = RsCheckDstInterface(phyId, "eth0", 1, isAll);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)snprintf_s, 2, -1);
    ret = RsCheckDstInterface(phyId, "eth0", 0, isAll);
    EXPECT_INT_EQ(ret, -EAGAIN);
    mocker_clean();
    free(gRsCb);
    gRsCb = NULL;
    DlHalDeinit();
    return;
}

void TcRsPeerGetIfaddrs()
{
    int ret = 0;
    int phyId = 0;
    unsigned int ifaddrNum = 1000;
    unsigned int ifNum = 1000;
    struct InterfaceInfo ifaddrInfos[1000] = {0};
    struct RsInitConfig cfg = {0};
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("RS INIT, ret:%d !\n", ret);

    ret = RsPeerGetIfnum(phyId, &ifNum);
    EXPECT_INT_EQ(ret, 0);

    ret = RsPeerGetIfaddrs(ifaddrInfos, &ifaddrNum, phyId);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(ifNum, ifaddrNum);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
}

void TcRsGetIfnum()
{
    DlHalInit();
    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->hccpMode = 1;
    int ret = 0;
    int phyId = 0;
    unsigned int ifnum = 0;
    bool isAll = false;

    ret = RsGetIfnum(phyId, isAll, &ifnum);
    EXPECT_INT_EQ(ret, 0);

    mocker((stub_fn_t)RsFillIfnum, 1, 1);
    ret = RsGetIfnum(phyId, isAll, &ifnum);
    EXPECT_INT_EQ(ret, 1);
    mocker_clean();

    ret = RsGetIfnum(phyId, isAll, NULL);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker((stub_fn_t)getifaddrs, 1, -1);
    ret = RsFillIfnum(phyId, isAll, &ifnum, 1);
    EXPECT_INT_EQ(ret, -ESYSFUNC);
    mocker_clean();

    mocker((stub_fn_t)RsCheckDstInterface, 1, -1);
    ret = RsFillIfnum(phyId, isAll, &ifnum, 0);
    EXPECT_INT_EQ(ret, -EAGAIN);
    mocker_clean();

    free(gRsCb);
    gRsCb = NULL;
    DlHalDeinit();
    return;
}

void TcRsGetCurTime()
{
    struct timeval time = {0};
    mocker(gettimeofday, 20, 1);
    mocker(memset_s, 20, 1);
    RsGetCurTime(&time);
    mocker_clean();
    return;
}

void tc_RsRdev2rdevCb()
{
    struct RsRdevCb* rdevCb = NULL;
    mocker(RsDev2rscb, 20, 0);
    mocker(RsGetRdevCb, 20, 1);
    RsRdev2rdevCb(1, 1, &rdevCb);
    mocker_clean();
    return;
}

void TcRsCompareIpGid()
{
    struct rdev rdevInfo = {0};
    union ibv_gid gid = {0};
    rdevInfo.family = 10;
    int ret = RsCompareIpGid(rdevInfo, &gid);
    EXPECT_INT_EQ(ret, -ENODEV);

    mocker(memcmp, 20, 0);
    ret = RsCompareIpGid(rdevInfo, &gid);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
    return;
}

void TcRsQueryGid()
{
    struct ibv_context ibCtxTmp = {0};
    struct RsRdevCb rdevCb = {0};
    struct rdev rdevInfo = {0};
    uint8_t ibPort = 1;
    int gidIdx = 0;
    int ret = 0;

    mocker(ibv_query_port, 20, 1);
    ret = RsQueryGid(rdevInfo, &ibCtxTmp, ibPort, &gidIdx);
    EXPECT_INT_EQ(ret, -EOPENSRC);
    mocker_clean();

    mocker(ibv_query_gid_type, 20, 1);
    ret = RsQueryGid(rdevInfo, &ibCtxTmp, ibPort, &gidIdx);
    EXPECT_INT_EQ(ret, -EOPENSRC);
    mocker_clean();

    mocker(ibv_query_gid, 20, 1);
    ret = RsQueryGid(rdevInfo, &ibCtxTmp, ibPort, &gidIdx);
    EXPECT_INT_EQ(ret, -EOPENSRC);
    mocker_clean();

    mocker(RsCompareIpGid, 20, 1);
    RsQueryGid(rdevInfo, &ibCtxTmp, ibPort, &gidIdx);
    mocker_clean();
    return;
}

void TcRsGetHostRdevIndex()
{
    struct RsRdevCb rdevCb = {0};
    struct rdev rdevInfo = {0};
    struct rs_cb rsCb = {0};
    int rdevIndex = 0;
    int ret = 0;

    rdevCb.devList = ibv_get_device_list(&(rdevCb.devNum));
    rdevCb.rsCb = &rsCb;
    mocker((stub_fn_t)pthread_mutex_lock, 20, 0);
    mocker((stub_fn_t)pthread_mutex_unlock, 20, 0);
    mocker_invoke(RsIbvGetDeviceName, stub_RsIbvGetDeviceName, 20);
    mocker(RsConvertIpAddr, 20, -EINVAL);
    ret = RsGetHostRdevIndex(rdevInfo, &rdevCb, &rdevIndex, 0);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();
}

void TcRsGetIbCtxAndRdevIndex()
{
    struct rdev rdevInfo = {0};
    int rdevIndex = 0;
    struct RsRdevCb rdevCb = {0};
    int ret = 0;

    rdevCb.devList = ibv_get_device_list(&(rdevCb.devNum));
    mocker(ibv_open_device, 20, NULL);
    ret = RsGetIbCtxAndRdevIndex(rdevInfo, &rdevCb, &rdevIndex);
    EXPECT_INT_EQ(ret, -ENODEV);
    mocker_clean();

    mocker(RsQueryGid, 20, -EEXIST);
    ret = RsGetIbCtxAndRdevIndex(rdevInfo, &rdevCb, &rdevIndex);
    EXPECT_INT_EQ(ret, -EEXIST);
    mocker_clean();

    mocker(RsQueryGid, 20, 1);
    ret = RsGetIbCtxAndRdevIndex(rdevInfo, &rdevCb, &rdevIndex);
    EXPECT_INT_EQ(ret, 1);
    mocker_clean();
    return;
}

void TcRsRdevCbInit()
{
    struct rdev rdevInfo = {0};
    struct rs_cb rsCb = {0};
    struct RsRdevCb rdevCb = {0};
    int rdevIndex = 0;
    mocker(RsInetNtop, 20, 0);
    mocker(pthread_mutex_init, 20, 1);
    int ret = RsRdevCbInit(rdevInfo, &rdevCb, &rsCb, &rdevIndex);
    EXPECT_INT_EQ(ret, -ESYSFUNC);
    mocker_clean();

    mocker(RsInetNtop, 20, 0);
    mocker(RsGetIbCtxAndRdevIndex, 20, 1);
    ret = RsRdevCbInit(rdevInfo, &rdevCb, &rsCb, &rdevIndex);
    EXPECT_INT_EQ(ret, 1);
    mocker_clean();

    mocker(RsInetNtop, 20, 0);
    mocker(RsGetIbCtxAndRdevIndex, 20, 0);
    mocker(RsGetSqDepthAndQpMaxNum, 20, 1);
    mocker(RsRoceUnmmapAiDbReg, 20, 1);
    ret = RsRdevCbInit(rdevInfo, &rdevCb, &rsCb, &rdevIndex);
    EXPECT_INT_EQ(ret, 1);
    mocker_clean();

    mocker(RsInetNtop, 20, 0);
    mocker(RsGetIbCtxAndRdevIndex, 20, 0);
    mocker(RsGetSqDepthAndQpMaxNum, 20, 0);
    mocker(RsRoceUnmmapAiDbReg, 20, 0);
    mocker(RsSetupPdAndNotify, 20, 1);
    ret = RsRdevCbInit(rdevInfo, &rdevCb, &rsCb, &rdevIndex);
    EXPECT_INT_EQ(ret, 1);
    mocker_clean();

    mocker(RsInetNtop, 20, 0);
    mocker(RsGetIbCtxAndRdevIndex, 20, 0);
    mocker(RsDrvQueryNotifyAndAllocPd, 20, 1);
    mocker(ibv_close_device, 20, 1);
    ret = RsRdevCbInit(rdevInfo, &rdevCb, &rsCb, &rdevIndex);
    EXPECT_INT_EQ(ret, 1);
    mocker_clean();

    mocker(RsInetNtop, 20, 0);
    mocker(RsGetIbCtxAndRdevIndex, 20, 0);
    mocker(RsDrvQueryNotifyAndAllocPd, 20, 0);
    mocker(RsDrvRegNotifyMr, 20, 1);
    mocker(ibv_dealloc_pd, 20, 1);
    mocker(ibv_close_device, 20, 1);
    ret = RsRdevCbInit(rdevInfo, &rdevCb, &rsCb, &rdevIndex);
    EXPECT_INT_EQ(ret, 1);
    mocker_clean();
    return;
}

void TcRsRdevInit()
{
    struct rdev rdevInfo = {0};
    int rdevIndex = 0;
    int ret = 0;
    mocker(RsGetRsCb, 20, 0);
    mocker_clean();

    mocker(RsGetRsCb, 20, 0);
    mocker(calloc, 20, NULL);
    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, -ENOMEM);
    mocker_clean();

    mocker(ibv_get_device_list, 20, NULL);
    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    mocker(RsApiInit, 1, -EINVAL);
    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    return;
}

void TcRsSslFree()
{
    struct rs_cb rscb = {0};
    struct RsCertSkidSubjectCb* skidSubjectCb = (struct RsCertSkidSubjectCb*)malloc(sizeof(struct RsCertSkidSubjectCb));
    SSL_CTX* clientSslCtx = (SSL_CTX*)malloc(sizeof(SSL_CTX));
    SSL_CTX* serverSslCtx = (SSL_CTX*)malloc(sizeof(SSL_CTX));
    rscb.sslEnable = 1;
    rscb.skidSubjectCb = skidSubjectCb;
    rscb.clientSslCtx = clientSslCtx;
    rscb.serverSslCtx = serverSslCtx;
    mocker(memset_s, 1, 1);
    RsSslFree(&rscb);
    mocker_clean();
    free(clientSslCtx);
    free(serverSslCtx);
    return;
}

void TcRsDrvConnect()
{
    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->hccpMode = 1;
    mocker(connect, 20, 1);
    RsDrvConnect(1, NULL, NULL, 1);
    mocker_clean();
    free(gRsCb);
    gRsCb = NULL;
    return;
}

void TcRsListenInvalidPort()
{
    int ret = 0;
    struct SocketListenInfo listen[2] = {0};
    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->hccpMode = 0;
    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    listen[0].port = 65536;
    ret = RsSocketListenStart(&listen[0], 1);
    EXPECT_INT_EQ(ret, -22);
    free(gRsCb);
    gRsCb = NULL;
    return;
}

void TcRsQpn2qpcb()
{
    struct RsQpCb* qpCb = NULL;
    RsQpn2qpcb(1, 1, 1, &qpCb);
    return;
}

void TcRsSslDeinit()
{
    struct rs_cb rscb = {0};
    rscb.sslEnable = 1;
    rscb.skidSubjectCb = malloc(sizeof(struct RsCertSkidSubjectCb));
    mocker(SSL_CTX_free, 20, 1);
    rs_ssl_deinit(&rscb);
    mocker_clean();
    free(rscb.skidSubjectCb);
    return;
}

void Tcrs_tls_inner_enable()
{
    struct rs_cb rscb = {0};
    rscb.sslEnable = 1;
    mocker(rs_ssl_inner_init, 20, 1);
    int ret = rs_tls_inner_enable(&rscb, 1);
    EXPECT_INT_EQ(1, ret);
    mocker_clean();
    return;
}

void TcRsSslInnerInit()
{
    struct rs_cb rscb = {0};
    mocker(SSL_CTX_new, 20, NULL);
    rscb.skidSubjectCb = malloc(sizeof(struct RsCertSkidSubjectCb));
    int ret = rs_ssl_inner_init(&rscb);
    EXPECT_INT_EQ(-ENOMEM, ret);
    mocker_clean();

    mocker(rs_ssl_ca_ky_init, 20, 1);
    ret = rs_ssl_inner_init(&rscb);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();
    return;
}

void TcRsSslCaKyInit()
{
    struct rs_cb rscb = {0};
    rscb.serverSslCtx = SSL_CTX_new(TLS_server_method());
    mocker(SSL_CTX_set_options, 20, 1);
    mocker(SSL_CTX_set_min_proto_version, 20, 0);
    mocker(SSL_CTX_set_cipher_list, 20, 1);
    mocker(rs_ssl_load_ca, 20, 0);
    mocker(rs_ssl_crl_init, 20, 0);
    mocker(rs_check_pridata, 20, 0);
    rs_ssl_ca_ky_init(rscb.serverSslCtx, &rscb);
    mocker_clean();

    mocker(SSL_CTX_set_options, 20, 1);
    mocker(SSL_CTX_set_min_proto_version, 20, 1);
    mocker(SSL_CTX_set_cipher_list, 20, 0);
    mocker(rs_ssl_load_ca, 20, 0);
    mocker(rs_ssl_crl_init, 20, 0);
    mocker(rs_check_pridata, 20, 0);
    rs_ssl_ca_ky_init(rscb.serverSslCtx, &rscb);
    mocker_clean();

    mocker(SSL_CTX_set_options, 20, 1);
    mocker(SSL_CTX_set_min_proto_version, 20, 1);
    mocker(SSL_CTX_set_cipher_list, 20, 1);
    mocker(rs_ssl_load_ca, 20, 1);
    mocker(rs_ssl_crl_init, 20, 0);
    mocker(rs_check_pridata, 20, 0);
    rs_ssl_ca_ky_init(rscb.serverSslCtx, &rscb);
    mocker_clean();

    mocker(SSL_CTX_set_options, 20, 1);
    mocker(SSL_CTX_set_min_proto_version, 20, 1);
    mocker(SSL_CTX_set_cipher_list, 20, 1);
    mocker(rs_ssl_load_ca, 20, 0);
    mocker(rs_ssl_crl_init, 20, 1);
    mocker(rs_check_pridata, 20, 0);
    rs_ssl_ca_ky_init(rscb.serverSslCtx, &rscb);
    mocker_clean();

    mocker(SSL_CTX_set_options, 20, 1);
    mocker(SSL_CTX_set_min_proto_version, 20, 1);
    mocker(SSL_CTX_set_cipher_list, 20, 1);
    mocker(rs_ssl_load_ca, 20, 0);
    mocker(rs_ssl_crl_init, 20, 0);
    mocker(rs_check_pridata, 20, 1);
    rs_ssl_ca_ky_init(rscb.serverSslCtx, &rscb);
    mocker_clean();

    mocker(SSL_CTX_set_options, 20, 1);
    mocker(SSL_CTX_set_min_proto_version, 20, 1);
    mocker(SSL_CTX_set_cipher_list, 20, 1);
    mocker(rs_ssl_load_ca, 20, 0);
    mocker(rs_ssl_crl_init, 20, 0);
    mocker(rs_check_pridata, 20, 0);
    rs_ssl_ca_ky_init(rscb.serverSslCtx, &rscb);
    mocker_clean();

    SSL_CTX_free(rscb.serverSslCtx);
    rscb.serverSslCtx = NULL;

    return;
}

void Tcrs_ssl_crl_init()
{
    SSL_CTX sslCtx;
    struct rs_cb rscb = {0};
    struct tls_cert_mng_info mngInfo = {0};
    int ret = 0;

    mocker(rs_ssl_get_crl_data, 20, 0);
    mocker(SSL_CTX_get_cert_store, 20, NULL);
    ret = rs_ssl_crl_init(&sslCtx, &rscb, &mngInfo);
    EXPECT_INT_EQ(-EFAULT, ret);
    mocker_clean();

    mocker(rs_ssl_get_crl_data, 20, 0);
    mocker(SSL_CTX_get_cert_store, 20, 1);
    mocker(X509_STORE_set_flags, 20, 0);
    ret = rs_ssl_crl_init(&sslCtx, &rscb, &mngInfo);
    EXPECT_INT_EQ(-EFAULT, ret);
    mocker_clean();

    mocker(rs_ssl_get_crl_data, 20, 0);
    mocker(SSL_CTX_get_cert_store, 20, 1);
    mocker(X509_STORE_add_crl, 20, 0);
    ret = rs_ssl_crl_init(&sslCtx, &rscb, &mngInfo);
    EXPECT_INT_EQ(-EFAULT, ret);
    mocker_clean();

    mocker(rs_ssl_get_crl_data, 20, 2);
    ret = rs_ssl_crl_init(&sslCtx, &rscb, &mngInfo);
    EXPECT_INT_EQ(2, ret);
    mocker_clean();
    return;
}

void Tcrs_check_pridata()
{
    SSL_CTX sslCtx;
    struct rs_cb rscb = {0};
    struct tls_cert_mng_info mngInfo = {0};
    int ret = 0;
    mocker(rs_get_pk, 20, 0);
    mocker(SSL_CTX_use_PrivateKey, 20, 0);
    rs_check_pridata(&sslCtx, &rscb, &mngInfo);

    mocker_clean();

    mocker(rs_get_pk, 20, 0);
    mocker(SSL_CTX_use_PrivateKey, 20, 1);
    mocker(SSL_CTX_check_private_key, 20, 0);
    ret = rs_check_pridata(&sslCtx, &rscb, &mngInfo);
    mocker_clean();

    mocker(rs_get_pk, 20, 0);
    mocker(SSL_CTX_use_PrivateKey, 20, 1);
    mocker(SSL_CTX_check_private_key, 20, 1);
    ret = rs_check_pridata(&sslCtx, &rscb, &mngInfo);
    return;
}

void Tcrs_ssl_load_ca()
{
    SSL_CTX sslCtx;
    struct rs_cb rscb = {0};
    struct tls_cert_mng_info mngInfo = {0};
    int ret = 0;

    mocker(NetCommGetSelfHome, 20, 1);
    ret = rs_ssl_load_ca(&sslCtx, &rscb, &mngInfo);
    EXPECT_INT_EQ(1, ret);
    mocker_clean();

    mocker(rs_ssl_get_ca_data, 20, 1);
    ret = rs_ssl_load_ca(&sslCtx, &rscb, &mngInfo);
    EXPECT_INT_EQ(1, ret);
    mocker_clean();

    mocker(rs_ssl_get_ca_data, 20, 0);
    mocker(SSL_CTX_load_verify_locations, 20, 0);
    ret = rs_ssl_load_ca(&sslCtx, &rscb, &mngInfo);
    mocker_clean();

    mocker(rs_ssl_get_ca_data, 20, 0);
    mocker(SSL_CTX_load_verify_locations, 20, 1);
    mocker(SSL_CTX_use_certificate_chain_file, 20, 0);
    ret = rs_ssl_load_ca(&sslCtx, &rscb, &mngInfo);
    mocker_clean();

    mocker(rs_ssl_get_ca_data, 20, 0);
    mocker(SSL_CTX_load_verify_locations, 20, 1);
    mocker(SSL_CTX_use_certificate_chain_file, 20, 1);
    mocker(rs_remove_certs, 20, 1);
    ret = rs_ssl_load_ca(&sslCtx, &rscb, &mngInfo);
    mocker_clean();

    return;
}

void Tcrs_ssl_get_ca_data()
{
    int ret = 0;
    char endFile = 0;
    char caFile = 0;
    struct rs_cb rscb = {0};
    struct tls_cert_mng_info mngInfo = {0};

    mocker(calloc, 20, NULL);
    ret = rs_ssl_get_ca_data(&rscb, &endFile, &caFile, &mngInfo);
    EXPECT_INT_EQ(-ENOMEM, ret);
    mocker_clean();

    mocker(rs_ssl_get_cert, 10, -1);
    ret = rs_ssl_get_ca_data(&rscb, &endFile, &caFile, &mngInfo);
    EXPECT_INT_EQ(ret, -EACCES);
    mocker_clean();

    mocker(rs_ssl_get_cert, 10, 0);
    mocker(rs_ssl_put_certs, 10, -1);
    ret = rs_ssl_get_ca_data(&rscb, &endFile, &caFile, &mngInfo);
    EXPECT_INT_EQ(ret, -EACCES);
    mocker_clean();

    mocker(rs_ssl_get_cert, 10, 0);
    mocker(rs_ssl_put_certs, 10, 0);
    mocker(memset_s, 10, 0);
    ret = rs_ssl_get_ca_data(&rscb, &endFile, &caFile, &mngInfo);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    return;
}

void Tcrs_ssl_get_crl_data1()
{
    struct rs_cb rscb = {0};
    FILE* fp = NULL;
    struct tls_cert_mng_info mngInfo = {0};
    X509_CRL* crl = NULL;
    int ret = 0;
    mocker(tls_get_user_config, 20, -2);
    ret = rs_ssl_get_crl_data(&rscb, fp, &mngInfo, &crl);
    EXPECT_INT_EQ(-ENODEV, ret);
    return;
}

void Tcrs_ssl_get_ca_data1()
{
    struct rs_cb rscb = {0};
    char endFile = 0;
    char caFile = 0;
    struct tls_cert_mng_info mngInfo = {0};
    int ret = 0;
    mocker(tls_get_user_config, 20, -2);
    ret = rs_ssl_get_ca_data(&rscb, &endFile, &caFile, &mngInfo);
    EXPECT_INT_EQ(-EACCES, ret);
    mocker_clean();
    return;
}

void Tcrs_ssl_put_certs()
{
    int ret = 0;
    struct rs_cb rscb = {0};
    struct CertFile fileName = {0};
    struct tls_cert_mng_info mngInfo = {0};
    struct RsCerts certs = {0};
    struct tls_ca_new_certs newCerts[RS_SSL_NEW_CERT_CB_NUM] = {{0}};

    mocker(rs_ssl_check_mng_and_cert_chain, 20, -1);
    ret = rs_ssl_put_certs(&rscb, &mngInfo, &certs, newCerts, &fileName);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(rs_ssl_check_mng_and_cert_chain, 20, 0);
    mocker(rs_ssl_put_cert_end_pem, 20, -1);
    ret = rs_ssl_put_certs(&rscb, &mngInfo, &certs, newCerts, &fileName);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(rs_ssl_check_mng_and_cert_chain, 20, 0);
    mocker(rs_ssl_put_cert_end_pem, 20, 0);
    mocker(rs_ssl_put_cert_ca_pem, 20, -1);
    ret = rs_ssl_put_certs(&rscb, &mngInfo, &certs, newCerts, &fileName);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();
    return;
}

void Tcrs_ssl_check_cert_chain()
{
    int ret = 0;
    struct tls_cert_mng_info mngInfo = {0};
    struct RsCerts certs = {0};
    struct tls_ca_new_certs newCerts = {{0}};

    mocker(X509_STORE_new, 20, NULL);
    ret = rs_ssl_check_cert_chain(&mngInfo, &certs, &newCerts);
    EXPECT_INT_EQ(ret, -ENOMEM);
    mocker_clean();

    mocker(X509_STORE_CTX_new, 20, NULL);
    ret = rs_ssl_check_cert_chain(&mngInfo, &certs, &newCerts);
    EXPECT_INT_EQ(ret, -ENOMEM);
    mocker_clean();

    mocker(rs_ssl_verify_cert_chain, 20, -1);
    ret = rs_ssl_check_cert_chain(&mngInfo, &certs, &newCerts);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(rs_ssl_verify_cert_chain, 20, 0);
    ret = rs_ssl_check_cert_chain(&mngInfo, &certs, &newCerts);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    return;
}

void Tcrs_ssl_skid_get_from_chain()
{
    struct tls_cert_mng_info mngInfo = {0};
    struct RsCerts certs = {0};
    struct rs_cb rscb = {0};
    struct tls_ca_new_certs newCerts[RS_SSL_NEW_CERT_CB_NUM] = {{0}};
    rscb.skidSubjectCb = NULL;

    mocker(calloc, 20, NULL);
    int ret = rs_ssl_skid_get_from_chain(&rscb, &mngInfo, &certs, newCerts);
    EXPECT_INT_EQ(-ENOMEM, ret);
    mocker_clean();

    mngInfo.cert_count = 2;
    mocker(tls_load_cert, 20, NULL);
    ret = rs_ssl_skid_get_from_chain(&rscb, &mngInfo, &certs, newCerts);
    EXPECT_INT_EQ(-22, ret);
    mocker_clean();

    mocker(rs_ssl_skids_subjects_get, 20, -1);
    mocker(memset_s, 20, -1);
    ret = rs_ssl_skid_get_from_chain(&rscb, &mngInfo, &certs, newCerts);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    return;
}

void Tcrs_ssl_verify_cert_chain()
{
    int ret = 0;
    X509_STORE_CTX ctx;
    X509_STORE store;
    struct RsCerts certs = {0};
    struct tls_cert_mng_info mngInfo = {0};
    struct tls_ca_new_certs newCerts[RS_SSL_NEW_CERT_CB_NUM] = {{0}};

    newCerts[0].ncert_count = 2;
    mocker(tls_load_cert, 20, NULL);
    ret = rs_ssl_verify_cert_chain(&ctx, &store, &certs, &mngInfo, newCerts);
    EXPECT_INT_EQ(ret, -22);
    mocker_clean();

    newCerts[0].ncert_count = 0;
    mocker(tls_load_cert, 20, NULL);
    ret = rs_ssl_verify_cert_chain(&ctx, &store, &certs, &mngInfo, newCerts);
    EXPECT_INT_EQ(ret, -22);
    mocker_clean();

    return;
}

void Tctls_get_cert_chain()
{
    X509_STORE store;
    struct RsCerts certs = {0};
    struct tls_cert_mng_info mngInfo = {0};
    mngInfo.cert_count = 2;
    mocker(tls_load_cert, 20, 0);

    int ret = tls_get_cert_chain(&store, &certs, &mngInfo);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();

    mocker(X509_STORE_add_cert, 20, 0);
    ret = tls_get_cert_chain(&store, &certs, &mngInfo);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();
    return;
}

void Tcrs_ssl_get_leaf_cert()
{
    struct RsCerts certs = {0};
    X509* leafCert;
    mocker(tls_load_cert, 20, NULL);
    int ret = rs_ssl_get_leaf_cert(&certs, &leafCert);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();
    return;
}

void Tctls_load_cert()
{
    char inbuf = 0;
    mocker(BIO_new_mem_buf, 20, NULL);
    X509* ret = tls_load_cert(&inbuf, 1, 1);
    EXPECT_ADDR_EQ(NULL, ret);
    mocker_clean();

    mocker(PEM_read_bio_X509, 20, NULL);
    ret = tls_load_cert(&inbuf, 1, 1);
    EXPECT_ADDR_EQ(NULL, ret);
    mocker_clean();

    ret = tls_load_cert(&inbuf, 1, 0);
    EXPECT_ADDR_EQ(NULL, ret);

    mocker(d2i_X509_bio, 20, NULL);
    ret = tls_load_cert(&inbuf, 1, 2);
    EXPECT_ADDR_EQ(NULL, ret);
    mocker_clean();
    return;
}

struct SSL {
    int fd;
};

void Tcrs_tls_peer_cert_verify()
{
    SSL ssl;
    int ret = rs_tls_peer_cert_verify(&ssl, gRsCb);
    EXPECT_INT_EQ(0, ret);

    mocker(SSL_get_verify_result, 20, 1);
    ret = rs_tls_peer_cert_verify(&ssl, gRsCb);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();

    mocker((stub_fn_t)SSL_get_verify_result, 10, X509_V_ERR_CERT_HAS_EXPIRED);
    ret = rs_tls_peer_cert_verify(&ssl, gRsCb);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
    return;
}

void Tcrs_ssl_err_string()
{
    rs_ssl_err_string(1, 1);
    return;
}

void TcRsServerSendWlistCheckResult()
{
    struct RsConnInfo conn = {0};
    int ret = 0;

    gRsCb = calloc(1, sizeof(struct rs_cb));

    ret = RsServerSendWlistCheckResult(&conn, 0);
    EXPECT_INT_NE(0, ret);

    ret = RsServerSendWlistCheckResult(&conn, 1);
    EXPECT_INT_NE(0, ret);
    free(gRsCb);
    gRsCb = NULL;
}

void TcRsDrvSslBindFd()
{
    struct RsConnInfo conn = {0};
    RsDrvSslBindFd(&conn, 1);
    return;
}

void tc_rs_socket_fill_wlist_by_phyID()
{
    struct SocketWlistInfoT whiteListNode = {0};
    struct RsConnInfo rsConn = {0};

    mocker(RsSocketIsVnicIp, 1, 1);
    rsConn.clientIp.family = AF_INET;
    rs_socket_fill_wlist_by_phyID(0, &whiteListNode, &rsConn);
    mocker_clean();
    return;
}

void TcRsGetVnicIp()
{
    unsigned int phyId = 0;
    unsigned int vnicIp = 0;
    int ret = 0;

    ret = RsGetVnicIp(phyId, &vnicIp);
    EXPECT_INT_EQ(0, ret);
}

int RsDev2rscb_stub(uint32_t devId, struct rs_cb** rsCb, bool initFlag)
{
    static struct rs_cb rsCbTmp = {0};
    *rsCb = &rsCbTmp;
    return 0;
}
void TcRsNotifyCfgSet()
{
    unsigned int devId = 0;
    unsigned long long va = 0x10000;
    unsigned long long size = 8192;
    int ret = 0;
    mocker(rsGetLocalDevIDByHostDevID, 1, 0);
    mocker_invoke(RsDev2rscb, RsDev2rscb_stub, 1);
    ret = RsNotifyCfgSet(devId, va, size);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    ret = RsNotifyCfgSet(129, va, size);
    EXPECT_INT_EQ(-EINVAL, ret);

    size = 8192;
    va = 0x10000;
    mocker(rsGetLocalDevIDByHostDevID, 1, -1);
    ret = RsNotifyCfgSet(devId, va, size);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    ret = RsNotifyCfgSet(devId, va, size);
}

void TcRsNotifyCfgGet()
{
    unsigned int devId = 0;
    unsigned long long va = 0;
    unsigned long long size = 0;
    int ret = 0;
    mocker(rsGetLocalDevIDByHostDevID, 1, 0);
    mocker_invoke(RsDev2rscb, RsDev2rscb_stub, 1);
    ret = RsNotifyCfgGet(devId, &va, &size);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    ret = RsNotifyCfgGet(129, &va, &size);
    EXPECT_INT_EQ(-EINVAL, ret);

    mocker(rsGetLocalDevIDByHostDevID, 1, -1);
    ret = RsNotifyCfgGet(devId, &va, &size);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    ret = RsNotifyCfgGet(devId, &va, &size);
}

void TcCryptoDecryptWithAesGcm() { return; }

void TcRsDrvQpNormalFail()
{
    struct RsQpCb qpCb = {0};
    struct ibv_qp ibQp = {0};
    int ret = 0;
    qpCb.ibQp = &ibQp;

    mocker(memset_s, 1, -1);
    ret = RsDrvQpNormal(&qpCb, 0);
    EXPECT_INT_EQ(-ESAFEFUNC, ret);
    mocker_clean();

    mocker_ret((stub_fn_t)memset_s, 0, 1, 1);
    ret = RsDrvQpNormal(&qpCb, 0);
    EXPECT_INT_EQ(-ESAFEFUNC, ret);
    mocker_clean();

    mocker(RsDrvNormalQpCreateInit, 1, -3);
    ret = RsDrvQpNormal(&qpCb, 0);
    EXPECT_INT_EQ(-3, ret);
    mocker_clean();

    mocker(RsDrvNormalQpCreateInit, 1, 0);
    mocker(RsIbvCreateQp, 1, NULL);
    ret = RsDrvQpNormal(&qpCb, 0);
    EXPECT_INT_EQ(-ENOMEM, ret);
    mocker_clean();

    mocker(RsDrvNormalQpCreateInit, 1, 0);
    mocker(RsIbvCreateQp, 1, 1);
    mocker(RsIbvQueryQp, 1, -1);
    mocker(RsIbvDestroyQp, 1, 0);
    ret = RsDrvQpNormal(&qpCb, 0);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();

    mocker(RsDrvNormalQpCreateInit, 1, 0);
    mocker(RsIbvQueryQp, 1, 0);
    mocker(RsDrvQpInfoRelated, 1, -1);
    mocker(RsIbvDestroyQp, 1, 0);
    ret = RsDrvQpNormal(&qpCb, 0);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    return;
}

extern void RsCloseRoceUserSo(void);
void tc_RsApiInit()
{
    RsRoceUserApiInit();
    RsCloseRoceUserSo();
    return;
}

void TcRsRecvWrlist()
{
    int ret = 0;
    struct RsWrlistBaseInfo baseInfo = {0};
    struct RecvWrlistData wr = {0};
    unsigned int recvNum = 0;
    unsigned int completeNum = 0;
    ret = RsRecvWrlist(baseInfo, &wr, recvNum, &completeNum);
    EXPECT_INT_EQ(-EINVAL, ret);

    recvNum = 1;
    mocker(RsQpn2qpcb, 1, -1);
    ret = RsRecvWrlist(baseInfo, &wr, recvNum, &completeNum);
    EXPECT_INT_EQ(-EACCES, ret);
    mocker_clean();

    mocker(RsQpn2qpcb, 1, 0);
    mocker(RsDrvPostRecv, 1, 0);
    ret = RsRecvWrlist(baseInfo, &wr, recvNum, &completeNum);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    return;
}

void TcRsDrvPostRecv()
{
    int ret = 0;
    struct RsQpCb qpCb = {0};
    struct ibv_qp ibQp = {0};
    struct RecvWrlistData wr = {0};
    unsigned int recvNum = 0;
    unsigned int completeNum = 0;

    qpCb.ibQp = &ibQp;

    ret = RsDrvPostRecv(&qpCb, &wr, recvNum, &completeNum);
    EXPECT_INT_EQ(-EINVAL, ret);

    recvNum = 1;
    mocker(RsIbvPostRecv, 1, 0);
    ret = RsDrvPostRecv(&qpCb, &wr, recvNum, &completeNum);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    mocker(RsIbvPostRecv, 1, -ENOMEM);
    ret = RsDrvPostRecv(&qpCb, &wr, recvNum, &completeNum);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    mocker(RsIbvPostRecv, 1, -1);
    ret = RsDrvPostRecv(&qpCb, &wr, recvNum, &completeNum);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(calloc, 1, NULL);
    ret = RsDrvPostRecv(&qpCb, &wr, recvNum, &completeNum);
    EXPECT_INT_EQ(-ENOSPC, ret);
    mocker_clean();
    return;
}

void TcRsDrvRegNotifyMr()
{
    int ret = 0;
    struct RsRdevCb rdevCb = {0};
    rdevCb.notifyType = NO_USE;

    ret = RsDrvRegNotifyMr(&rdevCb);
    EXPECT_INT_EQ(0, ret);

    rdevCb.notifyType = 1000;
    ret = RsDrvRegNotifyMr(&rdevCb);
    EXPECT_INT_EQ(-EINVAL, ret);

    rdevCb.notifyType = EVENTID;
    mocker(RsIbvExpRegMr, 1, NULL);
    ret = RsDrvRegNotifyMr(&rdevCb);
    EXPECT_INT_EQ(-EACCES, ret);
    mocker_clean();
    return;
}

void TcRsDrvQueryNotifyAndAllocPd()
{
    int ret = 0;
    struct RsRdevCb rdevCb = {0};
    rdevCb.notifyType = NOTIFY;
    rdevCb.backupInfo.backupFlag = true;

    mocker(RsOpenBackupIbCtx, 1, 0);
    mocker(RsIbvExpQueryNotify, 1, -1);
    ret = RsDrvQueryNotifyAndAllocPd(&rdevCb);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();
    return;
}

void TcRsSendNormalWrlist()
{
    int ret = 0;
    struct RsQpCb qpCb = {0};
    struct WrInfo wrList = {0};
    unsigned int sendNum = 1;
    unsigned int completeNum = 0;
    unsigned int keyFlag = 1;

    mocker(RsIbvPostSend, 1, 0);
    ret = RsSendNormalWrlist(&qpCb, &wrList, sendNum, &completeNum, keyFlag);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    wrList.memList.len = 0xffffffff;
    ret = RsSendNormalWrlist(&qpCb, &wrList, sendNum, &completeNum, keyFlag);
    EXPECT_INT_EQ(-EINVAL, ret);
    return;
}

void TcRsDrvSendExp()
{
    struct RsQpCb qpCb = {0};
    struct RsMrCb mrCb = {0};
    struct RsMrCb remMrCb = {0};
    struct SendWr wr = {0};
    struct SendWrRsp wrRsp = {0};
    int ret = 0;

    mocker(RsIbvExpPostSend, 1, 0);
    qpCb.qpMode = 2;
    ret = RsDrvSendExp(&qpCb, &mrCb, &remMrCb, &wr, &wrRsp);
    mocker_clean();

    mocker(RsIbvExpPostSend, 1, 0);
    qpCb.qpMode = 1;
    ret = RsDrvSendExp(&qpCb, &mrCb, &remMrCb, &wr, &wrRsp);
    mocker_clean();
    return;
}

void TcRsDrvNormalQpCreateInit()
{
    struct ibv_qp_init_attr qpInitAttr = {0};
    struct RsQpCb qpCb = {0};
    struct ibv_port_attr attr = {0};
    struct rs_cb rsCb = {0};
    struct RsRdevCb rdevCb = {0};
    int ret = 0;
    qpCb.rdevCb = &rdevCb;
    qpCb.rdevCb->rsCb = &rsCb;
    qpCb.rdevCb->rsCb->hccpMode == NETWORK_PEER_ONLINE;

    qpCb.txDepth = 10;
    qpCb.rxDepth = 10;

    mocker(memset_s, 10, 0);
    qpCb.qpMode = 2;
    ret = RsDrvNormalQpCreateInit(&qpInitAttr, &qpCb, &attr);
    qpCb.rdevCb->rsCb->hccpMode == NETWORK_OFFLINE;
    ret = RsDrvNormalQpCreateInit(&qpInitAttr, &qpCb, &attr);
    mocker_clean();

    return;
}

void TcRsRegisterMr()
{
    int ret = 0;
    unsigned int phyId = 0;
    unsigned int rdevIndex = 0;
    unsigned int qpn = 0;

    struct RsInitConfig cfg = {0};

    rs_ut_msg("resource prepare begin..................\n");

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;

    struct RdmaMrRegInfo mrRegInfo = {0};
    mrRegInfo.addr = (void*)0xabcdef;
    mrRegInfo.len = RS_TEST_MEM_SIZE;
    mrRegInfo.access = RS_ACCESS_LOCAL_WRITE;

    struct RdmaMrRegInfo mrRegInfo1 = {0};
    mrRegInfo1.addr = (void*)0xabcdef;
    mrRegInfo1.len = RS_TEST_MEM_SIZE;
    mrRegInfo1.access = RS_ACCESS_LOCAL_WRITE;

    gRsCb = malloc(sizeof(struct rs_cb));
    memset(gRsCb, 0xff, sizeof(struct rs_cb));
    struct rs_cb* temPtr = gRsCb;

    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    gRsCb->chipId = cfg.chipId;
    rs_ut_msg("RS INIT, ret:%d !\n", ret);

    struct rdev rdevInfo = {0};
    void* mrHandle = NULL;
    void* mrHandle1 = NULL;
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");

    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsRegisterMr(phyId, rdevIndex, &mrRegInfo, &mrHandle);
    EXPECT_INT_EQ(0, ret);

    ret = RsRegisterMr(64, rdevIndex, &mrRegInfo, &mrHandle);
    EXPECT_INT_NE(0, ret);

    mocker(RsDrvMrReg, 1, NULL);
    ret = RsRegisterMr(phyId, rdevIndex, &mrRegInfo1, &mrHandle1);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    ret = RsDeregisterMr(phyId, rdevIndex, mrHandle);
    EXPECT_INT_EQ(0, ret);

    ret = RsDeregisterMr(phyId, rdevIndex, NULL);
    EXPECT_INT_NE(0, ret);

    ret = RsRdevDeinit(phyId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    free(temPtr);
    return;
}

void TcRsEpollCtlAdd()
{
    struct SocketPeerInfo fdHandle[1] = {{0}};
    struct RsConnCb connCb = {0};
    int ret = 0;
    struct RsListHead list = {0};

    fdHandle[0].phyId = 0;
    fdHandle[0].fd = 0;

    RS_INIT_LIST_HEAD(&list);
    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->fdMap = (const void**)&fdHandle;
    gRsCb->connCb = connCb;
    gRsCb->heterogTcpFdList = list;

    mocker((stub_fn_t)calloc, 5, NULL);
    ret = RsEpollCtlAdd((const void*)fdHandle, RA_EPOLLONESHOT);
    EXPECT_INT_EQ(ret, -ENOMEM);
    mocker_clean();

    ret = RsEpollCtlAdd((const void*)fdHandle, RA_EPOLLONESHOT);
    EXPECT_INT_EQ(ret, -1);

    ret = RsEpollCtlAdd((const void*)fdHandle, RA_EPOLLERR);
    EXPECT_INT_EQ(ret, -EINVAL);

    free(gRsCb);
    gRsCb = NULL;
    return;
}

void TcRsEpollCtlAdd01()
{
    struct SocketPeerInfo fdHandle[1] = {{0}};
    struct RsConnCb connCb = {0};
    int ret = 0;
    struct RsListHead list = {0};

    fdHandle[0].phyId = 0;
    fdHandle[0].fd = 0;

    RS_INIT_LIST_HEAD(&list);
    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->fdMap = (const void**)&fdHandle;
    gRsCb->connCb = connCb;
    gRsCb->heterogTcpFdList = list;

    ret = RsEpollCtlAdd((const void*)fdHandle, RA_EPOLLIN);
    EXPECT_INT_EQ(ret, -1);

    free(gRsCb);
    gRsCb = NULL;
    return;
}

void TcRsEpollCtlAdd02()
{
    struct SocketPeerInfo fdHandle[1] = {{0}};
    struct RsConnCb connCb = {0};
    int ret = 0;
    struct RsListHead list = {0};

    fdHandle[0].phyId = 0;
    fdHandle[0].fd = 0;

    RS_INIT_LIST_HEAD(&list);
    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->fdMap = (const void**)&fdHandle;
    gRsCb->connCb = connCb;
    gRsCb->heterogTcpFdList = list;
    gRsCb->heterogTcpFdList.next = &(gRsCb->heterogTcpFdList);
    gRsCb->heterogTcpFdList.prev = &(gRsCb->heterogTcpFdList);

    mocker((stub_fn_t)RsEpollCtl, 5, 0);
    ret = RsEpollCtlAdd((const void*)fdHandle, RA_EPOLLIN);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
    ret = RsEpollCtlDel(0);
    free(gRsCb);
    gRsCb = NULL;
    return;
}

void TcRsEpollCtlAdd03()
{
    struct SocketPeerInfo fdHandle[1] = {{0}};
    struct RsConnCb connCb = {0};
    int ret = 0;

    fdHandle[0].phyId = 0;
    fdHandle[0].fd = 0;

    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->fdMap = (const void**)&fdHandle;
    gRsCb->connCb = connCb;
    RS_INIT_LIST_HEAD(&gRsCb->heterogTcpFdList);
    gRsCb->heterogTcpFdList.next = &(gRsCb->heterogTcpFdList);
    gRsCb->heterogTcpFdList.prev = &(gRsCb->heterogTcpFdList);

    mocker((stub_fn_t)RsEpollCtl, 5, -1);
    ret = RsEpollCtlAdd((const void*)fdHandle, RA_EPOLLIN);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
    free(gRsCb);
    gRsCb = NULL;
    return;
}

void TcRsEpollCtlMod()
{
    struct SocketPeerInfo fdHandle[1] = {{0}};
    struct RsConnCb connCb = {0};
    int ret = 0;
    struct RsListHead list = {0};

    fdHandle[0].phyId = 0;
    fdHandle[0].fd = 0;

    RS_INIT_LIST_HEAD(&list);
    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->fdMap = (const void**)&fdHandle;
    gRsCb->connCb = connCb;
    gRsCb->heterogTcpFdList = list;

    ret = RsEpollCtlMod((const void*)fdHandle, RA_EPOLLONESHOT);
    EXPECT_INT_EQ(ret, -1);

    ret = RsEpollCtlMod((const void*)fdHandle, RA_EPOLLERR);
    EXPECT_INT_EQ(ret, -EINVAL);

    free(gRsCb);
    gRsCb = NULL;
    return;
}

void TcRsEpollCtlMod01()
{
    struct SocketPeerInfo fdHandle[1] = {{0}};
    struct RsConnCb connCb = {0};
    int ret = 0;
    struct RsListHead list = {0};

    fdHandle[0].phyId = 0;
    fdHandle[0].fd = 0;

    RS_INIT_LIST_HEAD(&list);
    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->fdMap = (const void**)&fdHandle;
    gRsCb->connCb = connCb;
    gRsCb->heterogTcpFdList = list;

    ret = RsEpollCtlMod((const void*)fdHandle, RA_EPOLLIN);
    EXPECT_INT_EQ(ret, -1);

    free(gRsCb);
    gRsCb = NULL;
    return;
}

void TcRsEpollCtlMod02()
{
    struct SocketPeerInfo fdHandle[1] = {{0}};
    struct RsConnCb connCb = {0};
    int ret = 0;
    struct RsListHead list = {0};

    fdHandle[0].phyId = 0;
    fdHandle[0].fd = 0;

    RS_INIT_LIST_HEAD(&list);
    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->fdMap = (const void**)&fdHandle;
    gRsCb->connCb = connCb;
    gRsCb->heterogTcpFdList = list;

    mocker((stub_fn_t)RsEpollCtl, 5, 0);
    ret = RsEpollCtlMod((const void*)fdHandle, RA_EPOLLIN);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    free(gRsCb);
    gRsCb = NULL;
    return;
}

void TcRsEpollCtlMod03()
{
    struct SocketPeerInfo fdHandle[1] = {{0}};
    int ret = 0;

    fdHandle[0].phyId = 0;
    fdHandle[0].fd = 0;

    gRsCb = NULL;

    mocker((stub_fn_t)RsEpollCtl, 5, 0);
    ret = RsEpollCtlAdd((const void*)fdHandle, RA_EPOLLIN);
    EXPECT_INT_EQ(ret, -22);

    gRsCb = NULL;

    ret = RsEpollCtlMod((const void*)fdHandle, RA_EPOLLIN);
    EXPECT_INT_EQ(ret, -22);

    gRsCb = NULL;
    ret = RsEpollCtlDel(0);
    EXPECT_INT_EQ(ret, -22);

    RsSetCtx(0);
    mocker_clean();
    gRsCb = NULL;
    return;
}

void TcRsEpollCtlDel()
{
    int ret = 0;
    struct RsListHead list = {0};

    RS_INIT_LIST_HEAD(&list);
    gRsCb = malloc(sizeof(struct rs_cb));

    gRsCb->heterogTcpFdList = list;
    gRsCb->heterogTcpFdList.next = &(gRsCb->heterogTcpFdList);
    gRsCb->heterogTcpFdList.prev = &(gRsCb->heterogTcpFdList);
    ret = RsEpollCtlDel(0);
    EXPECT_INT_EQ(ret, -1);

    free(gRsCb);
    gRsCb = NULL;
    return;
}

void TcRsEpollCtlDel01()
{
    int ret = 0;
    struct RsListHead list = {0};

    RS_INIT_LIST_HEAD(&list);
    gRsCb = malloc(sizeof(struct rs_cb));

    gRsCb->heterogTcpFdList = list;
    gRsCb->heterogTcpFdList.next = &(gRsCb->heterogTcpFdList);
    gRsCb->heterogTcpFdList.prev = &(gRsCb->heterogTcpFdList);

    mocker((stub_fn_t)RsEpollCtl, 5, 0);
    ret = RsEpollCtlDel(0);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    free(gRsCb);
    gRsCb = NULL;
    return;
}

void TcRsSetTcpRecvCallback()
{
    (void)RsSetTcpRecvCallback(NULL);

    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->hccpMode = 1;

    (void)RsSetTcpRecvCallback(NULL);

    free(gRsCb);
    gRsCb = NULL;
    return;
}

void TcRsEpollEventInHandle()
{
    int ret = 0;
    struct rs_cb rsCb = {0};
    struct epoll_event events = {0};
    rsCb.sslEnable = 1;

    mocker((stub_fn_t)RsEpollEventListenInHandle, 1, -ENODEV);
    mocker((stub_fn_t)RsEpollEventSslAcceptInHandle, 5, 0);
    (void)RsEpollEventInHandle(&rsCb, &events);
    mocker_clean();

    return;
}

int* stub__errno_locations()
{
    static int errNo = 0;

    errNo = EADDRINUSE;
    return &errNo;
}

int* stub__errno_location()
{
    static int errNo = 0;

    errNo = EAGAIN;
    return &errNo;
}

void TcRsSocketListenBindListen()
{
    int ret = 0;
    struct RsConnCb connCb = {0};
    struct SocketListenInfo conn = {0};
    struct RsListenInfo listenInfo = {0};
    conn.localIp.addr.s_addr = 1;

    ret = RsSocketListenBindListen(-1, &connCb, &conn, &listenInfo, 0);
    EXPECT_INT_EQ(-ESYSFUNC, ret);

    mocker(setsockopt, 20, 0);
    mocker_invoke(__errno_location, stub__errno_location, 1);
    ret = RsSocketListenBindListen(-1, &connCb, &conn, &listenInfo, 0);
    EXPECT_INT_EQ(EAGAIN, ret);
    mocker_clean();

    mocker(setsockopt, 20, 0);
    mocker(bind, 20, 0);
    mocker_invoke(__errno_location, stub__errno_location, 1);
    ret = RsSocketListenBindListen(-1, &connCb, &conn, &listenInfo, 0);
    EXPECT_INT_EQ(EAGAIN, ret);
    mocker_clean();

    mocker(setsockopt, 20, 0);
    mocker(bind, 20, 1);
    mocker_invoke(__errno_location, stub__errno_location, 1);
    ret = RsSocketListenBindListen(-1, &connCb, &conn, &listenInfo, 0);
    EXPECT_INT_EQ(EAGAIN, ret);
    mocker_clean();

    mocker(setsockopt, 20, 0);
    mocker(bind, 20, 0);
    mocker(listen, 20, 1);
    mocker_invoke(__errno_location, stub__errno_location, 1);
    ret = RsSocketListenBindListen(-1, &connCb, &conn, &listenInfo, 0);
    EXPECT_INT_EQ(EAGAIN, ret);
    mocker_clean();

    mocker(setsockopt, 20, 0);
    mocker(bind, 20, 0);
    mocker(listen, 20, 1);
    mocker_invoke(__errno_location, stub__errno_locations, 1);
    ret = RsSocketListenBindListen(-1, &connCb, &conn, &listenInfo, 0);
    EXPECT_INT_EQ(EADDRINUSE, ret);
    mocker_clean();
}

void TcRsEpollEventInHandle01()
{
    int ret = 0;
    struct rs_cb rsCb = {0};
    struct epoll_event events = {0};
    rsCb.sslEnable = 0;

    mocker((stub_fn_t)RsEpollEventListenInHandle, 1, -ENODEV);
    mocker((stub_fn_t)RsEpollEventQpMrInHandle, 1, -ENODEV);
    mocker((stub_fn_t)RsEpollEventHeterogTcpRecvInHandle, 5, 0);
    (void)RsEpollEventInHandle(&rsCb, &events);
    mocker_clean();

    return;
}

void TcRsEpollTcpRecv()
{
    int ret = 0;
    struct rs_cb rsCb = {0};
    struct SocketPeerInfo fdHandle[1] = {{0}};
    int callback = 0;

    fdHandle[0].phyId = 0;
    fdHandle[0].fd = 0;

    gRsCb = malloc(sizeof(struct rs_cb));
    rsCb.fdMap = (const void**)&fdHandle;

    rsCb.tcpRecvCallback = RsSetTcpRecvCallback;
    ret = RsEpollTcpRecv(&rsCb, 0);
    EXPECT_INT_EQ(ret, 0);

    rsCb.tcpRecvCallback = NULL;
    ret = RsEpollTcpRecv(&rsCb, 0);
    EXPECT_INT_EQ(ret, -EINVAL);

    free(gRsCb);
    gRsCb = NULL;
    return;
}

void TcRsEpollEventSslAcceptInHandle()
{
    int ret = 0;
    struct RsListHead list = {0};
    struct rs_cb* rsCb = NULL;
    struct RsConnCb connCb = {0};

    RS_INIT_LIST_HEAD(&list);

    rsCb = malloc(sizeof(struct rs_cb));
    rsCb->connCb = connCb;
    rsCb->connCb.serverAcceptList = list;
    rsCb->connCb.serverAcceptList.next = &(rsCb->connCb.serverAcceptList);
    rsCb->connCb.serverAcceptList.prev = &(rsCb->connCb.serverAcceptList);
    ret = RsEpollEventSslAcceptInHandle(rsCb, 0);
    EXPECT_INT_EQ(ret, -ENODEV);

    free(rsCb);
    rsCb = NULL;
    return;
}
struct RsMrCb* gMrCb;
struct ibv_mr* gIbMr;

struct RsMrCb* gMrCb;
struct ibv_mr* gIbMr;

int StubRsGetMrcb(struct RsQpCb* qpCb, uint64_t addr, struct RsMrCb** mrCb, struct RsListHead* mrList)
{
    *mrCb = gMrCb;
    return 0;
}

int stub_RsIbvExpPostSend(
    struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** badWr, struct wr_exp_rsp* expRsp)
{
    int wqeIndex = 2;
    expRsp->wqe_index = wqeIndex;
    expRsp->db_info = 1;
    return 0;
}

void TcRsSendExpWrlist()
{
    DlHalInit();
    struct RsQpCb qpCb = {0};
    struct WrInfo wrlist[1] = {{0}};
    unsigned int sendNum = 1;
    struct SendWrRsp rsWrInfo[1] = {{0}};
    unsigned int completeNum = 0;
    struct DbInfo db = {0};
    struct ibv_qp ibQp = {0};
    struct WqeInfoT wqeTmp = {0};
    struct SendWrRsp wrRsp = {0};
    struct wr_exp_rsp expRsp = {0};

    int ret = 0;

    gMrCb = malloc(sizeof(struct RsMrCb));
    gIbMr = malloc(sizeof(struct ibv_mr));
    gMrCb->ibMr = gIbMr;

    wrlist[0].memList.len = RS_TEST_MEM_SIZE;
    wrlist[0].memList.addr = 0x15;
    wrlist[0].op = 1;
    qpCb.sendWrNum = 0;
    ibQp.qp_num = 1;
    qpCb.ibQp = &ibQp;
    qpCb.sqIndex = 1;
    mocker_invoke((stub_fn_t)RsGetMrcb, StubRsGetMrcb, 2);
    mocker_invoke(RsIbvExpPostSend, stub_RsIbvExpPostSend, 1);
    qpCb.qpMode = 3;
    rsWrInfo[0].db = db;
    rsWrInfo[0].wqeTmp = wqeTmp;
    ret = RsSendExpWrlist(&qpCb, wrlist, sendNum, rsWrInfo, &completeNum);
    mocker_clean();

    mocker_invoke((stub_fn_t)RsGetMrcb, StubRsGetMrcb, 2);
    mocker(RsIbvExpPostSend, 1, -12);
    ret = RsSendExpWrlist(&qpCb, wrlist, sendNum, rsWrInfo, &completeNum);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker_invoke((stub_fn_t)RsGetMrcb, StubRsGetMrcb, 2);
    mocker(RsIbvExpPostSend, 1, -1);
    ret = RsSendExpWrlist(&qpCb, wrlist, sendNum, rsWrInfo, &completeNum);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker_invoke((stub_fn_t)RsGetMrcb, StubRsGetMrcb, 2);
    mocker_invoke(RsIbvExpPostSend, stub_RsIbvExpPostSend, 1);
    wrlist[0].op = 0xf6;
    ret = RsSendExpWrlist(&qpCb, wrlist, sendNum, rsWrInfo, &completeNum);
    mocker_clean();

    free(gIbMr);
    free(gMrCb);

    EXPECT_INT_EQ(ret, 0);

    DlHalDeinit();
    return;
}

int StubIbvGetCqEvent(struct ibv_comp_channel* channel, struct ibv_cq** cq, void** cqContext)
{
    *cq = NULL;
    return 0;
}

void TcRsDrvPollCqHandle()
{
    int ret = 0;
    uint32_t devId = 0;
    uint32_t qpMode = 1;
    unsigned int rdevIndex = 0;
    int flag = 0; /* RC */
    struct RsQpResp resp = {0};
    struct RsQpResp resp2 = {0};
    int i = 0;
    struct RsInitConfig cfg = {0};
    struct RsQpCb* qpCb = NULL;
    struct ibv_cq *ibSendCqT, *ibRecvCqT;
    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");

    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    struct RsQpNorm qpNorm = {0};
    qpNorm.flag = flag;
    qpNorm.qpMode = qpMode;
    qpNorm.isExp = 1;

    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_EQ(ret, 0);

    ret = RsQpn2qpcb(0, rdevIndex, resp.qpn, &qpCb);

    mocker_invoke((stub_fn_t)ibv_get_cq_event, StubIbvGetCqEvent, 10);
    ibSendCqT = qpCb->ibSendCq;
    qpCb->ibSendCq = NULL;

    /* reach end ? */
    mocker((stub_fn_t)ibv_req_notify_cq, 10, 0);
    mocker((stub_fn_t)ibv_poll_cq, 10, 0);
    RsDrvPollCqHandle(qpCb);
    mocker_clean();

    mocker_invoke((stub_fn_t)ibv_get_cq_event, StubIbvGetCqEvent, 10);
    mocker((stub_fn_t)ibv_req_notify_cq, 10, 1);
    RsDrvPollCqHandle(qpCb);
    mocker_clean();

    mocker_invoke((stub_fn_t)ibv_get_cq_event, StubIbvGetCqEvent, 10);
    mocker((stub_fn_t)ibv_poll_cq, 10, -1);
    RsDrvPollCqHandle(qpCb);
    mocker_clean();

    mocker_invoke((stub_fn_t)ibv_get_cq_event, StubIbvGetCqEvent, 10);
    mocker((stub_fn_t)ibv_poll_cq, 10, 0);
    RsDrvPollCqHandle(qpCb);
    mocker_clean();

    mocker_invoke((stub_fn_t)ibv_get_cq_event, StubIbvGetCqEvent, 10);
    mocker((stub_fn_t)ibv_poll_cq, 10, -1);
    RsDrvPollCqHandle(qpCb);
    mocker_clean();

    qpCb->ibSendCq = ibSendCqT;

    qpCb->rdevCb->rsCb->hccpMode = NETWORK_PEER_ONLINE;
    struct RsCqContext srqContext = {0};
    qpCb->srqContext = &srqContext;
    mocker((stub_fn_t)RsIbvGetCqEvent, 10, 0);
    RsDrvPollSrqCqHandle(qpCb);
    mocker_clean();

    ret = RsQpDestroy(devId, rdevIndex, resp.qpn);

    ret = RsRdevDeinit(devId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    return;
}

void TcRsQpCreateWithAttrsV2()
{
    int ret = 0;
    uint32_t phyId = 0;
    unsigned int rdevIndex = 0;
    struct RsQpNormWithAttrs qpNorm = {0};
    struct RsQpRespWithAttrs qpResp = {0};
    qpNorm.isExp = 1;

    ret = RsQpCreateWithAttrs(64, rdevIndex, &qpNorm, &qpResp);
    EXPECT_INT_NE(ret, 0);
    ret = RsQpCreateWithAttrs(phyId, rdevIndex, &qpNorm, NULL);
    EXPECT_INT_NE(ret, 0);
    ret = RsQpCreateWithAttrs(phyId, rdevIndex, NULL, &qpResp);
    EXPECT_INT_NE(ret, 0);
    qpNorm.extAttrs.version = -1;
    ret = RsQpCreateWithAttrs(phyId, rdevIndex, &qpNorm, &qpResp);
    EXPECT_INT_NE(ret, 0);
    qpNorm.extAttrs.version = QP_CREATE_WITH_ATTR_VERSION;
    qpNorm.extAttrs.qpMode = -1;
    ret = RsQpCreateWithAttrs(phyId, rdevIndex, &qpNorm, &qpResp);
    EXPECT_INT_NE(ret, 0);
    qpNorm.extAttrs.qpMode = RA_RS_OP_QP_MODE_EXT;
    ret = RsQpCreateWithAttrs(phyId, rdevIndex, &qpNorm, &qpResp);
    EXPECT_INT_NE(ret, 0);
}

void TcRsQpCreateWithAttrs()
{
    TcRsQpCreateWithAttrsV1();
    TcRsQpCreateWithAttrsV2();
}

void TcRsNormalQpCreate()
{
    int ret = 0;
    uint32_t phyId = 0;
    uint32_t rdevIndex = 0;
    int flag = 0; /* RC */
    int qpMode = 1;
    uint32_t qpn, qpn2, qpn3;
    int i = 0;
    int tryNum = 10;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listen[2] = {0};
    struct SocketConnectInfo conn[2] = {0};
    struct RsSocketCloseInfoT sockClose[2] = {0};
    struct SocketFdData socketInfo[3] = {0};
    struct SocketWlistInfoT whiteList = {0};
    whiteList.remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    whiteList.connLimit = 1;

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_PEER_ONLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 1);

    rs_ut_msg("___________________after listen:\n");
    strcpy(whiteList.tag, "1234");
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");
    RsSocketWhiteListAdd(rdevInfo, &whiteList, 1);

    conn[0].phyId = 0;
    conn[0].family = AF_INET;
    conn[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    conn[0].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(conn[0].tag, "1234");
    conn[0].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 1);

    rs_ut_msg("___________________after connect:\n");

    usleep(SLEEP_TIME);

    i = 0;
    socketInfo[i].family = AF_INET;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo[i].tag, "1234");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [client]socket_info[0].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    struct ibv_cq* ibSendCq = NULL;
    struct ibv_cq* ibRecvCq = NULL;
    void* context = NULL;
    struct CqAttr attr = {0};
    attr.qpContext = &context;
    attr.ibSendCq = &ibSendCq;
    attr.ibRecvCq = &ibRecvCq;
    attr.sendCqDepth = 16384;
    attr.recvCqDepth = 16384;
    attr.sendCqEventId = 1;
    attr.recvCqEventId = 2;
    attr.sendChannel = NULL;
    attr.recvChannel = NULL;
    ret = RsCqCreate(phyId, rdevIndex, &attr);
    EXPECT_INT_EQ(ret, 0);
    struct RsQpResp qpResp = {0};
    struct RsQpResp qpResp2 = {0};

    struct ibv_qp_init_attr qpInitAttr = {0};
    qpInitAttr.qp_context = context;
    qpInitAttr.send_cq = ibSendCq;
    qpInitAttr.recv_cq = ibRecvCq;
    qpInitAttr.qp_type = 2;
    qpInitAttr.cap.max_inline_data = 32;
    qpInitAttr.cap.max_send_wr = 4096;
    qpInitAttr.cap.max_send_sge = 4096;
    qpInitAttr.cap.max_recv_wr = 4096;
    qpInitAttr.cap.max_recv_sge = 1;
    struct ibv_qp* qp = NULL;

    mocker((stub_fn_t)RsDrvNormalQpCreate, 10, -ENOMEM);
    ret = RsNormalQpCreate(phyId, rdevIndex, &qpInitAttr, &qpResp, (void**)&qp);
    EXPECT_INT_EQ(-ENOMEM, ret);
    mocker_clean();

    ret = RsNormalQpCreate(phyId, rdevIndex, &qpInitAttr, &qpResp, (void**)&qp);
    EXPECT_INT_EQ(0, ret);

    qpInitAttr.qp_context = NULL;
    ret = RsNormalQpCreate(phyId, rdevIndex, &qpInitAttr, &qpResp, (void**)&qp);
    EXPECT_INT_EQ(-22, ret);

    rs_ut_msg("___________________after qp create:\n");

    /* >>>>>>> RsQpConnectAsync test case begin <<<<<<<<<<< */
    struct RdmaMrRegInfo mrRegInfo = {0};
    mrRegInfo.addr = (void*)0xabcdef;
    mrRegInfo.len = RS_TEST_MEM_SIZE;
    mrRegInfo.access = RS_ACCESS_LOCAL_WRITE;
    ret = RsMrReg(phyId, rdevIndex, qpResp.qpn, &mrRegInfo);
    EXPECT_INT_EQ(ret, 0);
    /* >>>>>>> RsQpConnectAsync test case end <<<<<<<<<<< */

    rs_ut_msg("___________________after mr reg:\n");

    ret = RsQpConnectAsync(phyId, rdevIndex, qpResp.qpn, socketInfo[i].fd);
    rs_ut_msg("***RsQpConnectAsync: %d****\n", ret);

    rs_ut_msg("___________________after qp connect async:\n");
    usleep(SLEEP_TIME);
    rs_ut_msg("___________________after qp connect async & sleep:\n");

    i = 1;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.1");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.1");
    strcpy(socketInfo[i].tag, "1234");
    tryNum = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_SERVER, &socketInfo[i], 1);
        usleep(5000);
    } while (ret != 1 && tryNum--);
    rs_ut_msg("%s [server]socket_info[1].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    struct ibv_cq* ibSendCq2 = NULL;
    struct ibv_cq* ibRecvCq2 = NULL;
    void* context2 = NULL;
    struct CqAttr attr2 = {0};
    attr2.qpContext = &context2;
    attr2.ibSendCq = &ibSendCq2;
    attr2.ibRecvCq = &ibRecvCq2;
    attr2.sendCqDepth = 16384;
    attr2.recvCqDepth = 16384;
    attr2.sendCqEventId = -1;
    attr2.recvCqEventId = 0;
    attr2.sendChannel = NULL;
    attr2.recvChannel = NULL;
    ret = RsCqCreate(phyId, rdevIndex, &attr2);
    EXPECT_INT_EQ(ret, 0);

    struct ibv_qp_init_attr qpInitAttr2 = {0};
    qpInitAttr2.qp_context = context2;
    qpInitAttr2.send_cq = ibSendCq2;
    qpInitAttr2.recv_cq = ibRecvCq2;
    qpInitAttr2.qp_type = 2;
    qpInitAttr2.cap.max_inline_data = 32;
    qpInitAttr2.cap.max_send_wr = 4096;
    qpInitAttr2.cap.max_send_sge = 4096;
    qpInitAttr2.cap.max_recv_wr = 4096;
    qpInitAttr2.cap.max_recv_sge = 1;
    struct ibv_qp* qp2 = NULL;
    ret = RsNormalQpCreate(phyId, rdevIndex, &qpInitAttr2, &qpResp2, (void**)&qp2);
    EXPECT_INT_EQ(0, ret);

    rs_ut_msg("___________________after qp2 create:\n");

    ret = RsQpConnectAsync(phyId, rdevIndex, qpResp2.qpn, socketInfo[i].fd);

    rs_ut_msg("___________________after qp2 connect async:\n");

    struct ibv_cq* ibSendCq3 = NULL;
    struct ibv_cq* ibRecvCq3 = NULL;
    void* context3 = NULL;
    struct CqAttr attr3 = {0};
    attr3.qpContext = &context3;
    attr3.ibSendCq = &ibSendCq3;
    attr3.ibRecvCq = &ibSendCq3;
    attr3.sendCqDepth = 16384;
    attr3.recvCqDepth = 16384;
    attr3.sendCqEventId = 1;
    attr3.recvCqEventId = 2;
    attr3.sendChannel = (void*)0xabcd;
    attr3.recvChannel = (void*)0xabcd;

    ret = RsCqCreate(phyId, rdevIndex, &attr3);
    EXPECT_INT_EQ(ret, 0);

    struct ibv_cq* ibSendCq4 = NULL;
    struct ibv_cq* ibRecvCq4 = NULL;
    void* context4 = NULL;
    struct CqAttr attr4 = {0};
    attr4.qpContext = &context4;
    attr4.ibSendCq = &ibSendCq4;
    attr4.ibRecvCq = &ibSendCq4;
    attr4.sendCqDepth = 16384;
    attr4.recvCqDepth = 16384;
    attr4.sendCqEventId = 1;
    attr4.recvCqEventId = 2;
    attr4.sendChannel = NULL;
    attr4.recvChannel = (void*)0xabcd;

    ret = RsCqCreate(phyId, rdevIndex, &attr4);
    EXPECT_INT_EQ(ret, -1);

    usleep(SLEEP_TIME);

    ret = RsNormalQpDestroy(phyId, rdevIndex, qpResp2.qpn);
    ret = RsNormalQpDestroy(phyId, rdevIndex, qpResp.qpn);

    rs_ut_msg("___________________after qp1&2 destroy:\n");

    ret = RsCqDestroy(phyId, rdevIndex, &attr3);
    ret = RsCqDestroy(phyId, rdevIndex, &attr2);
    ret = RsCqDestroy(phyId, rdevIndex, &attr);

    rs_ut_msg("___________________after cq1&2 destroy:\n");

    sockClose[0].fd = socketInfo[0].fd;
    ret = RsSocketBatchClose(0, &sockClose[0], 1);

    rs_ut_msg("___________________after close socket 0:\n");

    sockClose[1].fd = socketInfo[1].fd;
    ret = RsSocketBatchClose(0, &sockClose[1], 1);

    rs_ut_msg("___________________after close socket 1:\n");

    /* ------Resource CLEAN-------- */
    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 1);

    rs_ut_msg("___________________after stop listen:\n");

    ret = RsRdevDeinit(phyId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsSocketDeinit(rdevInfo);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    rs_ut_msg("___________________after deinit:\n");

    return;
}

void TcRsQueryEvent()
{
    int ret = 0;
    int eventId = 1;
    struct event_summary* sendEvent = NULL;

    mocker((stub_fn_t)calloc, 10, NULL);
    ret = RsQueryEvent(eventId, &sendEvent);
    EXPECT_INT_EQ(-ENOMEM, ret);
    mocker_clean();
}

void TcRsCreateCq()
{
    int ret = 0;

    struct RsRdevCb rdevCb = {0};
    struct RsCqContext cqContext = {0};
    struct CqAttr attr = {0};
    cqContext.rdevCb = &rdevCb;
    cqContext.cqCreateMode = RS_NORMAL_CQ_CREATE;

    mocker((stub_fn_t)RsQueryEvent, 10, -1);
    ret = RsDrvCreateCqEvent(&cqContext, &attr);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker((stub_fn_t)RsIbvCreateCq, 10, NULL);
    ret = RsDrvCreateCqEvent(&cqContext, &attr);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();

    mocker((stub_fn_t)ibv_req_notify_cq, 10, -1);
    ret = RsDrvCreateCqEvent(&cqContext, &attr);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();

    cqContext.recvEvent = NULL;
    cqContext.sendEvent = NULL;
    mocker((stub_fn_t)RsIbvDestroyCq, 10, -1);
    ret = RsDrvDestroyCqEvent(&cqContext);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsCreateNormalQp()
{
    int ret = 0;
    struct RsQpCb qpCb = {0};
    struct ibv_qp_init_attr qpInitAttr = {0};

    mocker((stub_fn_t)RsIbvCreateQp, 10, NULL);
    ret = RsDrvNormalQpCreate(&qpCb, &qpInitAttr);
    EXPECT_INT_EQ(-ENOMEM, ret);
    mocker_clean();

    mocker((stub_fn_t)RsIbvQueryQp, 10, -1);
    mocker((stub_fn_t)RsIbvCreateQp, 10, 1);
    mocker((stub_fn_t)RsIbvDestroyQp, 10, 0);
    ret = RsDrvNormalQpCreate(&qpCb, &qpInitAttr);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();

    mocker((stub_fn_t)RsIbvQueryQp, 10, 0);
    mocker((stub_fn_t)RsIbvCreateQp, 10, 1);
    mocker((stub_fn_t)RsIbvDestroyQp, 10, 0);
    mocker((stub_fn_t)RsDrvQpInfoRelated, 10, -1);
    ret = RsDrvNormalQpCreate(&qpCb, &qpInitAttr);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();
}

void TcRsCreateCompChannel()
{
    int ret = 0;
    unsigned int phyId = 0;
    unsigned int rdevIndex = 0;

    struct RsInitConfig cfg = {0};

    rs_ut_msg("resource prepare begin..................\n");

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;

    gRsCb = malloc(sizeof(struct rs_cb));
    memset(gRsCb, 0xff, sizeof(struct rs_cb));
    struct rs_cb* temPtr = gRsCb;

    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    gRsCb->chipId = cfg.chipId;
    rs_ut_msg("RS INIT, ret:%d !\n", ret);

    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");

    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    void* compChannel = NULL;
    void* compChannel1 = NULL;
    ret = RsCreateCompChannel(phyId, rdevIndex, &compChannel);
    EXPECT_INT_EQ(0, ret);

    mocker(rsGetLocalDevIDByHostDevID, 1, -19);
    ret = RsCreateCompChannel(phyId, rdevIndex, &compChannel1);
    EXPECT_INT_EQ(-19, ret);
    mocker_clean();

    mocker(RsRdev2rdevCb, 1, -19);
    ret = RsCreateCompChannel(phyId, rdevIndex, &compChannel1);
    EXPECT_INT_EQ(-19, ret);
    mocker_clean();

    mocker(RsIbvCreateCompChannel, 1, NULL);
    ret = RsCreateCompChannel(phyId, rdevIndex, &compChannel1);
    EXPECT_INT_EQ(-259, ret);
    mocker_clean();

    mocker(RsIbvDestroyCompChannel, 1, -1);
    ret = RsDestroyCompChannel(compChannel1);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    ret = RsDestroyCompChannel(compChannel);
    EXPECT_INT_EQ(0, ret);

    ret = RsRdevDeinit(phyId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    free(temPtr);
    return;
}

void TcRsGetCqeErrInfo()
{
    int ret = 0;
    struct RsCqeErrInfo* errInfo = &gRsCqeErr;
    struct CqeErrInfo* tempInfo = &errInfo->info;
    struct CqeErrInfo cqeInfo = {0};

    tempInfo->status = 0;
    mocker((stub_fn_t)pthread_mutex_lock, 1, 0);
    mocker((stub_fn_t)pthread_mutex_unlock, 1, 0);
    ret = RsDrvGetCqeErrInfo(&cqeInfo);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    tempInfo->status = 1;
    mocker((stub_fn_t)memcpy_s, 1, 1);
    ret = RsDrvGetCqeErrInfo(&cqeInfo);
    EXPECT_INT_EQ(-ESAFEFUNC, ret);
    mocker_clean();

    tempInfo->status = 1;
    mocker((stub_fn_t)memset_s, 1, 1);
    mocker((stub_fn_t)pthread_mutex_lock, 1, 0);
    mocker((stub_fn_t)pthread_mutex_unlock, 1, 0);
    ret = RsDrvGetCqeErrInfo(&cqeInfo);
    EXPECT_INT_EQ(-ESAFEFUNC, ret);
    mocker_clean();

    tempInfo->status = 1;
    mocker((stub_fn_t)pthread_mutex_lock, 1, 0);
    mocker((stub_fn_t)pthread_mutex_unlock, 1, 0);
    ret = RsDrvGetCqeErrInfo(&cqeInfo);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    mocker((stub_fn_t)RsDrvGetCqeErrInfo, 1, 0);
    ret = RsGetCqeErrInfo(&cqeInfo);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsGetCqeErrInfoNum()
{
    unsigned int num = 0;
    unsigned int phyId = 0;
    int ret = 0;

    mocker((stub_fn_t)rsGetLocalDevIDByHostDevID, 10, 0);
    mocker((stub_fn_t)RsRdev2rdevCb, 10, 0);
    phyId = 128;
    ret = RsGetCqeErrInfoNum(phyId, 0, &num);
    EXPECT_INT_EQ(-EINVAL, ret);

    mocker((stub_fn_t)rsGetLocalDevIDByHostDevID, 10, 0);
    mocker((stub_fn_t)RsRdev2rdevCb, 10, 0);
    phyId = 0;
    ret = RsGetCqeErrInfoNum(0, 0, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    return;
}

int stub_RsRdev2rdevCb(unsigned int chipId, unsigned int rdevIndex, struct RsRdevCb** rdevCb)
{
    *rdevCb = &gRdevCb;
    return 0;
}

void TcRsGetCqeErrInfoList()
{
    struct CqeErrInfo info = {0};
    struct RsQpCb qpCb = {0};
    unsigned int num = 1;
    unsigned int phyId = 0;
    int ret = 0;

    mocker((stub_fn_t)rsGetLocalDevIDByHostDevID, 10, 0);
    mocker((stub_fn_t)RsRdev2rdevCb, 10, 0);
    phyId = 128;
    ret = RsGetCqeErrInfoList(phyId, 0, &info, &num);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();

    mocker((stub_fn_t)rsGetLocalDevIDByHostDevID, 10, 0);
    mocker_invoke((stub_fn_t)RsRdev2rdevCb, stub_RsRdev2rdevCb, 10);
    phyId = 0;
    RS_INIT_LIST_HEAD(&gRdevCb.qpList);
    ret = RsGetCqeErrInfoList(phyId, 0, &info, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    RsListAddTail(&qpCb.list, &gRdevCb.qpList);
    qpCb.rdevCb = &gRdevCb;
    qpCb.cqeErrInfo.info.status = 1;
    mocker((stub_fn_t)rsGetLocalDevIDByHostDevID, 10, 0);
    mocker_invoke((stub_fn_t)RsRdev2rdevCb, stub_RsRdev2rdevCb, 10);
    ret = RsGetCqeErrInfoList(phyId, 0, &info, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    return;
}

void TcRsSaveCqeErrInfo()
{
    int ret = 0;
    struct RsCqeErrInfo* errInfo = &gRsCqeErr;
    struct CqeErrInfo* tempInfo = &errInfo->info;
    struct RsQpCb qpCb = {0};

    tempInfo->status = 0;
    RsDrvSaveCqeErrInfo(0x15, &qpCb);
    mocker_clean();

    tempInfo->status = 1;
    RsDrvSaveCqeErrInfo(0x15, &qpCb);
    mocker_clean();
}

void TcRsCqeCallbackProcess()
{
    struct RsQpCb qpcbTmp = {0};
    struct ibv_wc wc = {0};
    struct ibv_cq evCqSq = {0};
    struct ibv_cq evCqRq = {0};
    struct RsRdevCb rdevCb = {0};

    qpcbTmp.ibSendCq = &evCqSq;
    qpcbTmp.ibRecvCq = &evCqRq;
    qpcbTmp.rdevCb = &rdevCb;

    wc.status = IBV_WC_MW_BIND_ERR;
    mocker((stub_fn_t)RsDrvSaveCqeErrInfo, 1, 0);
    RsCqeCallbackProcess(&qpcbTmp, &wc, &evCqRq);
    mocker_clean();
}

void TcRsCreateSrq()
{
    int ret = 0;
    unsigned int phyId = 0;
    unsigned int rdevIndex = 0;

    struct RsInitConfig cfg = {0};

    rs_ut_msg("resource prepare begin..................\n");

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;

    gRsCb = malloc(sizeof(struct rs_cb));
    memset(gRsCb, 0xff, sizeof(struct rs_cb));
    struct rs_cb* temPtr = gRsCb;

    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    gRsCb->chipId = cfg.chipId;
    rs_ut_msg("RS INIT, ret:%d !\n", ret);

    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");

    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    struct SrqAttr attr = {0};
    struct ibv_srq* ibSrq = NULL;
    struct ibv_cq* ibRecvCq = NULL;
    void* context = NULL;
    attr.ibSrq = &ibSrq;
    attr.ibRecvCq = &ibRecvCq;
    attr.maxSge = 1;
    attr.context = &context;
    attr.srqEventId = 1;
    attr.srqDepth = 63;
    attr.cqDepth = 64;
    struct SrqAttr attr1 = {0};
    struct ibv_srq* ibSrq1 = (struct ibv_srq*)0xab;
    struct ibv_cq* ibRecvCq1 = (struct ibv_cq*)0xab;
    struct RsCqContext cqContext1 = {0};
    cqContext1.ibSrqCq = ibRecvCq1;
    attr1.ibSrq = &ibSrq1;
    attr1.ibRecvCq = &ibRecvCq1;
    attr1.maxSge = 1;
    void* context1 = &cqContext1;
    attr1.context = &context1;
    attr1.srqEventId = 1;
    attr1.srqDepth = 63;
    attr1.cqDepth = 64;

    mocker(rsGetLocalDevIDByHostDevID, 1, -19);
    ret = RsCreateSrq(phyId, rdevIndex, &attr1);
    EXPECT_INT_EQ(-19, ret);
    mocker_clean();

    mocker(RsIbvCreateCompChannel, 1, NULL);
    ret = RsCreateSrq(phyId, rdevIndex, &attr1);
    EXPECT_INT_EQ(-22, ret);
    mocker_clean();

    mocker(calloc, 1, NULL);
    ret = RsCreateSrq(phyId, rdevIndex, &attr1);
    EXPECT_INT_EQ(-ENOMEM, ret);
    mocker_clean();

    mocker(RsIbvCreateSrq, 1, NULL);
    ret = RsCreateSrq(phyId, rdevIndex, &attr1);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();

    ret = RsCreateSrq(phyId, rdevIndex, &attr);
    EXPECT_INT_EQ(0, ret);

    struct ibv_cq* ibSendCq = NULL;
    struct ibv_cq* ibRecvCq3 = NULL;
    void* context2 = NULL;
    struct CqAttr attr2 = {0};
    attr2.qpContext = &context2;
    attr2.ibSendCq = &ibSendCq;
    attr2.ibRecvCq = &ibRecvCq;
    attr2.sendCqDepth = 16384;
    attr2.recvCqDepth = 16384;
    attr2.sendCqEventId = 1;
    attr2.recvCqEventId = 2;
    attr2.sendChannel = NULL;
    attr2.recvChannel = NULL;

    mocker(RsEpollCtl, 5, 0);
    ret = RsCqCreate(phyId, rdevIndex, &attr2);
    EXPECT_INT_EQ(ret, 0);

    ret = RsCqDestroy(phyId, rdevIndex, &attr2);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsIbvAckCqEvents, 1, NULL);
    ret = RsDestroySrq(phyId, rdevIndex, &attr);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    ret = RsRdevDeinit(phyId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    free(temPtr);
    return;
}

void TcRsGetIpv6ScopeId()
{
    int ret = 0;
    struct in6_addr localIp = {0};

    ret = RsGetIpv6ScopeId(localIp);
    EXPECT_INT_EQ(-EINVAL, ret);

    mocker(getifaddrs, 1, -1);
    ret = RsGetIpv6ScopeId(localIp);
    EXPECT_INT_EQ(-ESYSFUNC, ret);
}

void TcRsCreateEventHandle()
{
    int ret = 0;
    int fd = 0;

    ret = RsCreateEventHandle(NULL);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RsCreateEventHandle(&fd);
    EXPECT_INT_EQ(0, ret);
    ret = RsDestroyEventHandle(&fd);
    EXPECT_INT_EQ(0, ret);
}

void TcRsCtlEventHandle()
{
    struct SocketPeerInfo fdHandle[1] = {{0}};
    int ret = 0;
    int fd = 0;

    ret = RsCtlEventHandle(-1, NULL, 0, 100);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RsCreateEventHandle(&fd);
    EXPECT_INT_EQ(0, ret);
    ret = RsCtlEventHandle(fd, NULL, 0, 100);
    EXPECT_INT_EQ(-EINVAL, ret);

    fdHandle[0].phyId = 0;
    fdHandle[0].fd = 0;
    ret = RsCtlEventHandle(fd, (const void*)fdHandle, 0, 100);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RsCtlEventHandle(fd, (const void*)fdHandle, EPOLL_CTL_ADD, 100);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RsCtlEventHandle(fd, (const void*)fdHandle, EPOLL_CTL_ADD, RA_EPOLLONESHOT);

    ret = RsDestroyEventHandle(&fd);
    EXPECT_INT_EQ(0, ret);
}

void TcRsWaitEventHandle()
{
    struct SocketEventInfoT eventInfo = {0};
    unsigned int eventsNum = 0;
    int ret = 0;
    int fd = 0;

    ret = RsWaitEventHandle(-1, NULL, -2, -1, NULL);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RsCreateEventHandle(&fd);
    EXPECT_INT_EQ(0, ret);
    ret = RsWaitEventHandle(fd, NULL, -2, -1, NULL);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RsWaitEventHandle(fd, &eventInfo, -2, -1, NULL);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RsWaitEventHandle(fd, &eventInfo, 0, 1, &eventsNum);
    EXPECT_INT_EQ(0, ret);

    ret = RsDestroyEventHandle(&fd);
    EXPECT_INT_EQ(0, ret);
}

void TcRsDestroyEventHandle()
{
    int ret = 0;

    ret = RsDestroyEventHandle(NULL);
    EXPECT_INT_EQ(-EINVAL, ret);
}

void TcRsEpollCreateEpollfd()
{
    int ret = 0;

    ret = RsEpollCreateEpollfd(NULL);
    EXPECT_INT_EQ(-EINVAL, ret);
}

void TcRsEpollDestroyFd()
{
    int fd = 0;
    int ret = 0;

    ret = RsEpollDestroyFd(NULL);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RsEpollCreateEpollfd(&fd);
    EXPECT_INT_EQ(0, ret);
    ret = RsEpollDestroyFd(&fd);
    EXPECT_INT_EQ(-1, fd);
    EXPECT_INT_EQ(0, ret);
}

void TcRsEpollWaitHandle()
{
    int fd = 0;
    int ret = 0;

    ret = RsEpollWaitHandle(-1, NULL, 0, -1, 0);
    EXPECT_INT_EQ(-EINVAL, ret);
}

void TcSslverify_callback()
{
    X509_STORE_CTX ctx;
    int ret = 0;

    mocker((stub_fn_t)X509_STORE_CTX_get_error, 10, X509_V_ERR_CERT_HAS_EXPIRED);
    ret = verify_callback(0, &ctx);
    EXPECT_INT_EQ(ret, 1);
    mocker_clean();
}

void Tcrs_ssl_verify_cert()
{
    X509_STORE_CTX ctx;
    int ret = 0;

    mocker((stub_fn_t)X509_verify_cert, 10, 0);
    mocker((stub_fn_t)X509_STORE_CTX_get_error, 10, X509_V_ERR_CERT_HAS_EXPIRED);
    ret = rs_ssl_verify_cert(&ctx);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)X509_verify_cert, 10, 0);
    mocker((stub_fn_t)X509_STORE_CTX_get_error, 10, 11);
    ret = rs_ssl_verify_cert(&ctx);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();
}

void TcRsMemPool()
{
    struct LiteMemAttrResp memResp = {0};
    struct RsRdevCb rdevCb = {0};
    struct RsQpCb qpCb = {0};
    struct rs_cb rsCb = {0};
    int ret = 0;

    qpCb.qpMode = RA_RS_OP_QP_MODE;
    qpCb.memAlign = LITE_ALIGN_4KB;
    ret = RsInitMemPool(&qpCb);
    EXPECT_INT_EQ(ret, 0);
    RsDeinitMemPool(&qpCb);

    qpCb.memAlign = LITE_ALIGN_2MB;
    qpCb.memResp = memResp;
    qpCb.rdevCb = &rdevCb;
    rdevCb.rsCb = &rsCb;
    mocker(RsRoceInitMemPool, 100, -EINVAL);
    ret = RsInitMemPool(&qpCb);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();
    mocker(RsRoceDeinitMemPool, 100, -EINVAL);
    RsDeinitMemPool(&qpCb);
    mocker_clean();
}

void TcRsGetVnicIpInfo()
{
    int ret = 0;
    unsigned int ids[1] = {0};
    struct IpInfo info[1] = {0};

    DlHalInit();

    ret = RsGetVnicIpInfos(0, 0, NULL, 0, NULL);
    EXPECT_INT_NE(ret, 0);

    ret = RsGetVnicIpInfos(0, 0, ids, 1, NULL);
    EXPECT_INT_NE(ret, 0);

    ret = RsGetVnicIpInfos(0, 2, ids, 1, info);
    EXPECT_INT_NE(ret, 0);

    ret = RsGetVnicIpInfo(0, 0, 0, &info[0]);
    EXPECT_INT_EQ(ret, 0);

    ret = RsGetVnicIpInfo(0, 0, 1, &info[0]);
    EXPECT_INT_EQ(ret, 0);

    ret = RsGetVnicIpInfo(0, 0, 2, &info[0]);
    EXPECT_INT_NE(ret, 0);

    DlHalDeinit();
}

void TcRsTypicalRegisterMr()
{
    int ret = 0;
    uint32_t phyId = 0;
    uint32_t rdevIndex = 0;
    void* addr = NULL;
    struct RdmaMrRegInfo mrRegInfo = {0};
    struct ibv_mr* raRsMrHandle = NULL;
    struct RsInitConfig cfg = {0};

    addr = malloc(RS_TEST_MEM_SIZE);
    mrRegInfo.addr = addr;
    mrRegInfo.len = RS_TEST_MEM_SIZE;
    mrRegInfo.access = RS_ACCESS_LOCAL_WRITE;

    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");

    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsTypicalRegisterMrV1(phyId, rdevIndex, &mrRegInfo, (void**)&raRsMrHandle);
    EXPECT_INT_EQ(ret, 0);

    ret = RsTypicalDeregisterMr(phyId, rdevIndex, (uint64_t)addr);
    EXPECT_INT_EQ(ret, 0);

    ret = RsTypicalRegisterMr(phyId, rdevIndex, &mrRegInfo, (void**)&raRsMrHandle);
    EXPECT_INT_EQ(ret, 0);

    ret = RsTypicalDeregisterMr(phyId, rdevIndex, (uint64_t)raRsMrHandle);
    EXPECT_INT_EQ(ret, 0);

    ret = RsRdevDeinit(phyId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    cfg.chipId = 0;
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    free(addr);
}

int stub_RsIbvQueryQp_init(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attrMask, struct ibv_qp_init_attr* initAttr)
{
    if (attr == NULL) {
        return -EINVAL;
    }
    attr->qp_state = 1;
    return 0;
}

void TcRsTypicalQpModify()
{
    int ret = 0;
    uint32_t phyId = 0;
    uint32_t rdevIndex = 0;
    void* addr = NULL;
    struct RdmaMrRegInfo mrRegInfo = {0};
    struct ibv_mr* raRsMrHandle = NULL;
    struct RsInitConfig cfg = {0};

    addr = malloc(RS_TEST_MEM_SIZE);
    mrRegInfo.addr = addr;
    mrRegInfo.len = RS_TEST_MEM_SIZE;
    mrRegInfo.access = RS_ACCESS_LOCAL_WRITE;

    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");

    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    struct TypicalQp localQpInfo = {0};
    struct TypicalQp remoteQpInfo = {0};
    struct TypicalQpAttr qpAttr = {0};

    struct RsQpNorm qpNorm = {0};
    struct RsQpResp resp = {0};
    struct RsQpResp resp2 = {0};
    int flag = 0; /* RC */
    int qpMode = 4;
    qpNorm.flag = flag;
    qpNorm.qpMode = qpMode;
    qpNorm.isExp = 1;
    qpNorm.isExt = 1;
    int batchModifyQpn[2];

    ret = RsQpCreate(phyId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_EQ(ret, 0);
    ret = RsQpCreate(phyId, rdevIndex, qpNorm, &resp2);
    EXPECT_INT_EQ(ret, 0);

    localQpInfo.qpn = resp.qpn;
    localQpInfo.psn = resp.psn;
    localQpInfo.gidIdx = resp.gidIdx;
    (void)memcpy_s(localQpInfo.gid, HCCP_GID_RAW_LEN, resp.gid.raw, HCCP_GID_RAW_LEN);
    remoteQpInfo.qpn = resp2.qpn;
    remoteQpInfo.psn = resp2.psn;
    remoteQpInfo.gidIdx = resp2.gidIdx;
    (void)memcpy_s(remoteQpInfo.gid, HCCP_GID_RAW_LEN, resp2.gid.raw, HCCP_GID_RAW_LEN);

    mocker_invoke(RsIbvQueryQp, stub_RsIbvQueryQp_init, 10);
    mocker(RsRoceQueryQpc, 10, 1);
    ret = RsTypicalQpModify(phyId, rdevIndex, localQpInfo, remoteQpInfo, &qpAttr);
    EXPECT_INT_EQ(ret, 0);
    ret = RsTypicalQpModify(phyId, rdevIndex, remoteQpInfo, localQpInfo, &qpAttr);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    batchModifyQpn[0] = resp.qpn;
    batchModifyQpn[1] = resp2.qpn;
    ret = RsQpBatchModify(phyId, rdevIndex, 5, batchModifyQpn, 2);
    EXPECT_INT_EQ(ret, 0);

    ret = RsQpBatchModify(phyId, rdevIndex, 1, batchModifyQpn, 2);
    EXPECT_INT_EQ(ret, 0);

    ret = RsQpBatchModify(phyId, rdevIndex, 1, batchModifyQpn, 2);
    EXPECT_INT_NE(ret, 0);

    ret = RsQpDestroy(phyId, rdevIndex, resp.qpn);
    EXPECT_INT_EQ(ret, 0);
    ret = RsQpDestroy(phyId, rdevIndex, resp2.qpn);
    EXPECT_INT_EQ(ret, 0);

    ret = RsRdevDeinit(phyId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    cfg.chipId = 0;
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    free(addr);
}

void TcRsGetQpHyperFeature()
{
    int ret = 0;
    uint32_t phyId = 0;
    unsigned int rdevIndex = 0;
    struct RsInitConfig cfg = {0};
    struct rdev rdevInfo = {0};
    struct RsQpNorm qpNorm = {0};
    struct RsQpResp resp = {0};
    struct HyperFeature hyperFeature = {0};
    struct RsQpCb* qpCb = NULL;

    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");

    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    qpNorm.flag = 0;
    qpNorm.qpMode = 1;
    qpNorm.isExp = 1;
    ret = RsQpCreate(phyId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_EQ(ret, 0);

    ret = RsGetQpHyperFeature(phyId, rdevIndex, resp.qpn, &hyperFeature);
    EXPECT_INT_EQ(ret, 0);

    ret = RsGetQpHyperFeature(RS_MAX_DEV_NUM, rdevIndex, resp.qpn, &hyperFeature);
    EXPECT_INT_EQ(ret, -EINVAL);

    ret = RsGetQpHyperFeature(phyId, rdevIndex, 99999, &hyperFeature);
    EXPECT_INT_NE(ret, 0);

    mocker((stub_fn_t)ibv_query_qp_supported_hyroce_feature, 10, -1);
    ret = RsGetQpHyperFeature(phyId, rdevIndex, resp.qpn, &hyperFeature);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)memcpy_s, 10, 1);
    ret = RsGetQpHyperFeature(phyId, rdevIndex, resp.qpn, &hyperFeature);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    ret = RsQpn2qpcb(phyId, rdevIndex, resp.qpn, &qpCb);
    EXPECT_INT_EQ(ret, 0);
    qpCb->rdevCb->ibCtxEx = NULL;
    ret = RsGetQpHyperFeature(phyId, rdevIndex, resp.qpn, &hyperFeature);
    EXPECT_INT_EQ(ret, 0);

    ret = RsQpDestroy(phyId, rdevIndex, resp.qpn);
    EXPECT_INT_EQ(ret, 0);

    ret = RsRdevDeinit(phyId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
}

void Tcrs_ssl_get_cert()
{
    int ret = 0;
    struct tls_cert_mng_info mngInfo = {0};
    struct rs_cb rscb = {0};
    struct RsCertSkidSubjectCb skidSubjectCb = {0};
    struct RsCerts certs = {0};
    struct tls_ca_new_certs newCerts[RS_SSL_NEW_CERT_CB_NUM] = {{0}};

    mocker(tls_get_user_config, 10, -1);
    ret = rs_ssl_get_cert(&rscb, &certs, &mngInfo, newCerts);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(tls_get_user_config, 20, 0);
    ret = rs_ssl_get_cert(&rscb, &certs, &mngInfo, newCerts);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker_ret(tls_get_user_config, 0, 0, -1);
    rscb.hccpMode = NETWORK_OFFLINE;
    ret = rs_ssl_get_cert(&rscb, &certs, &mngInfo, newCerts);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker_ret(tls_get_user_config, 0, -2, -1);
    rscb.hccpMode = NETWORK_OFFLINE;
    ret = rs_ssl_get_cert(&rscb, &certs, &mngInfo, newCerts);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker_ret(tls_get_user_config, 0, -1, -1);
    rscb.hccpMode = NETWORK_OFFLINE;
    ret = rs_ssl_get_cert(&rscb, &certs, &mngInfo, newCerts);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker_ret(tls_get_user_config, 0, 0, -1);
    rscb.hccpMode = NETWORK_OFFLINE;
    ret = rs_ssl_get_cert(&rscb, &certs, &mngInfo, newCerts);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker_ret(tls_get_user_config, 0, -2, -1);
    rscb.hccpMode = NETWORK_OFFLINE;
    ret = rs_ssl_get_cert(&rscb, &certs, &mngInfo, newCerts);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker_ret(tls_get_user_config, -2, -2, -2);
    rscb.hccpMode = NETWORK_OFFLINE;
    ret = rs_ssl_get_cert(&rscb, &certs, &mngInfo, newCerts);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    return;
}

void tc_rs_ssl_X509_store_init()
{
    int ret = 0;
    X509_STORE_CTX ctx = {0};
    X509_STORE store = {0};
    struct tls_cert_mng_info mngInfo = {0};
    struct RsCerts certs = {0};
    struct tls_ca_new_certs newCerts[RS_SSL_NEW_CERT_CB_NUM] = {{0}};

    mocker(tls_get_cert_chain, 10, -1);
    ret = rs_ssl_x509_store_init(&store, &certs, &mngInfo, newCerts);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(tls_get_cert_chain, 10, 0);
    ret = rs_ssl_x509_store_init(&store, &certs, &mngInfo, newCerts);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    newCerts[0].ncert_count = 1;
    strcpy(newCerts[0].certs[0].ncert_info, "pub cert");
    ret = rs_ssl_x509_store_init(&store, &certs, &mngInfo, newCerts);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    newCerts[0].ncert_count = 2;
    strcpy(newCerts[0].certs[1].ncert_info, "root ca cert");
    mocker(tls_load_cert, 10, NULL);
    ret = rs_ssl_x509_store_init(&store, &certs, &mngInfo, newCerts);
    EXPECT_INT_EQ(ret, -22);
    mocker_clean();

    return;
}

void Tcrs_ssl_skids_subjects_get()
{
    int ret = 0;
    struct tls_cert_mng_info mngInfo = {0};
    struct RsCerts certs = {0};
    struct tls_ca_new_certs newCerts[RS_SSL_NEW_CERT_CB_NUM] = {{0}};
    struct rs_cb rscb = {0};

    mngInfo.cert_count = 2;
    mocker(tls_load_cert, 10, NULL);
    ret = rs_ssl_skids_subjects_get(&rscb, &mngInfo, &certs, newCerts);
    EXPECT_INT_EQ(ret, -22);
    mocker_clean();

    newCerts[0].ncert_count = 2;
    strcpy(newCerts[0].certs[1].ncert_info, "root ca cert");
    mocker(tls_load_cert, 10, NULL);
    ret = rs_ssl_skids_subjects_get(&rscb, &mngInfo, &certs, newCerts);
    EXPECT_INT_EQ(ret, -22);
    mocker_clean();

    return;
}

void Tcrs_ssl_put_cert_ca_pem()
{
    int ret = 0;
    char caFile[20];
    struct tls_cert_mng_info mngInfo = {0};
    struct RsCerts certs = {0};
    struct tls_ca_new_certs newCerts[RS_SSL_NEW_CERT_CB_NUM] = {{0}};

    strcpy(caFile, "ca file name");
    mocker(creat, 10, -1);
    ret = rs_ssl_put_cert_ca_pem(&certs, &mngInfo, newCerts, caFile);
    EXPECT_INT_EQ(ret, -EFILEOPER);
    mocker_clean();

    mocker(creat, 10, 0);
    mngInfo.cert_count = 2;
    mocker(write, 10, -1);
    ret = rs_ssl_put_cert_ca_pem(&certs, &mngInfo, newCerts, caFile);
    EXPECT_INT_EQ(ret, -22);
    mocker_clean();

    mocker(creat, 10, 0);
    newCerts[0].ncert_count = 2;
    strcpy(newCerts[0].certs[0].ncert_info, "pub cert");
    strcpy(newCerts[0].certs[1].ncert_info, "root ca cert");
    mocker(write, 10, 0);
    ret = rs_ssl_put_cert_ca_pem(&certs, &mngInfo, newCerts, caFile);
    EXPECT_INT_EQ(ret, -22);
    mocker_clean();

    return;
}

void Tcrs_ssl_put_cert_end_pem()
{
    int ret = 0;
    char endFile[20];
    struct RsCerts certs = {0};
    struct tls_ca_new_certs newCerts[RS_SSL_NEW_CERT_CB_NUM] = {{0}};

    strcpy(endFile, "end file name");
    newCerts[0].ncert_count = 2;
    mocker(creat, 10, 0);
    mocker(write, 10, -1);
    ret = rs_ssl_put_cert_end_pem(&certs, newCerts, endFile);
    EXPECT_INT_EQ(ret, -EFILEOPER);
    mocker_clean();

    strcpy(newCerts[0].certs[0].ncert_info, "pub cert");
    mocker(creat, 10, 0);
    mocker(write, 10, strlen(newCerts[0].certs[0].ncert_info));
    ret = rs_ssl_put_cert_end_pem(&certs, newCerts, endFile);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    newCerts[0].ncert_count = 0;
    mocker(rs_ssl_put_end_cert, 10, -1);
    ret = rs_ssl_put_cert_end_pem(&certs, newCerts, endFile);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    return;
}

void Tcrs_ssl_check_mng_and_cert_chain()
{
    int ret = 0;
    struct rs_cb rscb = {0};
    struct RsCerts certs = {0};
    struct tls_cert_mng_info mngInfo = {0};
    struct CertFile fileName = {0};
    struct tls_ca_new_certs newCerts[RS_SSL_NEW_CERT_CB_NUM] = {{0}};

    mngInfo.cert_count = 1;
    mngInfo.total_cert_len = 0;
    strcpy(certs.certs[0].certInfo, "pub cert");
    ret = rs_ssl_check_mng_and_cert_chain(&rscb, &mngInfo, &certs, newCerts, &fileName);
    EXPECT_INT_EQ(ret, -22);

    newCerts[0].ncert_count = 2;
    mocker(rs_remove_certs, 20, -1);
    ret = rs_ssl_check_mng_and_cert_chain(&rscb, &mngInfo, &certs, newCerts, &fileName);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(rs_remove_certs, 20, 0);
    mocker(rs_ssl_check_cert_chain, 20, -1);
    ret = rs_ssl_check_mng_and_cert_chain(&rscb, &mngInfo, &certs, newCerts, &fileName);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(rs_remove_certs, 20, 0);
    mocker(rs_ssl_check_cert_chain, 20, 0);
    mocker(rs_ssl_skid_get_from_chain, 20, -1);
    ret = rs_ssl_check_mng_and_cert_chain(&rscb, &mngInfo, &certs, newCerts, &fileName);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
    return;
}

void Tcrs_remove_certs()
{
    int ret = 0;
    char endFile[20];
    char caFile[20];

    mocker(remove, 10, -1);
    ret = rs_remove_certs(endFile, caFile);
    EXPECT_INT_EQ(ret, -EFILEOPER);
    mocker_clean();

    return;
}

void tc_rs_ssl_X509_store_add_cert()
{
    int ret = 0;
    char certInfo[20];
    X509_STORE store;

    mocker(tls_load_cert, 10, NULL);
    ret = rs_ssl_X509_store_add_cert(certInfo, &store);
    EXPECT_INT_EQ(ret, -22);
    mocker_clean();

    return;
}

void TcRsPeerSocketRecv()
{
    struct SocketConnectInfo conn[10] = {0};
    struct SocketErrInfo err[10] = {0};
    int ret = 0;

    mocker_clean();

    mocker(pthread_mutex_lock, 1, 0);
    mocker(pthread_mutex_unlock, 1, 0);
    mocker(RsGetConnInfo, 1, -1);
    ret = RsSocketGetClientSocketErrInfo(conn, err, 1);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    conn[0].family = AF_INET;
    mocker(RsSocketConnectCheckPara, 1, 0);
    mocker(RsSocketNodeid2vnic, 1, 0);
    mocker(rsGetLocalDevIDByHostDevID, 1, 0);
    mocker(RsDev2conncb, 1, 0);
    mocker(RsGetConnInfo, 1, 0);
    mocker(memcpy_s, 1, 0);
    mocker(memset_s, 1, 0);
    mocker(pthread_mutex_lock, 10, 0);
    mocker(pthread_mutex_unlock, 10, 0);
    ret = RsSocketGetClientSocketErrInfo(conn, err, 1);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    return;
}

void TcRsSocketGetServerSocketErrInfo()
{
    struct SocketListenInfo conn[10] = {0};
    struct ServerSocketErrInfo err[10] = {0};
    int ret = 0;

    mocker_clean();

    mocker(RsConvertIpAddr, 1, 0);
    mocker(pthread_mutex_lock, 1, 0);
    mocker(RsFindListenNode, 1, -1);
    mocker(pthread_mutex_unlock, 1, 0);
    ret = RsSocketGetServerSocketErrInfo(conn, err, 1);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    conn[0].family = AF_INET;
    mocker(RsSocketNodeid2vnic, 1, 0);
    mocker(rsGetLocalDevIDByHostDevID, 1, 0);
    mocker(RsDev2conncb, 1, 0);
    mocker(RsConvertIpAddr, 1, 0);
    mocker(RsFindListenNode, 1, 0);
    mocker(memcpy_s, 2, 0);
    mocker(memset_s, 1, 0);
    mocker(pthread_mutex_lock, 10, 0);
    mocker(pthread_mutex_unlock, 10, 0);
    ret = RsSocketGetServerSocketErrInfo(conn, err, 1);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    return;
}

int StubRsFindListenNode(
    struct RsConnCb* connCb, struct RsIpAddrInfo* ipAddr, uint32_t serverPort, struct RsListenInfo** listenInfo)
{
    *listenInfo = gPlistenInfo;

    return 0;
}

void TcRsSocketAcceptCreditAdd()
{
    struct SocketListenInfo conn[10] = {0};
    struct RsConnCb connCb = {0};
    int ret = 0;

    mocker_clean();

    mocker(RsConvertIpAddr, 1, 0);
    mocker(pthread_mutex_lock, 1, 0);
    mocker(pthread_mutex_unlock, 1, 0);
    mocker(RsFindListenNode, 1, -1);
    ret = RsSocketAcceptCreditAdd(conn, 1, 1);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RsConvertIpAddr, 1, 0);
    mocker(pthread_mutex_lock, 3, 0);
    mocker(pthread_mutex_unlock, 3, 0);
    mocker_invoke(RsFindListenNode, StubRsFindListenNode, 1);
    mocker(RsSocketListenAddToEpoll, 1, 0);
    ret = RsSocketAcceptCreditAdd(conn, 1, 1);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker(RsEpollCtl, 1, 0);
    mocker(pthread_mutex_lock, 3, 0);
    mocker(pthread_mutex_unlock, 3, 0);
    gListenInfo.fdState = LISTEN_FD_STATE_DELETED;
    ret = RsSocketListenAddToEpoll(&connCb, gPlistenInfo);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker(pthread_mutex_lock, 3, 0);
    mocker(pthread_mutex_unlock, 3, 0);
    gListenInfo.fdState = LISTEN_FD_STATE_DELETED;
    ret = RsSocketListenDelFromEpoll(&connCb, gPlistenInfo);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker(pthread_mutex_lock, 3, 0);
    mocker(pthread_mutex_unlock, 3, 0);
    mocker(RsEpollCtl, 1, -1);
    gListenInfo.fdState = LISTEN_FD_STATE_ADDED;
    ret = RsSocketListenDelFromEpoll(&connCb, gPlistenInfo);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    gListenInfo.acceptCreditFlag = 1;
    gListenInfo.acceptCreditLimit = 1;
    mocker(pthread_mutex_lock, 3, 0);
    mocker(pthread_mutex_unlock, 3, 0);
    mocker(RsEpollCtl, 1, 0);
    ret = RsSocketCheckCredit(&connCb, gPlistenInfo);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsEpollEventSslRecvTagInHandle()
{
    struct rs_cb rsCb = {0};
    struct RsAcceptInfo* acceptInfo = NULL;
    struct RsListHead list = {0};

    RS_INIT_LIST_HEAD(&list);
    acceptInfo = malloc(sizeof(struct RsAcceptInfo));
    acceptInfo->list = list;
    mocker_clean();
    mocker(RsSslRecvTagInHandle, 1, 0);
    mocker(RsWlistCheckConnAdd, 1, -1);
    mocker(SSL_free, 1, 0);
    mocker(pthread_mutex_lock, 1, 0);
    mocker(pthread_mutex_unlock, 1, 0);
    RsEpollEventSslRecvTagInHandle(&rsCb, acceptInfo);
    mocker_clean();
}

void TcRsRemapMr()
{
    struct MemRemapInfo memList[1] = {0};
    struct RsMrCb mrCb = {0};
    struct ibv_mr ibMr = {0};
    int ret = 0;

    mocker_clean();
    RS_INIT_LIST_HEAD(&gRdevCb.typicalMrList);
    mocker(rsGetLocalDevIDByHostDevID, 1, 0);
    mocker_invoke((stub_fn_t)RsRdev2rdevCb, stub_RsRdev2rdevCb, 10);
    mocker(RsRoceRemapMr, 1, 0);
    mocker(pthread_mutex_lock, 1, 0);
    mocker(pthread_mutex_unlock, 1, 0);
    ret = RsRemapMr(0, 0, memList, 1);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    RS_INIT_LIST_HEAD(&gRdevCb.typicalMrList);
    ibMr.length = 100;
    mrCb.ibMr = &ibMr;
    RsListAddTail(&mrCb.list, &gRdevCb.typicalMrList);
    mocker(rsGetLocalDevIDByHostDevID, 1, 0);
    mocker_invoke((stub_fn_t)RsRdev2rdevCb, stub_RsRdev2rdevCb, 10);
    mocker(RsRoceRemapMr, 1, 0);
    mocker(pthread_mutex_lock, 1, 0);
    mocker(pthread_mutex_unlock, 1, 0);
    ret = RsRemapMr(0, 0, memList, 1);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    RS_INIT_LIST_HEAD(&gRdevCb.typicalMrList);
    ibMr.length = 100;
    mrCb.ibMr = &ibMr;
    memList[0].size = (unsigned long long)-1;
    RsListAddTail(&mrCb.list, &gRdevCb.typicalMrList);
    mocker(rsGetLocalDevIDByHostDevID, 1, 0);
    mocker_invoke((stub_fn_t)RsRdev2rdevCb, stub_RsRdev2rdevCb, 10);
    mocker(RsRoceRemapMr, 1, 0);
    mocker(pthread_mutex_lock, 1, 0);
    mocker(pthread_mutex_unlock, 1, 0);
    ret = RsRemapMr(0, 0, memList, 1);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    RS_INIT_LIST_HEAD(&gRdevCb.typicalMrList);
    ibMr.length = 100;
    mrCb.ibMr = &ibMr;
    memList[0].size = 100;
    RsListAddTail(&mrCb.list, &gRdevCb.typicalMrList);
    mocker(rsGetLocalDevIDByHostDevID, 1, 0);
    mocker_invoke((stub_fn_t)RsRdev2rdevCb, stub_RsRdev2rdevCb, 10);
    mocker(RsRoceRemapMr, 1, 0);
    mocker(pthread_mutex_lock, 1, 0);
    mocker(pthread_mutex_unlock, 1, 0);
    ret = RsRemapMr(0, 0, memList, 1);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void tc_RsGetApiVersion()
{
    unsigned int apiVersion = 0;

    apiVersion = RsRoceGetApiVersion();
    EXPECT_INT_EQ(apiVersion, 0);

    apiVersion = RsNetGetApiVersion();
    EXPECT_INT_EQ(apiVersion, 0);
}

void TcRsGetTlsEnable()
{
    unsigned int phyId = 0;
    bool tlsEnable = false;
    int ret = 0;

    mocker(RsGetRsCb, 1, 1);
    ret = RsGetTlsEnable(phyId, &tlsEnable);
    EXPECT_INT_EQ(ret, 1);

    ret = RsGetTlsEnable(phyId, NULL);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();
}

int RsGetLinuxVersionStub(struct RsLinuxVersionInfo* verInfo)
{
    verInfo->major = 5;
    verInfo->minor = 19;
    verInfo->patch = 0;
    return 0;
}

static char gRsHccnCfgFilePath[32] = {0};
static char gRsHccnCfgKeyName[32] = {0};
static unsigned int gRsHccnCfgBufLen = 0;
static int gRsHccnCfgDevId = -1;

static int
StubRsHccnCfgFileRead(const char* filePath, int devId, const char* confName, char* confValue, unsigned int len)
{
    const char cfgValue[] = "stub_value";

    if (filePath == NULL || confName == NULL || strcpy_s(gRsHccnCfgFilePath, sizeof(gRsHccnCfgFilePath), filePath) != 0
        || strcpy_s(gRsHccnCfgKeyName, sizeof(gRsHccnCfgKeyName), confName) != 0) {
        return -EINVAL;
    }
    gRsHccnCfgBufLen = len;
    gRsHccnCfgDevId = devId;
    return (memcpy_s(confValue, len, cfgValue, sizeof(cfgValue)) == 0) ? 0 : -EINVAL;
}

void TcRsGetSecRandom()
{
    unsigned int value = 0;
    int ret;
    ret = RsGetSecRandom(&value);
    EXPECT_INT_EQ(ret, -257);

    mocker(strstr, 2, NULL);
    ret = RsGetSecRandom(&value);
    EXPECT_INT_NE(0, ret);

    mocker(read, 2, -1);
    ret = RsGetSecRandom(&value);
    EXPECT_INT_NE(0, ret);

    mocker(open, 2, -1);
    ret = RsGetSecRandom(&value);
    EXPECT_INT_NE(0, ret);

    mocker_invoke(RsGetLinuxVersion, RsGetLinuxVersionStub, 10);

    ret = RsGetSecRandom(&value);
    EXPECT_INT_NE(ret, 0);
}

void TcRsGetHccnCfg()
{
    char value[2048] = {0};
    unsigned int valueLen = 2048;
    struct RaInfo info = {
        .mode = NETWORK_OFFLINE,
        .phyId = 0,
    };

    int ret = 0;

    ret = RsGetHccnCfg(NULL, HCCN_CFG_UDP_PORT_MODE, value, &valueLen);
    EXPECT_INT_EQ(-EINVAL, ret);

    info.mode = NETWORK_ONLINE;
    ret = RsGetHccnCfg(&info, HCCN_CFG_UDP_PORT_MODE, value, &valueLen);
    EXPECT_INT_EQ(-ENOTSUPP, ret);

    mocker_invoke_p5(
        "FileReadCfg", "StubRsHccnCfgFileRead", (stub_fn_t)FileReadCfg, (stub_fn_t)StubRsHccnCfgFileRead, 2);
    info.mode = NETWORK_OFFLINE;
    ret = RsGetHccnCfg(&info, HCCN_CFG_UDP_PORT_MODE, value, &valueLen);
    EXPECT_INT_EQ(0, ret);
    EXPECT_STR_EQ("/etc/hccl.cfg", gRsHccnCfgFilePath);
    EXPECT_STR_EQ("udp_port_mode", gRsHccnCfgKeyName);
    EXPECT_INT_EQ(0, gRsHccnCfgDevId);
    EXPECT_INT_EQ(2048, (int)gRsHccnCfgBufLen);
    EXPECT_STR_EQ("stub_value", value);
    EXPECT_INT_EQ((int)sizeof("stub_value"), (int)valueLen);

    info.mode = NETWORK_PEER_ONLINE;
    info.phyId = 1;
    valueLen = sizeof(value);
    ret = RsGetHccnCfg(&info, HCCN_CFG_MULTI_QP_COUNT, value, &valueLen);
    EXPECT_INT_EQ(0, ret);
    EXPECT_STR_EQ("/etc/hcomm.cfg", gRsHccnCfgFilePath);
    EXPECT_STR_EQ("multi_qp_count", gRsHccnCfgKeyName);
    EXPECT_INT_EQ(1, gRsHccnCfgDevId);
    EXPECT_INT_EQ((int)sizeof(value), (int)gRsHccnCfgBufLen);
    EXPECT_STR_EQ("stub_value", value);
    EXPECT_INT_EQ((int)sizeof("stub_value"), (int)valueLen);
    mocker_clean();
}

void TcRsFreeDevList(void)
{
    struct rs_cb rscb = {0};

    RS_INIT_LIST_HEAD(&rscb.rdevList);
    RsFreeRdevList(&rscb);

    RS_INIT_LIST_HEAD(&rscb.udevList);
    RsFreeUdevList(&rscb);
}

void TcRsFreeRdevList(void)
{
    struct RsRdevCb rdevCb = {0};
    struct rs_cb rscb = {0};

    rscb.protocol = PROTOCOL_RDMA;
    mocker(rsGetDevIDByLocalDevID, 1, -1);
    RsFreeRdevList(&rscb);
    mocker_clean();

    RS_INIT_LIST_HEAD(&rscb.rdevList);
    RsListAddTail(&rdevCb.list, &rscb.rdevList);
    mocker(rsGetDevIDByLocalDevID, 1, 0);
    mocker(RsRdevDeinit, 1, -1);
    RsFreeRdevList(&rscb);
    mocker_clean();

    mocker(rsGetDevIDByLocalDevID, 1, 0);
    mocker(RsRdevDeinit, 1, 0);
    RsFreeRdevList(&rscb);
    mocker_clean();
}

void TcRsFreeUdevList(void)
{
    struct RsUbDevCb udevCb = {0};
    struct rs_cb rscb = {0};

    RS_INIT_LIST_HEAD(&rscb.udevList);
    mocker(RsUbCtxDeinit, 1, -1);
    RsListAddTail(&udevCb.list, &rscb.udevList);
    RsFreeUdevList(&rscb);
    mocker_clean();

    mocker_clean();
    mocker(RsUbCtxDeinit, 1, 0);
    RsFreeUdevList(&rscb);
    mocker_clean();
}

void TcRsRetryTimeoutExceptionCheck()
{
    struct SensorNode sensorNode = {0};
    struct ibv_wc wc = {0};
    int ret = 0;

    sensorNode.sensorHandle = 0;
    ret = RsRetryTimeoutExceptionCheck(&sensorNode);
    EXPECT_INT_EQ(ret, 0);

    sensorNode.sensorHandle = 1;
    mocker_clean();
    mocker(DlHalSensorNodeUpdateState, 1, -1);
    ret = RsRetryTimeoutExceptionCheck(&sensorNode);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(DlHalSensorNodeUpdateState, 1, 0);
    ret = RsRetryTimeoutExceptionCheck(&sensorNode);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    wc.status = IBV_WC_RETRY_EXC_ERR;
    mocker(RsRetryTimeoutExceptionCheck, 1, 0);
    RsRdmaRetryTimeoutExceptionCheck(&sensorNode, &wc);
    mocker_clean();
}

void TcRsSetQpLbValue()
{
    unsigned int rdevIndex = 0;
    unsigned int phyId = 0;
    unsigned int qpn = 0;
    int lbValue = 0;
    int ret = 0;

    ret = RsSetQpLbValue(RS_MAX_DEV_NUM, rdevIndex, qpn, lbValue);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker_clean();
    mocker_invoke(RsQpn2qpcb, ReplaceRsQpn2qpcb, 1);
    ret = RsSetQpLbValue(phyId, rdevIndex, qpn, lbValue);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsGetQpLbValue()
{
    unsigned int rdevIndex = 0;
    unsigned int phyId = 0;
    unsigned int qpn = 0;
    int lbValue = 0;
    int ret = 0;

    ret = RsGetQpLbValue(RS_MAX_DEV_NUM, rdevIndex, qpn, &lbValue);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker_clean();
    mocker_invoke(RsQpn2qpcb, ReplaceRsQpn2qpcb, 1);
    ret = RsGetQpLbValue(phyId, rdevIndex, qpn, NULL);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    mocker_clean();
    mocker_invoke(RsQpn2qpcb, ReplaceRsQpn2qpcb, 1);
    ret = RsGetQpLbValue(phyId, rdevIndex, qpn, &lbValue);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsGetLbMax()
{
    unsigned int rdevIndex = 0;
    unsigned int phyId = 0;
    int lbMax = 0;
    int ret = 0;

    ret = RsGetLbMax(RS_MAX_DEV_NUM, rdevIndex, &lbMax);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker_clean();
    mocker_invoke(RsQueryRdevCb, RsQueryRdevCbStub, 1);
    ret = RsGetLbMax(phyId, rdevIndex, NULL);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    mocker_clean();
    mocker_invoke(RsQueryRdevCb, RsQueryRdevCbStub, 1);
    ret = RsGetLbMax(phyId, rdevIndex, &lbMax);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}
