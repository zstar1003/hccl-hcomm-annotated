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
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <dlfcn.h>
#include <fcntl.h>

#include "ut_dispatch.h"
#include "stub/ibverbs.h"
#include "hccp_common.h"
#include "rs.h"
#include "rs_common_inner.h"
#include "rs_inner.h"
#include "rs_ping_inner.h"
#include "rs_ping.h"
#include "rs_drv_rdma.h"
#include "rs_drv_socket.h"
#include "rs_socket.h"
#include "ra_rs_err.h"
#include "tc_ut_rs.h"
#include "stub/verbs_exp.h"
#include "dl_hal_function.h"
#include "dl_ibverbs_function.h"
#include "dl.h"
#include "rs_esched.h"
#include "rs_ctx_inner.h"

extern __thread struct rs_cb* gRsCb;
int DlDrvGetLocalDevIdByHostDevId(unsigned int devId, unsigned int* chipId);

int SSL_CTX_set_min_proto_version(SSL_CTX* ctx, int version);

int SSL_CTX_set_cipher_list(SSL_CTX* ctx, const char* str);

int SSL_CTX_use_certificate_chain_file(SSL_CTX* ctx, const char* file);

int SSL_CTX_load_verify_locations(SSL_CTX* ctx, const char* CAfile, const char* CApath);

int SSL_CTX_check_private_key(const SSL_CTX* ctx);

int SSL_CTX_use_PrivateKey(SSL_CTX* ctx, EVP_PKEY* pkey);

void EVP_PKEY_free(EVP_PKEY* x);

X509_STORE* SSL_CTX_get_cert_store(const SSL_CTX* ctx);

int X509_STORE_set_flags(X509_STORE* ctx, unsigned long flags);

long SSL_ctrl(SSL* s, int cmd, long larg, void* parg);

int X509_STORE_add_crl(X509_STORE* ctx, X509_CRL* x);

void X509_STORE_free(X509_STORE* vfy);

const SSL_METHOD* TLS_server_method(void);

const SSL_METHOD* TLS_client_method(void);

SSL_CTX* SSL_CTX_new(const SSL_METHOD* meth);

void SSL_CTX_free(SSL_CTX* ctx);

int SSL_shutdown(SSL* s);

void SSL_free(SSL* ssl);

SSL* SSL_new(SSL_CTX* ctx);

int SSL_get_error(const SSL* s, int retCode);

int SSL_set_fd(SSL* s, int fd);

long SSL_ctrl(SSL* ssl, int cmd, long larg, void* parg);

void SSL_set_connect_state(SSL* s);

void SSL_set_accept_state(SSL* s);

int SSL_do_handshake(SSL* s);

long SSL_get_verify_result(const SSL* ssl);

X509* SSL_get_peer_certificate(const SSL* s);

int SSL_write(SSL* ssl, const void* buf, int num);

int SSL_read(SSL* ssl, void* buf, int num);
#define STACK_OF(type) struct stack_st_##type
BIO* BIO_new_mem_buf(const void* buf, int len);

X509* d2i_X509_bio(BIO* bp, X509** x509);

X509* PEM_read_bio_X509(BIO* bp, X509** x, pem_password_cb* cb, void* u);

X509_STORE* X509_STORE_new(void);

X509_STORE_CTX* X509_STORE_CTX_new(void);

int X509_STORE_CTX_init(X509_STORE_CTX* ctx, X509_STORE* store, X509* x509, STACK_OF(X509) * chain);

int X509_verify_cert(X509_STORE_CTX* ctx);

int X509_STORE_CTX_get_error(X509_STORE_CTX* ctx);

const char* X509_verify_cert_error_string(long n);

void X509_STORE_CTX_cleanup(X509_STORE_CTX* ctx);

void X509_STORE_CTX_free(X509_STORE_CTX* ctx);

void X509_STORE_free(X509_STORE* vfy);

void X509_free(X509* buf);

void X509_STORE_CTX_trusted_stack(X509_STORE_CTX* ctx, STACK_OF(X509) * sk);

int tls_get_user_config(
    unsigned int saveMode, unsigned int chipId, const char* name, unsigned char* buf, unsigned int* bufSize);

void TlsGetEnableInfo(unsigned int saveMode, unsigned int chipId, unsigned char* buf, unsigned int bufSize);

int rs_ssl_get_crl_data(struct rs_cb* rscb, FILE* fp, struct TlsCertManageInfo* mngInfo, X509_CRL* crl);

int rs_get_pridata(struct rs_cb* rscb, struct RsSecPara* rsPara, struct tls_cert_mng_info* mngInfo);

int rs_ssl_put_certs(
    struct rs_cb* rscb, struct tls_cert_mng_info* mngInfo, struct RsCerts* certs, struct tls_ca_new_certs* newCerts,
    struct CertFile* fileName);

#define SLEEP_TIME 10000
#define rs_ut_msg(fmt, args...) fprintf(stderr, "\t>>>>> " fmt, ##args)

int tryAgain;
struct RsQpCb qpCbTmp2;
const char* sTmp = "suc";
struct RsConnInfo* gConnInfo;

int RsAllocConnNode(struct RsConnInfo** conn, unsigned short serverPort);
int RsGetConnInfo(
    struct RsConnCb* connCb, struct SocketConnectInfo* conn, struct RsConnInfo** connInfo, unsigned int serverPort);
int RsGetRsCb(unsigned int phyId, struct rs_cb** rsCb);
int RsQpExpCreate(struct RsQpCb* qpCb);
int RsNotifyMrListAdd(struct RsQpCb* qpCb, char* buf);
void RsEpollRecvHandle(struct RsQpCb* qpCb, char* buf, int size);
void RsDrvPollCqHandle(struct RsQpCb* qpCb);
int RsCreateCq(struct RsQpCb* qpCb);
int RsQpStateModify(struct RsQpCb* qpCb);
void RsEpollEventInHandle(struct rs_cb* rsCb, struct epoll_event* events);
void RsEpollEventHandleOne(struct rs_cb* rsCb, struct epoll_event* events);
int RsMrInfoSync(struct RsMrCb* mrCb);
int RsDrvGetGidIndex(struct RsRdevCb* rdevCb, struct ibv_port_attr* attr, int* idx);
void RsEpollCtl(int epollfd, int op, int fd, int state);
int RsSocketConnect(struct RsConnInfo* conn);
int RsPostRecvStub(struct ibv_qp* qp, struct ibv_recv_wr* wr, struct ibv_recv_wr** badWr);
int RsDrvGetRandomNum(int* randNum);
void RsSocketTagSync(struct RsConnInfo* conn);
int RsSocketStateInit(unsigned int chipId, struct RsConnInfo* conn, uint32_t sslEnable, struct rs_cb* rscb);
int RsFindSockets(struct RsConnInfo* connTmp, struct SocketFdData conn[], int num, int role);
int RsAllocClientConnNode(
    struct RsConnCb* connCb, enum RsConnRole role, struct RsConnInfo** conn, struct SocketConnectInfo* socketConn,
    int serverPort);
uint32_t RsSocketVnic2nodeid(uint32_t ipAddr);
int roce_set_tsqp_depth(
    const char* devName, unsigned int rdevIndex, unsigned int tempDepth, unsigned int* qpNum, unsigned int* sqDepth);
int roce_get_tsqp_depth(
    const char* devName, unsigned int rdevIndex, unsigned int* tempDepth, unsigned int* qpNum, unsigned int* sqDepth);
int RsSocketNodeid2vnic(uint32_t nodeId, uint32_t* ipAddr);
int RsServerValidAsyncInit(unsigned int chipId, struct RsConnInfo* conn, struct SocketWlistInfoT* whiteListExpect);
extern int RsConnectBindClient(int fd, struct RsConnInfo* conn);
extern void RsSocketGetBindByChip(unsigned int chipId, bool* bindIp);
extern int RsInitRscbCfg(struct rs_cb* rscb);
extern void RsDeinitRscbCfg(struct rs_cb* rscb);
extern int RsSocketCloseFd(int fd);
extern int RsFindWhiteList(struct RsConnCb* connCb, struct RsIpAddrInfo* serverIp, struct RsWhiteList** whiteList);
extern int RsFindWhiteListNode(
    struct RsWhiteList* rsSocketWhiteList, struct SocketWlistInfoT* whiteListExpect, int family,
    struct RsWhiteListInfo** whiteListNode);
extern int RsServerSendWlistCheckResult(struct RsConnInfo* conn, bool flag);
extern uint32_t RsGenerateUeInfo(uint32_t dieId, uint32_t funcId);
extern uint32_t RsGenerateDevIndex(uint32_t devCnt, uint32_t dieId, uint32_t funcId);
extern uint32_t RsGenerateMmapResId(uint32_t id, uint32_t dieId, uint32_t funcId);
extern int RsNetAdaptApiInit(void);
extern int RsConnectHandle(struct RsInitConfig* cfg);
extern int RsSocketStateReset(unsigned int chipId, struct RsConnInfo* conn, uint32_t sslEnable, struct rs_cb* rscb);
extern int RsCallocMr(unsigned int phyId, struct RsMrCb** mrCb);
extern int RsCallocQpcb(unsigned int phyId, struct RsQpCb** qpCb);
extern void RsSocketsServeripConverter(struct SocketFdData* sockets, int num, int role);
extern void
RsEpollRecvHandleRemain(struct RsQpCb* qpCb, unsigned int totalSize, unsigned int curSize, bool flag, char* bufTmp);
extern int RsSetFdNonblock(int fd);
extern int RsSocketSslConnect(struct RsConnInfo* conn, struct rs_cb* rscb);
extern int RsSocketStateConnected(struct RsConnInfo* conn, uint32_t sslEnable, struct rs_cb* rscb);
extern int RsSocketStateSslFdBind(struct RsConnInfo* conn, uint32_t sslEnable, struct rs_cb* rscb);
extern void RsDoSslHandshake(struct RsAcceptInfo* acceptInfo, struct rs_cb* rscb);
extern int rs_get_pk(struct rs_cb* rscb, struct tls_cert_mng_info* mngInfo, EVP_PKEY** pky);
extern int rs_ssl_crl_init(SSL_CTX* sslCtx, struct rs_cb* rscb, struct tls_cert_mng_info* mngInfo);
extern int rs_ssl_check_cert_chain(struct tls_cert_mng_info* mngInfo, struct RsCerts* certs);
extern int rs_remove_certs(const char* endFile, const char* caFile);
extern int rs_ssl_skid_get_from_chain(
    struct rs_cb* rscb, struct tls_cert_mng_info* mngInfo, struct RsCerts* certs, struct tls_ca_new_certs* newCerts);
extern int rs_ssl_init(struct rs_cb* rscb);
extern int RsTcpRecvTagInHandle(
    struct RsListenInfo* listenInfo, int fd, struct RsConnInfo* connTmp, struct RsIpAddrInfo* remoteIp);
extern void RsEpollEventTcpListenInHandle(
    struct rs_cb* rsCb, struct RsListenInfo* listenInfo, int fd, struct RsIpAddrInfo* remoteIp);
extern void RsSslRecvTagInHandle(struct RsAcceptInfo* acceptInfo, struct RsConnInfo* connTmp);
extern void RsServerValidAsync(unsigned int chipId, struct RsConnCb* connCb, struct RsConnInfo* conn);
extern int RsNetApiInit(void);

long unsigned int StubCalloc(long unsigned int num, long unsigned int size)
{
    static int hit = 0;
    if (hit == 1) {
        return 0;
    }
    hit++;
    return (unsigned long)malloc(num * size);
}

int StubIbvGetCqEvent(struct ibv_comp_channel* channel, struct ibv_cq** cq, void** cqContext)
{
    *cq = NULL;
    return 0;
}

struct RsWhiteListInfo gWhiteListNodeTmp = {0};
int StubRsFindWhiteListNode(
    struct RsWhiteList* rsSocketWhiteList, struct SocketWlistInfoT* whiteListExpect, int family,
    struct RsWhiteListInfo** whiteListNode)
{
    *whiteListNode = &gWhiteListNodeTmp;
    return 0;
}

int RsPostSendStub(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** badWr);

int RsCreateEpoll(struct rs_cb* rsCb);
int RsGetMrcb(struct RsQpCb* qpCb, uint64_t addr, struct RsMrCb** mrCb, struct RsListHead* mrList);

void* RsEpollHandle();
int memcpy_s(void* dest, size_t destMax, const void* src, size_t count);
int memset_s(void* dest, size_t destMax, int c, size_t count);

void TcRsInit2()
{
    int ret = 0;
    struct RsInitConfig cfg = {0};

    struct RsConnInfo* info = NULL;
    ret = RsFd2conn(0, &info);
    EXPECT_INT_EQ(ret, -ENODEV);

    cfg.hccpMode = NETWORK_OFFLINE;

    mocker((stub_fn_t)pthread_mutex_init, 20, 1);
    ret = RsInit(&cfg);
    EXPECT_INT_NE(ret, 0);
    ret = RsDeinit(&cfg);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)epoll_create, 20, -1);
    mocker((stub_fn_t)pthread_create, 20, -1);
    ret = RsInit(&cfg);
    EXPECT_INT_NE(ret, 0);
    ret = RsDeinit(&cfg);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)pthread_create, 20, -1);
    ret = RsInit(&cfg);
    EXPECT_INT_NE(ret, 0);
    ret = RsDeinit(&cfg);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker_ret((stub_fn_t)pthread_create, 0, -1, -1);
    ret = RsInit(&cfg);
    EXPECT_INT_NE(ret, 0);
    ret = RsDeinit(&cfg);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)pthread_detach, 10, -1);
    RsConnectHandle(&cfg);
    mocker_clean();

    cfg.chipId = 3;
    cfg.hccpMode = NETWORK_ONLINE;
    mocker((stub_fn_t)calloc, 10, NULL);
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, -12);
    mocker_clean();

    return;
}

void TcRsDeinit2()
{
    int ret = 0;
    uint32_t devId = 0;
    struct RsInitConfig cfg = {0};
    struct rs_cb* rsCb = NULL;

    /* resource prepare... */
    cfg.hccpMode = NETWORK_OFFLINE;
    cfg.chipId = 0;
    cfg.whiteListStatus = 1;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    /* resource free... */
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("!!!!!!tc_rs_deinit2: rs_deinit1\n");
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDev2rscb(devId, &rsCb, false);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    mocker_clean();
    rs_ut_msg("!!!!!!tc_rs_deinit2: rs_deinit2\n");

    struct rs_cb* rscb = NULL;
    rscb = calloc(1, sizeof(struct rs_cb));
    rscb->hccpMode = NETWORK_OFFLINE;
    RS_INIT_LIST_HEAD(&rscb->connCb.clientConnList);
    RsInitRscbCfg(rscb);
    RsDeinitRscbCfg(rscb);
    mocker_clean();
    free(rscb);
    rscb = NULL;
    return;
}

void TcRsRdevInit()
{
    int ret = 0;
    unsigned int rdevIndex = 0;
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 10;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");

    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_NE(ret, 0);

    rdevInfo.phyId = 0;

    mocker((stub_fn_t)RsGetRsCb, 20, 1);
    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)calloc, 20, NULL);
    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)pthread_mutex_init, 20, 1);
    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)ibv_get_device_list, 10, -1);
    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)ibv_exp_query_notify, 20, 1);
    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)ibv_alloc_pd, 20, 0);
    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)ibv_reg_mr, 20, 0);
    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();
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

extern __thread struct rs_cb* gRsCb;
void TcRsSocketDeinit()
{
    int ret = 0;
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");
    struct rs_cb gRsCbTmp = {0};
    gRsCb = &gRsCbTmp;

    mocker((stub_fn_t)RsDev2rscb, 20, 1);
    ret = RsSocketDeinit(rdevInfo);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    rdevInfo.family = 3;
    ret = RsSocketDeinit(rdevInfo);
    EXPECT_INT_NE(ret, 0);

    rdevInfo.phyId = 10;
    ret = RsSocketDeinit(rdevInfo);
    EXPECT_INT_NE(ret, 0);
    gRsCb = NULL;
}

void TcRsRdevDeinit()
{
    int ret = 0;
    unsigned int rdevIndex = 00;

    ret = RsRdevDeinit(10, NOTIFY, 1);
    EXPECT_INT_NE(ret, 0);

    ret = RsRdevDeinit(0, NOTIFY, rdevIndex);
    EXPECT_INT_NE(ret, 0);
}

void TcRsSocketListenStart2()
{
    int ret = 0;
    uint32_t devId = 0;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listenNode[2] = {0};
    struct rs_cb* rsCb = NULL;

    /* resource prepare... */
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    listenNode[0].phyId = 0;
    listenNode[0].family = AF_INET;
    listenNode[0].localIp.addr.s_addr = inet_addr("127.0.0.1");
    mocker((stub_fn_t)calloc, 10, NULL);
    listenNode[0].port = 16666;
    ret = RsSocketListenStart(listenNode, 1);
    mocker_clean();

    mocker((stub_fn_t)socket, 10, -1);
    listenNode[0].port = 16666;
    ret = RsSocketListenStart(listenNode, 1);
    mocker_clean();

    mocker((stub_fn_t)socket, 10, 0);
    mocker((stub_fn_t)setsockopt, 10, -1);
    listenNode[0].port = 16666;
    ret = RsSocketListenStart(listenNode, 1);
    mocker_clean();

    listenNode[0].family = AF_INET;
    listenNode[0].localIp.addr.s_addr = inet_addr("127.0.0.3");
    mocker((stub_fn_t)bind, 10, 1);
    listenNode[0].port = 16666;
    ret = RsSocketListenStart(listenNode, 1);
    EXPECT_INT_NE(ret, 1);
    mocker_clean();

    mocker((stub_fn_t)listen, 10, -1);
    listenNode[0].port = 16666;
    ret = RsSocketListenStart(listenNode, 1);
    EXPECT_INT_NE(ret, -1);
    mocker_clean();

    /* twice listen but first failed */
    struct SocketListenInfo listenTwice[2] = {0};
    listenTwice[0].localIp.addr.s_addr = inet_addr("127.0.0.4");
    listenTwice[1].phyId = 1;
    listenTwice[0].family = AF_INET;
    listenTwice[1].family = AF_INET;
    listenTwice[0].port = 16666;
    listenTwice[1].port = 16666;
    ret = RsSocketListenStart(listenTwice, 2);
    mocker_clean();

    /* resource free... */
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("!!!!!!tc_rs_socket_listen_start2: RsDeinit\n");
    return;
}

void TcRsSocketBatchConnect2()
{
    int ret = 0;
    uint32_t devId = 0;
    struct RsInitConfig cfg = {0};
    struct SocketConnectInfo connNode[2] = {0};
    struct RsConnInfo connSocketErr = {0};

    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->hccpMode = 1;
    connSocketErr.state = 1;
    mocker((stub_fn_t)socket, 10, -1);
    ret = RsSocketStateReset(0, &connSocketErr, gRsCb->sslEnable, gRsCb);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();
    free(gRsCb);
    gRsCb = NULL;

    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->hccpMode = 0;
    connSocketErr.state = 1;
    mocker((stub_fn_t)socket, 10, 1);
    ret = RsSocketStateReset(0, &connSocketErr, gRsCb->sslEnable, gRsCb);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    memset(gRsCb, 0, sizeof(struct rs_cb));
    connSocketErr.state = 4;
    mocker((stub_fn_t)RsSocketTagSync, 10, -1);
    ret = RsSocketConnectAsync(&connSocketErr, gRsCb);
    mocker_clean();
    free(gRsCb);
    gRsCb = NULL;

    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->hccpMode = 1;
    gRsCb->connCb.wlistEnable = 1;
    gRsCb->sslEnable = 1;
    connSocketErr.state = 7;
    mocker((stub_fn_t)RsSocketRecv, 1, -11);
    ret = RsSocketConnectAsync(&connSocketErr, gRsCb);
    mocker_clean();
    mocker((stub_fn_t)RsSocketRecv, 1, 0);
    mocker((stub_fn_t)SSL_shutdown, 1, 0);
    mocker((stub_fn_t)SSL_free, 1, 0);
    ret = RsSocketConnectAsync(&connSocketErr, gRsCb);
    mocker_clean();
    free(gRsCb);
    gRsCb = NULL;

    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->hccpMode = 1;
    connSocketErr.state = 1;
    mocker((stub_fn_t)RsSocketStateInit, 10, -1);
    ret = RsSocketConnectAsync(&connSocketErr, gRsCb);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
    free(gRsCb);
    gRsCb = NULL;

    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->hccpMode = 1;
    connSocketErr.state = 1;
    mocker((stub_fn_t)socket, 10, 0);
    mocker((stub_fn_t)connect, 10, -1);
    ret = RsSocketStateReset(0, &connSocketErr, 0, gRsCb);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();
    free(gRsCb);
    gRsCb = NULL;

    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->hccpMode = 1;
    connSocketErr.state = 1;
    mocker((stub_fn_t)connect, 10, -1);
    mocker((stub_fn_t)RsSocketTagSync, 10, 0);
    ret = RsSocketStateInit(0, &connSocketErr, 0, gRsCb);
    free(gRsCb);
    gRsCb = NULL;
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->hccpMode = 1;
    connSocketErr.state = 1;
    mocker((stub_fn_t)connect, 10, 0);
    mocker((stub_fn_t)getsockname, 10, 0);
    mocker((stub_fn_t)RsSocketTagSync, 10, 0);
    ret = RsSocketStateInit(0, &connSocketErr, 0, gRsCb);
    free(gRsCb);
    gRsCb = NULL;
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    /* resource prepare... */
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    connNode[0].phyId = 0;

    mocker((stub_fn_t)calloc, 10, NULL);
    connNode[0].port = 16666;
    ret = RsSocketBatchConnect(connNode, 1);
    mocker_clean();

    mocker((stub_fn_t)RsAllocClientConnNode, 10, 1);
    connNode[0].port = 16666;
    ret = RsSocketBatchConnect(connNode, 1);
    mocker_clean();

    rs_ut_msg("--------RsSocketBatchConnect--------\n");

    connNode[0].phyId = 0;
    connNode[0].family = AF_INET;
    strcpy(connNode[0].tag, "1234");
    connNode[1].phyId = 0;
    connNode[1].family = AF_INET6;
    strcpy(connNode[1].tag, "5678");
    connNode[0].port = 16666;
    connNode[1].port = 16666;
    ret = RsSocketBatchConnect(connNode, 2);

    /* resource free... */
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("!!!!!!tc_rs_socket_batch_connect2: RsDeinit\n");
    return;
}

void TcRsSetTsqpDepthAbnormal()
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

    ret = RsSetTsqpDepth(129, rdevIndex, tempDepth, &qpNum);
    EXPECT_INT_EQ(ret, -EINVAL);

    ret = RsSetTsqpDepth(phyId, rdevIndex, 1, &qpNum);
    EXPECT_INT_EQ(ret, -EINVAL);

    ret = RsSetTsqpDepth(phyId, rdevIndex, tempDepth, NULL);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker((stub_fn_t)DlDrvQueryProcessHostPid, 10, 1);
    ret = RsSetTsqpDepth(phyId, rdevIndex, tempDepth, &qpNum);
    EXPECT_INT_EQ(ret, 1);
    mocker_clean();

    mocker((stub_fn_t)RsRdev2rdevCb, 10, 1);
    ret = RsSetTsqpDepth(phyId, rdevIndex, tempDepth, &qpNum);
    EXPECT_INT_EQ(ret, 1);
    mocker_clean();

    mocker((stub_fn_t)roce_set_tsqp_depth, 10, 1);
    ret = RsSetTsqpDepth(phyId, rdevIndex, tempDepth, &qpNum);
    EXPECT_INT_EQ(ret, 1);
    mocker_clean();

    ret = RsRdevDeinit(phyId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("!!!!!!tc_rs_set_tsqp_depth_abnormal: RsDeinit\n");
    return;
}

void TcRsGetTsqpDepthAbnormal()
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

    ret = RsGetTsqpDepth(129, rdevIndex, &tempDepth, &qpNum);
    EXPECT_INT_EQ(ret, -EINVAL);

    ret = RsGetTsqpDepth(phyId, rdevIndex, &tempDepth, NULL);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker((stub_fn_t)DlDrvQueryProcessHostPid, 10, 1);
    ret = RsGetTsqpDepth(phyId, rdevIndex, &tempDepth, &qpNum);
    EXPECT_INT_EQ(ret, 1);
    mocker_clean();

    mocker((stub_fn_t)RsRdev2rdevCb, 10, 1);
    ret = RsGetTsqpDepth(phyId, rdevIndex, &tempDepth, &qpNum);
    EXPECT_INT_EQ(ret, 1);
    mocker_clean();

    mocker((stub_fn_t)roce_get_tsqp_depth, 10, 1);
    ret = RsGetTsqpDepth(phyId, rdevIndex, &tempDepth, &qpNum);
    EXPECT_INT_EQ(ret, 1);
    mocker_clean();

    ret = RsRdevDeinit(phyId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("!!!!!!tc_rs_get_tsqp_depth_abnormal: RsDeinit\n");
    return;
}

int stub_RsIbvQueryQp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attrMask, struct ibv_qp_init_attr* initAttr)
{
    if (attr == NULL) {
        return -EINVAL;
    }
    attr->qp_state = 3;
    return 0;
}

void TcRsQpCreate2()
{
    int ret = 0;
    uint32_t devId = 0;
    uint32_t flag = 0;
    unsigned int rdevIndex = 0;
    struct RsInitConfig cfg = {0};
    struct RsQpResp resp = {0};
    struct RsQpCb* qpCbT = NULL;

    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");

    struct RsQpNorm qpNorm = {0};
    qpNorm.flag = flag;
    qpNorm.qpMode = 1;
    qpNorm.isExp = 1;

    qpCbT = calloc(1, sizeof(struct RsQpCb));
    rs_ut_msg("____qp_cb_t:%p\n", qpCbT);

    /* resource prepare... */
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    mocker((stub_fn_t)calloc, 10, NULL);
    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)ibv_create_comp_channel, 10, NULL);
    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)RsDrvCreateCq, 10, 1);
    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)RsDrvQpCreate, 10, 1);
    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)ibv_req_notify_cq, 10, 1);
    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    struct RsQpCb qpCbAbnormal = {0};
    struct RsRdevCb rdevCbAbnormal = {0};
    struct rs_cb rsCbAbnormal = {0};
    struct ibv_qp ibQpAbnormal = {0};
    rdevCbAbnormal.rsCb = &rsCbAbnormal;
    qpCbAbnormal.rdevCb = &rdevCbAbnormal;
    qpCbAbnormal.ibQp = &ibQpAbnormal;
    mocker((stub_fn_t)memset_s, 10, 1);
    ret = RsQpStateModify(&qpCbAbnormal);
    EXPECT_INT_NE(ret, 1);
    mocker_clean();

    mocker_invoke(RsIbvQueryQp, stub_RsIbvQueryQp, 10);
    ret = RsQpStateModify(&qpCbAbnormal);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)memset_s, 10, 1);
    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)ibv_exp_create_qp, 10, NULL);
    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)ibv_query_qp, 10, 1);
    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)ibv_modify_qp, 10, 1);
    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)ibv_query_port, 10, 1);
    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)RsDrvGetGidIndex, 10, 1);
    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)RsDrvGetGidIndex, 20, 0);
    mocker((stub_fn_t)ibv_query_gid, 20, 1);
    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)memcpy_s, 10, 1);
    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker_ret((stub_fn_t)memset_s, 0, 1, 0);
    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker_ret((stub_fn_t)memset_s, 0, 0, 1);
    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_EQ(ret, -ESAFEFUNC);
    mocker_clean();

    mocker((stub_fn_t)RsDrvGetRandomNum, 10, -1);
    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker((stub_fn_t)open, 10, -1);
    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_EQ(ret, -EFILEOPER);
    mocker_clean();

    mocker((stub_fn_t)read, 10, -1);
    mocker((stub_fn_t)close, 10, -1);
    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_EQ(ret, -EFILEOPER);
    mocker_clean();

    /* resource free... */
    ret = RsRdevDeinit(devId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("!!!!!!tc_rs_qp_create2: RsDeinit\n");
    free(qpCbT);

    return;
}

void TcRsEpollOps2()
{
    int ret = 0;
    uint32_t devId = 0;
    unsigned int rdevIndex = 0;
    int flag = 0; /* RC */
    struct RsQpResp resp = {0};
    struct RsQpResp resp2 = {0};
    int i = 0;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listen[2] = {0};
    struct SocketConnectInfo conn[2] = {0};
    struct RsSocketCloseInfoT sockClose[2] = {0};
    struct SocketFdData socketInfo[3] = {0};
    struct rs_cb* rsCb = NULL;

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    ret = RsDev2rscb(devId, &rsCb, false);
    EXPECT_INT_EQ(ret, 0);
    rsCb->connCb.wlistEnable = 0;

    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.3");
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 1);

    usleep(SLEEP_TIME);

    mocker((stub_fn_t)strcpy_s, 10, -1);
    conn[0].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 1);
    EXPECT_INT_EQ(ret, -22);
    mocker_clean();

    conn[0].phyId = 0;
    conn[0].family = AF_INET;
    conn[0].localIp.addr.s_addr = inet_addr("127.0.0.3");
    conn[0].remoteIp.addr.s_addr = inet_addr("127.0.0.3");
    conn[0].tag[0] = 1;
    conn[0].tag[1] = 2;
    conn[0].tag[2] = 3;
    conn[0].tag[3] = 4;
    conn[0].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 1);

    usleep(SLEEP_TIME);

    i = 0;
    socketInfo[i].family = AF_INET;
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].tag[0] = 1;
    socketInfo[i].tag[1] = 2;
    socketInfo[i].tag[2] = 3;
    socketInfo[i].tag[3] = 4;

    tryAgain = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        usleep(30000);
        tryAgain--;
    } while (ret != 1 && tryAgain);
    rs_ut_msg("%s [client]socket_info[0].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");

    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    struct RsQpNorm qpNorm = {0};
    qpNorm.flag = flag;
    qpNorm.qpMode = 1;
    qpNorm.isExp = 1;

    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("RsQpCreate: qpn %d, ret:%d\n", resp.qpn, ret);

    ret = RsQpConnectAsync(devId, rdevIndex, resp.qpn, socketInfo[i].fd);
    rs_ut_msg("***RsQpConnectAsync: %d****\n", ret);

    usleep(SLEEP_TIME);

    struct SocketConnectInfo connCtl = {0};
    connCtl.phyId = 0;
    connCtl.family = AF_INET;
    connCtl.localIp.addr.s_addr = inet_addr("127.0.0.3");
    connCtl.remoteIp.addr.s_addr = inet_addr("127.0.0.3");
    memset(connCtl.tag, 0, 128);
    strcpy(connCtl.tag, "abcde");
    connCtl.port = 16666;
    ret = RsSocketBatchConnect(&connCtl, 1);

    usleep(SLEEP_TIME);
    usleep(SLEEP_TIME);

    struct SocketFdData infoCtl = {0};
    infoCtl.phyId = 0;
    infoCtl.family = AF_INET;
    infoCtl.localIp.addr.s_addr = inet_addr("127.0.0.3");
    infoCtl.remoteIp.addr.s_addr = inet_addr("127.0.0.3");
    memset(infoCtl.tag, 0, 128);
    strcpy(infoCtl.tag, "abcde");

    tryAgain = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &infoCtl, 1);
        usleep(30000);
        tryAgain--;
    } while (ret != 1 && tryAgain);

    struct RsQpResp respCtl = {0};
    qpNorm.flag = flag;
    qpNorm.qpMode = 1;
    qpNorm.isExp = 1;
    ret = RsQpCreate(devId, rdevIndex, qpNorm, &respCtl);
    EXPECT_INT_EQ(ret, 0);

    mocker((stub_fn_t)RsEpollCtl, 10, -2);
    ret = RsQpConnectAsync(devId, rdevIndex, respCtl.qpn, infoCtl.fd);
    mocker_clean();

    mocker((stub_fn_t)RsSocketSend, 10, -1);
    ret = RsQpConnectAsync(devId, rdevIndex, respCtl.qpn, infoCtl.fd);
    mocker_clean();

    ret = RsQpDestroy(devId, rdevIndex, respCtl.qpn);

    /* ===RsEpollEventInHandle ut begin --- accept fail=== */
    struct epoll_event events = {0};
    struct rs_cb* rsCbT = NULL;
    struct RsRdevCb* rdevCb = NULL;
    struct RsListenInfo *listenInfo, *listenInfo2;

    ret = RsDev2rscb(0, &rsCbT, false);

    RS_LIST_GET_HEAD_ENTRY(listenInfo, listenInfo2, &rsCbT->connCb.listenList, list, struct RsListenInfo);
    for (; (&listenInfo->list) != &rsCbT->connCb.listenList;
         listenInfo = listenInfo2, listenInfo2 = list_entry(listenInfo2->list.next, struct RsListenInfo, list)) {
        events.data.fd = listenInfo->listenFd;
    }
    events.events = EPOLLIN;
    mocker((stub_fn_t)accept, 10, -1);
    RsEpollEventInHandle(rsCbT, &events);
    RsEpollEventInHandle(rsCbT, &events);
    RsEpollEventInHandle(rsCbT, &events);
    mocker_clean();

    mocker((stub_fn_t)calloc, 10, NULL);
    struct RsConnInfo* conCbNode = NULL;
    ret = RsAllocConnNode(&conCbNode, 16666);
    EXPECT_INT_EQ(ret, -12);

    mocker((stub_fn_t)accept, 1, 9900999);
    mocker((stub_fn_t)RsAllocConnNode, 1, -1);
    RsEpollEventInHandle(rsCbT, &events);
    mocker_clean();

    /* poll cq */
    struct RsQpCb *qpCbT, *qpCbT2;
    ret = RsGetRdevCb(rsCb, rdevIndex, &rdevCb);
    RS_LIST_GET_HEAD_ENTRY(qpCbT, qpCbT2, &rdevCb->qpList, list, struct RsQpCb);
    for (; (&qpCbT->list) != &rdevCb->qpList;
         qpCbT = qpCbT2, qpCbT2 = list_entry(qpCbT2->list.next, struct RsQpCb, list)) {
        events.data.fd = qpCbT->channel->fd;
    }
    RsEpollEventInHandle(rsCbT, &events);

    /* qp info message, RsSocketRecv = 0 error ! */
    ret = RsQpn2qpcb(devId, rdevIndex, resp.qpn, &qpCbT);
    EXPECT_INT_EQ(ret, 0);
    if (qpCbT->connInfo == NULL) {
        return;
    }
    int fdTmp = qpCbT->connInfo->connfd;
    qpCbT->connInfo->connfd = 99999;
    mocker((stub_fn_t)RsSocketRecv, 1, 0);
    events.data.fd = qpCbT->connInfo->connfd;
    RsEpollEventInHandle(rsCbT, &events);
    qpCbT->connInfo->connfd = fdTmp;
    mocker_clean();

    events.events = EPOLLOUT;
    RsEpollEventHandleOne(rsCbT, &events);

    mocker((stub_fn_t)pthread_detach, 1, 1);
    RsEpollHandle(rsCbT);
    mocker_clean();

    int epollfdT = rsCbT->connCb.epollfd;
    int eventfdT = rsCbT->connCb.eventfd;
    mocker((stub_fn_t)epoll_create, 1, 1);
    mocker((stub_fn_t)eventfd, 1, -1);

    ret = RsCreateEpoll(rsCbT);
    EXPECT_INT_NE(ret, 0);
    rsCbT->connCb.epollfd = epollfdT;
    rsCbT->connCb.eventfd = eventfdT;
    mocker_clean();

    i = 1;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].tag[0] = 1;
    socketInfo[i].tag[1] = 2;
    socketInfo[i].tag[2] = 3;
    socketInfo[i].tag[3] = 4;

    int tryAgain = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_SERVER, &socketInfo[i], 1);
        usleep(30000);
        tryAgain--;
    } while (ret != 1 && tryAgain);
    rs_ut_msg("%s [server]socket_info[1].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    qpNorm.flag = flag;
    qpNorm.qpMode = 1;
    qpNorm.isExp = 1;

    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp2);
    EXPECT_INT_EQ(ret, 0);

    rs_ut_msg("RsQpCreate: qpn2 %d, ret:%d\n", resp2.qpn, ret);

    ret = RsQpConnectAsync(devId, rdevIndex, resp2.qpn, socketInfo[i].fd);

    usleep(SLEEP_TIME);

    usleep(SLEEP_TIME);

    mocker((stub_fn_t)RsEpollCtl, 10, -1);
    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 1);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    struct SocketListenInfo listenCtl = {0};
    listenCtl.phyId = 0;
    listenCtl.localIp.addr.s_addr = inet_addr("127.0.0.9");
    listenCtl.port = 16666;
    mocker((stub_fn_t)RsEpollCtl, 10, -1);
    ret = RsSocketListenStart(&listenCtl, 1);
    mocker_clean();

    usleep(SLEEP_TIME);

    struct rs_cb rsCbCtl = {0};
    mocker((stub_fn_t)RsEpollCtl, 10, -1);
    ret = RsCreateEpoll(&rsCbCtl);
    mocker_clean();

    ret = RsQpDestroy(devId, rdevIndex, resp2.qpn);
    ret = RsQpDestroy(devId, rdevIndex, resp.qpn);

    sockClose[0].fd = socketInfo[0].fd;
    ret = RsSocketBatchClose(0, &sockClose[0], 1);

    sockClose[1].fd = socketInfo[1].fd;
    ret = RsSocketBatchClose(0, &sockClose[1], 1);
    /* ------Resource CLEAN-------- */
    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 1);

    ret = RsRdevDeinit(devId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("!!!!!!tc_rs_epoll_ops2: RsDeinit\n");

    return;
}

int StubRsEpollCtl(int epollfd, int op, int fd, int state)
{
    if (op == EPOLL_CTL_ADD)
        return 0;
    return 1;
}

void TcRsQpConnectAsync2()
{
    int ret = 0;
    uint32_t devId = 0;
    unsigned int rdevIndex = 0;
    int flag = 0; /* RC */
    struct RsQpResp resp = {0};
    struct RsQpResp resp2 = {0};
    int i = 0;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listen[2] = {0};
    struct SocketConnectInfo conn[2] = {0};
    struct RsSocketCloseInfoT sockClose[2] = {0};
    struct SocketFdData socketInfo[3] = {0};
    struct rs_cb* rsCb = NULL;

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    ret = RsDev2rscb(devId, &rsCb, false);
    EXPECT_INT_EQ(ret, 0);
    rsCb->connCb.wlistEnable = 0;

    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.3");
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 1);

    usleep(SLEEP_TIME);

    conn[0].phyId = 0;
    conn[0].family = AF_INET;
    conn[0].localIp.addr.s_addr = inet_addr("127.0.0.3");
    conn[0].remoteIp.addr.s_addr = inet_addr("127.0.0.3");
    conn[0].tag[0] = 1;
    conn[0].tag[1] = 2;
    conn[0].tag[2] = 3;
    conn[0].tag[3] = 4;
    conn[0].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 1);

    usleep(SLEEP_TIME);
    usleep(SLEEP_TIME);

    i = 0;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].tag[0] = 1;
    socketInfo[i].tag[1] = 2;
    socketInfo[i].tag[2] = 3;
    socketInfo[i].tag[3] = 4;

    int tryAgain = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        usleep(30000);
        tryAgain--;
    } while (ret != 1 && tryAgain);
    rs_ut_msg("%s [client]socket_info[0].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].tag[0] = 1;
    socketInfo[i].tag[1] = 2;
    socketInfo[i].tag[2] = 3;
    socketInfo[i].tag[3] = 4;

    mocker((stub_fn_t)RsFindSockets, 10, 2);
    ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 2);
    mocker_clean();

    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");

    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    struct RsQpNorm qpNorm = {0};
    qpNorm.flag = flag;
    qpNorm.qpMode = 1;
    qpNorm.isExp = 1;

    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_EQ(ret, 0);

    rs_ut_msg("RsQpCreate: qpn %d, ret:%d\n", resp.qpn, ret);

    mocker_invoke(RsEpollCtl, StubRsEpollCtl, 10);
    ret = RsQpConnectAsync(devId, rdevIndex, resp.qpn, socketInfo[i].fd);
    mocker_clean();

    struct RsQpCb* qpCb = NULL;
    ret = RsQpn2qpcb(devId, rdevIndex, resp.qpn, &qpCb);
    EXPECT_INT_EQ(ret, 0);
    qpCb->state = RS_QP_STATUS_DISCONNECT;

    ret = RsQpConnectAsync(devId, rdevIndex, resp.qpn, socketInfo[i].fd);
    rs_ut_msg("***RsQpConnectAsync: %d****\n", ret);

    usleep(SLEEP_TIME);

    /* ===RsQpConnectAsync ut begin === */
    struct RsQpCb* qpCbT = NULL;

    ret = RsQpn2qpcb(devId, rdevIndex, resp.qpn, &qpCbT);
    EXPECT_INT_EQ(ret, 0);
    int stateTmp = qpCbT->state;
    qpCbT->state = RS_QP_STATUS_CONNECTED;
    ret = RsQpConnectAsync(devId, rdevIndex, resp.qpn, socketInfo[i].fd);
    EXPECT_INT_NE(ret, 0);
    qpCbT->state = stateTmp;

    /* ===RsQpConnectAsync ut end === */

    i = 1;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].tag[0] = 1;
    socketInfo[i].tag[1] = 2;
    socketInfo[i].tag[2] = 3;
    socketInfo[i].tag[3] = 4;

    tryAgain = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_SERVER, &socketInfo[i], 1);
        usleep(30000);
        tryAgain--;
    } while (ret != 1 && tryAgain);
    rs_ut_msg("%s [server]socket_info[1].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp2);
    EXPECT_INT_EQ(ret, 0);

    rs_ut_msg("RsQpCreate: qpn2 %d, ret:%d\n", resp2.qpn, ret);

    ret = RsQpConnectAsync(devId, rdevIndex, resp2.qpn, socketInfo[i].fd);

    usleep(SLEEP_TIME);

    ret = RsQpDestroy(devId, rdevIndex, resp2.qpn);
    ret = RsQpDestroy(devId, rdevIndex, resp.qpn);

    sockClose[0].fd = socketInfo[0].fd;
    ret = RsSocketBatchClose(0, &sockClose[0], 1);

    sockClose[1].fd = socketInfo[1].fd;
    ret = RsSocketBatchClose(0, &sockClose[1], 1);

    /* ------Resource CLEAN-------- */
    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 1);

    ret = RsRdevDeinit(devId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("!!!!!!tc_rs_qp_connect_async2: RsDeinit\n");

    return;
}

void TcRsSendWr2()
{
    int ret = 0;
    uint32_t devId = 0;
    unsigned int rdevIndex = 0;
    int flag = 0; /* RC */
    struct RsQpResp resp = {0};
    struct RsQpResp resp2 = {0};
    int i = 0;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listen[2] = {0};
    struct SocketConnectInfo conn[2] = {0};
    struct RsSocketCloseInfoT sockClose[2] = {0};
    struct SocketFdData socketInfo[3] = {0};
    struct rs_cb* rsCb = NULL;

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    ret = RsDev2rscb(devId, &rsCb, false);
    EXPECT_INT_EQ(ret, 0);
    rsCb->connCb.wlistEnable = 0;

    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.3");
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 1);

    usleep(SLEEP_TIME);

    conn[0].phyId = 0;
    conn[0].family = AF_INET;
    conn[0].localIp.addr.s_addr = inet_addr("127.0.0.3");
    conn[0].remoteIp.addr.s_addr = inet_addr("127.0.0.3");
    conn[0].tag[0] = 1;
    conn[0].tag[1] = 2;
    conn[0].tag[2] = 3;
    conn[0].tag[3] = 4;
    conn[0].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 1);

    usleep(SLEEP_TIME);

    i = 0;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].tag[0] = 1;
    socketInfo[i].tag[1] = 2;
    socketInfo[i].tag[2] = 3;
    socketInfo[i].tag[3] = 4;

    tryAgain = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        usleep(30000);
        tryAgain--;
    } while (ret != 1 && tryAgain);
    rs_ut_msg("%s [client]socket_info[0].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");

    ret = RsRdevInit(rdevInfo, NOTIFY, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    struct RsQpNorm qpNorm = {0};
    qpNorm.flag = flag;
    qpNorm.qpMode = 1;
    qpNorm.isExp = 1;

    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("RsQpCreate: qpn %d, ret:%d\n", resp.qpn, ret);

    ret = RsQpConnectAsync(devId, rdevIndex, resp.qpn, socketInfo[i].fd);
    rs_ut_msg("***RsQpConnectAsync: %d****\n", ret);

    usleep(SLEEP_TIME);

    /* === rs_send_async ut begin === */

    /* === rs_send_async ut end === */

    i = 1;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].tag[0] = 1;
    socketInfo[i].tag[1] = 2;
    socketInfo[i].tag[2] = 3;
    socketInfo[i].tag[3] = 4;

    tryAgain = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_SERVER, &socketInfo[i], 1);
        usleep(30000);
        tryAgain--;
    } while (ret != 1 && tryAgain);
    rs_ut_msg("%s [server]socket_info[1].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp2);
    EXPECT_INT_EQ(ret, 0);

    rs_ut_msg("RsQpCreate: qpn2 %d, ret:%d\n", resp2.qpn, ret, 1);

    ret = RsQpConnectAsync(devId, rdevIndex, resp2.qpn, socketInfo[i].fd);

    usleep(SLEEP_TIME);

    ret = RsQpDestroy(devId, rdevIndex, resp2.qpn);
    ret = RsQpDestroy(devId, rdevIndex, resp.qpn);

    sockClose[0].fd = socketInfo[0].fd;
    ret = RsSocketBatchClose(0, &sockClose[0], 1);

    sockClose[1].fd = socketInfo[1].fd;
    ret = RsSocketBatchClose(0, &sockClose[1], 1);

    /* ------Resource CLEAN-------- */
    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 1);

    ret = RsRdevDeinit(devId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("!!!!!!tc_rs_send_wr2: RsDeinit\n");

    return;
}

void TcRsGetGidIndex2()
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
    struct SocketListenInfo listen[2] = {0};
    struct SocketConnectInfo conn[2] = {0};
    struct RsSocketCloseInfoT sockClose[2] = {0};
    struct SocketFdData socketInfo[3] = {0};
    struct rs_cb* rsCb = NULL;

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    ret = RsDev2rscb(devId, &rsCb, false);
    EXPECT_INT_EQ(ret, 0);
    rsCb->connCb.wlistEnable = 0;

    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.3");
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 1);

    usleep(SLEEP_TIME);

    conn[0].phyId = 0;
    conn[0].family = AF_INET;
    conn[0].localIp.addr.s_addr = inet_addr("127.0.0.3");
    conn[0].remoteIp.addr.s_addr = inet_addr("127.0.0.3");
    conn[0].tag[0] = 1;
    conn[0].tag[1] = 2;
    conn[0].tag[2] = 3;
    conn[0].tag[3] = 4;
    conn[0].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 1);

    usleep(1000);

    i = 0;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].tag[0] = 1;
    socketInfo[i].tag[1] = 2;
    socketInfo[i].tag[2] = 3;
    socketInfo[i].tag[3] = 4;

    tryAgain = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        usleep(30000);
        tryAgain--;
    } while (ret != 1 && tryAgain);
    rs_ut_msg("%s [client]socket_info[0].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

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
    rs_ut_msg("RsQpCreate: qpn %d, ret:%d\n", resp.qpn, ret);

    ret = RsQpConnectAsync(devId, rdevIndex, resp.qpn, socketInfo[i].fd);
    rs_ut_msg("***RsQpConnectAsync: %d****\n", ret);

    usleep(SLEEP_TIME);

    /* ===rs_get_gid_index ut begin=== */
    struct ibv_port_attr attr = {0};
    int index = 0;
    struct rs_cb* rsCbT = NULL;
    struct RsRdevCb rdevCb = {0};
    ret = RsDev2rscb(0, &rsCbT, false);
    attr.gid_tbl_len = 3;
    rdevCb.ibPort = 0;

    ret = RsDrvGetGidIndex(&rdevCb, &attr, &index);
    EXPECT_INT_NE(ret, 0);

    attr.state = IBV_PORT_ACTIVE;
    mocker((stub_fn_t)ibv_query_gid_type, 20, 1);
    ret = RsDrvGetGidIndex(&rdevCb, &attr, &index);
    mocker_clean();

    mocker((stub_fn_t)ibv_query_gid, 20, 1);
    ret = RsDrvGetGidIndex(&rdevCb, &attr, &index);
    mocker_clean();

    mocker_ret((stub_fn_t)ibv_query_gid, 0, 1, 1);
    ret = RsDrvGetGidIndex(&rdevCb, &attr, &index);
    mocker_clean();

    mocker((stub_fn_t)ibv_query_gid, 20, 0);
    mocker_ret((stub_fn_t)RsDrvCompareIpGid, 0, 1, 0);
    ret = RsDrvGetGidIndex(&rdevCb, &attr, &index);

    mocker((stub_fn_t)ibv_query_gid, 20, 0);
    mocker_ret((stub_fn_t)RsDrvCompareIpGid, 1, 0, 1);
    rdevCb.localIp.family = AF_INET6;
    ret = RsDrvGetGidIndex(&rdevCb, &attr, &index);
    mocker_clean();
    /* ===rs_get_gid_index ut end=== */

    i = 1;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].tag[0] = 1;
    socketInfo[i].tag[1] = 2;
    socketInfo[i].tag[2] = 3;
    socketInfo[i].tag[3] = 4;

    tryAgain = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_SERVER, &socketInfo[i], 1);
        usleep(30000);
        tryAgain--;
    } while (ret != 1 && tryAgain);
    rs_ut_msg("%s [server]socket_info[1].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp2);
    EXPECT_INT_EQ(ret, 0);

    rs_ut_msg("RsQpCreate: qpn2 %d, ret:%d\n", resp2.qpn, ret);

    ret = RsQpConnectAsync(devId, rdevIndex, resp2.qpn, socketInfo[i].fd);

    usleep(SLEEP_TIME);

    ret = RsQpDestroy(devId, rdevIndex, resp2.qpn);
    ret = RsQpDestroy(devId, rdevIndex, resp.qpn);

    sockClose[0].fd = socketInfo[0].fd;
    ret = RsSocketBatchClose(0, &sockClose[0], 1);

    sockClose[1].fd = socketInfo[1].fd;
    ret = RsSocketBatchClose(0, &sockClose[1], 1);

    /* ------Resource CLEAN-------- */
    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 1);

    ret = RsRdevDeinit(devId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("!!!!!!tc_rs_get_gid_index2:RsDeinit\n");

    return;
}

void TcRsMrAbnormal2()
{
    int ret = 0;
    uint32_t devId = 0;
    unsigned int rdevIndex = 0;
    int flag = 0; /* RC */
    uint32_t qpMode = 1;
    struct RsQpResp resp = {0};
    struct RsQpResp resp2 = {0};
    int i = 0;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listen[2] = {0};
    struct SocketConnectInfo conn[2] = {0};
    struct RsSocketCloseInfoT sockClose[2] = {0};
    struct SocketFdData socketInfo[3] = {0};
    struct rs_cb* rsCb = NULL;

    struct RsMrCb* mrCbNormal = NULL;
    ret = RsCallocMr(0, &mrCbNormal);
    EXPECT_INT_EQ(ret, -EINVAL);

    struct RsQpCb* qpCbNormal = NULL;
    ret = RsCallocQpcb(0, &qpCbNormal);
    EXPECT_INT_EQ(ret, -EINVAL);
    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    ret = RsDev2rscb(devId, &rsCb, false);
    EXPECT_INT_EQ(ret, 0);
    rsCb->connCb.wlistEnable = 0;

    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.3");
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 1);

    usleep(SLEEP_TIME);

    conn[0].phyId = 0;
    conn[0].family = AF_INET;
    conn[0].localIp.addr.s_addr = inet_addr("127.0.0.3");
    conn[0].remoteIp.addr.s_addr = inet_addr("127.0.0.3");
    conn[0].tag[0] = 1;
    conn[0].tag[1] = 2;
    conn[0].tag[2] = 3;
    conn[0].tag[3] = 4;
    conn[0].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 1);

    usleep(10000);

    i = 0;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].tag[0] = 1;
    socketInfo[i].tag[1] = 2;
    socketInfo[i].tag[2] = 3;
    socketInfo[i].tag[3] = 4;

    tryAgain = 100;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        usleep(300000);
        tryAgain--;
    } while (ret != 1 && tryAgain);
    rs_ut_msg("%s [client]socket_info[0].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

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

    rs_ut_msg("RsQpCreate: qpn %d, ret:%d\n", resp.qpn, ret);

    ret = RsQpConnectAsync(devId, rdevIndex, resp.qpn, socketInfo[i].fd);
    rs_ut_msg("***RsQpConnectAsync: %d****\n", ret);

    usleep(SLEEP_TIME);

    /* ===RsMrInfoSync ut begin=== */
    void* addr = NULL;
    struct RsMrCb* mrCb = NULL;
    addr = malloc(RS_TEST_MEM_SIZE);
    struct RdmaMrRegInfo mrRegInfo = {0};
    mrRegInfo.addr = addr;
    mrRegInfo.len = RS_TEST_MEM_SIZE;
    mrRegInfo.access = RS_ACCESS_LOCAL_WRITE;
    int tryNum = 3;
    do {
        ret = RsMrReg(devId, rdevIndex, resp.qpn, &mrRegInfo);
        EXPECT_INT_EQ(ret, 0);
        if (0 == ret)
            break;
        rs_ut_msg("MR REG1: qpn %d, ret:%d\n", resp.qpn, ret);
        tryNum--;
        usleep(3000);
    } while (tryNum && (-EAGAIN == ret));
    EXPECT_INT_EQ(ret, 0);

    struct RsQpCb* qpCbT = NULL;
    ret = RsQpn2qpcb(devId, rdevIndex, resp.qpn, &qpCbT);
    EXPECT_INT_EQ(ret, 0);
    ret = RsGetMrcb(qpCbT, (uint64_t)addr, &mrCb, &qpCbT->mrList);
    if (mrCb->qpCb->connInfo == NULL) {
        free(addr);
        return;
    }
    int connfdTmp = mrCb->qpCb->connInfo->connfd;
    mrCb->qpCb->connInfo->connfd = RS_FD_INVALID;
    mrCb->state = 0;
    ret = RsMrInfoSync(mrCb);
    mrCb->qpCb->connInfo->connfd = connfdTmp;

    int stateTmp2 = mrCb->state;
    mrCb->state = 0;
    mocker((stub_fn_t)RsSocketSend, 20, 0);
    ret = RsMrInfoSync(mrCb);
    EXPECT_INT_NE(ret, 0);
    mrCb->state = stateTmp2;

    mocker_clean();

    /* ===RsQpDestroy ut begin=== */
    ret = RsQpDestroy(devId, rdevIndex, resp.qpn);
    /* ===RsQpDestroy ut end=== */

    ret = RsMrDereg(devId, rdevIndex, resp.qpn, addr);
    free(addr);
    /* ===RsMrInfoSync ut end=== */

    i = 1;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].tag[0] = 1;
    socketInfo[i].tag[1] = 2;
    socketInfo[i].tag[2] = 3;
    socketInfo[i].tag[3] = 4;

    tryAgain = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_SERVER, &socketInfo[i], 1);
        usleep(30000);
        tryAgain--;
    } while (ret != 1 && tryAgain);
    rs_ut_msg("%s [server]socket_info[1].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp2);
    EXPECT_INT_EQ(ret, 0);

    rs_ut_msg("RsQpCreate: qpn2 %d, ret:%d\n", resp2.qpn, ret);

    ret = RsQpConnectAsync(devId, rdevIndex, resp2.qpn, socketInfo[i].fd);

    usleep(SLEEP_TIME);

    ret = RsQpDestroy(devId, rdevIndex, resp2.qpn);

    sockClose[0].fd = socketInfo[0].fd;
    ret = RsSocketBatchClose(0, &sockClose[0], 1);

    sockClose[1].fd = socketInfo[1].fd;
    ret = RsSocketBatchClose(0, &sockClose[1], 1);

    /* ------Resource CLEAN-------- */
    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 1);

    ret = RsRdevDeinit(devId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("!!!!!!tc_rs_mr_abnormal2:RsDeinit\n");

    return;
}

int stub_halGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t* value)
{
    *value = 1;
    return 0;
}

void TcRsSocketOps2()
{
    int ret = 0;
    uint32_t devId = 0;
    unsigned int rdevIndex = 0;
    int flag = 0; /* RC */
    struct RsQpResp resp = {0};
    struct RsQpResp resp2 = {0};
    uint32_t qpMode = 1;
    int i = 0;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listen[2] = {0};
    struct SocketConnectInfo conn[2] = {0};
    struct RsSocketCloseInfoT sockClose[2] = {0};
    struct SocketFdData socketInfo[3] = {0};
    struct SocketWlistInfoT whiteList = {0};
    struct rs_cb* rsCb = NULL;

    ret = RsSocketVnic2nodeid(0);
    EXPECT_INT_EQ(ret, 0);

    struct SocketFdData connServer = {0};
    connServer.localIp.addr.s_addr = 1;
    RsSocketsServeripConverter(&connServer, 1, 1);

    ret = RsGetSockets(0, &connServer, 1);
    EXPECT_INT_NE(ret, 0);

    struct RsConnInfo connTmp = {0};
    connTmp.state = 0;
    connTmp.serverIp.family = AF_INET;
    connTmp.serverIp.binAddr.addr.s_addr = 2;

    connTmp.state = 0;
    connServer.localIp.addr.s_addr = 1;
    connTmp.serverIp.family = AF_INET;
    connTmp.serverIp.binAddr.addr.s_addr = 1;
    memset(connServer.tag, 0, sizeof(connServer.tag));
    memset(connTmp.tag, 1, sizeof(connTmp.tag));

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 1;
    cfg.hccpMode = NETWORK_OFFLINE;
    mocker((stub_fn_t)halGetDeviceInfo, 10, -1);
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    mocker((stub_fn_t)system, 10, 0);
    mocker((stub_fn_t)access, 10, 0);
    mocker_invoke((stub_fn_t)halGetDeviceInfo, stub_halGetDeviceInfo, 10);
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDev2rscb(devId, &rsCb, false);
    EXPECT_INT_EQ(ret, 0);
    rsCb->connCb.wlistEnable = 0;

    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.3");
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 1);

    usleep(SLEEP_TIME);
    strcpy(whiteList.tag, "1234");

    conn[0].phyId = 0;
    conn[0].family = AF_INET;
    conn[0].localIp.addr.s_addr = inet_addr("127.0.0.3");
    conn[0].remoteIp.addr.s_addr = inet_addr("127.0.0.3");
    conn[0].tag[0] = 1;
    conn[0].tag[1] = 2;
    conn[0].tag[2] = 3;
    conn[0].tag[3] = 4;
    conn[0].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 1);

    i = 0;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].tag[0] = 1;
    socketInfo[i].tag[1] = 2;
    socketInfo[i].tag[2] = 3;
    socketInfo[i].tag[3] = 4;

    {
        /* ===RsGetSockets ut begin=== */
        struct RsConnCb* connCb = NULL;
        struct RsConnInfo *connTmp, *connTmp2;
        int stateTmp = 0;
        ret = RsDev2conncb(devId, &connCb);
        RS_LIST_GET_HEAD_ENTRY(connTmp, connTmp2, &connCb->clientConnList, list, struct RsConnInfo);
        for (; (&connTmp->list) != &connCb->clientConnList;
             connTmp = connTmp2, connTmp2 = list_entry(connTmp2->list.next, struct RsConnInfo, list)) {
            if (connTmp->serverIp.binAddr.addr.s_addr == socketInfo[i].localIp.addr.s_addr) {
                stateTmp = connTmp->state;
                break;
            }
        }
        rs_ut_msg("ori state:%d\n", stateTmp);
        connTmp->state = RS_CONN_STATE_INIT;
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        connTmp->state = stateTmp;
        mocker_clean();

        /* wrong server ip address */
        socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.4");
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.3");

        rs_ut_msg("conn_tmp->state:%d\n", connTmp->state);
        connTmp->state = RS_CONN_STATE_CONNECTED;
        mocker((stub_fn_t)send, 10, SOCK_CONN_TAG_SIZE);
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        connTmp->state = stateTmp;
        mocker_clean();

        connTmp->state = RS_CONN_STATE_TIMEOUT;
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        connTmp->state = stateTmp;
        mocker_clean();

        connTmp->state = RS_CONN_STATE_TAG_SYNC;
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        connTmp->state = stateTmp;
        mocker_clean();
    }

    tryAgain = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        usleep(30000);
        tryAgain--;
    } while (ret != 1 && tryAgain);
    rs_ut_msg("%s [client]socket_info[0].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

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
    rs_ut_msg("RsQpCreate: qpn %d, ret:%d\n", resp.qpn, ret);

    ret = RsQpConnectAsync(devId, rdevIndex, resp.qpn, socketInfo[i].fd);
    rs_ut_msg("***RsQpConnectAsync: %d****\n", ret);

    usleep(SLEEP_TIME);

    i = 1;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].tag[0] = 1;
    socketInfo[i].tag[1] = 2;
    socketInfo[i].tag[2] = 3;
    socketInfo[i].tag[3] = 4;

    tryAgain = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_SERVER, &socketInfo[i], 1);
        usleep(30000);
        tryAgain--;
    } while (ret != 1 && tryAgain);
    rs_ut_msg("[server]socket_info[1].fd:%d, status:%d\n", socketInfo[i].fd, socketInfo[i].status);

    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp2);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("RsQpCreate: qpn2 %d, ret:%d\n", resp2.qpn, ret);

    ret = RsQpConnectAsync(devId, rdevIndex, resp2.qpn, socketInfo[i].fd);

    usleep(SLEEP_TIME);

    ret = RsQpDestroy(devId, rdevIndex, resp2.qpn);
    ret = RsQpDestroy(devId, rdevIndex, resp.qpn);

    sockClose[0].fd = socketInfo[0].fd;
    ret = RsSocketBatchClose(0, &sockClose[0], 1);

    sockClose[1].fd = socketInfo[1].fd;
    ret = RsSocketBatchClose(0, &sockClose[1], 1);

    /* ------Resource CLEAN-------- */
    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 1);

    struct RsConnInfo connSendInc = {0};
    mocker((stub_fn_t)send, 10, -1);
    RsSocketTagSync(&connSendInc);
    mocker_clean();

    mocker(RsDrvSocketSend, 10, -EAGAIN);
    RsSocketTagSync(&connSendInc);
    mocker_clean();

    ret = RsRdevDeinit(devId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);
    cfg.chipId = 0;
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("!!!!!!tc_rs_socket_ops2: RsDeinit\n");

    return;
}

void TcRsSocketClose2()
{
    int ret = 0;
    uint32_t devId = 0;
    unsigned int rdevIndex = 0;
    int flag = 0; /* RC */
    struct RsQpResp resp = {0};
    struct RsQpResp resp2 = {0};
    uint32_t qpMode = 1;
    int i = 0;
    struct RsInitConfig cfg = {0};
    struct SocketListenInfo listen[2] = {0};
    struct SocketConnectInfo conn[2] = {0};
    struct RsSocketCloseInfoT sockClose[2] = {0};
    struct SocketFdData socketInfo[3] = {0};
    struct rs_cb* rsCb = NULL;

    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDev2rscb(devId, &rsCb, false);
    EXPECT_INT_EQ(ret, 0);
    rsCb->connCb.wlistEnable = 0;

    usleep(SLEEP_TIME);

    listen[0].phyId = 0;
    listen[0].family = AF_INET;
    listen[0].localIp.addr.s_addr = inet_addr("127.0.0.3");
    listen[0].port = 16666;
    ret = RsSocketListenStart(&listen[0], 1);

    usleep(SLEEP_TIME);

    conn[0].phyId = 0;
    conn[0].family = AF_INET;
    conn[0].localIp.addr.s_addr = inet_addr("127.0.0.3");
    conn[0].remoteIp.addr.s_addr = inet_addr("127.0.0.3");
    conn[0].tag[0] = 1;
    conn[0].tag[1] = 2;
    conn[0].tag[2] = 3;
    conn[0].tag[3] = 4;
    conn[0].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 1);

    usleep(SLEEP_TIME);

    i = 0;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].remoteIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].tag[0] = 1;
    socketInfo[i].tag[1] = 2;
    socketInfo[i].tag[2] = 3;
    socketInfo[i].tag[3] = 4;

    tryAgain = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_CLIENT, &socketInfo[i], 1);
        usleep(30000);
        tryAgain--;
    } while (ret != 1 && tryAgain);
    rs_ut_msg("%s [client]socket_info[0].fd:%d, status:%d\n", __func__, socketInfo[i].fd, socketInfo[i].status);

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
    rs_ut_msg("RsQpCreate: qpn %d, ret:%d\n", resp.qpn, ret);

    ret = RsQpConnectAsync(devId, rdevIndex, resp.qpn, socketInfo[i].fd);
    rs_ut_msg("***RsQpConnectAsync: %d****\n", ret);

    usleep(SLEEP_TIME);

    i = 1;
    socketInfo[i].family = AF_INET;
    socketInfo[i].localIp.addr.s_addr = inet_addr("127.0.0.3");
    socketInfo[i].tag[0] = 1;
    socketInfo[i].tag[1] = 2;
    socketInfo[i].tag[2] = 3;
    socketInfo[i].tag[3] = 4;

    tryAgain = 10;
    do {
        ret = RsGetSockets(RS_CONN_ROLE_SERVER, &socketInfo[i], 1);
        usleep(30000);
        tryAgain--;
    } while (ret != 1 && tryAgain);
    rs_ut_msg("[server]socket_info[1].fd:%d, status:%d\n", socketInfo[i].fd, socketInfo[i].status);

    ret = RsQpCreate(devId, rdevIndex, qpNorm, &resp2);
    rs_ut_msg("RsQpCreate: qpn2 %d, ret:%d\n", resp2.qpn, ret);

    ret = RsQpConnectAsync(devId, rdevIndex, resp2.qpn, socketInfo[i].fd);

    usleep(1000);

    ret = RsQpDestroy(devId, rdevIndex, resp.qpn);

    sockClose[0].fd = socketInfo[0].fd;
    ret = RsSocketBatchClose(0, &sockClose[0], 1);
    usleep(SLEEP_TIME);

    ret = RsQpConnectAsync(devId, rdevIndex, resp2.qpn, socketInfo[1].fd);

    void* addr;
    addr = malloc(RS_TEST_MEM_SIZE);
    struct RdmaMrRegInfo mrRegInfo = {0};
    mrRegInfo.addr = addr;
    mrRegInfo.len = RS_TEST_MEM_SIZE;
    mrRegInfo.access = RS_ACCESS_LOCAL_WRITE;
    ret = RsMrReg(devId, rdevIndex, resp2.qpn, &mrRegInfo);
    EXPECT_INT_EQ(ret, 0);
    free(addr);

    sockClose[1].fd = socketInfo[1].fd;
    ret = RsSocketBatchClose(0, &sockClose[1], 1);

    ret = RsQpDestroy(devId, rdevIndex, resp2.qpn);

    /* ------Resource CLEAN-------- */
    listen[0].port = 16666;
    ret = RsSocketListenStop(&listen[0], 1);

    ret = RsRdevDeinit(devId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("!!!!!!tc_rs_socket_close2: RsDeinit\n");

    return;
}
void TcRsAbnormal2()
{
    int ret = 0;
    uint32_t devId = 0;
    uint32_t rdevIndex = 0;
    uint32_t errDevId = 10;
    struct RsInitConfig cfg = {0};
    struct RsQpResp resp = {0};
    uint32_t qpMode = 1;
    struct RsQpCb* qpCb = NULL;
    char buf[64] = {0};
    uint32_t* cmd = NULL;
    struct ibv_cq *ibSendCqT, *ibRecvCqT;
    unsigned int totalSize = 2;
    unsigned int curSize = 1;
    bool flag = true;
    char* bufTmp = NULL;

    /* resource prepare... */
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

    ret = RsQpn2qpcb(devId, rdevIndex, resp.qpn, &qpCb);
    EXPECT_INT_EQ(ret, 0);

    mocker((stub_fn_t)calloc, 10, NULL);
    ret = RsNotifyMrListAdd(qpCb, buf);
    mocker_clean();

    mocker((stub_fn_t)memcpy_s, 10, 1);
    ret = RsNotifyMrListAdd(qpCb, buf);
    mocker_clean();

    cmd = (uint32_t*)buf;
    *cmd = RS_CMD_QP_INFO;
    mocker((stub_fn_t)memcpy_s, 10, 1);
    RsEpollRecvHandle(qpCb, buf, 64);
    mocker_clean();

    mocker((stub_fn_t)RsQpStateModify, 10, 1);
    RsEpollRecvHandle(qpCb, buf, 64);
    mocker_clean();

    *cmd = RS_CMD_MR_INFO;
    mocker((stub_fn_t)calloc, 10, NULL);
    RsEpollRecvHandle(qpCb, buf, 64);
    mocker_clean();

    mocker((stub_fn_t)memcpy_s, 10, 1);
    RsEpollRecvHandle(qpCb, buf, 64);
    mocker_clean();

    /* unknown cmd */
    *cmd = RS_CMD_QP_INFO + 444;
    RsEpollRecvHandle(qpCb, buf, 64);

    mocker((stub_fn_t)memcpy_s, 10, 1);
    RsEpollRecvHandleRemain(qpCb, totalSize, curSize, flag, bufTmp);
    mocker_clean();
    mocker((stub_fn_t)ibv_create_cq, 10, NULL);
    ibSendCqT = qpCb->ibSendCq;
    ibRecvCqT = qpCb->ibRecvCq;
    ret = RsDrvCreateCq(qpCb, 0);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker_ret((stub_fn_t)ibv_create_cq, 1, 0, 0);
    mocker((stub_fn_t)ibv_destroy_cq, 10, 0);
    ret = RsDrvCreateCq(qpCb, 0);
    qpCb->ibSendCq = ibSendCqT;
    qpCb->ibRecvCq = ibRecvCqT;
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker_ret((stub_fn_t)ibv_create_cq, 1, 0, 0);
    mocker((stub_fn_t)ibv_destroy_cq, 10, 0);
    ret = RsDrvCreateCq(qpCb, 0);
    qpCb->ibSendCq = ibSendCqT;
    qpCb->ibRecvCq = ibRecvCqT;
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker_ret((stub_fn_t)RsIbvExpCreateCq, 1, 0, 0);
    mocker((stub_fn_t)ibv_destroy_cq, 10, 0);
    ret = RsDrvCreateCq(qpCb, 1);
    qpCb->ibSendCq = ibSendCqT;
    qpCb->ibRecvCq = ibRecvCqT;
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)RsIbvQueryQp, 10, 1);
    RsQpStateModify(qpCb);
    mocker_clean();

    mocker((stub_fn_t)ibv_modify_qp, 10, 1);
    RsQpStateModify(qpCb);
    mocker_clean();

    mocker_ret((stub_fn_t)ibv_modify_qp, 0, 1, 0);
    RsQpStateModify(qpCb);
    mocker_clean();

    mocker((stub_fn_t)pthread_mutex_lock, 10, 1);
    mocker((stub_fn_t)pthread_mutex_unlock, 10, 1);
    (void)pthread_mutex_lock(NULL);
    (void)pthread_mutex_unlock(NULL);
    mocker_clean();

    RsDrvPollCqHandle(qpCb);

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

    qpCb->ibSendCq = ibSendCqT;

    /* resource free... */
    ret = RsQpDestroy(devId, rdevIndex, resp.qpn);
    EXPECT_INT_EQ(ret, 0);

    ret = RsRdevDeinit(devId, NOTIFY, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("!!!!!!tc_rs_abnormal2: RsDeinit\n");
    return;
}

struct RsConnInfo gConn = {0};

int stub_RsFd2conn(int fd, struct RsConnInfo** conn)
{
    *conn = &gConn;
    return 0;
}

void TcRsSocketNodeid2vnic()
{
    int ret;
    ret = RsSocketNodeid2vnic(0, NULL);
    EXPECT_INT_EQ(-EINVAL, ret);
}

void TcRsServerValidAsyncInit()
{
    int ret = 0;
    struct RsConnInfo conn = {0};
    struct SocketWlistInfoT whiteListExpect = {0};
    conn.state = 7;
    strcpy(conn.tag, "1234");
    conn.clientIp.family = AF_INET;
    conn.clientIp.binAddr.addr.s_addr = 16;
    ret = RsServerValidAsyncInit(0, &conn, &whiteListExpect);
    EXPECT_INT_EQ(0, ret);
}

void TcRsConnectHandle()
{
    int ret = 0;
    struct RsInitConfig cfg = {0};
    struct SocketConnectInfo conn[2] = {0};

    /* resource prepare... */
    cfg.hccpMode = NETWORK_OFFLINE;
    cfg.chipId = 0;
    cfg.whiteListStatus = 1;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsConnectBindClient, 100, -99);
    conn[0].phyId = 0;
    conn[0].family = AF_INET6;
    inet_pton(AF_INET6, "::1", &conn[0].localIp.addr6);
    inet_pton(AF_INET6, "::1", &conn[0].remoteIp.addr6);
    strcpy(conn[0].tag, "5678");
    conn[0].port = 16666;
    ret = RsSocketBatchConnect(&conn[0], 1);
    usleep(SLEEP_TIME);
    mocker_clean();

    /* resource free... */
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    ret = RsConnectHandle(NULL);
    EXPECT_INT_EQ(ret, 0);
    return;
}

int ReplaceRsQpn2qpcb(unsigned int phyId, unsigned int rdevIndex, uint32_t qpn, struct RsQpCb** qpCb)
{
    static struct RsQpCb aQpCb = {0};
    *qpCb = &aQpCb;
    return 0;
}

void TcRsGetQpContext()
{
    void *qp, *sendCq, *recvCq;
    RsGetQpContext(RS_MAX_DEV_NUM, 0, 0, &qp, &sendCq, &recvCq);

    mocker(RsQpn2qpcb, 1, -1);
    RsGetQpContext(0, 0, 0, &qp, &sendCq, &recvCq);
    mocker_clean();

    mocker_invoke(RsQpn2qpcb, ReplaceRsQpn2qpcb, 1);
    RsGetQpContext(0, 0, 0, &qp, &sendCq, &recvCq);
    mocker_clean();
}

void TcTlsAbnormal1()
{
    int ret = 0;
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

    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    gRsCb->sslEnable = 1;
    mocker_invoke((stub_fn_t)RsFd2conn, stub_RsFd2conn, 10);
    mocker((stub_fn_t)SSL_write, 10, -1);
    mocker((stub_fn_t)SSL_get_error, 10, 2);
    RsDrvSocketSend(socketInfo[0].fd, "1", 1, 0);
    mocker_clean();

    mocker_invoke((stub_fn_t)RsFd2conn, stub_RsFd2conn, 10);
    mocker((stub_fn_t)SSL_read, 10, -1);
    mocker((stub_fn_t)SSL_get_error, 10, 2);
    RsDrvSocketRecv(socketInfo[1].fd, "1", 1, 0);
    mocker_clean();

    mocker((stub_fn_t)RsFd2conn, 10, -1);
    RsDrvSocketRecv(socketInfo[1].fd, "1", 1, 0);
    mocker_clean();
    gRsCb->sslEnable = 0;
    RsDrvSocketRecv(socketInfo[1].fd, "1", 1, 0);

    mocker((stub_fn_t)fcntl, 10, -1);
    ret = RsSetFdNonblock(-1);
    EXPECT_INT_EQ(ret, -EFILEOPER);
    mocker_clean();

    mocker_ret((stub_fn_t)fcntl, 1, -1, 0);
    ret = RsSetFdNonblock(-1);
    EXPECT_INT_EQ(ret, -EFILEOPER);

    struct RsConnInfo connSsl = {0};
    connSsl.ssl = NULL;
    connSsl.clientIp.family = AF_INET;
    mocker((stub_fn_t)SSL_do_handshake, 10, -1);
    ret = RsSocketSslConnect(&connSsl, gRsCb);
    EXPECT_INT_EQ(ret, -EAGAIN);
    mocker_clean();

    mocker((stub_fn_t)SSL_get_verify_result, 10, -1);
    ret = RsSocketSslConnect(&connSsl, gRsCb);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    mocker((stub_fn_t)fcntl, 10, -1);
    ret = RsSocketStateReset(0, &connSsl, 1, gRsCb);
    EXPECT_INT_EQ(ret, -ESYSFUNC);
    mocker_clean();

    mocker((stub_fn_t)SSL_set_fd, 10, -1);
    ret = RsSocketStateConnected(&connSsl, 1, gRsCb);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    mocker((stub_fn_t)SSL_do_handshake, 10, -1);
    ret = RsSocketStateSslFdBind(&connSsl, 1, gRsCb);
    EXPECT_INT_EQ(ret, -EAGAIN);
    mocker_clean();

    connSsl.state = RS_CONN_STATE_SSL_BIND_FD;
    ret = RsSocketConnectAsync(&connSsl, gRsCb);
    EXPECT_INT_EQ(ret, 0);

    struct RsAcceptInfo sslInfo = {0};
    mocker((stub_fn_t)SSL_do_handshake, 10, -1);
    RsDoSslHandshake(&sslInfo, gRsCb);
    mocker_clean();

    mocker((stub_fn_t)SSL_do_handshake, 10, -1);
    mocker((stub_fn_t)SSL_get_error, 10, SSL_ERROR_WANT_WRITE);
    RsDoSslHandshake(&sslInfo, gRsCb);
    mocker_clean();

    mocker((stub_fn_t)SSL_do_handshake, 10, -1);
    mocker((stub_fn_t)SSL_get_error, 10, SSL_ERROR_WANT_READ);
    RsDoSslHandshake(&sslInfo, gRsCb);
    mocker_clean();

    mocker((stub_fn_t)calloc, 10, 0);
    ret = rs_get_pk(NULL, NULL, NULL);
    EXPECT_INT_EQ(ret, -ENOMEM);

    mocker((stub_fn_t)SSL_new, 10, 0);
    ret = RsDrvSslBindFd(&connSsl, 1);
    EXPECT_INT_EQ(ret, -ENOMEM);
    mocker_clean();

    ret = RsDrvSocketSend(-1, "abc", 3, 0);
    EXPECT_INT_EQ(ret, -EINVAL);

    ret = RsDrvSocketRecv(-1, "abc", 3, 0);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker((stub_fn_t)rs_ssl_get_crl_data, 10, 0);
    mocker((stub_fn_t)SSL_CTX_get_cert_store, 10, 0);
    ret = rs_ssl_crl_init(NULL, NULL, NULL);
    EXPECT_INT_EQ(ret, -EFAULT);
    mocker_clean();

    struct tls_cert_mng_info mngInfo = {0};
    struct rs_cb errRscb = {0};
    struct RsCerts certs = {0};
    struct tls_ca_new_certs newCerts[RS_SSL_NEW_CERT_CB_NUM] = {{0}};
    errRscb.chipId = 0;
    mngInfo.cert_count = 0;
    mngInfo.total_cert_len = 100;
    ret = rs_ssl_put_certs(&errRscb, &mngInfo, &certs, newCerts, NULL);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    mocker((stub_fn_t)X509_STORE_new, 10, 0);
    ret = rs_ssl_check_cert_chain(&mngInfo, &certs);
    EXPECT_INT_EQ(ret, -ENOMEM);
    mocker_clean();

    mocker((stub_fn_t)X509_STORE_CTX_new, 10, 0);
    ret = rs_ssl_check_cert_chain(&mngInfo, &certs);
    EXPECT_INT_EQ(ret, -ENOMEM);
    mocker_clean();

    int rs_ssl_verify_cert_chain(
        X509_STORE_CTX * ctx, X509_STORE * store, struct RsCerts * certs, STACK_OF(X509) * certChain,
        struct tls_cert_mng_info * mngInfo);
    mocker((stub_fn_t)rs_ssl_verify_cert_chain, 10, -1);
    ret = rs_ssl_check_cert_chain(&mngInfo, &certs);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker((stub_fn_t)rs_ssl_verify_cert_chain, 10, -1);
    ret = rs_ssl_check_cert_chain(&mngInfo, &certs);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    X509* tls_load_cert(const uint8_t* inbuf, uint32_t bufLen, int type);
    mocker((stub_fn_t)tls_load_cert, 10, 0);
    ret = rs_ssl_check_cert_chain(&mngInfo, &certs);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    mocker((stub_fn_t)X509_STORE_CTX_init, 10, 0);
    ret = rs_ssl_check_cert_chain(&mngInfo, &certs);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    mocker((stub_fn_t)X509_verify_cert, 10, 0);
    ret = rs_ssl_check_cert_chain(&mngInfo, &certs);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    mocker((stub_fn_t)BIO_new_mem_buf, 10, 0);
    tls_load_cert(NULL, 0, 0);
    mocker_clean();

    mocker((stub_fn_t)d2i_X509_bio, 10, 0);
    X509* cert = tls_load_cert(NULL, 0, SSL_FILETYPE_ASN1);
    free(cert);
    mocker_clean();

    mocker((stub_fn_t)d2i_X509_bio, 10, 0);
    tls_load_cert(NULL, 0, 100);
    mocker_clean();

    mocker((stub_fn_t)PEM_read_bio_X509, 10, 0);
    tls_load_cert(NULL, 0, SSL_FILETYPE_PEM);
    mocker_clean();

    ret = rs_remove_certs("123.txt", "456.txt");
    EXPECT_INT_EQ(ret, 0);

    mocker((stub_fn_t)calloc, 10, 0);
    errRscb.skidSubjectCb = NULL;
    ret = rs_ssl_skid_get_from_chain(&errRscb, NULL, NULL, NULL);
    EXPECT_INT_EQ(ret, -ENOMEM);
    mocker_clean();

    mngInfo.cert_count = 2;
    mocker((stub_fn_t)BIO_new_mem_buf, 10, 0);
    ret = rs_ssl_skid_get_from_chain(&errRscb, &mngInfo, &certs, newCerts);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    mocker((stub_fn_t)tls_get_user_config, 10, -1);
    ret = rs_ssl_init(&errRscb);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mngInfo.ky_len = RS_SSL_PRI_LEN + 1;
    mocker((stub_fn_t)rs_get_pridata, 10, 0);
    ret = rs_get_pk(&errRscb, &mngInfo, NULL);
    EXPECT_INT_EQ(ret, -EINVAL);
    mngInfo.ky_len = 0;
    mocker_clean();

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_ut_msg("!!!!!!tc_tls_abnormal1: RsDeinit\n");
}

int StubDlHalGetChipInfo91093(unsigned int devId, halChipInfo* chipInfo)
{
    strcpy(chipInfo->name, "910_93xx");

    return 0;
}

void TcRsSocketGetBindByChip()
{
    unsigned int chipId = 0;
    bool bindIp = false;

    mocker((stub_fn_t)DlDrvDeviceGetIndexByPhyId, 1, -1);
    RsSocketGetBindByChip(chipId, &bindIp);
    mocker_clean();

    mocker((stub_fn_t)DlDrvDeviceGetIndexByPhyId, 1, 0);
    mocker((stub_fn_t)DlHalGetDeviceInfo, 1, -2);
    RsSocketGetBindByChip(chipId, &bindIp);
    mocker_clean();

    mocker((stub_fn_t)DlDrvDeviceGetIndexByPhyId, 1, 0);
    mocker((stub_fn_t)DlHalGetDeviceInfo, 1, 0);
    mocker((stub_fn_t)DlHalGetChipInfo, 1, -2);
    RsSocketGetBindByChip(chipId, &bindIp);
    mocker_clean();

    mocker(DlDrvDeviceGetIndexByPhyId, 1, 0);
    mocker(DlHalGetDeviceInfo, 1, 0);
    mocker_invoke(DlHalGetChipInfo, StubDlHalGetChipInfo91093, 100);
    RsSocketGetBindByChip(chipId, &bindIp);
    EXPECT_INT_EQ(bindIp, true);
    mocker_clean();
}

int StubRsGetConnInfo(
    struct RsConnCb* connCb, struct SocketConnectInfo* conn, struct RsConnInfo** connInfo, int serverPort)
{
    (*connInfo) = gConnInfo;

    return 0;
}

void TcRsSocketBatchAbort()
{
    struct SocketConnectInfo conn[1] = {0};
    gRsCb = malloc(sizeof(struct rs_cb));
    gConnInfo = malloc(sizeof(struct RsConnInfo));
    int ret = 0;

    memset(gRsCb, 0, sizeof(struct rs_cb));
    memset(gConnInfo, 0, sizeof(struct RsConnInfo));

    mocker_clean();
    mocker(pthread_mutex_lock, 10, 0);
    mocker(pthread_mutex_unlock, 10, 0);
    mocker(RsGetConnInfo, 1, -1);
    ret = RsSocketBatchAbort(conn, 1);
    EXPECT_INT_EQ(ret, -1);

    mocker_clean();
    mocker(pthread_mutex_lock, 10, 0);
    mocker(pthread_mutex_unlock, 10, 0);
    mocker_invoke(RsGetConnInfo, StubRsGetConnInfo, 1);
    mocker(setsockopt, 1, -1);
    mocker(RsSocketCloseFd, 1, -1);

    gConnInfo->state = 2;
    gConnInfo->list.prev = &gConnInfo->list;
    gConnInfo->list.next = &gConnInfo->list;
    ret = RsSocketBatchAbort(conn, 1);
    EXPECT_INT_EQ(ret, -1);

    free(gRsCb);
    gRsCb = NULL;
}

int* stub__errno_location()
{
    static int errNo = 0;

    errNo = EAGAIN;
    return &errNo;
}

void TcRsSocketSendAndRecvLogTest()
{
    int ret = 0;

    gRsCb = malloc(sizeof(struct rs_cb));
    gRsCb->sslEnable = 0;

    mocker_clean();
    mocker(send, 1, -1);
    mocker_invoke(__errno_location, stub__errno_location, 1);
    ret = RsDrvSocketSend(1, "1", 1, 0);
    EXPECT_INT_EQ(ret, -EAGAIN);

    mocker_clean();
    mocker(send, 1, -1);
    ret = RsDrvSocketSend(1, "1", 1, 0);
    EXPECT_INT_EQ(ret, -EFILEOPER);

    mocker_clean();
    mocker(recv, 1, -1);
    mocker_invoke(__errno_location, stub__errno_location, 1);
    ret = RsDrvSocketRecv(1, "1", 1, 0);
    EXPECT_INT_EQ(ret, -EAGAIN);

    mocker_clean();
    mocker(recv, 1, -1);
    ret = RsDrvSocketRecv(1, "1", 1, 0);
    EXPECT_INT_EQ(ret, -EFILEOPER);

    mocker_clean();
    mocker(send, 1, -1);
    mocker_invoke(__errno_location, stub__errno_location, 1);
    ret = RsPeerSocketSend(0, 1, "1", 1);
    EXPECT_INT_EQ(ret, -EAGAIN);

    mocker_clean();
    mocker(send, 1, -1);
    ret = RsPeerSocketSend(0, 1, "1", 1);
    EXPECT_INT_EQ(ret, -EFILEOPER);

    mocker_clean();
    mocker(recv, 1, -1);
    mocker_invoke(__errno_location, stub__errno_location, 1);
    ret = RsPeerSocketRecv(0, 1, "1", 1);
    EXPECT_INT_EQ(ret, -EAGAIN);

    mocker_clean();
    mocker(recv, 1, -1);
    ret = RsPeerSocketRecv(0, 1, "1", 1);
    EXPECT_INT_EQ(ret, -EFILEOPER);

    free(gRsCb);
    gRsCb = NULL;
}

void StubHccpTimeMaxInterval(struct timeval* endTime, struct timeval* startTime, float* msec) { *msec = 90001.0; }

void stub_HccpTimeInterval(struct timeval* endTime, struct timeval* startTime, float* msec) { *msec = 5001.0; }

void TcRsTcpRecvTagInHandle()
{
    struct RsListenInfo listenInfo = {0};
    struct RsConnInfo connTmp = {0};
    struct RsIpAddrInfo remoteIp = {0};
    struct rs_cb* rsCb = NULL;
    struct RsAcceptInfo acceptInfo = {0};
    int ret = 0;

    mocker_clean();
    mocker(recv, 1, 0);
    ret = RsTcpRecvTagInHandle(&listenInfo, 0, &connTmp, &remoteIp);
    EXPECT_INT_EQ(ret, -ESOCKCLOSED);

    mocker_clean();
    mocker(recv, 1, 1);
    mocker_invoke(HccpTimeInterval, StubHccpTimeMaxInterval, 1);
    ret = RsTcpRecvTagInHandle(&listenInfo, 0, &connTmp, &remoteIp);
    EXPECT_INT_EQ(ret, -ETIME);

    mocker_clean();
    mocker(recv, 1, 256);
    mocker_invoke(HccpTimeInterval, stub_HccpTimeInterval, 1);
    ret = RsTcpRecvTagInHandle(&listenInfo, 0, &connTmp, &remoteIp);
    EXPECT_INT_EQ(ret, 0);

    mocker_clean();
    mocker(RsTcpRecvTagInHandle, 1, 1);
    mocker(close, 1, 1);
    mocker(pthread_mutex_lock, 1, 0);
    mocker(pthread_mutex_unlock, 1, 0);
    RsEpollEventTcpListenInHandle(rsCb, &listenInfo, 1, &remoteIp);

    mocker_clean();
    mocker(RsTcpRecvTagInHandle, 1, 0);
    mocker(RsWlistCheckConnAdd, 1, 1);
    mocker(pthread_mutex_lock, 1, 0);
    mocker(pthread_mutex_unlock, 1, 0);
    RsEpollEventTcpListenInHandle(rsCb, &listenInfo, 1, &remoteIp);
    mocker_clean();

    mocker((stub_fn_t)SSL_read, 10, -1);
    mocker_invoke(HccpTimeInterval, stub_HccpTimeInterval, 1);
    RsSslRecvTagInHandle(&acceptInfo, &connTmp);
    mocker_clean();
}

void TcRsServerValidAsyncAbnormal()
{
    struct RsConnInfo conn = {0};
    struct RsConnCb connCb = {0};

    mocker_clean();
    mocker(RsFindWhiteList, 1, 0);
    mocker(RsServerValidAsyncInit, 1, 0);
    mocker(pthread_mutex_lock, 1, 0);
    mocker(pthread_mutex_unlock, 1, 0);
    mocker(RsFindWhiteListNode, 1, -1);
    mocker(RsServerSendWlistCheckResult, 1, 0);
    RsServerValidAsync(0, &connCb, &conn);
    mocker_clean();
}

void TcRsServerValidAsyncAbnormal01()
{
    struct RsConnInfo conn = {0};
    struct RsConnCb connCb = {0};

    mocker_clean();
    mocker(RsServerValidAsyncInit, 1, 0);
    mocker(RsFindWhiteList, 1, 0);
    mocker(pthread_mutex_lock, 1, 0);
    mocker(pthread_mutex_unlock, 1, 0);
    mocker_invoke((stub_fn_t)RsFindWhiteListNode, StubRsFindWhiteListNode, 1);
    mocker(RsServerSendWlistCheckResult, 1, -1);
    RsServerValidAsync(0, &connCb, &conn);
    mocker_clean();
}

void TcRsNetApiInitFail()
{
    int ret = 0;

    mocker(RsNetAdaptApiInit, 1, -1);
    ret = RsNetApiInit();
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();
}
