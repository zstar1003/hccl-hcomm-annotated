/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RS_INNER_H
#define RS_INNER_H

#include <pthread.h>
#include <stdint.h>
#include <sys/time.h>
#include <infiniband/verbs.h>
#include <semaphore.h>
#ifndef CA_CONFIG_LLT
#include <openssl/crypto.h>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/rsa.h>
#include <crypto/x509.h>
#include <crypto/evp.h>
#include <crypto/asn1.h>
#include <openssl/x509v3.h>
#else
#include "stub_ssl.h"
#endif
#include "ascend_hal_external.h"
#ifndef HNS_ROCE_LLT
#include "dlog_pub.h"
#endif
#include "hccp_common.h"
#include "hccp_ping.h"
#include "hccp_nda.h"
#include "rs_rdma_inner.h"
#include "rs_common_inner.h"
#include "rs_ping_inner.h"
#include "rs.h"
#include "rs_list.h"

/* priority of algos and forbid unsafe algos */
#define CHIPER_LIST                                                                                                    \
    "ECDHE-RSA-AES256-GCM-SHA384:\
    !RC2:!RC4:!MD2:!MD4:!MD5:!DES:!3DES:!SHA1:!BLOWFISH:!CBC:!ECB:!ADH:!LOW:!PSK:!SRP:!DSS:!eNULL:!aNULL:!EXP:@STRENGTH"

#define RS_S6_ADDR32 2
#define RS_USLEEP_TIME 20000
#define RS_RECV_MAX_TIME (1000.0 * 5)      // ms
#define RS_RECV_TAG_MAX_TIME (1000.0 * 90) // ms
#define RS_DEVICE_NUM 0x3
#define RS_HOSTID2DEVID(dev_id) ((dev_id) & RS_DEVICE_NUM)
#define SOCK_CONN_DEV_ID_SIZE 64
#define RS_VNIC_MAX 128
#define RS_VNIC_IP_LEN 14
#define RS_IB_NAME_LEN 10

#define RS_VNIC_FIRST 192
#define RS_VNIC_SECOND 168
#ifndef CA_CONFIG_LLT
#define RS_VNIC_THIRD 2
#else
#define RS_VNIC_THIRD 1
#endif
#define RS_VNIC_FOUTH 199
#define RS_VNIC_FLAG 1

#define RS_TCP_DSCP_0 0
#define RS_ROCE_DSCP_33 33
#define RS_DSCP_MASK 0x3f
#define RS_DSCP_OFF 2

#define RS_MAX_FD_NUM 65536
#define TLS_CA_SSL_MAX_NEW_CERT_NUM 8

#define RS_CONN_EXIT_FLAG 2
#define RS_TRY_TIME 200
#define RS_WLIST_VALID_FLAG_SIZE 6
#define RS_SSL_CERT_LEN 2048
#define RS_SSL_MIN_CERT_NUM 2
#define RS_SSL_MAX_CERT_NUM 15
#define RS_SSL_MAX_ALL_CERT_NUM (RS_SSL_MAX_CERT_NUM + (TLS_CA_SSL_MAX_NEW_CERT_NUM * RS_SSL_NEW_CERT_CB_NUM))
#define RS_SSL_MIN_CERT_LEN (RS_SSL_CERT_LEN * RS_SSL_MIN_CERT_NUM)
#define RS_SSL_MAX_CERT_LEN (RS_SSL_CERT_LEN * RS_SSL_MAX_CERT_NUM)
#define RS_SSL_PRI_LEN 5120
#define RS_SSL_REVOKE_LEN 20480
#define RS_SSL_FALSH_HEAD_LEN 8
#define RS_SSL_ENC_MODE 1
#define RS_SSL_VERSION 2
#define HCCP_CERTS_STATR "-----BEGIN CERTIFICATE-----"
#define HCCP_CERTS_END "-----END CERTIFICATE-----"
#define RS_KID_MAX_LENGTH 512
#define RS_KID_MIN_LENGTH 8
#define RS_RSA_KY_BITS_MIN_LEN 2048
#define RS_DSA_KY_BITS_MIN_LEN 2048
#define RS_DH_KY_BITS_MIN_LEN 2048
#define RS_EC_KY_BITS_MIN_LEN 256
#define RS_SSL_ERR_MSG_LEN 256
#define RS_SOCKET_MAXLEN 2048
#define RS_INTERFACE_BOND_LEN 6
#define RS_INTERFACE_ETH_PREFIX_LEN 3
#define RS_INTERFACE_BOND_PREFIX_LEN 4

/* pcie card boardid rule: GPIO[75:73]=0x000 */
#define RS_BOARDID_PCIE_CARD_MASK 0xE00
#define RS_BOARDID_PCIE_CARD_MASK_VALUE 0x0
#define RS_BOARDID_AI_SERVER_MODULE 0x0
#define RS_BOARDID_ARM_SERVER_AG 0x20
#define RS_BOARDID_ARM_POD 0x30
#define RS_BOARDID_X86_16P 0x50
#define RS_BOARDID_ARM_SERVER_2DIE 0xB0

#define RS_MAX_RD_ATOMIC_NUM_PEER_ONLINE 16 // host RDMA adapt
#define RS_QP_TX_DEPTH_PEER_ONLINE 4096     // host RDMA adapt

#define RS_CLOSE_TIMEOUT 5

enum CaPtye { RS_EQPT_CA = 0, RS_ROOT_CA };

#define RS_SSL_DISABLE 0
#define RS_SSL_ENABLE 1

#define RS_EQPT_CERTS_PATH_LEN 256
#define RS_CA_CERTS_PATH_LEN 256

struct RsCertInfo {
    char certInfo[RS_SSL_CERT_LEN];
};

struct RsCerts {
    struct RsCertInfo certs[RS_SSL_MAX_CERT_NUM];
};

#define HCCP_NEW_CERTS_CB1_INDEX 0
#define HCCP_NEW_CERTS_CB2_INDEX 1
#define HCCP_NEW_CERTS_CB3_INDEX 2
#define HCCP_NEW_CERTS_CB4_INDEX 3
#define MAX_CERT_NUM_IN_CB 8
#define RS_SSL_NEW_CERT_CB_NUM 4

struct CertFile {
    const char *endFile;
    const char *caFile;
};

#define TLS_SALT_MAX_LEN 48
#define TLS_ENC_DEC_DIV_LEN 16
#define TLS_MAGIC_WORDS "1234567"
#define X509_VERIFY_SUCC 1

struct RsSecPara {
    unsigned char inBuf[RS_SSL_PRI_LEN];
    unsigned int inBufSize;
    unsigned char inSalt[TLS_SALT_MAX_LEN];
    unsigned int inSaltSize;
    unsigned char outBuf[RS_SSL_PRI_LEN];
    unsigned int outBufSize;
};

#define RS_CLOSE_RETRY_FOR_EINTR(ret, fd)                                                                              \
    do {                                                                                                               \
        do {                                                                                                           \
            (ret) = close((fd));                                                                                       \
        } while (((ret) < 0) && (errno == EINTR));                                                                     \
    } while (0)

#define RS_CHECK_RET_WITHOUT_RETURN(ret, fmt, val...)                                                                  \
    do {                                                                                                               \
        if (ret) {                                                                                                     \
            hccp_warn(fmt, ##val);                                                                                     \
        }                                                                                                              \
    } while (0)

#define RS_CHECK_POINTER_NULL_WITH_RET(ptr)                                                                            \
    do {                                                                                                               \
        if ((ptr) == NULL) {                                                                                           \
            hccp_err("pointer is NULL!");                                                                              \
            return (-EINVAL);                                                                                          \
        }                                                                                                              \
    } while (0)

#define RS_CHECK_POINTER_NULL_RETURN_VOID(ptr)                                                                         \
    do {                                                                                                               \
        if ((ptr) == NULL) {                                                                                           \
            hccp_err("pointer is NULL!");                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define RS_CHECK_POINTER_NULL_RETURN_NULL(ptr)                                                                         \
    do {                                                                                                               \
        if ((ptr) == NULL) {                                                                                           \
            hccp_err("null pointer exception!");                                                                       \
            return NULL;                                                                                               \
        }                                                                                                              \
    } while (0)

#define RS_CHECK_POINTER_NULL_RETURN_INT(ptr)                                                                          \
    do {                                                                                                               \
        if ((ptr) == NULL) {                                                                                           \
            hccp_err("null pointer exception!");                                                                       \
            return (-EINVAL);                                                                                          \
        }                                                                                                              \
    } while (0)

#define RS_PTHREAD_MUTEX_LOCK(conn_mutex)                                                                              \
    do {                                                                                                               \
        int ret_lock = pthread_mutex_lock(conn_mutex);                                                                 \
        if (ret_lock) {                                                                                                \
            hccp_warn("pthread_mutex_lock unsuccessful, ret[%d]", ret_lock);                                           \
        }                                                                                                              \
    } while (0)

#define RS_PTHREAD_MUTEX_ULOCK(conn_mutex)                                                                             \
    do {                                                                                                               \
        int ret_ulock = pthread_mutex_unlock(conn_mutex);                                                              \
        if (ret_ulock) {                                                                                               \
            hccp_warn("pthread_mutex_unlock unsuccessful, ret[%d]", ret_ulock);                                        \
        }                                                                                                              \
    } while (0)

#define RS_FD_INVALID (-1)

/*
 * mr_cb also used to sync to remote
 */
struct RsMrCb {
    struct RsMrInfo mrInfo; /* MUST be the first element */

    uint64_t wrId;
    uint32_t state;

    struct ibv_mr *ibMr;
    struct RsRdevCb *devCb;
    struct RsQpCb *qpCb;
    struct RsListHead list;
};

struct RsQpInfo {
    uint32_t cmd; /* MUST be the first element */

    int lid;
    int qpn;
    int psn;
    int gidIdx;
    union ibv_gid gid;
    struct RsMrInfo notifyMr;
};

enum RsConnState {
    RS_CONN_STATE_RESET,
    RS_CONN_STATE_INIT,
    RS_CONN_STATE_BIND,
    RS_CONN_STATE_LISTENING,
    RS_CONN_STATE_CONNECTED,
    RS_CONN_STATE_SSL_BIND_FD,
    RS_CONN_STATE_SSL_CONNECTED,
    RS_CONN_STATE_TAG_SYNC,
    RS_CONN_STATE_VALID_SYNC,
    RS_CONN_STATE_TX_TO_HCCL,

    RS_CONN_STATE_TIMEOUT,

    RS_CONN_STATE_ERR,

    RS_CONN_STATE_MAX,
};

struct RsConnInfo {
    struct RsIpAddrInfo serverIp;
    struct RsIpAddrInfo clientIp;
    uint16_t port;
    int scopeId;

    int connfd;
    SSL *ssl;
    uint32_t state; /* refer to enum rs_conn_state */
    struct timeval startTime;
    struct timeval endTime;
    bool isGot;

    /*
     * HCCL need classify the connection according by the tag.
     * when a client connects successfully, it need send the tag to Server,
     * Server return the tag to HCCL
     */
    char tag[SOCK_CONN_TAG_SIZE + SOCK_CONN_DEV_ID_SIZE];
    uint32_t tagSyncTimes;
    uint32_t tagEintrTimes;

    struct SocketErrInfo errInfo;

    struct RsListHead list;
};

enum ListenFdState {
    LISTEN_FD_STATE_ADDED = 0,
    LISTEN_FD_STATE_DELETED = 1,
};

struct RsListenInfo {
    struct RsIpAddrInfo serverIpAddr;
    struct RsIpAddrInfo clientIpAddr;
    uint16_t sockPort;

    int listenFd;
    uint32_t state; /* refer to enum rs_conn_state */
    int counter;

    int lastAcceptErrno; /* last accept errno, avoid log flush */
    struct SocketErrInfo errInfo;

    bool acceptCreditFlag;
    pthread_mutex_t acceptCreditMutex;
    enum ListenFdState fdState;
    unsigned int acceptCreditLimit;

    struct RsListHead list;
};

struct RsAcceptInfo {
    struct RsIpAddrInfo serverIpAddr;
    struct RsIpAddrInfo clientIpAddr;
    uint16_t sockPort;
    int connFd;
    SSL *ssl;
    uint32_t state;

    struct RsListHead list;
};

struct RsWhiteList {
    struct RsIpAddrInfo serverIp;
    struct RsListHead whiteList;
    struct RsListHead list;
};

struct RsWhiteListInfo {
    struct RsIpAddrInfo clientIp;
    unsigned int connLimit;
    char tag[SOCK_CONN_TAG_SIZE];
    struct RsListHead list;
};

struct RsHeterogTcpFdInfo {
    int fd;
    struct RsListHead list;
};

struct RsCqeErrInfo {
    pthread_mutex_t mutex;
    struct CqeErrInfo info;
};

struct RsConnCb {
    struct RsIpAddrInfo localIpAddr;
    unsigned int wlistEnable;
    int eventfd;
    int epollfd;
    int scopeId;

    pthread_mutex_t connMutex;
    struct rs_cb *rscb;
    struct SocketErrInfo epollErrInfo;

    struct RsListHead listenList;
    struct RsListHead serverAcceptList;
    struct RsListHead serverConnList;
    struct RsListHead clientConnList;
    struct RsListHead whiteList;
};

struct RsQpCb {
    struct RsRdevCb *rdevCb;
    struct ibv_pd *ibPd;
    struct ibv_qp *ibQp;
    struct ibv_qp_extend *ibQpEx;

    int eqNum;
    struct ibv_comp_channel *channel;
    struct ibv_cq *ibSendCq;
    int sendCqDepth;
    struct ibv_cq *ibRecvCq;
    int recvCqDepth;
    struct RsCqContext *srqContext;
    int numRecvCqEvents;
    int numSendCqEvents;

    unsigned int txDepth;
    unsigned int sendSgeNum;
    unsigned int rxDepth;
    unsigned int recvSgeNum;

    unsigned int sendWrNum;
    unsigned int recvWrNum;

    int sqIndex;
    int dbIndex;
    int qpMode;
    struct RsQpInfo qpInfoLo;
    struct RsQpInfo qpInfoRem;

    struct RsConnInfo *connInfo;
    int state;
    struct timeval startTime;
    struct timeval endTime;
    char qpMrBuf[RS_BUF_SIZE];
    unsigned int remainSize;

    pthread_mutex_t qpMutex;

    int mrNum;
    struct RsListHead list;
    struct RsListHead mrList;
    struct RsListHead remMrList;
    int isExp;

    uint32_t sendLen;
    uint32_t recvLen;
    uint32_t expectLen;

    struct event_summary *sendEvent;
    struct event_summary *recvEvent;

    struct QosAttr qosAttr;

    unsigned int timeout;
    unsigned int retryCnt;

    struct LiteQpCqAttrResp qpResp;

    struct LiteMemAttrResp memResp;
    int memAlign; // 0,1:4KB, 2:2MB
    uint32_t udpSport;

    unsigned int aiOpSupport;
    unsigned int grpId;
    unsigned int cqCstmFlag;

    struct RsCqeErrInfo cqeErrInfo;
    unsigned int useResvMem;
    unsigned int resvMemPoolId;
};

struct RsCqCreateAttr {
    struct RsRdevCb *rdevCb;
    int eqNum;
    int cqDepth;
    int cqEventId;
    struct ibv_cq *ibCq;
    struct ibv_comp_channel *channel;
    struct event_summary *event;
};

struct RsCqContext {
    struct RsRdevCb *rdevCb;
    int eqNum;
    int cqCreateMode;
    struct ibv_cq *ibSendCq;
    struct ibv_cq *ibRecvCq;
    struct ibv_cq *ibSrqCq;
    struct ibv_comp_channel *channel;
    struct event_summary *sendEvent;
    struct event_summary *recvEvent;
    struct RsCqContext *srqContext;
    int numRecvCqEvents;
};

/* rs_cb->state enum */
#define RS_STATE_HALT 4

struct RsAkid {
    char akidName[RS_KID_MAX_LENGTH];
};

struct RsIssuer {
    char issuerName[RS_KID_MAX_LENGTH];
};

struct RsCertAkidIssuerCb {
    struct RsAkid akids[RS_SSL_MAX_ALL_CERT_NUM];
    struct RsIssuer issers[RS_SSL_MAX_ALL_CERT_NUM];
};

struct RsSkid {
    char skidName[RS_KID_MAX_LENGTH];
};

struct RsSubject {
    char subjectName[RS_KID_MAX_LENGTH];
};

struct RsCertSkidSubjectCb {
    struct RsSkid skids[RS_SSL_MAX_ALL_CERT_NUM];
    struct RsSubject subjects[RS_SSL_MAX_ALL_CERT_NUM];
};

struct SensorNode {
    unsigned int logicDevid;
    int sensorUpdateCnt;
    uint64_t sensorHandle;
};

struct TlvBufInfo {
    unsigned int bufferSize;
    char *buf;
};

struct RsNslbCb {
    bool initFlag;
    void *netcoCb;
    pthread_mutex_t mutex;
};

struct RsTlvCb {
    unsigned int phyId;
    pthread_mutex_t mutex;
    struct TlvBufInfo bufInfo;
    bool initFlag;
    struct RsNslbCb nslbCb;
};

/*
 * Main Control block for device
 * for multi processor(device) in SMP system, each device have it's own rs_cb
 */
struct rs_cb {
    uint32_t chipId;
    uint32_t hccpMode;
    unsigned int logicId;
    enum ProtocolTypeT protocol;

    pthread_mutex_t mutex;

    sem_t connectTrigSem;
    uint32_t state;
    uint32_t sslEnable;
    SSL_CTX *serverSslCtx;
    SSL_CTX *clientSslCtx;
    struct RsCertSkidSubjectCb *skidSubjectCb;

    int connFlag;
    struct RsConnCb connCb;

    unsigned int devCnt;
    struct RsListHead rdevList;
    struct RsListHead udevList;
    struct RsListHead heterogTcpFdList;

    struct RsPingCtxCb pingCb;

    struct RsTlvCb tlvCb;

    char buf[RS_BUF_SIZE];
    struct ProcessRsSign pRsSign;

    unsigned long long notifyVaBase;
    unsigned long long notifySize;
    struct SensorNode sensorNode;

    void (*tcpRecvCallback)(const void *fdHandle);
    const void **fdMap;

    struct ifaddrs *ifaddrList;

    pid_t aicpuPid;
    unsigned int grpId;
    pid_t hostPid;
    bool grpSetupFlag;

    void *ndaCb;
    int ndaCbRefCnt;
    void *custom_ssl;
};

extern __thread struct rs_cb *gRsCb;

int RsSocketNodeid2vnic(uint32_t nodeId, uint32_t *ipAddr);
int RsGetHccpMode(unsigned int chipId);
int RsDev2conncb(uint32_t chipId, struct RsConnCb **connCb);
int RsDev2rscb(uint32_t chipId, struct rs_cb **rsCb, bool initFlag);
int RsQpn2qpcb(unsigned int phyId, unsigned int rdevIndex, uint32_t qpn, struct RsQpCb **qpCb);
int RsRdev2rdevCb(unsigned int chipId, unsigned int rdevIndex, struct RsRdevCb **rdevCb);
int RsGetRdevCb(struct rs_cb *rsCb, unsigned int rdevIndex, struct RsRdevCb **rdevCb);
void RsAccpetListNodeFree(struct rs_cb *rscb);
int RsWlistCheckConnAdd(struct rs_cb *rsCb, struct RsConnInfo *connTmp);
#ifdef CUSTOM_INTERFACE
int RsSetupSharemem(struct rs_cb *rsCb, bool backupFlag, unsigned int backupPhyid);
#endif
int RsQueryMrCb(struct RsRdevCb *devCb, uint64_t addr, struct RsMrCb **mrCb, struct RsListHead *mrList);
int RsGetRsCb(unsigned int phyId, struct rs_cb **rsCb);
int RsQueryGid(struct rdev rdevInfo, struct ibv_context *ibCtxTmp, uint8_t ibPort, int *gidIdx);
int RsEpollEventPingHandle(struct rs_cb *rsCb, int fd);
int RsSensorNodeRegister(unsigned int phyId, struct rs_cb *rsCb);
void RsSensorNodeUnregister(struct rs_cb *rsCb);
int RsRetryTimeoutExceptionCheck(struct SensorNode *snesorNode);
#endif // RS_INNER_H
