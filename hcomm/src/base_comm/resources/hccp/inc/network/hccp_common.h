/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __HCCP_COMMON_H__
#define __HCCP_COMMON_H__
#include <stdint.h>
#include <stdbool.h>
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include "hccp_opcode.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HCCP_ATTRI_VISI_DEF __attribute__((visibility("default")))

/**
 * @ingroup libsocket
 * implication of tag : tag is used for HCCL identification group information\n
 * macro : size of tag
 */
#define SOCK_CONN_TAG_SIZE 192
#define MAX_INTERFACE_NUM 8
#define MAX_INTERFACE_NAME_LEN 256
#define RA_QOS_ATTR_RESERVED 6
// 临时方案讲网卡数目提升到1024
#define MAX_INTERFACE_NUM_BAK 1024
#define MAX_SOCKET_EVENT_NUM 1024

enum NotifyTypeT {
    NO_USE = 0,
    NOTIFY = 1,
    EVENTID = 2,
};

/**
 * @ingroup libinit
 * white list switch
 */
enum WhiteListStatus {
    WHITE_LIST_DISABLE = 0, /**< enable white list */
    WHITE_LIST_ENABLE,      /**< disable white list */
};

/**
 * @ingroup librdma
 */
#define QP_CREATE_WITH_ATTR_VERSION 1
#define RA_QP_CREATE_WITH_ATTR_RESERVED 32
#define CQE_ERR_INFO_MAX_NUM 128
#define HCCP_GID_RAW_LEN 16U
#define REMAP_MR_MAX_NUM 128U

/**
 * @ingroup libinit
 * hccp operating environment
 */
enum NetworkMode {
    NETWORK_PEER_ONLINE = 0, /**< Third-party online mode */
    NETWORK_OFFLINE,         /**< offline mode */
    NETWORK_ONLINE,          /**< online mode */
};

/**
 * @ingroup libinit
 * hccp support protocol type
 */
enum ProtocolTypeT {
    PROTOCOL_RDMA = 0,
    PROTOCOL_UDMA,
    PROTOCOL_UNSUPPORT,
};

enum TransportModeT {
    CONN_RM = 1, /**< only for UB, Reliable Message */
};

/**
 * @ingroup libinit
 * rdma gid, aka infiniband ibv_gid
 */
union HccpGid {
    uint8_t raw[HCCP_GID_RAW_LEN];
    struct {
        uint64_t subnetPrefix;
        uint64_t interfaceId;
    } global;
};

/**
 * @ingroup libinit
 * rdma/ub gid/eid
 */
union HccpEid {
    uint8_t raw[16U]; /* Network Order */
    struct {
        uint64_t reserved; /* If IPv4 mapped to IPv6, == 0 */
        uint32_t prefix;   /* If IPv4 mapped to IPv6, == 0x0000ffff */
        uint32_t addr;     /* If IPv4 mapped to IPv6, == IPv4 addr */
    } in4;
    struct {
        uint64_t subnetPrefix;
        uint64_t interfaceId;
    } in6;
};

/**
 * @ingroup libinit
 * info need of rdma_agent
 */
struct RaInfo {
    int mode;           /**< reference to network_mode */
    unsigned int phyId; /**< physical device id */
};

/**
 * @ingroup libinit
 * ip address
 */
union HccpIpAddr {
    struct in_addr addr;
    struct in6_addr addr6;
};

/**
 * @ingroup libinit
 * hccp init info
 */
struct rdev {
    unsigned int phyId; /**< physical device id */
    int family;         /**< AF_INET(ipv4) or AF_INET6(ipv6) */
    union HccpIpAddr localIp;
};

struct RdevInitInfo {
    int mode;
    unsigned int notifyType;
    bool enabled910aLite;    /**< true will enable 910A lite, invalid if enabled_2mb_lite is false; default is false */
    bool disabledLiteThread; /**< true will not start lite thread, flag invalid if enabled_910a/2mb_lite is false */
    bool enabled2mbLite;     /**< true will enable 2MB lite(include 910A & 910B), default is false */
};

struct SocketInitInfoT {
    struct rdev rdevInfo;
    int scopeId;
};

/**
 * @ingroup libsocket
 * socket listen status
 */
enum ListenPhase {
    LISTEN_OK = 0,            /**< socket listen ok */
    LISTEN_CREATE_FD_ERR = 1, /**< socket create fd error */
    LISTEN_BIND_ERR = 2,      /**< socket bind socket port error */
    LISTEN_BEGIN_ERR = 3,     /**< socket listen error */
};

/**
 * @ingroup libsocket
 * struct of the listen info
 */
struct SocketListenInfoT {
    void *socketHandle; /**< socket handle */
    unsigned int port;  /**< Socket listening port number */
    unsigned int phase; /**< refer to enum listen_phase */
    unsigned int err;   /**< errno */
};

/**
 * @ingroup libsocket
 * struct of the client socket
 */
struct SocketConnectInfoT {
    void *socketHandle;           /**< socket handle */
    union HccpIpAddr remoteIp;    /**< IP address of remote socket, [0-7] is reserved for vnic */
    unsigned int port;            /**< Socket listening port number */
    char tag[SOCK_CONN_TAG_SIZE]; /**< tag must ended by '\0' */
};

/**
 * @ingroup libsocket
 * struct of the socket to be closed
 */
struct SocketCloseInfoT {
    void *socketHandle; /**< socket handle */
    void *fdHandle;     /**< fd handle */
    int disuseLinger;   /**< 0:use(default l_linger is RS_CLOSE_TIMEOUT), others:disuse */
};

/**
 * @ingroup libsocket
 * Details about socket after socket is linked
 */
struct SocketInfoT {
    void *socketHandle;           /**< socket handle */
    void *fdHandle;               /**< fd handle */
    union HccpIpAddr remoteIp;    /**< IP address of remote socket */
    int status;                   /**< socket status:0 not connected 1:connected 2:connect timeout 3:connecting */
    char tag[SOCK_CONN_TAG_SIZE]; /**< tag must ended by '\0' */
};

/**
 * @ingroup libsocket
 * struct of socket event info
 */
#ifdef __x86_64__
#define EPOLL_PACKED __attribute__((packed))
#else
#define EPOLL_PACKED
#endif

struct SocketEventInfoT {
    uint32_t event;
    union {
        void *fdHandle;
        uint64_t u64;
    };
} EPOLL_PACKED;
/**
 * @ingroup libinit
 * Configuration of rdma_agent initialization
 */
struct RaInitConfig {
    unsigned int phyId;       /**< physical device id */
    unsigned int nicPosition; /**< reference to network_mode */
    int hdcType;              /**< reference to drvHdcServiceType */
    bool enableHdcAsync;      /**< true will init an extra HDC session for async APIs */
};

struct RaGetIfattr {
    unsigned int phyId;       /**< physical device id */
    unsigned int nicPosition; /**< reference to network_mode */
    bool isAll; /**< valid when nic_position is NETWORK_OFFLINE. false: get specific rnic ip, true: get all rnic ip */
};

/**
 * @ingroup libinit
 * id type to get vnic ip
 */
enum IdType { PHY_ID_VNIC_IP, SDID_VNIC_IP };

/**
 * @ingroup libinit
 * struct of ip info
 */
struct IpInfo {
    int family;
    union HccpIpAddr ip;
    uint32_t resv[2U];
};

/**
 * @ingroup librdma
 * Flag of RDMA operations
 */
enum RaSendFlags {
    RA_SEND_FENCE = 1 << 0,     /**< RDMA operation with fence */
    RA_SEND_SIGNALED = 1 << 1,  /**< RDMA operation with signaled */
    RA_SEND_SOLICITED = 1 << 2, /**< RDMA operation with solicited */
    RA_SEND_INLINE = 1 << 3,    /**< RDMA operation with*/
};

/**
 * @ingroup librdma
 * Scatter and gather element
 */
struct SgList {
    uint64_t addr; /**< address of buf */
    uint32_t len;  /**< len of buf */
    uint32_t lkey; /**< local addr access key */
};

enum RaWrOpcode {
    RA_WR_RDMA_WRITE,
    RA_WR_RDMA_WRITE_WITH_IMM,
    RA_WR_SEND,
    RA_WR_SEND_WITH_IMM,
    RA_WR_RDMA_READ,
    RA_WR_RDMA_ATOMIC_WRITE = 0xf0,
    RA_WR_RDMA_WRITE_WITH_NOTIFY = 0xf2,
    RA_WR_RDMA_REDUCE_WRITE = 0xf5,
    RA_WR_RDMA_REDUCE_WRITE_WITH_NOTIFY = 0xf6,
};

/**
 * @ingroup librdma
 * port status
 */
enum PortStatus {
    PORT_STATUS_DOWN = 0,
    PORT_STATUS_ACTIVE = 1,
};

/**
 * @ingroup librdma
 * RDMA work request
 */
struct SendWr {
    struct SgList *bufList; /**< list of sg */
    uint16_t bufNum;        /**< num of buf_list */
    uint64_t dstAddr;       /**< destination address */
    uint32_t rkey;          /**< remote address access key */
    uint32_t op;            /**< operations of RDMA supported:RDMA_WRITE:0 */
    int sendFlag;           /**< reference to ra_send_flags */
};

struct WrAuxInfo {
    uint8_t dataType;
    uint8_t reduceType;
    uint32_t notifyOffset;
};

struct WrExtInfo {
    uint32_t immData;
    uint16_t resv;
};

struct SendWrlistData {
    unsigned long long dstAddr; /**< destination address */
    unsigned int op;            /**< operations of RDMA supported:RDMA_WRITE:0, RDMA_READ:4 */
    int sendFlags;              /**< reference to ra_send_flags */
    struct SgList memList;      /**< list of sg */
};

struct SendWrlistDataExt {
    unsigned long long dstAddr; /**< destination address */
    unsigned int op;            /**< operations of RDMA supported:RDMA_WRITE:0, RDMA_READ:4 */
    int sendFlags;              /**< reference to ra_send_flags */
    struct SgList memList;      /**< list of sg */
    union {
        struct WrAuxInfo aux; /**< aux info */
        struct WrExtInfo ext; /**< ext info */
    };
};

struct SendWrV2 {
    uint64_t wrId;          /**< user assigned work request ID */
    struct SgList *bufList; /**< list of sg */
    uint16_t bufNum;        /**< num of buf_list */
    uint64_t dstAddr;       /**< destination address */
    uint32_t rkey;          /**< remote address access key */
    uint32_t op;            /**< operations of RDMA supported:RDMA_WRITE:0 */
    int sendFlag;           /**< reference to ra_send_flags */
    union {
        struct WrAuxInfo aux; /**< aux info */
        struct WrExtInfo ext; /**< ext info */
    };
};

struct SendWrVerbs {
    uint64_t wrId;
    struct SgList *sgList;
    int numSge;
    uint32_t opcode;
    int sendFlags;
    uint32_t immData;
    uint64_t remoteAddr;
    uint32_t rkey;
};

struct RecvWrVerbs {
    uint64_t wrId;
    struct SgList *sgList;
    int numSge;
};

struct WrInfo {
    int sendFlags;         /**< reference to ra_send_flags */
    uint32_t rkey;         /**< remote address access key */
    uint32_t op;           /**< operations of RDMA supported:ra_wr_opcode */
    uint32_t immData;      /**< imm data */
    uint64_t wrId;         /**< user assigned work request ID */
    uint64_t dstAddr;      /**< destination address */
    struct SgList memList; /**< sg info */
    struct WrAuxInfo aux;  /**< aux info */
};

struct RecvWrlistData {
    uint64_t wrId;         /**< user assigned work request ID */
    struct SgList memList; /**< list of sg */
};

/**
 * @ingroup libsocket
 * Socket whitelist
 */
struct SocketWlistInfoT {
    union HccpIpAddr remoteIp;    /**< IP address of remote */
    unsigned int connLimit;       /**< limit of white list */
    char tag[SOCK_CONN_TAG_SIZE]; /**< tag used for whitelist must ended by '\0' */
};

/**
 * @ingroup librdma
 * Flag of mr access
 */
enum RaAccessFlags {
    RA_ACCESS_LOCAL_WRITE = 1,          /**< mr local write access */
    RA_ACCESS_REMOTE_WRITE = (1 << 1),  /**< mr remote write access */
    RA_ACCESS_REMOTE_READ = (1 << 2),   /**< mr remote read access */
    RA_ACCESS_REMOTE_ATOMIC = (1 << 3), /**< mr remote atomic access */
    RA_ACCESS_REDUCE = (1 << 8),
};

#define mem_mr_access_flags ra_access_flags

/**
 * @ingroup librdma
 * wqe template info
 */
struct WqeInfoT {
    unsigned int sqIndex;  /**< index of sq */
    unsigned int wqeIndex; /**< index of wqe */
};

/**
 * @ingroup librdma
 * doorbell info
 */
struct DbInfo {
    unsigned int dbIndex; /**< index of db */
    unsigned long dbInfo; /**< db content */
};

/**
 * @ingroup librdma
 * respond of sending work request
 */
struct SendWrRsp {
    union {
        struct WqeInfoT wqeTmp; /**< wqe template info */
        struct DbInfo db;       /**< doorbell info */
    };
};

struct IfaddrInfo {
    union HccpIpAddr ip; /* Address of interface */
    struct in_addr mask; /* Netmask of interface */
};

struct InterfaceInfo {
    int family;
    int scopeId;
    struct IfaddrInfo ifaddr;            /* Address and netmask of interface */
    char ifname[MAX_INTERFACE_NAME_LEN]; /* Name of interface */
};

struct MrInfoT {
    void *addr;              /**< starting address of mr */
    unsigned long long size; /**< size of mr */
    int access;              /**< access of mr, reference to ra_access_flags */
    unsigned int lkey;       /**< local addr access key */
    unsigned int rkey;       /**< remote addr access key */
};

struct MemRemapInfo {
    void *addr;              /**< starting address of needed remap memory */
    unsigned long long size; /**< size of needed remap memory */
};

enum RaWcOpcode {
    RA_WC_SEND,
    RA_WC_RDMA_WRITE,
    RA_WC_RDMA_READ,
    RA_WC_RECV = 1 << 7,
    RA_WC_RECV_RDMA_WITH_IMM,
};

enum RaEpollEvent {
    RA_EPOLLIN = 0,
    RA_EPOLLOUT,
    RA_EPOLLPRI,
    RA_EPOLLERR,
    RA_EPOLLHUP,
    RA_EPOLLET,
    RA_EPOLLONESHOT,
    RA_EPOLLOUT_LET_ONESHOT,
    RA_EPOLLINVALD
};

struct CqAttr {
    void **qpContext;
    struct ibv_cq **ibSendCq;
    struct ibv_cq **ibRecvCq;
    int sendCqDepth;
    int recvCqDepth;
    int sendCqEventId;
    int recvCqEventId;
    struct ibv_comp_channel *sendChannel;
    struct ibv_comp_channel *recvChannel;
    void *srqContext;
};

struct SrqAttr {
    void **context;
    struct ibv_srq **ibSrq;
    struct ibv_cq **ibRecvCq;
    int cqDepth;
    int srqDepth;
    int maxSge;
    int srqEventId;
};

struct QosAttr {
    unsigned char tc; // traffic class
    unsigned char sl; // priority(service level)
    unsigned char reserved[RA_QOS_ATTR_RESERVED];
};

struct CqeErrInfo {
    uint32_t status;
    uint32_t qpn;
    struct timeval time;
};

struct CqExtAttr {
    int sendCqDepth;
    int recvCqDepth;
    int sendCqCompVector;
    int recvCqCompVector;
};

struct AiDataPlaneWq {
    unsigned wqn;
    unsigned long long bufAddr;
    unsigned int wqebbSize;
    unsigned int depth;
    unsigned long long headAddr;
    unsigned long long tailAddr;
    unsigned long long swdbAddr;
    unsigned long long dbReg;
    unsigned int reserved[8U];
};

struct AiDataPlaneCq {
    unsigned int cqn;
    unsigned long long bufAddr;
    unsigned int cqeSize;
    unsigned int depth;
    unsigned long long headAddr;
    unsigned long long tailAddr;
    unsigned long long swdbAddr;
    unsigned long long dbReg;
    unsigned int reserved[2U];
};

struct AiDataPlaneInfo {
    struct AiDataPlaneWq sq;
    struct AiDataPlaneWq rq;
    struct AiDataPlaneCq scq;
    struct AiDataPlaneCq rcq;
    unsigned int reserved[8U];
};

union AiDataPlaneCstmFlag {
    struct {
        uint32_t cqCstm : 1; // 0: hccp poll cq; 1: caller poll cq
        uint32_t reserved : 31;
    } bs;
    uint32_t value;
};

enum {
    HCCP_RDMA_NOR_MODE = 0,
    HCCP_RDMA_GDR_TMPL_MODE = 1,
    HCCP_RDMA_OP_MODE = 2,
    HCCP_RDMA_GDR_ASYN_MODE = 3,
    HCCP_RDMA_OP_MODE_EXT = 4,
    HCCP_RDMA_ERR_MODE = 5
};

struct QpExtAttrs {
    int qpMode;
    // cq attr
    struct CqExtAttr cqAttr;
    // qp attr
    struct ibv_qp_init_attr qpAttr;
    // version control and reserved
    int version;
    int memAlign; // 0,1:4KB, 2:2MB
    uint32_t udpSport;
    union AiDataPlaneCstmFlag dataPlaneFlag; // only valid in ra_ai_qp_create
    union {
        struct {
            uint32_t useResvMem : 1;
            uint32_t reserved0 : 31;
        } bs;
        uint32_t value;
    } cstmFlag;             // only valid in RaQpCreateWithAttrs
    uint32_t resvMemPoolId; // valid when cstmFlag.bs.useResvMem was 1
    uint32_t reserved[27U];
};

struct AiQpInfo {
    unsigned long long aiQpAddr; // refer to struct ibv_qp *
    unsigned int sqIndex;        // index of sq
    unsigned int dbIndex;        // index of db

    // below cq related info valid when data_plane_flag.bs.cq_cstm was 1
    unsigned long long aiScqAddr; // refer to struct ibv_cq *scq
    unsigned long long aiRcqAddr; // refer to struct ibv_cq *rcq
    struct AiDataPlaneInfo dataPlaneInfo;
};

struct HyperFeature {
    char rawData[64U];
};

struct TypicalQp {
    uint32_t qpn;
    uint32_t psn;
    uint32_t gidIdx;
    uint8_t resv1[4U]; // for compatibility issue
    uint8_t gid[HCCP_GID_RAW_LEN];
    uint32_t tc; // tc must be consistent with RaSetQpAttr in hyper roce
    uint32_t sl; // sl must be consistent with RaSetQpAttr in hyper roce
    uint32_t retryCnt;
    uint32_t retryTime;
    // version control and reserved
    int version;
    struct HyperFeature feature;
    uint32_t udpSport;
    uint32_t reserved[15U];
    uint8_t resv2[4U]; // for compatibility issue
};

struct QpAttr {
    unsigned int qpn;
    unsigned int udpSport;
    unsigned int psn;
    unsigned int gidIdx;
    unsigned char gid[HCCP_GID_RAW_LEN];
    int pathMtu;
    uint64_t vendorPrivInfo;
    struct HyperFeature feature; // depend on RaSetQpAttrQos
};

struct LoopbackQpPair {
    void *ibvQp0;
    void *ibvQp1;
};

struct SocketErrInfo {
    struct timeval time;
    int errNo;
    int action; // refer to enum rs_conn_state
    char resv[32U];
};

struct ServerSocketErrInfo {
    struct SocketErrInfo epollWait;
    struct SocketErrInfo accept;
};

enum SaveSnapshotAction {
    SAVE_SNAPSHOT_ACTION_PRE_PROCESSING = 0,
    SAVE_SNAPSHOT_ACTION_POST_PROCESSING = 1,
    SAVE_SNAPSHOT_ACTION_MAX,
};

#ifdef __cplusplus
}
#endif
#endif
