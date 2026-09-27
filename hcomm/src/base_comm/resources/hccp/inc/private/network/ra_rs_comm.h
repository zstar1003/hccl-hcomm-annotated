/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_RS_COMM_H
#define RA_RS_COMM_H
#include <stdint.h>
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include "hccp_common.h"
#include "rdma_lite_common.h"
#include "ra_rs_opcode.h"

#define HOST_LITE_RESERVED 4
#define RA_MR_MAX_NUM 8

enum {
    RA_RS_NOR_QP_MODE = 0,
    RA_RS_GDR_TMPL_QP_MODE = 1,
    RA_RS_OP_QP_MODE = 2,
    RA_RS_GDR_ASYN_QP_MODE = 3,
    RA_RS_OP_QP_MODE_EXT = 4,
    RA_RS_ERR_QP_MODE = 5,
};

struct OpcodeInterfaceInfo {
    enum OpType opcode;
    unsigned int version;
};

struct WrlistSendCompleteNum {
    unsigned int sendNum;
    unsigned int *completeNum;
};

struct RdmaMrRegAttr {
    void *addr;
    unsigned long long len;
    int access;
    unsigned int resv;
};

struct RdmaMrRegInfo {
    void *addr;
    unsigned long long len;
    int access;
    unsigned int lkey;
    unsigned int rkey;
};

struct SocketConnectInfo {
    unsigned int phyId;
    int family;
    union HccpIpAddr localIp;
    union HccpIpAddr remoteIp;
    unsigned int port;
    char tag[SOCK_CONN_TAG_SIZE];
};

struct SocketListenInfo {
    unsigned int phyId;
    int family;
    union HccpIpAddr localIp;
    unsigned int port;
    unsigned int phase;
    unsigned int err;
};

struct SocketFdData {
    int fd;
    unsigned int phyId;
    int family; // AF_INET(ipv4) or AF_INET6(ipv6)
    union HccpIpAddr localIp;
    union HccpIpAddr remoteIp;
    int status;
    char tag[SOCK_CONN_TAG_SIZE];
};

struct SocketPeerInfo {
    int phyId;
    int fd;
    void *socketHandle;
    uint32_t sslEnable;
};

struct QpAttrDscpInfo {
    unsigned char tc;
    unsigned char sl;
};

struct LiteMrInfo {
    uint32_t key;
    uint64_t addr;
    uint64_t len;
};

struct LiteRdevCapResp {
    struct dev_cap_info cap;
    unsigned char reserved[HOST_LITE_RESERVED];
};

struct LiteQpCqAttrResp {
    struct rdma_lite_device_qp_attr qpData;
    struct rdma_lite_device_cq_attr sendCqData;
    struct rdma_lite_device_cq_attr recvCqData;
    unsigned char reserved[HOST_LITE_RESERVED];
};

struct LiteQpAttrResp {
    struct rdma_lite_device_qp_attr qpData;
    unsigned char reserved[HOST_LITE_RESERVED];
};

struct LiteConnectedInfoResp {
    unsigned int state;
    struct LiteMrInfo localMr[RA_MR_MAX_NUM];
    struct LiteMrInfo remMr[RA_MR_MAX_NUM];
    struct QosAttr qosAttr;
    unsigned char reserved[HOST_LITE_RESERVED];
};

struct LiteMemAttrResp {
    struct rdma_lite_device_mem_attr memData;
    unsigned char reserved[HOST_LITE_RESERVED];
};

struct RsQpNormWithAttrs {
    int isExp;
    int isExt;
    struct QpExtAttrs extAttrs;
    unsigned int aiOpSupport;
};

struct RsQpRespWithAttrs {
    unsigned int qpn;
    unsigned long long aiQpAddr; // refer to struct ibv_qp *
    unsigned int sqIndex;        // index of sq
    unsigned int dbIndex;        // index of db
    unsigned int psn;
    unsigned int gidIdx;

    // below cq related info valid when data_plane_flag.bs.cq_cstm was 1
    unsigned long long aiScqAddr; // refer to struct ibv_cq *scq
    unsigned long long aiRcqAddr; // refer to struct ibv_cq *rcq
    struct AiDataPlaneInfo dataPlaneInfo;
};

struct RsQpResp {
    unsigned int qpn;
    unsigned int psn;
    unsigned int gidIdx;
    union ibv_gid gid;
    int directFlag;
};

struct RaRsDevInfo {
    unsigned int phyId;
    unsigned int devIndex;
};

struct TypicalQpAttr {
    unsigned int udpSport;
    int pathMtu;
    uint64_t vendorPrivInfo;
};

enum {
    THREAD_HALT = 0,
    THREAD_RUNNING = 1,
    THREAD_DESTROYING = 2,
};

enum {
    HDC_UNCONNECTED = 0,
    HDC_CONNECTED = 1,
};

struct TlvRequestMsgHead {
    unsigned int phyId;
    unsigned int type;
    unsigned int moduleType;
    unsigned int totalBytes;
    unsigned int sendBytes;
    unsigned int offset;
};

enum {
    CONTEXT_TYPE_JETTY = 0,
    CONTEXT_TYPE_JFC = 1,
};

#define RA_THREAD_TRY_TIME 500
#define RA_THREAD_SLEEP_TIME 2000
#define RA_CONNECT_TRY_TIME (500 * 120)
#define HDC_TRY_TIME 200
#define HDC_USLEEP_TIME 20000
#define THREAD_SLEEP_TIME 100 /* 100us */
#define MAX_POOL_QUEUE_SIZE_V1 512U
#define MAX_POOL_QUEUE_SIZE 2048U
#define MAX_POOL_THREAD_NUM 2U
#define RA_POOL_THREAD_NUM 1U

#define MAX_WR_NUM_V1 1024
#define MAX_WR_NUM 64
#define MAX_PORT_NUM 65535
#define MAX_IP_INFO_NUM 128
#define MAX_IP_INFO_NUM_V1 256
#define MAX_SGE_NUM 16
#define RA_RS_4K_PAGE_SIZE 0x1000
#define HCCN_CFG_MSG_DATA_LEN 2048U
#define MAX_TLV_MSG_DATA_LEN 2048U
#define MAX_TLV_MSG_DATA_LEN_V2 3072U

#define RA_RS_GET_IFNUM_VERSION 1
#define RA_RS_WLIST_ADD_V2_VERSION 1
#define RA_RS_WLIST_DEL_V2_VERSION 1
#define RA_RS_SEND_WRLIST_V2_VERSION 1
#define RA_RS_SEND_WRLIST_EXT_V2_VERSION 1
#define RA_RS_SOCKET_CONN_VERSION 2
#define RA_RS_SOCKET_LISTEN_VERSION 2
#define RA_RS_GET_SOCKET_VERSION 2
#define RA_RS_GET_VNIC_IP_INFOS_VERSION 1
#define RA_RS_GET_NOTIFY_BA_VERSION 1
#define RA_RS_OPCODE_BASE_VERSION 1

#define LITE_VERSION_V2 2
#define LITE_SUPPORT_DEV_MEM_REGISTER 1
#define LITE_SUPPORT_PCIE_BAR_HUGE_MEM (1 << 1)
#define LITE_NOT_SUPPORT 0
#define LITE_ALIGN_4KB 1
#define LITE_ALIGN_2MB (1 << 1)

#define RA_RS_GET_ALL_IP_BIT_MASK (1U << 31)

#define CQ_DEFAULT_MIN_SEND_DEPTH 64
#define CQ_DEFAULT_MIN_RECV_DEPTH 64
#define QP_DEFAULT_MIN_CAP_SEND_WR CQ_DEFAULT_MIN_SEND_DEPTH
#define QP_DEFAULT_MIN_CAP_RECV_WR CQ_DEFAULT_MIN_RECV_DEPTH
#define QP_DEFAULT_MAX_CAP_INLINE_DATA 32
#define QP_DEFAULT_MIN_CAP_SEND_SGE 1
#define QP_DEFAULT_MIN_CAP_RECV_SGE 1
#define QP_DEFAULT_MAX_ATTR_TIMEOUT 20
#define QP_DEFAULT_MAX_ATTR_RETRY_CNT 7

static inline void RaRsSetDevInfo(struct RaRsDevInfo *devInfo, unsigned int phyId, unsigned int devIndex)
{
    devInfo->phyId = phyId;
    devInfo->devIndex = devIndex;
}
#endif // RA_RS_COMM_H
