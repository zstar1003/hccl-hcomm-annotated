/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RS_PING_INNER_H
#define RS_PING_INNER_H

#include <pthread.h>
#include <infiniband/verbs.h>
#include <urma_types.h>
#include "hccp_ping.h"
#include "rs_list.h"
#include "rs_common_inner.h"

#define RS_PING_SEC_TO_USEC 1000000
#define RS_PING_MSEC_TO_USEC 1000
#define RS_PING_PERIOD_TIME_USEC 10000
#define RS_PING_PAYLOAD_HEADER_RESV_GRH 40U
#define RS_PING_PAYLOAD_HEADER_MASK_SIZE 104U
#define RS_PING_PAYLOAD_HEADER_RESV_CUSTOM sizeof(struct RsPingPayloadHeader)

enum RsPingThreadStatus { RS_PING_THREAD_RESET = 0, RS_PING_THREAD_RUNNING = 1, RS_PING_THREAD_FINISH = 2 };

enum RsPingTaskStatus { RS_PING_TASK_RESET = 0, RS_PING_TASK_RUNNING = 1 };

struct RsPingMrCb {
    uint32_t payloadOffset;
    uint64_t len;

    pthread_mutex_t mutex;
    uint64_t addr;

    struct ibv_mr *ibMr;
    uint32_t sgeNum;
    struct ibv_sge *sgeList;
    uint32_t sgeIdx;
};

struct RsPingCqInfo {
    int depth;
    int compVector;
    struct ibv_cq *ibCq;
    uint32_t numEvents;
    int maxRecvWcNum;
};

struct RsPingLocalQpCb {
    struct ibv_comp_channel *channel;
    struct RsPingCqInfo sendCq;
    struct RsPingCqInfo recvCq;

    uint32_t qkey;
    struct ibv_qp_cap qpCap;
    uint32_t udpSport;
    struct ibv_qp *ibQp;

    struct RsPingMrCb sendMrCb;
    struct RsPingMrCb recvMrCb;
};

enum RsPingPongTargetState {
    RS_PING_PONG_TARGET_RESET = 0,
    RS_PING_PONG_TARGET_READY = 1,
    RS_PING_PONG_TARGET_FINISH = 2,
    RS_PING_PONG_TARGET_ERROR = 3,
    RS_PING_PONG_TARGET_MAX
};

enum RsPingType {
    RS_PING_TYPE_ROCE_DETECT = 1,
    RS_PING_TYPE_ROCE_RESPONSE = 2,
    RS_PING_TYPE_URMA_DETECT = 3,
    RS_PING_TYPE_URMA_RESPONSE = 4
};

struct RsPingTimestamp {
    uint64_t tvSec1;
    uint64_t tvUsec1;
    uint64_t tvSec2;
    uint64_t tvUsec2;
    uint64_t tvSec3;
    uint64_t tvUsec3;
    uint64_t tvSec4;
    uint64_t tvUsec4;
};

struct RsPingPayloadHeader {
    int version;
    enum RsPingType type;
    struct PingQpInfo server;
    struct PingQpInfo target;
    struct RsPingTimestamp timestamp;
    uint32_t taskId;
    uint8_t reserved[42U];
    uint16_t magic;
};

struct RsPongTargetInfo {
    struct PingQpInfo qpInfo;
    union {
        struct ibv_ah *ah;
        urma_target_jetty_t *importTjetty;
    };

    enum RsPingPongTargetState state;

    struct RsListHead list;
    uint64_t uuid;
};

struct RsPingTargetInfo {
    char *payloadBuffer;
    uint32_t payloadSize;

    struct PingQpInfo qpInfo;
    union {
        struct ibv_ah *ah;
        urma_target_jetty_t *importTjetty;
    };

    enum RsPingPongTargetState state;

    pthread_mutex_t tripMutex;
    struct PingResultSummary resultSummary;

    struct RsListHead list;
    uint64_t uuid;
};

struct RsPingRdevCb {
    struct RsIpAddrInfo ip;
    const char *devName;
    unsigned char ibPort;
    union ibv_gid gid;

    int devNum;
    struct ibv_device **devList;
    int gidIdx;
    struct ibv_context *ibCtx;
    struct ibv_pd *ibPd;
};

struct RsPingSegCb {
    uint32_t payloadOffset;
    uint64_t len;

    pthread_mutex_t mutex;
    uint64_t addr;

    urma_token_t tokenValue;
    urma_target_seg_t *segment;
    uint32_t sgeNum;
    urma_sge_t *sgeList;
    uint32_t sgeIdx;
};

struct RsPingJfcInfo {
    int depth;
    urma_jfc_t *jfc;
    uint32_t numEvents;
    int maxRecvWcNum;
};

struct RsPingLocalJettyCb {
    urma_jfce_t *jfce;
    struct RsPingJfcInfo sendJfc;
    struct RsPingJfcInfo recvJfc;

    uint32_t tokenValue;
    urma_jfr_t *jfr;
    urma_jetty_t *jetty;

    struct RsPingSegCb sendSegCb;
    struct RsPingSegCb recvSegCb;
};

struct RsPingUdevCb {
    struct HccpDevEidInfo eidInfo;
    urma_device_t *urmaDev;
    urma_context_t *urmaCtx;
};

struct RsPingCtxCb {
    enum ProtocolTypeT protocol;
    struct RsPingPongOps *pingPongOps;
    struct RsPingPongDfx *pingPongDfx;
    pthread_t tid;
    int threadStatus;

    pthread_mutex_t pingMutex;
    struct RsListHead pingList;
    unsigned int pingNum;

    pthread_mutex_t pongMutex;
    struct RsListHead pongList;
    unsigned int pongNum;

    struct PingLocalCommInfo commInfo;
    unsigned int logicDevid;
    unsigned int initCnt;

    unsigned int devIndex;
    pthread_mutex_t devMutex;
    union {
        struct RsPingRdevCb rdevCb;
        struct RsPingUdevCb udevCb;
    };

    union {
        struct RsPingLocalQpCb pingQp;
        struct RsPingLocalJettyCb pingJetty;
    };

    union {
        struct RsPingLocalQpCb pongQp;
        struct RsPingLocalJettyCb pongJetty;
    };

    int taskStatus;
    struct PingTaskAttr taskAttr;
    unsigned int taskId;
};

struct RsPingPongOps {
    bool (*checkPingFd)(struct RsPingCtxCb *pingCb, int fd);
    bool (*checkPongFd)(struct RsPingCtxCb *pingCb, int fd);
    int (*initPingCb)(unsigned int phyId, struct PingInitAttr *attr, struct PingInitInfo *info, unsigned int *devIndex,
        struct RsPingCtxCb *pingCb);
    int (*pingFindTargetNode)(struct RsPingCtxCb *pingCb, struct PingQpInfo *target, struct RsPingTargetInfo **node);
    int (*pingAllocTargetNode)(struct RsPingCtxCb *pingCb, struct PingTargetInfo *target,
        struct RsPingTargetInfo **node);
    void (*resetRecvBuffer)(struct RsPingCtxCb *pingCb);
    int (*pingPostSend)(struct RsPingCtxCb *pingCb, struct RsPingTargetInfo *target);
    int (*pingPollScq)(struct RsPingCtxCb *pingCb, struct RsPingTargetInfo *target);
    int (*pingPollRcq)(struct RsPingCtxCb *pingCb, int *polledCnt, struct timeval *timestamp2);
    void (*pongHandleSend)(struct RsPingCtxCb *pingCb, int polledCnt, struct timeval *timestamp2);
    void (*pongPollRcq)(struct RsPingCtxCb *pingCb);
    int (*getTargetResult)(struct RsPingCtxCb *pingCb, struct PingTargetCommInfo *target,
        struct PingResultInfo *result);
    void (*pingFreeTargetNode)(struct RsPingCtxCb *pingCb, struct RsPingTargetInfo *targetInfo);
    void (*deinitPingCb)(unsigned int phyId, struct RsPingCtxCb *pingCb);
};

struct RsPingPongDfx {
    void (*addTargetSuccess)(struct PingTargetInfo *target, struct RsPingTargetInfo *targetInfo);
    void (*initPingCbSuccess)(unsigned int phyId, struct PingInitAttr *attr, unsigned int devIndex);
    void (*pingCannotFindTargetNode)(unsigned int i, int ret, struct PingTargetCommInfo target, unsigned int phyId);
};

uint32_t RsPingGetTripTime(struct RsPingTimestamp *timestamp);

#endif // RS_PING_INNER_H
