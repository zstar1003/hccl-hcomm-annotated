/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCP_PING_H
#define HCCP_PING_H

#include "hccp_common.h"
#include "hccp_ctx.h"

#ifdef __cplusplus
extern "C" {
#endif

struct QpCap {
    uint32_t maxSendWr;
    uint32_t maxRecvWr;
    uint32_t maxSendSge;
    uint32_t maxRecvSge;
    uint32_t maxInlineData;
};

union PingQpAttr {
    struct {
        struct CqExtAttr cqAttr;
        struct {
            struct QpCap cap;
            uint32_t udpSport;
            uint32_t reserved[4U];
        } qpAttr;
        uint32_t reserved[4U];
    } rdma;
    struct {
        struct CqExtAttr cqAttr;
        struct {
            struct QpCap cap;
            uint32_t tokenValue; /**< refer to urma_token_t */
            uint32_t reserved[3U];
        } qpAttr;
        struct {
            uint32_t tokenValue;
        } segAttr;
        uint32_t reserved[4U];
    } ub;
};

struct PingLocalCommInfo {
    int version;
    union {
        struct {
            uint32_t flowLabel;
            uint8_t hopLimit;
            struct QosAttr qosAttr;
            uint32_t udpSport;
            uint32_t reserved[7U];
        } rdma;
        struct {
            struct QosAttr qosAttr;
            uint32_t reserved[7U];
        } ub;
    };
};

union PingDev {
    struct rdev rdma;
    struct {
        union HccpEid eid;
        uint32_t eidIndex;
    } ub;
};

struct PingInitAttr {
    int version;
    int mode;
    union PingDev dev;
    struct PingLocalCommInfo commInfo;
    union PingQpAttr client;
    union PingQpAttr server;
    uint32_t bufferSize;
    enum ProtocolTypeT protocol;
    union {
        struct {
            uint32_t reserved[31U];
        } rdma;
        struct {
            unsigned int phyId;
            uint32_t reserved[30U];
        } ub;
    };
};

struct PingQpInfo {
    int version;
    union {
        struct {
            union HccpGid gid;
            uint32_t qpn;
            uint32_t qkey;
            uint32_t reserved[4U];
        } rdma;
        struct {
            uint8_t size;
            uint8_t key[28U]; // refer to qp_key
            uint8_t reserved[7U];
            uint32_t tokenValue;
        } ub;
    };
};

struct PingBufferInfo {
    // all result buffer
    uint64_t bufferVa;
    uint32_t bufferSize;
    // each payload offset & header size
    uint32_t payloadOffset;
    uint32_t headerSize;
};

struct PingInitInfo {
    int version;
    struct PingQpInfo client;
    struct PingQpInfo server;
    struct PingBufferInfo result;
    uint32_t reserved[32U];
};

struct PingTaskAttr {
    uint32_t packetCnt;
    uint32_t packetInterval;
    uint32_t timeoutInterval;
};

#define PING_TOTAL_PAYLOAD_MAX_SIZE 2048U
#define PING_USER_PAYLOAD_MAX_SIZE 1500U

struct PingPayloadInfo {
    char buffer[PING_USER_PAYLOAD_MAX_SIZE];
    uint32_t size;
};

struct PingTargetCommInfo {
    union {
        union HccpIpAddr ip;
        union HccpEid eid;
    };
    struct PingQpInfo qpInfo;
};

struct PingTargetInfo {
    int version;
    struct PingLocalCommInfo localInfo;
    struct PingTargetCommInfo remoteInfo;
    struct PingPayloadInfo payload;
    uint32_t reserved[16U];
};

enum PingResultState {
    PING_RESULT_STATE_NOT_FOUND = 0,
    PING_RESULT_STATE_INVALID = 1,
    PING_RESULT_STATE_VALID = 2,
    PING_RESULT_STATE_MAX
};

struct PingResultSummary {
    int version;
    struct PingTaskAttr taskAttr;

    uint32_t rttMin; /**< tv_usec */
    uint32_t rttMax; /**< tv_usec */
    uint32_t rttAvg; /**< tv_usec */

    uint32_t sendCnt;
    uint32_t recvCnt;
    uint32_t timeoutCnt;

    uint32_t taskId;
    uint32_t reserved[31U];
};

struct PingResultInfo {
    enum PingResultState state;
    struct PingResultSummary summary;
};

struct PingTargetResult {
    struct PingTargetCommInfo remoteInfo;
    struct PingResultInfo result;
};

/**
 * @ingroup libinit
 * @brief Rdma_agent ping initialization
 * @param init_attr [IN] init attr
 * @param init_info [OUT] init info
 * @param ping_handle [OUT] ping handle info
 * @see ra_ping_deinit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaPingInit(struct PingInitAttr *initAttr, struct PingInitInfo *initInfo, void **pingHandle);

/**
 * @ingroup librdma
 * @brief Rdma_agent add target to list
 * @param ping_handle [IN] ping handle info
 * @param target [IN] ping target info
 * @param num [IN] num of target
 * @see ra_ping_target_del
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaPingTargetAdd(void *pingHandle, struct PingTargetInfo target[], uint32_t num);

/**
 * @ingroup librdma
 * @brief Rdma_agent trigger ping task start
 * @param ping_handle [IN] ping handle info
 * @param attr [IN] ping task attr
 * @see ra_ping_task_stop
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaPingTaskStart(void *pingHandle, struct PingTaskAttr *attr);

/**
 * @ingroup librdma
 * @brief Rdma_agent ping get results
 * @param ping_handle [IN] ping handle info
 * @param target [IN/OUT] ping result info
 * @param num [IN/OUT] num of target & num of results got
 * @see ra_ping_target_del
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaPingGetResults(void *pingHandle, struct PingTargetResult target[], uint32_t *num);

/**
 * @ingroup librdma
 * @brief Rdma_agent del target from list
 * @param ping_handle [IN] ping handle info
 * @param target [IN] ping target info
 * @param num [IN] num of target
 * @see ra_ping_target_add
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaPingTargetDel(void *pingHandle, struct PingTargetCommInfo target[], uint32_t num);

/**
 * @ingroup librdma
 * @brief Rdma_agent trigger ping task stop
 * @param ping_handle [IN] ping handle info
 * @see ra_ping_task_start
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaPingTaskStop(void *pingHandle);

/**
 * @ingroup libinit
 * @brief Rdma_agent ping deinitialization
 * @param ping_handle [IN] ping handle info
 * @see ra_ping_init
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaPingDeinit(void *pingHandle);

#ifdef __cplusplus
}
#endif
#endif // HCCP_PING_H