/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_HDC_PING_H
#define RA_HDC_PING_H

#include "hccp_ping.h"
#include "ra_ping.h"
#include "ra_rs_comm.h"
#include "ra_hdc.h"

#define RA_MAX_PING_TARGET_NUM 16

union OpPingInitData {
    struct {
        struct PingInitAttr attr;
        uint32_t reserved[RA_RSVD_NUM_4];
    } txData;

    struct {
        unsigned int devIndex;
        struct PingInitInfo info;
        uint32_t reserved[RA_RSVD_NUM_4];
    } rxData;
};

union OpPingAddData {
    struct {
        struct RaRsDevInfo rdev;
        struct PingTargetInfo target;
        uint32_t reserved[RA_RSVD_NUM_4];
    } txData;

    struct {
        uint32_t reserved[RA_RSVD_NUM_64];
    } rxData;
};

union OpPingStartData {
    struct {
        struct RaRsDevInfo rdev;
        struct PingTaskAttr attr;
        uint32_t reserved[RA_RSVD_NUM_4];
    } txData;

    struct {
        uint32_t reserved[RA_RSVD_NUM_64];
    } rxData;
};

union OpPingResultsData {
    struct {
        struct RaRsDevInfo rdev;
        unsigned int num;
        struct PingTargetCommInfo target[RA_MAX_PING_TARGET_NUM];
        uint32_t reserved[RA_RSVD_NUM_4];
    } txData;

    struct {
        unsigned int num;
        struct PingResultInfo target[RA_MAX_PING_TARGET_NUM];
        uint32_t reserved[RA_RSVD_NUM_4];
    } rxData;
};

union OpPingDelData {
    struct {
        struct RaRsDevInfo rdev;
        unsigned int num;
        struct PingTargetCommInfo target[RA_MAX_PING_TARGET_NUM];
        uint32_t reserved[RA_RSVD_NUM_4];
    } txData;

    struct {
        unsigned int num;
        uint32_t reserved[RA_RSVD_NUM_16];
    } rxData;
};

union OpPingStopData {
    struct {
        struct RaRsDevInfo rdev;
        uint32_t reserved[RA_RSVD_NUM_4];
    } txData;

    struct {
        uint32_t reserved[RA_RSVD_NUM_64];
    } rxData;
};

union OpPingDeinitData {
    struct {
        struct RaRsDevInfo rdev;
        uint32_t reserved[RA_RSVD_NUM_4];
    } txData;

    struct {
        uint32_t reserved[RA_RSVD_NUM_4];
    } rxData;
};

int RaHdcPingInit(struct RaPingHandle *pingHandle, struct PingInitAttr *initAttr, struct PingInitInfo *initInfo);
int RaHdcPingTargetAdd(struct RaPingHandle *pingHandle, struct PingTargetInfo target[], uint32_t num);
int RaHdcPingTaskStart(struct RaPingHandle *pingHandle, struct PingTaskAttr *attr);
int RaHdcPingGetResults(struct RaPingHandle *pingHandle, struct PingTargetResult target[], uint32_t *num);
int RaHdcPingTargetDel(struct RaPingHandle *pingHandle, struct PingTargetCommInfo target[], uint32_t num);
int RaHdcPingTaskStop(struct RaPingHandle *pingHandle);
int RaHdcPingDeinit(struct RaPingHandle *pingHandle);
#endif // RA_HDC_PING_H
