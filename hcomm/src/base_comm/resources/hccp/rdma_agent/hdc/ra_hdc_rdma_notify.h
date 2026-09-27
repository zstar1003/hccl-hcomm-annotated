/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_HDC_RDMA_NOTIFY_H
#define RA_HDC_RDMA_NOTIFY_H
#include "hccp_common.h"
#include "ra.h"
#include "ra_hdc.h"

#define RA_NOTIFY_TYPE_TOTAL_SIZE 1
#define RA_MEM_PHY_BIT 14
#define RA_MEM_TYPE_HBM (0x1 << RA_MEM_PHY_BIT)

union OpGetNotifyBaData {
    struct {
        unsigned int phyId;
        unsigned int rdevIndex;
        unsigned int qpn;
        unsigned long long rsvd[RA_RSVD_NUM_2];
    } txData;

    struct {
        unsigned long long va;
        int access;
        unsigned int lkey;
        unsigned long long size;
    } rxData;
};

union OpNotifyCfgSetData {
    struct {
        unsigned int phyId;
        unsigned long long va;
        unsigned long long size;
    } txData;

    struct {
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

union OpNotifyCfgGetData {
    struct {
        unsigned int phyId;
    } txData;

    struct {
        unsigned long long va;
        unsigned long long size;
        unsigned int rsvd[RA_RSVD_NUM_4];
    } rxData;
};

int RaHdcGetNotifyBaseAddr(struct RaRdmaHandle *rdmaHandle, unsigned long long *va, unsigned long long *size);
int RaHdcGetNotifyMrInfo(struct RaRdmaHandle *rdmaHandle, struct MrInfoT *info);
int RaHdcNotifyCfgSet(unsigned int phyId, unsigned long long va, unsigned long long size);
int RaHdcNotifyCfgGet(unsigned int phyId, unsigned long long *va, unsigned long long *size);
#endif // RA_HDC_RDMA_NOTIFY_H
