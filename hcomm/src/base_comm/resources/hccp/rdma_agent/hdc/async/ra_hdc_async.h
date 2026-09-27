/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_HDC_ASYNC_H
#define RA_HDC_ASYNC_H

#include "hccp_common.h"
#include "ra.h"
#include "ra_async.h"
#include "ra_rs_comm.h"
#include "ra_hdc.h"

union OpAsyncHdcConnectData {
    struct {
        unsigned int phyId;
        unsigned int queueSize;
        unsigned int threadNum;
        unsigned int resv[RA_RSVD_NUM_4];
    } txData;

    struct {
        unsigned int resv[RA_RSVD_NUM_4];
    } rxData;
};

union OpAsyncHdcCloseData {
    struct {
        unsigned int phyId;
        unsigned int resv[RA_RSVD_NUM_4];
    } txData;

    struct {
        unsigned int resv[RA_RSVD_NUM_4];
    } rxData;
};

int RaHdcInitAsync(struct RaInitConfig *cfg);
int RaHdcDeinitAsync(unsigned int phyId);
void RaHdcDeinitAsyncAll(void);
int RaHdcSendMsgAsync(unsigned int opcode, unsigned int phyId, char *data, unsigned int dataSize,
    struct RaRequestHandle *reqHandle);
void HdcAsyncDelResponse(struct RaRequestHandle *reqHandle);
int RaHdcAsyncSaveSnapshot(unsigned int phyId, enum SaveSnapshotAction action);
int RaHdcAsyncRestoreSnapshot(unsigned int phyId);
#endif // RA_HDC_ASYNC_H
