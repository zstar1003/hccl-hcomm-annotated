/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <errno.h>
#include <stdlib.h>
#include "ut_dispatch.h"
#include "securec.h"
#include "hccp_nda.h"
#include "ra_rs_err.h"
#include "ra.h"
#include "ra_peer_nda.h"
#include "rs_nda.h"
#include "tc_ra_nda.h"

extern int RaQpDestroy(void* qpHandle);
extern struct RaRdmaOps gRaPeerRdmaOps;

void TcRaNdaGetDirectFlag()
{
    struct RaRdmaHandle rdmaHandle = {0};
    int directFlag = 0;
    int ret = 0;

    ret = RaNdaGetDirectFlag(NULL, &directFlag);
    EXPECT_INT_EQ(128103, ret);

    ret = RaNdaGetDirectFlag(&rdmaHandle, &directFlag);
    EXPECT_INT_EQ(0, ret);
}

void TcRaPeerNdaGetDirectFlag()
{
    struct RaRdmaHandle rdmaHandle = {0};
    int directFlag = 0;
    int ret = 0;

    mocker_clean();
    mocker(RsNdaGetDirectFlag, 1, -1);
    ret = RaPeerNdaGetDirectFlag(&rdmaHandle, &directFlag);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();
}

void TcRaNdaQpCreate()
{
    struct RaRdmaHandle rdmaHandle = {0};
    struct NdaQpInitAttr attr = {0};
    struct NdaQpInfo info = {0};
    void* qpHandle = NULL;
    int ret = 0;

    rdmaHandle.rdmaOps = &gRaPeerRdmaOps;

    ret = RaNdaQpCreate(NULL, &attr, &info, &qpHandle);
    EXPECT_INT_EQ(128103, ret);

    ret = RaNdaQpCreate(&rdmaHandle, &attr, &info, &qpHandle);
    EXPECT_INT_EQ(0, ret);

    ret = RaQpDestroy(qpHandle);
    EXPECT_INT_EQ(0, ret);
}

void TcRaPeerNdaQpCreate()
{
    struct RaRdmaHandle rdmaHandle = {0};
    struct NdaQpInitAttr attr = {0};
    struct NdaQpInfo info = {0};
    void* qpHandle = NULL;
    int ret = 0;

    mocker(RsNdaQpCreate, 1, -1);
    ret = RaPeerNdaQpCreate(&rdmaHandle, &attr, &info, &qpHandle);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();
}

void TcRaPeerGetQpHyperFeature()
{
    struct RaQpHandle qpHandle = {0};
    struct HyperFeature hyperFeature = {0};
    int ret = 0;

    qpHandle.phyId = 0;
    qpHandle.rdevIndex = 0;
    qpHandle.qpn = 0;

    ret = RaPeerGetQpHyperFeature(&qpHandle, &hyperFeature);
    EXPECT_INT_EQ(0, ret);

    mocker((stub_fn_t)RsGetQpHyperFeature, 10, -1);
    ret = RaPeerGetQpHyperFeature(&qpHandle, &hyperFeature);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();
}
