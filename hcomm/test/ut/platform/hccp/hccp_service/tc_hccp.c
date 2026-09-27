/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <getopt.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include "tsd.h"
#include "stdio.h"
#include "ut_dispatch.h"
#include "hccp_pub.h"
#include "hccp_common.h"
#include "securec.h"
#include "param.h"
#include "dl_hal_function.h"

extern int LltMain(int argc, char* argv);
extern int HccpAddToCgroup(int logicId);
extern int HccpParamParse(int argc, char* argv[], struct HccpInitParam* param);
extern int HccpSetLogInfo(struct HccpInitParam* param);
extern void RsApiDeinit(void);
extern int RsApiInit(void);
extern int HccpChangeNumOfFile();
int dlDrvGetDevNum(unsigned int* numDev);
int dlDrvDeviceGetPhyIdByIndex(unsigned int devIndex, unsigned int* phyId);

extern int HccpParamParseId(const char* input, int* id);
extern int HccpParseLogicId(const char* input, struct HccpInitParam* param);
extern int HccpParsePid(const char* input, struct HccpInitParam* param);
extern int HccpParsePidSign(const char* input, struct HccpInitParam* param);
extern int HccpParseLogLevel(const char* input, struct HccpInitParam* param);
extern int HccpParseHdcType(const char* input, struct HccpInitParam* param);
extern int HccpParseWhiteListStatus(const char* input, struct HccpInitParam* param);
extern int HccpParseBackupPhyid(const char* input, struct HccpInitParam* param);

static void ResetParam(struct HccpInitParam* param)
{
    memset_s(param, sizeof(struct HccpInitParam), 0, sizeof(struct HccpInitParam));
    param->hdcType = HDC_SERVICE_TYPE_RDMA;
    param->whiteListStatus = WHITE_LIST_ENABLE;
    param->backupFlag = false;
}

static void ResetOptind(void) { optind = 1; }

void TcNormal()
{
    struct HccpInitParam param;
    int ret;

    ResetParam(&param);
    ret = HccpParamParseId("123", &param.logicId);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(param.logicId, 123);

    ResetParam(&param);
    ret = HccpParamParseId("-456", &param.logicId);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(param.logicId, -456);

    ResetParam(&param);
    ret = HccpParamParseId("abc", &param.logicId);
    EXPECT_INT_NE(ret, 0);

    ResetParam(&param);
    ret = HccpParamParseId("", &param.logicId);
    EXPECT_INT_NE(ret, 0);

    ResetParam(&param);
    ret = HccpParsePidSign("dummy", &param);
    EXPECT_INT_EQ(ret, 0);

    ResetParam(&param);
    ret = HccpParsePid("100", &param);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(param.pid, 100);

    ResetParam(&param);
    ret = HccpParsePid("0", &param);
    EXPECT_INT_NE(ret, 0);

    ResetParam(&param);
    ret = HccpParsePid("-1", &param);
    EXPECT_INT_NE(ret, 0);

    ResetParam(&param);
    ret = HccpParsePid("abc", &param);
    EXPECT_INT_NE(ret, 0);

    ResetParam(&param);
    ret = HccpParseLogLevel("3", &param);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(param.logLevel, 3);

    ResetParam(&param);
    ret = HccpParseLogLevel("0", &param);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(param.logLevel, 0);

    ResetParam(&param);
    ret = HccpParseLogLevel("abc", &param);
    EXPECT_INT_NE(ret, 0);

    ResetParam(&param);
    ret = HccpParseHdcType("6", &param);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(param.hdcType, HDC_SERVICE_TYPE_RDMA);

    ResetParam(&param);
    ret = HccpParseHdcType("18", &param);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(param.hdcType, HDC_SERVICE_TYPE_RDMA_V2);

    ResetParam(&param);
    ret = HccpParseHdcType("99", &param);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(param.hdcType, HDC_SERVICE_TYPE_RDMA);

    ResetParam(&param);
    ret = HccpParseHdcType("abc", &param);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(param.hdcType, HDC_SERVICE_TYPE_RDMA);

    ResetParam(&param);
    ret = HccpParseWhiteListStatus("1", &param);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(param.whiteListStatus, WHITE_LIST_ENABLE);

    ResetParam(&param);
    ret = HccpParseWhiteListStatus("0", &param);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(param.whiteListStatus, WHITE_LIST_DISABLE);

    ResetParam(&param);
    ret = HccpParseWhiteListStatus("abc", &param);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(param.whiteListStatus, WHITE_LIST_ENABLE);

    ResetParam(&param);
    ret = HccpParseBackupPhyid("100", &param);
    EXPECT_INT_EQ(ret, 0);

    ResetParam(&param);
    ret = HccpParseBackupPhyid("abc", &param);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(param.backupFlag, false);

    char* argv1[] = {"hccp", NULL};
    ResetParam(&param);
    ResetOptind();
    ret = HccpParamParse(1, argv1, &param);
    EXPECT_INT_NE(ret, 0);

    char* argv2[] = {"hccp", "--deviceId", "0", NULL};
    ResetParam(&param);
    ResetOptind();
    ret = HccpParamParse(3, argv2, &param);
    EXPECT_INT_NE(ret, 0);

    char* argv3[] = {"hccp", "--deviceId", "0", "--pid", "100", NULL};
    ResetParam(&param);
    ResetOptind();
    ret = HccpParamParse(5, argv3, &param);
    EXPECT_INT_NE(ret, 0);

    char* argv4[] = {"hccp", "--pid",         "100", "--logLevel", "3", "--hdcType", "6", "--whiteListStatus",
                     "1",    "--backupPhyId", "0",   NULL};
    ResetParam(&param);
    ResetOptind();
    ret = HccpParamParse(11, argv4, &param);
    EXPECT_INT_NE(ret, 0);

    char* argv5[] = {"hccp", "--deviceId",    "0", "--pid",     "100", "--pidSign",
                     "sign", "--logLevel",    "3", "--hdcType", "6",   "--whiteListStatus",
                     "1",    "--backupPhyId", "0", NULL};
    ResetParam(&param);
    ResetOptind();
    ret = HccpParamParse(15, argv5, &param);
    EXPECT_INT_NE(ret, 0);

    return;
}

void TcAbnormal()
{
    struct HccpInitParam param;
    int ret;

    ResetParam(&param);
    ret = HccpParseLogicId("0", &param);
    EXPECT_INT_NE(ret, 0);

    ResetParam(&param);
    ret = HccpParseLogicId("abc", &param);
    EXPECT_INT_NE(ret, 0);

    ResetParam(&param);
    ret = HccpParsePid("", &param);
    EXPECT_INT_NE(ret, 0);

    ResetParam(&param);
    ret = HccpParseLogLevel("", &param);
    EXPECT_INT_NE(ret, 0);

    ResetParam(&param);
    ret = HccpParseHdcType("", &param);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(param.hdcType, HDC_SERVICE_TYPE_RDMA);

    ResetParam(&param);
    ret = HccpParseWhiteListStatus("", &param);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(param.whiteListStatus, WHITE_LIST_ENABLE);

    ResetParam(&param);
    ret = HccpParseBackupPhyid("", &param);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(param.backupFlag, false);

    char* argv1[] = {"hccp", "--unknown", "value", NULL};
    ResetParam(&param);
    ResetOptind();
    ret = HccpParamParse(3, argv1, &param);
    EXPECT_INT_NE(ret, 0);

    char* argv2[] = {"hccp", "--pid", "abc", "--deviceId", "0", "--pidSign", "s", NULL};
    ResetParam(&param);
    ResetOptind();
    ret = HccpParamParse(7, argv2, &param);
    EXPECT_INT_NE(ret, 0);

    char* argv3[] = {"hccp", "--pid", "100", "--deviceId", "abc", "--pidSign", "s", NULL};
    ResetParam(&param);
    ResetOptind();
    ret = HccpParamParse(7, argv3, &param);
    EXPECT_INT_NE(ret, 0);

    return;
}
