/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARAM_H
#define PARAM_H
#include <stdbool.h>

#if defined(HNS_ROCE_LLT) || defined(DEFINE_HNS_LLT)
#define STATIC
#else
#define STATIC static
#endif

#define HCCP_DEFAULT_REQUIRED_ARGC_NUM 3
#define DEVID_PREFIX "deviceId" /* logic id */
#define PID_PREFIX "pid"
#define PID_SIGN_PREFIX "pidSign"
#define LOG_LEVEL_PREFIX "logLevelInPid"
#define HDC_TYPE_PREFIX "hdcType"
#define WHITE_LIST_STATUS_PREFIX "whiteListStatus"
#define BACKUP_PHYID_PREFIX "backupPhyId"

#define HCCP_CMD_MAX_LEN 128
#define HCCP_KEY_EXPIRED 127

enum {
    HCCP_ARGC_DEV = 0,
    HCCP_ARGC_PID = 1,
    HCCP_ARGC_PID_SIGN = 2,
    HCCP_ARGC_LOG_LEVEL = 3,
    HCCP_ARGC_HDC_TYPE = 4,
    HCCP_ARGC_WHITE_LIST_STATUS = 5,
    HCCP_ARGC_BACKUP_PHYID = 6,
    HCCP_ARGC_NUM = 7,
};

struct HccpInitParam {
    int logicId;
    unsigned int chipId;
    int pid;
    int logLevel;
    int hdcType;
    unsigned int whiteListStatus;
    bool backupFlag;
    unsigned int backupChipId;
};

struct ParamHandle {
    int (*optHandle)(const char *, struct HccpInitParam *);
    bool isDefaultRequired;
    int optVal;
};

int HccpParamParse(int argc, char *argv[], struct HccpInitParam *param);
#endif // PARAM_H
