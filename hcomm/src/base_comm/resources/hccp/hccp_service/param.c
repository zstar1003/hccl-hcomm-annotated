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
#include <errno.h>
#include "user_log.h"
#include "config_log.h"
#include "securec.h"
#include "dl_hal_function.h"
#include "ra_adp.h"
#include "param.h"

STATIC int HccpParamParseId(const char *input, int *id)
{
    int ret;

    ret = sscanf_s(input, "%d", id);
    CHK_PRT_RETURN(ret != 1, hccp_err("get id from str failed, ret %d", ret), -EINVAL);

    return 0;
}

STATIC int HccpParseLogicId(const char *input, struct HccpInitParam *param)
{
    int ret;

    ret = HccpParamParseId(input, &param->logicId);
    CHK_PRT_RETURN(ret, hccp_err("hccp_param_parse_id failed"), ret);

    ret = DlDrvDeviceGetPhyIdByIndex(param->logicId, &(param->chipId));
    CHK_PRT_RETURN(ret != 0 || param->chipId > HCCP_MAX_CHIP_ID,
        hccp_err("get chip id failed, ret:%d, chipId:%u > %u", ret, param->chipId, HCCP_MAX_CHIP_ID), -EINVAL);

    hccp_info_others("logic_id from TSD is [%d], chipId[%u]", param->logicId, param->chipId);
    return 0;
}

STATIC int HccpParsePid(const char *input, struct HccpInitParam *param)
{
    int ret;

    ret = HccpParamParseId(input, &param->pid);
    CHK_PRT_RETURN(ret, hccp_err("hccp parse pid failed"), ret);

    CHK_PRT_RETURN(param->pid <= 0, hccp_err("pid:%d <= 0", param->pid), -EINVAL);
    hccp_info_others("pid from TSD is [%d]", param->pid);
    return 0;
}

STATIC int HccpParsePidSign(const char *input, struct HccpInitParam *param)
{
    (void)input;
    (void)param;
    // no need to parse pid sign, skip
    return 0;
}

STATIC int HccpParseLogLevel(const char *input, struct HccpInitParam *param)
{
    int ret;

    ret = HccpParamParseId(input, &param->logLevel);
    CHK_PRT_RETURN(ret, hccp_err("hccp parse log level failed, ret[%d]", ret), ret);
    hccp_info_others("log_level from TSD is [%d]", param->logLevel);
    return 0;
}

STATIC int HccpParseHdcType(const char *input, struct HccpInitParam *param)
{
    int ret;

    ret = HccpParamParseId(input, &param->hdcType);
    if (ret != 0 || (param->hdcType != HDC_SERVICE_TYPE_RDMA && param->hdcType != HDC_SERVICE_TYPE_RDMA_V2)) {
        hccp_warn_others("parse hdcType unsuccessful ret:%d or hdc_type[%d] invalid. set to default hdc_type[%d]", ret,
            param->hdcType, HDC_SERVICE_TYPE_RDMA);
        param->hdcType = HDC_SERVICE_TYPE_RDMA;
    }

    hccp_info_others("hdc_type from TSD is [%d]", param->hdcType);
    return 0;
}

STATIC int HccpParseWhiteListStatus(const char *input, struct HccpInitParam *param)
{
    int ret;

    ret = HccpParamParseId(input, (int *)&param->whiteListStatus);
    if (ret != 0) {
        param->whiteListStatus = WHITE_LIST_ENABLE;
        hccp_warn_others("parse whiteListStatus unsuccessful ret:%d. set to default [%u]", ret, param->whiteListStatus);
    }

    param->whiteListStatus = (param->whiteListStatus != 0) ? WHITE_LIST_ENABLE : WHITE_LIST_DISABLE;
    hccp_info_others("white_list_status from TSD is [%u]", param->whiteListStatus);
    return 0;
}

STATIC int HccpParseBackupPhyid(const char *input, struct HccpInitParam *param)
{
    unsigned int backupPhyId = 0;
    int ret;

    ret = HccpParamParseId(input, (int *)(void *)&backupPhyId);
    if (ret != 0) {
        hccp_warn_others("parse backup phy_id unsuccessful ret:%d", ret);
        return 0;
    }

    ret = DlDrvGetLocalDevIdByHostDevId(backupPhyId, &param->backupChipId);
    if (ret != 0) {
        hccp_warn_others("dl_drv_get_local_dev_id_by_host_dev_id unsuccessful ret:%d, backupPhyId:%u", ret,
            backupPhyId);
        return 0;
    }

    hccp_info_others("backup_phy_id from TSD is [%u], backupChipId[%u]", backupPhyId, param->backupChipId);
    param->backupFlag = true;
    return 0;
}

int HccpParamParse(int argc, char *argv[], struct HccpInitParam *param)
{
    static struct option longOpts[] = {
        {DEVID_PREFIX, required_argument, NULL, HCCP_ARGC_DEV},
        {PID_PREFIX, required_argument, NULL, HCCP_ARGC_PID},
        {PID_SIGN_PREFIX, required_argument, NULL, HCCP_ARGC_PID_SIGN},
        {LOG_LEVEL_PREFIX, required_argument, NULL, HCCP_ARGC_LOG_LEVEL},
        {HDC_TYPE_PREFIX, required_argument, NULL, HCCP_ARGC_HDC_TYPE},
        {WHITE_LIST_STATUS_PREFIX, required_argument, NULL, HCCP_ARGC_WHITE_LIST_STATUS},
        {BACKUP_PHYID_PREFIX, required_argument, NULL, HCCP_ARGC_BACKUP_PHYID},
        {0, no_argument, NULL, HCCP_ARGC_NUM},
    };
    static struct ParamHandle paramHandles[] = {
        {HccpParseLogicId, true, HCCP_ARGC_DEV},
        {HccpParsePid, true, HCCP_ARGC_PID},
        {HccpParsePidSign, false, HCCP_ARGC_PID_SIGN},
        {HccpParseLogLevel, true, HCCP_ARGC_LOG_LEVEL},
        {HccpParseHdcType, false, HCCP_ARGC_HDC_TYPE},
        {HccpParseWhiteListStatus, false, HCCP_ARGC_WHITE_LIST_STATUS},
        {HccpParseBackupPhyid, false, HCCP_ARGC_BACKUP_PHYID},
        {NULL, false, HCCP_ARGC_NUM},
    };
    const char *optstring = "";
    int requiredCnt = 0;
    int optIdx;
    int ret;

    // set default attr
    param->hdcType = HDC_SERVICE_TYPE_RDMA;
    param->whiteListStatus = WHITE_LIST_ENABLE;
    param->backupFlag = false;

    while (getopt_long(argc, argv, optstring, longOpts, &optIdx) != -1) {
        // unrecognized option, skip to parse
        if (optarg == NULL) {
            continue;
        }

        CHK_PRT_RETURN(optIdx < 0 || optIdx >= HCCP_ARGC_NUM || paramHandles[optIdx].optHandle == NULL,
            hccp_err("opt_idx:%d invalid, valid range[0, %d), errno:%d", optIdx, HCCP_ARGC_NUM, errno), -EINVAL);

        CHK_PRT_RETURN(paramHandles[optIdx].optVal != optIdx,
            hccp_err("opt_val:%d != opt_idx:%d", paramHandles[optIdx].optVal, optIdx), -EINVAL);

        ret = paramHandles[optIdx].optHandle(optarg, param);
        CHK_PRT_RETURN(ret != 0, hccp_err("parse param failed, optIdx:%d, ret:%d", optIdx, ret), ret);

        if (paramHandles[optIdx].isDefaultRequired) {
            requiredCnt++;
        }
    }

    CHK_PRT_RETURN(requiredCnt != HCCP_DEFAULT_REQUIRED_ARGC_NUM,
        hccp_err("param num error, requiredCnt:%d, argc:%d", requiredCnt, HCCP_DEFAULT_REQUIRED_ARGC_NUM), -EINVAL);
    return 0;
}
