/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unistd.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <errno.h>
#include "tsd/tsd.h"
#ifndef CONFIG_HCCP_LLT
#include "dlog_pub.h"
#endif
#include "user_log.h"
#include "param.h"
#include "ra_adp.h"
#include "securec.h"
#include "dl_hal_function.h"

typedef void (*SighandlerT)(int);

STATIC int HccpExecCmd(const char *cmd)
{
    SighandlerT oldHandler = signal(SIGCHLD, SIG_DFL);
    int ret;

    ret = system(cmd);
    (void)signal(SIGCHLD, oldHandler);
    CHK_PRT_RETURN(ret == -1 || ret == HCCP_KEY_EXPIRED, hccp_err("exec failed, ret[%d], errno[%d]", ret, errno), -1);
    return 0;
}

STATIC int HccpAddToCgroup(int logicId)
{
    const char *addMemoryFile = "/var/add_to_cgroup_usermemory.sh";
    const char *addCtrlCpuFile = "/var/add_to_cgroup_ctrlcpu.sh";
    char cmd[HCCP_CMD_MAX_LEN] = {0};
    enum ProductType productType;
    int ret;

    // Cache result after first query, skip exception checking for subsequent queries
    productType = RsGetProductType(logicId);
    CHK_PRT_RETURN(productType == PRODUCT_TYPE_INVALID, hccp_err("RsGetProductType failed, logicId:%d", logicId),
        -EINVAL);

    // add user memory cgroup
    if ((productType == PRODUCT_TYPE_910 || productType == PRODUCT_TYPE_310p) && access(addMemoryFile, F_OK) == 0) {
        ret = snprintf_s(cmd, HCCP_CMD_MAX_LEN, HCCP_CMD_MAX_LEN - 1, "sudo %s hccp_service.bin", addMemoryFile);
        CHK_PRT_RETURN(ret <= 0, hccp_err("snprintf_s for cmd failed, ret:%d", ret), -EINVAL);
        ret = HccpExecCmd(cmd);
        CHK_PRT_RETURN(ret != 0, hccp_err("HccpExecCmd add usermemory cgroup failed, ret:%d", ret), ret);
    }

    // add ctrlcpu cgroup explicitly in VF scenario: check file due to compatibility issue
    if (productType == PRODUCT_TYPE_910_93 && access(addCtrlCpuFile, F_OK) == 0) {
        ret = snprintf_s(cmd, HCCP_CMD_MAX_LEN, HCCP_CMD_MAX_LEN - 1, "sudo %s %d", addCtrlCpuFile, getpid());
        CHK_PRT_RETURN(ret <= 0, hccp_err("snprintf_s for cmd failed, ret:%d", ret), -EINVAL);
        ret = HccpExecCmd(cmd);
        // ignore return value if add ctrlcpu cgroup unsuccessful, will result in HccpSetAffinity
        CHK_PRT_RETURN(ret != 0, hccp_run_warn("HccpExecCmd add ctrlcpu cgroup unsuccessful, ret:%d", ret), 0);
    }

    return 0;
}

STATIC int HccpChangeNumOfFile(void)
{
    struct rlimit limit = {0, 0};
    int ret;

    ret = getrlimit(RLIMIT_NOFILE, &limit);
    CHK_PRT_RETURN(ret, hccp_err("getrlimit failed, ret = %d, errno = %d\n", ret, errno), ret);

    limit.rlim_cur = limit.rlim_max;
    ret = setrlimit(RLIMIT_NOFILE, &limit);
    CHK_PRT_RETURN(ret, hccp_err("setrlimit failed, ret = %d, errno = %d\n", ret, errno), ret);

    return 0;
}

STATIC int HccpSetLogInfo(struct HccpInitParam *param)
{
#define HUNDREDS_DIGIT 100
    int ret;
    int enableEvent = param->logLevel / HUNDREDS_DIGIT;
    int level = param->logLevel % HUNDREDS_DIGIT;
#ifndef CONFIG_HCCP_LLT
    LogAttr logattr = {0};

    logattr.type = APPLICATION;
    logattr.pid = param->pid;
    logattr.deviceId = param->backupFlag ? param->backupChipId : param->chipId;
    ret = dlog_setlevel(-1, level, enableEvent);
    CHK_PRT_RETURN(ret,
        hccp_err("hccp set log level failed, ret:%d, log level:%d, enableEvent:%d", ret, level, enableEvent), ret);

    ret = DlogSetAttr(logattr);
    CHK_PRT_RETURN(ret,
        hccp_err("hccp set attr chip_id:%u, backupFlag:%d, backupChipId:%u failed, ret:%d", param->chipId,
            param->backupFlag, param->backupChipId, ret),
        ret);
#endif
    return 0;
}

#ifndef CONFIG_HCCP_LLT
int main(int argc, char *argv[])
#else
int llt_main(int argc, char *argv[])
#endif
{
    struct HccpInitParam param = {0};
    struct timeval start, end;
    float timeCost = 0.0;
    int ret;

    hccp_run_info("hccp init start!");
    ret = HccpChangeNumOfFile();
    CHK_PRT_RETURN(ret, hccp_err("hccp change limit of nofile failed, ret = %d", ret), ret);

    // use default logicId:0 to get product type
    ret = HccpAddToCgroup(0);
    CHK_PRT_RETURN(ret != 0, hccp_err("HccpAddToCgroup error[%d] logicId[%d]", ret, 0), ret);

    ret = DlHalInit();
    CHK_PRT_RETURN(ret, hccp_err("dl_hal_init error[%d]", ret), ret);

    ret = HccpParamParse(argc, argv, &param);
    if (ret != 0) {
        hccp_err("hccp_param_parse error[%d]", ret);
        goto out;
    }

    ret = HccpSetLogInfo(&param);
    if (ret != 0) {
        hccp_err("hccp_set_log_info error[%d]", ret);
        goto out;
    }

    RsGetCurTime(&start);

    ret = HccpInit(param.chipId, param.pid, param.hdcType, param.whiteListStatus);
    if (ret) {
        hccp_err("hccp init error[%d]", ret);
        goto hccp_init_fail;
    }

    RsGetCurTime(&end);
    HccpTimeInterval(&end, &start, &timeCost);
    hccp_run_info("hccp init ok cost [%f] ms logic_id[%d], tgid[%d]", timeCost, param.logicId, param.pid);

    ret = SendStartUpFinishMsg((uint32_t)param.logicId, 0, (uint32_t)param.pid, 0);
    if (ret) {
        hccp_err("SendStartUpFinishMsg error[%d]", ret);
    }

    ret = HccpDeinit(param.chipId);
    if (ret) {
        hccp_err("hccp deinit error[%d]", ret);
        goto hccp_init_fail;
    }

    hccp_run_info("hccp deinit ok! logic_id=%d", param.logicId);

hccp_init_fail:
out:
    DlHalDeinit();
    return ret;
}
