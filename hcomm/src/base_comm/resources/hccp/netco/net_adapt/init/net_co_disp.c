/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "net_co_disp.h"
#include "net_co_main_data.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif
void NetCoDispSummary(NetCo *co)
{
    BkfDisp *disp = co->disp;
    uint8_t buf[BKF_1K];

    BKF_DISP_PRINTF(disp, "===argInit===\n");
    BKF_DISP_PRINTF(disp, "epFd(%d)\n", co->argInit.epFd);
    BKF_DISP_PRINTF(disp, "inConnSelfUrl(%s)\n", BkfUrlGetStr(&co->argInit.inConnSelfUrl, buf, sizeof(buf)));
    BKF_DISP_PRINTF(disp, "inConnNetUrl(%s)\n", BkfUrlGetStr(&co->argInit.inConnNetUrl, buf, sizeof(buf)));
    BKF_DISP_PRINTF(disp, "outConnLsnUrl(%s)\n", BkfUrlGetStr(&co->argInit.outConnLsnUrl, buf, sizeof(buf)));
    BKF_DISP_PRINTF(disp, "===runtime info===\n");
    BKF_DISP_PRINTF(disp, "name(%s)\n", co->name);
    BKF_DISP_PRINTF(disp, "dbgOn(%u)\n", co->dbgOn);
    BKF_DISP_PRINTF(disp, "selfCid(%u)\n", co->selfCid);
    BKF_DISP_PRINTF(disp, "memMng(%#x)\n", BKF_MASK_ADDR(co->memMng));
    BKF_DISP_PRINTF(disp, "xMap(%#x)\n", BKF_MASK_ADDR(co->xMap));
    BKF_DISP_PRINTF(disp, "disp(%#x)\n", BKF_MASK_ADDR(co->disp));
    BKF_DISP_PRINTF(disp, "logCnt(%#x)\n", BKF_MASK_ADDR(co->logCnt));
    BKF_DISP_PRINTF(disp, "log(%#x)\n", BKF_MASK_ADDR(co->log));
    BKF_DISP_PRINTF(disp, "logFilePathName(%s)\n", co->logFilePathName);
    BKF_DISP_PRINTF(disp, "pfmOn(%u)\n", co->pfmOn);
    BKF_DISP_PRINTF(disp, "pfm(%#x)\n", BKF_MASK_ADDR(co->pfm));
    BKF_DISP_PRINTF(disp, "mux(%#x)/muxAdpee(%#x)\n", BKF_MASK_ADDR(co->mux), BKF_MASK_ADDR(co->muxAdpee));
    BKF_DISP_PRINTF(disp, "tmrMng(%#x)/tmrMngAdpee(%#x)\n", BKF_MASK_ADDR(co->tmrMng), BKF_MASK_ADDR(co->tmrMngAdpee));
    BKF_DISP_PRINTF(disp, "tmrIdDispOut(%#x)\n", BKF_MASK_ADDR(co->tmrIdDispOut));
    BKF_DISP_PRINTF(disp, "jobMng(%#x)/jobMngAdpee(%#x)\n", BKF_MASK_ADDR(co->jobMng), BKF_MASK_ADDR(co->jobMngAdpee));
    BKF_DISP_PRINTF(disp, "sysLogMng(%#x)\n", BKF_MASK_ADDR(co->sysLogMng));
    BKF_DISP_PRINTF(disp, "in(%#x)\n", BKF_MASK_ADDR(co->in));
    BKF_DISP_PRINTF(disp, "out(%#x)\n", BKF_MASK_ADDR(co->out));
}

uint32_t NetCoDispInit(NetCo *co)
{
    char name[BKF_NAME_LEN_MAX + 1];
    int err = snprintf_truncated_s(name, sizeof(name), "%s_disp", co->name);
    if (err <= 0) {
        return BKF_ERR;
    }
    BkfDispInitArg arg = {
        .name = name,
        .memMng = co->memMng,
        .mayProcFuncLvl = co->dbgOn ? BKF_DISP_FUNC_LVL_TEST : BKF_DISP_FUNC_LVL_DEFAULT,
    };
    co->disp = BkfDispInit(&arg);
    if (co->disp == VOS_NULL) {
        return BKF_ERR;
    }

    BkfDisp *disp = co->disp;
    char *objName = co->name;
    (void)BkfDispRegObj(disp, objName, co);
    (void)BKF_DISP_REG_FUNC(disp, NetCoDispSummary, "disp summary", objName, 0);
    return BKF_OK;
}

void NetCoDispUninit(NetCo *co)
{
    if (co->disp != VOS_NULL) {
        BkfDispUninit(co->disp);
        co->disp = VOS_NULL;
    }
}

#ifdef __cplusplus
}
#endif
