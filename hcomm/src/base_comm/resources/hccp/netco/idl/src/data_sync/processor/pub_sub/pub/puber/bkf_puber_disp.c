/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_puber_disp.h"
#include "bkf_puber_data.h"
#include "vos_base.h"

#ifdef __cplusplus
extern "C" {
#endif

STATIC void BkfPuberDispSelfInfo(BkfPuber *puber)
{
    BkfDisp *disp = puber->argInit.disp;
    BKF_DISP_PRINTF(disp, "%s\n", puber->name);
    BKF_DISP_PRINTF(disp, "svcName(%s/)\n", puber->svcName);
    BKF_DISP_PRINTF(disp, "idlVersionMajor/Minor(%u, %u)\n", puber->argInit.idlVersionMajor,
        puber->argInit.idlVersionMinor);
    BKF_DISP_PRINTF(disp, "dbgOn(%u)\n", puber->argInit.dbgOn);
    BKF_DISP_PRINTF(disp, "memMng(%#x)\n", BKF_MASK_ADDR(puber->argInit.memMng));
    BKF_DISP_PRINTF(disp, "disp(%#x)\n", BKF_MASK_ADDR(puber->argInit.disp));
    BKF_DISP_PRINTF(disp, "log(%#x)\n", BKF_MASK_ADDR(puber->argInit.log));
    BKF_DISP_PRINTF(disp, "xMap(%#x)\n", BKF_MASK_ADDR(puber->argInit.xMap));
    BKF_DISP_PRINTF(disp, "tmrMng(%#x)\n", BKF_MASK_ADDR(puber->argInit.tmrMng));
    BKF_DISP_PRINTF(disp, "jobMng(%#x)/jobTypeId(%u)/jobPrio(%u)\n", BKF_MASK_ADDR(puber->argInit.jobMng),
        puber->argInit.jobTypeId, puber->argInit.jobPrio);
    BKF_DISP_PRINTF(disp, "cookie(%#x)\n", BKF_MASK_ADDR(puber->argInit.cookie));
    BKF_DISP_PRINTF(disp, "verifyMayAccelerate(%#x)\n", BKF_MASK_ADDR(puber->argInit.verifyMayAccelerate));
    BKF_DISP_PRINTF(disp, "sysLogMng(%#x)\n", BKF_MASK_ADDR(puber->argInit.sysLogMng));
    BKF_DISP_PRINTF(disp, "dc(%#x)\n", BKF_MASK_ADDR(puber->argInit.dc));
    BKF_DISP_PRINTF(disp, "connCntMax(%d)\n", puber->argInit.connCntMax);
    BKF_DISP_PRINTF(disp, "xSeed(%" VOS_PRIu64 ")\n", puber->argInit.xSeed);
    BKF_DISP_PRINTF(disp, "tableTypeMng(%#x)\n", BKF_MASK_ADDR(puber->tableTypeMng));
    BKF_DISP_PRINTF(disp, "connMng(%#x)\n", BKF_MASK_ADDR(puber->connMng));
}

void BkfPuberDisp(BkfPuber *puber)
{
    BkfDisp *disp = puber->argInit.disp;
    BKF_DISP_PRINTF(disp, "====self info====\n");
    BkfPuberDispSelfInfo(puber);

    BKF_DISP_PRINTF(disp, "====child info====\n");
    BkfPuberTableTypeDispSummary(puber->tableTypeMng, disp);
    BkfPuberConnDispSummary(puber->connMng, disp);
}

void BkfPuberDispInit(BkfPuber *puber)
{
    BkfDisp *disp = puber->argInit.disp;
    char *objName = puber->name;
    uint32_t ret = BkfDispRegObj(disp, objName, puber);
    if (ret != BKF_OK) {
        BKF_ASSERT(0);
        return;
    }

    (void)BKF_DISP_REG_FUNC(disp, BkfPuberDisp, "disp puber", objName, 0);
    puber->dispInitOk = VOS_TRUE;
}

void BkfPuberDispUninit(BkfPuber *puber)
{
    if (puber->dispInitOk) {
        BkfDispUnregObj(puber->argInit.disp, puber->name);
    }
}

#ifdef __cplusplus
}
#endif
