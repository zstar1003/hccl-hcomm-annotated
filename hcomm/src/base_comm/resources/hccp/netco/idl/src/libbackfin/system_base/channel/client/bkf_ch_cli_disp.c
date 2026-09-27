/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_ch_cli_pri.h"
#include "bkf_ch_cli_adef.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

void BkfChCliMngDispCli(BkfChCliType *chCliType, BkfDisp *disp)
{
    BKF_RETURNvoid_IF(chCliType == VOS_NULL);
    BKF_RETURNvoid_IF(disp == VOS_NULL);

    BKF_DISP_PRINTF(disp, "Cli Type Info Name :%s, type: %u, chCli(%#x)\n", chCliType->vTbl.name,
        chCliType->vTbl.typeId, BKF_MASK_ADDR(chCliType->ch));
    BKF_DISP_PRINTF(disp, "Init(%#x), UnInit(%#x)\n", BKF_MASK_ADDR(chCliType->vTbl.init),
        BKF_MASK_ADDR(chCliType->vTbl.uninit));
    BKF_DISP_PRINTF(disp, "SelfCid(%#x), SelfUrl(%#x)\n", BKF_MASK_ADDR(chCliType->vTbl.setSelfCid),
        BKF_MASK_ADDR(chCliType->vTbl.setSelfUrl));
    BKF_DISP_PRINTF(disp, "Conn(%#x), DisConn(%#x)\n", BKF_MASK_ADDR(chCliType->vTbl.conn),
        BKF_MASK_ADDR(chCliType->vTbl.disConn));
    BKF_DISP_PRINTF(disp, "ConnTimeOut(%#x), enable(%#x)\n", BKF_MASK_ADDR(chCliType->vTbl.connTimeOut),
        BKF_MASK_ADDR(chCliType->vTbl.enable));
    BKF_DISP_PRINTF(disp, "MallocData(%#x), FreeData(%#x)\n", BKF_MASK_ADDR(chCliType->vTbl.mallocDataBuf),
        BKF_MASK_ADDR(chCliType->vTbl.freeDataBuf));
    BKF_DISP_PRINTF(disp, "ReadData(%#x), WriteData(%#x)\n", BKF_MASK_ADDR(chCliType->vTbl.read),
        BKF_MASK_ADDR(chCliType->vTbl.write));

    return;
}

typedef struct tagBkfChDispCliTypeCtx {
    int32_t chCliCnt;
    uint8_t typeId;
} BkfChDispCliTypeCtx;

void BkfChCliMngDispArgInfo(BkfChCliMng *chCliMng)
{
    BkfDisp *disp = VOS_NULL;
    disp = chCliMng->argInit.disp;

    BKF_DISP_PRINTF(disp, "Channel Client Name: %s, IsEnable: %u\n", chCliMng->name, chCliMng->hasEnable);
    BKF_DISP_PRINTF(disp, "-----------------------------------\n");
    BKF_DISP_PRINTF(disp, "ArgInit Parameter:\n");
    BKF_DISP_PRINTF(disp, "Name: %s, dbgFlag: %u\n", chCliMng->argInit.name, chCliMng->argInit.dbgOn);
    BKF_DISP_PRINTF(disp, "MemMng(%#x) disp(%#x)\n", BKF_MASK_ADDR(chCliMng->argInit.memMng),
        BKF_MASK_ADDR(chCliMng->argInit.disp));
    BKF_DISP_PRINTF(disp, "log(%#x), sysLog(%#x)\n", BKF_MASK_ADDR(chCliMng->argInit.log),
        BKF_MASK_ADDR(chCliMng->argInit.sysLogMng));
    BKF_DISP_PRINTF(disp, "logCnt(%#x), xMap(%#x)\n", BKF_MASK_ADDR(chCliMng->argInit.logCnt),
        BKF_MASK_ADDR(chCliMng->argInit.xMap));
    BKF_DISP_PRINTF(disp, "tmrMng(%#x), jobMng(%#x)\n", BKF_MASK_ADDR(chCliMng->argInit.tmrMng),
        BKF_MASK_ADDR(chCliMng->argInit.jobMng));
    BKF_DISP_PRINTF(disp, "jobType: %u, JobPri: %u\n", chCliMng->argInit.jobTypeId, chCliMng->argInit.jobPrio);
    BKF_DISP_PRINTF(disp, "dmsQueId: %u, dmsIntfId: %u, dmsSubIntfId: %u\n", chCliMng->argInit.dmsCliQueId,
        chCliMng->argInit.dmsIntfId, chCliMng->argInit.dmsSubIntfId);

    BKF_DISP_PRINTF(disp, "-----------------------------------\n");
    BKF_DISP_PRINTF(disp, "Enable FuncTable:\n");
    BKF_DISP_PRINTF(disp, "Cookies(%#x)\n", BKF_MASK_ADDR(chCliMng->argEnable.cookie));
    BKF_DISP_PRINTF(disp, "OnRcvData(%#x), OnRcvDataEvt(%#x)\n", BKF_MASK_ADDR(chCliMng->argEnable.onRcvData),
        BKF_MASK_ADDR(chCliMng->argEnable.onRcvDataEvent));
    BKF_DISP_PRINTF(disp, "onUnblock(%#x), onDisconn(%#x)\n", BKF_MASK_ADDR(chCliMng->argEnable.onUnblock),
        BKF_MASK_ADDR(chCliMng->argEnable.onDisconn));
    BKF_DISP_PRINTF(disp, "-----------------------------------\n");
    return;
}

void BkfChCliMngDisp(BkfChCliMng *chCliMng)
{
    BkfDisp *disp = VOS_NULL;
    BkfChCliType *chCliType = VOS_NULL;

    BkfChDispCliTypeCtx *lastCliTypeCtx = VOS_NULL;
    BkfChDispCliTypeCtx curCliTypeCtx = {0};

    BKF_RETURNvoid_IF(chCliMng == VOS_NULL);

    disp = chCliMng->argInit.disp;

    lastCliTypeCtx = BKF_DISP_GET_LAST_CTX(disp, VOS_NULL);
    if (lastCliTypeCtx == VOS_NULL) {
        BkfChCliMngDispArgInfo(chCliMng);
        chCliType = BkfChCliGetFirstType(chCliMng, VOS_NULL);
    } else {
        curCliTypeCtx = *lastCliTypeCtx;
        chCliType = BkfChCliFindNextType(chCliMng, curCliTypeCtx.typeId);
    }

    if (chCliType != VOS_NULL) {
        BkfChCliMngDispCli(chCliType, disp);
        curCliTypeCtx.chCliCnt++;
        curCliTypeCtx.typeId = chCliType->vTbl.typeId;
        BKF_DISP_SAVE_CTX(disp, &curCliTypeCtx, sizeof(curCliTypeCtx), VOS_NULL, 0);
    } else {
        BKF_DISP_PRINTF(disp, "ChCli Count: %u\n", curCliTypeCtx.chCliCnt);
    }
    return;
}

void BkfChCliDispInit(BkfChCliMng *chCliMng)
{
    BkfDisp *disp = chCliMng->argInit.disp;
    char *objName = (char *)chCliMng->name;
    uint32_t ret;

    ret = BkfDispRegObj(disp, objName, chCliMng);
    if (ret != BKF_OK) {
        BKF_ASSERT(0);
        return;
    }

    (void)BKF_DISP_REG_FUNC(disp, BkfChCliMngDisp, "disp channel for cliMng", objName, 0);
}

void BkfChCliDispUnInit(BkfChCliMng *chCliMng)
{
    BkfDispUnregObj(chCliMng->argInit.disp, chCliMng->name);
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
