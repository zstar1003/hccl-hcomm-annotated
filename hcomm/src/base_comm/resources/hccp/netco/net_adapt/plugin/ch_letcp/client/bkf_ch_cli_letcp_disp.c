/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_ch_cli_letcp_data.h"

void BkfChCliLetcpDisp(BkfChCli *chCli)
{
    BkfDisp *disp = VOS_NULL;
    uint8_t buf[BKF_1K / 8];
    uint32_t cnt = 0;
    BkfChCliConnId *connId = VOS_NULL;

    BKF_RETURNvoid_IF(chCli == VOS_NULL);

    disp = chCli->argInit.base->disp;

    BKF_DISP_PRINTF(disp, "Channel Client Name: %s\n", chCli->name);
    BKF_DISP_PRINTF(disp, "IsEnable: %u, SelfCidHasSet: %u\n", chCli->hasEnable, chCli->selfCidHasSet);
    BKF_DISP_PRINTF(disp, "SelfUrl: %s\n", BkfUrlGetStr(&chCli->selfUrl, buf, sizeof(buf)));
    BKF_DISP_PRINTF(disp, "-----------------------------------\n");
    BKF_DISP_PRINTF(disp, "Arg Parameter:\n");
    BKF_DISP_PRINTF(disp, "Name: %s, dbgFlag: %u\n", chCli->argInit.base->name, chCli->argInit.base->dbgOn);
    BKF_DISP_PRINTF(disp, "MemMng(%#x) disp(%#x)\n", BKF_MASK_ADDR(chCli->argInit.base->memMng),
        BKF_MASK_ADDR(chCli->argInit.base->disp));
    BKF_DISP_PRINTF(disp, "log(%#x), sysLog(%#x)\n", BKF_MASK_ADDR(chCli->argInit.base->log),
        BKF_MASK_ADDR(chCli->argInit.base->sysLogMng));
    BKF_DISP_PRINTF(disp, "logCnt(%#x)\n", BKF_MASK_ADDR(chCli->argInit.base->logCnt));
    BKF_DISP_PRINTF(disp, "tmrMng(%#x), jobMng(%#x)\n", BKF_MASK_ADDR(chCli->argInit.base->tmrMng),
        BKF_MASK_ADDR(chCli->argInit.base->jobMng));
    BKF_DISP_PRINTF(disp, "jobType: %u, JobPri: %u\n", chCli->argInit.base->jobTypeId, chCli->argInit.base->jobPrio);

    BKF_DISP_PRINTF(disp, "-----------------------------------\n");
    BKF_DISP_PRINTF(disp, "Enable FuncTable:\n");
    BKF_DISP_PRINTF(disp, "Cookies(%#x)\n", BKF_MASK_ADDR(chCli->argEnable.cookie));
    BKF_DISP_PRINTF(disp, "OnRcvData(%#x), OnRcvDataEvt(%#x)\n", BKF_MASK_ADDR(chCli->argEnable.onRcvData),
        BKF_MASK_ADDR(chCli->argEnable.onRcvDataEvent));
    BKF_DISP_PRINTF(disp, "onUnblock(%#x), onDisconn(%#x)\n", BKF_MASK_ADDR(chCli->argEnable.onUnblock),
        BKF_MASK_ADDR(chCli->argEnable.onDisconn));
    connId = VOS_AVLL_FIRST(chCli->connSet);
    while (connId != VOS_NULL) {
        cnt++;
        connId = VOS_AVLL_NEXT(chCli->connSet, connId->avlNode);
    }
    BKF_DISP_PRINTF(disp, "ConnId Count: %u\n", cnt);

    return;
}

typedef struct tagBkfChDispConnIdCtx {
    int32_t connIdCnt;
    BkfUrl urlSer;
    BkfUrl urlCli;
} BkfChDispConnIdCtx;

void BkfChCliLetcpDispOneConnId(BkfChCli *chCli, BkfChCliConnId *connId, int32_t idx)
{
    BkfDisp *disp = VOS_NULL;
    uint8_t buf[BKF_1K / 8];

    disp = chCli->argInit.base->disp;
    BKF_DISP_PRINTF(disp, "++++connId[%d]============\n", idx);
    BKF_DISP_PRINTF(disp, "sign(%#x)\n", connId->sign);
    BKF_DISP_PRINTF(disp, "connFd(%d)\n", connId->connFd);
    BKF_DISP_PRINTF(disp, "keyUrlSer(%s)\n", BkfUrlGetStr(&connId->keyUrlSer, buf, sizeof(buf)));
    BKF_DISP_PRINTF(disp, "keyUrlCli(%s)\n", BkfUrlGetStr(&connId->keyUrlCli, buf, sizeof(buf)));
    BKF_DISP_PRINTF(disp, "tmrIdWriteErr(%#x)\n", BKF_MASK_ADDR(connId->tmrIdWriteErr));
    BKF_DISP_PRINTF(disp, "connFdAttachOk(%u)\n", connId->connFdAttachOk);
    BKF_DISP_PRINTF(disp, "lockDel(%u)\n", connId->lockDel);
    BKF_DISP_PRINTF(disp, "softDel(%u)\n", connId->softDel);
    BKF_DISP_PRINTF(disp, "tcpstate(%u)\n", connId->tcpState);
    return;
}

BkfChCliConnId *BkfChCliLetcpDispGetCtx(BkfChCli *chCli, BkfChDispConnIdCtx *curConnCtx)
{
    BkfChCliConnId *connId = VOS_NULL;
    BkfChDispConnIdCtx *lastConnCtx = VOS_NULL;
    BkfDisp *disp = chCli->argInit.base->disp;

    lastConnCtx = BKF_DISP_GET_LAST_CTX(disp, VOS_NULL);
    if (lastConnCtx == VOS_NULL) {
        connId = BkfChCliLetcpGetFirstConnId(chCli, VOS_NULL);
    } else {
        *curConnCtx = *lastConnCtx;
        connId = BkfChCliLetcpFindNextConnId(chCli, &curConnCtx->urlSer, &curConnCtx->urlCli);
    }
    return connId;
}

void BkfChCliLetcpDispConnId(BkfChCli *chCli)
{
    BkfDisp *disp = VOS_NULL;
    BkfChCliConnId *connId = VOS_NULL;
    BkfChDispConnIdCtx curConnCtx = {0};

    BKF_RETURNvoid_IF(chCli == VOS_NULL);

    disp = chCli->argInit.base->disp;

    connId = BkfChCliLetcpDispGetCtx(chCli, &curConnCtx);
    if (connId != VOS_NULL) {
        BkfChCliLetcpDispOneConnId(chCli, connId, curConnCtx.connIdCnt);

        curConnCtx.connIdCnt++;
        curConnCtx.urlSer = connId->keyUrlSer;
        curConnCtx.urlCli = connId->keyUrlCli;
        BKF_DISP_SAVE_CTX(disp, &curConnCtx, sizeof(curConnCtx), VOS_NULL, 0);
    } else {
        BKF_DISP_PRINTF(disp, " ***total %d conn(s) ***\n", curConnCtx.connIdCnt);
    }
    return;
}

void BkfChCliLetcpDispSocketStat(BkfChCli *chCli)
{
    BkfDisp *disp = VOS_NULL;
    BkfChCliConnId *connId = VOS_NULL;
    BkfChDispConnIdCtx curConnCtx = {0};

    BKF_RETURNvoid_IF(chCli == VOS_NULL);

    disp = chCli->argInit.base->disp;

    connId = BkfChCliLetcpDispGetCtx(chCli, &curConnCtx);
    if (connId != VOS_NULL) {
        curConnCtx.connIdCnt++;
        curConnCtx.urlSer = connId->keyUrlSer;
        BKF_DISP_SAVE_CTX(disp, &curConnCtx, sizeof(curConnCtx), VOS_NULL, 0);
    } else {
        BKF_DISP_PRINTF(disp, " ***total %d conn(s) ***\n", curConnCtx.connIdCnt);
    }
    return;
}

void BkfChCliLetcpDispInit(BkfChCli *ch)
{
    BkfDisp *disp = ch->argInit.base->disp;
    char *objName = (char *)ch->name;
    uint32_t ret;

    ret = BkfDispRegObj(disp, objName, ch);
    if (ret != BKF_OK) {
        BKF_ASSERT(0);
        return;
    }

    (void)BKF_DISP_REG_FUNC(disp, BkfChCliLetcpDisp, "disp channel linux tcp for cli", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfChCliLetcpDispConnId, "disp channel linux tcp for connId", objName, 0);
    return;
}
void BkfChCliLetcpDispUninit(BkfChCli *ch)
{
    BkfDispUnregObj(ch->argInit.base->disp, ch->name);
    return;
}