/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((co)->log)
#define BKF_MOD_NAME ((co)->name)
#include "net_co_disp_out.h"
#include "net_co_main_data.h"
#include "net_co_log.h"
#include "v_stringlib.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif
#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
#define NET_CO_DISP_OUT_INTERVAL (10159)

STATIC uint32_t NetCoDispOutStartTmr(NetCo *co);
STATIC void NetCoDispOutStopTmr(NetCo *co);

#pragma pack()
#endif

#if BKF_BLOCK("表")
STATIC const char *g_NetCoDispOutRegCmdStr[] = {
    "BkfDispMemMng",
    "NetCoDispSummary",
    "NetCoInDispSummary",
    "NetCoOutDispSummary",
    "NetCoOutDispAllTbl",
};
#endif

#if BKF_BLOCK("公有函数定义")
uint32_t NetCoDispOutInit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");
    return (co->dbgOn ? NetCoDispOutStartTmr(co) : BKF_OK);
}

void NetCoDispOutUninit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "tmrIdDispOut(%#x)\n", BKF_MASK_ADDR(co->tmrIdDispOut));
    NetCoDispOutStopTmr(co);
}

void NetCoDispOutDisable(NetCo *co)
{
    BKF_LOG_DEBUG(BKF_LOG_HND, "tmrIdDispOut(%#x)\n", BKF_MASK_ADDR(co->tmrIdDispOut));
    NetCoDispOutStopTmr(co);
}

#if BKF_BLOCK("通过log，模拟disp输出")
#pragma pack(4)
#define NET_CO_DISP_OUT_CMD_STR_LEN_MAX (1023)
#define NET_CO_DISP_OUT_CMD_TOKEN_CNT_MAX (5 + 1 + 1) /* funcName + param(5) + objName(可选) */
#define NET_CO_DISP_OUT_STR_BUF_LEN_MAX (BKF_1K * 4)
typedef struct TagNetCoDispOutCmd {
    uint8_t cmdStrTokenCnt;
    char *cmdStrToken[NET_CO_DISP_OUT_CMD_TOKEN_CNT_MAX];
    char cmdStr[NET_CO_DISP_OUT_CMD_STR_LEN_MAX + 1];
} NetCoDispOutCmd;
#pragma pack()

STATIC uint32_t NetCoDispOutProcCmdParse(NetCo *co, char *cmdStr, NetCoDispOutCmd *cmd)
{
    int err = snprintf_truncated_s(cmd->cmdStr, sizeof(cmd->cmdStr), "%s", cmdStr);
    if (err <= 0) {
        return BKF_ERR;
    }
    char *temp = VOS_StrTrim(cmd->cmdStr);
    if (temp == VOS_NULL) {
        return BKF_ERR;
    }
    const char *delim = " ";
    char *itor;
    for (temp = strtok_s(temp, delim, &itor); temp != VOS_NULL; temp = strtok_s(VOS_NULL, delim, &itor)) {
        if (cmd->cmdStrTokenCnt >= NET_CO_DISP_OUT_CMD_TOKEN_CNT_MAX) {
            BKF_LOG_ERROR(BKF_LOG_HND,
                "token0(%s), hasParseTokenCnt(%u)/tokenCntMax(%u), temp(%#x, %s), has more token & ng\n",
                cmd->cmdStrToken[0], cmd->cmdStrTokenCnt, NET_CO_DISP_OUT_CMD_TOKEN_CNT_MAX, BKF_MASK_ADDR(temp), temp);
            return BKF_ERR;
        }
        cmd->cmdStrToken[cmd->cmdStrTokenCnt] = temp;
        cmd->cmdStrTokenCnt++;
    }

    return BKF_OK;
}
STATIC uint32_t NetCoDispOutProcCmdDo(NetCo *co, NetCoDispOutCmd *cmd)
{
    if (cmd->cmdStrTokenCnt == 0) {
        return BKF_ERR;
    }
    char *callFuncName = cmd->cmdStrToken[0];
    BKF_LOG_DEBUG(BKF_LOG_HND, "call ==> [%s]\n", callFuncName);

    char outStrBuf[NET_CO_DISP_OUT_STR_BUF_LEN_MAX];
    BkfDispProcCtx ctx = {0};
    for (;;) {
        outStrBuf[0] = '\0';
        uint32_t dispRet = BkfDispProc(co->disp, cmd->cmdStrToken, cmd->cmdStrTokenCnt, outStrBuf,
            sizeof(outStrBuf) - 1, &ctx);
        outStrBuf[NET_CO_DISP_OUT_STR_BUF_LEN_MAX - 1] = '\0'; /* 保护 */

        if (outStrBuf[0] != '\0') {
            /* 使用log文件输出 */
            NetCoLogOutput2File(co, (const char *)outStrBuf);
        }
        if (dispRet != BKF_DISP_PROC_HAS_MORE_DATA) {
            return dispRet;
        }
    }

    return BKF_OK;
}
void NetCoDispOutCmdFunc(NetCo *co, char *cmdStr)
{
    if ((co == VOS_NULL) || (cmdStr == VOS_NULL)) {
        return;
    }
    if (!co->dbgOn) {
        BKF_LOG_DEBUG(BKF_LOG_HND, "dbgOn(%u)\n", co->dbgOn);
        return;
    }

    NetCoDispOutCmd cmd = {0};
    uint32_t ret = NetCoDispOutProcCmdParse(co, cmdStr, &cmd);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return;
    }
    ret = NetCoDispOutProcCmdDo(co, &cmd);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return;
    }
}
#endif

void NetCoDispOutRegedCmd(NetCo *co)
{
    uint32_t i;
    for (i = 0; i < BKF_GET_ARY_COUNT(g_NetCoDispOutRegCmdStr); i++) {
        NetCoDispOutCmdFunc(co, (char *)g_NetCoDispOutRegCmdStr[i]);
    }
}
#endif

#if BKF_BLOCK("私有函数定义")
STATIC uint32_t NetCoDispOutOnTmrTO(NetCo *co, void *notUse)
{
    (void)notUse;
    NetCoDispOutRegedCmd(co);
    return BKF_OK;
}
STATIC uint32_t NetCoDispOutStartTmr(NetCo *co)
{
    if (co->tmrIdDispOut == VOS_NULL) {
        co->tmrIdDispOut = BkfTmrStartLoop(co->tmrMng, (F_BKF_TMR_TIMEOUT_PROC)NetCoDispOutOnTmrTO,
            NET_CO_DISP_OUT_INTERVAL, co);
        if (co->tmrIdDispOut == VOS_NULL) {
            BKF_LOG_ERROR(BKF_LOG_HND, "tmrIdDispOut(%#x), ng\n", BKF_MASK_ADDR(co->tmrIdDispOut));
            return BKF_ERR;
        }
    }

    return BKF_OK;
}

STATIC void NetCoDispOutStopTmr(NetCo *co)
{
    if (co->tmrIdDispOut != VOS_NULL) {
        BkfTmrStop(co->tmrMng, co->tmrIdDispOut);
        co->tmrIdDispOut = VOS_NULL;
    }
}
#endif

#ifdef __cplusplus
}
#endif
