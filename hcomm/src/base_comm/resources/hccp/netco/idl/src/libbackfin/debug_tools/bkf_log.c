/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_log.h"
#include "bkf_bufr.h"
#include "bkf_str.h"
#include "bkf_bas_type_mthd.h"
#include "bkf_assert.h"
#include "vos_systime.h"
#include "v_avll.h"
#include "v_stringlib.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif

#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
typedef struct tagBkfLogModLvl BkfLogModLvl;
struct tagBkfLog {
    BkfLogInitArg argInit; /* 放在第一个栏位 */
    char *name;
    BOOL outputEnable;
    F_BKF_DEBUG_OUTPUT debugOutputOrNull; /* debug输出函数 */
    uint8_t debugOutputEnable; /* debug使能判断函数, 一个比特位对应一个级别的开关，except, error,  warning , info
                                  ,默认全关 */
    uint8_t res[3];
    AVLL_TREE modLvlSet; /* 非默认，设置的值 */
    BkfLogModLvl *modLvlCache;
    uint32_t seqSeed;          /* 会溢出，没有关系 */
    AVLL_TREE blackBoxTypeSet; /* 单元log管理 */
    BkfBufr *memBuf;
};
BKF_STATIC_ASSERT(BKF_MBR_IS_FIRST(BkfLog, argInit));

struct tagBkfLogModLvl {
    AVLL_NODE avlNode;
    uint8_t lvl;
    char name[0];
};

/* data op */
STATIC BkfLogModLvl *BkfLogAddModLvl(BkfLog *log, char *modName, uint8_t lvl);
STATIC void BkfLogDelModLvl(BkfLog *log, BkfLogModLvl *modLvl);
STATIC void BkfLogDelAllModLvl(BkfLog *log);
STATIC BkfLogModLvl *BkfLogFindModLvl(BkfLog *log, char *modName);
STATIC BkfLogModLvl *BkfLogFindNextModLvl(BkfLog *log, char *modName);
STATIC BkfLogModLvl *BkfLogGetFirstModLvl(BkfLog *log, void **itorOutOrNull);
STATIC BkfLogModLvl *BkfLogGetNextModLvl(BkfLog *log, void **itorInOut);

/* disp */
STATIC void BkfLogDispInit(BkfLog *log);
STATIC void BkfLogDispUninit(BkfLog *log);
void BkfBlackBoxLogDispAll(BkfLog *log);
void BkfBlackBoxLogDispSpecType(BkfLog *log, uintptr_t blackboxType);

#define BKF_LOG_MEM_BUF_LEN_INIT (BKF_1K / 4)
#define BKF_LOG_MEM_BUF_LEN_DEFAULT (BKF_1K)

typedef struct { /* 模块启动时注册  */
    uint16_t logType;
    uint16_t uintLogBuffLen;
    AVLL_NODE avlNode;
    AVLL_TREE blackBoxInstSet;
} BkfBlackBoxType;

typedef struct {
    void *keyAddr; /* 记录日志实体的指针地址, key */
    uint32_t bid;
    AVLL_NODE avlNode;
    uint16_t writePos;   /* logbuf写位置 */
    uint16_t logBuffLen; /* 来自BkfBlackBoxType_的uintLogBuffLen */
    char *logBuf;
} BkfBlackBoxNode;

typedef struct {
    uint16_t logType;
    uint16_t pad;
    void *entityKey;
    uint32_t bid;
} BkfBlackBoxDispCtx;

void BkfBlackBoxUninit(BkfLog *log);

#pragma pack()
#endif

#if BKF_BLOCK("表")
const char *g_BkfLogLvlStrTbl[] = {
    "d",
    "i",
    "w",
    "e",
};

#endif

#if BKF_BLOCK("公有函数定义")
STATIC uint32_t BkfLogInitChkParam(BkfLogInitArg *arg)
{
    if (arg->name == VOS_NULL) {
        return BKF_ERR;
    }
    if (arg->memMng == VOS_NULL) {
        return BKF_ERR;
    }
    if (arg->disp == VOS_NULL) {
        return BKF_ERR;
    }
    if (arg->logCnt == VOS_NULL) {
        return BKF_ERR;
    }
    if (!BKF_LOG_LVL_IS_VALID(arg->moduleDefalutLvl)) {
        return BKF_ERR;
    }
    if (arg->memBufLen < 0) {
        return BKF_ERR;
    }
    return BKF_OK;
}
BkfLog *BkfLogInit(BkfLogInitArg *arg)
{
    uint32_t ret;
    BkfLog *log = VOS_NULL;
    uint32_t len;

    ret = BkfLogInitChkParam(arg);
    if (ret != BKF_OK) {
        goto error;
    }

    len = sizeof(BkfLog);
    log = BKF_MALLOC(arg->memMng, len);
    if (log == VOS_NULL) {
        goto error;
    }
    (void)memset_s(log, len, 0, len);
    log->argInit = *arg;
    log->name = BkfStrNew(arg->memMng, "%s_log", arg->name);
    log->argInit.name = log->name;
    log->outputEnable = arg->outputEnable;
    log->debugOutputEnable = 0; /* BKF在线debug功能默认关闭 */
    log->debugOutputOrNull = NULL;
    VOS_AVLL_INIT_TREE(log->modLvlSet, (AVLL_COMPARE)VOS_StrCmp, BKF_OFFSET(BkfLogModLvl, name[0]),
        BKF_OFFSET(BkfLogModLvl, avlNode));

    uint32_t initLen = arg->memBufLen > BKF_LOG_MEM_BUF_LEN_INIT ? BKF_LOG_MEM_BUF_LEN_INIT
                                                                 : ((uint32_t)arg->memBufLen);
    log->memBuf = BkfBufrInit(arg->memMng, initLen);

    VOS_AVLL_INIT_TREE(log->blackBoxTypeSet, (AVLL_COMPARE)Bkfuint16_tCmp, BKF_OFFSET(BkfBlackBoxType, logType),
        BKF_OFFSET(BkfBlackBoxType, avlNode));
    BkfLogDispInit(log);

    return log;

error:

    if (log != VOS_NULL) {
        BkfLogUninit(log);
    }
    return VOS_NULL;
}

void BkfLogUninit(BkfLog *log)
{
    if (log == VOS_NULL) {
        return;
    }

    BkfLogDelAllModLvl(log);
    BkfLogDispUninit(log);
    if (log->memBuf != VOS_NULL) {
        BkfBufrUninit(log->memBuf);
        log->memBuf = VOS_NULL;
    }
    BkfBlackBoxUninit(log);
    BkfStrDel(log->name);
    BKF_FREE(log->argInit.memMng, log);
    return;
}

uint32_t BkfLogSetModuleLvl(BkfLog *log, char *modName, uint8_t lvl)
{
    BkfLogModLvl *modLvl = VOS_NULL;
    BOOL reset2Default = VOS_FALSE;

    if ((log == VOS_NULL) || (modName == VOS_NULL) || !BKF_LOG_LVL_IS_VALID(lvl)) {
        return BKF_ERR;
    }

    modLvl = BkfLogFindModLvl(log, modName);
    reset2Default = (modLvl != VOS_NULL) && (lvl == log->argInit.moduleDefalutLvl);
    if (reset2Default) {
        BkfLogDelModLvl(log, modLvl);
        return BKF_OK;
    }

    if (modLvl == VOS_NULL) {
        modLvl = BkfLogAddModLvl(log, modName, lvl);
    } else {
        modLvl->lvl = lvl;
    }
    return (modLvl != VOS_NULL) ? BKF_OK : BKF_ERR;
}

STATIC BkfLogModLvl *BkfLogFindModLvl(BkfLog *log, char *modName)
{
    BkfLogModLvl *modLvl = VOS_NULL;

    if ((log->modLvlCache != VOS_NULL) && (VOS_StrCmp(log->modLvlCache->name, modName) == 0)) {
        return modLvl;
    } else if ((modLvl = VOS_AVLL_FIND(log->modLvlSet, modName)) != VOS_NULL) {
        log->modLvlCache = modLvl;
        return modLvl;
    }
    return VOS_NULL;
}

uint32_t BkfLogFirstHalf(BkfLog *log, char *modName, uint16_t line, uint8_t lvl)
{
    (void)line;
    BkfLogModLvl *modLvl = VOS_NULL;
    BOOL ret = (log == VOS_NULL) || (modName == VOS_NULL) || !BKF_LOG_LVL_IS_VALID(lvl);
    if (ret) {
        return BKF_ERR;
    }

    BOOL needOutDebug = (((log->debugOutputEnable) & (1 << lvl)) != 0);
    /* warning以及以上日志确保能写入内存日志中 */
    if (log->outputEnable == VOS_FALSE && lvl < BKF_LOG_LVL_WARN && !needOutDebug) {
        return BKF_ERR;
    }

    modLvl = BkfLogFindModLvl(log, modName);
    if (modLvl == VOS_NULL) {
        return (lvl >= log->argInit.moduleDefalutLvl) ? BKF_OK : BKF_ERR;
    } else {
        return (lvl >= modLvl->lvl) ? BKF_OK : BKF_ERR;
    }
}

STATIC int32_t BkfLogSecondHalfTime(BkfLog *log, char *buf, int32_t totalLen, int32_t usedLen)
{
    VOS_SYSTM_S sysTime;

    if (totalLen < usedLen) {
        return -1;
    }

    if (!log->argInit.logTime) {
        return 0;
    }

    (void)VOS_SystimeGet(&sysTime);
    return snprintf_truncated_s(&buf[usedLen], totalLen - usedLen, "[%02u:%02u:%02u:%03u], ", sysTime.ucHour,
        sysTime.ucMinute, sysTime.ucSecond, sysTime.uiMillSec);
}
STATIC int32_t BkfLogSecondHalfSeqAndModInfo(BkfLog *log, char *buf, int32_t totalLen, int32_t usedLen, char *modName,
    uint16_t line, uint8_t lvl, char *funcName)
{
    char *funcNameOutInfo = log->argInit.logFuncName ? funcName : "*";

    if (totalLen <= usedLen) {
        return -1;
    }

    return snprintf_truncated_s(&buf[usedLen], totalLen - usedLen, "[%-6u], [%-18s, %5u, %s@[%s]], ",
        BKF_GET_NEXT_VAL(log->seqSeed), modName, line, g_BkfLogLvlStrTbl[lvl], funcNameOutInfo);
}

STATIC int32_t BkfLogOutputToBufr(BkfBufr *memBuf, int32_t maxLen, uint8_t *buf, int32_t bufLen)
{
    int32_t freeLen = BkfBufrGetFreeLen(memBuf);
    int32_t usedLen = BkfBufrGetUsedLen(memBuf);
    int32_t curTotalLen = freeLen + usedLen;

    if (freeLen < bufLen && curTotalLen < maxLen) {
        /* buf 内存日志长度增加 */
        int32_t newLen = curTotalLen * 2;

        if (newLen > maxLen) {
            newLen = maxLen;
        }

        (void)BkfBufrResize(memBuf, newLen);
    }
    return BkfBufrPutForce(memBuf, (uint8_t *)buf, bufLen);
}

void BkfLogSecondHalf(BkfLog *log, char *modName, uint16_t line, uint8_t lvl, char *funcName, const char *fmt, ...)
{
    BOOL paramIsInvalid = (log == VOS_NULL) || (modName == VOS_NULL) || !BKF_LOG_LVL_IS_VALID(lvl) || (fmt == VOS_NULL);
    if (paramIsInvalid) {
        return;
    }

    BOOL needOutputFile = ((log->argInit.outputOrNull != VOS_NULL) && (log->outputEnable));
    BOOL needWrite2Mem = (log->memBuf != VOS_NULL) && (log->argInit.memBufLen > 0);
    BOOL needOutDebug = ((log->debugOutputOrNull != VOS_NULL) && (((log->debugOutputEnable) & (1 << lvl)) != 0));
    BOOL judgeFlg = !needOutputFile && !needOutDebug;
    /* 判断私有日志、内存日志、debug日志是否都不输出 */
    if (judgeFlg) {
        return;
    }

    char buf[BKF_1K];
    buf[0] = '\0';
    int32_t usedLen = 0;
    int32_t len = BkfLogSecondHalfTime(log, buf, sizeof(buf), usedLen);
    if (len < 0) {
        return;
    }
    usedLen += len;
    len = BkfLogSecondHalfSeqAndModInfo(log, buf, sizeof(buf), usedLen, modName, line, lvl, funcName);
    if (len < 0) {
        return;
    }
    usedLen += len;
    if ((size_t)usedLen >= sizeof(buf)) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    len = vsnprintf_truncated_s(&buf[usedLen], sizeof(buf) - usedLen, fmt, args);
    va_end(args);
    if (len < 0) {
        return;
    }
    usedLen += len;

    if (needWrite2Mem) {
        (void)BkfLogOutputToBufr(log->memBuf, log->argInit.memBufLen, (uint8_t *)buf, usedLen);
    }
    if (needOutputFile) {
        log->argInit.outputOrNull(log->argInit.cookie, (const char *)(&buf[0]));
    }
    if (needOutDebug) {
        log->debugOutputOrNull(log->argInit.cookie, (const char *)(&buf[0]));
    }
    return;
}

void BkfLogSetOutputEnable(BkfLog *log, BOOL enable)
{
    if (log == VOS_NULL) {
        return;
    }

    log->outputEnable = enable;
    return;
}

BOOL BkfLogGetOutputEnable(BkfLog *log)
{
    if (log == VOS_NULL) {
        return VOS_FALSE;
    }
    return log->outputEnable;
}

void BkfLogSetdebugOutputFun(BkfLog *log, F_BKF_DEBUG_OUTPUT fun)
{
    if (log == VOS_NULL) {
        return;
    }
    log->debugOutputOrNull = fun;
}

void BkfLogSetDebugOutputEnable(BkfLog *log, uint32_t debugLevel, BOOL enable)
{
    if (log == VOS_NULL) {
        return;
    }
    /* 所有level全使能/去使能 */
    if (debugLevel == BKF_LOG_LVL_CNT) {
        log->debugOutputEnable = enable ? BKF_DEBUG_LEVEL_ALL : 0;
        return;
    }

    if (enable) {
        log->debugOutputEnable |= (1 << debugLevel);
    } else {
        log->debugOutputEnable &= ~(1 << debugLevel);
        log->debugOutputEnable &= BKF_DEBUG_LEVEL_ALL; /* level只有四级，所以只关注后四位 */
    }
    return;
}

uint8_t BkfLogGetDebugOutputEnable(BkfLog *log)
{
    if (log == VOS_NULL) {
        return VOS_FALSE;
    }
    return log->debugOutputEnable;
}

void BkfLogTraceRcvDataFlow(BkfLog *log, char *recBuf, int32_t len)
{
    uint32_t i;
    char buf[BKF_1K / BKF_4BYTE] = {0};
    int32_t writeLen = 0;
    if (BkfLogGetOutputEnable(log) == VOS_FALSE) {
        return;
    }
    BKF_LOG_INFO(log, "TraceRecBuf_:\n");
    for (i = 0; i < (uint32_t)len; i++) {
        writeLen += snprintf_truncated_s(buf + writeLen, sizeof(buf) - (uint32_t)writeLen, "%02hhx ", recBuf[i]);
        if (i == 0) {
            continue;
        }
        if ((i % BKF_8BYTE) == 0) {
            writeLen += snprintf_truncated_s(buf + writeLen, sizeof(buf) - (uint32_t)writeLen, "\t");
        }
        if ((i % BKF_16BYTE) == 0) {
            writeLen += snprintf_truncated_s(buf + writeLen, sizeof(buf) - (uint32_t)writeLen, "\n\t");
            BKF_LOG_INFO(log, "%s", buf);
            writeLen = 0;
            (void)memset_s(buf, sizeof(buf), 0, sizeof(buf));
        }
    }
    if (writeLen > 0) {
        BKF_LOG_INFO(log, "%s\n", buf);
    }
    BKF_LOG_INFO(log, "\n");
}

#endif

#if BKF_BLOCK("私有函数定义")
STATIC BkfLogModLvl *BkfLogAddModLvl(BkfLog *log, char *modName, uint8_t lvl)
{
    BkfLogModLvl *modLvl = VOS_NULL;
    uint32_t strLen;
    uint32_t len;
    int32_t err;
    BOOL insOk = VOS_FALSE;

    strLen = VOS_StrLen(modName);
    len = sizeof(BkfLogModLvl) + strLen + 1;
    modLvl = BKF_MALLOC(log->argInit.memMng, len);
    if (modLvl == VOS_NULL) {
        goto error;
    }
    (void)memset_s(modLvl, len, 0, len);
    modLvl->lvl = lvl;
    err = snprintf_truncated_s(modLvl->name, strLen + 1, "%s", modName);
    if (err < 0) {
        goto error;
    }
    VOS_AVLL_INIT_NODE(modLvl->avlNode);
    insOk = VOS_AVLL_INSERT(log->modLvlSet, modLvl->avlNode);
    if (!insOk) {
        goto error;
    }

    log->modLvlCache = modLvl;
    return modLvl;

error:

    if (modLvl != VOS_NULL) {
        /* insOk一定为false */
        BKF_FREE(log->argInit.memMng, modLvl);
    }
    return VOS_NULL;
}

STATIC void BkfLogDelModLvl(BkfLog *log, BkfLogModLvl *modLvl)
{
    if (log->modLvlCache == modLvl) {
        log->modLvlCache = VOS_NULL;
    }
    VOS_AVLL_DELETE(log->modLvlSet, modLvl->avlNode);
    BKF_FREE(log->argInit.memMng, modLvl);
    return;
}

STATIC void BkfLogDelAllModLvl(BkfLog *log)
{
    BkfLogModLvl *modLvl = VOS_NULL;
    void *itor = VOS_NULL;

    for (modLvl = BkfLogGetFirstModLvl(log, &itor); modLvl != VOS_NULL; modLvl = BkfLogGetNextModLvl(log, &itor)) {
        BkfLogDelModLvl(log, modLvl);
    }
    return;
}

STATIC BkfLogModLvl *BkfLogFindNextModLvl(BkfLog *log, char *modName)
{
    BkfLogModLvl *modLvl = VOS_NULL;

    modLvl = VOS_AVLL_FIND_NEXT(log->modLvlSet, modName);
    return modLvl;
}

STATIC BkfLogModLvl *BkfLogGetFirstModLvl(BkfLog *log, void **itorOutOrNull)
{
    BkfLogModLvl *modLvl = VOS_NULL;

    modLvl = VOS_AVLL_FIRST(log->modLvlSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (modLvl != VOS_NULL) ? VOS_AVLL_NEXT(log->modLvlSet, modLvl->avlNode) : VOS_NULL;
    }
    return modLvl;
}

STATIC BkfLogModLvl *BkfLogGetNextModLvl(BkfLog *log, void **itorInOut)
{
    BkfLogModLvl *modLvl = VOS_NULL;

    modLvl = (*itorInOut);
    *itorInOut = (modLvl != VOS_NULL) ? VOS_AVLL_NEXT(log->modLvlSet, modLvl->avlNode) : VOS_NULL;
    return modLvl;
}

#endif

#if BKF_BLOCK("disp")
STATIC void BkfLogDispArg(BkfLog *log)
{
    BkfLogInitArg *arg = &log->argInit;
    BkfDisp *disp = arg->disp;
    uint32_t modLvlCnt;
    BkfLogModLvl *modLvl = VOS_NULL;
    void *itor = VOS_NULL;
    uint8_t buf[BKF_1K];

    modLvlCnt = 0;
    for (modLvl = BkfLogGetFirstModLvl(log, &itor); modLvl != VOS_NULL; modLvl = BkfLogGetNextModLvl(log, &itor)) {
        modLvlCnt++;
    }

    BKF_DISP_PRINTF(disp, "===argInit===\n");
    BKF_DISP_PRINTF(disp, "memMng(%#x)/disp(%#x)\n", BKF_MASK_ADDR(arg->memMng), BKF_MASK_ADDR(arg->disp));
    BKF_DISP_PRINTF(disp, "cookie(%#x)/debugOutputOrNull(%#x)", BKF_MASK_ADDR(arg->cookie),
        BKF_MASK_ADDR(log->debugOutputOrNull));
    BKF_DISP_PRINTF(disp, "cookie(%#x)/outputOrNull(%#x)", BKF_MASK_ADDR(arg->cookie),
        BKF_MASK_ADDR(arg->outputOrNull));
    BKF_DISP_PRINTF(disp, "outputEnable(%u)/logTime(%u)/moduleDefalutLvl(%u)/\n", arg->outputEnable, arg->logTime,
        arg->moduleDefalutLvl);
    BKF_DISP_PRINTF(disp, "name(%s)\n", log->name);
    BKF_DISP_PRINTF(disp, "===runtime===\n");
    BKF_DISP_PRINTF(disp, "outputEnable(%u)/modLvlCnt(%u)/modLvlCache(%#x)/seqSeed(%u)\n", log->outputEnable, modLvlCnt,
        BKF_MASK_ADDR(log->modLvlCache), log->seqSeed);
    BKF_DISP_PRINTF(disp, "memBuf(%#x), %s\n", BKF_MASK_ADDR(log->memBuf),
        BkfBufrGetStr(log->memBuf, buf, sizeof(buf)));
    return;
}
void BkfLogDisp(BkfLog *log)
{
    BkfDisp *disp = log->argInit.disp;
    char *lastModName = VOS_NULL;
    BkfLogModLvl *modLvl = VOS_NULL;
    uint32_t modLvlCnt = 0;

    lastModName = BKF_DISP_GET_LAST_CTX(disp, &modLvlCnt);
    if (lastModName == VOS_NULL) {
        BkfLogDispArg(log);
        modLvlCnt = 0;
        modLvl = BkfLogGetFirstModLvl(log, VOS_NULL);
    } else {
        modLvl = BkfLogFindNextModLvl(log, lastModName);
    }

    if (modLvl != VOS_NULL) {
        BKF_DISP_PRINTF(disp, "modName(%s)/lvl(%u)\n", modLvl->name, modLvl->lvl);

        modLvlCnt++;
        BKF_DISP_SAVE_CTX(disp, modLvl->name, VOS_StrLen(modLvl->name) + 1, &modLvlCnt, sizeof(modLvlCnt));
    } else {
        BKF_DISP_PRINTF(disp, " ***total(%u) modLvl(s)***\n", modLvlCnt);
    }
}

void BkfLogDispEnableOutput(BkfLog *log, uintptr_t enable)
{
    BkfDisp *disp = log->argInit.disp;

    if (enable && log->outputEnable) {
        BKF_DISP_PRINTF(disp, "has enable, not change\n");
        return;
    }
    if (!enable && !log->outputEnable) {
        BKF_DISP_PRINTF(disp, "has disable, not change\n");
        return;
    }

    BkfLogSetOutputEnable(log, (enable ? VOS_TRUE : VOS_FALSE));
    BKF_DISP_PRINTF(disp, "ok\n");
    return;
}

void BkfLogDispMemLog(BkfLog *log, uintptr_t lastKBytes)
{
    BkfDisp *disp = log->argInit.disp;
    BkfBufrDbgPeekItor *last = VOS_NULL;
    BkfBufrDbgPeekItor itor;
    uint8_t buf[BKF_1K / 4];
    int32_t lastBytes = 0;
    int32_t peekLen = 0;
    int32_t peekTotalLen = 0;

    if (log->memBuf == VOS_NULL) {
        BKF_DISP_PRINTF(disp, "memBuf off\n");
        return;
    }
    last = BKF_DISP_GET_LAST_CTX(disp, &peekTotalLen);
    if (last == VOS_NULL) {
        BKF_DISP_PRINTF(disp, "memBuf: %s\n", BkfBufrGetStr(log->memBuf, buf, sizeof(buf)));
        BKF_DISP_PRINTF(disp, "--------------------\n");
        lastBytes = (int32_t)BKF_GET_MIN(BKF_LOG_MEM_BUF_LEN_MAX, (uint64_t)lastKBytes * BKF_1K);
        peekTotalLen = 0;
        peekLen = BkfBufrDbgPeekLastFirst(log->memBuf, lastBytes, buf, sizeof(buf) - 1, &itor);
    } else {
        itor = *last;
        peekLen = BkfBufrDbgPeekLastNext(log->memBuf, buf, sizeof((buf)) - 1, &itor);
    }

    if (peekLen > 0) {
        buf[peekLen] = '\0';
        BKF_DISP_PRINTF(disp, "%s", buf);
        peekTotalLen += peekLen;
        BKF_DISP_SAVE_CTX(disp, &itor, sizeof(itor), &peekTotalLen, sizeof(peekTotalLen));
    } else if (peekLen == 0) {
        BKF_DISP_PRINTF(disp, "\n***total %d char(s), ***\n", peekTotalLen);
    } else {
        BKF_DISP_PRINTF(disp, "\nhas disp %d char(s), log maybe change, try again\n", peekTotalLen);
    }
}

void BkfLogDispSetMemBufLen(BkfLog *log, uintptr_t newMemBufLenKBytes)
{
    BkfDisp *disp = log->argInit.disp;
    uint64_t newLen;

    newLen = (uint64_t)newMemBufLenKBytes * BKF_1K;
    if (newLen > BKF_LOG_MEM_BUF_LEN_MAX) {
        BKF_DISP_PRINTF(disp, "max %u KB\n", BKF_LOG_MEM_BUF_LEN_MAX / BKF_1K);
        return;
    }

    BkfBufrUninit(log->memBuf);
    log->memBuf = VOS_NULL;
    if (newLen > 0) {
        log->memBuf = BkfBufrInit(log->argInit.memMng, (int32_t)newLen);
    }
    BKF_DISP_PRINTF(disp, "memBuf(%#x), ok\n", BKF_MASK_ADDR(log->memBuf));
}

STATIC void BkfLogDispInit(BkfLog *log)
{
    BkfDisp *disp = log->argInit.disp;
    char *objName = log->name;
    uint32_t ret;

    ret = BkfDispRegObj(disp, objName, log);
    if (ret != BKF_OK) {
        BKF_ASSERT(0);
        return;
    }

    (void)BKF_DISP_REG_FUNC(disp, BkfLogDisp, "disp log info", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfLogDispMemLog, "disp log in mem", objName, 1, "uniLastKBytes");
    (void)BKF_DISP_REG_TEST_FUNC(disp, BkfLogDispSetMemBufLen, "set mem buf len", objName, 1, "uniNewMemBufLenKBytes");
    (void)BKF_DISP_REG_TEST_FUNC(disp, BkfLogDispEnableOutput, "enable log output", objName, 1, "uniEnable");
    (void)BKF_DISP_REG_FUNC(disp, BkfBlackBoxLogDispAll, "disp allBlackBox info", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfBlackBoxLogDispSpecType, "disp spec type BlaclBox info", objName, 1,
        "uniBlackboxType");
}

STATIC void BkfLogDispUninit(BkfLog *log)
{
    BkfDispUnregObj(log->argInit.disp, log->name);
}
#endif

#if BKF_BLOCK("黑匣子日志")
void BkfBlackBoxFreeLogNode(BkfLog *log, BkfBlackBoxType *blackBoxType, BkfBlackBoxNode *node)
{
    VOS_AVLL_DELETE(blackBoxType->blackBoxInstSet, node->avlNode);
    BKF_FREE(log->argInit.memMng, node->logBuf);
    BKF_FREE(log->argInit.memMng, node);
}

void BkfBlackBoxUninitType(BkfLog *log, BkfBlackBoxType *blackBoxType)
{
    if (!log) {
        return;
    }
    BkfBlackBoxNode *entity = VOS_AVLL_FIRST(blackBoxType->blackBoxInstSet);
    while (entity) {
        BkfBlackBoxFreeLogNode(log, blackBoxType, entity);
        entity = VOS_AVLL_FIRST(blackBoxType->blackBoxInstSet);
    }
    VOS_AVLL_DELETE(log->blackBoxTypeSet, blackBoxType->avlNode);
    BKF_FREE(log->argInit.memMng, blackBoxType);
}

void BkfBlackBoxUninit(BkfLog *log)
{
    BkfBlackBoxType *blackBoxType = VOS_AVLL_FIRST(log->blackBoxTypeSet);
    while (blackBoxType) {
        BkfBlackBoxUninitType(log, blackBoxType);
        blackBoxType = VOS_AVLL_FIRST(log->blackBoxTypeSet);
    }
}

STATIC int32_t BkfBlackBoxLogNodeCmp(BkfBlackBoxNode *key1Input, BkfBlackBoxNode *key2InDs)
{
    int32_t ret = BKF_CMP_X(key1Input->keyAddr, key2InDs->keyAddr);
    if (ret != 0) {
        return ret;
    }
    return BKF_CMP_X(key1Input->bid, key2InDs->bid);
}

uint32_t BkfBlackBoxRegType(BkfLog *log, uint16_t type, uint16_t logBuffLen)
{
    if (!log || logBuffLen == 0) {
        return BKF_ERR;
    }
    BkfBlackBoxType *blackBoxType = (BkfBlackBoxType *)VOS_AVLL_FIND(log->blackBoxTypeSet, &type);
    if (blackBoxType) {
        return BKF_OK;
    }
    blackBoxType = BKF_MALLOC(log->argInit.memMng, sizeof(BkfBlackBoxType));
    if (!blackBoxType) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }
    blackBoxType->uintLogBuffLen = logBuffLen;
    blackBoxType->logType = type;
    VOS_AVLL_INIT_NODE(blackBoxType->avlNode);
    VOS_AVLL_INIT_TREE(blackBoxType->blackBoxInstSet, (AVLL_COMPARE)BkfBlackBoxLogNodeCmp, 0,
        BKF_OFFSET(BkfBlackBoxNode, avlNode));
    (void)VOS_AVLL_INSERT(log->blackBoxTypeSet, blackBoxType->avlNode);
    return BKF_OK;
}

void BkfBlackBoxDelRegType(BkfLog *log, uint16_t type)
{
    if (!log) {
        return;
    }
    BkfBlackBoxType *blackBoxType = (BkfBlackBoxType *)VOS_AVLL_FIND(log->blackBoxTypeSet, &type);
    if (!blackBoxType) {
        return;
    }
    BkfBlackBoxUninitType(log, blackBoxType);
}

void BkfBlackBoxWriteLog(BkfBlackBoxNode *logNode, char *str, uint32_t len)
{
    char *ch = str;
    uint32_t endPos = logNode->logBuffLen - 1;
    uint32_t beginPos = 0;
    uint32_t i;
    for (i = 0; i < len; i++) {
        logNode->logBuf[logNode->writePos] = *ch;
        ch++;
        logNode->writePos++;
        if (logNode->writePos == endPos) {
            logNode->writePos = beginPos;
        }
    }
}

BkfBlackBoxNode *BkfBlackBoxNewLogNode(BkfLog *log, BkfBlackBoxType *blackBoxType, void *key, uint32_t bid)
{
    BkfBlackBoxNode *logNode = BKF_MALLOC(log->argInit.memMng, sizeof(BkfBlackBoxNode));
    if (!logNode) {
        return NULL;
    }
    logNode->logBuffLen = blackBoxType->uintLogBuffLen;
    VOS_AVLL_INIT_NODE(logNode->avlNode);
    logNode->writePos = 0;
    logNode->keyAddr = key;
    logNode->bid = bid;
    logNode->logBuf = BKF_MALLOC(log->argInit.memMng, logNode->logBuffLen);
    if (!logNode->logBuf) {
        BKF_FREE(log->argInit.memMng, logNode);
        return NULL;
    }
    (void)memset_s(logNode->logBuf, logNode->logBuffLen, 0, logNode->logBuffLen);
    (void)VOS_AVLL_INSERT(blackBoxType->blackBoxInstSet, logNode->avlNode);
    return logNode;
}

void BkfBlackBoxLog(BkfLog *log, uint16_t type, void *key, uint32_t bid, const char *format, ...)
{
    if (!log) {
        return;
    }
    BkfBlackBoxType *blackBoxType = (BkfBlackBoxType *)VOS_AVLL_FIND(log->blackBoxTypeSet, &type);
    if (!blackBoxType) {
        return;
    }
    BkfBlackBoxNode tmpLogNode = {.keyAddr = key, .bid = bid};
    BkfBlackBoxNode *logNode = (BkfBlackBoxNode *)VOS_AVLL_FIND(blackBoxType->blackBoxInstSet, &tmpLogNode);
    if (!logNode) {
        logNode = BkfBlackBoxNewLogNode(log, blackBoxType, key, bid);
    }
    if (!logNode) {
        return;
    }
    uint8_t logBuf[BKF_1K] = {0};
    VOS_SYSTM_S sysTime;
    (void)VOS_SystimeGet(&sysTime);
    int32_t len = 0;
    int32_t ret = 0;
    len += snprintf_truncated_s((char *)logBuf + len, BKF_1K - (uint32_t)len,
        "\n<%02hhu%02hhu %02hhu:%02hhu:%02hhu:%03u> ", sysTime.ucMonth, sysTime.ucDate, sysTime.ucHour,
        sysTime.ucMinute, sysTime.ucSecond, sysTime.uiMillSec);
    va_list args;
    va_start(args, format);
    ret = vsnprintf_truncated_s((char *)logBuf + len, BKF_1K - (uint32_t)len, format, args);
    va_end(args);
    if (ret == -1) {
        BKF_ASSERT(0);
        return;
    }
    len += ret;
    len += snprintf_truncated_s((char *)logBuf + len, BKF_1K - (uint32_t)len, "%s", " ");
    BkfBlackBoxWriteLog(logNode, (char *)logBuf, (uint32_t)len);
    return;
}

void BkfBlackBoxDelLogInstNode(BkfLog *log, uint16_t type, void *key, uint32_t bid)
{
    if (!log) {
        return;
    }
    BkfBlackBoxType *logType = (BkfBlackBoxType *)VOS_AVLL_FIND(log->blackBoxTypeSet, &type);
    if (!logType) {
        return;
    }
    BkfBlackBoxNode tmpLogNode = {.keyAddr = key, .bid = bid};
    BkfBlackBoxNode *logNode = (BkfBlackBoxNode *)VOS_AVLL_FIND(logType->blackBoxInstSet, &tmpLogNode);
    if (!logNode) {
        return;
    }
    BkfBlackBoxFreeLogNode(log, logType, logNode);
}

void BkfBlackBoxDispOneInstLog(BkfLog *log, uint16_t type, void *key, uint32_t bid)
{
    if (!log) {
        return;
    }
    BkfBlackBoxType *blackBoxType = (BkfBlackBoxType *)VOS_AVLL_FIND(log->blackBoxTypeSet, &type);
    if (!blackBoxType) {
        return;
    }
    BkfBlackBoxNode tmpLogNode = {.keyAddr = key, .bid = bid};
    BkfBlackBoxNode *logInst = (BkfBlackBoxNode *)VOS_AVLL_FIND(blackBoxType->blackBoxInstSet, &tmpLogNode);
    if (!logInst) {
        return;
    }
    BKF_DISP_PRINTF(log->argInit.disp, "BlackBoxLog:%#x\n%s\n", BKF_MASK_ADDR(key), logInst->logBuf);
}

void BkfBlackBoxLogDispAll(BkfLog *log)
{
    BkfBlackBoxDispCtx currBlackBoxLogCtx = {0};
    BkfBlackBoxDispCtx *lastBlackBoxLogCtx = NULL;
    BkfDisp *disp = log->argInit.disp;
    lastBlackBoxLogCtx = BKF_DISP_GET_LAST_CTX(disp, NULL);
    BkfBlackBoxType *blackBoxType = NULL;
    BkfBlackBoxNode *entity = NULL;
    if (!lastBlackBoxLogCtx) {
        /* new disp */
        BKF_DISP_PRINTF(disp, "--------\n");
        blackBoxType = VOS_AVLL_FIRST(log->blackBoxTypeSet);
        if (!blackBoxType) {
            BKF_DISP_PRINTF(disp, "no BlackBox Log\n");
            return;
        }
        entity = VOS_AVLL_FIRST(blackBoxType->blackBoxInstSet);
    } else {
        blackBoxType = (BkfBlackBoxType *)VOS_AVLL_FIND(log->blackBoxTypeSet, &lastBlackBoxLogCtx->logType);
        if (!blackBoxType) {
            BKF_DISP_PRINTF(disp, "BlackBox log disp end\n");
            return;
        }
        BkfBlackBoxNode tmBlackBoxNode = {.keyAddr = lastBlackBoxLogCtx->entityKey, .bid = lastBlackBoxLogCtx->bid};
        entity = (BkfBlackBoxNode *)VOS_AVLL_FIND_NEXT(blackBoxType->blackBoxInstSet, &tmBlackBoxNode);
        if (!entity) {
            blackBoxType = (BkfBlackBoxType *)VOS_AVLL_FIND_NEXT(log->blackBoxTypeSet, &lastBlackBoxLogCtx->logType);
            if (!blackBoxType) {
                BKF_DISP_PRINTF(disp, "BlackBox log disp end\n");
                return;
            }
            entity = VOS_AVLL_FIRST(blackBoxType->blackBoxInstSet);
        }
    }
    currBlackBoxLogCtx.logType = blackBoxType->logType;
    BKF_DISP_PRINTF(disp, "BlackBoxType %u,   bufLen %u\n", blackBoxType->logType, blackBoxType->uintLogBuffLen);
    if (entity) {
        currBlackBoxLogCtx.entityKey = entity->keyAddr;
        currBlackBoxLogCtx.bid = entity->bid;
        BKF_DISP_PRINTF(disp, "key %#x,  bid %u  pos %u\n", BKF_MASK_ADDR(entity->keyAddr), entity->bid,
            entity->writePos);
        BKF_DISP_PRINTF(disp, "%s\n", entity->logBuf);
    }

    BKF_DISP_SAVE_CTX(disp, &currBlackBoxLogCtx, sizeof(BkfBlackBoxDispCtx), VOS_NULL, 0);
    return;
}

void BkfBlackBoxLogDispSpecType(BkfLog *log, uintptr_t blackboxType)
{
    BkfBlackBoxDispCtx currBlackBoxLogCtx = {0};
    BkfBlackBoxDispCtx *lastBlackBoxLogCtx = NULL;
    BkfDisp *disp = log->argInit.disp;
    lastBlackBoxLogCtx = BKF_DISP_GET_LAST_CTX(disp, NULL);
    BkfBlackBoxType *blackBoxType = NULL;
    BkfBlackBoxNode *boxNode = NULL;
    uint16_t logType = (uint16_t)blackboxType;
    if (!lastBlackBoxLogCtx) {
        /* new disp */
        BKF_DISP_PRINTF(disp, "--------\n");
        blackBoxType = (BkfBlackBoxType *)VOS_AVLL_FIND(log->blackBoxTypeSet, &logType);
        if (!blackBoxType) {
            BKF_DISP_PRINTF(disp, "no BlackBox Log:%u\n", logType);
            return;
        }
        boxNode = VOS_AVLL_FIRST(blackBoxType->blackBoxInstSet);
    } else {
        blackBoxType = (BkfBlackBoxType *)VOS_AVLL_FIND(log->blackBoxTypeSet, &lastBlackBoxLogCtx->logType);
        if (!blackBoxType) {
            BKF_DISP_PRINTF(disp, "BlackBox log disp end %u\n", logType);
            return;
        }
        BkfBlackBoxNode tmBlackBoxNode = {.keyAddr = lastBlackBoxLogCtx->entityKey, .bid = lastBlackBoxLogCtx->bid};
        boxNode = (BkfBlackBoxNode *)VOS_AVLL_FIND_NEXT(blackBoxType->blackBoxInstSet, &tmBlackBoxNode);
        if (!boxNode) {
            BKF_DISP_PRINTF(disp, "BlackBox log type %u disp end\n", logType);
            return;
        }
    }
    currBlackBoxLogCtx.logType = logType;
    BKF_DISP_PRINTF(disp, "BlackBoxType %u,   bufLen %u\n", logType, blackBoxType->uintLogBuffLen);
    if (boxNode) {
        currBlackBoxLogCtx.entityKey = boxNode->keyAddr;
        currBlackBoxLogCtx.bid = boxNode->bid;
        BKF_DISP_PRINTF(disp, "key %#x,  bid %u  pos %u\n", BKF_MASK_ADDR(boxNode->keyAddr), boxNode->bid,
            boxNode->writePos);
        BKF_DISP_PRINTF(disp, "%s\n", boxNode->logBuf);
    }
    BKF_DISP_SAVE_CTX(disp, &currBlackBoxLogCtx, sizeof(BkfBlackBoxDispCtx), VOS_NULL, 0);
    return;
}

#endif

#ifdef __cplusplus
}
#endif
