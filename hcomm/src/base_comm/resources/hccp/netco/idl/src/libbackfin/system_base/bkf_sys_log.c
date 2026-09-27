/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_sys_log.h"
#include "bkf_str.h"
#include "bkf_assert.h"
#include "securec.h"
#include "v_avll.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#pragma pack(4)

struct tagBkfSysLogMng {
    BkfSysLogInitArg argInit;
    AVLL_TREE tableTypeSet;
    char *name;
};

/* reg */

typedef struct tagBkfSysLogTableType {
    AVLL_TREE sysLogTree;
    AVLL_NODE avlNode;
    BkfSysLogTypeVTbl vTbl;
    char name[0];
} BkfSysLogTableType;

typedef struct tagBkfSysLog {
    AVLL_NODE avlNode;
    BkfTmrId *tmrId;
    void *val;
    BkfSysLogMng *sysLogMng;
    BkfSysLogTableType *tableType;
    char key[0];
} BkfSysLog;

#pragma pack()

/* 工具函数 */

STATIC uint32_t BkfSysLogTmrStart(BkfSysLogMng *sysLogMng, BkfSysLogTableType *tableType, BkfSysLog *sysLog);

/* 数据结构操作 */

STATIC BkfSysLog *BkfSysLogCreate(BkfSysLogMng *sysLogMng, BkfSysLogTableType *tableType, void *key, void *val)
{
    uint32_t sysLogLen;
    BkfSysLog *sysLog = VOS_NULL;

    sysLogLen = sizeof(BkfSysLog) + tableType->vTbl.keyLen + tableType->vTbl.valLen;

    sysLog = BKF_MALLOC(sysLogMng->argInit.memMng, sysLogLen);
    if (sysLog == VOS_NULL) {
        return VOS_NULL;
    }
    (void)memset_s(sysLog, sysLogLen, 0, sysLogLen);
    sysLog->val = sysLog->key + tableType->vTbl.keyLen;

    errno_t err = memcpy_s(sysLog->key, tableType->vTbl.keyLen, key, tableType->vTbl.keyLen);
    if (err != EOK) {
        goto error;
    }
    VOS_AVLL_INIT_NODE(sysLog->avlNode);
    if (!VOS_AVLL_INSERT(tableType->sysLogTree, sysLog->avlNode)) {
        goto error;
    }
    err = memcpy_s(sysLog->val, tableType->vTbl.valLen, val, tableType->vTbl.valLen);
    if (err != EOK) {
        goto error;
    }
    if (BkfSysLogTmrStart(sysLogMng, tableType, sysLog) != BKF_OK) {
        goto error;
    }
    return sysLog;

error:

    if (sysLog != VOS_NULL) {
        BKF_FREE(sysLogMng->argInit.memMng, sysLog);
    }
    return VOS_NULL;
}

STATIC void BkfSysLogDelete(BkfSysLogMng *sysLogMng, BkfSysLogTableType *tableType, BkfSysLog *sysLog)
{
    VOS_AVLL_DELETE(tableType->sysLogTree, sysLog->avlNode);
    BkfTmrStop(sysLogMng->argInit.tmrMng, sysLog->tmrId);
    BKF_FREE(sysLogMng->argInit.memMng, sysLog);
}

STATIC BkfSysLog *BkfSysLogGet(BkfSysLogTableType *tableType, void *key)
{
    return VOS_AVLL_FIND(tableType->sysLogTree, key);
}

STATIC BkfSysLogTableType *BkfSysLogTableTypeCreate(BkfSysLogMng *sysLogMng, BkfSysLogTypeVTbl *vTbl)
{
    BkfSysLogTableType *tableType = VOS_NULL;

    tableType = BKF_MALLOC(sysLogMng->argInit.memMng, sizeof(BkfSysLogTableType) + strlen(vTbl->name) + 1);
    if (tableType == VOS_NULL) {
        return VOS_NULL;
    }
    (void)memset_s(tableType, sizeof(BkfSysLogTableType), 0, sizeof(BkfSysLogTableType));
    tableType->vTbl = *vTbl;
    errno_t err = memcpy_s(tableType->name, strlen(vTbl->name), vTbl->name, strlen(vTbl->name));
    if (err != EOK) {
        BKF_FREE(sysLogMng->argInit.memMng, tableType);
        return VOS_NULL;
    }
    tableType->vTbl.name = tableType->name;
    VOS_AVLL_INIT_NODE(tableType->avlNode);
    if (!VOS_AVLL_INSERT(sysLogMng->tableTypeSet, tableType->avlNode)) {
        BKF_FREE(sysLogMng->argInit.memMng, tableType);
        return VOS_NULL;
    }
    VOS_AVLL_INIT_TREE(tableType->sysLogTree, (AVLL_COMPARE)tableType->vTbl.keyCmp, BKF_OFFSET(BkfSysLog, key[0]),
        BKF_OFFSET(BkfSysLog, avlNode));

    return tableType;
}

STATIC void BkfSysLogTableTypeDelete(BkfSysLogMng *sysLogMng, BkfSysLogTableType *tableType)
{
    BkfSysLog *sysLog = VOS_AVLL_FIRST(tableType->sysLogTree);
    while (sysLog != VOS_NULL) {
        BkfSysLog *sysLogNext = VOS_AVLL_NEXT(tableType->sysLogTree, sysLog->avlNode);
        BkfSysLogDelete(sysLogMng, tableType, sysLog);
        sysLog = sysLogNext;
    }

    VOS_AVLL_DELETE(sysLogMng->tableTypeSet, tableType->avlNode);
    BKF_FREE(sysLogMng->argInit.memMng, tableType);
}

STATIC BkfSysLogTableType *BkfSysLogTableTypeGet(BkfSysLogMng *sysLogMng, uint32_t typeId)
{
    return VOS_AVLL_FIND(sysLogMng->tableTypeSet, &typeId);
}

/* 业务逻辑 */

STATIC void BkfSysLogDispInit(BkfSysLogMng *sysLogMng)
{
    BkfDisp *disp = sysLogMng->argInit.disp;
    char *objName = sysLogMng->name;

    if (BkfDispRegObj(disp, objName, sysLogMng) != BKF_OK) {
        BKF_ASSERT(0);
    }
}

STATIC void BkfSysLogDispUninit(BkfSysLogMng *sysLogMng)
{
    BkfDispUnregObj(sysLogMng->argInit.disp, sysLogMng->name);
}

STATIC uint32_t BkfSysLogInitChkParam(BkfSysLogInitArg *arg)
{
    if (arg == VOS_NULL) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }
    if (arg->name == VOS_NULL) {
        return BKF_ERR;
    }
    if (arg->memMng == VOS_NULL) {
        return BKF_ERR;
    }
    if (arg->disp == VOS_NULL) {
        return BKF_ERR;
    }
    if (arg->tmrMng == VOS_NULL) {
        return BKF_ERR;
    }
    if (arg->printf == VOS_NULL) {
        return BKF_ERR;
    }

    return BKF_OK;
}

BkfSysLogMng *BkfSysLogInit(BkfSysLogInitArg *arg)
{
    BkfSysLogMng *sysLogMng = VOS_NULL;
    uint32_t sysLogSize;

    if (BkfSysLogInitChkParam(arg) != BKF_OK) {
        goto error;
    }
    sysLogSize = sizeof(BkfSysLogMng);
    sysLogMng = BKF_MALLOC(arg->memMng, sysLogSize);
    if (sysLogMng == VOS_NULL) {
        goto error;
    }

    (void)memset_s(sysLogMng, sysLogSize, 0, sysLogSize);

    sysLogMng->argInit = *arg;
    sysLogMng->name = BkfStrNew(arg->memMng, "%s_syslog", arg->name);
    sysLogMng->argInit.name = sysLogMng->name;
    VOS_AVLL_INIT_TREE(sysLogMng->tableTypeSet, (AVLL_COMPARE)Bkfuint32_tCmp,
        BKF_OFFSET(BkfSysLogTableType, vTbl) + BKF_OFFSET(BkfSysLogTypeVTbl, typeId),
        BKF_OFFSET(BkfSysLogTableType, avlNode));

    BkfSysLogDispInit(sysLogMng);

    return sysLogMng;

error:

    BkfSysLogUninit(sysLogMng);
    return VOS_NULL;
}

STATIC void BkfSysLogUnRegAll(BkfSysLogMng *sysLogMng)
{
    BkfSysLogTableType *tableType = VOS_AVLL_FIRST(sysLogMng->tableTypeSet);
    BkfSysLogTableType *tableTypeNext = tableType == VOS_NULL
                                            ? VOS_NULL
                                            : VOS_AVLL_NEXT(sysLogMng->tableTypeSet, tableType->avlNode);
    while (tableType != VOS_NULL) {
        BkfSysLogTableTypeDelete(sysLogMng, tableType);
        tableType = tableTypeNext;
        tableTypeNext = tableType == VOS_NULL ? VOS_NULL : VOS_AVLL_NEXT(sysLogMng->tableTypeSet, tableType->avlNode);
    }
}

void BkfSysLogUninit(BkfSysLogMng *sysLogMng)
{
    if (sysLogMng == VOS_NULL) {
        return;
    }
    BkfSysLogUnRegAll(sysLogMng);

    BkfSysLogDispUninit(sysLogMng);
    BkfStrDel(sysLogMng->name);
    BKF_FREE(sysLogMng->argInit.memMng, sysLogMng);
}

STATIC uint32_t BkfSysLogRegChkParam(BkfSysLogMng *sysLogMng, BkfSysLogTypeVTbl *vTbl)
{
    if (sysLogMng == VOS_NULL) {
        return BKF_ERR;
    }
    if (vTbl == VOS_NULL) {
        return BKF_ERR;
    }
    if (vTbl->name == VOS_NULL) {
        return BKF_ERR;
    }
    if (vTbl->keyCmp == VOS_NULL) {
        return BKF_ERR;
    }
    if (vTbl->out == VOS_NULL) {
        return BKF_ERR;
    }
    return BKF_OK;
}

uint32_t BkfSysLogReg(BkfSysLogMng *sysLogMng, BkfSysLogTypeVTbl *vTbl)
{
    BkfSysLogTableType *tableType = VOS_NULL;
    if (BkfSysLogRegChkParam(sysLogMng, vTbl) != BKF_OK) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(sysLogMng->argInit.log, "sysLog(%#x)/table(%u, %s), vTbl(%#x)\n", BKF_MASK_ADDR(sysLogMng),
        vTbl->typeId, vTbl->name, BKF_MASK_ADDR(vTbl));

    tableType = BkfSysLogTableTypeGet(sysLogMng, vTbl->typeId);
    if (tableType != VOS_NULL) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }
    tableType = BkfSysLogTableTypeCreate(sysLogMng, vTbl);
    if (tableType == VOS_NULL) {
        return BKF_ERR;
    }

    return BKF_OK;
}

STATIC void BkfSysLogTmrCallback(BkfSysLog *sysLog, void *paramTmrLibUnknown)
{
    (void)paramTmrLibUnknown;
    BkfSysLogMng *sysLogMng = sysLog->sysLogMng;
    BkfSysLogTableType *tableType = sysLog->tableType;

    if (VOS_AVLL_IN_TREE(sysLog->avlNode)) {
        BkfSysLogDelete(sysLogMng, tableType, sysLog);
    }
}

STATIC uint32_t BkfSysLogTmrStart(BkfSysLogMng *sysLogMng, BkfSysLogTableType *tableType, BkfSysLog *sysLog)
{
    if (sysLog->tmrId != VOS_NULL) {
        BKF_ASSERT(0);
        return BKF_OK;
    }
    sysLog->tableType = tableType;
    sysLog->sysLogMng = sysLogMng;
    sysLog->tmrId = BkfTmrStartLoop(sysLogMng->argInit.tmrMng, (F_BKF_TMR_TIMEOUT_PROC)BkfSysLogTmrCallback,
        sysLogMng->argInit.restrainIntervalMs, sysLog);
    return BKF_OK;
}

STATIC uint32_t BkfSysLogChkParam(BkfSysLogMng *sysLogMng, void *key, void *val)
{
    if (sysLogMng == VOS_NULL) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }
    if (key == VOS_NULL) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }
    if (val == VOS_NULL) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }
    return BKF_OK;
}

uint32_t BkfSysLogFunc(BkfSysLogMng *sysLogMng, uint32_t typeId, void *key, void *val)
{
    if (BkfSysLogChkParam(sysLogMng, key, val) != BKF_OK) {
        return BKF_ERR;
    }
    BkfSysLogTableType *tableType = BkfSysLogTableTypeGet(sysLogMng, typeId);
    if (tableType == VOS_NULL) {
        return BKF_ERR;
    }

    if (!tableType->vTbl.needRestrain) {
        (void)tableType->vTbl.out(tableType->vTbl.cookie, key, val, sysLogMng->argInit.printf,
            sysLogMng->argInit.printfParam1);
        return BKF_OK;
    }

    BkfSysLog *sysLog = BkfSysLogGet(tableType, key);
    if (sysLog == VOS_NULL) {
        sysLog = BkfSysLogCreate(sysLogMng, tableType, key, val);
        if (sysLog == VOS_NULL) {
            return BKF_ERR;
        }
        (void)tableType->vTbl.out(tableType->vTbl.cookie, key, val, sysLogMng->argInit.printf,
            sysLogMng->argInit.printfParam1);
    }

    return BKF_OK;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
