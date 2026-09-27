/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_disp.h"
#include "bkf_dl.h"
#include "bkf_str.h"
#include "bkf_assert.h"
#include "v_avll.h"
#include "securec.h"
#include "v_stringlib.h"
#include "vos_errno.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
enum {
    BKF_DISP_CNT_TEST,

    BKF_DISP_CNT_REG_OBJ_NAME_NULL,
    BKF_DISP_CNT_REG_OBJ_NAME_INVALID,
    BKF_DISP_CNT_REG_OBJ_ADDR_NULL,
    BKF_DISP_CNT_REG_OBJ_HAS_REGED,
    BKF_DISP_CNT_REG_OBJ_ADD_NG,
    BKF_DISP_CNT_REG_OBJ_OK,
    BKF_DISP_CNT_UNREG_OBJ_NFIND_NG,
    BKF_DISP_CNT_UNREG_OBJ_OK,

    BKF_DISP_CNT_REG_FUNC_NAME_NULL,
    BKF_DISP_CNT_REG_FUNC_ADDR_NULL,
    BKF_DISP_CNT_REG_FUNC_HELP_INFO_NULL,
    BKF_DISP_CNT_REG_FUNC_OBJ_NAME_NULL,
    BKF_DISP_CNT_REG_FUNC_OBJ_NOT_REG,
    BKF_DISP_CNT_REG_FUNC_PARAM_CNT_NG,
    BKF_DISP_CNT_REG_FUNC_PARAM_X_NULL,
    BKF_DISP_CNT_REG_FUNC_PARAM_X_NG,
    BKF_DISP_CNT_REG_FUNC_ADD_NG,
    BKF_DISP_CNT_REG_FUNC_OBJ_NAME_HAS_REGED,
    BKF_DISP_CNT_REG_FUNC_OBJ_NAME_ADD_NG,
    BKF_DISP_CNT_REG_FUNC_OK,

    BKF_DISP_CNT_PROC_CTX_NULL,
    BKF_DISP_CNT_PROC_IN_STR_NULL,
    BKF_DISP_CNT_PROC_IN_STR_CNT_NG,
    BKF_DISP_CNT_PROC_IN_STR_X_NULL,
    BKF_DISP_CNT_PROC_OUT_BUF_NULL,
    BKF_DISP_CNT_PROC_OUT_BUF_LEN_NG,
    BKF_DISP_CNT_PROC_NO_VALID_FUNC_FIND,
    BKF_DISP_CNT_PROC_PARAM_CNT_NG,
    BKF_DISP_CNT_PROC_PARAM_X_NG,
    BKF_DISP_CNT_PROC_OBJ_NAME_NULL,
    BKF_DISP_CNT_PROC_OBJ_NAME_NOT_REG,
    BKF_DISP_CNT_PROC_APP_CALL,

    BKF_DISP_CNT_PRINTF_OUT_BUF_NULL,
    BKF_DISP_CNT_PRINTF_FMT_NULL,
    BKF_DISP_CNT_PRINTF_OUT_STR_BUF_NOT_ENOUGH,

    BKF_DISP_CNT_SAVE_CTX_NOT_IN_CB,
    BKF_DISP_CNT_SAVE_CTX_OUT_STR_BUF_NOT_ENOUGH,
    BKF_DISP_CNT_SAVE_CTX1_PARAM_NG,
    BKF_DISP_CNT_SAVE_CTX2_PARAM_NG,
    BKF_DISP_CNT_SAVE_CTX_TOTAL_LEN_NG,
    BKF_DISP_CNT_SAVE_CTX1_NG,
    BKF_DISP_CNT_SAVE_CTX2_NG,
    BKF_DISP_CNT_SAVE_CTX_OK,

    BKF_DISP_CNT_GET_CTX_NOT_IN_CB,
    BKF_DISP_CNT_GET_CTX_NULL,
    BKF_DISP_CNT_GET_CTX2_NG,
    BKF_DISP_CNT_GET_CTX_OK,
    BKF_DISP_CNT_GET_CTX_VAL1_NULL,

    BKF_DISP_CNT_MAX
};

#define BKF_DISP_NAME_INVALID_CHAR "* %\\"

/* disp */
struct tagBkfDisp {
    BkfDispInitArg argInit;
    char *name;
    AVLL_TREE funcSet;
    BkfDl funcSetByTime;
    AVLL_TREE objSet;

    /* func proc info */
    char *outStrBuf;
    int32_t outStrBufLen;
    BkfDispProcCtx *ctx;
    uint8_t inCb : 1;
    uint8_t lastCbSaveCtx : 1;
    uint8_t curCbSaveCtx : 1;
    uint8_t curCbGetCtx : 1;
    uint8_t outStrBufNotEnough : 1;
    uint8_t appCbMaybeDeadloop : 1;
    uint8_t rsv : 2;
    uint8_t pad1[3];
    int32_t prinfTotalLen;
    int32_t prinfValidLen;

    /* stat */
    uint32_t statCnt[BKF_DISP_CNT_MAX];
};
#define BKF_DISP_INC_CNT(disp, whichCnt)                                                                               \
    do {                                                                                                               \
        if (((disp) != VOS_NULL) && ((whichCnt) < BKF_DISP_CNT_MAX)) {                                                 \
            (disp)->statCnt[(whichCnt)]++;                                                                             \
        }                                                                                                              \
    } while (0)

/* func */
#define BKF_DISP_FUNC_PARAM_CNT 5
typedef struct tagBkfDispFunc {
    AVLL_NODE avlNode;
    BkfDlNode dlNode;
    AVLL_TREE funcObjNameSet;

    /* func */
    F_BKF_DISP_FUNC funcAddr;
    const char *funcHelpInfo;
    uint8_t funcLvl;
    uint8_t funcParamCnt;
    uint8_t pad1[2];
    const char *funcFormalParam[BKF_DISP_FUNC_PARAM_CNT];

    /* stat */
    uint32_t appCallCnt; /* app调用此函数的次数 */
    uint32_t funcCbCnt;  /* 函数被回调的次数。app一次调用中，cb会被多次执行 */
    char funcName[0];
} BkfDispFunc;

#define BKF_DISP_FUNC_MAY_USE(disp, dispFunc) ((dispFunc)->funcLvl >= (disp)->argInit.mayProcFuncLvl)

typedef struct tagBkfDispFuncObjName {
    BkfDispFunc *dispFunc;
    AVLL_NODE avlNode;
    BkfDlNode dlNode;
    char objName[0];
} BkfDispFuncObjName;

/* reg obj */
typedef struct tagBkfDispObj {
    AVLL_NODE avlNode;
    BkfDl funcObjNameSet;
    void *objAddr;
    char objName[0];
} BkfDispObj;

#define BKF_DISP_APP_CALL_ONCE_FUNC_CB_TIME_MAX (BKF_1K * 100)
#define BKF_DISP_OUT_BUF_RSV_LEN_FOR_LAST_PROMPT (100)
#define BKF_DISP_APP_CALL_FIRST_TIME(ctx) ((ctx)->ctx1Len <= 0)

/* on msg & proc */
STATIC BOOL BkfDispFuncRealParam2Str(char *realParamStr, uintptr_t *outVal);
STATIC BOOL BkfDispFuncRealParam2Num(char *realParamStr, uintptr_t *outVal);

/* data op */
STATIC BkfDispFunc *BkfDispAddFunc(BkfDisp *disp, char *funcName, F_BKF_DISP_FUNC funcAddr, const char *funcHelpInfo,
    uint8_t funcLvl, uint8_t funcParamCnt, char **funcFormalParam);
STATIC void BkfDispDelFunc(BkfDisp *disp, BkfDispFunc *dispFunc);
STATIC void BkfDispDelAllFunc(BkfDisp *disp);
STATIC BkfDispFunc *BkfDispFindFunc(BkfDisp *disp, char *funcName);
STATIC BkfDispFunc *BkfDispFindNextFunc(BkfDisp *disp, char *funcName);
STATIC BkfDispFunc *BkfDispGetFirstFuncByTime(BkfDisp *disp, void **itorOutOrNull);
STATIC BkfDispFunc *BkfDispGetNextFuncByTime(BkfDisp *disp, void **itorInOut);

STATIC BkfDispFuncObjName *BkfDispAddFuncObjName(BkfDisp *disp, BkfDispFunc *dispFunc, BkfDispObj *dispObj);
STATIC void BkfDispDelFuncObjName(BkfDisp *disp, BkfDispFuncObjName *funcObjName);
STATIC void BkfDispDelFuncAllObjName(BkfDisp *disp, BkfDispFunc *dispFunc);
STATIC BkfDispFuncObjName *BkfDispFindFuncObjName(BkfDisp *disp, BkfDispFunc *dispFunc, char *objName);
STATIC BkfDispFuncObjName *BkfDispFindNextFuncObjName(BkfDisp *disp, BkfDispFunc *dispFunc, char *objName);
STATIC BkfDispFuncObjName *BkfDispGetFirstFuncObjNameOfFunc(BkfDisp *disp, BkfDispFunc *dispFunc, void **itorOutOrNull);
STATIC BkfDispFuncObjName *BkfDispGetNextFuncObjNameOfFunc(BkfDisp *disp, BkfDispFunc *dispFunc, void **itorInOut);
STATIC BkfDispFuncObjName *BkfDispGetFirstFuncObjNameOfObj(BkfDisp *disp, BkfDispObj *dispObj, void **itorOutOrNull);
STATIC BkfDispFuncObjName *BkfDispGetNextFuncObjNameOfObj(BkfDisp *disp, BkfDispObj *dispObj, void **itorInOut);

STATIC BkfDispObj *BkfDispAddObj(BkfDisp *disp, char *objName, void *objAddr);
STATIC void BkfDispDelObj(BkfDisp *disp, BkfDispObj *dispObj);
STATIC void BkfDispDelAllObj(BkfDisp *disp);
STATIC BkfDispObj *BkfDispFindObj(BkfDisp *disp, char *objName);
STATIC BkfDispObj *BkfDispFindNextObj(BkfDisp *disp, char *objName);
STATIC BkfDispObj *BkfDispGetFirstObj(BkfDisp *disp, void **itorOutOrNull);
STATIC BkfDispObj *BkfDispGetNextObj(BkfDisp *disp, void **itorInOut);

/* disp */
STATIC void BkfDispInitSelfDispFunc(BkfDisp *disp);
STATIC void BkfDispUninitSelfDispFunc(BkfDisp *disp);

#pragma pack()
#endif

#if BKF_BLOCK("表")
#pragma pack(4)
typedef uint32_t (*F_BKF_DISP_FUNC_REAL_PARAM_STR_2VAL)(char *realParamStr, uintptr_t *outVal);
typedef struct tagBkfDispFuncParamType {
    const char *formalParamPrefix;
    F_BKF_DISP_FUNC_REAL_PARAM_STR_2VAL realParamStr2Val;
} BkfDispFuncParamType;

const BkfDispFuncParamType g_BkfDispFuncParamType[] = {
    {"pc", BkfDispFuncRealParam2Str},
    {"uni", BkfDispFuncRealParam2Num},
};

const char *g_BkfDispStatStrTbl[] = {
    "BKF_DISP_CNT_TEST",
    "BKF_DISP_CNT_REG_OBJ_NAME_NULL",
    "BKF_DISP_CNT_REG_OBJ_NAME_INVALID",
    "BKF_DISP_CNT_REG_OBJ_ADDR_NULL",
    "BKF_DISP_CNT_REG_OBJ_HAS_REGED",
    "BKF_DISP_CNT_REG_OBJ_ADD_NG",
    "BKF_DISP_CNT_REG_OBJ_OK",
    "BKF_DISP_CNT_UNREG_OBJ_NFIND_NG",
    "BKF_DISP_CNT_UNREG_OBJ_OK",
    "BKF_DISP_CNT_REG_FUNC_NAME_NULL",
    "BKF_DISP_CNT_REG_FUNC_ADDR_NULL",
    "BKF_DISP_CNT_REG_FUNC_HELP_INFO_NULL",
    "BKF_DISP_CNT_REG_FUNC_OBJ_NAME_NULL",
    "BKF_DISP_CNT_REG_FUNC_OBJ_NOT_REG",
    "BKF_DISP_CNT_REG_FUNC_PARAM_CNT_NG",
    "BKF_DISP_CNT_REG_FUNC_PARAM_X_NULL",
    "BKF_DISP_CNT_REG_FUNC_PARAM_X_NG",
    "BKF_DISP_CNT_REG_FUNC_ADD_NG",
    "BKF_DISP_CNT_REG_FUNC_OBJ_NAME_HAS_REGED",
    "BKF_DISP_CNT_REG_FUNC_OBJ_NAME_ADD_NG",
    "BKF_DISP_CNT_REG_FUNC_OK",
    "BKF_DISP_CNT_PROC_CTX_NULL",
    "BKF_DISP_CNT_PROC_IN_STR_NULL",
    "BKF_DISP_CNT_PROC_IN_STR_CNT_NG",
    "BKF_DISP_CNT_PROC_IN_STR_X_NULL",
    "BKF_DISP_CNT_PROC_OUT_BUF_NULL",
    "BKF_DISP_CNT_PROC_OUT_BUF_LEN_NG",
    "BKF_DISP_CNT_PROC_NO_VALID_FUNC_FIND",
    "BKF_DISP_CNT_PROC_PARAM_CNT_NG",
    "BKF_DISP_CNT_PROC_PARAM_X_NG",
    "BKF_DISP_CNT_PROC_OBJ_NAME_NULL",
    "BKF_DISP_CNT_PROC_OBJ_NAME_NOT_REG",
    "BKF_DISP_CNT_PROC_APP_CALL",
    "BKF_DISP_CNT_PRINTF_OUT_BUF_NULL",
    "BKF_DISP_CNT_PRINTF_FMT_NULL",
    "BKF_DISP_CNT_PRINTF_OUT_STR_BUF_NOT_ENOUGH",
    "BKF_DISP_CNT_SAVE_CTX_NOT_IN_CB",
    "BKF_DISP_CNT_SAVE_CTX_OUT_STR_BUF_NOT_ENOUGH",
    "BKF_DISP_CNT_SAVE_CTX1_PARAM_NG",
    "BKF_DISP_CNT_SAVE_CTX2_PARAM_NG",
    "BKF_DISP_CNT_SAVE_CTX_TOTAL_LEN_NG",
    "BKF_DISP_CNT_SAVE_CTX1_NG",
    "BKF_DISP_CNT_SAVE_CTX2_NG",
    "BKF_DISP_CNT_SAVE_CTX_OK",
    "BKF_DISP_CNT_GET_CTX_NOT_IN_CB",
    "BKF_DISP_CNT_GET_CTX_NULL",
    "BKF_DISP_CNT_GET_CTX2_NG",
    "BKF_DISP_CNT_GET_CTX_OK",
    "BKF_DISP_CNT_GET_CTX_VAL1_NULL",
};
BKF_STATIC_ASSERT(BKF_GET_ARY_COUNT(g_BkfDispStatStrTbl) == BKF_DISP_CNT_MAX);

#pragma pack()
#endif

#if BKF_BLOCK("公有函数定义")
BkfDisp *BkfDispInit(BkfDispInitArg *arg)
{
    BOOL paramIsInvalid = VOS_FALSE;
    BkfDisp *disp = VOS_NULL;
    uint32_t len;

    paramIsInvalid = (arg == VOS_NULL) || (arg->name == VOS_NULL) || (arg->memMng == VOS_NULL);
    if (paramIsInvalid) {
        goto error;
    }

    len = sizeof(BkfDisp);
    disp = BKF_MALLOC(arg->memMng, len);
    if (disp == VOS_NULL) {
        goto error;
    }
    (void)memset_s(disp, len, 0, len);
    disp->argInit = *arg;
    disp->name = BkfStrNew(arg->memMng, "%s_disp", arg->name);
    disp->argInit.name = disp->name;
    BKF_DL_INIT(&disp->funcSetByTime);
    VOS_AVLL_INIT_TREE(disp->funcSet, (AVLL_COMPARE)VOS_StrCmp, BKF_OFFSET(BkfDispFunc, funcName[0]),
        BKF_OFFSET(BkfDispFunc, avlNode));
    VOS_AVLL_INIT_TREE(disp->objSet, (AVLL_COMPARE)VOS_StrCmp, BKF_OFFSET(BkfDispObj, objName[0]),
        BKF_OFFSET(BkfDispObj, avlNode));
    BkfDispInitSelfDispFunc(disp);

    return disp;

error:

    BkfDispUninit(disp);
    return VOS_NULL;
}

void BkfDispUninit(BkfDisp *disp)
{
    if (disp == VOS_NULL) {
        return;
    }

    BkfDispDelAllFunc(disp);
    BkfDispDelAllObj(disp);
    BkfDispUninitSelfDispFunc(disp);
    BkfStrDel(disp->name);
    BKF_FREE(disp->argInit.memMng, disp);
    return;
}

STATIC uint32_t BkfDispRegObjChkParam(BkfDisp *disp, char *objName, void *objAddr)
{
    if (disp == VOS_NULL) {
        return BKF_ERR;
    }
    if (objName == VOS_NULL) {
        return BKF_DISP_CNT_REG_OBJ_NAME_NULL;
    }
    if ((VOS_StrLen(objName) == 0) || (VOS_StrpBrk(objName, BKF_DISP_NAME_INVALID_CHAR) != VOS_NULL)) {
        return BKF_DISP_CNT_REG_OBJ_NAME_INVALID;
    }
    if (objAddr == VOS_NULL) {
        return BKF_DISP_CNT_REG_OBJ_ADDR_NULL;
    }

    return BKF_OK;
}
uint32_t BkfDispRegObj(BkfDisp *disp, char *objName, void *objAddr)
{
    BkfDispObj *dispObj = VOS_NULL;
    uint32_t ret;

    ret = BkfDispRegObjChkParam(disp, objName, objAddr);
    if (ret != BKF_OK) {
        BKF_DISP_INC_CNT(disp, ret);
        return ret;
    }

    dispObj = BkfDispFindObj(disp, objName);
    if (dispObj != VOS_NULL) {
        if (dispObj->objAddr == objAddr) {
            return BKF_OK;
        } else {
            BKF_DISP_INC_CNT(disp, BKF_DISP_CNT_REG_OBJ_HAS_REGED);
            return BKF_ERR;
        }
    }
    dispObj = BkfDispAddObj(disp, objName, objAddr);
    if (dispObj == VOS_NULL) {
        return BKF_ERR;
    }

    BKF_DISP_INC_CNT(disp, BKF_DISP_CNT_REG_OBJ_OK);
    return VOS_OK;
}

void BkfDispUnregObj(BkfDisp *disp, char *objName)
{
    BOOL paramIsInvalid = VOS_FALSE;
    BkfDispObj *dispObj = VOS_NULL;
    BkfDispFuncObjName *funcObjName = VOS_NULL;
    void *itor = VOS_NULL;
    BkfDispFunc *dispFunc = VOS_NULL;
    BOOL mayDel = FALSE;

    paramIsInvalid = (disp == VOS_NULL) || (objName == VOS_NULL);
    if (paramIsInvalid) {
        return;
    }

    dispObj = BkfDispFindObj(disp, objName);
    if (dispObj == VOS_NULL) {
        BKF_DISP_INC_CNT(disp, BKF_DISP_CNT_UNREG_OBJ_NFIND_NG);
        return;
    }
    for (funcObjName = BkfDispGetFirstFuncObjNameOfObj(disp, dispObj, &itor); funcObjName != VOS_NULL;
         funcObjName = BkfDispGetNextFuncObjNameOfObj(disp, dispObj, &itor)) {
        dispFunc = funcObjName->dispFunc;
        BkfDispDelFuncObjName(disp, funcObjName);
        mayDel = (BkfDispGetFirstFuncObjNameOfFunc(disp, dispFunc, VOS_NULL) == VOS_NULL);
        if (mayDel) {
            BkfDispDelFunc(disp, dispFunc);
        }
    }
    BkfDispDelObj(disp, dispObj);

    BKF_DISP_INC_CNT(disp, BKF_DISP_CNT_UNREG_OBJ_OK);
    return;
}

STATIC uint32_t BkfDispRegFuncChkParam(BkfDisp *disp, char *funcName, F_BKF_DISP_FUNC funcAddr,
    const char *funcHelpInfo, uint8_t funcLvl, char *funcMayProcObjName, uint8_t funcParamCnt, BkfDispObj **findDispObj)
{
    (void)funcLvl;
    BkfDispObj *dispObj = VOS_NULL;

    if (disp == VOS_NULL) {
        return BKF_ERR;
    }
    if (funcName == VOS_NULL) {
        return BKF_DISP_CNT_REG_FUNC_NAME_NULL;
    }
    if (funcAddr == VOS_NULL) {
        return BKF_DISP_CNT_REG_FUNC_ADDR_NULL;
    }
    if (funcHelpInfo == VOS_NULL) {
        return BKF_DISP_CNT_REG_FUNC_HELP_INFO_NULL;
    }
    if (funcMayProcObjName == VOS_NULL) {
        return BKF_DISP_CNT_REG_FUNC_OBJ_NAME_NULL;
    }
    dispObj = BkfDispFindObj(disp, funcMayProcObjName);
    if (dispObj == VOS_NULL) {
        return BKF_DISP_CNT_REG_FUNC_OBJ_NOT_REG;
    }
    *findDispObj = dispObj;
    if (funcParamCnt > BKF_DISP_FUNC_PARAM_CNT) {
        return BKF_DISP_CNT_REG_FUNC_PARAM_CNT_NG;
    }

    return BKF_OK;
}
STATIC BkfDispFuncParamType *BkfDispGetFuncFormalParamType(const char *formalParam)
{
    uint32_t i;
    char *find = VOS_NULL;
    BOOL findAtBegin = VOS_FALSE;

    if (VOS_StrpBrk(formalParam, BKF_DISP_NAME_INVALID_CHAR) != VOS_NULL) {
        return VOS_NULL;
    }

    for (i = 0; i < BKF_GET_ARY_COUNT(g_BkfDispFuncParamType); i++) {
        find = VOS_StrStr(formalParam, g_BkfDispFuncParamType[i].formalParamPrefix);
        findAtBegin = (find == formalParam);
        if (findAtBegin) {
            return (BkfDispFuncParamType *)&g_BkfDispFuncParamType[i];
        }
    }

    return VOS_NULL;
}
STATIC BOOL BkfDispRegFuncFormalParamIsValid(const char *formalParam)
{
    return (BkfDispGetFuncFormalParamType(formalParam) != VOS_NULL);
}
STATIC uint32_t BkfDispRegFuncAddFunc(BkfDisp *disp, char *funcName, F_BKF_DISP_FUNC funcAddr, const char *funcHelpInfo,
    uint8_t funcLvl, BkfDispObj *dispObj, uint8_t funcParamCnt, char **formalParam)
{
    BkfDispFunc *dispFunc = VOS_NULL;
    BkfDispFuncObjName *funcObjName = VOS_NULL;

    dispFunc = BkfDispFindFunc(disp, funcName);
    if (dispFunc == VOS_NULL) {
        dispFunc = BkfDispAddFunc(disp, funcName, funcAddr, funcHelpInfo, funcLvl, funcParamCnt, formalParam);
        if (dispFunc == VOS_NULL) {
            return BKF_DISP_CNT_REG_FUNC_ADD_NG;
        }
    }

    funcObjName = BkfDispFindFuncObjName(disp, dispFunc, dispObj->objName);
    if (funcObjName != VOS_NULL) {
        return BKF_DISP_CNT_REG_FUNC_OBJ_NAME_HAS_REGED;
    }
    funcObjName = BkfDispAddFuncObjName(disp, dispFunc, dispObj);
    if (funcObjName == VOS_NULL) {
        return BKF_DISP_CNT_REG_FUNC_OBJ_NAME_ADD_NG;
    }

    return BKF_OK;
}
uint32_t BkfDispRegFunc(BkfDisp *disp, char *funcName, F_BKF_DISP_FUNC funcAddr, const char *funcHelpInfo,
    uint8_t funcLvl, char *funcMayProcObjName, uint8_t funcParamCnt, ...)
{
    BkfDispObj *dispObj = VOS_NULL;
    uint32_t ret;
    va_list va;
    uint32_t i;
    char *formalParam[BKF_DISP_FUNC_PARAM_CNT] = {VOS_NULL};
    BOOL formalParamIsValid = VOS_TRUE;

    ret = BkfDispRegFuncChkParam(disp, funcName, funcAddr, funcHelpInfo, funcLvl, funcMayProcObjName, funcParamCnt,
        &dispObj);
    if (ret != BKF_OK) {
        BKF_DISP_INC_CNT(disp, ret);
        return ret;
    }

    formalParamIsValid = VOS_TRUE;
    va_start(va, funcParamCnt);
    for (i = 0; i < funcParamCnt; i++) {
        formalParam[i] = (char *)va_arg(va, char *);
        if (formalParam[i] == VOS_NULL) {
            BKF_DISP_INC_CNT(disp, BKF_DISP_CNT_REG_FUNC_PARAM_X_NULL);
            formalParamIsValid = VOS_FALSE;
            break;
        }
        if (!BkfDispRegFuncFormalParamIsValid(formalParam[i])) {
            BKF_DISP_INC_CNT(disp, BKF_DISP_CNT_REG_FUNC_PARAM_X_NG);
            formalParamIsValid = VOS_FALSE;
            break;
        }
    }
    va_end(va);
    if (!formalParamIsValid) {
        return BKF_ERR;
    }

    ret = BkfDispRegFuncAddFunc(disp, funcName, funcAddr, funcHelpInfo, funcLvl, dispObj, funcParamCnt, formalParam);
    if (ret != BKF_OK) {
        BKF_DISP_INC_CNT(disp, ret);
        return ret;
    }

    BKF_DISP_INC_CNT(disp, BKF_DISP_CNT_REG_FUNC_OK);
    return BKF_OK;
}

STATIC uint32_t BkfDispProcChkParam(BkfDisp *disp, char **inStr, uint8_t inStrCnt, char *outStrBuf,
    int32_t outStrBufLen, BkfDispProcCtx *ctx)
{
    uint8_t i;

    if (ctx == VOS_NULL) {
        BKF_DISP_PRINTF(disp, "ctx(%#x), ng\n", BKF_MASK_ADDR(ctx));
        return BKF_DISP_CNT_PROC_CTX_NULL;
    }
    if (BKF_DISP_APP_CALL_FIRST_TIME(ctx)) {
        BKF_DISP_PRINTF(disp, "====================.....\n");
    }
    if (outStrBuf == VOS_NULL) {
        BKF_DISP_PRINTF(disp, "outStrBuf(%#x) null, ng\n", BKF_MASK_ADDR(outStrBuf));
        return BKF_DISP_CNT_PROC_OUT_BUF_NULL;
    }
    if (outStrBufLen < BKF_DISP_PROC_OUT_BUF_LEN_MIN) {
        BKF_DISP_PRINTF(disp, "outStrBufLen(%d), ng\n", outStrBufLen);
        return BKF_DISP_CNT_PROC_OUT_BUF_LEN_NG;
    }
    if (inStr == VOS_NULL) {
        BKF_DISP_PRINTF(disp, "inStr(%#x) null, ng\n", BKF_MASK_ADDR(inStr));
        return BKF_DISP_CNT_PROC_IN_STR_NULL;
    }
    if (inStrCnt < 1) {
        BKF_DISP_PRINTF(disp, "inStrCnt(%d), no funcName, ng\n", inStrCnt);
        return BKF_DISP_CNT_PROC_IN_STR_CNT_NG;
    }
    for (i = 0; i < inStrCnt; i++) {
        if (inStr[i] == VOS_NULL) {
            BKF_DISP_PRINTF(disp, "inStr[%d](%#x) null, ng\n", i, BKF_MASK_ADDR(inStr[i]));
            return BKF_DISP_CNT_PROC_IN_STR_X_NULL;
        }
    }

    return VOS_OK;
}
STATIC BkfDispFunc *BkfDispProcParseDispFunc(BkfDisp *disp, char **inStr, uint8_t inStrCnt)
{
    (void)inStrCnt;
    char *funcName = VOS_NULL;
    BkfDispFunc *dispFunc = VOS_NULL;

    funcName = inStr[0];
    dispFunc = BkfDispFindFunc(disp, funcName);
    if ((dispFunc == VOS_NULL) || !BKF_DISP_FUNC_MAY_USE(disp, dispFunc)) {
        BKF_DISP_INC_CNT(disp, BKF_DISP_CNT_PROC_NO_VALID_FUNC_FIND);
        BKF_DISP_PRINTF(disp, "[%s(^)], no valid func found, BkfDispHelp for help\n", funcName);
        return VOS_NULL;
    }

    return dispFunc;
}
STATIC uint32_t BkfDispProcParseFuncOneParam(BkfDisp *disp, BkfDispFunc *dispFunc, uint8_t paramIdx, char *realParamStr,
    uintptr_t *realParamVal)
{
    (void)disp;
    BkfDispFuncParamType *paramType = VOS_NULL;
    BOOL str2ValSucc = VOS_FALSE;

    paramType = BkfDispGetFuncFormalParamType(dispFunc->funcFormalParam[paramIdx]);
    if (paramType == VOS_NULL) {
        return BKF_ERR;
    }

    str2ValSucc = paramType->realParamStr2Val(realParamStr, realParamVal);
    if (!str2ValSucc) {
        return BKF_ERR;
    }

    return BKF_OK;
}
STATIC uint32_t BkfDispProcParseFuncParam(BkfDisp *disp, char **inStr, uint8_t inStrCnt, BkfDispFunc *dispFunc,
    uintptr_t *realParamVal, uint8_t realParamValCnt)
{
    uint8_t i;
    uint8_t j;
    char *realParamStr = VOS_NULL;

    /* funcName, funcParam1, funcParam2, .. <funcProcObjName> */
    if (inStrCnt < (dispFunc->funcParamCnt + 1)) {
        BKF_DISP_INC_CNT(disp, BKF_DISP_CNT_PROC_PARAM_CNT_NG);
        BKF_DISP_PRINTF(disp, "[%s], reg %d param(s), count ng\n", dispFunc->funcName, dispFunc->funcParamCnt);
        return BKF_ERR;
    }

    if (dispFunc->funcParamCnt > realParamValCnt) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }
    for (i = 0; i < dispFunc->funcParamCnt; i++) {
        realParamStr = inStr[i + 1];
        if (BkfDispProcParseFuncOneParam(disp, dispFunc, i, realParamStr, &realParamVal[i]) != VOS_OK) {
            BKF_DISP_INC_CNT(disp, BKF_DISP_CNT_PROC_PARAM_X_NG);
            BKF_DISP_PRINTF(disp, "[%s]", dispFunc->funcName);
            for (j = 0; j < i; j++) {
                BKF_DISP_PRINTF(disp, " %s", inStr[j + 1]);
            }
            BKF_DISP_PRINTF(disp, " %s(^), this param ng\n", realParamStr);
            return BKF_ERR;
        }
    }

    return BKF_OK;
}
STATIC BkfDispObj *BkfDispProcParseDispObj(BkfDisp *disp, char **inStr, uint8_t inStrCnt, BkfDispFunc *dispFunc)
{
    BkfDispFuncObjName *funcObjName = VOS_NULL;
    char *objName = VOS_NULL;
    BkfDispObj *dispObj = VOS_NULL;
    /* 如果没有输入funcProcObjName， 取func注册过的obj最小的；否则，查找是否存在 */
    if (inStrCnt == (dispFunc->funcParamCnt + 1)) {
        funcObjName = BkfDispGetFirstFuncObjNameOfFunc(disp, dispFunc, VOS_NULL);
    } else {
        char *inputObjName = inStr[dispFunc->funcParamCnt + 1];
        funcObjName = BkfDispFindFuncObjName(disp, dispFunc, inputObjName);
    }

    if (funcObjName != VOS_NULL) {
        objName = funcObjName->objName;
        dispObj = BkfDispFindObj(disp, objName);
    }

    if (dispObj == VOS_NULL) {
        BKF_DISP_INC_CNT(disp, BKF_DISP_CNT_PROC_OBJ_NAME_NOT_REG);
        BKF_DISP_PRINTF(disp, "[%s]", dispFunc->funcName);
        uint8_t j = 0;
        for (j = 0; j < dispFunc->funcParamCnt; j++) {
            BKF_DISP_PRINTF(disp, " %s", inStr[j + 1]);
        }
        BKF_DISP_PRINTF(disp, " %s(^^), obj not reg to this func\n", objName);
        return VOS_NULL;
    }

    return dispObj;
}
STATIC void BkfDispProcCallFuncForPara(BkfDisp *disp, BkfDispFunc *dispFunc, uintptr_t *realParamVal,
    BkfDispObj *dispObj)
{
    (void)disp;
    switch (dispFunc->funcParamCnt) {
        case 0: {
            dispFunc->funcAddr(dispObj->objAddr);
            break;
        }
        case 1: {
            dispFunc->funcAddr(dispObj->objAddr, realParamVal[0]);
            break;
        }
        case 2: { /* 2 param */
            dispFunc->funcAddr(dispObj->objAddr, realParamVal[0], realParamVal[1]);
            break;
        }
        case 3: {                                                                                    /* 3 param */
            dispFunc->funcAddr(dispObj->objAddr, realParamVal[0], realParamVal[1], realParamVal[2]); /* param 2 */
            break;
        }
        case 4: {                                                                                   /* 4 param */
            dispFunc->funcAddr(dispObj->objAddr, realParamVal[0], realParamVal[1], realParamVal[2], /* param 2 */
                realParamVal[3]);                                                                   /* param 3 */
            break;
        }
        case 5: {                                                                                   /* 5 param */
            dispFunc->funcAddr(dispObj->objAddr, realParamVal[0], realParamVal[1], realParamVal[2], /* param 2 */
                realParamVal[3], realParamVal[4]);                                                  /* param 3, 4 */
            break;
        }
        default: {
            /* do nothing */
            break;
        }
    }

    return;
}
STATIC void BkfDispProcCallFunc(BkfDisp *disp, BkfDispFunc *dispFunc, uintptr_t *realParamVal, uint8_t realParamValCnt,
    BkfDispObj *dispObj)
{
    (void)realParamValCnt;
    int32_t i;
    BOOL lastCtxNotUse = VOS_FALSE;
    BOOL maybeDeadLoop = VOS_FALSE;
    BOOL needCbAgain = VOS_FALSE;

    if (BKF_DISP_APP_CALL_FIRST_TIME(disp->ctx)) {
        BKF_DISP_PRINTF(disp, "[%s]...\n", dispObj->objName);
    }

    disp->inCb = VOS_TRUE;
    BKF_DISP_INC_CNT(disp, BKF_DISP_CNT_PROC_APP_CALL);
    dispFunc->appCallCnt++;
    for (i = 0; i < BKF_DISP_APP_CALL_ONCE_FUNC_CB_TIME_MAX; i++) {
        dispFunc->funcCbCnt++;
        BkfDispProcCallFuncForPara(disp, dispFunc, realParamVal, dispObj);

        lastCtxNotUse = disp->lastCbSaveCtx && !disp->curCbGetCtx;
        if (lastCtxNotUse) {
            break;
        }

        maybeDeadLoop = (i == BKF_DISP_APP_CALL_ONCE_FUNC_CB_TIME_MAX - 1);
        if (maybeDeadLoop) {
            disp->appCbMaybeDeadloop = VOS_TRUE;
            break;
        }

        needCbAgain = disp->curCbSaveCtx && !disp->outStrBufNotEnough;
        if (!needCbAgain) {
            break;
        }

        disp->lastCbSaveCtx = disp->curCbSaveCtx;
        disp->curCbSaveCtx = VOS_FALSE;
        disp->curCbGetCtx = VOS_FALSE;
    }
    disp->inCb = VOS_FALSE;

    return;
}
STATIC uint32_t BkfDispProcFillRet(BkfDisp *disp)
{
    if (disp->lastCbSaveCtx && !disp->curCbGetCtx) {
        BKF_DISP_PRINTF(disp, "\nhas context, but not get this time, pls chk\n");
        return BKF_OK;
    } else if (disp->appCbMaybeDeadloop) {
        BKF_DISP_PRINTF(disp, "\nloop too many times & maybe deadloop, pls chk\n");
        return BKF_OK;
    } else if (disp->outStrBufNotEnough) {
        if (disp->prinfValidLen <= 0) {
            BKF_DISP_PRINTF(disp, "\nnot save context before out buf not enough and drop some data, pls chk\n");
            return BKF_OK;
        } else {
            return BKF_DISP_PROC_HAS_MORE_DATA;
        }
    } else {
        BKF_DISP_PRINTF(disp, "---[end]-----------------\n");
        return BKF_OK;
    }
}
uint32_t BkfDispProc(BkfDisp *disp, char **inStr, uint8_t inStrCnt, char *outStrBuf, int32_t outStrBufLen,
    BkfDispProcCtx *ctx)
{
    uint32_t ret;
    BkfDispFunc *dispFunc = VOS_NULL;
    uintptr_t realParamVal[BKF_DISP_FUNC_PARAM_CNT] = {0};
    BkfDispObj *dispObj = VOS_NULL;

    if (disp == VOS_NULL) {
        return BKF_ERR;
    }
    disp->outStrBuf = outStrBuf;
    disp->outStrBuf[0] = '\0';
    disp->outStrBufLen = outStrBufLen;
    disp->ctx = ctx;
    disp->inCb = VOS_FALSE;
    disp->lastCbSaveCtx = VOS_FALSE;
    disp->curCbSaveCtx = VOS_FALSE;
    disp->curCbGetCtx = VOS_FALSE;
    disp->outStrBufNotEnough = VOS_FALSE;
    disp->appCbMaybeDeadloop = VOS_FALSE;
    disp->prinfTotalLen = 0;
    disp->prinfValidLen = 0;
    ret = BkfDispProcChkParam(disp, inStr, inStrCnt, outStrBuf, outStrBufLen, ctx);
    if (ret != BKF_OK) {
        BKF_DISP_INC_CNT(disp, ret);
        return BKF_OK;
    }

    dispFunc = BkfDispProcParseDispFunc(disp, inStr, inStrCnt);
    if (dispFunc == VOS_NULL) {
        return BKF_OK;
    }
    ret = BkfDispProcParseFuncParam(disp, inStr, inStrCnt, dispFunc, realParamVal, BKF_GET_ARY_COUNT(realParamVal));
    if (ret != BKF_OK) {
        return BKF_OK;
    }
    dispObj = BkfDispProcParseDispObj(disp, inStr, inStrCnt, dispFunc);
    if (dispObj == VOS_NULL) {
        return BKF_OK;
    }
    BkfDispProcCallFunc(disp, dispFunc, realParamVal, BKF_GET_ARY_COUNT(realParamVal), dispObj);
    ret = BkfDispProcFillRet(disp);

    disp->outStrBuf = VOS_NULL;
    return ret;
}

STATIC uint32_t BkfDispPrinfChkParam(BkfDisp *disp, const char *fmt)
{
    if (disp == VOS_NULL) {
        return BKF_ERR;
    }
    if (fmt == VOS_NULL) {
        return BKF_DISP_CNT_PRINTF_FMT_NULL;
    }
    if (disp->outStrBuf == VOS_NULL) { /* 保护，如入参检查的时候，确实可能为空。 */
        return BKF_DISP_CNT_PRINTF_OUT_BUF_NULL;
    }
    if (disp->inCb && disp->outStrBufNotEnough) { /* 非cb状态有预留，不受此标记影响打印输出 */
        return BKF_DISP_CNT_PRINTF_OUT_STR_BUF_NOT_ENOUGH;
    }

    return BKF_OK;
}
static inline void BkfDispRollbackPrintfLen(BkfDisp *disp)
{
    /* 回滚到有效长度 */
    if (disp->prinfValidLen > 0) {
        disp->prinfTotalLen = disp->prinfValidLen;
        disp->outStrBuf[disp->prinfTotalLen] = '\0';
    }
}
STATIC void BkfDispWrite2OutBuf(BkfDisp *disp, char *buf, uint32_t len)
{
    int32_t mayWriteLen;
    int32_t writeLen;

    mayWriteLen = disp->outStrBufLen - disp->prinfTotalLen;
    if (disp->inCb) {
        mayWriteLen -= BKF_DISP_OUT_BUF_RSV_LEN_FOR_LAST_PROMPT;
    }
    if (mayWriteLen <= 0) {
        disp->outStrBufNotEnough = VOS_TRUE;
        BkfDispRollbackPrintfLen(disp);
        return;
    }
    writeLen = snprintf_truncated_s(&disp->outStrBuf[0] + disp->prinfTotalLen, mayWriteLen, "%s", buf);
    if (writeLen < (int32_t)len) {
        disp->outStrBufNotEnough = VOS_TRUE;
        BkfDispRollbackPrintfLen(disp);
        return;
    }
    disp->prinfTotalLen += writeLen;
}
void BkfDispPrintf(BkfDisp *disp, const char *fmt, ...)
{
    char buf[BKF_DISP_PRINTF_BUF_LEN_MAX] = {0};
    uint32_t ret;
    va_list va;
    errno_t err;

    ret = BkfDispPrinfChkParam(disp, fmt);
    if (ret != BKF_OK) {
        BKF_DISP_INC_CNT(disp, ret);
        return;
    }

    va_start(va, fmt);
    err = vsnprintf_truncated_s(buf, sizeof(buf), fmt, va);
    BKF_ASSERT(err >= 0);
    va_end(va);

    BkfDispWrite2OutBuf(disp, buf, VOS_StrLen(buf));
    return;
}

STATIC uint32_t BkfDispSaveCtxChkParam(BkfDisp *disp, void *ctx1, int32_t ctx1Len, void *ctx2OrNull, int32_t ctx2Len)
{
    BOOL invalid = VOS_FALSE;

    if (disp == VOS_NULL) {
        return BKF_ERR;
    }

    if (!disp->inCb) {
        return BKF_DISP_CNT_SAVE_CTX_NOT_IN_CB;
    }
    /* 无论后续是否能保存成功，都置上标记 */
    disp->curCbSaveCtx = VOS_TRUE;

    if (disp->outStrBufNotEnough) {
        return BKF_DISP_CNT_SAVE_CTX_OUT_STR_BUF_NOT_ENOUGH;
    }
    invalid = ((ctx1 == VOS_NULL) || (ctx1Len < 0));
    if (invalid) {
        return BKF_DISP_CNT_SAVE_CTX1_PARAM_NG;
    }
    invalid = (((ctx2OrNull == VOS_NULL) && (ctx2Len != 0)) || ((ctx2OrNull != VOS_NULL) && (ctx2Len < 0)));
    if (invalid) {
        return BKF_DISP_CNT_SAVE_CTX2_PARAM_NG;
    }
    if ((ctx1Len + ctx2Len) >= BKF_DISP_CTX_LEN_MAX) {
        return BKF_DISP_CNT_SAVE_CTX_TOTAL_LEN_NG;
    }

    return BKF_OK;
}
void BkfDispSaveCtx(BkfDisp *disp, void *ctx1, int32_t ctx1Len, void *ctx2OrNull, int32_t ctx2Len)
{
    uint32_t ret;
    errno_t err;

    ret = BkfDispSaveCtxChkParam(disp, ctx1, ctx1Len, ctx2OrNull, ctx2Len);
    if (ret != BKF_OK) {
        BKF_DISP_INC_CNT(disp, ret);
        return;
    }

    disp->ctx->ctx1Len = ctx1Len;
    disp->ctx->ctx2Len = ctx2Len;
    err = memmove_s(disp->ctx->buf, BKF_DISP_CTX_LEN_MAX, ctx1, ctx1Len);
    if (err != EOK) {
        BKF_DISP_INC_CNT(disp, BKF_DISP_CNT_SAVE_CTX1_NG);
        disp->ctx->ctx1Len = 0;
        disp->ctx->ctx2Len = 0;
        return;
    }

    if (ctx2OrNull != VOS_NULL) {
        err = memmove_s(&disp->ctx->buf[0] + ctx1Len, BKF_DISP_CTX_LEN_MAX - ctx1Len, ctx2OrNull, ctx2Len);
        if (err != EOK) {
            BKF_DISP_INC_CNT(disp, BKF_DISP_CNT_SAVE_CTX2_NG);
            disp->ctx->ctx1Len = 0;
            disp->ctx->ctx2Len = 0;
            return;
        }
    }

    disp->prinfValidLen = disp->prinfTotalLen;
    BKF_DISP_INC_CNT(disp, BKF_DISP_CNT_SAVE_CTX_OK);

    return;
}

STATIC uint32_t BkfDispGetLastCtxChkParam(BkfDisp *disp, void *ctx2OrNull)
{
    (void)ctx2OrNull;
    if (disp == VOS_NULL) {
        return BKF_ERR;
    }
    if (!disp->inCb) {
        return BKF_DISP_CNT_GET_CTX_NOT_IN_CB;
    }

    /* 无论后续是否能保存成功，都置上标记 */
    disp->curCbGetCtx = VOS_TRUE;
    return BKF_OK;
}
void *BkfDispGetLastCtx(BkfDisp *disp, void *ctx2OrNull)
{
    uint32_t ret;
    errno_t err;

    ret = BkfDispGetLastCtxChkParam(disp, ctx2OrNull);
    if (ret != BKF_OK) {
        BKF_DISP_INC_CNT(disp, ret);
        return VOS_NULL;
    }

    if (disp->ctx->ctx1Len <= 0) {
        BKF_DISP_INC_CNT(disp, BKF_DISP_CNT_GET_CTX_NULL);
        return VOS_NULL;
    }

    if ((disp->ctx->ctx2Len > 0) && (ctx2OrNull != VOS_NULL)) {
        err = memcpy_s(ctx2OrNull, disp->ctx->ctx2Len, &disp->ctx->buf[0] + disp->ctx->ctx1Len, disp->ctx->ctx2Len);
        if (err != EOK) {
            BKF_DISP_INC_CNT(disp, BKF_DISP_CNT_GET_CTX2_NG);
            return VOS_NULL;
        }
    }

    BKF_DISP_INC_CNT(disp, BKF_DISP_CNT_GET_CTX_OK);
    return disp->ctx->buf;
}

void BkfDispSave3Num(BkfDisp *disp, int32_t val1, int32_t val2, int32_t val3)
{
    int32_t val[3] = {val1, val2, val3};

    BKF_DISP_SAVE_CTX(disp, val, sizeof(val), VOS_NULL, 0);
}

BOOL BkfDispGetLast3Num(BkfDisp *disp, int32_t *val1Out, int32_t *val2OutOrNull, int32_t *val3OutOrNull)
{
    int32_t *val = VOS_NULL;

    if (disp == VOS_NULL) {
        return VOS_FALSE;
    }
    if (val1Out == VOS_NULL) {
        BKF_DISP_INC_CNT(disp, BKF_DISP_CNT_GET_CTX_VAL1_NULL);
        return VOS_FALSE;
    }

    val = BKF_DISP_GET_LAST_CTX(disp, VOS_NULL);
    if (val == VOS_NULL) {
        return VOS_FALSE;
    }
    *val1Out = val[0];
    if (val2OutOrNull != VOS_NULL) {
        *val2OutOrNull = val[1];
    }
    if (val3OutOrNull != VOS_NULL) {
        *val3OutOrNull = val[2]; /* param 2 */
    }
    return VOS_TRUE;
}

uint32_t BkfDispInitTempCtx(BkfDispTempCtx *tempCtx)
{
    if (tempCtx == VOS_NULL) {
        return BKF_ERR;
    }

    (void)memset_s(tempCtx, sizeof(BkfDispTempCtx), 0, sizeof(BkfDispTempCtx));
    return BKF_OK;
}

STATIC void BkfDispTestOneNoArgFuncWithObj(BkfDisp *disp, BkfDispFunc *dispFunc, BkfDispObj *dispObj)
{
    char inStrFuncName[BKF_1K / 2] = {0};
    char inStrObjName[BKF_1K / 2] = {0};
    char *inStr[2] = {inStrFuncName, inStrObjName};
    int32_t err;
    uint32_t ret;
    char outStrBuf[BKF_1K * 3] = {0};
    BkfDispProcCtx ctx = {0};

    do {
        err = snprintf_truncated_s(inStrFuncName, sizeof(inStrFuncName), "%s", dispFunc->funcName);
        if (err <= 0) {
            return;
        }
        err = snprintf_truncated_s(inStrObjName, sizeof(inStrObjName), "%s", dispObj->objName);
        if (err <= 0) {
            return;
        }
        ret = BkfDispProc(disp, inStr, BKF_GET_ARY_COUNT(inStr), outStrBuf, sizeof(outStrBuf), &ctx);
    } while (ret == BKF_DISP_PROC_HAS_MORE_DATA);
}
STATIC void BkfDispTestOneNoArgFunc(BkfDisp *disp, BkfDispFunc *dispFunc)
{
    BkfDispFuncObjName *funcObjName = VOS_NULL;
    void *itor = VOS_NULL;
    BkfDispObj *dispObj = VOS_NULL;

    for (funcObjName = BkfDispGetFirstFuncObjNameOfFunc(disp, dispFunc, &itor); funcObjName != VOS_NULL;
         funcObjName = BkfDispGetNextFuncObjNameOfFunc(disp, dispFunc, &itor)) {
        dispObj = BkfDispFindObj(disp, funcObjName->objName);
        if (dispObj != VOS_NULL) {
            BkfDispTestOneNoArgFuncWithObj(disp, dispFunc, dispObj);
        }
    }
}
STATIC void BkfDispTestNoArgFunc(BkfDisp *disp)
{
    BkfDispFunc *dispFunc = VOS_NULL;
    void *itor = VOS_NULL;

    for (dispFunc = BkfDispGetFirstFuncByTime(disp, &itor); dispFunc != VOS_NULL;
         dispFunc = BkfDispGetNextFuncByTime(disp, &itor)) {
        if (dispFunc->funcParamCnt == 0) {
            BkfDispTestOneNoArgFunc(disp, dispFunc);
        }
    }
}
void BkfDispTestFunc(BkfDisp *disp, char *funcName, char *objName, uint8_t funcParamCnt, ...)
{
    (void)funcName;
    (void)funcParamCnt;
    if ((disp == VOS_NULL) || (objName == VOS_NULL)) {
        return;
    }

    /* 暂时先支持这种 */
    BkfDispTestNoArgFunc(disp);
}

#endif

#if BKF_BLOCK("私有函数定义")
/* on msg & proc */
STATIC BOOL BkfDispFuncRealParam2Str(char *realParamStr, uintptr_t *outVal)
{
    *outVal = (uintptr_t)realParamStr;
    return VOS_TRUE;
}

STATIC BOOL BkfDispFuncRealParam2Num(char *realParamStr, uintptr_t *outVal)
{
    uintptr_t max = (uintptr_t)((uint32_t)-1);
    char *end = VOS_NULL;

    *outVal = VOS_strtoul(realParamStr, &end, 0);
    if ((end == VOS_NULL) || (*end != '\0')) {
        return VOS_FALSE;
    }

    *outVal = BKF_GET_MIN(*outVal, max);
    return VOS_TRUE;
}

/* data op */
STATIC BkfDispFunc *BkfDispAddFunc(BkfDisp *disp, char *funcName, F_BKF_DISP_FUNC funcAddr, const char *funcHelpInfo,
    uint8_t funcLvl, uint8_t funcParamCnt, char **funcFormalParam)
{
    BkfDispFunc *dispFunc = VOS_NULL;
    uint32_t strLen;
    uint32_t len;
    int32_t ret;
    uint32_t i;
    BOOL insOk = VOS_FALSE;

    strLen = VOS_StrLen(funcName);
    len = sizeof(BkfDispFunc) + strLen + 1;
    dispFunc = BKF_MALLOC(disp->argInit.memMng, len);
    if (dispFunc == VOS_NULL) {
        goto error;
    }
    (void)memset_s(dispFunc, len, 0, len);
    VOS_AVLL_INIT_TREE(dispFunc->funcObjNameSet, (AVLL_COMPARE)VOS_StrCmp, BKF_OFFSET(BkfDispFuncObjName, objName[0]),
        BKF_OFFSET(BkfDispFuncObjName, avlNode));
    ret = snprintf_truncated_s(dispFunc->funcName, strLen + 1, "%s", funcName);
    if (ret < 0) {
        goto error;
    }
    dispFunc->funcAddr = funcAddr;
    dispFunc->funcHelpInfo = funcHelpInfo;
    dispFunc->funcLvl = funcLvl;
    dispFunc->funcParamCnt = funcParamCnt;
    for (i = 0; i < funcParamCnt; i++) {
        dispFunc->funcFormalParam[i] = funcFormalParam[i];
    }
    BKF_DL_NODE_INIT(&dispFunc->dlNode);
    VOS_AVLL_INIT_NODE(dispFunc->avlNode);
    insOk = VOS_AVLL_INSERT(disp->funcSet, dispFunc->avlNode);
    if (!insOk) {
        goto error;
    }
    BKF_DL_ADD_LAST(&disp->funcSetByTime, &dispFunc->dlNode);

    return dispFunc;

error:

    if (dispFunc != VOS_NULL) {
        /* insOk一定为false */
        BKF_FREE(disp->argInit.memMng, dispFunc);
    }
    return VOS_NULL;
}

STATIC void BkfDispDelFunc(BkfDisp *disp, BkfDispFunc *dispFunc)
{
    BkfDispDelFuncAllObjName(disp, dispFunc);

    BKF_DL_REMOVE(&dispFunc->dlNode);
    VOS_AVLL_DELETE(disp->funcSet, dispFunc->avlNode);
    BKF_FREE(disp->argInit.memMng, dispFunc);
    return;
}

STATIC void BkfDispDelAllFunc(BkfDisp *disp)
{
    BkfDispFunc *dispFunc = VOS_NULL;
    void *itor = VOS_NULL;

    for (dispFunc = BkfDispGetFirstFuncByTime(disp, &itor); dispFunc != VOS_NULL;
         dispFunc = BkfDispGetNextFuncByTime(disp, &itor)) {
        BkfDispDelFunc(disp, dispFunc);
    }
    return;
}

STATIC BkfDispFunc *BkfDispFindFunc(BkfDisp *disp, char *funcName)
{
    BkfDispFunc *dispFunc = VOS_NULL;

    dispFunc = VOS_AVLL_FIND(disp->funcSet, funcName);
    return dispFunc;
}

STATIC BkfDispFunc *BkfDispFindNextFunc(BkfDisp *disp, char *funcName)
{
    BkfDispFunc *dispFunc = VOS_NULL;

    dispFunc = VOS_AVLL_FIND_NEXT(disp->funcSet, funcName);
    return dispFunc;
}

STATIC BkfDispFunc *BkfDispGetFirstFuncByTime(BkfDisp *disp, void **itorOutOrNull)
{
    BkfDispFunc *dispFunc = VOS_NULL;
    BkfDlNode *dlNode = VOS_NULL;

    dlNode = BKF_DL_GET_FIRST(&disp->funcSetByTime);
    if (dlNode != VOS_NULL) {
        dispFunc = BKF_DL_GET_ENTRY(dlNode, BkfDispFunc, dlNode);
        if (itorOutOrNull != VOS_NULL) {
            *itorOutOrNull = BKF_DL_GET_NEXT(&disp->funcSetByTime, dlNode);
        }
    } else {
        dispFunc = VOS_NULL;
        if (itorOutOrNull != VOS_NULL) {
            *itorOutOrNull = VOS_NULL;
        }
    }

    return dispFunc;
}

STATIC BkfDispFunc *BkfDispGetNextFuncByTime(BkfDisp *disp, void **itorInOut)
{
    BkfDispFunc *dispFunc = VOS_NULL;
    BkfDlNode *dlNode = VOS_NULL;

    dlNode = *itorInOut;
    if (dlNode != VOS_NULL) {
        dispFunc = BKF_DL_GET_ENTRY(dlNode, BkfDispFunc, dlNode);
        *itorInOut = BKF_DL_GET_NEXT(&disp->funcSetByTime, dlNode);
    } else {
        dispFunc = VOS_NULL;
        *itorInOut = VOS_NULL;
    }

    return dispFunc;
}

STATIC BkfDispFuncObjName *BkfDispAddFuncObjName(BkfDisp *disp, BkfDispFunc *dispFunc, BkfDispObj *dispObj)
{
    BkfDispFuncObjName *funcObjName = VOS_NULL;
    uint32_t strLen;
    uint32_t len;
    int32_t ret;
    BOOL insOk = VOS_FALSE;

    strLen = VOS_StrLen(dispObj->objName);
    len = sizeof(BkfDispFuncObjName) + strLen + 1;
    funcObjName = BKF_MALLOC(disp->argInit.memMng, len);
    if (funcObjName == VOS_NULL) {
        goto error;
    }
    (void)memset_s(funcObjName, len, 0, len);
    funcObjName->dispFunc = dispFunc;
    VOS_AVLL_INIT_NODE(funcObjName->avlNode);
    BKF_DL_NODE_INIT(&funcObjName->dlNode);
    BKF_DL_ADD_LAST(&dispObj->funcObjNameSet, &funcObjName->dlNode);
    ret = snprintf_truncated_s(funcObjName->objName, strLen + 1, "%s", dispObj->objName);
    if (ret < 0) {
        goto error;
    }
    insOk = VOS_AVLL_INSERT(dispFunc->funcObjNameSet, funcObjName->avlNode);
    if (!insOk) {
        goto error;
    }

    return funcObjName;

error:

    if (funcObjName != VOS_NULL) {
        /* insOk一定为false */
        if (BKF_DL_NODE_IS_IN(&funcObjName->dlNode)) {
            BKF_DL_REMOVE(&funcObjName->dlNode);
        }
        BKF_FREE(disp->argInit.memMng, funcObjName);
    }
    return VOS_NULL;
}

STATIC void BkfDispDelFuncObjName(BkfDisp *disp, BkfDispFuncObjName *funcObjName)
{
    BkfDispFunc *dispFunc = funcObjName->dispFunc;

    if (BKF_DL_NODE_IS_IN(&funcObjName->dlNode)) {
        BKF_DL_REMOVE(&funcObjName->dlNode);
    }
    VOS_AVLL_DELETE(dispFunc->funcObjNameSet, funcObjName->avlNode);
    BKF_FREE(disp->argInit.memMng, funcObjName);
    return;
}

STATIC void BkfDispDelFuncAllObjName(BkfDisp *disp, BkfDispFunc *dispFunc)
{
    BkfDispFuncObjName *funcObjName = VOS_NULL;
    void *itor = VOS_NULL;

    for (funcObjName = BkfDispGetFirstFuncObjNameOfFunc(disp, dispFunc, &itor); funcObjName != VOS_NULL;
         funcObjName = BkfDispGetNextFuncObjNameOfFunc(disp, dispFunc, &itor)) {
        BkfDispDelFuncObjName(disp, funcObjName);
    }
    return;
}

STATIC BkfDispFuncObjName *BkfDispFindFuncObjName(BkfDisp *disp, BkfDispFunc *dispFunc, char *objName)
{
    (void)disp;
    BkfDispFuncObjName *funcObjName = VOS_NULL;

    funcObjName = VOS_AVLL_FIND(dispFunc->funcObjNameSet, objName);
    return funcObjName;
}

STATIC BkfDispFuncObjName *BkfDispFindNextFuncObjName(BkfDisp *disp, BkfDispFunc *dispFunc, char *objName)
{
    (void)disp;
    BkfDispFuncObjName *funcObjName = VOS_NULL;

    funcObjName = VOS_AVLL_FIND_NEXT(dispFunc->funcObjNameSet, objName);
    return funcObjName;
}

STATIC BkfDispFuncObjName *BkfDispGetFirstFuncObjNameOfFunc(BkfDisp *disp, BkfDispFunc *dispFunc, void **itorOutOrNull)
{
    (void)disp;
    BkfDispFuncObjName *funcObjName = VOS_NULL;

    funcObjName = VOS_AVLL_FIRST(dispFunc->funcObjNameSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (funcObjName != VOS_NULL) ? VOS_AVLL_NEXT(dispFunc->funcObjNameSet, funcObjName->avlNode)
                                                   : VOS_NULL;
    }
    return funcObjName;
}

STATIC BkfDispFuncObjName *BkfDispGetNextFuncObjNameOfFunc(BkfDisp *disp, BkfDispFunc *dispFunc, void **itorInOut)
{
    (void)disp;
    BkfDispFuncObjName *funcObjName = VOS_NULL;

    funcObjName = (*itorInOut);
    *itorInOut = (funcObjName != VOS_NULL) ? VOS_AVLL_NEXT(dispFunc->funcObjNameSet, funcObjName->avlNode) : VOS_NULL;
    return funcObjName;
}

STATIC BkfDispFuncObjName *BkfDispGetFirstFuncObjNameOfObj(BkfDisp *disp, BkfDispObj *dispObj, void **itorOutOrNull)
{
    (void)disp;
    BkfDlNode *tempNode = VOS_NULL;
    BkfDispFuncObjName *funcObjName = VOS_NULL;

    tempNode = BKF_DL_GET_FIRST(&dispObj->funcObjNameSet);
    if (tempNode != VOS_NULL) {
        funcObjName = BKF_DL_GET_ENTRY(tempNode, BkfDispFuncObjName, dlNode);
        if (itorOutOrNull != VOS_NULL) {
            *itorOutOrNull = BKF_DL_GET_NEXT(&dispObj->funcObjNameSet, &funcObjName->dlNode);
        }
    } else {
        funcObjName = VOS_NULL;
        if (itorOutOrNull != VOS_NULL) {
            *itorOutOrNull = VOS_NULL;
        }
    }
    return funcObjName;
}

STATIC BkfDispFuncObjName *BkfDispGetNextFuncObjNameOfObj(BkfDisp *disp, BkfDispObj *dispObj, void **itorInOut)
{
    (void)disp;
    BkfDlNode *tempNode = VOS_NULL;
    BkfDispFuncObjName *funcObjName = VOS_NULL;

    tempNode = *itorInOut;
    if (tempNode != VOS_NULL) {
        funcObjName = BKF_DL_GET_ENTRY(tempNode, BkfDispFuncObjName, dlNode);
        *itorInOut = BKF_DL_GET_NEXT(&dispObj->funcObjNameSet, &funcObjName->dlNode);
    } else {
        funcObjName = VOS_NULL;
        *itorInOut = VOS_NULL;
    }
    return funcObjName;
}

STATIC BkfDispObj *BkfDispAddObj(BkfDisp *disp, char *objName, void *objAddr)
{
    BkfDispObj *dispObj = VOS_NULL;
    uint32_t strLen;
    uint32_t len;
    int32_t ret;
    BOOL insOk = VOS_FALSE;

    strLen = VOS_StrLen(objName);
    len = sizeof(BkfDispObj) + strLen + 1;
    dispObj = BKF_MALLOC(disp->argInit.memMng, len);
    if (dispObj == VOS_NULL) {
        goto error;
    }
    (void)memset_s(dispObj, len, 0, len);
    BKF_DL_INIT(&dispObj->funcObjNameSet);
    ret = snprintf_truncated_s(dispObj->objName, strLen + 1, "%s", objName);
    if (ret < 0) {
        goto error;
    }
    dispObj->objAddr = objAddr;
    VOS_AVLL_INIT_NODE(dispObj->avlNode);
    insOk = VOS_AVLL_INSERT(disp->objSet, dispObj->avlNode);
    if (!insOk) {
        goto error;
    }

    return dispObj;

error:

    if (dispObj != VOS_NULL) {
        /* insOk一定为false */
        BKF_FREE(disp->argInit.memMng, dispObj);
    }
    return VOS_NULL;
}

STATIC void BkfDispDelObj(BkfDisp *disp, BkfDispObj *dispObj)
{
    BOOL mayDel = VOS_FALSE;

    mayDel = (BkfDispGetFirstFuncObjNameOfObj(disp, dispObj, VOS_NULL) == VOS_NULL);
    if (!mayDel) {
        BKF_ASSERT(0);
        return;
    }

    VOS_AVLL_DELETE(disp->objSet, dispObj->avlNode);
    BKF_FREE(disp->argInit.memMng, dispObj);
    return;
}

STATIC void BkfDispDelAllObj(BkfDisp *disp)
{
    BkfDispObj *dispObj = VOS_NULL;
    void *itor = VOS_NULL;

    for (dispObj = BkfDispGetFirstObj(disp, &itor); dispObj != VOS_NULL; dispObj = BkfDispGetNextObj(disp, &itor)) {
        BkfDispDelObj(disp, dispObj);
    }
    return;
}

STATIC BkfDispObj *BkfDispFindObj(BkfDisp *disp, char *objName)
{
    BkfDispObj *dispObj = VOS_NULL;

    dispObj = VOS_AVLL_FIND(disp->objSet, objName);
    return dispObj;
}

STATIC BkfDispObj *BkfDispFindNextObj(BkfDisp *disp, char *objName)
{
    BkfDispObj *dispObj = VOS_NULL;

    dispObj = VOS_AVLL_FIND_NEXT(disp->objSet, objName);
    return dispObj;
}

STATIC BkfDispObj *BkfDispGetFirstObj(BkfDisp *disp, void **itorOutOrNull)
{
    BkfDispObj *dispObj = VOS_NULL;

    dispObj = VOS_AVLL_FIRST(disp->objSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (dispObj != VOS_NULL) ? VOS_AVLL_NEXT(disp->objSet, dispObj->avlNode) : VOS_NULL;
    }
    return dispObj;
}

STATIC BkfDispObj *BkfDispGetNextObj(BkfDisp *disp, void **itorInOut)
{
    BkfDispObj *dispObj = VOS_NULL;

    dispObj = (*itorInOut);
    *itorInOut = (dispObj != VOS_NULL) ? VOS_AVLL_NEXT(disp->objSet, dispObj->avlNode) : VOS_NULL;
    return dispObj;
}

#endif

#if BKF_BLOCK("disp")
STATIC void BkfDispOneFunc(BkfDisp *disp, BkfDispFunc *dispFunc, int32_t mayUseDispFuncTotal, int32_t funcObjNameCnt)
{
    uint8_t i;

    BKF_DISP_PRINTF(disp, "[%-4d] - %s(", mayUseDispFuncTotal, dispFunc->funcName);
    for (i = 0; i < dispFunc->funcParamCnt; i++) {
        BKF_DISP_PRINTF(disp, "%s", dispFunc->funcFormalParam[i]);
        if (i < (dispFunc->funcParamCnt - 1)) {
            BKF_DISP_PRINTF(disp, " ");
        }
    }
    BKF_DISP_PRINTF(disp, ") --- %s, (%d/%u/%u)\n", dispFunc->funcHelpInfo, funcObjNameCnt, dispFunc->appCallCnt,
        dispFunc->funcCbCnt);

    return;
}
STATIC int32_t BkfDispGetDispFuncCnt(BkfDisp *disp, BOOL onlyMayUseFunc)
{
    BkfDispFunc *dispFunc = VOS_NULL;
    void *itor = VOS_NULL;
    int32_t dispFuncCnt = 0;

    for (dispFunc = BkfDispGetFirstFuncByTime(disp, &itor); dispFunc != VOS_NULL;
         dispFunc = BkfDispGetNextFuncByTime(disp, &itor)) {
        if (onlyMayUseFunc && !BKF_DISP_FUNC_MAY_USE(disp, dispFunc)) {
            break;
        }

        dispFuncCnt++;
    }
    return dispFuncCnt;
}
STATIC int32_t BkfDispGetFuncObjNameCnt(BkfDisp *disp, BkfDispFunc *dispFunc)
{
    BkfDispFuncObjName *funcObjName = VOS_NULL;
    void *itor = VOS_NULL;
    int32_t funcObjNameCnt = 0;

    for (funcObjName = BkfDispGetFirstFuncObjNameOfFunc(disp, dispFunc, &itor); funcObjName != VOS_NULL;
         funcObjName = BkfDispGetNextFuncObjNameOfFunc(disp, dispFunc, &itor)) {
        funcObjNameCnt++;
    }
    return funcObjNameCnt;
}
STATIC int32_t BkfDispGetAllFuncObjNameCnt(BkfDisp *disp, BOOL onlyMayUseFunc)
{
    BkfDispFunc *dispFunc = VOS_NULL;
    BkfDispFuncObjName *funcObjName = VOS_NULL;
    void *itor1 = VOS_NULL;
    void *itor2 = VOS_NULL;
    int32_t funcObjNameCnt = 0;

    for (dispFunc = BkfDispGetFirstFuncByTime(disp, &itor1); dispFunc != VOS_NULL;
         dispFunc = BkfDispGetNextFuncByTime(disp, &itor1)) {
        if (onlyMayUseFunc && !BKF_DISP_FUNC_MAY_USE(disp, dispFunc)) {
            break;
        }

        for (funcObjName = BkfDispGetFirstFuncObjNameOfFunc(disp, dispFunc, &itor2); funcObjName != VOS_NULL;
             funcObjName = BkfDispGetNextFuncObjNameOfFunc(disp, dispFunc, &itor2)) {
            funcObjNameCnt++;
        }
    }
    return funcObjNameCnt;
}
STATIC int32_t BkfDispGetDispObjCnt(BkfDisp *disp)
{
    BkfDispObj *dispObj = VOS_NULL;
    void *itor = VOS_NULL;
    int32_t dispObjCnt = 0;

    for (dispObj = BkfDispGetFirstObj(disp, &itor); dispObj != VOS_NULL; dispObj = BkfDispGetNextObj(disp, &itor)) {
        dispObjCnt++;
    }
    return dispObjCnt;
}
void BkfDispHelp(BkfDisp *disp)
{
    int32_t mayUseDispFuncTotal;
    int32_t funcObjNameTotal;
    BkfDispFunc *dispFunc = VOS_NULL;
    int32_t i;
    void *itor = VOS_NULL;
    int32_t funcObjNameCnt;

    if (!BKF_DISP_GET_LAST_3NUM(disp, &mayUseDispFuncTotal, &funcObjNameTotal, VOS_NULL)) {
        mayUseDispFuncTotal = 0;
        funcObjNameTotal = 0;
    }
    i = 0;
    for (dispFunc = BkfDispGetFirstFuncByTime(disp, &itor); dispFunc != VOS_NULL;
         dispFunc = BkfDispGetNextFuncByTime(disp, &itor)) {
        if (BKF_DISP_FUNC_MAY_USE(disp, dispFunc)) {
            i++;
            if (i > mayUseDispFuncTotal) {
                break;
            }
        }
    }

    if (dispFunc != VOS_NULL) {
        funcObjNameCnt = BkfDispGetFuncObjNameCnt(disp, dispFunc);
        funcObjNameTotal += funcObjNameCnt;
        BkfDispOneFunc(disp, dispFunc, mayUseDispFuncTotal, funcObjNameCnt);
        mayUseDispFuncTotal++;
        BKF_DISP_SAVE_3NUM(disp, mayUseDispFuncTotal, funcObjNameTotal, 0);
    } else {
        BKF_DISP_PRINTF(disp, " ***total %d func(s), %d func obj name(s)***\n", mayUseDispFuncTotal, funcObjNameTotal);
    }
}

void BkfDispSelf(BkfDisp *disp)
{
    int32_t mayUseDispFuncCnt;
    int32_t dispFuncCnt;
    int32_t mayUseFuncObjNameCnt;
    int32_t funcObjNameCnt;
    int32_t dispObjCnt;
    int64_t memDispFunc;
    int64_t memFuncObjName;
    int64_t memDispObj;
    int64_t memTotal;

    mayUseDispFuncCnt = BkfDispGetDispFuncCnt(disp, VOS_TRUE);
    dispFuncCnt = BkfDispGetDispFuncCnt(disp, VOS_FALSE);
    mayUseFuncObjNameCnt = BkfDispGetAllFuncObjNameCnt(disp, VOS_TRUE);
    funcObjNameCnt = BkfDispGetAllFuncObjNameCnt(disp, VOS_FALSE);
    dispObjCnt = BkfDispGetDispObjCnt(disp);

    BKF_DISP_PRINTF(disp, "%s\n", disp->name);
    BKF_DISP_PRINTF(disp, "mayProcFuncLvl(%u)\n", disp->argInit.mayProcFuncLvl);
    BKF_DISP_PRINTF(disp, "%d/%d func(s), %d/%d func obj name(s)\n", mayUseDispFuncCnt, dispFuncCnt,
        mayUseFuncObjNameCnt, funcObjNameCnt);
    BKF_DISP_PRINTF(disp, "%d obj(s)\n", dispObjCnt);

    memDispFunc = (int64_t)sizeof(BkfDispFunc) * dispFuncCnt;
    memFuncObjName = (int64_t)sizeof(BkfDispFuncObjName) * funcObjNameCnt;
    memDispObj = (int64_t)sizeof(BkfDispObj) * dispObjCnt;
    memTotal = (int64_t)sizeof(*disp) + memDispFunc + memFuncObjName + memDispObj;
    BKF_DISP_PRINTF(disp, "memTotal       : ^%-10" VOS_PRId64 "\n", memTotal);
    BKF_DISP_PRINTF(disp, "memHandle      :  %-10" VOS_PRId64 "\n", sizeof(*disp));
    BKF_DISP_PRINTF(disp, "memDispFunc    : ^%-10" VOS_PRId64 " = %d^ * %d\n", memDispFunc, sizeof(BkfDispFunc),
        funcObjNameCnt);
    BKF_DISP_PRINTF(disp, "memFuncObjName : ^%-10" VOS_PRId64 " = %d^ * %d\n", memFuncObjName,
        sizeof(BkfDispFuncObjName), funcObjNameCnt);
    BKF_DISP_PRINTF(disp, "memDispObj     : ^%-10" VOS_PRId64 " = %d^ * %d\n", memDispObj, sizeof(BkfDispObj),
        dispObjCnt);
}

void BkfDispFuncObjNameFunc(BkfDisp *disp, char *funcName)
{
    BkfDispFunc *dispFunc = VOS_NULL;
    int32_t i;
    BkfDispFuncObjName *funcObjName = VOS_NULL;
    char *lastObjName = VOS_NULL;
    int32_t cnt = 0;

    if (funcName == VOS_NULL) {
        BKF_DISP_PRINTF(disp, "input func name is null\n");
        return;
    }

    /* func */
    dispFunc = BkfDispFindFunc(disp, funcName);
    if ((dispFunc == VOS_NULL) || !BKF_DISP_FUNC_MAY_USE(disp, dispFunc)) {
        BKF_DISP_PRINTF(disp, "%s, func not found\n", funcName);
        return;
    }

    lastObjName = BKF_DISP_GET_LAST_CTX(disp, &cnt);
    if (lastObjName == VOS_NULL) {
        BKF_DISP_PRINTF(disp, "%s(", dispFunc->funcName);
        for (i = 0; i < dispFunc->funcParamCnt; i++) {
            BKF_DISP_PRINTF(disp, "%s", dispFunc->funcFormalParam[i]);
            if (i < (dispFunc->funcParamCnt - 1)) {
                BKF_DISP_PRINTF(disp, " ");
            }
        }
        BKF_DISP_PRINTF(disp, ") --- %s\n", dispFunc->funcHelpInfo);

        cnt = 0;
        funcObjName = BkfDispGetFirstFuncObjNameOfFunc(disp, dispFunc, VOS_NULL);
    } else {
        funcObjName = BkfDispFindNextFuncObjName(disp, dispFunc, lastObjName);
    }

    /* obj name */
    if (funcObjName != VOS_NULL) {
        BKF_DISP_PRINTF(disp, "[%d] - %s\n", cnt, funcObjName->objName);

        cnt++;
        BKF_DISP_SAVE_CTX(disp, funcObjName->objName, VOS_StrLen(funcObjName->objName) + 1, &cnt, sizeof(cnt));
    } else {
        BKF_DISP_PRINTF(disp, " ***total %d func obj name(s)***\n", cnt);
    }
}

void BkfDispAllObj(BkfDisp *disp)
{
    BkfDispObj *dispObj = VOS_NULL;
    char *lastObjName = VOS_NULL;
    int32_t cnt = 0;

    lastObjName = BKF_DISP_GET_LAST_CTX(disp, &cnt);
    if (lastObjName == VOS_NULL) {
        cnt = 0;
        dispObj = BkfDispGetFirstObj(disp, VOS_NULL);
    } else {
        dispObj = BkfDispFindNextObj(disp, lastObjName);
    }

    /* obj name */
    if (dispObj != VOS_NULL) {
        BKF_DISP_PRINTF(disp, "[%d] - [%s : %#x]\n", cnt, dispObj->objName, BKF_MASK_ADDR(dispObj->objAddr));

        cnt++;
        BKF_DISP_SAVE_CTX(disp, dispObj->objName, VOS_StrLen(dispObj->objName) + 1, &cnt, sizeof(cnt));
    } else {
        BKF_DISP_PRINTF(disp, " ***total %d obj(s)***\n", cnt);
    }
}

void BkfDispStatNotZero(BkfDisp *disp)
{
    int32_t i;
    int32_t statNotZeroCnt;

    if (!BKF_DISP_GET_LAST_3NUM(disp, &i, &statNotZeroCnt, VOS_NULL)) {
        i = 0;
        statNotZeroCnt = 0;
    }

    if (i < BKF_DISP_CNT_MAX) {
        if (disp->statCnt[i] != 0) {
            BKF_DISP_PRINTF(disp, "[%-3d] - %-50s - %u\n", i, g_BkfDispStatStrTbl[i], disp->statCnt[i]);
            statNotZeroCnt++;
        }

        i++;
        BKF_DISP_SAVE_3NUM(disp, i, statNotZeroCnt, 0);
    } else {
        BKF_DISP_PRINTF(disp, " ***total %d stat(s), including %d not zero***\n", BKF_DISP_CNT_MAX, statNotZeroCnt);
    }
}

typedef struct tagBkfDispMemStatCtx {
    uint16_t num;
    uint8_t pad1[2];
    char str[BKF_DISP_CTX_LEN_MAX - 0x80];
} BkfDispMemStatCtx;
void BkfDispMemMng(BkfDisp *disp)
{
    BkfDispMemStatCtx *last = VOS_NULL;
    BkfDispMemStatCtx cur = {0};
    char *statStr = VOS_NULL;
    BkfMemStatKey key = {0};
    int32_t cnt = 0;
    errno_t err;
    uint8_t buf[BKF_1K];

    last = BKF_DISP_GET_LAST_CTX(disp, &cnt);
    if (last == VOS_NULL) {
        cnt = 0;
        BKF_DISP_PRINTF(disp, "%s\n", BkfMemGetSummaryStr(disp->argInit.memMng, buf, sizeof(buf)));
        statStr = BkfMemGetFirstStatStr(disp->argInit.memMng, &key, buf, sizeof(buf));
    } else {
        key.str = last->str;
        key.num = last->num;
        statStr = BkfMemGetNextStatStr(disp->argInit.memMng, &key, buf, sizeof(buf));
    }

    if (statStr != VOS_NULL) {
        BKF_DISP_PRINTF(disp, "[%-4d] - [%-30s : %-6u]: %s\n", cnt, BkfTrimStrPath(key.str), key.num, statStr);

        err = strcpy_s(cur.str, sizeof(cur.str), key.str);
        if (err != EOK) {
            BKF_DISP_PRINTF(disp, "err(%d)\n", err);
            return;
        }
        cur.num = key.num;
        cnt++;
        BKF_DISP_SAVE_CTX(disp, &cur, sizeof(cur), &cnt, sizeof(cnt));
    } else {
        BKF_DISP_PRINTF(disp, " ***total %d statInfo(s)***\n", cnt);
    }
}

void BkfDispTest1(BkfDisp *disp, char *numStr, uintptr_t num)
{
    BKF_DISP_PRINTF(disp, "%s : 0x%" VOS_PRIxPTR " : %" VOS_PRIuPTR "\n", numStr, num, num);
}

void BkfDispTestNotGetCtx(BkfDisp *disp)
{
    int32_t i = 0;

    BKF_DISP_PRINTF(disp, "testing not get ctx, will quit after show this and count twice\n");
    BKF_DISP_PRINTF(disp, "[%-3d] - %-50s - %u\n", i, g_BkfDispStatStrTbl[i], disp->statCnt[i]);
    i++;
    BKF_DISP_SAVE_3NUM(disp, i, 0, 0);
}

void BkfDispTestDeadloop(BkfDisp *disp)
{
    int32_t i = 0;

    if (!BKF_DISP_GET_LAST_3NUM(disp, &i, VOS_NULL, VOS_NULL)) {
        BKF_DISP_PRINTF(disp, "testing deadloop, will quit after show one count\n");
        BKF_DISP_PRINTF(disp, "[%-3d] - %-50s - %u\n", i, g_BkfDispStatStrTbl[i], disp->statCnt[i]);
    }
    i++;
    BKF_DISP_SAVE_3NUM(disp, i, 0, 0);
}

void BkfDispTestNotSaveCtxBeforeOutBufNotEnough(BkfDisp *disp)
{
    int32_t i;
    int32_t j;

    BKF_DISP_PRINTF(disp, "testing not save ctx before out buf not enough, will quit after show something\n");
    for (j = 0; j < 20; j++) { /* 20 times */
        for (i = 0; i < BKF_DISP_CNT_MAX; i++) {
            BKF_DISP_PRINTF(disp, "[%-3d] - %-50s - %u\n", i, g_BkfDispStatStrTbl[i], disp->statCnt[i]);
        }
    }
}

void BkfDispTestNormal(BkfDisp *disp)
{
    int32_t i;
    int32_t j;

    if (!BKF_DISP_GET_LAST_3NUM(disp, &i, &j, VOS_NULL)) {
        i = 0;
        j = 0;
        BKF_DISP_PRINTF(disp, "testing normal, will show 3 times, then quit\n");
    }

    if (j < 3) { /* 3 times */
        if (i < BKF_DISP_CNT_MAX) {
            BKF_DISP_PRINTF(disp, "[%-3d] - %-50s - %u\n", i, g_BkfDispStatStrTbl[i], disp->statCnt[i]);
            i++;
        } else {
            BKF_DISP_PRINTF(disp, "--------------------\n");
            j++;
            i = 0;
        }
        BKF_DISP_SAVE_3NUM(disp, i, j, 0);
    } else {
        /* do nothing */
    }
}

void BkfDispTestAddObjAndFuncDo(BkfDisp *disp)
{
    BKF_DISP_PRINTF(disp, "do\n");
}
void BkfDispTestAddObjAndFunc(BkfDisp *disp, char *objStr)
{
    (void)BkfDispRegObj(disp, objStr, disp);
    (void)BKF_DISP_REG_TEST_FUNC(disp, BkfDispTestAddObjAndFuncDo, "disp test add obj&func do", objStr, 0);
}
void BkfDispTestDelObj(BkfDisp *disp, char *objStr)
{
    BkfDispUnregObj(disp, objStr);
}

STATIC void BkfDispInitSelfDispFunc(BkfDisp *disp)
{
    char *objName = disp->name;
    uint32_t ret;

    ret = BkfDispRegObj(disp, objName, disp);
    if (ret != BKF_OK) {
        BKF_ASSERT(0);
        return;
    }

    (void)BKF_DISP_REG_FUNC(disp, BkfDispHelp, "disp all func(s)", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfDispSelf, "disp self info", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfDispFuncObjNameFunc, "disp func and it's obj(s)", objName, 1, "pcFuncName");
    (void)BKF_DISP_REG_FUNC(disp, BkfDispAllObj, "disp all obj(s)", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfDispStatNotZero, "disp not zero stat info", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, BkfDispMemMng, "disp mem mng info", objName, 0);
    (void)BKF_DISP_REG_TEST_FUNC(disp, BkfDispTest1, "disp test", objName, BKF_NUM2, "pcNumStr", "uniNum");
    (void)BKF_DISP_REG_TEST_FUNC(disp, BkfDispTestNotGetCtx, "disp test", objName, 0);
    (void)BKF_DISP_REG_TEST_FUNC(disp, BkfDispTestDeadloop, "disp test", objName, 0);
    (void)BKF_DISP_REG_TEST_FUNC(disp, BkfDispTestNotSaveCtxBeforeOutBufNotEnough, "disp test", objName, 0);
    (void)BKF_DISP_REG_TEST_FUNC(disp, BkfDispTestNormal, "disp test", objName, 0);
    (void)BKF_DISP_REG_TEST_FUNC(disp, BkfDispTestAddObjAndFunc, "disp test add obj&func", objName, 1, "pcObjStr");
    (void)BKF_DISP_REG_TEST_FUNC(disp, BkfDispTestDelObj, "disp test del obj", objName, 1, "pcObjStr");
}

STATIC void BkfDispUninitSelfDispFunc(BkfDisp *disp)
{
    BkfDispUnregObj(disp, disp->name);
}

#endif

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
