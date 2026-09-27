/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((fsmTmpl)->log)
#define BKF_MOD_NAME ((fsmTmpl)->name)

#include "bkf_fsm.h"
#include "bkf_str.h"
#include "bkf_assert.h"
#include "securec.h"
#include "vos_base.h"
#include "v_stringlib.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
/* fsm tmpl */
struct tagBkfFsmTmpl {
    BkfFsmTmplInitArg argInit;
    char *name;
    BkfLog *log;
    int32_t fsmCnt;
    uint64_t dispatchTotalCnt;
    uint64_t dispatchFsmCurStateNgCnt;
    uint64_t dispatchInputEvtNgCnt;
    uint64_t dispatchProcImpossibleCnt;
    uint64_t dispatchProcIgnoreCnt;
    uint64_t dispatchProcRetOkCnt;
    uint64_t dispatchProcRetNOkCnt;
    uint64_t chgStateTotalCnt;
    uint64_t chgStateNewStateInvalidCnt;
    uint64_t chgStateOkCnt;
};

#define BKF_FSM_STATE_IS_VALID(fsmTmpl, state) ((uint8_t)(state) < (fsmTmpl)->argInit.stateCnt)
#define BKF_FSM_EVT_IS_VALID(fsmTmpl, evt) ((uint8_t)(evt) < (fsmTmpl)->argInit.evtCnt)

/* fsm */
#define BKF_FSM_SIGN 0xa8

#define BKF_FSM_GET_APP_DATA_STR(fsm, buf, bufLen)                                                                     \
    ((fsm)->tmpl->argInit.getAppDataStrOrNull != VOS_NULL)                                                             \
        ? (fsm)->tmpl->argInit.getAppDataStrOrNull((fsm), (buf), (bufLen))                                             \
        : "-"

struct tagBkfFsmStatInfo {
    uint64_t dispatchTotalCnt;
    uint64_t dispatchFsmCurStateNgCnt;
    uint64_t dispatchInputEvtNgCnt;
    uint64_t dispatchProcImpossibleCnt;
    uint64_t dispatchProcIgnoreCnt;
    uint64_t dispatchProcRetOkCnt;
    uint64_t dispatchProcRetNOkCnt;
    uint32_t chgStateTotalCnt;
    uint32_t chgStateNewStateInvalidCnt;
    uint32_t chgStateOkCnt;
    uint32_t stateTransCnt[0]; /* 占位符 */
};
#define BKF_FSM_INC_STAT(fsm, statField)                                                                               \
    do {                                                                                                               \
        ((fsm)->tmpl->statField)++;                                                                                    \
        if ((fsm)->statInfo != VOS_NULL) {                                                                             \
            ((fsm)->statInfo->statField)++;                                                                            \
        }                                                                                                              \
    } while (0)

STATIC void BkfFsmTmplInitDisp(BkfFsmTmpl *fsmTmpl);
STATIC void BkfFsmTmplUninitDisp(BkfFsmTmpl *fsmTmpl);

#pragma pack()
#endif

#if BKF_BLOCK("公有函数定义")
BkfFsmTmpl *BkfFsmTmplInit(BkfFsmTmplInitArg *arg)
{
    BOOL argIsValid = VOS_FALSE;
    BkfFsmTmpl *fsmTmpl = VOS_NULL;
    uint32_t len;

    argIsValid = (arg != VOS_NULL) && (arg->name != VOS_NULL) && (arg->memMng != VOS_NULL) && (arg->disp != VOS_NULL) &&
                 (arg->stateStrTbl != VOS_NULL) && (arg->evtStrTbl != VOS_NULL) &&
                 (arg->stateEvtProcItemMtrx != VOS_NULL);
    if (!argIsValid) {
        goto error;
    }

    len = sizeof(BkfFsmTmpl);
    fsmTmpl = BKF_MALLOC(arg->memMng, len);
    if (fsmTmpl == VOS_NULL) {
        goto error;
    }
    (void)memset_s(fsmTmpl, len, 0, len);
    fsmTmpl->argInit = *arg;
    fsmTmpl->name = BkfStrNew(arg->memMng, "%s", arg->name);
    fsmTmpl->argInit.name = fsmTmpl->name;
    fsmTmpl->log = arg->log;

    BkfFsmTmplInitDisp(fsmTmpl);

    BKF_LOG_DEBUG(BKF_LOG_HND, "fsmTmpl(%#x), init ok\n", BKF_MASK_ADDR(fsmTmpl));
    return fsmTmpl;

error:

    if ((fsmTmpl != VOS_NULL) && (arg != VOS_NULL) && (arg->memMng != VOS_NULL)) {
        BKF_FREE(arg->memMng, fsmTmpl);
    }
    return VOS_NULL;
}

void BkfFsmTmplUninit(BkfFsmTmpl *fsmTmpl)
{
    if (fsmTmpl == VOS_NULL) {
        return;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "fsmTmpl(%#x), 2unit\n", BKF_MASK_ADDR(fsmTmpl));

    BkfFsmTmplUninitDisp(fsmTmpl);
    BkfStrDel(fsmTmpl->name);
    BKF_FREE(fsmTmpl->argInit.memMng, fsmTmpl);
    return;
}

const char *BkfFsmTmplGetStateStr(BkfFsmTmpl *fsmTmpl, uint8_t state)
{
    char *stateStr = VOS_NULL;
    char *outputStr = VOS_NULL;
    const char *findStr = "STATE_";

    if (fsmTmpl == VOS_NULL) {
        return "STATE_fsmTmpl_null";
    }
    if (!BKF_FSM_STATE_IS_VALID(fsmTmpl, state)) {
        return "STATE_fsmTmpl_invalid";
    }

    stateStr = (char *)fsmTmpl->argInit.stateStrTbl[state];
    outputStr = VOS_StrStr(stateStr, findStr);
    return (outputStr != VOS_NULL) ? outputStr : stateStr;
}

const char *BkfFsmTmplGetEvtStr(BkfFsmTmpl *fsmTmpl, uint8_t evt)
{
    char *evtStr = VOS_NULL;
    char *outputStr = VOS_NULL;
    const char *findStr = "EVT_";

    if (fsmTmpl == VOS_NULL) {
        return "EVT_fsmTmpl_null";
    }
    if (!BKF_FSM_EVT_IS_VALID(fsmTmpl, evt)) {
        return "EVT_fsmTmpl_invalid";
    }

    evtStr = (char *)fsmTmpl->argInit.evtStrTbl[evt];
    outputStr = VOS_StrStr(evtStr, findStr);
    return (outputStr != VOS_NULL) ? outputStr : evtStr;
}

uint32_t BkfFsmInit(BkfFsm *fsm, BkfFsmTmpl *fsmTmpl)
{
    uint32_t len;
    uint8_t buf[BKF_1K / 4];

    if ((fsm == VOS_NULL) || (fsmTmpl == VOS_NULL)) {
        goto error;
    }

    (void)memset_s(fsm, sizeof(BkfFsm), 0, sizeof(BkfFsm));
    fsm->state = fsmTmpl->argInit.stateInit;
    fsm->tmpl = fsmTmpl;
    if (fsmTmpl->argInit.dbgOn) {
        len = sizeof(BkfFsmStatInfo) + sizeof(uint32_t) * fsmTmpl->argInit.stateCnt * fsmTmpl->argInit.stateCnt;
        fsm->statInfo = BKF_MALLOC(fsmTmpl->argInit.memMng, len);
        if (fsm->statInfo != VOS_NULL) {
            (void)memset_s(fsm->statInfo, len, 0, len);
        } else {
            BKF_LOG_WARN(BKF_LOG_HND, "%#x, ng\n", BKF_MASK_ADDR(fsm->statInfo));
            /* 只是没有申请到诊断信息，继续 */
        }
    }
    BKF_SIGN_SET(fsm->sign, BKF_FSM_SIGN);

    fsmTmpl->fsmCnt++;
    BKF_LOG_DEBUG(BKF_LOG_HND, "fsm(%#x)/appData(%s), init ok\n", BKF_MASK_ADDR(fsm),
        BKF_FSM_GET_APP_DATA_STR(fsm, buf, sizeof(buf)));
    return BKF_OK;

error:

    return BKF_ERR;
}

void BkfFsmUninit(BkfFsm *fsm)
{
    BkfFsmTmpl *fsmTmpl = VOS_NULL;
    uint8_t buf[BKF_1K / 4];

    if ((fsm == VOS_NULL) || !BKF_SIGN_IS_VALID(fsm->sign, BKF_FSM_SIGN) || (fsm->tmpl == VOS_NULL)) {
        return;
    }
    fsmTmpl = fsm->tmpl;
    BKF_LOG_DEBUG(BKF_LOG_HND, "fsm(%#x)/appData(%s), 2uninit\n", BKF_MASK_ADDR(fsm),
        BKF_FSM_GET_APP_DATA_STR(fsm, buf, sizeof(buf)));

    fsm->tmpl->fsmCnt--;
    if (fsm->statInfo != VOS_NULL) {
        BKF_FREE(fsmTmpl->argInit.memMng, fsm->statInfo);
    }
    (void)memset_s(fsm, sizeof(BkfFsm), 0, sizeof(BkfFsm));
    return;
}

uint32_t BkfFsmDispatchChkParam(BkfFsm *fsm, uint8_t evt)
{
    BkfFsmTmpl *fsmTmpl = VOS_NULL;
    uint8_t buf[BKF_1K / 4];

    if ((fsm == VOS_NULL) || !BKF_SIGN_IS_VALID(fsm->sign, BKF_FSM_SIGN) || (fsm->tmpl == VOS_NULL)) {
        return BKF_ERR;
    }
    fsmTmpl = fsm->tmpl;
    BKF_FSM_INC_STAT(fsm, dispatchTotalCnt);

    if (!BKF_FSM_STATE_IS_VALID(fsmTmpl, fsm->state)) {
        BKF_LOG_ERROR(BKF_LOG_HND, "fsm(%#x)/appData(%s), fsm cur state(%u/[0, %u)) is invalid\n", BKF_MASK_ADDR(fsm),
            BKF_FSM_GET_APP_DATA_STR(fsm, buf, sizeof(buf)), fsm->state, fsmTmpl->argInit.stateCnt);
        BKF_FSM_INC_STAT(fsm, dispatchFsmCurStateNgCnt);
        return BKF_ERR;
    }
    if (!BKF_FSM_EVT_IS_VALID(fsmTmpl, evt)) {
        BKF_LOG_ERROR(BKF_LOG_HND, "fsm(%#x)/appData(%s), input evt(%u/[0, %u)) is invalid\n", BKF_MASK_ADDR(fsm),
            BKF_FSM_GET_APP_DATA_STR(fsm, buf, sizeof(buf)), evt, fsmTmpl->argInit.evtCnt);
        BKF_FSM_INC_STAT(fsm, dispatchInputEvtNgCnt);
        return BKF_ERR;
    }

    return BKF_OK;
}
uint32_t BkfFsmDispatch(BkfFsm *fsm, uint8_t evt, void *dispatchParam1, void *dispatchParam2, void *dispatchParam3)
{
    uint32_t ret;
    BkfFsmTmpl *fsmTmpl = VOS_NULL;
    BkfFsmProcItem *procItem = VOS_NULL;
    uint8_t buf[BKF_1K / 4];

    ret = BkfFsmDispatchChkParam(fsm, evt);
    if (ret != BKF_OK) {
        return ret;
    }
    fsmTmpl = fsm->tmpl;

    procItem = (BkfFsmProcItem *)BKF_GET_ARY_SIM_MTRX_ITEM(fsmTmpl->argInit.stateEvtProcItemMtrx,
        fsmTmpl->argInit.evtCnt, fsm->state, evt);
    BKF_LOG_DEBUG(BKF_LOG_HND, "before_dispatch: fsm(%#x)/appData(%s), state(%u/%s)/evt(%u/%s)/proc(%#x/%s)\n",
        BKF_MASK_ADDR(fsm), BKF_FSM_GET_APP_DATA_STR(fsm, buf, sizeof(buf)), fsm->state,
        BkfFsmTmplGetStateStr(fsmTmpl, fsm->state), evt, BkfFsmTmplGetEvtStr(fsmTmpl, evt),
        BKF_MASK_ADDR(procItem->proc), procItem->procStr);
    if (procItem->proc == BKF_FSM_PROC_IMPOSSIBLE) {
        BKF_LOG_ERROR(BKF_LOG_HND, "fsm(%#x)/appData(%s), state(%u/%s)/evt(%u/%s)/proc(%u/%s)\n", BKF_MASK_ADDR(fsm),
            BKF_FSM_GET_APP_DATA_STR(fsm, buf, sizeof(buf)), fsm->state, BkfFsmTmplGetStateStr(fsmTmpl, fsm->state),
            evt, BkfFsmTmplGetEvtStr(fsmTmpl, evt), BKF_MASK_ADDR(procItem->proc), procItem->procStr);
        BKF_FSM_INC_STAT(fsm, dispatchProcImpossibleCnt);
        return BKF_FSM_DISPATCH_RET_IMPOSSIBLE;
    }

    if (procItem->proc == BKF_FSM_PROC_IGNORE) {
        BKF_FSM_INC_STAT(fsm, dispatchProcIgnoreCnt);
        ret = BKF_OK;
    } else {
        ret = procItem->proc(dispatchParam1, dispatchParam2, dispatchParam3);
        fsm = VOS_NULL; /* 回调中可能删除 */
    }

    return ret;
}

uint8_t BkfFsmGetState(BkfFsm *fsm)
{
    if ((fsm == VOS_NULL) || !BKF_SIGN_IS_VALID(fsm->sign, BKF_FSM_SIGN) || (fsm->tmpl == VOS_NULL) ||
        !BKF_FSM_STATE_IS_VALID(fsm->tmpl, fsm->state)) {
        return BKF_FSM_STATE_INVALID;
    }

    return fsm->state;
}

uint32_t BkfFsmChgState(BkfFsm *fsm, uint8_t newState)
{
    BkfFsmTmpl *fsmTmpl = VOS_NULL;
    uint32_t *stateTransCnt = VOS_NULL;
    uint8_t buf[BKF_1K / 4];

    if ((fsm == VOS_NULL) || !BKF_SIGN_IS_VALID(fsm->sign, BKF_FSM_SIGN) || (fsm->tmpl == VOS_NULL)) {
        return BKF_ERR;
    }
    fsmTmpl = fsm->tmpl;
    BKF_FSM_INC_STAT(fsm, chgStateTotalCnt);

    if (!BKF_FSM_STATE_IS_VALID(fsmTmpl, newState)) {
        BKF_LOG_ERROR(BKF_LOG_HND, "fsm(%#x)/appData(%s), new state(%u/[0, %u)) is invalid\n", BKF_MASK_ADDR(fsm),
            BKF_FSM_GET_APP_DATA_STR(fsm, buf, sizeof(buf)), fsm->state, newState);
        BKF_FSM_INC_STAT(fsm, chgStateNewStateInvalidCnt);
        return BKF_ERR;
    }

    BKF_LOG_DEBUG(BKF_LOG_HND, "fsm(%#x)/appData(%s), state(%u, %s) -> (%u, %s), chg_ok\n", BKF_MASK_ADDR(fsm),
        BKF_FSM_GET_APP_DATA_STR(fsm, buf, sizeof(buf)), fsm->state, BkfFsmTmplGetStateStr(fsmTmpl, fsm->state),
        newState, BkfFsmTmplGetStateStr(fsmTmpl, newState));
    BKF_FSM_INC_STAT(fsm, chgStateOkCnt);
    if ((fsm->statInfo != VOS_NULL) && BKF_FSM_STATE_IS_VALID(fsmTmpl, fsm->state)) {
        stateTransCnt = BKF_GET_ARY_SIM_MTRX_ITEM(fsm->statInfo->stateTransCnt, fsmTmpl->argInit.stateCnt, fsm->state,
            newState);
        (*stateTransCnt)++;
    }
    BkfFsmStateChgSpy(fsm, newState);
    fsm->state = newState;
    return BKF_OK;
}

void BkfFsmStateChgSpy(BkfFsm *fsm, uint8_t newState)
{
    (void)fsm;
    (void)newState;
}

const char *BkfFsmGetStateStr(BkfFsm *fsm)
{
    if (fsm == VOS_NULL) {
        return "fsm_null";
    }
    if (!BKF_SIGN_IS_VALID(fsm->sign, BKF_FSM_SIGN)) {
        return "fsm_invalid";
    }

    return BkfFsmTmplGetStateStr(fsm->tmpl, fsm->state);
}

char *BkfFsmGetStrChkParam(BkfFsm *fsm, uint8_t *buf, int32_t bufLen)
{
    BkfFsmTmpl *fsmTmpl = VOS_NULL;
    int32_t ret;

    if (fsm == VOS_NULL) {
        return "fsm_null";
    }
    fsmTmpl = fsm->tmpl;

    if ((buf == VOS_NULL) || (bufLen <= 0)) {
        return "buf_invalid";
    }
    if (!BKF_SIGN_IS_VALID(fsm->sign, BKF_FSM_SIGN)) {
        ret = snprintf_truncated_s((char *)buf, (uint32_t)bufLen, "sign(%#x, %#x), not equ, ng", fsm->sign,
            BKF_FSM_SIGN);
        if (ret < 0) {
            goto error;
        }

        return (char *)buf;
    }
    if (fsmTmpl == VOS_NULL) {
        ret = snprintf_truncated_s((char *)buf, (uint32_t)bufLen, "tmpl(%#x), null & ng", BKF_MASK_ADDR(fsmTmpl));
        if (ret < 0) {
            goto error;
        }

        return (char *)buf;
    }

    return VOS_NULL;

error:

    return "buf_not_enough";
}
int32_t BkfFsmAppendStateTransInfoStrGetTitle(BkfFsmTmpl *fsmTmpl, uint8_t *buf, int32_t bufLen)
{
    int32_t ret;
    uint8_t stateTo;
    int32_t usedLen = 0;

    for (stateTo = 0; stateTo < fsmTmpl->argInit.stateCnt; stateTo++) {
        ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)bufLen - usedLen, "State%-3u: %s\n", stateTo,
            BkfFsmTmplGetStateStr(fsmTmpl, stateTo));
        if (ret < 0) {
            return ret;
        }
        usedLen += ret;
    }

    ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)bufLen - usedLen, "%-8s", "F~T>>");
    if (ret < 0) {
        return ret;
    }
    usedLen += ret;
    for (stateTo = 0; stateTo < fsmTmpl->argInit.stateCnt; stateTo++) {
        ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)bufLen - usedLen, "|State%-3u", stateTo);
        if (ret < 0) {
            return ret;
        }
        usedLen += ret;
    }

    ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)bufLen - usedLen, "\n");
    if (ret < 0) {
        return ret;
    }
    usedLen += ret;

    return usedLen;
}
int32_t BkfFsmAppendStateTransInfoStr(BkfFsm *fsm, uint8_t *buf, int32_t bufLen)
{
    BkfFsmTmpl *fsmTmpl = fsm->tmpl;
    int32_t usedLen = 0;
    int32_t ret;
    uint8_t stateFrom;
    uint8_t stateTo;
    uint32_t *stateTransCnt = VOS_NULL;

    /*
    打印状态迁移表统计信息：
            >>> toState0    toState1
    fromState0: 100          0
    fromState1: 123          200
    */
    ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)bufLen - usedLen, "%s", "===stateTransCnt===\n");
    if (ret < 0) {
        return ret;
    }
    usedLen += ret;

    ret = BkfFsmAppendStateTransInfoStrGetTitle(fsmTmpl, buf + usedLen, bufLen - usedLen);
    if (ret < 0) {
        return ret;
    }
    usedLen += ret;

    for (stateFrom = 0; stateFrom < fsmTmpl->argInit.stateCnt; stateFrom++) {
        ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)bufLen - usedLen, "State%-3u", stateFrom);
        if (ret < 0) {
            return ret;
        }
        usedLen += ret;

        for (stateTo = 0; stateTo < fsmTmpl->argInit.stateCnt; stateTo++) {
            stateTransCnt = BKF_GET_ARY_SIM_MTRX_ITEM(fsm->statInfo->stateTransCnt, fsmTmpl->argInit.stateCnt,
                stateFrom, stateTo);
            /* 注意，由于输出缓冲大小，需要控制输出字符数，因此，stateTransCnt不和title对齐。 */
            ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)bufLen - usedLen, "|%-8u", *stateTransCnt);
            if (ret < 0) {
                return ret;
            }
            usedLen += ret;
        }

        if (stateFrom != (fsmTmpl->argInit.stateCnt - 1)) {
            ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)bufLen - usedLen, "\n");
            if (ret < 0) {
                return ret;
            }
            usedLen += ret;
        }
    }

    return usedLen;
}
int32_t BkfFsmAppendDispatchStatInfoStr(BkfFsm *fsm, uint8_t *buf, int32_t bufLen)
{
    BkfFsmStatInfo *info = fsm->statInfo;
    int32_t usedLen = 0;
    int32_t ret;

    ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)bufLen - usedLen,
        "===dispatchStatInfo===\n"
        "TotalCnt(%" VOS_PRIu64 ")\n"
        "ProcRetOkCnt(%" VOS_PRIu64 ")\n"
        "ProcIgnoreCnt(%" VOS_PRIu64 ")\n",
        info->dispatchTotalCnt, info->dispatchProcRetOkCnt, info->dispatchProcIgnoreCnt);
    if (ret < 0) {
        return ret;
    }
    usedLen += ret;

    if (info->dispatchFsmCurStateNgCnt > 0) {
        ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)bufLen - usedLen,
            "FsmCurStateNgCnt(%" VOS_PRIu64 ")\n", info->dispatchFsmCurStateNgCnt);
        if (ret < 0) {
            return ret;
        }
        usedLen += ret;
    }
    if (info->dispatchInputEvtNgCnt > 0) {
        ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)bufLen - usedLen,
            "InputEvtNgCnt(%" VOS_PRIu64 ")\n", info->dispatchInputEvtNgCnt);
        if (ret < 0) {
            return ret;
        }
        usedLen += ret;
    }
    if (info->dispatchProcImpossibleCnt > 0) {
        ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)bufLen - usedLen,
            "ProcImpossibleCnt(%" VOS_PRIu64 ")\n", info->dispatchProcImpossibleCnt);
        if (ret < 0) {
            return ret;
        }
        usedLen += ret;
    }
    if (info->dispatchProcRetNOkCnt > 0) {
        ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)bufLen - usedLen,
            "ProcRetNOkCnt(%" VOS_PRIu64 ")\n", info->dispatchProcRetNOkCnt);
        if (ret < 0) {
            return ret;
        }
        usedLen += ret;
    }

    return usedLen;
}
int32_t BkfFsmAppendStatInfoStr(BkfFsm *fsm, uint8_t *buf, int32_t bufLen)
{
    BkfFsmStatInfo *info = fsm->statInfo;
    int32_t usedLen = 0;
    int32_t ret;

    ret = BkfFsmAppendDispatchStatInfoStr(fsm, buf, bufLen);
    if (ret < 0) {
        return ret;
    }
    usedLen += ret;

    ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)bufLen - usedLen,
        "===chgStateStatInfo===\n"
        "TotalCnt(%u)\n"
        "OkCnt(%u)\n"
        "NewStateInvalidCnt(%u)\n",
        info->chgStateTotalCnt, info->chgStateOkCnt, info->chgStateNewStateInvalidCnt);
    if (ret < 0) {
        return ret;
    }
    usedLen += ret;

    ret = BkfFsmAppendStateTransInfoStr(fsm, buf + usedLen, bufLen - usedLen);
    if (ret < 0) {
        return ret;
    }
    usedLen += ret;

    return usedLen;
}
char *BkfFsmGetStr(BkfFsm *fsm, uint8_t *buf, int32_t bufLen)
{
    BkfFsmTmpl *fsmTmpl = VOS_NULL;
    char *retStr = VOS_NULL;
    int32_t ret;
    int32_t usedLen = 0;

    retStr = BkfFsmGetStrChkParam(fsm, buf, bufLen);
    if (retStr != VOS_NULL) {
        return retStr;
    }
    fsmTmpl = fsm->tmpl;

    ret = snprintf_truncated_s((char *)buf, (uint32_t)bufLen, "sign(%#x)/state(%u, %s), tmpl(%#x, %s)/statInfo(%#x)\n",
        fsm->sign, fsm->state, BkfFsmTmplGetStateStr(fsmTmpl, fsm->state), BKF_MASK_ADDR(fsmTmpl), fsmTmpl->name,
        BKF_MASK_ADDR(fsm->statInfo));
    if (ret < 0) {
        goto error;
    }

    usedLen += ret;
    if (fsm->statInfo != VOS_NULL) {
        ret = BkfFsmAppendStatInfoStr(fsm, buf + usedLen, bufLen - usedLen);
        if (ret < 0) {
            goto error;
        }
    }

    return (char *)buf;

error:

    return "buf_not_enough";
}

#endif

#if BKF_BLOCK("disp")
void BkfFsmTmplDisp(BkfFsmTmpl *fsmTmpl)
{
    if ((fsmTmpl == VOS_NULL) || (fsmTmpl->argInit.disp == VOS_NULL)) {
        return;
    }
    BkfDisp *disp = fsmTmpl->argInit.disp;

    BKF_DISP_PRINTF(disp, "===argInit===\n");
    BKF_DISP_PRINTF(disp, "name(%s)/dbgOn(%u)\n", fsmTmpl->name, fsmTmpl->argInit.dbgOn);
    BKF_DISP_PRINTF(disp, "memMng(%#x)/disp(%#x)/log(%#x)\n", BKF_MASK_ADDR(fsmTmpl->argInit.memMng),
        BKF_MASK_ADDR(fsmTmpl->argInit.disp), BKF_MASK_ADDR(fsmTmpl->argInit.log));
    BKF_DISP_PRINTF(disp, "stateCnt(%u)/stateInit(%u, %s), stateStrTbl(%#x)\n", fsmTmpl->argInit.stateCnt,
        fsmTmpl->argInit.stateInit, BkfFsmTmplGetStateStr(fsmTmpl, fsmTmpl->argInit.stateInit),
        BKF_MASK_ADDR(fsmTmpl->argInit.stateStrTbl));
    BKF_DISP_PRINTF(disp, "evtCnt(%u)/evtStrTbl(%#x)\n", fsmTmpl->argInit.evtCnt,
        BKF_MASK_ADDR(fsmTmpl->argInit.evtStrTbl));
    BKF_DISP_PRINTF(disp, "stateEvtProcItemMtrx(%#x)\n", BKF_MASK_ADDR(fsmTmpl->argInit.stateEvtProcItemMtrx));
    BKF_DISP_PRINTF(disp, "getAppDataStrOrNull(%#x)\n", BKF_MASK_ADDR(fsmTmpl->argInit.getAppDataStrOrNull));
    BKF_DISP_PRINTF(disp, "===runTimeInfo===\n");
    BKF_DISP_PRINTF(disp, "log(%#x)\n", BKF_MASK_ADDR(fsmTmpl->log));
    BKF_DISP_PRINTF(disp, "fsmCnt(%d)\n", fsmTmpl->fsmCnt);
    BKF_DISP_PRINTF(disp, "---dispatchCntInfo---\n");
    BKF_DISP_PRINTF(disp, "dispatchTotalCnt(%" VOS_PRIu64 ")", fsmTmpl->dispatchTotalCnt);
    BKF_DISP_PRINTF(disp, "dispatchFsmCurStateNgCnt(%" VOS_PRIu64 ")\n", fsmTmpl->dispatchFsmCurStateNgCnt);
    BKF_DISP_PRINTF(disp, "dispatchInputEvtNgCnt(%" VOS_PRIu64 ")\n", fsmTmpl->dispatchInputEvtNgCnt);
    BKF_DISP_PRINTF(disp, "dispatchProcImpossibleCnt(%" VOS_PRIu64 ")\n", fsmTmpl->dispatchProcImpossibleCnt);
    BKF_DISP_PRINTF(disp, "dispatchProcIgnoreCnt(%" VOS_PRIu64 ")\n", fsmTmpl->dispatchProcIgnoreCnt);
    BKF_DISP_PRINTF(disp, "dispatchProcRetOkCnt(%" VOS_PRIu64 ")\n", fsmTmpl->dispatchProcRetOkCnt);
    BKF_DISP_PRINTF(disp, "dispatchProcRetNOkCnt(%" VOS_PRIu64 ")\n", fsmTmpl->dispatchProcRetNOkCnt);
    BKF_DISP_PRINTF(disp, "---chgStateCntInfo---\n");
    BKF_DISP_PRINTF(disp, "chgStateTotalCnt(%" VOS_PRIu64 ")\n", fsmTmpl->chgStateTotalCnt);
    BKF_DISP_PRINTF(disp, "chgStateNewStateInvalidCnt(%" VOS_PRIu64 ")\n", fsmTmpl->chgStateNewStateInvalidCnt);
    BKF_DISP_PRINTF(disp, "chgStateOkCnt(%" VOS_PRIu64 ")\n", fsmTmpl->chgStateOkCnt);
    return;
}

STATIC void BkfFsmTmplInitDisp(BkfFsmTmpl *fsmTmpl)
{
    BkfDisp *disp = fsmTmpl->argInit.disp;
    char *objName = fsmTmpl->name;
    uint32_t ret;

    ret = BkfDispRegObj(disp, objName, fsmTmpl);
    if (ret != BKF_OK) {
        BKF_ASSERT(0);
        return;
    }

    (void)BKF_DISP_REG_FUNC(disp, BkfFsmTmplDisp, "disp fsm template", objName, 0);
}

STATIC void BkfFsmTmplUninitDisp(BkfFsmTmpl *fsmTmpl)
{
    BkfDispUnregObj(fsmTmpl->argInit.disp, fsmTmpl->name);
}

#endif

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
