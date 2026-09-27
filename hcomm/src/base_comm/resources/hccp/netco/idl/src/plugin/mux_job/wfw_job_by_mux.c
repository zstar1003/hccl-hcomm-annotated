/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((jobxMng)->log)
#define BKF_MOD_NAME ((jobxMng)->name)

#include <sys/epoll.h>
#include <sys/timerfd.h>
#include <sys/eventfd.h>
#include <unistd.h>
#include <errno.h>
#include "wfw_linux_comm.h"
#include "bkf_bas_type_mthd.h"
#include "bkf_dl.h"
#include "bkf_str.h"
#include "bkf_assert.h"
#include "v_avll.h"
#include "v_stringlib.h"
#include "securec.h"
#include "wfw_job_by_mux.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
#define WFW_JOBX_SCHED_BY_EVT 1
#define WFW_JOBX_SCHED_BY_PIPE 2
#define WFW_JOBX_SCHED_MODE WFW_JOBX_SCHED_BY_EVT

#define WFW_JOBX_SIGN (0xaf98)
struct tagWfwJobxMng {
    uint16_t sign;
    uint8_t hasTrig : 1;
    uint8_t rsv1 : 7;
    uint8_t pad1[1];
    WfwJobxInitArg argInit;
    char *name;
    BkfLog *log;
    char jobName[BKF_NAME_LEN_MAX + 1];
    int fdTrigEvt;
    int fdTrigPipe[2];
    int fdTrigTmr; /* timerfd_settime() 除非参数错误，不会返回错误。因此，以此fd来兜底 */
    AVLL_TREE jobxTypeSet;
    AVLL_TREE jobxTypeNeedSchedSetByPrio;
    uint64_t seed;
    uint32_t cbCntMaxOfOneSched;
};

typedef struct tagBkfJobxTypePrioKey {
    uint32_t prio;
    uint64_t diffNum;
} BkfJobxTypePrioKey;
#define WFW_JOBX_INIT_PRIO_KEY(prioKey)                                                                                \
    do {                                                                                                               \
        (prioKey)->prio = 0;                                                                                           \
        (prioKey)->diffNum = 0xffffffffffffffff;                                                                       \
    } while (0)

typedef struct tagBkfJobxType {
    AVLL_NODE avlNode;
    AVLL_NODE avlNodePrio;
    uint32_t typeId;
    BkfJobxTypePrioKey prioKey;
    F_BKF_JOB_PROC proc;
    BkfDl jobIdSet;
} BkfJobxType;

#define WFW_JOBX_ID_SIGN (0xaf99)
struct tagBkfJobId {
    uint16_t sign;
    uint8_t lockDel : 1;
    uint8_t softDel : 1;
    uint8_t rsv1 : 6;
    uint8_t pad1[1];
    BkfJobxType *jobxType;
    BkfDlNode dlNode;
    void *paramCreate;
    char name[0];
};

typedef struct tagBkfJobxSchedCtx {
    uint32_t runCostTotal;
    uint32_t cbCntTotal;
    BOOL yield;
} BkfJobxSchedCtx;

/* proc & on msg */
STATIC uint32_t WfwJobxInitTrig(WfwJobxMng *jobxMng);
STATIC uint32_t WfwJobxTrigSched(WfwJobxMng *jobxMng);
STATIC void WfwJobxUninitTrig(WfwJobxMng *jobxMng);
STATIC void WfwJobxSched(WfwJobxMng *jobxMng);

STATIC uint32_t WfwJobxOnRegType(WfwJobxMng *jobxMng, uint32_t typeId, F_BKF_JOB_PROC proc, uint32_t prio);
STATIC BkfJobId *WfwJobxOnCreateJobId(WfwJobxMng *jobxMng, uint32_t typeId, char *name, void *param);
STATIC void WfwJobxOnDeleteJobId(WfwJobxMng *jobxMng, BkfJobId *jobId);

/* data op */
STATIC BkfJobxType *WfwJobxAddType(WfwJobxMng *jobxMng, uint32_t typeId, F_BKF_JOB_PROC proc, uint32_t prio);
STATIC void WfwJobxDelType(WfwJobxMng *jobxMng, BkfJobxType *jobxType);
STATIC void WfwJobxDelAllType(WfwJobxMng *jobxMng);
STATIC BkfJobxType *WfwJobxFindType(WfwJobxMng *jobxMng, uint32_t typeId);
STATIC BkfJobxType *WfwJobxGetFirstType(WfwJobxMng *jobxMng, void **itorOutOrNull);
STATIC BkfJobxType *WfwJobxGetNextType(WfwJobxMng *jobxMng, void **itorInOut);

STATIC int32_t WfwJobxCmpJobTypePrioKey(BkfJobxTypePrioKey *key1, BkfJobxTypePrioKey *key2);
STATIC uint32_t WfwJobxAttachTypePrio(WfwJobxMng *jobxMng, BkfJobxType *jobxType);
STATIC void WfwJobxDetachTypePrio(WfwJobxMng *jobxMng, BkfJobxType *jobxType);
STATIC BkfJobxType *WfwJobxGetPrioHighestType(WfwJobxMng *jobxMng);

STATIC BkfJobId *WfwJobxAddId(WfwJobxMng *jobxMng, BkfJobxType *jobxType, char *name, void *param);
STATIC void WfwJobxDelId(WfwJobxMng *jobxMng, BkfJobId *jobId);
STATIC void WfwJobxDelTypeAllId(WfwJobxMng *jobxMng, BkfJobxType *jobxType);
STATIC void WfwJobxMoveId2Tail(WfwJobxMng *jobxMng, BkfJobId *jobId);
STATIC BkfJobId *WfwJobxGetFirstId(WfwJobxMng *jobxMng, BkfJobxType *jobxType, void **itorOutOrNull);
STATIC BkfJobId *WfwJobxGetNextId(WfwJobxMng *jobxMng, BkfJobxType *jobxType, void **itorInOut);

#pragma pack()

#endif

#if BKF_BLOCK("公有函数定义")
/* func */
WfwJobxMng *WfwJobxInit(WfwJobxInitArg *arg)
{
    WfwJobxMng *jobxMng = VOS_NULL;
    BOOL argIsInvalid = VOS_FALSE;
    uint32_t len;
    uint32_t ret;

    argIsInvalid = (arg == VOS_NULL) || (arg->name == VOS_NULL) || (arg->memMng == VOS_NULL) ||
                   (arg->disp == VOS_NULL) || (arg->logCnt == VOS_NULL) || (arg->log == VOS_NULL) ||
                   (arg->pfm == VOS_NULL) || (arg->mux == VOS_NULL);
    if (argIsInvalid) {
        goto error;
    }

    len = sizeof(WfwJobxMng);
    jobxMng = BKF_MALLOC(arg->memMng, len);
    if (jobxMng == VOS_NULL) {
        goto error;
    }
    (void)memset_s(jobxMng, len, 0, len);
    jobxMng->fdTrigEvt = -1;
    jobxMng->fdTrigTmr = -1;
    jobxMng->fdTrigPipe[WFW_PIPE_FDR_IDX] = -1;
    jobxMng->fdTrigPipe[WFW_PIPE_FDW_IDX] = -1;
    jobxMng->argInit = *arg;
    jobxMng->name = BkfStrNew(arg->memMng, "%s", arg->name);
    jobxMng->argInit.name = jobxMng->name;
    jobxMng->log = arg->log;

    VOS_AVLL_INIT_TREE(jobxMng->jobxTypeSet, (AVLL_COMPARE)Bkfuint32_tCmp, BKF_OFFSET(BkfJobxType, typeId),
        BKF_OFFSET(BkfJobxType, avlNode));
    VOS_AVLL_INIT_TREE(jobxMng->jobxTypeNeedSchedSetByPrio, (AVLL_COMPARE)WfwJobxCmpJobTypePrioKey,
        BKF_OFFSET(BkfJobxType, prioKey), BKF_OFFSET(BkfJobxType, avlNodePrio));
    ret = WfwJobxInitTrig(jobxMng);
    if (ret != BKF_OK) {
        goto error;
    }

    BKF_SIGN_SET(jobxMng->sign, WFW_JOBX_SIGN);
    BKF_LOG_INFO(BKF_LOG_HND, "jobxMng(%#x), init ok\n", BKF_MASK_ADDR(jobxMng));
    return jobxMng;

error:

    WfwJobxUninit(jobxMng);
    return VOS_NULL;
}

void WfwJobxUninit(WfwJobxMng *jobxMng)
{
    if (jobxMng == VOS_NULL) {
        return;
    }
    BKF_LOG_INFO(BKF_LOG_HND, "jobxMng(%#x), 2uninit\n", BKF_MASK_ADDR(jobxMng));

    WfwJobxDelAllType(jobxMng);
    WfwJobxUninitTrig(jobxMng);
    BkfStrDel(jobxMng->name);
    BKF_SIGN_CLR(jobxMng->sign);
    BKF_FREE(jobxMng->argInit.memMng, jobxMng);
    return;
}

BkfIJob *WfwJobxBuildIJob(WfwJobxMng *jobxMng, BkfIJob *temp)
{
    int32_t err;

    if ((jobxMng == VOS_NULL) || (temp == VOS_NULL)) {
        return VOS_NULL;
    }

    (void)memset_s(temp, sizeof(BkfIJob), 0, sizeof(BkfIJob));
    err = snprintf_truncated_s(jobxMng->jobName, sizeof(jobxMng->jobName), "%s_job", jobxMng->name);
    if (err <= 0) {
        return VOS_NULL;
    }
    temp->name = jobxMng->jobName;
    temp->memMng = jobxMng->argInit.memMng;
    temp->runCostMax = jobxMng->argInit.runCostMax;
    temp->cookie = jobxMng;
    temp->regType = (F_BKF_IJOB_REG_TYPE)WfwJobxOnRegType;
    temp->createJobId = (F_BKF_IJOB_CREATE)WfwJobxOnCreateJobId;
    temp->deleteJobId = (F_BKF_IJOB_DELETE)WfwJobxOnDeleteJobId;
    return temp;
}

#endif

#if BKF_BLOCK("私有函数定义")
/* proc & on msg */
STATIC void WfwJobxOnTrigTmrFd(int fd, uint32_t curEvents, WfwJobxMng *jobxMng)
{
    (void)curEvents;
    ssize_t rlen;
    uint64_t temp;
    int ret;
    struct itimerspec newValue;

    rlen = read(fd, &temp, sizeof(temp));
    if (rlen != (ssize_t)sizeof(temp)) {
        BKF_LOG_WARN(BKF_LOG_HND, "rlen(%d)/errno(%d)\n", rlen, errno);
        /* 继续 */
    }
    (void)memset_s(&newValue, sizeof(struct itimerspec), 0, sizeof(struct itimerspec));
    ret = timerfd_settime(jobxMng->fdTrigTmr, 0, &newValue, NULL);
    if (ret == -1) {
        BKF_LOG_WARN(BKF_LOG_HND, "ret(%d)/errno(%d)\n", ret, errno);
        /* 继续 */
    }

    WfwJobxSched(jobxMng);
    return;
}
STATIC uint32_t WfwJobxInitTrigTmr(WfwJobxMng *jobxMng)
{
    int ret;
    struct itimerspec newValue;
    WfwArgForMux argForMux = {.selfCid = jobxMng->argInit.selfCid, .isMeshFd = VOS_FALSE};

    jobxMng->fdTrigTmr = timerfd_create(CLOCK_MONOTONIC, 0);
    if (jobxMng->fdTrigTmr == -1) {
        BKF_LOG_ERROR(BKF_LOG_HND, "fdTrigTmr(%d)/errno(%d)\n", jobxMng->fdTrigTmr, errno);
        return BKF_ERR;
    }
    ret = WfwSetFdNonBlock(jobxMng->fdTrigTmr);
    if (ret == -1) {
        BKF_LOG_ERROR(BKF_LOG_HND, "trigTmrFd(%d), ret(%d)/errno(%d)\n", jobxMng->fdTrigTmr, ret, errno);
        return BKF_ERR;
    }
    ret = WfwMuxAttachFd(jobxMng->argInit.mux, jobxMng->fdTrigTmr, EPOLLIN, (F_WFW_MUX_FD_PROC)WfwJobxOnTrigTmrFd,
        jobxMng, &argForMux);
    if (ret == -1) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%d)/errno(%d)\n", ret, errno);
        return BKF_ERR;
    }
    (void)memset_s(&newValue, sizeof(struct itimerspec), 0, sizeof(struct itimerspec));
    ret = timerfd_settime(jobxMng->fdTrigTmr, 0, &newValue, NULL);
    if (ret == -1) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%d)/errno(%d)\n", ret, errno);
        return BKF_ERR;
    }

    return BKF_OK;
}

#if (WFW_JOBX_SCHED_MODE == WFW_JOBX_SCHED_BY_EVT)
STATIC void WfwJobxOnTrigEvtFd(int fd, uint32_t curEvents, WfwJobxMng *jobxMng)
{
    (void)curEvents;
    int ret;
    eventfd_t value;

    ret = eventfd_read(fd, &value);
    if (ret == -1) {
        BKF_LOG_WARN(BKF_LOG_HND, "ret(%d)/errno(%d)\n", ret, errno);
        /* 继续 */
    }

    WfwJobxSched(jobxMng);
    return;
}
STATIC uint32_t WfwJobxInitTrig(WfwJobxMng *jobxMng)
{
    int ret;
    WfwArgForMux argForMux = {.selfCid = jobxMng->argInit.selfCid, .isMeshFd = VOS_FALSE};

    if (WfwJobxInitTrigTmr(jobxMng) != BKF_OK) {
        return BKF_ERR;
    }

    jobxMng->fdTrigEvt = eventfd(0, 0);
    if (jobxMng->fdTrigEvt == -1) {
        BKF_LOG_ERROR(BKF_LOG_HND, "fdEvtTrig(%d)/errno(%d)\n", jobxMng->fdTrigEvt, errno);
        return BKF_ERR;
    }
    ret = WfwSetFdNonBlock(jobxMng->fdTrigEvt);
    if (ret == -1) {
        BKF_LOG_ERROR(BKF_LOG_HND, "fdEvtTrig(%d), ret(%d)/errno(%d)\n", jobxMng->fdTrigEvt, ret, errno);
        return BKF_ERR;
    }
    ret = WfwMuxAttachFd(jobxMng->argInit.mux, jobxMng->fdTrigEvt, EPOLLIN, (F_WFW_MUX_FD_PROC)WfwJobxOnTrigEvtFd,
        jobxMng, &argForMux);
    if (ret == -1) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%d)/errno(%d)\n", ret, errno);
        return BKF_ERR;
    }

    return BKF_OK;
}

STATIC uint32_t WfwJobxTrigSched(WfwJobxMng *jobxMng)
{
    int ret;
    struct itimerspec newValue;

    if (jobxMng->hasTrig) {
        return BKF_OK;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "jobxMng(%#x)\n", BKF_MASK_ADDR(jobxMng));

    ret = eventfd_write(jobxMng->fdTrigEvt, 1);
    if (ret == -1) {
        BKF_LOG_WARN(BKF_LOG_HND, "ret(%d)/errno(%d)\n", ret, errno);
        newValue.it_value.tv_sec = 1;
        newValue.it_value.tv_nsec = 0;
        newValue.it_interval.tv_sec = 1;
        newValue.it_interval.tv_nsec = 0;
        ret = timerfd_settime(jobxMng->fdTrigTmr, 0, &newValue, NULL);
        if (ret == -1) {
            BKF_LOG_ERROR(BKF_LOG_HND, "ret(%d)/errno(%d)\n", ret, errno);
            BKF_ASSERT(0);
            return BKF_ERR;
        }
    }

    jobxMng->hasTrig = VOS_TRUE;
    return BKF_OK;
}

#elif (WFW_JOBX_SCHED_MODE == WFW_JOBX_SCHED_BY_PIPE)
STATIC void WfwJobxOnTrigPipeFdRead(int fd, uint32_t curEvents, WfwJobxMng *jobxMng)
{
    ssize_t rlen;
    char temp;

    rlen = read(fd, &temp, sizeof(temp));
    if (rlen != sizeof(temp)) {
        BKF_LOG_WARN(BKF_LOG_HND, "rlen(%d)/errno(%d)\n", rlen, errno);
        /* 继续 */
    }

    WfwJobxSched(jobxMng);
    return;
}
STATIC uint32_t WfwJobxInitTrig(WfwJobxMng *jobxMng)
{
    int ret;
    WfwArgForMux argForMux = {.selfCid = jobxMng->argInit.selfCid, .isMeshFd = VOS_FALSE};

    if (WfwJobxInitTrigTmr(jobxMng) != BKF_OK) {
        return BKF_ERR;
    }

    ret = pipe(jobxMng->fdTrigPipe);
    if (ret == -1) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%d)/errno(%d)\n", ret, errno);
        return BKF_ERR;
    }
    ret = WfwMuxAttachFd(jobxMng->argInit.mux, jobxMng->fdTrigPipe[WFW_PIPE_FDR_IDX], EPOLLIN,
        (F_WFW_MUX_FD_PROC)WfwJobxOnTrigPipeFdRead, jobxMng, &argForMux);
    if (ret == -1) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%d)/errno(%d)\n", ret, errno);
        return BKF_ERR;
    }

    return BKF_OK;
}

STATIC uint32_t WfwJobxTrigSched(WfwJobxMng *jobxMng)
{
    ssize_t wlen;
    struct itimerspec newValue;
    int ret;

    if (jobxMng->hasTrig) {
        return BKF_OK;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "jobxMng(%#x)\n", BKF_MASK_ADDR(jobxMng));

    wlen = write(jobxMng->fdTrigPipe[WFW_PIPE_FDW_IDX], "a", 1);
    if (wlen != 1) {
        BKF_LOG_WARN(BKF_LOG_HND, "wlen(%d)/errno(%d)\n", wlen, errno);

        newValue.it_value.tv_sec = 1;
        newValue.it_value.tv_nsec = 0;
        newValue.it_interval.tv_sec = 1;
        newValue.it_interval.tv_nsec = 0;
        ret = timerfd_settime(jobxMng->fdTrigTmr, 0, &newValue, NULL);
        if (ret == -1) {
            BKF_LOG_ERROR(BKF_LOG_HND, "ret(%d)/errno(%d)\n", ret, errno);
            BKF_ASSERT(0);
            return BKF_ERR;
        }
    }

    jobxMng->hasTrig = VOS_TRUE;
    return BKF_OK;
}

#else
STATIC uint32_t WfwJobxInitTrig(WfwJobxMng *jobxMng)
{
    return WfwJobxInitTrigTmr(jobxMng);
}

STATIC uint32_t WfwJobxTrigSched(WfwJobxMng *jobxMng)
{
    struct itimerspec newValue;
    int ret;

    if (jobxMng->hasTrig) {
        return BKF_OK;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "jobxMng(%#x)\n", BKF_MASK_ADDR(jobxMng));

    newValue.it_value.tv_sec = 0;
    newValue.it_value.tv_nsec = 1;
    newValue.it_interval.tv_sec = 0;
    newValue.it_interval.tv_nsec = 0;
    ret = timerfd_settime(jobxMng->fdTrigTmr, 0, &newValue, NULL);
    if (ret == -1) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%d)/errno(%d)\n", ret, errno);
        BKF_ASSERT(0);
        return BKF_ERR;
    }

    jobxMng->hasTrig = VOS_TRUE;
    return BKF_OK;
}

#endif

STATIC void WfwJobxUninitTrig(WfwJobxMng *jobxMng)
{
    if (jobxMng->fdTrigEvt != -1) {
        (void)WfwMuxDetachFd(jobxMng->argInit.mux, jobxMng->fdTrigEvt);
        (void)close(jobxMng->fdTrigEvt);
        jobxMng->fdTrigEvt = -1;
    }
    if (jobxMng->fdTrigPipe[WFW_PIPE_FDR_IDX] != -1) {
        (void)WfwMuxDetachFd(jobxMng->argInit.mux, jobxMng->fdTrigPipe[WFW_PIPE_FDR_IDX]);
        (void)close(jobxMng->fdTrigPipe[WFW_PIPE_FDR_IDX]);
        jobxMng->fdTrigPipe[WFW_PIPE_FDR_IDX] = -1;
    }
    if (jobxMng->fdTrigPipe[WFW_PIPE_FDW_IDX] != -1) {
        /* 写pipe没有加入mux。不需要detach */
        (void)close(jobxMng->fdTrigPipe[WFW_PIPE_FDW_IDX]);
        jobxMng->fdTrigPipe[WFW_PIPE_FDW_IDX] = -1;
    }
    if (jobxMng->fdTrigTmr != -1) {
        (void)WfwMuxDetachFd(jobxMng->argInit.mux, jobxMng->fdTrigTmr);
        (void)close(jobxMng->fdTrigTmr);
        jobxMng->fdTrigTmr = -1;
    }
}

STATIC void WfwJobxSchedOneJobId(WfwJobxMng *jobxMng, BkfJobxType *jobxType, BkfJobId *jobId, BkfJobxSchedCtx *ctx)
{
    uint32_t ret;
    uint32_t runcost;
    BOOL needDel = VOS_FALSE;

    /*
    1. 初始化&回调
    2. 更新数据
    */

    /* 1 */
    WfwJobxMoveId2Tail(jobxMng, jobId);
    runcost = 0;
    jobId->lockDel = VOS_TRUE;
    ret = jobxType->proc(jobId->paramCreate, &runcost);
    jobId->lockDel = VOS_FALSE;
    BKF_LOG_DEBUG(BKF_LOG_HND, "ret(%u)\n", ret);
    /* 2 */
    ctx->cbCntTotal++;
    needDel = (ret == BKF_JOB_FINSIH) || jobId->softDel;
    if (needDel) {
        ctx->runCostTotal += runcost;
        ctx->yield = (jobxMng->argInit.runCostMax == BKF_JOB_RUN_COST_INVALID) ||
                     (ctx->runCostTotal >= jobxMng->argInit.runCostMax);
        WfwJobxDelId(jobxMng, jobId);
    } else if (ret == BKF_JOB_CONTINUE) {
        ctx->yield = VOS_TRUE;
    } else {
        /* 继续调度 */
        BKF_LOG_INFO(BKF_LOG_HND, "ret(%u)\n", ret);
    }

    return;
}
STATIC void WfwJobxSchedOneJobType(WfwJobxMng *jobxMng, BkfJobxType *jobxType, BkfJobxSchedCtx *ctx)
{
    BkfJobId *jobId = VOS_NULL;
    BOOL jobIdHasSched = VOS_FALSE;

    while ((jobId = WfwJobxGetFirstId(jobxMng, jobxType, VOS_NULL)) != VOS_NULL) {
        if (!jobIdHasSched) {
            jobIdHasSched = VOS_TRUE;
            WfwJobxDetachTypePrio(jobxMng, jobxType);
            jobxType->prioKey.diffNum = BKF_GET_NEXT_VAL(jobxMng->seed);
            WfwJobxAttachTypePrio(jobxMng, jobxType);
        }

        WfwJobxSchedOneJobId(jobxMng, jobxType, jobId, ctx);
        jobId = VOS_NULL; /* 一定要置空 */
        if (ctx->yield) {
            break;
        }
    }

    if ((jobId = WfwJobxGetFirstId(jobxMng, jobxType, VOS_NULL)) == VOS_NULL) {
        WfwJobxDetachTypePrio(jobxMng, jobxType);
    }
    return;
}
STATIC void WfwJobxSched(WfwJobxMng *jobxMng)
{
    BkfJobxType *jobxType = VOS_NULL;
    BkfJobxSchedCtx ctx = {0};
    if (jobxMng == VOS_NULL) {
        return;
    }
    while ((jobxType = WfwJobxGetPrioHighestType(jobxMng)) != VOS_NULL) {
        WfwJobxSchedOneJobType(jobxMng, jobxType, &ctx);
        if (ctx.yield) {
            break;
        }
    }

    jobxMng->cbCntMaxOfOneSched = BKF_GET_MAX(jobxMng->cbCntMaxOfOneSched, ctx.cbCntTotal);
    jobxMng->hasTrig = VOS_FALSE;
    if ((jobxType = WfwJobxGetPrioHighestType(jobxMng)) != VOS_NULL) {
        (void)WfwJobxTrigSched(jobxMng);
    }
    return;
}

STATIC uint32_t WfwJobxOnRegType(WfwJobxMng *jobxMng, uint32_t typeId, F_BKF_JOB_PROC proc, uint32_t prio)
{
    BkfJobxType *jobxType = VOS_NULL;

    if (jobxMng == VOS_NULL || !BKF_SIGN_IS_VALID(jobxMng->sign, WFW_JOBX_SIGN)) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }
    BKF_LOG_INFO(BKF_LOG_HND, "jobxMng(%#x), typeId(%d)/proc(%#x)/prio(%#x)\n", BKF_MASK_ADDR(jobxMng), typeId,
        BKF_MASK_ADDR(proc), prio);

    jobxType = WfwJobxAddType(jobxMng, typeId, proc, prio);
    if (jobxType == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "jobxType(%#x)\n", BKF_MASK_ADDR(jobxType));
        return BKF_ERR;
    }

    return BKF_OK;
}

STATIC BkfJobId *WfwJobxOnCreateJobId(WfwJobxMng *jobxMng, uint32_t typeId, char *name, void *param)
{
    BkfJobxType *jobxType = VOS_NULL;
    BkfJobId *jobId = VOS_NULL;

    if (jobxMng == VOS_NULL || !BKF_SIGN_IS_VALID(jobxMng->sign, WFW_JOBX_SIGN)) {
        BKF_ASSERT(0);
        return VOS_NULL;
    }
    BKF_LOG_INFO(BKF_LOG_HND, "jobxMng(%#x), typeId(%d)/name(%s)/param(%#x)\n", BKF_MASK_ADDR(jobxMng), typeId, name,
        BKF_MASK_ADDR(param));

    jobxType = WfwJobxFindType(jobxMng, typeId);
    if (jobxType == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "jobxType(%#x)\n", BKF_MASK_ADDR(jobxType));
        return VOS_NULL;
    }
    jobId = WfwJobxAddId(jobxMng, jobxType, name, param);
    if (jobId == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "jobId(%#x)\n", BKF_MASK_ADDR(jobId));
        return VOS_NULL;
    }

    (void)WfwJobxAttachTypePrio(jobxMng, jobxType);
    (void)WfwJobxTrigSched(jobxMng);
    return jobId;
}

STATIC void WfwJobxOnDeleteJobId(WfwJobxMng *jobxMng, BkfJobId *jobId)
{
    if (jobxMng == VOS_NULL || !BKF_SIGN_IS_VALID(jobxMng->sign, WFW_JOBX_SIGN) || jobId == VOS_NULL) {
        BKF_ASSERT(0);
        return;
    }
    if (!BKF_SIGN_IS_VALID(jobId->sign, WFW_JOBX_ID_SIGN)) {
        BKF_ASSERT(0);
        return;
    }
    BKF_LOG_INFO(BKF_LOG_HND, "jobxMng(%#x), jobId(%#x)\n", BKF_MASK_ADDR(jobxMng), BKF_MASK_ADDR(jobId));

    WfwJobxDelId(jobxMng, jobId);
    return;
}

/* data op */
STATIC BkfJobxType *WfwJobxAddType(WfwJobxMng *jobxMng, uint32_t typeId, F_BKF_JOB_PROC proc, uint32_t prio)
{
    BkfJobxType *jobxType = VOS_NULL;
    uint32_t len;
    BOOL insOk = VOS_FALSE;

    len = sizeof(BkfJobxType);
    jobxType = BKF_MALLOC(jobxMng->argInit.memMng, len);
    if (jobxType == VOS_NULL) {
        goto error;
    }
    (void)memset_s(jobxType, len, 0, len);
    jobxType->typeId = typeId;
    jobxType->prioKey.prio = prio;
    jobxType->prioKey.diffNum = BKF_GET_NEXT_VAL(jobxMng->seed);
    jobxType->proc = proc;
    BKF_DL_INIT(&jobxType->jobIdSet);
    VOS_AVLL_INIT_NODE(jobxType->avlNode);
    VOS_AVLL_INIT_NODE(jobxType->avlNodePrio);
    insOk = VOS_AVLL_INSERT(jobxMng->jobxTypeSet, jobxType->avlNode);
    if (!insOk) {
        goto error;
    }

    return jobxType;

error:

    if (jobxType != VOS_NULL) {
        /* insOk 一定为false */
        BKF_FREE(jobxMng->argInit.memMng, jobxType);
    }
    return VOS_NULL;
}

STATIC void WfwJobxDelType(WfwJobxMng *jobxMng, BkfJobxType *jobxType)
{
    WfwJobxDelTypeAllId(jobxMng, jobxType);
    WfwJobxDetachTypePrio(jobxMng, jobxType);
    VOS_AVLL_DELETE(jobxMng->jobxTypeSet, jobxType->avlNode);
    BKF_FREE(jobxMng->argInit.memMng, jobxType);
    return;
}

STATIC void WfwJobxDelAllType(WfwJobxMng *jobxMng)
{
    BkfJobxType *jobxType = VOS_NULL;
    void *itor = VOS_NULL;

    for (jobxType = WfwJobxGetFirstType(jobxMng, &itor); jobxType != VOS_NULL;
         jobxType = WfwJobxGetNextType(jobxMng, &itor)) {
        WfwJobxDelType(jobxMng, jobxType);
    }
    return;
}

STATIC BkfJobxType *WfwJobxFindType(WfwJobxMng *jobxMng, uint32_t typeId)
{
    BkfJobxType *jobxType = VOS_NULL;

    jobxType = VOS_AVLL_FIND(jobxMng->jobxTypeSet, &typeId);
    return jobxType;
}

STATIC BkfJobxType *WfwJobxGetFirstType(WfwJobxMng *jobxMng, void **itorOutOrNull)
{
    BkfJobxType *jobxType = VOS_NULL;

    jobxType = VOS_AVLL_FIRST(jobxMng->jobxTypeSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (jobxType != VOS_NULL) ? VOS_AVLL_NEXT(jobxMng->jobxTypeSet, jobxType->avlNode) : VOS_NULL;
    }

    return jobxType;
}

STATIC BkfJobxType *WfwJobxGetNextType(WfwJobxMng *jobxMng, void **itorInOut)
{
    BkfJobxType *jobxType = VOS_NULL;

    jobxType = (*itorInOut);
    *itorInOut = (jobxType != VOS_NULL) ? VOS_AVLL_NEXT(jobxMng->jobxTypeSet, jobxType->avlNode) : VOS_NULL;
    return jobxType;
}

STATIC int32_t WfwJobxCmpJobTypePrioKey(BkfJobxTypePrioKey *key1, BkfJobxTypePrioKey *key2)
{
    int32_t ret;

    ret = BKF_CMP_X(key1->prio, key2->prio);
    if (ret != 0) {
        return ret;
    }
    return BKF_CMP_X(key1->diffNum, key2->diffNum);
}

STATIC uint32_t WfwJobxAttachTypePrio(WfwJobxMng *jobxMng, BkfJobxType *jobxType)
{
    BOOL insOk = VOS_FALSE;

    if (!VOS_AVLL_IN_TREE(jobxType->avlNodePrio)) {
        jobxType->prioKey.diffNum = BKF_GET_NEXT_VAL(jobxMng->seed);
        insOk = VOS_AVLL_INSERT(jobxMng->jobxTypeNeedSchedSetByPrio, jobxType->avlNodePrio);
        if (!insOk) {
            BKF_ASSERT(0);
            return BKF_ERR;
        }
    }

    return BKF_OK;
}

STATIC void WfwJobxDetachTypePrio(WfwJobxMng *jobxMng, BkfJobxType *jobxType)
{
    if (VOS_AVLL_IN_TREE(jobxType->avlNodePrio)) {
        VOS_AVLL_DELETE(jobxMng->jobxTypeNeedSchedSetByPrio, jobxType->avlNodePrio);
        VOS_AVLL_INIT_NODE(jobxType->avlNodePrio);
    }
    return;
}

STATIC BkfJobxType *WfwJobxGetPrioHighestType(WfwJobxMng *jobxMng)
{
    BkfJobxType *jobxType = VOS_NULL;

    jobxType = VOS_AVLL_LAST(jobxMng->jobxTypeNeedSchedSetByPrio);
    return jobxType;
}

STATIC BkfJobId *WfwJobxAddId(WfwJobxMng *jobxMng, BkfJobxType *jobxType, char *name, void *param)
{
    BkfJobId *jobId = VOS_NULL;
    uint32_t len;
    uint32_t strLen;
    int32_t ret;

    strLen = VOS_StrLen(name);
    len = sizeof(BkfJobId) + strLen + 1;
    jobId = BKF_MALLOC(jobxMng->argInit.memMng, len);
    if (jobId == VOS_NULL) {
        goto error;
    }
    (void)memset_s(jobId, len, 0, len);
    jobId->lockDel = VOS_FALSE;
    jobId->softDel = VOS_FALSE;
    jobId->jobxType = jobxType;
    jobId->paramCreate = param;
    ret = snprintf_truncated_s(jobId->name, strLen + 1, "%s", name);
    if (ret <= 0) {
        goto error;
    }
    BKF_DL_NODE_INIT(&jobId->dlNode);
    BKF_DL_ADD_LAST(&jobxType->jobIdSet, &jobId->dlNode);
    BKF_SIGN_SET(jobId->sign, WFW_JOBX_ID_SIGN);
    return jobId;

error:

    if (jobId != VOS_NULL) {
        BKF_FREE(jobxMng->argInit.memMng, jobId);
    }
    return VOS_NULL;
}

STATIC void WfwJobxDelId(WfwJobxMng *jobxMng, BkfJobId *jobId)
{
    if (jobId->lockDel) {
        jobId->softDel = VOS_TRUE;
    } else {
        BKF_DL_REMOVE(&jobId->dlNode);
        BKF_SIGN_CLR(jobId->sign);
        BKF_FREE(jobxMng->argInit.memMng, jobId);
    }
    return;
}

STATIC void WfwJobxDelTypeAllId(WfwJobxMng *jobxMng, BkfJobxType *jobxType)
{
    BkfJobId *jobId = VOS_NULL;
    void *itor = VOS_NULL;

    for (jobId = WfwJobxGetFirstId(jobxMng, jobxType, &itor); jobId != VOS_NULL;
         jobId = WfwJobxGetNextId(jobxMng, jobxType, &itor)) {
        WfwJobxDelId(jobxMng, jobId);
    }
    return;
}

STATIC void WfwJobxMoveId2Tail(WfwJobxMng *jobxMng, BkfJobId *jobId)
{
    (void)jobxMng;
    if (BKF_DL_NODE_IS_IN(&jobId->dlNode)) {
        BKF_DL_REMOVE(&jobId->dlNode);
    }
    BKF_DL_ADD_LAST(&jobId->jobxType->jobIdSet, &jobId->dlNode);
    return;
}

STATIC BkfJobId *WfwJobxGetFirstId(WfwJobxMng *jobxMng, BkfJobxType *jobxType, void **itorOutOrNull)
{
    (void)jobxMng;
    BkfDlNode *tempNode = VOS_NULL;
    BkfJobId *jobId = VOS_NULL;

    tempNode = BKF_DL_GET_FIRST(&jobxType->jobIdSet);
    if (tempNode != VOS_NULL) {
        jobId = BKF_DL_GET_ENTRY(tempNode, BkfJobId, dlNode);
        if (itorOutOrNull != VOS_NULL) {
            *itorOutOrNull = BKF_DL_GET_NEXT(&jobxType->jobIdSet, &jobId->dlNode);
        }
    } else {
        jobId = VOS_NULL;
        if (itorOutOrNull != VOS_NULL) {
            *itorOutOrNull = VOS_NULL;
        }
    }

    return jobId;
}

STATIC BkfJobId *WfwJobxGetNextId(WfwJobxMng *jobxMng, BkfJobxType *jobxType, void **itorInOut)
{
    (void)jobxMng;
    BkfDlNode *tempNode = VOS_NULL;
    BkfJobId *jobId = VOS_NULL;

    tempNode = *itorInOut;
    if (tempNode != VOS_NULL) {
        jobId = BKF_DL_GET_ENTRY(tempNode, BkfJobId, dlNode);
        *itorInOut = BKF_DL_GET_NEXT(&jobxType->jobIdSet, &jobId->dlNode);
    } else {
        jobId = VOS_NULL;
        *itorInOut = VOS_NULL;
    }
    return jobId;
}

#endif

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
